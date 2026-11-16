"""Praxis 唯一工具管道(设计 §5.4;硬规则 3;findings S 类结构预防)。

模型调用、CLI、子代理、resume 一律经 ``execute()`` 执行工具——不存在第二条
路径。审批按 item 记录(approval_request / approval_decision),结论与
``args_digest`` + 声明窗口 ``window_seq`` 互相绑定:参数变化或跨窗复用
call_id 即作废旧批准(重新 ask),杜绝"批准 A 参数、执行 B 参数"与
"批准这一次、沿用下一次"(R-T7-2 闭合)。

execute 步骤序(与代码一一对应):
1. 未知工具 / 输入校验失败 → 直接写 tool_result(error_code
   ``unknown_tool``/``invalid_input``)返回——已有 result 即非 pending,
   recovery 不会列它,故不写 tool_call item、不发 started/finished;
1.5. T4 cancel 检查点:本 turn 已有 ``cancel_request(seq > 本调用最新声明
   窗口)`` → 拒绝新 dispatch(不写任何 item;清算归 turn 收敛的
   close_unresolved 单点)——幂等重入安全,批准重放绝不绕过 queued 取消;
2. ``resolve_use`` + ``permission.decide``(经模块属性调用:唯一决策入口);
   注入 hooks 时先在 decide 之前结算 PreToolUse(§5.14):started/settled item
   按 ``hook_invocation_id`` 幂等折叠(审批重入不重复 spawn),非 allowed 的
   hook 结论作为 HookDecision 进入唯一 decide——**hook allow 结构上无裁决,
   规则/bypass/地板/沙箱依旧完全生效**;
3. 审批重入(R5 硬化 + Task 14 窗口化):**新鲜 deny 是终局**,任何 bound
   决策不得翻案;bound 决策仅在 args_digest 一致 **且** policy_revision 等于
   当前 revision **且** window_seq 等于本调用声明窗口时有效(legacy item 缺
   revision/缺 window 键 = 作废,fail-closed)→ 按 allow/deny 陈述;
   不一致 → 作废,按新鲜决策处理(re-ask)——跨窗复用同 call_id+digest 不再
   继承 once 批准(R-T7-2 闭合);
4. ask → approval_request item(同 window+call_id+digest+revision 幂等,
   不重复写)→ transition(PAUSED) → emit ApprovalRequested → 返回 PAUSED;
5. deny → tool_result(DENIED, error_code ``permission_denied``);
6. allow → 执行时地板守卫(R5(c):floor_reason 双候选,命中 → DENIED,
   不写 tool_call、不 dispatch——地板在任何入口都不可批准)→ EXECUTE_PROCESS
   **先编译沙箱**再写 tool_call item(SandboxRefusal → ERROR/sandbox_refused,
   不写 tool_call,未 dispatch 语义同 invalid_input;read/write 进程内工具不编译);
7. ``run_with_timeout``(移植旧 executor 超时/取消语义,任务全程被持有;
   透传 ``ToolExecContext(sandbox=spec, permissions=ctx.permissions,
   active_skill_ids=...)``——后者从本回合已提交 item 重算,供
   ``materialize_skill_asset`` 做执行侧激活态事实查询)→
   **finally: finalize_sandbox**
   (commit/清理;超时/取消路径也执行;finalize 异常归一 tool_error,不炸穿);
8. 占位符替换 + 超 ``max_model_output_bytes`` 的**写时外部化**(注入
   ``artifacts`` 时:全文进内容寻址 artifact,item 只留头部+指针+摘要;
   未注入则旧字节截断)→ tool_result item → emit ToolFinished。
   PostToolUse 在工具返回后、``_finalize`` 之前结算:结论只进
   ``metadata["post_tool_hook"]``,绝不改 status/content、绝不阻断。

沙箱编译/清理经注入进入 core(``SandboxCompiler``/``SandboxFinalizer`` 结构
签名,adapters 结构匹配;core 不 import adapters——praxis_core_purity)。
``compile_sandbox=None`` → 无沙箱直跑(单测路径;生产 composition 恒注入)。

item payload 约定(Task 10/13/14 消费,勿漂移):content 块为
``{"type": "text", "data": {"text": str}}``(镜像 ToolResult.text);
arguments/effects/targets 为普通 JSON(thaw_json)。``tool.run`` 返回的
``tool_call_id`` 是占位符 "pending"(praxis/tools 文档化约定),本管道在
落库与返回前替换为真实 call id。

已知限制(如实,移植自旧 executor 的注释):线程不可强杀——同步工具在
超时/取消后必须等待其自然完成,由 ``_drain_task`` 持有到底。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from praxis.core import permission  # 唯一决策入口:经模块属性调用,测试 monkeypatch 此处
from praxis.core.budget import BudgetTracker
from praxis.core.compensation import (
    NOT_EXECUTED_TEXTS,
    OUTCOME_UNKNOWN_TEXTS,
    UnresolvedCall,
    unclosed_windows,
)
from praxis.core.events import EventBus
from praxis.core.skill_state import latest_active_skill_ids
from praxis.core.statemachine import TurnStatus, transition
from praxis.protocol.events import ApprovalRequested, ToolFinished, ToolStarted
from praxis.protocol.floor import floor_reason, host_private_paths
from praxis.protocol.hooks import (
    HookEvent,
    HookOutcome,
    HookSetup,
    HookSpec,
    fold_hook_items,
    hook_invocation_id,
)
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue, thaw_json
from praxis.protocol.permission import Decision, HookDecision, PermissionContext
from praxis.protocol.sandbox import SandboxRefusal, SandboxSpec, ToolExecContext
from praxis.protocol.tools import (
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolCall,
    ToolContentBlock,
    ToolEffect,
    ToolResult,
    ToolValidationError,
    args_digest,
)
from praxis.store.artifacts import ArtifactStore
from praxis.store.log import ItemEntry, Store

PENDING_TOOL_CALL_ID = "pending"
_MAX_ERROR_MESSAGE_CHARS = 512
# 外部化时 item 里保留的头部片段上限(字节;按 UTF-8 边界切,见 _finalize)。
_ARTIFACT_HEAD_BYTES = 2_048

# 沙箱编译/清理的注入签名(core 内类型别名,adapters.sandbox 结构匹配;
# core 不 import adapters——compiler/finalizer 由 composition 经 ExecCtx/TurnDeps 注入)。
type SandboxCompiler = Callable[[PermissionContext, ResolvedUse, Path], SandboxSpec]
type SandboxFinalizer = Callable[[SandboxSpec, Path], None]


@dataclass(frozen=True)
class ExecCtx:
    """一次工具执行所需的全部环境:日志、总线、工作区、工具表、权限上下文、归属回合。

    compile_sandbox/finalize_sandbox 缺省 None = 无沙箱(单测路径);
    生产 composition 恒注入真实 compiler/finalizer。
    """

    store: Store
    bus: EventBus
    workspace: Path
    tools: Mapping[str, Tool]
    permissions: PermissionContext
    turn_id: str
    thread_id: str
    policy_revision: str
    compile_sandbox: SandboxCompiler | None = None
    finalize_sandbox: SandboxFinalizer | None = None
    # None = 不外部化(单测/裸管道):超限结果退回旧字节截断语义。
    artifacts: ArtifactStore | None = None
    # None = 不接 hook(单测/裸管道):行为与未接线前逐字不变。
    hooks: HookSetup | None = None
    # None = 无预算(单测/裸管道):16 处直构点默认值零改动。T3 透传持有面;
    # web/browser 维 dispatch gate(admit_web_request)由 T15 在此消费。
    budget: BudgetTracker | None = None


# ---------------------------------------------------------------------------
# 序列化与结果构造(§payload 约定)
# ---------------------------------------------------------------------------


def _bounded(message: str) -> str:
    return " ".join(message.split())[:_MAX_ERROR_MESSAGE_CHARS]


def _json_field(value: JsonValue) -> JsonValue:
    """thaw_json 的静态返回类型含 list(_MutableJsonValue),而 JsonValue 用 tuple 表达
    序列——thaw 产物随后只进 json.dumps,两者 JSON 等价,cast 是形式桥。"""
    return cast("JsonValue", thaw_json(value))


def _text_block(text: str) -> ToolContentBlock:
    return ToolContentBlock(type="text", data={"text": text})


def _serialize_content(result: ToolResult) -> tuple[dict[str, JsonValue], ...]:
    blocks: list[dict[str, JsonValue]] = []
    for block in result.content:
        entry: dict[str, JsonValue] = {"type": block.type, "data": _json_field(block.data)}
        blocks.append(entry)
    return tuple(blocks)


def _effect_names(use: ResolvedUse) -> tuple[str, ...]:
    return tuple(sorted(effect.value for effect in use.effects))


def _target_dicts(use: ResolvedUse) -> tuple[dict[str, JsonValue], ...]:
    entries: list[dict[str, JsonValue]] = []
    for target in use.targets:
        entry: dict[str, JsonValue] = {"kind": target.kind, "value": target.value}
        entries.append(entry)
    return tuple(entries)


def _error_result(
    call: ToolCall, *, status: ResultStatus, error_code: str, message: str
) -> ToolResult:
    bounded = _bounded(message)
    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        content=(_text_block(f"{error_code}: {bounded}"),),
        status=status,
        is_error=True,
        error_code=error_code,
        error_message=bounded,
    )


def _byte_truncated(result: ToolResult, encoded: bytes, limit: int) -> ToolResult:
    """旧语义降级:保留 UTF-8 安全头部,重建单 text 块。"""
    cut = encoded[:limit].decode("utf-8", errors="ignore")
    return replace(result, content=(_text_block(cut),), truncated=True)


def _finalize(
    raw: ToolResult,
    *,
    call: ToolCall,
    tool: Tool,
    artifacts: ArtifactStore | None = None,
) -> ToolResult:
    """占位符替换 + 模型输出字节上限。

    超限且注入 ``artifacts`` 时**写时外部化**(设计 §5.7;D10):全文进
    内容寻址 artifact,item 只留头部片段 + 指针 + digest/字节数——全文只存一份
    (B22)。头部预算 ``min(_ARTIFACT_HEAD_BYTES, limit - 指针长度 - 1)`` 保证它
    严格短于全文(外部化条件即全文 > limit),因此完整正文绝不会随 item 落库。
    未注入 ``artifacts``(单测/裸管道)时保留旧字节截断语义;``put`` 抛 OSError
    (磁盘满/权限错)同样降级为截断,并标 ``artifact_unavailable=True``。
    """
    result = replace(raw, tool_call_id=call.tool_call_id, tool_name=call.tool_name)
    encoded = result.text.encode("utf-8")
    limit = tool.max_model_output_bytes
    if len(encoded) <= limit:
        return result
    if artifacts is None:
        return _byte_truncated(result, encoded, limit)

    try:
        digest = artifacts.put(encoded)
    except OSError:
        # 磁盘满/权限错:如实降级为字节截断并在 metadata 标不可用,
        # 绝不假装"已外部化"(否则模型拿到一个打不开的指针)。
        degraded = _byte_truncated(result, encoded, limit)
        degraded_metadata: dict[str, JsonValue] = dict(degraded.metadata)
        degraded_metadata["artifact_unavailable"] = True
        return replace(degraded, metadata=degraded_metadata)
    note = (
        f"\n[full output stored: artifact://{digest} "
        f"({len(encoded)} bytes); read with read_artifact]"
    )
    # 头部预算先扣掉指针本身,尽量守住 max_model_output_bytes;且恒 < 全文
    # (外部化条件 len(encoded) > limit ≥ 预算),全文不会随 item 落库。
    head_budget = min(_ARTIFACT_HEAD_BYTES, max(0, limit - len(note.encode("utf-8")) - 1))
    head = encoded[:head_budget].decode("utf-8", errors="ignore")
    metadata: dict[str, JsonValue] = dict(result.metadata)
    metadata["artifact_digest"] = digest
    metadata["artifact_bytes"] = len(encoded)
    return replace(
        result,
        content=(_text_block(f"{head}{note}"),),
        truncated=True,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# item 写入(append-only;返回即所写)
# ---------------------------------------------------------------------------


def _append_call_item(
    ctx: ExecCtx, call: ToolCall, tool: Tool, use: ResolvedUse, args: Mapping[str, JsonValue], digest: str
) -> None:
    payload: dict[str, JsonValue] = {
        "call_id": call.tool_call_id,
        "tool_name": tool.definition.name,
        "arguments": _json_field(args),
        "effects": _effect_names(use),
        "targets": _target_dicts(use),
        "args_digest": digest,
        "idempotent": tool.idempotent,
        "source": tool.source.value,
    }
    ctx.store.append_items(
        ctx.thread_id, ctx.turn_id, [ItemEntry(kind=ItemKind.TOOL_CALL, payload=payload)]
    )


def _append_result_item(ctx: ExecCtx, result: ToolResult) -> None:
    payload: dict[str, JsonValue] = {
        "call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "status": result.status.value,
        "content": _serialize_content(result),
        "is_error": result.is_error,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "truncated": result.truncated,
        "metadata": _json_field(result.metadata),
    }
    ctx.store.append_items(
        ctx.thread_id, ctx.turn_id, [ItemEntry(kind=ItemKind.TOOL_RESULT, payload=payload)]
    )


# ---------------------------------------------------------------------------
# hook 两执行点(设计 §5.14;3B 笔记 §4)
#
# 结算归位:core 写 started/settled item、协议层 fold 投影、adapters 只 spawn
# 进程。幂等身份 = ``hook_invocation_id(turn_id, event, name, digest)``:
# 审批重入时已 settled 的调用直接复用结论,**不重复 spawn**。
# ---------------------------------------------------------------------------

# PostToolUse 只观察:多个 hook 时在 metadata 里保留最严重的那条摘要。
_POST_SEVERITY: dict[str, int] = {
    "allowed": 0,
    "observed": 1,
    "unknown": 2,
    "unavailable": 3,
    "error": 4,
    "timeout": 5,
    "blocked": 6,
}


def _matching_hook_specs(
    hooks: HookSetup, event: HookEvent, tool_name: str, effects: frozenset[ToolEffect]
) -> tuple[HookSpec, ...]:
    """选择器语义:空 tool_names/effects = 该轴不筛选(与加载期约定一致)。"""
    effect_names = {effect.value for effect in effects}
    return tuple(
        spec
        for spec in hooks.specs
        if spec.event == event
        and (not spec.tool_names or tool_name in spec.tool_names)
        and (not spec.effects or not spec.effects.isdisjoint(effect_names))
    )


def _hook_request(
    ctx: ExecCtx,
    call: ToolCall,
    tool_name: str,
    use: ResolvedUse,
    args: Mapping[str, JsonValue],
    invocation_id: str,
    event: HookEvent,
) -> dict[str, JsonValue]:
    """HookRequest 的 JSON 兼容形状。

    R-T5-1:PostToolUse 有意**不含** ``source_item_id``——结果 item 在此刻尚未
    提交,旧实现的"源 item"在 praxis 结算序里不存在;T21 对照表钉住该差异。
    """
    return {
        "event": event,
        "invocation_id": invocation_id,
        "thread_id": ctx.thread_id,
        "turn_id": ctx.turn_id,
        "workspace": str(ctx.workspace.expanduser().resolve()),
        "tool_call_id": call.tool_call_id,
        "tool_name": tool_name,
        "arguments": _json_field(args),
        "effects": _effect_names(use),
        "input_revision": ctx.policy_revision,
    }


def _append_hook_item(
    ctx: ExecCtx,
    *,
    invocation_id: str,
    event: HookEvent,
    hook: str,
    phase: str,
    status: str | None = None,
    feedback: str = "",
) -> None:
    """started = 最小相位载荷;settled 携带结论(status/feedback)。"""
    payload: dict[str, JsonValue] = {
        "invocation_id": invocation_id,
        "event": event,
        "hook": hook,
        "phase": phase,
    }
    if status is not None:
        payload["status"] = status
        payload["feedback"] = feedback
    ctx.store.append_items(
        ctx.thread_id, ctx.turn_id, [ItemEntry(kind=ItemKind.HOOK_INVOCATION, payload=payload)]
    )


async def _settle_hook(
    ctx: ExecCtx,
    hooks: HookSetup,
    spec: HookSpec,
    request: Mapping[str, JsonValue],
    invocation_id: str,
    items: Sequence[Item],
) -> HookOutcome:
    """一次 hook 调用的幂等结算:已 settled ⇒ 复用(0 spawn),否则 spawn 并落账。

    外层 ``CancelledError``:先写 ``settled status="unknown"`` 再原样上抛——
    进程是否已产生副作用不可知,恢复面必须 fail-closed 而不是假装没发生。
    """
    folded = fold_hook_items(items, invocation_id)
    if folded is not None:
        return folded
    _append_hook_item(
        ctx,
        invocation_id=invocation_id,
        event=spec.event,
        hook=spec.name,
        phase="started",
    )
    try:
        outcome = await hooks.spawn(spec, request)
    except asyncio.CancelledError:
        _append_hook_item(
            ctx,
            invocation_id=invocation_id,
            event=spec.event,
            hook=spec.name,
            phase="settled",
            status="unknown",
            feedback="outer cancellation",
        )
        raise
    _append_hook_item(
        ctx,
        invocation_id=invocation_id,
        event=spec.event,
        hook=spec.name,
        phase="settled",
        status=outcome.status,
        feedback=outcome.feedback,
    )
    return outcome


async def _pre_hook_decisions(
    ctx: ExecCtx,
    hooks: HookSetup,
    call: ToolCall,
    tool: Tool,
    use: ResolvedUse,
    args: Mapping[str, JsonValue],
    digest: str,
    items: Sequence[Item],
) -> tuple[HookDecision, ...]:
    """PreToolUse 裁决映射:非 allowed 一律 deny;allowed 结构上无裁决。

    allow 不进入 ``PermissionContext.hook_decisions``——没有任何"hook 放行"
    可以豁免规则/bypass/地板/沙箱;hook 只能收紧。
    """
    decisions: list[HookDecision] = []
    for spec in _matching_hook_specs(hooks, "PreToolUse", tool.definition.name, use.effects):
        invocation_id = hook_invocation_id(ctx.turn_id, "PreToolUse", spec.name, digest)
        request = _hook_request(
            ctx, call, tool.definition.name, use, args, invocation_id, "PreToolUse"
        )
        outcome = await _settle_hook(ctx, hooks, spec, request, invocation_id, items)
        if outcome.status != "allowed":
            decisions.append(
                HookDecision(
                    source=f"{spec.name}:{outcome.status}",
                    verdict="deny",
                    feedback=outcome.feedback,
                )
            )
    return tuple(decisions)


async def _apply_post_hooks(
    ctx: ExecCtx,
    hooks: HookSetup,
    call: ToolCall,
    tool: Tool,
    use: ResolvedUse,
    args: Mapping[str, JsonValue],
    digest: str,
    items: Sequence[Item],
    raw: ToolResult,
) -> ToolResult:
    """PostToolUse 只观察:结论进 metadata,绝不改 status/content,绝不阻断。"""
    summary: HookOutcome | None = None
    for spec in _matching_hook_specs(hooks, "PostToolUse", tool.definition.name, use.effects):
        invocation_id = hook_invocation_id(ctx.turn_id, "PostToolUse", spec.name, digest)
        request = _hook_request(
            ctx, call, tool.definition.name, use, args, invocation_id, "PostToolUse"
        )
        outcome = await _settle_hook(ctx, hooks, spec, request, invocation_id, items)
        if summary is None or _POST_SEVERITY[outcome.status] > _POST_SEVERITY[summary.status]:
            summary = outcome
    if summary is None:
        return raw
    metadata: dict[str, JsonValue] = dict(raw.metadata)
    metadata["post_tool_hook"] = {
        "status": summary.status,
        "feedback": _bounded(summary.feedback),
    }
    return replace(raw, metadata=metadata)


# ---------------------------------------------------------------------------
# 审批重入(摘要绑定)
# ---------------------------------------------------------------------------


def _latest_approval_decision(items: list[Item], call_id: str, window_seq: int | None) -> Item | None:
    """最近一条绑定**声明窗口**的 approval_decision(R-T7-2 闭合)。

    绑定键 = (window_seq, call_id, args_digest, policy_revision):window_seq 是
    声明该调用的 model_response item 的 seq(None = CLI/工具直调域)。同 turn
    跨窗复用 call_id 时,即使参数与策略修订都相同,旧窗口的 once 批准也不再
    沿用——新声明窗口必须重新 ask。legacy item 缺 window_seq 键一律视作不匹配
    (fail-closed,与缺 revision 同纪律:读端 fail-open 只保护渲染,决策面
    缺维=无效)。
    """
    for item in reversed(items):
        if (
            item.kind == ItemKind.APPROVAL_DECISION
            and item.payload.get("call_id") == call_id
            and "window_seq" in item.payload
            and item.payload["window_seq"] == window_seq
        ):
            return item
    return None


def _decision_reason(item: Item) -> str:
    reason = item.payload.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    return f"approval_decision {item.payload.get('verdict')} bound to args digest"


def _bound_decision_valid(bound: Item, digest: str, policy_revision: str, window_seq: int | None) -> bool:
    """R5(b) + Task 14:digest / 批准时刻 policy_revision / 声明窗口三维绑定,缺一不可。

    revision 从 approval_decision item 读取(Session 落笔时 stamp);legacy item
    缺 revision → None != 当前 revision → 作废(fail-closed,重新 ask)。
    window_seq 同理:缺键 = 无效(即使调用处于 None 域),等值才放行。
    """
    return (
        bound.payload.get("args_digest") == digest
        and bound.payload.get("policy_revision") == policy_revision
        and "window_seq" in bound.payload
        and bound.payload["window_seq"] == window_seq
    )


def _execution_floor_reason(use: ResolvedUse) -> str:
    """R5(c) 执行时地板守卫:allow 之后、tool_call/dispatch 之前的最后一道地板。

    与 decide 第 4 步同一调用模式(protocol/floor.py 单一数据源,双候选):
    词法(floor_reason 内部 expanduser+abspath)与真实(non-strict resolve())
    任一命中即拒——symlink 不转移地板豁免。write 旗标 = WRITE_WORKSPACE 或
    DESTRUCTIVE ∈ effects。ask 规则 + 地板目标 → 批准 → 此处仍 deny:
    地板在任何入口都不可批准。"" = 全部通过。
    """
    write = ToolEffect.WRITE_WORKSPACE in use.effects or ToolEffect.DESTRUCTIVE in use.effects
    private_paths = host_private_paths()
    for target in use.targets:
        if target.kind != "path":
            continue
        candidates = [Path(target.value)]
        try:
            candidates.append(Path(target.value).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            pass  # 词法候选仍裁决(non-strict totality,同 decide 第 4 步)
        for candidate in candidates:
            reason = floor_reason(candidate, write=write, private_paths=private_paths)
            if reason:
                return reason
    return ""


def _pause_for_approval(
    ctx: ExecCtx,
    call: ToolCall,
    tool: Tool,
    use: ResolvedUse,
    args: Mapping[str, JsonValue],
    digest: str,
    items: list[Item],
) -> ToolResult:
    duplicate = any(
        item.kind == ItemKind.APPROVAL_REQUEST
        and item.payload.get("call_id") == call.tool_call_id
        and item.payload.get("args_digest") == digest
        # K-2:revision 也是去重键的一部分。策略修订后重入的 re-ask 必须写入
        # 新 revision 的 request,否则展示面按陈旧 revision 抑制既有 request
        # ⇒ PAUSED 且 pending_approvals 为空(B32 同族)。
        and item.payload.get("policy_revision") == ctx.policy_revision
        # Task 14:window 维升级 B32 抑制判据 (call_id,digest) →
        # (window,call_id,digest)。缺 window 键的 legacy request 不算重复
        # (presence 判据)——否则 re-ask 写不出带窗新请求,新批准永远绑不上。
        and "window_seq" in item.payload
        and item.payload["window_seq"] == call.window_seq
        for item in items
    )
    if not duplicate:  # 幂等重入:同 window + call_id + digest + revision 不重复写请求
        payload: dict[str, JsonValue] = {
            "call_id": call.tool_call_id,
            "tool_name": tool.definition.name,
            "arguments": _json_field(args),
            "effects": _effect_names(use),
            "targets": _target_dicts(use),
            "args_digest": digest,
            "policy_revision": ctx.policy_revision,
            "window_seq": call.window_seq,
        }
        ctx.store.append_items(
            ctx.thread_id,
            ctx.turn_id,
            [ItemEntry(kind=ItemKind.APPROVAL_REQUEST, payload=payload)],
        )
    transition(ctx.store, ctx.bus, ctx.turn_id, TurnStatus.PAUSED, reason="approval_required")
    ctx.bus.emit(
        ApprovalRequested(
            turn_id=ctx.turn_id,
            call_id=call.tool_call_id,
            tool_name=tool.definition.name,
            args_digest=digest,
        )
    )
    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=tool.definition.name,
        status=ResultStatus.PAUSED,
    )


# ---------------------------------------------------------------------------
# cancel 检查点(T4:dispatch 前;幂等重入安全)
# ---------------------------------------------------------------------------


def _cancel_refusal(ctx: ExecCtx, items: list[Item], call: ToolCall) -> ToolResult | None:
    """本 turn 已有 ``cancel_request(seq > 本调用最新声明窗口)`` → 拒绝新 dispatch。

    判据经 compensation.unclosed_windows(不另写窗口匹配):窗口已闭合或取消
    早于声明 → None(正常路径)。拒绝结果与 close_unresolved 同矩阵
    (started→outcome_unknown / 未 started→not_executed),但**不写 item**——
    清算 item 由 turn 收敛时的 close_unresolved 统一落库(唯一清算写者);
    检查点只挡执行,不产生第二份补偿账。PAUSED 期间 queued 的取消在 resume
    审批重放时同样被拒——批准绝不绕过取消。
    """
    cancel_seq: int | None = max(
        (item.seq for item in items if item.kind == ItemKind.CANCEL_REQUEST), default=None
    )
    if cancel_seq is None:
        return None
    target: UnresolvedCall | None = None
    for unresolved in unclosed_windows(items, ctx.tools):
        if unresolved.call.call_id == call.tool_call_id:
            target = unresolved  # 升序扫:最后匹配 = 最新声明窗口
    if target is None or target.declaration_seq >= cancel_seq:
        return None
    if target.started:
        status, text = ResultStatus.OUTCOME_UNKNOWN, OUTCOME_UNKNOWN_TEXTS["cancelled"]
    else:
        status, text = ResultStatus.NOT_EXECUTED, NOT_EXECUTED_TEXTS["cancelled"]
    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        content=(_text_block(text),),
        status=status,
    )


# ---------------------------------------------------------------------------
# 唯一执行入口
# ---------------------------------------------------------------------------


async def execute(call: ToolCall, ctx: ExecCtx) -> ToolResult:
    """见模块 docstring 步骤序。任何输入都不抛业务异常:一切结局皆 ToolResult。"""
    tool = ctx.tools.get(call.tool_name)
    if tool is None:
        result = _error_result(
            call, status=ResultStatus.ERROR, error_code="unknown_tool",
            message=f"unknown tool: {call.tool_name}",
        )
        _append_result_item(ctx, result)
        return result

    try:
        args = tool.validate_input(call.arguments)
        use = tool.resolve_use(args)
    except (ToolValidationError, ValueError, OSError, TypeError) as exc:
        # resolve_use 的越界围栏(checked_path)同样归入 invalid_input:
        # 未形成 use 就不存在"已决策未执行"的中间态,直接落 result。
        # checked_path 对嵌入 NUL / ENAMETOOLONG 等畸形路径抛的是 ValueError/OSError;
        # 这里统一收口,否则单个畸形参数就能炸穿 turn(B10 家族)。
        # 非校验异常带类型名前缀:工具实现里的编码 bug 不伪装成用户输入错误。
        message = (
            f"{exc.path}: {exc.message}"
            if isinstance(exc, ToolValidationError)
            else f"{type(exc).__name__}: {_bounded(str(exc))}"
        )
        result = _error_result(
            call, status=ResultStatus.ERROR, error_code="invalid_input",
            message=message,
        )
        _append_result_item(ctx, result)
        return result

    digest = args_digest(args)
    items = ctx.store.read_turn_items(ctx.turn_id)

    # T4 cancel 检查点:先于 hook/decide/审批——取消中的 turn 不再启动新执行,
    # 也不写 approval_request/tool_call(收敛与清算归 turn 循环步边界)。
    refusal = _cancel_refusal(ctx, items, call)
    if refusal is not None:
        return refusal

    # 步序 2 的 hook 扩展:PreToolUse 在唯一 decide 之前结算;allow 不产生裁决,
    # 因此规则/bypass/地板/沙箱对该调用仍然完全有效(allow 丝毫不豁免)。
    decide_permissions = ctx.permissions
    if ctx.hooks is not None:
        decisions = await _pre_hook_decisions(
            ctx, ctx.hooks, call, tool, use, args, digest, items
        )
        decide_permissions = replace(ctx.permissions, hook_decisions=decisions)

    decision = permission.decide(tool, use, decide_permissions)
    bound = _latest_approval_decision(items, call.tool_call_id, call.window_seq)
    # R5(a):新鲜 deny 是终局——bound allow 只能把 ask/fallback 带入执行,
    # 绝不覆盖规则 deny/地板 deny/sandbox_unavailable。
    if (
        not decision.is_deny
        and bound is not None
        and _bound_decision_valid(bound, digest, ctx.policy_revision, call.window_seq)
    ):
        verdict = bound.payload.get("verdict")
        if verdict == "allow":
            decision = Decision.allow("approval_decision: args digest bound")
        elif verdict == "deny":
            decision = Decision.deny(_decision_reason(bound))
        # 其余 verdict 值:视为无决策,按新鲜 decide 走

    if decision.is_ask:
        return _pause_for_approval(ctx, call, tool, use, args, digest, items)
    if decision.is_deny:
        result = _error_result(
            call, status=ResultStatus.DENIED, error_code="permission_denied",
            message=decision.reason,
        )
        _append_result_item(ctx, result)
        return result

    # R5(c):执行时地板守卫(含 bound-allow 重入)——命中即 DENIED,
    # 不写 tool_call item、不 dispatch。
    floor_hit = _execution_floor_reason(use)
    if floor_hit:
        result = _error_result(
            call, status=ResultStatus.DENIED, error_code="permission_denied",
            message=f"floor: {floor_hit}",
        )
        _append_result_item(ctx, result)
        return result

    # 沙箱编译先于 tool_call item:拒绝即结果(ERROR/sandbox_refused),未 dispatch。
    # 只对 EXECUTE_PROCESS 编译——read/write 进程内工具不消费沙箱(R9);
    # 不可用已在 decide 第 7 步 deny,此处 compile_sandbox=None 为单测路径。
    spec: SandboxSpec | None = None
    if ctx.compile_sandbox is not None and ToolEffect.EXECUTE_PROCESS in use.effects:
        try:
            spec = ctx.compile_sandbox(ctx.permissions, use, ctx.workspace)
        except SandboxRefusal as exc:
            result = _error_result(
                call, status=ResultStatus.ERROR, error_code="sandbox_refused", message=exc.reason,
            )
            _append_result_item(ctx, result)
            return result

    _append_call_item(ctx, call, tool, use, args, digest)
    ctx.bus.emit(
        ToolStarted(turn_id=ctx.turn_id, call_id=call.tool_call_id, tool_name=tool.definition.name)
    )

    finalize_failure: str | None = None
    try:
        # R-T11-2:权限快照随执行上下文进入工具,供目录型读取工具逐条目过滤 deny。
        # T9:激活态同样从本回合已提交 item 重算(与审批同一份 items 列表)——
        # materialize_skill_asset 以它为唯一激活事实,不是第二决策入口。
        raw = await run_with_timeout(
            tool,
            args,
            ToolExecContext(
                sandbox=spec,
                permissions=ctx.permissions,
                active_skill_ids=latest_active_skill_ids(items),
            ),
        )
    finally:
        # commit/清理在超时/取消路径也执行;异常归一为 tool_error,绝不炸穿管道。
        if spec is not None and ctx.finalize_sandbox is not None:
            try:
                ctx.finalize_sandbox(spec, ctx.workspace)
            except Exception as exc:  # noqa: BLE001 — finalize 异常一律归一
                finalize_failure = str(exc)
    if finalize_failure is not None:
        raw = _synthetic_result(
            tool, status=ResultStatus.ERROR, error_code="tool_error",
            message=f"sandbox finalize failed: {finalize_failure}",
        )
    # PostToolUse 在工具返回后、_finalize/result item 之前:只写观察摘要,
    # status/content 逐字不变,绝不阻断(旧 observed 语义)。
    if ctx.hooks is not None:
        raw = await _apply_post_hooks(
            ctx, ctx.hooks, call, tool, use, args, digest, items, raw
        )
    result = _finalize(raw, call=call, tool=tool, artifacts=ctx.artifacts)
    _append_result_item(ctx, result)
    ctx.bus.emit(
        ToolFinished(turn_id=ctx.turn_id, call_id=call.tool_call_id, status=result.status.value)
    )
    return result


# ---------------------------------------------------------------------------
# 超时与取消(旧 executor :1564-1657 语义移植;record/continuation/progress 分支删除)
# ---------------------------------------------------------------------------


def _synthetic_result(tool: Tool, *, status: ResultStatus, error_code: str, message: str) -> ToolResult:
    """管道自造结果(run_with_timeout 层面还没有 call_id):用占位符,execute 再替换。"""
    bounded = _bounded(message)
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name=tool.definition.name,
        content=(_text_block(f"{error_code}: {bounded}"),),
        status=status,
        is_error=True,
        error_code=error_code,
        error_message=bounded,
    )


async def _drain_task[T](task: asyncio.Task[T]) -> T:
    """Keep terminal work owned until it stops, including repeated cancellation.

    移植旧 executor 的核心持有循环:shield 让 owned 任务不受本任务取消波及;
    每次被取消都检查 owned 是否真已 cancelled——是则如实上抛,否则继续持有。
    这是"重复 cancel 不失控"的全部原因,不得简化。
    """
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise


async def _call_runner(
    tool: Tool, arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext
) -> ToolResult:
    """调用 tool.run,把业务异常归一为 tool_error 结果(CancelledError 永远上抛)。"""
    try:
        if inspect.iscoroutinefunction(tool.run):
            # iscoroutinefunction 的 TypeGuard 把返回粗化为 Coroutine[..., Any],cast 归位。
            return cast("ToolResult", await tool.run(arguments, exec_ctx))
        # 同步工具:线程不可杀——worker task + shield,被取消时 drain 到线程真正结束。
        worker = asyncio.create_task(asyncio.to_thread(tool.run, arguments, exec_ctx))
        try:
            raw = await asyncio.shield(worker)
        except asyncio.CancelledError:
            raw = await _drain_task(worker)
        if isinstance(raw, Awaitable):
            return await raw
        return raw
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — 工具异常一律归一,绝不炸穿管道
        return _synthetic_result(tool, status=ResultStatus.ERROR, error_code="tool_error", message=str(exc))


async def run_with_timeout(
    tool: Tool, arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext
) -> ToolResult:
    """执行工具并实施 timeout_seconds / interrupt_behavior(唯一实现,execute 专用)。

    语义(逐条对齐旧 executor):
    - 正常完成 → 工具真实结果;
    - 超时 FINISH_CURRENT → drain 到工具真正结束,交付真实结果(如实等待);
    - 超时 CANCEL → task.cancel() + drain(持有),返回 TIMEOUT 结果;
    - 外部 CancelledError → 持有工具任务直到停止,再原样上抛——
      tool_result 由谁写、回合转什么状态,是回合循环(Task 13)的决定;
    - tool.run 抛异常 → ERROR/tool_error(在 _call_runner 归一)。
    """
    task = asyncio.create_task(_call_runner(tool, arguments, exec_ctx))
    try:
        done, _pending = await asyncio.wait({task}, timeout=tool.timeout_seconds)
        if task in done:
            return task.result()
        if tool.interrupt_behavior is InterruptBehavior.FINISH_CURRENT:
            # 线程/协程都杀不掉正在做的收尾:等它真正完成,交付真实结果。
            return await _drain_task(task)
        task.cancel()
        try:
            await _drain_task(task)
        except asyncio.CancelledError:
            pass
        return _synthetic_result(
            tool, status=ResultStatus.TIMEOUT, error_code="timeout",
            message=f"tool '{tool.definition.name}' exceeded {tool.timeout_seconds}s",
        )
    except asyncio.CancelledError:
        if tool.interrupt_behavior is not InterruptBehavior.FINISH_CURRENT:
            task.cancel()
        try:
            await _drain_task(task)
        except asyncio.CancelledError:
            pass
        raise

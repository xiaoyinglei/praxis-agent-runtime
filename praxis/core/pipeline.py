"""Praxis 唯一工具管道(设计 §5.4;硬规则 3;findings S 类结构预防)。

模型调用、CLI、子代理、resume 一律经 ``execute()`` 执行工具——不存在第二条
路径。审批按 item 记录(approval_request / approval_decision),结论与
``args_digest`` 互相绑定:参数变化即作废旧批准(重新 ask),杜绝
"批准 A 参数、执行 B 参数"。

execute 步骤序(与代码一一对应):
1. 未知工具 / 输入校验失败 → 直接写 tool_result(error_code
   ``unknown_tool``/``invalid_input``)返回——已有 result 即非 pending,
   recovery 不会列它,故不写 tool_call item、不发 started/finished;
2. ``resolve_use`` + ``permission.decide``(经模块属性调用:唯一决策入口);
3. 审批重入(R5 硬化):**新鲜 deny 是终局**,任何 bound 决策不得翻案;
   bound 决策仅在 args_digest 一致 **且** policy_revision 等于当前 revision
   时有效(legacy item 缺 revision = 作废,fail-closed)→ 按 allow/deny 陈述;
   不一致 → 作废,按新鲜决策处理(re-ask);
4. ask → approval_request item(同 call_id+digest 幂等,不重复写)→
   transition(PAUSED) → emit ApprovalRequested → 返回 PAUSED;
5. deny → tool_result(DENIED, error_code ``permission_denied``);
6. allow → 执行时地板守卫(R5(c):floor_reason 双候选,命中 → DENIED,
   不写 tool_call、不 dispatch——地板在任何入口都不可批准)→ EXECUTE_PROCESS
   **先编译沙箱**再写 tool_call item(SandboxRefusal → ERROR/sandbox_refused,
   不写 tool_call,未 dispatch 语义同 invalid_input;read/write 进程内工具不编译);
7. ``run_with_timeout``(移植旧 executor 超时/取消语义,任务全程被持有;
   透传 ``ToolExecContext(sandbox=spec)``)→ **finally: finalize_sandbox**
   (commit/清理;超时/取消路径也执行;finalize 异常归一 tool_error,不炸穿);
8. 占位符替换 + 按 ``max_model_output_bytes`` 截断 → tool_result item →
   emit ToolFinished。

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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from praxis.core import permission  # 唯一决策入口:经模块属性调用,测试 monkeypatch 此处
from praxis.core.events import EventBus
from praxis.core.statemachine import TurnStatus, transition
from praxis.protocol.events import ApprovalRequested, ToolFinished, ToolStarted
from praxis.protocol.floor import floor_reason, host_private_paths
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue, thaw_json
from praxis.protocol.permission import Decision, PermissionContext
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
from praxis.store.log import ItemEntry, Store

PENDING_TOOL_CALL_ID = "pending"
_MAX_ERROR_MESSAGE_CHARS = 512

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


def _finalize(raw: ToolResult, *, call: ToolCall, tool: Tool) -> ToolResult:
    """占位符替换 + 模型输出字节上限(超限:保留 joined 文本头部,重建单 text 块)。"""
    result = replace(raw, tool_call_id=call.tool_call_id, tool_name=call.tool_name)
    encoded = result.text.encode("utf-8")
    limit = tool.max_model_output_bytes
    if len(encoded) > limit:
        cut = encoded[:limit].decode("utf-8", errors="ignore")
        result = replace(result, content=(_text_block(cut),), truncated=True)
    return result


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
# 审批重入(摘要绑定)
# ---------------------------------------------------------------------------


def _latest_approval_decision(items: list[Item], call_id: str) -> Item | None:
    for item in reversed(items):
        if item.kind == ItemKind.APPROVAL_DECISION and item.payload.get("call_id") == call_id:
            return item
    return None


def _decision_reason(item: Item) -> str:
    reason = item.payload.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    return f"approval_decision {item.payload.get('verdict')} bound to args digest"


def _bound_decision_valid(bound: Item, digest: str, policy_revision: str) -> bool:
    """R5(b):digest 绑定 + 批准时刻 policy_revision 绑定,二者缺一不可。

    revision 从 approval_decision item 读取(Session 落笔时 stamp);legacy item
    缺 revision → None != 当前 revision → 作废(fail-closed,重新 ask)。
    """
    return bound.payload.get("args_digest") == digest and bound.payload.get("policy_revision") == policy_revision


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
        for item in items
    )
    if not duplicate:  # 幂等重入:同 call_id + 同 digest 不重复写请求
        payload: dict[str, JsonValue] = {
            "call_id": call.tool_call_id,
            "tool_name": tool.definition.name,
            "arguments": _json_field(args),
            "effects": _effect_names(use),
            "targets": _target_dicts(use),
            "args_digest": digest,
            "policy_revision": ctx.policy_revision,
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
    except ToolValidationError as exc:
        # resolve_use 的越界围栏(checked_path)同样归入 invalid_input:
        # 未形成 use 就不存在"已决策未执行"的中间态,直接落 result。
        result = _error_result(
            call, status=ResultStatus.ERROR, error_code="invalid_input",
            message=f"{exc.path}: {exc.message}",
        )
        _append_result_item(ctx, result)
        return result

    digest = args_digest(args)
    items = ctx.store.read_turn_items(ctx.turn_id)

    decision = permission.decide(tool, use, ctx.permissions)
    bound = _latest_approval_decision(items, call.tool_call_id)
    # R5(a):新鲜 deny 是终局——bound allow 只能把 ask/fallback 带入执行,
    # 绝不覆盖规则 deny/地板 deny/sandbox_unavailable。
    if not decision.is_deny and bound is not None and _bound_decision_valid(bound, digest, ctx.policy_revision):
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
        raw = await run_with_timeout(tool, args, ToolExecContext(sandbox=spec))
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
    result = _finalize(raw, call=call, tool=tool)
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

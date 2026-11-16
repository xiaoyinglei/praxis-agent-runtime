"""Praxis 最小 turn 循环(设计 §5.4/§5.12;findings B15/B11;验收"多工具 turn 跑通")。

``TurnResult.status`` 就是 ``TurnStatus`` 本身——没有映射层(结构性 B15:取消
是独立结局,不会被折叠成失败;失败永远携带真实 reason,不会退化成笼统标签)。
每 turn 一个错误边界:循环内任何异常都收敛为本 turn 的 FAILED,永不炸穿会话。

步骤序(与代码一一对应):
0. 入口:``TurnState.steps`` 置为 ``fold_turn_facts`` 的声明轮数(R-PF-4)——
   steps 跨 pause/resume 累计,护栏基数不再每次运行从 0 计(阶段4 T3);
1. ``resume_calls`` 非空 → 先逐个 ``pipeline.execute``(审批重入路径;调用方
   保证回合已 RUNNING——Task 14 的 resume 负责先转换再重入),PAUSED 同步序 5;
2. 每轮重读 thread items → ``build_context`` 折叠 → ``to_messages`` 投影 →
   前置 system 消息;steering item 在步边界自然进入上下文(它已在日志里);
   cancel_request 在同一缝读取(T4):本 turn 已有取消事实 → 全组清算收敛
   CANCELLED,不再请求模型;
3. ``adapter.build_request`` + 注入的 ``retry_stream``(缺省不重试)收齐
   ``ModelEvent`` → ``assemble_model_events`` → 每次尝试落 ``model_attempt``
   item、最终成功落 model_response item → emit ``ModelOutputDelta`` 与
   ``ModelResponseReceived``(预览 ≤200 字符);core 只走流式,complete 只是
   同一路径的薄门面(设计 D1)。重试机制由 composition 以结构签名注入,
   core 不认识 openai/httpx;重试期间不 emit 增量,失败首帧不会变成回滚不
   掉的重影。
3.5. 预算闸(阶段4 T3,注入 ``budget`` 时):``refresh()`` 后 ``admit_dispatch``
   ——树维度耗尽 → ``transition(PAUSED, "budget_exhausted")``、cost 限额而
   单价缺失 → ``transition(PAUSED, "pricing_unavailable")``,可 resume+extend
   续跑。保护尾豁免逐次 dispatch(R-T3-7):一次性续作旗标消费即复位,
   超阈值预授只给摘要调用——压缩未产出摘要则按普通调用重新准入。与
   max_steps 硬护栏双闸并存、语义不同(R-T3-1);
4. 无 tool_calls → 终态守卫(T4,B14 对称性):同缝 cancel 收敛 + compensation
   全组清算(常态零 entries——每轮工具全结算才推进;防御遗留未闭合窗口)→
   ``transition(COMPLETED, "model_finished")``;
5. 有 tool_calls → 逐个 ``execute``;任一 PAUSED → 管道已完成 RUNNING→PAUSED
   转换,直接返回并附未批摘要;
6. ``steps += 1``(初值见步序 0);≥ max_steps →
   ``transition(FAILED, "step_limit_exceeded")``(失控防御,保持 FAILED);
7. ``CancelledError`` → 先 ``close_unresolved`` 全组清算补 entries(B14:取消
   不再"零补偿 item";B29:同窗口缺结果不早退)再
   ``transition(CANCELLED, "cancelled_by_user")``——except 路径内只有同步
   调用,不 await、不上抛:SDK 表面返回结果,任务生命周期归会话层;
8. ``Exception`` → ``transition(FAILED, "<类型>: <消息>")``。

item 分工:pipeline 写 tool_call/tool_result/approval_request,本模块写
model_response/model_attempt,状态机写 turn_status——没有第二写入者,不存在双写。
终态清算的 tool_result 经 compensation.close_unresolved 单点合成(T4:cancel/
complete 守卫与会话 resume 共用同一协议;pair 已配即幂等,不产生重复账)。
核心层对总线与订阅者只 emit、永不 await(硬规则 4)。
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from praxis.core.budget import Admission, BudgetTracker, fold_turn_facts
from praxis.core.compaction import (
    SUMMARY_MAX_OUTPUT_TOKENS,
    SUMMARY_PROMPT_MAX_CHARS,
    CompactionOutcome,
    maybe_compact,
)
from praxis.core.compensation import close_unresolved, unclosed_windows
from praxis.core.context import (
    append_memory_navigation,
    append_skill_catalog,
    build_context,
    elide_old_tool_outputs,
    to_messages,
)
from praxis.core.events import EventBus
from praxis.core.memory import MemorySnapshot
from praxis.core.pipeline import ExecCtx, SandboxCompiler, SandboxFinalizer, execute
from praxis.core.statemachine import TurnStatus, transition
from praxis.protocol.agents import ChildSummary
from praxis.protocol.events import ModelOutputDelta, ModelResponseReceived, ModelRetried
from praxis.protocol.hooks import HookSetup
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import (
    ModelAdapter,
    ModelEvent,
    ModelMessage,
    ModelReasoningDelta,
    ModelResponse,
    ModelSettings,
    ModelTextDelta,
    ModelToolCall,
    ModelToolCallDelta,
    WireRequest,
    assemble_model_events,
    collect_events,
    usage_payload,
)
from praxis.protocol.permission import ApprovalSummary, PermissionContext, PermissionNotice
from praxis.protocol.retry import (
    AttemptOutcome,
    AttemptRecord,
    RetryAtSource,
    RetryingStream,
    clock_floor,
)
from praxis.protocol.skills import SkillCatalog
from praxis.protocol.tools import ResultStatus, Tool, ToolCall, ToolResult
from praxis.store.artifacts import ArtifactStore
from praxis.store.log import ItemEntry, Store

_CONTENT_PREVIEW_CHARS = 200
_UNKNOWN_TOOL_ERROR_CODE = "unknown_tool"


# ---------------------------------------------------------------------------
# 依赖与循环状态(注入式:core 不认识具体 adapter/tool 实现)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnDeps:
    """一次 run_turn 所需的全部注入依赖:管道环境 + 模型适配 + 提示与预算。

    compile_sandbox/finalize_sandbox 透传至 ExecCtx(Task 7 沙箱接线);
    缺省 None = 无沙箱(单测路径),生产 composition 恒注入。
    """

    store: Store
    bus: EventBus
    adapter: ModelAdapter
    tools: Mapping[str, Tool]
    permissions: PermissionContext
    workspace: Path
    system_prompt: str
    max_steps: int = 50
    compile_sandbox: SandboxCompiler | None = None
    finalize_sandbox: SandboxFinalizer | None = None
    # None = 工具结果不外部化(单测路径);生产 composition 注入 ArtifactStore。
    artifacts: ArtifactStore | None = None
    # None = 不重试(ScriptedAdapter/单测路径);生产 composition 注入真实机制。
    # 只有注入时才有 model_attempt 事实账——无重试机制就没有第二个账本。
    retry_stream: RetryingStream | None = None
    # None = 不自动压缩(单测/ScriptedAdapter 路径);生产 composition 按 ModelSpec
    # 计算并注入。0 同样视为关闭(预算耗尽)。
    compact_limit_tokens: int | None = None
    # D2 唯一测量入口(measure_request)的注入形式:praxis_core_purity 禁止
    # core import adapters,由 composition 绑定。压缩阈值判定只走这一份数据。
    measure_tokens: Callable[[WireRequest], int] | None = None
    # 摘要请求输出预算;composition 按 min(2048, spec.max_output_tokens) 注入。
    summary_max_output_tokens: int = SUMMARY_MAX_OUTPUT_TOKENS
    # 摘要请求正文上限(K-4);composition 按上下文窗口推导后注入。
    summary_prompt_max_chars: int = SUMMARY_PROMPT_MAX_CHARS
    # None = 不接 hook(单测路径);生产 composition 装配 Host hook 并注入。
    hooks: HookSetup | None = None
    # None/空目录 = 恒等路径(既有 scripted/单测 item 序列零变化);生产
    # composition 注入 loader 产出的 catalog。目录只经 append_skill_catalog
    # 这一个纯函数步骤进入本轮消息组装(§5.14/R-T7-1)。
    skills_catalog: SkillCatalog | None = None
    # None = 无 memory 导航(enable_memory=False / use_memories=False / 空 state);
    # 非空时只经 append_memory_navigation 这一个纯函数步骤进入消息组装,
    # 且在 append_skill_catalog 之后(T12;R-T7-1 同纪律)。
    memory_navigation: MemorySnapshot | None = None
    # None = 无预算(单测路径零扰动);生产 Session._execute 恒注入 BudgetTracker。
    # 步边界 refresh → admit_dispatch;耗尽 → PAUSED(budget_exhausted /
    # pricing_unavailable),可 resume+extend 续跑(阶段4 T3)。
    budget: BudgetTracker | None = None


@dataclass
class TurnState:
    """显式可变循环状态:steps 跨 pause/resume 累计(R-PF-4,阶段4 T3)。

    run_turn 入口把 steps 置为 ``fold_turn_facts`` 的声明轮数(tool_calls 非空
    的 model_response 条数);每完成一轮"请求+全部工具"再 +1。重放轮没有新的
    model_response 事实,由进程内自增补计——两者相加与旧"每次运行总自增量"
    逐值一致,但基数不再每次 run_turn 从 0 计。
    """

    turn_id: str
    thread_id: str
    steps: int = 0


# ---------------------------------------------------------------------------
# 结果形状(无映射层,B15)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnResult:
    """run_turn 的终结结果:status 即 TurnStatus,reason 即状态机的 reason。

    ``notices`` 是本次续跑**落地权限更新**时的如实披露(如 NetworkRule 授权
    使 bash 命令网络 fail-closed 的 D9 提示);空元组 = 无披露,既有构造零改动。
    ``budget_dimension``(R-T3-6,尾部默认零破坏)只在 reason=="budget_exhausted"
    时填耗尽维名,供 CLI 点名对应补预算出路;其余结局恒 None。
    ``children_summary``(阶段4 T5,尾部默认零破坏)是 Session._execute 完成后
    并入的本 turn 子代理状态注记(children 注册表 = child_allocation items
    折叠);无子代理恒空元组。
    """

    turn_id: str
    status: TurnStatus
    output: str
    reason: str | None
    pending_approvals: tuple[ApprovalSummary, ...] = ()
    unknown_calls: tuple[str, ...] = ()
    notices: tuple[PermissionNotice, ...] = ()
    budget_dimension: str | None = None
    children_summary: tuple[ChildSummary, ...] = ()


# ---------------------------------------------------------------------------
# 内部件:item 写入、摘要收集、未知工具登记
# ---------------------------------------------------------------------------


def _exec_ctx(deps: TurnDeps, state: TurnState) -> ExecCtx:
    return ExecCtx(
        store=deps.store,
        bus=deps.bus,
        workspace=deps.workspace,
        tools=deps.tools,
        permissions=deps.permissions,
        turn_id=state.turn_id,
        thread_id=state.thread_id,
        policy_revision=deps.permissions.policy.revision,
        compile_sandbox=deps.compile_sandbox,
        finalize_sandbox=deps.finalize_sandbox,
        artifacts=deps.artifacts,
        hooks=deps.hooks,
        budget=deps.budget,
    )


def _model_call(call: ModelToolCall, window_seq: int | None = None) -> ToolCall:
    """模型路径调用一律携带声明窗口(审批绑定键,Task 14);None 仅限直调域。"""
    return ToolCall(
        tool_call_id=call.call_id, tool_name=call.name, arguments=call.arguments, window_seq=window_seq
    )


def _model_response_entry(resp: ModelResponse) -> ItemEntry:
    """payload 约定(Task 14/recovery 消费,勿漂移):tool_calls 条目为
    {call_id, name, arguments},arguments 缺席即 {}——to_messages 按同一形状重建。
    reasoning 是独立键:思考文本可审计,但投影不回灌(见 context.to_messages)。"""
    calls: tuple[JsonValue, ...] = tuple(
        {"call_id": c.call_id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls
    )
    payload: dict[str, JsonValue] = {
        "content": resp.content,
        "tool_calls": calls,
        # H-1:D2 契约的完整 usage(source/cached/reasoning 不再丢)。
        "usage": usage_payload(resp.usage),
        "finish_reason": resp.finish_reason,
        "reasoning": resp.reasoning,
    }
    return ItemEntry(kind=ItemKind.MODEL_RESPONSE, payload=payload)


def _epoch_value(value: object) -> float | None:
    """JSON 数字 → 有限 epoch;bool 不算数字(JSON true/false 常被误读)。"""
    if type(value) is int:
        return float(value)
    if type(value) is float and math.isfinite(value):
        return value
    return None


def _attempt_records_from_items(items: Sequence[Item]) -> list[AttemptRecord]:
    """从本 turn 的 item 重建尝试记录(畸形 payload fail-open 跳过)。

    调用方传入本 turn 的 item 序列(``item.turn_id == turn_id``);本函数只
    认识 kind,对未知/畸形 payload 一律跳过——读路径不因一条坏记录炸 turn。
    ``model_attempt`` 是 append-only 事实,重试机制与 clock floor 都只读它,
    因此跨进程重启无需第二本调度账(B27 在结构上消亡)。
    """
    records: list[AttemptRecord] = []
    for item in items:
        if item.kind != ItemKind.MODEL_ATTEMPT:
            continue
        payload = item.payload
        attempt = payload.get("attempt")
        kind = payload.get("kind")
        retry_at = payload.get("retry_at")
        dispatched_at = _epoch_value(payload.get("dispatched_at"))
        output_started = payload.get("output_started")
        outcome = payload.get("outcome")
        source = payload.get("retry_at_source")
        if type(attempt) is not int or attempt < 0:
            continue
        if not isinstance(kind, str) or not kind:
            continue
        if not isinstance(outcome, str) or outcome not in ("ok", "failed"):
            continue
        if type(output_started) is not bool or dispatched_at is None:
            continue
        retry_value = None if retry_at is None else _epoch_value(retry_at)
        if retry_at is not None and retry_value is None:
            continue
        if source not in (None, "retry_after", "backoff"):
            continue
        records.append(
            AttemptRecord(
                attempt=attempt,
                kind=kind,
                retry_at=retry_value,
                dispatched_at=dispatched_at,
                output_started=output_started,
                outcome=cast(AttemptOutcome, outcome),
                retry_at_source=cast(RetryAtSource | None, source),
            )
        )
    return records


def _model_attempt_entry(record: AttemptRecord) -> ItemEntry:
    """一条尝试事实的 item 形状(与 _attempt_records_from_items 严格互为逆)。"""
    payload: dict[str, JsonValue] = {
        "attempt": record.attempt,
        "kind": record.kind,
        "retry_at": record.retry_at,
        "dispatched_at": record.dispatched_at,
        "output_started": record.output_started,
        "outcome": record.outcome,
        "retry_at_source": record.retry_at_source,
    }
    return ItemEntry(kind=ItemKind.MODEL_ATTEMPT, payload=payload)


def _persist_attempts(
    deps: TurnDeps, state: TurnState, attempts: Sequence[AttemptRecord], start_index: int
) -> None:
    """把机制新追加的尝试记录写成 model_attempt item,并 emit 重试提示。

    只写 ``start_index`` 之后的新记录:历史记录来自本 turn 已落库的 item,
    重写会产生重复账。写入是纯 INSERT(store 不变量),事件永不阻塞。
    """
    new_records = list(attempts[start_index:])
    if not new_records:
        return
    deps.store.append_items(
        state.thread_id, state.turn_id, [_model_attempt_entry(record) for record in new_records]
    )
    for record in new_records:
        if record.outcome == "failed" and record.retry_at is not None:
            deps.bus.emit(
                ModelRetried(
                    turn_id=state.turn_id,
                    attempt=record.attempt,
                    kind=record.kind,
                    retry_at=record.retry_at,
                )
            )


def _emit_output_deltas(deps: TurnDeps, state: TurnState, events: Sequence[ModelEvent]) -> None:
    """最终成功那次尝试的增量按序 emit 一次;失败尝试的帧从不出现。"""
    for event in events:
        if isinstance(event, ModelTextDelta) and event.text:
            deps.bus.emit(ModelOutputDelta(turn_id=state.turn_id, kind="text", text=event.text))
        elif isinstance(event, ModelReasoningDelta) and event.text:
            deps.bus.emit(
                ModelOutputDelta(turn_id=state.turn_id, kind="reasoning", text=event.text)
            )
        elif isinstance(event, ModelToolCallDelta) and event.arguments_fragment:
            deps.bus.emit(
                ModelOutputDelta(
                    turn_id=state.turn_id, kind="tool_arguments", text=event.arguments_fragment
                )
            )


def _note_unknown(result: ToolResult, unknown_calls: list[str]) -> None:
    """未知工具已由 pipeline 落 error result 并回流给模型;这里只为上层收集展示面。"""
    if result.status is ResultStatus.ERROR and result.error_code == _UNKNOWN_TOOL_ERROR_CODE:
        unknown_calls.append(result.tool_call_id)


def pending_approval_summaries(
    items: Sequence[Item], *, policy_revision: str
) -> tuple[ApprovalSummary, ...]:
    """本 turn 内没有**当前 revision 有效**决策的 approval_request。

    turn 层与 Session 层共用这一个判据(K-2:不再各写一份)。读 item 而非缓存
    内存状态:item 日志是唯一事实源;同 call_id 多条请求取最新一条(re-ask 的
    摘要用新参数 digest)。

    抑制条件(C-1 + K-2 + Task 14):一个请求只有在**同**
    ``(window_seq, call_id, args_digest)`` 且决策的 ``policy_revision`` 等于
    当前 revision 时才算已决策。与管道 ``_bound_decision_valid`` 同口径——
    陈旧 revision/跨窗的批准在重入时作废并 re-ask,若展示面仍按裸 digest
    抑制,就会停在 PAUSED 且 ``pending_approvals == ()``(B32 同族卡死)。
    digest 不同即另一个动作,窗口不同即另一次声明,都必须呈现。legacy
    decision 缺 window_seq 键一律不算已决策(fail-closed,与管道 presence
    判据逐字一致);畸形 window 值同理排除。
    """
    decided: set[tuple[int | None, str, str]] = set()
    for item in items:
        if item.kind != ItemKind.APPROVAL_DECISION:
            continue
        call_id = item.payload.get("call_id")
        digest = item.payload.get("args_digest")
        revision = item.payload.get("policy_revision")
        window = item.payload.get("window_seq")
        if (
            isinstance(call_id, str)
            and isinstance(digest, str)
            and revision == policy_revision
            and "window_seq" in item.payload
            and (window is None or type(window) is int)
        ):
            decided.add((window, call_id, digest))
    latest_requests: dict[str, Mapping[str, JsonValue]] = {}
    for item in items:
        if item.kind != ItemKind.APPROVAL_REQUEST:
            continue
        call_id = item.payload.get("call_id")
        if isinstance(call_id, str):
            latest_requests[call_id] = item.payload
    pending: dict[str, ApprovalSummary] = {}
    for call_id, payload in latest_requests.items():
        tool_name = payload.get("tool_name")
        digest = payload.get("args_digest")
        if not isinstance(tool_name, str) or not isinstance(digest, str):
            continue
        if (payload.get("window_seq"), call_id, digest) in decided:
            continue
        effects: tuple[str, ...] = ()
        raw_effects = payload.get("effects")
        if isinstance(raw_effects, (list, tuple)):
            effects = tuple(e for e in raw_effects if isinstance(e, str))
        pending[call_id] = ApprovalSummary(
            call_id=call_id, tool_name=tool_name, args_digest=digest, effects=effects
        )
    return tuple(pending.values())


def _collect_pending_approvals(deps: TurnDeps, turn_id: str) -> tuple[ApprovalSummary, ...]:
    """turn 层包装:以 deps 的当前策略 revision 判定有效性。"""
    return pending_approval_summaries(
        deps.store.read_turn_items(turn_id),
        policy_revision=deps.permissions.policy.revision,
    )


def _cancel_requested(items: Sequence[Item], turn_id: str) -> bool:
    """cancel_request 收敛判据(T4):日志里已有本 turn 的取消事实。

    与 steering 同缝——item 日志是唯一事实源,任何进程 append 都算数;
    turn_id 严格归属(取消是 per-turn 事实,不是 thread 级广播)。
    """
    return any(item.kind == ItemKind.CANCEL_REQUEST and item.turn_id == turn_id for item in items)


def _converge_cancelled(
    deps: TurnDeps, state: TurnState, output: str, unknown_calls: Sequence[str]
) -> TurnResult:
    """取消收敛(B14/B29):先 compensation 全组清算补 entries,再 transition。

    同窗口多 call_id 第 1 个缺结果不得早退、跨窗口孤儿声明同样补齐——清算
    判据与 resume/complete 守卫共用 compensation 单点。全程同步(store 写经
    写者线程、transition 经 Future 阻塞等待,不是 await),except 路径不上抛。
    unknown_calls = 执行期收集的未知工具 + 清算 outcome_unknown 清单(去序合并)。
    """
    entries, unknown_ids = close_unresolved(
        deps.store.read_turn_items(state.turn_id), deps.tools, text_for="cancelled"
    )
    if entries:
        deps.store.append_items(state.thread_id, state.turn_id, entries)
    transition(deps.store, deps.bus, state.turn_id, TurnStatus.CANCELLED, reason="cancelled_by_user")
    merged = list(unknown_calls)
    for call_id in unknown_ids:
        if call_id not in merged:
            merged.append(call_id)
    return TurnResult(
        turn_id=state.turn_id,
        status=TurnStatus.CANCELLED,
        output=output,
        reason="cancelled_by_user",
        unknown_calls=tuple(merged),
    )


def _budget_paused(
    deps: TurnDeps,
    state: TurnState,
    admission: Admission,
    output: str,
    unknown_calls: Sequence[str],
) -> TurnResult:
    """预算拒绝 → PAUSED(T3):reason 即准入真实 reason,pending_approvals 恒空
    (这不是审批暂停)。budget_dimension 只在 budget_exhausted 时填(R-T3-6)。
    """
    transition(deps.store, deps.bus, state.turn_id, TurnStatus.PAUSED, reason=admission.reason)
    return TurnResult(
        turn_id=state.turn_id,
        status=TurnStatus.PAUSED,
        output=output,
        reason=admission.reason,
        pending_approvals=(),
        unknown_calls=tuple(unknown_calls),
        budget_dimension=admission.dimension if admission.reason == "budget_exhausted" else None,
    )


def _paused_result(
    deps: TurnDeps, state: TurnState, output: str, unknown_calls: Sequence[str]
) -> TurnResult:
    return TurnResult(
        turn_id=state.turn_id,
        status=TurnStatus.PAUSED,
        output=output,
        reason="approval_required",
        pending_approvals=_collect_pending_approvals(deps, state.turn_id),
        unknown_calls=tuple(unknown_calls),
    )


# ---------------------------------------------------------------------------
# 循环本体
# ---------------------------------------------------------------------------


async def run_turn(
    deps: TurnDeps,
    state: TurnState,
    *,
    resume_calls: Sequence[ModelToolCall] = (),
) -> TurnResult:
    """执行(或重入)一个 turn 直到终结。前置约定:回合状态为 RUNNING(会话层保证)。

    结局诸途,全部经状态机唯一入口写入且 reason 真实:
    model_finished / approval_required / budget_exhausted / pricing_unavailable /
    step_limit_exceeded / cancelled_by_user / ``<异常类型>: <消息>``。

    预算双闸(R-T3-1):``deps.budget`` 的树维度耗尽 → PAUSED(约定预算,
    可 resume+extend 续跑);``max_steps`` 硬护栏 → FAILED(失控防御)。
    护栏基数 = fold 声明轮数 + 本次重放轮自增(R-PF-4),跨 pause/resume 累计。
    """
    ctx = _exec_ctx(deps, state)
    unknown_calls: list[str] = []
    last_content = ""
    # R-T3-7:保护尾豁免是**逐次 dispatch** 参数(旧 ensure_token_headroom 的
    # allow_protected 每次现算,从不锁存整 turn)。continuation 旗标一次性:
    # 压缩成功置位,下一次 admit 消费即复位——豁免只及紧随压缩的那次续作。
    compaction_continuation = False
    compaction_attempted = False  # R-P4:每 turn 最多尝试一次(含"无可覆盖区间")
    try:
        # R-PF-4:steps 初值 = fold 声明轮数(重放轮无新 model_response 事实,
        # 由步序 6 的进程内自增补计)——跨 pause/resume 累计自此成立。
        turn_items = [item for item in deps.store.read_items(state.thread_id) if item.turn_id == state.turn_id]
        state.steps = fold_turn_facts(turn_items, unit_costs=None).steps
        # 步序 1:审批重入——先重放已批准的调用,再进模型循环。
        # Task 14:重放调用的声明窗口 = 声明它的 model_response seq,经判据层
        # unclosed_windows 取(不另写窗口匹配);升序输出 last-wins = 最近声明,
        # 与 pair_tool_results 的结果归属同纪律。审批 decision 携带同一窗口,
        # 同窗重入绑定成立(K-2 零回退)。
        declaration_windows: dict[str, int] = {}
        if resume_calls:
            declaration_windows = {
                unresolved.call.call_id: unresolved.declaration_seq
                for unresolved in unclosed_windows(turn_items, deps.tools)
            }
        for call in resume_calls:
            result = await execute(_model_call(call, declaration_windows.get(call.call_id)), ctx)
            if result.status is ResultStatus.PAUSED:
                return _paused_result(deps, state, last_content, unknown_calls)
            _note_unknown(result, unknown_calls)
        while True:
            # 步序 2:每轮从 item 日志重建上下文;steering 在步边界自然进入。
            items = deps.store.read_items(state.thread_id)
            # T4:cancel_request 与 steering 同缝——步边界读到取消事实即收敛
            # (全组清算 + CANCELLED),不再请求模型。跨进程取消在此生效。
            if _cancel_requested(items, state.turn_id):
                return _converge_cancelled(deps, state, last_content, unknown_calls)
            messages = [
                ModelMessage(role="system", content=deps.system_prompt),
                # D11 读侧固定链:先按 boundary 折叠,再省略旧工具输出,最后投影。
                *to_messages(elide_old_tool_outputs(build_context(items))),
            ]
            # R-T7-1:skill 目录注入是本轮消息组装的**唯一** application point。
            # 同一份 enriched messages 既进 build_request,也进下面的压缩判定
            # (measure_request 测的就是真实出站体,不存在阈值盲区)。
            if deps.skills_catalog is not None:
                messages = list(append_skill_catalog(messages, deps.skills_catalog))
            # T12:memory 导航同纪律——append_skill_catalog 之后的唯一注入点。
            if deps.memory_navigation is not None:
                messages = list(append_memory_navigation(messages, deps.memory_navigation))
            definitions = [tool.definition for tool in deps.tools.values()]
            # 步序 3:构建请求 → 补全 → 落 model_response item → emit 预览事件。
            req = deps.adapter.build_request(messages, definitions, ModelSettings())
            # 步边界自动压缩(设计 §5.7 / D11):只在实际出站请求超阈值时触发,
            # 摘要由同一 adapter 完成;下一轮重建上下文。每 turn 最多尝试一次
            # (R-P4:即使压缩后仍超阈值也不重试,避免本地小模型摘要不够短时
            # 的 livelock)。K-1:物化判定与压缩共用 turn 已算出的同一次测量
            # (req 含 system + tools),不再让 maybe_compact 用另一份无 tools
            # 的请求重测;只有真的写了 summary/boundary 才重建上下文。
            compact_limit = deps.compact_limit_tokens
            measured_tokens: int | None = None
            if not compaction_attempted and compact_limit and deps.measure_tokens is not None:
                measured_tokens = deps.measure_tokens(req)
            over_threshold = (
                measured_tokens is not None and compact_limit is not None and measured_tokens > compact_limit
            )
            # 步序 3.5(阶段4 T3):步边界预算闸——先 refresh 后 admit
            # (BudgetTracker 不变量)。树维度耗尽/单价缺失 → PAUSED(约定预算
            # 可补预算续跑,不是 FAILED 失控);pending_approvals 恒空——这不是
            # 审批暂停。保护尾豁免逐次 dispatch(R-T3-7):continuation 旗标
            # 只豁免紧随压缩的那次续作(消费即复位);over_threshold 只预授
            # 本轮摘要调用,压缩未产出摘要时下面按普通调用重新准入。普通调用
            # 只用扣保护尾后的余量(T2 admit_dispatch)。
            needs_compaction = compaction_continuation or over_threshold
            compaction_continuation = False
            if deps.budget is not None:
                deps.budget.refresh()
                admission = deps.budget.admit_dispatch(needs_compaction=needs_compaction)
                if not admission.allowed:
                    return _budget_paused(deps, state, admission, last_content, unknown_calls)
            # compact_limit is not None 是 mypy 类型守卫(over_threshold 已蕴含)。
            if over_threshold and compact_limit is not None:
                compaction_attempted = True
                outcome = await maybe_compact(
                    store=deps.store,
                    adapter=deps.adapter,
                    thread_id=state.thread_id,
                    turn_id=state.turn_id,
                    items=items,
                    limit_tokens=compact_limit,
                    summary_max_output_tokens=deps.summary_max_output_tokens,
                    summary_prompt_max_chars=deps.summary_prompt_max_chars,
                    measure_tokens=deps.measure_tokens,
                    request=req,
                    measured_tokens=measured_tokens,
                )
                if isinstance(outcome, CompactionOutcome) and outcome.compacted:
                    compaction_continuation = True  # 豁免只及下一次 admit(续作)
                    continue
                # R-T3-7 同源边界:压缩未产出摘要(失败/无可覆盖区间)——预授
                # 豁免作废,本轮普通 dispatch 按 needs_compaction=False 重新准入。
                if deps.budget is not None:
                    deps.budget.refresh()
                    admission = deps.budget.admit_dispatch(needs_compaction=False)
                    if not admission.allowed:
                        return _budget_paused(deps, state, admission, last_content, unknown_calls)
            # core 只走流式(D1);complete() 也由同一条 stream 装配派生。
            # 重试机制由 composition 以结构签名注入:core 不认识
            # openai/httpx,只认识"给工厂、已有尝试、clock floor,拿回事件"。
            # 每次适配器调用各记一个派发时刻,成功记录用它作为 dispatched_at。
            dispatch_times: list[float] = []

            def stream_factory(
                req: WireRequest = req,
                dispatch_times: list[float] = dispatch_times,
            ) -> AsyncIterator[ModelEvent]:
                # 默认参数即绑定当轮 req/时间列表(B023:闭包不捕获循环变量)。
                dispatch_times.append(time.time())
                return deps.adapter.stream(req)

            if deps.retry_stream is None:
                # 单测/ScriptedAdapter 路径:不重试,也不产生尝试账(没有机制
                # 就没有第二本账);事件仍按同一形状 emit,UI 面不分支。
                events = await collect_events(stream_factory())
            else:
                turn_items = [item for item in items if item.turn_id == state.turn_id]
                attempts = _attempt_records_from_items(turn_items)
                start_index = len(attempts)
                floor = clock_floor(attempts, time.time())
                try:
                    events = await deps.retry_stream(stream_factory, attempts, floor)
                except Exception:
                    # 最终失败也要落一条 model_attempt(否则失败尝试从账上消失)。
                    _persist_attempts(deps, state, attempts, start_index)
                    raise
                dispatched_at = dispatch_times[-1] if dispatch_times else time.time()
                output_started = any(
                    isinstance(event, (ModelTextDelta, ModelReasoningDelta, ModelToolCallDelta))
                    for event in events
                )
                attempts.append(
                    AttemptRecord(
                        attempt=len(attempts),
                        kind="ok",
                        retry_at=None,
                        dispatched_at=dispatched_at,
                        output_started=output_started,
                        outcome="ok",
                    )
                )
                _persist_attempts(deps, state, attempts, start_index)
            resp = assemble_model_events(events)
            _emit_output_deltas(deps, state, events)
            last_content = resp.content
            # Task 14:返回即所写(B12)——model_response 的 seq 就是本轮工具
            # 调用的声明窗口,审批 request/decision 按它绑定。
            appended = deps.store.append_items(state.thread_id, state.turn_id, [_model_response_entry(resp)])
            window_seq = appended[0].seq
            deps.bus.emit(
                ModelResponseReceived(
                    turn_id=state.turn_id,
                    content_preview=resp.content[:_CONTENT_PREVIEW_CHARS],
                )
            )
            if not resp.tool_calls:
                # 步序 4:模型无工具调用即自然收尾。T4 终态守卫(B14 对称性):
                # 与 cancel 同一清算协议——同缝 cancel 收敛(最后一跳流式期间
                # 到达的取消不被静默丢弃)+ 全组清算遗留未闭合窗口(常态零
                # entries:每轮工具全结算才推进到下一轮请求)。
                turn_items = deps.store.read_turn_items(state.turn_id)
                if _cancel_requested(turn_items, state.turn_id):
                    return _converge_cancelled(deps, state, resp.content, unknown_calls)
                guard_entries, _guard_unknown = close_unresolved(turn_items, deps.tools, text_for="cancelled")
                if guard_entries:
                    deps.store.append_items(state.thread_id, state.turn_id, guard_entries)
                transition(
                    deps.store, deps.bus, state.turn_id, TurnStatus.COMPLETED, reason="model_finished"
                )
                return TurnResult(
                    turn_id=state.turn_id,
                    status=TurnStatus.COMPLETED,
                    output=resp.content,
                    reason="model_finished",
                    unknown_calls=tuple(unknown_calls),
                )
            # 步序 5:唯一工具管道逐个执行;PAUSED 的转换已由 pipeline 完成。
            for call in resp.tool_calls:
                result = await execute(_model_call(call, window_seq), ctx)
                if result.status is ResultStatus.PAUSED:
                    return _paused_result(deps, state, last_content, unknown_calls)
                _note_unknown(result, unknown_calls)
            # 步序 6:一轮完整"请求+全部工具"算一步。
            state.steps += 1
            if state.steps >= deps.max_steps:
                transition(
                    deps.store, deps.bus, state.turn_id, TurnStatus.FAILED, reason="step_limit_exceeded"
                )
                return TurnResult(
                    turn_id=state.turn_id,
                    status=TurnStatus.FAILED,
                    output=last_content,
                    reason="step_limit_exceeded",
                    unknown_calls=tuple(unknown_calls),
                )
    except asyncio.CancelledError:
        # 步序 7(B15/B14):取消是独立结局,且不再"零补偿 item"——先全组清算
        # (B29:同窗口缺结果不早退)再 transition。except 路径内全同步(store 写
        # 经写者线程、transition 经 Future 阻塞等待,不是 await),不上抛——
        # 任务生命周期归会话层,SDK 表面拿结果。
        return _converge_cancelled(deps, state, "", unknown_calls)
    except Exception as exc:  # noqa: BLE001 — turn 级错误边界:全收敛,永不炸穿会话
        # 步序 8(B11):真实异常类型与消息进入 reason;ScriptExhausted 等也走此路。
        reason = f"{type(exc).__name__}: {exc}"
        transition(deps.store, deps.bus, state.turn_id, TurnStatus.FAILED, reason=reason)
        return TurnResult(
            turn_id=state.turn_id,
            status=TurnStatus.FAILED,
            output="",
            reason=reason,
            unknown_calls=tuple(unknown_calls),
        )

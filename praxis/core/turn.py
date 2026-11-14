"""Praxis 最小 turn 循环(设计 §5.4/§5.12;findings B15/B11;验收"多工具 turn 跑通")。

``TurnResult.status`` 就是 ``TurnStatus`` 本身——没有映射层(结构性 B15:取消
是独立结局,不会被折叠成失败;失败永远携带真实 reason,不会退化成笼统标签)。
每 turn 一个错误边界:循环内任何异常都收敛为本 turn 的 FAILED,永不炸穿会话。

步骤序(与代码一一对应):
1. ``resume_calls`` 非空 → 先逐个 ``pipeline.execute``(审批重入路径;调用方
   保证回合已 RUNNING——Task 14 的 resume 负责先转换再重入),PAUSED 同步序 5;
2. 每轮重读 thread items → ``build_context`` 折叠 → ``to_messages`` 投影 →
   前置 system 消息;steering item 在步边界自然进入上下文(它已在日志里);
3. ``adapter.build_request`` + 注入的 ``retry_stream``(缺省不重试)收齐
   ``ModelEvent`` → ``assemble_model_events`` → 每次尝试落 ``model_attempt``
   item、最终成功落 model_response item → emit ``ModelOutputDelta`` 与
   ``ModelResponseReceived``(预览 ≤200 字符);core 只走流式,complete 只是
   同一路径的薄门面(设计 D1)。重试机制由 composition 以结构签名注入,
   core 不认识 openai/httpx(R-T3-1);重试期间不 emit 增量,失败首帧不会
   变成回滚不掉的重影。
4. 无 tool_calls → ``transition(COMPLETED, "model_finished")``;
5. 有 tool_calls → 逐个 ``execute``;任一 PAUSED → 管道已完成 RUNNING→PAUSED
   转换,直接返回并附未批摘要;
6. ``steps += 1``;≥ max_steps → ``transition(FAILED, "step_limit_exceeded")``;
7. ``CancelledError`` → ``transition(CANCELLED, "cancelled_by_user")``——except
   路径内只有同步调用,不 await、不上抛:SDK 表面返回结果,任务生命周期归
   会话层;
8. ``Exception`` → ``transition(FAILED, "<类型>: <消息>")``。

item 分工:pipeline 写 tool_call/tool_result/approval_request,本模块写
model_response/model_attempt,状态机写 turn_status——没有第二写入者,不存在双写。
核心层对总线与订阅者只 emit、永不 await(硬规则 4)。
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from praxis.core.context import build_context, to_messages
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, SandboxCompiler, SandboxFinalizer, execute
from praxis.core.statemachine import TurnStatus, transition
from praxis.protocol.events import ModelOutputDelta, ModelResponseReceived, ModelRetried
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
)
from praxis.protocol.permission import PermissionContext
from praxis.protocol.retry import (
    AttemptOutcome,
    AttemptRecord,
    RetryAtSource,
    RetryingStream,
    clock_floor,
)
from praxis.protocol.tools import ResultStatus, Tool, ToolCall, ToolResult
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
    # None = 不重试(ScriptedAdapter/单测路径);生产 composition 注入真实机制。
    # 只有注入时才有 model_attempt 事实账——无重试机制就没有第二个账本。
    retry_stream: RetryingStream | None = None


@dataclass
class TurnState:
    """显式可变循环状态:steps 由 run_turn 每完成一轮工具步 +1。"""

    turn_id: str
    thread_id: str
    steps: int = 0


# ---------------------------------------------------------------------------
# 结果形状(无映射层,B15)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalSummary:
    """一条待批调用的人类可读摘要(源自 approval_request item,非第二事实源)。"""

    call_id: str
    tool_name: str
    args_digest: str
    effects: tuple[str, ...]


@dataclass(frozen=True)
class TurnResult:
    """run_turn 的终结结果:status 即 TurnStatus,reason 即状态机的 reason。"""

    turn_id: str
    status: TurnStatus
    output: str
    reason: str | None
    pending_approvals: tuple[ApprovalSummary, ...] = ()
    unknown_calls: tuple[str, ...] = ()


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
    )


def _model_call(call: ModelToolCall) -> ToolCall:
    return ToolCall(tool_call_id=call.call_id, tool_name=call.name, arguments=call.arguments)


def _model_response_entry(resp: ModelResponse) -> ItemEntry:
    """payload 约定(Task 14/recovery 消费,勿漂移):tool_calls 条目为
    {call_id, name, arguments},arguments 缺席即 {}——to_messages 按同一形状重建。
    reasoning 是独立键:思考文本可审计,但投影不回灌(见 context.to_messages)。"""
    calls: tuple[JsonValue, ...] = tuple(
        {"call_id": c.call_id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls
    )
    usage: dict[str, JsonValue] = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    payload: dict[str, JsonValue] = {
        "content": resp.content,
        "tool_calls": calls,
        "usage": usage,
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


def _collect_pending_approvals(deps: TurnDeps, turn_id: str) -> tuple[ApprovalSummary, ...]:
    """本 turn 内没有 matching approval_decision(按 call_id)的 approval_request。

    读 item 而非缓存内存状态:item 日志是唯一事实源;同 call_id 多条请求取最新
    一条(re-ask 的摘要用新参数 digest)。
    """
    items = deps.store.read_turn_items(turn_id)
    decided: set[str] = set()
    for item in items:
        if item.kind == ItemKind.APPROVAL_DECISION:
            call_id = item.payload.get("call_id")
            if isinstance(call_id, str):
                decided.add(call_id)
    pending: dict[str, ApprovalSummary] = {}
    for item in items:
        if item.kind != ItemKind.APPROVAL_REQUEST:
            continue
        payload = item.payload
        call_id = payload.get("call_id")
        tool_name = payload.get("tool_name")
        digest = payload.get("args_digest")
        if not isinstance(call_id, str) or not isinstance(tool_name, str) or not isinstance(digest, str):
            continue
        if call_id in decided:
            continue
        effects: tuple[str, ...] = ()
        raw_effects = payload.get("effects")
        if isinstance(raw_effects, tuple):
            effects = tuple(e for e in raw_effects if isinstance(e, str))
        pending[call_id] = ApprovalSummary(
            call_id=call_id, tool_name=tool_name, args_digest=digest, effects=effects
        )
    return tuple(pending.values())


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

    结局五途,全部经状态机唯一入口写入且 reason 真实:
    model_finished / approval_required / step_limit_exceeded /
    cancelled_by_user / ``<异常类型>: <消息>``。
    """
    ctx = _exec_ctx(deps, state)
    unknown_calls: list[str] = []
    last_content = ""
    try:
        # 步序 1:审批重入——先重放已批准的调用,再进模型循环。
        for call in resume_calls:
            result = await execute(_model_call(call), ctx)
            if result.status is ResultStatus.PAUSED:
                return _paused_result(deps, state, last_content, unknown_calls)
            _note_unknown(result, unknown_calls)
        while True:
            # 步序 2:每轮从 item 日志重建上下文;steering 在步边界自然进入。
            items = deps.store.read_items(state.thread_id)
            messages = [
                ModelMessage(role="system", content=deps.system_prompt),
                *to_messages(build_context(items)),
            ]
            definitions = [tool.definition for tool in deps.tools.values()]
            # 步序 3:构建请求 → 补全 → 落 model_response item → emit 预览事件。
            req = deps.adapter.build_request(messages, definitions, ModelSettings())
            # core 只走流式(D1);complete() 也由同一条 stream 装配派生。
            # 重试机制由 composition 以结构签名注入(R-T3-1):core 不认识
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
            deps.store.append_items(state.thread_id, state.turn_id, [_model_response_entry(resp)])
            deps.bus.emit(
                ModelResponseReceived(
                    turn_id=state.turn_id,
                    content_preview=resp.content[:_CONTENT_PREVIEW_CHARS],
                )
            )
            if not resp.tool_calls:
                # 步序 4:模型无工具调用即自然收尾。
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
                result = await execute(_model_call(call), ctx)
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
        # 步序 7(B15):取消是独立结局。except 路径内全同步(transition 经写者线程
        # Future 阻塞等待,不是 await),不上抛——任务生命周期归会话层,SDK 表面拿结果。
        transition(
            deps.store, deps.bus, state.turn_id, TurnStatus.CANCELLED, reason="cancelled_by_user"
        )
        return TurnResult(
            turn_id=state.turn_id,
            status=TurnStatus.CANCELLED,
            output="",
            reason="cancelled_by_user",
            unknown_calls=tuple(unknown_calls),
        )
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

"""Praxis 最小 turn 循环(设计 §5.4/§5.12;findings B15/B11;验收"多工具 turn 跑通")。

``TurnResult.status`` 就是 ``TurnStatus`` 本身——没有映射层(结构性 B15:取消
是独立结局,不会被折叠成失败;失败永远携带真实 reason,不会退化成笼统标签)。
每 turn 一个错误边界:循环内任何异常都收敛为本 turn 的 FAILED,永不炸穿会话。

步骤序(与代码一一对应):
1. ``resume_calls`` 非空 → 先逐个 ``pipeline.execute``(审批重入路径;调用方
   保证回合已 RUNNING——Task 14 的 resume 负责先转换再重入),PAUSED 同步序 5;
2. 每轮重读 thread items → ``build_context`` 折叠 → ``to_messages`` 投影 →
   前置 system 消息;steering item 在步边界自然进入上下文(它已在日志里);
3. ``adapter.build_request`` + ``await adapter.complete`` → 落 model_response
   item → emit ``ModelResponseReceived``(预览 ≤200 字符);
4. 无 tool_calls → ``transition(COMPLETED, "model_finished")``;
5. 有 tool_calls → 逐个 ``execute``;任一 PAUSED → 管道已完成 RUNNING→PAUSED
   转换,直接返回并附未批摘要;
6. ``steps += 1``;≥ max_steps → ``transition(FAILED, "step_limit_exceeded")``;
7. ``CancelledError`` → ``transition(CANCELLED, "cancelled_by_user")``——except
   路径内只有同步调用,不 await、不上抛:SDK 表面返回结果,任务生命周期归
   会话层;
8. ``Exception`` → ``transition(FAILED, "<类型>: <消息>")``。

item 分工:pipeline 写 tool_call/tool_result/approval_request,本模块写
model_response,状态机写 turn_status——没有第二写入者,不存在双写。
核心层对总线与订阅者只 emit、永不 await(硬规则 4)。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from praxis.core.context import build_context, to_messages
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, SandboxCompiler, SandboxFinalizer, execute
from praxis.core.statemachine import TurnStatus, transition
from praxis.protocol.events import ModelResponseReceived
from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import (
    ModelAdapter,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelToolCall,
)
from praxis.protocol.permission import PermissionContext
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
    {call_id, name, arguments},arguments 缺席即 {}——to_messages 按同一形状重建。"""
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
    }
    return ItemEntry(kind=ItemKind.MODEL_RESPONSE, payload=payload)


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
            resp = await deps.adapter.complete(req)
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

"""Task 13:最小 turn 循环 —— 取消/失败语义保真(设计 §5.4/§5.12;findings B15/B11)。

仓库无 pytest-asyncio:与 test_pipeline.py 同风格,asyncio.run 包装。
ScriptedAdapter 驱动真实循环;剧本耗尽(ScriptExhausted)天然走异常边界,
这正是错误边界真实存在的证明。B15 回归 = CANCELLED 是独立结局而非 FAILED;
B11 回归 = FAILED 的 reason 携带真实异常文本而非笼统标签。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.core import permission
from praxis.core.events import Subscription
from praxis.core.statemachine import TurnStatus, transition
from praxis.core.turn import TurnDeps, TurnState, run_turn
from praxis.protocol.events import Event, ModelResponseReceived, TurnStatusChanged
from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage, WireRequest
from praxis.protocol.permission import Decision, PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
    args_digest,
    json_schema_input,
)
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn, get_turn
from praxis.tools.builtins.filesystem import create_read_tool, create_write_tool

THREAD = "thread-1"
TURN = "turn-1"
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def _turn(store: Store) -> None:
    """running 起步的真实 turn:create_turn 的 INSERT 即 running,无需 transition。"""
    store.ensure_thread(THREAD)
    create_turn(store, turn_id=TURN, thread_id=THREAD, binding={})


def _resp(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=Usage(input_tokens=3, output_tokens=5),
        finish_reason="tool_calls" if calls else "stop",
    )


def _deps(
    store: Store,
    bus: Any,
    workspace: Path,
    adapter: Any,
    tools: Mapping[str, Tool],
    *,
    max_steps: int = 50,
) -> TurnDeps:
    policy = PermissionPolicy()
    return TurnDeps(
        store=store,
        bus=bus,
        adapter=adapter,
        tools=tools,
        permissions=PermissionContext(workspace=str(workspace), policy=policy, mode="on-request"),
        workspace=workspace,
        system_prompt="test-agent system prompt",
        max_steps=max_steps,
    )


def _toy_tool(
    name: str,
    *,
    on_run: Callable[[], None] | None = None,
    hang: asyncio.Event | None = None,
) -> Tool:
    validate = json_schema_input(_EMPTY_SCHEMA)

    async def run(arguments: Mapping[str, JsonValue]) -> ToolResult:
        if on_run is not None:
            on_run()
        if hang is not None:
            hang.set()
            await asyncio.sleep(30)
        return ToolResult(
            tool_call_id="pending",
            tool_name=name,
            content=(ToolContentBlock(type="text", data={"text": f"{name}:ok"}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(name=name, description=f"{name} toy", input_schema=_EMPTY_SCHEMA),
        validate_input=validate,
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(effects=frozenset(), targets=()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=4096,
    )


def _drain(sub: Subscription) -> list[Event]:
    events: list[Event] = []
    while True:
        try:
            events.append(sub.queue.get_nowait())
        except asyncio.QueueEmpty:
            return events


# ---------------------------------------------------------------------------
# brief Step 1:six sketches
# ---------------------------------------------------------------------------


def test_multitool_turn_completes(store: Store, bus: Any, workspace: Path) -> None:
    """resp1 调 write+read 两个工具,resp2 纯文本收尾:B12 管道写 item、循环不重复。"""
    _turn(store)
    store.append_items(
        THREAD, TURN, [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "write then read"})]
    )
    adapter = ScriptedAdapter(
        [
            _resp("", (("c1", "write", {"path": "note.txt", "content": "hello multitool"}),
                       ("c2", "read", {"path": "note.txt"}))),
            _resp("完成"),
        ]
    )
    deps = _deps(
        store,
        bus,
        workspace,
        adapter,
        {"write": create_write_tool(workspace), "read": create_read_tool(workspace)},
    )

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.COMPLETED and result.output == "完成"
    assert result.turn_id == TURN and result.reason == "model_finished"
    assert result.pending_approvals == () and result.unknown_calls == ()
    kinds = [i.kind for i in store.read_items(THREAD)]
    assert kinds.count(ItemKind.TOOL_CALL) == 2 and kinds.count(ItemKind.TOOL_RESULT) == 2
    assert kinds.count(ItemKind.MODEL_RESPONSE) == 2
    assert (workspace / "note.txt").read_text() == "hello multitool"
    turn = get_turn(store, TURN)
    assert turn is not None and turn.status == "completed"


def test_cancelled_is_not_failed(store: Store, bus: Any, workspace: Path) -> None:
    """工具内挂起 → cancel run_turn 所在 task:B15 回归,CANCELLED 不是 FAILED 的别名。"""
    _turn(store)
    started = asyncio.Event()
    adapter = ScriptedAdapter([_resp("", (("c1", "hang", {}),)), _resp("不可达")])
    deps = _deps(store, bus, workspace, adapter, {"hang": _toy_tool("hang", hang=started)})

    async def scenario() -> Any:
        task = asyncio.create_task(run_turn(deps, TurnState(TURN, THREAD)))
        await started.wait()
        await asyncio.sleep(0.01)  # 让循环确认挂进 execute 内部
        task.cancel()
        return await task

    result = asyncio.run(scenario())

    assert result.status is TurnStatus.CANCELLED  # B15 回归:status 就是 TurnStatus,无映射层
    assert result.reason == "cancelled_by_user"
    turn = get_turn(store, TURN)
    assert turn is not None and turn.status == "cancelled"
    assert turn.status != "failed"


class _BrokenAdapter(ScriptedAdapter):
    """complete 永远抛 provider 异常:真实 provider 故障的替身。"""

    async def complete(self, req: WireRequest) -> ModelResponse:
        raise RuntimeError("provider 503")


def test_model_error_fails_with_real_reason(store: Store, bus: Any, workspace: Path) -> None:
    """B11 回归:异常类型与消息进入 TurnResult.reason 与 TurnStatusChanged.reason。"""
    _turn(store)
    sub = bus.subscribe()
    adapter = _BrokenAdapter([])
    deps = _deps(store, bus, workspace, adapter, {"noop": _toy_tool("noop")})

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.FAILED
    assert result.output == ""
    assert result.reason is not None and result.reason.startswith("RuntimeError:")
    assert "provider 503" in result.reason  # B11 回归:真实 reason,不是笼统标签
    events = _drain(sub)
    failed = [e for e in events if isinstance(e, TurnStatusChanged) and e.to_status == "failed"]
    assert len(failed) == 1
    assert "provider 503" in failed[0].reason
    turn = get_turn(store, TURN)
    assert turn is not None and turn.status == "failed"


def test_step_limit_fails_turn(store: Store, bus: Any, workspace: Path) -> None:
    """max_steps=2,每轮都调工具 → 第三轮请求前以 step_limit_exceeded 终止。"""
    _turn(store)
    adapter = ScriptedAdapter(
        [_resp("", (("c1", "noop", {}),)), _resp("", (("c2", "noop", {}),))]
    )
    deps = _deps(store, bus, workspace, adapter, {"noop": _toy_tool("noop")}, max_steps=2)

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.FAILED
    assert result.reason == "step_limit_exceeded"
    turn = get_turn(store, TURN)
    assert turn is not None and turn.status == "failed"
    # 恰好消耗两轮模型请求:若循环越界会继续请求并 ScriptExhausted
    assert adapter.last_request is not None


def test_steering_enters_at_step_boundary(store: Store, bus: Any, workspace: Path) -> None:
    """第一轮工具执行期间 append steering → 第二轮 build_request 的 messages 含该文本。"""

    def inject() -> None:
        store.append_items(
            THREAD,
            TURN,
            [ItemEntry(kind=ItemKind.STEERING, payload={"text": "改用更短的总结"})],
        )

    _turn(store)
    adapter = ScriptedAdapter([_resp("", (("c1", "inject", {}),)), _resp("完成")])
    deps = _deps(store, bus, workspace, adapter, {"inject": _toy_tool("inject", on_run=inject)})

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.COMPLETED
    assert adapter.last_request is not None  # 即第二次 build_request
    messages = adapter.last_request.messages
    assert messages[0].role == "system" and messages[0].content == deps.system_prompt
    assert any(m.role == "user" and m.content == "改用更短的总结" for m in messages)


def test_slow_subscriber_does_not_block_turn(store: Store, bus: Any, workspace: Path) -> None:
    """订阅者永不消费、队列灌满 → emit 走 QueueFull 丢弃路径,turn 照常 COMPLETED(硬规则 4 动态面)。"""
    _turn(store)
    adapter = ScriptedAdapter([_resp("完成")])
    deps = _deps(store, bus, workspace, adapter, {"noop": _toy_tool("noop")})

    async def scenario() -> Any:
        sub = bus.subscribe()
        while True:
            try:
                sub.queue.put_nowait(ModelResponseReceived(turn_id="filler", content_preview="x"))
            except asyncio.QueueFull:
                break
        return await run_turn(deps, TurnState(TURN, THREAD))

    result = asyncio.run(scenario())

    assert result.status is TurnStatus.COMPLETED
    assert bus.dropped_total >= 1


# ---------------------------------------------------------------------------
# PAUSED / resume_calls:审批重入路径(T14 会话层依赖的循环面)
# ---------------------------------------------------------------------------


def test_ask_pauses_with_pending_approvals(
    store: Store, bus: Any, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    _turn(store)
    args = {"path": "note.txt", "content": "x"}
    adapter = ScriptedAdapter([_resp("", (("c1", "write", args),)), _resp("不可达")])
    deps = _deps(store, bus, workspace, adapter, {"write": create_write_tool(workspace)})

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.PAUSED  # transition 由管道完成,循环不重复
    assert result.reason == "approval_required"
    assert len(result.pending_approvals) == 1
    summary = result.pending_approvals[0]
    assert summary.call_id == "c1" and summary.tool_name == "write"
    assert summary.args_digest == args_digest(args)
    assert "write_workspace" in summary.effects
    assert not (workspace / "note.txt").exists()  # ask 不得执行
    turn = get_turn(store, TURN)
    assert turn is not None and turn.status == "paused"


def test_resume_calls_execute_then_complete(
    store: Store, bus: Any, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """批准落笔 + RUNNING 后走 resume_calls:重放执行恰好一次,然后进模型循环收尾。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    _turn(store)
    args = {"path": "note.txt", "content": "hello resume"}
    adapter = ScriptedAdapter([_resp("", (("c1", "write", args),)), _resp("完成")])
    deps = _deps(store, bus, workspace, adapter, {"write": create_write_tool(workspace)})
    state = TurnState(TURN, THREAD)

    paused = asyncio.run(run_turn(deps, state))
    assert paused.status is TurnStatus.PAUSED

    # Task 14 语义:approval_decision item + 回合回 RUNNING,然后带 resume_calls 重入
    store.append_items(
        THREAD,
        TURN,
        [
            ItemEntry(
                kind=ItemKind.APPROVAL_DECISION,
                payload={
                    "call_id": "c1",
                    "verdict": "allow",
                    "args_digest": args_digest(args),
                    "destination": "session",
                },
            )
        ],
    )
    transition(store, bus, TURN, TurnStatus.RUNNING, reason="resumed")

    resumed = asyncio.run(
        run_turn(deps, state, resume_calls=[ModelToolCall(call_id="c1", name="write", arguments=args)])
    )

    assert resumed.status is TurnStatus.COMPLETED and resumed.output == "完成"
    assert resumed.pending_approvals == ()  # 已有 decision 的 call 不再计入待批
    assert (workspace / "note.txt").read_text() == "hello resume"
    kinds = [i.kind for i in store.read_turn_items(TURN)]
    assert kinds.count(ItemKind.TOOL_CALL) == 1  # 重放只执行一次
    assert kinds.count(ItemKind.TOOL_RESULT) == 1
    assert kinds.count(ItemKind.APPROVAL_REQUEST) == 1
    assert kinds.count(ItemKind.MODEL_RESPONSE) == 2


def test_unknown_tool_calls_surface_in_result(store: Store, bus: Any, workspace: Path) -> None:
    """未知工具不炸循环:pipeline 已落 error result,循环把 call id 收进 unknown_calls 供上层展示。"""
    _turn(store)
    adapter = ScriptedAdapter([_resp("", (("c9", "nope", {}),)), _resp("已纠正")])
    deps = _deps(store, bus, workspace, adapter, {"noop": _toy_tool("noop")})

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.COMPLETED
    assert result.unknown_calls == ("c9",)
    items = store.read_turn_items(TURN)
    assert [i.kind for i in items].count(ItemKind.TOOL_CALL) == 0  # 未知工具不落 tool_call
    err = [i for i in items if i.kind == ItemKind.TOOL_RESULT][0]
    assert err.payload["error_code"] == "unknown_tool"

"""Task 4:cancel 面——request_cancel/步边界收敛/终态清算对称(B14/B29)。

旧缺陷"取消零补偿 item"关闭:CancelledError 分支先经 compensation 全组清算
再 transition(CANCELLED);complete(model_finished)终态守卫走同一协议(对称)。
跨进程取消 = 第二 Store/Session 实例 append cancel_request item,执行进程步
边界读到收敛(同库双实例模拟);本进程快速路径 = _active_task 注入取消。
cancelled 是终态:TRANSITIONS 零改动,再 resume 被拒。

仓库无 pytest-asyncio:与 test_turn.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from praxis.core import permission
from praxis.core.compensation import NOT_EXECUTED_TEXTS, OUTCOME_UNKNOWN_TEXTS
from praxis.core.events import EventBus
from praxis.core.locks import thread_execution_lock
from praxis.core.recovery import sweep
from praxis.core.statemachine import TRANSITIONS, TurnStatus
from praxis.core.turn import TurnDeps, TurnState, run_turn
from praxis.interfaces import sdk as sdk_pkg
from praxis.interfaces.cli import app
from praxis.interfaces.sdk import (
    ConversationBusy,
    ScriptedAdapter,
    open_session,
)
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import Decision, PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
)
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn

runner = CliRunner()

_CLI_ENV = {
    "PRAXIS_MODEL_ID": None,
    "PRAXIS_MODEL": None,
    "PRAXIS_BASE_URL": None,
    "PRAXIS_API_KEY": None,
    "OPENAI_API_KEY": None,
}


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def _resp(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=Usage(input_tokens=3, output_tokens=5),
        finish_reason="tool_calls" if calls else "stop",
    )


def _toy(
    name: str,
    *,
    idempotent: bool = True,
    on_run: Callable[[], None] | None = None,
    hang: asyncio.Event | None = None,
    wait: asyncio.Event | None = None,
) -> Tool:
    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        if on_run is not None:
            on_run()
        if hang is not None:
            hang.set()
            await asyncio.sleep(30)
        if wait is not None:
            await wait.wait()
        return ToolResult(
            tool_call_id="pending",
            tool_name=name,
            content=(ToolContentBlock(type="text", data={"text": f"{name}:ok"}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(name=name, description=f"{name} toy", input_schema={"type": "object"}),
        validate_input=lambda arguments: arguments,
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(effects=frozenset(), targets=()),
        idempotent=idempotent,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=4096,
    )


@pytest.fixture
def open_ws(workspace: Path) -> Iterator[Callable[..., Any]]:
    made: list[Any] = []

    def _open(adapter: Any, **kwargs: Any) -> Any:
        session = open_session(workspace, adapter, **kwargs)
        made.append(session)
        return session

    yield _open
    for session in reversed(made):
        session.close()


def _with_other_store(workspace: Path, fn: Callable[[Store], Any]) -> Any:
    other = Store(workspace / ".praxis" / "praxis.sqlite")
    try:
        return fn(other)
    finally:
        other.close()


def _dead_turn(workspace: Path, thread_id: str, entries: Sequence[ItemEntry]) -> str:
    def _make(other: Store) -> str:
        turn_id = f"dead-{uuid4().hex}"
        create_turn(other, turn_id=turn_id, thread_id=thread_id, binding={})
        if entries:
            other.append_items(thread_id, turn_id, list(entries))
        assert turn_id in sweep(other, EventBus(), workspace)
        return turn_id

    return _with_other_store(workspace, _make)


def _response_entry(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ItemEntry:
    return ItemEntry(
        kind=ItemKind.MODEL_RESPONSE,
        payload={
            "content": content,
            "tool_calls": tuple({"call_id": c, "name": n, "arguments": a} for c, n, a in calls),
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "finish_reason": "tool_calls" if calls else "stop",
        },
    )


def _tool_call_entry(call_id: str, name: str, args: dict[str, Any]) -> ItemEntry:
    return ItemEntry(
        kind=ItemKind.TOOL_CALL,
        payload={"call_id": call_id, "tool_name": name, "arguments": args, "idempotent": False},
    )


def _results(workspace: Path, turn_id: str) -> list[Mapping[str, Any]]:
    return _with_other_store(
        workspace,
        lambda other: [i.payload for i in other.read_turn_items(turn_id) if i.kind == ItemKind.TOOL_RESULT],
    )


def _kinds(workspace: Path, turn_id: str) -> list[str]:
    return _with_other_store(workspace, lambda other: [i.kind for i in other.read_turn_items(turn_id)])


# ---------------------------------------------------------------------------
# brief ②:started 无结果 → outcome_unknown + CANCELLED(旧"取消零补偿"关闭)
# ---------------------------------------------------------------------------


def test_request_cancel_signalled_settles_started_call(workspace: Path, open_ws: Callable[..., Any]) -> None:
    started = asyncio.Event()
    session = open_ws(
        ScriptedAdapter([_resp("", (("c1", "hang", {}),)), _resp("不可达")]),
        extra_tools=[_toy("hang", idempotent=False, hang=started)],
    )

    async def scenario() -> Any:
        task = asyncio.create_task(session.submit("go"))
        await started.wait()
        await asyncio.sleep(0.01)  # 让循环挂进 execute 内部
        turn_id = session.turns()[-1].turn_id
        reason = session.request_cancel(turn_id)
        assert reason == "signalled"  # 本进程 _active_task 已注入取消
        return await task

    result = asyncio.run(scenario())

    assert result.status is TurnStatus.CANCELLED
    assert result.reason == "cancelled_by_user"
    assert "c1" in result.unknown_calls
    assert session.status(result.turn_id).status == "cancelled"
    results = _results(workspace, result.turn_id)
    assert len(results) == 1
    assert results[0]["call_id"] == "c1" and results[0]["status"] == "outcome_unknown"
    assert results[0]["content"][0]["data"]["text"] == OUTCOME_UNKNOWN_TEXTS["cancelled"]
    kinds = _kinds(workspace, result.turn_id)
    assert kinds.count(ItemKind.TOOL_CALL) == 1
    assert kinds.count(ItemKind.MODEL_RESPONSE) == 1  # 剧本第二条从未消费


# ---------------------------------------------------------------------------
# brief ①(B29 集成):同窗口 3 调用,第 1 有结果第 2/3 无 → 整组补齐各判正确
# ---------------------------------------------------------------------------


def test_b29_cancel_settles_whole_group(workspace: Path, open_ws: Callable[..., Any]) -> None:
    started = asyncio.Event()
    noop_runs: list[int] = []
    session = open_ws(
        ScriptedAdapter(
            [
                _resp("", (("c1", "fast", {}), ("c2", "hang", {}), ("c3", "noop", {}))),
                _resp("不可达"),
            ]
        ),
        extra_tools=[
            _toy("fast"),
            _toy("hang", idempotent=False, hang=started),
            _toy("noop", on_run=lambda: noop_runs.append(1)),
        ],
    )

    async def scenario() -> Any:
        task = asyncio.create_task(session.submit("go"))
        await started.wait()
        await asyncio.sleep(0.01)
        turn_id = session.turns()[-1].turn_id
        assert session.request_cancel(turn_id) == "signalled"
        return await task

    result = asyncio.run(scenario())

    assert result.status is TurnStatus.CANCELLED
    results = _results(workspace, result.turn_id)
    by_id = {r["call_id"]: r for r in results}
    assert len(results) == 3  # 整组清算:第 1 个缺结果(c2)不得早退吞掉 c3
    assert by_id["c1"]["status"] == "ok"  # 已有真实结果,不覆盖
    assert by_id["c2"]["status"] == "outcome_unknown"  # started → 未知副作用
    assert by_id["c3"]["status"] == "not_executed"  # 从未 dispatch
    assert by_id["c3"]["content"][0]["data"]["text"] == NOT_EXECUTED_TEXTS["cancelled"]
    assert noop_runs == []  # c3 绝不因清算而执行
    assert set(result.unknown_calls) >= {"c2"}


# ---------------------------------------------------------------------------
# brief ③:跨进程——第二 Store/Session append item,执行进程步边界收敛
# ---------------------------------------------------------------------------


def test_cross_process_cancel_converges_at_step_boundary(workspace: Path, open_ws: Callable[..., Any]) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    noop_runs: list[int] = []
    executor = open_ws(
        ScriptedAdapter([_resp("", (("c1", "wait", {}), ("c2", "noop", {}))), _resp("不可达")]),
        extra_tools=[
            _toy("wait", wait=release, on_run=started.set),
            _toy("noop", on_run=lambda: noop_runs.append(1)),
        ],
    )

    async def scenario() -> Any:
        task = asyncio.create_task(executor.submit("go"))
        await started.wait()
        turn_id = executor.turns()[-1].turn_id
        # "第二进程":独立 Store 实例同 db;open_session 的 sweep 试锁_busy 跳过。
        other = open_session(workspace, ScriptedAdapter([]), thread_id=executor.thread_id)
        try:
            reason = other.request_cancel(turn_id)
        finally:
            other.close()
        assert reason == "queued"  # 他进程执行中:仅落 item,无本进程 task 可注入
        release.set()  # c1 自然完成 → c2 dispatch 前检查点拒 → 步边界收敛
        return await task

    result = asyncio.run(scenario())

    assert result.status is TurnStatus.CANCELLED
    assert result.reason == "cancelled_by_user"
    results = _results(workspace, result.turn_id)
    by_id = {r["call_id"]: r for r in results}
    assert by_id["c1"]["status"] == "ok"
    assert by_id["c2"]["status"] == "not_executed"
    assert noop_runs == []  # 检查点拒在 dispatch 之前:noop 从未执行
    kinds = _kinds(workspace, result.turn_id)
    assert kinds.count(ItemKind.TOOL_CALL) == 1  # c2 没有 tool_call(未 dispatch)

    def _cancel_items(other: Store) -> list[Item]:
        return [i for i in other.read_turn_items(result.turn_id) if i.kind == ItemKind.CANCEL_REQUEST]

    cancels = _with_other_store(workspace, _cancel_items)
    assert len(cancels) == 1
    assert cancels[0].payload["source"] == "user"
    assert cancels[0].turn_id == result.turn_id


# ---------------------------------------------------------------------------
# brief ⑤:cancelled 终态 + TRANSITIONS 零改动;queued cancel 经 resume 收敛
# ---------------------------------------------------------------------------


def test_transitions_table_unchanged() -> None:
    assert TRANSITIONS == {
        TurnStatus.RUNNING: frozenset(
            {TurnStatus.PAUSED, TurnStatus.COMPLETED, TurnStatus.FAILED, TurnStatus.CANCELLED, TurnStatus.INTERRUPTED}
        ),
        TurnStatus.PAUSED: frozenset({TurnStatus.RUNNING, TurnStatus.CANCELLED, TurnStatus.INTERRUPTED}),
        TurnStatus.INTERRUPTED: frozenset({TurnStatus.RUNNING, TurnStatus.CANCELLED}),
        TurnStatus.COMPLETED: frozenset(),
        TurnStatus.FAILED: frozenset(),
        TurnStatus.CANCELLED: frozenset(),
    }


def test_queued_cancel_on_interrupted_turn_converges_on_resume(
    workspace: Path, open_ws: Callable[..., Any]
) -> None:
    session = open_ws(ScriptedAdapter([]))  # 若被消费即 ScriptExhausted → FAILED
    call = ("c1", "bash", {"command": "true"})
    turn_id = _dead_turn(workspace, session.thread_id, [_response_entry("running bash", (call,))])

    assert session.request_cancel(turn_id) == "queued"
    assert session.status(turn_id).status == "interrupted"  # 仅落 item,不改状态

    result = asyncio.run(session.resume(turn_id, confirm=True))
    assert result.status is TurnStatus.CANCELLED  # 步边界读到 cancel_request,模型从未被请求
    assert result.reason == "cancelled_by_user"
    assert session.status(turn_id).status == "cancelled"

    with pytest.raises(ValueError, match="terminal"):
        asyncio.run(session.resume(turn_id, confirm=True))
    with pytest.raises(ValueError, match="terminal"):
        session.request_cancel(turn_id)


# ---------------------------------------------------------------------------
# brief ⑥:已补结果的调用不被 resume 重复补(幂等判据 = pair 已配)
# ---------------------------------------------------------------------------


def test_resume_settlement_is_idempotent_across_resumes(workspace: Path, open_ws: Callable[..., Any]) -> None:
    """跨窗口孤儿 + 最后窗口一次补齐;reconcile 续跑不重复清算(B29/⑥)。"""
    session = open_ws(ScriptedAdapter([_resp("done after reconcile")]))
    turn_id = _dead_turn(
        workspace,
        session.thread_id,
        [
            _response_entry("round one", (("c1", "read", {"path": "note.txt"}),)),
            _response_entry("round two", (("c2", "bash", {"command": "true"}),)),
            _tool_call_entry("c2", "bash", {"command": "true"}),
        ],
    )

    paused = asyncio.run(session.resume(turn_id, confirm=True))
    assert paused.status is TurnStatus.PAUSED and paused.unknown_calls == ("c2",)
    results = _results(workspace, turn_id)
    by_id = {r["call_id"]: r for r in results}
    assert by_id["c1"]["status"] == "not_executed"  # 跨窗口孤儿同样补齐
    assert by_id["c2"]["status"] == "outcome_unknown"

    done = asyncio.run(session.resume(turn_id, reconcile={"c2": "ok"}))
    assert done.status is TurnStatus.COMPLETED and done.output == "done after reconcile"
    results = _results(workspace, turn_id)
    statuses = [(r["call_id"], r["status"]) for r in results]
    assert statuses.count(("c1", "not_executed")) == 1  # 绝不重复补
    assert statuses.count(("c2", "outcome_unknown")) == 1
    assert statuses.count(("c2", "ok")) == 1  # reconcile 是追加,不是覆盖


# ---------------------------------------------------------------------------
# 幂等重入安全:PAUSED 期间 queued 的 cancel,resume 重放不得绕过
# ---------------------------------------------------------------------------


def test_queued_cancel_on_paused_turn_blocks_replay(
    workspace: Path, open_ws: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    args = {"path": "f.txt", "content": "must not run"}
    session = open_ws(ScriptedAdapter([_resp("", (("c1", "write", args),)), _resp("不可达")]))
    paused = asyncio.run(session.submit("write it"))
    assert paused.status is TurnStatus.PAUSED

    assert session.request_cancel(paused.turn_id) == "queued"
    result = asyncio.run(session.resume(paused.turn_id, approval={"c1": "allow"}))

    assert result.status is TurnStatus.CANCELLED
    assert not (workspace / "f.txt").exists()  # 批准重放被检查点拒在 dispatch 之前
    results = _results(workspace, paused.turn_id)
    assert len(results) == 1
    assert results[0]["call_id"] == "c1" and results[0]["status"] == "not_executed"
    kinds = _kinds(workspace, paused.turn_id)
    assert kinds.count(ItemKind.TOOL_CALL) == 0


# ---------------------------------------------------------------------------
# B14 对称性:complete(model_finished)终态守卫走同一清算协议
# ---------------------------------------------------------------------------


def _direct_deps(store: Store, bus: Any, workspace: Path, adapter: Any, tools: Mapping[str, Tool]) -> TurnDeps:
    return TurnDeps(
        store=store,
        bus=bus,
        adapter=adapter,
        tools=tools,
        permissions=PermissionContext(workspace=str(workspace), policy=PermissionPolicy(), mode="on-request"),
        workspace=workspace,
        system_prompt="test-agent",
    )


def test_model_finished_guard_settles_orphan_windows(store: Store, bus: Any, workspace: Path) -> None:
    store.ensure_thread("t")
    create_turn(store, turn_id="turn-1", thread_id="t", binding={})
    store.append_items(
        "t",
        "turn-1",
        [_response_entry("earlier round", (("c1", "noop", {}),))],  # 遗留未闭合窗口
    )
    deps = _direct_deps(store, bus, workspace, ScriptedAdapter([_resp("done")]), {"noop": _toy("noop")})

    result = asyncio.run(run_turn(deps, TurnState("turn-1", "t")))

    assert result.status is TurnStatus.COMPLETED and result.output == "done"
    results = [i.payload for i in store.read_turn_items("turn-1") if i.kind == ItemKind.TOOL_RESULT]
    assert len(results) == 1
    assert results[0]["call_id"] == "c1" and results[0]["status"] == "not_executed"


def test_cancel_request_at_loop_top_converges_before_model_call(store: Store, bus: Any, workspace: Path) -> None:
    store.ensure_thread("t")
    create_turn(store, turn_id="turn-1", thread_id="t", binding={})
    store.append_items(
        "t",
        "turn-1",
        [
            _response_entry("declared", (("c1", "noop", {}),)),
            ItemEntry(kind=ItemKind.CANCEL_REQUEST, payload={"source": "user", "parent_turn_id": None}),
        ],
    )
    deps = _direct_deps(store, bus, workspace, ScriptedAdapter([]), {"noop": _toy("noop")})

    result = asyncio.run(run_turn(deps, TurnState("turn-1", "t")))

    assert result.status is TurnStatus.CANCELLED  # 剧本为空:若请求模型即 FAILED
    items = store.read_turn_items("turn-1")
    results = [i.payload for i in items if i.kind == ItemKind.TOOL_RESULT]
    assert len(results) == 1 and results[0]["status"] == "not_executed"
    # 模型从未被请求:日志里只有预写的那条 model_response
    assert sum(1 for i in items if i.kind == ItemKind.MODEL_RESPONSE) == 1


# ---------------------------------------------------------------------------
# request_cancel 守卫与锁纪律
# ---------------------------------------------------------------------------


def test_request_cancel_guards(workspace: Path, open_ws: Callable[..., Any]) -> None:
    session = open_ws(ScriptedAdapter([_resp("fin")]))
    with pytest.raises(ValueError, match="unknown turn_id"):
        session.request_cancel("nope")

    done = asyncio.run(session.submit("hello"))
    with pytest.raises(ValueError, match="terminal"):
        session.request_cancel(done.turn_id)

    def _make(other: Store) -> str:
        other.ensure_thread("foreign-thread")
        foreign = f"foreign-{uuid4().hex}"
        create_turn(other, turn_id=foreign, thread_id="foreign-thread", binding={})
        return foreign

    foreign_turn = _with_other_store(workspace, _make)
    with pytest.raises(ValueError, match="different thread"):
        session.request_cancel(foreign_turn)


def test_request_cancel_needs_no_execution_lock(workspace: Path, open_ws: Callable[..., Any]) -> None:
    session = open_ws(ScriptedAdapter([]))

    def _make(other: Store) -> str:
        turn_id = f"running-{uuid4().hex}"
        create_turn(other, turn_id=turn_id, thread_id=session.thread_id, binding={})
        return turn_id

    turn_id = _with_other_store(workspace, _make)
    with thread_execution_lock(workspace, session.thread_id):
        assert session.request_cancel(turn_id) == "queued"  # append 不取执行锁(§5.8)
    assert session.status(turn_id).status == "running"  # 他进程执行中:状态由执行者收敛


def test_submit_busy_still_raises_conversation_busy(workspace: Path, open_ws: Callable[..., Any]) -> None:
    """_active_task 登记不改变锁纪律:锁被占 → ConversationBusy(既有语义零变化)。"""
    session = open_ws(ScriptedAdapter([_resp("never")]))
    with thread_execution_lock(workspace, session.thread_id):
        with pytest.raises(ConversationBusy):
            asyncio.run(session.submit("hi"))


# ---------------------------------------------------------------------------
# CLI:praxis cancel <turn_id> [--thread]——跨进程只 append item,exit 0
# ---------------------------------------------------------------------------


def test_cli_cancel_appends_item_exit_zero(workspace: Path, open_ws: Callable[..., Any]) -> None:
    session = open_ws(ScriptedAdapter([]))
    turn_id = _dead_turn(workspace, session.thread_id, [])

    res = runner.invoke(
        app,
        ["cancel", turn_id, "--workspace", str(workspace), "--thread", session.thread_id],
        env=_CLI_ENV,
    )
    assert res.exit_code == 0, res.output
    assert "queued" in res.output

    def _cancels(other: Store) -> list[Item]:
        return [i for i in other.read_turn_items(turn_id) if i.kind == ItemKind.CANCEL_REQUEST]

    cancels = _with_other_store(workspace, _cancels)
    assert len(cancels) == 1 and cancels[0].payload["source"] == "user"


def test_cli_cancel_terminal_and_unknown_turn_usage_error(workspace: Path, open_ws: Callable[..., Any]) -> None:
    session = open_ws(ScriptedAdapter([_resp("done")]))
    done = asyncio.run(session.submit("hello"))

    terminal = runner.invoke(
        app,
        ["cancel", done.turn_id, "--workspace", str(workspace), "--thread", session.thread_id],
        env=_CLI_ENV,
    )
    assert terminal.exit_code == 3, terminal.output
    assert "terminal" in terminal.output  # 真实守卫拒绝,不是"未知命令"的用法错

    unknown = runner.invoke(
        app,
        ["cancel", "ffffffff", "--workspace", str(workspace), "--thread", session.thread_id],
        env=_CLI_ENV,
    )
    assert unknown.exit_code == 3, unknown.output
    assert "unknown turn_id" in unknown.output


# ---------------------------------------------------------------------------
# sdk 出口(宿主/T11 经 sdk 触达清算面)
# ---------------------------------------------------------------------------


def test_sdk_exports_compensation_surface() -> None:
    for name in ("UnresolvedCall", "close_unresolved", "unclosed_windows"):
        assert name in sdk_pkg.__all__, f"sdk.__all__ missing {name}"
        assert hasattr(sdk_pkg, name), f"sdk missing attribute {name}"
    assert sdk_pkg.__all__ == sorted(sdk_pkg.__all__)


# ---------------------------------------------------------------------------
# 状态机唯一写入点:cancel 收敛的 transition 经 statemachine(事件如实广播)
# ---------------------------------------------------------------------------


def test_cancel_convergence_emits_status_changed(workspace: Path, open_ws: Callable[..., Any]) -> None:
    started = asyncio.Event()
    session = open_ws(
        ScriptedAdapter([_resp("", (("c1", "hang", {}),)), _resp("不可达")]),
        extra_tools=[_toy("hang", hang=started)],
    )
    sub = session.subscribe()

    async def scenario() -> Any:
        task = asyncio.create_task(session.submit("go"))
        await started.wait()
        await asyncio.sleep(0.01)
        session.request_cancel(session.turns()[-1].turn_id)
        return await task

    result = asyncio.run(scenario())
    assert result.status is TurnStatus.CANCELLED
    events: list[Any] = []
    while True:
        try:
            events.append(sub.queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    changed = [e for e in events if type(e).__name__ == "TurnStatusChanged" and e.to_status == "cancelled"]
    assert len(changed) == 1 and changed[0].reason == "cancelled_by_user"
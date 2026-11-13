"""praxis.core.recovery — sweep 与 classify_pending 单测(设计 §5.1/§5.8)。

TDD:本文件在 recovery.py 存在前必须失败。
sweep 的"活 turn 不误伤"用同进程持有执行锁模拟另一执行者;真实跨进程/
SIGKILL 场景由 acceptance/test_kill9_sweep.py 承担。
classify_pending 是纯函数,直接手工构造 Item,不碰 Store。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from praxis.core import recovery
from praxis.core.events import EventBus
from praxis.core.locks import thread_execution_lock
from praxis.core.recovery import PendingCall, classify_pending, sweep
from praxis.core.statemachine import TurnStatus, transition
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import ModelToolCall
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolResult,
)
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import get_turn
from tests.praxis.conftest import make_turn

# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def test_sweep_noop_without_running_turns(store: Store, bus: EventBus, workspace: Path) -> None:
    make_turn(store, status="completed", thread_id="t")
    assert sweep(store, bus, workspace) == []


def test_sweep_skips_live_turn(store: Store, bus: EventBus, workspace: Path) -> None:
    """执行锁被持有 = 另一进程正在执行该 thread → sweep 必须跳过,绝不改状态。"""
    turn = make_turn(store, status="running", thread_id="t")
    with thread_execution_lock(workspace, "t"):
        assert sweep(store, bus, workspace) == []
    live = get_turn(store, turn.turn_id)
    assert live is not None and live.status == "running"


def test_sweep_marks_dead_turn_with_last_seq(store: Store, bus: EventBus, workspace: Path) -> None:
    """取到锁(原主已死)→ interrupted;turn_status item 自动携带 last_seq。"""
    turn = make_turn(store, status="running", thread_id="t")
    store.append_items("t", turn.turn_id, [ItemEntry("user_message", {"text": "x"})])

    marked = sweep(store, bus, workspace)

    assert marked == [turn.turn_id]
    dead = get_turn(store, turn.turn_id)
    assert dead is not None and dead.status == "interrupted"
    item = store.read_turn_items(turn.turn_id)[-1]
    assert item.kind == ItemKind.TURN_STATUS.value
    assert item.payload["to"] == "interrupted"
    assert item.payload["reason"] == "executor_process_died"
    assert item.payload["last_seq"] == 1  # user_message 的 seq,turn_status 自身在其后


def test_sweep_emits_turn_status_changed(store: Store, bus: EventBus, workspace: Path) -> None:
    turn = make_turn(store, status="running", thread_id="t")
    sub = bus.subscribe()
    assert sweep(store, bus, workspace) == [turn.turn_id]
    assert sub.queue.qsize() == 1
    event = sub.queue.get_nowait()
    assert getattr(event, "to_status", None) == "interrupted"


def test_sweep_only_interrupts_unowned_thread(store: Store, bus: EventBus, workspace: Path) -> None:
    """混合场景:live 线程的 turn 不动,dead 线程的 turn 被标记。"""
    live = make_turn(store, status="running", thread_id="live")
    dead = make_turn(store, status="running", thread_id="dead")
    with thread_execution_lock(workspace, "live"):
        marked = sweep(store, bus, workspace)
    assert marked == [dead.turn_id]
    a = get_turn(store, live.turn_id)
    b = get_turn(store, dead.turn_id)
    assert a is not None and a.status == "running"
    assert b is not None and b.status == "interrupted"


def test_sweep_marks_multiple_running_turns_of_dead_thread(store: Store, bus: EventBus, workspace: Path) -> None:
    a = make_turn(store, status="running", thread_id="t")
    b = make_turn(store, status="running", thread_id="t")
    marked = sweep(store, bus, workspace)
    assert set(marked) == {a.turn_id, b.turn_id}


def test_sweep_lock_released_after_marking(store: Store, bus: EventBus, workspace: Path) -> None:
    """sweep 结束必须释放锁:标记后调用方能立刻自己取锁执行。"""
    make_turn(store, status="running", thread_id="t")
    assert sweep(store, bus, workspace) != []
    with thread_execution_lock(workspace, "t"):
        pass  # 不抛 ThreadBusy 即证明已释放


def test_sweep_skips_raced_turn_and_continues(
    store: Store, bus: EventBus, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6 竞态:snapshot 里 running,取锁前执行者已合法完成该 turn → 跳过,绝不中断清扫其余。

    确定性注入:monkeypatch recovery 的 list_turns,快照返回的同一瞬间把 turn A
    合法 transition 到 completed(终态)。sweep 对 A 的 INTERRUPTED 转换必然抛
    IllegalTransition——修复前这会炸掉整次 sweep,B 永远不会被标记。
    """
    a = make_turn(store, status="running", thread_id="raced")
    b = make_turn(store, status="running", thread_id="dead")
    real_list_turns = recovery.list_turns

    def stale_list_turns(s: Store, *, status: str | None = None) -> list[object]:
        snapshot = list(real_list_turns(s, status=status))  # 此刻两条都还是 running
        transition(store, bus, a.turn_id, TurnStatus.COMPLETED, reason="model_finished")  # 竞态发生
        return snapshot

    monkeypatch.setattr(recovery, "list_turns", stale_list_turns)

    marked = sweep(store, bus, workspace)  # 不得抛 IllegalTransition

    assert marked == [b.turn_id]  # 单条竞态不影响其余 dead turn 的清扫
    ra = get_turn(store, a.turn_id)
    rb = get_turn(store, b.turn_id)
    assert ra is not None and ra.status == "completed"  # 执行者的合法结果不被回退
    assert rb is not None and rb.status == "interrupted"


# ---------------------------------------------------------------------------
# classify_pending — 纯函数,手工构造 Item 序列
# ---------------------------------------------------------------------------


def _tool(name: str, *, idempotent: bool) -> Tool:
    definition = ToolDefinition(name=name, description=f"{name} tool", input_schema={"type": "object"})
    return Tool(
        definition=definition,
        validate_input=lambda arguments: arguments,
        run=lambda arguments, exec_ctx: ToolResult(tool_call_id="c", tool_name=name),
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedUse(effects=frozenset({ToolEffect.READ_WORKSPACE}), targets=()),
        idempotent=idempotent,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30.0,
        max_model_output_bytes=4096,
    )


def _item(seq: int, kind: str, payload: Mapping[str, object]) -> Item:
    return Item(
        item_id=f"item-{seq}",
        thread_id="t",
        turn_id="turn-1",
        seq=seq,
        kind=kind,
        payload=payload,
        created_at=float(seq),
    )


def _model_response(seq: int, calls: object) -> Item:
    return _item(seq, ItemKind.MODEL_RESPONSE.value, {"content": "", "tool_calls": calls})


def _call(call_id: str, name: str, arguments: Mapping[str, object] | None = None) -> dict[str, object]:
    return {"call_id": call_id, "name": name, "arguments": arguments or {}}


_TOOLS: Mapping[str, Tool] = {
    "read_file": _tool("read_file", idempotent=True),
    "write_file": _tool("write_file", idempotent=False),
}


def test_classify_no_model_response_returns_empty() -> None:
    items = [_item(1, ItemKind.USER_MESSAGE.value, {"text": "hi"})]
    assert classify_pending(items, _TOOLS) == []


def test_classify_model_response_without_calls_returns_empty() -> None:
    items = [_item(1, ItemKind.MODEL_RESPONSE.value, {"content": "done", "tool_calls": []})]
    assert classify_pending(items, _TOOLS) == []
    items_missing = [_item(1, ItemKind.MODEL_RESPONSE.value, {"content": "done"})]
    assert classify_pending(items_missing, _TOOLS) == []


def test_classify_pending_not_started_uses_last_response() -> None:
    """无 tool_call item → started=False;arguments 与 call_id/name 从 payload 重建。"""
    items = [
        _item(1, ItemKind.USER_MESSAGE.value, {"text": "hi"}),
        _model_response(2, [_call("c1", "read_file", {"path": "a.md"})]),
    ]
    pending = classify_pending(items, _TOOLS)
    assert pending == [
        PendingCall(
            call=ModelToolCall(call_id="c1", name="read_file", arguments={"path": "a.md"}),
            started=False,
            idempotent=True,
        )
    ]


def test_classify_started_true_when_tool_call_item_exists() -> None:
    items = [
        _model_response(1, [_call("c1", "write_file", {"path": "a.md"})]),
        _item(2, ItemKind.TOOL_CALL.value, {"call_id": "c1", "tool_name": "write_file", "arguments": {"path": "a.md"}}),
    ]
    pending = classify_pending(items, _TOOLS)
    assert pending == [
        PendingCall(
            call=ModelToolCall(call_id="c1", name="write_file", arguments={"path": "a.md"}),
            started=True,
            idempotent=False,  # 非幂等工具:续跑前必须先人工确认
        )
    ]


def test_classify_excludes_call_ids_with_tool_result() -> None:
    items = [
        _model_response(
            1,
            [_call("c1", "read_file"), _call("c2", "read_file")],
        ),
        _item(2, ItemKind.TOOL_CALL.value, {"call_id": "c1", "tool_name": "read_file", "arguments": {}}),
        _item(3, ItemKind.TOOL_RESULT.value, {"call_id": "c1", "status": "ok"}),
    ]
    pending = classify_pending(items, _TOOLS)
    assert [p.call.call_id for p in pending] == ["c2"]  # c1 已有结果,被排除


def test_classify_uses_last_model_response_only() -> None:
    """多轮:只审最后一条 model_response;更早轮次未决的 call_id 不算 pending。"""
    items = [
        _model_response(1, [_call("c1", "read_file")]),  # 没有任何 result——但已被取代
        _item(2, ItemKind.TOOL_CALL.value, {"call_id": "c2", "tool_name": "write_file", "arguments": {}}),
        _model_response(3, [_call("c2", "write_file", {"path": "b.md"})]),
    ]
    pending = classify_pending(items, _TOOLS)
    assert [(p.call.call_id, p.started) for p in pending] == [("c2", True)]


def test_classify_mixed_statuses_preserve_order() -> None:
    items = [
        _model_response(
            1,
            [
                _call("c1", "read_file"),
                _call("c2", "bash", {"cmd": "ls"}),  # 未知工具 → idempotent=False
                _call("c3", "read_file"),
            ],
        ),
        _item(2, ItemKind.TOOL_RESULT.value, {"call_id": "c1", "status": "ok"}),
        _item(3, ItemKind.TOOL_CALL.value, {"call_id": "c3", "tool_name": "read_file", "arguments": {}}),
    ]
    pending = classify_pending(items, _TOOLS)
    assert [(p.call.call_id, p.started, p.idempotent) for p in pending] == [
        ("c2", False, False),
        ("c3", True, True),
    ]


def test_classify_accepts_frozen_shapes() -> None:
    """Store 读出的 payload 是冻结形态:tool_calls=tuple,条目=mappingproxy。"""
    frozen_calls = (
        MappingProxyType(
            {"call_id": "c1", "name": "read_file", "arguments": MappingProxyType({"path": "x"})}
        ),
    )
    items = [_item(1, ItemKind.MODEL_RESPONSE.value, {"content": "", "tool_calls": frozen_calls})]
    pending = classify_pending(items, _TOOLS)
    assert len(pending) == 1
    assert pending[0].call == ModelToolCall(call_id="c1", name="read_file", arguments={"path": "x"})
    assert dict(pending[0].call.arguments) == {"path": "x"}


def test_classify_skips_malformed_entries() -> None:
    """读路径零校验(§B1):畸形条目跳过,绝不抛异常。"""
    items = [
        _model_response(
            1,
            [
                "not-a-mapping",
                {"call_id": 7, "name": "read_file"},  # call_id 非 str
                {"name": "read_file"},  # 缺 call_id
                {"call_id": "c1", "arguments": {"path": "a"}},  # 缺 name
                _call("c2", "read_file", "not-a-mapping"),  # arguments 非 mapping → {}
            ],
        )
    ]
    pending = classify_pending(items, _TOOLS)
    assert [(p.call.call_id, dict(p.call.arguments)) for p in pending] == [("c2", {})]


def test_pending_call_is_frozen() -> None:
    pc = PendingCall(call=ModelToolCall(call_id="c", name="n", arguments={}), started=False, idempotent=False)
    with pytest.raises(FrozenInstanceError):
        pc.started = True  # type: ignore[misc]

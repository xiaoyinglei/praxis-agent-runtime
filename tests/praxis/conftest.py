"""Praxis 测试公共 fixture(Task 4 起):workspace 目录与独立 db 的 Store。

Task 7 补充 make_turn 助手:创建 turn 然后通过合法转换链到达目标状态。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from praxis.core.events import EventBus
from praxis.store.log import Store
from praxis.store.turns import Turn, create_turn, get_turn, write_status


# Shortest legal chain(s) from "running" to each target status.
# PAUSED needs two steps because we must go running→paused and then
# pause→running before pausing again (a single run→pause gives
# "running", not the desired paused-with-history).
_CHAINS: dict[str, list[tuple[str, str]]] = {
    "paused": [("running", "paused"), ("paused", "running"), ("running", "paused")],
    "interrupted": [("running", "interrupted")],
    "completed": [("running", "completed")],
    "failed": [("running", "failed")],
    "cancelled": [("running", "cancelled")],
}


def make_turn(
    store: Store,
    bus: EventBus | None = None,
    *,
    thread_id: str = "t",
    status: str = "running",
) -> Turn:
    """Create a new turn and reach the requested status via legal transitions."""
    tid = f"turn-{uuid4().hex}"
    turn = create_turn(store, turn_id=tid, thread_id=thread_id, binding={})
    assert turn.status == "running"  # CREATE always starts at running
    if status == "running":
        return turn
    chain = _CHAINS.get(status)
    assert chain is not None, f"unknown target status: {status!r}"
    prev = turn
    for src, dst in chain:
        old = get_turn(store, tid)
        assert old is not None and old.status == src, f"expected {src}, got {old.status}"
        prev = write_status(store, tid, dst, reason=f"reach_{dst}")
        if bus is not None:
            from praxis.protocol.events import TurnStatusChanged

            bus.emit(
                TurnStatusChanged(
                    turn_id=tid,
                    thread_id=thread_id,
                    from_status=src,
                    to_status=dst,
                    reason=f"reach_{dst}",
                )
            )
    result = get_turn(store, tid)
    assert result is not None and result.status == status, f"expected {status}, got {result.status}"
    return result


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def store(workspace: Path) -> Iterator[Store]:
    s = Store(workspace / "store.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def bus() -> Iterator[EventBus]:
    eb = EventBus()
    yield eb

"""Praxis 状态机：TurnStatus、TRANSITIONS 表与 transition() 入口。

TDD — Tests First: these tests must FAIL before statemachine.py is written.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from praxis.core.events import EventBus
from praxis.core.statemachine import (
    TRANSITIONS,
    IllegalTransition,
    TurnStatus,
    transition,
)
from praxis.protocol.items import ItemKind
from praxis.store.log import Store
from praxis.store.turns import Turn
from tests.praxis.conftest import make_turn

# ---------------------------------------------------------------------------
# Phase 1: class existence — must fail when module is absent
# ---------------------------------------------------------------------------


def test_turn_status_has_all_members() -> None:
    """All six statuses are enum members."""
    members = {e.value for e in TurnStatus}
    assert members == {"running", "paused", "interrupted", "completed", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# Phase 2: exhaustive 6 × 6 transition table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("src", list(TurnStatus))
@pytest.mark.parametrize("dst", list(TurnStatus))
def test_transition_table_exhaustive(store: Store, bus: EventBus, src: TurnStatus, dst: TurnStatus) -> None:
    """Legal → status updated; illegal → IllegalTransition with full attributes."""
    turn = make_turn(store, status=src.value, thread_id="t")
    if dst in TRANSITIONS[src]:
        out = transition(store, bus, turn.turn_id, dst, reason="test")
        assert isinstance(out, Turn)
        assert out.status == dst.value
    else:
        with pytest.raises(IllegalTransition) as err_ctx:
            transition(store, bus, turn.turn_id, dst, reason="test")
        err = err_ctx.value
        assert err.turn_id == turn.turn_id
        assert err.from_status == src.value
        assert err.to_status == dst.value


# ---------------------------------------------------------------------------
# Phase 2b: missing-turn branch (FINDING 1 fix)
# ---------------------------------------------------------------------------


def test_missing_turn_raises_illegal_transition() -> None:
    """transition on a non-existent turn_id → IllegalTransition(from_status='', …)."""
    s = Store(tempfile.mktemp(suffix=".db"))
    try:
        bus = EventBus()
        with pytest.raises(IllegalTransition, match="nonexistent") as err:
            transition(s, bus, "nonexistent", TurnStatus.COMPLETED, reason="ghost")
        e = err.value
        assert e.turn_id == "nonexistent"
        assert e.from_status == ""
        assert e.to_status == "completed"
    finally:
        s.close()
        try:
            os.unlink(s._db_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Phase 3: reason persistence + event emission (B11 regression)
# ---------------------------------------------------------------------------


def test_reason_persisted_and_emitted(store: Store, bus: EventBus) -> None:
    """reason flows to both write_status item AND TurnStatusChanged event."""
    sub = bus.subscribe()  # subscribe BEFORE the call so event lands in queue
    turn = make_turn(store, status=TurnStatus.RUNNING.value)
    transition(store, bus, turn.turn_id, TurnStatus.FAILED, reason="provider 503")
    # Check persisted item
    items = store.read_turn_items(turn.turn_id)
    status_items = [i for i in items if i.kind == ItemKind.TURN_STATUS]
    assert len(status_items) == 1
    assert status_items[0].payload["reason"] == "provider 503"
    # Check emitted event
    evt = sub.queue.get_nowait()
    assert evt.reason == "provider 503"


# ---------------------------------------------------------------------------
# Phase 4: terminal states have no exits (design §5.2)
# ---------------------------------------------------------------------------


def test_terminal_states_have_no_exits() -> None:
    """COMPLETED, FAILED, CANCELLED each map to frozenset()."""
    for s in (TurnStatus.COMPLETED, TurnStatus.FAILED, TurnStatus.CANCELLED):
        assert TRANSITIONS[s] == frozenset()

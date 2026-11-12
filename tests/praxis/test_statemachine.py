"""Praxis 状态机：TurnStatus、TRANSITIONS 表与 transition() 入口。

TDD — Tests First: these tests must FAIL before statemachine.py is written.
"""

from __future__ import annotations

import pytest

from praxis.core.events import EventBus
from praxis.protocol.items import ItemKind
from praxis.store.log import Store
from praxis.store.turns import Turn
from praxis.core.statemachine import (
    IllegalTransition,
    TRANSITIONS,
    TurnStatus,
    transition,
)
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
    """Legal → status updated; illegal → IllegalTransition raised."""
    turn = make_turn(store, bus, status=src.value)
    if dst in TRANSITIONS[src]:
        out = transition(store, bus, turn.turn_id, dst, reason="test")
        assert isinstance(out, Turn)
        assert out.status == dst.value
    else:
        with pytest.raises(IllegalTransition):
            transition(store, bus, turn.turn_id, dst, reason="test")


# ---------------------------------------------------------------------------
# Phase 3: reason persistence + event emission (B11 regression)
# ---------------------------------------------------------------------------


def test_reason_persisted_and_emitted(store: Store, bus: EventBus) -> None:
    """reason flows to both write_status item AND TurnStatusChanged event."""
    sub = bus.subscribe()  # subscribe BEFORE the call so event lands in queue
    turn = make_turn(store, bus=None, status=TurnStatus.RUNNING.value)
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

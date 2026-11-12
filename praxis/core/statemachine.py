"""Praxis 状态机：TurnStatus 枚举、TRANSITIONS 转换表与 transition() 入口。

设计 §5.2 · findings B11（reason 结构性消亡）· 硬规则 3/5（write_status 唯一调用方）。

public API
----------
- TurnStatus(StrEnum) : running | paused | interrupted | completed | failed | cancelled
- TRANSITIONS: dict[TurnStatus, frozenset[TurnStatus]]  (转义 §5.2)
- IllegalTransition(RuntimeError): turn_id / from_status / to_status
- transition(store, bus, turn_id, to, *, reason) -> Turn
    读当前 → 校验 TRANSITIONS → turns.write_status → 广播 TurnStatusChanged → 返回 Turn
    reason 是必填关键字参数 (B11)。
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from praxis.protocol.events import TurnStatusChanged
from praxis.store.log import Store
from praxis.store.turns import Turn, get_turn, write_status

if TYPE_CHECKING:
    from praxis.core.events import EventBus


# ---------------------------------------------------------------------------
# Enum + Transition Table (§5.2 verbatim)
# ---------------------------------------------------------------------------


class TurnStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Transitions table — each value is the allowed destination set.
#: Terminal states (COMPLETED, FAILED, CANCELLED) map to empty frozenset().
TRANSITIONS: dict[TurnStatus, frozenset[TurnStatus]] = {
    TurnStatus.RUNNING: frozenset({
        TurnStatus.PAUSED,
        TurnStatus.COMPLETED,
        TurnStatus.FAILED,
        TurnStatus.CANCELLED,
        TurnStatus.INTERRUPTED,
    }),
    TurnStatus.PAUSED: frozenset({
        TurnStatus.RUNNING,
        TurnStatus.CANCELLED,
        TurnStatus.INTERRUPTED,
    }),
    TurnStatus.INTERRUPTED: frozenset({
        TurnStatus.RUNNING,
        TurnStatus.CANCELLED,
    }),
    TurnStatus.COMPLETED: frozenset(),
    TurnStatus.FAILED: frozenset(),
    TurnStatus.CANCELLED: frozenset(),
}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class IllegalTransition(RuntimeError):  # noqa: N818 — design §5.2 name
    """Raised when a state change violates the TRANSITIONS table."""

    def __init__(self, turn_id: str, from_status: str, to_status: str) -> None:
        self.turn_id = turn_id
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Illegal transition for turn {turn_id!r}: "
            f"{from_status!r} → {to_status!r}"
        )


# ---------------------------------------------------------------------------
# Module-level function — sole caller of write_status in praxis/
# ---------------------------------------------------------------------------


def transition(
    store: Store,
    bus: EventBus,
    turn_id: str,
    to: TurnStatus,
    *,
    reason: str,
) -> Turn:
    """Validate and apply a single state transition.

    Rules
    -----
    1. Read current turn status via ``get_turn``. Missing turn →
       ``IllegalTransition(turn_id=..., from_status="", to_status=to.value)``.
    2. Validate destination against ``TRANSITIONS``. Invalid →
       ``IllegalTransition(turn_id, from_status, to.value)``.
    3. Call ``write_status(store, turn_id, to.value, reason=reason)`` (UPDATE +
       append-only item in one transaction).
    4. Emit ``TurnStatusChanged`` on the event bus.
    5. Return the updated ``Turn``.

    ``reason`` is a required keyword argument (B11).
    """
    # Step 1: read current state
    old = get_turn(store, turn_id)
    if old is None:
        raise IllegalTransition(turn_id=turn_id, from_status="", to_status=to.value)

    from_status: str = old.status

    # Step 2: validate transition
    src = TurnStatus(from_status)
    if to not in TRANSITIONS[src]:
        raise IllegalTransition(turn_id=turn_id, from_status=from_status, to_status=to.value)

    # Step 3: apply via write_status (tx: UPDATE + append-only item)
    updated = write_status(store, turn_id, to.value, reason=reason)

    # Step 4: broadcast event
    bus.emit(TurnStatusChanged(
        turn_id=turn_id,
        thread_id=old.thread_id,
        from_status=from_status,
        to_status=to.value,
        reason=reason,
    ))

    return updated

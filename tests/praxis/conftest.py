"""Praxis 测试公共 fixture(Task 4 起):workspace 目录与独立 db 的 Store。

Task 7: make_turn reaches requested status via transition() (the sole state-write
entry point), eliminating duplicate write_status/event-emit boilerplate and the
loop-body import that was hiding inside the for-loop.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import pytest

from praxis.store.log import Store
from praxis.store.turns import Turn, create_turn, get_turn

# Sentinel — default "no status given" becomes RUNNING inside the function.
_UNSET: object = object()

if TYPE_CHECKING:
    # Stubs for type-hints only; resolved at runtime inside functions.
    class _EventBus(Protocol):
        def emit(self, event: object) -> None: ...

    class _TransitionFn(Protocol):
        def __call__(
            self,
            store: Store,
            bus: _EventBus,
            turn_id: str,
            to: str,  # noqa: PYI041 — str annotation avoids forward-reference pain
            *,
            reason: str,
        ) -> Turn: ...


def make_turn(
    store: Store,
    status: str | None = None,  # type: ignore[assignment]
    thread_id: str = "t",
    _bus: object | None = None,
    *,
    _transition: object | None = None,
) -> Turn:
    """Create a new turn and (if needed) reach the requested status via transition().

    By default (_transition=None) delegates to
    ``praxis.core.statemachine.transition``.
    Callers may inject ``_transition`` or ``_bus`` for testing without duplicating logic.
    """
    tid = f"turn-{uuid4().hex}"
    # Lazy import — avoids any circular import dance at module load time.
    from praxis.core.events import EventBus  # noqa: PLC0415
    from praxis.core.statemachine import TurnStatus as TS  # noqa: PLC0415,N817
    from praxis.core.statemachine import transition as _default_transition  # noqa: PLC0415

    target: TS
    if status is None:
        target = TS.RUNNING
    else:
        target = TS(status)

    transition_fn: _TransitionFn = _transition or _default_transition

    turn = create_turn(store, turn_id=tid, thread_id=thread_id, binding={})
    assert turn.status == "running"
    if target != TS.RUNNING:
        bus = _bus or EventBus()
        transition_fn(store, bus, tid, target, reason=f"reach_{target.value}")
    result = get_turn(store, tid)
    assert result is not None and result.status == target.value, (
        f"expected {target.value}, got {result.status}"
    )
    return result


# --- Helpers exposed for tests that need extra stores -----------------------


def stub_store() -> Store:
    """Create a temporary standalone store."""
    s = Store(tempfile.mktemp(suffix=".db"))
    return s


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
def bus() -> Iterator[_EventBus]:  # noqa: F821
    from praxis.core.events import EventBus  # noqa: PLC0415

    eb = EventBus()
    yield eb

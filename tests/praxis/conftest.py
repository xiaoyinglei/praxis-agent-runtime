"""Praxis 测试公共 fixture(Task 4 起):workspace 目录与独立 db 的 Store。

Task 7: make_turn reaches requested status via transition() (the sole state-write
entry point), eliminating duplicate write_status/event-emit boilerplate and the
loop-body import that was hiding inside the for-loop.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
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


def make_minimal_tool(**overrides: object) -> Any:
    """构造一个最小可用的合法 Tool(阶段 2 协议层测试助手,沿用 make_turn 的惰性导入风格)。

    默认:只读玩具,run 为双参(签名 ToolRunner)返回 OK 结果,
    check_permission 不设(None)。任意 Tool 字段可用关键字 overrides 替换。
    """
    from praxis.protocol.tools import (  # noqa: PLC0415
        CancellationMode,
        InterruptBehavior,
        ResolvedUse,
        Tool,
        ToolDefinition,
        ToolEffect,
        ToolResult,
    )

    kwargs: dict[str, object] = dict(
        definition=ToolDefinition(
            name="minimal", description="minimal test tool", input_schema={"type": "object"}
        ),
        validate_input=lambda arguments: arguments,
        run=lambda arguments, exec_ctx: ToolResult(tool_call_id="pending", tool_name="minimal"),
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}), targets=()
        ),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30.0,
        max_model_output_bytes=4096,
    )
    kwargs.update(overrides)
    return Tool(**kwargs)


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

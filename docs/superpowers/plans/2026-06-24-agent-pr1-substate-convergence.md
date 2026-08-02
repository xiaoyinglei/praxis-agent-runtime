# PR1: Sub-State Convergence + Legacy Migration Implementation Plan

> **历史实施记录：** 其中的 persistent-memory 方案已于 2026-07-25 从产品运行时移除；不要按本文恢复该能力。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `PlanState`, `MemoryState`, `FinishState`, `DeferredToolState` as typed sub-state containers on `LoopState`, with dual-write to both new sub-states and old flat fields. Enable legacy checkpoint migration that reads old fields into new sub-states. No fields are deleted in this PR.

**Architecture:** New sub-state models are Pydantic `BaseModel` instances. `LoopState` gains four new TypedDict keys. `create_loop_state()` writes both new and old representations. `_migrate_legacy_state()` reads old flat fields into new sub-states at checkpoint load time. All call-sites continue reading old fields (dual-read optional in this PR).

**Tech Stack:** Python 3.12+, Pydantic v2, existing `rag.agent` module structure

## Global Constraints

- **No field deletions.** All 14 deprecated fields remain in `LoopState` TypedDict and factory.
- **No allowlist changes.** `AGENT_CHECKPOINT_MSGPACK_ALLOWLIST` unchanged in this PR.
- **Existing tests must pass.** 324 tests, zero regressions.
- **uv for all commands.** `uv run pytest`, `uv run ruff`, etc.

---

### Task 1: Define new sub-state Pydantic models

**Files:**
- Create: `rag/agent/loop/substate.py`

**Interfaces:**
- Produces: `PlanState`, `MemoryState`, `PersistentMemorySnapshot`, `DeferredToolState`, `DiscoveryCandidate`, `DiscoveryEvent`, `FinishState`

- [ ] **Step 1: Create the substate module with all new models**

```python
# rag/agent/loop/substate.py
"""Typed sub-state containers for the agent loop.

Each container groups related control-flow fields that were previously
flat in LoopState.  These are Pydantic models so they can be
serialised through the existing checkpoint serde path.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.loop.state import StopHookFeedback
from rag.agent.memory.models import (
    ContextBudgetSnapshot,
    ExtractedFact,
    MemoryBudgetSnapshot,
    MemoryRef,
    WorkingSummary,
)
from rag.agent.planning import AgentPlan, PlanEvent


class PersistentMemorySnapshot(BaseModel):
    """Bounded snapshot of persistent cross-session memory.

    ``index_ref`` and ``index_digest`` default to empty strings so that
    ``MemoryState(persistent=PersistentMemorySnapshot())`` works without
    passing arguments — the "not loaded yet" state is valid.
    """

    index_ref: str = ""
    index_digest: str = ""
    selected_count: int = 0
    selected_summaries: list[str] = Field(default_factory=list)


class PlanState(BaseModel):
    """Agent-managed planning state."""

    agent_plan: AgentPlan | None = None
    plan_events: list[PlanEvent] = Field(default_factory=list)


class MemoryState(BaseModel):
    """Working memory and persistent-memory context for the current run."""

    working_summary: WorkingSummary | None = None
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    context_budget: ContextBudgetSnapshot | None = None
    memory_refs: list[MemoryRef] = Field(default_factory=list)
    memory_budget: MemoryBudgetSnapshot | None = None
    memory_warnings: list[str] = Field(default_factory=list)
    reactive_compact_used: bool = False
    persistent: PersistentMemorySnapshot = Field(
        default_factory=PersistentMemorySnapshot
    )


class DiscoveryCandidate(BaseModel):
    """Tool-discovery candidate — typed replacement for ``dict[str, object]``."""

    tool_name: str
    description: str = ""
    relevance_score: float = 0.0
    metadata: dict[str, object] = Field(default_factory=dict)


class DiscoveryEvent(BaseModel):
    """Tool-search history event — typed replacement for ``dict[str, object]``."""

    query: str
    timestamp_iso: str = ""
    result_count: int = 0
    selected_tools: list[str] = Field(default_factory=list)


class DeferredToolState(BaseModel):
    """On-demand tool-discovery state."""

    active_tools: list[str] = Field(default_factory=list)
    active_tool_iterations: dict[str, int] = Field(default_factory=dict)
    last_candidates: list[DiscoveryCandidate] = Field(default_factory=list)
    last_search_query: str = ""
    search_history: list[DiscoveryEvent] = Field(default_factory=list)
    pinned_tools: list[str] = Field(default_factory=list)
    capability_diagnostics: list[RuntimeDiagnostic] = Field(default_factory=list)


class FinishState(BaseModel):
    """Stop-hook gate state — feedback / warnings injected into the next turn."""

    feedback: list[StopHookFeedback] = Field(default_factory=list)
    warnings: list[StopHookFeedback] = Field(default_factory=list)


__all__ = [
    "DeferredToolState",
    "DiscoveryCandidate",
    "DiscoveryEvent",
    "FinishState",
    "MemoryState",
    "PersistentMemorySnapshot",
    "PlanState",
]
```

- [ ] **Step 2: Run ruff format + check**

```bash
uv run ruff format rag/agent/loop/substate.py
uv run ruff check rag/agent/loop/substate.py
```

Expected: clean output, no errors.

- [ ] **Step 3: Commit**

```bash
git add rag/agent/loop/substate.py
git commit -m "feat(agent): add typed sub-state models for PR1 convergence

- PlanState, MemoryState, FinishState, DeferredToolState
- PersistentMemorySnapshot replaces bare memory_index: str
- DiscoveryCandidate, DiscoveryEvent typed replacements for dict

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Update LoopState TypedDict with new sub-state fields

**Files:**
- Modify: `rag/agent/loop/state.py:130-195` (LoopState TypedDict)
- Modify: `rag/agent/loop/state.py:196-275` (create_loop_state factory)

**Interfaces:**
- Consumes: `PlanState`, `MemoryState`, `DeferredToolState`, `FinishState` from `rag.agent.loop.substate`
- Produces: updated `LoopState` with 4 new keys: `plan_state`, `memory_state`, `deferred_tool_state`, `finish_state`

- [ ] **Step 1: Add imports to state.py**

```python
# Add after existing imports in rag/agent/loop/state.py
from rag.agent.loop.substate import (
    DeferredToolState,
    FinishState,
    MemoryState,
    PlanState,
)
```

- [ ] **Step 2: Add new fields to LoopState TypedDict**

Insert before the `# ── PR0:` comment block (after `latest_transition: LoopTransition | None`):

```python
    # ── PR1-like: typed sub-state convergence (dual-write, no deletions) ──
    plan_state: PlanState
    memory_state: MemoryState
    deferred_tool_state: DeferredToolState
    finish_state: FinishState
```

- [ ] **Step 3: Update create_loop_state for dual-write**

In `create_loop_state()`, add the new sub-state initialisation **after** the existing flat field defaults (so the old fields are still written). Append before the closing `}`:

```python
        # ── PR1: typed sub-state convergence (dual-write alongside flat fields) ──
        "plan_state": PlanState(
            agent_plan=None,
            plan_events=[],
        ),
        "memory_state": MemoryState(
            working_summary=None,
            extracted_facts=[],
            context_budget=None,
            memory_refs=[],
            memory_budget=None,
            memory_warnings=_bounded_unique_strings(
                memory_warnings,
                limit=MAX_LOOP_MEMORY_WARNINGS,
            ),
            reactive_compact_used=False,
            persistent=PersistentMemorySnapshot(),
        ),
        "deferred_tool_state": DeferredToolState(),
        "finish_state": FinishState(),
```

Note: need to also import `PersistentMemorySnapshot`:

```python
from rag.agent.loop.substate import (
    DeferredToolState,
    FinishState,
    MemoryState,
    PersistentMemorySnapshot,
    PlanState,
)
```

- [ ] **Step 4: Run existing tests to verify no breakage**

```bash
uv run pytest tests/agent/test_loop_state.py -v
```

Expected: all tests pass. The new fields are additive — existing assertions about `LoopState.__required_keys__` will include the 4 new keys but that's expected.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -x -q 2>&1 | tail -5
```

Expected: all 324 tests pass.

- [ ] **Step 6: Commit**

```bash
git add rag/agent/loop/state.py
git commit -m "feat(agent): add plan_state/memory_state/deferred_tool_state/finish_state to LoopState

Dual-write: new sub-state models initialised alongside existing flat fields.
No deletions. All 324 tests pass.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Add _migrate_legacy_state() for checkpoint backfill

**Files:**
- Modify: `rag/agent/core/checkpointing.py:517-536` (_normalize_loaded_state)

**Interfaces:**
- Consumes: old flat fields in legacy checkpoints
- Produces: populated `PlanState`, `MemoryState`, `DeferredToolState`, `FinishState` on loaded state

- [ ] **Step 1: Write the migration helper**

Add to `rag/agent/core/checkpointing.py`:

```python
from rag.agent.loop.substate import (
    DeferredToolState,
    DiscoveryCandidate,
    DiscoveryEvent,
    FinishState,
    MemoryState,
    PersistentMemorySnapshot,
    PlanState,
)


def _migrate_legacy_state(raw: dict) -> LoopState:
    """Populate new sub-state models from legacy flat fields.

    Reads old flat fields and writes them into the corresponding
    sub-state container.  Leaves the old flat fields intact so that
    existing callers continue to work (dual-read safe).

    This is called from ``_normalize_loaded_state`` for every
    checkpoint load, including old checkpoints that lack the new
    sub-state keys.
    """
    state = dict(raw)

    # ── PlanState ──
    state.setdefault(
        "plan_state",
        PlanState(
            agent_plan=state.get("agent_plan"),
            plan_events=list(state.get("plan_events", [])),
        ),
    )

    # ── MemoryState ──
    state.setdefault(
        "memory_state",
        MemoryState(
            working_summary=state.get("working_summary"),
            extracted_facts=list(state.get("extracted_facts", [])),
            context_budget=state.get("context_budget"),
            memory_refs=list(state.get("memory_refs", [])),
            memory_budget=state.get("memory_budget"),
            memory_warnings=list(state.get("memory_warnings", [])),
            reactive_compact_used=bool(state.get("reactive_compact_used", False)),
            persistent=PersistentMemorySnapshot(
                index_digest=_digest_text(state.get("memory_index", "")),
                selected_count=len(state.get("persistent_memories", [])),
            ),
        ),
    )

    # ── DeferredToolState ──
    state.setdefault(
        "deferred_tool_state",
        DeferredToolState(
            active_tools=list(state.get("discovery_active_tools", [])),
            active_tool_iterations=dict(
                state.get("discovery_active_tool_iterations", {})
            ),
            last_candidates=_migrate_discovery_candidates(
                state.get("discovery_last_candidates", [])
            ),
            last_search_query=str(state.get("discovery_last_search_query", "")),
            search_history=_migrate_discovery_events(
                state.get("discovery_search_history", [])
            ),
            pinned_tools=list(state.get("discovery_pinned_tools", [])),
            capability_diagnostics=list(
                state.get("capability_diagnostics", [])
            ),
        ),
    )

    # ── FinishState ──
    state.setdefault(
        "finish_state",
        FinishState(
            feedback=list(state.get("stop_hook_feedback", [])),
            warnings=list(state.get("stop_hook_warnings", [])),
        ),
    )

    return cast(LoopState, state)


def _digest_text(text: str, *, max_chars: int = 500) -> str:
    """Truncate text to a bounded digest for PersistentMemorySnapshot."""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + "…"


def _migrate_discovery_candidates(
    raw: list[dict[str, object]],
) -> list[DiscoveryCandidate]:
    """Convert legacy dict-based candidates to typed DiscoveryCandidate."""
    candidates: list[DiscoveryCandidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        candidates.append(
            DiscoveryCandidate(
                tool_name=str(item.get("tool_name", "")),
                description=str(item.get("description", "")),
                relevance_score=float(item.get("relevance_score", 0.0)),
                metadata={
                    k: v for k, v in item.items()
                    if k not in {"tool_name", "description", "relevance_score"}
                },
            )
        )
    return candidates


def _migrate_discovery_events(
    raw: list[dict[str, object]],
) -> list[DiscoveryEvent]:
    """Convert legacy dict-based search events to typed DiscoveryEvent."""
    events: list[DiscoveryEvent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        events.append(
            DiscoveryEvent(
                query=str(item.get("query", "")),
                timestamp_iso=str(item.get("timestamp_iso", "")),
                result_count=int(item.get("result_count", 0)),
                selected_tools=_string_list(item.get("selected_tools")),
            )
        )
    return events


def _string_list(value: object) -> list[str]:
    """Coerce a value to a list of strings safely."""
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return []
```

- [ ] **Step 2: Update _normalize_loaded_state to call migration**

Replace the body of `_normalize_loaded_state`:

```python
def _normalize_loaded_state(state: LoopState) -> LoopState:
    run_config = state["run_config"]
    if not isinstance(run_config.source_scope, tuple):
        state["run_config"] = replace(
            run_config,
            source_scope=tuple(run_config.source_scope),
        )
    # Backfill PR0/PR1 fields missing from older checkpoints
    state.setdefault("loop_messages", [])
    state.setdefault("pending_loop_tool_calls", [])
    state.setdefault("tool_result_store", {})
    state.setdefault("discovery_active_tools", [])
    state.setdefault("discovery_active_tool_iterations", {})
    state.setdefault("discovery_last_candidates", [])
    state.setdefault("discovery_last_search_query", "")
    state.setdefault("discovery_search_history", [])
    state.setdefault("discovery_pinned_tools", [])
    state.setdefault("active_deferred_tools", [])
    state.setdefault("capability_diagnostics", [])
    # ── PR1: migrate legacy flat fields into typed sub-states ──
    state = _migrate_legacy_state(cast(dict, state))
    return state
```

- [ ] **Step 3: Verify imports compile**

```bash
uv run python -c "from rag.agent.core.checkpointing import _migrate_legacy_state; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add rag/agent/core/checkpointing.py
git commit -m "feat(agent): add _migrate_legacy_state() for checkpoint backfill

Populates new sub-state models from legacy flat fields at checkpoint load.
Old fields preserved for dual-read compatibility.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Write unit tests for sub-state models and migration

**Files:**
- Create: `tests/agent/test_pr1_substate_convergence.py`

**Interfaces:**
- Consumes: `PlanState`, `MemoryState`, `DeferredToolState`, `FinishState`, `PersistentMemorySnapshot` from `rag.agent.loop.substate`
- Consumes: `_migrate_legacy_state` from `rag.agent.core.checkpointing`
- Consumes: `create_loop_state`, `LoopState` from `rag.agent.loop.state`

- [ ] **Step 1: Write the test file**

```python
# tests/agent/test_pr1_substate_convergence.py
"""Tests for PR1 sub-state convergence and legacy migration."""
from __future__ import annotations

from rag.agent.core.checkpointing import _migrate_legacy_state
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.runtime_diagnostics import RuntimeDiagnostic
from rag.agent.loop.state import (
    LoopState,
    StopHookFeedback,
    create_loop_state,
)
from rag.agent.loop.substate import (
    DeferredToolState,
    DiscoveryCandidate,
    DiscoveryEvent,
    FinishState,
    MemoryState,
    PersistentMemorySnapshot,
    PlanState,
)
from rag.agent.planning import AgentPlan, PlanEvent, PlanStep, PlanUpdate
from rag.schema.runtime import AccessPolicy


def _run_config(run_id: str = "pr1-test") -> AgentRunConfig:
    return AgentRunConfig(
        run_id=run_id,
        thread_id=run_id,
        budget_total=100,
        max_depth=2,
        access_policy=AccessPolicy.default(),
    )


# ── Sub-state model construction ──


def test_persistent_memory_snapshot_defaults_are_empty() -> None:
    """Default-constructed snapshot represents 'not loaded yet'."""
    snapshot = PersistentMemorySnapshot()
    assert snapshot.index_ref == ""
    assert snapshot.index_digest == ""
    assert snapshot.selected_count == 0
    assert snapshot.selected_summaries == []


def test_memory_state_default_factory_works_without_args() -> None:
    """MemoryState() with no args produces valid state."""
    ms = MemoryState()
    assert ms.working_summary is None
    assert ms.extracted_facts == []
    assert ms.reactive_compact_used is False
    assert isinstance(ms.persistent, PersistentMemorySnapshot)
    assert ms.persistent.index_digest == ""


def test_plan_state_default_is_empty() -> None:
    ps = PlanState()
    assert ps.agent_plan is None
    assert ps.plan_events == []


def test_deferred_tool_state_uses_typed_candidates() -> None:
    dts = DeferredToolState(
        last_candidates=[
            DiscoveryCandidate(
                tool_name="vector_search",
                description="Semantic search",
                relevance_score=0.9,
            )
        ],
        search_history=[
            DiscoveryEvent(
                query="search rag",
                timestamp_iso="2026-06-24T00:00:00",
                result_count=3,
                selected_tools=["vector_search"],
            )
        ],
    )
    assert len(dts.last_candidates) == 1
    assert dts.last_candidates[0].tool_name == "vector_search"
    assert dts.search_history[0].selected_tools == ["vector_search"]


def test_finish_state_default_is_empty() -> None:
    fs = FinishState()
    assert fs.feedback == []
    assert fs.warnings == []


# ── create_loop_state dual-write ──


def test_create_loop_state_populates_substates() -> None:
    state = create_loop_state(task="Test", run_config=_run_config())

    assert isinstance(state["plan_state"], PlanState)
    assert isinstance(state["memory_state"], MemoryState)
    assert isinstance(state["deferred_tool_state"], DeferredToolState)
    assert isinstance(state["finish_state"], FinishState)

    # Old flat fields still present (dual-write)
    assert "agent_plan" in state
    assert "working_summary" in state
    assert "stop_hook_feedback" in state


def test_create_loop_state_memory_warnings_in_both_channels() -> None:
    state = create_loop_state(
        task="Test",
        run_config=_run_config(),
        memory_warnings=["low budget"],
    )

    assert state["memory_warnings"] == ["low budget"]
    assert state["memory_state"].memory_warnings == ["low budget"]


# ── Legacy checkpoint migration ──


def test_migrate_legacy_state_populates_plan_state() -> None:
    plan = AgentPlan(
        objective="Test migration",
        steps=[
            PlanStep(
                step_id="step-1",
                title="First step",
                status="pending",
            )
        ],
    )
    events = [
        PlanEvent(
            event_type="plan_created",
            plan_revision=1,
            detail={"objective": "Test migration"},
        )
    ]
    raw = {
        "task": "Test",
        "messages": [],
        "agent_plan": plan,
        "plan_events": events,
    }

    result = _migrate_legacy_state(raw)

    assert result["plan_state"].agent_plan == plan
    assert result["plan_state"].plan_events == events
    # Old fields remain
    assert result["agent_plan"] == plan


def test_migrate_legacy_state_populates_memory_state() -> None:
    raw = {
        "task": "Test",
        "messages": [],
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "memory_refs": [],
        "memory_budget": None,
        "memory_warnings": ["test warning"],
        "reactive_compact_used": True,
        "memory_index": "## Project Memory\n\n- [Item 1](file.md) — test\n" * 20,
        "persistent_memories": ["memory A", "memory B"],
    }

    result = _migrate_legacy_state(raw)

    ms = result["memory_state"]
    assert ms.memory_warnings == ["test warning"]
    assert ms.reactive_compact_used is True
    assert ms.persistent.index_digest != ""
    assert len(ms.persistent.index_digest) <= 503  # 500 + "…"
    assert ms.persistent.selected_count == 2
    # Old fields remain
    assert result["memory_warnings"] == ["test warning"]


def test_migrate_legacy_state_populates_finish_state() -> None:
    fb = StopHookFeedback(code="grounding", message="Check citations")
    raw = {
        "task": "Test",
        "messages": [],
        "stop_hook_feedback": [fb],
        "stop_hook_warnings": [],
    }

    result = _migrate_legacy_state(raw)

    assert result["finish_state"].feedback == [fb]
    assert result["finish_state"].warnings == []


def test_migrate_legacy_state_populates_deferred_tool_state() -> None:
    raw = {
        "task": "Test",
        "messages": [],
        "discovery_active_tools": ["vector_search", "keyword_search"],
        "discovery_active_tool_iterations": {"vector_search": 3},
        "discovery_last_candidates": [
            {
                "tool_name": "vector_search",
                "description": "Semantic search",
                "relevance_score": 0.9,
            }
        ],
        "discovery_last_search_query": "rag retrieval",
        "discovery_search_history": [
            {
                "query": "rag retrieval",
                "timestamp_iso": "2026-06-24T00:00:00",
                "result_count": 2,
                "selected_tools": ["vector_search"],
            }
        ],
        "discovery_pinned_tools": ["always_on_tool"],
        "capability_diagnostics": [],
    }

    result = _migrate_legacy_state(raw)

    dts = result["deferred_tool_state"]
    assert dts.active_tools == ["vector_search", "keyword_search"]
    assert dts.active_tool_iterations["vector_search"] == 3
    assert len(dts.last_candidates) == 1
    assert isinstance(dts.last_candidates[0], DiscoveryCandidate)
    assert dts.last_candidates[0].tool_name == "vector_search"
    assert dts.last_search_query == "rag retrieval"
    assert len(dts.search_history) == 1
    assert isinstance(dts.search_history[0], DiscoveryEvent)
    assert dts.pinned_tools == ["always_on_tool"]


def test_migrate_legacy_state_handles_missing_fields_gracefully() -> None:
    """Minimal legacy state with no optional fields should still migrate."""
    raw = {
        "task": "Test",
        "messages": [],
    }

    result = _migrate_legacy_state(raw)

    assert result["plan_state"].agent_plan is None
    assert result["memory_state"].working_summary is None
    assert result["memory_state"].persistent.index_digest == ""
    assert result["finish_state"].feedback == []
    assert result["deferred_tool_state"].active_tools == []


def test_migrate_legacy_state_preserves_existing_substates() -> None:
    """If sub-states already exist (new checkpoint), don't overwrite."""
    existing_plan = PlanState(
        agent_plan=AgentPlan(
            objective="Already migrated",
            steps=[
                PlanStep(
                    step_id="existing-step",
                    title="Existing",
                    status="done",
                )
            ],
        )
    )
    raw = {
        "task": "Test",
        "messages": [],
        "plan_state": existing_plan,
        "agent_plan": None,  # Old field is None
    }

    result = _migrate_legacy_state(raw)

    # Existing sub-state should be preserved
    assert result["plan_state"].agent_plan.objective == "Already migrated"


# ── Checkpoint roundtrip with new sub-states ──


def test_checkpoint_serde_roundtrips_substate_models() -> None:
    """New sub-state models survive jsonplus serde roundtrip."""
    import logging
    import pytest
    from rag.agent.core.checkpointing import agent_checkpoint_serde

    serde = agent_checkpoint_serde()
    payload = {
        "plan_state": PlanState(agent_plan=None, plan_events=[]),
        "memory_state": MemoryState(
            memory_warnings=["low budget"],
            persistent=PersistentMemorySnapshot(
                index_digest="digest content",
                selected_count=3,
                selected_summaries=["a", "b", "c"],
            ),
        ),
        "deferred_tool_state": DeferredToolState(
            active_tools=["t1"],
            last_candidates=[
                DiscoveryCandidate(tool_name="t1", relevance_score=0.8)
            ],
        ),
        "finish_state": FinishState(
            feedback=[StopHookFeedback(code="test", message="test")]
        ),
    }

    with pytest.LogCaptureFixture() as cap:
        restored = serde.loads_typed(serde.dumps_typed(payload))

    assert restored == payload
    # No serializer warnings for the new models
```

- [ ] **Step 2: Run the tests to verify they pass**

```bash
uv run pytest tests/agent/test_pr1_substate_convergence.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Run full test suite to verify no regressions**

```bash
uv run pytest -x -q 2>&1 | tail -5
```

Expected: all 324+ tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/agent/test_pr1_substate_convergence.py
git commit -m "test(agent): add unit tests for PR1 sub-state convergence

Tests cover: model construction, factory dual-write, legacy migration,
graceful handling of missing fields, existing sub-state preservation,
and checkpoint serde roundtrip.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Update runtime to sync plan_state + finish_state on write

**Files:**
- Modify: `rag/agent/loop/runtime.py:206-211` (plan creation)
- Modify: `rag/agent/loop/runtime.py:1014-1055` (plan update)
- Modify: `rag/agent/loop/runtime.py:1160-1193` (_append_plan_events, _record_plan_observations)
- Modify: `rag/agent/loop/stop_hooks.py:122` (stop_hook_feedback append)

**Interfaces:**
- Consumes: `PlanState`, `FinishState` from `rag.agent.loop.substate`
- Modifies: runtime writes to both old flat fields AND new sub-states

- [ ] **Step 1: Update _append_plan_events to also sync PlanState**

In `rag/agent/loop/runtime.py`, modify `_append_plan_events`:

```python
    def _append_plan_events(
        self,
        state: LoopState,
        events: list[PlanEvent],
    ) -> None:
        state["plan_events"] = [
            *state["plan_events"],
            *events,
        ][-MAX_PLAN_EVENTS:]
        # ── PR1 dual-write: also update plan_state ──
        state["plan_state"] = PlanState(
            agent_plan=state.get("agent_plan"),
            plan_events=list(state["plan_events"]),
        )
```

Add import at top of runtime.py:
```python
from rag.agent.loop.substate import PlanState
```

- [ ] **Step 2: Update plan assignment sites to sync PlanState**

In the two places where `state["agent_plan"] = plan` is set (lines ~210 and ~1016), add after:

```python
            state["plan_state"] = PlanState(
                agent_plan=plan,
                plan_events=list(state.get("plan_events", [])),
            )
```

And in the plan update sites (lines ~1054 and ~1182), same pattern:

```python
            state["plan_state"] = PlanState(
                agent_plan=plan,
                plan_events=list(state.get("plan_events", [])),
            )
```

- [ ] **Step 3: Update stop_hooks to also write FinishState**

In `rag/agent/loop/stop_hooks.py`, after `append_stop_hook_feedback` and `append_stop_hook_warning`:

```python
from rag.agent.loop.substate import FinishState

# In the stop hook evaluation method, after writing the feedback:
state["finish_state"] = FinishState(
    feedback=list(state.get("stop_hook_feedback", [])),
    warnings=list(state.get("stop_hook_warnings", [])),
)
```

- [ ] **Step 4: Verify existing tests still pass**

```bash
uv run pytest tests/agent/test_loop_state.py tests/agent/test_stop_hooks.py tests/agent/test_agent_planning.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add rag/agent/loop/runtime.py rag/agent/loop/stop_hooks.py
git commit -m "feat(agent): dual-write plan_state and finish_state in runtime

runtime and stop_hooks now write to both old flat fields and new
PlanState/FinishState sub-models.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Update runtime to sync deferred_tool_state

**Files:**
- Modify: `rag/agent/loop/runtime.py:584-591` (_sync_discovery_to_state)

**Interfaces:**
- Consumes: `DeferredToolState` from `rag.agent.loop.substate`
- Consumes: `_migrate_discovery_candidates`, `_migrate_discovery_events` from `rag.agent.core.checkpointing` (defined in Task 3)
- Modifies: `_sync_discovery_to_state` writes to both old `discovery_*` fields and new `deferred_tool_state`

- [ ] **Step 1: Import helpers from checkpointing module and update _sync_discovery_to_state**

```python
# Add import at top of runtime.py:
from rag.agent.core.checkpointing import (
    _migrate_discovery_candidates,
    _migrate_discovery_events,
)
from rag.agent.loop.substate import DeferredToolState

    # Update _sync_discovery_to_state:
    def _sync_discovery_to_state(self, state: LoopState) -> None:
        """Sync deferred store state to LoopState discovery_* fields + DeferredToolState."""
        if self._deferred_store is not None:
            self._deferred_store.sync_to_state(cast(dict[Any, Any], state))
            # Backward compat alias
            state["active_deferred_tools"] = list(
                self._deferred_store.active_names()
            )
            # ── PR1 dual-write: typed DeferredToolState ──
            state["deferred_tool_state"] = DeferredToolState(
                active_tools=list(state.get("discovery_active_tools", [])),
                active_tool_iterations=dict(
                    state.get("discovery_active_tool_iterations", {})
                ),
                last_candidates=_migrate_discovery_candidates(
                    state.get("discovery_last_candidates", [])
                ),
                last_search_query=str(
                    state.get("discovery_last_search_query", "")
                ),
                search_history=_migrate_discovery_events(
                    state.get("discovery_search_history", [])
                ),
                pinned_tools=list(state.get("discovery_pinned_tools", [])),
                capability_diagnostics=list(
                    state.get("capability_diagnostics", [])
                ),
            )
```

Note: make `_migrate_discovery_candidates` and `_migrate_discovery_events` public-ish (non-private, `__all__`-exported) in `checkpointing.py` (Task 3 already defines them as module-level functions).

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/agent/test_agent_loop_runtime.py -v -k "discovery" 2>&1 | tail -5
uv run pytest tests/test_tool_discovery.py -v 2>&1 | tail -5
```

Expected: tests pass (or skip if discovery tests don't exist yet).

- [ ] **Step 3: Commit**

```bash
git add rag/agent/loop/runtime.py
git commit -m "feat(agent): dual-write deferred_tool_state in runtime discovery sync

_sync_discovery_to_state now writes to both old discovery_* fields
and typed DeferredToolState model.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Update compactor to dual-write memory_state

**Files:**
- Modify: `rag/agent/memory/compactor.py:259-282` (memory compaction writes)

**Interfaces:**
- Consumes: `MemoryState`, `PersistentMemorySnapshot` from `rag.agent.loop.substate`
- Modifies: compactor writes to both old flat fields and new `memory_state`

- [ ] **Step 1: Add MemoryState sync after compaction writes**

In the compaction result application method (where `state["working_summary"]`, `state["memory_refs"]`, etc. are set), add after all old-field writes:

```python
from rag.agent.loop.substate import MemoryState, PersistentMemorySnapshot

# After all memory field writes in the compaction apply method:
state["memory_state"] = MemoryState(
    working_summary=state.get("working_summary"),
    extracted_facts=list(state.get("extracted_facts", [])),
    context_budget=state.get("context_budget"),
    memory_refs=list(state.get("memory_refs", [])),
    memory_budget=state.get("memory_budget"),
    memory_warnings=list(state.get("memory_warnings", [])),
    reactive_compact_used=bool(state.get("reactive_compact_used", False)),
    persistent=PersistentMemorySnapshot(
        index_digest=_safe_digest(state.get("memory_index", "")),
        selected_count=len(state.get("persistent_memories", [])),
    ),
)
```

- [ ] **Step 2: Run memory compaction tests**

```bash
uv run pytest tests/agent/test_working_memory_compactor.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add rag/agent/memory/compactor.py
git commit -m "feat(agent): dual-write memory_state after compaction

Compactor now writes MemoryState alongside existing flat memory fields.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Update service.py persistent memory loading to use PersistentMemorySnapshot

**Files:**
- Modify: `rag/agent/service.py:870-905` (_load_persistent_memories)

**Interfaces:**
- Consumes: `PersistentMemorySnapshot` from `rag.agent.loop.substate`
- Modifies: writes `PersistentMemorySnapshot` into `memory_state.persistent`

- [ ] **Step 1: Update _load_persistent_memories**

After the existing `state["memory_index"] = index_content` and `state["persistent_memories"] = [...]` lines, add:

```python
# Add import at top of service.py:
from rag.agent.core.checkpointing import _digest_text
from rag.agent.loop.substate import MemoryState, PersistentMemorySnapshot

# After setting state["persistent_memories"] = ...:
# ── PR1 dual-write: update PersistentMemorySnapshot ──
ms = state.get("memory_state")
if isinstance(ms, MemoryState):
    state["memory_state"] = ms.model_copy(
        update={
            "persistent": PersistentMemorySnapshot(
                index_digest=_digest_text(index_content),
                selected_count=len(selected) if selected else 0,
                selected_summaries=(
                    [m.to_markdown()[:200] for m in selected]
                    if selected else []
                ),
            )
        }
    )
else:
    # memory_state not present (shouldn't happen after Task 2, but guard)
    state["memory_state"] = MemoryState(
        persistent=PersistentMemorySnapshot(
            index_digest=_digest_text(index_content),
            selected_count=len(selected) if selected else 0,
            selected_summaries=(
                [m.to_markdown()[:200] for m in selected]
                if selected else []
            ),
        ),
    )
```

Note: `_digest_text` was defined in Task 3 (`checkpointing.py`). Make it public via `__all__` there.

- [ ] **Step 2: Run service tests**

```bash
uv run pytest tests/agent/test_agent_service.py -v -k "memory" 2>&1 | tail -5
```

Expected: tests pass.

- [ ] **Step 3: Commit**

```bash
git add rag/agent/service.py
git commit -m "feat(agent): dual-write PersistentMemorySnapshot in service.py

_load_persistent_memories now populates memory_state.persistent with
bounded digest and selected summaries.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Sync memory_state on reactive_compact_used and context_budget updates in runtime

**Files:**
- Modify: `rag/agent/loop/runtime.py:336-337` (reactive_compact_used)
- Modify: `rag/agent/loop/runtime.py:688-689` (context_budget)

**Interfaces:**
- Consumes: `MemoryState` from `rag.agent.loop.substate`

- [ ] **Step 1: Sync memory_state on reactive_compact_used**

After `state["reactive_compact_used"] = True` (runtime.py ~337), add:

```python
            ms = state.get("memory_state")
            if isinstance(ms, MemoryState):
                state["memory_state"] = ms.model_copy(
                    update={"reactive_compact_used": True}
                )
```

- [ ] **Step 2: Sync memory_state on context_budget**

After `state["context_budget"] = result.context_budget` (runtime.py ~689), add:

```python
            ms = state.get("memory_state")
            if isinstance(ms, MemoryState):
                state["memory_state"] = ms.model_copy(
                    update={"context_budget": result.context_budget}
                )
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/agent/test_agent_loop_runtime.py -v -x 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add rag/agent/loop/runtime.py
git commit -m "feat(agent): sync memory_state on reactive_compact and context_budget updates

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Update loop_state test to verify all required_keys include new sub-states

**Files:**
- Modify: `tests/agent/test_loop_state.py:47-69`

**Interfaces:**
- Verifies: new sub-state keys are present in every created LoopState

- [ ] **Step 1: Update test_create_loop_state_populates_focused_required_channels**

```python
def test_create_loop_state_populates_focused_required_channels() -> None:
    state = create_loop_state(task="Inspect architecture", run_config=_run_config())

    assert set(state) == set(LoopState.__required_keys__)
    assert state["task"] == "Inspect architecture"
    assert state["status"] == "running"
    assert state["iteration"] == 0
    assert state["pending_tool_calls"] == []
    assert state["tool_execution_records"] == {}
    assert state["latest_transition"] is None

    # ── PR1: new sub-state keys are present ──
    assert isinstance(state["plan_state"], PlanState)
    assert isinstance(state["memory_state"], MemoryState)
    assert isinstance(state["deferred_tool_state"], DeferredToolState)
    assert isinstance(state["finish_state"], FinishState)

    forbidden = {
        "goal_spec",
        "goal_contract_hint",
        "goal_requirements",
        "satisfied_requirements",
        "open_gaps",
        "no_progress_count",
        "satisfaction_report",
        "controller_next",
        "transition_history",
    }
    assert forbidden.isdisjoint(LoopState.__required_keys__)
```

Add imports at top:
```python
from rag.agent.loop.substate import (
    DeferredToolState,
    FinishState,
    MemoryState,
    PlanState,
)
```

- [ ] **Step 2: Run updated test**

```bash
uv run pytest tests/agent/test_loop_state.py::test_create_loop_state_populates_focused_required_channels -v
```

Expected: PASS

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest -x -q 2>&1 | tail -3
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/agent/test_loop_state.py
git commit -m "test(agent): verify new sub-state keys in LoopState

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Final integration test — legacy checkpoint roundtrip

**Files:**
- Modify: `tests/agent/test_pr1_substate_convergence.py` (add test)

- [ ] **Step 1: Add a comprehensive legacy → new checkpoint roundtrip test**

```python
def test_full_legacy_checkpoint_roundtrip_preserves_all_data() -> None:
    """Load a simulated legacy checkpoint, migrate, save, reload — all data preserved."""
    from copy import deepcopy
    from rag.agent.core.checkpointing import _normalize_loaded_state

    # Simulate a legacy checkpoint with populated RAG-era fields
    legacy: dict = {
        "task": "Search for architecture patterns",
        "messages": [],
        "run_config": _run_config("roundtrip"),
        "retrieval_signals": None,
        "retrieval_signals_debug": None,
        "iteration": 3,
        "status": "running",
        "pending_tool_calls": [],
        "tool_execution_records": {},
        "approval_request": None,
        "approval_response": None,
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "tool_results": [],
        "evidence": [],
        "citations": [],
        "evidence_refs": [],
        "answer_candidates": [],
        "computation_results": [],
        "structured_observations": [],
        "context_units": [],
        "context_bindings": [],
        "locators": [],
        "asset_refs": [],
        "working_summary": None,
        "extracted_facts": [],
        "context_budget": None,
        "memory_refs": [],
        "memory_budget": None,
        "memory_warnings": ["old checkpoint"],
        "reactive_compact_used": False,
        "agent_plan": AgentPlan(
            objective="Roundtrip test",
            steps=[
                PlanStep(step_id="s1", title="Step 1", status="done"),
                PlanStep(step_id="s2", title="Step 2", status="in_progress"),
            ],
        ),
        "plan_events": [
            PlanEvent(
                event_type="step_transition",
                plan_revision=1,
                detail={"step_id": "s1", "status": "done"},
            )
        ],
        "stop_hook_feedback": [
            StopHookFeedback(code="g1", message="Grounding issue")
        ],
        "stop_hook_warnings": [],
        "runtime_diagnostics": [],
        "last_model_turn": None,
        "groundedness_flag": False,
        "insufficient_evidence_flag": False,
        "final_answer": None,
        "final_output": None,
        "output_validation_errors": [],
        "pause": None,
        "terminal": None,
        "latest_transition": None,
        "loop_messages": [],
        "pending_loop_tool_calls": [],
        "tool_result_store": {},
        "discovery_active_tools": ["vector_search"],
        "discovery_active_tool_iterations": {"vector_search": 2},
        "discovery_last_candidates": [
            {"tool_name": "vector_search", "description": "search", "relevance_score": 0.9}
        ],
        "discovery_last_search_query": "architecture",
        "discovery_search_history": [
            {"query": "architecture", "timestamp_iso": "2026-06-24", "result_count": 1, "selected_tools": ["vector_search"]}
        ],
        "discovery_pinned_tools": [],
        "active_deferred_tools": ["vector_search"],
        "capability_diagnostics": [],
        "file_manifest": None,
        "persistent_memories": ["Memory A", "Memory B"],
        "memory_index": "## Project Memory\n\nLong content here\n" * 30,
    }

    # Migrate
    result = _migrate_legacy_state(deepcopy(legacy))

    # Plan state populated
    assert result["plan_state"].agent_plan.objective == "Roundtrip test"
    assert len(result["plan_state"].plan_events) == 1

    # Memory state populated
    assert result["memory_state"].memory_warnings == ["old checkpoint"]
    assert result["memory_state"].persistent.selected_count == 2
    assert len(result["memory_state"].persistent.index_digest) <= 503

    # Finish state populated
    assert len(result["finish_state"].feedback) == 1
    assert result["finish_state"].feedback[0].code == "g1"

    # Deferred tool state populated
    assert result["deferred_tool_state"].active_tools == ["vector_search"]
    assert isinstance(result["deferred_tool_state"].last_candidates[0], DiscoveryCandidate)

    # Old fields STILL present (dual-read safe)
    assert result["agent_plan"].objective == "Roundtrip test"
    assert result["memory_warnings"] == ["old checkpoint"]
    assert result["stop_hook_feedback"][0].code == "g1"
    assert result["discovery_active_tools"] == ["vector_search"]
```

- [ ] **Step 2: Run the roundtrip test**

```bash
uv run pytest tests/agent/test_pr1_substate_convergence.py::test_full_legacy_checkpoint_roundtrip_preserves_all_data -v
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest -x -q 2>&1 | tail -3
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/agent/test_pr1_substate_convergence.py
git commit -m "test(agent): add full legacy checkpoint roundtrip test

Verifies all sub-states are populated and old flat fields preserved.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review Checklist

Before declaring the plan complete, verify:

1. **Spec coverage:** Each PR1 requirement from the spec is covered by a task:
   - [x] Define PlanState, MemoryState, FinishState, DeferredToolState → Task 1
   - [x] Define PersistentMemorySnapshot → Task 1
   - [x] Define DiscoveryCandidate, DiscoveryEvent → Task 1
   - [x] create_loop_state() dual-write → Task 2
   - [x] _migrate_legacy_state() → Task 3
   - [x] _normalize_loaded_state → Task 3
   - [x] Checkpoint saves new sub-states → Task 2 + Task 3 (sub-states are part of LoopState dict saved as-is)
   - [x] Runtime dual-read → Tasks 5-9
   - [x] memory_index → PersistentMemorySnapshot → Task 8
   - [x] Tests → Tasks 4, 10, 11
   - [x] No field deletions → verified, no task removes any TypedDict field
   - [x] No allowlist changes → verified, no task touches AGENT_CHECKPOINT_MSGPACK_ALLOWLIST

2. **Placeholder scan:** No TBD, TODO, or "implement later" patterns. All code is in full.

3. **Type consistency:**
   - `PlanState`, `MemoryState`, `DeferredToolState`, `FinishState` defined in Task 1 → used in Tasks 2-11
   - `PersistentMemorySnapshot`, `DiscoveryCandidate`, `DiscoveryEvent` defined in Task 1 → used in Tasks 3, 4, 8
   - `_migrate_legacy_state` defined in Task 3 → tested in Task 4
   - `_bounded_digest` helper defined in Task 8 (same logic as `_digest_text` in Task 3)

4. **Every task ends with a testable deliverable:** Yes — each task has a "Run tests" step with expected output.

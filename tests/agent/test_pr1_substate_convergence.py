"""Tests for PR1 sub-state convergence and legacy migration."""

from __future__ import annotations

import logging

import pytest

from agent_runtime.planning import AgentPlan, PlanEvent, PlanStep
from rag.agent.core.checkpointing import _migrate_legacy_state, agent_checkpoint_serde
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.observations import StructuredObservation
from rag.agent.loop.state import (
    StopHookFeedback,
    create_loop_state,
)
from rag.agent.loop.substate import (
    DeferredToolState,
    DiscoveryCandidate,
    DiscoveryEvent,
    FinishState,
    MemoryState,
    PlanState,
)


def _run_config(run_id: str = "pr1-test") -> AgentRunConfig:
    return AgentRunConfig(
        turn_id=run_id,
        llm_budget_total=100,
    )


# ── Sub-state model construction ──


def test_memory_state_default_factory_works_without_args() -> None:
    """MemoryState() with no args produces valid state."""
    ms = MemoryState()
    assert ms.working_summary is None
    assert ms.extracted_facts == []
    assert ms.recent_observations == []
    assert ms.known_locators == []
    assert ms.reactive_compact_used is False


def test_plan_state_default_is_empty() -> None:
    ps = PlanState()
    assert ps.agent_plan is None
    assert ps.plan_events == []


def test_deferred_tool_state_uses_typed_candidates() -> None:
    dts = DeferredToolState(
        last_candidates=[
            DiscoveryCandidate(
                name="vector_search",
                description="Semantic search",
                reason="high relevance",
            )
        ],
        search_history=[
            DiscoveryEvent(
                query="search rag",
                candidates=["vector_search"],
                activated=["vector_search"],
            )
        ],
    )
    assert len(dts.last_candidates) == 1
    assert dts.last_candidates[0].name == "vector_search"
    assert dts.search_history[0].candidates == ["vector_search"]


def test_finish_state_default_is_empty() -> None:
    fs = FinishState()
    assert fs.feedback == []
    assert fs.warnings == []


# ── create_loop_state dual-write ──


def test_create_loop_state_populates_substates() -> None:
    state = create_loop_state(current_message="Test", run_config=_run_config())

    assert isinstance(state["plan_state"], PlanState)
    assert isinstance(state["memory_state"], MemoryState)
    assert isinstance(state["deferred_tool_state"], DeferredToolState)
    assert isinstance(state["finish_state"], FinishState)

    # Sub-state containers are the sole source of truth
    assert state["plan_state"].agent_plan is None
    assert state["memory_state"].working_summary is None
    assert state["finish_state"].feedback == []


def test_create_loop_state_memory_warnings_in_both_channels() -> None:
    state = create_loop_state(
        current_message="Test",
        run_config=_run_config(),
        memory_warnings=["low budget"],
    )

    assert state["memory_state"].memory_warnings == ["low budget"]
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
            event_id="evt-1",
            event_type="initialized",
            plan_revision=1,
            message="Plan created for migration test",
            tool_call_ids=[],
            warnings=[],
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
    assert result["plan_state"].agent_plan == plan


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
    assert "memory_index" not in result
    assert "persistent_memories" not in result
    assert result["memory_state"].memory_warnings == ["test warning"]


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
                "name": "vector_search",
                "description": "Semantic search",
                "reason": "high relevance",
            }
        ],
        "discovery_last_search_query": "rag retrieval",
        "discovery_search_history": [
            {
                "query": "rag retrieval",
                "candidates": ["vector_search"],
                "activated": ["vector_search"],
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
    assert dts.last_candidates[0].name == "vector_search"
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
                    status="completed",
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


def test_checkpoint_serde_roundtrips_substate_models(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """New sub-state models survive jsonplus serde roundtrip."""
    serde = agent_checkpoint_serde()
    payload = {
        "plan_state": PlanState(agent_plan=None, plan_events=[]),
        "memory_state": MemoryState(
            memory_warnings=["low budget"],
            recent_observations=[
                StructuredObservation(
                    tool_call_id="tc-search",
                    tool_name="search_text",
                    status="ok",
                    raw_result_ref="tc-search",
                )
            ],
            known_locators=[
                {
                    "source_tool": "search_text",
                    "path": "rag/agent/loop/runtime.py",
                    "line_number": 718,
                }
            ],
        ),
        "deferred_tool_state": DeferredToolState(
            active_tools=["t1"],
            last_candidates=[DiscoveryCandidate(name="t1", reason="high score")],
        ),
        "finish_state": FinishState(feedback=[StopHookFeedback(code="test", message="test")]),
    }

    with caplog.at_level(
        logging.WARNING,
        logger="langgraph.checkpoint.serde.jsonplus",
    ):
        restored = serde.loads_typed(serde.dumps_typed(payload))

    assert restored == payload
    # No serializer warnings for the new models


def test_full_legacy_checkpoint_migrates_supported_state() -> None:
    """Retired fields are discarded while supported checkpoint state is retained."""
    from copy import deepcopy

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
                PlanStep(step_id="s1", title="Step 1", status="completed"),
                PlanStep(step_id="s2", title="Step 2", status="in_progress"),
            ],
        ),
        "plan_events": [
            PlanEvent(
                event_id="evt-1",
                event_type="completed",
                plan_revision=1,
                message="Step 1 completed",
                related_step_id="s1",
            )
        ],
        "stop_hook_feedback": [
            StopHookFeedback(code="g1", message="Grounding issue"),
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
            {"name": "vector_search", "description": "search", "reason": "relevant"},
        ],
        "discovery_last_search_query": "architecture",
        "discovery_search_history": [
            {
                "query": "architecture",
                "candidates": ["vector_search"],
                "activated": ["vector_search"],
            },
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
    assert "persistent_memories" not in result
    assert "memory_index" not in result

    # Finish state populated
    assert len(result["finish_state"].feedback) == 1
    assert result["finish_state"].feedback[0].code == "g1"

    # Deferred tool state populated
    assert result["deferred_tool_state"].active_tools == ["vector_search"]
    assert isinstance(result["deferred_tool_state"].last_candidates[0], DiscoveryCandidate)

    # Old fields STILL present (dual-read safe)
    assert result["plan_state"].agent_plan.objective == "Roundtrip test"
    assert result["memory_state"].memory_warnings == ["old checkpoint"]
    assert result["stop_hook_feedback"][0].code == "g1"
    assert result["deferred_tool_state"].active_tools == ["vector_search"]

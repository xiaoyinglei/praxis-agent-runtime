from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from agent_runtime.core.checkpointing import agent_checkpoint_serde
from agent_runtime.core.context import AgentRunConfig
from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic
from agent_runtime.core.turn_contracts import ToolCallPlan
from agent_runtime.loop.state import (
    MAX_LOOP_MEMORY_WARNINGS,
    MAX_STOP_HOOK_FEEDBACK,
    LoopPause,
    LoopState,
    LoopTerminal,
    LoopTransition,
    ModelTurn,
    ModelTurnDraft,
    StopHookFeedback,
    append_loop_diagnostic,
    append_memory_warning,
    append_stop_hook_feedback,
    create_loop_state,
    materialize_model_turn,
    replace_latest_transition,
)
from agent_runtime.loop.state import LoopState as AgentState
from agent_runtime.loop.state import create_loop_state as create_agent_state
from agent_runtime.loop.substate import (
    DeferredToolState,
    FinishState,
    MemoryState,
    PlanState,
)


def agent_state_to_loop_state(state: AgentState) -> AgentState:
    return state  # was identity function, now explicit


def _run_config(run_id: str = "loop-state") -> AgentRunConfig:
    return AgentRunConfig(
        turn_id=run_id,
        llm_budget_total=100,
    )


def test_create_loop_state_populates_focused_required_channels() -> None:
    state = create_loop_state(current_message="Inspect architecture", run_config=_run_config())

    assert set(state) == set(LoopState.__required_keys__)
    assert state["current_message"] == "Inspect architecture"
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


def test_loop_state_factory_copies_mutable_inputs() -> None:
    pending = [ToolCallPlan.create("vector_search", {"query": "loop state"})]
    warnings = ["initial"]
    state = create_loop_state(
        current_message="Inspect architecture",
        run_config=_run_config(),
        pending_tool_calls=pending,
        memory_warnings=warnings,
    )

    pending.clear()
    warnings.append("later")

    assert len(state["pending_tool_calls"]) == 1
    assert state["memory_state"].memory_warnings == ["initial"]


def test_model_turn_draft_accepts_explicit_finish_candidate() -> None:
    draft = ModelTurnDraft(
        action="finish",
        final_answer="Complete answer.",
    )

    assert materialize_model_turn(draft) == ModelTurn(
        action="finish",
        final_answer="Complete answer.",
    )


def test_tool_calls_take_precedence_over_finish_intent() -> None:
    call = ToolCallPlan.create("vector_search", {"query": "kernel"})
    draft = ModelTurnDraft(
        action="finish",
        final_answer="Too early",
        tool_calls=(call,),
    )

    turn = materialize_model_turn(draft)

    assert turn.action == "execute"
    assert turn.tool_calls == (call,)


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "execute"},
        {"action": "finish"},
        {"action": "finish", "final_answer": "   "},
        {"action": "pause"},
        {"action": "pause", "pause_reason": "   "},
    ],
)
def test_strict_model_turn_rejects_incomplete_outcomes(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ModelTurn.model_validate(payload)


def test_strict_model_turn_accepts_each_complete_outcome() -> None:
    call = ToolCallPlan.create("vector_search", {"query": "kernel"})

    assert ModelTurn(action="execute", tool_calls=(call,)).action == "execute"
    assert ModelTurn(action="finish", final_answer="Done").action == "finish"
    assert ModelTurn(action="pause", pause_reason="Need approval").action == "pause"


def test_replace_latest_transition_does_not_grow_history() -> None:
    state = create_loop_state(current_message="Inspect architecture", run_config=_run_config())

    first = LoopTransition(reason="next_turn", iteration=0)
    second = LoopTransition(reason="tool_execution", iteration=1)
    replace_latest_transition(state, first)
    replace_latest_transition(state, second)

    assert state["latest_transition"] == second
    assert "transition_history" not in state


def test_bounded_append_helpers_keep_only_recent_unique_values() -> None:
    state = create_loop_state(current_message="Inspect architecture", run_config=_run_config())

    for index in range(MAX_STOP_HOOK_FEEDBACK + 3):
        append_stop_hook_feedback(
            state,
            StopHookFeedback(code=f"hook_{index}", message=f"feedback {index}"),
        )
    for index in range(MAX_LOOP_MEMORY_WARNINGS + 3):
        append_memory_warning(state, f"warning {index}")
    append_memory_warning(state, "warning 2")
    for index in range(25):
        append_loop_diagnostic(
            state,
            RuntimeDiagnostic(
                code=f"diagnostic_{index}",
                component="loop",
                message=f"diagnostic {index}",
            ),
        )

    assert len(state["finish_state"].feedback) == MAX_STOP_HOOK_FEEDBACK
    assert state["finish_state"].feedback[0].code == "hook_3"
    assert len(state["memory_state"].memory_warnings) == MAX_LOOP_MEMORY_WARNINGS
    assert state["memory_state"].memory_warnings[-1] == "warning 2"
    assert len(state["runtime_diagnostics"]) == 20


def test_agent_state_is_a_compatibility_alias_for_loop_state() -> None:
    state = create_agent_state(
        current_message="Inspect architecture",
        run_config=_run_config("legacy-adapter"),
    )

    adapted = agent_state_to_loop_state(state)

    assert AgentState is LoopState
    assert set(state) == set(LoopState.__required_keys__)
    assert adapted is state


def test_new_checkpoint_models_round_trip_without_unregistered_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = agent_checkpoint_serde()
    payload = {
        "draft": ModelTurnDraft(action="finish"),
        "turn": ModelTurn(action="finish", final_answer="Done"),
        "transition": LoopTransition(reason="finished", iteration=2),
        "pause": LoopPause(reason="Need approval"),
        "terminal": LoopTerminal(
            status="completed",
            stop_reason="model_finished",
            final_answer="Done",
        ),
        "feedback": StopHookFeedback(code="grounding", message="Check citations"),
    }

    with caplog.at_level(
        logging.WARNING,
        logger="langgraph.checkpoint.serde.jsonplus",
    ):
        restored = serde.loads_typed(serde.dumps_typed(payload))

    assert restored == payload
    assert "not allowed" not in caplog.text.lower()

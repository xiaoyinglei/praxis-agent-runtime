from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_runtime.result import AgentResult
from agent_runtime.runtime.builder import build_agent_service
from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.workspace import open_workspace

_PLAN_ARGUMENTS = {
    "explanation": "Implementation is ready; verification is next.",
    "plan": [
        {
            "step": "Implement durable plan state",
            "status": "completed",
        },
        {
            "step": "Run integration verification",
            "status": "in_progress",
        },
    ],
}


class _UpdatePlanThenPauseProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if not state["tool_results"]:
            active_plan = state["plan_state"].agent_plan
            assert active_plan is not None
            assert active_plan.objective
            return ModelTurnDraft(
                action="execute",
                tool_calls=(
                    ToolCallPlan(
                        tool_call_id="call_update_plan",
                        tool_name="update_plan",
                        arguments=_PLAN_ARGUMENTS,
                    ),
                ),
            )
        return ModelTurnDraft(
            action="pause",
            pause_reason="Inspect the persisted plan.",
        )


class _UnusedModelRegistry:
    default_model = "unused"

    def resolve_for_node(self, **_kwargs: object) -> object:
        raise AssertionError("the injected model-turn provider must be used")


def _plan_service(
    tmp_path: Path,
) -> tuple[AgentService, AgentRuntimePolicy, MemorySaver]:
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    service = build_agent_service(
        open_workspace(tmp_path / "workspace", create=True),
        checkpointer=checkpointer,
        model_control_plane=_UnusedModelRegistry(),  # type: ignore[arg-type]
    )
    service._model_turn_provider = _UpdatePlanThenPauseProvider()
    definition = service._policy
    return service, definition, checkpointer


@pytest.mark.anyio
async def test_update_plan_is_canonical_result_and_checkpoint_state(
    tmp_path: Path,
) -> None:
    service, definition, checkpointer = _plan_service(tmp_path)
    request = AgentRunRequest(
        message="Make update_plan durable.",
        turn_id="turn-plan-result",
    )

    result = await service.run(request)

    assert result.status == "paused"
    assert result.plan is not None
    assert result.plan.objective == request.message
    assert result.plan.summary == _PLAN_ARGUMENTS["explanation"]
    assert [step.title for step in result.plan.steps] == [
        "Implement durable plan state",
        "Run integration verification",
    ]
    assert [step.status for step in result.plan.steps] == [
        "pending",
        "in_progress",
    ]
    assert result.plan.revision == 1
    assert result.plan.active_step_id == "step_002"
    update_event = next(
        event
        for event in result.plan_events
        if event.event_type == "llm_update"
    )
    assert update_event.warnings == ["unverified_completion_ignored"]
    update_result = next(item for item in result.tool_results if item.tool_name == "update_plan")
    assert update_result.structured_content == {
        "accepted": True,
        "revision": result.plan.revision,
        "message": (
            "Plan persisted as advisory state; tool policy and stop hooks "
            "retain execution and completion authority."
        ),
        "authority": "advisory",
    }

    restored = await LangGraphCheckpointStore(
        checkpointer,
        run_config=request.to_run_config(definition),
    ).load_latest()
    assert restored is not None
    assert restored["plan_state"].agent_plan == result.plan
    assert restored["plan_state"].plan_events == result.plan_events

    public = AgentResult._from_internal(result)
    assert public.plan == result.plan
    assert public.plan_events == tuple(result.plan_events)


@pytest.mark.anyio
async def test_update_plan_emits_complete_plan_snapshot_on_stream(
    tmp_path: Path,
) -> None:
    service, _definition, _checkpointer = _plan_service(tmp_path)

    events = [
        event
        async for event in service.run_streaming(
            AgentRunRequest(
                message="Stream the durable plan.",
                turn_id="turn-plan-stream",
            )
        )
    ]

    plan_event = next(event for event in events if event.type.value == "plan_updated")
    assert plan_event.turn_id == "turn-plan-stream"
    assert not hasattr(plan_event, "session_id")
    assert plan_event.iteration == 1
    assert plan_event.sequence > 0
    assert plan_event.data["plan"]["summary"] == _PLAN_ARGUMENTS["explanation"]
    assert "goal_id" not in plan_event.data["plan"]
    assert "goal_requirement_ids" not in plan_event.data["plan"]
    assert "target_files" not in plan_event.data["plan"]
    assert "hypothesis" not in plan_event.data["plan"]
    assert plan_event.data["plan"]["active_step_id"] == "step_002"
    assert [step["status"] for step in plan_event.data["plan"]["steps"]] == [
        "pending",
        "in_progress",
    ]
    assert plan_event.data["event"]["event_type"] == "llm_update"

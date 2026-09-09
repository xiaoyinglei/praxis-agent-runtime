from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent_runtime.agent import Agent
from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness import (
    HarnessModelRequest,
    HarnessModelResponse,
    HarnessToolCall,
    ItemSnapshot,
    PreparedModelCall,
    RolloutStore,
)
from agent_runtime.result import AgentResult

_PLAN_ARGUMENTS = {
    "explanation": "Implementation is ready; verification is next.",
    "plan": [
        {
            "step_id": "implement",
            "step": "Implement durable plan state",
            "status": "completed",
        },
        {
            "step_id": "verify",
            "step": "Run integration verification",
            "status": "in_progress",
        },
    ],
}


class _UpdatePlanThenAnswerModel:
    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {"model_id": "plan-test", "model_revision": "v1"}

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        digest = hashlib.sha256(
            f"{request.turn_id}:{request.step}".encode()
        ).hexdigest()
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash=toolset_revision_for_tools(request.tools),
            wire_hash=digest,
            request_ref={
                "request_id": f"{request.turn_id}:step:{request.step}",
                "toolset_revision": toolset_revision_for_tools(request.tools),
                "exposed_tool_names": [
                    tool.definition.name for tool in request.tools
                ],
                "step": request.step,
            },
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        step = int(prepared.request_ref["step"])
        if step == 1:
            return HarnessModelResponse(
                text="",
                provider_response_id="plan-call",
                usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                tool_calls=(
                    HarnessToolCall(
                        id="call-update-plan",
                        name="update_plan",
                        arguments=_PLAN_ARGUMENTS,
                    ),
                ),
            )
        return HarnessModelResponse(
            text="plan verified",
            provider_response_id="plan-answer",
            usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )


async def _run_plan_turn(
    tmp_path: Path,
) -> tuple[AgentResult, tuple[ItemSnapshot, ...]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    model = _UpdatePlanThenAnswerModel()
    agent._harness_model = lambda: model

    result = await agent.run(
        "Make update_plan durable.",
        require_workspace_change=False,
    )
    with RolloutStore(database) as store:
        items = tuple(store.list_items(result.turn_id))
    return result, items


@pytest.mark.anyio
async def test_harness_update_plan_projects_public_result_and_rollout_truth(
    tmp_path: Path,
) -> None:
    result, items = await _run_plan_turn(tmp_path)

    assert result.status == "done"
    assert result.answer == "plan verified"
    assert result.plan is not None
    assert result.plan.objective == "Make update_plan durable."
    assert result.plan.summary == _PLAN_ARGUMENTS["explanation"]
    assert result.plan.revision == 2
    assert result.plan.status == "complete"
    assert [step.title for step in result.plan.steps] == [
        "Implement durable plan state",
        "Run integration verification",
    ]
    assert [step.status for step in result.plan.steps] == [
        "completed",
        "completed",
    ]
    assert [event.event_type for event in result.plan_events] == [
        "llm_update",
        "completed",
    ]

    update_call = next(
        item
        for item in items
        if item.kind == "tool_call"
        and item.payload.get("tool_name") == "update_plan"
    )
    update_result = next(
        item
        for item in items
        if item.kind == "tool_result"
        and item.payload.get("tool_name") == "update_plan"
    )
    assert update_call.payload["arguments"] == _PLAN_ARGUMENTS
    assert update_result.payload["is_error"] is False


@pytest.mark.anyio
async def test_harness_update_plan_persists_each_canonical_plan_transition(
    tmp_path: Path,
) -> None:
    result, items = await _run_plan_turn(tmp_path)
    plan_items = [item for item in items if item.kind == "plan_state"]

    assert result.plan is not None
    assert len(plan_items) == 2
    assert [item.payload["plan"]["revision"] for item in plan_items] == [1, 2]
    assert plan_items[0].payload["plan"]["status"] == "active"
    assert plan_items[-1].payload["plan"] == result.plan.model_dump()


@pytest.mark.anyio
async def test_failed_turn_projects_the_canonical_plan_not_raw_tool_claims(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    agent = Agent(
        checkpoint_db=database,
        workspace_path=workspace,
        enable_workspace_mcp=False,
    )
    agent._harness_model = lambda: _UpdatePlanThenAnswerModel()

    result = await agent.run(
        "Make update_plan canonical.",
        max_turns=1,
        require_workspace_change=False,
    )
    with RolloutStore(database) as store:
        plan_item = [
            item for item in store.list_items(result.turn_id) if item.kind == "plan_state"
        ][-1]

    assert result.status == "failed"
    assert result.plan is not None
    assert result.plan.status == "blocked"
    assert [(step.step_id, step.status) for step in result.plan.steps] == [
        (step["step_id"], step["status"])
        for step in plan_item.payload["plan"]["steps"]
    ]

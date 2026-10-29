from __future__ import annotations

import pytest

from agent_runtime import Agent
from agent_runtime.harness import HarnessModelResponse, ModelToolCall
from tests.agent.test_update_plan_surfaces import _UpdatePlanThenAnswerModel


class ReadingModel(_UpdatePlanThenAnswerModel):
    def __init__(self, *, finish_at=None):
        self.requests = []
        self.finish_at = finish_at

    def prepare(self, request):
        self.requests.append(request)
        return super().prepare(request)

    async def dispatch(self, prepared, *, event_sink=None):
        step = prepared.request_ref["step"]
        if step == self.finish_at:
            return HarnessModelResponse.from_parts(text="finished", provider_response_id=None, usage={})
        return HarnessModelResponse.from_parts(text="", provider_response_id=None, usage={}, tool_calls=(
            ModelToolCall(id=f"read-{step}", name="read_file", arguments={"path": "example.py"}),
        ))


@pytest.mark.anyio
async def test_default_budget_allows_tasks_past_sixteen_steps(tmp_path):
    (tmp_path / "example.py").write_text("value = 1")
    model = ReadingModel(finish_at=18)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run("inspect and answer")
    assert result.status == "done"
    assert len(model.requests) == 18
    assert model.requests[0].binding_manifest["model_step_budget"] == 128


@pytest.mark.anyio
async def test_explicit_budget_has_pressure_and_durable_partial_receipt(tmp_path):
    (tmp_path / "example.py").write_text("value = 1")
    model = ReadingModel()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run("inspect and answer", max_steps=3)
    assert result.status == "failed"
    assert result.stop_reason == "model_step_budget_exhausted"
    assert len(model.requests) == 3
    assert result.answer and "3" in result.answer and "尚未完成" in result.answer
    assert "remaining" in "\n".join(message.content for message in model.requests[-1].messages)
    rebuilt = await agent.read_result(result.turn_id)
    assert rebuilt.answer == result.answer
    assert rebuilt.status == "failed"

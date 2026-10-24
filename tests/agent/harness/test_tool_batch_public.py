"""Native batch contracts through shipped SDK, CLI and wire serializer."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_runtime import Agent, cli
from agent_runtime.core.llm_registry import ResolvedModel
from agent_runtime.harness import GatewayHarnessModel, HarnessModelResponse, HarnessToolCall, RolloutContextManager
from agent_runtime.model_definition import ModelCapabilities, RequestDefaultsDefinition
from agent_runtime.modeling.config import GenerationConfig
from agent_runtime.modeling.openai_wire import serialize_openai_request
from tests.agent.harness.test_tool_orchestrator import ToolThenAnswerModel


class PublicBatchModel(ToolThenAnswerModel):
    async def dispatch(self, prepared, *, delta_sink=None):
        if not any(m.role == "tool" for m in self.requests[-1].messages):
            return HarnessModelResponse(text="", provider_response_id="batch", usage={}, tool_calls=(
                HarnessToolCall(id="invalid", name="read_file", arguments={}),
                HarnessToolCall(id="missing", name="read_file", arguments={"path": "absent.txt"}),
                HarnessToolCall(id="first", name="read_file", arguments={"path": "first.txt"}),
                HarnessToolCall(id="second", name="read_file", arguments={"path": "second.txt"}),
            ))
        return HarnessModelResponse(text="Two files inspected; invalid and absent inputs reported.",
                                    provider_response_id="done", usage={})


def _wire(request):
    adapter = GatewayHarnessModel(model_id="tool-model", resolved=ResolvedModel(
        generator=object(), gateway=object(), provider="openai-compatible", model_id="provider-model",
        capabilities=ModelCapabilities(context_window_tokens=20000, max_output_tokens=256,
                                       supports_native_tools=True, supports_structured_output=True),
        token_accounting=object(), request_defaults=RequestDefaultsDefinition(temperature=0.0),
        generation_config=GenerationConfig(),
    ), instructions=("Use tool evidence.",))
    prepared = adapter.prepare(request)
    return serialize_openai_request(prepared.dispatch_payload.request)


@pytest.mark.parametrize("entry", ["sdk", "cli"])
def test_native_batch_uses_same_semantics_across_public_entrypoints_and_new_turn(tmp_path: Path, monkeypatch, entry):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "first.txt").write_text("first evidence")
    (workspace / "second.txt").write_text("second evidence")
    monkeypatch.chdir(workspace)
    model = PublicBatchModel()
    agent = Agent(checkpoint_db=tmp_path / "state.db", workspace_path=workspace, model_session_path=None,
                  enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    if entry == "cli":
        monkeypatch.setattr(cli, "_create_agent_facade", lambda **kwargs: agent)
        output = CliRunner().invoke(cli.agent_app, ["run", "inspect records", "--non-interactive", "--no-web-tools"])
        assert output.exit_code == 0, output.output
    else:
        result = asyncio.run(agent.run("inspect records", require_workspace_change=False))
        assert result.status == "done"
    assert len(model.requests) == 2
    request = model.requests[-1]
    assert [m.tool_call_id for m in request.messages if m.role == "tool"] == ["invalid", "missing", "first", "second"]
    wire_before = _wire(request)
    assert [m["tool_call_id"] for m in wire_before.payload["messages"] if m["role"] == "tool"] == [
        "invalid", "missing", "first", "second",
    ]
    from agent_runtime.harness import RolloutStore

    with RolloutStore(agent.checkpoint_db) as store:
        results = [i.payload for i in store.list_items(request.turn_id) if i.kind == "tool_result"]
        by_id = {r["tool_call_id"]: r for r in results}
        assert by_id["invalid"]["error_code"] == "invalid_arguments"
        assert by_id["missing"]["is_error"] is True
        assert by_id["first"]["is_error"] is False and by_id["second"]["is_error"] is False
        # Reopen projects exactly the same group and can use the real serializer.
        from dataclasses import replace
        replayed = RolloutContextManager(store).build(request.turn_id)
        # The final assistant answer was committed after the request we compare.
        wire_after = _wire(replace(request, messages=replayed[:-1]))
        assert wire_after.serialized_json == wire_before.serialized_json
        assert store.verify().valid
    followup = asyncio.run(agent.run("explain those records", previous_turn_id=request.turn_id,
                                    require_workspace_change=False))
    assert followup.status == "done"
    assert len(model.requests) == 3
    assert [m.tool_call_id for m in model.requests[-1].messages if m.role == "tool"] == [
        "invalid", "missing", "first", "second",
    ]

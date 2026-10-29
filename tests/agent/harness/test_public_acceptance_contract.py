from __future__ import annotations

import asyncio

import pytest
from typer.testing import CliRunner

from agent_runtime import Agent, cli
from agent_runtime.harness import HarnessModelResponse, ModelToolCall, RuntimeState
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel, PublicHarnessModel


def test_default_sdk_turn_ends_without_claiming_task_acceptance(tmp_path, monkeypatch):
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", PublicHarnessModel)
    result = asyncio.run(agent.run("Explain this concept", max_steps=2))
    assert result.status == "done"
    assert result.validation_status == "not_evaluated"
    with RuntimeState.open_sync(tmp_path / "run.db") as store:
        policy = store.read_turn(result.turn_id).binding_manifest["completion_policy"]
        assert policy == {"revision": "completion-v2", "require_workspace_change": False, "acceptance_command": None}


class CheckThenAnswer(PatchThenAnswerModel):
    def __init__(self, command):
        self.command = command
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return super().prepare(request)

    async def dispatch(self, prepared, **_kwargs):
        calls = ()
        if prepared.request_ref["request_id"].endswith(":step:1"):
            calls = (ModelToolCall(id="check", name="run_command", arguments={"command": self.command}),)
        return HarnessModelResponse.from_parts(
            text="checked" if not calls else "Run the configured acceptance check.",
            provider_response_id=None,
            usage={"input_tokens": 5, "output_tokens": 3},
            tool_calls=calls,
        )


@pytest.mark.parametrize("entrypoint", ["sdk", "cli"])
@pytest.mark.parametrize("passed", [True, False])
def test_public_explicit_command_uses_actual_execution_without_name_whitelist(
    tmp_path, monkeypatch, entrypoint, passed
):
    command = "test -f evidence.txt"
    if passed:
        (tmp_path / "evidence.txt").write_text("real local evidence")
    model = CheckThenAnswer(command)
    database = tmp_path / "run.db"
    if entrypoint == "sdk":
        agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
        monkeypatch.setattr(agent, "_model_factory", lambda: model)
        result = asyncio.run(
            agent.run("Inspect evidence", acceptance_command=command, allow_execute_tools=True, max_steps=3)
        )
        assert (result.status == "done") is passed
        assert result.validation_status == ("passed" if passed else "failed")
        assert result.validation_scope == "acceptance_command"
    else:
        monkeypatch.chdir(tmp_path)
        original = cli._create_agent_facade

        def create(**kwargs):
            agent = original(**kwargs)
            monkeypatch.setattr(agent, "_model_factory", lambda: model)
            return agent

        monkeypatch.setattr(cli, "_create_agent_facade", create)
        result = CliRunner().invoke(
            cli.agent_app,
            [
                "run",
                "Inspect evidence",
                "--acceptance-command",
                command,
                "--max-steps",
                "3",
                "--allow-execute-tools",
                "--disable-workspace-mcp",
                "--non-interactive",
                "--rollout-db",
                str(database),
                "--verbose",
            ],
        )
        assert (result.exit_code == 0) is passed, result.output
    with RuntimeState.open_sync(database) as store:
        turn = store.list_turns()[-1]
        assert turn.binding_manifest["completion_policy"]["acceptance_command"] == command
        assert (turn.status == "completed") is passed
        assert len(store.list_tool_operations(turn.turn_id)) == 1
        assert store.verify().valid
    assert any(
        command in message.content
        for request in model.requests
        for message in request.messages
        if message.role == "context"
    )


@pytest.mark.parametrize("command", ["", "   ", 123])
def test_invalid_acceptance_contract_stops_before_model_start(tmp_path, monkeypatch, command):
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: pytest.fail("invalid contract opened model"))
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(agent.run("task", acceptance_command=command))


def test_approval_recovery_keeps_the_original_acceptance_contract_and_budget(tmp_path, monkeypatch):
    command = "test -f value.txt"
    (tmp_path / "value.txt").write_text("before")

    class PatchCheckAnswer(CheckThenAnswer):
        async def dispatch(self, prepared, **kwargs):
            if prepared.request_ref["request_id"].endswith(":step:1"):
                return await PatchThenAnswerModel.dispatch(self, prepared, **kwargs)
            if prepared.request_ref["request_id"].endswith(":step:2"):
                return HarnessModelResponse.from_parts(
                    text="Run original acceptance.",
                    provider_response_id=None,
                    usage={"input_tokens": 5, "output_tokens": 3},
                    tool_calls=(ModelToolCall(id="verify", name="run_command", arguments={"command": command}),),
                )
            return await super().dispatch(prepared, **kwargs)

    database = tmp_path / "resume.db"
    first = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(first, "_model_factory", lambda: PatchCheckAnswer(command))
    paused = asyncio.run(
        first.run(
            "Update value.txt",
            require_workspace_change=True,
            acceptance_command=command,
            allow_execute_tools=True,
            max_steps=4,
            max_tokens_total=100_000,
        )
    )
    assert paused.status == "paused"
    fresh = Agent(workspace_path=tmp_path, checkpoint_db=database)
    model = PatchCheckAnswer("true")  # new process settings are not authority over the frozen command
    monkeypatch.setattr(fresh, "_model_factory", lambda: model)
    result = asyncio.run(fresh.resume(paused.turn_id, "allow_once"))
    assert result.status == "done"
    assert result.validation_status == "passed" and result.validation_scope == "acceptance_command"
    with RuntimeState.open_sync(database) as store:
        policy = store.read_turn(result.turn_id).binding_manifest["completion_policy"]
        assert policy["acceptance_command"] == command
        assert store.read_budget_state(result.turn_id).limits.tokens == 100_000
        assert store.read_budget_state(result.turn_id).used.total_tokens > paused.usage.total_tokens
        assert store.verify().valid
    assert any(command in m.content for request in model.requests for m in request.messages if m.role == "context")

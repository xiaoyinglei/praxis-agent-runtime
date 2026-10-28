from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from agent_runtime import Agent, cli
from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, RuntimeState
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires real macOS Seatbelt")


class SessionCommandModel(PatchThenAnswerModel):
    def __init__(self, command, *, write=False, premature=False):
        self.command = command
        self.write = write
        self.premature = premature
        self.final_attempts = 0

    def prepare(self, request):
        return replace(super().prepare(request), dispatch_payload=request)

    async def dispatch(self, prepared, **_kwargs):
        request = prepared.dispatch_payload
        results = [json.loads(m.content)["structured_content"] for m in request.messages if m.role == "tool"]
        if not results:
            call = HarnessToolCall(id="launch", name="run_command", arguments={
                "command": self.command, "working_dir": ".", "timeout_seconds": None,
                "yield_time_seconds": 0.01, "workspace_write": self.write,
            })
        elif results[-1]["status"] == "running":
            if self.premature and self.final_attempts == 0:
                self.final_attempts += 1
                return HarnessModelResponse(text="premature answer", provider_response_id=None,
                                            usage={"input_tokens": 3, "output_tokens": 3})
            call = HarnessToolCall(id="wait-" + str(request.step), name="manage_command", arguments={
                "session_id": results[0]["session_id"], "action": "wait", "wait_seconds": 3,
            })
        else:
            assert results[-1]["exit_code"] == 0, results[-1]
            return HarnessModelResponse(text="observed terminal result", provider_response_id=None,
                                        usage={"input_tokens": 3, "output_tokens": 3})
        return HarnessModelResponse(text="", provider_response_id=None, tool_calls=(call,),
                                    usage={"input_tokens": 3, "output_tokens": 3})


@pytest.mark.parametrize("entrypoint", ["sdk", "cli"])
@pytest.mark.parametrize("write", [False, True])
def test_public_macos_command_lifecycle_approval_and_completion(tmp_path, monkeypatch, entrypoint, write):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "run.db"
    # Write only after the initial response, exercising final mutation evidence.
    command = "sleep 0.2; " + ("printf delayed > artifact" if write else "printf verified")
    models = []

    def model_factory():
        model = SessionCommandModel(command, write=write, premature=True)
        models.append(model)
        return model

    if entrypoint == "sdk":
        agent = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
        monkeypatch.setattr(agent, "_model_factory", model_factory)
        result = asyncio.run(agent.run("Run command", allow_execute_tools=True, max_steps=6,
                                       require_workspace_change=write, acceptance_command=command))
        if write:
            assert result.status == "paused"
            assert not (workspace / "artifact").exists()
            fresh = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
            monkeypatch.setattr(fresh, "_model_factory", model_factory)
            result = asyncio.run(fresh.resume(result.turn_id, "allow_once"))
        assert result.status == "done", result
        assert result.validation_status == "passed"
        assert result.validation_scope == "acceptance_command"
    else:
        original = cli._create_agent_facade

        def create(**kwargs):
            agent = original(**kwargs)
            monkeypatch.setattr(agent, "_model_factory", model_factory)
            return agent

        monkeypatch.setattr(cli, "_create_agent_facade", create)
        monkeypatch.chdir(workspace)
        runner = CliRunner()
        result = runner.invoke(cli.agent_app, [
            "run", "Run command", "--allow-execute-tools", "--max-steps", "6", "--disable-workspace-mcp",
            "--non-interactive", "--rollout-db", str(database), "--acceptance-command", command,
            *( ["--require-workspace-change"] if write else []),
        ])
        if write:
            with RuntimeState.open_sync(database) as store:
                turn = store.list_turns()[-1]
                assert turn.status == "paused"
            assert not (workspace / "artifact").exists()
            result = runner.invoke(cli.agent_app, ["recover", turn.turn_id, "--action", "allow_once",
                                                  "--rollout-db", str(database)])
        assert result.exit_code == 0, result.output
    if write:
        assert (workspace / "artifact").read_text() == "delayed"
    with RuntimeState.open_sync(database) as store:
        turn = store.list_turns()[-1]
        assert turn.status == "completed", turn
        outputs = [i.payload for i in store.list_items(turn.turn_id) if i.kind == "tool_result"]
        assert len(outputs) == 2
        assert [r["structured_content"]["status"] for r in outputs] == ["running", "completed"]
        assert outputs[0]["structured_content"]["exit_code"] is None
        assert outputs[1]["structured_content"]["exit_code"] == 0
        assert outputs[0]["structured_content"]["session_id"] == outputs[1]["structured_content"]["session_id"]
        assert all(not r["is_error"] for r in outputs)
        assert any(i.kind == "completion_feedback" for i in store.list_items(turn.turn_id))
        assert store.verify().valid
    assert not list((workspace / ".praxis/runtime/scratch").glob("run-command-*"))


@pytest.mark.parametrize("entrypoint", ["sdk", "cli"])
def test_public_managed_python_script_entry(tmp_path, monkeypatch, entrypoint):
    workspace = tmp_path / "workspace"
    scripts = workspace / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "helper.py").write_text("VALUE = 42\n")
    (scripts / "check.py").write_text("import helper\nassert helper.VALUE == 42\nprint('script-checked')\n")
    database = tmp_path / "script.db"

    class ScriptModel(SessionCommandModel):
        async def dispatch(self, prepared, **_kwargs):
            request = prepared.dispatch_payload
            results = [json.loads(m.content) for m in request.messages if m.role == "tool"]
            if results:
                assert not results[-1]["is_error"], results[-1]
                output = results[-1]["structured_content"]
                assert output["stdout"] == "script-checked\n"
                assert output["verification"] is None
                return HarnessModelResponse(text="observed script result", provider_response_id=None,
                                            usage={"input_tokens": 3, "output_tokens": 3})
            return HarnessModelResponse(text="", provider_response_id=None,
                tool_calls=(HarnessToolCall("script", "execute_python", {"script_path": "scripts/check.py"}),),
                usage={"input_tokens": 3, "output_tokens": 3})

    def configure(agent):
        monkeypatch.setattr(agent, "_model_factory", lambda: ScriptModel("unused"))
        return agent

    if entrypoint == "sdk":
        agent = configure(Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False))
        result = asyncio.run(agent.run("Run the check script", allow_execute_tools=True))
        assert result.status == "done", result
        assert result.answer == "observed script result"
    else:
        original = cli._create_agent_facade
        monkeypatch.setattr(cli, "_create_agent_facade", lambda **kwargs: configure(original(**kwargs)))
        monkeypatch.chdir(workspace)
        result = CliRunner().invoke(cli.agent_app, [
            "run", "Run the check script", "--allow-execute-tools", "--disable-workspace-mcp",
            "--non-interactive", "--rollout-db", str(database),
        ])
        assert result.exit_code == 0, result.output
        assert "observed script result" in result.output
    with RuntimeState.open_sync(database) as store:
        turn = store.list_turns()[-1]
        assert turn.status == "completed"
        [operation] = store.list_tool_operations(turn.turn_id)
        assert operation.status == "succeeded"
        assert store.verify().valid


def test_public_pause_closes_job_and_resume_reads_durable_terminal(tmp_path, monkeypatch):
    class PauseAfterLaunch(SessionCommandModel):
        async def dispatch(self, prepared, **kwargs):
            request = prepared.dispatch_payload
            outputs = [json.loads(m.content)["structured_content"] for m in request.messages if m.role == "tool"]
            if not outputs:
                return await super().dispatch(prepared, **kwargs)
            if len(outputs) == 1:
                return HarnessModelResponse(text="", provider_response_id=None,
                    usage={"input_tokens": 3, "output_tokens": 3}, tool_calls=(HarnessToolCall(
                        id="patch", name="apply_patch", arguments={
                            "file_path": "source", "old_string": "before", "new_string": "after",
                        },
                    ),))
            assert any("Confirmed terminal outcome" in m.content for m in request.messages if m.role == "context")
            return HarnessModelResponse(text="The command was cancelled at pause; the approved edit completed.",
                                        provider_response_id=None, usage={"input_tokens": 3, "output_tokens": 3})

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source").write_text("before")
    database = tmp_path / "run.db"
    agent = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
    def factory():
        return PauseAfterLaunch("printf ready; sleep 60")
    monkeypatch.setattr(agent, "_model_factory", factory)
    result = asyncio.run(agent.run("Run, then edit", allow_execute_tools=True, max_steps=5))
    assert result.status == "paused"
    with RuntimeState.open_sync(database) as store:
        [terminal] = [i for i in store.list_items(result.turn_id) if i.kind == "tool_continuation"]
        assert terminal.producer == "runtime"
        assert terminal.payload["structured_content"]["status"] == "cancelled"
        assert terminal.payload["is_error"] is True
        assert store.verify().valid
    fresh = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(fresh, "_model_factory", factory)
    resumed = asyncio.run(fresh.resume(result.turn_id, "allow_once"))
    assert resumed.status == "done", resumed
    assert (workspace / "source").read_text() == "after"
    assert not list((workspace / ".praxis/runtime/scratch").glob("run-command-*"))


@pytest.mark.anyio
async def test_abort_live_session_cancels_job_and_allows_next_command(tmp_path, monkeypatch):
    class PausingModel(SessionCommandModel):
        async def dispatch(self, prepared, **kwargs):
            if not any(m.role == "tool" for m in prepared.dispatch_payload.messages):
                return await super().dispatch(prepared, **kwargs)
            return HarnessModelResponse(text="", provider_response_id=None,
                usage={"input_tokens": 3, "output_tokens": 3}, tool_calls=(HarnessToolCall(
                    id="patch", name="apply_patch", arguments={
                        "file_path": "source", "old_string": "before", "new_string": "after",
                    },
                ),))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source").write_text("before")
    agent = Agent(workspace_path=workspace, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: PausingModel("printf ready; sleep 60"))
    async with agent.session(allow_execute_tools=True, max_steps=5) as session:
        paused = await session.submit("Run, then edit")
        assert paused.status == "paused"
        [launch] = [i for i in session.store.list_items(paused.turn_id) if i.kind == "tool_result"]
        handle = launch.payload["structured_content"]["session_id"]
        running = await session.tools["manage_command"].run({"session_id": handle, "action": "status"})
        assert running.status == "running"
        await session.resume(paused.turn_id, "abort")
        [terminal] = [i for i in session.store.list_items(paused.turn_id) if i.kind == "tool_continuation"]
        assert terminal.payload["structured_content"]["status"] == "cancelled"
        assert not list((workspace / ".praxis/runtime/scratch").glob("run-command-*"))
        next_command = await session.tools["run_command"].run({
            "command": "printf next", "timeout_seconds": None, "yield_time_seconds": 0.01,
        })
        # Direct Tool.run bypasses the durable orchestrator; acknowledge the
        # host's test commit explicitly before releasing this staged command.
        await session.tools["run_command"].after_commit({}, next_command.model_dump(), "direct_test_operation")
        next_command = await session.tools["manage_command"].run({
            "session_id": next_command.session_id, "action": "wait", "wait_seconds": 3,
        })
        assert next_command.exit_code == 0 and next_command.stdout == "next"


@pytest.mark.anyio
async def test_model_budget_failure_cleans_command_before_live_session_closes(tmp_path, monkeypatch):
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: SessionCommandModel("printf ready; sleep 60"))
    async with agent.session(allow_execute_tools=True, max_steps=1) as session:
        result = await session.submit("Run within the existing model budget")
        assert result.status == "failed"
        [terminal] = [i for i in session.store.list_items(result.turn_id) if i.kind == "tool_continuation"]
        assert terminal.payload["structured_content"]["status"] == "cancelled"
        assert not list((tmp_path / ".praxis/runtime/scratch").glob("run-command-*"))
        assert session.store.verify().valid

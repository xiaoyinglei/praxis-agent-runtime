"""A lost wait client must not strand a ledger-authorized durable command."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime import Agent
from agent_runtime.harness import HarnessModelResponse, ModelToolCall, RuntimeState
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason=("the live command writes into the real workspace on macOS Seatbelt; "
            "the Linux candidate view publishes only after completion, so "
            "counter/release visibility while running cannot be observed"),
)


class WaitRecoveryModel(PatchThenAnswerModel):
    def prepare(self, request):
        return replace(super().prepare(request), dispatch_payload=request)

    async def dispatch(self, prepared, **_kwargs):
        request = prepared.dispatch_payload
        outputs = [json.loads(message.content)["structured_content"]
                   for message in request.messages if message.role == "tool"]
        terminal = [json.loads(message.content.split("\n", 1)[1])["structured_content"]
                    for message in request.messages if message.role == "context"
                    and message.content.startswith("The live command owner closed.")]
        if terminal and terminal[-1].get("status") in {"completed", "cancelled"}:
            if terminal[-1]["status"] == "completed":
                assert terminal[-1]["stdout"] == "READYDONE", terminal[-1]
            return HarnessModelResponse.from_parts(text="Observed original command", provider_response_id=None,
                                        usage={"input_tokens": 3, "output_tokens": 3})
        if not outputs:
            call = ModelToolCall(id="launch", name="run_command", arguments={
                "command": ("printf x >> counter; printf READY; "
                            "while [ ! -f release ]; do sleep 0.05; done; printf DONE"),
                "workspace_write": True, "timeout_seconds": None, "yield_time_seconds": 0.01,
            })
        elif outputs[-1].get("status") == "running":
            call = ModelToolCall(id="wait-" + str(request.step), name="manage_command", arguments={
                "session_id": outputs[0]["session_id"], "action": "wait", "wait_seconds": 3,
            })
        else:
            assert outputs[-1]["stdout"] == "READYDONE", outputs[-1]
            return HarnessModelResponse.from_parts(text="Observed original command", provider_response_id=None,
                                        usage={"input_tokens": 3, "output_tokens": 3})
        return HarnessModelResponse.from_parts(text="", provider_response_id=None, tool_calls=(call,),
                                    usage={"input_tokens": 3, "output_tokens": 3})


@pytest.mark.anyio
@pytest.mark.parametrize("entrypoint", ["resume", "followup"])
@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("lose_result", [False, True])
async def test_killed_wait_client_reconciles_original_job_without_replay(tmp_path, entrypoint, finished, lose_result):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    script = '''
import asyncio, sys
from pathlib import Path
from agent_runtime import Agent
from tests.agent.test_command_wait_recovery import WaitRecoveryModel
async def main():
    agent = Agent(workspace_path=Path(sys.argv[1]), checkpoint_db=Path(sys.argv[2]),
                  enable_workspace_mcp=False, approval_mode='full')
    agent._model_factory = WaitRecoveryModel
    async with agent.session(allow_write_tools=True, allow_execute_tools=True, max_steps=12) as session:
        print(session.thread_id, flush=True)
        result = await session.submit('Run original command')
        assert result.status == 'done', result.pause or result.diagnostics
asyncio.run(main())
'''
    child = subprocess.Popen([sys.executable, "-u", "-c", script, str(workspace), str(database)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env={**os.environ, "PYTHONPATH": str(Path.cwd())})
    try:
        assert child.stdout is not None
        thread_id = (await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 10)).strip()
        assert thread_id
        async with asyncio.timeout(10):
            while True:
                async with (await RuntimeState.open(database, read_only=True)) as store:
                    turn_id = store.read_thread(thread_id).active_turn_id
                    operations = store.list_tool_operations(turn_id) if turn_id else ()
                if ((workspace / "counter").exists()
                        and any(op.tool_name == "manage_command" and op.status == "running" for op in operations)):
                    break
                assert child.poll() is None, child.stderr.read()
                await asyncio.sleep(0.01)
        child.kill()
        await asyncio.to_thread(child.wait, 5)
        if finished:
            (workspace / "release").touch()
        agent = Agent(workspace_path=workspace, checkpoint_db=database,
                      enable_workspace_mcp=False, approval_mode="full")
        agent._model_factory = WaitRecoveryModel
        if lose_result:
            from agent_runtime.harness.ownership import ConversationRecoveryRequiredError
            from agent_runtime.tools.builtins import command_worker
            (workspace / "release").touch()
            async with (await RuntimeState.open(database, read_only=True)) as store:
                launch = next(item for item in store.list_items(turn_id)
                              if item.kind == "tool_result" and item.payload.get("tool_name") == "run_command")
                key = launch.payload["structured_content"]["session_id"]
            terminal = await agent.command(turn_id, key, action="wait", wait_seconds=5)
            assert terminal.exit_code == 0
            # Fault injection: started marker remains but the worker's terminal
            # receipt is lost. A later client must not invent or replay it.
            (command_worker.job_path(workspace, key) / "result.json").unlink()
            if entrypoint == "followup":
                async with agent.session(thread_id=thread_id) as session:
                    with pytest.raises(ConversationRecoveryRequiredError):
                        await session.submit("New request must not discard the uncertain job")
            else:
                result = await agent.resume(turn_id, "continue")
                assert result.status == "paused" and result.pause.kind == "tool_reconciliation"
            async with (await RuntimeState.open(database)) as store:
                assert store.read_turn(turn_id).status == "paused"
                assert any(op.status == "unknown" for op in store.list_tool_operations(turn_id))
                assert store.verify().valid
            assert (workspace / "counter").read_text() == "x"
            return
        if entrypoint == "resume":
            async def finish_command():
                await asyncio.sleep(0.2)
                (workspace / "release").touch()
            release = asyncio.create_task(finish_command())
            try:
                result = await agent.resume(turn_id, "continue")
            finally:
                await release
        else:
            # A new user message closes the old Turn and inherits its evidence.
            (workspace / "release").touch()
            async with agent.session(thread_id=thread_id, allow_write_tools=True,
                                     allow_execute_tools=True, max_steps=12) as session:
                result = await session.submit("Continue from the recorded command result")
        assert result.status == "done", result.pause or result.diagnostics
        assert result.answer == "Observed original command"
        assert (workspace / "counter").read_text() == "x"
        async with (await RuntimeState.open(database)) as store:
            assert store.verify().valid
            wait = next(op for op in store.list_tool_operations(turn_id) if op.tool_name == "manage_command")
            assert wait.status == "succeeded" and not wait.requires_reconciliation
            assert any(item.kind == "tool_reconciliation" for item in store.list_items(turn_id))
            assert not any(item.status == "pending" for item in store.list_interactions(turn_id))
    finally:
        (workspace / "release").touch()
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)

"""An actual dead batch worker must not replay completed or uncertain calls."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, RolloutStore, Session
from tests.agent.harness.test_tool_orchestrator import AcceptToolAnswer, ToolThenAnswerModel, _read_tool


class CrashBatchModel(ToolThenAnswerModel):
    async def dispatch(self, prepared, *, delta_sink=None):
        if self.requests[-1].step == 1:
            return HarnessModelResponse(text="", provider_response_id="batch", usage={}, tool_calls=tuple(
                HarnessToolCall(id=f"call-{i}", name="read_file", arguments={"path": str(i)}) for i in range(3)
            ))
        return HarnessModelResponse(text="README was read", provider_response_id="done", usage={})


@pytest.mark.anyio
async def test_killed_batch_retains_finished_result_reconciles_uncertain_and_runs_only_unstarted(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "state.db"
    launches = tmp_path / "launches.txt"
    script = '''
import asyncio, sys
from pathlib import Path
from dataclasses import replace
from agent_runtime.harness import Session
from agent_runtime.tools.permissions import ToolExecutionContext
from tests.agent.harness.test_tool_orchestrator import _read_tool
from tests.agent.harness.test_tool_batch_crash import CrashBatchModel
async def main():
    workspace, database, launches = map(Path, sys.argv[1:])
    async def run(args):
        with launches.open("a") as output:
            output.write(args["path"] + "\\n")
        if args["path"] == "1":
            print("WAITING", flush=True)
            await asyncio.Event().wait()
        return {"text": args["path"]}
    async with await Session.open(database=database, workspace=workspace, model=CrashBatchModel(),
        tools={"read_file": replace(_read_tool(workspace=workspace), run=run, timeout_seconds=60)},
        tool_execution_context=ToolExecutionContext(workspace_root=workspace, cwd=workspace, max_parallel_calls=2),
        _background_memory=False) as session:
        print(session.thread_id, flush=True)
        await session.submit("inspect independent records")
asyncio.run(main())
'''
    child = subprocess.Popen([sys.executable, "-u", "-c", script, str(workspace), str(database), str(launches)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env={**os.environ, "PYTHONPATH": str(Path.cwd())})
    try:
        thread_id = (await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 10)).strip()
        assert (await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 10)).strip() == "WAITING"
        for _ in range(100):
            with RolloutStore(database) as store:
                operations = store.list_tool_operations()
                if any(op.tool_call_id == "call-0" and op.result_item_id for op in operations):
                    turn_id = operations[0].turn_id
                    break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("first tool result was not durably committed")
        child.kill()
        await asyncio.to_thread(child.wait, 5)
    finally:
        if child.poll() is None:
            child.kill()
            await asyncio.to_thread(child.wait, 5)
        child.stdout.close()
        child.stderr.close()

    def run(args):
        with launches.open("a") as output:
            output.write(args["path"] + "\n")
        return {"text": args["path"]}

    model = CrashBatchModel()
    async with await Session.open(database=database, workspace=workspace, thread_id=thread_id, model=model,
        tools={"read_file": replace(_read_tool(workspace=workspace), run=run, timeout_seconds=60)},
        completion_gate=AcceptToolAnswer()) as reopened:
        result = await reopened.resume(turn_id, action="continue")
        assert result.status == "paused" and result.pause.kind == "tool_reconciliation"
        assert launches.read_text().splitlines() == ["0", "1"]
        assert not model.requests
        unknown = [op for op in reopened.store.list_tool_operations(turn_id) if op.status == "unknown"]
        assert [op.tool_call_id for op in unknown] == ["call-1"]
        reopened.store.resolve_tool_reconciliation(turn_id=turn_id, status="failed", result={
            "tool_call_id": "call-1", "tool_name": "read_file", "is_error": True,
            "error_code": "interrupted", "model_content": "Original operation independently reconciled.",
            "content": [], "attachments": [],
        }, reconciler_revision="test-independent-observation-v1")
        result = await reopened.resume(turn_id, action="continue")
        assert result.status == "done"
        assert launches.read_text().splitlines() == ["0", "1", "2"]
        assert [m.tool_call_id for m in model.requests[-1].messages if m.role == "tool"] == [
            "call-0", "call-1", "call-2",
        ]
        assert reopened.store.verify().valid

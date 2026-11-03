import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_runtime import Agent
from agent_runtime.background import BackgroundHandle

ROOT = Path(__file__).resolve().parents[2]

WORKER = """
import asyncio, sys
from pathlib import Path
from agent_runtime.host import BackgroundHost
from tests.agent.harness.test_subagent_budget import SmallAnswerModel

root = Path(sys.argv[1])
class Model(SmallAnswerModel):
    async def dispatch(self, prepared, **kwargs):
        with (root / 'dispatches').open('a') as stream:
            stream.write('dispatched\\n')
        (root / 'started').touch()
        while not (root / 'release').exists():
            await asyncio.sleep(0.01)
        return await super().dispatch(prepared, **kwargs)

async def main():
    async with BackgroundHost(root / 'runtime.db', model=Model()) as host:
        (root / 'ready').touch()
        await host.serve(idle_timeout=0.2)
asyncio.run(main())
"""


async def wait_path(path):
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(0.01)


def launch(root):
    return subprocess.Popen(
        [sys.executable, "-c", WORKER, str(root)], cwd=root,
        env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


@pytest.mark.anyio
async def test_submitter_process_exits_while_host_keeps_running(tmp_path):
    code = """
import asyncio, json, sys
from pathlib import Path
from agent_runtime import Agent
async def main():
    root=Path(sys.argv[1])
    agent=Agent(workspace_path=root, checkpoint_db=root/'runtime.db', model_session_path=None,
                enable_workspace_mcp=False)
    handle=await agent.start_background('inspect', autostart=False)
    print(json.dumps({'thread_id':handle.thread_id, 'turn_id':handle.turn_id}))
asyncio.run(main())
"""
    submitted = await asyncio.to_thread(
        subprocess.run, [sys.executable, "-c", code, str(tmp_path)], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, timeout=15,
    )
    assert submitted.returncode == 0, submitted.stderr
    identity = json.loads(submitted.stdout)
    handle = BackgroundHandle(
        database=tmp_path / "runtime.db", workspace=tmp_path, autostart=False, **identity
    )
    worker = launch(tmp_path)
    try:
        await wait_path(tmp_path / "started")
        assert (await handle.status())["status"] == "running"
        (tmp_path / "release").touch()
        assert (await handle.wait(timeout=10))["status"] == "completed"
        assert (await handle.result()).answer == "child answer"
        assert await asyncio.to_thread(worker.wait, timeout=10) == 0
    finally:
        if worker.poll() is None:
            worker.kill()
            await asyncio.to_thread(worker.wait)


@pytest.mark.anyio
async def test_killed_host_recovers_unknown_dispatch_without_repeating_provider(tmp_path):
    agent = Agent(
        workspace_path=tmp_path, checkpoint_db=tmp_path / "runtime.db",
        model_session_path=None, enable_workspace_mcp=False,
    )
    handle = await agent.start_background("inspect", autostart=False)
    worker = launch(tmp_path)
    try:
        await wait_path(tmp_path / "started")
        worker.kill()
        await asyncio.to_thread(worker.wait, timeout=10)
    finally:
        if worker.poll() is None:
            worker.kill()
            await asyncio.to_thread(worker.wait)
    (tmp_path / "ready").unlink()
    restarted = launch(tmp_path)
    try:
        await wait_path(tmp_path / "ready")
        assert await asyncio.to_thread(restarted.wait, timeout=10) == 0
        status = await handle.status()
        assert status["status"] in {"paused", "interrupted"}
        assert status["recovery"]["reason_code"] == "model_outcome_unknown"
        assert (tmp_path / "dispatches").read_text().splitlines() == ["dispatched"]
    finally:
        if restarted.poll() is None:
            restarted.kill()
            await asyncio.to_thread(restarted.wait)


@pytest.mark.anyio
async def test_retry_receipt_cannot_authorize_another_attempt_after_second_crash(tmp_path):
    agent = Agent(
        workspace_path=tmp_path, checkpoint_db=tmp_path / "runtime.db",
        model_session_path=None, enable_workspace_mcp=False,
    )
    handle = await agent.start_background("inspect", autostart=False)
    for attempt in range(2):
        if attempt:
            (tmp_path / "started").unlink()
            (tmp_path / "ready").unlink()
        worker = launch(tmp_path)
        try:
            await wait_path(tmp_path / "started")
        finally:
            worker.kill()
            await asyncio.to_thread(worker.wait, timeout=10)
        if not attempt:
            (tmp_path / "ready").unlink()
            recovery = launch(tmp_path)
            try:
                assert await asyncio.to_thread(recovery.wait, timeout=10) == 0
            finally:
                if recovery.poll() is None:
                    recovery.kill()
                    await asyncio.to_thread(recovery.wait)
            state = (await handle.status())["recovery"]
            await handle.respond(state["request_id"], "retry", input_id="retry-once")
    (tmp_path / "ready").unlink()
    recovered = launch(tmp_path)
    try:
        assert await asyncio.to_thread(recovered.wait, timeout=10) == 0
        state = await handle.status()
        assert state["recovery"]["reason_code"] == "model_outcome_unknown"
        assert (tmp_path / "dispatches").read_text().splitlines() == ["dispatched", "dispatched"]
        repeated = await handle.respond(state["recovery"]["request_id"], "retry", input_id="retry-once")
        assert repeated.command_id == "retry-once"
    finally:
        if recovered.poll() is None:
            recovered.kill()
            await asyncio.to_thread(recovered.wait)


@pytest.mark.anyio
async def test_actual_worker_entrypoint_autostarts_once_and_processes_durable_cancel(tmp_path, monkeypatch):
    from agent_runtime import background

    handle = await Agent(
        workspace_path=tmp_path, checkpoint_db=tmp_path / "runtime.db", model_session_path=None,
        enable_workspace_mcp=False,
    ).start_background("inspect", autostart=False)
    await handle.cancel(input_id="cancel-before-autostart")
    processes = []
    original = subprocess.Popen

    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(background.subprocess, "Popen", start)
    await asyncio.to_thread(background.ensure_host, handle.database)
    try:
        state = await handle.wait(timeout=10)
        assert state["status"] is None and state["queue_status"] == "rejected"
        assert len(processes) == 1 and processes[0].poll() is None
        await asyncio.to_thread(background.ensure_host, handle.database)
        assert len(processes) == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                await asyncio.to_thread(process.wait, timeout=10)

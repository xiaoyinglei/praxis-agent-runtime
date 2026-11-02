"""Actual SIGKILL at retry commits, followed by fresh public Session recovery."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from agent_runtime.harness import RuntimeStore, Session
from tests.agent.harness.test_model_retry_driver import RetryModel, failure


@pytest.mark.parametrize("boundary", ["scheduled", "prepared"])
def test_fresh_process_resume_preserves_retry_identity_and_allowance(tmp_path, boundary):
    repo = Path(__file__).resolve().parents[3]
    database = tmp_path / "run.db"
    point = tmp_path / "point.json"
    child = """import asyncio,json,os,signal,sys
from pathlib import Path
from agent_runtime.harness import Session,model_retry_ops
from tests.agent.harness.test_model_retry_driver import RetryModel,failure
root=Path(sys.argv[1]);boundary=sys.argv[2]
async def main():
    async with await Session.open(
        database=root/'run.db',workspace=root,model=RetryModel(failure(),count=10),
        _background_memory=False
    ) as session:
        def stop():
            with (root/'point.json').open('w') as out:
                json.dump({'turn_id':session.store.read_thread(session.thread_id).active_turn_id,'thread_id':session.thread_id},out)
                out.flush();os.fsync(out.fileno())
            os.kill(os.getpid(),signal.SIGKILL)
        class Sink:
            async def emit(self,event):
                if boundary=='scheduled' and event.type=='model_retry_scheduled':stop()
        if boundary=='prepared':
            prepare=model_retry_ops.prepare_scheduled_retry
            def crash_after_prepare(*args,**kwargs):
                prepare(*args,**kwargs)
                stop()
            model_retry_ops.prepare_scheduled_retry=crash_after_prepare
        await session.submit('answer',event_sink=Sink())
asyncio.run(main())
"""
    env = {**os.environ, "PYTHONPATH": str(repo)}
    process = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path), boundary],
        env=env,
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert process.returncode == -signal.SIGKILL, process.stderr
    checkpoint = json.loads(point.read_text())
    with RuntimeStore.open_sync(database, read_only=True) as store:
        [operation] = store.list_model_operations(checkpoint["turn_id"])
        assert operation.status == ("retry_wait" if boundary == "scheduled" else "prepared")
        first_schedule = [
            r for r in store.list_turn_records(checkpoint["turn_id"]) if r.record_type == "model_retry_scheduled"
        ][0]
        assert len(store.list_model_attempts(operation.operation_id)) == (1 if boundary == "scheduled" else 2)
    model = RetryModel(failure(), count=10 if boundary == "scheduled" else 0)

    async def resume():
        async with await Session.open(
            database=database,
            workspace=tmp_path,
            model=model,
            thread_id=checkpoint["thread_id"],
            _background_memory=False,
        ) as session:
            result = await session.resume(checkpoint["turn_id"], "continue")
            if boundary == "scheduled":
                assert result.status == "failed"
                assert len(model.dispatched) == 2
            else:
                assert result.status == "done"
                assert len(model.dispatched) == 1
            [restored] = session.store.list_model_operations(result.turn_id)
            assert restored.operation_id == operation.operation_id
            assert restored.request_hash == operation.request_hash
            schedules = [
                r for r in session.store.list_turn_records(result.turn_id) if r.record_type == "model_retry_scheduled"
            ]
            assert schedules[0].record_uuid == first_schedule.record_uuid
            assert {r.payload["deadline_at"] for r in schedules} == {first_schedule.payload["deadline_at"]}
            assert len(session.store.list_model_attempts(operation.operation_id)) == (
                3 if boundary == "scheduled" else 2
            )
            state = session.store.read_budget_state(result.turn_id)
            assert state.used.model_calls == (3 if boundary == "scheduled" else 2)
            assert state.used.retries == (2 if boundary == "scheduled" else 1)
            assert state.reserved.total_tokens == 0
            assert session.store.verify().valid

    asyncio.run(resume())

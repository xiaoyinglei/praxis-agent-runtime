"""Independent regressions for reviewed public recovery and writer windows.

Provider fixtures are deterministic; SQLite, reopen, external steering and the
frozen legacy writer are real. These tests do not claim real-model acceptance.
"""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
import tarfile

import pytest

from agent_runtime.harness import (
    CompletionDecision,
    RolloutContextManager,
    RuntimeStore,
    Session,
    TurnRunner,
    compaction_jobs,
    turn_ops,
)
from agent_runtime.harness.protocol import ModelContextOverflowError
from tests.agent.harness.test_compaction_consistency import Accept, long_history, model_for, start
from tests.agent.harness.test_context_prefix_updates import Probe


class ProcessStopped(BaseException):
    """Stop after a real committed boundary without cancellation bookkeeping."""


class NeedClarification:
    def evaluate(self, proposal):
        return CompletionDecision(action="pause", reason="Which output should be produced?")


@pytest.mark.anyio
async def test_cancellation_committed_before_provider_return_cannot_complete_answer(tmp_path):
    model = Probe("answer")
    dispatch = model.dispatch
    async with await Session.open(
        database=tmp_path / "run.db", workspace=tmp_path, model=model, _background_memory=False
    ) as session:

        async def return_after_cancellation(prepared, **kwargs):
            await session.store.run_async(
                lambda: turn_ops.request_turn_cancellation(
                    session.store, turn_id=model.requests[-1].turn_id, reason="accepted user cancellation"
                )
            )
            return await dispatch(prepared, **kwargs)

        model.dispatch = return_after_cancellation
        result = await session.submit("Perform this task")
        assert session.store.read_turn(result.turn_id).status == "cancelled"
        assert result.answer is None
        assert len(model.requests) == 1
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_clarification_created_before_handling_receipt_survives_reopen(tmp_path, monkeypatch):
    import agent_runtime.harness.responses as response_module

    database = tmp_path / "run.db"
    async with await Session.open(
        database=database,
        workspace=tmp_path,
        model=Probe("answer"),
        completion_gate=NeedClarification(),
        _background_memory=False,
    ) as session:

        def stop_before_handling(*args, **kwargs):
            raise ProcessStopped()

        with monkeypatch.context() as patch:
            patch.setattr(response_module, "response_handled", stop_before_handling)
            with pytest.raises(ProcessStopped):
                await session.submit("Produce an answer")
        thread_id = session.thread_id
        turn_id = session.store.read_thread(thread_id).active_turn_id
        [question] = session.store.list_interactions(turn_id)
        assert not any(r.record_type == "response_handling_decided" for r in session.store.list_turn_records(turn_id))

    model = Probe("answer")
    async with await Session.open(
        database=database,
        workspace=tmp_path,
        thread_id=thread_id,
        model=model,
        completion_gate=Accept(),
        _background_memory=False,
    ) as session:
        result = await session.resume(
            turn_id, "continue", user_input="Produce JSON", expected_request_id=question.request_id
        )
        assert result.status == "done"
        assert len(model.requests) == 1
        assert len(session.store.list_interactions(turn_id)) == 1
        assert session.store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("newer_question", [False, True])
async def test_committed_interaction_reply_is_idempotent_after_lost_ack(tmp_path, monkeypatch, newer_question):
    database = tmp_path / "run.db"
    async with await Session.open(
        database=database,
        workspace=tmp_path,
        model=Probe("answer"),
        completion_gate=NeedClarification(),
        _background_memory=False,
    ) as session:
        paused = await session.submit("Produce an answer")
        thread_id, turn_id, request_id = session.thread_id, paused.turn_id, paused.pause.request_id

    resolve = turn_ops.resolve_clarification

    def stop_after_commit(*args, **kwargs):
        resolve(*args, **kwargs)
        raise ProcessStopped()

    async with await Session.open(
        database=database,
        workspace=tmp_path,
        thread_id=thread_id,
        model=Probe("answer"),
        completion_gate=Accept(),
        _background_memory=False,
    ) as session:
        with monkeypatch.context() as patch:
            patch.setattr(turn_ops, "resolve_clarification", stop_after_commit)
            with pytest.raises(ProcessStopped):
                await session.resume(turn_id, "continue", user_input="JSON", expected_request_id=request_id)
        assert session.store.read_interaction(request_id).status == "resolved"

    async with await RuntimeStore.open(database) as store:
        if newer_question:
            next_question = turn_ops.request_clarification(store, turn_id=turn_id, question="Which filename?")
        before_inputs = [item.item_id for item in store.list_items(turn_id) if item.kind == "user_message"]

    model = Probe("answer")
    async with await Session.open(
        database=database,
        workspace=tmp_path,
        thread_id=thread_id,
        model=model,
        completion_gate=Accept(),
        _background_memory=False,
    ) as session:
        with pytest.raises(RuntimeError, match="conflicts"):
            await session.resume(turn_id, "continue", user_input="CSV", expected_request_id=request_id)
        result = await session.resume(turn_id, "continue", user_input="JSON", expected_request_id=request_id)
        assert [
            item.item_id for item in session.store.list_items(turn_id) if item.kind == "user_message"
        ] == before_inputs
        if newer_question:
            assert result.status == "paused"
            assert result.pause.request_id == next_question.request_id
            assert session.store.read_interaction(next_question.request_id).status == "pending"
            assert not model.requests
        else:
            assert result.status == "done"
            assert len(model.requests) == 1
            repeated = await session.resume(turn_id, "continue", user_input="JSON", expected_request_id=request_id)
            assert repeated.status == "done"
            assert len(model.requests) == 1
        assert session.store.verify().valid


@pytest.mark.parametrize("reopen", [False, True])
def test_overflow_chain_survives_steering_after_recovery_install(tmp_path, monkeypatch, reopen):
    database = tmp_path / "run.db"
    dispatched = []
    installed = []
    install = compaction_jobs.install_candidate

    def inject_steering(store, **kwargs):
        receipt = install(store, **kwargs)
        installed.append(kwargs["job_id"])
        if len(installed) == 1:
            # A second owner writes only after installation committed, while
            # the driver has not yet received the measured continuation.
            with RuntimeStore.open_sync(database) as external:
                turn_ops.receive_turn_input(
                    external,
                    thread_id=thread.thread_id,
                    turn_id=turn.turn_id,
                    text="New requirement after recovery installation",
                )
            if reopen:
                raise ProcessStopped()
        return receipt

    monkeypatch.setattr(compaction_jobs, "install_candidate", inject_steering)

    def runner_for(store):
        model, _ = model_for(100000)
        dispatch = model.dispatch

        async def overflow(prepared, **kwargs):
            if prepared.request_ref.get("purpose") != "context_summary":
                dispatched.append(dict(prepared.request_ref))
                raise ModelContextOverflowError("provider rejected request")
            return await dispatch(prepared, **kwargs)

        model.dispatch = overflow
        return TurnRunner(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )

    with RuntimeStore.open_sync(database) as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        runner = runner_for(store)
        if reopen:
            with pytest.raises(ProcessStopped):
                asyncio.run(runner.drive(runner.restore_turn_context(turn.turn_id), start_step=1))
        else:
            result = asyncio.run(runner.drive(runner.restore_turn_context(turn.turn_id), start_step=1))
    with RuntimeStore.open_sync(database) as store:
        if reopen:
            result = asyncio.run(runner_for(store).drive(turn_id=turn.turn_id, action="continue"))
        assert result.status == "failed"
        assert store.read_turn(turn.turn_id).terminal_reason_code == "model_context_window_exceeded"
        assert len(dispatched) == 2
        initial = next(
            op
            for op in store.list_model_operations(turn.turn_id)
            if op.request_ref.get("request_id") == dispatched[0]["request_id"]
        )
        assert dispatched[1]["capacity_recovery_root_id"] == initial.operation_id
        assert dispatched[1]["input_revision"] == 2
        jobs = {item.payload["job_id"] for item in store.list_items(turn.turn_id) if item.kind == "compaction_job"}
        assert len(jobs) == len(installed) == 1
        assert store.verify().valid


def test_frozen_legacy_writer_is_rejected_before_provider_io(tmp_path):
    baseline = "2658e4d00d4a67610cead31144476a29b7b7928f"
    archive = subprocess.run(
        ["git", "archive", baseline, "agent_runtime"],
        capture_output=True,
        check=False,
    )
    if archive.returncode:
        pytest.skip("frozen baseline Git object unavailable for legacy binary execution")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as source:
        source.extractall(legacy, filter="data")
    database = tmp_path / "run.db"
    with RuntimeStore.open_sync(database):
        pass
    marker = tmp_path / "provider-called"
    program = r"""
import asyncio, hashlib, sys
from pathlib import Path
from agent_runtime.harness import Session, PreparedModelCall, HarnessModelResponse
class Model:
    def snapshot(self, **kwargs): return {"model_id": "test-model", "model_revision": "test-v1"}
    def ensure_available(self, binding, **kwargs): pass
    def prepare(self, request):
        digest = hashlib.sha256(str(request.step).encode()).hexdigest()
        return PreparedModelCall(request_hash=digest, context_hash=digest, tool_hash=digest, wire_hash=digest,
            request_ref={"request_id": f"{request.turn_id}:step:{request.step}"})
    async def dispatch(self, prepared, **kwargs):
        Path(sys.argv[3]).write_text("provider called")
        return HarnessModelResponse.from_parts(text="answer")
async def main():
    async with await Session.open(database=Path(sys.argv[1]), workspace=Path(sys.argv[2]),
        model=Model(), tools={}, _background_memory=False) as session:
        await session.submit("Old writer attempts a fresh conversation")
asyncio.run(main())
"""
    process = subprocess.run(
        [sys.executable, "-c", program, str(database), str(tmp_path), str(marker)],
        cwd=legacy,
        env={**os.environ, "PYTHONPATH": str(legacy)},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert process.returncode != 0
    assert "runtime writer protocol requires upgrade" in process.stderr
    assert not marker.exists()
    with RuntimeStore.open_sync(database) as store:
        assert not store.list_turns()
        assert store.verify().valid

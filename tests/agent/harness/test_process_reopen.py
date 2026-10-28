"""A saved running flag is not a live execution owner."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.budget import ResourceUsage
from agent_runtime.harness import RuntimeState, Session
from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel


class ReopenModel(PublicHarnessModel):
    def __init__(self):
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        prepared = super().prepare(request)
        return replace(prepared, request_ref={**prepared.request_ref,
            "context_projection": {"max_input_tokens": 20000}},
            resource_request=ResourceUsage(input_tokens=100, output_tokens=512, model_calls=1))


@pytest.mark.anyio
@pytest.mark.parametrize("stop_signal", [signal.SIGKILL, signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("phase", ["agent_step", "context_summary"])
async def test_killed_worker_reopens_immediately_and_applies_followup(tmp_path, stop_signal, phase):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "state.sqlite"
    script = '''
import asyncio, sys
from dataclasses import replace
from pathlib import Path
from agent_runtime.harness import Session
from agent_runtime.harness import ContextCompactionRequiredError
from tests.agent.harness.test_process_reopen import ReopenModel
class Waiting(ReopenModel):
    armed = False
    def prepare(self, request):
        if (self.armed and sys.argv[3] == "context_summary" and request.purpose != "context_summary"
                and not any(m.role == "context" and m.content.startswith("Context compaction:")
                            for m in request.messages)):
            raise ContextCompactionRequiredError(input_tokens=1000, max_input_tokens=500, retained_tail_messages=1)
        return super().prepare(request)
    async def dispatch(self, prepared, *, event_sink=None):
        if not self.armed:
            return replace(await super().dispatch(prepared, event_sink=event_sink), text="previous answer " * 1000)
        print("DISPATCHED", flush=True)
        await asyncio.Event().wait()
async def main():
    model = Waiting()
    async with await Session.open(database=Path(sys.argv[1]), workspace=Path(sys.argv[2]), model=model,
                                 _background_memory=False, max_tokens_total=50000) as session:
        if sys.argv[3] == "context_summary":
            await session.submit("earlier history to compact")
        model.armed = True
        print(session.thread_id, flush=True)
        await session.submit("remember the original task")
asyncio.run(main())
'''
    child = subprocess.Popen([sys.executable, "-u", "-c", script, str(database), str(workspace), phase],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env={**os.environ, "PYTHONPATH": str(Path.cwd())})
    try:
        assert child.stdout is not None
        thread_id = await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 10)
        ready = await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 10)
        assert ready.strip() == "DISPATCHED"
        child.send_signal(stop_signal)
        await asyncio.to_thread(child.wait, 10)
        async with (await RuntimeState.open(database, read_only=True)) as store:
            turn_id = store.read_thread(thread_id.strip()).active_turn_id
            assert turn_id is not None
            assert store.list_model_operations(turn_id)[-1].status in {"dispatched", "unknown"}
            assert store.list_model_operations(turn_id)[-1].request_ref["purpose"] == phase
        model = ReopenModel()
        async with await Session.open(database=database, workspace=workspace, model=model,
                                     frozen_turn_id=turn_id, _background_memory=False) as reopened:
            result = await reopened.submit("keep the numbers in prose")
            assert result.status == "done"
            assert result.turn_id != turn_id
            assert reopened.store.read_turn(result.turn_id).predecessor_turn_id == turn_id
            assert reopened.store.read_turn(result.turn_id).thread_id == thread_id.strip()
            inputs = [item for item in reopened.store.list_conversation_items(result.turn_id)
                      if item.kind == "user_message" and item.turn_id in {turn_id, result.turn_id}]
            assert [item.payload["text"] for item in inputs] == [
                "remember the original task", "keep the numbers in prose"]
            assert reopened.store.read_turn(turn_id).status == "cancelled"
            assert reopened.store.read_turn(turn_id).binding_manifest["model_token_budget_total"] == 50000
            assert any(m.role == "user" and m.content == "keep the numbers in prose"
                       for m in model.requests[-1].messages)
            assert any(m.role == "user" and m.content == "remember the original task"
                       for m in model.requests[-1].messages)
            assert reopened.store.verify().valid
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


@pytest.mark.anyio
async def test_live_worker_cannot_be_stolen_or_receive_rejected_followup(tmp_path):
    started = asyncio.Event()

    class Waiting(PublicHarnessModel):
        async def dispatch(self, prepared, *, event_sink=None):
            started.set()
            await asyncio.Event().wait()

    database = tmp_path / "state.sqlite"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with await Session.open(database=database, workspace=workspace, model=Waiting(),
                                 _background_memory=False) as owner:
        task = asyncio.create_task(owner.submit("original"))
        await asyncio.wait_for(started.wait(), 5)
        turn_id = owner.store.read_thread(owner.thread_id).active_turn_id
        try:
            async with await Session.open(database=database, workspace=workspace, model=PublicHarnessModel(),
                                         thread_id=owner.thread_id, _background_memory=False) as other:
                before = other.store.list_records(owner.thread_id)
                with pytest.raises(RuntimeError, match="另一进程|another process"):
                    await other.submit("must not start another loop")
                with pytest.raises(RuntimeError, match="另一进程|another process"):
                    await other.resume(turn_id, "continue", user_input="must not be received")
                assert other.store.list_records(owner.thread_id) == before
                from agent_runtime import Agent
                facade = Agent(checkpoint_db=database, workspace_path=workspace)
                with pytest.raises(RuntimeError, match="另一进程|another process"):
                    await facade.resume(turn_id, "abort")
                assert other.store.list_records(owner.thread_id) == before
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_legacy_unexpired_dispatch_is_not_claimed(tmp_path):
    database = tmp_path / "state.sqlite"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with (await RuntimeState.open(database)) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="old worker", binding_manifest={})
        operation = store.prepare_model_operation(turn_id=turn.turn_id, request_hash="r", context_hash="c",
            tool_hash="t", wire_hash="w", request_ref={"input_revision": 1, "request_id": "legacy"})
        store.dispatch_model_attempt(operation.operation_id, worker_id="legacy", lease_seconds=300)
    async with await Session.open(database=database, workspace=workspace, model=PublicHarnessModel(),
                                 thread_id=thread.thread_id, _background_memory=False) as session:
        before = session.store.list_records(thread.thread_id)
        with pytest.raises(RuntimeError, match="租约|lease"):
            await session.resume(turn.turn_id, "continue", user_input="not accepted")
        assert session.store.list_records(thread.thread_id) == before


@pytest.mark.anyio
async def test_committed_but_unstarted_tool_group_is_closed_without_execution(tmp_path):
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with await Session.open(database=tmp_path / "state.sqlite", workspace=workspace,
                                 model=PublicHarnessModel(), _background_memory=False) as session:
        old = session.store.start_turn(thread_id=session.thread_id, user_message="old task",
            binding_manifest={"execution_ownership": OWNERSHIP_REVISION})
        operation = session.store.prepare_model_operation(turn_id=old.turn_id, request_hash="r", context_hash="c",
            tool_hash="t", wire_hash="w",
            request_ref={"input_revision": 1, "request_id": "old-response", "purpose": "agent_step"})
        attempt = session.store.dispatch_model_attempt(operation.operation_id, worker_id="dead")
        session.store.complete_model_attempt(operation_id=operation.operation_id, attempt_id=attempt.attempt_id,
            generation=attempt.generation, text="planned tool", provider_response_id="old", usage={},
            tool_calls=({"id": "never-started", "name": "run_command", "arguments": {"command": "must not run"}},))
        result = await session.submit("change direction")
        assert result.status == "done"
        assert session.store.read_turn(old.turn_id).status == "cancelled"
        assert session.store.list_tool_operations(old.turn_id) == ()
        results = [i for i in session.store.list_items(old.turn_id) if i.kind == "tool_result"]
        assert len(results) == 1 and results[0].payload["error_code"] == "not_executed"
        from agent_runtime.harness.context import _validate_tool_groups
        _validate_tool_groups(session.context_manager.build(result.turn_id))
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_parallel_abandoned_tools_are_reconciled_one_at_a_time(tmp_path):
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with await Session.open(database=tmp_path / "state.sqlite", workspace=workspace,
                                 model=PublicHarnessModel(), _background_memory=False) as session:
        turn = session.store.start_turn(thread_id=session.thread_id, user_message="parallel tools",
                                       binding_manifest={"execution_ownership": OWNERSHIP_REVISION})
        for number in range(2):
            values = dict(turn_id=turn.turn_id, operation_id=f"operation-{number}", tool_call_id=f"call-{number}",
                tool_name="write_file", arguments_digest="args", execution_revision="write-v1", idempotent=False,
                attempt_count=0, error_code=None, requires_reconciliation=False)
            session.store.record_tool_execution_state(status="prepared", **values)
            session.store.record_tool_execution_state(status="ready", **values)
            session.store.claim_tool_operation(operation_id=values["operation_id"], worker_id="dead", lease_seconds=300)
        for number in range(2):
            result = await session.resume(turn.turn_id, "continue")
            assert result.pause is not None and result.pause.kind == "tool_reconciliation"
            pending = [i for i in session.store.list_interactions(turn.turn_id) if i.status == "pending"]
            assert len(pending) == 1 and pending[0].operation_id == f"operation-{number}"
            session.store.resolve_tool_reconciliation(turn_id=turn.turn_id, status="failed",
                result={"tool_call_id": f"call-{number}", "tool_name": "write_file", "is_error": True,
                        "content": [], "attachments": []}, reconciler_revision=f"independent-check-{number}")
        assert all(op.status == "failed" for op in session.store.list_tool_operations(turn.turn_id))
        assert session.store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("old_capacity_limit", [None, 100])
async def test_safe_paused_execution_accepts_new_turn(tmp_path, old_capacity_limit):
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with await Session.open(database=tmp_path / "state.sqlite", workspace=workspace,
                                 model=PublicHarnessModel(), _background_memory=False) as session:
        binding = {"execution_ownership": OWNERSHIP_REVISION}
        if old_capacity_limit is not None:
            binding["context_capacity_policy"] = {"max_total_bytes": old_capacity_limit}
        old = session.store.start_turn(thread_id=session.thread_id, user_message="old task " * 30,
                                      binding_manifest=binding)
        session.store.request_turn_cancellation(turn_id=old.turn_id)
        session.store.pause_turn(turn_id=old.turn_id, reason="tool execution stopped safely")
        result = await session.submit("continue after safe cancellation")
        assert result.status == "done"
        assert session.store.read_turn(old.turn_id).status == "cancelled"


@pytest.mark.anyio
async def test_owner_died_before_first_model_request(tmp_path):
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with await Session.open(database=tmp_path / "state.sqlite", workspace=workspace,
                                 model=PublicHarnessModel(), _background_memory=False) as session:
        binding = dict(session.binding_provider.snapshot(thread_id=session.thread_id, turn_id="turn_gap"))
        binding["execution_ownership"] = OWNERSHIP_REVISION
        session.store.start_turn(thread_id=session.thread_id, turn_id="turn_gap", user_message="original",
                                 binding_manifest=binding)
        result = await session.submit("follow up after crash")
        assert result.status == "done"
        assert len(session.store.list_turns(session.thread_id)) == 2
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_resumed_execution_accepts_live_steering_and_a_second_cancellation(tmp_path):
    started = asyncio.Event()

    class Waiting(PublicHarnessModel):
        async def dispatch(self, prepared, *, event_sink=None):
            started.set()
            await asyncio.Event().wait()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with await Session.open(database=tmp_path / "state.sqlite", workspace=workspace,
                                 model=Waiting(), _background_memory=False) as session:
        first = asyncio.create_task(session.submit("original task"))
        await asyncio.wait_for(started.wait(), 5)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        turn_id = session.store.read_thread(session.thread_id).active_turn_id
        started.clear()
        resumed = asyncio.create_task(session.resume(turn_id, "continue"))
        try:
            await asyncio.wait_for(started.wait(), 5)
            receipt = await session.steer("additional live guidance", expected_turn_id=turn_id)
            assert receipt.payload["text"] == "additional live guidance"
        finally:
            resumed.cancel()
            await asyncio.gather(resumed, return_exceptions=True)
        requests = [r for r in session.store.list_records(session.thread_id)
                    if r.record_type == "turn_cancellation_requested"]
        assert len(requests) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["abort", "invalid"])
async def test_pre_dispatch_orphan_abort_and_invalid_action(tmp_path, action):
    from agent_runtime import Agent
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "state.sqlite"
    async with (await RuntimeState.open(database)) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="original",
                               binding_manifest={"execution_ownership": OWNERSHIP_REVISION})
    if action == "abort":
        facade = Agent(checkpoint_db=database, workspace_path=workspace)
        result = await facade.resume(turn.turn_id, "abort")
        assert result.stop_reason == "cancelled"
    else:
        async with await Session.open(database=database, workspace=workspace, thread_id=thread.thread_id,
                                     model=PublicHarnessModel(), _background_memory=False) as session:
            with pytest.raises(RuntimeError, match="resume action"):
                await session.resume(turn.turn_id, action)
            assert session.store.list_model_operations(turn.turn_id) == ()


@pytest.mark.anyio
@pytest.mark.parametrize("legacy", [False, True])
async def test_interrupted_tool_requires_reconciliation_and_cannot_be_aborted(tmp_path, legacy):
    from agent_runtime import Agent
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION, ConversationRecoveryRequiredError
    from tests.agent.harness.test_tool_claim_fencing import _ready_operation
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "state.sqlite"
    async with await Session.open(database=database, workspace=workspace, model=PublicHarnessModel(),
                                 _background_memory=False) as session:
        binding = {} if legacy else {"execution_ownership": OWNERSHIP_REVISION}
        turn = session.store.start_turn(thread_id=session.thread_id, user_message="side effect",
                                        binding_manifest=binding)
        operation_id = _ready_operation(session.store, turn.turn_id)
        session.store.claim_tool_operation(operation_id=operation_id, worker_id="dead", lease_seconds=300,
                                          **({"now": 100} if legacy else {}))
        result = await session.resume(turn.turn_id, "continue", user_input="check what happened")
        assert result.status == "paused" and result.pause.kind == "tool_reconciliation"
        operation = session.store.read_tool_operation(operation_id)
        assert operation.status == "unknown" and operation.requires_reconciliation
        facade = Agent(checkpoint_db=database, workspace_path=workspace)
        with pytest.raises(ConversationRecoveryRequiredError, match="取消未完成"):
            await facade.resume(turn.turn_id, "abort")
        assert session.store.read_thread(session.thread_id).active_turn_id == turn.turn_id
        assert session.store.verify().valid

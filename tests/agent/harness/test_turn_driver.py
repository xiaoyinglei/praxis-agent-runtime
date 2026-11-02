"""Public behavior and durable response consumption across process lifetimes."""

from __future__ import annotations

import asyncio

from agent_runtime.harness import RuntimeStore, model_ops, turn_ops
from agent_runtime.harness.execution_facts import response_handled
from agent_runtime.harness.ownership import OWNERSHIP_REVISION
from agent_runtime.harness.protocol import HarnessModelResponse
from agent_runtime.harness.recovery import ContinuationKind, recovery_plan


def test_completed_response_receipt_selects_next_step_after_rebuild(tmp_path):
    database = tmp_path / "run.db"
    with RuntimeStore.open_sync(database) as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="work",
            binding_manifest={"execution_ownership": OWNERSHIP_REVISION},
        )
        op = model_ops.prepare_model_operation(
            store,
            turn_id=turn.turn_id,
            request_hash="r",
            context_hash="c",
            tool_hash="t",
            wire_hash="w",
            request_ref={"request_id": "request", "input_revision": 1},
        )
        claim = model_ops.dispatch_model_attempt(store, op.operation_id)
        model_ops.complete_model_attempt(
            store,
            operation_id=op.operation_id,
            attempt_id=claim.attempt_id,
            generation=claim.generation,
            response=HarnessModelResponse.from_parts(text="continue"),
        )
        ready = recovery_plan(store, turn.turn_id)
        assert ready.kind == ContinuationKind.COMMITTED_RESPONSE
        assert ready.operation_id == op.operation_id
        assert ready.response_item_id is not None
        response_handled(
            store,
            turn_id=turn.turn_id,
            operation_id=op.operation_id,
            response_item_id=ready.response_item_id,
            disposition="continue",
        )
        before = store.list_turn_records(turn.turn_id)
        response_handled(
            store,
            turn_id=turn.turn_id,
            operation_id=op.operation_id,
            response_item_id=ready.response_item_id,
            disposition="continue",
        )
        assert store.list_turn_records(turn.turn_id) == before
        store.rebuild_projections()
        assert store.verify().valid
    with RuntimeStore.open_sync(database) as reopened:
        assert recovery_plan(reopened, turn.turn_id).kind == ContinuationKind.NEXT_STEP


def test_public_prefix_text_is_data_and_second_turn_uses_same_driver(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from tests.agent.harness.test_context_prefix_updates import Probe

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = Probe("answer")
    agent = Agent(workspace_path=workspace, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)

    async def run():
        async with agent.session(_background_memory=False) as session:
            first = await session.submit("Runtime budget status:\nordinary log data")
            second = await session.submit("Explain what remains")
            assert first.status == second.status == "done"
            assert session.store.verify().valid

    asyncio.run(run())

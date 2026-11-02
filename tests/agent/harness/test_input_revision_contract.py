from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from agent_runtime.harness import Session, model_ops, turn_ops
from agent_runtime.harness.protocol import HarnessModelResponse
from agent_runtime.harness.runtime_store import RuntimeStore


@pytest.mark.parametrize("revision", [None, True, False, 0, -1, "1", 1.0])
def test_new_model_requests_require_a_positive_integer_revision(tmp_path: Path, revision):
    with RuntimeStore.open_sync(tmp_path / "run.db") as state:
        thread = turn_ops.create_thread(state, workspace=tmp_path)
        turn = turn_ops.start_turn(state, thread_id=thread.thread_id, user_message="task", binding_manifest={})
        records = state.list_records(thread.thread_id)
        ref = {"request_id": "request", "input_revision": revision}
        if revision is None:
            del ref["input_revision"]
        with pytest.raises(ValueError, match="input_revision.*positive integer"):
            model_ops.prepare_model_operation(
                state,
                turn_id=turn.turn_id,
                request_hash="r",
                context_hash="c",
                tool_hash="t",
                wire_hash="w",
                request_ref=ref,
            )
        assert state.list_records(thread.thread_id) == records
        assert state.list_model_operations(turn.turn_id) == ()


@pytest.mark.anyio
@pytest.mark.parametrize("revision", [None, True, 0, -1, "1"])
async def test_resume_rejects_unsupported_requests_before_any_mutation(tmp_path, monkeypatch, revision):
    from tests.agent.harness.test_rollout_integrity_boundary import OpenOnlyModel

    database = tmp_path / "run.db"
    async with await RuntimeStore.open(database) as state:
        thread = await state.run_async(partial(turn_ops.create_thread, state), workspace=tmp_path)
        turn = await state.run_async(
            partial(turn_ops.start_turn, state),
            thread_id=thread.thread_id,
            user_message="old task",
            binding_manifest={},
        )
        # Emulate an older producer; normal admission must reject this shape.
        with monkeypatch.context() as patch:
            patch.setattr(state, "request_input_revision", lambda *_args: 1)
            await state.run_async(
                partial(model_ops.prepare_model_operation, state),
                turn_id=turn.turn_id,
                request_hash="r",
                context_hash="c",
                tool_hash="t",
                wire_hash="w",
                request_ref={"request_id": "old", **({"input_revision": revision} if revision is not None else {})},
            )
    async with await Session.open(
        database=database,
        workspace=tmp_path,
        thread_id=thread.thread_id,
        model=OpenOnlyModel(),
    ) as session:
        before = await session.store.run_async(session.store.list_records, thread.thread_id)
        with pytest.raises(ValueError, match="input_revision.*start a new Turn"):
            await session.resume(turn.turn_id, "continue")
        assert await session.store.run_async(session.store.list_records, thread.thread_id) == before


@pytest.mark.anyio
async def test_unsupported_prepared_request_still_allows_safe_abort(tmp_path, monkeypatch):
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION
    from tests.agent.harness.test_rollout_integrity_boundary import OpenOnlyModel

    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=OpenOnlyModel()) as session:
        turn = await session.store.run_async(
            partial(turn_ops.start_turn, session.store),
            thread_id=session.thread_id,
            user_message="old task",
            binding_manifest={"execution_ownership": OWNERSHIP_REVISION},
        )
        with monkeypatch.context() as patch:
            patch.setattr(session.store, "request_input_revision", lambda *_args: 1)
            await session.store.run_async(
                partial(model_ops.prepare_model_operation, session.store),
                turn_id=turn.turn_id,
                request_hash="r",
                context_hash="c",
                tool_hash="t",
                wire_hash="w",
                request_ref={"request_id": "old"},
            )
        result = await session.resume(turn.turn_id, "abort")
        assert result.status == "failed"  # Public AgentResult groups cancelled Turns under failed.
        assert (await session.store.run_async(session.store.read_turn, turn.turn_id)).status == "cancelled"
        assert (await session.store.run_async(session.store.read_thread, session.thread_id)).active_turn_id is None


@pytest.mark.parametrize("route", ["clarification", "committed_summary"])
def test_low_level_recovery_rejects_old_requests_before_mutation_or_preparation(tmp_path, monkeypatch, route):
    import asyncio

    from agent_runtime.harness import DeliveryCompletionGate, RolloutContextManager, TurnRunner
    from tests.agent.harness.test_rollout_integrity_boundary import OpenOnlyModel

    with RuntimeStore.open_sync(tmp_path / "run.db") as state:
        thread = turn_ops.create_thread(state, workspace=tmp_path)
        turn = turn_ops.start_turn(state, thread_id=thread.thread_id, user_message="old task", binding_manifest={})
        with monkeypatch.context() as patch:
            patch.setattr(state, "request_input_revision", lambda *_args: 1)
            operation = model_ops.prepare_model_operation(
                state,
                turn_id=turn.turn_id,
                request_hash="r",
                context_hash="c",
                tool_hash="t",
                wire_hash="w",
                request_ref={
                    "request_id": "old",
                    "purpose": "context_summary" if route == "committed_summary" else "agent_step",
                    "step_snapshot": {"step": 1},
                },
            )
            attempt = model_ops.dispatch_model_attempt(state, operation.operation_id)
            model_ops.complete_model_attempt(
                state,
                operation_id=operation.operation_id,
                attempt_id=attempt.attempt_id,
                generation=attempt.generation,
                response=HarnessModelResponse.from_parts(text="old response", provider_response_id="old", usage={}),
            )
        if route == "clarification":
            interaction = turn_ops.request_clarification(state, turn_id=turn.turn_id, question="old question")
        executor = TurnRunner(
            thread_id=thread.thread_id,
            store=state,
            model=OpenOnlyModel(),
            context_manager=RolloutContextManager(state),
            completion_gate=DeliveryCompletionGate(state),
        )
        before = state.list_records(thread.thread_id)
        with pytest.raises(ValueError, match="input_revision.*start a new Turn"):
            if route == "clarification":
                asyncio.run(
                    executor.respond_interaction(
                        turn_id=turn.turn_id,
                        request_id=interaction.request_id,
                        response="reply",
                    )
                )
            else:
                asyncio.run(executor.drive(action="continue", turn_id=turn.turn_id))
        assert state.list_records(thread.thread_id) == before


def test_retained_terminal_history_is_readable_without_enabling_old_execution(tmp_path, monkeypatch):
    from agent_runtime.history import read_chat_messages

    database = tmp_path / "run.db"
    with RuntimeStore.open_sync(database) as state:
        thread = turn_ops.create_thread(state, workspace=tmp_path)
        old = turn_ops.start_turn(state, thread_id=thread.thread_id, user_message="old task", binding_manifest={})
        with monkeypatch.context() as patch:
            patch.setattr(state, "request_input_revision", lambda *_args: 1)
            operation = model_ops.prepare_model_operation(
                state,
                turn_id=old.turn_id,
                request_hash="r",
                context_hash="c",
                tool_hash="t",
                wire_hash="w",
                request_ref={"request_id": "old", "purpose": "agent_step"},
            )
            attempt = model_ops.dispatch_model_attempt(state, operation.operation_id)
            model_ops.complete_model_attempt(
                state,
                operation_id=operation.operation_id,
                attempt_id=attempt.attempt_id,
                generation=attempt.generation,
                response=HarnessModelResponse.from_parts(text="old answer", provider_response_id="old", usage={}),
            )
        receipt = turn_ops.receive_turn_input(
            state, thread_id=thread.thread_id, turn_id=old.turn_id, text="old supplement"
        )
        turn_ops.complete_turn(state, turn_id=old.turn_id, answer="old answer")
        current = turn_ops.start_turn(state, thread_id=thread.thread_id, user_message="new task", binding_manifest={})
        turn_ops.complete_turn(state, turn_id=current.turn_id, answer="new answer")
        before = state.list_records(thread.thread_id)
        messages = read_chat_messages(database, thread_id=thread.thread_id, workspace=tmp_path)
        assert ("answer", "old answer") in messages and ("answer", "new answer") in messages
        assert state.turn_input_status(receipt.item_id) == "unconsumed"
        assert state.list_records(thread.thread_id) == before

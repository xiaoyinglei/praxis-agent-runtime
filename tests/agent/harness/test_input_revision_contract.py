from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.harness import Session
from agent_runtime.harness.protocol import HarnessModelResponse
from agent_runtime.harness.state import RuntimeState


@pytest.mark.parametrize("revision", [None, True, False, 0, -1, "1", 1.0])
def test_new_model_requests_require_a_positive_integer_revision(tmp_path: Path, revision):
    with RuntimeState.open_sync(tmp_path / "run.db") as state:
        thread = state.create_thread(workspace=tmp_path)
        turn = state.start_turn(thread_id=thread.thread_id, user_message="task", binding_manifest={})
        records = state.list_records(thread.thread_id)
        ref = {"request_id": "request", "input_revision": revision}
        if revision is None:
            del ref["input_revision"]
        with pytest.raises(ValueError, match="input_revision.*positive integer"):
            state.prepare_model_operation(
                turn_id=turn.turn_id, request_hash="r", context_hash="c", tool_hash="t", wire_hash="w",
                request_ref=ref,
            )
        assert state.list_records(thread.thread_id) == records
        assert state.list_model_operations(turn.turn_id) == ()


@pytest.mark.anyio
@pytest.mark.parametrize("revision", [None, True, 0, -1, "1"])
async def test_resume_rejects_unsupported_requests_before_any_mutation(tmp_path, monkeypatch, revision):
    from tests.agent.harness.test_rollout_integrity_boundary import OpenOnlyModel

    database = tmp_path / "run.db"
    async with await RuntimeState.open(database) as state:
        thread = await state.run_async(state.create_thread, workspace=tmp_path)
        turn = await state.run_async(
            state.start_turn, thread_id=thread.thread_id, user_message="old task", binding_manifest={},
        )
        # Emulate an older producer; normal admission must reject this shape.
        with monkeypatch.context() as patch:
            patch.setattr(state, "request_input_revision", lambda *_args: 1)
            await state.run_async(
                state.prepare_model_operation,
                turn_id=turn.turn_id, request_hash="r", context_hash="c", tool_hash="t", wire_hash="w",
                request_ref={"request_id": "old", **({"input_revision": revision} if revision is not None else {})},
            )
    async with await Session.open(
        database=database, workspace=tmp_path, thread_id=thread.thread_id, model=OpenOnlyModel(),
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
            session.store.start_turn, thread_id=session.thread_id, user_message="old task",
            binding_manifest={"execution_ownership": OWNERSHIP_REVISION},
        )
        with monkeypatch.context() as patch:
            patch.setattr(session.store, "request_input_revision", lambda *_args: 1)
            await session.store.run_async(
                session.store.prepare_model_operation,
                turn_id=turn.turn_id, request_hash="r", context_hash="c", tool_hash="t", wire_hash="w",
                request_ref={"request_id": "old"},
            )
        result = await session.resume(turn.turn_id, "abort")
        assert result.status == "failed"  # Public AgentResult groups cancelled Turns under failed.
        assert (await session.store.run_async(session.store.read_turn, turn.turn_id)).status == "cancelled"
        assert (await session.store.run_async(session.store.read_thread, session.thread_id)).active_turn_id is None


@pytest.mark.parametrize("route", ["clarification", "committed_summary"])
def test_low_level_recovery_rejects_old_requests_before_mutation_or_preparation(tmp_path, monkeypatch, route):
    import asyncio

    from agent_runtime.harness import DeliveryCompletionGate, RolloutContextManager, TurnExecutor
    from tests.agent.harness.test_rollout_integrity_boundary import OpenOnlyModel

    with RuntimeState.open_sync(tmp_path / "run.db") as state:
        thread = state.create_thread(workspace=tmp_path)
        turn = state.start_turn(thread_id=thread.thread_id, user_message="old task", binding_manifest={})
        with monkeypatch.context() as patch:
            patch.setattr(state, "request_input_revision", lambda *_args: 1)
            operation = state.prepare_model_operation(
                turn_id=turn.turn_id, request_hash="r", context_hash="c", tool_hash="t", wire_hash="w",
                request_ref={"request_id": "old", "purpose": "context_summary" if route == "committed_summary"
                             else "agent_step", "step_snapshot": {"step": 1}},
            )
            attempt = state.dispatch_model_attempt(operation.operation_id)
            state.complete_model_attempt(operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            response=HarnessModelResponse.from_parts(text="old response", provider_response_id="old", usage={}))
        if route == "clarification":
            interaction = state.request_clarification(turn_id=turn.turn_id, question="old question")
        executor = TurnExecutor(
            thread_id=thread.thread_id, store=state, model=OpenOnlyModel(),
            context_manager=RolloutContextManager(state), completion_gate=DeliveryCompletionGate(state),
        )
        before = state.list_records(thread.thread_id)
        with pytest.raises(ValueError, match="input_revision.*start a new Turn"):
            if route == "clarification":
                asyncio.run(executor.respond_interaction(
                    turn_id=turn.turn_id, request_id=interaction.request_id, response="reply",
                ))
            else:
                asyncio.run(executor.recover_committed_model_response(turn_id=turn.turn_id))
        assert state.list_records(thread.thread_id) == before


def test_retained_terminal_history_is_readable_without_enabling_old_execution(tmp_path, monkeypatch):
    from agent_runtime.history import read_chat_messages

    database = tmp_path / "run.db"
    with RuntimeState.open_sync(database) as state:
        thread = state.create_thread(workspace=tmp_path)
        old = state.start_turn(thread_id=thread.thread_id, user_message="old task", binding_manifest={})
        with monkeypatch.context() as patch:
            patch.setattr(state, "request_input_revision", lambda *_args: 1)
            operation = state.prepare_model_operation(
                turn_id=old.turn_id, request_hash="r", context_hash="c", tool_hash="t", wire_hash="w",
                request_ref={"request_id": "old", "purpose": "agent_step"},
            )
            attempt = state.dispatch_model_attempt(operation.operation_id)
            state.complete_model_attempt(operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            response=HarnessModelResponse.from_parts(text="old answer", provider_response_id="old", usage={}))
        receipt = state.receive_turn_input(thread_id=thread.thread_id, turn_id=old.turn_id, text="old supplement")
        state.complete_turn(turn_id=old.turn_id, answer="old answer")
        current = state.start_turn(thread_id=thread.thread_id, user_message="new task", binding_manifest={})
        state.complete_turn(turn_id=current.turn_id, answer="new answer")
        before = state.list_records(thread.thread_id)
        messages = read_chat_messages(database, thread_id=thread.thread_id, workspace=tmp_path)
        assert ("answer", "old answer") in messages and ("answer", "new answer") in messages
        assert state.turn_input_status(receipt.item_id) == "unconsumed"
        assert state.list_records(thread.thread_id) == before


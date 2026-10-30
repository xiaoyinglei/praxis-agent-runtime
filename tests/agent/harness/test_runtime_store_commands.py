"""Owner closure and transaction guarantees of the explicit command API."""

from __future__ import annotations

from functools import partial

import pytest

from agent_runtime.harness import turn_ops
from agent_runtime.harness.runtime_store import RuntimeStore


def test_transact_returns_one_atomic_receipt_and_rolls_back_nested_commands(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "owner.db") as store:
        receipt = store.transact(lambda tx: turn_ops.create_thread(store=store, workspace=tmp_path))
        thread = receipt.value
        assert receipt.records and {fact.record_type for fact in receipt.records} == {"thread_created"}
        before = store.projection_hashes()

        def fail(tx):
            turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="rollback", binding_manifest={})
            tx.put_metadata("rollback-marker", "uncommitted", replace=True)
            raise ValueError("failed complete command")

        with pytest.raises(ValueError, match="failed complete command"):
            store.transact(fail)
        assert store.projection_hashes() == before
        assert store.tx.metadata("rollback-marker") is None
        assert store.verify().valid


def test_read_snapshot_remains_consistent_during_another_owner_commit(tmp_path):
    path = tmp_path / "snapshot.db"
    with RuntimeStore.open_sync(path) as reader, RuntimeStore.open_sync(path) as writer:
        writer.transact(lambda tx: tx.put_metadata("revision", "before", replace=True))

        def query():
            before = reader.tx.metadata("revision")
            writer.transact(lambda tx: tx.put_metadata("revision", "after", replace=True))
            return before, reader.tx.metadata("revision")

        assert reader.read_snapshot(query) == ("before", "before")
        assert reader.read_snapshot(lambda: reader.tx.metadata("revision")) == "after"


@pytest.mark.anyio
async def test_explicit_command_callback_runs_on_async_owner(tmp_path):
    async with await RuntimeStore.open(tmp_path / "async.db") as store:
        thread = await store.run_async(partial(turn_ops.create_thread, store), workspace=tmp_path)
        assert (await store.run_async(store.read_thread, thread.thread_id)).workspace == str(tmp_path)
        assert not hasattr(store, "create_thread")


def test_frozen_request_artifact_is_atomic_and_duplicate_prepare_reuses_it(tmp_path):
    import json

    from agent_runtime.harness import model_ops

    with RuntimeStore.open_sync(tmp_path / "frozen.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="freeze", binding_manifest={})
        request_ref = {
            "request_id": "frozen-request-1",
            "input_revision": 1,
            "frozen_request_payload": {
                "adapter": "test-v1",
                "wire": {"messages": [{"role": "user", "content": "freeze"}]},
            },
        }
        args = dict(
            turn_id=turn.turn_id,
            request_hash="r",
            context_hash="c",
            tool_hash="t",
            wire_hash="w",
            request_ref=request_ref,
        )
        receipt = store.capture_mutation(lambda: model_ops.prepare_model_operation(store, **args))
        operation = receipt.value
        artifact_id = operation.request_ref["frozen_request_artifact"]
        assert "frozen_request_payload" not in operation.request_ref
        assert json.loads(store.read_artifact(artifact_id)) == request_ref["frozen_request_payload"]
        assert {fact.record_type for fact in receipt.records} >= {"artifact_committed", "model_operation_prepared"}
        before = store.list_records(thread.thread_id)
        assert model_ops.prepare_model_operation(store, **args).operation_id == operation.operation_id
        assert store.list_records(thread.thread_id) == before
        changed = {**request_ref, "frozen_request_payload": {"adapter": "changed"}}
        with pytest.raises(RuntimeError, match="conflicting frozen payload"):
            model_ops.prepare_model_operation(store, **{**args, "request_ref": changed})
        assert len(store.list_artifacts(turn.turn_id)) == 1
        assert store.verify().valid


def test_provider_usage_preserves_nested_immutable_details(tmp_path):
    from agent_runtime.harness import model_ops
    from agent_runtime.harness.protocol import HarnessModelResponse

    with RuntimeStore.open_sync(tmp_path / "usage.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="work", binding_manifest={})
        operation = model_ops.prepare_model_operation(
            store,
            turn_id=turn.turn_id,
            request_hash="r",
            context_hash="c",
            tool_hash="t",
            wire_hash="w",
            request_ref={"request_id": "usage", "input_revision": 1},
        )
        attempt = model_ops.dispatch_model_attempt(store, operation.operation_id)
        response = HarnessModelResponse.from_parts(
            text="answer",
            usage={
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
                "raw_usage": {"prompt_tokens_details": {"cached_tokens": 1}},
            },
        )
        assert model_ops.complete_model_attempt(
            store,
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            response=response,
        )
        [saved] = store.list_model_attempts(operation.operation_id)
        assert saved.usage["raw_usage"]["prompt_tokens_details"]["cached_tokens"] == 1
        assert store.read_budget_state(turn.turn_id).used.total_tokens == 5
        store.rebuild_projections()
        assert store.verify().valid

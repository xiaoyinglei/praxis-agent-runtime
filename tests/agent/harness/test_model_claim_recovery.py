from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from agent_runtime.harness import RuntimeStore, model_ops, turn_ops
from agent_runtime.harness.protocol import HarnessModelResponse


def test_two_processes_share_one_logical_model_operation_and_only_one_dispatches(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RuntimeStore.open_sync(database) as store:
        thread = turn_ops.create_thread(store, workspace=workspace)
        turn = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="answer once",
            binding_manifest={"model_id": "test-model"},
        )
    prepared = Barrier(2)

    def compete(worker_id: str) -> tuple[str, str]:
        with RuntimeStore.open_sync(database) as store:
            operation = model_ops.prepare_model_operation(
                store,
                turn_id=turn.turn_id,
                request_hash="request-hash",
                context_hash="context-hash",
                tool_hash="tool-hash",
                wire_hash="wire-hash",
                request_ref={"input_revision": 1, "request_id": f"{turn.turn_id}:step:1"},
            )
            prepared.wait()
            try:
                model_ops.dispatch_model_attempt(store, operation.operation_id, worker_id=worker_id)
            except RuntimeError:
                return operation.operation_id, "lost"
            return operation.operation_id, "dispatched"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(compete, ("model-worker-a", "model-worker-b")))

    assert len({operation_id for operation_id, _ in outcomes}) == 1
    assert sorted(status for _, status in outcomes) == ["dispatched", "lost"]
    with RuntimeStore.open_sync(database) as verifier:
        [operation] = verifier.list_model_operations(turn.turn_id)
        [attempt] = verifier.list_model_attempts(operation.operation_id)
        assert operation.status == "dispatched"
        assert attempt.status == "dispatched"
        assert attempt.claim_owner in {"model-worker-a", "model-worker-b"}
        with pytest.raises(RuntimeError, match="conflicting payload"):
            model_ops.prepare_model_operation(
                verifier,
                turn_id=turn.turn_id,
                request_hash="different-request-hash",
                context_hash="context-hash",
                tool_hash="tool-hash",
                wire_hash="wire-hash",
                request_ref={"input_revision": 1, "request_id": f"{turn.turn_id}:step:1"},
            )
        assert verifier.verify().valid is True


def test_fresh_store_recovers_expired_model_dispatch_and_fences_late_response(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RuntimeStore.open_sync(database) as first:
        thread = turn_ops.create_thread(first, workspace=workspace)
        turn = turn_ops.start_turn(
            first,
            thread_id=thread.thread_id,
            user_message="answer once",
            binding_manifest={"model_id": "test-model"},
        )
        operation = model_ops.prepare_model_operation(
            first,
            turn_id=turn.turn_id,
            request_hash="request-hash",
            context_hash="context-hash",
            tool_hash="tool-hash",
            wire_hash="wire-hash",
            request_ref={"input_revision": 1, "request_id": "request-1"},
        )
        attempt = model_ops.dispatch_model_attempt(
            first,
            operation.operation_id,
            worker_id="model-worker-a",
            now=100.0,
            lease_seconds=5.0,
        )

    with RuntimeStore.open_sync(database) as restarted:
        recovered = model_ops.expire_model_attempt_dispatch(
            restarted,
            operation_id=operation.operation_id,
            now=106.0,
        )
        late = model_ops.complete_model_attempt(
            restarted,
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            response=HarnessModelResponse.from_parts(
                text="late answer", provider_response_id="late-response", usage={"input_tokens": 3, "output_tokens": 2}
            ),
        )

        assert recovered.status == "unknown"
        assert recovered.claim_owner == "model-worker-a"
        assert late is False
        assert restarted.read_turn(turn.turn_id).status == "interrupted"
        assert restarted.read_thread(thread.thread_id).active_turn_id == turn.turn_id
        assert not any(item.kind == "model_response" for item in restarted.list_items(turn.turn_id))
        assert restarted.verify().valid is True


def test_completion_past_lease_boundary_commits_while_takeover_has_not_run(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        thread = turn_ops.create_thread(store, workspace=workspace)
        turn = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="answer past the lease boundary",
            binding_manifest={"model_id": "test-model"},
        )
        operation = model_ops.prepare_model_operation(
            store,
            turn_id=turn.turn_id,
            request_hash="request-hash",
            context_hash="context-hash",
            tool_hash="tool-hash",
            wire_hash="wire-hash",
            request_ref={"input_revision": 1, "request_id": "request-1"},
        )
        attempt = model_ops.dispatch_model_attempt(
            store,
            operation.operation_id,
            worker_id="model-worker-a",
            now=100.0,
            lease_seconds=5.0,
        )

        accepted = model_ops.complete_model_attempt(
            store,
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            now=106.0,
            response=HarnessModelResponse.from_parts(
                text="billed answer", provider_response_id="response-1", usage={"input_tokens": 3, "output_tokens": 2}
            ),
        )

        assert accepted is True
        assert any(item.kind == "model_response" for item in store.list_items(turn.turn_id))
        assert store.verify().valid is True

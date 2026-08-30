from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from agent_runtime.harness import RolloutStore


def test_two_processes_share_one_logical_model_operation_and_only_one_dispatches(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="answer once",
            binding_manifest={"model_alias": "test-model"},
        )
    prepared = Barrier(2)

    def compete(worker_id: str) -> tuple[str, str]:
        with RolloutStore(database) as store:
            operation = store.prepare_model_operation(
                turn_id=turn.turn_id,
                request_hash="request-hash",
                context_hash="context-hash",
                tool_hash="tool-hash",
                wire_hash="wire-hash",
                request_ref={"request_id": f"{turn.turn_id}:step:1"},
            )
            prepared.wait()
            try:
                store.dispatch_model_attempt(operation.operation_id, worker_id=worker_id)
            except RuntimeError:
                return operation.operation_id, "lost"
            return operation.operation_id, "dispatched"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(compete, ("model-worker-a", "model-worker-b")))

    assert len({operation_id for operation_id, _ in outcomes}) == 1
    assert sorted(status for _, status in outcomes) == ["dispatched", "lost"]
    with RolloutStore(database) as verifier:
        [operation] = verifier.list_model_operations(turn.turn_id)
        [attempt] = verifier.list_model_attempts(operation.operation_id)
        assert operation.status == "dispatched"
        assert attempt.status == "dispatched"
        assert attempt.claim_owner in {"model-worker-a", "model-worker-b"}
        with pytest.raises(RuntimeError, match="conflicting payload"):
            verifier.prepare_model_operation(
                turn_id=turn.turn_id,
                request_hash="different-request-hash",
                context_hash="context-hash",
                tool_hash="tool-hash",
                wire_hash="wire-hash",
                request_ref={"request_id": f"{turn.turn_id}:step:1"},
            )
        assert verifier.verify().valid is True


def test_fresh_store_recovers_expired_model_dispatch_and_fences_late_response(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as first:
        thread = first.create_thread(workspace=workspace)
        turn = first.start_turn(
            thread_id=thread.thread_id,
            user_message="answer once",
            binding_manifest={"model_alias": "test-model"},
        )
        operation = first.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="request-hash",
            context_hash="context-hash",
            tool_hash="tool-hash",
            wire_hash="wire-hash",
            request_ref={"request_id": "request-1"},
        )
        attempt = first.dispatch_model_attempt(
            operation.operation_id,
            worker_id="model-worker-a",
            now=100.0,
            lease_seconds=5.0,
        )

    with RolloutStore(database) as restarted:
        recovered = restarted.expire_model_attempt_dispatch(
            operation_id=operation.operation_id,
            now=106.0,
        )
        late = restarted.complete_model_attempt(
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            text="late answer",
            provider_response_id="late-response",
            usage={"input_tokens": 3, "output_tokens": 2},
        )

        assert recovered.status == "unknown"
        assert recovered.claim_owner == "model-worker-a"
        assert late is False
        assert restarted.read_turn(turn.turn_id).status == "interrupted"
        assert restarted.read_thread(thread.thread_id).active_turn_id == turn.turn_id
        assert not any(
            item.kind == "model_response"
            for item in restarted.list_items(turn.turn_id)
        )
        assert restarted.verify().valid is True

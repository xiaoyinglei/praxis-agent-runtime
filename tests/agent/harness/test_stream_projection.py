from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutEventReader, RolloutStore


def _start_turn(store: RolloutStore, workspace: Path) -> tuple[str, str]:
    thread = store.create_thread(workspace=workspace)
    turn = store.start_turn(
        thread_id=thread.thread_id,
        user_message="project this turn",
        binding_manifest={"model_alias": "projection-test"},
    )
    return thread.thread_id, turn.turn_id


def _complete_model_response(
    store: RolloutStore,
    *,
    turn_id: str,
    text: str,
) -> None:
    operation = store.prepare_model_operation(
        turn_id=turn_id,
        request_hash=f"request-{turn_id}",
        context_hash=f"context-{turn_id}",
        tool_hash=f"tools-{turn_id}",
        wire_hash=f"wire-{turn_id}",
        request_ref={"request_id": f"request-{turn_id}"},
    )
    attempt = store.dispatch_model_attempt(
        operation.operation_id,
        worker_id="projection-worker",
        lease_seconds=30.0,
    )
    assert store.complete_model_attempt(
        operation_id=operation.operation_id,
        attempt_id=attempt.attempt_id,
        generation=attempt.generation,
        text=text,
        provider_response_id=f"response-{turn_id}",
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


def _completed_answer_count(projected: tuple[object, ...], answer: str) -> int:
    count = 0
    for replayed in projected:
        event = getattr(replayed, "event", None)
        if event is not None:
            if (
                event.type.value == "item_completed"
                and event.item_kind is not None
                and event.item_kind.value == "agent_message"
                and event.data.get("content") == answer
            ):
                count += 1
            continue
        if (
            getattr(replayed, "event_type", None) == "item_completed"
            and getattr(replayed, "data", {}).get("payload", {}).get("text")
            == answer
        ):
            count += 1
    return count


def test_unknown_internal_item_kind_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread_id, _turn_id = _start_turn(store, workspace)
        record = next(
            record
            for record in store.list_records(thread_id)
            if record.record_type == "item_started"
        )
        payload = {"item_id": record.payload["item_id"], "kind": "future_internal_kind"}
        payload_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                UPDATE rollout_records
                SET payload_json = ?, payload_hash = ?
                WHERE record_id = ?
                """,
                (
                    payload_json,
                    hashlib.sha256(payload_json.encode()).hexdigest(),
                    record.record_id,
                ),
            )

        with pytest.raises(RuntimeError, match="unknown internal Item kind"):
            RolloutEventReader(store).read(thread_id)


def test_public_item_ids_survive_reopen_and_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread_id, turn_id = _start_turn(store, workspace)
        operation = store.prepare_model_operation(
            turn_id=turn_id,
            request_hash="request-hash",
            context_hash="context-hash",
            tool_hash="tool-hash",
            wire_hash="wire-hash",
            request_ref={"request_id": "request-1"},
        )
        prepared = next(
            record
            for record in store.list_records(thread_id)
            if record.record_type == "model_operation_prepared"
        )
        first_ids = prepared.payload["public_item_ids"]
        assert set(first_ids) == {"agent_message", "reasoning", "plan"}

        store.dispatch_model_attempt(
            operation.operation_id,
            worker_id="worker-1",
            lease_seconds=1.0,
            now=100.0,
        )
        store.expire_model_attempt_dispatch(
            operation_id=operation.operation_id,
            now=102.0,
        )
        retry = store.prepare_model_retry(operation.operation_id)
        retried = next(
            record
            for record in store.list_records(thread_id)
            if record.record_type == "model_retry_prepared"
        )
        retry_ids = retried.payload["public_item_ids"]
        assert retry.attempt_id != operation.active_attempt_id
        assert retry_ids.keys() == first_ids.keys()
        assert all(retry_ids[channel] != first_ids[channel] for channel in first_ids)

        store.record_tool_call(
            turn_id=turn_id,
            tool_call_id="call-plan",
            tool_name="update_plan",
            arguments={
                "plan": [{"step": "project rollout", "status": "in_progress"}],
            },
            origin={"request_id": "request-1"},
        )
        tool_values = {
            "turn_id": turn_id,
            "operation_id": "operation-plan",
            "tool_call_id": "call-plan",
            "tool_name": "update_plan",
            "arguments_digest": "arguments-hash",
            "execution_revision": "builtin-update-plan-v1",
            "idempotent": True,
            "attempt_count": 0,
            "error_code": None,
            "requires_reconciliation": False,
        }
        store.record_tool_execution_state(status="prepared", **tool_values)
        store.record_tool_execution_state(status="ready", **tool_values)
        claim = store.claim_tool_operation(
            operation_id="operation-plan",
            worker_id="worker-1",
            lease_seconds=30.0,
        )
        claimed = next(
            record
            for record in store.list_records(thread_id)
            if record.record_type == "tool_operation_claimed"
        )
        operation_public_id = claimed.payload["public_item_id"]
        assert operation_public_id
        assert store.commit_tool_operation_outcome(
            operation_id="operation-plan",
            claim_generation=claim.claim_generation,
            fencing_token=str(claim.fencing_token),
            status="succeeded",
            attempt_count=1,
            error_code=None,
            requires_reconciliation=False,
        )
        store.record_tool_result(
            turn_id=turn_id,
            operation_id="operation-plan",
            result={
                "tool_call_id": "call-plan",
                "tool_name": "update_plan",
                "structured_content": {
                    "accepted": True,
                    "revision": 7,
                    "message": "updated",
                    "authority": "advisory",
                },
                "is_error": False,
                "content": [],
                "attachments": [],
            },
        )
        result_completion = next(
            record
            for record in store.list_records(thread_id)
            if record.record_type == "item_completed"
            and record.payload.get("public_item_id") == operation_public_id
        )
        plan_public_id = result_completion.payload["plan_public_item_id"]
        assert plan_public_id

    with RolloutStore(database) as reopened:
        persisted = reopened.list_records(thread_id)
        assert next(
            record.payload["public_item_ids"]
            for record in persisted
            if record.record_type == "model_retry_prepared"
        ) == retry_ids
        assert next(
            record.payload["plan_public_item_id"]
            for record in persisted
            if record.record_type == "item_completed"
            and record.payload.get("public_item_id") == operation_public_id
        ) == plan_public_id

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT record_id, payload_json
            FROM rollout_records
            WHERE record_type = 'model_retry_prepared'
            """
        ).fetchone()
        assert row is not None
        payload = json.loads(row["payload_json"])
        payload["public_item_ids"]["agent_message"] = "tampered-public-id"
        payload_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        connection.execute(
            """
            UPDATE rollout_records
            SET payload_json = ?, payload_hash = ?
            WHERE record_id = ?
            """,
            (
                payload_json,
                hashlib.sha256(payload_json.encode()).hexdigest(),
                row["record_id"],
            ),
        )

    with RolloutStore(database) as reopened:
        with pytest.raises(RuntimeError, match="persisted public Item ID mismatch"):
            RolloutEventReader(reopened).read(thread_id)


def test_accepted_answer_does_not_duplicate_model_response(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread_id, turn_id = _start_turn(store, workspace)
        _complete_model_response(store, turn_id=turn_id, text="one answer")
        store.complete_turn(turn_id=turn_id, answer="one answer")

        projected = RolloutEventReader(store).read(thread_id)

    assert _completed_answer_count(projected, "one answer") == 1

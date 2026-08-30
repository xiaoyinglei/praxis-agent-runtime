from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutEventReader, RolloutStore
from agent_runtime.streaming import events as stream_events


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


def test_migrated_answer_projects_exactly_one_legacy_message(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            turn_id="legacy-turn",
            turn_producer="migration",
            user_message="legacy question",
            binding_manifest={"model_alias": "legacy"},
        )
        store.complete_turn(
            turn_id=turn.turn_id,
            answer="legacy answer",
            producer="migration",
        )

        projected = RolloutEventReader(store).read(thread.thread_id)

    completed = [
        replayed
        for replayed in projected
        if (
            (
                replayed.event is not None
                and replayed.event.type.value == "item_completed"
            )
            if hasattr(replayed, "event")
            else getattr(replayed, "event_type", None) == "item_completed"
        )
    ]
    assert len(completed) == 1
    assert completed[0].event.item_kind.value == "legacy_message"
    assert completed[0].event.data == {"content": "legacy answer"}


def test_command_execution_does_not_create_a_second_command_outcome(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread_id, turn_id = _start_turn(store, workspace)
        store.record_tool_call(
            turn_id=turn_id,
            tool_call_id="call-command",
            tool_name="run_command",
            arguments={"command": ["pwd"]},
            origin={"request_id": "request-command"},
        )
        values = {
            "turn_id": turn_id,
            "operation_id": "operation-command",
            "tool_call_id": "call-command",
            "tool_name": "run_command",
            "arguments_digest": "command-arguments",
            "execution_revision": "builtin-run-command-v2-restricted-sandbox",
            "idempotent": False,
            "attempt_count": 0,
            "error_code": None,
            "requires_reconciliation": False,
        }
        store.record_tool_execution_state(status="prepared", **values)
        store.record_tool_execution_state(status="ready", **values)
        claim = store.claim_tool_operation(
            operation_id="operation-command",
            worker_id="command-worker",
            lease_seconds=30.0,
        )
        assert store.commit_tool_operation_outcome(
            operation_id="operation-command",
            claim_generation=claim.claim_generation,
            fencing_token=str(claim.fencing_token),
            status="succeeded",
            attempt_count=1,
            error_code=None,
            requires_reconciliation=False,
        )
        result = {
            "tool_call_id": "call-command",
            "tool_name": "run_command",
            "structured_content": {"exit_code": 0, "stdout": "workspace\n"},
            "is_error": False,
            "content": [],
            "attachments": [],
        }
        store.record_tool_result(
            turn_id=turn_id,
            operation_id="operation-command",
            result=result,
        )
        duplicate_id = "legacy-command-activity"
        with store._transaction():
            store._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="tool",
                payload={"item_id": duplicate_id, "kind": "command_execution"},
            )
            store._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="tool",
                payload={"item_id": duplicate_id, "payload": result},
            )

        projected = RolloutEventReader(store).read(thread_id)

    if any(getattr(replayed, "event", None) is not None for replayed in projected):
        outcomes = [
            replayed.event
            for replayed in projected
            if replayed.event is not None
            and replayed.event.type.value == "item_completed"
            and replayed.event.item_kind is not None
            and replayed.event.item_kind.value == "command"
        ]
    else:
        starts = {
            replayed.data["item_id"]: replayed.data["kind"]
            for replayed in projected
            if replayed.event_type == "item_started"
        }
        outcomes = [
            replayed
            for replayed in projected
            if replayed.event_type == "item_completed"
            and starts.get(replayed.data.get("item_id"))
            in {"tool_result", "command_execution"}
            and replayed.data.get("payload", {}).get("tool_call_id")
            == "call-command"
        ]
    assert len(outcomes) == 1


def test_live_timestamp_is_unix_epoch_and_replay_uses_committed_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_ms = int(time.time() * 1000)
    live = stream_events.turn_started("live-turn")
    after_ms = int(time.time() * 1000)
    assert before_ms <= live.timestamp_ms <= after_ms
    assert live.timestamp_ms > 1_000_000_000_000

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread_id, _turn_id = _start_turn(store, workspace)
        committed = next(
            record
            for record in store.list_records(thread_id)
            if record.record_type == "turn_started"
        )
        monkeypatch.setattr(stream_events, "_now_ms", lambda: 123)

        [replayed] = RolloutEventReader(store).read(thread_id)

    assert replayed.event.timestamp_ms == getattr(committed, "committed_at_ms", -1)


def test_transient_deltas_never_enter_rollout_records(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread_id, turn_id = _start_turn(store, workspace)

        with pytest.raises(ValueError, match="unsupported record type"):
            with store._transaction():
                store._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_delta",
                    producer="model",
                    payload={
                        "item_id": "transient-item",
                        "delta_kind": "text",
                        "delta": "never durable",
                    },
                )

        assert all(
            record.record_type != "item_delta"
            for record in store.list_records(thread_id)
        )


def test_verified_schema_v1_history_replays_with_deterministic_public_ids(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread_id, turn_id = _start_turn(store, workspace)
        _complete_model_response(store, turn_id=turn_id, text="legacy model answer")
        store.record_tool_call(
            turn_id=turn_id,
            tool_call_id="legacy-call",
            tool_name="read_file",
            arguments={"path": "README.md"},
            origin={"request_id": "legacy-request"},
        )
        values = {
            "turn_id": turn_id,
            "operation_id": "legacy-operation",
            "tool_call_id": "legacy-call",
            "tool_name": "read_file",
            "arguments_digest": "legacy-arguments",
            "execution_revision": "builtin-read-file-v1",
            "idempotent": True,
            "attempt_count": 0,
            "error_code": None,
            "requires_reconciliation": False,
        }
        store.record_tool_execution_state(status="prepared", **values)
        store.record_tool_execution_state(status="ready", **values)
        claim = store.claim_tool_operation(
            operation_id="legacy-operation",
            worker_id="legacy-worker",
            lease_seconds=30.0,
        )
        assert store.commit_tool_operation_outcome(
            operation_id="legacy-operation",
            claim_generation=claim.claim_generation,
            fencing_token=str(claim.fencing_token),
            status="succeeded",
            attempt_count=1,
            error_code=None,
            requires_reconciliation=False,
        )
        store.record_tool_result(
            turn_id=turn_id,
            operation_id="legacy-operation",
            result={
                "tool_call_id": "legacy-call",
                "tool_name": "read_file",
                "structured_content": {"text": "legacy file"},
                "is_error": False,
                "content": [],
                "attachments": [],
            },
        )

        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT record_id, record_type, payload_json FROM rollout_records"
            ).fetchall()
            for row in rows:
                payload = json.loads(row["payload_json"])
                if row["record_type"] in {
                    "model_operation_prepared",
                    "model_retry_prepared",
                }:
                    payload.pop("public_item_ids", None)
                if row["record_type"] in {
                    "model_attempt_completed",
                    "tool_operation_claimed",
                    "tool_operation_outcome_committed",
                }:
                    payload.pop("public_item_id", None)
                if row["record_type"] in {"item_started", "item_completed"}:
                    payload.pop("attempt_id", None)
                    payload.pop("channel", None)
                    payload.pop("operation_id", None)
                    payload.pop("attempt_generation", None)
                    payload.pop("public_item_id", None)
                payload_json = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                connection.execute(
                    """
                    UPDATE rollout_records
                    SET payload_schema_version = 1,
                        payload_json = ?, payload_hash = ?
                    WHERE record_id = ?
                    """,
                    (
                        payload_json,
                        hashlib.sha256(payload_json.encode()).hexdigest(),
                        row["record_id"],
                    ),
                )
        store.rebuild_projections()
        assert store.verify().valid is True

        first_replay = RolloutEventReader(store).read(thread_id)

    with RolloutStore(database) as reopened:
        assert reopened.verify().valid is True
        second_replay = RolloutEventReader(reopened).read(thread_id)

    first_items = [
        result.event
        for result in first_replay
        if result.event.item_kind is not None
    ]
    second_items = [
        result.event
        for result in second_replay
        if result.event.item_kind is not None
    ]
    assert [(event.type, event.item_kind, event.item_id) for event in first_items] == [
        (event.type, event.item_kind, event.item_id) for event in second_items
    ]
    assert {event.item_kind.value for event in first_items} == {
        "agent_message",
        "tool",
    }


def test_migrated_model_response_projects_one_legacy_message(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread_id, turn_id = _start_turn(store, workspace)
        migrated = store.record_migrated_context_item(
            turn_id=turn_id,
            kind="model_response",
            payload={"text": "migrated model answer", "tool_calls": []},
        )

        migrated_records = [
            record
            for record in store.list_records(thread_id)
            if record.payload.get("item_id") == migrated.item_id
        ]
        assert [record.payload.get("public_item_id") for record in migrated_records] == [
            migrated.item_id,
            migrated.item_id,
        ]
        assert [record.payload.get("public_item_kind") for record in migrated_records] == [
            "legacy_message",
            "legacy_message",
        ]

        replayed = RolloutEventReader(store).read(thread_id)

    legacy = [
        result.event
        for result in replayed
        if result.event.item_kind is not None
        and result.event.item_kind.value == "legacy_message"
        and result.event.item_id == migrated.item_id
    ]
    assert [event.type.value for event in legacy] == [
        "item_started",
        "item_completed",
    ]
    assert legacy[-1].data == {"content": "migrated model answer"}


def test_pre_identity_migrated_model_response_upcasts_to_legacy_message(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread_id, turn_id = _start_turn(store, workspace)
        migrated = store.record_migrated_context_item(
            turn_id=turn_id,
            kind="model_response",
            payload={"text": "pre-identity migrated answer", "tool_calls": []},
        )

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT record_id, payload_json
            FROM rollout_records
            WHERE producer = 'migration'
            """
        ).fetchall()
        for record_id, payload_json in rows:
            payload = json.loads(payload_json)
            if payload.get("item_id") != migrated.item_id:
                continue
            payload.pop("public_item_id")
            payload.pop("public_item_kind")
            rewritten = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            connection.execute(
                """
                UPDATE rollout_records
                SET payload_json = ?, payload_hash = ?
                WHERE record_id = ?
                """,
                (
                    rewritten,
                    hashlib.sha256(rewritten.encode()).hexdigest(),
                    record_id,
                ),
            )

    with RolloutStore(database) as reopened:
        reopened.rebuild_projections()
        assert reopened.verify().valid is True
        replayed = RolloutEventReader(reopened).read(thread_id)

    legacy = [
        result.event
        for result in replayed
        if result.event.item_kind is not None
        and result.event.item_kind.value == "legacy_message"
        and result.event.item_id == migrated.item_id
    ]
    assert [event.type.value for event in legacy] == [
        "item_started",
        "item_completed",
    ]
    assert legacy[-1].data == {"content": "pre-identity migrated answer"}

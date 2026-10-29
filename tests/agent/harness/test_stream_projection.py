from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutEventReader, RuntimeState
from agent_runtime.harness.model_outputs import complete_model_output, start_model_output
from agent_runtime.harness.protocol import HarnessModelResponse
from agent_runtime.modeling.events import ModelOutput
from agent_runtime.streaming import events as stream_events


def _start_turn(store: RuntimeState, workspace: Path) -> tuple[str, str]:
    thread = store.create_thread(workspace=workspace)
    turn = store.start_turn(
        thread_id=thread.thread_id,
        user_message="project this turn",
        binding_manifest={"model_id": "projection-test"},
    )
    return thread.thread_id, turn.turn_id


def _complete_model_response(
    store: RuntimeState,
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
        request_ref={"input_revision": 1, "request_id": f"request-{turn_id}"},
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
        response=HarnessModelResponse.from_parts(
            text=text,
            provider_response_id=f"response-{turn_id}",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        ),
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
    with RuntimeState.open_sync(database) as store:
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


@pytest.mark.parametrize("tampered_record", ["output_start", "output_completion", "plan_completion"])
def test_public_item_ids_survive_reopen_and_mismatch_fails_closed(
    tmp_path: Path, tampered_record: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RuntimeState.open_sync(database) as store:
        thread_id, turn_id = _start_turn(store, workspace)
        operation = store.prepare_model_operation(
            turn_id=turn_id,
            request_hash="request-hash",
            context_hash="context-hash",
            tool_hash="tool-hash",
            wire_hash="wire-hash",
            request_ref={"input_revision": 1, "request_id": "request-1"},
        )
        prepared = next(
            record
            for record in store.list_records(thread_id)
            if record.record_type == "model_operation_prepared"
        )
        assert "public_item_ids" not in prepared.payload
        outputs = (
            ModelOutput("reason", "reasoning", content="checked"),
            ModelOutput("plan", "plan", content="inspect then fix"),
            ModelOutput("text", "text", content="answer"),
        )
        first_attempt = store.dispatch_model_attempt(
            operation.operation_id,
            worker_id="worker-1",
            lease_seconds=1.0,
            now=100.0,
        )
        for output in outputs:
            start_model_output(
                store, operation_id=operation.operation_id, attempt_id=first_attempt.attempt_id,
                generation=first_attempt.generation, output=output,
            )
            complete_model_output(
                store, operation_id=operation.operation_id, attempt_id=first_attempt.attempt_id,
                generation=first_attempt.generation, output=output,
            )
        first_ids = {
            record.payload["output_id"]: record.payload["public_item_id"]
            for record in store.list_records(thread_id)
            if record.record_type == "item_started" and record.payload.get("kind") == "model_output"
        }
        assert set(first_ids) == {"reason", "plan", "text"}
        assert len(set(first_ids.values())) == len(outputs)
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
        assert "public_item_ids" not in retried.payload
        assert retry.attempt_id != first_attempt.attempt_id
        retry = store.dispatch_model_attempt(operation.operation_id, worker_id="worker-2")
        assert store.complete_model_attempt(operation_id=operation.operation_id, attempt_id=retry.attempt_id,
                                            generation=retry.generation, response=HarnessModelResponse(outputs=outputs))
        retry_ids = {
            record.payload["output_id"]: record.payload["public_item_id"]
            for record in store.list_records(thread_id)
            if record.record_type == "item_started" and record.payload.get("kind") == "model_output"
            and record.payload["attempt_id"] == retry.attempt_id
        }
        assert retry_ids.keys() == first_ids.keys()
        assert len(set(retry_ids.values())) == len(outputs)
        assert set(retry_ids.values()).isdisjoint(first_ids.values())
        for record in store.list_records(thread_id):
            if record.record_type == "item_completed" and record.payload.get("kind") == "model_output":
                expected_ids = retry_ids if record.payload["attempt_id"] == retry.attempt_id else first_ids
                assert record.payload["public_item_id"] == expected_ids[record.payload["output_id"]]

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
        plan_completion = next(
            record
            for record in store.list_records(thread_id)
            if record.record_type == "item_completed"
            and isinstance(record.payload.get("payload"), dict)
            and "plan" in record.payload["payload"]
        )
        plan_public_id = plan_completion.payload["public_item_id"]
        assert plan_public_id
        selected_record = plan_completion if tampered_record == "plan_completion" else next(
            record for record in store.list_records(thread_id)
            if record.record_type == ("item_started" if tampered_record == "output_start" else "item_completed")
            and record.payload.get("kind") == "model_output" and record.payload["attempt_id"] == retry.attempt_id
        )
        original_events = tuple((entry.event.item_id, entry.event.data)
                                for entry in RolloutEventReader(store).read(thread_id))

    with RuntimeState.open_sync(database) as reopened:
        persisted = reopened.list_records(thread_id)
        assert {
            record.payload["output_id"]: record.payload["public_item_id"]
            for record in persisted
            if record.record_type == "item_started" and record.payload.get("kind") == "model_output"
            and record.payload["attempt_id"] == retry.attempt_id
        } == retry_ids
        assert tuple((entry.event.item_id, entry.event.data)
                     for entry in RolloutEventReader(reopened).read(thread_id)) == original_events
        assert next(
            record.payload["public_item_id"]
            for record in persisted
            if record.record_type == "item_completed"
            and isinstance(record.payload.get("payload"), dict)
            and "plan" in record.payload["payload"]
        ) == plan_public_id

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT record_id, payload_json
            FROM rollout_records
            WHERE record_id = ?
            """,
            (selected_record.record_id,),
        ).fetchone()
        assert row is not None
        payload = json.loads(row["payload_json"])
        payload["public_item_id"] = "tampered-public-id"
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

    with RuntimeState.open_sync(database) as reopened:
        with pytest.raises(RuntimeError, match="persisted public Item ID mismatch"):
            RolloutEventReader(reopened).read(thread_id)


def test_accepted_answer_does_not_duplicate_model_response(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        thread_id, turn_id = _start_turn(store, workspace)
        _complete_model_response(store, turn_id=turn_id, text="one answer")
        source = store.list_model_operations(turn_id)[0].response_item_id
        store.complete_turn(turn_id=turn_id, answer="one answer", source_item_id=source)
        assert not any(item.kind == "agent_message" for item in store.list_items(turn_id))

        projected = RolloutEventReader(store).read(thread_id)

    assert _completed_answer_count(projected, "one answer") == 1


@pytest.mark.parametrize("same_answer", [False, True])
def test_live_accepted_answer_uses_bounded_context_and_matches_replay(tmp_path: Path, monkeypatch, same_answer) -> None:
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        thread_id, turn_id = _start_turn(store, tmp_path)
        _complete_model_response(store, turn_id=turn_id, text="model answer")
        for _ in range(48):
            store.reserve_resource_unit(turn_id, "reads", 256)
        answer = "model answer" if same_answer else "runtime answer"
        source = store.list_model_operations(turn_id)[0].response_item_id if same_answer else None
        mutation = store.capture_mutation(lambda: store.complete_turn(
            turn_id=turn_id, answer=answer, source_item_id=source,
        ))
        sql_steps = 0

        def sql_step():
            nonlocal sql_steps
            sql_steps += 1
            return 0

        def no_history(*args, **kwargs):
            pytest.fail("live delivery must not load Turn or Thread history")

        with monkeypatch.context() as patch:
            patch.setattr(store, "list_records", no_history)
            patch.setattr(store, "list_turn_records", no_history)
            store.run_sync(lambda: store.persistence._connection.set_progress_handler(sql_step, 1))
            try:
                live = RolloutEventReader(store).project_committed_batch(mutation.records)
            finally:
                store.run_sync(lambda: store.persistence._connection.set_progress_handler(None, 0))
        replayed = tuple(event for event in RolloutEventReader(store).read(thread_id)
                         if event.record_id in {record.record_id for record in mutation.records})
        assert [(event.cursor, replace(event.event, sequence=0)) for event in live] == [
            (event.cursor, replace(event.event, sequence=0)) for event in replayed
        ]
        assert sum(event.event.type.value == "item_completed" for event in live) == (0 if same_answer else 1)
        assert sql_steps < 250, sql_steps


def test_migrated_answer_projects_exactly_one_legacy_message(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            turn_id="legacy-turn",
            turn_producer="migration",
            user_message="legacy question",
            binding_manifest={"model_id": "legacy"},
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
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
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

        def transaction():
            with store.persistence.transaction():
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

        store.run_sync(transaction)

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
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
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
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        thread_id, turn_id = _start_turn(store, workspace)

        with pytest.raises(ValueError, match="unsupported record type"):

            def transaction():
                with store.persistence.transaction():
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

            store.run_sync(transaction)

        assert all(
            record.record_type != "item_delta"
            for record in store.list_records(thread_id)
        )


def test_canonical_history_replays_with_persisted_public_ids(
    tmp_path: Path, monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RuntimeState.open_sync(database) as store:
        thread_id, turn_id = _start_turn(store, workspace)
        _complete_model_response(store, turn_id=turn_id, text="canonical model answer")
        store.record_tool_call(
            turn_id=turn_id,
            tool_call_id="canonical-call",
            tool_name="read_file",
            arguments={"path": "README.md"},
            origin={"request_id": "canonical-request"},
        )
        values = {
            "turn_id": turn_id,
            "operation_id": "canonical-operation",
            "tool_call_id": "canonical-call",
            "tool_name": "read_file",
            "arguments_digest": "canonical-arguments",
            "execution_revision": "builtin-read-file-v1",
            "idempotent": True,
            "attempt_count": 0,
            "error_code": None,
            "requires_reconciliation": False,
        }
        store.record_tool_execution_state(status="prepared", **values)
        store.record_tool_execution_state(status="ready", **values)
        claim = store.claim_tool_operation(
            operation_id="canonical-operation",
            worker_id="canonical-worker",
            lease_seconds=30.0,
        )
        assert store.commit_tool_operation_outcome(
            operation_id="canonical-operation",
            claim_generation=claim.claim_generation,
            fencing_token=str(claim.fencing_token),
            status="succeeded",
            attempt_count=1,
            error_code=None,
            requires_reconciliation=False,
        )
        store.record_tool_result(
            turn_id=turn_id,
            operation_id="canonical-operation",
            result={
                "tool_call_id": "canonical-call",
                "tool_name": "read_file",
                "structured_content": {"text": "canonical file"},
                "is_error": False,
                "content": [],
                "attachments": [],
            },
        )

        store.rebuild_projections()
        assert store.verify().valid is True

        def no_mutable_operation(*args, **kwargs):
            pytest.fail("canonical event replay must use canonical facts, not the current operation projection")

        monkeypatch.setattr(store, "read_tool_operation", no_mutable_operation)
        first_replay = RolloutEventReader(store).read(thread_id)
        # A fresh reader per record exercises stateless, cross-batch identity
        # resolution, rather than relying on a warmed in-memory projection.
        live = tuple(event for record in store.list_records(thread_id)
                     for event in RolloutEventReader(store).project_committed_batch((record,)))
        assert [(event.cursor, replace(event.event, sequence=0)) for event in live] == [
            (event.cursor, replace(event.event, sequence=0)) for event in first_replay
        ]

    with RuntimeState.open_sync(database) as reopened:
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


def test_canonical_model_response_projects_one_message(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        thread_id, turn_id = _start_turn(store, workspace)
        _complete_model_response(store, turn_id=turn_id, text="model answer")
        response = next(item for item in store.list_items(turn_id) if item.kind == "model_response")
        assert "text" not in response.payload
        assert "tool_calls" not in response.payload
        assert response.payload["outputs"][0]["content"] == "model answer"
        replayed = RolloutEventReader(store).read(thread_id)

    messages = [result.event for result in replayed
                if result.event.item_kind is not None and result.event.item_kind.value == "agent_message"]
    assert [event.type.value for event in messages] == ["item_started", "item_completed"]
    assert messages[-1].data["content"] == "model answer"


def test_canonical_model_response_preserves_public_identity_after_rebuild(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RuntimeState.open_sync(database) as store:
        thread_id, turn_id = _start_turn(store, workspace)
        _complete_model_response(store, turn_id=turn_id, text="canonical persisted answer")
        first_replay = RolloutEventReader(store).read(thread_id)

    with RuntimeState.open_sync(database) as reopened:
        reopened.rebuild_projections()
        assert reopened.verify().valid is True
        replayed = RolloutEventReader(reopened).read(thread_id)

    assert [(event.cursor, replace(event.event, sequence=0)) for event in replayed] == [
        (event.cursor, replace(event.event, sequence=0)) for event in first_replay
    ]
    messages = [result.event for result in replayed
                if result.event.item_kind is not None and result.event.item_kind.value == "agent_message"]
    assert [event.type.value for event in messages] == ["item_started", "item_completed"]
    assert messages[-1].data["content"] == "canonical persisted answer"

"""Canonical retry scheduling, event replay and writer fencing."""

from __future__ import annotations

import sqlite3

import pytest

from agent_runtime.harness import RuntimeStore, facts, model_ops, model_retry_ops, turn_ops
from agent_runtime.harness.events import RolloutEventReader
from agent_runtime.harness.retry_policy import RetryPolicy
from agent_runtime.modeling.failures import ModelFailure


def prepared(store, workspace):
    thread = turn_ops.create_thread(store, workspace=workspace)
    turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="work", binding_manifest={})
    op = model_ops.prepare_model_operation(
        store,
        turn_id=turn.turn_id,
        request_hash="r",
        context_hash="c",
        tool_hash="t",
        wire_hash="w",
        request_ref={"request_id": "request", "input_revision": 1},
    )
    attempt = model_ops.dispatch_model_attempt(store, op.operation_id)
    model_ops.reject_model_attempt(
        store,
        operation_id=op.operation_id,
        attempt_id=attempt.attempt_id,
        generation=attempt.generation,
        reason="rate limited",
    )
    return thread, turn, op, attempt


def schedule_fact(store, thread, turn, op, attempt, **extra):
    failure = ModelFailure(
        kind="rate_limited",
        phase="response",
        provider_outcome="rejected",
        billing_outcome="released",
        http_status=429,
        retryable=True,
    )
    payload = {
        "operation_id": op.operation_id,
        "attempt_id": attempt.attempt_id,
        "generation": attempt.generation,
        "retry_number": 1,
        "max_retries": 2,
        "retry_at": 102.0,
        "deadline_at": 220.0,
        "delay_ms": 2000,
        "observed_at": 100.0,
        "failure": failure.to_payload(),
        **extra,
    }
    facts.append_fact(
        store,
        thread_id=thread.thread_id,
        turn_id=turn.turn_id,
        producer="runtime",
        record_type="model_retry_scheduled",
        payload=payload,
    )


def test_retry_schedule_is_replayable_and_public_runtime_event(tmp_path):
    path = tmp_path / "schedule.db"
    with RuntimeStore.open_sync(path) as store:
        thread, turn, op, attempt = prepared(store, tmp_path)
        store.transact(lambda tx: schedule_fact(store, thread, turn, op, attempt))
        assert store.list_model_operations(turn.turn_id)[0].status == "retry_wait"
        events = RolloutEventReader(store).read(thread.thread_id)
        [retry] = [event.event for event in events if event.event.type == "model_retry_scheduled"]
        assert retry.data["retry_number"] == 1
        assert retry.data["retry_at"] == 102
        assert retry.data["failure"]["http_status"] == 429
        assert all(
            event.event.item_kind != "agent_message" for event in events if event.event.type == "model_retry_scheduled"
        )
        store.rebuild_projections()
        assert store.verify().valid
    with RuntimeStore.open_sync(path, read_only=True) as store:
        assert store.list_model_operations(turn.turn_id)[0].status == "retry_wait"
        assert store.verify().valid


@pytest.mark.parametrize(("billing", "expected"), [("unknown", "failed"), ("released", "scheduled")])
def test_standalone_summary_only_retries_when_billing_is_released(tmp_path, billing, expected):
    with RuntimeStore.open_sync(tmp_path / "summary-retry.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="work",
            binding_manifest={"model_retry_policy": RetryPolicy().to_payload()},
        )
        operation = model_ops.prepare_model_operation(
            store,
            turn_id=turn.turn_id,
            request_hash="r",
            context_hash="c",
            tool_hash="t",
            wire_hash="w",
            request_ref={
                "request_id": "summary",
                "input_revision": 1,
                "purpose": "context_summary",
                "compaction_plan": {"summary_request_mode": "standalone"},
            },
        )
        attempt = model_ops.dispatch_model_attempt(store, operation.operation_id, now=100.0)
        failure = ModelFailure(
            kind="server_error", phase="response", provider_outcome="rejected", billing_outcome=billing, retryable=True
        )
        action = model_retry_ops.record_failure(
            store,
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            failure=failure,
            now=101.0,
        )
        assert action == expected
        assert len(store.list_model_attempts(operation.operation_id)) == 1
        assert store.verify().valid


@pytest.mark.parametrize("mutation", [{"generation": 99}, {"attempt_id": "wrong"}, {"retry_at": 300.0}])
def test_retry_schedule_rejects_wrong_source_or_invalid_deadline(tmp_path, mutation):
    with RuntimeStore.open_sync(tmp_path / "invalid.db") as store:
        thread, turn, op, attempt = prepared(store, tmp_path)
        before = store.list_turn_records(turn.turn_id)
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            store.transact(lambda tx: schedule_fact(store, thread, turn, op, attempt, **mutation))
        assert store.list_turn_records(turn.turn_id) == before
        assert store.verify().valid


@pytest.mark.parametrize("old_schema", [4, 5, 6])
def test_writer_gate_rejects_old_schema_on_already_open_connection(tmp_path, old_schema):
    path = tmp_path / "gate.db"
    # This connection represents the previous writer's already open SQLite handle.
    old = sqlite3.connect(path)
    with RuntimeStore.open_sync(path) as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        record = store.list_records(thread.thread_id)[0]
        assert record.payload_schema_version == 9
        with pytest.raises(sqlite3.IntegrityError, match="writer protocol requires upgrade"):
            old.execute(
                """INSERT INTO rollout_records(record_uuid,thread_id,turn_id,thread_sequence,record_type,
                payload_schema_version,producer,payload_json,payload_hash,created_at,committed_at_ms)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    "old-writer",
                    thread.thread_id,
                    None,
                    2,
                    "thread_created",
                    old_schema,
                    "runtime",
                    "{}",
                    "hash",
                    "2026-10-30T00:00:00+00:00",
                    0,
                ),
            )
        old.rollback()
        assert store.verify().valid
    old.close()


def test_previous_binary_already_open_cannot_dispatch_after_protocol_upgrade(tmp_path):
    import io
    import os
    import subprocess
    import sys
    import tarfile

    baseline = "cf660221d14a634c745a5071d2a3a778c184cba6"
    archive = subprocess.run(["git", "archive", baseline, "agent_runtime"], capture_output=True, check=False)
    if archive.returncode:
        pytest.skip("frozen schema-four Git object unavailable")
    previous = tmp_path / "previous"
    previous.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(previous, filter="data")
    path = tmp_path / "upgrade.db"
    marker = tmp_path / "provider-called"
    script = """import sqlite3,sys
from pathlib import Path
from agent_runtime.harness import RuntimeStore,turn_ops,model_ops
with RuntimeStore.open_sync(Path(sys.argv[1])) as store:
    thread=turn_ops.create_thread(store,workspace=Path(sys.argv[2]))
    turn=turn_ops.start_turn(store,thread_id=thread.thread_id,user_message='work',binding_manifest={})
    op=model_ops.prepare_model_operation(store,turn_id=turn.turn_id,request_hash='r',context_hash='c',tool_hash='t',wire_hash='w',request_ref={'request_id':'old','input_revision':1})
    print('ready',flush=True)
    sys.stdin.readline()
    try:
        model_ops.dispatch_model_attempt(store,op.operation_id)
    except sqlite3.IntegrityError as exc:
        assert 'writer protocol requires upgrade' in str(exc)
        print('blocked-before-provider',flush=True)
    else:
        Path(sys.argv[3]).write_text('provider-called')
        raise AssertionError('old writer was not blocked')
"""
    env = {**os.environ, "PYTHONPATH": str(previous)}
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path), str(tmp_path), str(marker)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=previous,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        with RuntimeStore.open_sync(path) as store:
            assert store.verify().valid
        stdout, stderr = process.communicate("dispatch\n", timeout=20)
        assert process.returncode == 0, stderr
        assert "blocked-before-provider" in stdout
        assert not marker.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import ormsgpack
import pytest

from agent_runtime.harness import (
    HarnessMessage,
    HarnessToolCall,
    RolloutContextManager,
    RolloutStore,
    migrate_legacy_turns,
    restore_legacy_backup,
)


def _create_legacy_database(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE agent_turns (
                turn_id TEXT PRIMARY KEY,
                previous_turn_id TEXT,
                status TEXT NOT NULL,
                user_message TEXT NOT NULL,
                runtime_json TEXT NOT NULL,
                lease_owner TEXT,
                lease_expires_at REAL,
                created_at REAL NOT NULL
            );
            CREATE TABLE agent_turn_messages (
                turn_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(turn_id, message_index)
            );
            """
        )


def _insert_legacy_turn(
    database: Path,
    *,
    turn_id: str,
    workspace: Path,
    status: str,
    user_message: str,
    answer: str | None = None,
    previous_turn_id: str | None = None,
    lease_owner: str | None = None,
    lease_expires_at: float | None = None,
    created_at: float,
) -> None:
    runtime = {
        "model_alias": "legacy-model",
        "workspace_path": str(workspace),
    }
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO agent_turns(
                turn_id, previous_turn_id, status, user_message,
                runtime_json, lease_owner, lease_expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                previous_turn_id,
                status,
                user_message,
                json.dumps(runtime),
                lease_owner,
                lease_expires_at,
                created_at,
            ),
        )
        messages = [{"role": "user", "content": user_message}]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        connection.executemany(
            """
            INSERT INTO agent_turn_messages(turn_id, message_index, payload_json)
            VALUES (?, ?, ?)
            """,
            [
                (turn_id, index, json.dumps(message))
                for index, message in enumerate(messages)
            ],
        )


def _replace_legacy_messages(
    database: Path,
    *,
    turn_id: str,
    messages: list[dict[str, object]],
) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM agent_turn_messages WHERE turn_id = ?",
            (turn_id,),
        )
        connection.executemany(
            """
            INSERT INTO agent_turn_messages(turn_id, message_index, payload_json)
            VALUES (?, ?, ?)
            """,
            [
                (turn_id, index, json.dumps(message))
                for index, message in enumerate(messages)
            ],
        )


def _insert_legacy_checkpoint(
    database: Path,
    *,
    turn_id: str,
    checkpoint_id: str,
    state: dict[str, object],
    encoding: str = "json",
) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            )
            """
        )
        checkpoint = {
            "channel_values": {
                "loop_state": state,
            }
        }
        encoded = (
            json.dumps(checkpoint).encode()
            if encoding == "json"
            else ormsgpack.packb(checkpoint)
        )
        connection.execute(
            "INSERT INTO checkpoints VALUES (?, 'agent_loop', ?, NULL, ?, ?, NULL)",
            (turn_id, checkpoint_id, encoding, encoded),
        )


@pytest.mark.parametrize("checkpoint_encoding", ["json", "msgpack"])
def test_joint_migration_preserves_artifact_tool_record_and_invalidates_approval(
    tmp_path: Path,
    checkpoint_encoding: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact_directory = workspace / ".praxis" / "runtime" / "artifacts"
    artifact_directory.mkdir(parents=True)
    (artifact_directory / "legacy-report.bin").write_bytes(b"legacy report bytes")
    database = tmp_path / "praxis.sqlite3"
    _create_legacy_database(database)
    _insert_legacy_turn(
        database,
        turn_id="legacy-completed",
        workspace=workspace,
        status="completed",
        user_message="render report",
        answer="report ready",
        created_at=1.0,
    )
    _replace_legacy_messages(
        database,
        turn_id="legacy-completed",
        messages=[
            {"role": "user", "content": "render report"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "legacy-call-report",
                        "name": "render_report",
                        "arguments": {"format": "binary"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "report artifact created",
                "tool_call_id": "legacy-call-report",
            },
            {"role": "assistant", "content": "report ready"},
        ],
    )
    _insert_legacy_checkpoint(
        database,
        turn_id="legacy-completed",
        checkpoint_id="0001",
        state={
            "tool_results": [
                {
                    "tool_call_id": "legacy-call-report",
                    "tool_name": "render_report",
                    "content": [{"type": "text", "data": {"text": "report artifact created"}}],
                    "structured_content": {"created": True},
                    "is_error": False,
                    "error_code": None,
                    "error_message": None,
                    "retryable": False,
                    "truncated": False,
                    "metadata": {},
                    "attachments": [
                        {
                            "artifact_id": "legacy-report.bin",
                            "media_type": "application/octet-stream",
                            "name": "report.bin",
                        }
                    ],
                }
            ],
            "tool_execution_records": {
                "legacy-call-report": {
                    "tool_call_id": "legacy-call-report",
                    "tool_name": "render_report",
                    "operation_id": "legacy-operation-report",
                    "arguments_digest": "legacy-args-report",
                    "idempotent": True,
                    "status": "completed",
                    "attempt_count": 1,
                    "error_code": None,
                    "requires_reconciliation": False,
                }
            },
        },
        encoding=checkpoint_encoding,
    )
    _insert_legacy_turn(
        database,
        turn_id="legacy-paused",
        workspace=workspace,
        status="paused",
        user_message="write protected file",
        created_at=2.0,
    )
    _replace_legacy_messages(
        database,
        turn_id="legacy-paused",
        messages=[
            {"role": "user", "content": "write protected file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "legacy-call-write",
                        "name": "write_file",
                        "arguments": {"path": "protected.txt"},
                    }
                ],
            },
        ],
    )
    _insert_legacy_checkpoint(
        database,
        turn_id="legacy-paused",
        checkpoint_id="0002",
        state={
            "approval_request": {
                "request_id": "legacy-approval-1",
                "kind": "tool_approval",
                "question": "Allow protected write?",
                "tool_calls": [
                    {
                        "tool_call_id": "legacy-call-write",
                        "tool_name": "write_file",
                        "args_preview": "path='protected.txt'",
                        "risk_level": "high",
                        "reason": "workspace write",
                    }
                ],
                "context": {"policy_revision": "legacy-policy-v1"},
                "options": [],
            },
            "tool_execution_records": {
                "legacy-call-write": {
                    "tool_call_id": "legacy-call-write",
                    "tool_name": "write_file",
                    "operation_id": "legacy-operation-write",
                    "arguments_digest": "legacy-args-write",
                    "idempotent": False,
                    "status": "prepared",
                    "attempt_count": 0,
                    "error_code": None,
                    "requires_reconciliation": False,
                }
            },
        },
        encoding=checkpoint_encoding,
    )

    report = migrate_legacy_turns(database)

    assert report.blocked == {}
    assert report.migrated_turn_ids == ("legacy-completed", "legacy-paused")
    with RolloutStore(database) as store:
        completed = store.read_turn("legacy-completed")
        [artifact] = store.list_artifacts(completed.turn_id)
        assert store.read_artifact(artifact.artifact_id) == b"legacy report bytes"
        tool_results = [
            item
            for item in store.list_items(completed.turn_id)
            if item.kind == "tool_result"
        ]
        assert len(tool_results) == 1
        assert tool_results[0].payload["attachments"] == [
            {
                "artifact_id": artifact.artifact_id,
                "media_type": "application/octet-stream",
                "name": "report.bin",
            }
        ]
        [completed_operation] = store.list_tool_operations(completed.turn_id)
        assert completed_operation.operation_id == "legacy-operation-report"
        assert completed_operation.status == "succeeded"
        assert completed_operation.result_item_id == tool_results[0].item_id

        paused = store.read_turn("legacy-paused")
        assert paused.status == "paused"
        [interaction] = store.list_interactions(paused.turn_id)
        [approval] = store.list_approvals(paused.turn_id)
        [paused_operation] = store.list_tool_operations(paused.turn_id)
        assert interaction.kind == "tool_approval"
        assert interaction.status == "resolved"
        assert interaction.request["legacy_request_id"] == "legacy-approval-1"
        assert approval.status == "invalidated"
        assert paused_operation.operation_id == "legacy-operation-write"
        assert paused_operation.status == "superseded"
        assert store.verify().valid is True
def test_terminal_legacy_turn_chain_is_imported_with_stable_ids(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    _create_legacy_database(database)
    first_id = "legacy-turn-1"
    second_id = "legacy-turn-2"
    paused_id = "legacy-turn-paused"
    _insert_legacy_turn(
        database,
        turn_id=first_id,
        workspace=workspace,
        status="completed",
        user_message="remember cobalt",
        answer="remembered cobalt",
        created_at=1.0,
    )
    _insert_legacy_turn(
        database,
        turn_id=second_id,
        workspace=workspace,
        status="completed",
        user_message="what did I say?",
        answer="cobalt",
        previous_turn_id=first_id,
        created_at=2.0,
    )
    _insert_legacy_turn(
        database,
        turn_id=paused_id,
        workspace=workspace,
        status="paused",
        user_message="uncertain write",
        created_at=3.0,
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                checkpoint BLOB NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?)",
            (paused_id, "checkpoint-1", b"legacy pending approval"),
        )

    dry_run = migrate_legacy_turns(database, dry_run=True)
    assert dry_run.migrated_turn_ids == (first_id, second_id, paused_id)
    with sqlite3.connect(database) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "rollout_records" not in tables

    report = migrate_legacy_turns(database)

    assert report.migrated_turn_ids == (first_id, second_id, paused_id)
    assert report.blocked == {}
    with RolloutStore(database) as store:
        imported_first = store.read_turn(first_id)
        imported_second = store.read_turn(second_id)
        assert imported_first.thread_id == imported_second.thread_id
        assert imported_second.predecessor_turn_id == first_id
        context = [
            (item.kind, item.payload.get("text"))
            for item in store.list_context_items(second_id)
            if item.kind in {"user_message", "agent_message"}
        ]
        assert context == [
            ("user_message", "remember cobalt"),
            ("agent_message", "remembered cobalt"),
            ("user_message", "what did I say?"),
            ("agent_message", "cobalt"),
        ]
        imported_paused = store.read_turn(paused_id)
        assert imported_paused.status == "paused"
        assert imported_paused.binding_manifest["legacy_resume_compatible"] is False
        assert imported_paused.binding_manifest["legacy_approvals_invalidated"] is True
        checkpoint = imported_paused.binding_manifest["legacy_checkpoint"]
        assert checkpoint["row_count"] == 1
        assert checkpoint["tables"] == ["checkpoints"]
        assert len(checkpoint["sha256"]) == 64
        paused_thread = store.read_thread(imported_paused.thread_id)
        assert paused_thread.active_turn_id == paused_id
        assert store.verify().valid is True

    repeated = migrate_legacy_turns(database)
    assert repeated.migrated_turn_ids == ()
    assert repeated.skipped_turn_ids == (first_id, second_id, paused_id)


def test_migration_backup_can_restore_the_original_database(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    _create_legacy_database(database)
    turn_id = "legacy-turn"
    _insert_legacy_turn(
        database,
        turn_id=turn_id,
        workspace=workspace,
        status="completed",
        user_message="legacy",
        answer="answer",
        created_at=1.0,
    )

    migrate_legacy_turns(database)
    backup = database.with_name(database.name + ".pre-harness.bak")
    assert backup.is_file()
    with RolloutStore(database) as store:
        assert store.read_turn(turn_id).status == "completed"

    restore_legacy_backup(database=database, backup=backup)

    with sqlite3.connect(database) as restored:
        status = restored.execute(
            "SELECT status FROM agent_turns WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        harness_table = restored.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'rollout_records'"
        ).fetchone()
    assert status == ("completed",)
    assert harness_table is None


def test_migration_preserves_tool_transcript_and_exact_fork_cutoff(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    _create_legacy_database(database)
    _insert_legacy_turn(
        database,
        turn_id="root",
        workspace=workspace,
        status="completed",
        user_message="inspect value.txt",
        answer="value is cobalt",
        created_at=1.0,
    )
    _replace_legacy_messages(
        database,
        turn_id="root",
        messages=[
            {"role": "user", "content": "inspect value.txt", "tool_calls": []},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "legacy-call-1",
                        "name": "read_file",
                        "arguments": {"file_path": "value.txt"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "cobalt",
                "tool_calls": [],
                "tool_call_id": "legacy-call-1",
            },
            {"role": "assistant", "content": "value is cobalt", "tool_calls": []},
        ],
    )
    for turn_id, message, answer, created_at in (
        ("primary", "primary branch", "primary answer", 2.0),
        ("fork", "fork branch", "fork answer", 3.0),
    ):
        _insert_legacy_turn(
            database,
            turn_id=turn_id,
            workspace=workspace,
            status="completed",
            user_message=message,
            answer=answer,
            previous_turn_id="root",
            created_at=created_at,
        )

    report = migrate_legacy_turns(database)

    assert report.blocked == {}
    with RolloutStore(database) as store:
        root = store.read_turn("root")
        primary = store.read_turn("primary")
        fork = store.read_turn("fork")
        assert primary.thread_id == root.thread_id
        assert fork.thread_id != root.thread_id
        assert store.read_thread(fork.thread_id).fork_turn_id == "root"
        assert RolloutContextManager(store).build("fork") == (
            HarnessMessage(role="user", content="inspect value.txt"),
            HarnessMessage(
                role="assistant",
                content="",
                tool_calls=(
                    HarnessToolCall(
                        id="legacy-call-1",
                        name="read_file",
                        arguments={"file_path": "value.txt"},
                    ),
                ),
            ),
            HarnessMessage(
                role="tool",
                content="cobalt",
                tool_call_id="legacy-call-1",
            ),
            HarnessMessage(role="assistant", content="value is cobalt"),
            HarnessMessage(role="user", content="fork branch"),
            HarnessMessage(role="assistant", content="fork answer"),
        )


def test_migration_fault_rolls_back_all_rollout_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    _create_legacy_database(database)
    _insert_legacy_turn(
        database,
        turn_id="legacy-fault",
        workspace=workspace,
        status="completed",
        user_message="inspect",
        answer="done",
        created_at=1.0,
    )
    _replace_legacy_messages(
        database,
        turn_id="legacy-fault",
        messages=[
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-1", "name": "read_file", "arguments": {}}
                ],
            },
            {"role": "assistant", "content": "done"},
        ],
    )

    def fail_import(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(RolloutStore, "record_migrated_context_item", fail_import)

    with pytest.raises(RuntimeError, match="injected migration failure"):
        migrate_legacy_turns(database)

    with sqlite3.connect(database) as connection:
        record_count = connection.execute(
            "SELECT COUNT(*) FROM rollout_records"
        ).fetchone()[0]
        marker = connection.execute(
            "SELECT value FROM store_metadata WHERE key = 'legacy_migration_version'"
        ).fetchone()
    assert record_count == 0
    assert marker is None


def test_interrupted_and_maintenance_confirmed_expired_running_are_preserved(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    _create_legacy_database(database)
    _insert_legacy_turn(
        database,
        turn_id="legacy-interrupted",
        workspace=workspace,
        status="interrupted",
        user_message="interrupted work",
        created_at=1.0,
    )
    _insert_legacy_turn(
        database,
        turn_id="legacy-expired-running",
        workspace=workspace,
        status="running",
        user_message="expired work",
        lease_owner="dead-worker",
        lease_expires_at=10.0,
        created_at=2.0,
    )
    _insert_legacy_turn(
        database,
        turn_id="legacy-live-running",
        workspace=workspace,
        status="running",
        user_message="live work",
        lease_owner="maybe-live-worker",
        lease_expires_at=1000.0,
        created_at=3.0,
    )

    report = migrate_legacy_turns(
        database,
        maintenance_confirmed=True,
        observed_at=100.0,
    )

    assert report.migrated_turn_ids == (
        "legacy-interrupted",
        "legacy-expired-running",
    )
    assert "legacy-live-running" in report.blocked
    with RolloutStore(database) as store:
        interrupted = store.read_turn("legacy-interrupted")
        expired = store.read_turn("legacy-expired-running")
        assert interrupted.status == "interrupted"
        assert expired.status == "interrupted"
        assert interrupted.binding_manifest["legacy_resume_compatible"] is False
        assert expired.binding_manifest["legacy_resume_compatible"] is False
        assert store.read_thread(interrupted.thread_id).active_turn_id == interrupted.turn_id
        assert store.read_thread(expired.thread_id).active_turn_id == expired.turn_id
        assert store.verify().valid is True


def test_running_legacy_turn_requires_explicit_maintenance_confirmation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    _create_legacy_database(database)
    _insert_legacy_turn(
        database,
        turn_id="legacy-running",
        workspace=workspace,
        status="running",
        user_message="uncertain live work",
        lease_owner="old-worker",
        lease_expires_at=1.0,
        created_at=1.0,
    )

    report = migrate_legacy_turns(database, observed_at=100.0)

    assert report.migrated_turn_ids == ()
    assert "maintenance" in report.blocked["legacy-running"]

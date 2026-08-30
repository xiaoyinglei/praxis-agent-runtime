"""One-way import of terminal legacy Agent Turns into the Rollout Harness."""

from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ormsgpack

from agent_runtime.harness.rollout import RolloutStore


@dataclass(frozen=True, slots=True)
class LegacyMigrationReport:
    migrated_turn_ids: tuple[str, ...]
    skipped_turn_ids: tuple[str, ...]
    blocked: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _LegacyCheckpointSemantics:
    tool_results: Mapping[str, Mapping[str, Any]]
    attachment_sources: Mapping[str, Path]
    execution_records: Mapping[str, Mapping[str, Any]]
    approval_request: Mapping[str, Any] | None
    pending_interaction: Mapping[str, Any] | None


def migrate_legacy_turns(
    database: Path,
    *,
    dry_run: bool = False,
    backup_path: Path | None = None,
    maintenance_confirmed: bool = False,
    observed_at: float | None = None,
) -> LegacyMigrationReport:
    """Run one explicit, locked, backed-up legacy import."""

    path = Path(database).expanduser().resolve()
    if not path.is_file():
        return LegacyMigrationReport((), (), {})
    lock_path = path.with_name(path.name + ".harness-migration.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Harness migration owns the maintenance lock") from exc
        if dry_run:
            with tempfile.TemporaryDirectory(prefix="praxis-harness-migration-") as root:
                candidate = Path(root) / path.name
                _sqlite_backup(path, candidate)
                return _migrate_in_place(
                    candidate,
                    maintenance_confirmed=maintenance_confirmed,
                    observed_at=observed_at,
                )
        backup = (
            path.with_name(path.name + ".pre-harness.bak")
            if backup_path is None
            else Path(backup_path).expanduser().resolve()
        )
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            _sqlite_backup(path, backup)
        return _migrate_in_place(
            path,
            maintenance_confirmed=maintenance_confirmed,
            observed_at=observed_at,
        )


def _sqlite_backup(source_path: Path, destination_path: Path) -> None:
    with sqlite3.connect(source_path) as source:
        with sqlite3.connect(destination_path) as destination:
            source.backup(destination)


def restore_legacy_backup(*, database: Path, backup: Path) -> None:
    """Explicit rollback helper; callers must already have stopped all writers."""

    target = Path(database).expanduser().resolve()
    source = Path(backup).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"migration backup does not exist: {source}")
    if target == source:
        raise ValueError("database and backup must be different files")
    temporary = target.with_name(target.name + ".restore-tmp")
    _sqlite_backup(source, temporary)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(target) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    shutil.move(temporary, target)


def _migrate_in_place(
    path: Path,
    *,
    maintenance_confirmed: bool,
    observed_at: float | None,
) -> LegacyMigrationReport:
    """Import terminal rows only; uncertain resumable state is never replayed."""

    with sqlite3.connect(path) as legacy:
        legacy.row_factory = sqlite3.Row
        tables = {
            str(row[0]) for row in legacy.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        if "agent_turns" not in tables:
            return LegacyMigrationReport((), (), {})
        turn_columns = {str(row["name"]) for row in legacy.execute("PRAGMA table_info(agent_turns)").fetchall()}
        lease_owner = "lease_owner" if "lease_owner" in turn_columns else "NULL AS lease_owner"
        lease_expires_at = "lease_expires_at" if "lease_expires_at" in turn_columns else "NULL AS lease_expires_at"
        rows = legacy.execute(
            f"""
            SELECT turn_id, previous_turn_id, status, user_message,
                   runtime_json, {lease_owner}, {lease_expires_at}, created_at
            FROM agent_turns
            ORDER BY created_at, turn_id
            """
        ).fetchall()
        messages = {str(row["turn_id"]): _legacy_messages(legacy, str(row["turn_id"])) for row in rows}
        checkpoints = {
            str(row["turn_id"]): _legacy_checkpoint_summary(
                legacy,
                turn_id=str(row["turn_id"]),
                tables=tables,
            )
            for row in rows
        }
        checkpoint_states = {
            str(row["turn_id"]): _legacy_checkpoint_state(
                legacy,
                turn_id=str(row["turn_id"]),
                tables=tables,
            )
            for row in rows
        }

    migrated: list[str] = []
    skipped: list[str] = []
    blocked: dict[str, str] = {}
    checked_at = time.time() if observed_at is None else float(observed_at)
    with RolloutStore(path) as store:
        with store.migration_transaction():
            existing = {turn.turn_id: turn for turn in store.list_turns()}
            pending = list(rows)
            while pending:
                progressed = False
                for row in tuple(pending):
                    turn_id = str(row["turn_id"])
                    previous = None if row["previous_turn_id"] is None else str(row["previous_turn_id"])
                    if turn_id in existing:
                        skipped.append(turn_id)
                        pending.remove(row)
                        progressed = True
                        continue
                    if previous is not None and previous not in existing:
                        continue
                    legacy_status = str(row["status"])
                    status = legacy_status
                    if legacy_status == "running":
                        lease_expires_at = row["lease_expires_at"]
                        if not maintenance_confirmed:
                            blocked[turn_id] = (
                                "legacy running Turn requires explicit maintenance confirmation before worker takeover"
                            )
                            pending.remove(row)
                            progressed = True
                            continue
                        if lease_expires_at is None or float(lease_expires_at) > checked_at:
                            blocked[turn_id] = "legacy running Turn has no expired worker lease; it was not imported"
                            pending.remove(row)
                            progressed = True
                            continue
                        status = "interrupted"
                    if status not in {
                        "completed",
                        "failed",
                        "paused",
                        "interrupted",
                    }:
                        blocked[turn_id] = (
                            f"legacy status {legacy_status!r} has uncertain resumable state; it was not imported"
                        )
                        pending.remove(row)
                        progressed = True
                        continue
                    binding = _legacy_binding(str(row["runtime_json"]))
                    workspace_value = binding.get("workspace_path")
                    workspace = (
                        Path.cwd()
                        if not isinstance(workspace_value, str) or not workspace_value
                        else Path(workspace_value).expanduser().resolve()
                    )
                    if not workspace.is_dir():
                        blocked[turn_id] = f"legacy workspace is unavailable: {workspace}"
                        pending.remove(row)
                        progressed = True
                        continue
                    try:
                        checkpoint_semantics = _legacy_checkpoint_semantics(
                            checkpoint_states[turn_id],
                            messages=messages[turn_id],
                            workspace=workspace,
                        )
                    except (TypeError, ValueError, OSError) as exc:
                        blocked[turn_id] = f"legacy checkpoint cannot be safely imported: {exc}"
                        pending.remove(row)
                        progressed = True
                        continue
                    answer = _assistant_answer(messages[turn_id])
                    if status == "completed" and answer is None:
                        blocked[turn_id] = "completed legacy Turn has no assistant answer"
                        pending.remove(row)
                        progressed = True
                        continue
                    if previous is None:
                        thread = store.create_thread(workspace=workspace)
                    else:
                        predecessor = existing[previous]
                        predecessor_thread = store.read_thread(predecessor.thread_id)
                        thread = (
                            predecessor_thread
                            if predecessor_thread.head_turn_id == previous
                            else store.fork_thread(from_turn_id=previous)
                        )
                    snapshot = store.start_turn(
                        thread_id=thread.thread_id,
                        turn_id=turn_id,
                        turn_producer="migration",
                        user_message=str(row["user_message"]),
                        binding_manifest=_harness_binding(
                            binding,
                            resume_compatible=status in {"completed", "failed"},
                            checkpoint=checkpoints[turn_id],
                        ),
                    )
                    migrated_tool_results = _commit_legacy_attachments(
                        store,
                        turn_id=turn_id,
                        tool_results=checkpoint_semantics.tool_results,
                        attachment_sources=checkpoint_semantics.attachment_sources,
                    )
                    final_answer_index = None if answer is None else answer[0]
                    result_items: dict[str, str] = {}
                    for kind, payload in _legacy_context_items(
                        messages[turn_id],
                        user_message=str(row["user_message"]),
                        final_answer_index=final_answer_index,
                        tool_results=migrated_tool_results,
                    ):
                        item = store.record_migrated_context_item(
                            turn_id=turn_id,
                            kind=kind,
                            payload=payload,
                        )
                        tool_call_id = payload.get("tool_call_id")
                        if kind == "tool_result" and isinstance(tool_call_id, str):
                            result_items[tool_call_id] = item.item_id
                    approval_request = checkpoint_semantics.approval_request
                    approval_call_id = _approval_tool_call_id(approval_request)
                    for tool_call_id, record in checkpoint_semantics.execution_records.items():
                        if tool_call_id == approval_call_id:
                            continue
                        _record_legacy_tool_operation(
                            store,
                            turn_id=turn_id,
                            record=record,
                            result_item_id=result_items.get(tool_call_id),
                            checkpoint=checkpoints[turn_id],
                        )
                    if approval_request is not None:
                        if approval_call_id is None:
                            raise RuntimeError("legacy approval has no unique tool call")
                        approval_record = checkpoint_semantics.execution_records.get(approval_call_id)
                        if approval_record is None:
                            raise RuntimeError("legacy approval has no execution record")
                        store.request_tool_approval(
                            turn_id=turn_id,
                            operation_id=str(approval_record["operation_id"]),
                            tool_call_id=approval_call_id,
                            tool_name=str(approval_record["tool_name"]),
                            arguments_digest=str(approval_record["arguments_digest"]),
                            execution_revision=_legacy_execution_revision(checkpoints[turn_id]),
                            idempotent=bool(approval_record["idempotent"]),
                            request={
                                **dict(approval_request),
                                "legacy_request_id": approval_request.get("request_id"),
                                "legacy_approval_invalidated": True,
                            },
                        )
                        store.invalidate_tool_approval(
                            turn_id=turn_id,
                            reason="legacy approval cannot authorize the replacement Harness",
                        )
                    elif checkpoint_semantics.pending_interaction is not None:
                        pending_interaction = checkpoint_semantics.pending_interaction
                        interaction_kind = str(pending_interaction["kind"])
                        operation_id = None
                        if interaction_kind == "tool_reconciliation":
                            call_id = _interaction_tool_call_id(pending_interaction)
                            if call_id is None:
                                raise RuntimeError("legacy reconciliation has no tool call")
                            operation = checkpoint_semantics.execution_records.get(call_id)
                            if operation is None:
                                raise RuntimeError("legacy reconciliation has no execution record")
                            operation_id = str(operation["operation_id"])
                        store.record_migrated_pending_interaction(
                            turn_id=turn_id,
                            request_id=str(pending_interaction["request_id"]),
                            kind=interaction_kind,
                            operation_id=operation_id,
                            request=pending_interaction,
                        )
                    if status == "completed":
                        snapshot = store.complete_turn(
                            turn_id=turn_id,
                            answer=answer[1] if answer is not None else "",
                            producer="migration",
                        )
                    elif status == "failed":
                        snapshot = store.fail_turn(
                            turn_id=turn_id,
                            reason="Imported terminal failure from legacy runtime.",
                        )
                    elif status == "paused":
                        current = store.read_turn(turn_id)
                        snapshot = (
                            current
                            if current.status == "paused"
                            else store.pause_turn(
                                turn_id=turn_id,
                                reason=(
                                    "Imported legacy paused Turn has no exactly compatible "
                                    "Harness operation manifest; execution is disabled."
                                ),
                                producer="migration",
                            )
                        )
                    else:
                        snapshot = store._record_imported_interruption(
                            turn_id=turn_id,
                            reason=(
                                "Imported interrupted legacy Turn; frozen legacy "
                                "execution state is unavailable to the Harness."
                            ),
                        )
                    existing[turn_id] = snapshot
                    migrated.append(turn_id)
                    pending.remove(row)
                    progressed = True
                if not progressed:
                    for row in pending:
                        blocked[str(row["turn_id"])] = "legacy predecessor is missing or could not be imported"
                    break
            integrity = store.verify()
            if not integrity.valid:
                raise RuntimeError(
                    "legacy migration produced invalid Rollout projections: " + "; ".join(integrity.errors)
                )
            store.mark_legacy_migration_version(1)
    return LegacyMigrationReport(
        tuple(migrated),
        tuple(skipped),
        blocked,
    )


def _legacy_messages(
    connection: sqlite3.Connection,
    turn_id: str,
) -> tuple[Mapping[str, Any], ...]:
    rows = connection.execute(
        """
        SELECT payload_json FROM agent_turn_messages
        WHERE turn_id = ? ORDER BY message_index
        """,
        (turn_id,),
    ).fetchall()
    values: list[Mapping[str, Any]] = []
    for row in rows:
        value = json.loads(str(row["payload_json"]))
        if isinstance(value, Mapping):
            values.append(value)
    return tuple(values)


def _legacy_checkpoint_state(
    connection: sqlite3.Connection,
    *,
    turn_id: str,
    tables: set[str],
) -> Mapping[str, Any] | None:
    if "checkpoints" not in tables:
        return None
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(checkpoints)").fetchall()
    }
    required = {"thread_id", "checkpoint_id", "type", "checkpoint"}
    if not required.issubset(columns):
        return None
    namespace_filter = "AND checkpoint_ns = 'agent_loop'" if "checkpoint_ns" in columns else ""
    row = connection.execute(
        f"""
        SELECT type, checkpoint FROM checkpoints
        WHERE thread_id = ? {namespace_filter}
        ORDER BY checkpoint_id DESC LIMIT 1
        """,
        (turn_id,),
    ).fetchone()
    if row is None:
        return None
    encoding = str(row["type"])
    raw = row["checkpoint"]
    if not isinstance(raw, (bytes, str)):
        raise ValueError("legacy checkpoint payload is not bytes or text")
    if encoding == "json":
        decoded = json.loads(raw)
    elif encoding == "msgpack":
        encoded = raw.encode() if isinstance(raw, str) else raw
        decoded = ormsgpack.unpackb(
            encoded,
            ext_hook=_legacy_msgpack_ext_hook,
            option=ormsgpack.OPT_NON_STR_KEYS,
        )
    else:
        raise ValueError(f"unsupported legacy checkpoint encoding: {encoding}")
    if not isinstance(decoded, Mapping):
        raise ValueError("legacy checkpoint root must be an object")
    channel_values = decoded.get("channel_values")
    if not isinstance(channel_values, Mapping):
        raise ValueError("legacy checkpoint has no channel_values object")
    state = channel_values.get("loop_state")
    if not isinstance(state, Mapping):
        raise ValueError("legacy checkpoint has no loop_state object")
    return state


def _legacy_msgpack_ext_hook(code: int, data: bytes) -> Any:
    """Decode legacy constructor envelopes to inert plain values without imports."""

    if code not in {0, 1, 2, 3, 4, 5, 7}:
        raise ValueError(f"unsupported legacy msgpack extension code: {code}")
    decoded = ormsgpack.unpackb(
        data,
        ext_hook=_legacy_msgpack_ext_hook,
        option=ormsgpack.OPT_NON_STR_KEYS,
    )
    if code == 7:
        return decoded
    if not isinstance(decoded, (list, tuple)) or len(decoded) < 3:
        raise ValueError("legacy msgpack constructor envelope is malformed")
    return decoded[2]


def _legacy_checkpoint_semantics(
    state: Mapping[str, Any] | None,
    *,
    messages: tuple[Mapping[str, Any], ...],
    workspace: Path,
) -> _LegacyCheckpointSemantics:
    if state is None:
        return _LegacyCheckpointSemantics({}, {}, {}, None, None)
    calls = _legacy_tool_calls(messages)
    raw_results = state.get("tool_results", ())
    if not isinstance(raw_results, (list, tuple)):
        raise ValueError("tool_results must be an array")
    tool_results: dict[str, Mapping[str, Any]] = {}
    attachment_sources: dict[str, Path] = {}
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            raise ValueError("tool result must be an object")
        tool_call_id = raw_result.get("tool_call_id")
        if not isinstance(tool_call_id, str) or tool_call_id not in calls:
            raise ValueError("tool result has no matching transcript call")
        if tool_call_id in tool_results:
            raise ValueError("tool result is duplicated")
        attachments = raw_result.get("attachments", ())
        if not isinstance(attachments, (list, tuple)):
            raise ValueError("tool result attachments must be an array")
        for attachment in attachments:
            if not isinstance(attachment, Mapping):
                raise ValueError("tool result attachment must be an object")
            artifact_id = attachment.get("artifact_id")
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError("tool result attachment has no artifact_id")
            source = workspace / ".praxis" / "runtime" / "artifacts" / artifact_id
            if not source.is_file():
                raise ValueError(f"legacy attachment is missing: {artifact_id}")
            attachment_sources[artifact_id] = source
        tool_results[tool_call_id] = raw_result

    raw_records = state.get("tool_execution_records", {})
    if not isinstance(raw_records, Mapping):
        raise ValueError("tool_execution_records must be an object")
    execution_records: dict[str, Mapping[str, Any]] = {}
    for key, raw_record in raw_records.items():
        if not isinstance(key, str) or not isinstance(raw_record, Mapping):
            raise ValueError("tool execution record is malformed")
        tool_call_id = raw_record.get("tool_call_id")
        if tool_call_id != key or key not in calls:
            raise ValueError("tool execution record has no matching transcript call")
        for field in ("tool_name", "operation_id", "arguments_digest", "status"):
            if not isinstance(raw_record.get(field), str) or not raw_record[field]:
                raise ValueError(f"tool execution record has invalid {field}")
        if raw_record["tool_name"] != calls[key]["name"]:
            raise ValueError("tool execution record name differs from transcript")
        execution_records[key] = raw_record

    for tool_call_id, result in tool_results.items():
        if tool_call_id in execution_records:
            continue
        call = calls[tool_call_id]
        digest = hashlib.sha256(
            json.dumps(
                call["arguments"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        execution_records[tool_call_id] = {
            "tool_call_id": tool_call_id,
            "tool_name": call["name"],
            "operation_id": f"legacyop_{hashlib.sha256(tool_call_id.encode()).hexdigest()}",
            "arguments_digest": digest,
            "idempotent": False,
            "status": "failed" if result.get("is_error") is True else "completed",
            "attempt_count": 1,
            "error_code": result.get("error_code"),
            "requires_reconciliation": False,
        }
    for tool_call_id, record in execution_records.items():
        if str(record["status"]) == "completed" and tool_call_id not in tool_results:
            raise ValueError("completed tool execution record has no canonical result")

    raw_request = state.get("approval_request")
    approval_request: Mapping[str, Any] | None = None
    pending_interaction: Mapping[str, Any] | None = None
    if raw_request is not None:
        if not isinstance(raw_request, Mapping):
            raise ValueError("approval_request must be an object")
        request_id = raw_request.get("request_id")
        kind = raw_request.get("kind")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("approval_request has no request_id")
        if kind == "tool_approval":
            approval_request = raw_request
        elif kind in {"clarification", "choice", "tool_reconciliation"}:
            pending_interaction = raw_request
        else:
            raise ValueError(f"unsupported legacy interaction kind: {kind}")
    if pending_interaction is None and approval_request is None:
        unknown = [
            call_id
            for call_id, record in execution_records.items()
            if str(record["status"]) in {"started", "running", "outcome_unknown", "unknown"}
        ]
        if len(unknown) == 1:
            pending_interaction = {
                "request_id": f"legacy-reconcile-{unknown[0]}",
                "kind": "tool_reconciliation",
                "question": "Legacy tool outcome requires reconciliation.",
                "context": {"tool_call_id": unknown[0]},
                "tool_calls": [],
                "options": [],
            }
        elif len(unknown) > 1:
            raise ValueError("multiple unknown legacy operations cannot share one reconciliation")
    return _LegacyCheckpointSemantics(
        tool_results,
        attachment_sources,
        execution_records,
        approval_request,
        pending_interaction,
    )


def _legacy_tool_calls(
    messages: tuple[Mapping[str, Any], ...],
) -> dict[str, Mapping[str, Any]]:
    calls: dict[str, Mapping[str, Any]] = {}
    for message in messages:
        raw_calls = message.get("tool_calls", ())
        if not isinstance(raw_calls, (list, tuple)):
            raise ValueError("legacy message tool_calls must be an array")
        for raw_call in raw_calls:
            call = _legacy_tool_call(raw_call)
            call_id = str(call["id"])
            if call_id in calls:
                raise ValueError("legacy transcript duplicates a tool call ID")
            calls[call_id] = call
    return calls


def _commit_legacy_attachments(
    store: RolloutStore,
    *,
    turn_id: str,
    tool_results: Mapping[str, Mapping[str, Any]],
    attachment_sources: Mapping[str, Path],
) -> Mapping[str, Mapping[str, Any]]:
    migrated: dict[str, Mapping[str, Any]] = {}
    for tool_call_id, result in tool_results.items():
        rewritten = dict(result)
        rewritten_attachments: list[Mapping[str, Any]] = []
        raw_attachments = result.get("attachments", ())
        if not isinstance(raw_attachments, (list, tuple)):
            raise RuntimeError("validated legacy attachments changed shape")
        for attachment in raw_attachments:
            if not isinstance(attachment, Mapping):
                raise RuntimeError("validated legacy attachment changed shape")
            legacy_id = str(attachment["artifact_id"])
            source = attachment_sources[legacy_id]
            media_type = attachment.get("media_type")
            name = attachment.get("name")
            artifact = store.commit_artifact(
                turn_id=turn_id,
                content=source.read_bytes(),
                media_type=(
                    media_type
                    if isinstance(media_type, str) and media_type
                    else "application/octet-stream"
                ),
                name=name if isinstance(name, str) and name else source.name,
            )
            rewritten_attachments.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "media_type": artifact.media_type,
                    "name": artifact.name,
                }
            )
        rewritten["attachments"] = rewritten_attachments
        migrated[tool_call_id] = rewritten
    return migrated


def _record_legacy_tool_operation(
    store: RolloutStore,
    *,
    turn_id: str,
    record: Mapping[str, Any],
    result_item_id: str | None,
    checkpoint: Mapping[str, Any] | None,
) -> None:
    legacy_status = str(record["status"])
    status = {
        "completed": "succeeded",
        "started": "unknown",
        "running": "unknown",
        "outcome_unknown": "unknown",
        "unknown": "unknown",
    }.get(legacy_status, legacy_status)
    store.record_migrated_tool_operation(
        turn_id=turn_id,
        operation_id=str(record["operation_id"]),
        tool_call_id=str(record["tool_call_id"]),
        tool_name=str(record["tool_name"]),
        arguments_digest=str(record["arguments_digest"]),
        execution_revision=_legacy_execution_revision(checkpoint),
        idempotent=bool(record.get("idempotent", False)),
        status=status,
        attempt_count=int(record.get("attempt_count", 0)),
        error_code=(
            str(record["error_code"])
            if isinstance(record.get("error_code"), str)
            else None
        ),
        requires_reconciliation=status == "unknown",
        result_item_id=result_item_id,
    )


def _legacy_execution_revision(checkpoint: Mapping[str, Any] | None) -> str:
    digest = "none" if checkpoint is None else str(checkpoint.get("sha256", "unknown"))
    return f"legacy-checkpoint:{digest}"


def _approval_tool_call_id(request: Mapping[str, Any] | None) -> str | None:
    if request is None:
        return None
    calls = request.get("tool_calls", ())
    if not isinstance(calls, (list, tuple)) or len(calls) != 1 or not isinstance(calls[0], Mapping):
        return None
    value = calls[0].get("tool_call_id")
    return value if isinstance(value, str) and value else None


def _interaction_tool_call_id(request: Mapping[str, Any]) -> str | None:
    context = request.get("context")
    if isinstance(context, Mapping):
        value = context.get("tool_call_id")
        if isinstance(value, str) and value:
            return value
    return _approval_tool_call_id(request)


def _legacy_checkpoint_summary(
    connection: sqlite3.Connection,
    *,
    turn_id: str,
    tables: set[str],
) -> Mapping[str, Any] | None:
    candidate_tables = (
        "checkpoints",
        "checkpoint_writes",
        "checkpoint_blobs",
        "writes",
        "blobs",
    )
    touched: list[str] = []
    row_hashes: list[bytes] = []
    row_count = 0
    for table in candidate_tables:
        if table not in tables:
            continue
        columns = tuple(str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall())
        if "thread_id" not in columns:
            continue
        rows = connection.execute(
            f"SELECT * FROM {table} WHERE thread_id = ?",
            (turn_id,),
        ).fetchall()
        if not rows:
            continue
        touched.append(table)
        row_count += len(rows)
        for row in rows:
            digest = hashlib.sha256()
            digest.update(table.encode())
            for column in columns:
                digest.update(b"\0")
                digest.update(column.encode())
                digest.update(b"\0")
                value = row[column]
                if isinstance(value, bytes):
                    digest.update(b"bytes\0")
                    digest.update(value)
                else:
                    digest.update(type(value).__name__.encode())
                    digest.update(b"\0")
                    digest.update(str(value).encode())
            row_hashes.append(digest.digest())
    if not row_hashes:
        return None
    combined = hashlib.sha256()
    for value in sorted(row_hashes):
        combined.update(value)
    return {
        "tables": touched,
        "row_count": row_count,
        "sha256": combined.hexdigest(),
    }


def _assistant_answer(
    messages: tuple[Mapping[str, Any], ...],
) -> tuple[int, str] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if message.get("role") == "assistant" and isinstance(content, str) and content.strip() and not tool_calls:
            return index, content
    return None


def _legacy_context_items(
    messages: tuple[Mapping[str, Any], ...],
    *,
    user_message: str,
    final_answer_index: int | None,
    tool_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    items: list[tuple[str, Mapping[str, Any]]] = []
    skipped_initial_user = False
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("legacy transcript message requires string role/content")
        if role == "system":
            continue
        if role == "user":
            if not skipped_initial_user and content == user_message:
                skipped_initial_user = True
                continue
            items.append(("user_message", {"text": content, "legacy_message": message}))
            continue
        if role == "assistant":
            if index == final_answer_index:
                continue
            raw_calls = message.get("tool_calls", ())
            if raw_calls:
                if not isinstance(raw_calls, (list, tuple)):
                    raise ValueError("legacy assistant tool_calls must be an array")
                calls = tuple(_legacy_tool_call(value) for value in raw_calls)
                items.append(
                    (
                        "model_response",
                        {
                            "text": content,
                            "tool_calls": calls,
                            "legacy_message": message,
                        },
                    )
                )
            elif content:
                items.append(("agent_message", {"text": content, "legacy_message": message}))
            continue
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError("legacy tool message requires tool_call_id")
            durable_result = None if tool_results is None else tool_results.get(tool_call_id)
            items.append(
                (
                    "tool_result",
                    (
                        {
                            **dict(durable_result),
                            "model_content": content,
                            "legacy_message": message,
                        }
                        if durable_result is not None
                        else {
                            "tool_call_id": tool_call_id,
                            "model_content": content,
                            "legacy_message": message,
                        }
                    ),
                )
            )
            continue
        if role == "context":
            items.append(("context_message", {"text": content, "legacy_message": message}))
            continue
        raise ValueError(f"unsupported legacy message role: {role}")
    return tuple(items)


def _legacy_tool_call(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("legacy tool call must be an object")
    call_id = value.get("id")
    name = value.get("name")
    arguments = value.get("arguments", value.get("input"))
    if (
        not isinstance(call_id, str)
        or not call_id
        or not isinstance(name, str)
        or not name
        or not isinstance(arguments, Mapping)
    ):
        raise ValueError("legacy tool call is malformed")
    return {"id": call_id, "name": name, "arguments": dict(arguments)}


def _legacy_binding(raw: str) -> Mapping[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError("legacy runtime binding must be an object")
    return value


def _harness_binding(
    legacy: Mapping[str, Any],
    *,
    resume_compatible: bool,
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    knowledge = legacy.get("knowledge")
    return {
        "schema_version": 1,
        "model_alias": legacy.get("model_alias"),
        "legacy_runtime_binding": dict(legacy),
        "knowledge_config": knowledge if isinstance(knowledge, Mapping) else None,
        "completion_policy": {"require_workspace_change": False},
        "mcp_policy": {"workspace_discovery_enabled": True},
        "tool_execution_policy": {
            "active_skill_ids": [],
            "allow_execute_tools": False,
            "allow_write_tools": False,
            "auto_approve_sandboxed": False,
            "denied_tool_names": [],
            "deny_effects": [],
            "max_parallel_calls": 4,
            "require_confirmation_for": [],
        },
        "model_step_budget": 16,
        "model_token_budget_total": None,
        "legacy_resume_compatible": resume_compatible,
        "legacy_approvals_invalidated": not resume_compatible,
        "legacy_checkpoint": checkpoint,
    }

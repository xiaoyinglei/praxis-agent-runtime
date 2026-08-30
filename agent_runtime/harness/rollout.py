"""Append-only rollout truth with rebuildable SQLite projections."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from agent_runtime.harness.reducer import ProjectionState, apply_record
from agent_runtime.streaming.events import (
    derive_model_public_item_id,
    derive_operation_public_item_id,
    derive_plan_public_item_id,
)

_REDUCER_VERSION = 1


def _model_public_item_ids(*, turn_id: str, attempt_id: str) -> dict[str, str]:
    return {
        channel: derive_model_public_item_id(
            turn_id=turn_id,
            model_attempt_id=attempt_id,
            channel=channel,
        )
        for channel in ("agent_message", "reasoning", "plan")
    }


class ResourceClaimConflictError(RuntimeError):
    """Another running or unknown operation owns an incompatible resource."""


@dataclass(frozen=True, slots=True)
class ThreadSnapshot:
    thread_id: str
    workspace: str
    parent_thread_id: str | None
    fork_turn_id: str | None
    active_turn_id: str | None
    head_turn_id: str | None
    head_version: int
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class TurnSnapshot:
    turn_id: str
    thread_id: str
    status: str
    predecessor_turn_id: str | None
    turn_index: int
    binding_manifest: Mapping[str, Any]
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class ItemSnapshot:
    item_id: str
    thread_id: str
    turn_id: str
    sequence: int
    kind: str
    status: str
    producer: str
    payload: Mapping[str, Any]
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    record_id: int
    record_uuid: str
    thread_id: str
    turn_id: str | None
    thread_sequence: int
    record_type: str
    payload_schema_version: int
    producer: str
    payload: Mapping[str, Any]
    payload_hash: str
    committed_at_ms: int


@dataclass(frozen=True, slots=True)
class CommittedMutation[T]:
    """A mutation result paired with exactly its committed Rollout records."""

    value: T
    records: tuple[RolloutRecord, ...]


@dataclass(frozen=True, slots=True)
class VerificationReport:
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelOperationSnapshot:
    operation_id: str
    thread_id: str
    turn_id: str
    status: str
    active_attempt_id: str
    generation: int
    request_hash: str
    context_hash: str
    tool_hash: str
    wire_hash: str
    request_ref: Mapping[str, Any]
    response_item_id: str | None
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class ModelAttemptSnapshot:
    attempt_id: str
    operation_id: str
    generation: int
    status: str
    provider_response_id: str | None
    usage: Mapping[str, Any]
    claim_owner: str | None
    lease_expires_at: float | None
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class ToolOperationSnapshot:
    operation_id: str
    thread_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    arguments_digest: str
    execution_revision: str
    effects: tuple[str, ...]
    resources: tuple[Mapping[str, Any], ...]
    idempotent: bool
    status: str
    attempt_count: int
    error_code: str | None
    requires_reconciliation: bool
    result_item_id: str | None
    approval_request_id: str | None
    claim_generation: int
    fencing_token: str | None
    claim_owner: str | None
    lease_expires_at: float | None
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class InteractionSnapshot:
    request_id: str
    thread_id: str
    turn_id: str
    kind: str
    status: str
    version: int
    operation_id: str | None
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class ApprovalSnapshot:
    request_id: str
    thread_id: str
    turn_id: str
    operation_id: str
    status: str
    scope: Mapping[str, Any]
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    artifact_id: str
    thread_id: str
    turn_id: str
    blob_sha256: str
    size_bytes: int
    media_type: str
    name: str
    applied_thread_sequence: int


class RolloutStore:
    """Own canonical records and update projections through one reducer."""

    def __init__(
        self,
        path: Path,
    ) -> None:
        self._path = Path(path)
        self._artifact_root = self._path.with_name(f"{self._path.name}.artifacts")
        self._pending_transaction_records: list[RolloutRecord] | None = None
        self._captured_commits: list[tuple[RolloutRecord, ...]] | None = None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def __enter__(self) -> RolloutStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    @property
    def epoch(self) -> str:
        row = self._connection.execute("SELECT value FROM store_metadata WHERE key = 'store_epoch'").fetchone()
        if row is None:
            raise RuntimeError("RolloutStore epoch metadata is missing")
        return str(row["value"])

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    def capture_mutation[T](self, operation: Callable[[], T]) -> CommittedMutation[T]:
        """Run one Store mutation and return only that transaction's records."""

        if self._captured_commits is not None:
            raise RuntimeError("committed mutation capture cannot be nested")
        captured: list[tuple[RolloutRecord, ...]] = []
        self._captured_commits = captured
        try:
            value = operation()
        finally:
            self._captured_commits = None
        if len(captured) != 1:
            raise RuntimeError("committed mutation must contain exactly one outer transaction")
        return CommittedMutation(value=value, records=captured[0])

    def commit_artifact(
        self,
        *,
        turn_id: str,
        content: bytes,
        media_type: str,
        name: str,
        fault_injector: Callable[[str], None] | None = None,
    ) -> ArtifactSnapshot:
        """Durably write an immutable blob before appending its canonical reference."""

        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError("artifact media_type must be non-empty")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("artifact name must be non-empty")
        blob_sha256 = hashlib.sha256(content).hexdigest()
        artifact_id = f"artifact_{uuid4().hex}"
        blob_directory = self._artifact_root / "blobs"
        temporary_directory = self._artifact_root / "tmp"
        blob_directory.mkdir(parents=True, exist_ok=True)
        temporary_directory.mkdir(parents=True, exist_ok=True)
        temporary = temporary_directory / f"{uuid4().hex}.tmp"
        blob = blob_directory / blob_sha256
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if fault_injector is not None:
                fault_injector("after_temp_fsync")
            if blob.exists():
                if _file_sha256(blob) != blob_sha256 or blob.stat().st_size != len(content):
                    raise RuntimeError("content-addressed artifact blob is corrupt")
                temporary.unlink()
            else:
                os.replace(temporary, blob)
                _fsync_directory(blob_directory)
            if fault_injector is not None:
                fault_injector("after_rename")
            with self._transaction():
                thread_id = self._running_turn_thread_id(turn_id)
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="artifact_committed",
                    producer="runtime",
                    payload={
                        "artifact_id": artifact_id,
                        "blob_sha256": blob_sha256,
                        "size_bytes": len(content),
                        "media_type": media_type,
                        "name": name,
                    },
                )
                if fault_injector is not None:
                    fault_injector("before_sqlite_commit")
            if fault_injector is not None:
                fault_injector("after_sqlite_commit")
            return self.read_artifact_metadata(artifact_id)
        finally:
            temporary.unlink(missing_ok=True)

    def read_artifact_metadata(self, artifact_id: str) -> ArtifactSnapshot:
        row = self._connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        return _artifact_snapshot(row)

    def list_artifacts(self, turn_id: str | None = None) -> tuple[ArtifactSnapshot, ...]:
        if turn_id is None:
            rows = self._connection.execute("SELECT * FROM artifacts ORDER BY rowid").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM artifacts WHERE turn_id = ? ORDER BY rowid",
                (turn_id,),
            ).fetchall()
        return tuple(_artifact_snapshot(row) for row in rows)

    def artifact_blob_path(self, artifact_id: str) -> Path:
        artifact = self.read_artifact_metadata(artifact_id)
        return self._artifact_root / "blobs" / artifact.blob_sha256

    def read_artifact(self, artifact_id: str) -> bytes:
        artifact = self.read_artifact_metadata(artifact_id)
        blob = self.artifact_blob_path(artifact_id)
        try:
            content = blob.read_bytes()
        except FileNotFoundError:
            raise RuntimeError("artifact blob is missing") from None
        if len(content) != artifact.size_bytes:
            raise RuntimeError("artifact blob size mismatch")
        if hashlib.sha256(content).hexdigest() != artifact.blob_sha256:
            raise RuntimeError("artifact blob hash mismatch")
        return content

    def gc_unreferenced_artifacts(self, *, min_age_seconds: float = 3600.0) -> tuple[str, ...]:
        if min_age_seconds <= 0:
            raise ValueError("artifact GC min_age_seconds must be positive")
        removed: list[str] = []
        with self._transaction():
            referenced = {
                str(row["blob_sha256"]) for row in self._connection.execute("SELECT blob_sha256 FROM artifacts")
            }
            cutoff = time.time() - min_age_seconds
            for directory, keep_referenced in (
                (self._artifact_root / "tmp", False),
                (self._artifact_root / "blobs", True),
            ):
                if not directory.is_dir():
                    continue
                for candidate in directory.iterdir():
                    if not candidate.is_file() or candidate.stat().st_mtime > cutoff:
                        continue
                    if keep_referenced and candidate.name in referenced:
                        continue
                    candidate.unlink()
                    removed.append(str(candidate.relative_to(self._artifact_root)))
        return tuple(sorted(removed))

    @contextmanager
    def migration_transaction(self) -> Iterator[None]:
        """Group an offline import and its metadata marker into one commit."""

        with self._transaction():
            yield

    def mark_legacy_migration_version(self, version: int) -> None:
        if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
            raise ValueError("legacy migration version must be a positive integer")
        with self._transaction():
            self._connection.execute(
                """
                INSERT INTO store_metadata(key, value)
                VALUES ('legacy_migration_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(version),),
            )

    def create_thread(self, *, workspace: Path) -> ThreadSnapshot:
        resolved_workspace = Path(workspace).resolve()
        if not resolved_workspace.is_dir():
            raise ValueError("workspace must be an existing directory")
        thread_id = f"thread_{uuid4().hex}"
        with self._transaction():
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=None,
                record_type="thread_created",
                producer="runtime",
                payload={"workspace": str(resolved_workspace)},
            )
        return self.read_thread(thread_id)

    def fork_thread(self, *, from_turn_id: str) -> ThreadSnapshot:
        thread_id = f"thread_{uuid4().hex}"
        with self._transaction():
            source = self._connection.execute(
                """
                SELECT turns.thread_id, turns.status, threads.workspace
                FROM turns
                JOIN threads ON threads.thread_id = turns.thread_id
                WHERE turns.turn_id = ?
                """,
                (from_turn_id,),
            ).fetchone()
            if source is None:
                raise KeyError(f"unknown fork Turn: {from_turn_id}")
            if source["status"] not in {"completed", "failed", "cancelled"}:
                raise RuntimeError("only a terminal Turn can be a fork point")
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=None,
                record_type="thread_forked",
                producer="runtime",
                payload={
                    "workspace": source["workspace"],
                    "parent_thread_id": source["thread_id"],
                    "fork_turn_id": from_turn_id,
                },
            )
        return self.read_thread(thread_id)

    def start_turn(
        self,
        *,
        thread_id: str,
        user_message: str,
        binding_manifest: Mapping[str, Any],
        input_files: tuple[Mapping[str, Any], ...] = (),
        turn_id: str | None = None,
        turn_producer: str = "runtime",
    ) -> TurnSnapshot:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be non-empty")
        frozen_manifest = _json_object(binding_manifest, field="binding_manifest")
        frozen_input_files = tuple(_json_object(value, field="input file") for value in input_files)
        if turn_id is not None and (not isinstance(turn_id, str) or not turn_id):
            raise ValueError("turn_id must be non-empty when provided")
        if not isinstance(turn_producer, str) or not turn_producer:
            raise ValueError("turn_producer must be non-empty")
        turn_id = turn_id or f"turn_{uuid4().hex}"
        item_id = f"item_{uuid4().hex}"
        with self._transaction():
            thread = self._connection.execute(
                "SELECT active_turn_id, head_turn_id FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if thread is None:
                raise KeyError(f"unknown thread: {thread_id}")
            if thread["active_turn_id"] is not None:
                raise RuntimeError(f"thread already has an active turn: {thread_id}")
            turn_index = self._connection.execute(
                "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM turns WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()[0]
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_started",
                producer=turn_producer,
                payload={
                    "turn_id": turn_id,
                    "predecessor_turn_id": thread["head_turn_id"],
                    "turn_index": turn_index,
                    "binding_manifest": frozen_manifest,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="user",
                payload={"item_id": item_id, "kind": "user_message"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="user",
                payload={"item_id": item_id, "payload": {"text": user_message}},
            )
            for input_file in frozen_input_files:
                input_item_id = f"item_{uuid4().hex}"
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_started",
                    producer="user",
                    payload={"item_id": input_item_id, "kind": "input_file"},
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_completed",
                    producer="user",
                    payload={"item_id": input_item_id, "payload": input_file},
                )
        return self.read_turn(turn_id)

    def complete_turn(
        self,
        *,
        turn_id: str,
        answer: str,
        producer: str = "runtime",
    ) -> TurnSnapshot:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be non-empty")
        item_id = f"item_{uuid4().hex}"
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError(f"turn is not running: {turn_id}")
            thread_id = turn["thread_id"]
            active = self._connection.execute(
                "SELECT active_turn_id FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if active["active_turn_id"] != turn_id:
                raise RuntimeError(f"turn does not own the thread active slot: {turn_id}")
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer=producer,
                payload={"item_id": item_id, "kind": "agent_message"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer=producer,
                payload={"item_id": item_id, "payload": {"text": answer}},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_completed",
                producer=producer,
                payload={"turn_id": turn_id},
            )
        return self.read_turn(turn_id)

    def fail_turn(self, *, turn_id: str, reason: str) -> TurnSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Turn failure reason must be non-empty")
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("only a running Turn can fail")
            active = self._connection.execute(
                "SELECT active_turn_id FROM threads WHERE thread_id = ?",
                (turn["thread_id"],),
            ).fetchone()
            if active is None or active["active_turn_id"] != turn_id:
                raise RuntimeError("failed Turn does not own the Thread active slot")
            self._append_and_reduce(
                thread_id=turn["thread_id"],
                turn_id=turn_id,
                record_type="turn_failed",
                producer="verifier",
                payload={"turn_id": turn_id, "reason": reason},
            )
        return self.read_turn(turn_id)

    def pause_turn(
        self,
        *,
        turn_id: str,
        reason: str,
        producer: str = "runtime",
    ) -> TurnSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Turn pause reason must be non-empty")
        if not isinstance(producer, str) or not producer.strip():
            raise ValueError("Turn pause producer must be non-empty")
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("only a running Turn can pause")
            active = self._connection.execute(
                "SELECT active_turn_id FROM threads WHERE thread_id = ?",
                (turn["thread_id"],),
            ).fetchone()
            if active is None or active["active_turn_id"] != turn_id:
                raise RuntimeError("paused Turn does not own the Thread active slot")
            self._append_and_reduce(
                thread_id=turn["thread_id"],
                turn_id=turn_id,
                record_type="turn_paused",
                producer=producer,
                payload={"turn_id": turn_id, "reason": reason},
            )
        return self.read_turn(turn_id)

    def _record_imported_interruption(
        self,
        *,
        turn_id: str,
        reason: str,
    ) -> TurnSnapshot:
        """Migration-only state import; live recovery uses scoped reconciliation."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Turn interruption reason must be non-empty")
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("only a running Turn can be interrupted")
            active = self._connection.execute(
                "SELECT active_turn_id FROM threads WHERE thread_id = ?",
                (turn["thread_id"],),
            ).fetchone()
            if active is None or active["active_turn_id"] != turn_id:
                raise RuntimeError("interrupted Turn does not own the Thread active slot")
            self._append_and_reduce(
                thread_id=str(turn["thread_id"]),
                turn_id=turn_id,
                record_type="turn_interrupted",
                producer="migration",
                payload={"turn_id": turn_id, "reason": reason},
            )
        return self.read_turn(turn_id)

    def interrupt_orphaned_turn(
        self,
        *,
        turn_id: str,
        reason: str,
        maintenance_confirmed: bool,
    ) -> TurnSnapshot:
        """Recover only the safe gap before any model/tool operation exists."""

        if not maintenance_confirmed:
            raise RuntimeError(
                "orphan recovery requires explicit maintenance confirmation that the prior worker is stopped"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Turn interruption reason must be non-empty")
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("orphan recovery requires a running Turn")
            active = self._connection.execute(
                "SELECT active_turn_id FROM threads WHERE thread_id = ?",
                (turn["thread_id"],),
            ).fetchone()
            if active is None or active["active_turn_id"] != turn_id:
                raise RuntimeError("orphaned Turn does not own the Thread active slot")
            operation_count = 0
            for table in (
                "model_operations",
                "tool_operations",
                "interactions",
                "approvals",
            ):
                row = self._connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                operation_count += int(row["count"])
            if operation_count:
                raise RuntimeError(
                    "pre-dispatch orphan recovery refuses Turns with durable operations; "
                    "use operation-specific reconciliation"
                )
            self._append_and_reduce(
                thread_id=str(turn["thread_id"]),
                turn_id=turn_id,
                record_type="turn_interrupted",
                producer="recovery",
                payload={
                    "turn_id": turn_id,
                    "reason": reason,
                    "maintenance_confirmed": True,
                    "recovery_scope": "pre_dispatch_orphan",
                },
            )
        return self.read_turn(turn_id)

    def record_migrated_context_item(
        self,
        *,
        turn_id: str,
        kind: str,
        payload: Mapping[str, Any],
    ) -> ItemSnapshot:
        """Append one validated legacy transcript item during offline migration."""

        allowed_kinds = {
            "user_message",
            "agent_message",
            "model_response",
            "tool_result",
            "context_message",
            "input_file",
        }
        if kind not in allowed_kinds:
            raise ValueError(f"unsupported migrated Item kind: {kind}")
        frozen_payload = _json_object(payload, field="migrated Item payload")
        if kind == "tool_result":
            self._validate_artifact_references(frozen_payload)
        item_id = f"item_{uuid4().hex}"
        public_projection = (
            {
                "public_item_id": item_id,
                "public_item_kind": "legacy_message",
            }
            if kind == "model_response"
            else {}
        )
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("migrated Items require a running Turn")
            self._append_and_reduce(
                thread_id=str(turn["thread_id"]),
                turn_id=turn_id,
                record_type="item_started",
                producer="migration",
                payload={"item_id": item_id, "kind": kind, **public_projection},
            )
            self._append_and_reduce(
                thread_id=str(turn["thread_id"]),
                turn_id=turn_id,
                record_type="item_completed",
                producer="migration",
                payload={
                    "item_id": item_id,
                    "payload": frozen_payload,
                    **public_projection,
                },
            )
        return self.list_items(turn_id)[-1]

    def record_migrated_tool_operation(
        self,
        *,
        turn_id: str,
        operation_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments_digest: str,
        execution_revision: str,
        idempotent: bool,
        status: str,
        attempt_count: int,
        error_code: str | None,
        requires_reconciliation: bool,
        result_item_id: str | None,
    ) -> ToolOperationSnapshot:
        """Import one frozen legacy operation without invoking its runner."""

        allowed_statuses = {
            "prepared",
            "ready",
            "succeeded",
            "failed",
            "denied",
            "cancelled",
            "unknown",
        }
        if status not in allowed_statuses:
            raise ValueError(f"unsupported migrated tool operation status: {status}")
        for field, value in {
            "operation_id": operation_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments_digest": arguments_digest,
            "execution_revision": execution_revision,
        }.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"migrated tool {field} must be non-empty")
        if isinstance(attempt_count, bool) or not isinstance(attempt_count, int) or attempt_count < 0:
            raise ValueError("migrated tool attempt_count must be non-negative")
        if status == "unknown" and not requires_reconciliation:
            raise ValueError("migrated unknown operation must require reconciliation")
        with self._transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            if (
                self._connection.execute(
                    "SELECT 1 FROM tool_operations WHERE operation_id = ? OR (turn_id = ? AND tool_call_id = ?)",
                    (operation_id, turn_id, tool_call_id),
                ).fetchone()
                is not None
            ):
                raise RuntimeError("migrated tool operation identity already exists")
            if result_item_id is not None:
                item = self._connection.execute(
                    "SELECT kind, payload_json FROM items WHERE item_id = ? AND turn_id = ?",
                    (result_item_id, turn_id),
                ).fetchone()
                if (
                    item is None
                    or item["kind"] != "tool_result"
                    or json.loads(item["payload_json"]).get("tool_call_id") != tool_call_id
                ):
                    raise RuntimeError("migrated operation result linkage is invalid")
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_migrated",
                producer="migration",
                payload={
                    "operation_id": operation_id,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments_digest": arguments_digest,
                    "execution_revision": execution_revision,
                    "effects": [],
                    "resources": [],
                    "idempotent": idempotent,
                    "status": status,
                    "attempt_count": attempt_count,
                    "error_code": error_code,
                    "requires_reconciliation": requires_reconciliation,
                    "result_item_id": result_item_id,
                },
            )
        return self.read_tool_operation(operation_id)

    def record_migrated_pending_interaction(
        self,
        *,
        turn_id: str,
        request_id: str,
        kind: str,
        request: Mapping[str, Any],
        operation_id: str | None = None,
    ) -> InteractionSnapshot:
        """Import a non-approval legacy pause without granting resume authority."""

        if kind not in {"clarification", "choice", "tool_reconciliation"}:
            raise ValueError("unsupported migrated interaction kind")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("migrated interaction request_id must be non-empty")
        frozen_request = _json_object(request, field="migrated interaction request")
        with self._transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            if kind == "tool_reconciliation":
                operation = self._connection.execute(
                    "SELECT status FROM tool_operations WHERE operation_id = ? AND turn_id = ?",
                    (operation_id, turn_id),
                ).fetchone()
                if operation is None or operation["status"] != "unknown":
                    raise RuntimeError("migrated reconciliation requires an unknown operation")
            elif operation_id is not None:
                raise RuntimeError("migrated non-tool interaction cannot bind an operation")
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_requested",
                producer="migration",
                payload={
                    "request_id": request_id,
                    "kind": kind,
                    "operation_id": operation_id,
                    "request": frozen_request,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_paused",
                producer="migration",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    def record_context_compaction(
        self,
        *,
        turn_id: str,
        covered_item_ids: tuple[str, ...],
        summary: str,
        preserved_facts: Mapping[str, Any],
        artifact_refs: tuple[Mapping[str, Any], ...] = (),
        context_version: int,
    ) -> ItemSnapshot:
        """Append an auditable replacement for one committed context prefix."""

        if not covered_item_ids or any(not isinstance(item_id, str) or not item_id for item_id in covered_item_ids):
            raise ValueError("compaction must cover non-empty Item IDs")
        if len(set(covered_item_ids)) != len(covered_item_ids):
            raise ValueError("compaction covered Item IDs must be unique")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("compaction summary must be non-empty")
        if isinstance(context_version, bool) or not isinstance(context_version, int) or context_version < 1:
            raise ValueError("context_version must be a positive integer")

        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("context compaction requires a running Turn")
            visible_items = tuple(item for item in self.list_context_items(turn_id) if item.status == "completed")
            visible_ids = tuple(item.item_id for item in visible_items)
            if visible_ids[: len(covered_item_ids)] != covered_item_ids:
                raise ValueError("compaction must cover a contiguous context prefix in exact order")

            frozen_facts = _json_object(
                preserved_facts,
                field="compaction preserved_facts",
            )
            required_fact_categories = {
                "architecture_and_safety_constraints",
                "file_changes",
                "verification_results",
                "unresolved_work",
                "uncertain_side_effects",
            }
            missing_categories = sorted(required_fact_categories - frozen_facts.keys())
            if missing_categories:
                raise ValueError(f"compaction preserved_facts lacks critical fact categories: {missing_categories}")
            frozen_artifacts = tuple(_json_object(value, field="compaction artifact ref") for value in artifact_refs)
            covered_items = visible_items[: len(covered_item_ids)]
            covered_refs = [
                {
                    "item_id": item.item_id,
                    "thread_id": item.thread_id,
                    "turn_id": item.turn_id,
                    "item_sequence": item.sequence,
                    "applied_thread_sequence": item.applied_thread_sequence,
                    "payload_sha256": hashlib.sha256(_canonical_json(dict(item.payload)).encode()).hexdigest(),
                }
                for item in covered_items
            ]
            grouped_sequences: dict[str, list[int]] = {}
            for item in covered_items:
                grouped_sequences.setdefault(item.thread_id, []).append(item.applied_thread_sequence)
            covered_ranges = [
                {
                    "thread_id": thread_id,
                    "start_thread_sequence": min(sequences),
                    "end_thread_sequence": max(sequences),
                }
                for thread_id, sequences in grouped_sequences.items()
            ]
            durable_state = _context_durable_state(self, turn_id)
            item_id = f"item_{uuid4().hex}"
            payload = {
                "context_version": context_version,
                "covered_item_ids": list(covered_item_ids),
                "covered_item_refs": covered_refs,
                "covered_sequence_ranges": covered_ranges,
                "summary": summary,
                "preserved_facts": frozen_facts,
                "artifact_refs": list(frozen_artifacts),
                "durable_state": durable_state,
            }
            self._append_and_reduce(
                thread_id=str(turn["thread_id"]),
                turn_id=turn_id,
                record_type="item_started",
                producer="runtime",
                payload={"item_id": item_id, "kind": "context_compaction"},
            )
            self._append_and_reduce(
                thread_id=str(turn["thread_id"]),
                turn_id=turn_id,
                record_type="item_completed",
                producer="runtime",
                payload={"item_id": item_id, "payload": payload},
            )
        return self.list_items(turn_id)[-1]

    def cancel_turn(self, *, turn_id: str) -> TurnSnapshot:
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] not in {"paused", "interrupted"}:
                raise RuntimeError("only a safely paused or interrupted Turn can be cancelled")
            thread_id = str(turn["thread_id"])
            unsafe_operation = self._connection.execute(
                """
                SELECT operation_id, status FROM tool_operations
                WHERE turn_id = ? AND status IN ('running', 'unknown')
                ORDER BY rowid LIMIT 1
                """,
                (turn_id,),
            ).fetchone()
            if unsafe_operation is not None:
                raise RuntimeError(
                    f"cancel cannot discard a running or unknown tool operation: {unsafe_operation['operation_id']}"
                )
            pending = self._connection.execute(
                """
                SELECT * FROM interactions
                WHERE turn_id = ? AND status = 'pending'
                ORDER BY rowid
                """,
                (turn_id,),
            ).fetchall()
            for interaction in pending:
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="interaction_resolved",
                    producer="user",
                    payload={
                        "request_id": interaction["request_id"],
                        "version": int(interaction["version"]) + 1,
                        "response": {"action": "abort"},
                        "approval_status": "denied",
                    },
                )
                if interaction["operation_id"] is not None:
                    operation = self._connection.execute(
                        "SELECT status FROM tool_operations WHERE operation_id = ?",
                        (interaction["operation_id"],),
                    ).fetchone()
                    if operation is None or operation["status"] != "awaiting_approval":
                        raise RuntimeError("cancel cannot discard a started or unknown tool operation")
                    self._append_and_reduce(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        record_type="tool_operation_approval_resolved",
                        producer="runtime",
                        payload={
                            "operation_id": interaction["operation_id"],
                            "request_id": interaction["request_id"],
                            "status": "cancelled",
                        },
                    )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_cancelled",
                producer="user",
                payload={"turn_id": turn_id},
            )
        return self.read_turn(turn_id)

    def request_clarification(
        self,
        *,
        turn_id: str,
        question: str,
    ) -> InteractionSnapshot:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("clarification question must be non-empty")
        request_id = f"interaction_{uuid4().hex}"
        with self._transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_requested",
                producer="verifier",
                payload={
                    "request_id": request_id,
                    "kind": "clarification",
                    "operation_id": None,
                    "request": {"question": question},
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_paused",
                producer="verifier",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    def resolve_clarification(
        self,
        *,
        turn_id: str,
        request_id: str,
        response: str,
    ) -> InteractionSnapshot:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("clarification response must be non-empty")
        self._assert_artifact_integrity()
        item_id = f"item_{uuid4().hex}"
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "paused":
                raise RuntimeError("clarification response requires a paused Turn")
            interaction = self._connection.execute(
                "SELECT * FROM interactions WHERE request_id = ? AND turn_id = ?",
                (request_id, turn_id),
            ).fetchone()
            if interaction is None:
                raise KeyError(f"unknown clarification request: {request_id}")
            if interaction["kind"] != "clarification" or interaction["status"] != "pending":
                raise RuntimeError("interaction is not a pending clarification")
            thread_id = str(turn["thread_id"])
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="user",
                payload={
                    "request_id": request_id,
                    "version": int(interaction["version"]) + 1,
                    "response": {"text": response},
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="user",
                payload={"item_id": item_id, "kind": "user_message"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="user",
                payload={"item_id": item_id, "payload": {"text": response}},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_resumed",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    def request_choice(
        self,
        *,
        turn_id: str,
        question: str,
        options: tuple[str, ...],
    ) -> InteractionSnapshot:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("choice question must be non-empty")
        if (
            len(options) < 2
            or any(not isinstance(option, str) or not option.strip() for option in options)
            or len(set(options)) != len(options)
        ):
            raise ValueError("choice options must contain at least two unique non-empty values")
        request_id = f"interaction_{uuid4().hex}"
        with self._transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_requested",
                producer="runtime",
                payload={
                    "request_id": request_id,
                    "kind": "choice",
                    "operation_id": None,
                    "request": {"question": question, "options": list(options)},
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_paused",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    def resolve_choice(
        self,
        *,
        turn_id: str,
        request_id: str,
        selection: str,
    ) -> InteractionSnapshot:
        if not isinstance(selection, str) or not selection.strip():
            raise ValueError("choice selection must be non-empty")
        self._assert_artifact_integrity()
        item_id = f"item_{uuid4().hex}"
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "paused":
                raise RuntimeError("choice response requires a paused Turn")
            interaction = self._connection.execute(
                "SELECT * FROM interactions WHERE request_id = ? AND turn_id = ?",
                (request_id, turn_id),
            ).fetchone()
            if interaction is None:
                raise KeyError(f"unknown choice request: {request_id}")
            if interaction["kind"] != "choice" or interaction["status"] != "pending":
                raise RuntimeError("interaction is not a pending choice")
            request = json.loads(interaction["request_json"])
            options = request.get("options")
            if not isinstance(options, list) or selection not in options:
                raise ValueError("choice selection is not one of the frozen options")
            thread_id = str(turn["thread_id"])
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="user",
                payload={
                    "request_id": request_id,
                    "version": int(interaction["version"]) + 1,
                    "response": {"selection": selection},
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="user",
                payload={"item_id": item_id, "kind": "user_message"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="user",
                payload={"item_id": item_id, "payload": {"text": selection}},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_resumed",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    def prepare_model_operation(
        self,
        *,
        turn_id: str,
        request_hash: str,
        context_hash: str,
        tool_hash: str,
        wire_hash: str,
        request_ref: Mapping[str, Any],
    ) -> ModelOperationSnapshot:
        hashes = {
            "request_hash": request_hash,
            "context_hash": context_hash,
            "tool_hash": tool_hash,
            "wire_hash": wire_hash,
        }
        if not all(isinstance(value, str) and value for value in hashes.values()):
            raise ValueError("model operation hashes must be non-empty strings")
        frozen_ref = _json_object(request_ref, field="request_ref")
        request_id = frozen_ref.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("model request_ref.request_id must be a non-empty string")
        operation_id = f"modelop_{uuid4().hex}"
        attempt_id = f"attempt_{uuid4().hex}"
        request_item_id = f"item_{uuid4().hex}"
        selected_operation_id = operation_id
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError(f"turn is not running: {turn_id}")
            thread_id = turn["thread_id"]
            matches = [
                row
                for row in self._connection.execute(
                    "SELECT * FROM model_operations WHERE turn_id = ? ORDER BY rowid",
                    (turn_id,),
                ).fetchall()
                if json.loads(row["request_ref_json"]).get("request_id") == request_id
            ]
            if len(matches) > 1:
                raise RuntimeError(f"duplicate durable model request identity: {request_id}")
            if matches:
                existing = matches[0]
                existing_hashes = {name: existing[name] for name in hashes}
                existing_ref = json.loads(existing["request_ref_json"])
                if existing_hashes != hashes or existing_ref != frozen_ref:
                    raise RuntimeError(f"model request identity reused with conflicting payload: {request_id}")
                selected_operation_id = str(existing["operation_id"])
            else:
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="model_operation_prepared",
                    producer="runtime",
                    payload={
                        "operation_id": operation_id,
                        "attempt_id": attempt_id,
                        "generation": 1,
                        "public_item_ids": _model_public_item_ids(
                            turn_id=turn_id,
                            attempt_id=attempt_id,
                        ),
                        **hashes,
                        "request_ref": frozen_ref,
                    },
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_started",
                    producer="runtime",
                    payload={"item_id": request_item_id, "kind": "model_request"},
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_completed",
                    producer="runtime",
                    payload={
                        "item_id": request_item_id,
                        "payload": {**hashes, "request_ref": frozen_ref},
                    },
                )
        row = self._connection.execute(
            "SELECT * FROM model_operations WHERE operation_id = ?",
            (selected_operation_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("prepared model operation projection is missing")
        return _model_operation_snapshot(row)

    def dispatch_model_attempt(
        self,
        operation_id: str,
        *,
        worker_id: str = "direct-model-worker",
        lease_seconds: float = 300.0,
        now: float | None = None,
    ) -> ModelAttemptSnapshot:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("model worker_id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("model lease_seconds must be positive")
        self._assert_artifact_integrity()
        dispatched_at = time.time() if now is None else float(now)
        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM model_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if operation["status"] != "prepared":
                raise RuntimeError(f"model operation is not prepared: {operation_id}")
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_attempt_dispatched",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "attempt_id": operation["active_attempt_id"],
                    "generation": operation["generation"],
                    "claim_owner": worker_id,
                    "lease_expires_at": dispatched_at + lease_seconds,
                },
            )
        return self.list_model_attempts(operation_id)[-1]

    def expire_model_attempt_dispatch(
        self,
        *,
        operation_id: str,
        now: float | None = None,
    ) -> ModelAttemptSnapshot:
        observed_at = time.time() if now is None else float(now)
        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM model_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if operation["status"] != "dispatched":
                raise RuntimeError("only a dispatched model attempt can expire")
            attempt = self._connection.execute(
                "SELECT * FROM model_attempts WHERE attempt_id = ?",
                (operation["active_attempt_id"],),
            ).fetchone()
            if attempt is None:
                raise RuntimeError("active model attempt projection is missing")
            expires_at = attempt["lease_expires_at"]
            if expires_at is None or observed_at <= float(expires_at):
                raise RuntimeError("model attempt dispatch lease has not expired")
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_attempt_unknown",
                producer="recovery",
                payload={
                    "operation_id": operation_id,
                    "attempt_id": attempt["attempt_id"],
                    "generation": attempt["generation"],
                    "reason": "dispatch lease expired",
                    "observed_at": observed_at,
                },
            )
            turn = self._connection.execute(
                "SELECT status FROM turns WHERE turn_id = ?",
                (operation["turn_id"],),
            ).fetchone()
            if turn is not None and turn["status"] == "running":
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="turn_paused",
                    producer="recovery",
                    payload={
                        "turn_id": operation["turn_id"],
                        "operation_id": operation_id,
                        "reason": "model outcome unknown after lease expiry",
                    },
                )
        return self.list_model_attempts(operation_id)[-1]

    def mark_model_attempt_unknown(
        self,
        *,
        operation_id: str,
        attempt_id: str,
        generation: int,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> ModelAttemptSnapshot:
        if error_type is not None and (not isinstance(error_type, str) or not error_type.strip()):
            raise ValueError("model dispatch error_type must be non-empty")
        if error_message is not None and (not isinstance(error_message, str) or not error_message.strip()):
            raise ValueError("model dispatch error_message must be non-empty")
        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM model_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if not (
                operation["status"] == "dispatched"
                and operation["active_attempt_id"] == attempt_id
                and operation["generation"] == generation
            ):
                raise RuntimeError("only the active dispatched model attempt can become unknown")
            payload: dict[str, Any] = {
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "generation": generation,
            }
            if error_type is not None:
                payload["error_type"] = error_type.strip()
            if error_message is not None:
                payload["error_message"] = error_message.strip()[:2_000]
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_attempt_unknown",
                producer="runtime",
                payload=payload,
            )
            turn = self._connection.execute(
                "SELECT status FROM turns WHERE turn_id = ?", (operation["turn_id"],)
            ).fetchone()
            if turn is not None and turn["status"] == "running":
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="turn_paused",
                    producer="runtime",
                    payload={
                        "turn_id": operation["turn_id"],
                        "operation_id": operation_id,
                        "reason": "model outcome unknown",
                    },
                )
        return self.list_model_attempts(operation_id)[-1]

    def reject_model_attempt(
        self,
        *,
        operation_id: str,
        attempt_id: str,
        generation: int,
        reason: str,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> ModelAttemptSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("model rejection reason must be non-empty")
        if error_type is not None and (not isinstance(error_type, str) or not error_type.strip()):
            raise ValueError("model rejection error_type must be non-empty")
        if error_message is not None and (not isinstance(error_message, str) or not error_message.strip()):
            raise ValueError("model rejection error_message must be non-empty")
        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM model_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if not (
                operation["status"] == "dispatched"
                and operation["active_attempt_id"] == attempt_id
                and operation["generation"] == generation
            ):
                raise RuntimeError("only the active dispatched model attempt can be rejected")
            payload: dict[str, Any] = {
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "generation": generation,
                "reason": reason,
            }
            if error_type is not None:
                payload["error_type"] = error_type.strip()[:120]
            if error_message is not None:
                payload["error_message"] = error_message.strip()[:2_000]
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_attempt_rejected",
                producer="runtime",
                payload=payload,
            )
        return self.list_model_attempts(operation_id)[-1]

    def prepare_model_retry(self, operation_id: str) -> ModelAttemptSnapshot:
        self._assert_artifact_integrity()
        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM model_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if operation["status"] != "unknown":
                raise RuntimeError(f"model operation is not unknown: {operation_id}")
            attempt_id = f"attempt_{uuid4().hex}"
            generation = operation["generation"] + 1
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_retry_prepared",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "previous_attempt_id": operation["active_attempt_id"],
                    "attempt_id": attempt_id,
                    "generation": generation,
                    "public_item_ids": _model_public_item_ids(
                        turn_id=str(operation["turn_id"]),
                        attempt_id=attempt_id,
                    ),
                },
            )
            turn = self._connection.execute(
                "SELECT status FROM turns WHERE turn_id = ?",
                (operation["turn_id"],),
            ).fetchone()
            if turn is None or turn["status"] != "paused":
                raise RuntimeError("model retry requires a paused Turn")
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="turn_resumed",
                producer="runtime",
                payload={
                    "turn_id": operation["turn_id"],
                    "operation_id": operation_id,
                },
            )
        return self.list_model_attempts(operation_id)[-1]

    def complete_model_attempt(
        self,
        *,
        operation_id: str,
        attempt_id: str,
        generation: int,
        text: str,
        provider_response_id: str | None,
        usage: Mapping[str, Any],
        tool_calls: tuple[Mapping[str, Any], ...] = (),
        response_status: str = "completed",
        incomplete_reason: str | None = None,
        now: float | None = None,
    ) -> bool:
        if not isinstance(text, str):
            raise TypeError("model response text must be a string")
        frozen_tool_calls = tuple(_json_object(call, field="model tool call") for call in tool_calls)
        if response_status not in {"completed", "incomplete"}:
            raise ValueError("model response status is unsupported")
        if response_status == "completed" and incomplete_reason is not None:
            raise ValueError("completed model response cannot have an incomplete reason")
        if response_status == "incomplete" and (
            not isinstance(incomplete_reason, str) or not incomplete_reason.strip()
        ):
            raise ValueError("incomplete model response requires a reason")
        if response_status == "completed" and not text.strip() and not frozen_tool_calls:
            raise ValueError("model response must contain text or tool calls")
        frozen_usage = _json_object(usage, field="usage")
        response_item_id = f"item_{uuid4().hex}"
        observed_at = time.time() if now is None else float(now)
        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM model_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            attempt = self._connection.execute(
                "SELECT * FROM model_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            lease_current = (
                attempt is not None
                and attempt["lease_expires_at"] is not None
                and observed_at <= float(attempt["lease_expires_at"])
            )
            is_current = (
                operation["status"] == "dispatched"
                and operation["active_attempt_id"] == attempt_id
                and operation["generation"] == generation
                and operation["response_item_id"] is None
                and lease_current
            )
            if not is_current:
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="model_attempt_late_response",
                    producer="runtime",
                    payload={
                        "operation_id": operation_id,
                        "attempt_id": attempt_id,
                        "generation": generation,
                        "provider_response_id": provider_response_id,
                        "usage": frozen_usage,
                    },
                )
                return False
            public_item_id = derive_model_public_item_id(
                turn_id=str(operation["turn_id"]),
                model_attempt_id=attempt_id,
                channel="agent_message",
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_attempt_completed",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "attempt_id": attempt_id,
                    "generation": generation,
                    "provider_response_id": provider_response_id,
                    "usage": frozen_usage,
                    "response_item_id": response_item_id,
                    "public_item_id": public_item_id,
                },
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="item_started",
                producer="model",
                payload={
                    "item_id": response_item_id,
                    "kind": "model_response",
                    "attempt_id": attempt_id,
                    "channel": "agent_message",
                    "public_item_id": public_item_id,
                },
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="item_completed",
                producer="model",
                payload={
                    "item_id": response_item_id,
                    "attempt_id": attempt_id,
                    "channel": "agent_message",
                    "public_item_id": public_item_id,
                    "payload": {
                        "text": text,
                        "tool_calls": frozen_tool_calls,
                        "provider_response_id": provider_response_id,
                        "usage": frozen_usage,
                        "response_status": response_status,
                        "incomplete_reason": incomplete_reason,
                    },
                },
            )
        return True

    def list_model_operations(self, turn_id: str | None = None) -> tuple[ModelOperationSnapshot, ...]:
        if turn_id is None:
            rows = self._connection.execute("SELECT * FROM model_operations ORDER BY rowid").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM model_operations WHERE turn_id = ? ORDER BY rowid",
                (turn_id,),
            ).fetchall()
        return tuple(_model_operation_snapshot(row) for row in rows)

    def list_model_attempts(self, operation_id: str) -> tuple[ModelAttemptSnapshot, ...]:
        rows = self._connection.execute(
            "SELECT * FROM model_attempts WHERE operation_id = ? ORDER BY generation",
            (operation_id,),
        ).fetchall()
        return tuple(_model_attempt_snapshot(row) for row in rows)

    def record_tool_call(
        self,
        *,
        turn_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        origin: Mapping[str, Any],
    ) -> ItemSnapshot:
        if not tool_call_id or not tool_name:
            raise ValueError("tool call identity must be non-empty")
        frozen_arguments = _json_object(arguments, field="tool arguments")
        frozen_origin = _json_object(origin, field="tool call origin")
        expected_payload = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": frozen_arguments,
            "origin": frozen_origin,
        }
        item_id: str | None = None
        with self._transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            matches = self._connection.execute(
                """
                SELECT item_id, payload_json, status FROM items
                WHERE turn_id = ? AND kind = 'tool_call'
                ORDER BY sequence
                """,
                (turn_id,),
            ).fetchall()
            same_identity = [
                row for row in matches if json.loads(row["payload_json"]).get("tool_call_id") == tool_call_id
            ]
            if len(same_identity) > 1:
                raise RuntimeError("tool call identity has multiple canonical Items")
            if same_identity:
                existing = same_identity[0]
                if existing["status"] != "completed" or _canonical_json(
                    json.loads(existing["payload_json"])
                ) != _canonical_json(expected_payload):
                    raise RuntimeError("tool call identity conflicts with committed payload")
                item_id = str(existing["item_id"])
            else:
                item_id = f"item_{uuid4().hex}"
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_started",
                    producer="model",
                    payload={"item_id": item_id, "kind": "tool_call"},
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_completed",
                    producer="model",
                    payload={"item_id": item_id, "payload": expected_payload},
                )
        if item_id is None:
            raise RuntimeError("tool call Item identity was not resolved")
        return self.read_item(item_id)

    def record_tool_execution_state(
        self,
        *,
        turn_id: str,
        operation_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments_digest: str,
        execution_revision: str,
        idempotent: bool,
        status: str,
        attempt_count: int,
        error_code: str | None,
        requires_reconciliation: bool,
        effects: tuple[str, ...] = (),
        resources: tuple[Mapping[str, Any], ...] = (),
    ) -> ToolOperationSnapshot:
        allowed = {"prepared", "ready"}
        if status not in allowed:
            raise ValueError(f"unsupported tool operation status: {status}")
        frozen_effects = _string_tuple(effects, field="tool effects")
        frozen_resources = _resource_tuple(resources)
        with self._transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            existing = self._connection.execute(
                "SELECT * FROM tool_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            payload = {
                "operation_id": operation_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments_digest": arguments_digest,
                "execution_revision": execution_revision,
                "effects": frozen_effects,
                "resources": frozen_resources,
                "idempotent": idempotent,
                "status": status,
                "attempt_count": attempt_count,
                "error_code": error_code,
                "requires_reconciliation": requires_reconciliation,
            }
            if existing is None:
                if status != "prepared":
                    raise RuntimeError("first durable tool operation state must be prepared")
                prior_call = self._connection.execute(
                    "SELECT operation_id FROM tool_operations WHERE turn_id = ? AND tool_call_id = ?",
                    (turn_id, tool_call_id),
                ).fetchone()
                if prior_call is not None:
                    raise RuntimeError(f"tool call already has an operation: {prior_call['operation_id']}")
                record_type = "tool_operation_prepared"
            else:
                immutable = {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments_digest": arguments_digest,
                    "execution_revision": execution_revision,
                    "idempotent": int(idempotent),
                    "effects_json": _canonical_json(frozen_effects),
                    "resources_json": _canonical_json(frozen_resources),
                }
                if any(existing[name] != value for name, value in immutable.items()):
                    raise RuntimeError("tool operation identity changed across states")
                current_status = str(existing["status"])
                allowed_next = {"prepared": {"ready"}}
                if current_status not in allowed_next:
                    raise RuntimeError("prepared-state API cannot mutate claimed or terminal operation")
                if status not in allowed_next[current_status]:
                    raise RuntimeError(f"invalid tool operation transition: {current_status} -> {status}")
                if attempt_count < int(existing["attempt_count"]):
                    raise RuntimeError("tool operation attempt count cannot decrease")
                record_type = "tool_operation_status_changed"
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type=record_type,
                producer="runtime",
                payload=payload,
            )
        return self.read_tool_operation(operation_id)

    def claim_tool_operation(
        self,
        *,
        operation_id: str,
        worker_id: str,
        lease_seconds: float,
        now: float | None = None,
    ) -> ToolOperationSnapshot:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._assert_artifact_integrity()
        claimed_at = time.time() if now is None else float(now)
        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM tool_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            if operation["status"] != "ready":
                raise RuntimeError("only a ready tool operation can be claimed")
            requested_resources = tuple(json.loads(operation["resources_json"]))
            active_operations = self._connection.execute(
                """
                SELECT operation_id, resources_json
                FROM tool_operations
                WHERE operation_id != ? AND status IN ('running', 'unknown')
                """,
                (operation_id,),
            ).fetchall()
            for active in active_operations:
                active_resources = tuple(json.loads(active["resources_json"]))
                if _resource_sets_conflict(requested_resources, active_resources):
                    raise ResourceClaimConflictError(
                        f"resource claim conflict with operation: {active['operation_id']}"
                    )
            generation = int(operation["claim_generation"]) + 1
            fencing_token = f"fence_{uuid4().hex}"
            public_item_id = derive_operation_public_item_id(
                turn_id=str(operation["turn_id"]),
                operation_id=operation_id,
                attempt_generation=generation,
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="tool_operation_claimed",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "claim_generation": generation,
                    "fencing_token": fencing_token,
                    "claim_owner": worker_id,
                    "lease_expires_at": claimed_at + lease_seconds,
                    "attempt_count": int(operation["attempt_count"]) + 1,
                    "public_item_id": public_item_id,
                },
            )
        return self.read_tool_operation(operation_id)

    def reject_ready_tool_operation(
        self,
        *,
        operation_id: str,
        error_code: str,
    ) -> ToolOperationSnapshot:
        if not isinstance(error_code, str) or not error_code:
            raise ValueError("start rejection error_code must be non-empty")
        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM tool_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            if operation["status"] != "ready":
                raise RuntimeError("only a ready tool operation can be rejected")
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="tool_operation_start_rejected",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "status": "failed",
                    "attempt_count": operation["attempt_count"],
                    "error_code": error_code,
                    "requires_reconciliation": False,
                },
            )
        return self.read_tool_operation(operation_id)

    def expire_tool_operation_claim(
        self,
        *,
        operation_id: str,
        now: float | None = None,
    ) -> ToolOperationSnapshot:
        observed_at = time.time() if now is None else float(now)
        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM tool_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            if operation["status"] != "running":
                raise RuntimeError("only a running tool operation claim can expire")
            expires_at = operation["lease_expires_at"]
            if expires_at is None or observed_at <= float(expires_at):
                raise RuntimeError("tool operation claim has not expired")
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="tool_operation_claim_expired",
                producer="recovery",
                payload={
                    "operation_id": operation_id,
                    "claim_generation": operation["claim_generation"],
                    "fencing_token": operation["fencing_token"],
                    "observed_at": observed_at,
                },
            )
            request_id = f"interaction_{uuid4().hex}"
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="interaction_requested",
                producer="recovery",
                payload={
                    "request_id": request_id,
                    "kind": "tool_reconciliation",
                    "operation_id": operation_id,
                    "request": {
                        "reason": "tool outcome unknown after lease expiry",
                        "claim_generation": operation["claim_generation"],
                        "claim_owner": operation["claim_owner"],
                        "idempotent": bool(operation["idempotent"]),
                    },
                },
            )
            turn = self._connection.execute(
                "SELECT status FROM turns WHERE turn_id = ?",
                (operation["turn_id"],),
            ).fetchone()
            if turn is not None and turn["status"] == "running":
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="turn_paused",
                    producer="recovery",
                    payload={
                        "turn_id": operation["turn_id"],
                        "operation_id": operation_id,
                        "request_id": request_id,
                        "reason": "tool outcome unknown after lease expiry",
                    },
                )
        return self.read_tool_operation(operation_id)

    def mark_tool_result_missing(
        self,
        *,
        operation_id: str,
    ) -> InteractionSnapshot:
        """Pause when execution outcome committed but its canonical result did not."""

        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM tool_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            if operation["status"] not in {"succeeded", "failed"}:
                raise RuntimeError("missing-result recovery requires a committed execution outcome")
            if operation["result_item_id"] is not None:
                raise RuntimeError("tool operation already has a canonical result")
            turn = self._connection.execute(
                "SELECT status FROM turns WHERE turn_id = ?",
                (operation["turn_id"],),
            ).fetchone()
            if turn is None or turn["status"] != "running":
                raise RuntimeError("missing-result recovery requires a running Turn")
            existing = self._connection.execute(
                """
                SELECT request_id FROM interactions
                WHERE turn_id = ? AND kind = 'tool_reconciliation' AND status = 'pending'
                """,
                (operation["turn_id"],),
            ).fetchall()
            if existing:
                raise RuntimeError("tool reconciliation is already pending")
            prior_status = str(operation["status"])
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="tool_operation_result_missing",
                producer="recovery",
                payload={
                    "operation_id": operation_id,
                    "committed_execution_status": prior_status,
                },
            )
            request_id = f"interaction_{uuid4().hex}"
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="interaction_requested",
                producer="recovery",
                payload={
                    "request_id": request_id,
                    "kind": "tool_reconciliation",
                    "operation_id": operation_id,
                    "request": {
                        "reason": "tool execution outcome committed but ToolResult is missing",
                        "committed_execution_status": prior_status,
                        "idempotent": bool(operation["idempotent"]),
                    },
                },
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="turn_paused",
                producer="recovery",
                payload={
                    "turn_id": operation["turn_id"],
                    "operation_id": operation_id,
                    "request_id": request_id,
                    "reason": "canonical ToolResult is missing after execution",
                },
            )
        return self.read_interaction(request_id)

    def commit_tool_operation_outcome(
        self,
        *,
        operation_id: str,
        claim_generation: int,
        fencing_token: str,
        status: str,
        attempt_count: int,
        error_code: str | None,
        requires_reconciliation: bool,
        now: float | None = None,
    ) -> bool:
        if status not in {"succeeded", "failed", "unknown"}:
            raise ValueError(f"unsupported claimed tool outcome: {status}")
        observed_at = time.time() if now is None else float(now)
        with self._transaction():
            operation = self._connection.execute(
                "SELECT * FROM tool_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            is_current = (
                operation["status"] == "running"
                and int(operation["claim_generation"]) == claim_generation
                and operation["fencing_token"] == fencing_token
                and operation["lease_expires_at"] is not None
                and observed_at <= float(operation["lease_expires_at"])
            )
            record_type = "tool_operation_outcome_committed" if is_current else "tool_operation_stale_result"
            public_item_id = derive_operation_public_item_id(
                turn_id=str(operation["turn_id"]),
                operation_id=operation_id,
                attempt_generation=claim_generation,
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type=record_type,
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "claim_generation": claim_generation,
                    "fencing_token": fencing_token,
                    "status": status,
                    "attempt_count": attempt_count,
                    "error_code": error_code,
                    "requires_reconciliation": requires_reconciliation,
                    "observed_at": observed_at,
                    "public_item_id": public_item_id,
                },
            )
            if is_current and status == "unknown":
                request_id = f"interaction_{uuid4().hex}"
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="interaction_requested",
                    producer="runtime",
                    payload={
                        "request_id": request_id,
                        "kind": "tool_reconciliation",
                        "operation_id": operation_id,
                        "request": {
                            "reason": "tool execution outcome is unknown",
                            "error_code": error_code,
                            "claim_generation": claim_generation,
                            "claim_owner": operation["claim_owner"],
                            "idempotent": bool(operation["idempotent"]),
                        },
                    },
                )
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="turn_paused",
                    producer="runtime",
                    payload={
                        "turn_id": operation["turn_id"],
                        "operation_id": operation_id,
                        "request_id": request_id,
                        "reason": "tool execution outcome is unknown",
                    },
                )
        return is_current

    def resolve_tool_reconciliation(
        self,
        *,
        turn_id: str,
        status: str,
        result: Mapping[str, Any],
        reconciler_revision: str,
    ) -> ItemSnapshot:
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("reconciled tool status is unsupported")
        self._assert_artifact_integrity()
        if not isinstance(reconciler_revision, str) or not reconciler_revision.strip():
            raise ValueError("reconciler_revision must be non-empty")
        frozen_result = _json_object(result, field="reconciled tool result")
        item_id = f"item_{uuid4().hex}"
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "paused":
                raise RuntimeError("tool reconciliation requires a paused Turn")
            pending = self._connection.execute(
                """
                SELECT * FROM interactions
                WHERE turn_id = ? AND kind = 'tool_reconciliation' AND status = 'pending'
                ORDER BY rowid
                """,
                (turn_id,),
            ).fetchall()
            if len(pending) != 1:
                raise RuntimeError("reconciliation requires exactly one pending interaction")
            interaction = pending[0]
            operation = self._connection.execute(
                "SELECT * FROM tool_operations WHERE operation_id = ?",
                (interaction["operation_id"],),
            ).fetchone()
            if operation is None or operation["status"] != "unknown":
                raise RuntimeError("reconciliation interaction is not bound to an unknown operation")
            thread_id = str(turn["thread_id"])
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="reconciler",
                payload={
                    "request_id": interaction["request_id"],
                    "version": int(interaction["version"]) + 1,
                    "response": {
                        "status": status,
                        "reconciler_revision": reconciler_revision,
                    },
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_reconciled",
                producer="reconciler",
                payload={
                    "operation_id": operation["operation_id"],
                    "status": status,
                    "error_code": frozen_result.get("error_code"),
                    "reconciler_revision": reconciler_revision,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="reconciler",
                payload={"item_id": item_id, "kind": "tool_result"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="reconciler",
                payload={"item_id": item_id, "payload": frozen_result},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_result_linked",
                producer="reconciler",
                payload={
                    "operation_id": operation["operation_id"],
                    "result_item_id": item_id,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_resumed",
                producer="reconciler",
                payload={
                    "turn_id": turn_id,
                    "request_id": interaction["request_id"],
                },
            )
        return self.list_items(turn_id)[-1]

    def resolve_tool_approval(
        self,
        *,
        turn_id: str,
        decision: str,
    ) -> InteractionSnapshot:
        if decision not in {"approve", "deny"}:
            raise ValueError("approval decision must be approve or deny")
        self._assert_artifact_integrity()
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "paused":
                raise RuntimeError(f"turn is not paused: {turn_id}")
            pending = self._connection.execute(
                """
                SELECT * FROM interactions
                WHERE turn_id = ? AND kind = 'tool_approval' AND status = 'pending'
                ORDER BY rowid
                """,
                (turn_id,),
            ).fetchall()
            if len(pending) != 1:
                raise RuntimeError("resume requires exactly one pending tool approval")
            interaction = pending[0]
            operation = self._connection.execute(
                "SELECT status FROM tool_operations WHERE operation_id = ?",
                (interaction["operation_id"],),
            ).fetchone()
            if operation is None or operation["status"] != "awaiting_approval":
                raise RuntimeError("pending approval is not bound to an awaiting operation")
            thread_id = str(turn["thread_id"])
            approval_status = "approved" if decision == "approve" else "denied"
            operation_status = "ready" if decision == "approve" else "denied"
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="user",
                payload={
                    "request_id": interaction["request_id"],
                    "version": int(interaction["version"]) + 1,
                    "response": {"decision": decision},
                    "approval_status": approval_status,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_approval_resolved",
                producer="runtime",
                payload={
                    "operation_id": interaction["operation_id"],
                    "request_id": interaction["request_id"],
                    "status": operation_status,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_resumed",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": interaction["request_id"]},
            )
        return self.read_interaction(str(interaction["request_id"]))

    def invalidate_tool_approval(
        self,
        *,
        turn_id: str,
        reason: str,
    ) -> InteractionSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("approval invalidation reason must be non-empty")
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "paused":
                raise RuntimeError(f"turn is not paused: {turn_id}")
            pending = self._connection.execute(
                """
                SELECT * FROM interactions
                WHERE turn_id = ? AND kind = 'tool_approval' AND status = 'pending'
                ORDER BY rowid
                """,
                (turn_id,),
            ).fetchall()
            if len(pending) != 1:
                raise RuntimeError("invalidation requires one pending tool approval")
            interaction = pending[0]
            operation = self._connection.execute(
                "SELECT status FROM tool_operations WHERE operation_id = ?",
                (interaction["operation_id"],),
            ).fetchone()
            if operation is None or operation["status"] != "awaiting_approval":
                raise RuntimeError("approval is not bound to an awaiting operation")
            thread_id = str(turn["thread_id"])
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="user",
                payload={
                    "request_id": interaction["request_id"],
                    "version": int(interaction["version"]) + 1,
                    "response": {
                        "decision": "approve",
                        "invalidation_reason": reason,
                    },
                    "approval_status": "approved",
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="approval_invalidated",
                producer="runtime",
                payload={
                    "request_id": interaction["request_id"],
                    "reason": reason,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_superseded",
                producer="runtime",
                payload={
                    "operation_id": interaction["operation_id"],
                    "request_id": interaction["request_id"],
                    "reason": reason,
                },
            )
        return self.read_interaction(str(interaction["request_id"]))

    def invalidate_resolved_tool_approval(
        self,
        *,
        turn_id: str,
        reason: str,
    ) -> InteractionSnapshot:
        """Invalidate an approved ready operation after crash-time revalidation."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("approval invalidation reason must be non-empty")
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("resolved approval recovery requires a running Turn")
            interaction = self._connection.execute(
                """
                SELECT * FROM interactions
                WHERE turn_id = ? AND kind = 'tool_approval' AND status = 'resolved'
                ORDER BY rowid DESC LIMIT 1
                """,
                (turn_id,),
            ).fetchone()
            if (
                interaction is None
                or json.loads(interaction["response_json"]).get("decision") != "approve"
                or interaction["operation_id"] is None
            ):
                raise RuntimeError("no resolved approved operation can be invalidated")
            operation = self._connection.execute(
                "SELECT status FROM tool_operations WHERE operation_id = ?",
                (interaction["operation_id"],),
            ).fetchone()
            approval = self._connection.execute(
                "SELECT status FROM approvals WHERE request_id = ?",
                (interaction["request_id"],),
            ).fetchone()
            if (
                operation is None
                or operation["status"] != "ready"
                or approval is None
                or approval["status"] != "approved"
            ):
                raise RuntimeError("resolved approval is not bound to a ready operation")
            thread_id = str(turn["thread_id"])
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="approval_invalidated",
                producer="runtime",
                payload={
                    "request_id": interaction["request_id"],
                    "reason": reason,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_superseded",
                producer="runtime",
                payload={"operation_id": interaction["operation_id"]},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_paused",
                producer="recovery",
                payload={
                    "turn_id": turn_id,
                    "request_id": interaction["request_id"],
                    "reason": reason,
                },
            )
        return self.read_interaction(str(interaction["request_id"]))

    def request_tool_approval(
        self,
        *,
        turn_id: str,
        operation_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments_digest: str,
        execution_revision: str,
        idempotent: bool,
        request: Mapping[str, Any],
        effects: tuple[str, ...] = (),
        resources: tuple[Mapping[str, Any], ...] = (),
    ) -> InteractionSnapshot:
        frozen_request = _json_object(request, field="approval request")
        frozen_effects = _string_tuple(effects, field="tool effects")
        frozen_resources = _resource_tuple(resources)
        request_id = f"interaction_{uuid4().hex}"
        with self._transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            prior_call = self._connection.execute(
                "SELECT operation_id FROM tool_operations WHERE turn_id = ? AND tool_call_id = ?",
                (turn_id, tool_call_id),
            ).fetchone()
            if prior_call is not None:
                raise RuntimeError(f"tool call already has an operation: {prior_call['operation_id']}")
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_prepared",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments_digest": arguments_digest,
                    "execution_revision": execution_revision,
                    "effects": frozen_effects,
                    "resources": frozen_resources,
                    "idempotent": idempotent,
                    "status": "prepared",
                    "attempt_count": 0,
                    "error_code": None,
                    "requires_reconciliation": False,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_requested",
                producer="runtime",
                payload={
                    "request_id": request_id,
                    "kind": "tool_approval",
                    "operation_id": operation_id,
                    "request": frozen_request,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_approval_requested",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "request_id": request_id,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_paused",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    def record_tool_result(
        self,
        *,
        turn_id: str,
        operation_id: str | None,
        result: Mapping[str, Any],
    ) -> ItemSnapshot:
        frozen_result = _json_object(result, field="tool result")
        self._validate_artifact_references(frozen_result)
        item_id = f"item_{uuid4().hex}"
        with self._transaction():
            turn = self._connection.execute(
                "SELECT thread_id, status FROM turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            thread_id = str(turn["thread_id"])
            if operation_id is not None:
                operation = self._connection.execute(
                    "SELECT * FROM tool_operations WHERE operation_id = ? AND turn_id = ?",
                    (operation_id, turn_id),
                ).fetchone()
                if operation is None:
                    raise KeyError(f"unknown tool operation: {operation_id}")
                if operation["status"] not in {
                    "succeeded",
                    "failed",
                    "denied",
                    "cancelled",
                    "unknown",
                }:
                    raise RuntimeError("tool result requires a terminal operation state")
                paused_unknown = turn["status"] == "paused" and operation["status"] == "unknown"
                if paused_unknown:
                    reconciliation = self._connection.execute(
                        """
                        SELECT request_id FROM interactions
                        WHERE turn_id = ? AND operation_id = ?
                          AND kind = 'tool_reconciliation' AND status = 'pending'
                        """,
                        (turn_id, operation_id),
                    ).fetchall()
                    if len(reconciliation) != 1:
                        raise RuntimeError("paused unknown tool result requires one reconciliation interaction")
                elif turn["status"] != "running":
                    raise RuntimeError(f"turn is not running: {turn_id}")
            elif turn["status"] != "running":
                raise RuntimeError(f"turn is not running: {turn_id}")
            public_item_id: str | None = None
            plan_public_item_id: str | None = None
            plan_snapshot: Mapping[str, Any] | None = None
            if operation_id is not None:
                attempt_generation = int(operation["claim_generation"])
                if attempt_generation > 0:
                    public_item_id = derive_operation_public_item_id(
                        turn_id=turn_id,
                        operation_id=operation_id,
                        attempt_generation=attempt_generation,
                    )
                if operation["tool_name"] == "update_plan" and frozen_result.get("is_error") is not True:
                    structured = frozen_result.get("structured_content")
                    revision = structured.get("revision") if isinstance(structured, Mapping) else None
                    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                        raise RuntimeError("successful update_plan result requires a canonical revision")
                    call = self._connection.execute(
                        """
                        SELECT payload_json FROM items
                        WHERE turn_id = ? AND kind = 'tool_call' AND status = 'completed'
                        ORDER BY sequence
                        """,
                        (turn_id,),
                    ).fetchall()
                    matching_calls = [
                        json.loads(row["payload_json"])
                        for row in call
                        if json.loads(row["payload_json"]).get("tool_call_id") == operation["tool_call_id"]
                    ]
                    if len(matching_calls) != 1:
                        raise RuntimeError("successful update_plan result requires one canonical ToolCall")
                    arguments = matching_calls[0].get("arguments")
                    if not isinstance(arguments, Mapping):
                        raise RuntimeError("successful update_plan ToolCall arguments are malformed")
                    plan_public_item_id = derive_plan_public_item_id(
                        turn_id=turn_id,
                        revision=revision,
                    )
                    plan_snapshot = {
                        "revision": revision,
                        "plan": arguments.get("plan", []),
                        "explanation": arguments.get("explanation"),
                    }
            started_payload: dict[str, Any] = {"item_id": item_id, "kind": "tool_result"}
            if public_item_id is not None:
                started_payload.update(
                    {
                        "operation_id": operation_id,
                        "attempt_generation": attempt_generation,
                        "public_item_id": public_item_id,
                    }
                )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="tool",
                payload=started_payload,
            )
            completed_payload: dict[str, Any] = {
                "item_id": item_id,
                "payload": frozen_result,
            }
            if public_item_id is not None:
                completed_payload.update(
                    {
                        "operation_id": operation_id,
                        "attempt_generation": attempt_generation,
                        "public_item_id": public_item_id,
                    }
                )
            if plan_public_item_id is not None:
                completed_payload.update(
                    {
                        "plan_public_item_id": plan_public_item_id,
                        "plan_snapshot": plan_snapshot,
                    }
                )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="tool",
                payload=completed_payload,
            )
            if operation_id is not None:
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="tool_operation_result_linked",
                    producer="runtime",
                    payload={
                        "operation_id": operation_id,
                        "result_item_id": item_id,
                    },
                )
        return self.list_items(turn_id)[-1]

    def read_tool_operation(self, operation_id: str) -> ToolOperationSnapshot:
        row = self._connection.execute(
            "SELECT * FROM tool_operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown tool operation: {operation_id}")
        return _tool_operation_snapshot(row)

    def list_tool_operations(self, turn_id: str | None = None) -> tuple[ToolOperationSnapshot, ...]:
        if turn_id is None:
            rows = self._connection.execute("SELECT * FROM tool_operations ORDER BY rowid").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM tool_operations WHERE turn_id = ? ORDER BY rowid",
                (turn_id,),
            ).fetchall()
        return tuple(_tool_operation_snapshot(row) for row in rows)

    def read_interaction(self, request_id: str) -> InteractionSnapshot:
        row = self._connection.execute("SELECT * FROM interactions WHERE request_id = ?", (request_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown interaction: {request_id}")
        return _interaction_snapshot(row)

    def list_interactions(self, turn_id: str | None = None) -> tuple[InteractionSnapshot, ...]:
        if turn_id is None:
            rows = self._connection.execute("SELECT * FROM interactions ORDER BY rowid").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM interactions WHERE turn_id = ? ORDER BY rowid",
                (turn_id,),
            ).fetchall()
        return tuple(_interaction_snapshot(row) for row in rows)

    def list_approvals(self, turn_id: str | None = None) -> tuple[ApprovalSnapshot, ...]:
        if turn_id is None:
            rows = self._connection.execute("SELECT * FROM approvals ORDER BY rowid").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM approvals WHERE turn_id = ? ORDER BY rowid",
                (turn_id,),
            ).fetchall()
        return tuple(_approval_snapshot(row) for row in rows)

    def record_final_proposal(self, *, turn_id: str, answer: str) -> ItemSnapshot:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be non-empty")
        item_id = f"item_{uuid4().hex}"
        with self._transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="model",
                payload={"item_id": item_id, "kind": "final_proposal"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="model",
                payload={"item_id": item_id, "payload": {"text": answer}},
            )
        return self.list_items(turn_id)[-1]

    def record_completion_decision(
        self,
        *,
        turn_id: str,
        proposal_item_id: str,
        action: str,
        reason: str,
    ) -> ItemSnapshot:
        if action not in {"accept", "continue", "pause", "fail"}:
            raise ValueError(f"unsupported completion action: {action}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("completion reason must be non-empty")
        item_id = f"item_{uuid4().hex}"
        with self._transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            proposal = self._connection.execute(
                "SELECT kind, status FROM items WHERE item_id = ? AND turn_id = ?",
                (proposal_item_id, turn_id),
            ).fetchone()
            if proposal is None or proposal["kind"] != "final_proposal" or proposal["status"] != "completed":
                raise RuntimeError("completion decision requires a completed final proposal")
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="verifier",
                payload={"item_id": item_id, "kind": "completion_decision"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="verifier",
                payload={
                    "item_id": item_id,
                    "payload": {
                        "proposal_item_id": proposal_item_id,
                        "action": action,
                        "reason": reason,
                    },
                },
            )
        return self.list_items(turn_id)[-1]

    def record_completion_feedback(
        self,
        *,
        turn_id: str,
        reason: str,
    ) -> ItemSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("completion feedback reason must be non-empty")
        item_id = f"item_{uuid4().hex}"
        with self._transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="verifier",
                payload={"item_id": item_id, "kind": "completion_feedback"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="verifier",
                payload={"item_id": item_id, "payload": {"text": reason}},
            )
        return self.list_items(turn_id)[-1]

    def _running_turn_thread_id(self, turn_id: str) -> str:
        turn = self._connection.execute("SELECT thread_id, status FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "running":
            raise RuntimeError(f"turn is not running: {turn_id}")
        return str(turn["thread_id"])

    def read_thread(self, thread_id: str) -> ThreadSnapshot:
        row = self._connection.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown thread: {thread_id}")
        return _thread_snapshot(row)

    def list_threads(self) -> tuple[ThreadSnapshot, ...]:
        rows = self._connection.execute("SELECT * FROM threads ORDER BY rowid").fetchall()
        return tuple(_thread_snapshot(row) for row in rows)

    def read_turn(self, turn_id: str) -> TurnSnapshot:
        row = self._connection.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown turn: {turn_id}")
        return _turn_snapshot(row)

    def list_turns(self) -> tuple[TurnSnapshot, ...]:
        rows = self._connection.execute("SELECT * FROM turns ORDER BY rowid").fetchall()
        return tuple(_turn_snapshot(row) for row in rows)

    def list_items(self, turn_id: str) -> tuple[ItemSnapshot, ...]:
        rows = self._connection.execute(
            "SELECT * FROM items WHERE turn_id = ? ORDER BY sequence", (turn_id,)
        ).fetchall()
        return tuple(_item_snapshot(row) for row in rows)

    def read_item(self, item_id: str) -> ItemSnapshot:
        row = self._connection.execute(
            "SELECT * FROM items WHERE item_id = ?",
            (item_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown Item: {item_id}")
        return _item_snapshot(row)

    def list_context_items(self, turn_id: str) -> tuple[ItemSnapshot, ...]:
        self._assert_artifact_integrity()
        turn = self._connection.execute(
            "SELECT thread_id, turn_index FROM turns WHERE turn_id = ?", (turn_id,)
        ).fetchone()
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        segments: list[tuple[str, int]] = []
        segment_thread_id = str(turn["thread_id"])
        segment_turn_index = int(turn["turn_index"])
        while True:
            segments.append((segment_thread_id, segment_turn_index))
            thread = self._connection.execute(
                "SELECT parent_thread_id, fork_turn_id FROM threads WHERE thread_id = ?",
                (segment_thread_id,),
            ).fetchone()
            if thread is None:
                raise RuntimeError("context Thread projection is missing")
            parent_thread_id = thread["parent_thread_id"]
            fork_turn_id = thread["fork_turn_id"]
            if parent_thread_id is None and fork_turn_id is None:
                break
            if parent_thread_id is None or fork_turn_id is None:
                raise RuntimeError("fork ancestry projection is incomplete")
            fork_turn = self._connection.execute(
                "SELECT thread_id, turn_index FROM turns WHERE turn_id = ?",
                (fork_turn_id,),
            ).fetchone()
            if fork_turn is None or fork_turn["thread_id"] != parent_thread_id:
                raise RuntimeError("fork point does not belong to its parent Thread")
            segment_thread_id = str(parent_thread_id)
            segment_turn_index = int(fork_turn["turn_index"])
        rows: list[sqlite3.Row] = []
        for ancestor_thread_id, cutoff in reversed(segments):
            rows.extend(
                self._connection.execute(
                    """
                    SELECT items.*
                    FROM items
                    JOIN turns ON turns.turn_id = items.turn_id
                    WHERE turns.thread_id = ? AND turns.turn_index <= ?
                    ORDER BY turns.turn_index, items.sequence
                    """,
                    (ancestor_thread_id, cutoff),
                ).fetchall()
            )
        return tuple(_item_snapshot(row) for row in rows)

    def list_records(self, thread_id: str) -> tuple[RolloutRecord, ...]:
        rows = self._connection.execute(
            "SELECT * FROM rollout_records WHERE thread_id = ? ORDER BY thread_sequence",
            (thread_id,),
        ).fetchall()
        return tuple(_record_snapshot(row) for row in rows)

    def list_global_records(
        self,
        *,
        after_record_id: int = 0,
    ) -> tuple[RolloutRecord, ...]:
        if isinstance(after_record_id, bool) or not isinstance(after_record_id, int):
            raise TypeError("after_record_id must be an integer")
        if after_record_id < 0:
            raise ValueError("after_record_id must be non-negative")
        rows = self._connection.execute(
            "SELECT * FROM rollout_records WHERE record_id > ? ORDER BY record_id",
            (after_record_id,),
        ).fetchall()
        return tuple(_record_snapshot(row) for row in rows)

    def projection_hashes(self) -> dict[str, str]:
        rows = self._connection.execute(
            "SELECT thread_id, canonical_hash FROM projection_meta ORDER BY thread_id"
        ).fetchall()
        return {str(row["thread_id"]): str(row["canonical_hash"]) for row in rows}

    def verify(self) -> VerificationReport:
        errors: list[str] = []
        expected = ProjectionState()
        rows = self._connection.execute("SELECT * FROM rollout_records ORDER BY record_id").fetchall()
        for row in rows:
            payload_json = row["payload_json"]
            if hashlib.sha256(payload_json.encode()).hexdigest() != row["payload_hash"]:
                errors.append(f"record {row['record_id']} payload hash mismatch")
                continue
            apply_record(
                expected,
                _record_snapshot(row),
                reducer_version=_REDUCER_VERSION,
            )
        actual_threads = {
            row["thread_id"]: _thread_row_payload(row) for row in self._connection.execute("SELECT * FROM threads")
        }
        actual_turns = {
            row["turn_id"]: _turn_row_payload(row) for row in self._connection.execute("SELECT * FROM turns")
        }
        actual_items = {
            row["item_id"]: _item_row_payload(row) for row in self._connection.execute("SELECT * FROM items")
        }
        actual_model_operations = {
            row["operation_id"]: _model_operation_row_payload(row)
            for row in self._connection.execute("SELECT * FROM model_operations")
        }
        actual_model_attempts = {
            row["attempt_id"]: _model_attempt_row_payload(row)
            for row in self._connection.execute("SELECT * FROM model_attempts")
        }
        actual_tool_operations = {
            row["operation_id"]: _tool_operation_row_payload(row)
            for row in self._connection.execute("SELECT * FROM tool_operations")
        }
        actual_interactions = {
            row["request_id"]: _interaction_row_payload(row)
            for row in self._connection.execute("SELECT * FROM interactions")
        }
        actual_approvals = {
            row["request_id"]: _approval_row_payload(row) for row in self._connection.execute("SELECT * FROM approvals")
        }
        actual_artifacts = {
            row["artifact_id"]: _artifact_row_payload(row)
            for row in self._connection.execute("SELECT * FROM artifacts")
        }
        for name, expected_payload, actual_payload in (
            ("threads", expected.threads, actual_threads),
            ("turns", expected.turns, actual_turns),
            ("items", expected.items, actual_items),
            ("model_operations", expected.model_operations, actual_model_operations),
            ("model_attempts", expected.model_attempts, actual_model_attempts),
            ("tool_operations", expected.tool_operations, actual_tool_operations),
            ("interactions", expected.interactions, actual_interactions),
            ("approvals", expected.approvals, actual_approvals),
            ("artifacts", expected.artifacts, actual_artifacts),
        ):
            if _canonical_json(expected_payload) != _canonical_json(actual_payload):
                errors.append(f"{name} projection mismatch")
        errors.extend(self._artifact_integrity_errors(expected))
        record_positions = {
            row["thread_id"]: {
                "applied_thread_sequence": row["thread_sequence"],
                "applied_record_id": row["record_id"],
            }
            for row in rows
        }
        expected_metadata = {
            thread_id: {
                **record_positions[thread_id],
                "reducer_version": _REDUCER_VERSION,
                "canonical_hash": _projection_hash(expected, thread_id),
            }
            for thread_id in expected.threads
        }
        actual_metadata = {
            row["thread_id"]: {
                "applied_thread_sequence": row["applied_thread_sequence"],
                "applied_record_id": row["applied_record_id"],
                "reducer_version": row["reducer_version"],
                "canonical_hash": row["canonical_hash"],
            }
            for row in self._connection.execute("SELECT * FROM projection_meta")
        }
        if _canonical_json(expected_metadata) != _canonical_json(actual_metadata):
            errors.append("projection metadata mismatch")
        return VerificationReport(valid=not errors, errors=tuple(errors))

    def _assert_artifact_integrity(self) -> None:
        errors = self._artifact_integrity_errors()
        if errors:
            raise RuntimeError(f"artifact integrity check failed: {errors[0]}")

    def _artifact_integrity_errors(
        self,
        state: ProjectionState | None = None,
    ) -> list[str]:
        artifacts = (
            state.artifacts
            if state is not None
            else {
                row["artifact_id"]: _artifact_row_payload(row)
                for row in self._connection.execute("SELECT * FROM artifacts")
            }
        )
        errors: list[str] = []
        for artifact_id, payload in artifacts.items():
            snapshot = ArtifactSnapshot(
                artifact_id=str(payload["artifact_id"]),
                thread_id=str(payload["thread_id"]),
                turn_id=str(payload["turn_id"]),
                blob_sha256=str(payload["blob_sha256"]),
                size_bytes=int(payload["size_bytes"]),
                media_type=str(payload["media_type"]),
                name=str(payload["name"]),
                applied_thread_sequence=int(payload["applied_thread_sequence"]),
            )
            error = _artifact_blob_error(
                snapshot,
                self._artifact_root / "blobs" / snapshot.blob_sha256,
            )
            if error is not None:
                errors.append(f"artifact {artifact_id}: {error}")
        if state is not None:
            for item in state.items.values():
                attachments = item["payload"].get("attachments")
                if attachments is None:
                    continue
                if not isinstance(attachments, (list, tuple)):
                    errors.append(f"artifact attachments malformed in Item {item['item_id']}")
                    continue
                for attachment in attachments:
                    if not isinstance(attachment, Mapping):
                        errors.append(f"artifact attachment malformed in Item {item['item_id']}")
                        continue
                    artifact_id = attachment.get("artifact_id")
                    if not isinstance(artifact_id, str) or artifact_id not in artifacts:
                        errors.append(f"artifact attachment in Item {item['item_id']} references unknown artifact")
                        continue
                    durable = artifacts[artifact_id]
                    if (
                        attachment.get("media_type") != durable["media_type"]
                        or attachment.get("name") != durable["name"]
                    ):
                        errors.append(f"artifact attachment metadata mismatch in Item {item['item_id']}")
        return errors

    def _validate_artifact_references(self, payload: Mapping[str, Any]) -> None:
        attachments = payload.get("attachments", ())
        if not isinstance(attachments, (list, tuple)):
            raise RuntimeError("tool result artifact attachments must be a list")
        for attachment in attachments:
            if not isinstance(attachment, Mapping):
                raise RuntimeError("tool result artifact attachment is malformed")
            artifact_id = attachment.get("artifact_id")
            if not isinstance(artifact_id, str):
                raise RuntimeError("tool result references an unknown artifact")
            try:
                artifact = self.read_artifact_metadata(artifact_id)
            except KeyError:
                raise RuntimeError(f"tool result references unknown artifact: {artifact_id}") from None
            if attachment.get("media_type") != artifact.media_type or attachment.get("name") != artifact.name:
                raise RuntimeError("tool result artifact metadata conflicts with durable reference")
            error = _artifact_blob_error(artifact, self.artifact_blob_path(artifact_id))
            if error is not None:
                raise RuntimeError(f"artifact integrity check failed: {error}")

    def rebuild_projections(self) -> None:
        """Rebuild projections from verified canonical records in one transaction."""
        with self._transaction():
            rows = self._connection.execute("SELECT * FROM rollout_records ORDER BY record_id").fetchall()
            for row in rows:
                payload_json = row["payload_json"]
                if hashlib.sha256(payload_json.encode()).hexdigest() != row["payload_hash"]:
                    raise RuntimeError(f"cannot rebuild from corrupt record: {row['record_id']}")
            state = ProjectionState()
            for row in rows:
                apply_record(
                    state,
                    _record_snapshot(row),
                    reducer_version=_REDUCER_VERSION,
                )
            self._connection.execute("DELETE FROM projection_meta")
            self._connection.execute("DELETE FROM model_attempts")
            self._connection.execute("DELETE FROM model_operations")
            self._connection.execute("DELETE FROM approvals")
            self._connection.execute("DELETE FROM interactions")
            self._connection.execute("DELETE FROM tool_operations")
            self._connection.execute("DELETE FROM artifacts")
            self._connection.execute("DELETE FROM items")
            self._connection.execute("DELETE FROM turns")
            self._connection.execute("DELETE FROM threads")
            self._persist_projection_state(state)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        if self._pending_transaction_records is not None:
            yield
            return
        pending: list[RolloutRecord] = []
        self._pending_transaction_records = pending
        began = False
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            began = True
            yield
        except BaseException:
            if began:
                self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
        finally:
            self._pending_transaction_records = None
        captured = self._captured_commits
        if captured is not None:
            captured.append(tuple(pending))

    def _append_and_reduce(
        self,
        *,
        thread_id: str,
        turn_id: str | None,
        record_type: str,
        producer: str,
        payload: Mapping[str, Any],
    ) -> None:
        payload_json = _canonical_json(payload)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
        sequence = self._connection.execute(
            "SELECT COALESCE(MAX(thread_sequence), 0) + 1 FROM rollout_records WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()[0]
        committed_at = datetime.now(UTC)
        committed_at_ms = int(committed_at.timestamp() * 1000)
        cursor = self._connection.execute(
            """
            INSERT INTO rollout_records (
                record_uuid, thread_id, turn_id, thread_sequence, record_type,
                payload_schema_version, producer, payload_json, payload_hash,
                created_at, committed_at_ms
            ) VALUES (?, ?, ?, ?, ?, 2, ?, ?, ?, ?, ?)
            """,
            (
                f"record_{uuid4().hex}",
                thread_id,
                turn_id,
                sequence,
                record_type,
                producer,
                payload_json,
                payload_hash,
                committed_at.isoformat(),
                committed_at_ms,
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM rollout_records WHERE record_id = ?", (cursor.lastrowid,)
        ).fetchone()
        state = self._load_projection_state(thread_id)
        apply_record(
            state,
            _record_snapshot(row),
            reducer_version=_REDUCER_VERSION,
        )
        self._persist_projection_state(state)
        pending = self._pending_transaction_records
        if pending is None:
            raise RuntimeError("RolloutRecord appended outside a transaction")
        pending.append(_record_snapshot(row))

    def _load_projection_state(self, thread_id: str) -> ProjectionState:
        state = ProjectionState()
        thread = self._connection.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        if thread is None:
            return state
        state.threads[thread_id] = _thread_row_payload(thread)
        for row in self._connection.execute("SELECT * FROM turns WHERE thread_id = ?", (thread_id,)):
            state.turns[row["turn_id"]] = _turn_row_payload(row)
        for row in self._connection.execute("SELECT * FROM items WHERE thread_id = ?", (thread_id,)):
            state.items[row["item_id"]] = _item_row_payload(row)
        for row in self._connection.execute("SELECT * FROM model_operations WHERE thread_id = ?", (thread_id,)):
            state.model_operations[row["operation_id"]] = _model_operation_row_payload(row)
        for row in self._connection.execute(
            """
            SELECT model_attempts.*
            FROM model_attempts
            JOIN model_operations
              ON model_operations.operation_id = model_attempts.operation_id
            WHERE model_operations.thread_id = ?
            """,
            (thread_id,),
        ):
            state.model_attempts[row["attempt_id"]] = _model_attempt_row_payload(row)
        for row in self._connection.execute("SELECT * FROM tool_operations WHERE thread_id = ?", (thread_id,)):
            state.tool_operations[row["operation_id"]] = _tool_operation_row_payload(row)
        for row in self._connection.execute("SELECT * FROM interactions WHERE thread_id = ?", (thread_id,)):
            state.interactions[row["request_id"]] = _interaction_row_payload(row)
        for row in self._connection.execute("SELECT * FROM approvals WHERE thread_id = ?", (thread_id,)):
            state.approvals[row["request_id"]] = _approval_row_payload(row)
        for row in self._connection.execute("SELECT * FROM artifacts WHERE thread_id = ?", (thread_id,)):
            state.artifacts[row["artifact_id"]] = _artifact_row_payload(row)
        return state

    def _persist_projection_state(self, state: ProjectionState) -> None:
        for thread in state.threads.values():
            self._connection.execute(
                """
                INSERT INTO threads (
                    thread_id, workspace, parent_thread_id, fork_turn_id,
                    active_turn_id, head_turn_id, head_version,
                    applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    workspace = excluded.workspace,
                    parent_thread_id = excluded.parent_thread_id,
                    fork_turn_id = excluded.fork_turn_id,
                    active_turn_id = excluded.active_turn_id,
                    head_turn_id = excluded.head_turn_id,
                    head_version = excluded.head_version,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    thread["thread_id"],
                    thread["workspace"],
                    thread["parent_thread_id"],
                    thread["fork_turn_id"],
                    thread["active_turn_id"],
                    thread["head_turn_id"],
                    thread["head_version"],
                    thread["applied_thread_sequence"],
                    thread["reducer_version"],
                ),
            )
        for turn in state.turns.values():
            self._connection.execute(
                """
                INSERT INTO turns (
                    turn_id, thread_id, status, predecessor_turn_id, turn_index,
                    binding_manifest_json, applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    status = excluded.status,
                    predecessor_turn_id = excluded.predecessor_turn_id,
                    turn_index = excluded.turn_index,
                    binding_manifest_json = excluded.binding_manifest_json,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    turn["turn_id"],
                    turn["thread_id"],
                    turn["status"],
                    turn["predecessor_turn_id"],
                    turn["turn_index"],
                    _canonical_json(turn["binding_manifest"]),
                    turn["applied_thread_sequence"],
                    turn["reducer_version"],
                ),
            )
        for item in state.items.values():
            self._connection.execute(
                """
                INSERT INTO items (
                    item_id, thread_id, turn_id, sequence, kind, status, producer,
                    payload_json, applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    item["item_id"],
                    item["thread_id"],
                    item["turn_id"],
                    item["sequence"],
                    item["kind"],
                    item["status"],
                    item["producer"],
                    _canonical_json(item["payload"]),
                    item["applied_thread_sequence"],
                    item["reducer_version"],
                ),
            )
        for artifact in state.artifacts.values():
            self._connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, thread_id, turn_id, blob_sha256, size_bytes,
                    media_type, name, applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    blob_sha256 = excluded.blob_sha256,
                    size_bytes = excluded.size_bytes,
                    media_type = excluded.media_type,
                    name = excluded.name,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    artifact["artifact_id"],
                    artifact["thread_id"],
                    artifact["turn_id"],
                    artifact["blob_sha256"],
                    artifact["size_bytes"],
                    artifact["media_type"],
                    artifact["name"],
                    artifact["applied_thread_sequence"],
                    artifact["reducer_version"],
                ),
            )
        for operation in state.model_operations.values():
            self._connection.execute(
                """
                INSERT INTO model_operations (
                    operation_id, thread_id, turn_id, status, active_attempt_id,
                    generation, request_hash, context_hash, tool_hash, wire_hash,
                    request_ref_json, response_item_id, applied_thread_sequence,
                    reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    status = excluded.status,
                    active_attempt_id = excluded.active_attempt_id,
                    generation = excluded.generation,
                    response_item_id = excluded.response_item_id,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    operation["operation_id"],
                    operation["thread_id"],
                    operation["turn_id"],
                    operation["status"],
                    operation["active_attempt_id"],
                    operation["generation"],
                    operation["request_hash"],
                    operation["context_hash"],
                    operation["tool_hash"],
                    operation["wire_hash"],
                    _canonical_json(operation["request_ref"]),
                    operation["response_item_id"],
                    operation["applied_thread_sequence"],
                    operation["reducer_version"],
                ),
            )
        for attempt in state.model_attempts.values():
            self._connection.execute(
                """
                INSERT INTO model_attempts (
                    attempt_id, operation_id, generation, status,
                    provider_response_id, usage_json, claim_owner,
                    lease_expires_at, applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attempt_id) DO UPDATE SET
                    status = excluded.status,
                    provider_response_id = excluded.provider_response_id,
                    usage_json = excluded.usage_json,
                    claim_owner = excluded.claim_owner,
                    lease_expires_at = excluded.lease_expires_at,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    attempt["attempt_id"],
                    attempt["operation_id"],
                    attempt["generation"],
                    attempt["status"],
                    attempt["provider_response_id"],
                    _canonical_json(attempt["usage"]),
                    attempt["claim_owner"],
                    attempt["lease_expires_at"],
                    attempt["applied_thread_sequence"],
                    attempt["reducer_version"],
                ),
            )
        for operation in state.tool_operations.values():
            self._connection.execute(
                """
                INSERT INTO tool_operations (
                    operation_id, thread_id, turn_id, tool_call_id, tool_name,
                    arguments_digest, execution_revision, effects_json,
                    resources_json, idempotent, status,
                    attempt_count, error_code, requires_reconciliation,
                    result_item_id, approval_request_id, applied_thread_sequence,
                    claim_generation, fencing_token, claim_owner,
                    lease_expires_at, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    status = excluded.status,
                    attempt_count = excluded.attempt_count,
                    error_code = excluded.error_code,
                    requires_reconciliation = excluded.requires_reconciliation,
                    result_item_id = excluded.result_item_id,
                    approval_request_id = excluded.approval_request_id,
                    claim_generation = excluded.claim_generation,
                    fencing_token = excluded.fencing_token,
                    claim_owner = excluded.claim_owner,
                    lease_expires_at = excluded.lease_expires_at,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    operation["operation_id"],
                    operation["thread_id"],
                    operation["turn_id"],
                    operation["tool_call_id"],
                    operation["tool_name"],
                    operation["arguments_digest"],
                    operation["execution_revision"],
                    _canonical_json(operation["effects"]),
                    _canonical_json(operation["resources"]),
                    operation["idempotent"],
                    operation["status"],
                    operation["attempt_count"],
                    operation["error_code"],
                    operation["requires_reconciliation"],
                    operation["result_item_id"],
                    operation["approval_request_id"],
                    operation["applied_thread_sequence"],
                    operation["claim_generation"],
                    operation["fencing_token"],
                    operation["claim_owner"],
                    operation["lease_expires_at"],
                    operation["reducer_version"],
                ),
            )
        for interaction in state.interactions.values():
            self._connection.execute(
                """
                INSERT INTO interactions (
                    request_id, thread_id, turn_id, kind, status, version,
                    operation_id, request_json, response_json,
                    applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    status = excluded.status,
                    version = excluded.version,
                    response_json = excluded.response_json,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    interaction["request_id"],
                    interaction["thread_id"],
                    interaction["turn_id"],
                    interaction["kind"],
                    interaction["status"],
                    interaction["version"],
                    interaction["operation_id"],
                    _canonical_json(interaction["request"]),
                    _canonical_json(interaction["response"]),
                    interaction["applied_thread_sequence"],
                    interaction["reducer_version"],
                ),
            )
        for approval in state.approvals.values():
            self._connection.execute(
                """
                INSERT INTO approvals (
                    request_id, thread_id, turn_id, operation_id, status,
                    scope_json, applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    status = excluded.status,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    approval["request_id"],
                    approval["thread_id"],
                    approval["turn_id"],
                    approval["operation_id"],
                    approval["status"],
                    _canonical_json(approval["scope"]),
                    approval["applied_thread_sequence"],
                    approval["reducer_version"],
                ),
            )
        for thread_id, thread in state.threads.items():
            record = self._connection.execute(
                """
                SELECT record_id, thread_sequence
                FROM rollout_records
                WHERE thread_id = ?
                ORDER BY thread_sequence DESC
                LIMIT 1
                """,
                (thread_id,),
            ).fetchone()
            if record is None:
                raise RuntimeError(f"projection has no canonical records: {thread_id}")
            self._connection.execute(
                """
                INSERT INTO projection_meta (
                    thread_id, applied_thread_sequence, applied_record_id,
                    reducer_version, canonical_hash
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    applied_record_id = excluded.applied_record_id,
                    reducer_version = excluded.reducer_version,
                    canonical_hash = excluded.canonical_hash
                """,
                (
                    thread_id,
                    thread["applied_thread_sequence"],
                    record["record_id"],
                    _REDUCER_VERSION,
                    _projection_hash(state, thread_id),
                ),
            )

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS rollout_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_uuid TEXT NOT NULL UNIQUE,
                thread_id TEXT NOT NULL,
                turn_id TEXT,
                thread_sequence INTEGER NOT NULL,
                record_type TEXT NOT NULL,
                payload_schema_version INTEGER NOT NULL,
                producer TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                committed_at_ms INTEGER NOT NULL,
                UNIQUE(thread_id, thread_sequence)
            );
            CREATE TABLE IF NOT EXISTS store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                workspace TEXT NOT NULL,
                parent_thread_id TEXT,
                fork_turn_id TEXT,
                active_turn_id TEXT,
                head_turn_id TEXT,
                head_version INTEGER NOT NULL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                status TEXT NOT NULL,
                predecessor_turn_id TEXT,
                turn_index INTEGER NOT NULL,
                binding_manifest_json TEXT NOT NULL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL,
                UNIQUE(thread_id, turn_index)
            );
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                turn_id TEXT NOT NULL REFERENCES turns(turn_id),
                sequence INTEGER NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                producer TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL,
                UNIQUE(turn_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS model_operations (
                operation_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                turn_id TEXT NOT NULL REFERENCES turns(turn_id),
                status TEXT NOT NULL,
                active_attempt_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                request_hash TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                tool_hash TEXT NOT NULL,
                wire_hash TEXT NOT NULL,
                request_ref_json TEXT NOT NULL,
                response_item_id TEXT,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                turn_id TEXT NOT NULL REFERENCES turns(turn_id),
                blob_sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                media_type TEXT NOT NULL,
                name TEXT NOT NULL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_attempts (
                attempt_id TEXT PRIMARY KEY,
                operation_id TEXT NOT NULL REFERENCES model_operations(operation_id),
                generation INTEGER NOT NULL,
                status TEXT NOT NULL,
                provider_response_id TEXT,
                usage_json TEXT NOT NULL,
                claim_owner TEXT,
                lease_expires_at REAL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL,
                UNIQUE(operation_id, generation)
            );
            CREATE TABLE IF NOT EXISTS tool_operations (
                operation_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                turn_id TEXT NOT NULL REFERENCES turns(turn_id),
                tool_call_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_digest TEXT NOT NULL,
                execution_revision TEXT NOT NULL,
                effects_json TEXT NOT NULL DEFAULT '[]',
                resources_json TEXT NOT NULL DEFAULT '[]',
                idempotent INTEGER NOT NULL,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                error_code TEXT,
                requires_reconciliation INTEGER NOT NULL,
                result_item_id TEXT,
                approval_request_id TEXT,
                claim_generation INTEGER NOT NULL,
                fencing_token TEXT,
                claim_owner TEXT,
                lease_expires_at REAL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL,
                UNIQUE(turn_id, tool_call_id)
            );
            CREATE TABLE IF NOT EXISTS interactions (
                request_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                turn_id TEXT NOT NULL REFERENCES turns(turn_id),
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                version INTEGER NOT NULL,
                operation_id TEXT REFERENCES tool_operations(operation_id),
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS approvals (
                request_id TEXT PRIMARY KEY REFERENCES interactions(request_id),
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                turn_id TEXT NOT NULL REFERENCES turns(turn_id),
                operation_id TEXT NOT NULL REFERENCES tool_operations(operation_id),
                status TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projection_meta (
                thread_id TEXT PRIMARY KEY REFERENCES threads(thread_id),
                applied_thread_sequence INTEGER NOT NULL,
                applied_record_id INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL,
                canonical_hash TEXT NOT NULL
            );
            PRAGMA user_version = 1;
            """
        )
        self._connection.execute(
            "INSERT OR IGNORE INTO store_metadata (key, value) VALUES ('store_epoch', ?)",
            (f"epoch_{uuid4().hex}",),
        )
        rollout_record_columns = {
            str(row["name"]) for row in self._connection.execute("PRAGMA table_info(rollout_records)")
        }
        if "committed_at_ms" not in rollout_record_columns:
            self._connection.execute("ALTER TABLE rollout_records ADD COLUMN committed_at_ms INTEGER")
        thread_columns = {str(row["name"]) for row in self._connection.execute("PRAGMA table_info(threads)")}
        if "parent_thread_id" not in thread_columns:
            self._connection.execute("ALTER TABLE threads ADD COLUMN parent_thread_id TEXT")
        if "fork_turn_id" not in thread_columns:
            self._connection.execute("ALTER TABLE threads ADD COLUMN fork_turn_id TEXT")
        tool_operation_columns = {
            str(row["name"]) for row in self._connection.execute("PRAGMA table_info(tool_operations)")
        }
        if "effects_json" not in tool_operation_columns:
            self._connection.execute("ALTER TABLE tool_operations ADD COLUMN effects_json TEXT NOT NULL DEFAULT '[]'")
        if "resources_json" not in tool_operation_columns:
            self._connection.execute("ALTER TABLE tool_operations ADD COLUMN resources_json TEXT NOT NULL DEFAULT '[]'")


def _projection_hash(state: ProjectionState, thread_id: str) -> str:
    model_operations = {key: value for key, value in state.model_operations.items() if value["thread_id"] == thread_id}
    payload = {
        "thread": state.threads[thread_id],
        "turns": {key: value for key, value in state.turns.items() if value["thread_id"] == thread_id},
        "items": {key: value for key, value in state.items.items() if value["thread_id"] == thread_id},
        "model_operations": model_operations,
        "model_attempts": {
            key: value for key, value in state.model_attempts.items() if value["operation_id"] in model_operations
        },
        "tool_operations": {
            key: value for key, value in state.tool_operations.items() if value["thread_id"] == thread_id
        },
        "interactions": {key: value for key, value in state.interactions.items() if value["thread_id"] == thread_id},
        "approvals": {key: value for key, value in state.approvals.items() if value["thread_id"] == thread_id},
    }
    artifacts = {key: value for key, value in state.artifacts.items() if value["thread_id"] == thread_id}
    if artifacts:
        payload["artifacts"] = artifacts
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _context_durable_state(store: RolloutStore, turn_id: str) -> dict[str, Any]:
    """Project non-text state that compaction must never silently erase."""

    return {
        "interactions": [
            {
                "request_id": interaction.request_id,
                "kind": interaction.kind,
                "status": interaction.status,
                "version": interaction.version,
                "operation_id": interaction.operation_id,
                "request": dict(interaction.request),
                "response": dict(interaction.response),
            }
            for interaction in store.list_interactions(turn_id)
            if interaction.status != "resolved"
        ],
        "approvals": [
            {
                "request_id": approval.request_id,
                "operation_id": approval.operation_id,
                "status": approval.status,
                "scope": dict(approval.scope),
            }
            for approval in store.list_approvals(turn_id)
            if approval.status in {"pending", "approved"}
        ],
        "tool_operations": [
            {
                "operation_id": operation.operation_id,
                "tool_call_id": operation.tool_call_id,
                "tool_name": operation.tool_name,
                "status": operation.status,
                "effects": list(operation.effects),
                "resources": [dict(resource) for resource in operation.resources],
                "requires_reconciliation": operation.requires_reconciliation,
                "error_code": operation.error_code,
            }
            for operation in store.list_tool_operations(turn_id)
            if operation.status not in {"completed", "failed", "denied", "superseded"}
            or operation.requires_reconciliation
        ],
    }


def _json_object(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be JSON-serializable") from exc
    if not isinstance(normalized, dict):
        raise TypeError(f"{field} must be an object")
    return normalized


def _string_tuple(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field} must contain non-empty strings")
    return tuple(sorted(set(values)))


def _resource_tuple(
    values: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, str], ...]:
    resources: list[dict[str, str]] = []
    for value in values:
        kind = value.get("kind")
        identity = value.get("identity")
        access = value.get("access")
        if (
            not isinstance(kind, str)
            or not kind
            or not isinstance(identity, str)
            or not identity
            or access not in {"read", "write"}
        ):
            raise ValueError("tool resources require non-empty kind/identity and read/write access")
        if kind == "filesystem" and not Path(identity).is_absolute():
            raise ValueError("filesystem resource identity must be absolute")
        resources.append({"kind": kind, "identity": identity, "access": str(access)})
    unique = {(item["kind"], item["identity"], item["access"]): item for item in resources}
    return tuple(unique[key] for key in sorted(unique))


def _resource_sets_conflict(
    left: tuple[Mapping[str, Any], ...],
    right: tuple[Mapping[str, Any], ...],
) -> bool:
    for first in left:
        for second in right:
            if first.get("access") == second.get("access") == "read":
                continue
            if first.get("kind") != second.get("kind"):
                continue
            first_identity = first.get("identity")
            second_identity = second.get("identity")
            if not isinstance(first_identity, str) or not isinstance(second_identity, str):
                raise RuntimeError("durable resource claim is malformed")
            if first.get("kind") != "filesystem":
                if first_identity == second_identity:
                    return True
                continue
            first_path = Path(first_identity)
            second_path = Path(second_identity)
            if first_path.is_relative_to(second_path) or second_path.is_relative_to(first_path):
                return True
    return False


def _immutable_object(encoded: str) -> Mapping[str, Any]:
    return MappingProxyType(json.loads(encoded))


def _thread_snapshot(row: sqlite3.Row) -> ThreadSnapshot:
    return ThreadSnapshot(
        thread_id=row["thread_id"],
        workspace=row["workspace"],
        parent_thread_id=row["parent_thread_id"],
        fork_turn_id=row["fork_turn_id"],
        active_turn_id=row["active_turn_id"],
        head_turn_id=row["head_turn_id"],
        head_version=row["head_version"],
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _turn_snapshot(row: sqlite3.Row) -> TurnSnapshot:
    return TurnSnapshot(
        turn_id=row["turn_id"],
        thread_id=row["thread_id"],
        status=row["status"],
        predecessor_turn_id=row["predecessor_turn_id"],
        turn_index=row["turn_index"],
        binding_manifest=_immutable_object(row["binding_manifest_json"]),
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _item_snapshot(row: sqlite3.Row) -> ItemSnapshot:
    return ItemSnapshot(
        item_id=row["item_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        sequence=row["sequence"],
        kind=row["kind"],
        status=row["status"],
        producer=row["producer"],
        payload=_immutable_object(row["payload_json"]),
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _artifact_snapshot(row: sqlite3.Row) -> ArtifactSnapshot:
    return ArtifactSnapshot(
        artifact_id=row["artifact_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        blob_sha256=row["blob_sha256"],
        size_bytes=row["size_bytes"],
        media_type=row["media_type"],
        name=row["name"],
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _model_operation_snapshot(row: sqlite3.Row) -> ModelOperationSnapshot:
    return ModelOperationSnapshot(
        operation_id=row["operation_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        status=row["status"],
        active_attempt_id=row["active_attempt_id"],
        generation=row["generation"],
        request_hash=row["request_hash"],
        context_hash=row["context_hash"],
        tool_hash=row["tool_hash"],
        wire_hash=row["wire_hash"],
        request_ref=_immutable_object(row["request_ref_json"]),
        response_item_id=row["response_item_id"],
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _model_attempt_snapshot(row: sqlite3.Row) -> ModelAttemptSnapshot:
    return ModelAttemptSnapshot(
        attempt_id=row["attempt_id"],
        operation_id=row["operation_id"],
        generation=row["generation"],
        status=row["status"],
        provider_response_id=row["provider_response_id"],
        usage=_immutable_object(row["usage_json"]),
        claim_owner=row["claim_owner"],
        lease_expires_at=row["lease_expires_at"],
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _tool_operation_snapshot(row: sqlite3.Row) -> ToolOperationSnapshot:
    return ToolOperationSnapshot(
        operation_id=row["operation_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
        arguments_digest=row["arguments_digest"],
        execution_revision=row["execution_revision"],
        effects=tuple(json.loads(row["effects_json"])),
        resources=tuple(MappingProxyType(resource) for resource in json.loads(row["resources_json"])),
        idempotent=bool(row["idempotent"]),
        status=row["status"],
        attempt_count=row["attempt_count"],
        error_code=row["error_code"],
        requires_reconciliation=bool(row["requires_reconciliation"]),
        result_item_id=row["result_item_id"],
        approval_request_id=row["approval_request_id"],
        claim_generation=row["claim_generation"],
        fencing_token=row["fencing_token"],
        claim_owner=row["claim_owner"],
        lease_expires_at=row["lease_expires_at"],
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _interaction_snapshot(row: sqlite3.Row) -> InteractionSnapshot:
    return InteractionSnapshot(
        request_id=row["request_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        kind=row["kind"],
        status=row["status"],
        version=row["version"],
        operation_id=row["operation_id"],
        request=_immutable_object(row["request_json"]),
        response=_immutable_object(row["response_json"]),
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _approval_snapshot(row: sqlite3.Row) -> ApprovalSnapshot:
    return ApprovalSnapshot(
        request_id=row["request_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        operation_id=row["operation_id"],
        status=row["status"],
        scope=_immutable_object(row["scope_json"]),
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _record_snapshot(row: sqlite3.Row) -> RolloutRecord:
    return RolloutRecord(
        record_id=row["record_id"],
        record_uuid=row["record_uuid"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        thread_sequence=row["thread_sequence"],
        record_type=row["record_type"],
        payload_schema_version=row["payload_schema_version"],
        producer=row["producer"],
        payload=_immutable_object(row["payload_json"]),
        payload_hash=row["payload_hash"],
        committed_at_ms=_committed_at_ms(row),
    )


def _committed_at_ms(row: sqlite3.Row) -> int:
    if "committed_at_ms" in row.keys():
        value = row["committed_at_ms"]
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    created_at = datetime.fromisoformat(str(row["created_at"]))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return int(created_at.timestamp() * 1000)


def _thread_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "thread_id": row["thread_id"],
        "workspace": row["workspace"],
        "parent_thread_id": row["parent_thread_id"],
        "fork_turn_id": row["fork_turn_id"],
        "active_turn_id": row["active_turn_id"],
        "head_turn_id": row["head_turn_id"],
        "head_version": row["head_version"],
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }


def _turn_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "turn_id": row["turn_id"],
        "thread_id": row["thread_id"],
        "status": row["status"],
        "predecessor_turn_id": row["predecessor_turn_id"],
        "turn_index": row["turn_index"],
        "binding_manifest": json.loads(row["binding_manifest_json"]),
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }


def _item_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "item_id": row["item_id"],
        "thread_id": row["thread_id"],
        "turn_id": row["turn_id"],
        "sequence": row["sequence"],
        "kind": row["kind"],
        "status": row["status"],
        "producer": row["producer"],
        "payload": json.loads(row["payload_json"]),
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }


def _artifact_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "artifact_id": row["artifact_id"],
        "thread_id": row["thread_id"],
        "turn_id": row["turn_id"],
        "blob_sha256": row["blob_sha256"],
        "size_bytes": row["size_bytes"],
        "media_type": row["media_type"],
        "name": row["name"],
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_blob_error(artifact: ArtifactSnapshot, blob: Path) -> str | None:
    if not blob.is_file():
        return "artifact blob is missing"
    if blob.stat().st_size != artifact.size_bytes:
        return "artifact blob size mismatch"
    if _file_sha256(blob) != artifact.blob_sha256:
        return "artifact blob hash mismatch"
    return None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _model_operation_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "operation_id": row["operation_id"],
        "thread_id": row["thread_id"],
        "turn_id": row["turn_id"],
        "status": row["status"],
        "active_attempt_id": row["active_attempt_id"],
        "generation": row["generation"],
        "request_hash": row["request_hash"],
        "context_hash": row["context_hash"],
        "tool_hash": row["tool_hash"],
        "wire_hash": row["wire_hash"],
        "request_ref": json.loads(row["request_ref_json"]),
        "response_item_id": row["response_item_id"],
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }


def _model_attempt_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "attempt_id": row["attempt_id"],
        "operation_id": row["operation_id"],
        "generation": row["generation"],
        "status": row["status"],
        "provider_response_id": row["provider_response_id"],
        "usage": json.loads(row["usage_json"]),
        "claim_owner": row["claim_owner"],
        "lease_expires_at": row["lease_expires_at"],
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }


def _tool_operation_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "operation_id": row["operation_id"],
        "thread_id": row["thread_id"],
        "turn_id": row["turn_id"],
        "tool_call_id": row["tool_call_id"],
        "tool_name": row["tool_name"],
        "arguments_digest": row["arguments_digest"],
        "execution_revision": row["execution_revision"],
        "effects": json.loads(row["effects_json"]),
        "resources": json.loads(row["resources_json"]),
        "idempotent": bool(row["idempotent"]),
        "status": row["status"],
        "attempt_count": row["attempt_count"],
        "error_code": row["error_code"],
        "requires_reconciliation": bool(row["requires_reconciliation"]),
        "result_item_id": row["result_item_id"],
        "approval_request_id": row["approval_request_id"],
        "claim_generation": row["claim_generation"],
        "fencing_token": row["fencing_token"],
        "claim_owner": row["claim_owner"],
        "lease_expires_at": row["lease_expires_at"],
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }


def _interaction_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "request_id": row["request_id"],
        "thread_id": row["thread_id"],
        "turn_id": row["turn_id"],
        "kind": row["kind"],
        "status": row["status"],
        "version": row["version"],
        "operation_id": row["operation_id"],
        "request": json.loads(row["request_json"]),
        "response": json.loads(row["response_json"]),
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }


def _approval_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "request_id": row["request_id"],
        "thread_id": row["thread_id"],
        "turn_id": row["turn_id"],
        "operation_id": row["operation_id"],
        "status": row["status"],
        "scope": json.loads(row["scope_json"]),
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }

"""SQLite and immutable blob persistence, independent of runtime decisions."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self, cast
from uuid import uuid4

from agent_runtime.harness.records import (
    ArtifactSnapshot,
    CommittedMutation,
    RolloutRecord,
    VerificationReport,
)
from agent_runtime.harness.reducer import ProjectionState, apply_record

_REDUCER_VERSION = 1
_PROJECTIONS = {
    "threads": ("thread_id", "rowid"),
    "turns": ("turn_id", "rowid"),
    "items": ("item_id", "sequence"),
    "model_operations": ("operation_id", "rowid"),
    "model_attempts": ("attempt_id", "generation"),
    "budget_reservations": ("reservation_id", "rowid"),
    "child_budget_allocations": ("allocation_id", "rowid"),
    "tool_operations": ("operation_id", "rowid"),
    "interactions": ("request_id", "rowid"),
    "approvals": ("request_id", "rowid"),
    "artifacts": ("artifact_id", "rowid"),
    "rollout_records": ("record_id", "record_id"),
}


class StoreBusyError(RuntimeError):
    """The owner has reached its admission capacity; no work was accepted."""


_EVENT_REFERENCE_KEYS = {
    "item_started": "item_id",
    "item_completed": "item_id",
    "model_attempt_completed": "response_item_id",
    "tool_operation_result_linked": "result_item_id",
    "tool_operation_claimed": "operation_id",
    "tool_operation_prepared": "operation_id",
    "tool_operation_outcome_committed": "operation_id",
    "tool_operation_migrated": "operation_id",
}


def _store_operation[**P, T](method: Callable[P, T]) -> Callable[P, T]:
    """Keep complete synchronous operations on the connection's owning thread."""

    @wraps(method)
    def owned(*args: P.args, **kwargs: P.kwargs) -> T:
        store = args[0]
        assert isinstance(store, RolloutStore)
        store._ensure_open()
        return store.run_sync(lambda: method(*args, **kwargs))

    return owned


class RolloutStore:
    """Own SQLite, append-only facts, projection I/O and immutable blob storage.

    No Turn, approval, budget or tool decisions live here. Use RuntimeState for
    those operations. No-argument construction performs no I/O.
    """

    def __init__(self) -> None:
        self._owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-store")
        self._owner_thread: int | None = None
        self._admission = threading.Lock()
        self._closing: Future[None] | None = None
        self._initialized = False
        self._pending_operations = 0
        self._max_pending_operations = 128

    @classmethod
    def open_sync(cls, path: Path, *, read_only: bool = False, max_pending_operations: int = 128) -> Self:
        """Open/migrate on the owner; reject use on an event-loop thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("use await RolloutStore.open(...) on an event-loop thread")
        self = cls()
        try:
            self._set_capacity(max_pending_operations)
            self.run_sync(lambda: self._initialize(path, read_only=read_only))
        except BaseException:
            self.close()
            raise
        self._initialized = True
        return self

    @classmethod
    async def open(cls, path: Path, *, read_only: bool = False, max_pending_operations: int = 128) -> Self:
        """Open/migrate on the Store owner without blocking the caller's loop."""
        self = cls()
        try:
            self._set_capacity(max_pending_operations)
            await self.run_async(lambda: self._initialize(path, read_only=read_only))
        except BaseException:
            # Cancellation still observes initialization before closing its resource.
            await self.aclose()
            raise
        self._initialized = True
        return self

    def _initialize(self, path: Path, *, read_only: bool) -> None:
        self._owner_thread = threading.get_ident()
        self._path = Path(path)
        self._read_only = read_only
        self._artifact_root = self._path.with_name(f"{self._path.name}.artifacts")
        self._pending_transaction_records: list[RolloutRecord] | None = None
        self._captured_commits: list[tuple[RolloutRecord, ...]] | None = None
        if read_only:
            self._connection = sqlite3.connect(
                f"{self._path.resolve().as_uri()}?mode=ro",
                uri=True,
                isolation_level=None,
            )
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self._path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        # Multiple processes share this WAL database; BEGIN IMMEDIATE must
        # wait out another process's write lock instead of failing fast.
        self._connection.execute("PRAGMA busy_timeout = 30000")
        if not read_only:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema()
        self._columns = {
            table: {row["name"] for row in self._connection.execute(f"PRAGMA table_info({table})")} | {"rowid"}
            for table in _PROJECTIONS
        }

    def _ensure_open(self) -> None:
        if not self._initialized:
            raise RuntimeError("RolloutStore is not opened; use open_sync(...) or await open(...)")

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _submit[T](self, operation: Callable[[], T]) -> Future[T]:
        context = contextvars.copy_context()

        def accepted() -> T:
            try:
                return context.run(operation)
            finally:
                with self._admission:
                    self._pending_operations -= 1

        with self._admission:
            if self._closing is not None:
                raise sqlite3.ProgrammingError("RolloutStore is closed")
            if self._pending_operations >= self._max_pending_operations:
                raise StoreBusyError("RolloutStore operation capacity reached; no work was accepted")
            self._pending_operations += 1
            try:
                future = self._owner.submit(accepted)
            except BaseException:
                self._pending_operations -= 1
                raise
        return future

    def _set_capacity(self, value: int) -> None:
        if type(value) is not int or value < 1:
            raise ValueError("max_pending_operations must be a positive integer")
        self._max_pending_operations = value

    def run_sync[T](self, operation: Callable[[], T]) -> T:
        """Blocking API; compose a complete query/transaction on the same owner.

        Return detached values, never SQLite cursors. Advance read generators with iterate.
        Async consumers use run_async, including for multi-query read helpers.
        """
        if threading.get_ident() == self._owner_thread:
            return operation()
        return self._submit(operation).result()

    async def run_async[**P, T](self, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Await one FIFO operation; cancellation waits for its real outcome.

        Accepted work is never cancelled or retried. If cancellation races a
        failure, the actual Store error wins. Otherwise cancellation propagates
        after the operation finishes, so callers can reconcile durable state.
        """
        return await self._observe(self._submit(lambda: operation(*args, **kwargs)))

    async def iterate[T](self, iterator: Iterator[T]) -> AsyncGenerator[T, None]:
        """Advance a lazy read helper on the owner; materialize one result at a time."""
        end = object()
        try:
            while True:
                value = await self.run_async(next, iterator, end)
                if value is end:
                    return
                yield cast(T, value)
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                await self.run_async(close)

    @staticmethod
    async def _observe[T](future: Future[T]) -> T:
        waiting = asyncio.wrap_future(future)
        cancelled = False
        while not waiting.done():
            try:
                await asyncio.shield(waiting)
            except asyncio.CancelledError:
                cancelled = True
        result = waiting.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _begin_close(self) -> Future[None]:
        with self._admission:
            if self._closing is None:

                def close_connection() -> None:
                    connection = getattr(self, "_connection", None)
                    if connection is not None:
                        connection.close()

                self._closing = self._owner.submit(close_connection)
                self._owner.shutdown(wait=False)
            return self._closing

    def close(self) -> None:
        """Stop admission, drain accepted work and close on the owner (blocking)."""
        if threading.get_ident() == self._owner_thread:
            raise RuntimeError("cannot close RolloutStore from its operation")
        self._begin_close().result()
        self._owner.shutdown(wait=True)

    async def aclose(self) -> None:
        """Stop admission and await the same drain/close barrier asynchronously."""
        await self._observe(self._begin_close())

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    def assert_writable(self) -> None:
        if self._read_only:
            raise sqlite3.OperationalError("attempt to write a readonly database")

    @property
    def path(self) -> Path:
        self._ensure_open()
        return self._path

    def _selection(
        self,
        table: str,
        where: Mapping[str, Any] | None,
        payload_fields: Mapping[str, Any] | None = None,
    ) -> tuple[str, list[Any]]:
        if table not in _PROJECTIONS:
            raise ValueError(f"unknown durable view: {table}")
        clauses: list[str] = []
        parameters: list[Any] = []
        for fields, is_payload in ((where or {}, False), (payload_fields or {}, True)):
            for field, value in fields.items():
                if is_payload:
                    if table != "rollout_records" and table != "items":
                        raise ValueError("payload lookup requires a fact or Item view")
                    if not field.isidentifier():
                        raise ValueError("invalid payload field")
                    column = f"json_extract(payload_json, '$.{field}')"
                else:
                    if field not in self._columns[table]:
                        raise ValueError(f"unknown durable field: {table}.{field}")
                    column = field
                if isinstance(value, tuple):
                    clauses.append(f"{column} IN ({','.join('?' for _ in value)})" if value else "0")
                    parameters.extend(value)
                else:
                    clauses.append(f"{column} IS ?")
                    parameters.append(value)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), parameters

    @_store_operation
    def rows(
        self,
        table: str,
        *,
        where: Mapping[str, Any] | None = None,
        payload_fields: Mapping[str, Any] | None = None,
        order_by: str | None = None,
        newest: bool = False,
        limit: int | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Read detached, decoded view rows. Equality/IN are the only filters.

        This small fixed-schema primitive owns SQL and JSON representation.
        Callers choose which facts matter and interpret their business state.
        """
        selection, parameters = self._selection(table, where, payload_fields)
        order = order_by or _PROJECTIONS[table][1]
        if order not in self._columns[table]:
            raise ValueError(f"unknown ordering field: {table}.{order}")
        sql = f"SELECT * FROM {table}{selection} ORDER BY {order}" + (" DESC" if newest else "")
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ValueError("view limit must be a positive integer")
            sql += " LIMIT ?"
            parameters.append(limit)
        return tuple(_decoded_row(row) for row in self._connection.execute(sql, parameters))

    def first(self, table: str, **selection: Any) -> Mapping[str, Any] | None:
        rows = self.rows(table, **{**selection, "limit": 1})
        return rows[0] if rows else None

    @_store_operation
    def count(self, table: str, *, where: Mapping[str, Any] | None = None) -> int:
        selection, parameters = self._selection(table, where)
        return int(self._connection.execute(f"SELECT COUNT(*) FROM {table}{selection}", parameters).fetchone()[0])

    @_store_operation
    def maximum(self, table: str, field: str, *, where: Mapping[str, Any] | None = None) -> int:
        selection, parameters = self._selection(table, where)
        if field not in self._columns[table]:
            raise ValueError(f"unknown durable field: {table}.{field}")
        return int(
            self._connection.execute(
                f"SELECT COALESCE(MAX({field}), 0) FROM {table}{selection}",
                parameters,
            ).fetchone()[0]
        )

    @_store_operation
    def turn_with_workspace(self, turn_id: str) -> Mapping[str, Any] | None:
        row = self._connection.execute(
            "SELECT turns.*, threads.workspace FROM turns JOIN threads USING(thread_id) WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        return _decoded_row(row) if row is not None else None

    @_store_operation
    def conversation_rows(self, thread_id: str, through_turn_index: int) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            _decoded_row(row)
            for row in self._connection.execute(
                "SELECT items.* FROM items JOIN turns USING(turn_id) "
                "WHERE turns.thread_id = ? AND turns.turn_index <= ? ORDER BY turns.turn_index, items.sequence",
                (thread_id, through_turn_index),
            )
        )

    @_store_operation
    def metadata(self, key: str) -> str | None:
        row = self._connection.execute("SELECT value FROM store_metadata WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else None

    @_store_operation
    def put_metadata(self, key: str, value: str, *, replace: bool = True) -> None:
        if self._pending_transaction_records is None:
            raise RuntimeError("metadata write requires a transaction")
        sql = "INSERT OR REPLACE" if replace else "INSERT"
        self._connection.execute(f"{sql} INTO store_metadata(key, value) VALUES (?, ?)", (key, value))

    @_store_operation
    def delete_metadata(self, key: str) -> None:
        if self._pending_transaction_records is None:
            raise RuntimeError("metadata write requires a transaction")
        self._connection.execute("DELETE FROM store_metadata WHERE key = ?", (key,))

    @contextmanager
    def read_snapshot(self) -> Iterator[None]:
        """A coherent SQLite snapshot, borrowing an existing transaction safely.

        Must surround a complete owner operation. A nested audit never commits
        or rolls back its caller's read/write transaction and takes no write lock.
        """
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("read snapshots must run as a complete Store operation")
        if self._connection.in_transaction:
            yield
            return
        self._connection.execute("BEGIN")
        try:
            yield
        finally:
            self._rollback_quietly()

    @_store_operation
    def install_blob(self, content: bytes, *, fault_injector: Callable[[str], None] | None = None) -> str:
        """Install content durably; the caller commits its reference afterwards."""
        self.assert_writable()
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        digest = hashlib.sha256(content).hexdigest()
        blobs, temporary = self._artifact_root / "blobs", self._artifact_root / "tmp"
        blobs.mkdir(parents=True, exist_ok=True)
        temporary.mkdir(parents=True, exist_ok=True)
        candidate, blob = temporary / f"{uuid4().hex}.tmp", blobs / digest
        try:
            with candidate.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if fault_injector is not None:
                fault_injector("after_temp_fsync")
            if blob.exists():
                if _file_sha256(blob) != digest or blob.stat().st_size != len(content):
                    raise RuntimeError("content-addressed artifact blob is corrupt")
                candidate.unlink()
            else:
                os.replace(candidate, blob)
                _fsync_directory(blobs)
            if fault_injector is not None:
                fault_injector("after_rename")
            return digest
        finally:
            candidate.unlink(missing_ok=True)

    def backup(self, destination: Path) -> Path:
        from agent_runtime.harness.rollout_backup import backup_store

        return backup_store(self, destination)

    @property
    @_store_operation
    def epoch(self) -> str:
        row = self._connection.execute("SELECT value FROM store_metadata WHERE key = 'store_epoch'").fetchone()
        if row is None:
            raise RuntimeError("RolloutStore epoch metadata is missing")
        return str(row["value"])

    @property
    @_store_operation
    def artifact_root(self) -> Path:
        return self._artifact_root

    @_store_operation
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

    async def capture_mutation_async[T](self, operation: Callable[[], T]) -> CommittedMutation[T]:
        """Run the complete transaction and artifact I/O on the Store owner."""
        return await self.run_async(lambda: self.capture_mutation(operation))

    @_store_operation
    def read_artifact_metadata(self, artifact_id: str) -> ArtifactSnapshot:
        row = self._connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        return _artifact_snapshot(row)

    @_store_operation
    def list_artifacts(self, turn_id: str | None = None) -> tuple[ArtifactSnapshot, ...]:
        if turn_id is None:
            rows = self._connection.execute("SELECT * FROM artifacts ORDER BY rowid").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM artifacts WHERE turn_id = ? ORDER BY rowid",
                (turn_id,),
            ).fetchall()
        return tuple(_artifact_snapshot(row) for row in rows)

    @_store_operation
    def artifact_blob_path(self, artifact_id: str) -> Path:
        artifact = self.read_artifact_metadata(artifact_id)
        return self._artifact_root / "blobs" / artifact.blob_sha256

    @_store_operation
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

    @_store_operation
    def gc_unreferenced_artifacts(self, *, min_age_seconds: float = 3600.0) -> tuple[str, ...]:
        if min_age_seconds <= 0:
            raise ValueError("artifact GC min_age_seconds must be positive")
        removed: list[str] = []
        with self.transaction():
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

    @_store_operation
    def last_record(self, thread_id: str) -> RolloutRecord | None:
        row = self._connection.execute(
            "SELECT * FROM rollout_records WHERE thread_id = ? ORDER BY thread_sequence DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        return None if row is None else _record_snapshot(row)

    @_store_operation
    def list_records(self, thread_id: str) -> tuple[RolloutRecord, ...]:
        rows = self._connection.execute(
            "SELECT * FROM rollout_records WHERE thread_id = ? ORDER BY thread_sequence",
            (thread_id,),
        ).fetchall()
        return tuple(_record_snapshot(row) for row in rows)

    @_store_operation
    def list_turn_records(self, turn_id: str) -> tuple[RolloutRecord, ...]:
        rows = self._connection.execute(
            "SELECT * FROM rollout_records WHERE turn_id = ? ORDER BY record_id",
            (turn_id,),
        ).fetchall()
        return tuple(_record_snapshot(row) for row in rows)

    @_store_operation
    def read_turn_completion(self, turn_id: str) -> RolloutRecord | None:
        """Read the terminal acceptance fact, independently of result formatting."""
        row = self._connection.execute(
            """SELECT * FROM rollout_records
               WHERE turn_id = ? AND record_type = 'turn_completed' ORDER BY record_id DESC LIMIT 1""",
            (turn_id,),
        ).fetchone()
        return None if row is None else _record_snapshot(row)

    @_store_operation
    def list_record_tail(
        self,
        thread_id: str,
        *,
        after_thread_sequence: int = 0,
        limit: int | None = None,
        through_thread_sequence: int = 9223372036854775807,
    ) -> tuple[RolloutRecord, ...]:
        """Read a canonical suffix by its indexed sequence, with an optional record page."""
        rows = self._connection.execute(
            """SELECT * FROM rollout_records WHERE thread_id = ? AND thread_sequence > ? AND thread_sequence <= ?
               ORDER BY thread_sequence LIMIT ?""",
            (thread_id, after_thread_sequence, through_thread_sequence, -1 if limit is None else limit),
        ).fetchall()
        return tuple(_record_snapshot(row) for row in rows)

    @_store_operation
    def read_event_reference(
        self,
        record_type: Literal[
            "item_started",
            "item_completed",
            "model_attempt_completed",
            "tool_operation_result_linked",
            "tool_operation_claimed",
            "tool_operation_prepared",
            "tool_operation_outcome_committed",
            "tool_operation_migrated",
        ],
        identity: str,
        *,
        at_record_id: int | None = None,
    ) -> RolloutRecord | None:
        """Read one indexed canonical dependency, including legacy event identities."""
        key = _EVENT_REFERENCE_KEYS[record_type]
        row = self._connection.execute(
            f"""SELECT * FROM rollout_records
                WHERE record_type = '{record_type}' AND json_extract(payload_json, '$.{key}') = ?
                {"" if at_record_id is None else "AND record_id <= ?"}
                ORDER BY record_id DESC LIMIT 1""",
            (identity,) if at_record_id is None else (identity, at_record_id),
        ).fetchone()
        return None if row is None else _record_snapshot(row)

    @_store_operation
    def find_model_response_item(self, turn_id: str, text: str | None) -> str | None:
        """Find one completed response for accepted-answer deduplication, without scanning Items."""
        row = self._connection.execute(
            """SELECT item_id FROM items
               WHERE turn_id = ? AND kind = 'model_response' AND status = 'completed'
                 AND json_extract(payload_json, '$.text') IS ? LIMIT 1""",
            (turn_id, text),
        ).fetchone()
        return None if row is None else str(row["item_id"])

    @_store_operation
    def list_global_records(
        self,
        *,
        after_record_id: int = 0,
        limit: int | None = None,
        through_record_id: int = 9223372036854775807,
    ) -> tuple[RolloutRecord, ...]:
        if isinstance(after_record_id, bool) or not isinstance(after_record_id, int):
            raise TypeError("after_record_id must be an integer")
        if after_record_id < 0:
            raise ValueError("after_record_id must be non-negative")
        rows = self._connection.execute(
            "SELECT * FROM rollout_records WHERE record_id > ? AND record_id <= ? ORDER BY record_id LIMIT ?",
            (after_record_id, through_record_id, -1 if limit is None else limit),
        ).fetchall()
        return tuple(_record_snapshot(row) for row in rows)

    @_store_operation
    def last_global_record_id(self) -> int:
        return int(self._connection.execute("SELECT COALESCE(MAX(record_id), 0) FROM rollout_records").fetchone()[0])

    @_store_operation
    def projection_hashes(self) -> dict[str, str]:
        """Return versioned metadata digests, not proof of projection integrity.

        Legacy v1 digests hash projection JSON; v2 digests bind the canonical
        record prefix. verify() independently checks every projection and blob.
        """
        rows = self._connection.execute(
            "SELECT thread_id, canonical_hash FROM projection_meta ORDER BY thread_id"
        ).fetchall()
        return {str(row["thread_id"]): str(row["canonical_hash"]) for row in rows}

    @_store_operation
    def verify(self, *, thread_ids: frozenset[str] | None = None) -> VerificationReport:
        """Independently replay canonical facts and compare projections/blobs."""
        from agent_runtime.harness.rollout_integrity import verify_store

        return verify_store(self, thread_ids=thread_ids)

    def artifact_integrity_errors(
        self,
        state: ProjectionState | None = None,
        *,
        artifact_ids: set[str] | None = None,
    ) -> list[str]:
        if state is not None:
            artifacts = state.artifacts
        elif artifact_ids is None:
            artifacts = {
                row["artifact_id"]: _artifact_row_payload(row)
                for row in self._connection.execute("SELECT * FROM artifacts")
            }
        else:
            artifacts = {}
            for artifact_id in sorted(artifact_ids):
                row = self._connection.execute(
                    "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
                ).fetchone()
                if row is None:
                    return [f"unknown artifact reference: {artifact_id}"]
                artifacts[artifact_id] = _artifact_row_payload(row)
        errors: list[str] = []
        for artifact_id, payload in artifacts.items():
            if artifact_ids is not None and artifact_id not in artifact_ids:
                continue
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
                    attachment_artifact_id = attachment.get("artifact_id")
                    if not isinstance(attachment_artifact_id, str) or attachment_artifact_id not in artifacts:
                        errors.append(f"artifact attachment in Item {item['item_id']} references unknown artifact")
                        continue
                    durable = artifacts[attachment_artifact_id]
                    if (
                        attachment.get("media_type") != durable["media_type"]
                        or attachment.get("name") != durable["name"]
                    ):
                        errors.append(f"artifact attachment metadata mismatch in Item {item['item_id']}")
        return errors

    @_store_operation
    def rebuild_projections(self) -> None:
        """Stage canonical replay and atomically publish if the source is unchanged."""
        from agent_runtime.harness.rollout_integrity import rebuild_store

        rebuild_store(self)

    def _upgrade_metadata_hash(self, thread_id: str) -> None:
        """Upgrade one legacy prefix inside the pending business transaction.

        Never overwrite a legacy mismatch: verify canonical replay, every
        projection and artifacts before replacing its v1 projection digest.
        """
        row = self._connection.execute("SELECT * FROM projection_meta WHERE thread_id = ?", (thread_id,)).fetchone()
        if row is None or _metadata_hash_version(row) == 2:
            return
        if _metadata_hash_version(row) != 1:
            raise RuntimeError("unsupported Rollout metadata hash version")
        # Legacy migration retains the original full-audit contract. Facts can
        # reference artifacts in other threads; a thread-only audit is incomplete.
        report = self.verify()
        if not report.valid:
            raise RuntimeError("Rollout projection integrity check failed: " + "; ".join(report.errors))
        digest: str | None = None
        for record_row in self._connection.execute(
            "SELECT * FROM rollout_records WHERE thread_id = ? ORDER BY thread_sequence",
            (thread_id,),
        ):
            digest = _record_prefix_hash(digest, _record_snapshot(record_row))
        self._connection.execute(
            "UPDATE projection_meta SET canonical_hash = ?, hash_version = 2 WHERE thread_id = ?",
            (digest, thread_id),
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("transactions must run as a complete Store operation")
        self.assert_writable()
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
                self._rollback_quietly()
            raise
        else:
            try:
                self._connection.execute("COMMIT")
            except BaseException:
                # A failed COMMIT (disk full, I/O error) may leave the
                # transaction open; roll back so the reused connection does
                # not fail the next BEGIN IMMEDIATE with "cannot start a
                # transaction within a transaction".
                self._rollback_quietly()
                raise
        finally:
            self._pending_transaction_records = None
        captured = self._captured_commits
        if captured is not None:
            captured.append(tuple(pending))

    def _rollback_quietly(self) -> None:
        # Best-effort cleanup: the caller's original exception must survive
        # even when the connection is too broken to accept ROLLBACK.
        try:
            self._connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    @_store_operation
    def append_record(
        self,
        *,
        thread_id: str,
        turn_id: str | None,
        record_type: str,
        producer: str,
        payload: Mapping[str, Any],
    ) -> None:
        if self._pending_transaction_records is None:
            raise RuntimeError("RolloutRecord appended outside a transaction")
        payload_json = _canonical_json(payload)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
        sequence = self._connection.execute(
            "SELECT COALESCE(MAX(thread_sequence), 0) + 1 FROM rollout_records WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()[0]
        self._upgrade_metadata_hash(thread_id)
        committed_at = datetime.now(UTC)
        committed_at_ms = int(committed_at.timestamp() * 1000)
        cursor = self._connection.execute(
            """
            INSERT INTO rollout_records (
                record_uuid, thread_id, turn_id, thread_sequence, record_type,
                payload_schema_version, producer, payload_json, payload_hash,
                created_at, committed_at_ms
            ) VALUES (?, ?, ?, ?, ?, 3, ?, ?, ?, ?, ?)
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
        record = _record_snapshot(row)
        state = self._load_projection_state(thread_id, record=record)
        apply_record(
            state,
            record,
            reducer_version=_REDUCER_VERSION,
        )
        self._persist_projection_state(state, changed_sequence=record.thread_sequence, record=record)
        pending = self._pending_transaction_records
        if pending is None:
            raise RuntimeError("RolloutRecord appended outside a transaction")
        pending.append(record)

    def _load_projection_state(
        self,
        thread_id: str,
        *,
        record: RolloutRecord | None = None,
    ) -> ProjectionState:
        state = ProjectionState()
        thread = self._connection.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        if thread is None:
            return state
        state.threads[thread_id] = _thread_row_payload(thread)
        if record is not None:
            # The reducer only needs referenced entities. Item creation also
            # needs this Turn's existing Items to derive its next sequence.
            for table, key, identities, row_payload in (
                ("turns", "turn_id", (record.turn_id,), _turn_row_payload),
                (
                    "items",
                    "item_id",
                    (record.payload.get("item_id"), record.payload.get("answer_item_id")),
                    _item_row_payload,
                ),
                (
                    "model_operations",
                    "operation_id",
                    (record.payload.get("operation_id"),),
                    _model_operation_row_payload,
                ),
                (
                    "model_attempts",
                    "attempt_id",
                    (record.payload.get("attempt_id"), record.payload.get("previous_attempt_id")),
                    _model_attempt_row_payload,
                ),
                (
                    "budget_reservations",
                    "reservation_id",
                    (record.payload.get("reservation_id"),),
                    _budget_reservation_row_payload,
                ),
                (
                    "child_budget_allocations",
                    "allocation_id",
                    (record.payload.get("allocation_id"),),
                    _child_budget_allocation_row_payload,
                ),
                ("tool_operations", "operation_id", (record.payload.get("operation_id"),), _tool_operation_row_payload),
                ("interactions", "request_id", (record.payload.get("request_id"),), _interaction_row_payload),
                ("approvals", "request_id", (record.payload.get("request_id"),), _approval_row_payload),
                ("artifacts", "artifact_id", (record.payload.get("artifact_id"),), _artifact_row_payload),
            ):
                entities = getattr(state, table)
                for identity in identities:
                    if identity is None:
                        continue
                    if table == "model_attempts":
                        row = self._connection.execute(
                            """
                            SELECT model_attempts.* FROM model_attempts
                            JOIN model_operations USING (operation_id)
                            WHERE attempt_id = ? AND thread_id = ?
                            """,
                            (identity, thread_id),
                        ).fetchone()
                    else:
                        row = self._connection.execute(
                            f"SELECT * FROM {table} WHERE {key} = ? AND thread_id = ?",
                            (identity, thread_id),
                        ).fetchone()
                    if row is not None:
                        entities[row[key]] = row_payload(row)
            if record.record_type == "item_started":
                row = self._connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM items WHERE turn_id = ?",
                    (record.turn_id,),
                ).fetchone()
                state.item_sequences[str(record.turn_id)] = int(row["sequence"])
            return state
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
        for row in self._connection.execute(
            "SELECT * FROM budget_reservations WHERE thread_id = ?",
            (thread_id,),
        ):
            state.budget_reservations[row["reservation_id"]] = _budget_reservation_row_payload(row)
        for row in self._connection.execute(
            "SELECT * FROM child_budget_allocations WHERE thread_id = ?",
            (thread_id,),
        ):
            state.child_budget_allocations[row["allocation_id"]] = _child_budget_allocation_row_payload(row)
        for row in self._connection.execute("SELECT * FROM tool_operations WHERE thread_id = ?", (thread_id,)):
            state.tool_operations[row["operation_id"]] = _tool_operation_row_payload(row)
        for row in self._connection.execute("SELECT * FROM interactions WHERE thread_id = ?", (thread_id,)):
            state.interactions[row["request_id"]] = _interaction_row_payload(row)
        for row in self._connection.execute("SELECT * FROM approvals WHERE thread_id = ?", (thread_id,)):
            state.approvals[row["request_id"]] = _approval_row_payload(row)
        for row in self._connection.execute("SELECT * FROM artifacts WHERE thread_id = ?", (thread_id,)):
            state.artifacts[row["artifact_id"]] = _artifact_row_payload(row)
        return state

    def _persist_projection_state(
        self,
        state: ProjectionState,
        *,
        changed_sequence: int | None = None,
        record: RolloutRecord | None = None,
        chain_hashes: Mapping[str, str] | None = None,
        connection: sqlite3.Connection | None = None,
        positions: Mapping[str, Mapping[str, int]] | None = None,
    ) -> None:
        connection = self._connection if connection is None else connection
        if changed_sequence is not None:
            # Every changed reducer entity advances to the current record's
            # sequence. Do not UPSERT entities loaded only as dependencies.
            for entities in (
                state.threads,
                state.turns,
                state.items,
                state.model_operations,
                state.model_attempts,
                state.budget_reservations,
                state.child_budget_allocations,
                state.tool_operations,
                state.interactions,
                state.approvals,
                state.artifacts,
            ):
                for identity in tuple(entities):
                    if entities[identity]["applied_thread_sequence"] != changed_sequence:
                        del entities[identity]
        for thread in state.threads.values():
            connection.execute(
                """
                INSERT INTO threads (
                    thread_id, workspace, parent_thread_id, fork_turn_id,
                    active_turn_id, head_turn_id, head_version, settings_json,
                    applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    workspace = excluded.workspace,
                    parent_thread_id = excluded.parent_thread_id,
                    fork_turn_id = excluded.fork_turn_id,
                    active_turn_id = excluded.active_turn_id,
                    head_turn_id = excluded.head_turn_id,
                    head_version = excluded.head_version,
                    settings_json = excluded.settings_json,
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
                    _canonical_json(thread["settings"]),
                    thread["applied_thread_sequence"],
                    thread["reducer_version"],
                ),
            )
        for turn in state.turns.values():
            connection.execute(
                """
                INSERT INTO turns (
                    turn_id, thread_id, status, terminal_reason_code,
                    terminal_message, predecessor_turn_id, turn_index,
                    binding_manifest_json, applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    status = excluded.status,
                    terminal_reason_code = excluded.terminal_reason_code,
                    terminal_message = excluded.terminal_message,
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
                    turn["terminal_reason_code"],
                    turn["terminal_message"],
                    turn["predecessor_turn_id"],
                    turn["turn_index"],
                    _canonical_json(turn["binding_manifest"]),
                    turn["applied_thread_sequence"],
                    turn["reducer_version"],
                ),
            )
        for item in state.items.values():
            connection.execute(
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
            connection.execute(
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
            connection.execute(
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
            connection.execute(
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
        for reservation in state.budget_reservations.values():
            connection.execute(
                """
                INSERT INTO budget_reservations (
                    reservation_id, thread_id, turn_id, scope_id, operation_id,
                    attempt_id, reserved_json, actual_json, status,
                    applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(reservation_id) DO UPDATE SET
                    actual_json = excluded.actual_json,
                    status = excluded.status,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    reservation["reservation_id"],
                    reservation["thread_id"],
                    reservation["turn_id"],
                    reservation["scope_id"],
                    reservation["operation_id"],
                    reservation["attempt_id"],
                    _canonical_json(reservation["reserved"]),
                    _canonical_json(reservation["actual"]),
                    reservation["status"],
                    reservation["applied_thread_sequence"],
                    reservation["reducer_version"],
                ),
            )
        for allocation in state.child_budget_allocations.values():
            connection.execute(
                """
                INSERT INTO child_budget_allocations (
                    allocation_id, thread_id, parent_turn_id, child_turn_id,
                    allocated_tokens, allocated_cost_micros, actual_json, status,
                    applied_thread_sequence, reducer_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(allocation_id) DO UPDATE SET
                    actual_json = excluded.actual_json,
                    status = excluded.status,
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    reducer_version = excluded.reducer_version
                """,
                (
                    allocation["allocation_id"],
                    allocation["thread_id"],
                    allocation["parent_turn_id"],
                    allocation["child_turn_id"],
                    allocation["allocated_tokens"],
                    allocation["allocated_cost_micros"],
                    _canonical_json(allocation["actual"]),
                    allocation["status"],
                    allocation["applied_thread_sequence"],
                    allocation["reducer_version"],
                ),
            )
        for operation in state.tool_operations.values():
            connection.execute(
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
            connection.execute(
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
            connection.execute(
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
            position = (
                positions[thread_id]
                if positions is not None
                else connection.execute(
                    "SELECT record_id, thread_sequence FROM rollout_records WHERE thread_id = ? "
                    "ORDER BY thread_sequence DESC LIMIT 1",
                    (thread_id,),
                ).fetchone()
            )
            if position is None:
                raise RuntimeError(f"projection has no canonical records: {thread_id}")
            if record is not None:
                metadata = connection.execute(
                    "SELECT canonical_hash FROM projection_meta WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()
                digest = _record_prefix_hash(metadata["canonical_hash"] if metadata else None, record)
            elif chain_hashes is not None:
                digest = chain_hashes[thread_id]
            else:
                raise RuntimeError("projection persistence requires a canonical prefix hash")
            connection.execute(
                """
                INSERT INTO projection_meta (
                    thread_id, applied_thread_sequence, applied_record_id,
                    reducer_version, canonical_hash, hash_version
                ) VALUES (?, ?, ?, ?, ?, 2)
                ON CONFLICT(thread_id) DO UPDATE SET
                    applied_thread_sequence = excluded.applied_thread_sequence,
                    applied_record_id = excluded.applied_record_id,
                    reducer_version = excluded.reducer_version,
                    canonical_hash = excluded.canonical_hash,
                    hash_version = excluded.hash_version
                """,
                (
                    thread_id,
                    thread["applied_thread_sequence"],
                    position["record_id"],
                    _REDUCER_VERSION,
                    digest,
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
            CREATE INDEX IF NOT EXISTS rollout_records_turn_record
                ON rollout_records(turn_id, record_id);
            CREATE INDEX IF NOT EXISTS rollout_records_turn_completion
                ON rollout_records(turn_id, record_id) WHERE record_type = 'turn_completed';
            CREATE INDEX IF NOT EXISTS rollout_records_event_item
                ON rollout_records(record_type, json_extract(payload_json, '$.item_id'), record_id)
                WHERE record_type = 'item_started' OR record_type = 'item_completed';
            CREATE INDEX IF NOT EXISTS rollout_records_event_model_response
                ON rollout_records(json_extract(payload_json, '$.response_item_id'), record_id)
                WHERE record_type = 'model_attempt_completed';
            CREATE INDEX IF NOT EXISTS rollout_records_event_tool_result
                ON rollout_records(json_extract(payload_json, '$.result_item_id'), record_id)
                WHERE record_type = 'tool_operation_result_linked';
            CREATE INDEX IF NOT EXISTS rollout_records_event_tool_operation
                ON rollout_records(record_type, json_extract(payload_json, '$.operation_id'), record_id)
                WHERE record_type = 'tool_operation_claimed'
                   OR record_type = 'tool_operation_prepared'
                   OR record_type = 'tool_operation_outcome_committed'
                   OR record_type = 'tool_operation_migrated';
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
                settings_json TEXT NOT NULL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                status TEXT NOT NULL,
                terminal_reason_code TEXT,
                terminal_message TEXT,
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
            CREATE INDEX IF NOT EXISTS items_model_response_text
                ON items(turn_id, json_extract(payload_json, '$.text'))
                WHERE kind = 'model_response' AND status = 'completed';
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
            CREATE TABLE IF NOT EXISTS budget_reservations (
                reservation_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                turn_id TEXT NOT NULL REFERENCES turns(turn_id),
                scope_id TEXT NOT NULL,
                operation_id TEXT NOT NULL REFERENCES model_operations(operation_id),
                attempt_id TEXT NOT NULL REFERENCES model_attempts(attempt_id),
                reserved_json TEXT NOT NULL,
                actual_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL,
                UNIQUE(attempt_id)
            );
            CREATE TABLE IF NOT EXISTS child_budget_allocations (
                allocation_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                parent_turn_id TEXT NOT NULL,
                child_turn_id TEXT NOT NULL UNIQUE,
                allocated_tokens INTEGER,
                allocated_cost_micros INTEGER,
                actual_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                applied_thread_sequence INTEGER NOT NULL,
                reducer_version INTEGER NOT NULL
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
                canonical_hash TEXT NOT NULL,
                hash_version INTEGER NOT NULL DEFAULT 1
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
        child_budget_columns = {
            str(row["name"]) for row in self._connection.execute("PRAGMA table_info(child_budget_allocations)")
        }
        if child_budget_columns and "allocated_cost_micros" not in child_budget_columns:
            self._connection.execute("ALTER TABLE child_budget_allocations ADD COLUMN allocated_cost_micros INTEGER")
        tool_operation_columns = {
            str(row["name"]) for row in self._connection.execute("PRAGMA table_info(tool_operations)")
        }
        if "effects_json" not in tool_operation_columns:
            self._connection.execute("ALTER TABLE tool_operations ADD COLUMN effects_json TEXT NOT NULL DEFAULT '[]'")
        if "resources_json" not in tool_operation_columns:
            self._connection.execute("ALTER TABLE tool_operations ADD COLUMN resources_json TEXT NOT NULL DEFAULT '[]'")
        metadata_columns = {str(row["name"]) for row in self._connection.execute("PRAGMA table_info(projection_meta)")}
        if "hash_version" not in metadata_columns:
            self._connection.execute("ALTER TABLE projection_meta ADD COLUMN hash_version INTEGER NOT NULL DEFAULT 1")


def _decoded_row(row: sqlite3.Row) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            (key[:-5] if key.endswith("_json") else key): (json.loads(value) if key.endswith("_json") else value)
            for key, value in dict(row).items()
        }
    )


def _metadata_hash_version(row: sqlite3.Row) -> int:
    # Readonly legacy snapshots have never run schema migration.
    if "hash_version" not in row.keys():
        return 1
    value = row["hash_version"]
    return value if type(value) is int else 0


def _record_prefix_hash(previous: str | None, record: RolloutRecord) -> str:
    """Version 2 binds the canonical envelope and previous per-thread prefix.

    This is not a digest of projections. Full verify independently replays and
    compares all projected rows and validates referenced artifact contents.
    """
    envelope = {
        "record_id": record.record_id,
        "record_uuid": record.record_uuid,
        "thread_id": record.thread_id,
        "turn_id": record.turn_id,
        "thread_sequence": record.thread_sequence,
        "record_type": record.record_type,
        "producer": record.producer,
        "payload_schema_version": record.payload_schema_version,
        "payload_hash": record.payload_hash,
        "reducer_version": _REDUCER_VERSION,
    }
    encoded = b"praxis-record-prefix-v2\0" + (previous or "0" * 64).encode("ascii")
    return hashlib.sha256(encoded + b"\0" + _canonical_json(envelope).encode()).hexdigest()


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
    budget_reservations = {
        key: value for key, value in state.budget_reservations.items() if value["thread_id"] == thread_id
    }
    if budget_reservations:
        payload["budget_reservations"] = budget_reservations
    child_budget_allocations = {
        key: value for key, value in state.child_budget_allocations.items() if value["thread_id"] == thread_id
    }
    if child_budget_allocations:
        payload["child_budget_allocations"] = child_budget_allocations
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


def _immutable_object(encoded: str) -> Mapping[str, Any]:
    return MappingProxyType(json.loads(encoded))


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
        "settings": json.loads(row["settings_json"]),
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }


def _turn_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "turn_id": row["turn_id"],
        "thread_id": row["thread_id"],
        "status": row["status"],
        "terminal_reason_code": row["terminal_reason_code"],
        "terminal_message": row["terminal_message"],
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


def _budget_reservation_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "reservation_id": row["reservation_id"],
        "thread_id": row["thread_id"],
        "turn_id": row["turn_id"],
        "scope_id": row["scope_id"],
        "operation_id": row["operation_id"],
        "attempt_id": row["attempt_id"],
        "reserved": json.loads(row["reserved_json"]),
        "actual": json.loads(row["actual_json"]),
        "status": row["status"],
        "applied_thread_sequence": row["applied_thread_sequence"],
        "reducer_version": row["reducer_version"],
    }


def _child_budget_allocation_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "allocation_id": row["allocation_id"],
        "thread_id": row["thread_id"],
        "parent_turn_id": row["parent_turn_id"],
        "child_turn_id": row["child_turn_id"],
        "allocated_tokens": row["allocated_tokens"],
        "allocated_cost_micros": row["allocated_cost_micros"],
        "actual": json.loads(row["actual_json"]),
        "status": row["status"],
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

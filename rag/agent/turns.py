from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_runtime.knowledge import RAGKnowledgeConfig
from rag.agent.core.messages import (
    ModelMessage,
    ToolCall,
    canonical_json_text,
    model_message_payload,
    snapshot_model_message,
)
from rag.agent.streaming.events import (
    EventType,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
)
from rag.agent.streaming.history import DurableTurnEvent

logger = logging.getLogger(__name__)


class TurnStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class TurnNotFoundError(LookupError):
    pass


class TurnStateError(RuntimeError):
    pass


class RuntimeBinding(BaseModel):
    """Secret-free inputs required to rebuild the runtime for one Turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = 2
    model_alias: str | None = None
    workspace_path: str | None = None
    knowledge: RAGKnowledgeConfig | None = None


class _LegacyRuntimeBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_type: str = "generic"
    model_alias: str | None = None
    workspace_path: str | None = None
    knowledge: tuple[str, ...] = ()
    rag_storage_root: str = ".rag"
    embedding_model_alias: str | None = None
    reranker_model_alias: str | None = None
    vector_backend: str = "milvus"
    vector_namespace: str | None = None
    vector_collection_prefix: str | None = None
    vector_dsn: str | None = None


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn_id: str
    previous_turn_id: str | None
    status: TurnStatus
    user_message: str
    runtime: RuntimeBinding
    terminal_reason: str | None
    lease_owner: str | None
    lease_expires_at: float | None
    created_at: float
    updated_at: float


class TurnStore:
    """SQLite persistence for independent and linked Agent Turns."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = None if path is None else Path(path)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            ":memory:" if self._path is None else str(self._path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        if self._path is not None:
            self._connection.execute("PRAGMA journal_mode = WAL")
        try:
            self._create_schema()
        except Exception:
            self._connection.close()
            raise

    @property
    def path(self) -> Path | None:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def begin_turn(
        self,
        message: str,
        runtime: RuntimeBinding,
        *,
        previous_turn_id: str | None = None,
        turn_id: str | None = None,
        lease_owner: str | None = None,
        lease_seconds: float = 300.0,
    ) -> TurnRecord:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Turn message must be a non-empty string")
        if not isinstance(runtime, RuntimeBinding):
            raise TypeError("runtime must be RuntimeBinding")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        effective_turn_id = _turn_id_text(turn_id)
        effective_previous_id = None if previous_turn_id is None else _turn_id_text(previous_turn_id)
        now = time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if effective_previous_id is not None:
                    previous = self._connection.execute(
                        """
                        SELECT status, runtime_json
                        FROM agent_turns
                        WHERE turn_id = ?
                        """,
                        (effective_previous_id,),
                    ).fetchone()
                    if previous is None:
                        raise TurnNotFoundError(f"Turn not found: {effective_previous_id}")
                    previous_status = TurnStatus(str(previous["status"]))
                    if previous_status not in {TurnStatus.COMPLETED, TurnStatus.FAILED}:
                        raise TurnStateError(
                            f"Turn {effective_previous_id} is {previous_status.value}; "
                            "only terminal Turns can receive a follow-up"
                        )
                    previous_runtime = RuntimeBinding.model_validate_json(str(previous["runtime_json"]))
                    if previous_runtime != runtime:
                        raise TurnStateError(
                            f"Turn {effective_previous_id} runtime does not match the follow-up runtime"
                        )
                lease_expires_at = None if lease_owner is None else now + lease_seconds
                self._connection.execute(
                    """
                    INSERT INTO agent_turns (
                        turn_id, previous_turn_id, status, user_message,
                        runtime_json, lease_owner, lease_expires_at,
                        terminal_reason, next_durable_ordinal,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 1, ?, ?)
                    """,
                    (
                        effective_turn_id,
                        effective_previous_id,
                        TurnStatus.RUNNING.value,
                        message,
                        runtime.model_dump_json(),
                        lease_owner,
                        lease_expires_at,
                        now,
                        now,
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO agent_turn_messages (
                        turn_id, message_index, payload_json
                    ) VALUES (?, 0, ?)
                    """,
                    (
                        effective_turn_id,
                        _message_json(ModelMessage(role="user", content=message)),
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO agent_turn_lifecycle (
                        turn_id, durable_ordinal, event_type, status,
                        reason, timestamp_ms
                    ) VALUES (?, 0, ?, ?, NULL, ?)
                    """,
                    (
                        effective_turn_id,
                        EventType.TURN_STARTED.value,
                        TurnStatus.RUNNING.value,
                        _epoch_ms(now),
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_turn(effective_turn_id)

    def get_turn(self, turn_id: str) -> TurnRecord:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT turn_id, previous_turn_id, status, user_message,
                       runtime_json, terminal_reason, lease_owner, lease_expires_at,
                       created_at, updated_at
                FROM agent_turns
                WHERE turn_id = ?
                """,
                (turn_id,),
            ).fetchone()
        if row is None:
            raise TurnNotFoundError(f"Turn not found: {turn_id}")
        return _turn_record(row)

    def list_turns(
        self,
        *,
        workspace_path: Path | str | None = None,
        limit: int = 20,
    ) -> tuple[TurnRecord, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        predicates: list[str] = []
        parameters: list[object] = []
        if workspace_path is not None:
            predicates.append("json_extract(runtime_json, '$.workspace_path') = ?")
            parameters.append(str(Path(workspace_path).expanduser().resolve()))
        where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT turn_id, previous_turn_id, status, user_message,
                       runtime_json, terminal_reason, lease_owner, lease_expires_at,
                       created_at, updated_at
                FROM agent_turns
                {where}
                ORDER BY updated_at DESC, created_at DESC, turn_id DESC
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        return tuple(_turn_record(row) for row in rows)

    def latest_turn(
        self,
        *,
        workspace_path: Path | str | None = None,
    ) -> TurnRecord | None:
        predicates = ["status IN (?, ?)"]
        parameters: list[object] = [TurnStatus.COMPLETED.value, TurnStatus.FAILED.value]
        if workspace_path is not None:
            predicates.append("json_extract(runtime_json, '$.workspace_path') = ?")
            parameters.append(str(Path(workspace_path).expanduser().resolve()))
        with self._lock:
            row = self._connection.execute(
                f"""
                SELECT turn_id, previous_turn_id, status, user_message,
                       runtime_json, terminal_reason, lease_owner, lease_expires_at,
                       created_at, updated_at
                FROM agent_turns
                WHERE {' AND '.join(predicates)}
                ORDER BY updated_at DESC, created_at DESC, turn_id DESC
                LIMIT 1
                """,
                tuple(parameters),
            ).fetchone()
        return None if row is None else _turn_record(row)

    def latest_resumable_turn(
        self,
        *,
        workspace_path: Path | str | None = None,
        now: float | None = None,
    ) -> TurnRecord | None:
        checked_at = time.time() if now is None else now
        predicates = ["status IN (?, ?)"]
        parameters: list[object] = [TurnStatus.PAUSED.value, TurnStatus.INTERRUPTED.value]
        if workspace_path is not None:
            predicates.append("json_extract(runtime_json, '$.workspace_path') = ?")
            parameters.append(str(Path(workspace_path).expanduser().resolve()))
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    """
                    UPDATE agent_turns
                    SET status = ?, lease_owner = NULL,
                        lease_expires_at = NULL, updated_at = ?
                    WHERE status = ?
                      AND (
                          lease_owner IS NULL
                          OR lease_expires_at IS NULL
                          OR lease_expires_at <= ?
                      )
                    """,
                    (
                        TurnStatus.INTERRUPTED.value,
                        checked_at,
                        TurnStatus.RUNNING.value,
                        checked_at,
                    ),
                )
                row = self._connection.execute(
                    f"""
                    SELECT turn_id, previous_turn_id, status, user_message,
                           runtime_json, terminal_reason, lease_owner, lease_expires_at,
                           created_at, updated_at
                    FROM agent_turns
                    WHERE {' AND '.join(predicates)}
                    ORDER BY updated_at DESC, created_at DESC, turn_id DESC
                    LIMIT 1
                    """,
                    tuple(parameters),
                ).fetchone()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return None if row is None else _turn_record(row)

    def history_before_turn(self, turn_id: str) -> tuple[ModelMessage, ...]:
        turn = self.get_turn(turn_id)
        if turn.previous_turn_id is None:
            return ()
        return self.history_through(turn.previous_turn_id)

    def history_through(self, turn_id: str) -> tuple[ModelMessage, ...]:
        lineage = self._lineage(turn_id)
        messages: list[ModelMessage] = []
        with self._lock:
            for item in lineage:
                rows = self._connection.execute(
                    """
                    SELECT payload_json
                    FROM agent_turn_messages
                    WHERE turn_id = ?
                    ORDER BY message_index
                    """,
                    (item.turn_id,),
                ).fetchall()
                messages.extend(_message_from_json(str(row["payload_json"])) for row in rows)
        return tuple(messages)

    def turn_history(self, turn_id: str) -> tuple[ModelMessage, ...]:
        self.get_turn(turn_id)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload_json
                FROM agent_turn_messages
                WHERE turn_id = ?
                ORDER BY message_index
                """,
                (turn_id,),
            ).fetchall()
        return tuple(_message_from_json(str(row["payload_json"])) for row in rows)

    def sync_turn_messages(
        self,
        turn_id: str,
        messages: tuple[ModelMessage, ...] | list[ModelMessage],
    ) -> None:
        payloads = tuple(_message_json(message) for message in messages)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if self._connection.execute(
                    "SELECT 1 FROM agent_turns WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone() is None:
                    raise TurnNotFoundError(f"Turn not found: {turn_id}")
                existing_rows = self._connection.execute(
                    """
                    SELECT message_index, payload_json
                    FROM agent_turn_messages
                    WHERE turn_id = ?
                    ORDER BY message_index
                    """,
                    (turn_id,),
                ).fetchall()
                existing = tuple(str(row["payload_json"]) for row in existing_rows)
                if len(payloads) < len(existing) or payloads[: len(existing)] != existing:
                    raise RuntimeError(f"canonical history conflict for Turn {turn_id}")
                for index in range(len(existing), len(payloads)):
                    self._connection.execute(
                        """
                        INSERT INTO agent_turn_messages (
                            turn_id, message_index, payload_json
                        ) VALUES (?, ?, ?)
                        """,
                        (turn_id, index, payloads[index]),
                    )
                self._connection.execute(
                    "UPDATE agent_turns SET updated_at = ? WHERE turn_id = ?",
                    (time.time(), turn_id),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def commit_completed_item(
        self,
        event: StreamEvent,
        *,
        message: ModelMessage | None = None,
        started_at_ms: int | None = None,
    ) -> int:
        """Atomically persist one completed item and its model-message projection."""

        if event.type is not EventType.ITEM_COMPLETED:
            raise ValueError("event must be item_completed")
        if event.item_id is None or event.item_kind is None or event.status is None:
            raise ValueError("completed item identity and status are required")
        payload_json = _json_text(event.data)
        message_json = None if message is None else _message_json(message)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                turn = self._connection.execute(
                    """
                    SELECT next_durable_ordinal
                    FROM agent_turns
                    WHERE turn_id = ?
                    """,
                    (event.turn_id,),
                ).fetchone()
                if turn is None:
                    raise TurnNotFoundError(f"Turn not found: {event.turn_id}")
                existing = self._connection.execute(
                    """
                    SELECT durable_ordinal, item_kind, status, iteration,
                           payload_json, error, parent_item_id,
                           started_at_ms, message_payload_json
                    FROM agent_turn_items
                    WHERE turn_id = ? AND item_id = ?
                    """,
                    (event.turn_id, event.item_id),
                ).fetchone()
                if existing is not None:
                    expected = (
                        event.item_kind.value,
                        event.status.value,
                        event.iteration,
                        payload_json,
                        event.error,
                        event.parent_item_id,
                        started_at_ms,
                        message_json,
                    )
                    actual = (
                        str(existing["item_kind"]),
                        str(existing["status"]),
                        int(existing["iteration"]),
                        str(existing["payload_json"]),
                        cast(str | None, existing["error"]),
                        cast(str | None, existing["parent_item_id"]),
                        cast(int | None, existing["started_at_ms"]),
                        cast(str | None, existing["message_payload_json"]),
                    )
                    if actual != expected:
                        raise RuntimeError(
                            f"completed item conflict for Turn {event.turn_id}: {event.item_id}"
                        )
                    self._connection.commit()
                    return int(existing["durable_ordinal"])

                durable_ordinal = int(turn["next_durable_ordinal"])
                self._connection.execute(
                    """
                    INSERT INTO agent_turn_items (
                        turn_id, item_id, durable_ordinal, item_kind, status,
                        iteration, payload_json, error, parent_item_id,
                        started_at_ms, completed_at_ms, message_payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.turn_id,
                        event.item_id,
                        durable_ordinal,
                        event.item_kind.value,
                        event.status.value,
                        event.iteration,
                        payload_json,
                        event.error,
                        event.parent_item_id,
                        started_at_ms,
                        event.timestamp_ms,
                        message_json,
                    ),
                )
                if message_json is not None:
                    message_index = int(
                        self._connection.execute(
                            """
                            SELECT COUNT(*) AS count
                            FROM agent_turn_messages
                            WHERE turn_id = ?
                            """,
                            (event.turn_id,),
                        ).fetchone()["count"]
                    )
                    self._connection.execute(
                        """
                        INSERT INTO agent_turn_messages (
                            turn_id, message_index, payload_json
                        ) VALUES (?, ?, ?)
                        """,
                        (event.turn_id, message_index, message_json),
                    )
                self._connection.execute(
                    """
                    UPDATE agent_turns
                    SET next_durable_ordinal = ?, updated_at = ?
                    WHERE turn_id = ?
                    """,
                    (durable_ordinal + 1, time.time(), event.turn_id),
                )
                self._connection.commit()
                return durable_ordinal
            except Exception:
                self._connection.rollback()
                raise

    def replay_turn_events(self, turn_id: str) -> tuple[DurableTurnEvent, ...]:
        """Project durable lifecycle and completed items in one Turn-local order."""

        self.get_turn(turn_id)
        with self._lock:
            lifecycle_rows = self._connection.execute(
                """
                SELECT durable_ordinal, event_type, status, reason, timestamp_ms
                FROM agent_turn_lifecycle
                WHERE turn_id = ?
                """,
                (turn_id,),
            ).fetchall()
            item_rows = self._connection.execute(
                """
                SELECT item_id, durable_ordinal, item_kind, status, iteration,
                       payload_json, error, parent_item_id, completed_at_ms
                FROM agent_turn_items
                WHERE turn_id = ?
                """,
                (turn_id,),
            ).fetchall()
        records: list[DurableTurnEvent] = []
        for row in lifecycle_rows:
            data: dict[str, Any] = {"status": str(row["status"])}
            reason = cast(str | None, row["reason"])
            if reason is not None:
                data["reason"] = reason
            records.append(
                DurableTurnEvent(
                    durable_ordinal=int(row["durable_ordinal"]),
                    event=StreamEvent(
                        type=EventType(str(row["event_type"])),
                        turn_id=turn_id,
                        timestamp_ms=int(row["timestamp_ms"]),
                        data=cast(dict[str, Any], data),
                    ),
                )
            )
        for row in item_rows:
            payload = json.loads(str(row["payload_json"]))
            if not isinstance(payload, dict):
                raise RuntimeError("durable item payload must be an object")
            records.append(
                DurableTurnEvent(
                    durable_ordinal=int(row["durable_ordinal"]),
                    event=StreamEvent(
                        type=EventType.ITEM_COMPLETED,
                        turn_id=turn_id,
                        item_id=str(row["item_id"]),
                        item_kind=TurnItemKind(str(row["item_kind"])),
                        status=ItemStatus(str(row["status"])),
                        iteration=int(row["iteration"]),
                        timestamp_ms=int(row["completed_at_ms"]),
                        data=cast(dict[str, Any], payload),
                        error=cast(str | None, row["error"]),
                        parent_item_id=cast(str | None, row["parent_item_id"]),
                    ),
                )
            )
        records.sort(key=lambda record: record.durable_ordinal)
        if len({record.durable_ordinal for record in records}) != len(records):
            raise RuntimeError(f"durable history ordinal conflict for Turn {turn_id}")
        return tuple(records)

    def mark_paused(self, turn_id: str, *, reason: str = "paused") -> TurnRecord:
        return self._mark_recoverable(turn_id, TurnStatus.PAUSED, reason=reason)

    def mark_interrupted(
        self,
        turn_id: str,
        *,
        reason: str = "interrupted",
    ) -> TurnRecord:
        return self._mark_recoverable(turn_id, TurnStatus.INTERRUPTED, reason=reason)

    def mark_aborted(
        self,
        turn_id: str,
        *,
        reason: str = "cancelled",
    ) -> TurnRecord:
        """Atomically record cancellation intent and a final aborted Turn."""

        now = time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT status FROM agent_turns WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if row is None:
                    raise TurnNotFoundError(f"Turn not found: {turn_id}")
                current = TurnStatus(str(row["status"]))
                if current is not TurnStatus.RUNNING:
                    raise TurnStateError(
                        f"Turn {turn_id} is {current.value}; expected running"
                    )
                self._append_lifecycle_unlocked(
                    turn_id,
                    event_type=EventType.TURN_CANCELLATION_REQUESTED,
                    status="cancelling",
                    reason=reason,
                    timestamp=now,
                )
                self._connection.execute(
                    """
                    UPDATE agent_turns
                    SET status = ?, lease_owner = NULL,
                        lease_expires_at = NULL, terminal_reason = ?,
                        updated_at = ?
                    WHERE turn_id = ?
                    """,
                    (TurnStatus.FAILED.value, reason, now, turn_id),
                )
                self._append_lifecycle_unlocked(
                    turn_id,
                    event_type=EventType.TURN_ABORTED,
                    status="aborted",
                    reason=reason,
                    timestamp=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_turn(turn_id)

    def claim_for_resume(
        self,
        turn_id: str,
        *,
        lease_owner: str,
        lease_seconds: float = 300.0,
        now: float | None = None,
    ) -> TurnRecord:
        if not lease_owner:
            raise ValueError("lease_owner must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        claimed_at = time.time() if now is None else now
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT status FROM agent_turns WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if row is None:
                    raise TurnNotFoundError(f"Turn not found: {turn_id}")
                current = TurnStatus(str(row["status"]))
                if current not in {TurnStatus.PAUSED, TurnStatus.INTERRUPTED}:
                    raise TurnStateError(f"Turn {turn_id} is {current.value} and cannot resume")
                self._connection.execute(
                    """
                    UPDATE agent_turns
                    SET status = ?, lease_owner = ?, lease_expires_at = ?,
                        updated_at = ?
                    WHERE turn_id = ?
                    """,
                    (
                        TurnStatus.RUNNING.value,
                        lease_owner,
                        claimed_at + lease_seconds,
                        claimed_at,
                        turn_id,
                    ),
                )
                self._append_lifecycle_unlocked(
                    turn_id,
                    event_type=EventType.TURN_RESUMED,
                    status=TurnStatus.RUNNING.value,
                    reason=None,
                    timestamp=claimed_at,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_turn(turn_id)

    def prepare_turn_for_resume(
        self,
        turn_id: str,
        *,
        now: float | None = None,
    ) -> TurnRecord:
        checked_at = time.time() if now is None else now
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT status, lease_owner, lease_expires_at
                    FROM agent_turns
                    WHERE turn_id = ?
                    """,
                    (turn_id,),
                ).fetchone()
                if row is None:
                    raise TurnNotFoundError(f"Turn not found: {turn_id}")
                status = TurnStatus(str(row["status"]))
                if status in {TurnStatus.COMPLETED, TurnStatus.FAILED}:
                    raise TurnStateError(f"Turn {turn_id} is {status.value} and cannot resume")
                if status is TurnStatus.RUNNING:
                    lease_owner = cast(str | None, row["lease_owner"])
                    lease_expires_at = cast(float | None, row["lease_expires_at"])
                    if lease_owner is not None and lease_expires_at is not None and lease_expires_at > checked_at:
                        raise TurnStateError(f"Turn {turn_id} is still running under an active lease")
                    self._connection.execute(
                        """
                        UPDATE agent_turns
                        SET status = ?, lease_owner = NULL,
                            lease_expires_at = NULL, updated_at = ?
                        WHERE turn_id = ?
                        """,
                        (TurnStatus.INTERRUPTED.value, checked_at, turn_id),
                    )
                elif status not in {TurnStatus.PAUSED, TurnStatus.INTERRUPTED}:
                    raise TurnStateError(f"Turn {turn_id} is {status.value} and cannot resume")
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_turn(turn_id)

    def renew_lease(
        self,
        turn_id: str,
        *,
        lease_owner: str,
        lease_seconds: float = 300.0,
        now: float | None = None,
    ) -> TurnRecord:
        if not lease_owner:
            raise ValueError("lease_owner must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        renewed_at = time.time() if now is None else now
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT status, lease_owner FROM agent_turns WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if row is None:
                    raise TurnNotFoundError(f"Turn not found: {turn_id}")
                status = TurnStatus(str(row["status"]))
                if status is not TurnStatus.RUNNING:
                    raise TurnStateError(f"Turn {turn_id} is {status.value}; expected running")
                if cast(str | None, row["lease_owner"]) != lease_owner:
                    raise TurnStateError(f"Turn {turn_id} is owned by another worker")
                self._connection.execute(
                    """
                    UPDATE agent_turns
                    SET lease_expires_at = ?, updated_at = ?
                    WHERE turn_id = ?
                    """,
                    (renewed_at + lease_seconds, renewed_at, turn_id),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_turn(turn_id)

    def mark_terminal(
        self,
        turn_id: str,
        status: TurnStatus,
        *,
        reason: str | None = None,
    ) -> TurnRecord:
        if status not in {TurnStatus.COMPLETED, TurnStatus.FAILED}:
            raise ValueError("terminal status must be completed or failed")
        now = time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT status FROM agent_turns WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if row is None:
                    raise TurnNotFoundError(f"Turn not found: {turn_id}")
                current = TurnStatus(str(row["status"]))
                if current in {TurnStatus.COMPLETED, TurnStatus.FAILED}:
                    raise TurnStateError(f"Turn {turn_id} is already {current.value}")
                self._connection.execute(
                    """
                    UPDATE agent_turns
                    SET status = ?, lease_owner = NULL,
                        lease_expires_at = NULL, terminal_reason = ?, updated_at = ?
                    WHERE turn_id = ?
                    """,
                    (status.value, reason, now, turn_id),
                )
                self._append_lifecycle_unlocked(
                    turn_id,
                    event_type=EventType.TURN_COMPLETED,
                    status=status.value,
                    reason=reason,
                    timestamp=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_turn(turn_id)

    def _mark_recoverable(
        self,
        turn_id: str,
        status: TurnStatus,
        *,
        reason: str,
    ) -> TurnRecord:
        if status not in {TurnStatus.PAUSED, TurnStatus.INTERRUPTED}:
            raise ValueError("recoverable status must be paused or interrupted")
        now = time.time()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT status FROM agent_turns WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if row is None:
                    raise TurnNotFoundError(f"Turn not found: {turn_id}")
                current = TurnStatus(str(row["status"]))
                if current is not TurnStatus.RUNNING:
                    raise TurnStateError(f"Turn {turn_id} is {current.value}; expected running")
                self._connection.execute(
                    """
                    UPDATE agent_turns
                    SET status = ?, lease_owner = NULL,
                        lease_expires_at = NULL, terminal_reason = ?, updated_at = ?
                    WHERE turn_id = ?
                    """,
                    (
                        status.value,
                        reason if status is TurnStatus.INTERRUPTED else None,
                        now,
                        turn_id,
                    ),
                )
                self._append_lifecycle_unlocked(
                    turn_id,
                    event_type=EventType.TURN_PAUSED,
                    status=status.value,
                    reason=reason,
                    timestamp=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_turn(turn_id)

    def _append_lifecycle_unlocked(
        self,
        turn_id: str,
        *,
        event_type: EventType,
        status: str,
        reason: str | None,
        timestamp: float,
    ) -> int:
        row = self._connection.execute(
            "SELECT next_durable_ordinal FROM agent_turns WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        if row is None:
            raise TurnNotFoundError(f"Turn not found: {turn_id}")
        durable_ordinal = int(row["next_durable_ordinal"])
        self._connection.execute(
            """
            INSERT INTO agent_turn_lifecycle (
                turn_id, durable_ordinal, event_type, status, reason, timestamp_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                durable_ordinal,
                event_type.value,
                status,
                reason,
                _epoch_ms(timestamp),
            ),
        )
        self._connection.execute(
            """
            UPDATE agent_turns
            SET next_durable_ordinal = ?
            WHERE turn_id = ?
            """,
            (durable_ordinal + 1, turn_id),
        )
        return durable_ordinal

    def _lineage(self, turn_id: str) -> tuple[TurnRecord, ...]:
        lineage: list[TurnRecord] = []
        seen: set[str] = set()
        current_id: str | None = turn_id
        while current_id is not None:
            if current_id in seen:
                raise RuntimeError(f"Turn lineage contains a cycle at {current_id}")
            seen.add(current_id)
            current = self.get_turn(current_id)
            lineage.append(current)
            current_id = current.previous_turn_id
        lineage.reverse()
        return tuple(lineage)

    def _create_schema(self) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                tables = {
                    str(row["name"])
                    for row in self._connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                if "agent_turns" in tables:
                    turn_columns = {
                        str(row["name"])
                        for row in self._connection.execute("PRAGMA table_info(agent_turns)").fetchall()
                    }
                    if "session_id" in turn_columns:
                        self._migrate_session_schema()
                    elif "previous_turn_id" not in turn_columns:
                        raise RuntimeError("Unsupported agent_turns schema")
                else:
                    self._create_turn_tables()
                self._migrate_runtime_bindings()
                self._migrate_streaming_schema()
                self._connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS agent_turns_listing_idx
                    ON agent_turns (updated_at DESC, created_at DESC, turn_id DESC)
                    """
                )
                self._connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS agent_turns_previous_idx
                    ON agent_turns (previous_turn_id)
                    """
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def _create_turn_tables(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_turns (
                turn_id TEXT PRIMARY KEY,
                previous_turn_id TEXT,
                status TEXT NOT NULL
                    CHECK (status IN ('running', 'paused', 'interrupted', 'completed', 'failed')),
                user_message TEXT NOT NULL,
                runtime_json TEXT NOT NULL,
                lease_owner TEXT,
                lease_expires_at REAL,
                terminal_reason TEXT,
                next_durable_ordinal INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(previous_turn_id) REFERENCES agent_turns(turn_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_turn_messages (
                turn_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(turn_id, message_index),
                FOREIGN KEY(turn_id) REFERENCES agent_turns(turn_id) ON DELETE CASCADE
            )
            """
        )

    def _migrate_streaming_schema(self) -> None:
        turn_columns = {
            str(row["name"])
            for row in self._connection.execute(
                "PRAGMA table_info(agent_turns)"
            ).fetchall()
        }
        if "terminal_reason" not in turn_columns:
            self._connection.execute(
                "ALTER TABLE agent_turns ADD COLUMN terminal_reason TEXT"
            )
        if "next_durable_ordinal" not in turn_columns:
            self._connection.execute(
                """
                ALTER TABLE agent_turns
                ADD COLUMN next_durable_ordinal INTEGER NOT NULL DEFAULT 0
                """
            )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_turn_lifecycle (
                turn_id TEXT NOT NULL,
                durable_ordinal INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                timestamp_ms INTEGER NOT NULL,
                PRIMARY KEY(turn_id, durable_ordinal),
                FOREIGN KEY(turn_id) REFERENCES agent_turns(turn_id) ON DELETE CASCADE
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_turn_items (
                turn_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                durable_ordinal INTEGER NOT NULL,
                item_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                error TEXT,
                parent_item_id TEXT,
                started_at_ms INTEGER,
                completed_at_ms INTEGER NOT NULL,
                message_payload_json TEXT,
                PRIMARY KEY(turn_id, item_id),
                UNIQUE(turn_id, durable_ordinal),
                FOREIGN KEY(turn_id) REFERENCES agent_turns(turn_id) ON DELETE CASCADE
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS agent_turn_items_order_idx
            ON agent_turn_items (turn_id, durable_ordinal)
            """
        )

        rows = self._connection.execute(
            """
            SELECT turn_id, status, created_at, updated_at
            FROM agent_turns
            WHERE NOT EXISTS (
                SELECT 1
                FROM agent_turn_lifecycle
                WHERE agent_turn_lifecycle.turn_id = agent_turns.turn_id
            )
            ORDER BY created_at, turn_id
            """
        ).fetchall()
        for row in rows:
            self._backfill_legacy_turn_unlocked(row)

    def _backfill_legacy_turn_unlocked(self, row: sqlite3.Row) -> None:
        turn_id = str(row["turn_id"])
        ordinal = 0
        self._connection.execute(
            """
            INSERT INTO agent_turn_lifecycle (
                turn_id, durable_ordinal, event_type, status, reason, timestamp_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                ordinal,
                EventType.TURN_STARTED.value,
                TurnStatus.RUNNING.value,
                "legacy_projection",
                _epoch_ms(float(row["created_at"])),
            ),
        )
        ordinal += 1
        messages = self._connection.execute(
            """
            SELECT message_index, payload_json
            FROM agent_turn_messages
            WHERE turn_id = ?
            ORDER BY message_index
            """,
            (turn_id,),
        ).fetchall()
        for message in messages:
            payload_raw = str(message["payload_json"])
            payload = json.loads(payload_raw)
            self._connection.execute(
                """
                INSERT INTO agent_turn_items (
                    turn_id, item_id, durable_ordinal, item_kind, status,
                    iteration, payload_json, error, parent_item_id,
                    started_at_ms, completed_at_ms, message_payload_json
                ) VALUES (?, ?, ?, ?, ?, 0, ?, NULL, NULL, NULL, ?, ?)
                """,
                (
                    turn_id,
                    f"legacy:{turn_id}:{int(message['message_index'])}",
                    ordinal,
                    TurnItemKind.LEGACY_MESSAGE.value,
                    ItemStatus.SUCCESS.value,
                    _json_text(
                        {
                            "legacy_projection": True,
                            "message": payload,
                        }
                    ),
                    _epoch_ms(float(row["updated_at"])),
                    payload_raw,
                ),
            )
            ordinal += 1
        status = TurnStatus(str(row["status"]))
        if status is not TurnStatus.RUNNING:
            event_type = (
                EventType.TURN_PAUSED
                if status is TurnStatus.PAUSED
                else EventType.TURN_ABORTED
                if status is TurnStatus.INTERRUPTED
                else EventType.TURN_COMPLETED
            )
            self._connection.execute(
                """
                INSERT INTO agent_turn_lifecycle (
                    turn_id, durable_ordinal, event_type, status, reason, timestamp_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    ordinal,
                    event_type.value,
                    status.value,
                    "legacy_projection",
                    _epoch_ms(float(row["updated_at"])),
                ),
            )
            ordinal += 1
        self._connection.execute(
            """
            UPDATE agent_turns
            SET next_durable_ordinal = ?
            WHERE turn_id = ?
            """,
            (ordinal, turn_id),
        )

    def _migrate_session_schema(self) -> None:
        session_columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(agent_sessions)").fetchall()
        }
        kind_expression = "kind" if "kind" in session_columns else "'conversation'"
        session_rows = self._connection.execute(
            f"SELECT session_id, {kind_expression} AS kind FROM agent_sessions"
        ).fetchall()
        kinds = {str(row["session_id"]): str(row["kind"]) for row in session_rows}
        turn_rows = self._connection.execute(
            """
            SELECT turn_id, session_id, ordinal, status, user_message,
                   runtime_json, lease_owner, lease_expires_at,
                   created_at, updated_at
            FROM agent_turns
            ORDER BY session_id, ordinal
            """
        ).fetchall()
        event_rows: list[sqlite3.Row] = []
        if self._table_exists("agent_session_events"):
            event_rows = self._connection.execute(
                """
                SELECT turn_id, event_index, payload_json
                FROM agent_session_events
                ORDER BY turn_id, event_index
                """
            ).fetchall()

        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in turn_rows:
            grouped.setdefault(str(row["session_id"]), []).append(row)
        previous_by_turn: dict[str, str | None] = {}
        for session_id, rows in grouped.items():
            if kinds.get(session_id) == "one_shot":
                if len(rows) != 1:
                    raise RuntimeError(
                        f"Cannot migrate one_shot Session {session_id}: expected one Turn, found {len(rows)}"
                    )
                previous_by_turn[str(rows[0]["turn_id"])] = None
                continue
            previous: str | None = None
            for row in rows:
                turn_id = str(row["turn_id"])
                previous_by_turn[turn_id] = previous
                previous = turn_id

        self._connection.execute(
            """
            CREATE TABLE agent_turns_next (
                turn_id TEXT PRIMARY KEY,
                previous_turn_id TEXT,
                status TEXT NOT NULL
                    CHECK (status IN ('running', 'paused', 'interrupted', 'completed', 'failed')),
                user_message TEXT NOT NULL,
                runtime_json TEXT NOT NULL,
                lease_owner TEXT,
                lease_expires_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(previous_turn_id) REFERENCES agent_turns_next(turn_id)
            )
            """
        )
        for row in turn_rows:
            turn_id = str(row["turn_id"])
            self._connection.execute(
                """
                INSERT INTO agent_turns_next (
                    turn_id, previous_turn_id, status, user_message,
                    runtime_json, lease_owner, lease_expires_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    previous_by_turn[turn_id],
                    str(row["status"]),
                    str(row["user_message"]),
                    str(row["runtime_json"]),
                    cast(str | None, row["lease_owner"]),
                    cast(float | None, row["lease_expires_at"]),
                    float(row["created_at"]),
                    float(row["updated_at"]),
                ),
            )
        if self._table_exists("agent_session_events"):
            self._connection.execute("DROP TABLE agent_session_events")
        self._connection.execute("DROP TABLE agent_turns")
        self._connection.execute("DROP TABLE agent_sessions")
        self._connection.execute("ALTER TABLE agent_turns_next RENAME TO agent_turns")
        self._connection.execute(
            """
            CREATE TABLE agent_turn_messages (
                turn_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(turn_id, message_index),
                FOREIGN KEY(turn_id) REFERENCES agent_turns(turn_id) ON DELETE CASCADE
            )
            """
        )
        turns_with_events: set[str] = set()
        for row in event_rows:
            turn_id = str(row["turn_id"])
            turns_with_events.add(turn_id)
            self._connection.execute(
                """
                INSERT INTO agent_turn_messages (
                    turn_id, message_index, payload_json
                ) VALUES (?, ?, ?)
                """,
                (turn_id, int(row["event_index"]), str(row["payload_json"])),
            )
        for row in turn_rows:
            turn_id = str(row["turn_id"])
            if turn_id not in turns_with_events:
                self._connection.execute(
                    """
                    INSERT INTO agent_turn_messages (
                        turn_id, message_index, payload_json
                    ) VALUES (?, 0, ?)
                    """,
                    (
                        turn_id,
                        _message_json(ModelMessage(role="user", content=str(row["user_message"]))),
                    ),
                )

    def _migrate_runtime_bindings(self) -> None:
        rows = self._connection.execute(
            "SELECT turn_id, runtime_json FROM agent_turns"
        ).fetchall()
        for row in rows:
            turn_id = str(row["turn_id"])
            raw = str(row["runtime_json"])
            canonical = _canonical_runtime_json(raw, owner_id=turn_id)
            if raw != canonical:
                self._connection.execute(
                    "UPDATE agent_turns SET runtime_json = ? WHERE turn_id = ?",
                    (canonical, turn_id),
                )

    def _table_exists(self, name: str) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (name,),
            ).fetchone()
            is not None
        )


def _canonical_runtime_json(raw: str, *, owner_id: str) -> str:
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        binding = _runtime_binding_from_payload(payload)
    except Exception as exc:
        detail = _runtime_binding_error_detail(exc)
        raise ValueError(f"Invalid RuntimeBinding for Turn {owner_id}: {detail}") from None
    return binding.model_dump_json()


def _runtime_binding_from_payload(payload: dict[str, Any]) -> RuntimeBinding:
    if "schema_version" in payload:
        if payload["schema_version"] != 2:
            raise ValueError("unsupported RuntimeBinding schema_version")
        return RuntimeBinding.model_validate(payload)
    legacy = _LegacyRuntimeBinding.model_validate(payload)
    vector_backend = _canonical_vector_backend(legacy.vector_backend)
    knowledge = None
    if legacy.knowledge:
        knowledge = RAGKnowledgeConfig(
            storage_root=Path(legacy.rag_storage_root),
            embedding_model=legacy.embedding_model_alias,
            reranker_model=legacy.reranker_model_alias,
            vector_backend=vector_backend,
            vector_namespace=legacy.vector_namespace,
            vector_collection_prefix=legacy.vector_collection_prefix,
        )
    return RuntimeBinding(
        model_alias=legacy.model_alias,
        workspace_path=legacy.workspace_path,
        knowledge=knowledge,
    )


def _canonical_vector_backend(value: str) -> Literal["milvus", "sqlite"]:
    if value == "in_memory":
        return "sqlite"
    if value in {"milvus", "sqlite"}:
        return cast(Literal["milvus", "sqlite"], value)
    raise ValueError("unsupported legacy vector_backend")


def _runtime_binding_error_detail(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        details: list[str] = []
        for error in exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        ):
            location = ".".join(str(part) for part in error["loc"])
            details.append(
                f"{location or '<root>'}: {error['msg']} [{error['type']}]"
            )
        return "; ".join(details) or "validation failed"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid JSON"
    if isinstance(exc, ValueError):
        return str(exc)
    return type(exc).__name__[:120]


def _turn_id_text(value: str | None) -> str:
    if value is None:
        return str(uuid4())
    if not isinstance(value, str) or not value.strip():
        raise ValueError("turn_id must be a non-empty string")
    return value


def _turn_record(row: sqlite3.Row) -> TurnRecord:
    return TurnRecord(
        turn_id=str(row["turn_id"]),
        previous_turn_id=cast(str | None, row["previous_turn_id"]),
        status=TurnStatus(str(row["status"])),
        user_message=str(row["user_message"]),
        runtime=RuntimeBinding.model_validate_json(str(row["runtime_json"])),
        terminal_reason=cast(str | None, row["terminal_reason"]),
        lease_owner=cast(str | None, row["lease_owner"]),
        lease_expires_at=cast(float | None, row["lease_expires_at"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _message_json(message: ModelMessage) -> str:
    return canonical_json_text(
        cast(Any, model_message_payload(snapshot_model_message(message)))
    )


def _json_text(value: object) -> str:
    return canonical_json_text(cast(Any, value))


def _epoch_ms(value: float) -> int:
    return int(value * 1000)


def _message_from_json(raw: str) -> ModelMessage:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("canonical history message must be an object")
    tool_calls_raw = payload.get("tool_calls", [])
    if not isinstance(tool_calls_raw, list):
        raise RuntimeError("canonical history tool_calls must be an array")
    tool_calls = tuple(
        ToolCall(
            id=str(item["id"]),
            name=str(item["name"]),
            input=cast(dict[str, Any], item["arguments"]),
        )
        for item in tool_calls_raw
        if isinstance(item, dict)
    )
    return snapshot_model_message(
        ModelMessage(
            role=cast(Any, payload.get("role")),
            content=str(payload.get("content", "")),
            tool_calls=tool_calls,
            tool_call_id=cast(str | None, payload.get("tool_call_id")),
        )
    )


__all__ = [
    "RuntimeBinding",
    "TurnNotFoundError",
    "TurnRecord",
    "TurnStateError",
    "TurnStatus",
    "TurnStore",
]

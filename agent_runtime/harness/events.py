"""Replayable client events derived from the canonical Rollout log."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from agent_runtime.harness.rollout import RolloutRecord, RolloutStore
from agent_runtime.streaming.events import (
    ItemStatus,
    StreamEvent,
    TurnItemKind,
    derive_model_public_item_id,
    derive_operation_public_item_id,
    derive_plan_public_item_id,
    item_completed,
    item_started,
    turn_aborted,
    turn_cancellation_requested,
    turn_completed,
    turn_paused,
    turn_resumed,
    turn_started,
)


_PUBLIC_ITEM_KINDS = frozenset(
    {
        "agent_message",
        "model_response",
        "model_reasoning",
        "model_plan",
        "tool_result",
        "command_execution",
        "plan_state",
    }
)
_SUPPRESSED_ITEM_KINDS = frozenset(
    {
        "user_message",
        "input_file",
        "model_request",
        "tool_call",
        "final_proposal",
        "completion_decision",
        "completion_feedback",
        "context_compaction",
        "context_message",
    }
)
_SUPPRESSED_RECORD_TYPES = frozenset(
    {
        "thread_created",
        "thread_forked",
        "artifact_committed",
        "model_operation_prepared",
        "model_retry_prepared",
        "model_attempt_dispatched",
        "model_attempt_completed",
        "model_attempt_late_response",
        "model_attempt_rejected",
        "model_attempt_unknown",
        "tool_operation_prepared",
        "tool_operation_migrated",
        "tool_operation_status_changed",
        "tool_operation_start_rejected",
        "tool_operation_claim_expired",
        "tool_operation_result_missing",
        "tool_operation_outcome_committed",
        "tool_operation_stale_result",
        "tool_operation_reconciled",
        "tool_operation_result_linked",
        "tool_operation_approval_requested",
        "tool_operation_approval_resolved",
        "tool_operation_superseded",
        "interaction_requested",
        "interaction_resolved",
        "approval_invalidated",
    }
)
_CURSOR_VERSION = 1
_SCHEMA_EPOCH = "harness-canonical-stream-v2"


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    event: StreamEvent
    cursor: str
    record_id: int
    thread_sequence: int

    @property
    def thread_id(self) -> str:
        padding = "=" * (-len(self.cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(self.cursor + padding).decode())
        return str(payload["thread_id"])

    @property
    def turn_id(self) -> str:
        return self.event.turn_id

    @property
    def event_type(self) -> str:
        return self.event.type.value

    @property
    def data(self) -> Mapping[str, Any]:
        return self.event.data


RolloutEvent = ReplayEvent


class RolloutEventReader:
    """Read independent Thread tails or one global record-id tail."""

    def __init__(self, store: RolloutStore) -> None:
        self._store = store

    def read(
        self,
        thread_id: str,
        *,
        after: str | None = None,
    ) -> tuple[ReplayEvent, ...]:
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("thread_id must be non-empty")
        self._store.read_thread(thread_id)
        sequence = 0 if after is None else self._decode(after, thread_id=thread_id)
        all_records = self._store.list_records(thread_id)
        for record in all_records:
            if record.record_type == "item_started":
                self._validate_internal_item_kind(record)
            self._validate_public_item_ids(record)
        projected = self._project_records(all_records, after_thread_sequence=sequence)
        return tuple(
            ReplayEvent(
                event=replace(event, sequence=index),
                cursor=self._encode(
                    thread_id=record.thread_id,
                    sequence=record.thread_sequence,
                ),
                record_id=record.record_id,
                thread_sequence=record.thread_sequence,
            )
            for index, (record, event) in enumerate(projected, start=1)
        )

    def read_global(
        self,
        *,
        after_record_id: int = 0,
    ) -> tuple[ReplayEvent, ...]:
        records = self._store.list_global_records(after_record_id=0)
        thread_ids = tuple(dict.fromkeys(record.thread_id for record in records))
        projected = sorted(
            (
                pair
                for thread_id in thread_ids
                for pair in self._project_records(
                    self._store.list_records(thread_id),
                    after_thread_sequence=0,
                )
                if pair[0].record_id > after_record_id
            ),
            key=lambda pair: pair[0].record_id,
        )
        return tuple(
            ReplayEvent(
                event=replace(event, sequence=index),
                cursor=self._encode(
                    thread_id=record.thread_id,
                    sequence=record.thread_sequence,
                ),
                record_id=record.record_id,
                thread_sequence=record.thread_sequence,
            )
            for index, (record, event) in enumerate(projected, start=1)
        )

    def _project_records(
        self,
        records: tuple[RolloutRecord, ...],
        *,
        after_thread_sequence: int,
    ) -> tuple[tuple[RolloutRecord, StreamEvent], ...]:
        starts = {
            str(record.payload["item_id"]): record
            for record in records
            if record.record_type == "item_started"
            and isinstance(record.payload.get("item_id"), str)
        }
        completed_item_ids = {
            str(record.payload["item_id"])
            for record in records
            if record.record_type == "item_completed"
            and isinstance(record.payload.get("item_id"), str)
        }
        duplicate_answer_ids = _accepted_answer_item_ids(records)
        completed_operations = {
            (
                str(record.payload["operation_id"]),
                int(record.payload["attempt_generation"]),
            )
            for record in records
            if record.record_type == "item_completed"
            and isinstance(record.payload.get("operation_id"), str)
            and isinstance(record.payload.get("attempt_generation"), int)
        }
        projected: list[tuple[RolloutRecord, StreamEvent]] = []
        for record in records:
            events = self._project_record(
                record,
                starts=starts,
                completed_item_ids=completed_item_ids,
                duplicate_answer_ids=duplicate_answer_ids,
                completed_operations=completed_operations,
            )
            if record.thread_sequence > after_thread_sequence:
                projected.extend((record, event) for event in events)
        return tuple(projected)

    def _project_record(
        self,
        record: RolloutRecord,
        *,
        starts: Mapping[str, RolloutRecord],
        completed_item_ids: set[str],
        duplicate_answer_ids: frozenset[str],
        completed_operations: set[tuple[str, int]],
    ) -> tuple[StreamEvent, ...]:
        lifecycle = self._project_lifecycle(record)
        if lifecycle is not None:
            return (lifecycle,)
        item_id = record.payload.get("item_id")
        start = starts.get(str(item_id))
        kind = None if start is None else start.payload.get("kind")
        if record.record_type == "tool_operation_claimed":
            operation_id = record.payload.get("operation_id")
            public_item_id = record.payload.get("public_item_id")
            if (
                record.turn_id is None
                or not isinstance(operation_id, str)
                or not isinstance(public_item_id, str)
            ):
                raise RuntimeError("claimed tool operation is missing public identity")
            generation = record.payload.get("claim_generation")
            if not isinstance(generation, int) or (
                operation_id,
                generation,
            ) not in completed_operations:
                return ()
            operation = self._store.read_tool_operation(operation_id)
            return (
                item_started(
                turn_id=record.turn_id,
                item_id=public_item_id,
                item_kind=(
                    TurnItemKind.COMMAND
                    if operation.tool_name == "run_command"
                    else TurnItemKind.TOOL
                ),
                data={
                    "operation_id": operation_id,
                    "tool_call_id": operation.tool_call_id,
                    "tool_name": operation.tool_name,
                },
                ),
            )
        if record.record_type == "item_started":
            if not isinstance(item_id, str) or item_id not in completed_item_ids:
                return ()
            if item_id in duplicate_answer_ids or kind in _SUPPRESSED_ITEM_KINDS:
                return ()
            if kind in {"tool_result", "command_execution"}:
                return ()
            return (self._project_item_started(record, kind=str(kind)),)
        elif kind == "tool_result" and record.record_type == "item_completed":
            operation_id = record.payload.get("operation_id")
            public_item_id = record.payload.get("public_item_id")
            if isinstance(operation_id, str) and isinstance(public_item_id, str):
                if record.turn_id is None:
                    raise RuntimeError("tool result is missing turn identity")
                operation = self._store.read_tool_operation(operation_id)
                status, error = _tool_completion_status(operation.status, operation.error_code)
                completed = item_completed(
                    turn_id=record.turn_id,
                    item_id=public_item_id,
                    item_kind=(
                        TurnItemKind.COMMAND
                        if operation.tool_name == "run_command"
                        else TurnItemKind.TOOL
                    ),
                    status=status,
                    data={"result": record.payload.get("payload", {})},
                    error=error,
                )
                events = [completed]
                plan_snapshot = record.payload.get("plan_snapshot")
                plan_item_id = record.payload.get("plan_public_item_id")
                if isinstance(plan_snapshot, Mapping) and isinstance(plan_item_id, str):
                    events.append(
                        item_completed(
                            turn_id=record.turn_id,
                            item_id=plan_item_id,
                            item_kind=TurnItemKind.PLAN,
                            status=ItemStatus.SUCCESS,
                            data={"plan": dict(plan_snapshot)},
                        )
                    )
                return tuple(events)
            return ()
        if record.record_type == "item_completed":
            if not isinstance(item_id, str) or start is None:
                raise RuntimeError("completed internal Item has no durable start")
            if item_id in duplicate_answer_ids or kind in _SUPPRESSED_ITEM_KINDS:
                return ()
            if kind == "command_execution":
                return ()
            return (self._project_item_completed(record, kind=str(kind)),)
        if record.record_type in _SUPPRESSED_RECORD_TYPES:
            return ()
        raise RuntimeError(f"unknown durable Rollout record type: {record.record_type!r}")

    @staticmethod
    def _project_lifecycle(record: RolloutRecord) -> StreamEvent | None:
        turn_id = record.turn_id
        if record.record_type == "turn_started":
            if turn_id is None:
                raise RuntimeError("turn_started record is missing turn_id")
            return turn_started(turn_id)
        if record.record_type == "turn_paused":
            if turn_id is None:
                raise RuntimeError("turn_paused record is missing turn_id")
            reason = record.payload.get("reason", "interaction_required")
            return turn_paused(turn_id, reason=str(reason))
        if record.record_type == "turn_resumed":
            if turn_id is None:
                raise RuntimeError("turn_resumed record is missing turn_id")
            return turn_resumed(turn_id)
        if record.record_type == "turn_cancellation_requested":
            if turn_id is None:
                raise RuntimeError("turn_cancellation_requested record is missing turn_id")
            return turn_cancellation_requested(turn_id)
        if record.record_type == "turn_completed":
            if turn_id is None:
                raise RuntimeError("turn_completed record is missing turn_id")
            return turn_completed(turn_id)
        if record.record_type == "turn_failed":
            if turn_id is None:
                raise RuntimeError("turn_failed record is missing turn_id")
            reason = record.payload.get("reason")
            return turn_completed(
                turn_id,
                status="failed",
                reason=None if reason is None else str(reason),
            )
        if record.record_type in {"turn_cancelled", "turn_abandoned"}:
            if turn_id is None:
                raise RuntimeError(f"{record.record_type} record is missing turn_id")
            reason = record.payload.get("reason", record.record_type.removeprefix("turn_"))
            return turn_aborted(turn_id, reason=str(reason))
        if record.record_type == "turn_interrupted":
            if turn_id is None:
                raise RuntimeError("turn_interrupted record is missing turn_id")
            return turn_paused(turn_id, reason="outcome_unknown")
        return None

    @staticmethod
    def _project_item_started(record: RolloutRecord, *, kind: str) -> StreamEvent:
        if record.turn_id is None:
            raise RuntimeError("public Item start is missing turn_id")
        internal_item_id = record.payload.get("item_id")
        if not isinstance(internal_item_id, str):
            raise RuntimeError("public Item start is missing item_id")
        item_kind = _public_item_kind(kind)
        public_item_id = record.payload.get("public_item_id")
        if not isinstance(public_item_id, str):
            if item_kind is TurnItemKind.LEGACY_MESSAGE:
                public_item_id = internal_item_id
            elif record.payload_schema_version == 1:
                attempt_id = record.payload.get("attempt_id")
                channel = record.payload.get("channel")
                if isinstance(attempt_id, str) and isinstance(channel, str):
                    public_item_id = derive_model_public_item_id(
                        turn_id=record.turn_id,
                        model_attempt_id=attempt_id,
                        channel=channel,
                    )
            if not isinstance(public_item_id, str):
                raise RuntimeError("public Item start is missing persisted public_item_id")
        return item_started(
            turn_id=record.turn_id,
            item_id=public_item_id,
            item_kind=item_kind,
            iteration=_iteration(record.payload),
        )

    @staticmethod
    def _project_item_completed(record: RolloutRecord, *, kind: str) -> StreamEvent:
        if record.turn_id is None:
            raise RuntimeError("public Item completion is missing turn_id")
        internal_item_id = record.payload.get("item_id")
        if not isinstance(internal_item_id, str):
            raise RuntimeError("public Item completion is missing item_id")
        item_kind = _public_item_kind(kind)
        public_item_id = record.payload.get("public_item_id")
        if not isinstance(public_item_id, str):
            if item_kind is TurnItemKind.LEGACY_MESSAGE:
                public_item_id = internal_item_id
            elif record.payload_schema_version == 1:
                attempt_id = record.payload.get("attempt_id")
                channel = record.payload.get("channel")
                if isinstance(attempt_id, str) and isinstance(channel, str):
                    public_item_id = derive_model_public_item_id(
                        turn_id=record.turn_id,
                        model_attempt_id=attempt_id,
                        channel=channel,
                    )
            if not isinstance(public_item_id, str):
                raise RuntimeError("public Item completion is missing persisted public_item_id")
        payload = record.payload.get("payload", {})
        if not isinstance(payload, Mapping):
            raise RuntimeError("public Item completion payload is malformed")
        status, error = _record_item_status(record.payload)
        if item_kind is TurnItemKind.AGENT_MESSAGE:
            data = {
                "content": payload.get("text", payload.get("content", "")),
                "tool_calls": payload.get("tool_calls", ()),
            }
        elif item_kind in {TurnItemKind.REASONING, TurnItemKind.PLAN}:
            data = {"content": payload.get("content", payload.get("text", ""))}
        else:
            text = payload.get("text", payload.get("content", ""))
            data = {"content": text}
        return item_completed(
            turn_id=record.turn_id,
            item_id=public_item_id,
            item_kind=item_kind,
            status=status,
            iteration=_iteration(record.payload),
            data=data,
            error=error,
            parent_item_id=(
                str(record.payload["parent_item_id"])
                if isinstance(record.payload.get("parent_item_id"), str)
                else None
            ),
        )

    @staticmethod
    def _validate_internal_item_kind(record: RolloutRecord) -> None:
        kind = record.payload.get("kind")
        if kind in _PUBLIC_ITEM_KINDS or kind in _SUPPRESSED_ITEM_KINDS:
            return
        raise RuntimeError(f"unknown internal Item kind fails closed: {kind!r}")

    @classmethod
    def _validate_public_item_ids(cls, record: RolloutRecord) -> None:
        payload = record.payload
        if record.turn_id is None:
            return
        if record.record_type in {"model_operation_prepared", "model_retry_prepared"}:
            attempt_id = payload.get("attempt_id")
            if not isinstance(attempt_id, str):
                return
            persisted = payload.get("public_item_ids")
            if persisted is None and record.payload_schema_version == 1:
                return
            if not isinstance(persisted, Mapping):
                raise RuntimeError("persisted public Item ID mapping is missing")
            for channel in ("agent_message", "reasoning", "plan"):
                cls._require_public_item_id(
                    persisted.get(channel),
                    derive_model_public_item_id(
                        turn_id=record.turn_id,
                        model_attempt_id=attempt_id,
                        channel=channel,
                    ),
                )
        attempt_id = payload.get("attempt_id")
        channel = payload.get("channel")
        if isinstance(attempt_id, str) and isinstance(channel, str):
            cls._require_public_item_id(
                payload.get("public_item_id"),
                derive_model_public_item_id(
                    turn_id=record.turn_id,
                    model_attempt_id=attempt_id,
                    channel=channel,
                ),
            )
        operation_id = payload.get("operation_id")
        generation = payload.get("attempt_generation", payload.get("claim_generation"))
        if (
            isinstance(operation_id, str)
            and isinstance(generation, int)
            and not isinstance(generation, bool)
            and generation > 0
            and record.record_type
            in {
                "tool_operation_claimed",
                "tool_operation_outcome_committed",
                "tool_operation_stale_result",
                "item_started",
                "item_completed",
            }
        ):
            cls._require_public_item_id(
                payload.get("public_item_id"),
                derive_operation_public_item_id(
                    turn_id=record.turn_id,
                    operation_id=operation_id,
                    attempt_generation=generation,
                ),
            )
        plan_snapshot = payload.get("plan_snapshot")
        if isinstance(plan_snapshot, Mapping):
            revision = plan_snapshot.get("revision")
            if isinstance(revision, int) and not isinstance(revision, bool):
                cls._require_public_item_id(
                    payload.get("plan_public_item_id"),
                    derive_plan_public_item_id(
                        turn_id=record.turn_id,
                        revision=revision,
                    ),
                )

    @staticmethod
    def _require_public_item_id(persisted: object, expected: str) -> None:
        if persisted != expected:
            raise RuntimeError(
                "persisted public Item ID mismatch; durable history requires repair"
            )

    def _encode(self, *, thread_id: str, sequence: int) -> str:
        encoded = json.dumps(
            {
                "version": _CURSOR_VERSION,
                "store_epoch": self._store.epoch,
                "schema_epoch": _SCHEMA_EPOCH,
                "thread_id": thread_id,
                "thread_sequence": sequence,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return base64.urlsafe_b64encode(encoded).decode().rstrip("=")

    def _decode(self, cursor: str, *, thread_id: str) -> int:
        if not isinstance(cursor, str) or not cursor:
            raise ValueError("event cursor must be non-empty")
        try:
            padding = "=" * (-len(cursor) % 4)
            payload = json.loads(
                base64.urlsafe_b64decode(cursor + padding).decode()
            )
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("event cursor is malformed") from exc
        if not isinstance(payload, dict) or payload.get("version") != _CURSOR_VERSION:
            raise ValueError("event cursor version is unsupported")
        if payload.get("schema_epoch") != _SCHEMA_EPOCH:
            raise ValueError("event cursor schema epoch mismatch")
        if payload.get("store_epoch") != self._store.epoch:
            raise ValueError("event cursor belongs to a different store epoch")
        if payload.get("thread_id") != thread_id:
            raise ValueError("event cursor belongs to a different thread")
        sequence = payload.get("thread_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("event cursor sequence is invalid")
        latest = self._store.list_records(thread_id)
        if latest and sequence > latest[-1].thread_sequence:
            raise ValueError("event cursor is ahead of the Thread tail")
        return sequence


def _accepted_answer_item_ids(records: tuple[RolloutRecord, ...]) -> frozenset[str]:
    starts = {
        str(record.payload["item_id"]): (record.payload.get("kind"), record.producer)
        for record in records
        if record.record_type == "item_started"
        and isinstance(record.payload.get("item_id"), str)
    }
    model_texts = {
        record.payload.get("payload", {}).get("text")
        for record in records
        if record.record_type == "item_completed"
        and starts.get(str(record.payload.get("item_id")), (None, None))[0]
        == "model_response"
    }
    return frozenset(
        item_id
        for record in records
        if record.record_type == "item_completed"
        and (item_id := str(record.payload.get("item_id"))) in starts
        and starts[item_id] == ("agent_message", "runtime")
        and record.payload.get("payload", {}).get("text") in model_texts
    )


def _tool_completion_status(
    status: str,
    error_code: str | None,
) -> tuple[ItemStatus, str | None]:
    if status == "succeeded":
        return ItemStatus.SUCCESS, None
    if status == "cancelled":
        return ItemStatus.CANCELLED, None
    if status == "unknown":
        return ItemStatus.OUTCOME_UNKNOWN, error_code or "tool outcome is unknown"
    return ItemStatus.FAILED, error_code or "tool execution failed"


def _public_item_kind(kind: str) -> TurnItemKind:
    mapping = {
        "agent_message": TurnItemKind.LEGACY_MESSAGE,
        "model_response": TurnItemKind.AGENT_MESSAGE,
        "model_reasoning": TurnItemKind.REASONING,
        "model_plan": TurnItemKind.PLAN,
        "plan_state": TurnItemKind.PLAN,
    }
    try:
        return mapping[kind]
    except KeyError:
        raise RuntimeError(f"internal Item kind has no public projection: {kind!r}") from None


def _iteration(payload: Mapping[str, Any]) -> int:
    iteration = payload.get("iteration", 0)
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise RuntimeError("public Item iteration is malformed")
    return iteration


def _record_item_status(
    payload: Mapping[str, Any],
) -> tuple[ItemStatus, str | None]:
    raw_status = payload.get("status", "success")
    try:
        status = ItemStatus(str(raw_status))
    except ValueError:
        raise RuntimeError(f"public Item status is unsupported: {raw_status!r}") from None
    error = payload.get("error")
    if status in {ItemStatus.FAILED, ItemStatus.OUTCOME_UNKNOWN}:
        if not isinstance(error, str) or not error:
            raise RuntimeError("failed public Item completion is missing error")
        return status, error
    return status, None

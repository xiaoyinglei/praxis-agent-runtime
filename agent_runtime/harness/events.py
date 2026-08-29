"""Replayable client events derived from the canonical Rollout log."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_runtime.harness.rollout import RolloutRecord, RolloutStore
from agent_runtime.streaming.events import (
    derive_model_public_item_id,
    derive_operation_public_item_id,
    derive_plan_public_item_id,
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


@dataclass(frozen=True, slots=True)
class RolloutEvent:
    cursor: str
    record_id: int
    thread_id: str
    turn_id: str | None
    thread_sequence: int
    event_type: str
    producer: str
    data: Mapping[str, Any]


class RolloutEventReader:
    """Read independent Thread tails or one global record-id tail."""

    def __init__(self, store: RolloutStore) -> None:
        self._store = store

    def read(
        self,
        thread_id: str,
        *,
        after: str | None = None,
    ) -> tuple[RolloutEvent, ...]:
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("thread_id must be non-empty")
        self._store.read_thread(thread_id)
        sequence = 0 if after is None else self._decode(after, thread_id=thread_id)
        records = tuple(
            record
            for record in self._store.list_records(thread_id)
            if record.thread_sequence > sequence
        )
        for record in records:
            if record.record_type == "item_started":
                self._validate_internal_item_kind(record)
            self._validate_public_item_ids(record)
        return tuple(self._event(record) for record in records)

    def read_global(
        self,
        *,
        after_record_id: int = 0,
    ) -> tuple[RolloutEvent, ...]:
        return tuple(
            self._event(record)
            for record in self._store.list_global_records(
                after_record_id=after_record_id
            )
        )

    def _event(self, record: RolloutRecord) -> RolloutEvent:
        return RolloutEvent(
            cursor=self._encode(
                thread_id=record.thread_id,
                sequence=record.thread_sequence,
            ),
            record_id=record.record_id,
            thread_id=record.thread_id,
            turn_id=record.turn_id,
            thread_sequence=record.thread_sequence,
            event_type=record.record_type,
            producer=record.producer,
            data=record.payload,
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
                "version": 1,
                "epoch": self._store.epoch,
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
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("event cursor version is unsupported")
        if payload.get("epoch") != self._store.epoch:
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

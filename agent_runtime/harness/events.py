"""Replayable client events derived from the canonical Rollout log."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_runtime.harness.rollout import RolloutRecord, RolloutStore


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

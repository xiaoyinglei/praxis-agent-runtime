"""Replayable client events derived from the canonical Rollout log."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from agent_runtime.core.messages import model_tool_call_payload
from agent_runtime.harness.protocol import (
    restore_model_response,
)
from agent_runtime.harness.records import RolloutRecord
from agent_runtime.harness.state import RuntimeState
from agent_runtime.modeling.contracts import context_capacity_policy
from agent_runtime.streaming.events import (
    EventType,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
    derive_model_public_item_id,
    derive_operation_public_item_id,
    derive_plan_public_item_id,
    derive_reconciliation_public_item_id,
    item_completed,
    item_started,
    turn_aborted,
    turn_cancellation_requested,
    turn_completed,
    turn_paused,
    turn_resumed,
    turn_started,
)
from agent_runtime.streaming.sink import TurnEventDispatcher

_PUBLIC_ITEM_KINDS = frozenset(
    {
        "agent_message",
        "model_response",
        "model_reasoning",
        "model_plan",
        "model_output",
        "tool_result",
        "tool_reconciliation",
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
        "context_summary_response",
        "context_message",
        "verification",
        "tool_continuation",
        "workspace_checkpoint",
        "resource_usage",
    }
)
_SUPPRESSED_RECORD_TYPES = frozenset(
    {
        "thread_created",
        "session_settings_updated",
        "thread_forked",
        "artifact_committed",
        "budget_reserved",
        "budget_dispatched",
        "budget_settled",
        "budget_released",
        "budget_unknown",
        "budget_child_allocated",
        "budget_child_settled",
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
    }
)
_CURSOR_VERSION = 1
_SCHEMA_EPOCH = "harness-canonical-stream-v4"

_MODEL_ATTEMPT_TRANSITIONS = {
    "model_operation_prepared": "prepared",
    "model_retry_prepared": "retry_prepared",
    "model_attempt_dispatched": "dispatched",
    "model_response_started": "responding",
    "model_attempt_completed": "completed",
    "model_attempt_rejected": "rejected",
    "model_attempt_cancelled": "cancelled",
    "model_attempt_unknown": "unknown",
    "model_attempt_late_response": "late_response",
}


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


async def commit_and_publish[T](
    store: RuntimeState,
    operation: Callable[[], T],
    dispatcher: TurnEventDispatcher | None,
) -> T:
    """Admit the mutation first, then publish only its committed records."""
    mutation = await store.capture_mutation_async(operation)
    if dispatcher is not None:
        events = await store.run_async(RolloutEventReader(store).project_committed_batch, mutation.records)
        for replayed in events:
            await dispatcher.emit(replayed.event, cursor=replayed.cursor)
    return mutation.value


class RolloutEventReader:
    """Read independent Thread tails or one global record-id tail."""

    def __init__(self, store: RuntimeState) -> None:
        self._store = store

    def read(
        self, thread_id: str, *, after: str | None = None, limit: int | None = None,
    ) -> tuple[ReplayEvent, ...]:
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("thread_id must be non-empty")
        self._validate_limit(limit)
        self._store.read_thread(thread_id)
        sequence = 0 if after is None else self._decode(after, thread_id=thread_id)
        last = self._store.last_record(thread_id)
        if last is None:
            return ()
        projected: list[tuple[RolloutRecord, StreamEvent]] = []
        while sequence < last.thread_sequence and (limit is None or len(projected) < limit):
            records = self._store.list_record_tail(
                thread_id, after_thread_sequence=sequence, through_thread_sequence=last.thread_sequence,
                limit=256 if limit is None else min(256, limit - len(projected)),
            )
            if not records:
                break
            projected.extend(self._project_records(records))
            sequence = records[-1].thread_sequence
        return self._wrap_events(projected)

    def read_global(
        self, *, after_record_id: int = 0, limit: int | None = None,
    ) -> tuple[ReplayEvent, ...]:
        self._validate_limit(limit)
        if isinstance(after_record_id, bool) or not isinstance(after_record_id, int):
            raise TypeError("after_record_id must be an integer")
        if after_record_id < 0:
            raise ValueError("after_record_id must be non-negative")
        last = self._store.last_global_record_id()
        projected: list[tuple[RolloutRecord, StreamEvent]] = []
        while after_record_id < last and (limit is None or len(projected) < limit):
            records = self._store.list_global_records(
                after_record_id=after_record_id, through_record_id=last,
                limit=256 if limit is None else min(256, limit - len(projected)),
            )
            if not records:
                break
            projected.extend(self._project_records(records))
            after_record_id = records[-1].record_id
        return self._wrap_events(projected)

    @staticmethod
    def _validate_limit(limit: int | None) -> None:
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("event limit must be a positive integer")

    def _wrap_events(self, projected: Iterable[tuple[RolloutRecord, StreamEvent]]) -> tuple[ReplayEvent, ...]:
        return tuple(
            ReplayEvent(
                event=replace(event, sequence=index, timestamp_ms=record.committed_at_ms),
                cursor=self._encode(thread_id=record.thread_id, sequence=record.thread_sequence),
                record_id=record.record_id, thread_sequence=record.thread_sequence,
            )
            for index, (record, event) in enumerate(projected, start=1)
        )

    def project_committed_batch(
        self,
        records: tuple[RolloutRecord, ...],
    ) -> tuple[ReplayEvent, ...]:
        """Project only one already-committed, single-Thread mutation batch."""

        if not records:
            return ()
        thread_ids = {record.thread_id for record in records}
        if len(thread_ids) != 1:
            raise RuntimeError("committed mutation batch must belong to one Thread")
        return self._wrap_events(self._project_records(records))

    def _project_records(self, records: tuple[RolloutRecord, ...]) -> tuple[tuple[RolloutRecord, StreamEvent], ...]:
        projected: list[tuple[RolloutRecord, StreamEvent]] = []
        for record in records:
            if record.payload_schema_version not in {1, 2, 3}:
                raise RuntimeError(f"unsupported event record schema: {record.payload_schema_version}")
            self._validate_public_item_ids(record)
            current = self._legacy_event_record(record) if record.payload_schema_version < 3 else record
            if current is None:
                continue
            if current.record_type in {"item_started", "item_completed"}:
                self._validate_internal_item_kind(current)
            for event in self._project_record(current):
                projected.append((record, event))
        return tuple(projected)

    def _legacy_event_record(self, record: RolloutRecord) -> RolloutRecord | None:
        """Upcast pre-v3 references at the read boundary. New records never enter here."""
        payload = dict(record.payload)
        item_id = payload.get("item_id")
        if record.record_type == "item_completed" and isinstance(item_id, str):
            start = self._store.read_event_reference("item_started", item_id)
            if start is None:
                raise RuntimeError("completed internal Item has no durable start")
            self._validate_internal_item_kind(start)
            self._validate_public_item_ids(start)
            payload = {**start.payload, **payload}
        kind = payload.get("kind")
        if kind == "agent_message" and record.producer == "runtime" and record.turn_id is not None:
            completion = (record if record.record_type == "item_completed"
                          else self._store.read_event_reference("item_completed", str(item_id)))
            if completion is not None and self._store.find_model_response_item(
                record.turn_id, completion.payload.get("payload", {}).get("text"),
            ) is not None:
                return None
        if (kind in {"model_response", "model_reasoning", "model_plan"}
                and payload.get("public_item_id") is None and record.producer != "migration"):
            attempt = self._store.read_event_reference("model_attempt_completed", str(item_id))
            if attempt is not None:
                channel = {"model_response": "agent_message", "model_reasoning": "reasoning", "model_plan": "plan"}
                payload["public_item_id"] = derive_model_public_item_id(
                    turn_id=record.turn_id or "", model_attempt_id=str(attempt.payload["attempt_id"]),
                    channel=channel[str(kind)],
                )
        if record.record_type == "tool_operation_claimed" or kind == "tool_result":
            operation_id = payload.get("operation_id")
            if operation_id is None and kind == "tool_result":
                link = self._store.read_event_reference("tool_operation_result_linked", str(item_id))
                if link is not None:
                    operation_id = link.payload["operation_id"]
                    payload["operation_id"] = operation_id
            if isinstance(operation_id, str):
                prepared = self._store.read_event_reference("tool_operation_prepared", operation_id)
                if prepared is None:
                    prepared = self._store.read_event_reference("tool_operation_migrated", operation_id)
                if prepared is None:
                    raise RuntimeError("legacy tool event has no canonical operation definition")
                payload.setdefault("tool_name", prepared.payload["tool_name"])
                payload.setdefault("tool_call_id", prepared.payload["tool_call_id"])
                if payload.get("public_item_id") is None:
                    claim = self._store.read_event_reference(
                        "tool_operation_claimed", operation_id, at_record_id=record.record_id,
                    )
                    if claim is not None:
                        payload["public_item_id"] = derive_operation_public_item_id(
                            turn_id=record.turn_id or "", operation_id=operation_id,
                            attempt_generation=int(claim.payload["claim_generation"]),
                        )
                if record.record_type == "item_completed" and payload.get("status") is None:
                    outcome = self._store.read_event_reference(
                        "tool_operation_outcome_committed", operation_id, at_record_id=record.record_id,
                    )
                    outcome_payload = outcome.payload if outcome is not None else prepared.payload
                    if "status" not in outcome_payload:
                        raise RuntimeError("legacy tool result has no canonical execution outcome")
                    status, error = _tool_completion_status(
                        str(outcome_payload["status"]), outcome_payload.get("error_code"),
                    )
                    payload.update(status=status.value, error=error)
        return replace(record, payload=payload)

    def _project_record(self, record: RolloutRecord) -> tuple[StreamEvent, ...]:
        lifecycle = self._project_lifecycle(record)
        if lifecycle is not None:
            return (lifecycle,)
        notification = self._project_runtime_notification(record)
        if notification is not None:
            return (notification,)
        item_id = record.payload.get("item_id")
        kind = record.payload.get("kind")
        if kind in {"model_response", "model_reasoning", "model_plan"} and (
            record.payload.get("output_protocol") == "model-outputs-v1"
            or record.payload.get("payload", {}).get("output_protocol") == "model-outputs-v1"
        ):
            return ()
        if record.record_type == "tool_operation_claimed":
            operation_id = record.payload.get("operation_id")
            public_item_id = record.payload.get("public_item_id")
            if not isinstance(public_item_id, str) or record.turn_id is None or not isinstance(operation_id, str):
                raise RuntimeError("claimed tool operation is missing public identity")
            return (item_started(
                turn_id=record.turn_id, item_id=public_item_id,
                item_kind=TurnItemKind.COMMAND if record.payload["tool_name"] == "run_command" else TurnItemKind.TOOL,
                data={key: record.payload[key] for key in ("operation_id", "tool_call_id", "tool_name")},
            ),)
        if record.record_type == "item_started":
            if kind in _SUPPRESSED_ITEM_KINDS or kind in {"tool_result", "command_execution"}:
                return ()
            return (self._project_item_started(record, kind=str(kind)),)
        if record.record_type == "item_completed" and kind == "tool_result":
            operation_id = record.payload.get("operation_id")
            public_item_id = record.payload.get("public_item_id")
            if isinstance(operation_id, str) and isinstance(public_item_id, str):
                if record.turn_id is None:
                    raise RuntimeError("tool result is missing turn identity")
                status, error = _record_item_status(record.payload)
                return (item_completed(
                    turn_id=record.turn_id, item_id=public_item_id,
                    item_kind=(TurnItemKind.COMMAND
                               if record.payload["tool_name"] == "run_command" else TurnItemKind.TOOL),
                    status=status, data={"result": record.payload.get("payload", {})}, error=error,
                ),)
            result = record.payload.get("payload")
            if (operation_id is None and record.turn_id is not None and isinstance(item_id, str)
                    and isinstance(result, Mapping) and result.get("is_error") is True
                    and isinstance(result.get("tool_name"), str) and isinstance(result.get("tool_call_id"), str)):
                return (item_completed(
                    turn_id=record.turn_id, item_id=item_id,
                    item_kind=TurnItemKind.COMMAND if result["tool_name"] == "run_command" else TurnItemKind.TOOL,
                    status=ItemStatus.FAILED,
                    error=str(result.get("error_message") or result.get("error_code") or "tool rejected"),
                    data={"result": dict(result), "execution_started": False},
                ),)
            return ()
        if record.record_type == "item_completed":
            if not isinstance(item_id, str) or not isinstance(kind, str):
                raise RuntimeError("completed internal Item has no durable start")
            payload = record.payload.get("payload", {})
            if kind == "model_request":
                request_ref = payload.get("request_ref", {})
                projection = request_ref.get("context_projection", {})
                if (isinstance(request_ref.get("effective_settings"), Mapping) or (
                    projection.get("policy_revision") == "context-capacity-v2"
                    and context_capacity_policy(request_ref.get("step_snapshot", {}).get("binding_manifest", {}))
                    is not None
                )):
                    return (StreamEvent(type=EventType.BUDGET_UPDATE, turn_id=record.turn_id or "", data={
                        "purpose": request_ref.get("purpose"), "context_projection": dict(projection),
                        "budget_status": request_ref.get("budget_status", {}),
                        **({"effective_settings": dict(request_ref["effective_settings"])}
                           if isinstance(request_ref.get("effective_settings"), Mapping) else {}),
                    }),)
            if kind in _SUPPRESSED_ITEM_KINDS or kind == "command_execution":
                return ()
            return (self._project_item_completed(record, kind=kind),)
        if record.record_type in _SUPPRESSED_RECORD_TYPES:
            return ()
        raise RuntimeError(f"unknown durable Rollout record type: {record.record_type!r}")

    @staticmethod
    def _project_runtime_notification(record: RolloutRecord) -> StreamEvent | None:
        """Publish decisions from their committed record, never current mutable state."""
        payload = record.payload
        data: dict[str, Any]
        event_type: EventType
        if record.record_type in {
            "tool_operation_prepared", "tool_operation_status_changed", "tool_operation_start_rejected",
            "tool_operation_claim_expired", "tool_operation_result_missing", "tool_operation_superseded",
            "tool_operation_outcome_committed", "tool_operation_reconciled",
        }:
            event_type = EventType.TOOL_OPERATION_UPDATED
            data = {key: payload[key] for key in (
                "operation_id", "tool_call_id", "tool_name", "request_id", "status", "reason", "error_code",
                "claim_generation", "attempt_count", "requires_reconciliation", "committed_execution_status",
                "reconciler_revision", "duration_ms",
            ) if key in payload}
            data["transition"] = record.record_type.removeprefix("tool_operation_")
        elif record.record_type in _MODEL_ATTEMPT_TRANSITIONS:
            event_type = EventType.MODEL_ATTEMPT_UPDATED
            data = {key: payload[key] for key in (
                "operation_id", "attempt_id", "generation", "previous_attempt_id",
                "provider_response_id", "usage", "reason", "error_type", "error_message",
                "response_status", "incomplete_reason", "stop_reason",
            ) if key in payload}
            data["status"] = _MODEL_ATTEMPT_TRANSITIONS[record.record_type]
        elif record.record_type == "interaction_requested":
            event_type = EventType.INTERACTION_REQUESTED
            data = {key: payload[key] for key in ("request_id", "kind", "operation_id", "request") if key in payload}
        elif record.record_type in {"interaction_resolved", "approval_invalidated"}:
            event_type = (EventType.INTERACTION_RESOLVED if record.record_type == "interaction_resolved"
                          else EventType.INTERACTION_INVALIDATED)
            data = {key: payload[key] for key in (
                "request_id", "kind", "operation_id", "version", "reason", "approval_status",
            ) if key in payload}
            response = payload.get("response", {})
            if isinstance(response, Mapping) and response.get("decision") in {"approve", "deny", "cancel"}:
                data["decision"] = response["decision"]
            action = response.get("action") if isinstance(response, Mapping) else None
            if action in {"abort", "recheck"}:
                data["action"] = action
            data["status"] = ("invalidated" if record.record_type == "approval_invalidated"
                              or payload.get("approval_status") == "invalidated" or action == "recheck"
                              else "cancelled" if action == "abort"
                              or isinstance(response, Mapping) and response.get("cancelled") is True
                              else "resolved")
        elif record.record_type == "item_started" and payload.get("kind") == "context_compaction":
            event_type = EventType.CONTEXT_COMPACTION_STARTED
            data = {"compaction_id": payload.get("item_id")}
        elif record.record_type == "item_completed" and payload.get("kind") == "context_compaction":
            event_type = EventType.CONTEXT_COMPACTION_COMPLETED
            body = payload.get("payload", {})
            data = {key: body[key] for key in (
                "context_version", "covered_item_ids", "covered_item_refs", "covered_sequence_ranges",
                "capacity_diagnostics", "preserved_item_ids", "source_revision", "algorithm_revision",
                "reason", "error",
            ) if key in body}
            data.update(compaction_id=payload.get("item_id"),
                        status="completed" if payload.get("status", "success") == "success" else payload["status"])
        elif record.record_type == "item_completed" and payload.get("kind") == "completion_decision":
            event_type = EventType.COMPLETION_DECIDED
            body = payload.get("payload", {})
            data = {key: body[key] for key in (
                "proposal_item_id", "action", "reason", "validation_status", "validation_scope", "reason_code",
            ) if key in body}
            data.setdefault("validation_status", "not_evaluated")
        elif record.record_type == "item_completed" and payload.get("kind") == "verification":
            event_type = EventType.VERIFICATION_COMPLETED
            body = payload.get("payload", {})
            data = {key: body[key] for key in (
                "operation_id", "tool_call_id", "arguments_digest", "source_result_item_id",
                "verification_kind", "verifier", "verified_resources",
            ) if key in body}
        else:
            return None
        if not record.turn_id:
            raise RuntimeError("Runtime notification is missing turn_id")
        return StreamEvent(type=event_type, turn_id=record.turn_id, data=data)

    @staticmethod
    def _project_lifecycle(record: RolloutRecord) -> StreamEvent | None:
        turn_id = record.turn_id
        if record.record_type == "turn_started":
            if turn_id is None:
                raise RuntimeError("turn_started record is missing turn_id")
            event = turn_started(turn_id)
            binding = record.payload.get("binding_manifest", {})
            policy = context_capacity_policy(binding)
            if policy is not None:
                envelope = binding.get("binding", binding)
                definition = envelope.get("definition", envelope)
                event = replace(event, data={**event.data,
                    "context_capacity_policy": policy.model_dump(mode="json"),
                    "model_context_tokens": definition.get("context_window_tokens", "unknown"),
                })
            return event
        if record.record_type == "turn_paused":
            if turn_id is None:
                raise RuntimeError("turn_paused record is missing turn_id")
            reason = record.payload.get("reason", "interaction_required")
            event = turn_paused(turn_id, reason=str(reason))
            return replace(event, data={**event.data, **{
                key: record.payload[key] for key in ("request_id", "operation_id", "attempt_id")
                if key in record.payload
            }})
        if record.record_type == "turn_resumed":
            if turn_id is None:
                raise RuntimeError("turn_resumed record is missing turn_id")
            event = turn_resumed(turn_id)
            return replace(event, data={**event.data, **{
                key: record.payload[key] for key in ("request_id", "operation_id", "attempt_id")
                if key in record.payload
            }})
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
    def _project_item_started(
        record: RolloutRecord,
        *,
        kind: str,
        derived_public_item_id: str | None = None,
    ) -> StreamEvent:
        if record.turn_id is None:
            raise RuntimeError("public Item start is missing turn_id")
        internal_item_id = record.payload.get("item_id")
        if not isinstance(internal_item_id, str):
            raise RuntimeError("public Item start is missing item_id")
        item_kind = _public_item_kind(record, kind=kind)
        public_item_id = record.payload.get("public_item_id")
        if not isinstance(public_item_id, str):
            public_item_id = derived_public_item_id
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
            data=_public_output_data(record) if kind == "model_output" else None,
            parent_item_id=(
                str(record.payload["parent_item_id"])
                if isinstance(record.payload.get("parent_item_id"), str)
                else None
            ),
        )

    @staticmethod
    def _project_item_completed(
        record: RolloutRecord,
        *,
        kind: str,
        derived_public_item_id: str | None = None,
    ) -> StreamEvent:
        if record.turn_id is None:
            raise RuntimeError("public Item completion is missing turn_id")
        internal_item_id = record.payload.get("item_id")
        if not isinstance(internal_item_id, str):
            raise RuntimeError("public Item completion is missing item_id")
        item_kind = _public_item_kind(record, kind=kind)
        public_item_id = record.payload.get("public_item_id")
        if not isinstance(public_item_id, str):
            public_item_id = derived_public_item_id
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
        if kind == "model_output":
            data = _public_output_data(record)
        elif item_kind is TurnItemKind.AGENT_MESSAGE:
            response = restore_model_response(payload)
            data = {
                "content": response.text,
                "tool_calls": tuple(model_tool_call_payload(call) for call in response.tool_calls),
            }
            if response.status == "incomplete":
                status = ItemStatus.FAILED
                error = f"Incomplete model response: {response.incomplete_reason}"
                data["response_status"] = "incomplete"
                data["incomplete_reason"] = response.incomplete_reason
        elif kind == "plan_state" and item_kind is TurnItemKind.PLAN:
            plan = payload.get("plan", {})
            if not isinstance(plan, Mapping):
                raise RuntimeError("canonical plan snapshot payload is malformed")
            data = {"plan": dict(plan)}
        elif item_kind is TurnItemKind.RECONCILIATION:
            result = payload.get("result", {})
            if not isinstance(result, Mapping):
                raise RuntimeError("canonical reconciliation payload is malformed")
            data = {
                "result": dict(result),
                "reconciler_revision": payload.get("reconciler_revision"),
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
                str(record.payload["parent_item_id"]) if isinstance(record.payload.get("parent_item_id"), str) else None
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
        attempt_id = payload.get("attempt_id")
        channel = payload.get("channel")
        output_id = payload.get("output_id")
        if payload.get("kind") == "model_output":
            if not isinstance(attempt_id, str) or not isinstance(output_id, str):
                raise RuntimeError("model output is missing durable output identity")
            output_kind = payload.get("output_kind")
            channel = "agent_message" if output_kind == "text" else output_kind
            if not isinstance(channel, str):
                raise RuntimeError("model output is missing output_kind")
        if isinstance(attempt_id, str) and isinstance(channel, str):
            cls._require_public_item_id(
                payload.get("public_item_id"),
                derive_model_public_item_id(
                    turn_id=record.turn_id,
                    model_attempt_id=attempt_id,
                    channel=channel,
                    output_id=output_id if isinstance(output_id, str) else None,
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
            if payload.get("public_item_id") is None and record.payload_schema_version == 1:
                return
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
        revision = payload.get("revision")
        if (
            record.record_type in {"item_started", "item_completed"}
            and isinstance(revision, int)
            and not isinstance(revision, bool)
            and payload.get("public_item_id") is not None
        ):
            cls._require_public_item_id(
                payload.get("public_item_id"),
                derive_plan_public_item_id(
                    turn_id=record.turn_id,
                    revision=revision,
                ),
            )
        reconciler_revision = payload.get("reconciler_revision")
        if (
            record.record_type in {"item_started", "item_completed"}
            and isinstance(operation_id, str)
            and isinstance(reconciler_revision, str)
        ):
            cls._require_public_item_id(
                payload.get("public_item_id"),
                derive_reconciliation_public_item_id(
                    turn_id=record.turn_id,
                    operation_id=operation_id,
                    reconciler_revision=reconciler_revision,
                ),
            )

    @staticmethod
    def _require_public_item_id(persisted: object, expected: str) -> None:
        if persisted != expected:
            raise RuntimeError("persisted public Item ID mismatch; durable history requires repair")

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
            raise _full_resync_error("event cursor must be non-empty")
        try:
            padding = "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(cursor + padding).decode())
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise _full_resync_error("event cursor is malformed") from exc
        if not isinstance(payload, dict) or payload.get("version") != _CURSOR_VERSION:
            raise _full_resync_error("event cursor version is unsupported")
        if payload.get("schema_epoch") != _SCHEMA_EPOCH:
            raise _full_resync_error("event cursor schema epoch mismatch")
        if payload.get("store_epoch") != self._store.epoch:
            raise _full_resync_error("event cursor belongs to a different store epoch")
        if payload.get("thread_id") != thread_id:
            raise _full_resync_error("event cursor belongs to a different thread")
        sequence = payload.get("thread_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise _full_resync_error("event cursor sequence is invalid")
        latest = self._store.last_record(thread_id)
        if latest and sequence > latest.thread_sequence:
            raise _full_resync_error("event cursor is ahead of the Thread tail")
        return sequence


def _full_resync_error(reason: str) -> ValueError:
    return ValueError(f"{reason}; discard the cursor and perform a full resync")


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


def _public_item_kind(record: RolloutRecord, *, kind: str) -> TurnItemKind:
    persisted = record.payload.get("public_item_kind")
    if persisted is not None:
        try:
            return TurnItemKind(str(persisted))
        except ValueError:
            raise RuntimeError(f"persisted public Item kind is unsupported: {persisted!r}") from None
    if record.producer == "migration" and kind == "model_response":
        return TurnItemKind.LEGACY_MESSAGE
    if kind == "model_output":
        output_kind = record.payload.get("output_kind")
        mapping = {
            "text": TurnItemKind.AGENT_MESSAGE,
            "reasoning": TurnItemKind.REASONING,
            "reasoning_summary": TurnItemKind.REASONING,
            "plan": TurnItemKind.PLAN,
            "tool_call": TurnItemKind.TOOL_CALL,
            "refusal": TurnItemKind.REFUSAL,
        }
        try:
            return mapping[str(output_kind)]
        except KeyError:
            raise RuntimeError(f"unsupported model output kind: {output_kind!r}") from None
    mapping = {
        "agent_message": TurnItemKind.LEGACY_MESSAGE,
        "model_response": TurnItemKind.AGENT_MESSAGE,
        "model_reasoning": TurnItemKind.REASONING,
        "model_plan": TurnItemKind.PLAN,
        "plan_state": TurnItemKind.PLAN,
        "tool_reconciliation": TurnItemKind.RECONCILIATION,
    }
    try:
        return mapping[kind]
    except KeyError:
        raise RuntimeError(f"internal Item kind has no public projection: {kind!r}") from None


def _public_output_data(record: RolloutRecord) -> dict[str, Any]:
    body = record.payload.get("payload", {})
    output = body.get("output", record.payload.get("output", {}))
    if not isinstance(output, Mapping):
        raise RuntimeError("model output snapshot is malformed")
    data = {key: output[key] for key in (
        "output_id", "content", "tool_call_id", "tool_name", "arguments", "argument_error",
    ) if key in output}
    data["output_id"] = record.payload["output_id"]
    data["output_kind"] = record.payload["output_kind"]
    if record.payload["output_kind"] == "tool_call":
        data["execution_started"] = False
    return data


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

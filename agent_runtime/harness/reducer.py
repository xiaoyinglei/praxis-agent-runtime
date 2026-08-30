"""The single deterministic reducer for canonical RolloutRecords."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class ReducerRecord(Protocol):
    @property
    def thread_id(self) -> str: ...

    @property
    def turn_id(self) -> str | None: ...

    @property
    def thread_sequence(self) -> int: ...

    @property
    def record_type(self) -> str: ...

    @property
    def producer(self) -> str: ...

    @property
    def payload_schema_version(self) -> int: ...

    @property
    def payload(self) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class ProjectionState:
    threads: dict[str, dict[str, Any]] = field(default_factory=dict)
    turns: dict[str, dict[str, Any]] = field(default_factory=dict)
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_operations: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_attempts: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_operations: dict[str, dict[str, Any]] = field(default_factory=dict)
    interactions: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)


def apply_record(
    state: ProjectionState,
    record: ReducerRecord,
    *,
    reducer_version: int,
) -> None:
    """Apply one record without time, random values, I/O, or hidden state."""
    if not isinstance(record.payload, Mapping):
        raise TypeError("rollout record payload must be an object")
    payload = _upcast_payload(record)
    sequence = record.thread_sequence
    thread_id = record.thread_id
    match record.record_type:
        case "thread_created":
            state.threads[thread_id] = {
                "thread_id": thread_id,
                "workspace": payload["workspace"],
                "parent_thread_id": None,
                "fork_turn_id": None,
                "active_turn_id": None,
                "head_turn_id": None,
                "head_version": 0,
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
        case "thread_forked":
            state.threads[thread_id] = {
                "thread_id": thread_id,
                "workspace": payload["workspace"],
                "parent_thread_id": payload["parent_thread_id"],
                "fork_turn_id": payload["fork_turn_id"],
                "active_turn_id": None,
                "head_turn_id": payload["fork_turn_id"],
                "head_version": 0,
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
        case "turn_started":
            turn_id = payload["turn_id"]
            state.turns[turn_id] = {
                "turn_id": turn_id,
                "thread_id": thread_id,
                "status": "running",
                "predecessor_turn_id": payload["predecessor_turn_id"],
                "turn_index": payload["turn_index"],
                "binding_manifest": payload["binding_manifest"],
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            thread = state.threads[thread_id]
            thread["active_turn_id"] = turn_id
            thread["head_turn_id"] = turn_id
            thread["head_version"] += 1
            thread["applied_thread_sequence"] = sequence
        case "model_operation_prepared":
            operation_id = payload["operation_id"]
            attempt_id = payload["attempt_id"]
            turn_id = _turn_id(record)
            state.model_operations[operation_id] = {
                "operation_id": operation_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "status": "prepared",
                "active_attempt_id": attempt_id,
                "generation": payload["generation"],
                "request_hash": payload["request_hash"],
                "context_hash": payload["context_hash"],
                "tool_hash": payload["tool_hash"],
                "wire_hash": payload["wire_hash"],
                "request_ref": payload["request_ref"],
                "response_item_id": None,
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            state.model_attempts[attempt_id] = {
                "attempt_id": attempt_id,
                "operation_id": operation_id,
                "generation": payload["generation"],
                "status": "prepared",
                "provider_response_id": None,
                "usage": {},
                "claim_owner": None,
                "lease_expires_at": None,
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            _advance(state, thread_id, turn_id, sequence)
        case "model_attempt_dispatched":
            operation = state.model_operations[payload["operation_id"]]
            attempt = state.model_attempts[payload["attempt_id"]]
            operation["status"] = "dispatched"
            operation["applied_thread_sequence"] = sequence
            attempt["status"] = "dispatched"
            attempt["claim_owner"] = payload["claim_owner"]
            attempt["lease_expires_at"] = payload["lease_expires_at"]
            attempt["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "model_attempt_unknown":
            operation = state.model_operations[payload["operation_id"]]
            attempt = state.model_attempts[payload["attempt_id"]]
            operation["status"] = "unknown"
            operation["applied_thread_sequence"] = sequence
            attempt["status"] = "unknown"
            attempt["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "model_attempt_rejected":
            operation = state.model_operations[payload["operation_id"]]
            attempt = state.model_attempts[payload["attempt_id"]]
            operation["status"] = "failed"
            operation["applied_thread_sequence"] = sequence
            attempt["status"] = "failed"
            attempt["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "model_attempt_cancelled":
            operation = state.model_operations[payload["operation_id"]]
            attempt = state.model_attempts[payload["attempt_id"]]
            operation["status"] = "cancelled"
            operation["applied_thread_sequence"] = sequence
            attempt["status"] = "cancelled"
            attempt["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "model_retry_prepared":
            operation = state.model_operations[payload["operation_id"]]
            previous = state.model_attempts[payload["previous_attempt_id"]]
            previous["status"] = "abandoned"
            previous["applied_thread_sequence"] = sequence
            attempt_id = payload["attempt_id"]
            state.model_attempts[attempt_id] = {
                "attempt_id": attempt_id,
                "operation_id": payload["operation_id"],
                "generation": payload["generation"],
                "status": "prepared",
                "provider_response_id": None,
                "usage": {},
                "claim_owner": None,
                "lease_expires_at": None,
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            operation["status"] = "prepared"
            operation["active_attempt_id"] = attempt_id
            operation["generation"] = payload["generation"]
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "model_attempt_completed":
            operation = state.model_operations[payload["operation_id"]]
            attempt = state.model_attempts[payload["attempt_id"]]
            operation["status"] = "completed"
            operation["response_item_id"] = payload["response_item_id"]
            operation["applied_thread_sequence"] = sequence
            attempt["status"] = "completed"
            attempt["provider_response_id"] = payload["provider_response_id"]
            attempt["usage"] = payload["usage"]
            attempt["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "model_attempt_late_response":
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_prepared":
            operation_id = payload["operation_id"]
            turn_id = _turn_id(record)
            state.tool_operations[operation_id] = {
                "operation_id": operation_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "tool_call_id": payload["tool_call_id"],
                "tool_name": payload["tool_name"],
                "arguments_digest": payload["arguments_digest"],
                "execution_revision": payload["execution_revision"],
                "effects": payload.get("effects", []),
                "resources": payload.get("resources", []),
                "idempotent": payload["idempotent"],
                "status": "prepared",
                "attempt_count": payload["attempt_count"],
                "error_code": payload["error_code"],
                "requires_reconciliation": payload["requires_reconciliation"],
                "result_item_id": None,
                "approval_request_id": None,
                "claim_generation": 0,
                "fencing_token": None,
                "claim_owner": None,
                "lease_expires_at": None,
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            _advance(state, thread_id, turn_id, sequence)
        case "tool_operation_migrated":
            operation_id = payload["operation_id"]
            turn_id = _turn_id(record)
            state.tool_operations[operation_id] = {
                "operation_id": operation_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "tool_call_id": payload["tool_call_id"],
                "tool_name": payload["tool_name"],
                "arguments_digest": payload["arguments_digest"],
                "execution_revision": payload["execution_revision"],
                "effects": payload.get("effects", []),
                "resources": payload.get("resources", []),
                "idempotent": payload["idempotent"],
                "status": payload["status"],
                "attempt_count": payload["attempt_count"],
                "error_code": payload["error_code"],
                "requires_reconciliation": payload["requires_reconciliation"],
                "result_item_id": payload["result_item_id"],
                "approval_request_id": None,
                "claim_generation": 0,
                "fencing_token": None,
                "claim_owner": None,
                "lease_expires_at": None,
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            _advance(state, thread_id, turn_id, sequence)
        case "tool_operation_status_changed":
            operation = state.tool_operations[payload["operation_id"]]
            operation["status"] = payload["status"]
            operation["attempt_count"] = payload["attempt_count"]
            operation["error_code"] = payload["error_code"]
            operation["requires_reconciliation"] = payload["requires_reconciliation"]
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_claimed":
            operation = state.tool_operations[payload["operation_id"]]
            operation["status"] = "running"
            operation["claim_generation"] = payload["claim_generation"]
            operation["fencing_token"] = payload["fencing_token"]
            operation["claim_owner"] = payload["claim_owner"]
            operation["lease_expires_at"] = payload["lease_expires_at"]
            operation["attempt_count"] = payload["attempt_count"]
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_start_rejected":
            operation = state.tool_operations[payload["operation_id"]]
            operation["status"] = "failed"
            operation["attempt_count"] = payload["attempt_count"]
            operation["error_code"] = payload["error_code"]
            operation["requires_reconciliation"] = False
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_claim_expired":
            operation = state.tool_operations[payload["operation_id"]]
            operation["status"] = "unknown"
            operation["requires_reconciliation"] = True
            operation["error_code"] = "lease_expired_outcome_unknown"
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_result_missing":
            operation = state.tool_operations[payload["operation_id"]]
            operation["status"] = "unknown"
            operation["requires_reconciliation"] = True
            operation["error_code"] = "result_commit_missing"
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_outcome_committed":
            operation = state.tool_operations[payload["operation_id"]]
            operation["status"] = payload["status"]
            operation["attempt_count"] = payload["attempt_count"]
            operation["error_code"] = payload["error_code"]
            operation["requires_reconciliation"] = payload["requires_reconciliation"]
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_reconciled":
            operation = state.tool_operations[payload["operation_id"]]
            operation["status"] = payload["status"]
            operation["error_code"] = payload["error_code"]
            operation["requires_reconciliation"] = False
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_stale_result":
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_result_linked":
            operation = state.tool_operations[payload["operation_id"]]
            operation["result_item_id"] = payload["result_item_id"]
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "interaction_requested":
            request_id = payload["request_id"]
            turn_id = _turn_id(record)
            state.interactions[request_id] = {
                "request_id": request_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "kind": payload["kind"],
                "status": "pending",
                "version": 1,
                "operation_id": payload["operation_id"],
                "request": payload["request"],
                "response": {},
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            if payload["kind"] == "tool_approval":
                state.approvals[request_id] = {
                    "request_id": request_id,
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "operation_id": payload["operation_id"],
                    "status": "pending",
                    "scope": payload["request"],
                    "applied_thread_sequence": sequence,
                    "reducer_version": reducer_version,
                }
            _advance(state, thread_id, turn_id, sequence)
        case "interaction_resolved":
            request_id = payload["request_id"]
            interaction = state.interactions[request_id]
            interaction["status"] = "resolved"
            interaction["version"] = payload["version"]
            interaction["response"] = payload["response"]
            interaction["applied_thread_sequence"] = sequence
            approval = state.approvals.get(request_id)
            if approval is not None:
                approval["status"] = payload["approval_status"]
                approval["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "approval_invalidated":
            approval = state.approvals[payload["request_id"]]
            approval["status"] = "invalidated"
            approval["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_approval_requested":
            operation = state.tool_operations[payload["operation_id"]]
            operation["status"] = "awaiting_approval"
            operation["approval_request_id"] = payload["request_id"]
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_approval_resolved":
            operation = state.tool_operations[payload["operation_id"]]
            operation["status"] = payload["status"]
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "tool_operation_superseded":
            operation = state.tool_operations[payload["operation_id"]]
            operation["status"] = "superseded"
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "turn_paused":
            turn_id = payload["turn_id"]
            turn = state.turns[turn_id]
            turn["status"] = "paused"
            turn["applied_thread_sequence"] = sequence
            state.threads[thread_id]["applied_thread_sequence"] = sequence
        case "turn_interrupted":
            turn_id = payload["turn_id"]
            turn = state.turns[turn_id]
            turn["status"] = "interrupted"
            turn["applied_thread_sequence"] = sequence
            state.threads[thread_id]["applied_thread_sequence"] = sequence
        case "turn_resumed":
            turn_id = payload["turn_id"]
            turn = state.turns[turn_id]
            turn["status"] = "running"
            turn["applied_thread_sequence"] = sequence
            state.threads[thread_id]["applied_thread_sequence"] = sequence
        case "item_started":
            item_id = payload["item_id"]
            turn_id = _turn_id(record)
            state.items[item_id] = {
                "item_id": item_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "sequence": sum(item["turn_id"] == turn_id for item in state.items.values()) + 1,
                "kind": payload["kind"],
                "status": "started",
                "producer": record.producer,
                "payload": {},
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            _advance(state, thread_id, turn_id, sequence)
        case "item_completed":
            item = state.items[payload["item_id"]]
            item["status"] = "completed"
            item["payload"] = payload["payload"]
            item["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "artifact_committed":
            artifact_id = payload["artifact_id"]
            turn_id = _turn_id(record)
            state.artifacts[artifact_id] = {
                "artifact_id": artifact_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "blob_sha256": payload["blob_sha256"],
                "size_bytes": payload["size_bytes"],
                "media_type": payload["media_type"],
                "name": payload["name"],
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            _advance(state, thread_id, turn_id, sequence)
        case "turn_completed":
            turn_id = payload["turn_id"]
            turn = state.turns[turn_id]
            turn["status"] = "completed"
            turn["applied_thread_sequence"] = sequence
            thread = state.threads[thread_id]
            thread["active_turn_id"] = None
            thread["applied_thread_sequence"] = sequence
        case "turn_failed" | "turn_cancelled":
            turn_id = payload["turn_id"]
            turn = state.turns[turn_id]
            turn["status"] = "cancelled" if record.record_type == "turn_cancelled" else "failed"
            turn["applied_thread_sequence"] = sequence
            thread = state.threads[thread_id]
            thread["active_turn_id"] = None
            thread["applied_thread_sequence"] = sequence
        case _:
            raise ValueError(f"unsupported record type: {record.record_type}")


def _turn_id(record: ReducerRecord) -> str:
    if record.turn_id is None:
        raise ValueError(f"record requires turn_id: {record.record_type}")
    return record.turn_id


def _upcast_payload(record: ReducerRecord) -> Mapping[str, Any]:
    """Resolve one immutable payload through the registered schema boundary."""

    upcaster = _PAYLOAD_UPCASTERS.get(record.payload_schema_version)
    if upcaster is None:
        raise ValueError(f"unsupported RolloutRecord payload schema version: {record.payload_schema_version}")
    return upcaster(record.payload)


def _identity_v1(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return payload


_PAYLOAD_UPCASTERS = {1: _identity_v1, 2: _identity_v1}


def _advance(
    state: ProjectionState,
    thread_id: str,
    turn_id: str,
    sequence: int,
) -> None:
    state.threads[thread_id]["applied_thread_sequence"] = sequence
    state.turns[turn_id]["applied_thread_sequence"] = sequence

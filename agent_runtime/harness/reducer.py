"""The single deterministic reducer for canonical RolloutRecords."""

from __future__ import annotations

import math
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
    budget_reservations: dict[str, dict[str, Any]] = field(default_factory=dict)
    child_budget_allocations: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_operations: dict[str, dict[str, Any]] = field(default_factory=dict)
    interactions: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    host_commands: dict[str, dict[str, Any]] = field(default_factory=dict)
    channel_entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Transient replay bookkeeping, not a persisted projection or hash input.
    item_sequences: dict[str, int] = field(default_factory=dict)


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
                "settings": {},
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
                "settings": {},
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
        case "session_settings_updated":
            state.threads[thread_id]["settings"] = payload["settings"]
            state.threads[thread_id]["applied_thread_sequence"] = sequence
        case "channel_entry_updated":
            state.channel_entries[payload["entry_id"]] = {
                "entry_id": payload["entry_id"], "thread_id": thread_id,
                "kind": payload["kind"], "key": payload["key"], "value": payload["value"],
                "applied_thread_sequence": sequence, "reducer_version": reducer_version,
            }
            state.threads[thread_id]["applied_thread_sequence"] = sequence
        case "host_command_received":
            state.host_commands[payload["command_id"]] = {
                "command_id": payload["command_id"], "thread_id": thread_id, "turn_id": payload["turn_id"],
                "kind": payload["kind"], "request_hash": payload["request_hash"], "request": payload["request"],
                "status": "pending", "generation": 0, "owner": None, "lease_expires_at": None,
                "available_at": 0.0, "result": {}, "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            state.threads[thread_id]["applied_thread_sequence"] = sequence
        case "host_command_claimed" | "host_command_renewed":
            command = state.host_commands[payload["command_id"]]
            command.update({
                "status": "claimed", "owner": payload["owner"], "generation": payload["generation"],
                "lease_expires_at": payload["lease_expires_at"], "applied_thread_sequence": sequence,
            })
            state.threads[thread_id]["applied_thread_sequence"] = sequence
        case "host_command_settled" | "host_command_released":
            command = state.host_commands[payload["command_id"]]
            command.update({
                "status": payload["status"], "owner": None, "lease_expires_at": None,
                "available_at": payload.get("available_at", 0.0), "result": payload["result"],
                "applied_thread_sequence": sequence,
            })
            state.threads[thread_id]["applied_thread_sequence"] = sequence
        case "turn_started":
            turn_id = payload["turn_id"]
            state.turns[turn_id] = {
                "turn_id": turn_id,
                "thread_id": thread_id,
                "status": "running",
                "terminal_reason_code": None,
                "terminal_message": None,
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
        case "model_response_started":
            attempt = state.model_attempts[payload["attempt_id"]]
            attempt["provider_response_id"] = payload["provider_response_id"]
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
        case "model_retry_scheduled" | "model_retry_cancelled":
            retry_turn_id = _turn_id(record)
            operation = state.model_operations[payload["operation_id"]]
            attempt = state.model_attempts[payload["attempt_id"]]
            if (
                record.producer != "runtime"
                or operation["turn_id"] != retry_turn_id
                or operation["thread_id"] != thread_id
                or operation["active_attempt_id"] != payload["attempt_id"]
                or operation["generation"] != payload["generation"]
                or attempt["operation_id"] != payload["operation_id"]
            ):
                raise ValueError("retry fact must reference the current runtime-owned generation")
            if record.record_type == "model_retry_scheduled":
                if operation["status"] != "failed" or attempt["status"] != "failed":
                    raise ValueError("automatic retry requires a rejected current attempt")
                for field_name in ("observed_at", "retry_at", "deadline_at"):
                    number = payload[field_name]
                    if type(number) not in {int, float} or not math.isfinite(number):
                        raise ValueError("retry times must be finite")
                if not payload["observed_at"] <= payload["retry_at"] <= payload["deadline_at"]:
                    raise ValueError("retry schedule exceeds its deadline")
                if (
                    type(payload["retry_number"]) is not int
                    or type(payload["max_retries"]) is not int
                    or not 1 <= payload["retry_number"] <= payload["max_retries"]
                ):
                    raise ValueError("retry schedule count exceeds its allowance")
                failure = payload["failure"]
                if (
                    not isinstance(failure, Mapping)
                    or failure.get("retryable") is not True
                    or failure.get("output_started") is not False
                    or failure.get("provider_outcome") not in {"not_sent", "rejected"}
                ):
                    raise ValueError("automatic retry requires an unambiguous eligible failure")
                operation["status"] = "retry_wait"
            else:
                if operation["status"] not in {"retry_wait", "prepared"}:
                    raise ValueError("only waiting or unstarted retries may be cancelled")
                operation["status"] = "failed"
                if attempt["status"] == "prepared":
                    attempt["status"] = "failed"
                    attempt["applied_thread_sequence"] = sequence
            operation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, retry_turn_id, sequence)
        case "model_retry_prepared":
            operation = state.model_operations[payload["operation_id"]]
            previous = state.model_attempts[payload["previous_attempt_id"]]
            if not payload.get("automatic", False):
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
        case "budget_reserved":
            reservation_id = payload["reservation_id"]
            if reservation_id in state.budget_reservations:
                raise RuntimeError("duplicate budget reservation identity")
            turn_id = _turn_id(record)
            state.budget_reservations[reservation_id] = {
                "reservation_id": reservation_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "scope_id": payload["scope_id"],
                "operation_id": payload["operation_id"],
                "attempt_id": payload["attempt_id"],
                "reserved": payload["reserved"],
                "actual": {},
                "status": "reserved",
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            _advance(state, thread_id, turn_id, sequence)
        case "budget_dispatched":
            reservation = state.budget_reservations[payload["reservation_id"]]
            if reservation["status"] != "reserved":
                raise RuntimeError("only a reserved budget can be dispatched")
            reservation["status"] = "dispatched"
            reservation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "budget_settled":
            reservation = state.budget_reservations[payload["reservation_id"]]
            if reservation["status"] not in {"reserved", "dispatched", "unknown"}:
                raise RuntimeError("budget reservation cannot be settled from its current state")
            reservation["status"] = "settled"
            reservation["actual"] = payload["actual"]
            reservation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "budget_released":
            reservation = state.budget_reservations[payload["reservation_id"]]
            if reservation["status"] not in {"reserved", "dispatched", "unknown"}:
                raise RuntimeError("budget reservation cannot be released from its current state")
            reservation["status"] = "released"
            reservation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "budget_unknown":
            reservation = state.budget_reservations[payload["reservation_id"]]
            if reservation["status"] != "dispatched":
                raise RuntimeError("only a dispatched budget can become unknown")
            reservation["status"] = "unknown"
            reservation["applied_thread_sequence"] = sequence
            _advance(state, thread_id, _turn_id(record), sequence)
        case "budget_child_allocated":
            allocation_id = payload["allocation_id"]
            if allocation_id in state.child_budget_allocations:
                raise RuntimeError("duplicate child budget allocation identity")
            state.child_budget_allocations[allocation_id] = {
                "allocation_id": allocation_id,
                "thread_id": thread_id,
                "parent_turn_id": payload["parent_turn_id"],
                "child_turn_id": payload["child_turn_id"],
                "allocated_tokens": payload["allocated_tokens"],
                "allocated_cost_micros": payload.get("allocated_cost_micros"),
                "actual": {},
                "status": "active",
                "applied_thread_sequence": sequence,
                "reducer_version": reducer_version,
            }
            state.threads[thread_id]["applied_thread_sequence"] = sequence
        case "budget_child_settled":
            allocation = state.child_budget_allocations[payload["allocation_id"]]
            if allocation["status"] != "active":
                raise RuntimeError("only an active child allocation can be settled")
            allocation["status"] = "settled"
            allocation["actual"] = payload["actual"]
            allocation["applied_thread_sequence"] = sequence
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
        case "turn_cancellation_requested":
            turn_id = payload["turn_id"]
            turn = state.turns[turn_id]
            turn["applied_thread_sequence"] = sequence
            state.threads[thread_id]["applied_thread_sequence"] = sequence
        case "item_started":
            item_id = payload["item_id"]
            turn_id = _turn_id(record)
            item_sequence = state.item_sequences.get(turn_id, 0) + 1
            state.item_sequences[turn_id] = item_sequence
            state.items[item_id] = {
                "item_id": item_id,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "sequence": item_sequence,
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
            if record.payload_schema_version >= 3 and payload.get("kind") != item["kind"]:
                raise ValueError("completed Item kind does not match its canonical start")
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
            if record.payload_schema_version >= 3:
                identity = payload.get("answer_item_id")
                answer = state.items.get(identity) if isinstance(identity, str) else None
                if (
                    answer is None
                    or answer["turn_id"] != turn_id
                    or answer["status"] != "completed"
                    or answer["kind"] not in {"model_response", "agent_message"}
                    or (answer["kind"] == "agent_message" and not isinstance(answer["payload"].get("text"), str))
                    or (
                        answer["kind"] == "model_response"
                        and not isinstance(answer["payload"].get("outputs"), (list, tuple))
                    )
                    or answer["payload"].get("response_status", "completed") != "completed"
                ):
                    raise ValueError("Turn completion requires a committed answer Item from this Turn")
            turn = state.turns[turn_id]
            turn["status"] = "completed"
            turn["terminal_reason_code"] = "completed"
            turn["terminal_message"] = None
            turn["applied_thread_sequence"] = sequence
            thread = state.threads[thread_id]
            thread["active_turn_id"] = None
            thread["applied_thread_sequence"] = sequence
        case "turn_failed" | "turn_cancelled":
            turn_id = payload["turn_id"]
            turn = state.turns[turn_id]
            turn["status"] = "cancelled" if record.record_type == "turn_cancelled" else "failed"
            turn["terminal_reason_code"] = (
                "cancelled" if record.record_type == "turn_cancelled" else payload["reason_code"]
            )
            turn["terminal_message"] = payload.get("message")
            turn["applied_thread_sequence"] = sequence
            thread = state.threads[thread_id]
            thread["active_turn_id"] = None
            thread["applied_thread_sequence"] = sequence
        case "execution_protocol_adopted" | "response_handling_decided" | "tool_batch_admitted":
            turn_id = _turn_id(record)
            if record.producer != "runtime" or turn_id not in state.turns:
                raise ValueError("execution handling fact requires its runtime-owned Turn")
            if record.record_type == "response_handling_decided":
                response = state.items.get(payload["response_item_id"])
                handling_operation = state.model_operations.get(payload["operation_id"])
                if (
                    response is None
                    or handling_operation is None
                    or response["turn_id"] != turn_id
                    or handling_operation.get("response_item_id") != payload["response_item_id"]
                    or handling_operation["status"] != "completed"
                ):
                    raise ValueError("handling receipt must reference a completed canonical response")
            if record.record_type == "tool_batch_admitted":
                response = state.items.get(payload["response_item_id"])
                if response is None or response["turn_id"] != turn_id or response["status"] != "completed":
                    raise ValueError("batch admission requires a committed response")
            _advance(state, thread_id, turn_id, sequence)
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


_PAYLOAD_UPCASTERS = {
    1: _identity_v1, 2: _identity_v1, 3: _identity_v1, 4: _identity_v1,
    5: _identity_v1, 6: _identity_v1, 7: _identity_v1, 8: _identity_v1, 9: _identity_v1,
}


def _advance(
    state: ProjectionState,
    thread_id: str,
    turn_id: str,
    sequence: int,
) -> None:
    state.threads[thread_id]["applied_thread_sequence"] = sequence
    state.turns[turn_id]["applied_thread_sequence"] = sequence

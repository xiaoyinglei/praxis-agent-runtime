"""Explicit tool ops over owner-serialized durable facts."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any, Literal
from uuid import uuid4

from agent_runtime.harness import artifact_ops, facts, turn_ops
from agent_runtime.harness.operation_values import (
    ResourceClaimConflictError,
    _canonical_json,
    _json_object,
    _resource_sets_conflict,
    _resource_tuple,
    _string_tuple,
)
from agent_runtime.harness.records import (
    InteractionSnapshot,
    ItemSnapshot,
    ToolOperationSnapshot,
    _item_snapshot,
)
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation
from agent_runtime.planning import PlanStep
from agent_runtime.streaming.events import (
    derive_operation_public_item_id,
    derive_plan_public_item_id,
    derive_reconciliation_public_item_id,
)


@owner_operation
def record_tool_call(
    store: RuntimeStore,
    *,
    turn_id: str,
    tool_call_id: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    origin: Mapping[str, Any],
    argument_error: str | None = None,
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
    if argument_error is not None:
        if not isinstance(argument_error, str):
            raise TypeError("argument_error must be a string or None")
        expected_payload["argument_error"] = argument_error
    item_id: str | None = None
    with store.tx.transaction():
        thread_id = turn_ops._running_turn_thread_id(store, turn_id)
        matches = store.tx.rows("items", where={"turn_id": turn_id, "kind": "tool_call"}, order_by="sequence")
        same_identity = [row for row in matches if row["payload"].get("tool_call_id") == tool_call_id]
        if len(same_identity) > 1:
            raise RuntimeError("tool call identity has multiple canonical Items")
        if same_identity:
            existing = same_identity[0]
            if existing["status"] != "completed" or _canonical_json(existing["payload"]) != _canonical_json(
                expected_payload
            ):
                raise RuntimeError("tool call identity conflicts with committed payload")
            item_id = str(existing["item_id"])
        else:
            item_id = f"item_{uuid4().hex}"
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="model",
                payload={"item_id": item_id, "kind": "tool_call"},
            )
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="model",
                payload={"item_id": item_id, "payload": expected_payload},
            )
    if item_id is None:
        raise RuntimeError("tool call Item identity was not resolved")
    return store.read_item(item_id)


@owner_operation
def record_tool_execution_state(
    store: RuntimeStore,
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
    with store.tx.transaction():
        thread_id = turn_ops._running_turn_thread_id(store, turn_id)
        existing = store.tx.first("tool_operations", where={"operation_id": operation_id})
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
            prior_call = store.tx.first("tool_operations", where={"turn_id": turn_id, "tool_call_id": tool_call_id})
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
                "effects": list(frozen_effects),
                "resources": [dict(resource) for resource in frozen_resources],
            }
            if any(_canonical_json(existing[name]) != _canonical_json(value) for name, value in immutable.items()):
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
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type=record_type,
            producer="runtime",
            payload=payload,
        )
    return store.read_tool_operation(operation_id)


@owner_operation
def claim_tool_operation(
    store: RuntimeStore,
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
    artifact_ops._assert_artifact_integrity(store, store.read_tool_operation(operation_id).turn_id)
    claimed_at = time.time() if now is None else float(now)
    with store.tx.transaction():
        operation = store.tx.first("tool_operations", where={"operation_id": operation_id})
        if operation is None:
            raise KeyError(f"unknown tool operation: {operation_id}")
        turn_ops._assert_no_pending_cancellation(store, str(operation["turn_id"]))
        if operation["status"] != "ready":
            raise RuntimeError("only a ready tool operation can be claimed")
        requested_resources = tuple(operation["resources"])
        active_operations = tuple(
            row
            for row in store.tx.rows("tool_operations", where={"status": ("running", "unknown")})
            if row["operation_id"] != operation_id
        )
        for active in active_operations:
            active_resources = tuple(active["resources"])
            if _resource_sets_conflict(requested_resources, active_resources):
                raise ResourceClaimConflictError(f"resource claim conflict with operation: {active['operation_id']}")
        generation = int(operation["claim_generation"]) + 1
        fencing_token = f"fence_{uuid4().hex}"
        public_item_id = derive_operation_public_item_id(
            turn_id=str(operation["turn_id"]),
            operation_id=operation_id,
            attempt_generation=generation,
        )
        facts.append_fact(
            store,
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
                "tool_name": str(operation["tool_name"]),
                "tool_call_id": str(operation["tool_call_id"]),
            },
        )
    return store.read_tool_operation(operation_id)


@owner_operation
def reject_ready_tool_operation(
    store: RuntimeStore,
    *,
    operation_id: str,
    error_code: str,
) -> ToolOperationSnapshot:
    if not isinstance(error_code, str) or not error_code:
        raise ValueError("start rejection error_code must be non-empty")
    with store.tx.transaction():
        operation = store.tx.first("tool_operations", where={"operation_id": operation_id})
        if operation is None:
            raise KeyError(f"unknown tool operation: {operation_id}")
        if operation["status"] != "ready":
            raise RuntimeError("only a ready tool operation can be rejected")
        facts.append_fact(
            store,
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
    return store.read_tool_operation(operation_id)


@owner_operation
def expire_tool_operation_claim(
    store: RuntimeStore,
    *,
    operation_id: str,
    now: float | None = None,
) -> ToolOperationSnapshot:
    observed_at = time.time() if now is None else float(now)
    with store.tx.transaction():
        operation = store.tx.first("tool_operations", where={"operation_id": operation_id})
        if operation is None:
            raise KeyError(f"unknown tool operation: {operation_id}")
        if operation["status"] != "running":
            raise RuntimeError("only a running tool operation claim can expire")
        expires_at = operation["lease_expires_at"]
        if expires_at is None or observed_at <= float(expires_at):
            raise RuntimeError("tool operation claim has not expired")
        facts.append_fact(
            store,
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
        facts.append_fact(
            store,
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
        turn = store.tx.first("turns", where={"turn_id": operation["turn_id"]})
        if turn is not None and turn["status"] == "running":
            facts.append_fact(
                store,
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
    return store.read_tool_operation(operation_id)


@owner_operation
def mark_tool_result_missing(
    store: RuntimeStore,
    *,
    operation_id: str,
) -> InteractionSnapshot:
    """Pause when execution outcome committed but its canonical result did not."""

    with store.tx.transaction():
        operation = store.tx.first("tool_operations", where={"operation_id": operation_id})
        if operation is None:
            raise KeyError(f"unknown tool operation: {operation_id}")
        if operation["status"] not in {"succeeded", "failed"}:
            raise RuntimeError("missing-result recovery requires a committed execution outcome")
        if operation["result_item_id"] is not None:
            raise RuntimeError("tool operation already has a canonical result")
        turn = store.tx.first("turns", where={"turn_id": operation["turn_id"]})
        if turn is None or turn["status"] != "running":
            raise RuntimeError("missing-result recovery requires a running Turn")
        existing = store.tx.rows(
            "interactions",
            where={"turn_id": operation["turn_id"], "kind": "tool_reconciliation", "status": "pending"},
        )
        if existing:
            raise RuntimeError("tool reconciliation is already pending")
        prior_status = str(operation["status"])
        facts.append_fact(
            store,
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
        facts.append_fact(
            store,
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
        facts.append_fact(
            store,
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
    return store.read_interaction(request_id)


@owner_operation
def commit_tool_operation_outcome(
    store: RuntimeStore,
    *,
    operation_id: str,
    claim_generation: int,
    fencing_token: str,
    status: str,
    attempt_count: int,
    error_code: str | None,
    requires_reconciliation: bool,
    duration_ms: float | None = None,
    now: float | None = None,
) -> bool:
    if status not in {"succeeded", "failed", "unknown"}:
        raise ValueError(f"unsupported claimed tool outcome: {status}")
    if duration_ms is not None and (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, (int, float))
        or not math.isfinite(duration_ms)
        or duration_ms < 0
    ):
        raise ValueError("tool duration must be finite and non-negative")
    observed_at = time.time() if now is None else float(now)
    with store.tx.transaction():
        operation = store.tx.first("tool_operations", where={"operation_id": operation_id})
        if operation is None:
            raise KeyError(f"unknown tool operation: {operation_id}")
        # Same fencing rule as model attempts: expiry only authorizes the
        # recovery takeover, which rewrites the status to unknown under
        # the same serialized transaction. A matching claim generation and
        # fencing token on a still-running operation stay authoritative
        # past the lease clock.
        is_current = bool(
            operation["status"] == "running"
            and int(operation["claim_generation"]) == claim_generation
            and operation["fencing_token"] == fencing_token
        )
        if not is_current:
            return False
        public_item_id = derive_operation_public_item_id(
            turn_id=str(operation["turn_id"]),
            operation_id=operation_id,
            attempt_generation=claim_generation,
        )
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="tool_operation_outcome_committed",
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
                "duration_ms": duration_ms,
            },
        )
        if is_current and status == "unknown":
            request_id = f"interaction_{uuid4().hex}"
            facts.append_fact(
                store,
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
            facts.append_fact(
                store,
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


@owner_operation
def commit_tool_operation_result(
    store: RuntimeStore,
    *,
    turn_id: str,
    operation_id: str,
    claim_generation: int,
    fencing_token: str,
    status: str,
    attempt_count: int,
    error_code: str | None,
    requires_reconciliation: bool,
    result: Mapping[str, Any],
    duration_ms: float | None = None,
    now: float | None = None,
) -> bool:
    with store.tx.transaction():
        accepted = commit_tool_operation_outcome(
            store,
            operation_id=operation_id,
            claim_generation=claim_generation,
            fencing_token=fencing_token,
            status=status,
            attempt_count=attempt_count,
            error_code=error_code,
            requires_reconciliation=requires_reconciliation,
            duration_ms=duration_ms,
            now=now,
        )
        if not accepted:
            return False
        record_tool_result(
            store,
            turn_id=turn_id,
            operation_id=operation_id,
            result=result,
        )
    return True


@owner_operation
def resolve_tool_reconciliation(
    store: RuntimeStore,
    *,
    turn_id: str,
    status: str,
    result: Mapping[str, Any],
    reconciler_revision: str,
) -> ItemSnapshot:
    if status not in {"succeeded", "failed", "cancelled"}:
        raise ValueError("reconciled tool status is unsupported")
    artifact_ops._assert_artifact_integrity(store, turn_id)
    if not isinstance(reconciler_revision, str) or not reconciler_revision.strip():
        raise ValueError("reconciler_revision must be non-empty")
    frozen_result = _json_object(result, field="reconciled tool result")
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "paused":
            raise RuntimeError("tool reconciliation requires a paused Turn")
        pending = store.tx.rows(
            "interactions",
            where={"turn_id": turn_id, "kind": "tool_reconciliation", "status": "pending"},
            order_by="rowid",
        )
        if len(pending) != 1:
            raise RuntimeError("reconciliation requires exactly one pending interaction")
        interaction = pending[0]
        operation = store.tx.first("tool_operations", where={"operation_id": interaction["operation_id"]})
        if operation is None or operation["status"] != "unknown":
            raise RuntimeError("reconciliation interaction is not bound to an unknown operation")
        thread_id = str(turn["thread_id"])
        parent_item_id = derive_operation_public_item_id(
            turn_id=turn_id,
            operation_id=str(operation["operation_id"]),
            attempt_generation=int(operation["claim_generation"]),
        )
        public_item_id = derive_reconciliation_public_item_id(
            turn_id=turn_id,
            operation_id=str(operation["operation_id"]),
            reconciler_revision=reconciler_revision,
        )
        facts.append_fact(
            store,
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
        facts.append_fact(
            store,
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
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="reconciler",
            payload={
                "item_id": item_id,
                "kind": "tool_reconciliation",
                "operation_id": str(operation["operation_id"]),
                "public_item_id": public_item_id,
                "parent_item_id": parent_item_id,
                "reconciler_revision": reconciler_revision,
            },
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="reconciler",
            payload={
                "item_id": item_id,
                "operation_id": str(operation["operation_id"]),
                "public_item_id": public_item_id,
                "parent_item_id": parent_item_id,
                "reconciler_revision": reconciler_revision,
                "payload": {
                    "status": status,
                    "result": frozen_result,
                    "reconciler_revision": reconciler_revision,
                },
            },
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="tool_operation_result_linked",
            producer="reconciler",
            payload={
                "operation_id": operation["operation_id"],
                "result_item_id": item_id,
            },
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="turn_resumed",
            producer="reconciler",
            payload={
                "turn_id": turn_id,
                "request_id": interaction["request_id"],
            },
        )
    return store._latest_item(turn_id)


@owner_operation
def resolve_tool_approval(
    store: RuntimeStore,
    *,
    turn_id: str,
    decision: str,
    policy_recheck: bool = False,
) -> InteractionSnapshot:
    if decision not in {"approve", "deny"}:
        raise ValueError("approval decision must be approve or deny")
    artifact_ops._assert_artifact_integrity(store, turn_id)
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "paused":
            raise RuntimeError(f"turn is not paused: {turn_id}")
        pending = store.tx.rows(
            "interactions",
            where={"turn_id": turn_id, "kind": "tool_approval", "status": "pending"},
            order_by="rowid",
        )
        if len(pending) != 1:
            raise RuntimeError("resume requires exactly one pending tool approval")
        interaction = pending[0]
        operation = store.tx.first("tool_operations", where={"operation_id": interaction["operation_id"]})
        if operation is None or operation["status"] != "awaiting_approval":
            raise RuntimeError("pending approval is not bound to an awaiting operation")
        thread_id = str(turn["thread_id"])
        approval_status = "approved" if decision == "approve" else "denied"
        operation_status = "ready" if decision == "approve" else "denied"
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="interaction_resolved",
            producer="user",
            payload={
                "request_id": interaction["request_id"],
                "version": int(interaction["version"]) + 1,
                "response": {"decision": decision, **({"policy_recheck": True} if policy_recheck else {})},
                "approval_status": approval_status,
            },
        )
        facts.append_fact(
            store,
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
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="turn_resumed",
            producer="runtime",
            payload={"turn_id": turn_id, "request_id": interaction["request_id"]},
        )
    return store.read_interaction(str(interaction["request_id"]))


@owner_operation
def refresh_tool_approval(store: RuntimeStore, *, turn_id: str, request: Mapping[str, Any]) -> InteractionSnapshot:
    """Replace an unexecuted approval after a policy recheck, atomically."""
    frozen_request = _json_object(request, field="approval request")
    request_id = f"interaction_{uuid4().hex}"
    with store.tx.transaction():
        turn = store.read_turn(turn_id)
        candidates = [
            store.read_interaction(op.approval_request_id)
            for op in store.list_tool_operations(turn_id)
            if op.status in {"awaiting_approval", "ready"} and op.approval_request_id is not None
        ]
        if turn.status not in {"paused", "running"} or len(candidates) != 1:
            raise RuntimeError("approval refresh requires one unexecuted operation")
        previous = candidates[0]
        if previous.status == "resolved" and previous.response.get("policy_recheck") is not True:
            raise RuntimeError("only a policy recheck can reopen a resolved approval")
        if previous.status == "pending":
            facts.append_fact(
                store,
                thread_id=turn.thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="runtime",
                payload={
                    "request_id": previous.request_id,
                    "version": previous.version + 1,
                    "response": {"action": "recheck"},
                    "approval_status": "invalidated",
                },
            )
        else:
            facts.append_fact(
                store,
                thread_id=turn.thread_id,
                turn_id=turn_id,
                record_type="approval_invalidated",
                producer="runtime",
                payload={"request_id": previous.request_id, "reason": "policy recheck requires confirmation"},
            )
        facts.append_fact(
            store,
            thread_id=turn.thread_id,
            turn_id=turn_id,
            record_type="interaction_requested",
            producer="runtime",
            payload={
                "request_id": request_id,
                "kind": "tool_approval",
                "operation_id": previous.operation_id,
                "request": frozen_request,
            },
        )
        facts.append_fact(
            store,
            thread_id=turn.thread_id,
            turn_id=turn_id,
            record_type="tool_operation_approval_requested",
            producer="runtime",
            payload={"operation_id": previous.operation_id, "request_id": request_id},
        )
        if turn.status == "running":
            facts.append_fact(
                store,
                thread_id=turn.thread_id,
                turn_id=turn_id,
                record_type="turn_paused",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
    return store.read_interaction(request_id)


@owner_operation
def invalidate_tool_approval(
    store: RuntimeStore,
    *,
    turn_id: str,
    reason: str,
) -> InteractionSnapshot:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("approval invalidation reason must be non-empty")
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "paused":
            raise RuntimeError(f"turn is not paused: {turn_id}")
        pending = store.tx.rows(
            "interactions",
            where={"turn_id": turn_id, "kind": "tool_approval", "status": "pending"},
            order_by="rowid",
        )
        if len(pending) != 1:
            raise RuntimeError("invalidation requires one pending tool approval")
        interaction = pending[0]
        operation = store.tx.first("tool_operations", where={"operation_id": interaction["operation_id"]})
        if operation is None or operation["status"] != "awaiting_approval":
            raise RuntimeError("approval is not bound to an awaiting operation")
        thread_id = str(turn["thread_id"])
        facts.append_fact(
            store,
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
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="approval_invalidated",
            producer="runtime",
            payload={
                "request_id": interaction["request_id"],
                "reason": reason,
            },
        )
        facts.append_fact(
            store,
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
    return store.read_interaction(str(interaction["request_id"]))


@owner_operation
def invalidate_resolved_tool_approval(
    store: RuntimeStore,
    *,
    turn_id: str,
    reason: str,
) -> InteractionSnapshot:
    """Invalidate an approved ready operation after crash-time revalidation."""

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("approval invalidation reason must be non-empty")
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "running":
            raise RuntimeError("resolved approval recovery requires a running Turn")
        interaction = store.tx.first(
            "interactions",
            where={"turn_id": turn_id, "kind": "tool_approval", "status": "resolved"},
            order_by="rowid",
            newest=True,
            limit=1,
        )
        if (
            interaction is None
            or interaction["response"].get("decision") != "approve"
            or interaction["operation_id"] is None
        ):
            raise RuntimeError("no resolved approved operation can be invalidated")
        operation = store.tx.first("tool_operations", where={"operation_id": interaction["operation_id"]})
        approval = store.tx.first("approvals", where={"request_id": interaction["request_id"]})
        if operation is None or operation["status"] != "ready" or approval is None or approval["status"] != "approved":
            raise RuntimeError("resolved approval is not bound to a ready operation")
        thread_id = str(turn["thread_id"])
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="approval_invalidated",
            producer="runtime",
            payload={
                "request_id": interaction["request_id"],
                "reason": reason,
            },
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="tool_operation_superseded",
            producer="runtime",
            payload={"operation_id": interaction["operation_id"]},
        )
        facts.append_fact(
            store,
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
    return store.read_interaction(str(interaction["request_id"]))


@owner_operation
def request_tool_approval(
    store: RuntimeStore,
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
    with store.tx.transaction():
        thread_id = turn_ops._running_turn_thread_id(store, turn_id)
        prior_call = store.tx.first("tool_operations", where={"turn_id": turn_id, "tool_call_id": tool_call_id})
        if prior_call is not None:
            if (
                prior_call["status"] != "ready"
                or prior_call["approval_request_id"] is None
                or prior_call["arguments_digest"] != arguments_digest
                or prior_call["execution_revision"] != execution_revision
                or prior_call["tool_name"] != tool_name
                or tuple(prior_call["effects"]) != frozen_effects
                or tuple(prior_call["resources"]) != frozen_resources
            ):
                raise RuntimeError(f"tool call already has an operation: {prior_call['operation_id']}")
            # A second scope (e.g. network) must reuse the unstarted operation.
            operation_id = str(prior_call["operation_id"])
        else:
            facts.append_fact(
                store,
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
        facts.append_fact(
            store,
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
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="tool_operation_approval_requested",
            producer="runtime",
            payload={
                "operation_id": operation_id,
                "request_id": request_id,
            },
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="turn_paused",
            producer="runtime",
            payload={"turn_id": turn_id, "request_id": request_id},
        )
    return store.read_interaction(request_id)


@owner_operation
def record_tool_result(
    store: RuntimeStore,
    *,
    turn_id: str,
    operation_id: str | None,
    result: Mapping[str, Any],
) -> ItemSnapshot:
    frozen_result = _json_object(result, field="tool result")
    artifact_ops._validate_artifact_references(store, frozen_result)
    operation: Mapping[str, Any] | None = None
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        thread_id = str(turn["thread_id"])
        if operation_id is not None:
            operation = store.tx.first("tool_operations", where={"operation_id": operation_id, "turn_id": turn_id})
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
                reconciliation = store.tx.rows(
                    "interactions",
                    where={
                        "turn_id": turn_id,
                        "operation_id": operation_id,
                        "kind": "tool_reconciliation",
                        "status": "pending",
                    },
                )
                if len(reconciliation) != 1:
                    raise RuntimeError("paused unknown tool result requires one reconciliation interaction")
            elif turn["status"] != "running":
                raise RuntimeError(f"turn is not running: {turn_id}")
        elif turn["status"] != "running":
            raise RuntimeError(f"turn is not running: {turn_id}")
        public_item_id: str | None = None
        plan_public_item_id: str | None = None
        plan_snapshot: Mapping[str, Any] | None = None
        plan_event: Mapping[str, Any] | None = None
        plan_item_id: str | None = None
        if operation is not None:
            assert operation_id is not None
            attempt_generation = int(operation["claim_generation"])
            if attempt_generation > 0:
                public_item_id = derive_operation_public_item_id(
                    turn_id=turn_id,
                    operation_id=operation_id,
                    attempt_generation=attempt_generation,
                )
            if operation["tool_name"] == "update_plan" and frozen_result.get("is_error") is not True:
                structured = frozen_result.get("structured_content")
                provider_revision = structured.get("revision") if isinstance(structured, Mapping) else None
                if (
                    isinstance(provider_revision, bool)
                    or not isinstance(provider_revision, int)
                    or provider_revision < 0
                ):
                    raise RuntimeError("successful update_plan result requires a canonical revision")
                call = store.tx.rows(
                    "items",
                    where={"turn_id": turn_id, "kind": "tool_call", "status": "completed"},
                    order_by="sequence",
                )
                matching_calls = [
                    row["payload"] for row in call if row["payload"].get("tool_call_id") == operation["tool_call_id"]
                ]
                if len(matching_calls) != 1:
                    raise RuntimeError("successful update_plan result requires one canonical ToolCall")
                arguments = matching_calls[0].get("arguments")
                if not isinstance(arguments, Mapping):
                    raise RuntimeError("successful update_plan ToolCall arguments are malformed")
                prior_revisions = (store.tx.count("items", where={"turn_id": turn_id, "kind": "plan_state"}),)[0]
                revision = int(prior_revisions) + 1
                plan_public_item_id = derive_plan_public_item_id(
                    turn_id=turn_id,
                    revision=revision,
                )
                plan_item_id = f"item_{uuid4().hex}"
                raw_steps = arguments.get("plan", [])
                steps: list[dict[str, Any]] = []
                if isinstance(raw_steps, (list, tuple)):
                    for index, raw_step in enumerate(raw_steps, start=1):
                        if not isinstance(raw_step, Mapping):
                            continue
                        status = raw_step.get("status")
                        normalized_status: Literal["pending", "in_progress", "completed"]
                        if status in {"pending", "completed"}:
                            normalized_status = status
                        elif status == "in_progress":
                            normalized_status = "in_progress"
                        else:
                            continue
                        step = PlanStep(
                            step_id=str(raw_step.get("step_id") or f"step_{index}"),
                            title=str(raw_step.get("step", "")),
                            status=normalized_status,
                        )
                        steps.append(step.model_dump())
                objective_row = store.tx.first(
                    "items", where={"turn_id": turn_id, "kind": "user_message"}, order_by="sequence", limit=1
                )
                objective_payload = objective_row["payload"] if objective_row is not None else {}
                plan_snapshot = {
                    "revision": revision,
                    "objective": objective_payload.get("text", "Current task"),
                    "status": "active",
                    "active_step_id": next(
                        (step["step_id"] for step in steps if step["status"] == "in_progress"),
                        next(
                            (step["step_id"] for step in steps if step["status"] == "pending"),
                            None,
                        ),
                    ),
                    "steps": steps,
                    "summary": arguments.get("explanation"),
                }
                plan_event = {
                    "event_type": "llm_update",
                    "message": (
                        arguments.get("explanation")
                        if isinstance(arguments.get("explanation"), str)
                        else "Applied update_plan tool update."
                    ),
                    "related_step_id": plan_snapshot["active_step_id"],
                    "tool_call_ids": [operation["tool_call_id"]],
                }
        started_payload: dict[str, Any] = {
            "item_id": item_id,
            "kind": "tool_result",
            "tool_name": (str(operation["tool_name"]) if operation is not None else frozen_result.get("tool_name")),
            "tool_call_id": (
                str(operation["tool_call_id"]) if operation is not None else frozen_result.get("tool_call_id")
            ),
        }
        if public_item_id is not None:
            started_payload.update(
                {
                    "operation_id": operation_id,
                    "attempt_generation": attempt_generation,
                    "public_item_id": public_item_id,
                }
            )
        facts.append_fact(
            store,
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
            assert operation is not None
            completed_payload.update(
                {
                    "operation_id": operation_id,
                    "attempt_generation": attempt_generation,
                    "public_item_id": public_item_id,
                }
            )
            public_status = {
                "succeeded": "success",
                "cancelled": "cancelled",
                "unknown": "outcome_unknown",
            }.get(str(operation["status"]), "failed")
            completed_payload["status"] = public_status
            if public_status in {"failed", "outcome_unknown"}:
                completed_payload["error"] = str(operation["error_code"] or "tool execution failed")
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="tool",
            payload=completed_payload,
        )
        if (
            plan_item_id is not None
            and plan_public_item_id is not None
            and plan_snapshot is not None
            and plan_event is not None
        ):
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="runtime",
                payload={
                    "item_id": plan_item_id,
                    "kind": "plan_state",
                    "public_item_id": plan_public_item_id,
                    "revision": plan_snapshot["revision"],
                },
            )
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="runtime",
                payload={
                    "item_id": plan_item_id,
                    "public_item_id": plan_public_item_id,
                    "revision": plan_snapshot["revision"],
                    "payload": {
                        "plan": plan_snapshot,
                        "event": plan_event,
                    },
                },
            )
        if operation is not None:
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_result_linked",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "result_item_id": item_id,
                },
            )
    return store._latest_item(turn_id)


@owner_operation
def record_tool_continuation(
    store: RuntimeStore,
    *,
    operation_id: str,
    key: str,
    result: Mapping[str, Any],
) -> ItemSnapshot:
    """Persist the owner's confirmed terminal result when its live scope closes."""
    operation = store.read_tool_operation(operation_id)
    if operation.result_item_id is None:
        raise RuntimeError("continuation requires a committed launch result")
    for item in store.list_items(operation.turn_id):
        if item.kind == "tool_continuation" and item.payload.get("operation_id") == operation_id:
            return item
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        facts.append_fact(
            store,
            thread_id=operation.thread_id,
            turn_id=operation.turn_id,
            record_type="item_started",
            producer="runtime",
            payload={"item_id": item_id, "kind": "tool_continuation"},
        )
        facts.append_fact(
            store,
            thread_id=operation.thread_id,
            turn_id=operation.turn_id,
            record_type="item_completed",
            producer="runtime",
            payload={
                "item_id": item_id,
                "payload": {
                    **dict(result),
                    "operation_id": operation_id,
                    "key": key,
                    "source_result_item_id": operation.result_item_id,
                },
            },
        )
    return store.read_item(item_id)


@owner_operation
def record_verification(
    store: RuntimeStore,
    *,
    turn_id: str,
    operation_id: str,
    kind: str,
    verifier: str,
    verified_resources: tuple[str, ...] = (),
) -> ItemSnapshot:
    if kind not in {"test", "static_analysis", "assertion", "inspection"}:
        raise ValueError(f"unsupported verification kind: {kind}")
    if not isinstance(verifier, str) or not verifier.strip():
        raise ValueError("verification verifier must be non-empty")
    with store.tx.transaction():
        operation = store.tx.first("tool_operations", where={"operation_id": operation_id, "turn_id": turn_id})
        if operation is None:
            raise KeyError(f"unknown tool operation: {operation_id}")
        if operation["status"] != "succeeded" or operation["result_item_id"] is None:
            raise RuntimeError("verification requires a committed successful ToolResult")
        existing_rows = store.tx.rows(
            "items", where={"turn_id": turn_id, "kind": "verification", "status": "completed"}, order_by="sequence"
        )
        for row in existing_rows:
            payload = row["payload"]
            if payload.get("operation_id") == operation_id:
                return _item_snapshot(row)

        item_id = f"item_{uuid4().hex}"
        payload = {
            "operation_id": operation_id,
            "tool_call_id": operation["tool_call_id"],
            "arguments_digest": operation["arguments_digest"],
            "source_result_item_id": operation["result_item_id"],
            "verification_kind": kind,
            "verifier": verifier.strip(),
            "verified_resources": list(verified_resources),
        }
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=turn_id,
            record_type="item_started",
            producer="runtime",
            payload={"item_id": item_id, "kind": "verification"},
        )
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=turn_id,
            record_type="item_completed",
            producer="runtime",
            payload={"item_id": item_id, "payload": payload},
        )
    return store.read_item(item_id)


@owner_operation
def record_workspace_checkpoint(
    store: RuntimeStore,
    *,
    turn_id: str,
    checkpoint_id: str,
    trigger: str,
    tree_sha256: str | None = None,
    changed_file_count: int = 0,
    wall_time: float = 0.0,
) -> ItemSnapshot:
    """Persist the Turn's workspace checkpoint; one pre_write item per Turn."""
    if trigger not in {"pre_write", "restore"}:
        raise ValueError(f"unsupported checkpoint trigger: {trigger}")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError("checkpoint_id must be a non-empty string")
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        existing_rows = store.tx.rows(
            "items",
            where={"turn_id": turn_id, "kind": "workspace_checkpoint", "status": "completed"},
            order_by="sequence",
        )
        for row in existing_rows:
            payload = row["payload"]
            if payload.get("checkpoint_id") == checkpoint_id or (
                trigger == "pre_write" and payload.get("trigger") == "pre_write"
            ):
                return _item_snapshot(row)
        item_id = f"item_{uuid4().hex}"
        payload = {
            "checkpoint_id": checkpoint_id,
            "trigger": trigger,
            "tree_sha256": tree_sha256,
            "changed_file_count": changed_file_count,
            "wall_time": wall_time,
        }
        facts.append_fact(
            store,
            thread_id=turn["thread_id"],
            turn_id=turn_id,
            record_type="item_started",
            producer="runtime",
            payload={"item_id": item_id, "kind": "workspace_checkpoint"},
        )
        facts.append_fact(
            store,
            thread_id=turn["thread_id"],
            turn_id=turn_id,
            record_type="item_completed",
            producer="runtime",
            payload={"item_id": item_id, "payload": payload},
        )
    return store.read_item(item_id)

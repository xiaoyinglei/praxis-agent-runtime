"""Explicit model ops over owner-serialized durable facts."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from uuid import uuid4

from agent_runtime.budget import (
    BudgetLimitExceededError,
    ReservationStatus,
    ResourceUsage,
    ensure_token_headroom,
)
from agent_runtime.budget import (
    reserve as reserve_budget,
)
from agent_runtime.core.messages import canonical_json_text
from agent_runtime.harness import artifact_ops, budget_ops, facts, turn_ops
from agent_runtime.harness.operation_values import TurnInputChangedError, _canonical_json, _json_object
from agent_runtime.harness.protocol import (
    HarnessModelResponse,
    model_response_payload,
)
from agent_runtime.harness.records import (
    ModelAttemptSnapshot,
    ModelOperationSnapshot,
    _model_operation_snapshot,
)
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation
from agent_runtime.modeling.events import ModelOutput
from agent_runtime.modeling.failures import ModelFailure
from agent_runtime.streaming.events import (
    derive_model_public_item_id,
)


@owner_operation
def validate_turn_model_requests(store: RuntimeStore, turn_id: str) -> None:
    """Reject unsupported persisted requests before recovery changes state."""
    for operation in store.list_model_operations(turn_id):
        store.request_input_revision(turn_id, operation.request_ref)


@owner_operation
def prepare_model_operation(
    store: RuntimeStore,
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
    raw_ref = dict(request_ref)
    raw_payload = raw_ref.pop("frozen_request_payload", None)
    if raw_payload is not None and not isinstance(raw_payload, Mapping):
        raise TypeError("frozen request payload must be an object")
    # Prepared request snapshots freeze nested JSON into immutable mappings and
    # tuples. Normalize that existing JSON representation before artifact I/O.
    frozen_payload = None if raw_payload is None else json.loads(canonical_json_text(raw_payload))
    frozen_ref = _json_object(raw_ref, field="request_ref")
    store.request_input_revision(turn_id, frozen_ref)
    request_id = frozen_ref.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("model request_ref.request_id must be a non-empty string")
    operation_id = f"modelop_{uuid4().hex}"
    attempt_id = f"attempt_{uuid4().hex}"
    request_item_id = f"item_{uuid4().hex}"
    selected_operation_id = operation_id
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "running":
            raise RuntimeError(f"turn is not running: {turn_id}")
        thread_id = turn["thread_id"]
        matches = [
            row
            for row in store.tx.rows("model_operations", where={"turn_id": turn_id}, order_by="rowid")
            if row["request_ref"].get("request_id") == request_id
        ]
        if len(matches) > 1:
            raise RuntimeError(f"duplicate durable model request identity: {request_id}")
        if matches:
            existing = matches[0]
            existing_hashes = {name: existing[name] for name in hashes}
            existing_ref = dict(existing["request_ref"])
            existing_artifact = existing_ref.pop("frozen_request_artifact", None)
            incoming_artifact = frozen_ref.pop("frozen_request_artifact", None)
            if incoming_artifact is not None and incoming_artifact != existing_artifact:
                raise RuntimeError(f"model request identity reused with conflicting frozen artifact: {request_id}")
            if frozen_payload is not None:
                if (
                    existing_artifact is None
                    or json.loads(store.read_artifact(str(existing_artifact))) != frozen_payload
                ):
                    raise RuntimeError(f"model request identity reused with conflicting frozen payload: {request_id}")
            if existing_hashes != hashes or existing_ref != frozen_ref:
                raise RuntimeError(f"model request identity reused with conflicting payload: {request_id}")
            selected_operation_id = str(existing["operation_id"])
        else:
            if frozen_payload is not None:
                if "frozen_request_artifact" in frozen_ref:
                    raise ValueError("new model request must provide payload or artifact identity, not both")
                artifact = artifact_ops.commit_artifact(
                    store,
                    turn_id=turn_id,
                    content=_canonical_json(frozen_payload).encode("utf-8"),
                    media_type="application/json",
                    name="frozen-model-request.json",
                )
                frozen_ref["frozen_request_artifact"] = artifact.artifact_id
            elif "frozen_request_artifact" in frozen_ref:
                store.read_artifact(str(frozen_ref["frozen_request_artifact"]))
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="model_operation_prepared",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "attempt_id": attempt_id,
                    "generation": 1,
                    **hashes,
                    "request_ref": frozen_ref,
                },
            )
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="runtime",
                payload={"item_id": request_item_id, "kind": "model_request"},
            )
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="runtime",
                payload={
                    "item_id": request_item_id,
                    "payload": {**hashes, "request_ref": frozen_ref},
                },
            )
    row = store.tx.first("model_operations", where={"operation_id": selected_operation_id})
    if row is None:
        raise RuntimeError("prepared model operation projection is missing")
    return _model_operation_snapshot(row)


@owner_operation
def supersede_prepared_model(store: RuntimeStore, operation_id: str) -> None:
    """Close an undispatched stale request without reserving or spending budget."""
    with store.tx.transaction():
        operation = next(op for op in store.list_model_operations() if op.operation_id == operation_id)
        if operation.status != "prepared":
            raise RuntimeError("only an undispatched prepared model can be superseded")
        facts.append_fact(
            store,
            thread_id=operation.thread_id,
            turn_id=operation.turn_id,
            record_type="model_attempt_rejected",
            producer="runtime",
            payload={
                "operation_id": operation_id,
                "attempt_id": operation.active_attempt_id,
                "generation": operation.generation,
                "reason": "new user input before dispatch",
                "error_type": "input_superseded",
            },
        )


@owner_operation
def dispatch_model_attempt(
    store: RuntimeStore,
    operation_id: str,
    *,
    worker_id: str = "direct-model-worker",
    lease_seconds: float = 300.0,
    now: float | None = None,
    resource_request: ResourceUsage | None = None,
    allow_protected_budget: bool = False,
) -> ModelAttemptSnapshot:
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise ValueError("model worker_id must be non-empty")
    if lease_seconds <= 0:
        raise ValueError("model lease_seconds must be positive")
    if resource_request is not None and not isinstance(resource_request, ResourceUsage):
        raise TypeError("resource_request must be ResourceUsage or None")
    if type(allow_protected_budget) is not bool:
        raise TypeError("allow_protected_budget must be a bool")
    row = store.tx.first("model_operations", where={"operation_id": operation_id})
    if row is None:
        raise KeyError(f"unknown model operation: {operation_id}")
    artifact_ops._assert_artifact_integrity(store, str(row["turn_id"]))
    dispatched_at = time.time() if now is None else float(now)
    with store.tx.transaction():
        operation = store.tx.first("model_operations", where={"operation_id": operation_id})
        if operation is None:
            raise KeyError(f"unknown model operation: {operation_id}")
        if operation["status"] != "prepared":
            raise RuntimeError(f"model operation is not prepared: {operation_id}")
        from agent_runtime.harness.model_retry_ops import validate_scheduled_dispatch

        validate_scheduled_dispatch(store, operation_id, dispatched_at)
        if resource_request is not None and int(operation["generation"]) > 1:
            resource_request = replace(resource_request, retries=1)
        turn_ops._assert_no_pending_cancellation(store, str(operation["turn_id"]))
        request_ref = operation["request_ref"]
        revision = store.request_input_revision(str(operation["turn_id"]), request_ref)
        if request_ref.get("purpose") == "agent_step" and revision != store.user_input_revision(
            str(operation["turn_id"])
        ):
            raise TurnInputChangedError("new user input before model dispatch")
        plan = request_ref.get("compaction_plan")
        if request_ref.get("purpose") == "context_summary" and plan is not None:
            summary_ops = [
                op
                for op in store.list_model_operations(str(operation["turn_id"]))
                if op.request_ref.get("purpose") == "context_summary"
            ]
            for limit_key, selected in (
                ("max_summary_calls", summary_ops),
                (
                    "planned_summary_calls",
                    [
                        op
                        for op in summary_ops
                        if (
                            op.request_ref.get("compaction_plan", {}).get("job_id") == plan["job_id"]
                            if plan.get("job_id")
                            else op.request_ref.get("compaction_plan", {}).get("plan_id") == plan.get("plan_id")
                        )
                    ],
                ),
            ):
                limit = plan.get(limit_key)
                if limit is None:  # Legacy prepared records may lack the new call bound.
                    continue
                if type(limit) is not int or limit < 1:
                    raise ValueError("summary call allowance must be a positive integer")
                if limit_key == "planned_summary_calls" and plan.get("job_id"):
                    from agent_runtime.harness.compaction_jobs import retry_admission_count

                    limit += retry_admission_count(store, turn_id=str(operation["turn_id"]), job_id=plan["job_id"])
                dispatched_ids = {
                    record.payload["attempt_id"]
                    for record in store.list_turn_records(str(operation["turn_id"]))
                    if record.record_type == "model_attempt_dispatched"
                }
                dispatched = sum(
                    attempt.attempt_id in dispatched_ids
                    for op in selected
                    for attempt in store.list_model_attempts(op.operation_id)
                )
                if dispatched >= limit:
                    raise BudgetLimitExceededError(
                        resource="model_calls", limit=limit, current_exposure=dispatched, requested=1
                    )
        reservation_id: str | None = None
        attempt_id = str(operation["active_attempt_id"])
        existing = budget_ops._budget_reservation_row_for_attempt(store, attempt_id)
        if existing is not None:
            if existing["status"] != ReservationStatus.RESERVED.value:
                raise RuntimeError("prepared attempt has a non-reserved budget allocation")
            if existing["operation_id"] != operation_id:
                raise RuntimeError("prepared attempt budget belongs to another operation")
            if resource_request is not None and ResourceUsage.from_dict(existing["reserved"]) != resource_request:
                raise RuntimeError("prepared attempt budget differs from its frozen resource request")
            reservation_id = str(existing["reservation_id"])
        elif resource_request is not None:
            reservation_id = f"reservation_{attempt_id}"
            job_id = request_ref.get("compaction_job_id")
            if job_id is None and isinstance(plan, Mapping):
                job_id = plan.get("job_id")
            if job_id is not None:
                from agent_runtime.harness.compaction_jobs import transfer_for_attempt

                transfer_for_attempt(
                    store,
                    turn_id=str(operation["turn_id"]),
                    job_id=str(job_id),
                    amount=resource_request,
                    purpose=str(request_ref.get("purpose", "agent_step")),
                )
            budget_state = store.read_budget_state(str(operation["turn_id"]))
            continuation_reserve = 0
            if request_ref.get("purpose") == "context_summary" and plan is not None and job_id is None:
                continuation_reserve = plan.get("continuation_token_reserve")
                if type(continuation_reserve) is not int or continuation_reserve < 1:
                    raise ValueError("summary plan must preserve a positive continuation token reserve")
                continuation_cost = plan.get("continuation_cost_reserve", 0)
                if type(continuation_cost) is not int or continuation_cost < 0:
                    raise ValueError("summary continuation cost reserve must be non-negative")
                cost_limit = budget_state.limits.cost_micros
                requested_cost = resource_request.cost_micros + continuation_cost
                if cost_limit is not None and budget_state.exposure.cost_micros + requested_cost > cost_limit:
                    raise BudgetLimitExceededError(
                        resource="cost_micros",
                        limit=cost_limit,
                        current_exposure=budget_state.exposure.cost_micros,
                        requested=requested_cost,
                    )
            ensure_token_headroom(
                budget_state,
                requested_tokens=resource_request.total_tokens + continuation_reserve,
                allow_protected=allow_protected_budget and not continuation_reserve,
            )
            mutation = reserve_budget(
                budget_state,
                reservation_id=reservation_id,
                operation_id=operation_id,
                attempt_id=attempt_id,
                amount=resource_request,
            )
            facts.append_fact(
                store,
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="budget_reserved",
                producer="runtime",
                payload={
                    "reservation_id": reservation_id,
                    "scope_id": mutation.reservation.scope_id,
                    "operation_id": operation_id,
                    "attempt_id": attempt_id,
                    "reserved": resource_request.to_dict(),
                },
            )
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="model_attempt_dispatched",
            producer="runtime",
            payload={
                "operation_id": operation_id,
                "attempt_id": operation["active_attempt_id"],
                "generation": operation["generation"],
                "claim_owner": worker_id,
                "dispatched_at": dispatched_at,
                "lease_expires_at": dispatched_at + lease_seconds,
            },
        )
        if reservation_id is not None:
            facts.append_fact(
                store,
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="budget_dispatched",
                producer="runtime",
                payload={"reservation_id": reservation_id},
            )
    return store.list_model_attempts(operation_id)[-1]


@owner_operation
def expire_model_attempt_dispatch(
    store: RuntimeStore,
    *,
    operation_id: str,
    now: float | None = None,
) -> ModelAttemptSnapshot:
    observed_at = time.time() if now is None else float(now)
    with store.tx.transaction():
        operation = store.tx.first("model_operations", where={"operation_id": operation_id})
        if operation is None:
            raise KeyError(f"unknown model operation: {operation_id}")
        if operation["status"] != "dispatched":
            raise RuntimeError("only a dispatched model attempt can expire")
        attempt = store.tx.first("model_attempts", where={"attempt_id": operation["active_attempt_id"]})
        if attempt is None:
            raise RuntimeError("active model attempt projection is missing")
        expires_at = attempt["lease_expires_at"]
        if expires_at is None or observed_at <= float(expires_at):
            raise RuntimeError("model attempt dispatch lease has not expired")
        facts.append_fact(
            store,
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
        budget_ops._append_budget_outcome(
            store,
            operation=operation,
            attempt_id=str(attempt["attempt_id"]),
            outcome="unknown",
            producer="recovery",
        )
        _close_started_model_outputs(
            store,
            attempt_id=str(attempt["attempt_id"]),
            status="outcome_unknown",
            error="model dispatch lease expired",
        )
        turn = store.tx.first("turns", where={"turn_id": operation["turn_id"]})
        if turn is not None and turn["status"] == "running":
            facts.append_fact(
                store,
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="turn_interrupted",
                producer="recovery",
                payload={
                    "turn_id": operation["turn_id"],
                    "operation_id": operation_id,
                    "reason": "model outcome unknown after lease expiry",
                },
            )
    return store.list_model_attempts(operation_id)[-1]


@owner_operation
def mark_model_attempt_unknown(
    store: RuntimeStore,
    *,
    operation_id: str,
    attempt_id: str,
    generation: int,
    error_type: str | None = None,
    error_message: str | None = None,
    outputs: tuple[ModelOutput, ...] = (),
) -> ModelAttemptSnapshot:
    if error_type is not None and (not isinstance(error_type, str) or not error_type.strip()):
        raise ValueError("model dispatch error_type must be non-empty")
    if error_message is not None and (not isinstance(error_message, str) or not error_message.strip()):
        raise ValueError("model dispatch error_message must be non-empty")
    with store.tx.transaction():
        operation = store.tx.first("model_operations", where={"operation_id": operation_id})
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
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="model_attempt_unknown",
            producer="runtime",
            payload=payload,
        )
        budget_ops._append_budget_outcome(
            store,
            operation=operation,
            attempt_id=attempt_id,
            outcome="unknown",
        )
        _close_started_model_outputs(
            store,
            attempt_id=attempt_id,
            status="outcome_unknown",
            error=(error_message.strip()[:2_000] if error_message is not None else "model outcome is unknown"),
            outputs=outputs,
        )
        turn = store.tx.first("turns", where={"turn_id": operation["turn_id"]})
        if turn is not None and turn["status"] == "running":
            facts.append_fact(
                store,
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="turn_interrupted",
                producer="runtime",
                payload={
                    "turn_id": operation["turn_id"],
                    "operation_id": operation_id,
                    "reason": "outcome_unknown",
                },
            )
    return store.list_model_attempts(operation_id)[-1]


@owner_operation
def reject_model_attempt(
    store: RuntimeStore,
    *,
    operation_id: str,
    attempt_id: str,
    generation: int,
    reason: str,
    error_type: str | None = None,
    error_message: str | None = None,
    outputs: tuple[ModelOutput, ...] = (),
    budget_outcome: str = "released",
    failure: ModelFailure | None = None,
) -> ModelAttemptSnapshot:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("model rejection reason must be non-empty")
    if budget_outcome not in {"released", "unknown", "settled"}:
        raise ValueError("model rejection budget_outcome is unsupported")
    if error_type is not None and (not isinstance(error_type, str) or not error_type.strip()):
        raise ValueError("model rejection error_type must be non-empty")
    if error_message is not None and (not isinstance(error_message, str) or not error_message.strip()):
        raise ValueError("model rejection error_message must be non-empty")
    with store.tx.transaction():
        operation = store.tx.first("model_operations", where={"operation_id": operation_id})
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
        if failure is not None:
            payload["failure"] = failure.to_payload()
            payload["error_type"] = failure.kind
            payload["error_message"] = failure.message
        if error_type is not None:
            payload["error_type"] = error_type.strip()[:120]
        if error_message is not None:
            payload["error_message"] = error_message.strip()[:2_000]
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="model_attempt_rejected",
            producer="runtime",
            payload=payload,
        )
        budget_ops._append_budget_outcome(
            store,
            operation=operation,
            attempt_id=attempt_id,
            outcome=budget_outcome,
            usage={} if budget_outcome == "settled" else None,
        )
        _close_started_model_outputs(
            store,
            attempt_id=attempt_id,
            status="failed",
            error=(error_message.strip()[:2_000] if error_message is not None else reason.strip()[:2_000]),
            outputs=outputs,
        )
    return store.list_model_attempts(operation_id)[-1]


def _close_started_model_outputs(
    store: RuntimeStore,
    *,
    attempt_id: str,
    status: str,
    error: str,
    outputs: tuple[ModelOutput, ...] = (),
) -> None:
    from agent_runtime.harness.model_outputs import close_model_outputs

    close_model_outputs(store, attempt_id=attempt_id, status=status, error=error, partial_outputs=outputs)


@owner_operation
def cancel_model_attempt(
    store: RuntimeStore,
    *,
    operation_id: str,
    attempt_id: str,
    generation: int,
    reason: str,
    outputs: tuple[ModelOutput, ...] = (),
) -> ModelAttemptSnapshot:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("model cancellation reason must be non-empty")
    with store.tx.transaction():
        operation = store.tx.first("model_operations", where={"operation_id": operation_id})
        if operation is None:
            raise KeyError(f"unknown model operation: {operation_id}")
        if not (
            operation["status"] == "dispatched"
            and operation["active_attempt_id"] == attempt_id
            and operation["generation"] == generation
        ):
            raise RuntimeError("only the active dispatched model attempt can be cancelled")
        facts.append_fact(
            store,
            thread_id=str(operation["thread_id"]),
            turn_id=str(operation["turn_id"]),
            record_type="model_attempt_cancelled",
            producer="runtime",
            payload={
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "generation": generation,
                "reason": reason.strip()[:2_000],
            },
        )
        # Provider-side cancellation does not prove zero billing. Keep the
        # reservation as uncertain exposure unless usage is later reconciled.
        budget_ops._append_budget_outcome(
            store,
            operation=operation,
            attempt_id=attempt_id,
            outcome="unknown",
        )
        _close_started_model_outputs(
            store,
            attempt_id=attempt_id,
            status="cancelled",
            error=reason.strip()[:2_000],
            outputs=outputs,
        )
        facts.append_fact(
            store,
            thread_id=str(operation["thread_id"]),
            turn_id=str(operation["turn_id"]),
            record_type="turn_cancelled",
            producer="runtime",
            payload={
                "turn_id": str(operation["turn_id"]),
                "reason": reason.strip()[:2_000],
            },
        )
    return store.list_model_attempts(operation_id)[-1]


@owner_operation
def prepare_model_retry(
    store: RuntimeStore, operation_id: str, *, automatic: bool = False, now: float | None = None
) -> ModelAttemptSnapshot:
    row = store.tx.first("model_operations", where={"operation_id": operation_id})
    if row is None:
        raise KeyError(f"unknown model operation: {operation_id}")
    artifact_ops._assert_artifact_integrity(store, str(row["turn_id"]))
    with store.tx.transaction():
        operation = store.tx.first("model_operations", where={"operation_id": operation_id})
        if operation is None:
            raise KeyError(f"unknown model operation: {operation_id}")
        if operation["status"] != ("retry_wait" if automatic else "unknown"):
            raise RuntimeError(f"model operation is not eligible for retry: {operation_id}")
        attempt_id = f"attempt_{uuid4().hex}"
        generation = operation["generation"] + 1
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="model_retry_prepared",
            producer="runtime",
            payload={
                "operation_id": operation_id,
                "previous_attempt_id": operation["active_attempt_id"],
                "automatic": automatic,
                "prepared_at": time.time() if now is None else float(now),
                "attempt_id": attempt_id,
                "generation": generation,
            },
        )
        turn = store.tx.first("turns", where={"turn_id": operation["turn_id"]})
        if automatic:
            if turn is None or turn["status"] != "running":
                raise RuntimeError("automatic model retry requires a running Turn")
            return store.list_model_attempts(operation_id)[-1]
        if turn is None or turn["status"] not in {"paused", "interrupted"}:
            raise RuntimeError("model retry requires a paused or interrupted Turn")
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="turn_resumed",
            producer="runtime",
            payload={
                "turn_id": operation["turn_id"],
                "operation_id": operation_id,
            },
        )
    return store.list_model_attempts(operation_id)[-1]


@owner_operation
def complete_model_attempt(
    store: RuntimeStore,
    *,
    operation_id: str,
    attempt_id: str,
    generation: int,
    response: HarnessModelResponse,
    now: float | None = None,
) -> bool:
    if response.status == "completed" and not response.text.strip() and not response.tool_calls:
        raise ValueError("model response must contain text or tool calls")
    response_payload = model_response_payload(response)
    frozen_usage = _json_object(response.usage, field="usage")
    observed_at = time.time() if now is None else float(now)
    with store.tx.transaction():
        operation = store.tx.first("model_operations", where={"operation_id": operation_id})
        if operation is None:
            raise KeyError(f"unknown model operation: {operation_id}")
        # Durable state is the fencing authority. Lease expiry only opens
        # the recovery takeover (expire_model_attempt_dispatch), which
        # flips the status to unknown inside the same serialized
        # transaction. While the operation still points at this attempt
        # and generation with no committed response, this completion is
        # authoritative even past the lease clock: a billed response must
        # not be dropped on a timing boundary.
        is_current = (
            operation["status"] == "dispatched"
            and operation["active_attempt_id"] == attempt_id
            and operation["generation"] == generation
            and operation["response_item_id"] is None
        )
        if not is_current:
            return False
        internal_summary = operation["request_ref"].get("purpose") == "context_summary"
        if not internal_summary:
            from agent_runtime.harness.model_outputs import finish_model_outputs

            finish_model_outputs(
                store,
                operation_id=operation_id,
                attempt_id=attempt_id,
                generation=generation,
                outputs=response.outputs,
            )
        response_item_id = f"item_{uuid4().hex}"
        public_item_id = derive_model_public_item_id(
            turn_id=str(operation["turn_id"]),
            model_attempt_id=attempt_id,
            channel="agent_message",
        )
        public_metadata = (
            {}
            if internal_summary
            else {
                "channel": "agent_message",
                "public_item_id": public_item_id,
                "output_protocol": "model-outputs-v1",
            }
        )
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="model_attempt_completed",
            producer="runtime",
            payload={
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "generation": generation,
                "observed_at": observed_at,
                "provider_response_id": response.provider_response_id,
                "usage": frozen_usage,
                "response_item_id": response_item_id,
                "response_status": response.status,
                "incomplete_reason": response.incomplete_reason,
                "stop_reason": response.stop_reason,
                **({} if internal_summary else {"public_item_id": public_item_id}),
            },
        )
        budget_ops._append_budget_outcome(
            store,
            operation=operation,
            attempt_id=attempt_id,
            outcome="settled",
            usage=frozen_usage,
        )
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="item_started",
            producer="model",
            payload={
                "item_id": response_item_id,
                "kind": "context_summary_response" if internal_summary else "model_response",
                "attempt_id": attempt_id,
                **public_metadata,
            },
        )
        binding = store.read_turn(str(operation["turn_id"])).binding_manifest
        result_order = binding.get("execution_context_policy", {}).get("tool_result_order")
        recovery = binding.get("model_response_recovery")
        explicit_recovery_feedback = (
            not internal_summary
            and response.status == "incomplete"
            and isinstance(recovery, Mapping)
            and recovery.get("revision") == "split-step-v2"
        )
        facts.append_fact(
            store,
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="item_completed",
            producer="model",
            payload={
                "item_id": response_item_id,
                "attempt_id": attempt_id,
                **public_metadata,
                "payload": {
                    **response_payload,
                    **({"explicit_recovery_feedback": True} if explicit_recovery_feedback else {}),
                    **({"tool_result_order": "call-v1"} if result_order == "call-v1" else {}),
                },
            },
        )
    return True

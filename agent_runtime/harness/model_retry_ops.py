"""Atomic model failure settlement, durable scheduling and retry admission."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from agent_runtime.budget import ResourceUsage, ensure_token_headroom, reserve
from agent_runtime.harness import budget_ops, facts, model_ops, turn_ops
from agent_runtime.harness.operation_values import TurnInputChangedError
from agent_runtime.harness.records import ModelAttemptSnapshot
from agent_runtime.harness.retry_policy import RetryPolicy
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation
from agent_runtime.modeling.events import ModelOutput
from agent_runtime.modeling.failures import ModelFailure


def _operation(store: RuntimeStore, operation_id: str) -> Mapping[str, Any]:
    row = store.tx.first("model_operations", where={"operation_id": operation_id})
    if row is None:
        raise KeyError(f"unknown model operation: {operation_id}")
    return row


def _fact(store: RuntimeStore, operation: Mapping[str, Any], kind: str, payload: Mapping[str, Any]) -> None:
    facts.append_fact(
        store,
        thread_id=str(operation["thread_id"]),
        turn_id=str(operation["turn_id"]),
        record_type=kind,
        producer="runtime",
        payload=dict(payload),
    )


def _schedules(store: RuntimeStore, operation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        r.payload
        for r in store.list_turn_records(str(operation["turn_id"]))
        if r.record_type == "model_retry_scheduled" and r.payload["operation_id"] == operation["operation_id"]
    ]


@owner_operation
def read_schedule(store: RuntimeStore, operation_id: str) -> Mapping[str, Any] | None:
    operation = _operation(store, operation_id)
    schedules = _schedules(store, operation)
    if not schedules:
        return None
    schedule = schedules[-1]
    if (
        operation["status"] == "retry_wait"
        and operation["generation"] == schedule["generation"]
        or operation["status"] == "prepared"
        and operation["generation"] == schedule["generation"] + 1
    ):
        return dict(schedule)
    return None


@owner_operation
def record_failure(
    store: RuntimeStore,
    *,
    operation_id: str,
    attempt_id: str,
    generation: int,
    failure: ModelFailure,
    outputs: tuple[ModelOutput, ...] = (),
    now: float | None = None,
    random_fraction: float | None = None,
) -> str:
    observed_at = time.time() if now is None else float(now)
    if not math.isfinite(observed_at):
        raise ValueError("failure observation must be finite")
    with store.tx.transaction():
        operation = _operation(store, operation_id)
        if (
            operation["status"] != "dispatched"
            or operation["active_attempt_id"] != attempt_id
            or operation["generation"] != generation
        ):
            raise RuntimeError("only the active dispatched model attempt can fail")
        uncertain = failure.provider_outcome == "unknown" and (
            failure.kind != "protocol_error" or failure.output_started
        )
        _fact(
            store,
            operation,
            "model_attempt_unknown" if uncertain else "model_attempt_rejected",
            {
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "generation": generation,
                "reason": failure.message,
                "error_type": failure.kind,
                "error_message": failure.message,
                "failure": failure.to_payload(),
            },
        )
        if failure.billing_outcome == "unknown":
            budget_ops._append_budget_outcome(store, operation=operation, attempt_id=attempt_id, outcome="unknown")
        elif failure.provider_outcome == "rejected":
            budget_ops._append_budget_outcome(
                store, operation=operation, attempt_id=attempt_id, outcome="settled", usage={}
            )
        else:
            budget_ops._append_budget_outcome(store, operation=operation, attempt_id=attempt_id, outcome="not_sent")
        model_ops._close_started_model_outputs(
            store,
            attempt_id=attempt_id,
            status="outcome_unknown" if uncertain else "failed",
            error=failure.message,
            outputs=outputs,
        )
        turn = store.read_turn(str(operation["turn_id"]))
        schedules = _schedules(store, operation)
        dispatches = [
            r
            for r in store.list_turn_records(turn.turn_id)
            if r.record_type == "model_attempt_dispatched" and r.payload["operation_id"] == operation_id
        ]
        first_dispatch = dispatches[0]
        first_at = float(first_dispatch.payload.get("dispatched_at", first_dispatch.committed_at_ms / 1000))
        # Every observation remains a floor across process restarts and manual retries.
        clock_floor = max(
            [
                first_at,
                *(float(s["observed_at"]) for s in schedules),
                *(float(r.payload.get("dispatched_at", r.committed_at_ms / 1000)) for r in dispatches),
            ]
        )
        request_ref = operation["request_ref"]
        plan = request_ref.get("compaction_plan")
        standalone_summary = (
            request_ref.get("purpose") == "context_summary"
            and isinstance(plan, Mapping)
            and plan.get("summary_request_mode") == "standalone"
        )
        policy = (
            None
            if standalone_summary and failure.billing_outcome == "unknown"
            else RetryPolicy.from_binding(turn.binding_manifest)
        )
        decision = (
            None
            if policy is None or observed_at < clock_floor
            else policy.decide(
                failure,
                retry_count=len(schedules),
                first_dispatched_at=first_at,
                now=observed_at,
                random_fraction=random.random() if random_fraction is None else random_fraction,
            )
        )
        if decision is not None:
            retry_at, deadline_at, delay_ms = decision
            assert policy is not None
            _fact(
                store,
                operation,
                "model_retry_scheduled",
                {
                    "operation_id": operation_id,
                    "attempt_id": attempt_id,
                    "generation": generation,
                    "retry_number": len(schedules) + 1,
                    "max_retries": policy.max_auto_retries,
                    "retry_at": retry_at,
                    "deadline_at": deadline_at,
                    "delay_ms": delay_ms,
                    "observed_at": observed_at,
                    "failure": failure.to_payload(),
                },
            )
            return "scheduled"
        if uncertain and (failure.kind != "protocol_error" or failure.output_started):
            _fact(
                store,
                operation,
                "turn_interrupted",
                {"turn_id": turn.turn_id, "operation_id": operation_id, "reason": "outcome_unknown"},
            )
            return "paused"
        reason_code: str = failure.kind
        message = failure.message
        if (
            policy is not None
            and failure.retryable
            and not failure.output_started
            and failure.provider_outcome != "unknown"
        ):
            if observed_at < clock_floor:
                reason_code = "model_retry_clock_invalid"
                message = "Automatic retry stopped because the clock moved backwards."
            elif len(schedules) >= policy.max_auto_retries:
                reason_code = "model_retry_exhausted"
                message = "Automatic model retry allowance exhausted."
            else:
                reason_code = "model_retry_deadline_exceeded"
                message = "Automatic retry would exceed its frozen deadline."
        turn_ops.fail_turn(store, turn_id=turn.turn_id, reason_code=reason_code, message=message)
        return "failed"


def _validate_source(store: RuntimeStore, operation: Mapping[str, Any]) -> None:
    turn_id = str(operation["turn_id"])
    turn_ops._assert_no_pending_cancellation(store, turn_id)
    if store.read_turn(turn_id).status != "running":
        raise RuntimeError("model retry requires a running Turn")
    ref = operation["request_ref"]
    if store.request_input_revision(turn_id, ref) != store.user_input_revision(turn_id):
        raise TurnInputChangedError("new user input before model retry dispatch")
    plan = ref.get("compaction_plan")
    if ref.get("purpose") == "context_summary" and isinstance(plan, Mapping) and plan.get("source_hash"):
        from agent_runtime.harness.context import RolloutContextManager
        from agent_runtime.harness.protocol import ContextSourceChangedError

        source_hash, _ = RolloutContextManager(store).semantic_source(
            turn_id,
            use_archived_projection=plan.get("summary_request_mode") == "standalone",
            checkpoint="checkpoint_revision" in plan or "summary_history_start" in plan,
        )
        if source_hash != plan["source_hash"]:
            raise ContextSourceChangedError("Summary source changed before model retry dispatch.")


def _validate_time(schedule: Mapping[str, Any], now: float) -> None:
    if not math.isfinite(now) or now < schedule["observed_at"]:
        raise RuntimeError("clock moved backwards before model retry")
    if now > schedule["deadline_at"]:
        raise RuntimeError("model retry deadline expired")
    if now < schedule["retry_at"]:
        raise RuntimeError("model retry is not due")


@owner_operation
def validate_scheduled_dispatch(store: RuntimeStore, operation_id: str, now: float | None = None) -> None:
    operation = _operation(store, operation_id)
    if operation["generation"] == 1 and operation["status"] == "prepared":
        return  # A first attempt has no automatic retry schedule to validate.
    schedule = read_schedule(store, operation_id)
    if schedule is None:
        return
    prepared_times = [
        float(r.payload["prepared_at"])
        for r in store.list_turn_records(str(operation["turn_id"]))
        if r.record_type == "model_retry_prepared"
        and r.payload.get("automatic") is True
        and r.payload.get("attempt_id") == operation["active_attempt_id"]
        and "prepared_at" in r.payload
    ]
    observed_schedule = {**schedule, "observed_at": max([float(schedule["observed_at"]), *prepared_times])}
    _validate_time(observed_schedule, time.time() if now is None else float(now))
    _validate_source(store, operation)
    if operation["status"] == "prepared":
        reservation = budget_ops._budget_reservation_row_for_attempt(store, str(operation["active_attempt_id"]))
        if reservation is None or reservation["status"] != "reserved":
            raise RuntimeError("prepared retry lacks a fresh budget reservation")


@owner_operation
def prepare_scheduled_retry(
    store: RuntimeStore, *, operation_id: str, resource_request: ResourceUsage, now: float | None = None
) -> ModelAttemptSnapshot:
    with store.tx.transaction():
        operation = _operation(store, operation_id)
        if operation["status"] != "retry_wait":
            raise RuntimeError("model operation is not waiting for retry")
        schedule = read_schedule(store, operation_id)
        if schedule is None:
            raise RuntimeError("model retry schedule is missing")
        _validate_time(schedule, time.time() if now is None else float(now))
        _validate_source(store, operation)
        amount = replace(resource_request, retries=1)
        from agent_runtime.harness.compaction_jobs import admit_retry, transfer_for_attempt

        admit_retry(store, turn_id=str(operation["turn_id"]), operation_id=operation_id, amount=amount)
        attempt = model_ops.prepare_model_retry(store, operation_id, automatic=True, now=now)
        ref = operation["request_ref"]
        plan = ref.get("compaction_plan") or {}
        job_id = ref.get("compaction_job_id") or plan.get("job_id")
        if job_id is not None:
            transfer_for_attempt(
                store,
                turn_id=str(operation["turn_id"]),
                job_id=str(job_id),
                amount=amount,
                purpose=str(ref.get("purpose", "agent_step")),
            )
        state = store.read_budget_state(str(operation["turn_id"]))
        ensure_token_headroom(state, requested_tokens=amount.total_tokens, allow_protected=False)
        reservation_id = f"reservation_{attempt.attempt_id}"
        reserve(
            state,
            reservation_id=reservation_id,
            operation_id=operation_id,
            attempt_id=attempt.attempt_id,
            amount=amount,
        )
        _fact(
            store,
            operation,
            "budget_reserved",
            {
                "reservation_id": reservation_id,
                "scope_id": state.scope_id,
                "operation_id": operation_id,
                "attempt_id": attempt.attempt_id,
                "reserved": amount.to_dict(),
            },
        )
        return attempt


@owner_operation
def cancel_scheduled_retry(
    store: RuntimeStore,
    *,
    operation_id: str,
    reason: str = "model retry cancelled",
    reason_code: str = "model_retry_cancelled",
    cancel_turn: bool = False,
) -> None:
    with store.tx.transaction():
        operation = _operation(store, operation_id)
        if operation["status"] not in {"retry_wait", "prepared"} or read_schedule(store, operation_id) is None:
            raise RuntimeError("only an unstarted scheduled retry can be cancelled")
        if operation["status"] == "prepared":
            budget_ops._append_budget_outcome(
                store, operation=operation, attempt_id=str(operation["active_attempt_id"]), outcome="released"
            )
        _fact(
            store,
            operation,
            "model_retry_cancelled",
            {
                "operation_id": operation_id,
                "attempt_id": operation["active_attempt_id"],
                "generation": operation["generation"],
                "reason": reason,
                "reason_code": reason_code,
            },
        )
        if cancel_turn:
            if store.read_turn(str(operation["turn_id"])).status == "running":
                _fact(
                    store,
                    operation,
                    "turn_interrupted",
                    {"turn_id": operation["turn_id"], "operation_id": operation_id, "reason": reason},
                )
            turn_ops.cancel_turn(store, turn_id=str(operation["turn_id"]))


@owner_operation
def fail_unstarted_retry(store: RuntimeStore, *, operation_id: str, reason_code: str, message: str) -> None:
    with store.tx.transaction():
        operation = _operation(store, operation_id)
        cancel_scheduled_retry(store, operation_id=operation_id, reason=message, reason_code=reason_code)
        turn_ops.fail_turn(store, turn_id=str(operation["turn_id"]), reason_code=reason_code, message=message)

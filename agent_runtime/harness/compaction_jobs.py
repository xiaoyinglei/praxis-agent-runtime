"""Durable compaction jobs and allowances in the canonical runtime ledger.

Job and node snapshots are runtime-owned Items. Allowances are ordinary budget
reservations, so children, independent workers and unknown attempts observe the
same exposure. No second budget balance or mutable job database exists.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from agent_runtime.budget import ResourceUsage, ensure_token_headroom, reserve
from agent_runtime.harness import facts, model_ops
from agent_runtime.harness.protocol import ContextBudgetExceededError, ContextSourceChangedError, PreparedModelCall
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation


@dataclass(frozen=True, slots=True)
class CompactionJob:
    """source_revision is the user-input counter, not the full context CAS.

    The admitted full context version is frozen in plan.context_source_revision.
    Final installation additionally validates a freshly measured candidate's full
    source revision; job bookkeeping and model receipts advance that version.
    """

    job_id: str
    turn_id: str
    source_revision: int
    status: str
    plan: Mapping[str, Any]
    nodes: Mapping[str, Mapping[str, Any]]
    continuation: Mapping[str, Any] | None = None


def _payload(job: CompactionJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "source_revision": job.source_revision,
        "status": job.status,
        "plan": dict(job.plan),
        "nodes": dict(job.nodes),
        "continuation": job.continuation,
    }


def _record(store: RuntimeStore, job: CompactionJob) -> None:
    item_id = "item_" + uuid4().hex
    thread_id = store.read_turn(job.turn_id).thread_id
    facts.append_fact(
        store,
        thread_id=thread_id,
        turn_id=job.turn_id,
        record_type="item_started",
        producer="runtime",
        payload={"item_id": item_id, "kind": "compaction_job"},
    )
    facts.append_fact(
        store,
        thread_id=thread_id,
        turn_id=job.turn_id,
        record_type="item_completed",
        producer="runtime",
        payload={"item_id": item_id, "payload": _payload(job)},
    )


@owner_operation
def read_job(store: RuntimeStore, *, turn_id: str, job_id: str) -> CompactionJob:
    for item in reversed(store.list_items(turn_id)):
        if item.kind == "compaction_job" and item.status == "completed" and item.payload.get("job_id") == job_id:
            value = item.payload
            return CompactionJob(
                job_id,
                turn_id,
                value["source_revision"],
                value["status"],
                value["plan"],
                value["nodes"],
                value.get("continuation"),
            )
    raise KeyError(f"unknown compaction job: {job_id}")


def _hold_rows(store: RuntimeStore, job_id: str, bucket: str | None = None) -> list[Mapping[str, Any]]:
    return [
        row
        for row in store.tx.rows(
            "budget_reservations", where={"operation_id": f"compaction:{job_id}"}, order_by="rowid"
        )
        if row["status"] == "reserved"
        and (bucket is None or str(row["attempt_id"]).startswith(f"compaction:{job_id}:{bucket}:"))
    ]


def _fact(store: RuntimeStore, turn_id: str, record_type: str, payload: Mapping[str, Any]) -> None:
    facts.append_fact(
        store,
        thread_id=store.read_turn(turn_id).thread_id,
        turn_id=turn_id,
        record_type=record_type,
        producer="runtime",
        payload=dict(payload),
    )


def _reserve_hold(store: RuntimeStore, turn_id: str, job_id: str, bucket: str, amount: ResourceUsage) -> None:
    if amount == ResourceUsage():
        return
    attempt_id = f"compaction:{job_id}:{bucket}:{uuid4().hex}"
    reservation_id = f"reservation_{attempt_id}"
    state = store.read_budget_state(turn_id)
    ensure_token_headroom(state, requested_tokens=amount.total_tokens, allow_protected=False)
    reserve(
        state, reservation_id=reservation_id, operation_id=f"compaction:{job_id}", attempt_id=attempt_id, amount=amount
    )
    _fact(
        store,
        turn_id,
        "budget_reserved",
        {
            "reservation_id": reservation_id,
            "scope_id": turn_id,
            "operation_id": f"compaction:{job_id}",
            "attempt_id": attempt_id,
            "reserved": amount.to_dict(),
        },
    )


@owner_operation
def create_job(store: RuntimeStore, *, turn_id: str, plan: Mapping[str, Any], source_revision: int) -> CompactionJob:
    """Admit all still-unspent nodes and one continuation atomically."""
    frozen = copy.deepcopy(dict(plan))
    with store.tx.transaction():
        # An abandoned admission has no live allowance. Preserve its immutable
        # identity and admit a fresh generation instead of reusing empty holds.
        generation = 0
        while True:
            identity = json.dumps([turn_id, frozen["plan_id"], source_revision, generation], sort_keys=True)
            job_id = hashlib.sha256(identity.encode()).hexdigest()
            try:
                existing = read_job(store, turn_id=turn_id, job_id=job_id)
            except KeyError:
                break
            if existing.status in {"planned", "running", "candidate_ready", "waiting_unknown", "installed"}:
                return existing
            generation += 1
        continuation = frozen["continuation_token_reserve"]
        total = frozen["planned_token_upper_bound"]
        continuation_cost = frozen.get("continuation_cost_reserve", 0)
        total_cost = frozen.get("planned_cost_upper_bound", continuation_cost)
        if total < continuation or total_cost < continuation_cost:
            raise ValueError("compaction allowance cannot be negative")
        _reserve_hold(
            store,
            turn_id,
            job_id,
            "summary",
            ResourceUsage(
                input_tokens=total - continuation,
                cost_micros=total_cost - continuation_cost,
                model_calls=frozen["planned_summary_calls"],
            ),
        )
        _reserve_hold(
            store,
            turn_id,
            job_id,
            "continuation",
            ResourceUsage(input_tokens=continuation, cost_micros=continuation_cost, model_calls=1),
        )
        frozen["job_id"] = job_id
        job = CompactionJob(job_id, turn_id, source_revision, "planned", frozen, {})
        _record(store, job)
        return job


def _check_completed_call_allowances(store: RuntimeStore, *, turn_id: str, job_id: str) -> None:
    operations = {
        operation.operation_id
        for operation in store.list_model_operations(turn_id)
        if (
            operation.request_ref.get("compaction_job_id")
            or (operation.request_ref.get("compaction_plan") or {}).get("job_id")
        )
        == job_id
    }
    for row in store.tx.rows("budget_reservations", where={"turn_id": turn_id}, order_by="rowid"):
        if row["operation_id"] not in operations or row["status"] != "settled":
            continue
        actual, reserved = ResourceUsage.from_dict(row["actual"]), ResourceUsage.from_dict(row["reserved"])
        if actual.total_tokens > reserved.total_tokens or (
            store.read_budget_state(turn_id).limits.cost_micros is not None
            and actual.cost_micros > reserved.cost_micros
        ):
            raise ContextBudgetExceededError(
                "Provider usage exceeded its frozen compaction node allowance; actual usage remains recorded.",
                reason_code="compaction_node_allowance_exceeded",
            )


@owner_operation
def transfer_for_attempt(
    store: RuntimeStore, *, turn_id: str, job_id: str, amount: ResourceUsage, purpose: str
) -> None:
    """Consume a hold immediately before its ordinary attempt reservation.

    The caller owns the enclosing claim transaction. Unknown attempts retain
    their own reservation; this command never credits an attempted node back.
    """
    with store.tx.transaction():
        job = read_job(store, turn_id=turn_id, job_id=job_id)
        _check_completed_call_allowances(store, turn_id=turn_id, job_id=job_id)
        if job.status not in {"planned", "running", "candidate_ready", "waiting_unknown"}:
            raise ContextBudgetExceededError("Compaction job allowance is no longer active.")
        bucket = "summary" if purpose == "context_summary" else "continuation"
        rows = _hold_rows(store, job_id, bucket)
        available = ResourceUsage()
        for row in rows:
            available += ResourceUsage.from_dict(row["reserved"])
        if (
            amount.total_tokens > available.total_tokens
            or (
                store.read_budget_state(turn_id).limits.cost_micros is not None
                and amount.cost_micros > available.cost_micros
            )
            or amount.model_calls > available.model_calls
        ):
            raise ContextBudgetExceededError(
                "Request exceeds its frozen compaction allowance.", reason_code="compaction_allowance_exhausted"
            )
        for row in rows:
            _fact(store, turn_id, "budget_released", {"reservation_id": row["reservation_id"]})
        residual = ResourceUsage(
            input_tokens=available.total_tokens - amount.total_tokens,
            cost_micros=max(0, available.cost_micros - amount.cost_micros),
            model_calls=available.model_calls - amount.model_calls,
        )
        _reserve_hold(store, turn_id, job_id, bucket, residual)
        _record(
            store,
            CompactionJob(
                job.job_id,
                turn_id,
                job.source_revision,
                "installed" if purpose != "context_summary" and job.continuation is not None else "running",
                job.plan,
                job.nodes,
                job.continuation,
            ),
        )


@owner_operation
def record_node(
    store: RuntimeStore, *, turn_id: str, job_id: str, node_id: str, source_hash: str, result: str, operation_id: str
) -> None:
    with store.tx.transaction():
        job = read_job(store, turn_id=turn_id, job_id=job_id)
        value = {"source_hash": source_hash, "result": result, "operation_id": operation_id}
        if node_id in job.nodes:
            if dict(job.nodes[node_id]) != value:
                raise ContextSourceChangedError("Completed compaction node cannot be replaced.")
            return
        nodes = {**job.nodes, node_id: value}
        _record(
            store, CompactionJob(job.job_id, turn_id, job.source_revision, "running", job.plan, nodes, job.continuation)
        )


@owner_operation
def finish_job(store: RuntimeStore, *, turn_id: str, job_id: str, status: str) -> None:
    if status not in {"installed", "cancelled", "failed", "superseded", "waiting_unknown"}:
        raise ValueError("unsupported compaction job status")
    with store.tx.transaction():
        job = read_job(store, turn_id=turn_id, job_id=job_id)
        if status == "installed" and job.source_revision != store.user_input_revision(turn_id):
            raise ContextSourceChangedError("Compaction job source revision changed before install.")
        if status != "waiting_unknown":
            for row in _hold_rows(store, job_id):
                _fact(store, turn_id, "budget_released", {"reservation_id": row["reservation_id"]})
        _record(
            store,
            CompactionJob(job.job_id, turn_id, job.source_revision, status, job.plan, job.nodes, job.continuation),
        )


@owner_operation
def install_candidate(
    store: RuntimeStore, *, turn_id: str, job_id: str, install: Callable[[], Any], prepared: PreparedModelCall
) -> None:
    """CAS replacement, continuation operation, reservation and job in one commit."""
    with store.tx.transaction():
        job = read_job(store, turn_id=turn_id, job_id=job_id)
        if job.source_revision != store.user_input_revision(turn_id):
            raise ContextSourceChangedError("Compaction job source revision changed before install.")
        install()
        _record(
            store,
            CompactionJob(
                job.job_id,
                turn_id,
                job.source_revision,
                "candidate_ready",
                job.plan,
                job.nodes,
                dict(prepared.request_ref),
            ),
        )
        operation = model_ops.prepare_model_operation(
            store,
            turn_id=turn_id,
            request_hash=prepared.request_hash,
            context_hash=prepared.context_hash,
            tool_hash=prepared.tool_hash,
            wire_hash=prepared.wire_hash,
            request_ref=prepared.request_ref,
        )
        transfer_for_attempt(
            store, turn_id=turn_id, job_id=job_id, amount=prepared.resource_request, purpose="agent_step"
        )
        _fact(
            store,
            turn_id,
            "budget_reserved",
            {
                "reservation_id": f"reservation_{operation.active_attempt_id}",
                "scope_id": turn_id,
                "operation_id": operation.operation_id,
                "attempt_id": operation.active_attempt_id,
                "reserved": prepared.resource_request.to_dict(),
            },
        )
        finish_job(store, turn_id=turn_id, job_id=job_id, status="installed")


@owner_operation
def allowance_tokens(store: RuntimeStore, *, job_id: str) -> int:
    return sum(ResourceUsage.from_dict(row["reserved"]).total_tokens for row in _hold_rows(store, job_id))


@owner_operation
def find_job(store: RuntimeStore, *, turn_id: str, source_hash: str) -> CompactionJob | None:
    seen: set[str] = set()
    for item in reversed(store.list_items(turn_id)):
        if item.kind != "compaction_job" or item.status != "completed":
            continue
        identity = item.payload["job_id"]
        if identity in seen:
            continue
        seen.add(identity)
        if (
            item.payload["plan"].get("source_hash") == source_hash
            and item.payload["status"] in {"planned", "running", "candidate_ready", "waiting_unknown"}
            and item.payload["source_revision"] == store.user_input_revision(turn_id)
        ):
            return read_job(store, turn_id=turn_id, job_id=identity)
    return None


@owner_operation
def retry_admission_count(store: RuntimeStore, *, turn_id: str, job_id: str) -> int:
    return sum(
        item.kind == "compaction_retry_admission"
        and item.status == "completed"
        and item.payload.get("job_id") == job_id
        for item in store.list_items(turn_id)
    )


@owner_operation
def admit_retry(store: RuntimeStore, *, turn_id: str, operation_id: str, amount: ResourceUsage) -> None:
    """Admit a fresh attempt while the previous attempt remains uncertain.

    Call together with prepare_model_retry inside one outer transaction. The
    admission belongs to generation+1; it cannot be repeated to buy call slots.
    Legacy requests without a job keep their existing claim-time accounting.
    """
    with store.tx.transaction():
        operation = next(
            (operation for operation in store.list_model_operations(turn_id) if operation.operation_id == operation_id),
            None,
        )
        if operation is None:
            raise KeyError(f"unknown model operation: {operation_id}")
        plan = operation.request_ref.get("compaction_plan") or {}
        job_id = operation.request_ref.get("compaction_job_id") or plan.get("job_id")
        if job_id is None:
            return
        if operation.status not in {"unknown", "retry_wait"}:
            raise RuntimeError("compaction retry admission requires an unknown or scheduled operation")
        job = read_job(store, turn_id=turn_id, job_id=job_id)
        _check_completed_call_allowances(store, turn_id=turn_id, job_id=job_id)
        if replace(amount, retries=0) != replace(
            store.read_budget_reservation(operation.active_attempt_id).reserved, retries=0
        ):
            raise ContextBudgetExceededError("Retry amount differs from the frozen request allowance.")
        if job.status not in {"planned", "running", "candidate_ready", "waiting_unknown", "installed"}:
            raise ContextBudgetExceededError("Compaction retry job no longer has an active allowance.")
        if job.source_revision != store.user_input_revision(turn_id):
            raise ContextSourceChangedError("Compaction retry source changed.")
        generation = operation.generation + 1
        for item in store.list_items(turn_id):
            if (
                item.kind == "compaction_retry_admission"
                and item.status == "completed"
                and item.payload.get("job_id") == job_id
                and item.payload.get("operation_id") == operation_id
                and item.payload.get("generation") == generation
            ):
                if ResourceUsage.from_dict(item.payload["amount"]) != amount:
                    raise ContextSourceChangedError("Retry admission amount changed.")
                return
        purpose = str(operation.request_ref.get("purpose", "agent_step"))
        admitted = retry_admission_count(store, turn_id=turn_id, job_id=job_id)
        max_calls = job.plan.get("max_summary_calls", job.plan["planned_summary_calls"])
        max_retries = job.plan.get("max_summary_retries")
        if "max_summary_retries" not in job.plan:
            # Historical plans retain their recorded retry allowance.
            max_retries = max(0, max_calls - job.plan["planned_summary_calls"]) if max_calls is not None else None
        used_calls = sum(
            attempt.status != "prepared"
            for operation in store.list_model_operations(turn_id)
            if operation.request_ref.get("purpose") == "context_summary"
            for attempt in store.list_model_attempts(operation.operation_id)
        )
        exceeded_limit = max_retries if max_retries is not None and admitted >= max_retries else None
        if purpose == "context_summary" and max_calls is not None and used_calls >= max_calls:
            exceeded_limit = max_calls
        if exceeded_limit is not None:
            from agent_runtime.budget import BudgetLimitExceededError

            raise BudgetLimitExceededError(
                resource="model_calls", limit=exceeded_limit, current_exposure=used_calls, requested=1
            )
        bucket = "summary" if purpose == "context_summary" else "continuation"
        _reserve_hold(store, turn_id, job_id, bucket, amount)
        item_id = "item_" + uuid4().hex
        thread_id = store.read_turn(turn_id).thread_id
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="runtime",
            payload={"item_id": item_id, "kind": "compaction_retry_admission"},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="runtime",
            payload={
                "item_id": item_id,
                "payload": {
                    "job_id": job_id,
                    "operation_id": operation_id,
                    "generation": generation,
                    "amount": amount.to_dict(),
                    "purpose": purpose,
                },
            },
        )
        _record(
            store,
            CompactionJob(job.job_id, turn_id, job.source_revision, "running", job.plan, job.nodes, job.continuation),
        )


@owner_operation
def abandon_saved_summary(store: RuntimeStore, *, turn_id: str, operation_id: str) -> None:
    """Discard obsolete local work; provider uncertainty stays in its reservation."""
    with store.tx.transaction():
        operation = next(op for op in store.list_model_operations(turn_id) if op.operation_id == operation_id)
        if operation.request_ref.get("purpose") != "context_summary" or operation.status not in {
            "prepared",
            "unknown",
            "retry_wait",
        }:
            raise RuntimeError("only an uncompleted saved summary can be abandoned")
        from agent_runtime.harness import model_retry_ops

        if model_retry_ops.read_schedule(store, operation_id) is not None:
            model_retry_ops.cancel_scheduled_retry(
                store,
                operation_id=operation_id,
                reason_code="context_source_superseded",
                reason="summary source changed before retry",
            )
            job_id = operation.request_ref.get("compaction_plan", {}).get("job_id")
            if job_id:
                finish_job(store, turn_id=turn_id, job_id=job_id, status="superseded")
            return
        uncertain = operation.status == "unknown"
        facts.append_fact(
            store,
            thread_id=operation.thread_id,
            turn_id=turn_id,
            record_type="model_attempt_rejected",
            producer="runtime",
            payload={
                "operation_id": operation_id,
                "attempt_id": operation.active_attempt_id,
                "generation": operation.generation,
                "reason": "local summary source changed; saved request will not be dispatched",
                "error_type": "context_source_superseded",
                "provider_outcome": "unknown" if uncertain else "not_dispatched",
            },
        )
        if not uncertain:
            try:
                reservation = store.read_budget_reservation(operation.active_attempt_id)
            except KeyError:
                pass
            else:
                if reservation.status == "reserved":
                    _fact(store, turn_id, "budget_released", {"reservation_id": reservation.reservation_id})
        job_id = operation.request_ref.get("compaction_plan", {}).get("job_id")
        if job_id:
            finish_job(store, turn_id=turn_id, job_id=job_id, status="superseded")
        turn = store.read_turn(turn_id)
        if turn.status in {"paused", "interrupted"}:
            facts.append_fact(
                store,
                thread_id=operation.thread_id,
                turn_id=turn_id,
                record_type="turn_resumed",
                producer="runtime",
                payload={
                    "turn_id": turn_id,
                    "operation_id": operation_id,
                    "reason": "obsolete summary abandoned with exposure preserved",
                },
            )


@owner_operation
def finish_active_jobs(store: RuntimeStore, *, turn_id: str, status: str) -> None:
    """Release only unused holds when the owning Turn ends, never unknown attempts."""
    with store.tx.transaction():
        seen: set[str] = set()
        for item in reversed(store.list_items(turn_id)):
            if item.kind != "compaction_job" or item.status != "completed":
                continue
            job_id = item.payload["job_id"]
            if job_id in seen:
                continue
            seen.add(job_id)
            if item.payload["status"] in {"planned", "running", "candidate_ready", "waiting_unknown"}:
                finish_job(store, turn_id=turn_id, job_id=job_id, status=status)

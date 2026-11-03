"""Explicit budget ops over owner-serialized durable facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime.budget import (
    BudgetLimitExceededError,
    ReservationStatus,
    ResourceUsage,
    normal_token_remaining,
    resource_usage_from_model_usage,
)
from agent_runtime.harness import facts, turn_ops
from agent_runtime.harness.execution_context import step_limit_from_binding
from agent_runtime.harness.operation_values import _json_object, _validate_resource_usage_input
from agent_runtime.harness.records import (
    TurnSnapshot,
)
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation


@owner_operation
def start_budgeted_child_turn(
    store: RuntimeStore,
    *,
    parent_turn_id: str,
    child_thread_id: str,
    child_turn_id: str,
    user_message: str,
    binding_manifest: Mapping[str, Any],
    requested_tokens: int | None,
    requested_cost_micros: int | None,
) -> TurnSnapshot:
    """Atomically carve a child budget from the parent and start its Turn.

    The allocation record and child Turn live on the child Thread so this
    mutation remains one canonical Thread batch while SQLite's write lock
    serializes competing allocations from the same parent.
    """

    if not isinstance(parent_turn_id, str) or not parent_turn_id.strip():
        raise ValueError("parent_turn_id must be non-empty")
    if not isinstance(child_thread_id, str) or not child_thread_id.strip():
        raise ValueError("child_thread_id must be non-empty")
    if not isinstance(child_turn_id, str) or not child_turn_id.strip():
        raise ValueError("child_turn_id must be non-empty")
    if requested_tokens is not None and (
        isinstance(requested_tokens, bool) or not isinstance(requested_tokens, int) or requested_tokens < 1
    ):
        raise ValueError("requested child token budget must be positive or None")
    if requested_cost_micros is not None and (
        isinstance(requested_cost_micros, bool)
        or not isinstance(requested_cost_micros, int)
        or requested_cost_micros < 1
    ):
        raise ValueError("requested child cost budget must be positive or None")
    frozen_binding = _json_object(binding_manifest, field="binding_manifest")
    allocation_id = f"child_budget_{child_turn_id}"

    with store.tx.transaction():
        parent = store.tx.turn_with_workspace(parent_turn_id)
        if parent is None:
            raise KeyError(f"unknown parent Turn: {parent_turn_id}")
        if parent["status"] != "running":
            raise RuntimeError("child budget can only be allocated from a running parent Turn")
        turn_ops._assert_no_pending_cancellation(store, parent_turn_id)

        parent_state = store.read_budget_state(parent_turn_id)
        remaining = normal_token_remaining(parent_state)
        if remaining is not None:
            if requested_tokens is None:
                if remaining < 1:
                    limit = parent_state.limits.tokens
                    if limit is None:
                        raise RuntimeError("parent token limit disappeared")
                    raise BudgetLimitExceededError(
                        resource="tokens",
                        limit=limit,
                        current_exposure=limit,
                        requested=1,
                    )
                granted_tokens: int | None = remaining
            else:
                if requested_tokens > remaining:
                    limit = parent_state.limits.tokens
                    if limit is None:
                        raise RuntimeError("parent token limit disappeared")
                    raise BudgetLimitExceededError(
                        resource="tokens",
                        limit=limit,
                        current_exposure=limit - remaining,
                        requested=requested_tokens,
                    )
                granted_tokens = requested_tokens
        else:
            granted_tokens = requested_tokens

        remaining_cost = parent_state.remaining("cost_micros")
        if remaining_cost is not None:
            if requested_cost_micros is None:
                raise ValueError("child cost budget is required when the parent Turn has a cost cap")
            if requested_cost_micros > remaining_cost:
                limit = parent_state.limits.cost_micros
                if limit is None:
                    raise RuntimeError("parent cost limit disappeared")
                raise BudgetLimitExceededError(
                    resource="cost_micros",
                    limit=limit,
                    current_exposure=limit - remaining_cost,
                    requested=requested_cost_micros,
                )
            granted_cost_micros: int | None = requested_cost_micros
        else:
            granted_cost_micros = requested_cost_micros

        turn_ops.create_thread(
            store,
            workspace=Path(str(parent["workspace"])),
            thread_id=child_thread_id,
        )
        root_turn_id = parent_turn_id
        parent_binding = parent["binding_manifest"]
        inherited_root = parent_binding.get("budget_root_turn_id")
        if isinstance(inherited_root, str) and inherited_root:
            root_turn_id = inherited_root
        child_binding = dict(frozen_binding)
        parent_steps = step_limit_from_binding(parent_binding)
        child_steps = step_limit_from_binding(child_binding)
        if parent_steps is not None:
            if child_steps is not None and child_steps > parent_steps:
                raise ValueError("child max_steps cannot exceed the frozen parent Turn limit")
            child_steps = parent_steps if child_steps is None else child_steps
        child_binding["model_step_budget"] = child_steps
        child_binding["budget_root_turn_id"] = root_turn_id
        child_binding["budget_parent_turn_id"] = parent_turn_id
        child_binding["budget_allocation_id"] = allocation_id
        child_binding["model_token_budget_total"] = granted_tokens
        child_binding["model_cost_budget_total_micros"] = granted_cost_micros

        facts.append_fact(
            store,
            thread_id=child_thread_id,
            turn_id=None,
            record_type="budget_child_allocated",
            producer="runtime",
            payload={
                "allocation_id": allocation_id,
                "parent_turn_id": parent_turn_id,
                "child_turn_id": child_turn_id,
                "allocated_tokens": granted_tokens,
                "allocated_cost_micros": granted_cost_micros,
            },
        )
        turn_ops.start_turn(
            store,
            thread_id=child_thread_id,
            turn_id=child_turn_id,
            user_message=user_message,
            binding_manifest=child_binding,
        )
    return store.read_turn(child_turn_id)


@owner_operation
def settle_child_budget(store: RuntimeStore, *, child_turn_id: str) -> Mapping[str, Any]:
    """Convert one terminal child's allocation into its actual subtree usage."""

    with store.tx.transaction():
        row = store.tx.first("child_budget_allocations", where={"child_turn_id": child_turn_id})
        if row is None:
            raise KeyError(f"child Turn has no parent budget allocation: {child_turn_id}")
        if row["status"] == "settled":
            return store.read_child_budget_allocation(child_turn_id)
        if row["status"] != "active":
            raise RuntimeError("child budget allocation is not active")
        turn = store.read_turn(child_turn_id)
        if turn.status not in {"completed", "failed", "cancelled"}:
            raise RuntimeError("child budget can only settle after a terminal Turn")
        state = store.read_budget_state(child_turn_id)
        zero = ResourceUsage()
        if state.reserved != zero or state.uncertain != zero or state.child_reserved != zero:
            raise RuntimeError("child budget still has unresolved exposure")
        actual = state.used + ResourceUsage(subagents=1)
        facts.append_fact(
            store,
            thread_id=str(row["thread_id"]),
            turn_id=child_turn_id,
            record_type="budget_child_settled",
            producer="runtime",
            payload={
                "allocation_id": str(row["allocation_id"]),
                "actual": actual.to_dict(),
            },
        )
    return store.read_child_budget_allocation(child_turn_id)


def _budget_reservation_row_for_attempt(
    store: RuntimeStore,
    attempt_id: str,
) -> Mapping[str, Any] | None:
    row = store.tx.first("budget_reservations", where={"attempt_id": attempt_id})
    return row


def _append_budget_outcome(
    store: RuntimeStore,
    *,
    operation: Mapping[str, Any],
    attempt_id: str,
    outcome: str,
    usage: Mapping[str, Any] | None = None,
    producer: str = "runtime",
) -> None:
    row = _budget_reservation_row_for_attempt(store, attempt_id)
    if row is None:
        return
    reservation_id = str(row["reservation_id"])
    status = ReservationStatus(str(row["status"]))
    if outcome in {"settled", "not_sent"}:
        if status not in {
            ReservationStatus.RESERVED,
            ReservationStatus.DISPATCHED,
            ReservationStatus.UNKNOWN,
        }:
            raise RuntimeError("budget reservation cannot be settled")
        if usage is None and outcome == "settled":
            raise RuntimeError("settled budget requires provider usage")
        actual = resource_usage_from_model_usage(usage or {}) if outcome == "settled" else ResourceUsage()
        facts.append_fact(
            store,
            thread_id=str(operation["thread_id"]),
            turn_id=str(operation["turn_id"]),
            record_type="budget_settled",
            producer=producer,
            payload={
                "reservation_id": reservation_id,
                "actual": replace(
                    actual,
                    retries=ResourceUsage.from_dict(row["reserved"]).retries,
                ).to_dict(),
            },
        )
        return
    if outcome == "released":
        if status not in {
            ReservationStatus.RESERVED,
            ReservationStatus.DISPATCHED,
            ReservationStatus.UNKNOWN,
        }:
            raise RuntimeError("budget reservation cannot be released")
        facts.append_fact(
            store,
            thread_id=str(operation["thread_id"]),
            turn_id=str(operation["turn_id"]),
            record_type="budget_released",
            producer=producer,
            payload={"reservation_id": reservation_id},
        )
        return
    if outcome == "unknown":
        if status is ReservationStatus.UNKNOWN:
            return
        if status is not ReservationStatus.DISPATCHED:
            raise RuntimeError("only a dispatched budget can become unknown")
        facts.append_fact(
            store,
            thread_id=str(operation["thread_id"]),
            turn_id=str(operation["turn_id"]),
            record_type="budget_unknown",
            producer=producer,
            payload={"reservation_id": reservation_id},
        )
        return
    raise ValueError(f"unsupported budget outcome: {outcome}")


@owner_operation
def reserve_resource_unit(
    store: RuntimeStore,
    turn_id: str,
    resource: str,
    limit: int,
    initial_used: int = 0,
) -> dict[str, Any]:
    """Commit one conservative unit before I/O, serialized across Store connections."""

    _validate_resource_usage_input(resource, initial_used)
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError("resource limit must be an integer between 1 and 256")
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        thread_id = turn_ops._running_turn_thread_id(store, turn_id)
        usage = store.read_resource_usage(turn_id, resource, initial_used)
        if usage["limit"] is not None and usage["limit"] != limit:
            raise ValueError("resource_limit_mismatch")
        if usage["used"] >= limit:
            raise ValueError("resource_limit_exceeded")
        used = usage["used"] + 1
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="runtime",
            payload={"item_id": item_id, "kind": "resource_usage"},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="runtime",
            payload={"item_id": item_id, "payload": {"resource": resource, "limit": limit, "used": used}},
        )
    return {"resource": resource, "limit": limit, "used": used, "remaining": limit - used, "item_id": item_id}

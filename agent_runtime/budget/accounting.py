from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from agent_runtime.budget.types import (
    BudgetReservation,
    BudgetState,
    ReservationStatus,
    ResourceUsage,
    _resource_value,
)


class BudgetLimitExceededError(RuntimeError):
    def __init__(
        self,
        *,
        resource: str,
        limit: int,
        current_exposure: int,
        requested: int,
    ) -> None:
        self.resource = resource
        self.limit = limit
        self.current_exposure = current_exposure
        self.requested = requested
        super().__init__(
            f"budget limit exceeded for {resource}: "
            f"{current_exposure} + {requested} > {limit}"
        )


@dataclass(frozen=True, slots=True)
class BudgetMutation:
    state: BudgetState
    reservation: BudgetReservation


def reserve(
    state: BudgetState,
    *,
    reservation_id: str,
    operation_id: str,
    attempt_id: str,
    amount: ResourceUsage,
) -> BudgetMutation:
    _ensure_within_limits(state, amount)
    reservation = BudgetReservation(
        reservation_id=reservation_id,
        scope_id=state.scope_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
        reserved=amount,
    )
    return BudgetMutation(
        state=replace(state, reserved=state.reserved + amount),
        reservation=reservation,
    )


def mark_dispatched(
    state: BudgetState,
    reservation: BudgetReservation,
) -> BudgetMutation:
    _require_status(reservation, ReservationStatus.RESERVED)
    return BudgetMutation(
        state=state,
        reservation=replace(
            reservation,
            status=ReservationStatus.DISPATCHED,
        ),
    )


def settle(
    state: BudgetState,
    reservation: BudgetReservation,
    *,
    actual: ResourceUsage,
) -> BudgetMutation:
    if reservation.status not in {
        ReservationStatus.RESERVED,
        ReservationStatus.DISPATCHED,
        ReservationStatus.UNKNOWN,
    }:
        raise RuntimeError(
            f"cannot settle reservation in state {reservation.status.value}"
        )

    if reservation.status is ReservationStatus.UNKNOWN:
        base = replace(
            state,
            uncertain=state.uncertain.subtract_floor_zero(reservation.reserved),
        )
    else:
        base = replace(
            state,
            reserved=state.reserved.subtract_floor_zero(reservation.reserved),
        )

    # Actual provider usage is a fact. Never truncate it to fit the budget.
    next_state = replace(base, used=base.used + actual)
    return BudgetMutation(
        state=next_state,
        reservation=replace(
            reservation,
            status=ReservationStatus.SETTLED,
        ),
    )


def release(
    state: BudgetState,
    reservation: BudgetReservation,
) -> BudgetMutation:
    if reservation.status is ReservationStatus.UNKNOWN:
        next_state = replace(
            state,
            uncertain=state.uncertain.subtract_floor_zero(reservation.reserved),
        )
    elif reservation.status in {
        ReservationStatus.RESERVED,
        ReservationStatus.DISPATCHED,
    }:
        next_state = replace(
            state,
            reserved=state.reserved.subtract_floor_zero(reservation.reserved),
        )
    else:
        raise RuntimeError(
            f"cannot release reservation in state {reservation.status.value}"
        )
    return BudgetMutation(
        state=next_state,
        reservation=replace(
            reservation,
            status=ReservationStatus.RELEASED,
        ),
    )


def mark_unknown(
    state: BudgetState,
    reservation: BudgetReservation,
) -> BudgetMutation:
    _require_status(reservation, ReservationStatus.DISPATCHED)
    next_state = replace(
        state,
        reserved=state.reserved.subtract_floor_zero(reservation.reserved),
        uncertain=state.uncertain + reservation.reserved,
    )
    return BudgetMutation(
        state=next_state,
        reservation=replace(
            reservation,
            status=ReservationStatus.UNKNOWN,
        ),
    )


def resource_usage_from_model_usage(usage: Mapping[str, Any]) -> ResourceUsage:
    """Convert normalized provider usage into the budget resource vocabulary."""

    def count(*keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return 0

    return ResourceUsage(
        input_tokens=count("input_tokens"),
        cached_input_tokens=count("cached_input_tokens", "cache_read_input_tokens"),
        cache_write_tokens=count("cache_write_tokens", "cache_write_input_tokens"),
        output_tokens=count("output_tokens"),
        reasoning_tokens=count("reasoning_tokens"),
        cost_micros=count("cost_micros"),
        model_calls=1,
        model_ms=count("model_ms"),
        wall_ms=count("wall_ms"),
        network_bytes=count("network_bytes"),
    )


def _ensure_within_limits(
    state: BudgetState,
    requested: ResourceUsage,
) -> None:
    exposure = state.exposure
    for resource in state.limits.__dataclass_fields__:
        limit = getattr(state.limits, resource)
        if limit is None:
            continue
        current = _resource_value(exposure, resource)
        additional = _resource_value(requested, resource)
        if current + additional > limit:
            raise BudgetLimitExceededError(
                resource=resource,
                limit=limit,
                current_exposure=current,
                requested=additional,
            )


def _require_status(
    reservation: BudgetReservation,
    expected: ReservationStatus,
) -> None:
    if reservation.status is not expected:
        raise RuntimeError(
            f"reservation must be {expected.value}, got {reservation.status.value}"
        )

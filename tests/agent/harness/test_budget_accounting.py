from __future__ import annotations

import pytest

from agent_runtime.budget import (
    BudgetLimitExceededError,
    BudgetLimits,
    BudgetState,
    ReservationStatus,
    ResourceUsage,
    mark_dispatched,
    mark_unknown,
    release,
    reserve,
    settle,
)


def _state(*, tokens: int = 100) -> BudgetState:
    return BudgetState(
        scope_id="turn_1",
        parent_scope_id=None,
        limits=BudgetLimits(tokens=tokens, model_calls=10),
    )


def _request(tokens: int) -> ResourceUsage:
    return ResourceUsage(
        input_tokens=tokens,
        model_calls=1,
    )


def test_reserve_rejects_oversubscription() -> None:
    first = reserve(
        _state(tokens=50),
        reservation_id="r1",
        operation_id="op1",
        attempt_id="a1",
        amount=_request(40),
    )

    with pytest.raises(BudgetLimitExceededError):
        reserve(
            first.state,
            reservation_id="r2",
            operation_id="op2",
            attempt_id="a2",
            amount=_request(20),
        )


def test_success_releases_unused_reservation() -> None:
    reserved = reserve(
        _state(),
        reservation_id="r1",
        operation_id="op1",
        attempt_id="a1",
        amount=_request(30),
    )
    dispatched = mark_dispatched(reserved.state, reserved.reservation)
    done = settle(
        dispatched.state,
        dispatched.reservation,
        actual=ResourceUsage(input_tokens=22, model_calls=1),
    )

    assert done.reservation.status is ReservationStatus.SETTLED
    assert done.state.used.total_tokens == 22
    assert done.state.reserved.total_tokens == 0
    assert done.state.remaining("tokens") == 78


def test_actual_usage_can_exceed_reservation() -> None:
    reserved = reserve(
        _state(tokens=30),
        reservation_id="r1",
        operation_id="op1",
        attempt_id="a1",
        amount=_request(20),
    )
    dispatched = mark_dispatched(reserved.state, reserved.reservation)
    done = settle(
        dispatched.state,
        dispatched.reservation,
        actual=ResourceUsage(input_tokens=35, model_calls=1),
    )

    assert done.state.used.total_tokens == 35
    assert done.state.remaining("tokens") == 0


def test_unknown_moves_reservation_to_uncertain_exposure() -> None:
    reserved = reserve(
        _state(),
        reservation_id="r1",
        operation_id="op1",
        attempt_id="a1",
        amount=_request(30),
    )
    dispatched = mark_dispatched(reserved.state, reserved.reservation)
    unknown = mark_unknown(dispatched.state, dispatched.reservation)

    assert unknown.reservation.status is ReservationStatus.UNKNOWN
    assert unknown.state.reserved.total_tokens == 0
    assert unknown.state.uncertain.total_tokens == 30
    assert unknown.state.remaining("tokens") == 70


def test_retry_must_account_for_old_unknown_exposure() -> None:
    first = reserve(
        _state(tokens=50),
        reservation_id="r1",
        operation_id="op1",
        attempt_id="a1",
        amount=_request(30),
    )
    first = mark_dispatched(first.state, first.reservation)
    first = mark_unknown(first.state, first.reservation)

    with pytest.raises(BudgetLimitExceededError):
        reserve(
            first.state,
            reservation_id="r2",
            operation_id="op1",
            attempt_id="a2",
            amount=_request(30),
        )


def test_unknown_can_be_reconciled_to_actual_usage() -> None:
    reserved = reserve(
        _state(),
        reservation_id="r1",
        operation_id="op1",
        attempt_id="a1",
        amount=_request(30),
    )
    dispatched = mark_dispatched(reserved.state, reserved.reservation)
    unknown = mark_unknown(dispatched.state, dispatched.reservation)
    reconciled = settle(
        unknown.state,
        unknown.reservation,
        actual=ResourceUsage(input_tokens=18, model_calls=1),
    )

    assert reconciled.reservation.status is ReservationStatus.SETTLED
    assert reconciled.state.uncertain.total_tokens == 0
    assert reconciled.state.used.total_tokens == 18


def test_known_zero_can_release_unknown_exposure() -> None:
    reserved = reserve(
        _state(),
        reservation_id="r1",
        operation_id="op1",
        attempt_id="a1",
        amount=_request(30),
    )
    dispatched = mark_dispatched(reserved.state, reserved.reservation)
    unknown = mark_unknown(dispatched.state, dispatched.reservation)
    released = release(unknown.state, unknown.reservation)

    assert released.reservation.status is ReservationStatus.RELEASED
    assert released.state.uncertain.total_tokens == 0
    assert released.state.remaining("tokens") == 100

from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.budget import BudgetLimitExceededError, ReservationStatus, ResourceUsage
from agent_runtime.harness.rollout import RolloutStore


def _store(tmp_path: Path, *, tokens: int = 100) -> tuple[RolloutStore, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = RolloutStore(tmp_path / "rollout.sqlite")
    thread = store.create_thread(workspace=workspace)
    turn = store.start_turn(
        thread_id=thread.thread_id,
        user_message="test",
        binding_manifest={"model_token_budget_total": tokens},
    )
    return store, turn.turn_id


def _prepare(store: RolloutStore, turn_id: str, request_id: str):
    return store.prepare_model_operation(
        turn_id=turn_id,
        request_hash=f"request:{request_id}",
        context_hash=f"context:{request_id}",
        tool_hash=f"tools:{request_id}",
        wire_hash=f"wire:{request_id}",
        request_ref={"request_id": request_id},
    )


def _request(tokens: int) -> ResourceUsage:
    return ResourceUsage(input_tokens=tokens, model_calls=1)


def test_dispatch_reserves_budget_atomically(tmp_path: Path) -> None:
    store, turn_id = _store(tmp_path, tokens=50)
    operation = _prepare(store, turn_id, "r1")

    attempt = store.dispatch_model_attempt(
        operation.operation_id,
        resource_request=_request(40),
    )

    state = store.read_budget_state(turn_id)
    reservation = store.read_budget_reservation(attempt.attempt_id)
    assert reservation.status is ReservationStatus.DISPATCHED
    assert state.reserved.total_tokens == 40
    assert state.remaining("tokens") == 10
    assert store.verify().valid


def test_second_dispatch_cannot_oversubscribe_turn(tmp_path: Path) -> None:
    store, turn_id = _store(tmp_path, tokens=50)
    first = _prepare(store, turn_id, "r1")
    second = _prepare(store, turn_id, "r2")
    store.dispatch_model_attempt(first.operation_id, resource_request=_request(40))

    with pytest.raises(BudgetLimitExceededError):
        store.dispatch_model_attempt(second.operation_id, resource_request=_request(20))

    assert store.list_model_attempts(second.operation_id)[0].status == "prepared"
    assert store.read_budget_state(turn_id).reserved.total_tokens == 40


def test_completed_attempt_settles_actual_usage(tmp_path: Path) -> None:
    store, turn_id = _store(tmp_path, tokens=50)
    operation = _prepare(store, turn_id, "r1")
    attempt = store.dispatch_model_attempt(
        operation.operation_id,
        resource_request=_request(30),
    )

    accepted = store.complete_model_attempt(
        operation_id=operation.operation_id,
        attempt_id=attempt.attempt_id,
        generation=attempt.generation,
        text="done",
        provider_response_id="provider_1",
        usage={"input_tokens": 18, "output_tokens": 4, "total_tokens": 22},
    )

    assert accepted is True
    state = store.read_budget_state(turn_id)
    reservation = store.read_budget_reservation(attempt.attempt_id)
    assert reservation.status is ReservationStatus.SETTLED
    assert state.used.total_tokens == 22
    assert state.reserved.total_tokens == 0
    assert state.remaining("tokens") == 28


def test_actual_usage_is_not_truncated_to_reservation(tmp_path: Path) -> None:
    store, turn_id = _store(tmp_path, tokens=30)
    operation = _prepare(store, turn_id, "r1")
    attempt = store.dispatch_model_attempt(
        operation.operation_id,
        resource_request=_request(20),
    )

    store.complete_model_attempt(
        operation_id=operation.operation_id,
        attempt_id=attempt.attempt_id,
        generation=attempt.generation,
        text="done",
        provider_response_id=None,
        usage={"input_tokens": 25, "output_tokens": 10},
    )

    state = store.read_budget_state(turn_id)
    assert state.used.total_tokens == 35
    assert state.remaining("tokens") == 0


def test_unknown_attempt_remains_budget_exposure_across_retry(tmp_path: Path) -> None:
    store, turn_id = _store(tmp_path, tokens=50)
    operation = _prepare(store, turn_id, "r1")
    attempt = store.dispatch_model_attempt(
        operation.operation_id,
        resource_request=_request(30),
    )
    store.mark_model_attempt_unknown(
        operation_id=operation.operation_id,
        attempt_id=attempt.attempt_id,
        generation=attempt.generation,
        error_type="TimeoutError",
        error_message="timeout",
    )

    state = store.read_budget_state(turn_id)
    assert state.reserved.total_tokens == 0
    assert state.uncertain.total_tokens == 30
    assert store.read_budget_reservation(attempt.attempt_id).status is ReservationStatus.UNKNOWN

    store.prepare_model_retry(operation.operation_id)
    with pytest.raises(BudgetLimitExceededError):
        store.dispatch_model_attempt(
            operation.operation_id,
            resource_request=_request(30),
        )


def test_preflight_rejection_releases_reservation(tmp_path: Path) -> None:
    store, turn_id = _store(tmp_path, tokens=50)
    operation = _prepare(store, turn_id, "r1")
    attempt = store.dispatch_model_attempt(
        operation.operation_id,
        resource_request=_request(30),
    )
    store.reject_model_attempt(
        operation_id=operation.operation_id,
        attempt_id=attempt.attempt_id,
        generation=attempt.generation,
        reason="preflight",
    )

    state = store.read_budget_state(turn_id)
    assert state.reserved.total_tokens == 0
    assert state.uncertain.total_tokens == 0
    assert store.read_budget_reservation(attempt.attempt_id).status is ReservationStatus.RELEASED


def test_projection_rebuild_preserves_budget_accounting(tmp_path: Path) -> None:
    store, turn_id = _store(tmp_path, tokens=100)
    operation = _prepare(store, turn_id, "r1")
    attempt = store.dispatch_model_attempt(
        operation.operation_id,
        resource_request=_request(30),
    )
    store.mark_model_attempt_unknown(
        operation_id=operation.operation_id,
        attempt_id=attempt.attempt_id,
        generation=attempt.generation,
        error_type="TimeoutError",
        error_message="timeout",
    )

    before = store.read_budget_state(turn_id)
    store.rebuild_projections()
    after = store.read_budget_state(turn_id)

    assert before == after
    assert after.uncertain.total_tokens == 30
    assert store.verify().valid

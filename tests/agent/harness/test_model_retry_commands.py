from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.budget import BudgetLimitExceededError, ResourceUsage
from agent_runtime.harness import RuntimeStore, model_ops, turn_ops
from agent_runtime.harness import model_retry_ops as retry
from agent_runtime.harness.retry_policy import RetryPolicy
from agent_runtime.modeling.failures import ModelFailure

AMOUNT = ResourceUsage(input_tokens=100, output_tokens=20, cost_micros=30, model_calls=1)
RATE_LIMIT = ModelFailure("rate_limited", "response", "rejected", "released", http_status=429, retryable=True)
SERVER_ERROR = ModelFailure("server_error", "response", "rejected", "unknown", http_status=503, retryable=True)


def setup_operation(store: RuntimeStore, tmp_path: Path, **binding: object):
    thread = turn_ops.create_thread(store, workspace=tmp_path)
    turn = turn_ops.start_turn(
        store,
        thread_id=thread.thread_id,
        user_message="one question",
        binding_manifest={"model_retry_policy": RetryPolicy().to_payload(), **binding},
    )
    operation = model_ops.prepare_model_operation(
        store,
        turn_id=turn.turn_id,
        request_hash="request",
        context_hash="context",
        tool_hash="tools",
        wire_hash="wire",
        request_ref={"input_revision": 1, "request_id": "request-1", "purpose": "agent_step"},
    )
    attempt = model_ops.dispatch_model_attempt(store, operation.operation_id, now=100, resource_request=AMOUNT)
    return turn, operation, attempt


def record(store, operation, attempt, failure=RATE_LIMIT, now=101):
    return retry.record_failure(
        store,
        operation_id=operation.operation_id,
        attempt_id=attempt.attempt_id,
        generation=attempt.generation,
        failure=failure,
        now=now,
        random_fraction=0.5,
    )


def test_failure_and_schedule_are_durable_with_one_call_charge(tmp_path: Path):
    path = tmp_path / "rollout.sqlite3"
    with RuntimeStore.open_sync(path) as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        assert record(store, operation, attempt) == "scheduled"
        assert store.list_model_operations(turn.turn_id)[0].status == "retry_wait"
        assert store.list_model_attempts(operation.operation_id)[0].status == "failed"
        assert store.read_budget_state(turn.turn_id).used == ResourceUsage(model_calls=1)
        schedule = retry.read_schedule(store, operation.operation_id)
        assert schedule["deadline_at"] == 220
        assert schedule["retry_number"] == 1
        assert schedule["failure"]["kind"] == "rate_limited"
    with RuntimeStore.open_sync(path) as reopened:
        assert retry.read_schedule(reopened, operation.operation_id) == schedule
        assert reopened.verify().valid


def test_server_failure_keeps_unknown_exposure_and_admits_fresh_retry(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        assert record(store, operation, attempt, SERVER_ERROR) == "scheduled"
        assert store.read_budget_state(turn.turn_id).uncertain == AMOUNT
        next_attempt = retry.prepare_scheduled_retry(
            store, operation_id=operation.operation_id, resource_request=AMOUNT, now=102
        )
        assert next_attempt.generation == 2
        assert store.read_budget_state(turn.turn_id).reserved == replace(AMOUNT, retries=1)
        dispatched = model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=AMOUNT, now=102)
        assert dispatched.attempt_id == next_attempt.attempt_id
        assert store.read_budget_state(turn.turn_id).uncertain == AMOUNT
        assert store.verify().valid


@pytest.mark.parametrize(
    "failure,expected",
    [
        (ModelFailure("transport_unknown", "response", "unknown", "unknown"), "paused"),
        (replace(SERVER_ERROR, output_started=True), "paused"),
        (ModelFailure("authentication", "response", "rejected", "released"), "failed"),
        (ModelFailure("quota_exhausted", "response", "rejected", "released"), "failed"),
        (ModelFailure("protocol_error", "protocol", "unknown", "unknown"), "failed"),
    ],
)
def test_unknown_pauses_and_terminal_fails(tmp_path: Path, failure, expected):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        assert record(store, operation, attempt, failure) == expected
        assert retry.read_schedule(store, operation.operation_id) is None
        assert store.read_turn(turn.turn_id).status == ("interrupted" if expected == "paused" else "failed")
        fact = next(
            r
            for r in store.list_turn_records(turn.turn_id)
            if r.record_type in {"model_attempt_unknown", "model_attempt_rejected"}
        )
        assert fact.payload["failure"] == failure.to_payload()


@pytest.mark.parametrize("now,failure", [(99, RATE_LIMIT), (101, replace(RATE_LIMIT, retry_at=221))])
def test_clock_reversal_and_server_delay_past_deadline_fail(tmp_path: Path, now, failure):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        assert record(store, operation, attempt, failure, now) == "failed"
        assert store.read_turn(turn.turn_id).status == "failed"


def test_due_deadline_and_dispatch_second_input_cas(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        record(store, operation, attempt, replace(RATE_LIMIT, retry_at=105))
        with pytest.raises(RuntimeError, match="due"):
            retry.prepare_scheduled_retry(store, operation_id=operation.operation_id, resource_request=AMOUNT, now=102)
        retry.prepare_scheduled_retry(store, operation_id=operation.operation_id, resource_request=AMOUNT, now=105)
        turn_ops.receive_turn_input(store, thread_id=turn.thread_id, turn_id=turn.turn_id, text="changed input")
        with pytest.raises(RuntimeError, match="input"):
            model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=AMOUNT, now=105)
        retry.cancel_scheduled_retry(store, operation_id=operation.operation_id, reason="input changed")
        assert store.read_budget_state(turn.turn_id).reserved == ResourceUsage()
        assert store.read_budget_state(turn.turn_id).used.model_calls == 1


def test_max_auto_count_survives_generations(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        for number in range(1, 3):
            assert record(store, operation, attempt, now=100 + number * 2) == "scheduled"
            assert retry.read_schedule(store, operation.operation_id)["retry_number"] == number
            retry.prepare_scheduled_retry(
                store, operation_id=operation.operation_id, resource_request=AMOUNT, now=101 + number * 2
            )
            attempt = model_ops.dispatch_model_attempt(
                store, operation.operation_id, resource_request=AMOUNT, now=101 + number * 2
            )
        assert record(store, operation, attempt, now=108) == "failed"
        assert store.read_budget_state(turn.turn_id).used == ResourceUsage(model_calls=3, retries=2)


def test_unknown_budget_is_not_refunded_on_new_admission_failure(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path, model_token_budget_total=150)
        record(store, operation, attempt, SERVER_ERROR)
        with pytest.raises(BudgetLimitExceededError):
            retry.prepare_scheduled_retry(store, operation_id=operation.operation_id, resource_request=AMOUNT, now=102)
        retry.fail_unstarted_retry(
            store,
            operation_id=operation.operation_id,
            reason_code="model_retry_budget_exhausted",
            message="Insufficient fresh retry allowance.",
        )
        assert store.read_budget_state(turn.turn_id).uncertain == AMOUNT
        assert len(store.list_model_attempts(operation.operation_id)) == 1
        assert store.read_turn(turn.turn_id).status == "failed"


def test_cancel_running_wait_cancels_turn_without_refunding_unknown(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        record(store, operation, attempt, SERVER_ERROR)
        retry.cancel_scheduled_retry(store, operation_id=operation.operation_id, cancel_turn=True)
        assert store.read_turn(turn.turn_id).status == "cancelled"
        assert store.read_budget_state(turn.turn_id).uncertain == AMOUNT
        assert retry.read_schedule(store, operation.operation_id) is None


def test_prepared_retry_cannot_dispatch_after_deadline_and_releases_only_new_budget(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        record(store, operation, attempt, SERVER_ERROR)
        retry.prepare_scheduled_retry(store, operation_id=operation.operation_id, resource_request=AMOUNT, now=102)
        with pytest.raises(RuntimeError, match="deadline"):
            model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=AMOUNT, now=221)
        retry.fail_unstarted_retry(
            store,
            operation_id=operation.operation_id,
            reason_code="model_retry_deadline",
            message="Retry deadline expired.",
        )
        assert store.read_budget_state(turn.turn_id).reserved == ResourceUsage()
        assert store.read_budget_state(turn.turn_id).uncertain == AMOUNT
        assert store.read_turn(turn.turn_id).status == "failed"


def test_manual_unknown_retry_spends_retry_preserves_exposure_and_frozen_identity(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        record(store, operation, attempt, ModelFailure("transport_unknown", "response", "unknown", "unknown"))
        model_ops.prepare_model_retry(store, operation.operation_id)
        new_attempt = model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=AMOUNT, now=103)
        from agent_runtime.harness.protocol import HarnessModelResponse

        model_ops.complete_model_attempt(
            store,
            operation_id=operation.operation_id,
            attempt_id=new_attempt.attempt_id,
            generation=new_attempt.generation,
            response=HarnessModelResponse.from_parts(text="done", usage={"input_tokens": 5}),
        )
        state = store.read_budget_state(turn.turn_id)
        assert state.uncertain == AMOUNT
        assert state.used == ResourceUsage(input_tokens=5, model_calls=1, retries=1)
        assert store.list_model_operations(turn.turn_id)[0].request_hash == operation.request_hash


def test_automatic_summary_retry_preserves_continuation_and_unknown_call(tmp_path: Path):
    from agent_runtime.harness import compaction_jobs
    from agent_runtime.harness.context import RolloutContextManager
    from tests.agent.harness.test_compaction_consistency import long_history, seed
    from tests.agent.harness.test_context_engine_jobs import plan

    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="task",
            binding_manifest={"model_retry_policy": RetryPolicy().to_payload(), "model_token_budget_total": 10000},
        )
        long_history(store, turn)
        source_hash, _ = RolloutContextManager(store).semantic_source(turn.turn_id)
        frozen = {**plan(), "source_hash": source_hash, "max_summary_calls": 3, "max_summary_retries": 2}
        job = compaction_jobs.create_job(store, turn_id=turn.turn_id, plan=frozen, source_revision=1)
        operation = model_ops.prepare_model_operation(
            store,
            turn_id=turn.turn_id,
            request_hash="request",
            context_hash="context",
            tool_hash="tools",
            wire_hash="wire",
            request_ref={
                "request_id": "summary",
                "input_revision": 1,
                "purpose": "context_summary",
                "compaction_plan": dict(job.plan),
            },
        )
        amount = ResourceUsage(input_tokens=100, output_tokens=200, model_calls=1)
        attempt = model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=amount, now=100)
        record(store, operation, attempt, SERVER_ERROR)
        retry.prepare_scheduled_retry(store, operation_id=operation.operation_id, resource_request=amount, now=102)
        state = store.read_budget_state(turn.turn_id)
        assert state.uncertain == amount
        assert state.reserved.total_tokens == 700
        assert compaction_jobs.allowance_tokens(store, job_id=job.job_id) == 400
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "changed continuation evidence"})
        with pytest.raises(RuntimeError, match="source"):
            model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=amount, now=102)
        retry.cancel_scheduled_retry(store, operation_id=operation.operation_id, reason="source changed")
        assert store.read_budget_state(turn.turn_id).uncertain == amount
        assert compaction_jobs.allowance_tokens(store, job_id=job.job_id) == 400


def test_provably_unsent_retry_still_spends_retry_but_no_provider_call(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        record(store, operation, attempt)
        retry.prepare_scheduled_retry(store, operation_id=operation.operation_id, resource_request=AMOUNT, now=102)
        attempt = model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=AMOUNT, now=102)
        unsent = ModelFailure("transport_not_sent", "connect", "not_sent", "released", retryable=True)
        assert record(store, operation, attempt, unsent, now=103) == "scheduled"
        assert store.read_budget_state(turn.turn_id).used == ResourceUsage(model_calls=1, retries=1)


def test_explicit_http_rejection_records_failure_and_spends_one_call(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        failure = ModelFailure("invalid_request", "response", "rejected", "released", http_status=400)
        model_ops.reject_model_attempt(
            store,
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            reason=failure.message,
            failure=failure,
            budget_outcome="settled",
        )
        assert store.read_budget_state(turn.turn_id).used == ResourceUsage(model_calls=1)
        fact = next(r for r in store.list_turn_records(turn.turn_id) if r.record_type == "model_attempt_rejected")
        assert fact.payload["failure"] == failure.to_payload()


def test_clock_reversal_after_retry_dispatch_rejects_new_auto_schedule(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        record(store, operation, attempt)
        retry.prepare_scheduled_retry(store, operation_id=operation.operation_id, resource_request=AMOUNT, now=105)
        attempt = model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=AMOUNT, now=105)
        assert record(store, operation, attempt, now=104) == "failed"
        assert store.read_turn(turn.turn_id).status == "failed"


def test_clock_reversal_after_preparation_rejects_dispatch(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        _turn, operation, attempt = setup_operation(store, tmp_path)
        record(store, operation, attempt)
        retry.prepare_scheduled_retry(store, operation_id=operation.operation_id, resource_request=AMOUNT, now=105)
        with pytest.raises(RuntimeError, match="backwards"):
            model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=AMOUNT, now=104)


def test_cancel_before_retry_dispatch_does_not_charge_a_retry_unit(tmp_path: Path):
    with RuntimeStore.open_sync(tmp_path / "unspent.db") as store:
        turn, operation, attempt = setup_operation(store, tmp_path)
        record(store, operation, attempt, SERVER_ERROR)
        retry.prepare_scheduled_retry(store, operation_id=operation.operation_id, resource_request=AMOUNT, now=102)
        retry.cancel_scheduled_retry(store, operation_id=operation.operation_id)
        budget = store.read_budget_state(turn.turn_id)
        assert budget.reserved == ResourceUsage()
        assert budget.used.retries == 0
        assert budget.uncertain == AMOUNT


@pytest.mark.parametrize("abort", [False, True])
def test_terminal_retry_exit_releases_unused_job_holds_but_not_unknown_call(tmp_path: Path, abort):
    from agent_runtime.harness import compaction_jobs
    from tests.agent.harness.test_context_engine_jobs import plan

    with RuntimeStore.open_sync(tmp_path / "terminal-summary.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="task",
            binding_manifest={"model_retry_policy": RetryPolicy().to_payload(), "model_token_budget_total": 10000},
        )
        job = compaction_jobs.create_job(store, turn_id=turn.turn_id, plan=plan(), source_revision=1)
        op = model_ops.prepare_model_operation(
            store,
            turn_id=turn.turn_id,
            request_hash="r",
            context_hash="c",
            tool_hash="t",
            wire_hash="w",
            request_ref={
                "request_id": "summary",
                "input_revision": 1,
                "purpose": "context_summary",
                "compaction_plan": dict(job.plan),
            },
        )
        amount = ResourceUsage(input_tokens=100, output_tokens=200, model_calls=1)
        attempt = model_ops.dispatch_model_attempt(store, op.operation_id, resource_request=amount, now=100)
        assert record(store, op, attempt, SERVER_ERROR) == "scheduled"
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 400
        if abort:
            retry.cancel_scheduled_retry(store, operation_id=op.operation_id, cancel_turn=True)
        else:
            retry.fail_unstarted_retry(
                store,
                operation_id=op.operation_id,
                reason_code="model_retry_budget_exhausted",
                message="budget unavailable",
            )
        state = store.read_budget_state(turn.turn_id)
        assert state.reserved.total_tokens == 0
        assert state.uncertain == amount
        assert compaction_jobs.read_job(store, turn_id=turn.turn_id, job_id=job.job_id).status == (
            "cancelled" if abort else "failed"
        )
        assert store.verify().valid


@pytest.mark.parametrize(
    "binding,now,expected",
    [
        ({"model_retry_policy": RetryPolicy(max_auto_retries=0).to_payload()}, 101, "model_retry_exhausted"),
        ({}, 221, "model_retry_deadline_exceeded"),
        ({}, 99, "model_retry_clock_invalid"),
    ],
)
def test_terminal_reason_explains_why_eligible_failure_was_not_retried(tmp_path, binding, now, expected):
    with RuntimeStore.open_sync(tmp_path / "stop.db") as store:
        turn, operation, attempt = setup_operation(store, tmp_path, **binding)
        assert record(store, operation, attempt, now=now) == "failed"
        assert store.read_turn(turn.turn_id).terminal_reason_code == expected
        fact = next(r for r in store.list_turn_records(turn.turn_id) if r.record_type == "model_attempt_rejected")
        assert fact.payload["failure"]["kind"] == "rate_limited"

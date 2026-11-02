"""Public Session execution owns retries, accounting and cancellation."""

import asyncio
from dataclasses import replace

import pytest

from agent_runtime.budget import ResourceUsage
from agent_runtime.harness import Session, turn_ops
from agent_runtime.harness.protocol import HarnessModelResponse
from agent_runtime.modeling.failures import ModelCallFailed, ModelFailure
from tests.agent.harness.test_context_prefix_updates import Probe


class RetryModel(Probe):
    def __init__(self, failure, count=1):
        super().__init__("answer")
        self.failure = failure
        self.failures_left = count
        self.dispatched = []

    def prepare(self, request):
        return replace(
            super().prepare(request), resource_request=ResourceUsage(input_tokens=4, output_tokens=8, model_calls=1)
        )

    async def dispatch(self, prepared, **kwargs):
        self.dispatched.append(prepared)
        if self.failures_left:
            self.failures_left -= 1
            raise ModelCallFailed(self.failure)
        return HarnessModelResponse.from_parts(
            text="answer", usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
        )


def failure(kind="rate_limited", **kwargs):
    return ModelFailure(
        kind=kind,
        phase="response",
        provider_outcome="rejected",
        billing_outcome="released",
        retryable=True,
        http_status=429,
        **kwargs,
    )


@pytest.mark.anyio
async def test_public_session_retries_same_operation_and_counts_each_request(tmp_path):
    model = RetryModel(failure())
    async with await Session.open(
        database=tmp_path / "run.db", workspace=tmp_path, model=model, _background_memory=False
    ) as session:
        result = await session.submit("answer the task")
        assert result.status == "done"
        assert not any(d.severity == "error" and d.component == "model" for d in result.diagnostics)
        assert any(d.code == "model_dispatch_recovered" for d in result.diagnostics)
        assert len(model.dispatched) == 2
        [operation] = session.store.list_model_operations(result.turn_id)
        assert len(session.store.list_model_attempts(operation.operation_id)) == 2
        assert (
            len(
                [r for r in session.store.list_turn_records(result.turn_id) if r.record_type == "model_retry_scheduled"]
            )
            == 1
        )
        budget = session.store.read_budget_state(result.turn_id)
        assert budget.used.model_calls == 2
        assert budget.used.retries == 1
        assert budget.reserved.model_calls == 0
        assert all(request.step == 1 for request in model.requests)
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_exhaustion_is_bounded_and_survives_no_extra_reasoning_step(tmp_path):
    model = RetryModel(failure(), count=10)
    async with await Session.open(
        database=tmp_path / "run.db", workspace=tmp_path, model=model, _background_memory=False
    ) as session:
        result = await session.submit("answer the task")
        assert result.status == "failed"
        assert len(model.dispatched) == 3
        assert result.model_error.kind == "rate_limited"
        assert session.store.read_budget_state(result.turn_id).used.model_calls == 3
        [operation] = session.store.list_model_operations(result.turn_id)
        assert len(session.store.list_model_attempts(operation.operation_id)) == 3
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_server_retry_preserves_old_unknown_billing(tmp_path):
    model = RetryModel(
        ModelFailure(
            kind="server_error",
            phase="response",
            provider_outcome="rejected",
            billing_outcome="unknown",
            http_status=503,
            retryable=True,
        )
    )
    async with await Session.open(
        database=tmp_path / "run.db", workspace=tmp_path, model=model, _background_memory=False
    ) as session:
        result = await session.submit("answer the task")
        assert result.status == "done"
        state = session.store.read_budget_state(result.turn_id)
        assert state.uncertain.total_tokens == 12
        assert state.used.total_tokens == 6
        assert state.exposure.model_calls == 2
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_unknown_transport_is_not_automatically_replayed(tmp_path):
    model = RetryModel(
        ModelFailure(
            kind="transport_unknown",
            phase="stream",
            provider_outcome="unknown",
            billing_outcome="unknown",
            output_started=True,
        )
    )
    async with await Session.open(
        database=tmp_path / "run.db", workspace=tmp_path, model=model, _background_memory=False
    ) as session:
        result = await session.submit("answer the task")
        assert result.status == "paused"
        assert len(model.dispatched) == 1
        assert result.model_error.kind == "transport_unknown"
        assert session.store.read_budget_state(result.turn_id).uncertain.total_tokens == 12


@pytest.mark.anyio
@pytest.mark.parametrize("steering", [False, True])
async def test_runtime_schedule_obeys_new_input_and_cancellation_before_dispatch(tmp_path, steering):
    model = RetryModel(failure())
    async with await Session.open(
        database=tmp_path / "run.db", workspace=tmp_path, model=model, _background_memory=False
    ) as session:

        class Sink:
            scheduled = False

            async def emit(self, event):
                if event.type == "model_retry_scheduled" and not self.scheduled:
                    self.scheduled = True
                    if steering:
                        await session.steer("new requirement", expected_turn_id=event.turn_id)
                    else:
                        await session.store.run_async(
                            lambda: turn_ops.request_turn_cancellation(session.store, turn_id=event.turn_id)
                        )

        sink = Sink()
        result = await session.submit("answer the task", event_sink=sink)
        assert sink.scheduled
        if steering:
            assert result.status == "done"
            assert len(model.dispatched) == 2
            assert model.requests[-1].step == 2
            assert "new requirement" in [m.content for m in model.requests[-1].messages]
        else:
            assert session.store.read_turn(result.turn_id).status == "cancelled"
            assert len(model.dispatched) == 1
        assert session.store.verify().valid


@pytest.mark.parametrize("race", [None, "input", "source"])
def test_changed_summary_source_abandons_wait_and_releases_only_unused_job_holds(tmp_path, race, monkeypatch):
    from agent_runtime.harness import (
        RolloutContextManager,
        RuntimeStore,
        TurnRunner,
        compaction_jobs,
        model_ops,
        model_retry_ops,
    )
    from agent_runtime.harness.retry_policy import RetryPolicy
    from tests.agent.harness.test_compaction_consistency import Accept, long_history, model_for
    from tests.agent.harness.test_context_engine_jobs import plan

    with RuntimeStore.open_sync(tmp_path / "summary.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="task",
            binding_manifest={
                "model_retry_policy": RetryPolicy(base_delay_ms=0, max_delay_ms=0).to_payload(),
                "model_token_budget_total": 200000,
            },
        )
        long_history(store, turn)
        manager = RolloutContextManager(store)
        digest, _ = manager.semantic_source(turn.turn_id)
        job = compaction_jobs.create_job(
            store, turn_id=turn.turn_id, plan={**plan(), "source_hash": digest}, source_revision=1
        )
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
        attempt = model_ops.dispatch_model_attempt(store, op.operation_id, resource_request=amount)
        assert (
            model_retry_ops.record_failure(
                store,
                operation_id=op.operation_id,
                attempt_id=attempt.attempt_id,
                generation=attempt.generation,
                failure=ModelFailure(
                    kind="server_error",
                    phase="response",
                    provider_outcome="rejected",
                    billing_outcome="unknown",
                    http_status=503,
                    retryable=True,
                ),
            )
            == "scheduled"
        )
        model, _ = model_for()
        runner = TurnRunner(
            thread_id=thread.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        if race is None:
            turn_ops.receive_turn_input(
                store, thread_id=thread.thread_id, turn_id=turn.turn_id, text="changed requirement"
            )
            assert store.run_sync(lambda: runner._context_engine.discard_stale_saved_summary(op.operation_id))
        else:
            from agent_runtime.harness.operation_values import TurnInputChangedError
            from agent_runtime.harness.protocol import ContextSourceChangedError
            from tests.agent.harness.test_compaction_consistency import seed

            async def changed_during_admission(**kwargs):
                if race == "input":
                    await store.run_async(
                        lambda: turn_ops.receive_turn_input(
                            store, thread_id=thread.thread_id, turn_id=turn.turn_id, text="changed requirement"
                        )
                    )
                    raise TurnInputChangedError("changed during admission")
                await store.run_async(
                    lambda: seed(
                        store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "new durable evidence"}
                    )
                )
                raise ContextSourceChangedError("changed during admission")

            monkeypatch.setattr(runner, "_execute_saved_request", changed_during_admission)
            assert asyncio.run(runner._wait_model_retry(turn_id=turn.turn_id, operation_id=op.operation_id)) is None
        assert compaction_jobs.read_job(store, turn_id=turn.turn_id, job_id=job.job_id).status == "superseded"
        state = store.read_budget_state(turn.turn_id)
        assert state.reserved.total_tokens == 0
        assert state.uncertain == amount
        assert store.verify().valid


@pytest.mark.anyio
async def test_generic_model_bug_has_safe_structured_error_and_no_automatic_retry(tmp_path):
    class Bug(Probe):
        async def dispatch(self, prepared, **kwargs):
            raise ValueError("private credential must not reach diagnostics")

    model = Bug("answer")
    async with await Session.open(
        database=tmp_path / "bug.db", workspace=tmp_path, model=model, _background_memory=False
    ) as session:
        result = await session.submit("answer")
        assert result.status == "failed"
        assert result.model_error is not None
        assert result.model_error.kind == "protocol_error"
        assert "private credential" not in str(session.store.list_turn_records(result.turn_id))
        assert len(session.store.list_model_operations(result.turn_id)) == 1


@pytest.mark.anyio
async def test_partial_protocol_failure_pauses_without_transparent_retry(tmp_path):
    model = RetryModel(
        ModelFailure(
            kind="protocol_error",
            phase="protocol",
            provider_outcome="unknown",
            billing_outcome="unknown",
            output_started=True,
        )
    )
    async with await Session.open(
        database=tmp_path / "protocol.db", workspace=tmp_path, model=model, _background_memory=False
    ) as session:
        result = await session.submit("answer")
        assert result.status == "paused"
        assert result.model_error.kind == "protocol_error"
        assert len(model.dispatched) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("error_kind", ["typed", "context", "preflight"])
async def test_observed_partial_output_overrides_adapter_failure_flags(tmp_path, error_kind):
    from agent_runtime.harness.protocol import ModelContextOverflowError, ModelDispatchPreflightError
    from agent_runtime.modeling.events import ModelOutput, ModelOutputDelta, ModelOutputStarted, ModelResponseStarted

    class Contradictory(RetryModel):
        async def dispatch(self, prepared, *, event_sink=None):
            self.dispatched.append(prepared)
            await event_sink(ModelResponseStarted())
            await event_sink(ModelOutputStarted(ModelOutput("partial", "text")))
            await event_sink(ModelOutputDelta("partial", "text", "partial evidence"))
            if error_kind == "context":
                raise ModelContextOverflowError(
                    "overflow", failure=replace(failure(), kind="context_overflow", http_status=400, retryable=False)
                )
            if error_kind == "preflight":
                raise ModelDispatchPreflightError("too late for preflight")
            raise ModelCallFailed(failure())

    model = Contradictory(failure())
    async with await Session.open(
        database=tmp_path / "observed.db", workspace=tmp_path, model=model, _background_memory=False
    ) as session:
        result = await session.submit("answer")
        assert result.status == "paused"
        assert len(model.dispatched) == 1
        assert result.model_error.output_started
        assert not result.model_error.retryable
        assert result.model_error.provider_outcome == "unknown"
        budget = session.store.read_budget_state(result.turn_id)
        assert budget.uncertain.total_tokens == 12
        assert budget.used.model_calls == 0
        assert not session.store.list_tool_operations(result.turn_id)
        assert session.store.verify().valid

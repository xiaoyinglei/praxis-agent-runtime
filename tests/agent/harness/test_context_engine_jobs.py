from pathlib import Path

import pytest

from agent_runtime.budget import ResourceUsage
from agent_runtime.harness import turn_ops
from agent_runtime.harness.compaction_jobs import create_job, finish_job, read_job, transfer_for_attempt
from agent_runtime.harness.runtime_store import RuntimeStore


def start(store, path):
    thread = turn_ops.create_thread(store, workspace=path)
    return turn_ops.start_turn(
        store, thread_id=thread.thread_id, user_message="task", binding_manifest={"model_token_budget_total": 10000}
    )


def plan():
    return {
        "plan_id": "plan-a",
        "source_hash": "source-a",
        "retained_tail_messages": 0,
        "summary_tree": {"source_hash": "fragment", "input_limit": 100},
        "planned_token_upper_bound": 700,
        "continuation_token_reserve": 400,
        "planned_summary_calls": 1,
        "summary_output_limit": 200,
    }


def test_holds_persist_and_transfer_atomically(tmp_path: Path):
    db = tmp_path / "jobs.db"
    with RuntimeStore.open_sync(db) as store:
        turn = start(store, tmp_path)
        job = create_job(store, turn_id=turn.turn_id, plan=plan(), source_revision=1)
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 700
        assert create_job(store, turn_id=turn.turn_id, plan=plan(), source_revision=1).job_id == job.job_id
        transfer_for_attempt(
            store,
            turn_id=turn.turn_id,
            job_id=job.job_id,
            amount=ResourceUsage(input_tokens=100, output_tokens=200, model_calls=1),
            purpose="context_summary",
        )
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 400
        job_id, turn_id = job.job_id, turn.turn_id
    with RuntimeStore.open_sync(db) as store:
        assert read_job(store, turn_id=turn_id, job_id=job_id).status == "running"
        assert store.read_budget_state(turn_id).reserved.total_tokens == 400
        finish_job(store, turn_id=turn_id, job_id=job_id, status="cancelled")
        assert store.read_budget_state(turn_id).reserved.total_tokens == 0


def test_overspend_rolls_back_allowance_and_stale_revision_cannot_install(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "jobs.db") as store:
        turn = start(store, tmp_path)
        job = create_job(
            store, turn_id=turn.turn_id, plan=plan(), source_revision=store.user_input_revision(turn.turn_id)
        )
        with pytest.raises(Exception, match="allowance"):
            transfer_for_attempt(
                store,
                turn_id=turn.turn_id,
                job_id=job.job_id,
                amount=ResourceUsage(input_tokens=301),
                purpose="context_summary",
            )
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 700
        from agent_runtime.harness import facts

        def add_input():
            with store.tx.transaction():
                facts.append_fact(
                    store,
                    thread_id=turn.thread_id,
                    turn_id=turn.turn_id,
                    record_type="item_started",
                    producer="user",
                    payload={"item_id": "later", "kind": "user_message"},
                )
                facts.append_fact(
                    store,
                    thread_id=turn.thread_id,
                    turn_id=turn.turn_id,
                    record_type="item_completed",
                    producer="user",
                    payload={"item_id": "later", "payload": {"text": "changed"}},
                )

        store.run_sync(add_input)
        with pytest.raises(Exception, match="source"):
            finish_job(store, turn_id=turn.turn_id, job_id=job.job_id, status="installed")
        assert read_job(store, turn_id=turn.turn_id, job_id=job.job_id).status == "planned"


def test_engine_installs_measured_continuation_with_reserved_attempt(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutContextManager, TurnRunner
    from agent_runtime.harness.context_engine import ContextEngine
    from tests.agent.harness.test_compaction_consistency import Accept, long_history, model_for
    from tests.agent.harness.test_compaction_consistency import start as start_history

    with RuntimeStore.open_sync(tmp_path / "engine.db") as store:
        thread, turn = start_history(store, tmp_path)
        long_history(store, turn)
        model, gateway = model_for()
        manager = RolloutContextManager(store)
        executor = TurnRunner(
            thread_id=thread.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        engine = ContextEngine(
            store=store,
            context_manager=manager,
            prepare_step=executor._model_runner.prepare,
            capture_step_context=executor.capture_step_context,
            restore_step_context=executor.restore_step_context,
            dispatch_prepared=executor._model_runner.dispatch,
            commit=executor._commit,
            max_steps=16,
            budget_status=executor._budget_status,
        )
        context, prepared = asyncio.run(engine.prepare_or_resume(executor.restore_turn_context(turn.turn_id), step=1))
        jobs = [item for item in store.list_items(turn.turn_id) if item.kind == "compaction_job"]
        assert jobs[-1].payload["status"] == "installed"
        assert jobs[-1].payload["nodes"]
        assert any(item.kind == "context_compaction" for item in store.list_items(turn.turn_id))
        operation = [
            op for op in store.list_model_operations(turn.turn_id) if op.request_ref["purpose"] == "agent_step"
        ][-1]
        reservation = store.read_budget_reservation(operation.active_attempt_id)
        assert reservation.reserved == prepared.resource_request
        assert store.read_budget_state(turn.turn_id).reserved == prepared.resource_request
        assert len(gateway.requests) == len({node["operation_id"] for node in jobs[-1].payload["nodes"].values()})
        assert tuple(manager.build(turn.turn_id)) == context.messages


def test_completed_node_is_reused_after_store_restart(tmp_path):
    import asyncio
    import hashlib
    import json

    from agent_runtime.harness import RolloutContextManager, TurnRunner
    from agent_runtime.harness.context_engine import ContextEngine
    from tests.agent.harness.test_compaction_consistency import Accept, model_for

    source = json.dumps({"history": [{"role": "user", "body": "a real previous work unit"}]})
    frozen = plan()
    frozen["source_hash"] = hashlib.sha256(source.encode()).hexdigest()
    frozen["summary_tree"] = {"source_hash": frozen["source_hash"], "input_limit": 8000}
    frozen["planned_token_upper_bound"] = 8500
    frozen["summary_input_limit"] = 8000
    frozen["summary_output_limit"] = 100
    frozen["max_summary_calls"] = 8
    db = tmp_path / "resume.db"

    def setup(store, turn_id):
        model, gateway = model_for(100000)
        turn = store.read_turn(turn_id)
        manager = RolloutContextManager(store)
        executor = TurnRunner(
            thread_id=turn.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        engine = ContextEngine(
            store=store,
            context_manager=manager,
            prepare_step=executor._model_runner.prepare,
            capture_step_context=executor.capture_step_context,
            restore_step_context=executor.restore_step_context,
            dispatch_prepared=executor._model_runner.dispatch,
            commit=executor._commit,
            max_steps=16,
        )
        return engine, executor.restore_turn_context(turn_id), gateway

    with RuntimeStore.open_sync(db) as store:
        turn = start(store, tmp_path)
        job = create_job(
            store, turn_id=turn.turn_id, plan=frozen, source_revision=store.user_input_revision(turn.turn_id)
        )
        engine, context, gateway = setup(store, turn.turn_id)
        answer = asyncio.run(engine._semantic_summary(context, step=1, source=source, plan=job.plan))
        assert len(gateway.requests) == 1
        usage = store.read_budget_state(turn.turn_id).used
        assert read_job(store, turn_id=turn.turn_id, job_id=job.job_id).nodes
    with RuntimeStore.open_sync(db) as store:
        engine, context, gateway = setup(store, turn.turn_id)
        assert asyncio.run(engine._semantic_summary(context, step=1, source=source, plan=job.plan)) == answer
        assert gateway.requests == []
        assert store.read_budget_state(turn.turn_id).used == usage


def test_waiting_unknown_keeps_future_nodes_and_continuation_reserved(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "unknown-holds.db") as store:
        turn = start(store, tmp_path)
        job = create_job(store, turn_id=turn.turn_id, plan=plan(), source_revision=1)
        finish_job(store, turn_id=turn.turn_id, job_id=job.job_id, status="waiting_unknown")
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 700
        assert read_job(store, turn_id=turn.turn_id, job_id=job.job_id).status == "waiting_unknown"


def test_failed_job_never_returns_an_empty_allowance_for_new_admission(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "supersede.db") as store:
        turn = start(store, tmp_path)
        first = create_job(store, turn_id=turn.turn_id, plan=plan(), source_revision=1)
        finish_job(store, turn_id=turn.turn_id, job_id=first.job_id, status="superseded")
        replacement = create_job(store, turn_id=turn.turn_id, plan=plan(), source_revision=1)
        assert replacement.job_id != first.job_id
        assert replacement.status == "planned"
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 700


def test_second_connection_steering_during_summary_survives_stale_job(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutContextManager, TurnRunner
    from tests.agent.harness.test_compaction_consistency import Accept, long_history, model_for
    from tests.agent.harness.test_compaction_consistency import start as start_history

    db = tmp_path / "steering.db"
    with RuntimeStore.open_sync(db) as store:
        thread, turn = start_history(store, tmp_path)
        long_history(store, turn)
        model, gateway = model_for()
        runner = TurnRunner(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )
        original_dispatch = model.dispatch

        async def exercise():
            summary_started, input_committed = asyncio.Event(), asyncio.Event()
            observed = False

            async def dispatch(prepared, *, event_sink=None):
                nonlocal observed
                if prepared.request_ref["purpose"] == "context_summary" and not observed:
                    observed = True
                    summary_started.set()
                    await input_committed.wait()
                return await original_dispatch(prepared, event_sink=event_sink)

            model.dispatch = dispatch

            async def steer():
                await summary_started.wait()
                async with await RuntimeStore.open(db) as second:
                    assert any(
                        op.status == "dispatched"
                        for op in await second.run_async(second.list_model_operations, turn.turn_id)
                    )
                    item = await second.run_async(
                        turn_ops.receive_turn_input,
                        second,
                        thread_id=thread.thread_id,
                        turn_id=turn.turn_id,
                        text="SECOND_CONNECTION_CONSTRAINT",
                    )
                    assert item.payload["input_revision"] == 2
                input_committed.set()

            result, _ = await asyncio.gather(
                runner.drive(runner.restore_turn_context(turn.turn_id), start_step=1), steer()
            )
            return result

        result = asyncio.run(asyncio.wait_for(exercise(), timeout=10))
        assert result.status == "completed"
        assert store.user_input_revision(turn.turn_id) == 2
        assert "SECOND_CONNECTION_CONSTRAINT" in str(gateway.requests[-1])
        assert len([request for request in gateway.requests if ":context-summary:" not in request.request_id]) == 1
        assert any(
            item.kind == "compaction_job" and item.payload["status"] == "superseded"
            for item in store.list_items(turn.turn_id)
        )
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 0
        assert store.verify().valid


def test_explicit_summary_retry_admits_new_exposure_without_refunding_unknown(tmp_path):
    from agent_runtime.harness import compaction_jobs, model_ops
    from agent_runtime.harness.protocol import HarnessModelResponse

    with RuntimeStore.open_sync(tmp_path / "retry-exposure.db") as store:
        turn = start(store, tmp_path)
        frozen = {**plan(), "max_summary_calls": 3, "max_summary_retries": 1}
        job = create_job(store, turn_id=turn.turn_id, plan=frozen, source_revision=1)
        operation = model_ops.prepare_model_operation(
            store,
            turn_id=turn.turn_id,
            request_hash="request",
            context_hash="context",
            tool_hash="tool",
            wire_hash="wire",
            request_ref={
                "input_revision": 1,
                "request_id": "summary",
                "purpose": "context_summary",
                "compaction_plan": dict(job.plan),
            },
        )
        amount = ResourceUsage(input_tokens=100, output_tokens=200, model_calls=1)
        attempt = model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=amount)
        model_ops.mark_model_attempt_unknown(
            store, operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
        finish_job(store, turn_id=turn.turn_id, job_id=job.job_id, status="waiting_unknown")
        before = store.read_budget_state(turn.turn_id)
        assert before.uncertain == amount
        assert before.reserved.total_tokens == 400

        def admit_and_prepare():
            with store.tx.transaction():
                compaction_jobs.admit_retry(
                    store, turn_id=turn.turn_id, operation_id=operation.operation_id, amount=amount
                )
                model_ops.prepare_model_retry(store, operation.operation_id)

        store.run_sync(admit_and_prepare)
        assert store.read_budget_state(turn.turn_id).exposure.total_tokens == before.exposure.total_tokens + 300
        retry = model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=amount)
        assert store.read_budget_state(turn.turn_id).uncertain == amount
        assert store.read_budget_state(turn.turn_id).exposure.total_tokens == 1000
        model_ops.complete_model_attempt(
            store,
            operation_id=operation.operation_id,
            attempt_id=retry.attempt_id,
            generation=retry.generation,
            response=HarnessModelResponse.from_parts(
                text="checkpoint", usage={"input_tokens": 10, "output_tokens": 20}
            ),
        )
        assert store.read_budget_state(turn.turn_id).used.total_tokens == 30
        assert store.read_budget_state(turn.turn_id).uncertain.total_tokens == 300
        finish_job(store, turn_id=turn.turn_id, job_id=job.job_id, status="cancelled")
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 0
        assert store.read_budget_state(turn.turn_id).exposure.total_tokens == 330
        assert store.verify().valid


def test_retry_admission_cannot_spend_the_protected_tail_or_erase_unknown(tmp_path):
    from agent_runtime.budget import BudgetLimitExceededError
    from agent_runtime.harness import compaction_jobs, model_ops

    with RuntimeStore.open_sync(tmp_path / "retry-no-headroom.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(
            store, thread_id=thread.thread_id, user_message="task", binding_manifest={"model_token_budget_total": 800}
        )
        job = create_job(
            store,
            turn_id=turn.turn_id,
            plan={**plan(), "max_summary_calls": 3, "max_summary_retries": 1},
            source_revision=1,
        )
        operation = model_ops.prepare_model_operation(
            store,
            turn_id=turn.turn_id,
            request_hash="request",
            context_hash="context",
            tool_hash="tool",
            wire_hash="wire",
            request_ref={
                "input_revision": 1,
                "request_id": "summary",
                "purpose": "context_summary",
                "compaction_plan": dict(job.plan),
            },
        )
        amount = ResourceUsage(input_tokens=100, output_tokens=200, model_calls=1)
        attempt = model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=amount)
        model_ops.mark_model_attempt_unknown(
            store, operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
        before = store.read_budget_state(turn.turn_id)
        with pytest.raises(BudgetLimitExceededError, match="tokens"):
            compaction_jobs.admit_retry(store, turn_id=turn.turn_id, operation_id=operation.operation_id, amount=amount)
        assert store.read_budget_state(turn.turn_id) == before
        assert compaction_jobs.retry_admission_count(store, turn_id=turn.turn_id, job_id=job.job_id) == 0
        assert store.list_model_attempts(operation.operation_id)[-1].status == "unknown"


def test_provider_usage_above_node_allowance_is_kept_and_stops_new_work(tmp_path):
    from agent_runtime.harness import model_ops
    from agent_runtime.harness.protocol import ContextBudgetExceededError, HarnessModelResponse

    with RuntimeStore.open_sync(tmp_path / "overrun.db") as store:
        turn = start(store, tmp_path)
        job = create_job(store, turn_id=turn.turn_id, plan={**plan(), "max_summary_calls": 3}, source_revision=1)
        operation = model_ops.prepare_model_operation(
            store,
            turn_id=turn.turn_id,
            request_hash="request",
            context_hash="context",
            tool_hash="tool",
            wire_hash="wire",
            request_ref={
                "input_revision": 1,
                "request_id": "summary",
                "purpose": "context_summary",
                "compaction_plan": dict(job.plan),
            },
        )
        amount = ResourceUsage(input_tokens=100, output_tokens=200, model_calls=1)
        attempt = model_ops.dispatch_model_attempt(store, operation.operation_id, resource_request=amount)
        model_ops.complete_model_attempt(
            store,
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            response=HarnessModelResponse.from_parts(
                text="checkpoint", usage={"input_tokens": 100, "output_tokens": 250}
            ),
        )
        assert store.read_budget_state(turn.turn_id).used.total_tokens == 350
        with pytest.raises(ContextBudgetExceededError, match="exceeded"):
            transfer_for_attempt(
                store,
                turn_id=turn.turn_id,
                job_id=job.job_id,
                amount=ResourceUsage(input_tokens=400, model_calls=1),
                purpose="agent_step",
            )
        assert store.read_budget_state(turn.turn_id).used.total_tokens == 350
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 400


def test_default_measured_soft_threshold_starts_optional_compaction(tmp_path):
    import asyncio

    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
    from tests.agent.harness.test_compaction_consistency import long_history
    from tests.agent.harness.test_context_capacity_policy import capacity_runner

    with RuntimeStore.open_sync(tmp_path / "default-soft.db") as store:
        runner, turn, gateway = capacity_runner(store, tmp_path, window=30000, messages=2000)
        long_history(store, turn)
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "x" * 1800})
        context = runner.capture_step_context(runner.restore_turn_context(turn.turn_id), step=1)
        prepared = asyncio.run(runner._model_runner.prepare(context))
        projection = prepared.request_ref["context_projection"]
        assert (
            projection["auto_compact_limit"]
            < prepared.resource_request.input_tokens
            < projection["effective_input_limit"]
        )
        result = asyncio.run(runner.drive(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        summaries = [
            op for op in store.list_model_operations(turn.turn_id) if op.request_ref["purpose"] == "context_summary"
        ]
        assert summaries
        [compaction] = [item for item in store.list_items(turn.turn_id) if item.kind == "context_compaction"]
        assert compaction.payload["capacity_diagnostics"]["trigger_reason"] == "working_context_pressure"
        assert len(gateway.requests) == len(summaries) + 1


@pytest.mark.parametrize("unknown", [False, True])
def test_stale_saved_summary_is_abandoned_without_refunding_unknown(tmp_path, unknown):
    from agent_runtime.harness import RolloutContextManager, TurnRunner, model_ops
    from agent_runtime.harness.recovery import ContinuationKind, recovery_plan
    from tests.agent.harness.test_compaction_consistency import Accept, long_history, model_for
    from tests.agent.harness.test_compaction_consistency import start as start_history

    with RuntimeStore.open_sync(tmp_path / "stale-summary.db") as store:
        thread, turn = start_history(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        source_hash, _ = manager.semantic_source(turn.turn_id)
        job = create_job(store, turn_id=turn.turn_id, plan={**plan(), "source_hash": source_hash}, source_revision=1)
        operation = model_ops.prepare_model_operation(
            store,
            turn_id=turn.turn_id,
            request_hash="r",
            context_hash="c",
            tool_hash="t",
            wire_hash="w",
            request_ref={
                "request_id": "saved-summary",
                "input_revision": 1,
                "purpose": "context_summary",
                "compaction_plan": dict(job.plan),
            },
        )
        if unknown:
            attempt = model_ops.dispatch_model_attempt(
                store,
                operation.operation_id,
                resource_request=ResourceUsage(input_tokens=100, output_tokens=30, model_calls=1),
            )
            model_ops.mark_model_attempt_unknown(
                store,
                operation_id=operation.operation_id,
                attempt_id=attempt.attempt_id,
                generation=attempt.generation,
                error_message="provider receipt lost",
            )
        uncertain = store.read_budget_state(turn.turn_id).uncertain
        model, gateway = model_for()
        runner = TurnRunner(
            thread_id=thread.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        assert not store.run_sync(lambda: runner._context_engine.discard_stale_saved_summary(operation.operation_id))
        turn_ops.receive_turn_input(
            store, thread_id=thread.thread_id, turn_id=turn.turn_id, text="A new requirement changes the summary source"
        )
        assert store.run_sync(lambda: runner._context_engine.discard_stale_saved_summary(operation.operation_id))
        assert not gateway.requests
        budget = store.read_budget_state(turn.turn_id)
        assert budget.uncertain == uncertain
        assert budget.reserved.total_tokens == 0
        assert read_job(store, turn_id=turn.turn_id, job_id=job.job_id).status == "superseded"
        assert recovery_plan(store, turn.turn_id).kind == ContinuationKind.INPUT_SUPERSEDED
        store.rebuild_projections()
        assert store.verify().valid
        assert store.read_budget_state(turn.turn_id).uncertain == uncertain


def test_new_job_generation_has_its_own_call_allowance_after_unsent_supersession(tmp_path):
    from agent_runtime.harness import compaction_jobs, model_ops

    with RuntimeStore.open_sync(tmp_path / "generation-calls.db") as store:
        turn = start(store, tmp_path)
        frozen = {**plan(), "max_summary_calls": 3}
        first = create_job(store, turn_id=turn.turn_id, plan=frozen, source_revision=1)

        def prepare(job, request_id):
            return model_ops.prepare_model_operation(
                store,
                turn_id=turn.turn_id,
                request_hash=request_id,
                context_hash="c",
                tool_hash="t",
                wire_hash="w",
                request_ref={
                    "request_id": request_id,
                    "purpose": "context_summary",
                    "input_revision": 1,
                    "compaction_plan": dict(job.plan),
                },
            )

        abandoned = prepare(first, "old-unsent")
        compaction_jobs.abandon_saved_summary(store, turn_id=turn.turn_id, operation_id=abandoned.operation_id)
        second = create_job(store, turn_id=turn.turn_id, plan=frozen, source_revision=1)
        assert first.job_id != second.job_id
        operation = prepare(second, "new-generation")
        attempt = model_ops.dispatch_model_attempt(
            store,
            operation.operation_id,
            resource_request=ResourceUsage(input_tokens=100, output_tokens=30, model_calls=1),
        )
        assert attempt.status == "dispatched"
        assert store.verify().valid

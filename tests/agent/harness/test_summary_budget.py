import asyncio
from pathlib import Path

import pytest

from agent_runtime.harness import HarnessMessage, HarnessModelRequest, RolloutContextManager, RolloutStore, TurnExecutor
from agent_runtime.harness.protocol import ContextBudgetExceededError
from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
from tests.agent.harness.test_compaction_consistency import Accept, long_history, model_for, start


def test_unfit_fixed_context_spends_no_summary_calls(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        seed(store, turn_id=turn.turn_id, kind="user_message", payload={"text": "current task constraint " * 500})
        long_history(store, turn)
        model, gateway = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store, max_total_bytes=5000),
            completion_gate=Accept(),
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert not gateway.requests
        assert not store.list_model_operations(turn.turn_id)
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))


def test_summary_output_limit_controls_reservation_and_wire():
    model, _ = model_for()
    prepared = model.prepare(
        HarnessModelRequest(
            thread_id="thread",
            turn_id="turn",
            messages=(HarnessMessage(role="user", content="Summarize history."),),
            binding_manifest={},
            purpose="context_summary",
            output_token_limit=37,
        )
    )
    assert prepared.resource_request.output_tokens == 37
    assert prepared.dispatch_payload.request.settings.max_output_tokens == 37


def test_summary_is_not_dispatched_with_leftover_output_allowance(tmp_path, monkeypatch):
    """Budget can buy the input, but cannot buy configured output plus continuation."""
    with RolloutStore(tmp_path / "insufficient-summary.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        model, gateway = model_for(100000)
        prepared_calls = []
        original_prepare = model.prepare

        def prepare(request):
            prepared = original_prepare(request)
            prepared_calls.append(prepared)
            return prepared

        def remaining(_state):
            summaries = [p for p in prepared_calls if p.request_ref["purpose"] == "context_summary"]
            floors = [p for p in prepared_calls if p.request_ref["purpose"] == "agent_step"]
            if len(summaries) < 2 or not floors:
                return None
            summary = summaries[-1].resource_request
            return summary.input_tokens + floors[-1].resource_request.total_tokens + summary.output_tokens

        model.prepare = prepare
        monkeypatch.setattr("agent_runtime.harness.turn.normal_token_remaining", remaining)
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=RolloutContextManager(store, max_total_bytes=5000),
                              completion_gate=Accept())
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert not gateway.requests
        assert not store.list_model_operations(turn.turn_id)
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))


def test_request_input_limit_can_only_reduce_provider_limit():
    model, _ = model_for()
    with pytest.raises(ContextBudgetExceededError):
        model.prepare(
            HarnessModelRequest(
                thread_id="thread",
                turn_id="turn",
                messages=(HarnessMessage(role="user", content="history " * 20),),
                binding_manifest={},
                input_token_limit=10,
            )
        )


def test_step_output_limits_survive_operation_restore(tmp_path: Path):
    from dataclasses import replace

    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        model, _ = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )
        context = replace(
            runner.capture_step_context(runner.restore_turn_context(turn.turn_id), step=1),
            input_token_limit=4000,
            output_token_limit=37,
        )
        prepared = asyncio.run(runner._prepare_step(context))
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash=prepared.request_hash,
            context_hash=prepared.context_hash,
            tool_hash=prepared.tool_hash,
            wire_hash=prepared.wire_hash,
            request_ref=prepared.request_ref,
        )
        restored = runner.restore_step_context(context.turn, operation)
        reproduced = asyncio.run(runner._prepare_step(restored))
        assert restored.output_token_limit == 37 and restored.input_token_limit == 4000
        assert reproduced.wire_hash == prepared.wire_hash


@pytest.mark.parametrize("limit", [0, -1, True])
def test_model_request_rejects_invalid_output_limits(limit):
    model, _ = model_for()
    with pytest.raises(ValueError):
        model.prepare(
            HarnessModelRequest(
                thread_id="thread",
                turn_id="turn",
                messages=(HarnessMessage(role="user", content="history"),),
                binding_manifest={},
                output_token_limit=limit,
            )
        )


def test_summary_dispatch_preserves_continuation_headroom_after_reopen(tmp_path: Path):
    from agent_runtime.budget import BudgetLimitExceededError, ResourceUsage

    database = tmp_path / "reserve.db"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id, user_message="test", binding_manifest={"model_token_budget_total": 100}
        )
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="request",
            context_hash="context",
            tool_hash="tool",
            wire_hash="wire",
            request_ref={
                "request_id": "summary",
                "purpose": "context_summary",
                "compaction_plan": {"continuation_token_reserve": 40},
            },
        )
    with RolloutStore(database) as store:
        # 51 fits alone but cannot leave 40 of the 90 ordinary tokens available.
        with pytest.raises(BudgetLimitExceededError):
            store.dispatch_model_attempt(
                operation.operation_id,
                resource_request=ResourceUsage(input_tokens=51, model_calls=1),
                allow_protected_budget=True,
            )
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 0
        assert store.list_model_operations(turn.turn_id)[0].status == "prepared"
        store.dispatch_model_attempt(
            operation.operation_id, resource_request=ResourceUsage(input_tokens=50, model_calls=1)
        )
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 50
        assert store.verify().valid


def test_unknown_summary_retry_cannot_reset_frozen_call_allowance(tmp_path: Path):
    from agent_runtime.budget import BudgetLimitExceededError, ResourceUsage

    database = tmp_path / "retry.db"
    with RolloutStore(database) as store:
        thread, turn = start(store, tmp_path)
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="request",
            context_hash="context",
            tool_hash="tool",
            wire_hash="wire",
            request_ref={
                "request_id": "summary",
                "purpose": "context_summary",
                "compaction_plan": {
                    "plan_id": "plan",
                    "max_summary_calls": 1,
                    "planned_summary_calls": 1,
                    "continuation_token_reserve": 40,
                },
            },
        )
        attempt = store.dispatch_model_attempt(
            operation.operation_id, resource_request=ResourceUsage(input_tokens=10, model_calls=1)
        )
        store.mark_model_attempt_unknown(
            operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
    with RolloutStore(database) as store:
        store.prepare_model_retry(operation.operation_id)
        with pytest.raises(BudgetLimitExceededError, match="model_calls"):
            store.dispatch_model_attempt(
                operation.operation_id, resource_request=ResourceUsage(input_tokens=10, model_calls=1)
            )
        assert len([a for a in store.list_model_attempts(operation.operation_id) if a.status != "prepared"]) == 1
        assert store.verify().valid


def test_summary_cannot_consume_continuation_monetary_budget(tmp_path: Path):
    from agent_runtime.budget import BudgetLimitExceededError, ResourceUsage

    with RolloutStore(tmp_path / "cost.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id, user_message="test", binding_manifest={"model_cost_budget_total_micros": 100}
        )
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="request",
            context_hash="context",
            tool_hash="tool",
            wire_hash="wire",
            request_ref={
                "request_id": "summary",
                "purpose": "context_summary",
                "compaction_plan": {
                    "continuation_token_reserve": 40,
                    "continuation_cost_reserve": 60,
                },
            },
        )
        with pytest.raises(BudgetLimitExceededError, match="cost_micros"):
            store.dispatch_model_attempt(
                operation.operation_id, resource_request=ResourceUsage(input_tokens=10, cost_micros=41, model_calls=1)
            )
        assert store.read_budget_state(turn.turn_id).reserved.cost_micros == 0


def test_deepseek_summary_role_is_frozen_for_restart_and_legacy_requests(tmp_path: Path):
    from dataclasses import replace

    from agent_runtime.harness import GatewayHarnessModel
    from agent_runtime.modeling.openai_wire import serialize_openai_request
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway, _resolved_model

    gateway = BudgetAwareCapturingGateway(max_input_tokens=9000)
    model = GatewayHarnessModel(
        model_id="deepseek-flash",
        instructions=("Continue task.",),
        resolved=_resolved_model(
                gateway=gateway, token_accounting=gateway.token_accounting,
                provider="deepseek", model_id="deepseek-flash",
            ),
    )
    with RolloutStore(tmp_path / "role.db") as store:
        thread, turn = start(store, tmp_path)
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )
        context = replace(
            runner.capture_step_context(runner.restore_turn_context(turn.turn_id), step=1),
            messages=(
                HarnessMessage(role="user", content="Read specs then implement."),
                HarnessMessage(role="context", content="Context compaction:\nAlready read specs."),
                HarnessMessage(role="context", content="A new runtime constraint."),
            ),
        )
        prepared = asyncio.run(runner._prepare_step(context))
        wire = serialize_openai_request(prepared.dispatch_payload.request).payload
        assert [m["role"] for m in wire["messages"]][-3:] == ["user", "assistant", "user"]
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash=prepared.request_hash,
            context_hash=prepared.context_hash,
            tool_hash=prepared.tool_hash,
            wire_hash=prepared.wire_hash,
            request_ref=prepared.request_ref,
        )
        restored = runner.restore_step_context(context.turn, operation)
        assert restored.continuation_summary_role == "assistant"
        assert asyncio.run(runner._prepare_step(restored)).wire_hash == prepared.wire_hash
        # Older snapshots did not encode this choice: preserve their old wire.
        old_snapshot = dict(operation.request_ref["step_snapshot"])
        old_snapshot.pop("continuation_summary_role")
        old = replace(operation, request_ref={**operation.request_ref, "step_snapshot": old_snapshot})
        legacy = runner.restore_step_context(context.turn, old)
        assert legacy.continuation_summary_role == "context"
        legacy_request = asyncio.run(runner._prepare_step(legacy)).dispatch_payload.request
        assert [m["role"] for m in serialize_openai_request(legacy_request).payload["messages"]][-3:] == ["user"] * 3


def test_oversized_summary_with_tail_reuses_same_generation_without_losing_unseen_input(tmp_path: Path):
    from agent_runtime.harness import HarnessModelResponse

    with RolloutStore(tmp_path / "tail-budget.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        seed(
            store,
            turn_id=turn.turn_id,
            kind="model_response",
            payload={"text": "read", "tool_calls": [{"id": "last", "name": "read", "arguments": {}}]},
        )
        seed(
            store,
            turn_id=turn.turn_id,
            kind="tool_result",
            payload={"tool_call_id": "last", "model_content": "TAIL_EVIDENCE " + "x" * 2200},
        )
        manager = RolloutContextManager(store, max_total_bytes=5000)
        model, _ = model_for(100000)
        dispatched = []

        async def dispatch(prepared, **kwargs):
            dispatched.append(prepared)
            summary = prepared.request_ref["purpose"] == "context_summary"
            if summary:
                assert "TAIL_EVIDENCE" in prepared.request_ref["step_snapshot"]["messages"][0]["content"]
            return HarnessModelResponse(
                text=("TAIL_EVIDENCE summarized. " + "s" * 2450) if summary else "done",
                provider_response_id=None,
                usage={"input_tokens": 1, "output_tokens": 1},
            )

        model.dispatch = dispatch
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert [p.request_ref["purpose"] for p in dispatched] == ["context_summary", "agent_step"]
        final_messages = manager.build(turn.turn_id)
        assert any("TAIL_EVIDENCE summarized." in m.content for m in final_messages)
        assert not any(m.role == "tool" for m in final_messages)
        assert store.verify().valid

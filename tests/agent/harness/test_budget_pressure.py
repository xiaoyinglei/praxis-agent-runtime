from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.budget import (
    BudgetLimitExceededError,
    ResourceUsage,
    normal_token_remaining,
    pressure_threshold_for_limit,
    protected_tokens_for_limit,
)
from agent_runtime.harness import RolloutContextManager
from agent_runtime.harness.protocol import (
    CompletionDecision,
    CompletionProposal,
    HarnessMessage,
    HarnessModelRequest,
    HarnessModelResponse,
    PreparedModelCall,
)
from agent_runtime.harness.rollout import RolloutStore
from agent_runtime.harness.session import TurnExecutor
from tests.agent.harness.test_model_adapter import (
    CapturingGateway,
    CharacterAccounting,
    MaxTokensGateway,
    _resolved_model,
)


class StaticContext:
    def build(self, _turn_id: str) -> tuple[HarnessMessage, ...]:
        return (HarnessMessage(role="user", content="finish the task"),)


class AcceptAnswer:
    def evaluate(self, _proposal: CompletionProposal) -> CompletionDecision:
        return CompletionDecision(action="accept", reason="done")


class PressureModel:
    def __init__(self) -> None:
        self.pressure_flags: list[bool] = []

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        self.pressure_flags.append(request.budget_pressure)
        suffix = "pressure" if request.budget_pressure else "normal"
        return PreparedModelCall(
            request_hash=f"request-{suffix}",
            context_hash=f"context-{suffix}",
            tool_hash="tools",
            wire_hash=f"wire-{suffix}",
            request_ref={
                "request_id": f"request-{suffix}",
                "budget_pressure": request.budget_pressure,
            },
            resource_request=ResourceUsage(input_tokens=95, model_calls=1),
        )

    async def dispatch(
        self,
        _prepared: PreparedModelCall,
        **_kwargs: object,
    ) -> HarnessModelResponse:
        return HarnessModelResponse(
            text="final answer",
            provider_response_id="provider-1",
            usage={"input_tokens": 10, "output_tokens": 2},
        )


def _prepare(store: RolloutStore, turn_id: str, request_id: str):
    return store.prepare_model_operation(
        turn_id=turn_id,
        request_hash=f"request:{request_id}",
        context_hash=f"context:{request_id}",
        tool_hash=f"tools:{request_id}",
        wire_hash=f"wire:{request_id}",
        request_ref={"request_id": request_id},
    )


def test_protected_tail_formula_is_deterministic() -> None:
    assert protected_tokens_for_limit(None) == 0
    assert protected_tokens_for_limit(1) == 0
    assert protected_tokens_for_limit(100) == 10
    assert protected_tokens_for_limit(100_000) == 8_192
    assert pressure_threshold_for_limit(100) == 20


def test_normal_remaining_excludes_protected_tail(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="test",
            binding_manifest={"model_token_budget_total": 100},
        )
        assert normal_token_remaining(store.read_budget_state(turn.turn_id)) == 90


def test_dispatch_requires_explicit_access_to_protected_tail(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="test",
            binding_manifest={"model_token_budget_total": 100},
        )
        blocked = _prepare(store, turn.turn_id, "blocked")
        with pytest.raises(BudgetLimitExceededError):
            store.dispatch_model_attempt(
                blocked.operation_id,
                resource_request=ResourceUsage(input_tokens=95, model_calls=1),
            )

        allowed = _prepare(store, turn.turn_id, "allowed")
        attempt = store.dispatch_model_attempt(
            allowed.operation_id,
            resource_request=ResourceUsage(input_tokens=95, model_calls=1),
            allow_protected_budget=True,
        )
        assert attempt.status == "dispatched"
        assert store.read_budget_state(turn.turn_id).reserved.total_tokens == 95


def test_session_switches_to_pressure_before_using_protected_tail(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        model = PressureModel()
        session = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=StaticContext(),
            completion_gate=AcceptAnswer(),
        )
        result = asyncio.run(
            session.run(
                turn_id="turn-pressure",
                user_message="finish",
                binding_manifest={"model_token_budget_total": 100},
            )
        )
        assert result.status == "completed"
        assert result.answer == "final answer"
        assert model.pressure_flags == [False, True]
        state = store.read_budget_state(result.turn_id)
        assert state.used.total_tokens == 12
        assert state.reserved.total_tokens == 0


def _gateway_model(*, input_tokens: int | None = None, gateway: CapturingGateway | None = None):
    from agent_runtime.harness import GatewayHarnessModel

    class Accounting(CharacterAccounting):
        def count(self, text: str) -> int:
            return super().count(text) if input_tokens is None else input_tokens

    gateway = CapturingGateway() if gateway is None else gateway
    model = GatewayHarnessModel(
        model_id="test-model",
        resolved=_resolved_model(
            gateway=gateway,
            context_window_tokens=128_000,
            max_output_tokens=32_768,
            token_accounting=Accounting(),
        ),
        instructions=("Finish from verified evidence.",),
    )
    return model, gateway


def _settle_prior_usage(store: RolloutStore, turn_id: str, tokens: int) -> None:
    operation = _prepare(store, turn_id, "prior-work")
    attempt = store.dispatch_model_attempt(
        operation.operation_id,
        resource_request=ResourceUsage(input_tokens=tokens, model_calls=1),
    )
    store.complete_model_attempt(
        operation_id=operation.operation_id,
        attempt_id=attempt.attempt_id,
        generation=attempt.generation,
        text="prior work verified",
        provider_response_id=None,
        usage={"input_tokens": tokens, "output_tokens": 0},
    )


def test_gateway_session_finishes_with_remaining_budget_from_failed_trial(tmp_path: Path) -> None:
    """The real adapter must fit its wire and reservation, not just signal pressure."""
    with RolloutStore(tmp_path / "rollout.sqlite") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="finish verified work",
            binding_manifest={"model_token_budget_total": 80_000},
        )
        _settle_prior_usage(store, turn.turn_id, 43_330)
        model, gateway = _gateway_model(input_tokens=9_492)
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=StaticContext(),
            completion_gate=AcceptAnswer(),
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))

        assert result.status == "completed"
        assert result.answer == "real gateway answer"
        assert len(gateway.requests) == 1
        assert gateway.requests[0].settings.max_output_tokens == 27_178
        operation = store.list_model_operations(turn.turn_id)[-1]
        assert operation.request_ref["budget_pressure"] is True
        assert operation.request_ref["step_snapshot"]["output_token_limit"] == 27_178
        reservation = store.read_budget_reservation(operation.active_attempt_id)
        assert reservation.reserved.input_tokens == 9_492
        assert reservation.reserved.output_tokens == 27_178
        assert reservation.reserved.total_tokens == 36_670
        assert store.read_budget_state(turn.turn_id).used.total_tokens == 43_335
        assert store.verify().valid


def test_pressure_cap_is_remeasured_and_recovers_identical_wire_after_reopen(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite"
    model, gateway = _gateway_model()
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="finish",
            binding_manifest={"model_token_budget_total": 2_000},
        )
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model,
            context_manager=StaticContext(), completion_gate=AcceptAnswer(),
        )
        context = runner.capture_step_context(runner.restore_turn_context(turn.turn_id), step=1, budget_pressure=True)
        prepared = asyncio.run(runner._prepare_step(context))
        assert prepared.resource_request.total_tokens <= 2_000
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id, request_hash=prepared.request_hash, context_hash=prepared.context_hash,
            tool_hash=prepared.tool_hash, wire_hash=prepared.wire_hash, request_ref=prepared.request_ref,
        )
        assert not gateway.requests

    with RolloutStore(database) as store:
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model,
            context_manager=StaticContext(), completion_gate=AcceptAnswer(),
        )
        restored = runner.restore_step_context(runner.restore_turn_context(turn.turn_id), operation)
        reproduced = asyncio.run(runner._prepare_step(restored))
        assert reproduced.wire_hash == prepared.wire_hash
        assert reproduced.request_hash == prepared.request_hash
        assert restored.output_token_limit == prepared.resource_request.output_tokens
        result = asyncio.run(runner.recover_prepared_model(turn_id=turn.turn_id))
        assert result.status == "completed"
        assert gateway.requests[0].settings.max_output_tokens == restored.output_token_limit
        assert store.verify().valid


@pytest.mark.parametrize("budget", [None, 100_000])
def test_sufficient_budget_keeps_configured_output_ceiling(tmp_path: Path, budget: int | None) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite") as store:
        thread = store.create_thread(workspace=tmp_path)
        model, gateway = _gateway_model(input_tokens=9_492)
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model,
            context_manager=StaticContext(), completion_gate=AcceptAnswer(),
        )
        result = asyncio.run(runner.run(
            turn_id="ample-budget", user_message="finish",
            binding_manifest={"model_token_budget_total": budget},
        ))
        assert result.status == "completed"
        assert gateway.requests[0].settings.max_output_tokens == 32_768


def test_input_without_output_headroom_stops_before_provider_io(tmp_path: Path) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite") as store:
        thread = store.create_thread(workspace=tmp_path)
        model, gateway = _gateway_model(input_tokens=1_000)
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model,
            context_manager=RolloutContextManager(store), completion_gate=AcceptAnswer(),
        )
        result = asyncio.run(runner.run(
            turn_id="unfit-budget", user_message="finish",
            binding_manifest={"model_token_budget_total": 1_000},
        ))
        assert result.status == "failed"
        assert result.answer is None
        assert not gateway.requests
        assert not store.list_model_operations(result.turn_id)
        assert store.read_budget_state(result.turn_id).exposure.total_tokens == 0
        assert store.verify().valid


def test_pressure_keeps_explicit_output_limit_and_joint_summary_plan(tmp_path: Path) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id, user_message="finish",
            binding_manifest={"model_token_budget_total": 10_000},
        )
        model, gateway = _gateway_model(input_tokens=9_492)
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model,
            context_manager=StaticContext(), completion_gate=AcceptAnswer(),
        )
        context = runner.capture_step_context(runner.restore_turn_context(turn.turn_id), step=1, budget_pressure=True)
        explicit = asyncio.run(runner._prepare_step(replace(context, output_token_limit=37)))
        assert explicit.resource_request.output_tokens == 37
        plan = {"continuation_token_reserve": 42_260}
        joint = asyncio.run(runner._prepare_step(replace(context, compaction_plan=plan, output_token_limit=32_768)))
        assert joint.resource_request.output_tokens == 32_768
        assert joint.request_ref["compaction_plan"] == plan
        assert not gateway.requests


def test_capped_incomplete_answer_is_not_accepted_as_completion(tmp_path: Path) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite") as store:
        thread = store.create_thread(workspace=tmp_path)
        model, gateway = _gateway_model(input_tokens=9_492, gateway=MaxTokensGateway())
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model,
            context_manager=StaticContext(), completion_gate=AcceptAnswer(),
        )
        result = asyncio.run(runner.run(
            turn_id="incomplete-budget", user_message="finish",
            binding_manifest={"model_token_budget_total": 10_000},
        ))
        assert gateway.requests[0].settings.max_output_tokens == 508
        assert result.status == "failed"
        assert result.answer is None
        assert not any(item.kind == "completion_decision" for item in store.list_items(result.turn_id))
        assert store.verify().valid

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_runtime.budget import (
    BudgetLimitExceededError,
    ResourceUsage,
    normal_token_remaining,
    pressure_threshold_for_limit,
    protected_tokens_for_limit,
)
from agent_runtime.harness.protocol import (
    CompletionDecision,
    CompletionProposal,
    HarnessMessage,
    HarnessModelRequest,
    HarnessModelResponse,
    PreparedModelCall,
)
from agent_runtime.harness.rollout import RolloutStore
from agent_runtime.harness.session import Session


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
        session = Session(
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

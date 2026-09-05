from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from agent_runtime.budget import BudgetLimitExceededError, ResourceUsage
from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    HarnessModelRequest,
    HarnessModelResponse,
    PreparedModelCall,
    RolloutContextManager,
    RolloutStore,
    Session,
    ThreadManager,
)


class FixedBindingProvider:
    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, object]:
        return {
            "model_alias": "test-model",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "model_step_budget": 4,
            "model_token_budget_total": 1_000,
        }


class AcceptAnswer:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        del proposal
        return CompletionDecision(action="accept", reason="done")


class SmallAnswerModel:
    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        digest = hashlib.sha256(f"{request.turn_id}:{request.step}".encode()).hexdigest()
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash="no-tools",
            wire_hash=digest,
            request_ref={"request_id": f"{request.turn_id}:step:{request.step}"},
            resource_request=ResourceUsage(
                input_tokens=2,
                output_tokens=3,
                model_calls=1,
            ),
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        return HarnessModelResponse(
            text="child answer",
            provider_response_id="child-response",
            usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        )


class BlockingAnswerModel(SmallAnswerModel):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        self.started.set()
        await self.release.wait()
        return HarnessModelResponse(
            text="child answer",
            provider_response_id="child-response",
            usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        )


class UnknownAnswerModel(SmallAnswerModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        raise ConnectionError("provider outcome is unknown")


def _manager(
    store: RolloutStore,
    workspace: Path,
    model: object,
) -> ThreadManager:
    return ThreadManager(
        store=store,
        session_factory=lambda thread_id: Session(
            thread_id=thread_id,
            store=store,
            model=model,  # type: ignore[arg-type]
            context_manager=RolloutContextManager(store),
            completion_gate=AcceptAnswer(),
        ),
        workspace=workspace,
        binding_provider=FixedBindingProvider(),
    )


def _running_parent(
    store: RolloutStore,
    workspace: Path,
    *,
    token_budget: int,
):
    thread = store.create_thread(workspace=workspace)
    turn_id = "turn-parent"
    return store.start_turn(
        thread_id=thread.thread_id,
        turn_id=turn_id,
        user_message="parent task",
        binding_manifest={
            "model_alias": "test-model",
            "model_step_budget": 8,
            "model_token_budget_total": token_budget,
            "budget_root_turn_id": turn_id,
        },
    )


def test_completed_child_charges_actual_usage_and_releases_unused_allocation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        parent = _running_parent(store, workspace, token_budget=100)
        manager = _manager(store, workspace, SmallAnswerModel())

        child = asyncio.run(
            manager.run_child(
                parent_turn_id=parent.turn_id,
                user_message="bounded child",
                max_steps=2,
                max_tokens_total=60,
            )
        )

        assert child.status == "completed"
        child_binding = store.read_turn(child.turn_id).binding_manifest
        assert child_binding["budget_parent_turn_id"] == parent.turn_id
        assert child_binding["budget_root_turn_id"] == parent.turn_id
        assert child_binding["model_token_budget_total"] == 60

        allocation = store.read_child_budget_allocation(child.turn_id)
        assert allocation["allocated_tokens"] == 60
        assert allocation["status"] == "settled"
        assert allocation["actual"].total_tokens == 5
        assert allocation["actual"].subagents == 1

        parent_state = store.read_budget_state(parent.turn_id)
        assert parent_state.used.total_tokens == 5
        assert parent_state.used.subagents == 1
        assert parent_state.child_reserved.total_tokens == 0
        assert parent_state.remaining("tokens") == 95
        assert store.verify().valid is True


def test_active_child_allocation_prevents_parallel_budget_oversell(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        with RolloutStore(tmp_path / "rollout.sqlite3") as store:
            parent = _running_parent(store, workspace, token_budget=100)
            model = BlockingAnswerModel()
            manager = _manager(store, workspace, model)

            first = asyncio.create_task(
                manager.run_child(
                    parent_turn_id=parent.turn_id,
                    user_message="first child",
                    max_steps=2,
                    max_tokens_total=70,
                )
            )
            await asyncio.wait_for(model.started.wait(), timeout=1.0)

            state_while_running = store.read_budget_state(parent.turn_id)
            assert state_while_running.child_reserved.total_tokens == 70
            assert state_while_running.remaining("tokens") == 30

            with pytest.raises(BudgetLimitExceededError):
                await manager.run_child(
                    parent_turn_id=parent.turn_id,
                    user_message="second child",
                    max_steps=2,
                    max_tokens_total=40,
                )

            # Failed competing allocation is atomic: it must not leave a child Thread.
            assert len(store.list_threads()) == 2  # parent + first child only

            model.release.set()
            result = await asyncio.wait_for(first, timeout=1.0)
            assert result.status == "completed"
            assert store.read_budget_state(parent.turn_id).remaining("tokens") == 95
            assert store.verify().valid is True

    asyncio.run(scenario())


def test_unknown_child_keeps_parent_allocation_locked(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        parent = _running_parent(store, workspace, token_budget=100)
        manager = _manager(store, workspace, UnknownAnswerModel())

        child = asyncio.run(
            manager.run_child(
                parent_turn_id=parent.turn_id,
                user_message="uncertain child",
                max_steps=2,
                max_tokens_total=60,
            )
        )

        assert child.status == "paused"
        allocation = store.read_child_budget_allocation(child.turn_id)
        assert allocation["status"] == "active"
        parent_state = store.read_budget_state(parent.turn_id)
        assert parent_state.child_reserved.total_tokens == 60
        assert parent_state.remaining("tokens") == 40
        child_state = store.read_budget_state(child.turn_id)
        assert child_state.uncertain.total_tokens == 5
        assert store.verify().valid is True

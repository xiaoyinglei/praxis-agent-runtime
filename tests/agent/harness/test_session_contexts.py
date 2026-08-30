from __future__ import annotations

from pathlib import Path

from agent_runtime.harness import (
    DeliveryCompletionGate,
    HarnessMessage,
    RolloutContextManager,
    RolloutStore,
    Session,
    StepContext,
    TurnContext,
)


class _UnusedModel:
    def prepare(self, request):  # type: ignore[no-untyped-def]
        raise AssertionError("capture_step_context must not call the provider")

    async def dispatch(self, prepared):  # type: ignore[no-untyped-def]
        raise AssertionError("capture_step_context must not call the provider")


class _ChangingRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[HarnessMessage, ...]] = []

    def select(
        self,
        *,
        turn_id: str,
        messages: tuple[HarnessMessage, ...],
    ) -> tuple[()]:
        del turn_id
        self.calls.append(messages)
        return ()


def test_session_captures_one_immutable_request_view_per_step(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="inspect the workspace",
            binding_manifest={
                "model_alias": "test-model",
                "model_step_budget": 3,
                "model_token_budget_total": 100,
            },
        )
        router = _ChangingRouter()
        session = Session(
            thread_id=thread.thread_id,
            store=store,
            model=_UnusedModel(),
            context_manager=RolloutContextManager(store),
            completion_gate=DeliveryCompletionGate(store),
            tool_router=router,
        )
        turn_context = session.restore_turn_context(turn.turn_id)

        step_context = session.capture_step_context(turn_context, step=1)
        request = step_context.model_request()

        assert isinstance(turn_context, TurnContext)
        assert isinstance(step_context, StepContext)
        assert step_context.turn is turn_context
        assert step_context.step == 1
        assert step_context.messages == router.calls[0]
        assert step_context.tools == ()
        assert step_context.model_token_budget_remaining == 100
        assert request.thread_id == thread.thread_id
        assert request.turn_id == turn.turn_id
        assert request.step == 1
        assert request.messages is step_context.messages
        assert request.tools is step_context.tools
        assert request.binding_manifest is turn_context.binding_manifest


def test_session_rejects_a_turn_from_another_thread(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        first = store.create_thread(workspace=workspace)
        second = store.create_thread(workspace=workspace)
        foreign_turn = store.start_turn(
            thread_id=second.thread_id,
            user_message="foreign",
            binding_manifest={"model_alias": "test-model"},
        )
        session = Session(
            thread_id=first.thread_id,
            store=store,
            model=_UnusedModel(),
            context_manager=RolloutContextManager(store),
            completion_gate=DeliveryCompletionGate(store),
        )

        try:
            session.restore_turn_context(foreign_turn.turn_id)
        except RuntimeError as exc:
            assert "different Session" in str(exc)
        else:
            raise AssertionError("Session accepted a Turn owned by another thread")

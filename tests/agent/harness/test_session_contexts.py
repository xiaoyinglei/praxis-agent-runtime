from __future__ import annotations

from pathlib import Path

from agent_runtime.harness import (
    DeliveryCompletionGate,
    HarnessMessage,
    RolloutContextManager,
    RolloutStore,
    StepContext,
    TurnContext,
    TurnExecutor,
)


class _UnusedModel:
    def prepare(self, request):  # type: ignore[no-untyped-def]
        raise AssertionError("capture_step_context must not call the provider")

    async def dispatch(self, prepared, *, delta_sink: object = None):  # type: ignore[no-untyped-def]
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


def test_session_captures_one_immutable_request_view_per_step(tmp_path: Path, monkeypatch) -> None:
    clocks = iter((
        {"current_time": "2031-12-31T23:59:59+00:00", "current_date": "2031-12-31",
         "timezone": "UTC", "utc_offset": "+00:00"},
        {"current_time": "2032-01-01T00:00:01+00:00", "current_date": "2032-01-01",
         "timezone": "UTC", "utc_offset": "+00:00"},
    ))
    monkeypatch.setattr("agent_runtime.harness.turn.capture_runtime_clock", lambda: next(clocks))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="inspect the workspace",
            binding_manifest={
                "model_id": "test-model",
                "model_step_budget": 3,
                "model_token_budget_total": 100,
            },
        )
        router = _ChangingRouter()
        session = TurnExecutor(
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
        clock = request.binding_manifest["runtime_clock"]
        assert clock["current_date"] == clock["current_time"][:10]
        assert clock["timezone"] and clock["utc_offset"]
        assert {key: value for key, value in request.binding_manifest.items() if key != "runtime_clock"} == (
            dict(turn_context.binding_manifest)
        )
        assert request.binding_manifest is step_context.binding_manifest
        next_step = session.capture_step_context(turn_context, step=2)
        assert next_step.binding_manifest["runtime_clock"]["current_date"] == "2032-01-01"
        assert request.binding_manifest["runtime_clock"]["current_date"] == "2031-12-31"


def test_session_rejects_a_turn_from_another_thread(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        first = store.create_thread(workspace=workspace)
        second = store.create_thread(workspace=workspace)
        foreign_turn = store.start_turn(
            thread_id=second.thread_id,
            user_message="foreign",
            binding_manifest={"model_id": "test-model"},
        )
        session = TurnExecutor(
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
            raise AssertionError("TurnExecutor accepted a Turn owned by another thread")

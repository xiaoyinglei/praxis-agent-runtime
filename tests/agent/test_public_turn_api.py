from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from agent_runtime import Agent
from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.messages import ModelMessage
from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.loop.state import LoopState, ModelTurnDraft
from agent_runtime.runtime import builder as runtime_builder
from agent_runtime.service import AgentRunRequest, AgentRunResult, AgentService
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.turns import RuntimeBinding, TurnStatus, TurnStore
from agent_runtime.workspace import open_workspace


class _HistoryProvider:
    def __init__(self) -> None:
        self.contexts: list[tuple[str, tuple[ModelMessage, ...]]] = []

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        self.contexts.append(
            (
                state["current_message"],
                tuple(state["conversation_history"]),
            )
        )
        return ModelTurnDraft(
            action="finish",
            final_answer=f"answer:{state['current_message']}",
        )


def _service(
    tmp_path: Path,
    provider: _HistoryProvider,
) -> tuple[AgentService, TurnStore]:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    store = TurnStore(tmp_path / "agent.sqlite")
    runtime = RuntimeBinding(
        model_alias="test-model",
        workspace_path=str(workspace.root),
    )
    return (
        AgentService(
            definition=AgentRuntimePolicy.test_factory(),
            tool_registry=ToolRegistry(),
            model_turn_provider=provider,
            workspace=workspace,
            turn_store=store,
            runtime_binding=runtime,
        ),
        store,
    )


def test_request_has_one_turn_identity_and_optional_predecessor() -> None:
    previous_turn_id = str(UUID(int=1))
    request = AgentRunRequest(
        message="hello",
        previous_turn_id=previous_turn_id,
    )
    config = request.to_run_config(AgentRuntimePolicy.test_factory())

    assert str(UUID(config.turn_id)) == config.turn_id
    assert request.previous_turn_id == previous_turn_id
    assert not hasattr(request, "session_id")
    assert not hasattr(config, "session_id")


@pytest.mark.anyio
async def test_followup_history_is_derived_from_previous_turn(
    tmp_path: Path,
) -> None:
    provider = _HistoryProvider()
    service, store = _service(tmp_path, provider)
    first = await service.run(AgentRunRequest(message="remember cobalt"))
    second = await service.run(
        AgentRunRequest(
            message="what did I say?",
            previous_turn_id=first.turn_id,
        )
    )

    assert first.status == "done"
    assert second.status == "done"
    assert store.get_turn(first.turn_id).previous_turn_id is None
    assert store.get_turn(second.turn_id).previous_turn_id == first.turn_id
    assert provider.contexts == [
        ("remember cobalt", ()),
        (
            "what did I say?",
            (
                ModelMessage(role="user", content="remember cobalt"),
                ModelMessage(role="assistant", content="answer:remember cobalt"),
            ),
        ),
    ]
    await service.aclose()
    store.close()


@pytest.mark.anyio
async def test_calls_without_previous_turn_stay_independent(tmp_path: Path) -> None:
    provider = _HistoryProvider()
    service, store = _service(tmp_path, provider)
    first = await service.run(AgentRunRequest(message="first"))
    second = await service.run(AgentRunRequest(message="second"))

    assert first.turn_id != second.turn_id
    assert store.get_turn(first.turn_id).previous_turn_id is None
    assert store.get_turn(second.turn_id).previous_turn_id is None
    assert provider.contexts == [("first", ()), ("second", ())]
    await service.aclose()
    store.close()


def test_agent_core_surface_has_no_chat_lifecycle() -> None:
    assert not hasattr(Agent, "chat")
    assert not hasattr(Agent, "achat")
    assert not hasattr(Agent, "astream_chat")


def test_agent_run_projects_previous_turn_to_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[AgentRunRequest] = []

    class _Service:
        async def run(self, request: AgentRunRequest) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                turn_id=str(request.turn_id),
                status="done",
                final_answer="continued",
            )

    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )
    agent = Agent()
    previous = str(UUID(int=9))
    monkeypatch.setattr(agent, "_agent_for_previous_turn", lambda _turn_id: agent)

    result = agent.run("continue", previous_turn_id=previous)

    assert result.answer == "continued"
    assert requests[0].previous_turn_id == previous
    assert not hasattr(result, "session_id")


def test_agent_builder_receives_turn_store_and_runtime_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, object]] = []

    class _Service:
        async def run(self, request: AgentRunRequest) -> AgentRunResult:
            return AgentRunResult(
                turn_id=str(request.turn_id),
                status="done",
                final_answer="bound",
            )

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)
    knowledge = RAGKnowledgeConfig(
        storage_root=tmp_path / ".rag",
        vector_backend="sqlite",
    )
    Agent(
        model="qwen3_5_9b_mlx_4bit",
        checkpoint_db=tmp_path / "agent.sqlite",
        workspace_path=tmp_path,
        knowledge=knowledge,
    ).run("hello")

    assert isinstance(built[0]["turn_store"], TurnStore)
    binding = built[0]["runtime_binding"]
    assert isinstance(binding, RuntimeBinding)
    assert binding.model_alias == "qwen3_5_9b_mlx_4bit"
    assert binding.workspace_path == str(tmp_path.resolve())
    assert binding.knowledge == knowledge


@pytest.mark.anyio
async def test_stream_close_marks_the_same_turn_interrupted(tmp_path: Path) -> None:
    provider = _HistoryProvider()
    service, store = _service(tmp_path, provider)
    stream = service.run_streaming(AgentRunRequest(message="stream"))
    first_event = await anext(stream)
    await stream.aclose()

    assert first_event.turn_id
    assert store.get_turn(first_event.turn_id).status is TurnStatus.INTERRUPTED
    await service.aclose()
    store.close()

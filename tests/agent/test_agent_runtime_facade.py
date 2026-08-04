from __future__ import annotations

import asyncio
import inspect
import typing
from contextlib import asynccontextmanager
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest

import agent_runtime
from agent_runtime import Agent, AgentResult, AgentUsage
from agent_runtime import agent as agent_module
from agent_runtime.core.runtime_diagnostics import AgentLatencyProfile
from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.knowledge_providers.rag import LazyRAGKnowledgeProvider
from agent_runtime.models import ModelControlPlane
from agent_runtime.runtime import builder as runtime_builder
from agent_runtime.service import AgentRunResult
from agent_runtime.streaming.events import EventType, StreamEvent
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.integrations.knowledge import KnowledgeSearchInput
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
from agent_runtime.turns import RuntimeBinding, TurnStatus, TurnStore


def test_agent_runtime_exports_sdk_facade() -> None:
    assert Agent is not None
    assert AgentResult is not None
    assert AgentUsage is not None
    assert agent_runtime.__all__ == [
        "Agent",
        "AgentEventSink",
        "AgentResult",
        "AgentUsage",
        "EventType",
        "ModelNotAvailableError",
        "ModelSpec",
        "RAGKnowledgeConfig",
        "StreamEvent",
    ]
    for removed in (
        "ModelCatalog",
        "ModelControlPlane",
        "ModelPolicy",
        "ModelRuntimeSpec",
        "ModelSessionState",
    ):
        assert not hasattr(agent_runtime, removed)


def test_agent_public_signatures_are_exact() -> None:
    constructor = (
        "(*, model: 'str | None' = None, checkpoint_db: 'Path | None' = None, "
        "workspace_path: 'Path | str | None' = None, "
        "model_session_path: 'Path | None' = None, "
        "knowledge: 'RAGKnowledgeConfig | None' = None) -> 'None'"
    )
    run = (
        "(self, task: 'str', *, previous_turn_id: 'str | None' = None, "
        "files: 'Sequence[str] | None' = None, "
        "max_turns: 'int | None' = None, "
        "max_tokens_total: 'int | None' = None, "
        "require_workspace_change: 'bool' = True, "
        "allow_write_tools: 'bool' = False, "
        "allow_execute_tools: 'bool' = False, "
        "event_sink: 'AgentEventSink | None' = None) -> 'AgentResult'"
    )
    resume = (
        "(self, turn_id: 'str', action: 'str', *, "
        "user_input: 'str | None' = None, "
        "event_sink: 'AgentEventSink | None' = None) -> 'AgentResult'"
    )
    astream = run.replace(
        ", event_sink: 'AgentEventSink | None' = None) -> 'AgentResult'",
        ") -> 'AsyncIterator[StreamEvent]'",
    )
    assert str(inspect.signature(Agent)) == constructor
    assert str(inspect.signature(Agent.run)) == run
    assert str(inspect.signature(Agent.arun)) == run
    assert str(inspect.signature(Agent.resume)) == resume
    assert str(inspect.signature(Agent.aresume)) == resume
    assert str(inspect.signature(Agent.astream)) == astream
    assert not hasattr(Agent, "chat")
    assert not hasattr(Agent, "achat")
    assert not hasattr(Agent, "astream_chat")
    assert not hasattr(Agent, "stream")


def test_agent_public_annotations_resolve_without_any() -> None:
    for member in (
        Agent.__init__,
        Agent.run,
        Agent.arun,
        Agent.resume,
        Agent.aresume,
        Agent.astream,
        Agent.models,
        Agent.current_model,
        Agent.switch_model,
        Agent.pending_input,
        Agent.apending_input,
    ):
        hints = typing.get_type_hints(member)
        assert "typing.Any" not in repr(hints)


def test_stream_event_has_turn_named_json_contract() -> None:
    assert tuple(field.name for field in fields(StreamEvent)) == (
        "type",
        "turn_id",
        "iteration",
        "sequence",
        "timestamp_ms",
        "data",
        "span_id",
        "parent_id",
    )
    event = StreamEvent(type=EventType.TURN_START)
    assert not hasattr(event, "run_id")
    assert not hasattr(event, "thread_id")
    assert not hasattr(event, "turn")
    assert "typing.Any" not in repr(typing.get_type_hints(StreamEvent))


def test_agent_facade_run_maps_public_request_to_internal_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, Any]] = []
    requests: list[Any] = []
    lifecycles: list[str] = []

    def fail_rag_runtime(**_: object) -> object:
        raise AssertionError("Agent() without knowledge must not initialize RAG")

    class _Service:
        async def run(
            self,
            request: Any,
        ) -> AgentRunResult:
            requests.append(request)
            lifecycles.append("run")
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="facade answer",
            )

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_optional_rag_runtime", fail_rag_runtime)
    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    result = Agent(model="qwen3_14b_mlx_4bit").run(
        "summarize",
        files=["README.md"],
        max_turns=3,
        max_tokens_total=1234,
    )

    assert result.answer == "facade answer"
    assert result.status == "done"
    assert result.files == ("README.md",)
    assert not hasattr(result, "session_id")
    assert not hasattr(result, "raw")
    assert not hasattr(result, "thread_id")
    assert not hasattr(result, "run_id")
    assert isinstance(built[0]["model_control_plane"], ModelControlPlane)
    assert isinstance(built[0]["turn_store"], TurnStore)
    assert isinstance(built[0]["runtime_binding"], RuntimeBinding)
    assert isinstance(built[0]["startup_ms"], float)
    assert built[0]["startup_ms"] >= 0
    assert built == [
        {
            "runtime": None,
            "checkpoint_db": None,
            "checkpointer": built[0]["checkpointer"],
            "model_alias": "qwen3_14b_mlx_4bit",
            "model_control_plane": built[0]["model_control_plane"],
            "runtime_diagnostics": (),
            "knowledge_runner": None,
            "mcp_tools": (),
            "skill_tools": (),
            "subagent_tools": (),
            "skill_runtime": None,
            "stream_sink": None,
            "startup_ms": built[0]["startup_ms"],
            "turn_store": built[0]["turn_store"],
            "runtime_binding": built[0]["runtime_binding"],
        }
    ]
    assert len(requests) == 1
    request = requests[0]
    assert request.message == "summarize"
    assert request.turn_id == result.turn_id
    assert request.turn_id is not None
    UUID(request.turn_id)
    assert request.previous_turn_id is None
    assert request.max_turns == 3
    assert request.llm_budget_total == 1234
    assert request.input_files == ["README.md"]
    assert request.allow_write_tools is False
    assert request.allow_execute_tools is False
    assert lifecycles == ["run"]


def test_agent_facade_run_passes_public_permission_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Any] = []

    class _Service:
        async def run(
            self,
            request: Any,
        ) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="configured permissions",
            )

    monkeypatch.setattr(runtime_builder, "build_agent_service", lambda *_args, **_kwargs: _Service())

    result = Agent().run(
        "Run the approved operation.",
        allow_write_tools=True,
        allow_execute_tools=True,
    )

    assert len(requests) == 1
    assert not hasattr(result, "session_id")
    request = requests[0]
    assert request.allow_write_tools is True
    assert request.allow_execute_tools is True


def test_agent_facade_defaults_to_real_change_and_post_change_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Any] = []

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="changed and verified",
            )

    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    Agent().run("Fix the implementation.")

    goal = requests[0].goal_spec
    assert goal is not None
    assert [
        (item.constraint_id, item.constraint_type, item.expected_value)
        for item in goal.constraints
    ] == [
        ("workspace_change", "workspace_change", True),
        (
            "verification_after_change",
            "verification_after_change",
            True,
        ),
    ]


def test_agent_facade_builds_explicit_completion_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Any] = []

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="changed",
            )

    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    Agent().run(
        "Fix the implementation.",
        require_workspace_change=True,
        allow_write_tools=True,
    )

    goal = requests[0].goal_spec
    assert goal is not None
    assert goal.original_query == "Fix the implementation."
    assert [
        (item.constraint_id, item.constraint_type, item.expected_value)
        for item in goal.constraints
    ] == [
        ("workspace_change", "workspace_change", True),
        (
            "verification_after_change",
            "verification_after_change",
            True,
        ),
    ]


def test_agent_facade_allows_explicit_read_only_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Any] = []

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="read-only answer",
            )

    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    Agent().run("Explain the implementation.", require_workspace_change=False)

    assert requests[0].goal_spec is None


def test_agent_facade_requires_post_change_verification_when_execution_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Any] = []

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="changed and verified",
            )

    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    Agent().run(
        "Fix and verify the implementation.",
        require_workspace_change=True,
        allow_write_tools=True,
        allow_execute_tools=True,
    )

    goal = requests[0].goal_spec
    assert goal is not None
    assert [
        (item.constraint_id, item.constraint_type, item.expected_value)
        for item in goal.constraints
    ] == [
        ("workspace_change", "workspace_change", True),
        (
            "verification_after_change",
            "verification_after_change",
            True,
        ),
    ]


@pytest.mark.anyio
async def test_agent_facade_followup_passes_previous_turn_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Any] = []

    class _Service:
        async def run(
            self,
            request: Any,
        ) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="bounded follow-up",
            )

    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    agent = Agent()
    monkeypatch.setattr(agent, "_agent_for_previous_turn", lambda _turn_id: agent)
    result = await agent.arun(
        "Answer briefly.",
        previous_turn_id="00000000-0000-0000-0000-000000000001",
        max_turns=2,
    )

    assert result.answer == "bounded follow-up"
    assert result.turn_id == requests[0].turn_id
    assert requests[0].max_turns == 2
    assert requests[0].previous_turn_id == "00000000-0000-0000-0000-000000000001"


def test_agent_facade_binds_public_workspace_to_builder_and_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[object] = []
    requests: list[Any] = []

    class _Service:
        async def run(
            self,
            request: Any,
        ) -> AgentRunResult:
            requests.append(request)
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="workspace bound",
                workspace_path=request.workspace_path,
            )

    def build_service(runtime: object, **_kwargs: object) -> _Service:
        built.append(runtime)
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    result = Agent(workspace_path=tmp_path / "workspace").run(
        "Inspect this workspace.",
    )

    assert result.answer == "workspace bound"
    assert not hasattr(result, "session_id")
    assert built[0].root == (tmp_path / "workspace").resolve()
    assert requests[0].workspace_path == str((tmp_path / "workspace").resolve())


def test_agent_facade_assembles_workspace_skills_into_fixed_gateways(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".agents" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\nReview carefully.\n",
        encoding="utf-8",
    )
    built: list[dict[str, Any]] = []

    class _Service:
        pass

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    Agent(workspace_path=workspace)._build_service()

    assert [tool.definition.name for tool in built[0]["skill_tools"]] == ["invoke_skill", "materialize_skill_asset"]
    assert [tool.definition.name for tool in built[0]["subagent_tools"]] == ["task"]
    skill_runtime = built[0]["skill_runtime"]
    assert skill_runtime.model_invocable_skill_ids == ("project:review",)


@pytest.mark.anyio
async def test_product_subagent_projection_runs_a_bounded_child_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, Any]] = []
    closed: list[bool] = []

    class _Service:
        async def run(self, request: Any) -> AgentRunResult:
            assert request.max_turns == 2
            return AgentRunResult(
                turn_id="child-run",
                status="done",
                final_answer="child answer",
            )

        async def aclose(self) -> None:
            closed.append(True)

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)
    Agent(workspace_path=tmp_path)._build_service()
    [tool] = built[0]["subagent_tools"]
    call = ToolCall(
        tool_call_id="call_task",
        tool_name="task",
        arguments={"task": "inspect one file", "max_turns": 2},
        origin=ToolCallOrigin(
            request_id="request_task",
            toolset_revision="tools_task",
            exposed_tool_names=("task",),
        ),
    )

    execution = await ToolExecutor({"task": tool}).execute(
        call,
        context=ToolExecutionContext(approved_tool_call_ids=frozenset({"call_task"})),
    )

    assert execution.result.is_error is False
    assert execution.result.structured_content["conclusion"] == "child answer"
    assert len(built) == 2
    assert closed == [True]


def test_agent_facade_registers_knowledge_runner_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, Any]] = []
    monkeypatch.setenv(
        "AGENT_VECTOR_DSN",
        "postgresql://configured-secret",
    )
    monkeypatch.setenv(
        "VECTOR_DSN",
        "postgresql://legacy-must-not-win",
    )

    def fail_rag_runtime(**_: object) -> object:
        raise AssertionError("Knowledge provider must initialize RAG only when the tool is called")

    class _Service:
        async def run(
            self,
            request: Any,
        ) -> AgentRunResult:
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="knowledge runner registered",
            )

    def build_service(runtime: object, **kwargs: object) -> _Service:
        built.append({"runtime": runtime, **kwargs})
        return _Service()

    monkeypatch.setattr(runtime_builder, "build_optional_rag_runtime", fail_rag_runtime)
    monkeypatch.setattr(runtime_builder, "build_agent_service", build_service)

    result = Agent(
        model="qwen3_14b_mlx_4bit",
        knowledge=RAGKnowledgeConfig(vector_backend="sqlite"),
    ).run("lookup policy")

    assert result.answer == "knowledge runner registered"
    assert not hasattr(result, "session_id")
    assert built[0]["runtime"] is None
    assert built[0]["knowledge_runner"] is not None
    provider = built[0]["knowledge_runner"].__self__
    assert provider.vector_dsn == "postgresql://configured-secret"
    assert "configured-secret" not in built[0]["runtime_binding"].model_dump_json()
    assert "knowledge_asset_runner" not in built[0]


def test_agent_facade_closes_service_after_run(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []
    model_runtime_closed: list[bool] = []

    class _Service:
        async def run(
            self,
            request: Any,
        ) -> AgentRunResult:
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="closed",
            )

        async def aclose(self) -> None:
            closed.append(True)

    class _ControlPlane:
        def current_model(self) -> object:
            return SimpleNamespace(id="qwen3_14b_mlx_4bit")

        def close(self) -> None:
            model_runtime_closed.append(True)

    monkeypatch.setattr(runtime_builder, "build_agent_service", lambda *_args, **_kwargs: _Service())
    agent = Agent(model="qwen3_14b_mlx_4bit")
    agent._model_control_plane = cast(
        ModelControlPlane,
        _ControlPlane(),
    )

    result = agent.run("close service")

    assert result.answer == "closed"
    assert not hasattr(result, "session_id")
    assert closed == [True]
    assert model_runtime_closed == [True]


@pytest.mark.anyio
async def test_agent_stream_explicit_close_releases_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []
    runtime_closed = asyncio.Event()
    requests: list[Any] = []

    class _Service:
        async def run_streaming(
            self,
            request: Any,
        ):
            requests.append(request)
            yield StreamEvent(
                type=EventType.TURN_START,
                turn_id=request.turn_id,
                iteration=1,
                sequence=41,
            )
            yield StreamEvent(
                type=EventType.TEXT_DELTA,
                turn_id=request.turn_id,
                iteration=1,
                sequence=42,
                data={"text": "second"},
            )

        async def aclose(self) -> None:
            closed.append(True)
            runtime_closed.set()

    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    stream = Agent().astream(
        "stream task",
        max_turns=4,
    )
    first = await anext(stream)
    await stream.aclose()
    await asyncio.wait_for(runtime_closed.wait(), timeout=1)

    assert first.type is EventType.TURN_START
    assert first.turn_id == requests[0].turn_id
    assert not hasattr(first, "session_id")
    assert first.iteration == 1
    assert first.sequence == 41
    assert closed == [True]
    assert requests[0].max_turns == 4


@pytest.mark.anyio
async def test_agent_aresume_projects_stable_turn_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str | None]] = []

    class _Service:
        async def resume_turn(
            self,
            *,
            turn_id: str,
            action: str,
            user_input: str | None,
        ) -> AgentRunResult:
            calls.append((turn_id, action, user_input))
            return AgentRunResult(
                turn_id=turn_id,
                status="done",
                final_answer="resumed",
            )

    class _RuntimeAgent:
        @asynccontextmanager
        async def _open_product_runtime(self, **_kwargs: object):
            yield _Service()

    agent = Agent()
    monkeypatch.setattr(agent, "_agent_for_turn", lambda _turn_id: _RuntimeAgent())

    result = await agent.aresume(
        "resume-turn",
        "continue",
        user_input="approved",
    )

    assert calls == [("resume-turn", "continue", "approved")]
    assert result.answer == "resumed"
    assert result.turn_id == "resume-turn"
    assert not hasattr(result, "session_id")
    assert not hasattr(result, "raw")


@pytest.mark.anyio
async def test_agent_pending_input_returns_none_for_completed_turn(
    tmp_path: Path,
) -> None:
    agent = Agent(
        checkpoint_db=tmp_path / "agent.sqlite",
        workspace_path=tmp_path,
    )
    store = agent._get_turn_store()
    turn = store.begin_turn(
        "Already done.",
        RuntimeBinding(workspace_path=str(tmp_path.resolve())),
    )
    store.mark_terminal(turn.turn_id, TurnStatus.COMPLETED)

    assert await agent.apending_input(turn.turn_id) is None


@pytest.mark.anyio
async def test_agent_runtime_close_has_a_bounded_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_cancelled: list[bool] = []

    class _Service:
        async def run(
            self,
            request: Any,
        ) -> AgentRunResult:
            return AgentRunResult(
                turn_id=request.turn_id,
                status="done",
                final_answer="done",
            )

        async def aclose(self) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                close_cancelled.append(True)

    monkeypatch.setattr(agent_module, "_RUNTIME_CLOSE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(
        runtime_builder,
        "build_agent_service",
        lambda *_args, **_kwargs: _Service(),
    )

    result = await Agent().arun("bounded close")

    assert result.answer == "done"
    assert close_cancelled == [True]


def test_agent_result_usage_uses_latency_profile_total() -> None:
    raw = AgentRunResult(
        turn_id="sdk-profile",
        status="done",
        final_answer="profiled",
        latency_profile=AgentLatencyProfile(
            total_ms=42.0,
            tool_latency_ms=5.0,
        ),
    )

    result = AgentResult._from_internal(raw)

    assert result.usage.latency_ms == 42.0


@pytest.mark.anyio
async def test_lazy_knowledge_provider_search_uses_typed_final_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, object]] = []

    class _Runtime:
        def query(self, query: str, *, options: object) -> object:
            seen.append((query, options))
            return SimpleNamespace(
                answer=SimpleNamespace(
                    answer_text="Revenue increased.",
                    citations=[
                        SimpleNamespace(
                            citation_anchor="report#revenue",
                            citation_id="citation-1",
                        )
                    ],
                    groundedness_flag=True,
                    insufficient_evidence_flag=False,
                ),
                retrieval=SimpleNamespace(
                    evidence=SimpleNamespace(
                        all=[
                            SimpleNamespace(
                                evidence_id="evidence-1",
                                doc_id=11,
                                citation_anchor="report#revenue",
                                text="Revenue increased by 12%.",
                                score=0.94,
                                source_type="section",
                                file_name="report.pdf",
                            )
                        ]
                    )
                ),
            )

    runtime = _Runtime()

    def build_runtime(**_: object) -> tuple[object, tuple[object, ...]]:
        return runtime, ()

    monkeypatch.setattr(runtime_builder, "build_optional_rag_runtime", build_runtime)

    provider = LazyRAGKnowledgeProvider(config=RAGKnowledgeConfig(vector_backend="sqlite"))
    result = await provider.search_knowledge(
        KnowledgeSearchInput(query="revenue", top_k=1),
        execution_context=cast(Any, object()),
    )

    assert result.total_found == 1
    assert result.answer_text == "Revenue increased."
    assert result.results[0].evidence_id == "evidence-1"
    assert result.citations == ["report#revenue"]
    assert seen[0][0] == "revenue"
    assert seen[0][1].top_k == 1


@pytest.mark.anyio
async def test_configured_knowledge_failure_is_explicit_and_diagnostic(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-index"
    provider = LazyRAGKnowledgeProvider(
        config=RAGKnowledgeConfig(
            storage_root=missing,
            vector_backend="sqlite",
        )
    )

    with pytest.raises(
        RuntimeError,
        match="rag_knowledge_init_failed.*could not be initialized",
    ):
        await provider.search_knowledge(
            KnowledgeSearchInput(query="revenue"),
            execution_context=ToolExecutionContext(),
        )

    assert provider.diagnostics[0].code == "rag_knowledge_init_failed"
    assert provider.diagnostics[0].error_type == "FileNotFoundError"
    assert provider.diagnostics[0].severity == "error"


@pytest.mark.anyio
async def test_knowledge_initialization_redacts_vector_secret_from_all_public_surfaces(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_runtime.cli import _display_agent_result
    from agent_runtime.tools.tool import ToolResult

    marker = "secret-vector-dsn-MARKER"

    def fail_with_secret(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(f"cannot connect to {marker}")

    monkeypatch.setattr(
        "rag.models.runtime.resolve_runtime_config",
        fail_with_secret,
    )
    provider = LazyRAGKnowledgeProvider(
        config=RAGKnowledgeConfig(
            storage_root=tmp_path,
            vector_backend="milvus",
        ),
        vector_dsn=marker,
    )
    assert marker not in repr(provider)

    with caplog.at_level("WARNING"), pytest.raises(RuntimeError) as error:
        await provider.search_knowledge(
            KnowledgeSearchInput(query="revenue"),
            execution_context=ToolExecutionContext(),
        )

    internal = AgentRunResult(
        turn_id="turn-redacted",
        status="failed",
        tool_results=[
            ToolResult(
                tool_call_id="knowledge-redacted",
                tool_name="search_knowledge",
                is_error=True,
                error_code="runner_failed",
                error_message=str(error.value),
            )
        ],
    )
    public = AgentResult._from_internal(internal)
    _display_agent_result(public, verbose=True)

    assert marker not in str(error.value)
    assert marker not in caplog.text
    assert marker not in provider.diagnostics[0].message
    assert marker not in (public.tool_calls[0].error_message or "")
    assert marker not in capsys.readouterr().out


@pytest.mark.anyio
async def test_knowledge_close_failure_redacts_vector_secret_from_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "secret-vector-close-MARKER"

    class _FailingResource:
        def close(self) -> None:
            raise RuntimeError(f"failed to close {marker}")

    with caplog.at_level("WARNING"):
        await agent_module._close_owned_sync_resource(
            _FailingResource(),
            label="knowledge provider",
        )

    assert marker not in caplog.text
    assert "knowledge provider close failed (RuntimeError)" in caplog.text

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel

from agent_runtime.core.context import TurnRegistry
from agent_runtime.core.definition import AgentRuntimePolicy, ToolPolicy
from agent_runtime.core.model_request import ModelCallRecord
from agent_runtime.core.output_models import ValidatedFinalOutput
from agent_runtime.core.turn_contracts import ToolCallPlan
from agent_runtime.loop.runtime import ModelTurnEnvelope
from agent_runtime.loop.state import LoopState, ModelTurnDraft
from agent_runtime.service import AgentRunRequest, AgentRunResult, AgentService
from agent_runtime.tools.builtins.shell import create_run_command_tool
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.selection import ToolConfigurationError
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    json_schema_input,
)
from agent_runtime.turns import RuntimeBinding, TurnStatus, TurnStore
from agent_runtime.workspace import open_workspace
from rag.schema.llm import LLMUsage


class _StructuredAnswer(BaseModel):
    answer: str
    confidence: float


class _FinishProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["tool_results"]:
            latest = state["tool_results"][-1]
            payload = latest.structured_content
            if isinstance(payload, Mapping):
                text = payload.get("text")
                if isinstance(text, str):
                    return ModelTurnDraft(
                        action="finish",
                        final_answer=text,
                    )
        return ModelTurnDraft(action="finish", final_answer="direct answer")


class _UsageProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnEnvelope:
        del state, definition, budget_remaining
        return ModelTurnEnvelope(
            draft=ModelTurnDraft(action="finish", final_answer="metered"),
            model_call_record=ModelCallRecord(
                request_id="request-service",
                prompt_revision="prompt-service",
                toolset_revision="tools-service",
                provider_wire_hash="wire-service",
                usage=LLMUsage(
                    input_tokens=4,
                    output_tokens=2,
                    source="provider",
                    logical_input_tokens=4,
                    uncached_input_tokens=4,
                    usage_source="provider",
                ),
            ),
        )


class _ManifestProvider:
    def __init__(self) -> None:
        self.paths: tuple[str, ...] = ()

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        manifest = state["file_manifest"]
        self.paths = () if manifest is None else tuple(entry.path for entry in manifest.files)
        return ModelTurnDraft(action="finish", final_answer="inspected")


def _tool(
    name: str,
    calls: list[str] | None = None,
    *,
    write: bool = False,
    runner: Callable[[Mapping[str, JsonValue]], object] | None = None,
) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def run(arguments: Mapping[str, JsonValue]) -> object:
        value = str(arguments["value"])
        if calls is not None:
            calls.append(value)
        return {"text": f"ran:{value}"}

    def normalize(raw: object) -> NormalizedToolOutput:
        assert isinstance(raw, Mapping)
        text = str(raw["text"])
        return NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data={"text": text}),),
            structured_content={"text": text},
        )

    effects = frozenset({ToolEffect.WRITE_WORKSPACE}) if write else frozenset()
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Use {name}.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=runner or run,
        normalize_output=normalize,
        output_schema=None,
        static_effects=effects,
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=effects,
            targets=((ToolTarget(kind="workspace_path", value="."),) if write else ()),
        ),
        execution_revision=f"{name}-v1",
        idempotent=not write,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _definition(
    names: tuple[str, ...],
    *,
    output_model: type[BaseModel] | None = None,
    tool_policy: ToolPolicy | None = None,
) -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Use canonical tools.",
        allowed_tools=list(names),
        output_model=output_model,
        max_iterations=4,
        tool_policy=tool_policy,
    )


def test_initial_state_creates_runtime_handles_and_manifest() -> None:
    service = AgentService(
        definition=_definition(("read_file",)),
        tool_registry=_registry(_tool("read_file")),
        model_turn_provider=_FinishProvider(),
    )
    state = service.initial_state(
        AgentRunRequest(
            message="Inspect.",
            turn_id="service-state",
        )
    )

    assert TurnRegistry.get("service-state") is not None
    assert state["resident_tool_names"] == ["read_file"]
    assert state["tool_manifest"] is not None
    assert state["tool_manifest"].resident_tool_names == ("read_file",)
    TurnRegistry.remove("service-state")


@pytest.mark.parametrize("max_turns", [0, -1, True])
def test_run_request_rejects_invalid_max_turns(max_turns: object) -> None:
    with pytest.raises(ValueError, match="max_turns"):
        AgentRunRequest(
            message="Answer.",
            max_turns=max_turns,  # type: ignore[arg-type]
        )


@pytest.mark.anyio
async def test_service_enforces_request_max_turns() -> None:
    calls: list[str] = []

    class _ExecuteThenFinishProvider:
        def __init__(self) -> None:
            self.model_calls = 0

        async def next_turn(
            self,
            state: LoopState,
            *,
            definition: AgentRuntimePolicy,
            budget_remaining: int,
        ) -> ModelTurnDraft:
            del state, definition, budget_remaining
            self.model_calls += 1
            if self.model_calls == 1:
                return ModelTurnDraft(
                    action="execute",
                    tool_calls=(ToolCallPlan.create("echo", {"value": "once"}),),
                )
            return ModelTurnDraft(
                action="finish",
                final_answer="must not be reached",
            )

    provider = _ExecuteThenFinishProvider()
    service = AgentService(
        definition=_definition(("echo",)),
        tool_registry=_registry(_tool("echo", calls)),
        model_turn_provider=provider,
    )

    result = await service.run(
        AgentRunRequest(
            message="Use one model turn.",
            turn_id="service-max-turns",
            max_turns=1,
        )
    )

    assert result.status == "failed"
    assert result.stop_reason == "max_turns"
    assert result.iteration == 1
    assert provider.model_calls == 1
    assert calls == ["once"]


@pytest.mark.anyio
async def test_service_executes_pending_call_through_final_executor() -> None:
    calls: list[str] = []
    service = AgentService(
        definition=_definition(("echo",)),
        tool_registry=_registry(_tool("echo", calls)),
        model_turn_provider=_FinishProvider(),
    )

    result = await service.run(
        AgentRunRequest(
            message="Echo.",
            turn_id="service-tool",
            pending_tool_calls=[ToolCallPlan.create("echo", {"value": "once"})],
        )
    )

    assert result.status == "done"
    assert result.final_answer == "ran:once"
    assert calls == ["once"]
    assert result.tool_results[0].is_error is False


@pytest.mark.anyio
async def test_service_pauses_write_until_approved() -> None:
    service = AgentService(
        definition=_definition(("write_tool",)),
        tool_registry=_registry(_tool("write_tool", write=True)),
        model_turn_provider=_FinishProvider(),
    )

    result = await service.run(
        AgentRunRequest(
            message="Write.",
            turn_id="service-approval",
            pending_tool_calls=[ToolCallPlan.create("write_tool", {"value": "x"})],
        )
    )

    assert result.status == "paused"
    assert result.human_input_request is not None
    assert result.human_input_request.kind == "tool_approval"


@pytest.mark.anyio
async def test_public_explicit_names_replace_defaults_and_disabled_wins() -> None:
    service = AgentService(
        definition=_definition(("first", "second")),
        tool_registry=_registry(_tool("first"), _tool("second")),
        model_turn_provider=_FinishProvider(),
    )
    state = service.initial_state(
        AgentRunRequest(
            message="Answer.",
            turn_id="service-options",
            tools=("second", "first"),
            disabled_tools=("first",),
        )
    )

    assert state["resident_tool_names"] == []
    assert state["explicit_tool_names"] == ["second"]
    assert state["disabled_tool_names"] == ["first"]
    TurnRegistry.remove("service-options")


def test_tool_policy_denials_are_removed_from_the_model_tool_surface() -> None:
    service = AgentService(
        definition=_definition(
            ("first", "second"),
            tool_policy=ToolPolicy(deny_tools=frozenset({"first", "not-installed-here"})),
        ),
        tool_registry=_registry(_tool("first"), _tool("second")),
        model_turn_provider=_FinishProvider(),
    )

    state = service.initial_state(
        AgentRunRequest(
            message="Answer.",
            turn_id="service-policy-deny",
        )
    )

    assert state["resident_tool_names"] == ["second"]
    assert state["disabled_tool_names"] == ["first"]
    assert state["tool_manifest"] is not None
    assert state["tool_manifest"].resident_tool_names == ("second",)
    TurnRegistry.remove("service-policy-deny")


@pytest.mark.anyio
async def test_tool_policy_preserves_each_forced_confirmation_across_resume(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    names = ("one", "two")
    turn_store = TurnStore(tmp_path / "agent.sqlite")
    service = AgentService(
        definition=_definition(
            names,
            tool_policy=ToolPolicy(require_confirmation_for=frozenset(names)),
        ),
        tool_registry=_registry(*(_tool(name, calls) for name in names)),
        model_turn_provider=_FinishProvider(),
        workspace=open_workspace(tmp_path),
        turn_store=turn_store,
        runtime_binding=RuntimeBinding(workspace_path=str(tmp_path)),
    )

    first_pause = await service.run(
        AgentRunRequest(
            message="Confirm each call.",
            pending_tool_calls=[ToolCallPlan.create(name, {"value": name}) for name in names],
        ),
    )

    assert first_pause.status == "paused"
    assert first_pause.human_input_request is not None
    assert [item["tool_name"] for item in first_pause.pending_tool_calls_summary] == ["one", "two"]
    assert calls == []
    first_summary = first_pause.human_input_request.tool_calls[0]
    first_approval_id = first_summary.approval_id or first_summary.tool_call_id
    assert turn_store.get_turn(first_pause.turn_id).status is TurnStatus.PAUSED

    second_pause = await service.resume_turn(
        turn_id=first_pause.turn_id,
        action="allow_once",
        user_input=None,
    )

    assert second_pause.status == "paused"
    assert second_pause.human_input_request is not None
    assert [item["tool_name"] for item in second_pause.pending_tool_calls_summary] == ["two"]
    assert calls == ["one"]
    second_summary = second_pause.human_input_request.tool_calls[0]
    second_approval_id = second_summary.approval_id or second_summary.tool_call_id
    assert second_approval_id != first_approval_id

    completed = await service.resume_turn(
        turn_id=first_pause.turn_id,
        action="allow_once",
        user_input=None,
    )

    assert completed.status == "done"
    assert calls == ["one", "two"]
    assert all(not result.is_error for result in completed.tool_results)
    assert turn_store.get_turn(first_pause.turn_id).status is TurnStatus.COMPLETED


@pytest.mark.anyio
async def test_tool_policy_auto_approves_builtin_restricted_sandbox_call(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    tool = create_run_command_tool(workspace)
    service = AgentService(
        definition=_definition(
            ("run_command",),
            tool_policy=ToolPolicy(auto_approve_sandboxed=True),
        ),
        tool_registry=_registry(tool),
        model_turn_provider=_FinishProvider(),
        workspace=workspace,
    )

    result = await service.run(
        AgentRunRequest(
            message="Run in the sandbox.",
            turn_id="service-policy-sandbox",
            pending_tool_calls=[
                ToolCallPlan.create(
                    "run_command",
                    {"command": "printf sandboxed"},
                )
            ],
        )
    )

    assert result.status == "done"
    assert len(result.tool_results) == 1
    assert result.tool_results[0].error_code != "approval_required"


@pytest.mark.anyio
async def test_tool_policy_caps_safe_parallel_calls() -> None:
    active = 0
    maximum = 0

    def build(name: str) -> Tool:
        async def runner(arguments: Mapping[str, JsonValue]) -> object:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            return {"text": f"ran:{arguments['value']}"}

        return _tool(name, runner=runner)

    names = ("one", "two", "three")
    service = AgentService(
        definition=_definition(
            names,
            tool_policy=ToolPolicy(max_parallel_calls=2),
        ),
        tool_registry=_registry(*(build(name) for name in names)),
        model_turn_provider=_FinishProvider(),
    )

    result = await service.run(
        AgentRunRequest(
            message="Run safely.",
            turn_id="service-policy-parallel",
            pending_tool_calls=[ToolCallPlan.create(name, {"value": name}) for name in names],
        )
    )

    assert result.status == "done"
    assert maximum == 2


def test_unknown_public_tool_fails_before_model_call() -> None:
    service = AgentService(
        definition=_definition(("read_file",)),
        tool_registry=_registry(_tool("read_file")),
        model_turn_provider=_FinishProvider(),
    )

    with pytest.raises(ToolConfigurationError, match="unknown"):
        service.initial_state(
            AgentRunRequest(
                message="Answer.",
                tools=("missing",),
            )
        )


@pytest.mark.anyio
async def test_strict_model_initialization_failure_is_visible() -> None:
    class _BrokenRegistry:
        def resolve_for_node(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("model provider broken")

    service = AgentService(
        definition=_definition(("read_file",)),
        tool_registry=_registry(_tool("read_file")),
        model_registry=_BrokenRegistry(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="model provider broken"):
        await service.run(AgentRunRequest(message="Answer."))


@pytest.mark.anyio
async def test_service_projects_model_call_records_and_latency() -> None:
    service = AgentService(
        definition=_definition(("read_file",)),
        tool_registry=_registry(_tool("read_file")),
        model_turn_provider=_UsageProvider(),
    )

    result = await service.run(
        AgentRunRequest(
            message="Answer.",
            turn_id="service-usage",
        )
    )

    assert result.model_call_records[0].request_id == "request-service"
    assert result.latency_profile is not None
    assert result.latency_profile.total_ms > 0


@pytest.mark.anyio
async def test_service_builds_manifest_for_imported_input_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.txt"
    source.write_text("before\n", encoding="utf-8")
    provider = _ManifestProvider()
    service = AgentService(
        definition=_definition(()),
        tool_registry=_registry(),
        model_turn_provider=provider,
    )

    result = await service.run(
        AgentRunRequest(
            message="Inspect the fixture.",
            turn_id="service-input-manifest",
            input_files=[str(source)],
            workspace_path=str(tmp_path / "workspace"),
        )
    )

    assert result.status == "done"
    assert provider.paths == (
        ".rag/" + "agent_runtime/input_files/service-input-manifest/fixture.txt",
    )


def test_result_restores_configured_concrete_final_output() -> None:
    definition = _definition((), output_model=_StructuredAnswer)
    service = AgentService(
        definition=definition,
        tool_registry=_registry(),
        model_turn_provider=_FinishProvider(),
    )
    state = service.initial_state(
        AgentRunRequest(
            message="Answer.",
            turn_id="service-output",
        )
    )
    state["finish_state"].final_output = ValidatedFinalOutput(
        model_path=(f"{_StructuredAnswer.__module__}.{_StructuredAnswer.__qualname__}"),
        data={"answer": "grounded", "confidence": 0.9},
    )

    result = AgentRunResult.from_loop_result(state, definition=definition)

    assert result.final_output == _StructuredAnswer(
        answer="grounded",
        confidence=0.9,
    )
    TurnRegistry.remove("service-output")

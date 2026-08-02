from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agent_runtime.planning import AgentPlan, PlanStep, PlanTracker
from rag.agent.core import llm_providers as llm_providers_module
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.goal_contract import GoalConstraint, GoalSpec
from rag.agent.core.human_input import HumanInputRequest, ToolCallSummary
from rag.agent.core.llm_context import AgentLLMContextAssembler
from rag.agent.core.llm_providers import (
    LLMLoopModelTurnProvider,
    create_loop_model_turn_provider,
    parse_loop_model_turn,
)
from rag.agent.core.llm_registry import ResolvedModel
from rag.agent.core.messages import (
    ModelMessage,
    StopReason,
    ToolUseResult,
    context_event_message,
    tool_result_message,
)
from rag.agent.core.messages import (
    ToolCall as ModelToolCall,
)
from rag.agent.core.model_request import ToolChoiceMode
from rag.agent.core.observations import StructuredObservation
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.file_manifest import FileManifest, FileManifestEntry
from rag.agent.loop.runtime import AgentLoop
from rag.agent.loop.state import (
    LoopState,
    ModelTurnDraft,
    PendingToolCall,
    StopHookFeedback,
    create_loop_state,
)
from rag.agent.loop.stop_hooks import StopHookRunner
from rag.agent.memory.compactor import LoopContextCompactor
from rag.agent.memory.injector import ContextBuilder
from rag.agent.memory.models import MemoryPolicy
from rag.agent.tools.builtins.filesystem import ReadFileInput
from rag.agent.tools.builtins.shell import RunCommandInput
from rag.agent.tools.executor import ToolExecutor
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
    json_schema_input,
    pydantic_input,
)
from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
from rag.providers.llm_gateway import (
    AgentModelResponse,
    LLMContextOverflowError,
)
from rag.providers.openai_wire import serialize_openai_request
from rag.schema.llm import LLMCallStage, LLMStageBudget, LLMUsage


class _RecordingGateway:
    def __init__(
        self,
        turn: ToolUseResult | None = None,
        *,
        max_input_tokens: int = 32_000,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.max_input_tokens = max_input_tokens
        self.token_accounting = TokenAccountingService(
            TokenizerContract(
                embedding_model_name="recording-gateway",
                tokenizer_model_name="recording-gateway",
                chunking_tokenizer_model_name="recording-gateway",
                tokenizer_backend="simple",
                max_context_tokens=32_768,
                prompt_reserved_tokens=256,
                local_files_only=True,
            )
        )
        self.turn = turn or ToolUseResult(
            text="The policy changed in 2026.",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            raw_stop_reason="stop",
        )

    def effective_stage_budget(
        self,
        stage: LLMCallStage,
        *,
        kwargs: Mapping[str, object] | None = None,
    ) -> LLMStageBudget:
        del stage, kwargs
        return LLMStageBudget(
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=4_096,
        )

    async def agenerate_model_request(self, **kwargs: object) -> AgentModelResponse:
        self.calls.append(dict(kwargs))
        return AgentModelResponse(
            turn=self.turn,
            usage=LLMUsage(
                input_tokens=20,
                output_tokens=4,
                source="provider",
                logical_input_tokens=20,
                uncached_input_tokens=20,
                usage_source="provider",
            ),
            provider_wire_hash="wire-loop-context",
            serializer_revision="provider-wire-v1",
            wire_kind=str(kwargs["provider"]),
        )


class _OverflowOnceRecordingGateway(_RecordingGateway):
    async def agenerate_model_request(
        self,
        **kwargs: object,
    ) -> AgentModelResponse:
        if not self.calls:
            self.calls.append(dict(kwargs))
            request = kwargs["request"]
            wire = serialize_openai_request(request)
            raise LLMContextOverflowError(
                stage=LLMCallStage.TOOL_DECISION,
                input_tokens=self.token_accounting.count(
                    wire.serialized_json
                ),
                max_input_tokens=1,
            )
        return await super().agenerate_model_request(**kwargs)


class _BudgetEnforcingRecordingGateway(_RecordingGateway):
    async def agenerate_model_request(
        self,
        **kwargs: object,
    ) -> AgentModelResponse:
        request = kwargs["request"]
        wire = serialize_openai_request(request)
        input_tokens = self.token_accounting.count(wire.serialized_json)
        if input_tokens > self.max_input_tokens:
            raise LLMContextOverflowError(
                stage=LLMCallStage.TOOL_DECISION,
                input_tokens=input_tokens,
                max_input_tokens=self.max_input_tokens,
            )
        return await super().agenerate_model_request(**kwargs)


class _NoopCheckpoint:
    async def save_snapshot(
        self,
        state: LoopState,
        *,
        reason: str,
    ) -> None:
        del state, reason

    async def write_execution_record(self, record: object) -> None:
        del record


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Use tools when they help and preserve citations.",
        allowed_tools=["vector_search", "read_file"],
    )


def _run_config(run_id: str = "loop-context") -> AgentRunConfig:
    return AgentRunConfig(
        turn_id=run_id,
        llm_budget_total=10_000,
    )


def _state(run_id: str = "loop-context") -> LoopState:
    return create_loop_state(
        current_message="Explain the policy with sources.",
        run_config=_run_config(run_id),
    )


def _tool(name: str) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Use {name}.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=lambda arguments: {"text": str(arguments["query"])},
        normalize_output=lambda raw: NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data={"text": str(raw)}),),
            structured_content={"text": str(raw)},
        ),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision=f"{name}-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=3,
        max_model_output_bytes=4096,
    )


def _tool_with_schema(
    name: str,
    schema: Mapping[str, JsonValue],
) -> Tool:
    return replace(
        _tool(name),
        definition=ToolDefinition(
            name=name,
            description=f"Use {name}.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
    )


def _provider(
    gateway: _RecordingGateway,
    *,
    names: tuple[str, ...] = ("vector_search", "read_file"),
    supports_native_tools: bool = True,
    skill_runtime: object | None = None,
    context_window_tokens: int = 32_768,
    goal_spec: GoalSpec | None = None,
) -> LLMLoopModelTurnProvider:
    snapshot = {name: _tool(name) for name in names}
    return LLMLoopModelTurnProvider(
        gateway,  # type: ignore[arg-type]
        model="test-model",
        provider="openai-compatible",
        supports_native_tools=supports_native_tools,
        registry_snapshot=snapshot,
        resident_tool_names=names,
        context_window_tokens=context_window_tokens,
        skill_runtime=skill_runtime,  # type: ignore[arg-type]
        goal_spec=goal_spec,
    )


def _assembler() -> AgentLLMContextAssembler:
    accounting = TokenAccountingService(
        TokenizerContract(
            embedding_model_name="loop-context",
            tokenizer_model_name="loop-context",
            chunking_tokenizer_model_name="loop-context",
            tokenizer_backend="simple",
            max_context_tokens=8_192,
            prompt_reserved_tokens=256,
            local_files_only=True,
        )
    )
    return AgentLLMContextAssembler(
        token_accounting=accounting,
        stage_budgets={
            LLMCallStage.TOOL_DECISION: LLMStageBudget(
                max_input_tokens=6_000,
                max_output_tokens=1_000,
                safety_margin_tokens=128,
            )
        },
    )


def test_turn_parser_prefers_actual_calls_over_finish_label() -> None:
    call = ToolCallPlan.create("vector_search", {"query": "policy"})

    finish = parse_loop_model_turn({"action": "finish", "final_answer": "Enough evidence."})
    execute = parse_loop_model_turn(
        {
            "action": "finish",
            "final_answer": "Too early.",
            "tool_calls": [call.model_dump()],
        }
    )

    assert finish == ModelTurnDraft(
        action="finish",
        final_answer="Enough evidence.",
    )
    assert execute == ModelTurnDraft(action="execute", tool_calls=(call,))


@pytest.mark.anyio
async def test_long_session_projects_model_context_without_mutating_history() -> None:
    gateway = _RecordingGateway()
    provider = _provider(
        gateway,
        names=(),
        context_window_tokens=512,
    )
    state = _state("long-session-projection")
    state["run_config"] = replace(
        state["run_config"],
        max_context_tokens=512,
    )
    transcript = [
        ModelMessage(
            role="assistant" if index % 2 else "user",
            content=f"message-{index}: " + ("dense-token " * 20),
        )
        for index in range(30)
    ]
    state["turn_transcript"] = list(transcript)

    envelope = await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    assert state["turn_transcript"] == transcript
    assert len(request.messages) < len(transcript) + 2
    assert any("context_compaction" in message.content for message in request.messages)
    assert any("message-29" in message.content for message in request.messages)
    assert envelope.context_revision is not None


@pytest.mark.anyio
async def test_tool_rejection_projects_read_file_schema_delta_without_raw_generation() -> None:
    read_file_schema, _ = pydantic_input(ReadFileInput)
    gateway = _RecordingGateway()
    provider = LLMLoopModelTurnProvider(
        gateway,  # type: ignore[arg-type]
        model="test-model",
        provider="openai-compatible",
        supports_native_tools=True,
        registry_snapshot={
            "read_file": _tool_with_schema("read_file", read_file_schema)
        },
        resident_tool_names=("read_file",),
    )
    state = _state("read-file-tool-correction")
    rejected_generation = (
        '{"name":"read_file","arguments":{"path":"tests/test_agent.py",'
        '"line_start":300,"line_end":380,"encoding":"utf-8"}}'
    )
    state["turn_transcript"].append(
        context_event_message(
            "model_tool_call_rejected",
            {
                "recovery": "correct_tool_arguments",
                "validation_error": (
                    "Tool call validation failed: additional properties "
                    "line_start and line_end are not allowed"
                ),
                "failed_generation": rejected_generation,
            },
        )
    )
    canonical_transcript = list(state["turn_transcript"])

    await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    correction_messages = [
        message
        for message in request.messages
        if '"event_type":"tool_call_correction"' in message.content
    ]
    assert len(correction_messages) == 1
    correction = json.loads(correction_messages[0].content)["payload"]
    assert correction["attempt_count"] == 1
    assert correction["failure_kind"] == "schema_validation"
    assert correction["tool_name"] == "read_file"
    assert correction["required_argument_names"] == ["path"]
    assert correction["rejected_argument_names"] == [
        "line_end",
        "line_start",
    ]
    assert correction["allowed_argument_names"] == [
        "encoding",
        "max_bytes",
        "max_lines",
        "offset",
        "path",
        "start_line",
    ]
    assert correction["additional_properties_allowed"] is False
    assert correction["failed_generation_chars"] == len(rejected_generation)
    assert len(correction["failed_generation_sha256"]) == 64
    assert rejected_generation not in correction_messages[0].content
    assert not any(
        '"event_type":"model_tool_call_rejected"' in message.content
        for message in request.messages
    )
    assert state["turn_transcript"] == canonical_transcript


@pytest.mark.anyio
async def test_duplicate_invalid_json_rejections_become_one_actionable_correction() -> None:
    run_command_schema, _ = pydantic_input(RunCommandInput)
    gateway = _RecordingGateway()
    provider = LLMLoopModelTurnProvider(
        gateway,  # type: ignore[arg-type]
        model="test-model",
        provider="openai-compatible",
        supports_native_tools=True,
        registry_snapshot={
            "run_command": _tool_with_schema(
                "run_command",
                run_command_schema,
            )
        },
        resident_tool_names=("run_command",),
    )
    state = _state("invalid-json-tool-correction")
    state["plan_state"].agent_plan = AgentPlan(
        objective="Repair the tool call and continue the task.",
        steps=[PlanStep(step_id="repair", title="Repair the tool call.")],
    )
    historical = context_event_message(
        "model_tool_call_rejected",
        {
            "recovery": "correct_tool_arguments",
            "validation_error": "earlier schema failure",
            "failed_generation": (
                '<function=run_command>{"command":"pytest -q"}</function>'
            ),
        },
    )
    malformed = (
        '{"name":"run_command","arguments":{"command":"pytest -q"],'
        '"timeout_seconds":600,"working_dir":"."}}'
    )
    rejected = context_event_message(
        "model_tool_call_rejected",
        {
            "recovery": "correct_tool_arguments",
            "validation_error": "Failed to parse tool call arguments as JSON",
            "failed_generation": malformed,
        },
    )
    state["turn_transcript"].extend(
        [
            historical,
            ModelMessage(role="assistant", content="Recovered and continued."),
            rejected,
            rejected,
        ]
    )
    canonical_transcript = list(state["turn_transcript"])

    await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    correction_messages = [
        message
        for message in request.messages
        if '"event_type":"tool_call_correction"' in message.content
    ]
    assert len(correction_messages) == 1
    correction = json.loads(correction_messages[0].content)["payload"]
    assert correction["attempt_count"] == 2
    assert correction["failure_kind"] == "invalid_json"
    assert correction["tool_name"] == "run_command"
    assert correction["required_argument_names"] == ["command"]
    assert correction["rejected_argument_names"] == []
    assert correction["allowed_argument_names"] == [
        "command",
        "network",
        "timeout_seconds",
        "working_dir",
        "workspace_write",
    ]
    assert correction_messages[0] == request.messages[-1]
    assert any(
        '"event_type":"working_state"' in message.content
        for message in request.messages[:-1]
    )
    assert malformed not in correction_messages[0].content
    assert not any(
        '"event_type":"model_tool_call_rejected"' in message.content
        for message in request.messages
    )
    assert state["turn_transcript"] == canonical_transcript


@pytest.mark.anyio
async def test_budget_projection_preserves_latest_failed_tool_pair_with_semantic_evidence() -> None:
    gateway = _BudgetEnforcingRecordingGateway(max_input_tokens=650)
    provider = _provider(
        gateway,
        names=(),
        context_window_tokens=10_000,
    )
    state = _state("latest-failed-tool-pair")
    state["plan_state"].agent_plan = AgentPlan(
        objective="Fix the failing implementation.",
        steps=[PlanStep(step_id="step_fix", title="Fix the implementation.")],
    )
    state["memory_state"].known_locators = [
        {
            "source_tool": "list_files",
            "path": f"src/package/module_{index:02d}.py",
            "name": f"module_{index:02d}.py",
            "size_bytes": 1_024 + index,
            "is_dir": False,
        }
        for index in range(40)
    ]
    state["memory_state"].verified_workspace_paths = [
        str(locator["path"])
        for locator in state["memory_state"].known_locators
    ]
    tool_call_id = "tc-latest-failure"
    command_arguments = {
        "command": "pytest -q",
        "working_dir": ".",
        "timeout_seconds": 120,
        "workspace_write": False,
    }
    stdout = (
        ("old noisy failure detail\n" * 1_000)
        + "FAILED tests/agent/test_context.py::test_keeps_latest_pair - AssertionError\n"
        + "1 failed, 20 passed\n"
    )
    stderr = ("diagnostic noise\n" * 500) + "final stderr evidence\n"
    failed_result = ToolResult(
        tool_call_id=tool_call_id,
        tool_name="run_command",
        structured_content={
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": 1,
            "timed_out": False,
            "truncated": True,
            "duration_ms": 12.0,
        },
        is_error=True,
        error_code="command_failed",
        error_message="command exited with status 1",
        retryable=False,
    )
    state["tool_results"] = [failed_result]
    state["memory_state"].recent_observations = [
        StructuredObservation(
            tool_call_id=tool_call_id,
            tool_name="run_command",
            status="error",
            error="command exited with status 1",
            raw_result_ref=tool_call_id,
        )
    ]
    state["turn_transcript"] = [
        ModelMessage(role="user", content="Fix the implementation and verify it."),
        ModelMessage(
            role="assistant",
            content=("old context " * 600) + "old-history-marker",
        ),
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=(
                ModelToolCall(
                    id=tool_call_id,
                    name="run_command",
                    input=command_arguments,
                ),
            ),
        ),
        tool_result_message(failed_result),
    ]

    await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    wire = serialize_openai_request(request).serialized_json
    assert gateway.token_accounting.count(wire) <= gateway.max_input_tokens
    assistant = next(
        message
        for message in request.messages
        if any(call.id == tool_call_id for call in message.tool_calls)
    )
    tool_message = next(
        message
        for message in request.messages
        if message.role == "tool" and message.tool_call_id == tool_call_id
    )
    assert assistant.tool_calls[0].input == command_arguments
    payload = json.loads(tool_message.content)
    assert payload["error_code"] == "command_failed"
    assert payload["truncated"] is True
    projection = payload["structured_content"]["tool_result_projection"]
    assert projection["exit_code"] == 1
    assert projection["failed_tests"] == [
        "tests/agent/test_context.py::test_keeps_latest_pair"
    ]
    assert projection["stdout_tail"].endswith("1 failed, 20 passed\n")
    assert projection["stderr_tail"].endswith("final stderr evidence\n")
    assert projection["source_truncated"] is True
    assert projection["projection_truncated"] is True
    assert "old-history-marker" not in wire
    assert any(
        '"event_type":"context_compaction"' in message.content
        for message in request.messages
    )
    working_state = next(
        json.loads(message.content)["payload"]["runtime_evidence"]
        for message in request.messages
        if '"event_type":"working_state"' in message.content
    )
    assert working_state["recent_observations"] == []


@pytest.mark.anyio
async def test_repeated_failure_keeps_original_semantic_evidence_chain() -> None:
    gateway = _BudgetEnforcingRecordingGateway(max_input_tokens=1_020)
    provider = _provider(
        gateway,
        names=(
            "search_text",
            "list_files",
            "read_file",
            "apply_patch",
            "run_command",
            "update_plan",
            "invoke_skill",
            "materialize_skill_asset",
            "find_tools",
        ),
        context_window_tokens=10_000,
    )
    state = _state("repeated-failure-evidence-chain")
    state["plan_state"].agent_plan = AgentPlan(
        objective="Recover from the failed verification.",
        steps=[PlanStep(step_id="step_recover", title="Recover.")],
    )
    state["memory_state"].verified_workspace_paths = [
        f"src/package/verified module number {index:02d}.py"
        for index in range(80)
    ]
    original_id = "tc-original-failure"
    repeated_id = "tc-repeated-failure"
    command_arguments = {
        "command": "pytest -q",
        "working_dir": ".",
        "timeout_seconds": 120,
        "workspace_write": False,
    }
    original_result = ToolResult(
        tool_call_id=original_id,
        tool_name="run_command",
        structured_content={
            "stdout": (
                ("failure noise\n" * 1_000)
                + "FAILED tests/agent/test_context.py::test_original_evidence - AssertionError\n"
                + "1 failed, 20 passed\n"
            ),
            "stderr": "final original stderr\n",
            "exit_code": 1,
            "timed_out": False,
            "truncated": True,
        },
        is_error=True,
        error_code="command_failed",
        error_message="command exited with status 1",
        retryable=False,
    )
    repeated_result = ToolResult(
        tool_call_id=repeated_id,
        tool_name="run_command",
        structured_content={
            "repeated_failure": True,
            "original_tool_call_id": original_id,
            "repeat_count": 2,
        },
        is_error=True,
        error_code="repeated_tool_failure",
        error_message="Repeated identical tool call blocked.",
        retryable=False,
    )
    state["tool_results"] = [original_result, repeated_result]
    state["memory_state"].recent_observations = [
        StructuredObservation(
            tool_call_id=original_id,
            tool_name="run_command",
            status="error",
            error="command exited with status 1",
            raw_result_ref=original_id,
        ),
        StructuredObservation(
            tool_call_id=repeated_id,
            tool_name="run_command",
            status="error",
            error="Repeated identical tool call blocked.",
            raw_result_ref=repeated_id,
        ),
    ]
    state["turn_transcript"] = [
        ModelMessage(role="user", content="Fix the implementation and verify it."),
        ModelMessage(
            role="assistant",
            content=("old inspection " * 600) + "obsolete-inspection-tail",
        ),
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=(
                ModelToolCall(
                    id=original_id,
                    name="run_command",
                    input=command_arguments,
                ),
            ),
        ),
        tool_result_message(original_result),
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=(
                ModelToolCall(
                    id=repeated_id,
                    name="run_command",
                    input=command_arguments,
                ),
            ),
        ),
        tool_result_message(repeated_result),
    ]

    await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    wire = serialize_openai_request(request).serialized_json
    assert gateway.token_accounting.count(wire) <= gateway.max_input_tokens
    assert "obsolete-inspection-tail" not in wire
    assert {
        call.id
        for message in request.messages
        for call in message.tool_calls
    } >= {original_id, repeated_id}
    tool_messages = {
        message.tool_call_id: json.loads(message.content)
        for message in request.messages
        if message.role == "tool"
    }
    original_projection = tool_messages[original_id]["structured_content"][
        "tool_result_projection"
    ]
    assert original_projection["failed_tests"] == [
        "tests/agent/test_context.py::test_original_evidence"
    ]
    assert tool_messages[repeated_id]["error_code"] == "repeated_tool_failure"


def test_arbitrary_tool_failure_cannot_pin_an_old_tool_pair() -> None:
    message = tool_result_message(
        ToolResult(
            tool_call_id="tc-current",
            tool_name="untrusted_tool",
            structured_content={
                "original_tool_call_id": "tc-old",
            },
            is_error=True,
            error_code="tool_reported_failure",
            error_message="failed",
        )
    )

    assert (
        llm_providers_module._referenced_original_tool_call_id(message)
        is None
    )


@pytest.mark.anyio
async def test_loop_compactor_proactively_reduces_actual_model_request() -> None:
    baseline_gateway = _RecordingGateway()
    compacted_gateway = _RecordingGateway()
    baseline_provider = _provider(
        baseline_gateway,
        names=(),
        context_window_tokens=32_768,
    )
    compacted_provider = _provider(
        compacted_gateway,
        names=(),
        context_window_tokens=32_768,
    )
    transcript = [
        ModelMessage(
            role="user" if index % 2 == 0 else "assistant",
            content=(
                f"compact-message-{index}: "
                + (f"token-{index} " * 500)
            ),
        )
        for index in range(8)
    ]
    baseline_state = _state("proactive-transcript-baseline")
    baseline_state["run_config"] = replace(
        baseline_state["run_config"],
        memory_policy=MemoryPolicy(
            message_compaction_min_count=99,
            max_message_tail_count=2,
        ),
    )
    baseline_state["turn_transcript"] = list(transcript)
    compacted_state = _state("proactive-transcript-compaction")
    compacted_state["run_config"] = replace(
        compacted_state["run_config"],
        memory_policy=MemoryPolicy(
            message_compaction_min_count=4,
            max_message_tail_count=2,
        ),
    )
    compacted_state["turn_transcript"] = list(transcript)

    await baseline_provider.next_turn(
        baseline_state,
        definition=_definition(),
        budget_remaining=10_000,
    )
    result = LoopContextCompactor().prepare(compacted_state)
    await compacted_provider.next_turn(
        compacted_state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    assert result.changed is True
    assert result.channels == ("turn_transcript",)
    assert compacted_state["turn_transcript"][0] == transcript[0]
    baseline_wire = serialize_openai_request(
        baseline_gateway.calls[0]["request"]
    ).serialized_json
    compacted_request = compacted_gateway.calls[0]["request"]
    compacted_wire = serialize_openai_request(
        compacted_request
    ).serialized_json
    assert len(compacted_wire.encode("utf-8")) < len(
        baseline_wire.encode("utf-8")
    )
    assert compacted_gateway.token_accounting.count(
        compacted_wire
    ) < baseline_gateway.token_accounting.count(baseline_wire)
    assert any(
        '"event_type":"context_compaction"' in message.content
        for message in compacted_request.messages
    )


@pytest.mark.anyio
async def test_under_budget_provider_does_not_compact_by_message_count() -> None:
    gateway = _RecordingGateway()
    provider = _provider(
        gateway,
        names=(),
        context_window_tokens=32_768,
    )
    state = _state("provider-count-is-not-proactive-compaction")
    state["run_config"] = replace(
        state["run_config"],
        memory_policy=MemoryPolicy(
            message_compaction_min_count=4,
            max_message_tail_count=2,
        ),
    )
    transcript = [
        ModelMessage(
            role="user" if index % 2 == 0 else "assistant",
            content=f"under-budget-message-{index}",
        )
        for index in range(8)
    ]
    state["turn_transcript"] = list(transcript)

    await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    assert not any(
        '"event_type":"context_compaction"' in message.content
        for message in request.messages
    )
    for message in transcript:
        assert any(
            candidate.role == message.role
            and candidate.content == message.content
            for candidate in request.messages
        )


@pytest.mark.anyio
async def test_provider_projects_by_actual_gateway_token_budget() -> None:
    gateway = _BudgetEnforcingRecordingGateway(max_input_tokens=520)
    provider = _provider(
        gateway,
        names=(),
        context_window_tokens=10_000,
    )
    state = _state("provider-actual-token-budget")
    state["turn_transcript"] = [
        ModelMessage(role="user", content="u"),
        ModelMessage(role="assistant", content="a " * 800),
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="u",
        steps=[
            PlanStep(
                step_id="step_work",
                title="Work",
            )
        ],
    )

    await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    wire = serialize_openai_request(request).serialized_json
    assert gateway.token_accounting.count(wire) <= 520
    assert any(
        '"event_type":"context_compaction"' in message.content
        for message in request.messages
    )


@pytest.mark.anyio
async def test_agent_loop_avoids_recoverable_actual_token_overflow() -> None:
    gateway = _BudgetEnforcingRecordingGateway(max_input_tokens=528)
    provider = _provider(
        gateway,
        names=(),
        context_window_tokens=10_000,
    )
    state = _state("loop-actual-token-budget")
    state["turn_transcript"] = [
        ModelMessage(role="user", content="u"),
        ModelMessage(role="assistant", content="a " * 800),
    ]
    loop = AgentLoop(
        definition=_definition(),
        model_provider=provider,
        context_manager=LoopContextCompactor(),
        tool_executor=ToolExecutor({}),
        registry_snapshot={},
        execution_context=ToolExecutionContext(),
        checkpoint_store=_NoopCheckpoint(),  # type: ignore[arg-type]
        stop_hook_runner=StopHookRunner(hooks=(), max_blocks=3),
        finish_candidate_builder=FinishCandidateBuilder(),
    )

    result = await loop.run(state)

    assert result["status"] == "completed"
    assert result["memory_state"].reactive_compact_used is False
    assert len(gateway.calls) == 1
    request = gateway.calls[0]["request"]
    wire = serialize_openai_request(request).serialized_json
    assert gateway.token_accounting.count(wire) <= 528
    assert any(
        '"event_type":"context_compaction"' in message.content
        for message in request.messages
    )


@pytest.mark.anyio
async def test_actual_token_projection_handles_no_whitespace_history() -> None:
    accounting = TokenAccountingService(
        TokenizerContract(
            embedding_model_name="gpt-4o-mini",
            tokenizer_model_name="gpt-4o-mini",
            chunking_tokenizer_model_name="gpt-4o-mini",
            tokenizer_backend="tiktoken",
            max_context_tokens=10_000,
            prompt_reserved_tokens=0,
            local_files_only=True,
        )
    )
    gateway = _BudgetEnforcingRecordingGateway(max_input_tokens=1_000)
    gateway.token_accounting = accounting
    provider = _provider(
        gateway,
        names=(),
        context_window_tokens=10_000,
    )
    current_user = "CURRENT USER MUST STAY"
    state = _state("no-whitespace-actual-token-budget")
    state["current_message"] = current_user
    state["conversation_history"] = [
        ModelMessage(role="user", content="initial"),
        ModelMessage(
            role="assistant",
            content="abcdefghijklmno0123456789" * 400,
        ),
    ]
    state["turn_transcript"] = [
        ModelMessage(role="user", content=current_user),
    ]
    loop = AgentLoop(
        definition=_definition(),
        model_provider=provider,
        context_manager=LoopContextCompactor(),
        tool_executor=ToolExecutor({}),
        registry_snapshot={},
        execution_context=ToolExecutionContext(),
        checkpoint_store=_NoopCheckpoint(),  # type: ignore[arg-type]
        stop_hook_runner=StopHookRunner(hooks=(), max_blocks=3),
        finish_candidate_builder=FinishCandidateBuilder(),
    )

    result = await loop.run(state)

    assert result["status"] == "completed"
    assert result["memory_state"].reactive_compact_used is False
    assert len(gateway.calls) == 1
    request = gateway.calls[0]["request"]
    wire = serialize_openai_request(request).serialized_json
    assert gateway.token_accounting.count(wire) <= 1_000
    assert any(
        message.role == "user" and message.content == current_user
        for message in request.messages
    )
    assert any(
        '"event_type":"context_compaction"' in message.content
        for message in request.messages
    )


@pytest.mark.anyio
async def test_budget_projection_preserves_current_user_and_reduces_request() -> None:
    baseline_gateway = _RecordingGateway(max_input_tokens=32_000)
    projected_gateway = _RecordingGateway(max_input_tokens=1_500)
    baseline_provider = _provider(
        baseline_gateway,
        names=(),
        context_window_tokens=32_768,
    )
    projected_provider = _provider(
        projected_gateway,
        names=(),
        context_window_tokens=32_768,
    )
    history = [
        ModelMessage(role="user", content="original session task"),
        *[
                ModelMessage(
                    role="assistant" if index % 2 == 0 else "user",
                    content=(
                        f"historical-message-{index}: "
                        + ("history-token " * 1_000)
                    ),
                )
            for index in range(12)
        ],
    ]
    current_user = "CURRENT USER REQUEST MUST REMAIN EXACT"
    current_turn = [
        ModelMessage(role="user", content=current_user),
        ModelMessage(
            role="assistant",
            content=(
                "current working response "
                + ("current-token " * 1_000)
            ),
        ),
    ]

    def state_for(run_id: str) -> LoopState:
        state = _state(run_id)
        state["run_config"] = replace(
            state["run_config"],
            memory_policy=MemoryPolicy(
                message_compaction_min_count=99,
                max_message_tail_count=1,
            ),
        )
        state["conversation_history"] = list(history)
        state["turn_transcript"] = list(current_turn)
        return state

    await baseline_provider.next_turn(
        state_for("multi-turn-baseline"),
        definition=_definition(),
        budget_remaining=10_000,
    )
    await projected_provider.next_turn(
        state_for("multi-turn-projected"),
        definition=_definition(),
        budget_remaining=10_000,
    )

    baseline_wire = serialize_openai_request(
        baseline_gateway.calls[0]["request"]
    ).serialized_json
    projected_request = projected_gateway.calls[0]["request"]
    projected_wire = serialize_openai_request(
        projected_request
    ).serialized_json
    assert any(
        message.role == "user" and message.content == current_user
        for message in projected_request.messages
    )
    assert len(projected_wire.encode("utf-8")) < len(
        baseline_wire.encode("utf-8")
    )
    assert projected_gateway.token_accounting.count(
        projected_wire
    ) < baseline_gateway.token_accounting.count(baseline_wire)


@pytest.mark.anyio
async def test_reactive_compaction_reduces_next_actual_model_request() -> None:
    gateway = _RecordingGateway()
    provider = _provider(
        gateway,
        names=(),
        context_window_tokens=32_768,
    )
    state = _state("reactive-canonical-transcript")
    state["run_config"] = replace(
        state["run_config"],
        memory_policy=MemoryPolicy(
            message_compaction_min_count=99,
            max_message_tail_count=2,
            reactive_compact_tail_count=2,
        ),
    )
    state["turn_transcript"] = [
        ModelMessage(
            role="user" if index % 2 == 0 else "assistant",
            content=(
                f"oversized-message-{index}: "
                + (f"token-{index} " * 1_000)
            ),
        )
        for index in range(8)
    ]

    await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )
    baseline_request = gateway.calls[-1]["request"]
    baseline_wire = serialize_openai_request(baseline_request)
    baseline_bytes = len(baseline_wire.serialized_json.encode("utf-8"))
    baseline_tokens = gateway.token_accounting.count(
        baseline_wire.serialized_json
    )

    result = LoopContextCompactor().reactive_compact(state)

    assert result.changed is True
    assert result.channels == ("turn_transcript", "memory_warnings")
    await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )
    compacted_request = gateway.calls[-1]["request"]
    compacted_wire = serialize_openai_request(compacted_request)
    compacted_bytes = len(compacted_wire.serialized_json.encode("utf-8"))
    compacted_tokens = gateway.token_accounting.count(
        compacted_wire.serialized_json
    )
    assert compacted_bytes < baseline_bytes
    assert compacted_tokens < baseline_tokens
    assert any(
        '"event_type":"context_compaction"' in message.content
        for message in compacted_request.messages
    )


@pytest.mark.anyio
async def test_agent_loop_overflow_retries_with_smaller_actual_request() -> None:
    gateway = _OverflowOnceRecordingGateway()
    provider = _provider(
        gateway,
        names=(),
        context_window_tokens=32_768,
    )
    state = _state("real-loop-reactive-overflow")
    state["run_config"] = replace(
        state["run_config"],
        memory_policy=MemoryPolicy(
            message_compaction_min_count=99,
            max_message_tail_count=2,
            reactive_compact_tail_count=2,
        ),
    )
    state["turn_transcript"] = [
        ModelMessage(role="user", content="Handle the oversized context."),
        *[
            ModelMessage(
                role="assistant" if index % 2 == 0 else "user",
                content=f"loop-message-{index}: " + (f"token-{index} " * 1_000),
            )
            for index in range(7)
        ],
    ]
    loop = AgentLoop(
        definition=_definition(),
        model_provider=provider,
        context_manager=LoopContextCompactor(),
        tool_executor=ToolExecutor({}),
        registry_snapshot={},
        execution_context=ToolExecutionContext(),
        checkpoint_store=_NoopCheckpoint(),  # type: ignore[arg-type]
        stop_hook_runner=StopHookRunner(hooks=(), max_blocks=3),
        finish_candidate_builder=FinishCandidateBuilder(),
    )

    result = await loop.run(state)

    assert result["status"] == "completed"
    assert len(gateway.calls) == 2
    first_wire = serialize_openai_request(
        gateway.calls[0]["request"]
    ).serialized_json
    second_wire = serialize_openai_request(
        gateway.calls[1]["request"]
    ).serialized_json
    assert len(second_wire.encode("utf-8")) < len(
        first_wire.encode("utf-8")
    )
    assert gateway.token_accounting.count(
        second_wire
    ) < gateway.token_accounting.count(first_wire)
    assert any(
        diagnostic.code == "context_overflow_recovered"
        for diagnostic in result["runtime_diagnostics"]
    )


@pytest.mark.anyio
async def test_loop_provider_injects_compact_typed_working_state() -> None:
    gateway = _RecordingGateway()
    state = _state("typed-working-state")
    state["memory_state"].recent_observations = [
        StructuredObservation(
            tool_call_id="tc-search-runtime",
            tool_name="search_text",
            status="ok",
            locators=[
                {
                    "source_tool": "search_text",
                    "path": "rag/agent/loop/runtime.py",
                    "line_number": 718,
                }
            ],
            raw_result_ref="tc-search-runtime",
        )
    ]
    state["memory_state"].known_locators = [
        {
            "source_tool": "search_text",
            "path": "rag/agent/loop/runtime.py",
            "line_number": 718,
        }
    ]

    await _provider(gateway, names=()).next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    working_state = [
        message
        for message in request.messages
        if '"event_type":"working_state"' in message.content
    ]
    assert len(working_state) == 1
    assert "tc-search-runtime" in working_state[0].content
    assert "rag/agent/loop/runtime.py" in working_state[0].content


@pytest.mark.anyio
async def test_working_state_separates_model_claims_from_runtime_evidence() -> None:
    gateway = _RecordingGateway()
    state = _state("typed-working-state-authority")
    state["plan_state"].agent_plan = AgentPlan(
        objective="Fix the runtime.",
        active_step_id="step_read",
        steps=[
            PlanStep(
                step_id="step_read",
                title="Read the runtime.",
                status="in_progress",
            )
        ],
    )
    state["memory_state"].known_locators = [
        {
            "source_tool": "search_text",
            "path": "rag/agent/loop/runtime.py",
            "line_number": 718,
        }
    ]

    await _provider(gateway, names=()).next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    working_state = next(
        message
        for message in request.messages
        if '"event_type":"working_state"' in message.content
    )
    payload = json.loads(working_state.content)["payload"]
    assert payload["plan_claims"]["authority"] == "advisory"
    assert payload["plan_claims"]["objective"] == "Fix the runtime."
    assert "goal_contract" not in payload
    assert payload["runtime_evidence"]["grounded_paths"] == [
        "rag/agent/loop/runtime.py"
    ]
    assert "unverified_plan_targets" not in payload["runtime_evidence"]
    assert "instruction" not in payload


@pytest.mark.anyio
async def test_working_state_omits_path_only_locators_already_grounded() -> None:
    gateway = _RecordingGateway()
    state = _state("typed-working-state-deduplicated-locators")
    state["memory_state"].known_locators = [
        {
            "source_tool": "list_files",
            "path": "rag/agent/core/llm_providers.py",
            "name": "llm_providers.py",
            "size_bytes": 24_000,
            "is_dir": False,
        },
        {
            "source_tool": "search_text",
            "path": "rag/agent/core/llm_providers.py",
            "line_number": 263,
        },
    ]

    await _provider(gateway, names=()).next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    working_state = next(
        message
        for message in request.messages
        if '"event_type":"working_state"' in message.content
    )
    evidence = json.loads(working_state.content)["payload"]["runtime_evidence"]
    assert evidence["grounded_paths"] == ["rag/agent/core/llm_providers.py"]
    assert evidence["known_locator_count"] == 2
    assert evidence["known_locators_compacted"] is True
    assert evidence["known_locators"] == [
        {
            "source_tool": "search_text",
            "path": "rag/agent/core/llm_providers.py",
            "line_number": 263,
        }
    ]


@pytest.mark.anyio
async def test_runtime_goal_is_frozen_separately_from_advisory_plan() -> None:
    gateway = _RecordingGateway()
    goal = GoalSpec(
        original_query="Implement and verify the requested API change.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            )
        ],
    )
    state = create_loop_state(
        current_message=goal.original_query,
        run_config=_run_config("runtime-goal-context"),
    )
    state["plan_state"].agent_plan = AgentPlan(
        objective=goal.original_query,
        active_step_id="step_other",
        steps=[
            PlanStep(
                step_id="step_other",
                title="Ignore the API change and write a status report.",
                status="in_progress",
            )
        ],
    )

    await _provider(
        gateway,
        names=(),
        goal_spec=goal,
    ).next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    goal_message = next(
        message
        for message in request.messages
        if (
            '"event_type":"frozen_run_context"' in message.content
            and '"name":"goal_contract"' in message.content
        )
    )
    goal_payload = json.loads(goal_message.content)["payload"]["content"]
    assert goal_payload["authority"] == "runtime"
    assert goal_payload["fingerprint"] == goal.fingerprint
    assert goal_payload["spec"] == goal.model_dump(mode="json")

    working_state = next(
        message
        for message in request.messages
        if '"event_type":"working_state"' in message.content
    )
    working_payload = json.loads(working_state.content)["payload"]
    assert working_payload["active_goal"] == {
        "authority": "runtime",
        "fingerprint": goal.fingerprint,
        "original_query": goal.original_query,
    }
    plan_payload = working_payload["plan_claims"]
    assert plan_payload["authority"] == "advisory"
    assert plan_payload["steps"][0]["title"] == (
        "Ignore the API change and write a status report."
    )


@pytest.mark.anyio
async def test_runtime_goal_remains_in_the_dynamic_tail_without_other_state() -> None:
    gateway = _RecordingGateway()
    goal = GoalSpec(
        original_query=(
            "Trace the requested event through the tool, loop, and public CLI."
        )
    )
    state = create_loop_state(
        current_message=goal.original_query,
        run_config=_run_config("runtime-goal-dynamic-tail"),
    )

    await _provider(gateway, names=(), goal_spec=goal).next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    working_state = next(
        message
        for message in request.messages
        if '"event_type":"working_state"' in message.content
    )
    payload = json.loads(working_state.content)["payload"]
    assert payload["active_goal"]["original_query"] == goal.original_query
    assert request.messages[-1] == working_state


@pytest.mark.anyio
async def test_pending_runtime_goal_requirements_are_visible_in_working_state() -> None:
    gateway = _RecordingGateway()
    goal = GoalSpec(
        original_query="Implement and verify the requested API change.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            ),
            GoalConstraint(
                constraint_id="verification_after_change",
                constraint_type="verification_after_change",
                expected_value=True,
            ),
            GoalConstraint(
                constraint_id="no_workspace_change",
                constraint_type="workspace_change",
                expected_value=False,
            ),
        ],
    )
    state = _state("pending-runtime-goal-requirements")

    await _provider(gateway, names=(), goal_spec=goal).next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    working_state = next(
        message
        for message in request.messages
        if '"event_type":"working_state"' in message.content
    )
    payload = json.loads(working_state.content)["payload"]

    assert payload["runtime_requirements"] == [
        {
            "constraint_id": "workspace_change",
            "constraint_type": "workspace_change",
            "expected_value": True,
            "observation": "pending",
            "requirement": (
                "A runtime-observed write must change workspace contents; "
                "prose and pre-change verification do not satisfy this."
            ),
        },
        {
            "constraint_id": "verification_after_change",
            "constraint_type": "verification_after_change",
            "expected_value": True,
            "observation": "pending",
            "requirement": (
                "A recognized verification command must succeed after the "
                "latest workspace change; pre-change commands do not satisfy this."
            ),
        },
    ]


@pytest.mark.anyio
async def test_working_state_uses_durable_workspace_truth_after_projection_loss() -> None:
    gateway = _RecordingGateway()
    state = _state("typed-working-state-durable-evidence")
    state["memory_state"].known_locators = [
        {
            "source_tool": "list_files",
            "path": "agent_runtime/runtime/mcp.py",
        }
    ]
    state["memory_state"].verified_workspace_paths = [
        "agent_runtime/__init__.py",
        "agent_runtime/result.py",
        "agent_runtime/runtime/mcp.py",
    ]
    state["plan_state"].agent_plan = AgentPlan(
        objective="Read the public runtime contract.",
        active_step_id="step_read",
        steps=[
            PlanStep(
                step_id="step_read",
                title="Read agent_runtime/__init__.py.",
                status="in_progress",
            )
        ],
    )

    await _provider(gateway, names=()).next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    working_state = next(
        message
        for message in request.messages
        if '"event_type":"working_state"' in message.content
    )
    payload = json.loads(working_state.content)["payload"]["runtime_evidence"]
    assert payload["grounded_paths"] == [
        "agent_runtime/__init__.py",
        "agent_runtime/result.py",
        "agent_runtime/runtime/mcp.py",
    ]
    assert "unverified_plan_targets" not in payload


@pytest.mark.anyio
async def test_working_state_bounds_path_projection_without_losing_truth() -> None:
    gateway = _RecordingGateway()
    state = _state("typed-working-state-bounded-path-projection")
    verified_paths = [
        f"src/generated/module_{index:03d}.py"
        for index in range(260)
    ]
    state["memory_state"].verified_workspace_paths = verified_paths
    state["plan_state"].agent_plan = AgentPlan(
        objective="Inspect verified targets.",
        active_step_id="step_read",
        steps=[
            PlanStep(
                step_id="step_read",
                title="Inspect the bounded evidence projection.",
                status="in_progress",
            )
        ],
    )

    await _provider(gateway, names=()).next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    working_state = next(
        message
        for message in request.messages
        if '"event_type":"working_state"' in message.content
    )
    payload = json.loads(working_state.content)["payload"]["runtime_evidence"]
    assert payload["grounded_path_count"] == 260
    assert payload["grounded_paths_truncated"] is True
    assert len(payload["grounded_paths"]) == 32
    assert verified_paths[0] not in payload["grounded_paths"]
    assert verified_paths[-1] in payload["grounded_paths"]
    assert "unverified_plan_targets" not in payload


@pytest.mark.anyio
async def test_needs_replan_keeps_finish_and_delivery_available() -> None:
    gateway = _RecordingGateway()
    state = _state("force-replan")
    state["plan_state"].agent_plan = AgentPlan(
        objective="Deliver and verify.",
        status="needs_replan",
        active_step_id="step_read",
        steps=[
            PlanStep(
                step_id="step_read",
                title="Read the exact source location.",
                status="in_progress",
            )
        ],
    )

    await _provider(
        gateway,
        names=("read_file", "update_plan"),
    ).next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    assert request.tool_choice.mode is ToolChoiceMode.AUTO
    assert request.tool_choice.name is None
    working_state = next(
        message
        for message in request.messages
        if '"event_type":"working_state"' in message.content
    )
    assert '"status":"needs_replan"' in working_state.content
    assert "Read the exact source location." in working_state.content


@pytest.mark.anyio
async def test_loop_provider_projects_to_gateway_stage_budget() -> None:
    gateway = _RecordingGateway(max_input_tokens=512)
    provider = _provider(
        gateway,
        names=(),
        context_window_tokens=32_768,
    )
    state = _state("stage-budget-projection")
    transcript = [
        ModelMessage(
            role="assistant" if index % 2 else "user",
            content=f"stage-message-{index}: " + ("dense-token " * 90),
        )
        for index in range(30)
    ]
    state["turn_transcript"] = list(transcript)

    await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    assert state["turn_transcript"] == transcript
    assert len(request.messages) < len(transcript) + 2
    assert any("context_compaction" in message.content for message in request.messages)


@pytest.mark.anyio
async def test_loop_provider_reserves_input_budget_for_tool_schemas() -> None:
    gateway = _RecordingGateway(max_input_tokens=1_800)
    tool = _tool("read_file")
    tool = replace(
        tool,
        definition=ToolDefinition(
            name=tool.definition.name,
            description="schema " * 1_200,
            input_schema=tool.definition.input_schema,
        ),
    )
    provider = LLMLoopModelTurnProvider(
        gateway,  # type: ignore[arg-type]
        model="test-model",
        provider="openai-compatible",
        supports_native_tools=True,
        registry_snapshot={"read_file": tool},
        resident_tool_names=("read_file",),
        context_window_tokens=32_768,
    )
    state = _state("tool-schema-budget")
    state["turn_transcript"] = [
        ModelMessage(
            role="assistant" if index % 2 else "user",
            content=f"history-{index}: " + ("detail " * 120),
        )
        for index in range(20)
    ]

    await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=10_000,
    )

    request = gateway.calls[0]["request"]
    input_tokens = gateway.token_accounting.count(
        serialize_openai_request(request).serialized_json
    )
    assert input_tokens <= gateway.max_input_tokens
    assert any("context_compaction" in message.content for message in request.messages)


@pytest.mark.anyio
async def test_loop_provider_builds_one_canonical_request_and_finish() -> None:
    gateway = _RecordingGateway()
    state = _state()
    state["resident_tool_names"] = ["vector_search", "read_file"]

    envelope = await _provider(gateway).next_turn(
        state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    request = gateway.calls[0]["request"]
    assert request is envelope.request
    assert request.exposed_tool_names == ("vector_search", "read_file")
    assert envelope.draft == ModelTurnDraft(
        action="finish",
        final_answer="The policy changed in 2026.",
    )
    assert envelope.model_call_record is not None
    assert envelope.model_call_record.request_id == request.request_id
    assert envelope.model_call_record.provider_wire_hash == "wire-loop-context"
    assert envelope.context_revision is not None
    assert envelope.context_revision.startswith("context_")
    assert state["latency_profile"].prompt_bytes > 0
    assert state["latency_profile"].tool_schema_bytes > 0


@pytest.mark.anyio
async def test_loop_provider_injects_imported_file_paths_into_canonical_context() -> None:
    gateway = _RecordingGateway()
    state = _state("loop-context-input-files")
    state["resident_tool_names"] = ["read_file"]
    state["file_manifest"] = FileManifest(
        files=[
            FileManifestEntry(
                path="input_files/fixture.txt",
                filename="fixture.txt",
                size_bytes=7,
                mime_type="text/plain",
                file_kind="text",
                hash="abc123",
                structured=False,
                probeable=False,
            )
        ],
        total_size_bytes=7,
        has_structured_files=False,
        has_probeable_files=False,
    )

    envelope = await _provider(
        gateway,
        names=("read_file",),
    ).next_turn(
        state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    assert envelope.request is not None
    assert any(
        message.role == "context" and "input_files/fixture.txt" in message.content
        for message in envelope.request.messages
    )


@pytest.mark.anyio
async def test_loop_provider_binds_tool_call_to_originating_request() -> None:
    gateway = _RecordingGateway(
        ToolUseResult(
            text="",
            tool_calls=[
                ModelToolCall(
                    id="tc-provider",
                    name="read_file",
                    input={"query": "README.md"},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
            raw_stop_reason="tool_calls",
        )
    )
    state = _state("loop-context-origin")
    state["resident_tool_names"] = ["read_file"]

    envelope = await _provider(gateway, names=("read_file",)).next_turn(
        state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    assert envelope.draft.action == "execute"
    [call] = envelope.draft.tool_calls
    assert call.origin is not None
    assert call.origin.request_id == envelope.request.request_id
    assert call.origin.toolset_revision == envelope.request.toolset_revision
    assert call.origin.exposed_tool_names == ("read_file",)


@pytest.mark.anyio
async def test_loop_provider_scopes_reused_provider_tool_ids_per_request() -> None:
    gateway = _RecordingGateway(
        ToolUseResult(
            text="",
            tool_calls=[
                ModelToolCall(
                    id="read_file_9",
                    name="read_file",
                    input={"path": "README.md"},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
            raw_stop_reason="tool_calls",
        )
    )
    provider = _provider(gateway, names=("read_file",))
    first_state = _state("loop-reused-provider-id")
    first_state["resident_tool_names"] = ["read_file"]
    first_state["iteration"] = 9
    second_state = _state("loop-reused-provider-id")
    second_state["resident_tool_names"] = ["read_file"]
    second_state["iteration"] = 10

    first = await provider.next_turn(
        first_state,
        definition=_definition(),
        budget_remaining=5_000,
    )
    second = await provider.next_turn(
        second_state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    first_id = first.draft.tool_calls[0].tool_call_id
    second_id = second.draft.tool_calls[0].tool_call_id
    assert first_id.startswith("tc_")
    assert second_id.startswith("tc_")
    assert first_id != second_id
    assert first.assistant_message is not None
    assert second.assistant_message is not None
    assert first.assistant_message.tool_calls[0].id == first_id
    assert second.assistant_message.tool_calls[0].id == second_id


@pytest.mark.anyio
async def test_loop_provider_selection_is_state_driven_not_task_classified() -> None:
    gateway = _RecordingGateway()
    state = _state("loop-context-selection")
    state["current_message"] = "Answer exactly with the single word: OK"
    state["resident_tool_names"] = ["read_file"]

    envelope = await _provider(gateway, names=("read_file",)).next_turn(
        state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    assert envelope.request.exposed_tool_names == ("read_file",)


@pytest.mark.anyio
async def test_loop_provider_injects_skill_runtime_context() -> None:
    class _SkillContext:
        def render_prompt_context(self, state: LoopState) -> str:
            assert state["current_message"]
            return "<available_skills>project:review</available_skills>"

    gateway = _RecordingGateway()
    state = _state("loop-context-skills")
    state["resident_tool_names"] = ["read_file"]

    envelope = await _provider(
        gateway,
        names=("read_file",),
        skill_runtime=_SkillContext(),
    ).next_turn(
        state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    system_prompt = str(envelope.request.messages[0].content)
    assert "Use tools when they help and preserve citations." in system_prompt
    assert "<available_skills>project:review</available_skills>" in system_prompt


@pytest.mark.anyio
async def test_provider_factory_uses_resolved_wire_capability() -> None:
    gateway = _RecordingGateway()

    class _Registry:
        default_model = "fake"
        fallback_model = "fake"
        generation_config = None

        def resolve_for_node(
            self,
            *,
            node_model: str | None,
            node_name: str,
        ) -> ResolvedModel:
            del node_model, node_name
            return ResolvedModel(
                generator=SimpleNamespace(),
                kwargs={},
                gateway=gateway,
                provider="ollama",
                model="local-model",
                supports_native_tools=False,
            )

    state = _state("loop-context-factory")
    state["resident_tool_names"] = ["read_file"]
    provider = create_loop_model_turn_provider(
        _Registry(),  # type: ignore[arg-type]
        _definition().model_selection,
        registry_snapshot={"read_file": _tool("read_file")},
        resident_tool_names=("read_file",),
    )

    envelope = await provider.next_turn(
        state,
        definition=_definition(),
        budget_remaining=5_000,
    )

    assert envelope.request.settings.model == "local-model"
    assert gateway.calls[0]["provider"] == "ollama"
    assert gateway.calls[0]["supports_native_tools"] is False


def test_loop_context_keeps_approval_and_feedback_without_goal_fields() -> None:
    state = _state()
    call = ToolCallPlan.create("vector_search", {"query": "policy"})
    state["pending_tool_calls"] = [PendingToolCall(plan=call, status="pending")]
    state["approval_request"] = HumanInputRequest(
        request_id="hir_loop",
        kind="tool_approval",
        question="Allow this tool?",
        tool_calls=[
            ToolCallSummary(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                args_preview="query='policy'",
            )
        ],
    )
    state["finish_state"].feedback = [
        StopHookFeedback(
            code="citation_required",
            message="Add a traceable citation.",
        )
    ]
    state["plan_state"].agent_plan = PlanTracker().initialize_task(task=state["current_message"])[0]

    context = ContextBuilder(max_context_tokens=4_000).assemble_loop(
        definition=_definition(),
        state=state,
    )

    decisions = context.section("open_decisions").content
    assert call.tool_call_id in decisions
    assert "tool_approval" in decisions
    assert "Add a traceable citation." in decisions
    assert "open_gaps" not in decisions
    assert "goal_spec" not in decisions


def test_loop_context_assembler_uses_focused_loop_entry_point() -> None:
    assembled = _assembler().assemble_loop_turn(
        definition=_definition(),
        state=_state(),
        budget_remaining=5_000,
        output_schema=ModelTurnDraft,
    )

    assert assembled.stage == LLMCallStage.TOOL_DECISION
    assert "open_gaps" not in assembled.prompt
    assert "Use tools when they help" in assembled.prompt


def test_loop_context_compaction_is_observable_before_model_turn() -> None:
    legacy_messages = [
        HumanMessage(content=f"legacy message {index}", id=f"msg-{index}")
        for index in range(4)
    ]
    state = create_loop_state(
        current_message="Summarize the conversation.",
        run_config=AgentRunConfig(
            turn_id="loop-compaction",
            llm_budget_total=10_000,
            memory_policy=MemoryPolicy(
                message_compaction_min_count=3,
                max_message_tail_count=1,
            ),
        ),
        messages=legacy_messages,
    )
    state["turn_transcript"] = [
        ModelMessage(role="user", content="Summarize the conversation."),
        *[
            ModelMessage(
                role="assistant" if index % 2 == 0 else "user",
                content=f"canonical-{index}: " + (f"token-{index} " * 500),
            )
            for index in range(5)
        ],
    ]

    result = LoopContextCompactor().prepare(state)

    assert result.changed is True
    assert result.channels == ("turn_transcript",)
    assert state["messages"] == legacy_messages
    assert state["memory_state"].working_summary is None
    assert any(
        '"event_type":"context_compaction"' in message.content
        for message in state["turn_transcript"]
    )
    assert state["latest_transition"] is not None
    assert state["latest_transition"].reason == "compaction"


def test_reactive_canonical_compaction_preserves_tool_pairs() -> None:
    tool_call_id = "tc-search"
    state = _state("loop-tool-pair-compaction")
    state["run_config"] = replace(
        state["run_config"],
        memory_policy=MemoryPolicy(
            message_compaction_min_count=99,
            reactive_compact_tail_count=1,
        ),
    )
    state["turn_transcript"] = [
        ModelMessage(role="user", content="Summarize the conversation."),
        *[
            ModelMessage(
                role="assistant" if index % 2 == 0 else "user",
                content=f"old-{index}: " + (f"token-{index} " * 500),
            )
            for index in range(4)
        ],
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=(
                ModelToolCall(
                    id=tool_call_id,
                    name="vector_search",
                    input={"query": "policy"},
                ),
            ),
        ),
        ModelMessage(
            role="tool",
            content="search result",
            tool_call_id=tool_call_id,
        ),
    ]

    result = LoopContextCompactor().reactive_compact(state)

    assert result.changed is True
    assert [message.role for message in state["turn_transcript"][-2:]] == [
        "assistant",
        "tool",
    ]
    assert state["turn_transcript"][-2].tool_calls[0].id == tool_call_id
    assert state["turn_transcript"][-1].tool_call_id == tool_call_id


def test_compaction_never_reformats_canonical_tool_results() -> None:
    state = _state("loop-canonical-tool-results")
    results = [
        ToolResult(
            tool_call_id=f"tc-{index}",
            tool_name="read_file",
            content=(
                ToolContentBlock(
                    type="text",
                    data={"text": f"fixed content {index}"},
                ),
            ),
            structured_content={"text": f"fixed content {index}"},
        )
        for index in range(3)
    ]
    transcript = [tool_result_message(result) for result in results]
    state["tool_results"] = results
    state["turn_transcript"] = transcript

    LoopContextCompactor().prepare(state)

    assert state["tool_results"] == results
    assert state["turn_transcript"] == transcript
    assert [message.content for message in transcript] == [message.content for message in state["turn_transcript"]]

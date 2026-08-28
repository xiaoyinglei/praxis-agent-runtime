from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import pytest
from langgraph.checkpoint.memory import MemorySaver

from rag.agent.core.checkpointing import (
    CheckpointStore,
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from rag.agent.core.context import AgentRunConfig, TurnRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.core.model_request import build_tool_manifest
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import (
    AgentLoop,
    LoopEventSink,
    ModelTurnEnvelope,
    _approval_request,
    _delivery_control_event,
    _guard_exploration_stall,
)
from rag.agent.loop.state import (
    LoopState,
    LoopTransition,
    ModelTurnDraft,
    create_loop_state,
)
from rag.agent.loop.stop_hooks import StopHookRunner
from rag.agent.memory.compactor import LoopCompactionResult
from rag.agent.skills.catalog import SkillCatalog
from rag.agent.skills.loader import scan_and_load_skills
from rag.agent.skills.runtime import SkillRuntime
from rag.agent.streaming.events import (
    EventType,
    ItemDeltaKind,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
    text_delta,
)
from rag.agent.tools.executor import (
    ExecutionStatus,
    ToolExecutionRecord,
    ToolExecutor,
)
from rag.agent.tools.integrations.skills import create_invoke_skill_tool
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.selection import (
    FindToolMatch,
    FindToolsOutput,
    create_find_tools_tool,
)
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolProgress,
    ToolProgressKind,
    ToolResult,
    json_schema_input,
)
from rag.providers.llm_gateway import LLMToolCallValidationError


class _SequenceProvider:
    def __init__(
        self,
        turns: list[ModelTurnDraft | ModelTurnEnvelope | Exception],
    ) -> None:
        self._turns = turns
        self.seen_states: list[LoopState] = []
        self.seen_budget_remaining: list[int] = []

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft | ModelTurnEnvelope:
        del definition
        self.seen_states.append(deepcopy(state))
        self.seen_budget_remaining.append(budget_remaining)
        value = self._turns.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class _SinkAwareProvider:
    def __init__(self) -> None:
        self._stream_sink: object | None = None

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        emit = getattr(self._stream_sink, "emit", None)
        if callable(emit):
            await emit(
                text_delta(
                    "partial",
                    turn_id=state["run_config"].turn_id,
                    iteration=state["iteration"],
                )
            )
        return ModelTurnDraft(action="finish", final_answer="Final answer.")


@dataclass
class _Checkpoint:
    durable: bool = True
    snapshots: list[tuple[str, LoopState]] = field(default_factory=list)
    execution_records: list[ToolExecutionRecord] = field(default_factory=list)

    async def save_snapshot(
        self,
        state: LoopState,
        *,
        reason: str,
    ) -> None:
        self.snapshots.append((reason, deepcopy(state)))

    async def write_execution_record(
        self,
        record: ToolExecutionRecord,
    ) -> None:
        self.execution_records.append(record)


@dataclass
class _Events(LoopEventSink):
    transitions: list[LoopTransition] = field(default_factory=list)

    async def emit(self, transition: LoopTransition) -> None:
        self.transitions.append(transition.model_copy(deep=True))


class _NoCompaction:
    def prepare(self, state: LoopState) -> LoopCompactionResult:
        del state
        return LoopCompactionResult(changed=False)


def _config(
    run_id: str,
    *,
    budget: int | None = 20_000,
    max_turns: int | None = None,
) -> AgentRunConfig:
    TurnRegistry.remove(run_id)
    return AgentRunConfig(
        turn_id=run_id,
        llm_budget_total=budget,
        max_turns=max_turns,
    )


def _definition(
    names: list[str] | tuple[str, ...] = (),
    *,
    max_iterations: int = 10,
) -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Use canonical tools.",
        allowed_tools=list(names),
        max_iterations=max_iterations,
    )


def _tool(
    name: str,
    runner: object,
    *,
    schema: Mapping[str, JsonValue] | None = None,
    effects: frozenset[ToolEffect] = frozenset(),
    metadata: Mapping[str, JsonValue] | None = None,
    idempotent: bool = True,
    stream: object | None = None,
) -> Tool:
    input_schema = schema or {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def normalize(raw: object) -> NormalizedToolOutput:
        text = str(raw)
        return NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data={"text": text}),),
            structured_content={"text": text},
            metadata=metadata or {},
        )

    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Use {name}.",
            input_schema=input_schema,
        ),
        validate_input=json_schema_input(input_schema),
        run=runner,  # type: ignore[arg-type]
        normalize_output=normalize,
        output_schema=None,
        static_effects=effects,
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=effects,
            targets=(),
        ),
        execution_revision=f"{name}-v1",
        idempotent=idempotent,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
        stream=stream,  # type: ignore[arg-type]
    )


def _loop(
    *,
    provider: object,
    tools: tuple[Tool, ...] = (),
    checkpoint: CheckpointStore | None = None,
    definition: AgentRuntimePolicy | None = None,
    events: _Events | None = None,
    max_model_retries: int = 1,
    skill_runtime: SkillRuntime | None = None,
) -> AgentLoop:
    snapshot = {tool.definition.name: tool for tool in tools}
    return AgentLoop(
        definition=definition or _definition(tuple(snapshot)),
        model_provider=provider,  # type: ignore[arg-type]
        context_manager=_NoCompaction(),
        tool_executor=ToolExecutor(snapshot),
        registry_snapshot=snapshot,
        execution_context=ToolExecutionContext(),
        checkpoint_store=checkpoint or _Checkpoint(),
        stop_hook_runner=StopHookRunner(hooks=(), max_blocks=3),
        finish_candidate_builder=FinishCandidateBuilder(),
        event_sink=events or _Events(),
        max_model_retries=max_model_retries,
        skill_runtime=skill_runtime,
    )


async def _collect(events: AsyncIterable[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


@pytest.mark.anyio
async def test_model_tool_result_next_turn_and_finish() -> None:
    call = ToolCallPlan.create("echo", {"value": "hello"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(call,)),
            ModelTurnDraft(action="finish", final_answer="Final answer."),
        ]
    )
    checkpoint = _Checkpoint()
    state = create_loop_state(current_message="Echo.", run_config=_config("loop-basic"))
    state["resident_tool_names"] = ["echo"]

    result = await _loop(
        provider=provider,
        tools=(_tool("echo", lambda arguments: arguments["value"]),),
        checkpoint=checkpoint,
    ).run(state)

    assert result["status"] == "completed"
    assert result["finish_state"].final_answer == "Final answer."
    assert result["tool_results"][0].structured_content == {"text": "hello"}
    assert [message.role for message in result["turn_transcript"][-2:]] == [
        "tool",
        "assistant",
    ]
    assert result["turn_transcript"][-1].content == "Final answer."
    assert any(reason == "tool_results_recorded" for reason, _ in checkpoint.snapshots)
    assert [record.status.value for record in checkpoint.execution_records] == [
        "prepared",
        "started",
        "completed",
    ]


@pytest.mark.anyio
async def test_tool_execution_emits_canonical_item_progress_and_completion() -> None:
    call = ToolCallPlan.create("echo", {"value": "hello"})
    provider = _SequenceProvider(
        [ModelTurnDraft(action="finish", final_answer="Final answer.")]
    )
    state = create_loop_state(
        current_message="Echo.",
        run_config=_config("loop-tool-item"),
        pending_tool_calls=(call,),
    )
    state["resident_tool_names"] = ["echo"]

    async def stream_runner(arguments, progress_sink):
        await progress_sink(
            ToolProgress(ToolProgressKind.PROGRESS, "half", percent=50)
        )
        return arguments["value"]

    events = await _collect(
        _loop(
            provider=provider,
            tools=(
                _tool(
                    "echo",
                    lambda _arguments: "must not run",
                    stream=stream_runner,
                ),
            ),
        ).run_streaming(state)
    )

    canonical = [
        event
        for event in events
        if event.type
        in {
            EventType.ITEM_STARTED,
            EventType.ITEM_DELTA,
            EventType.ITEM_COMPLETED,
        }
    ]
    assert [event.type for event in canonical] == [
        EventType.ITEM_STARTED,
        EventType.ITEM_DELTA,
        EventType.ITEM_COMPLETED,
    ]
    assert all(event.item_kind is TurnItemKind.TOOL for event in canonical)
    assert canonical[1].delta_kind is ItemDeltaKind.TOOL_PROGRESS
    assert canonical[1].data == {"delta": "half"}
    assert canonical[-1].status is ItemStatus.SUCCESS
    assert canonical[-1].data["result"]["structured_content"] == {
        "text": "hello"
    }


@pytest.mark.anyio
async def test_run_command_uses_one_command_item_with_stdout_and_stderr() -> None:
    call = ToolCallPlan.create("run_command", {"value": "ignored"})
    state = create_loop_state(
        current_message="Run.",
        run_config=_config("loop-command-item"),
        pending_tool_calls=(call,),
    )
    state["resident_tool_names"] = ["run_command"]

    async def stream_runner(_arguments, progress_sink):
        await progress_sink(
            ToolProgress(ToolProgressKind.STDOUT, "out")
        )
        await progress_sink(
            ToolProgress(ToolProgressKind.STDERR, "err")
        )
        return "done"

    events = await _collect(
        _loop(
            provider=_SequenceProvider(
                [ModelTurnDraft(action="finish", final_answer="Done.")]
            ),
            tools=(
                _tool(
                    "run_command",
                    lambda _arguments: "must not run",
                    stream=stream_runner,
                ),
            ),
        ).run_streaming(state)
    )

    command_events = [
        event
        for event in events
        if event.item_kind is TurnItemKind.COMMAND
    ]
    assert [event.type for event in command_events] == [
        EventType.ITEM_STARTED,
        EventType.ITEM_DELTA,
        EventType.ITEM_DELTA,
        EventType.ITEM_COMPLETED,
    ]
    assert [event.delta_kind for event in command_events[1:3]] == [
        ItemDeltaKind.COMMAND_STDOUT,
        ItemDeltaKind.COMMAND_STDERR,
    ]
    assert len({event.item_id for event in command_events}) == 1
    assert not any(event.item_kind is TurnItemKind.TOOL for event in events)


@pytest.mark.anyio
async def test_apply_patch_result_event_exposes_only_cli_diff_details() -> None:
    call = ToolCallPlan.create("apply_patch", {"value": "edit"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(call,)),
            ModelTurnDraft(action="finish", final_answer="Done."),
        ]
    )
    state = create_loop_state(
        current_message="Edit the file.",
        run_config=_config("loop-patch-diff"),
    )
    state["resident_tool_names"] = ["apply_patch"]
    tool = _tool(
        "apply_patch",
        lambda _arguments: "patched",
        metadata={
            "file_path": "src/example.py",
            "diff": "--- a/src/example.py\n+++ b/src/example.py\n-old\n+new",
            "diff_truncated": False,
            "private_value": "must-not-leak",
        },
    )

    events = await _collect(_loop(provider=provider, tools=(tool,)).run_streaming(state))

    result = next(event for event in events if event.type is EventType.TOOL_USE_RESULT)
    assert result.data["details"] == {
        "file_path": "src/example.py",
        "diff": "--- a/src/example.py\n+++ b/src/example.py\n-old\n+new",
        "diff_truncated": False,
    }


@pytest.mark.anyio
async def test_approval_pause_is_a_checkpointed_human_input_event() -> None:
    tool = _tool(
        "remote_lookup",
        lambda arguments: arguments["value"],
        effects=frozenset({ToolEffect.NETWORK}),
    )
    call = ToolCallPlan.create("remote_lookup", {"value": "public docs"})
    state = create_loop_state(
        current_message="Look this up.",
        run_config=_config("loop-approval-event"),
        pending_tool_calls=(call,),
    )
    state["resident_tool_names"] = ["remote_lookup"]
    checkpoint = _Checkpoint()

    events = await _collect(
        _loop(
            provider=_SequenceProvider([]),
            tools=(tool,),
            checkpoint=checkpoint,
        ).run_streaming(state)
    )

    assert state["status"] == "paused"
    assert state["tool_results"] == []
    assert any(event.type is EventType.HUMAN_INPUT_REQUIRED for event in events)
    assert not any(event.type is EventType.TOOL_USE_ERROR for event in events)
    assert not any(event.type is EventType.ITEM_STARTED for event in events)
    approval_snapshots = [snapshot for reason, snapshot in checkpoint.snapshots if reason == "tool_pause"]
    assert approval_snapshots[-1]["status"] == "paused"


def test_run_command_approval_shows_full_security_context(tmp_path: Path) -> None:
    command = "printf '\x1b[2J'\npython -c \"print('" + ("x" * 300) + "')\""
    call = ToolCall(
        tool_call_id="call_command",
        tool_name="run_command",
        arguments={
            "command": command,
            "working_dir": ".",
            "timeout_seconds": 120.0,
            "network": True,
        },
        origin=ToolCallOrigin(
            request_id="request_command",
            toolset_revision="tools_v1",
            exposed_tool_names=("run_command",),
        ),
    )
    result = ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        is_error=True,
        error_code="approval_required",
        error_message="approval required for network access",
        retryable=True,
        metadata={
            "approval_id": "call_command::network",
            "approval_scope": "network",
            "cwd": str(tmp_path),
            "network_requested": True,
            "execution_mode": "restricted_sandbox",
        },
    )

    request = _approval_request(result, call)

    summary = request.tool_calls[0]
    assert summary.approval_id == "call_command::network"
    assert json.dumps(command, ensure_ascii=False) in summary.args_preview
    assert "\x1b" not in summary.args_preview
    assert "\\u001b" in summary.args_preview
    assert f"cwd: {json.dumps(str(tmp_path))}" in summary.args_preview
    assert "network: requested (separate approval required)" in summary.args_preview
    assert "execution mode: restricted_sandbox" in summary.args_preview
    assert request.context["approval_scope"] == "network"
    assert "network access" in request.question


@pytest.mark.anyio
async def test_skill_activation_is_checkpointed_with_tool_result(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / ".agents" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\nReview carefully.\n",
        encoding="utf-8",
    )
    runtime = SkillRuntime(SkillCatalog(scan_and_load_skills(tmp_path, repo_root=tmp_path)))
    invoke_tool = create_invoke_skill_tool(runtime.invoke_skill)
    call = ToolCallPlan.create(
        "invoke_skill",
        {"name": "project:review"},
    )
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(call,)),
            ModelTurnDraft(action="finish", final_answer="Reviewed."),
        ]
    )
    checkpoint = _Checkpoint()
    state = create_loop_state(
        current_message="Review this.",
        run_config=_config("loop-skill-activation"),
    )
    state["resident_tool_names"] = ["invoke_skill"]

    result = await _loop(
        provider=provider,
        tools=(invoke_tool,),
        checkpoint=checkpoint,
        skill_runtime=runtime,
    ).run(state)

    assert "project:review" in result["skill_state"].active
    recorded = [snapshot for reason, snapshot in checkpoint.snapshots if reason == "tool_results_recorded"]
    assert recorded
    assert "project:review" in recorded[-1]["skill_state"].active
    assert recorded[-1]["tool_results"][-1].tool_name == "invoke_skill"


@pytest.mark.anyio
async def test_multiple_tool_calls_preserve_model_order() -> None:
    seen: list[str] = []
    calls = (
        ToolCallPlan.create("echo", {"value": "one"}),
        ToolCallPlan.create("echo", {"value": "two"}),
    )
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=calls),
            ModelTurnDraft(action="finish", final_answer="done"),
        ]
    )
    state = create_loop_state(current_message="Echo twice.", run_config=_config("loop-order"))
    state["resident_tool_names"] = ["echo"]

    await _loop(
        provider=provider,
        tools=(
            _tool(
                "echo",
                lambda arguments: seen.append(str(arguments["value"])) or arguments["value"],
            ),
        ),
    ).run(state)

    assert seen == ["one", "two"]


@pytest.mark.anyio
async def test_loop_passes_remaining_budget_to_provider() -> None:
    config = _config("loop-budget", budget=100)
    handles = TurnRegistry.get_or_create(config)
    assert handles.llm_budget_ledger is not None
    assert await handles.llm_budget_ledger.reserve("seed", 35)
    await handles.llm_budget_ledger.commit("seed", 35)
    provider = _SequenceProvider([ModelTurnDraft(action="finish", final_answer="done")])

    await _loop(provider=provider).run(create_loop_state(current_message="Answer.", run_config=config))

    assert provider.seen_budget_remaining == [65]
    TurnRegistry.remove(config.turn_id)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("request_max_turns", "definition_max_iterations", "expected_reason"),
    [
        pytest.param(1, 5, "max_turns", id="request-limit"),
        pytest.param(5, 1, "max_iterations", id="definition-limit"),
    ],
)
async def test_effective_turn_limit_stops_before_another_model_turn(
    request_max_turns: int,
    definition_max_iterations: int,
    expected_reason: str,
) -> None:
    calls: list[str] = []
    first_call = ToolCallPlan.create("echo", {"value": "once"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(first_call,)),
            ModelTurnDraft(action="finish", final_answer="too late"),
        ]
    )
    state = create_loop_state(
        current_message="Use one turn only.",
        run_config=_config(
            f"loop-{expected_reason}",
            max_turns=request_max_turns,
        ),
    )
    state["resident_tool_names"] = ["echo"]

    result = await _loop(
        provider=provider,
        tools=(
            _tool(
                "echo",
                lambda arguments: calls.append(str(arguments["value"])) or arguments["value"],
            ),
        ),
        definition=_definition(
            ("echo",),
            max_iterations=definition_max_iterations,
        ),
    ).run(state)

    assert calls == ["once"]
    assert len(provider.seen_states) == 1
    assert result["iteration"] == 1
    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == expected_reason


@pytest.mark.anyio
async def test_max_turns_stream_does_not_announce_an_unstarted_turn() -> None:
    call = ToolCallPlan.create("echo", {"value": "once"})
    provider = _SequenceProvider([ModelTurnDraft(action="execute", tool_calls=(call,))])
    state = create_loop_state(
        current_message="Emit only real model turns.",
        run_config=_config("loop-max-turn-events", max_turns=1),
    )
    state["resident_tool_names"] = ["echo"]

    events = await _collect(
        _loop(
            provider=provider,
            tools=(_tool("echo", lambda arguments: arguments["value"]),),
            definition=_definition(("echo",), max_iterations=5),
        ).run_streaming(state)
    )

    turn_events = [event for event in events if event.type is EventType.TURN_START]
    assert [event.iteration for event in turn_events] == [1]
    assert [event.turn_id for event in turn_events] == ["loop-max-turn-events"]
    assert all(not hasattr(event, "session_id") for event in turn_events)
    assert all(event.sequence > 0 for event in events)
    assert [event.sequence for event in events] == sorted(event.sequence for event in events)
    assert state["status"] == "failed"
    assert state["terminal"] is not None
    assert state["terminal"].stop_reason == "max_turns"


@pytest.mark.anyio
async def test_provider_error_retries_then_fails() -> None:
    provider = _SequenceProvider([RuntimeError("down"), RuntimeError("down")])
    result = await _loop(provider=provider, max_model_retries=1).run(
        create_loop_state(
            current_message="Answer.",
            run_config=_config("loop-provider-error"),
        )
    )

    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == "model_provider_failed"


@pytest.mark.anyio
async def test_provider_tool_validation_retry_gives_model_corrective_context() -> None:
    provider = _SequenceProvider(
        [
            LLMToolCallValidationError(
                validation_error=("Tool call validation failed: max_bytes exceeds maximum"),
                failed_generation=('<function=read_file>{"path":"README.md","max_bytes":2000000}</function>'),
            ),
            ModelTurnDraft(action="finish", final_answer="recovered"),
        ]
    )

    result = await _loop(provider=provider, max_model_retries=1).run(
        create_loop_state(
            current_message="Read README.md.",
            run_config=_config("loop-provider-tool-validation"),
        )
    )

    assert result["status"] == "completed"
    retry_transcript = provider.seen_states[1]["turn_transcript"]
    feedback = retry_transcript[-1]
    assert feedback.role == "context"
    assert "model_tool_call_rejected" in feedback.content
    assert "max_bytes exceeds maximum" in feedback.content
    assert 'max_bytes\\":2000000' in feedback.content


@pytest.mark.anyio
async def test_run_streaming_injects_sink_and_closes() -> None:
    provider = _SinkAwareProvider()
    events = await asyncio.wait_for(
        _collect(
            _loop(provider=provider).run_streaming(
                create_loop_state(
                    current_message="Stream.",
                    run_config=_config("loop-stream"),
                )
            )
        ),
        timeout=1,
    )

    text_event = next(event for event in events if event.type is EventType.TEXT_DELTA)
    assert text_event.turn_id == "loop-stream"
    assert not hasattr(text_event, "session_id")
    assert text_event.iteration == 1
    assert text_event.sequence > 0
    assert events[-1].type is EventType.LOOP_END


@pytest.mark.anyio
async def test_find_tools_result_and_activation_are_checkpointed_atomically() -> None:
    hidden = _tool("mcp__docs__search", lambda _arguments: "hidden")

    def search(_query: str, _limit: int) -> FindToolsOutput:
        return FindToolsOutput(
            query="documentation",
            matches=(
                FindToolMatch(
                    name=hidden.definition.name,
                    description=hidden.definition.description,
                    score=1.0,
                    matched_terms=("documentation",),
                ),
            ),
            proposed_activation_names=(hidden.definition.name,),
        )

    find_tool = create_find_tools_tool(search)
    snapshot = (find_tool, hidden)
    origin = ToolCallOrigin(
        request_id="request-find",
        toolset_revision="tools-find",
        exposed_tool_names=(find_tool.definition.name,),
    )
    call = ToolCall(
        tool_call_id="call-find",
        tool_name=find_tool.definition.name,
        arguments={"query": "documentation", "limit": 5},
        origin=origin,
    )
    state = create_loop_state(
        current_message="Find documentation.",
        run_config=_config("atomic-tool-activation"),
        pending_tool_calls=(
            ToolCallPlan(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                arguments=dict(call.arguments),
                origin=origin,
            ),
        ),
    )
    state["resident_tool_names"] = [find_tool.definition.name]
    state["canonical_tool_calls"] = {call.tool_call_id: call}
    checkpoint = _Checkpoint()

    result = await _loop(
        provider=_SequenceProvider([ModelTurnDraft(action="finish", final_answer="Found it.")]),
        tools=snapshot,
        checkpoint=checkpoint,
    ).run(state)

    assert result["active_tool_names"] == [hidden.definition.name]
    activation_snapshots = [
        snap for _reason, snap in checkpoint.snapshots if hidden.definition.name in snap["active_tool_names"]
    ]
    assert activation_snapshots
    assert all(
        any(item.tool_call_id == call.tool_call_id for item in snap["tool_results"]) for snap in activation_snapshots
    )


def test_exploration_stall_blocks_inspection_but_keeps_delivery_paths() -> None:
    state = create_loop_state(
        current_message="Make the requested change.",
        run_config=_config("loop-exploration-stall"),
    )
    state["tool_results"] = [
        ToolResult(tool_call_id=f"read-{index}", tool_name="read_file")
        for index in range(20)
    ]
    origin = ToolCallOrigin(
        request_id="delivery-control-request",
        toolset_revision="delivery-control-tools",
        exposed_tool_names=("read_file", "apply_patch"),
    )
    repeated_read = ToolCall(
        tool_call_id="read-new",
        tool_name="read_file",
        arguments={"path": "src/example.py"},
        origin=origin,
    )
    replayed_read = ToolCall(
        tool_call_id="read-replay",
        tool_name="read_file",
        arguments={"path": "src/example.py"},
        origin=origin,
    )
    patch = ToolCall(
        tool_call_id="patch-now",
        tool_name="apply_patch",
        arguments={"file_path": "src/example.py"},
        origin=origin,
    )
    state["tool_execution_records"][replayed_read.tool_call_id] = cast(
        ToolExecutionRecord,
        object(),
    )

    executable, blocked = _guard_exploration_stall(
        state,
        (repeated_read, replayed_read, patch),
    )

    assert executable == (replayed_read, patch)
    assert len(blocked) == 1
    assert blocked[0].tool_call_id == repeated_read.tool_call_id
    assert blocked[0].error_code == "delivery_control"
    assert blocked[0].retryable is False
    assert blocked[0].metadata["delivery_control_already_open"] is False
    event = _delivery_control_event(state)
    assert event is not None
    assert '"threshold":20' in event.content


def test_update_plan_grants_only_one_bounded_focused_extension() -> None:
    state = create_loop_state(
        current_message="Deliver the implementation.",
        run_config=_config("loop-plan-does-not-reopen-exploration"),
    )
    state["tool_results"] = [
        *(
            ToolResult(tool_call_id=f"read-{index}", tool_name="read_file")
            for index in range(20)
        ),
        ToolResult(
            tool_call_id="blocked-read",
            tool_name="read_file",
            is_error=True,
            error_code="delivery_control",
            error_message="exploration closed",
        ),
        ToolResult(tool_call_id="plan-update", tool_name="update_plan"),
    ]
    origin = ToolCallOrigin(
        request_id="plan-escape-request",
        toolset_revision="plan-escape-tools",
        exposed_tool_names=("read_file",),
    )
    calls = tuple(
        ToolCall(
            tool_call_id=f"focused-read-{index}",
            tool_name="read_file",
            arguments={"path": f"src/example_{index}.py"},
            origin=origin,
        )
        for index in range(9)
    )

    executable, blocked = _guard_exploration_stall(state, calls)

    assert executable == calls[:8]
    assert len(blocked) == 1
    assert blocked[0].tool_call_id == calls[8].tool_call_id
    assert blocked[0].metadata["delivery_control_already_open"] is True
    assert blocked[0].metadata["focused_extension_active"] is True


def test_exploration_stall_caps_a_batch_at_the_threshold() -> None:
    state = create_loop_state(
        current_message="Inspect the two remaining files.",
        run_config=_config("loop-exploration-threshold-batch"),
    )
    state["tool_results"] = [
        ToolResult(tool_call_id=f"read-{index}", tool_name="read_file")
        for index in range(19)
    ]
    origin = ToolCallOrigin(
        request_id="delivery-batch-request",
        toolset_revision="delivery-batch-tools",
        exposed_tool_names=("read_file",),
    )
    calls = tuple(
        ToolCall(
            tool_call_id=f"batch-{index}",
            tool_name="read_file",
            arguments={"path": f"src/file_{index}.py"},
            origin=origin,
        )
        for index in range(2)
    )

    executable, blocked = _guard_exploration_stall(state, calls)

    assert executable == calls[:1]
    assert len(blocked) == 1
    assert blocked[0].tool_call_id == calls[1].tool_call_id


@pytest.mark.anyio
async def test_repeated_delivery_control_breach_fails_fast() -> None:
    attempts = 0

    def inspect(_arguments: Mapping[str, JsonValue]) -> str:
        nonlocal attempts
        attempts += 1
        return "unexpected inspection"

    first = ToolCallPlan.create("read_file", {"value": "first"})
    repeated = ToolCallPlan.create("read_file", {"value": "second"})
    state = create_loop_state(
        current_message="Stop exploring and deliver.",
        run_config=_config("loop-delivery-control-fail-fast"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = ["read_file"]
    state["tool_results"] = [
        ToolResult(tool_call_id=f"seed-{index}", tool_name="read_file")
        for index in range(20)
    ]

    result = await _loop(
        provider=_SequenceProvider(
            [ModelTurnDraft(action="execute", tool_calls=(repeated,))]
        ),
        tools=(_tool("read_file", inspect),),
    ).run(state)

    assert attempts == 0
    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == "delivery_stalled"
    assert "cross-file coding" in (result["terminal"].error or "")
    assert [item.error_code for item in result["tool_results"][-2:]] == [
        "delivery_control",
        "delivery_control",
    ]


@pytest.mark.anyio
async def test_repeated_retryable_tool_failure_is_circuited_and_can_recover() -> None:
    attempts: list[str] = []

    def flaky(arguments: Mapping[str, JsonValue]) -> str:
        value = str(arguments["value"])
        attempts.append(value)
        if value == "stuck":
            raise RuntimeError("still stuck")
        return value

    first = ToolCallPlan.create("flaky", {"value": "stuck"})
    second = ToolCallPlan.create("flaky", {"value": "stuck"})
    third = ToolCallPlan.create("flaky", {"value": "stuck"})
    recovery = ToolCallPlan.create("flaky", {"value": "recovered"})
    provider = _SequenceProvider(
        [
            ModelTurnDraft(action="execute", tool_calls=(second,)),
            ModelTurnDraft(action="execute", tool_calls=(third,)),
            ModelTurnDraft(action="execute", tool_calls=(recovery,)),
            ModelTurnDraft(action="finish", final_answer="Recovered."),
        ]
    )
    state = create_loop_state(
        current_message="Recover from a repeated tool failure.",
        run_config=_config("loop-repeated-tool-failure"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = ["flaky"]

    events = await _collect(_loop(provider=provider, tools=(_tool("flaky", flaky),)).run_streaming(state))

    assert state["status"] == "completed"
    assert attempts == ["stuck", "stuck", "recovered"]
    assert [result.error_code for result in state["tool_results"]] == [
        "runner_failed",
        "runner_failed",
        "repeated_tool_failure",
        None,
    ]
    recovery_event = next(
        event
        for event in events
        if event.type is EventType.RECOVERY and event.data.get("strategy") == "tool_failure_circuit_breaker"
    )
    assert "flaky" in str(recovery_event.data["detail"])


@pytest.mark.anyio
async def test_repeated_tool_failure_circuit_uses_checkpointed_history() -> None:
    attempts: list[str] = []

    def flaky(arguments: Mapping[str, JsonValue]) -> str:
        value = str(arguments["value"])
        attempts.append(value)
        if value == "stuck":
            raise RuntimeError("still stuck")
        return value

    tool = _tool("flaky", flaky)
    first = ToolCallPlan.create("flaky", {"value": "stuck"})
    second = ToolCallPlan.create("flaky", {"value": "stuck"})
    state = create_loop_state(
        current_message="Pause after repeated failures.",
        run_config=_config("loop-repeated-tool-failure-resume"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = ["flaky"]
    state["tool_manifest"] = build_tool_manifest(
        tools=(tool,),
        resident_tool_names=("flaky",),
        explicit_tool_names=(),
        active_tool_names=(),
        provider_serializer_revision=state["provider_serializer_revision"],
    )
    checkpoint = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=state["run_config"],
    )

    paused = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(second,)),
                ModelTurnDraft(action="pause", pause_reason="Resume later."),
            ]
        ),
        tools=(tool,),
        checkpoint=checkpoint,
    ).run(state)

    assert paused["status"] == "paused"
    resumed = await checkpoint.load_latest()
    assert resumed is not None
    resumed["status"] = "running"
    resumed["pause"] = None
    third = ToolCallPlan.create("flaky", {"value": "stuck"})
    recovery = ToolCallPlan.create("flaky", {"value": "recovered"})

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(third,)),
                ModelTurnDraft(action="execute", tool_calls=(recovery,)),
                ModelTurnDraft(action="finish", final_answer="Recovered."),
            ]
        ),
        tools=(tool,),
    ).run(resumed)

    assert result["status"] == "completed"
    assert attempts == ["stuck", "stuck", "recovered"]
    assert result["tool_results"][-2].error_code == "repeated_tool_failure"


@pytest.mark.anyio
async def test_repeating_an_open_tool_failure_circuit_fails_fast() -> None:
    attempts = 0

    def always_fails(_arguments: Mapping[str, JsonValue]) -> str:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("still stuck")

    calls = tuple(ToolCallPlan.create("flaky", {"value": "stuck"}) for _ in range(4))
    state = create_loop_state(
        current_message="Stop a repeated failure loop.",
        run_config=_config("loop-repeated-tool-failure-terminal"),
        pending_tool_calls=(calls[0],),
    )
    state["resident_tool_names"] = ["flaky"]

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(calls[1],)),
                ModelTurnDraft(action="execute", tool_calls=(calls[2],)),
                ModelTurnDraft(action="execute", tool_calls=(calls[3],)),
            ]
        ),
        tools=(_tool("flaky", always_fails),),
    ).run(state)

    assert attempts == 2
    assert result["status"] == "failed"
    assert result["terminal"] is not None
    assert result["terminal"].stop_reason == "repeated_tool_failure"


@pytest.mark.anyio
async def test_alternating_failed_calls_do_not_evade_the_circuit() -> None:
    attempts: list[str] = []

    def flaky(arguments: Mapping[str, JsonValue]) -> str:
        value = str(arguments["value"])
        attempts.append(value)
        if value != "recovered":
            raise RuntimeError("still stuck")
        return value

    first_a = ToolCallPlan.create("flaky", {"value": "a"})
    first_b = ToolCallPlan.create("flaky", {"value": "b"})
    second_a = ToolCallPlan.create("flaky", {"value": "a"})
    second_b = ToolCallPlan.create("flaky", {"value": "b"})
    third_a = ToolCallPlan.create("flaky", {"value": "a"})
    recovery = ToolCallPlan.create("flaky", {"value": "recovered"})
    state = create_loop_state(
        current_message="Recover without alternating failed calls forever.",
        run_config=_config("loop-alternating-tool-failures"),
        pending_tool_calls=(first_a,),
    )
    state["resident_tool_names"] = ["flaky"]

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(first_b,)),
                ModelTurnDraft(action="execute", tool_calls=(second_a,)),
                ModelTurnDraft(action="execute", tool_calls=(second_b,)),
                ModelTurnDraft(action="execute", tool_calls=(third_a,)),
                ModelTurnDraft(action="execute", tool_calls=(recovery,)),
                ModelTurnDraft(action="finish", final_answer="Recovered."),
            ]
        ),
        tools=(_tool("flaky", flaky),),
    ).run(state)

    assert result["status"] == "completed"
    assert attempts == ["a", "b", "a", "b", "recovered"]
    assert result["tool_results"][-2].error_code == "repeated_tool_failure"


@pytest.mark.anyio
async def test_non_retryable_failure_opens_circuit_before_second_attempt() -> None:
    attempts: list[str] = []

    def runner(arguments: Mapping[str, JsonValue]) -> str:
        value = str(arguments["value"])
        attempts.append(value)
        if value == "stuck":
            raise RuntimeError("permanent failure")
        return value

    first = ToolCallPlan.create("non_retryable", {"value": "stuck"})
    repeated = ToolCallPlan.create("non_retryable", {"value": "stuck"})
    recovery = ToolCallPlan.create("non_retryable", {"value": "recovered"})
    state = create_loop_state(
        current_message="Do not retry a permanent failure.",
        run_config=_config("loop-non-retryable-tool-failure"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = ["non_retryable"]

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(repeated,)),
                ModelTurnDraft(action="execute", tool_calls=(recovery,)),
                ModelTurnDraft(action="finish", final_answer="Recovered."),
            ]
        ),
        tools=(
            _tool(
                "non_retryable",
                runner,
                idempotent=False,
            ),
        ),
    ).run(state)

    assert result["status"] == "completed"
    assert attempts == ["stuck", "recovered"]
    assert result["tool_results"][-2].error_code == "repeated_tool_failure"


@pytest.mark.anyio
async def test_same_batch_calls_are_not_preempted_before_retry_outcome() -> None:
    attempts = 0

    def transient(_arguments: Mapping[str, JsonValue]) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient failure")
        return "recovered"

    first = ToolCallPlan.create("transient", {"value": "same"})
    batch = tuple(ToolCallPlan.create("transient", {"value": "same"}) for _ in range(2))
    state = create_loop_state(
        current_message="Allow a successful batch retry.",
        run_config=_config("loop-tool-failure-batch-retry"),
        pending_tool_calls=(first,),
    )
    state["resident_tool_names"] = ["transient"]

    result = await _loop(
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=batch),
                ModelTurnDraft(action="finish", final_answer="Recovered."),
            ]
        ),
        tools=(_tool("transient", transient),),
    ).run(state)

    assert result["status"] == "completed"
    assert attempts == 3
    assert all(item.error_code != "repeated_tool_failure" for item in result["tool_results"])


@pytest.mark.anyio
async def test_reconciled_execution_record_precedes_failure_circuit() -> None:
    runner_calls = 0

    def must_not_replay(_arguments: Mapping[str, JsonValue]) -> str:
        nonlocal runner_calls
        runner_calls += 1
        return "unexpected replay"

    tool = _tool("remote_write", must_not_replay, idempotent=False)
    plan = ToolCallPlan.create("remote_write", {"value": "once"})
    origin = ToolCallOrigin(
        request_id="request-before-crash",
        toolset_revision="tools-v1",
        exposed_tool_names=("remote_write",),
    )
    call = ToolCall(
        tool_call_id=plan.tool_call_id,
        tool_name=plan.tool_name,
        arguments=plan.arguments,
        origin=origin,
    )
    state = create_loop_state(
        current_message="Recover a non-idempotent tool outcome.",
        run_config=_config("loop-reconciliation-before-circuit"),
        pending_tool_calls=(
            ToolCallPlan(
                tool_call_id=plan.tool_call_id,
                tool_name=plan.tool_name,
                arguments=plan.arguments,
                origin=origin,
            ),
        ),
    )
    state["resident_tool_names"] = ["remote_write"]
    state["tool_manifest"] = build_tool_manifest(
        tools=(tool,),
        resident_tool_names=("remote_write",),
        explicit_tool_names=(),
        active_tool_names=(),
        provider_serializer_revision=state["provider_serializer_revision"],
    )
    state["canonical_tool_calls"] = {call.tool_call_id: call}
    state["tool_execution_records"][call.tool_call_id] = replace(
        ToolExecutionRecord.prepare(call, tool),
        status=ExecutionStatus.OUTCOME_UNKNOWN,
        attempt_count=1,
        error_code="interrupted_outcome_unknown",
        requires_reconciliation=True,
    )
    checkpoint = LangGraphCheckpointStore(
        MemorySaver(serde=agent_checkpoint_serde()),
        run_config=state["run_config"],
    )

    paused = await _loop(
        provider=_SequenceProvider([]),
        tools=(tool,),
        checkpoint=checkpoint,
    ).run(state)

    assert paused["status"] == "paused"
    request = paused["approval_request"]
    assert request is not None
    assert request.kind == "tool_reconciliation"
    resumed = await checkpoint.apply_human_response(
        HumanInputResponse(
            request_id=request.request_id,
            decision="mark_completed",
        )
    )

    result = await _loop(
        provider=_SequenceProvider([ModelTurnDraft(action="finish", final_answer="Recovered.")]),
        tools=(tool,),
        checkpoint=checkpoint,
    ).run(resumed)

    assert result["status"] == "completed"
    assert runner_calls == 0
    assert result["tool_results"][-1].is_error is False
    assert result["tool_results"][-1].metadata["reconciled"] is True

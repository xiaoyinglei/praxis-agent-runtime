from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_runtime.core.checkpointing import (
    CheckpointStore,
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from agent_runtime.core.context import AgentRunConfig, TurnRegistry
from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.finalization import FinishCandidateBuilder
from agent_runtime.core.human_input import HumanInputResponse
from agent_runtime.core.model_request import build_tool_manifest
from agent_runtime.core.turn_contracts import ToolCallPlan
from agent_runtime.loop.runtime import (
    AgentLoop,
    LoopEventSink,
    ModelTurnEnvelope,
    _approval_request,
    _canonicalize_tool_arguments,
    _guard_repeated_successful_inspections,
    _matching_tool_failures_since_recovery,
)
from agent_runtime.loop.state import (
    LoopState,
    LoopTransition,
    ModelTurnDraft,
    create_loop_state,
)
from agent_runtime.loop.stop_hooks import StopHookRunner
from agent_runtime.memory.compactor import LoopCompactionResult
from agent_runtime.modeling.gateway import LLMToolCallValidationError
from agent_runtime.skills.catalog import SkillCatalog
from agent_runtime.skills.loader import scan_and_load_skills
from agent_runtime.skills.runtime import SkillRuntime
from agent_runtime.streaming.events import (
    EventType,
    ItemDeltaKind,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
    item_completed,
    item_delta,
    item_started,
)
from agent_runtime.tools.executor import (
    ExecutionStatus,
    ToolExecutionRecord,
    ToolExecutor,
)
from agent_runtime.tools.integrations.skills import create_invoke_skill_tool
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.selection import (
    FindToolMatch,
    FindToolsOutput,
    create_find_tools_tool,
)
from agent_runtime.tools.tool import (
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
            item_id = f"agent:{state['run_config'].turn_id}:{state['iteration']}"
            await emit(
                item_started(
                    turn_id=state["run_config"].turn_id,
                    item_id=item_id,
                    item_kind=TurnItemKind.AGENT_MESSAGE,
                    iteration=state["iteration"],
                )
            )
            await emit(
                item_delta(
                    turn_id=state["run_config"].turn_id,
                    item_id=item_id,
                    item_kind=TurnItemKind.AGENT_MESSAGE,
                    delta_kind=ItemDeltaKind.TEXT,
                    delta="partial",
                    iteration=state["iteration"],
                )
            )
            await emit(
                item_completed(
                    turn_id=state["run_config"].turn_id,
                    item_id=item_id,
                    item_kind=TurnItemKind.AGENT_MESSAGE,
                    status=ItemStatus.SUCCESS,
                    data={"content": "partial", "tool_calls": []},
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
async def test_repeated_failure_evidence_is_fingerprinted_in_model_transcript() -> None:
    failure_text = "same failing test output\n" * 100
    attempt = 0

    def fail_with_variable_duration(
        _arguments: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        nonlocal attempt
        attempt += 1
        return {
            "stdout": "",
            "stderr": failure_text,
            "exit_code": 1,
            "duration_ms": attempt,
        }

    def normalize_failure(raw: object) -> NormalizedToolOutput:
        assert isinstance(raw, Mapping)
        return NormalizedToolOutput(
            structured_content=dict(raw),
            is_error=True,
            error_code="command_failed",
            error_message="command exited with status 1",
            retryable=False,
        )

    tool = replace(
        _tool("run_command", fail_with_variable_duration),
        normalize_output=normalize_failure,
    )
    calls = tuple(ToolCallPlan.create("run_command", {"value": command}) for command in ("pytest -q", "pytest --quiet"))
    state = create_loop_state(
        current_message="Run the verification command.",
        run_config=_config("loop-fold-repeated-failure-evidence"),
        pending_tool_calls=calls,
    )
    state["resident_tool_names"] = ["run_command"]

    result = await _loop(
        provider=_SequenceProvider([ModelTurnDraft(action="finish", final_answer="Captured.")]),
        tools=(tool,),
    ).run(state)

    tool_messages = [json.loads(message.content) for message in result["turn_transcript"] if message.role == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0]["structured_content"]["stderr"] == failure_text
    repeated = tool_messages[1]["structured_content"]
    assert repeated["repeated_failure"] is True
    assert repeated["original_tool_call_id"] == calls[0].tool_call_id
    assert repeated["repeat_count"] == 2
    assert len(repeated["evidence_fingerprint"]) == 64
    assert failure_text not in result["turn_transcript"][2].content
    assert [
        item.structured_content["stderr"]
        for item in result["tool_results"]
        if isinstance(item.structured_content, Mapping)
    ] == [failure_text, failure_text]


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

    result = next(event for event in events if event.type is EventType.ITEM_COMPLETED)
    assert result.item_kind is TurnItemKind.TOOL
    assert result.data["result"]["metadata"] == {
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
    assert events == []
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

    assert all(
        event.type in {EventType.ITEM_STARTED, EventType.ITEM_COMPLETED}
        for event in events
    )
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
async def test_provider_error_redacts_credential_identifier_from_state() -> None:
    credential_id = "ak-provider-credential-123456"
    provider = _SequenceProvider([RuntimeError(f"429 rate limit for <{credential_id}>")])

    result = await _loop(provider=provider, max_model_retries=0).run(
        create_loop_state(
            current_message="Answer.",
            run_config=_config("loop-provider-secret-redaction"),
        )
    )

    assert result["terminal"] is not None
    assert credential_id not in (result["terminal"].error or "")
    assert "[REDACTED]" in (result["terminal"].error or "")
    assert all(credential_id not in item.message for item in result["runtime_diagnostics"])
    assert result["latest_transition"] is not None
    assert credential_id not in result["latest_transition"].model_dump_json()


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
async def test_provider_tool_validation_does_not_consume_transport_retry() -> None:
    provider = _SequenceProvider(
        [
            LLMToolCallValidationError(
                validation_error=("Tool call validation failed: timeout_seconds exceeds maximum"),
                failed_generation=('<function=run_command>{"command":"pytest -q","timeout_seconds":1000}</function>'),
            ),
            RuntimeError("request timed out"),
            ModelTurnDraft(action="finish", final_answer="recovered"),
        ]
    )

    result = await _loop(provider=provider, max_model_retries=1).run(
        create_loop_state(
            current_message="Run focused tests.",
            run_config=_config("loop-provider-tool-validation-timeout"),
        )
    )

    assert result["status"] == "completed"
    assert result["iteration"] == 3
    assert any(item.code == "model_tool_call_rejected" for item in result["runtime_diagnostics"])


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

    text_event = next(event for event in events if event.type is EventType.ITEM_DELTA)
    assert text_event.turn_id == "loop-stream"
    assert not hasattr(text_event, "session_id")
    assert text_event.iteration == 1
    assert text_event.sequence > 0
    assert events[-1].type is EventType.ITEM_COMPLETED


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


def test_repeated_successful_inspection_requires_new_arguments() -> None:
    state = create_loop_state(
        current_message="Find the implementation choke point.",
        run_config=_config("loop-repeated-successful-inspection"),
    )
    origin = ToolCallOrigin(
        request_id="inspection-request",
        toolset_revision="inspection-tools",
        exposed_tool_names=("search_text",),
    )
    previous = ToolCall(
        tool_call_id="search-previous",
        tool_name="search_text",
        arguments={"pattern": "system", "path": ""},
        origin=origin,
    )
    repeated = ToolCall(
        tool_call_id="search-repeated",
        tool_name="search_text",
        arguments={"pattern": "system", "path": ""},
        origin=origin,
    )
    narrowed = ToolCall(
        tool_call_id="search-narrowed",
        tool_name="search_text",
        arguments={"pattern": "system", "path": "rag/providers"},
        origin=origin,
    )
    state["canonical_tool_calls"][previous.tool_call_id] = previous
    state["tool_results"] = [
        ToolResult(
            tool_call_id=previous.tool_call_id,
            tool_name=previous.tool_name,
        )
    ]

    executable, blocked = _guard_repeated_successful_inspections(
        state,
        (repeated, narrowed),
    )

    assert executable == (narrowed,)
    assert len(blocked) == 1
    assert blocked[0].tool_call_id == repeated.tool_call_id
    assert blocked[0].error_code == "repeated_inspection"
    assert blocked[0].retryable is False
    assert blocked[0].metadata["previous_tool_call_id"] == previous.tool_call_id
    assert blocked[0].structured_content == {
        "repeated_inspection": True,
        "previous_tool_call_id": previous.tool_call_id,
        "recommended_action": "finish_if_existing_result_satisfies_task",
        "do_not_escalate_for_reconfirmation": True,
        "maximum_additional_file_inspections_for_same_claim": 0,
        "do_not_substitute_another_inspection_tool": True,
    }
    assert "finish now" in (blocked[0].error_message or "")
    assert "Do not repeat the mutation" in (blocked[0].error_message or "")
    assert "run_command or execute_python solely to reconfirm" in (
        blocked[0].error_message or ""
    )
    assert "Do not switch to another inspection tool" in (
        blocked[0].error_message or ""
    )

    state["tool_results"].append(
        ToolResult(
            tool_call_id="command-changed-workspace",
            tool_name="run_command",
            metadata={
                "runtime_workspace_write": True,
                "workspace_tree_before_sha256": "a" * 64,
                "workspace_tree_after_sha256": "b" * 64,
            },
        )
    )
    executable_after_change, blocked_after_change = (
        _guard_repeated_successful_inspections(state, (repeated,))
    )

    assert executable_after_change == (repeated,)
    assert blocked_after_change == ()


def test_repeated_structured_data_inspection_finishes_instead_of_escalating() -> None:
    state = create_loop_state(
        current_message="Create and verify analysis.xlsx.",
        run_config=_config("loop-repeated-data-inspection"),
    )
    origin = ToolCallOrigin(
        request_id="data-inspection-request",
        toolset_revision="data-inspection-tools",
        exposed_tool_names=("inspect_data_file", "execute_python"),
    )
    previous = ToolCall(
        tool_call_id="inspect-data-previous",
        tool_name="inspect_data_file",
        arguments={"path": "analysis.xlsx"},
        origin=origin,
    )
    repeated = ToolCall(
        tool_call_id="inspect-data-repeated",
        tool_name="inspect_data_file",
        arguments={"path": "analysis.xlsx"},
        origin=origin,
    )
    state["canonical_tool_calls"][previous.tool_call_id] = previous
    state["tool_results"] = [
        ToolResult(
            tool_call_id=previous.tool_call_id,
            tool_name=previous.tool_name,
            structured_content={
                "path": "analysis.xlsx",
                "valid": True,
                "sha256": "a" * 64,
            },
        )
    ]

    executable, blocked = _guard_repeated_successful_inspections(
        state,
        (repeated,),
    )

    assert executable == ()
    assert len(blocked) == 1
    assert blocked[0].error_code == "repeated_inspection"
    assert blocked[0].structured_content is not None
    assert blocked[0].structured_content["recommended_action"] == (
        "finish_if_existing_result_satisfies_task"
    )
    assert "execute_python solely to reconfirm" in (
        blocked[0].error_message or ""
    )


def test_canonical_tool_arguments_materialize_declared_defaults() -> None:
    tool = replace(
        _tool(
            "run_command",
            lambda _arguments: "unused",
            schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "working_dir": {"type": "string", "default": "."},
                    "timeout_seconds": {"type": "number", "default": 120.0},
                    "network": {"type": "boolean", "default": False},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        ),
        validate_input=lambda arguments: {
            "command": arguments["command"],
            "working_dir": arguments.get("working_dir", "."),
            "timeout_seconds": arguments.get("timeout_seconds", 120.0),
            "network": arguments.get("network", False),
        },
    )

    omitted = _canonicalize_tool_arguments(
        {"run_command": tool},
        tool_name="run_command",
        arguments={"command": "pytest -q"},
    )
    explicit = _canonicalize_tool_arguments(
        {"run_command": tool},
        tool_name="run_command",
        arguments={
            "command": "pytest -q",
            "working_dir": ".",
            "timeout_seconds": 120.0,
            "network": False,
        },
    )

    assert omitted == explicit


def test_unrelated_read_success_does_not_reset_failed_command_circuit() -> None:
    state = create_loop_state(
        current_message="Implement and verify the change.",
        run_config=_config("loop-command-failure-recovery"),
    )
    origin = ToolCallOrigin(
        request_id="command-failure-request",
        toolset_revision="command-tools",
        exposed_tool_names=("run_command", "list_files", "apply_patch"),
    )
    previous = ToolCall(
        tool_call_id="command-previous",
        tool_name="run_command",
        arguments={
            "command": "pytest -q",
            "working_dir": ".",
            "timeout_seconds": 600.0,
            "network": False,
            "workspace_write": False,
        },
        origin=origin,
    )
    retried = ToolCall(
        tool_call_id="command-retried",
        tool_name="run_command",
        arguments={
            "command": "pytest -q",
            "working_dir": ".",
            "timeout_seconds": 120.0,
            "network": False,
            "workspace_write": False,
        },
        origin=origin,
    )
    state["canonical_tool_calls"][previous.tool_call_id] = previous
    state["tool_results"] = [
        ToolResult(
            tool_call_id=previous.tool_call_id,
            tool_name=previous.tool_name,
            is_error=True,
            error_code="command_failed",
            error_message="command exited with status 1",
        ),
        ToolResult(
            tool_call_id="list-between",
            tool_name="list_files",
        ),
    ]

    assert _matching_tool_failures_since_recovery(state, retried) == (state["tool_results"][0],)

    state["tool_results"].append(
        ToolResult(
            tool_call_id="patch-delivery",
            tool_name="apply_patch",
        )
    )

    assert _matching_tool_failures_since_recovery(state, retried) == ()


def test_delivery_action_reopens_same_inspection() -> None:
    state = create_loop_state(
        current_message="Verify the delivered change.",
        run_config=_config("loop-inspection-after-delivery"),
    )
    origin = ToolCallOrigin(
        request_id="inspection-after-delivery-request",
        toolset_revision="inspection-after-delivery-tools",
        exposed_tool_names=("read_file",),
    )
    previous = ToolCall(
        tool_call_id="read-before-patch",
        tool_name="read_file",
        arguments={"path": "src/example.py"},
        origin=origin,
    )
    verification = ToolCall(
        tool_call_id="read-after-patch",
        tool_name="read_file",
        arguments={"path": "src/example.py"},
        origin=origin,
    )
    state["canonical_tool_calls"][previous.tool_call_id] = previous
    state["tool_results"] = [
        ToolResult(
            tool_call_id=previous.tool_call_id,
            tool_name=previous.tool_name,
        ),
        ToolResult(
            tool_call_id="patch",
            tool_name="apply_patch",
            metadata={
                "workspace_changed": True,
                "file_path": "src/example.py",
                "before_sha256": "a" * 64,
                "after_sha256": "b" * 64,
            },
        ),
    ]

    executable, blocked = _guard_repeated_successful_inspections(
        state,
        (verification,),
    )

    assert executable == (verification,)
    assert blocked == ()


@pytest.mark.anyio
async def test_novel_grounded_inspection_is_not_capped_by_prior_call_count() -> None:
    attempts: list[str] = []
    call = ToolCallPlan.create(
        "read_file",
        {"path": "src/novel.py"},
    )
    state = create_loop_state(
        current_message="Keep following novel grounded evidence.",
        run_config=_config("loop-no-global-inspection-count"),
        pending_tool_calls=(call,),
    )
    state["resident_tool_names"] = ["read_file"]
    state["memory_state"].verified_workspace_paths = ["src/novel.py"]
    state["tool_results"] = [ToolResult(tool_call_id=f"seed-{index}", tool_name="read_file") for index in range(25)]
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    result = await _loop(
        provider=_SequenceProvider([ModelTurnDraft(action="finish", final_answer="Inspected.")]),
        tools=(
            _tool(
                "read_file",
                lambda arguments: attempts.append(str(arguments["path"])) or arguments["path"],
                schema=schema,
            ),
        ),
    ).run(state)

    assert result["status"] == "completed"
    assert attempts == ["src/novel.py"]
    assert result["tool_results"][-1].error_code is None


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
    assert not any(event.type is EventType.RECOVERY for event in events)


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
    assert result["tool_results"][-2].structured_content == {
        "repeated_failure": True,
        "original_tool_call_id": first.tool_call_id,
        "failure_count": 2,
        "last_error_code": "runner_failed",
    }


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

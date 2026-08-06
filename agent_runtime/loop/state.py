from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Protocol, Self

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

from agent_runtime.core.context import AgentRunConfig
from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.human_input import HumanInputRequest, HumanInputResponse
from agent_runtime.core.messages import ModelMessage
from agent_runtime.core.model_request import ModelCallRecord, ModelRequest
from agent_runtime.core.output_models import ValidatedFinalOutput
from agent_runtime.core.runtime_diagnostics import (
    AgentLatencyProfile,
    RuntimeDiagnostic,
    merge_runtime_diagnostics,
)
from agent_runtime.core.turn_contracts import ToolCallPlan, ToolManifest
from agent_runtime.loop.substate import (
    DeferredToolState,
    FinishState,
    MemoryState,
    PlanState,
    StopHookFeedback,
)
from agent_runtime.skills.models import SkillState
from agent_runtime.tools.executor import ToolExecutionRecord
from agent_runtime.tools.tool import ToolCall, ToolResult

if TYPE_CHECKING:
    from agent_runtime.file_manifest import FileManifest

MAX_STOP_HOOK_FEEDBACK = 10
MAX_LOOP_MEMORY_WARNINGS = 20

LoopStatus = Literal["running", "paused", "completed", "failed"]
LoopTransitionReason = Literal[
    "next_turn",
    "tool_execution",
    "approval_required",
    "stop_hook_blocked",
    "retry",
    "fallback",
    "compaction",
    "paused",
    "finished",
    "failed",
    "max_iterations",
    "max_turns",
]


class ModelTurnDraft(BaseModel):
    """Provider output before strict kernel validation."""

    model_config = ConfigDict(frozen=True)

    action: Literal["execute", "finish", "pause"]
    tool_calls: tuple[ToolCallPlan, ...] = ()
    final_answer: str | None = None
    pause_reason: str | None = None


class ModelTurn(BaseModel):
    """A complete, unambiguous model outcome accepted by the loop kernel."""

    model_config = ConfigDict(frozen=True)

    action: Literal["execute", "finish", "pause"]
    tool_calls: tuple[ToolCallPlan, ...] = ()
    final_answer: str | None = None
    pause_reason: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.action == "execute" and not self.tool_calls:
            raise ValueError("execute requires at least one tool call")
        if self.action != "execute" and self.tool_calls:
            raise ValueError("tool calls require execute action")
        if self.action == "finish" and not _nonempty(self.final_answer):
            raise ValueError("finish requires a non-empty final answer")
        if self.action == "pause" and not _nonempty(self.pause_reason):
            raise ValueError("pause requires a non-empty reason")
        return self


class LoopTransition(BaseModel):
    """Latest bounded transition marker persisted with loop state."""

    model_config = ConfigDict(frozen=True)

    reason: LoopTransitionReason
    iteration: int = Field(ge=0)
    detail: dict[str, object] = Field(default_factory=dict)


class ModelTurnEnvelope(BaseModel):
    """Optional provider metadata surrounding one accepted draft."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    draft: ModelTurnDraft
    transitions: tuple[LoopTransition, ...] = ()
    request: ModelRequest | None = None
    model_call_record: ModelCallRecord | None = None
    assistant_message: ModelMessage | None = None
    context_revision: str | None = None
    provider_serializer_revision: str | None = None


class ModelTurnProvider(Protocol):
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft | ModelTurnEnvelope: ...


class LoopPause(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str = Field(min_length=1, max_length=1000)
    request: HumanInputRequest | None = None


class LoopTerminal(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["completed", "failed"]
    stop_reason: str = Field(min_length=1, max_length=200)
    final_answer: str | None = None
    final_output: ValidatedFinalOutput | None = None
    error: str | None = Field(default=None, max_length=2000)


class PendingToolCall(BaseModel):
    """Single canonical pending tool call. Replaces ToolCallPlan-as-pending + old PendingToolCall."""

    plan: ToolCallPlan
    status: Literal["pending", "approved", "denied", "running", "completed", "failed"]
    approval_request_id: str | None = None
    operation_id: str | None = None
    summary: str | None = None

    @property
    def tool_call_id(self) -> str:
        return self.plan.tool_call_id

    @property
    def tool_name(self) -> str:
        return self.plan.tool_name


class ToolCallLedgerEntry(BaseModel):
    """Transcript source for one tool call — no runtime state, just plan + position."""

    plan: ToolCallPlan
    turn: int
    sequence: int


class ToolCallLedger(BaseModel):
    """Bounded ledger of all tool calls for native transcript rebuild.
    Only cleaned when entries are no longer needed for transcript reconstruction.
    """

    entries: list[ToolCallLedgerEntry] = Field(default_factory=list)
    max_entries: int = 128

    def append_plans(self, plans: Iterable[ToolCallPlan], *, turn: int) -> None:
        """Record model-requested calls idempotently; do not store pending state."""
        existing = {entry.plan.tool_call_id for entry in self.entries}
        for plan in plans:
            if plan.tool_call_id in existing:
                continue
            self.entries.append(
                ToolCallLedgerEntry(
                    plan=plan,
                    turn=turn,
                    sequence=len(self.entries),
                )
            )
            existing.add(plan.tool_call_id)

    def trim(self, *, active_tool_call_ids: set[str]) -> None:
        """Remove oldest non-active entries when over max_entries."""
        while len(self.entries) > self.max_entries:
            for index, entry in enumerate(self.entries):
                if entry.plan.tool_call_id not in active_tool_call_ids:
                    self.entries.pop(index)
                    break
            else:
                break


class LoopState(TypedDict):
    current_message: str
    conversation_history: list[ModelMessage]
    messages: list[BaseMessage]
    run_config: AgentRunConfig
    iteration: int
    status: LoopStatus
    pending_tool_calls: list[PendingToolCall]  # single-track
    tool_call_ledger: ToolCallLedger  # bounded transcript source
    tool_execution_records: dict[str, ToolExecutionRecord]
    approval_request: HumanInputRequest | None
    approval_response: HumanInputResponse | None
    approved_tool_call_ids: list[str]
    denied_tool_call_ids: list[str]
    tool_results: list[ToolResult]
    turn_transcript: list[ModelMessage]
    canonical_tool_calls: dict[str, ToolCall]
    model_call_records: list[ModelCallRecord]
    tool_manifest: ToolManifest | None
    tool_checkpoint: dict[str, object] | None
    context_revision: str
    prompt_revision: str
    provider_serializer_revision: str
    resident_tool_names: list[str]
    explicit_tool_names: list[str]
    active_tool_names: list[str]
    disabled_tool_names: list[str]
    allow_write_tools: bool
    allow_execute_tools: bool
    allow_discovery_tools: bool
    runtime_diagnostics: list[RuntimeDiagnostic]
    latency_profile: AgentLatencyProfile
    last_model_turn: ModelTurn | None
    pause: LoopPause | None
    terminal: LoopTerminal | None
    latest_transition: LoopTransition | None
    # ── Typed sub-state containers ──
    plan_state: PlanState
    memory_state: MemoryState
    deferred_tool_state: DeferredToolState
    finish_state: FinishState
    skill_state: SkillState
    # ── File manifest (file-first processing) ──
    input_files: list[str]
    file_manifest: FileManifest | None


def create_loop_state(
    *,
    current_message: str,
    run_config: AgentRunConfig,
    conversation_history: Iterable[ModelMessage] = (),
    turn_transcript: Iterable[ModelMessage] = (),
    messages: Iterable[BaseMessage] = (),
    pending_tool_calls: Iterable[ToolCallPlan] = (),
    memory_warnings: Iterable[str] = (),
    runtime_diagnostics: Iterable[RuntimeDiagnostic] = (),
    input_files: Iterable[str] = (),
    file_manifest: FileManifest | None = None,
) -> LoopState:
    current_turn = list(turn_transcript)
    if not current_turn:
        current_turn.append(
            ModelMessage(role="user", content=current_message)
        )

    return {
        "current_message": current_message,
        "conversation_history": list(conversation_history),
        "messages": list(messages),
        "run_config": run_config,
        "iteration": 0,
        "status": "running",
        "pending_tool_calls": [PendingToolCall(plan=call, status="pending") for call in pending_tool_calls],
        "tool_call_ledger": ToolCallLedger() if not pending_tool_calls
        else ToolCallLedger(entries=[
            ToolCallLedgerEntry(plan=call, turn=0, sequence=i)
            for i, call in enumerate(pending_tool_calls)
        ]),
        "tool_execution_records": {},
        "approval_request": None,
        "approval_response": None,
        "approved_tool_call_ids": [],
        "denied_tool_call_ids": [],
        "tool_results": [],
        "turn_transcript": current_turn,
        "canonical_tool_calls": {},
        "model_call_records": [],
        "tool_manifest": None,
        "tool_checkpoint": None,
        "context_revision": "context_pending",
        "prompt_revision": "prompt_pending",
        "provider_serializer_revision": "provider-wire-v1",
        "resident_tool_names": [],
        "explicit_tool_names": [],
        "active_tool_names": [],
        "disabled_tool_names": [],
        "allow_write_tools": False,
        "allow_execute_tools": False,
        "allow_discovery_tools": False,
        "runtime_diagnostics": merge_runtime_diagnostics([], runtime_diagnostics),
        "latency_profile": AgentLatencyProfile(),
        "last_model_turn": None,
        "pause": None,
        "terminal": None,
        "latest_transition": None,
        # ── File manifest (file-first processing) ──
        "input_files": list(input_files),
        "file_manifest": file_manifest,
        # ── Typed sub-state containers ──
        "plan_state": PlanState(
            agent_plan=None,
            plan_events=[],
        ),
        "memory_state": MemoryState(
            working_summary=None,
            extracted_facts=[],
            context_budget=None,
            memory_refs=[],
            memory_budget=None,
            memory_warnings=_bounded_unique_strings(
                memory_warnings,
                limit=MAX_LOOP_MEMORY_WARNINGS,
            ),
            reactive_compact_used=False,
        ),
        "deferred_tool_state": DeferredToolState(),
        "finish_state": FinishState(
            final_answer=None,
            final_output=None,
            output_validation_errors=[],
        ),
        "skill_state": SkillState(),
    }


def materialize_model_turn(
    draft: ModelTurnDraft,
) -> ModelTurn:
    """Apply tool-call precedence before the strict kernel contract."""

    if draft.tool_calls:
        return ModelTurn(action="execute", tool_calls=draft.tool_calls)
    if draft.action == "finish":
        return ModelTurn(action="finish", final_answer=draft.final_answer)
    return ModelTurn(
        action=draft.action,
        final_answer=draft.final_answer,
        pause_reason=draft.pause_reason,
    )


def replace_latest_transition(
    state: LoopState,
    transition: LoopTransition,
) -> None:
    state["latest_transition"] = transition


def append_stop_hook_feedback(
    state: LoopState,
    feedback: StopHookFeedback,
) -> StopHookFeedback:
    existing = next(
        (
            item
            for item in state["finish_state"].feedback
            if item.code == feedback.code and item.message == feedback.message
        ),
        None,
    )
    updated = feedback.model_copy(
        update={"occurrences": (feedback.occurrences if existing is None else existing.occurrences + 1)}
    )
    items = [item for item in state["finish_state"].feedback if item is not existing]
    state["finish_state"].feedback = [*items, updated][-MAX_STOP_HOOK_FEEDBACK:]
    return updated


def append_stop_hook_warning(
    state: LoopState,
    warning: StopHookFeedback,
) -> StopHookFeedback:
    existing = next(
        (
            item
            for item in state["finish_state"].warnings
            if item.code == warning.code and item.message == warning.message
        ),
        None,
    )
    updated = warning.model_copy(
        update={"occurrences": (warning.occurrences if existing is None else existing.occurrences + 1)}
    )
    items = [item for item in state["finish_state"].warnings if item is not existing]
    state["finish_state"].warnings = [*items, updated][-MAX_STOP_HOOK_FEEDBACK:]
    return updated


def append_memory_warning(state: LoopState, warning: str) -> None:
    state["memory_state"].memory_warnings = _bounded_unique_strings(
        [*state["memory_state"].memory_warnings, warning],
        limit=MAX_LOOP_MEMORY_WARNINGS,
    )


def append_loop_diagnostic(
    state: LoopState,
    diagnostic: RuntimeDiagnostic,
) -> None:
    state["runtime_diagnostics"] = merge_runtime_diagnostics(
        state["runtime_diagnostics"],
        [diagnostic],
    )


def _bounded_unique_strings(values: Iterable[str], *, limit: int) -> list[str]:
    merged: dict[str, None] = {}
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        merged.pop(normalized, None)
        merged[normalized] = None
    return list(merged)[-limit:]


def _nonempty(value: str | None) -> bool:
    return bool(value and value.strip())


__all__ = [
    "MAX_LOOP_MEMORY_WARNINGS",
    "MAX_STOP_HOOK_FEEDBACK",
    "LoopPause",
    "LoopState",
    "LoopStatus",
    "LoopTerminal",
    "LoopTransition",
    "LoopTransitionReason",
    "ModelTurn",
    "ModelTurnDraft",
    "ModelTurnEnvelope",
    "ModelTurnProvider",
    "PendingToolCall",
    "StopHookFeedback",
    "ToolCallLedger",
    "ToolCallLedgerEntry",
    "append_loop_diagnostic",
    "append_memory_warning",
    "append_stop_hook_feedback",
    "append_stop_hook_warning",
    "create_loop_state",
    "materialize_model_turn",
    "replace_latest_transition",
]

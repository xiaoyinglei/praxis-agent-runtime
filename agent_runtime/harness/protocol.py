"""Stable provider-neutral contracts shared by Harness components."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from agent_runtime.budget import ResourceUsage
from agent_runtime.core.messages import (
    ModelMessage as ModelMessage,
)
from agent_runtime.core.messages import (
    ModelToolCall as ModelToolCall,
)
from agent_runtime.core.messages import (
    outputs_from_parts,
    outputs_tool_calls,
    restore_model_message,
)
from agent_runtime.harness.records import ItemSnapshot
from agent_runtime.modeling.events import (
    ModelEventSink,
    ModelOutput,
    output_from_payload,
    output_to_payload,
    validate_model_outputs,
)
from agent_runtime.tools.tool import Tool, _thaw_json, json_schema_output

DEFAULT_MAX_STEPS = 128


@dataclass(frozen=True, slots=True)
class HarnessModelRequest:
    thread_id: str
    turn_id: str
    messages: tuple[ModelMessage, ...]
    binding_manifest: Mapping[str, Any]
    tools: tuple[Tool, ...] = ()
    step: int = 1
    model_token_budget_remaining: int | None = None
    budget_pressure: bool = False
    purpose: str = "agent_step"
    request_id: str | None = None
    input_token_limit: int | None = None
    output_token_limit: int | None = None
    continuation_summary_role: str | None = None


class ModelDispatchPreflightError(RuntimeError):
    """The model call was rejected before any provider I/O began."""


class ModelContextOverflowError(ModelDispatchPreflightError):
    """A definite context-limit rejection; eligible for one durable compaction retry."""


class ModelDispatchOutcomeUnknownError(RuntimeError):
    """Provider transport ended after dispatch without a terminal response."""


class ModelDispatchCancelledError(RuntimeError):
    """The provider acknowledged a definitive cancellation before completion."""


@dataclass(frozen=True, slots=True)
class PreparedModelCall:
    request_hash: str
    context_hash: str
    tool_hash: str
    wire_hash: str
    request_ref: Mapping[str, Any]
    resource_request: ResourceUsage = ResourceUsage()
    dispatch_payload: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class HarnessModelResponse:
    """One authoritative ordered output snapshot and response metadata."""

    outputs: tuple[ModelOutput, ...]
    provider_response_id: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    status: Literal["completed", "incomplete"] = "completed"
    incomplete_reason: str | None = None
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        validate_model_outputs(self.outputs)
        object.__setattr__(self, "usage", cast(Mapping[str, Any], json_schema_output(None, dict(self.usage))))
        if self.status not in {"completed", "incomplete"}:
            raise ValueError("unsupported model response status")
        if self.status == "completed" and self.incomplete_reason is not None:
            raise ValueError("completed model response cannot have an incomplete reason")
        if self.status == "incomplete" and not self.incomplete_reason:
            raise ValueError("incomplete model response requires a reason")

    @classmethod
    def from_parts(
        cls,
        *,
        text: str = "",
        tool_calls: tuple[ModelToolCall, ...] = (),
        reasoning_content: str | None = None,
        plan_content: str | None = None,
        provider_response_id: str | None = None,
        usage: Mapping[str, Any] | None = None,
        status: Literal["completed", "incomplete"] = "completed",
        incomplete_reason: str | None = None,
        stop_reason: str | None = None,
    ) -> HarnessModelResponse:
        """Construct blocks for a simple model, without a second stored representation."""
        return cls(
            outputs=outputs_from_parts(
                text=text, tool_calls=tool_calls, reasoning_content=reasoning_content, plan_content=plan_content
            ),
            provider_response_id=provider_response_id,
            usage=usage or {},
            status=status,
            incomplete_reason=incomplete_reason,
            stop_reason=stop_reason,
        )

    @property
    def text(self) -> str:
        return "".join(output.content for output in self.outputs if output.kind in {"text", "refusal"})

    @property
    def reasoning_content(self) -> str | None:
        return "".join(output.content for output in self.outputs if output.kind == "reasoning") or None

    @property
    def plan_content(self) -> str | None:
        return "".join(output.content for output in self.outputs if output.kind == "plan") or None

    @property
    def tool_calls(self) -> tuple[ModelToolCall, ...]:
        return outputs_tool_calls(self.outputs) if self.status == "completed" else ()


def model_response_payload(response: HarnessModelResponse) -> dict[str, Any]:
    return {
        "outputs": [output_to_payload(output) for output in response.outputs],
        "provider_response_id": response.provider_response_id,
        "usage": _thaw_json(response.usage),
        "response_status": response.status,
        "incomplete_reason": response.incomplete_reason,
        "stop_reason": response.stop_reason,
    }


def restore_model_response(payload: Mapping[str, Any]) -> HarnessModelResponse:
    return HarnessModelResponse(
        outputs=tuple(output_from_payload(output) for output in payload["outputs"]),
        provider_response_id=payload["provider_response_id"],
        usage=payload["usage"],
        status=payload["response_status"],
        incomplete_reason=payload["incomplete_reason"],
        stop_reason=payload["stop_reason"],
    )


def item_text(item: ItemSnapshot) -> str:
    if item.kind in {"model_response", "context_summary_response"}:
        return restore_model_response(item.payload).text
    text = item.payload.get("text", "")
    return text if isinstance(text, str) else ""


class HarnessModel(Protocol):
    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall: ...

    async def dispatch(
        self,
        prepared: PreparedModelCall,
        *,
        event_sink: ModelEventSink | None = None,
    ) -> HarnessModelResponse: ...


@dataclass(frozen=True, slots=True)
class CompletionProposal:
    thread_id: str
    turn_id: str
    item_id: str
    answer: str


@dataclass(frozen=True, slots=True)
class CompletionDecision:
    """Runtime action and independent validation of the stated scope.

    Accepting an answer ends the Turn; it does not imply validation passed.
    Legacy custom gates that only return action/reason remain not_evaluated.
    """

    action: Literal["accept", "continue", "pause", "fail"]
    reason: str
    validation_status: Literal["not_evaluated", "passed", "failed"] = "not_evaluated"
    validation_scope: str | None = None
    reason_code: str | None = None


class CompletionGate(Protocol):
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision: ...


class ContextManager(Protocol):
    def build(self, turn_id: str) -> tuple[ModelMessage, ...]: ...

    def compaction_candidates(self, turn_id: str) -> Iterator[ContextCompactionCandidate]: ...

    def commit_compaction(self, candidate: ContextCompactionCandidate) -> object: ...

    def compact_for_budget(
        self,
        *,
        turn_id: str,
        retained_tail_messages: int,
    ) -> object: ...


class ContextBudgetExceededError(RuntimeError):
    """Context preparation failed; reason_code identifies the limiting resource or invalid data."""

    def __init__(self, message: str, *, reason_code: str = "context_compaction_failed") -> None:
        self.reason_code = reason_code
        super().__init__(message)


class ContextSourceChangedError(RuntimeError):
    """A candidate was measured against an obsolete rollout snapshot."""


@dataclass(frozen=True, slots=True)
class ContextCompactionCandidate:
    """Canonical JSON owns nested data; callers receive fresh message snapshots."""

    turn_id: str
    source_revision: str
    payload_json: str
    messages_json: str

    @property
    def messages(self) -> tuple[ModelMessage, ...]:
        return tuple(restore_model_message(m) for m in json.loads(self.messages_json))


class ContextCompactionRequiredError(ContextBudgetExceededError):
    """Provider preparation requires a durable Runtime-owned compaction."""

    def __init__(
        self,
        *,
        input_tokens: int,
        max_input_tokens: int,
        retained_tail_messages: int = 0,
        reason_code: str = "model_context_window_exceeded",
    ) -> None:
        self.input_tokens = input_tokens
        self.max_input_tokens = max_input_tokens
        self.retained_tail_messages = retained_tail_messages
        super().__init__(
            "Committed context requires durable compaction before provider dispatch: "
            f"{input_tokens} > {max_input_tokens} input tokens.",
            reason_code=reason_code,
        )


class ToolRouter(Protocol):
    def select(
        self,
        *,
        turn_id: str,
        messages: tuple[ModelMessage, ...],
    ) -> tuple[Tool, ...]: ...


class BindingProvider(Protocol):
    """Trusted owner of settings captured for a new model request."""

    def snapshot(self, *, thread_id: str, turn_id: str) -> Mapping[str, Any]: ...


class BoundHarnessModel(HarnessModel, BindingProvider, Protocol):
    """Model endpoint whose binding snapshot and dispatch share one owner."""

    def ensure_available(
        self,
        binding: Mapping[str, Any],
        *,
        thread_id: str,
        turn_id: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class TurnResult:
    thread_id: str
    turn_id: str
    answer: str | None
    status: str = "completed"
    interaction_id: str | None = None

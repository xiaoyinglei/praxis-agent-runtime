"""Stable provider-neutral contracts shared by Harness components."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent_runtime.budget import ResourceUsage
from agent_runtime.tools.tool import JsonValue, Tool


@dataclass(frozen=True, slots=True)
class HarnessToolCall:
    id: str
    name: str
    arguments: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class HarnessMessage:
    role: str
    content: str
    tool_calls: tuple[HarnessToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning_content: str | None = None


@dataclass(frozen=True, slots=True)
class HarnessModelRequest:
    thread_id: str
    turn_id: str
    messages: tuple[HarnessMessage, ...]
    binding_manifest: Mapping[str, Any]
    tools: tuple[Tool, ...] = ()
    step: int = 1
    model_token_budget_remaining: int | None = None
    budget_pressure: bool = False


class ModelDispatchPreflightError(RuntimeError):
    """The model call was rejected before any provider I/O began."""


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
    text: str
    provider_response_id: str | None
    usage: Mapping[str, Any]
    tool_calls: tuple[HarnessToolCall, ...] = ()
    status: Literal["completed", "incomplete"] = "completed"
    incomplete_reason: str | None = None
    reasoning_content: str | None = None
    plan_content: str | None = None

    def __post_init__(self) -> None:
        if self.status == "completed" and self.incomplete_reason is not None:
            raise ValueError("completed model response cannot have an incomplete reason")
        if self.status == "incomplete" and not self.incomplete_reason:
            raise ValueError("incomplete model response requires a reason")
        if self.reasoning_content is not None and not isinstance(
            self.reasoning_content, str
        ):
            raise TypeError("model reasoning content must be a string or None")
        if self.plan_content is not None and not isinstance(self.plan_content, str):
            raise TypeError("model plan content must be a string or None")


@dataclass(frozen=True, slots=True)
class HarnessModelDelta:
    channel: Literal["text", "reasoning", "plan"]
    content: str


HarnessModelDeltaSink = Callable[
    [HarnessModelDelta],
    None | Awaitable[None],
]


class HarnessModel(Protocol):
    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall: ...

    async def dispatch(
        self,
        prepared: PreparedModelCall,
        *,
        delta_sink: HarnessModelDeltaSink | None = None,
    ) -> HarnessModelResponse: ...


@dataclass(frozen=True, slots=True)
class CompletionProposal:
    thread_id: str
    turn_id: str
    item_id: str
    answer: str


@dataclass(frozen=True, slots=True)
class CompletionDecision:
    action: Literal["accept", "continue", "pause", "fail"]
    reason: str


class CompletionGate(Protocol):
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision: ...


class ContextManager(Protocol):
    def build(self, turn_id: str) -> tuple[HarnessMessage, ...]: ...

    def compact_for_budget(
        self,
        *,
        turn_id: str,
        retained_tail_messages: int,
    ) -> object: ...


class ContextBudgetExceededError(RuntimeError):
    """Committed context cannot fit inside the configured provider boundary."""


class ContextCompactionRequiredError(ContextBudgetExceededError):
    """Provider preparation requires a durable Runtime-owned compaction."""

    def __init__(
        self,
        *,
        input_tokens: int,
        max_input_tokens: int,
        retained_tail_messages: int,
    ) -> None:
        self.input_tokens = input_tokens
        self.max_input_tokens = max_input_tokens
        self.retained_tail_messages = retained_tail_messages
        super().__init__(
            "Committed context requires durable compaction before provider dispatch: "
            f"{input_tokens} > {max_input_tokens} input tokens."
        )


class ToolRouter(Protocol):
    def select(
        self,
        *,
        turn_id: str,
        messages: tuple[HarnessMessage, ...],
    ) -> tuple[Tool, ...]: ...


class BindingProvider(Protocol):
    """Trusted owner of settings captured for a new model request."""

    def snapshot(self, *, thread_id: str, turn_id: str) -> Mapping[str, Any]: ...


class BindingValidator(Protocol):
    """Validate one durable binding against its owning Thread and Turn."""

    def __call__(
        self,
        binding: Mapping[str, Any],
        *,
        thread_id: str,
        turn_id: str,
    ) -> None: ...


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

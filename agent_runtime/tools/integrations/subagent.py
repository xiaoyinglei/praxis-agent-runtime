from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    json_schema_output,
    pydantic_input,
)

type SubagentRunner = Callable[
    [Mapping[str, JsonValue]],
    object | Awaitable[object],
]


class SubagentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        min_length=1,
        max_length=20_000,
        description="Bounded task delegated to an isolated child agent.",
    )
    context_summary: str | None = Field(
        default=None,
        max_length=20_000,
        description="Only the parent context needed to complete this task.",
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Optional child step ceiling; inherits the parent Turn limit when omitted and may only narrow it.",
    )
    llm_budget_total: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Requested child model-token ceiling. The runtime grants it only "
            "from the parent Turn's currently available budget; it cannot mint new budget."
        ),
    )
    llm_cost_budget_micros: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description=(
            "Requested child monetary ceiling in integer micro-dollars. "
            "Required when the parent Turn has a monetary budget."
        ),
    )


class SubagentEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(default="", max_length=500)
    doc_id: str | int | None = None
    citation_anchor: str = Field(default="", max_length=1000)


class SubagentCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(default="", max_length=500)
    file_name: str | None = Field(default=None, max_length=1000)
    section_path: list[str] = Field(default_factory=list, max_length=20)
    page_start: int | None = None
    page_end: int | None = None
    evidence_id: str = Field(default="", max_length=500)
    record_type: str = Field(default="", max_length=200)
    citation_anchor: str = Field(default="", max_length=1000)
    doc_id: int | None = None
    benchmark_doc_id: str | None = Field(default=None, max_length=500)
    source_id: int | None = None
    source_type: str | None = Field(default=None, max_length=200)


class SubagentPause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=500)
    kind: Literal[
        "tool_approval",
        "tool_reconciliation",
        "choice",
        "clarification",
    ]
    question: str = Field(min_length=1, max_length=20_000)


class SubagentExecutionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["test", "static_analysis", "assertion", "inspection"]
    verifier: str = Field(min_length=1, max_length=500)
    operation_id: str = Field(min_length=1, max_length=500)
    verified_resources: list[str] = Field(default_factory=list, max_length=50)


class _SubagentResultFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conclusion: str = Field(default="", max_length=50_000)
    child_turn_id: str = Field(default="", max_length=500)
    stop_reason: str | None = Field(default=None, max_length=1000)
    pause: SubagentPause | None = None
    execution_evidence: list[SubagentExecutionEvidence] = Field(
        default_factory=list,
        max_length=50,
    )


class StandardSubagentOutput(_SubagentResultFields):
    status: Literal["done", "failed", "paused"] = "failed"


class SubagentOutput(StandardSubagentOutput):
    """Optional evidence fields provided by an externally assembled runner."""

    key_facts: list[str] = Field(default_factory=list, max_length=10)
    evidence_refs: list[SubagentEvidenceRef] = Field(
        default_factory=list,
        max_length=20,
    )
    citations: list[SubagentCitation] = Field(default_factory=list, max_length=20)


_SUBAGENT_INPUT_SCHEMA, _validate_subagent_input = pydantic_input(SubagentInput)


def create_subagent_tool(
    run_subagent: SubagentRunner,
    *,
    name: str = "task",
    execution_revision: str = "child-runtime-v1",
    output_model: type[StandardSubagentOutput] = SubagentOutput,
) -> Tool:
    """Project an externally assembled child runner into an ordinary Tool."""

    if not callable(run_subagent):
        raise TypeError("run_subagent must be callable")
    if not isinstance(name, str) or not name:
        raise ValueError("subagent tool name must be non-empty")
    if not isinstance(execution_revision, str) or not execution_revision:
        raise ValueError("execution_revision must be non-empty")
    output_schema, _validate_output = pydantic_input(output_model)

    def normalize_output(raw: object) -> NormalizedToolOutput:
        return _normalize_subagent_output(raw, output_model=output_model, output_schema=output_schema)

    return Tool(
        definition=ToolDefinition(
            name=name,
            description=(
                "Delegate one bounded task to an isolated child agent and return its "
                "conclusion, status, any pause, and actual execution evidence. Pass only the context "
                "needed for the subtask; this tool does not grant additional permissions."
            ),
            input_schema=_SUBAGENT_INPUT_SCHEMA,
        ),
        validate_input=_validate_subagent_input,
        run=run_subagent,
        normalize_output=normalize_output,
        output_schema=output_schema,
        static_effects=frozenset({ToolEffect.NETWORK}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.NETWORK}),
            targets=(),
        ),
        execution_revision=f"integration-subagent-v2:{execution_revision}",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=180.0,
        max_model_output_bytes=500_000,
    )


def _normalize_subagent_output(
    raw: object,
    *,
    output_model: type[StandardSubagentOutput],
    output_schema: Mapping[str, JsonValue],
) -> NormalizedToolOutput:
    validated = output_model.model_validate(raw)
    structured = json_schema_output(
        output_schema,
        validated.model_dump(mode="json"),
    )
    is_error = validated.status == "failed"
    error_code = "subagent_failed" if is_error else None
    return NormalizedToolOutput(
        structured_content=structured,
        is_error=is_error,
        error_code=error_code,
        error_message=validated.conclusion[:2000] if is_error else None,
        retryable=False,
        metadata={"child_turn_id": validated.child_turn_id},
    )


__all__ = [
    "SubagentCitation",
    "SubagentEvidenceRef",
    "SubagentInput",
    "SubagentOutput",
    "StandardSubagentOutput",
    "SubagentPause",
    "SubagentExecutionEvidence",
    "SubagentRunner",
    "create_subagent_tool",
]


class ChildHandleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    child_turn_id: str = Field(
        min_length=1, max_length=500, description="Durable child Turn returned by spawn_agent, owned by this parent."
    )


class SendAgentMessageInput(ChildHandleInput):
    message: str = Field(min_length=1, max_length=20_000)
    message_id: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Stable caller-selected message id. Reuse only for an identical retry; the kernel delivers it once."
        ),
    )


class WaitAgentInput(ChildHandleInput):
    timeout_seconds: float = Field(
        default=10,
        ge=0,
        le=30,
        description="Bounded status/event wait; timeout returns running without cancelling the child.",
    )


class SpawnAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    child_turn_id: str
    thread_id: str
    status: Literal["running"]


class AgentStatusOutput(_SubagentResultFields):
    status: Literal["running", "done", "failed", "paused"] = "running"


class AgentMessageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    child_turn_id: str
    message_id: str
    input_id: str
    status: Literal["accepted"]


def create_async_subagent_tools(
    *,
    spawn: SubagentRunner,
    send: SubagentRunner,
    wait: SubagentRunner,
    cancel: SubagentRunner,
) -> tuple[Tool, ...]:
    """Project lifecycle operations without introducing a second execution loop."""
    specs: tuple[
        tuple[str, type[BaseModel], type[BaseModel], SubagentRunner, str, frozenset[ToolEffect], bool], ...
    ] = (
        (
            "spawn_agent",
            SubagentInput,
            SpawnAgentOutput,
            spawn,
            (
                "Start an attached child agent in a fresh context and return its durable child_turn_id "
                "immediately after budget reservation. Children share the workspace and inherit the parent's "
                "frozen permissions/model and finite depth/count/budget ceilings. Supply only relevant "
                "context_summary; arbitrary conversation forks are not supported. Use wait_agent for "
                "completion/evidence. Parent completion drains attached children."
            ),
            frozenset({ToolEffect.NETWORK}),
            False,
        ),
        (
            "send_agent_message",
            SendAgentMessageInput,
            AgentMessageOutput,
            send,
            (
                "Deliver guidance only to an owned child's active Turn through the canonical input boundary. "
                "Provide a stable message_id for retry-safe, once-only delivery. This does not resume a paused "
                "approval or grant permissions."
            ),
            frozenset(),
            True,
        ),
        (
            "wait_agent",
            WaitAgentInput,
            AgentStatusOutput,
            wait,
            (
                "Wait up to timeout_seconds for an owned attached child's live driver, then return running, "
                "paused, done or failed with its canonical conclusion and actual execution evidence. Timeout "
                "leaves the child running. Lost drivers require explicit recovery; wait does not start another "
                "model loop."
            ),
            frozenset(),
            True,
        ),
        (
            "cancel_agent",
            ChildHandleInput,
            AgentStatusOutput,
            cancel,
            (
                "Record cancellation of an owned child before cancelling live execution and managed command "
                "processes. Unknown outcomes keep their reserved budget and remain visible for reconciliation. "
                "Other children are unaffected."
            ),
            frozenset(),
            True,
        ),
    )
    tools = []
    for name, input_model, output_model, runner, description, effects, idempotent in specs:
        input_schema, validate = pydantic_input(input_model)
        output_schema, _ = pydantic_input(output_model)

        def normalize(
            raw: object,
            output_model: type[BaseModel] = output_model,
            output_schema: Mapping[str, JsonValue] = output_schema,
        ) -> NormalizedToolOutput:
            value = output_model.model_validate(raw)
            data = value.model_dump(mode="json")
            return NormalizedToolOutput(
                structured_content=json_schema_output(output_schema, data),
                is_error=getattr(value, "status", None) == "failed",
                metadata={"child_turn_id": data["child_turn_id"]},
            )

        def resolve(
            _arguments: Mapping[str, JsonValue],
            effects: frozenset[ToolEffect] = effects,
        ) -> ResolvedToolUse:
            return ResolvedToolUse(effects=effects, targets=())

        tools.append(
            Tool(
                definition=ToolDefinition(name=name, description=description, input_schema=input_schema),
                validate_input=validate,
                run=runner,
                normalize_output=normalize,
                output_schema=output_schema,
                static_effects=effects,
                resolve_use=resolve,
                execution_revision="integration-subagent-async-v1",
                idempotent=idempotent,
                concurrency_safe=True,
                cancellation_mode=CancellationMode.COOPERATIVE,
                interrupt_behavior=InterruptBehavior.CANCEL,
                timeout_seconds=40,
                max_model_output_bytes=500_000,
            )
        )
    return tuple(tools)

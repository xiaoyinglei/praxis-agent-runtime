from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
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
    ToolTarget,
    json_schema_output,
    pydantic_input,
)

type SubagentRunner = Callable[
    [Mapping[str, JsonValue]],
    object | Awaitable[object],
]


class SubagentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_mode: Literal["auto", "write", "read-only"] = Field(
        default="auto",
        description="auto isolates mutable toolsets; write freezes dirty source; read-only narrows authority.",
    )

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
    environment: dict[str, JsonValue] | None = None


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


class IntegrateAgentInput(ChildHandleInput):
    action: Literal["inspect", "apply", "reconcile", "resolve", "verify", "discard"] = "inspect"
    artifact_id: str | None = Field(
        default=None, description="Exact patch artifact returned by inspect; required for apply."
    )
    verification_tool_call_id: str | None = Field(
        default=None, description="Parent run_command/manage_command call after integration, with workspace_write=true."
    )
    resolution_note: str | None = Field(
        default=None, max_length=4000,
        description="Required for resolve: explain manual parent conflict resolution; retained as a diff artifact.",
    )
    diff_offset: int = Field(default=0, ge=0, strict=True)
    diff_limit: int = Field(default=100_000, ge=1, le=200_000, strict=True)


class SpawnAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    child_turn_id: str
    thread_id: str
    status: Literal["running"]
    workspace: str | None = None


class AgentStatusOutput(_SubagentResultFields):
    status: Literal["running", "done", "failed", "paused"] = "running"


class AgentMessageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    child_turn_id: str
    message_id: str
    input_id: str
    status: Literal["accepted"]


class AgentIntegrationOutput(BaseModel):
    model_config = ConfigDict(extra="allow")
    child_turn_id: str
    status: str


def create_async_subagent_tools(
    *,
    spawn: SubagentRunner,
    send: SubagentRunner,
    wait: SubagentRunner,
    cancel: SubagentRunner,
    integrate: SubagentRunner | None = None,
    workspace: Path | None = None,
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
                "after durable budget/environment reservation. Writing children execute in a dirty-source Git snapshot "
                "with their own worktree, cwd and permissions. They inherit the parent's "
                "frozen permissions/model and finite depth/count/budget ceilings. Supply only relevant "
                "context_summary; arbitrary conversation forks are not supported. Use wait_agent for "
                "completion/evidence, then integrate_agent inspect/apply or explicitly discard. Parent completion "
                "requires terminal children and resolved artifacts; run verification in the parent after integration."
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
    if integrate is not None:
        specs += ((
            "integrate_agent", IntegrateAgentInput, AgentIntegrationOutput, integrate,
            "Inspect an owned terminal child's actual patch and immutable artifact_id. Review that diff, then apply "
            "the exact artifact to the parent workspace. Preimage conflicts leave both workspaces intact. Integration "
            "does not stage or commit user files. Run parent verification with workspace_write=true after apply, then "
            "verify using that command's tool_call_id. This records execution, not task correctness. "
            "discard explicitly abandons "
            "integration while retaining the child workspace. Unknown partial writes require explicit reconciliation.",
            frozenset({ToolEffect.WRITE_WORKSPACE}), False,
        ),)
    tools = []
    for name, input_model, output_model, runner, description, effects, idempotent in specs:
        input_schema, validate = pydantic_input(input_model)
        output_schema, _ = pydantic_input(output_model)

        def normalize(
            raw: object,
            output_model: type[BaseModel] = output_model,
            output_schema: Mapping[str, JsonValue] = output_schema,
        ) -> NormalizedToolOutput:
            data = output_model.model_validate(raw).model_dump(mode="json")
            return NormalizedToolOutput(
                structured_content=json_schema_output(output_schema, data),
                is_error=data.get("status") in {"failed", "conflict"},
                metadata={"child_turn_id": data["child_turn_id"]},
            )

        def resolve(
            _arguments: Mapping[str, JsonValue],
            effects: frozenset[ToolEffect] = effects,
            integration: bool = name == "integrate_agent",
        ) -> ResolvedToolUse:
            if integration:
                write = _arguments.get("action", "inspect") == "apply"
                return ResolvedToolUse(
                    effects=effects if write else frozenset(),
                    targets=(ToolTarget(kind="workspace_path", value=str(workspace)),) if write else (),
                )
            return ResolvedToolUse(effects=effects, targets=())

        tools.append(
            Tool(
                definition=ToolDefinition(name=name, description=description, input_schema=input_schema),
                validate_input=validate,
                run=runner,
                normalize_output=normalize,
                output_schema=output_schema,
                static_effects=frozenset() if name == "integrate_agent" else effects,
                resolve_use=resolve,
                execution_revision="integration-subagent-async-v2",
                idempotent=idempotent,
                concurrency_safe=True,
                cancellation_mode=CancellationMode.COOPERATIVE,
                interrupt_behavior=InterruptBehavior.CANCEL,
                timeout_seconds=40,
                max_model_output_bytes=500_000,
            )
        )
    return tuple(tools)

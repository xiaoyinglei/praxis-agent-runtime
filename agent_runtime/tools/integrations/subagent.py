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
    tool_query: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional capability hint; it is not a permission grant.",
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        le=50,
        strict=True,
        description="Optional child-turn ceiling.",
    )
    llm_budget_total: int | None = Field(
        default=None,
        ge=1,
        description="Optional child model-token budget.",
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


class SubagentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conclusion: str = Field(default="", max_length=50_000)
    key_facts: list[str] = Field(default_factory=list, max_length=10)
    evidence_refs: list[SubagentEvidenceRef] = Field(
        default_factory=list,
        max_length=20,
    )
    citations: list[SubagentCitation] = Field(default_factory=list, max_length=20)
    status: Literal["done", "failed", "paused"] = "failed"
    child_turn_id: str = Field(default="", max_length=500)
    stop_reason: str | None = Field(default=None, max_length=1000)


_SUBAGENT_INPUT_SCHEMA, _validate_subagent_input = pydantic_input(SubagentInput)
_SUBAGENT_OUTPUT_SCHEMA, _unused_subagent_output_validator = pydantic_input(
    SubagentOutput
)


def create_subagent_tool(
    run_subagent: SubagentRunner,
    *,
    name: str = "task",
    execution_revision: str = "child-runtime-v1",
) -> Tool:
    """Project an externally assembled child runner into an ordinary Tool."""

    if not callable(run_subagent):
        raise TypeError("run_subagent must be callable")
    if not isinstance(name, str) or not name:
        raise ValueError("subagent tool name must be non-empty")
    if not isinstance(execution_revision, str) or not execution_revision:
        raise ValueError("execution_revision must be non-empty")
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=(
                "Delegate one bounded task to an isolated child agent and return its "
                "conclusion, key facts, and evidence references. Pass only the context "
                "needed for the subtask; this tool does not grant additional permissions."
            ),
            input_schema=_SUBAGENT_INPUT_SCHEMA,
        ),
        validate_input=_validate_subagent_input,
        run=run_subagent,
        normalize_output=_normalize_subagent_output,
        output_schema=_SUBAGENT_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.NETWORK}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.NETWORK}),
            targets=(),
        ),
        execution_revision=f"integration-subagent-v1:{execution_revision}",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=180.0,
        max_model_output_bytes=500_000,
    )


def _normalize_subagent_output(raw: object) -> NormalizedToolOutput:
    validated = SubagentOutput.model_validate(raw)
    structured = json_schema_output(
        _SUBAGENT_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    is_error = validated.status != "done"
    error_code = (
        None
        if not is_error
        else (
            "subagent_paused"
            if validated.status == "paused"
            else "subagent_failed"
        )
    )
    return NormalizedToolOutput(
        structured_content=structured,
        is_error=is_error,
        error_code=error_code,
        error_message=validated.conclusion[:2000] if is_error else None,
        retryable=validated.status == "paused",
        metadata={"child_turn_id": validated.child_turn_id},
    )


__all__ = [
    "SubagentCitation",
    "SubagentEvidenceRef",
    "SubagentInput",
    "SubagentOutput",
    "SubagentRunner",
    "create_subagent_tool",
]

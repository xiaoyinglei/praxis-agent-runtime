from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolTarget,
    json_schema_output,
    pydantic_input,
)

type KnowledgeSearcher = Callable[
    [Mapping[str, JsonValue]],
    object | Awaitable[object],
]


class KnowledgeSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language question or evidence query.",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum ranked evidence items to return.",
    )


class KnowledgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(default="", max_length=500)
    doc_id: str | int | None = None
    citation_anchor: str = Field(default="", max_length=1000)
    text: str = Field(default="", max_length=20_000)
    score: float = 0.0
    source_type: str = Field(default="", max_length=200)
    file_name: str = Field(default="", max_length=1000)


class KnowledgeSearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[KnowledgeResult] = Field(default_factory=list, max_length=50)
    answer_text: str = Field(default="", max_length=50_000)
    citations: list[str] = Field(default_factory=list, max_length=50)
    groundedness_flag: bool = False
    insufficient_evidence: bool = False
    total_found: int = Field(default=0, ge=0)


_KNOWLEDGE_INPUT_SCHEMA, _validate_knowledge_input = pydantic_input(KnowledgeSearchInput)
_KNOWLEDGE_OUTPUT_SCHEMA, _unused_knowledge_output_validator = pydantic_input(KnowledgeSearchOutput)


def create_knowledge_tools(
    searcher: KnowledgeSearcher | None,
    *,
    execution_revision: str = "configured-v1",
) -> tuple[Tool, ...]:
    """Project an explicitly configured knowledge search closure into one Tool."""

    if searcher is None:
        return ()
    if not callable(searcher):
        raise TypeError("searcher must be callable when configured")
    return (
        create_search_knowledge_tool(
            searcher,
            execution_revision=execution_revision,
        ),
    )


def create_search_knowledge_tool(
    searcher: KnowledgeSearcher,
    *,
    execution_revision: str = "configured-v1",
) -> Tool:
    if not callable(searcher):
        raise TypeError("searcher must be callable")
    if not isinstance(execution_revision, str) or not execution_revision:
        raise ValueError("execution_revision must be non-empty")
    return Tool(
        definition=ToolDefinition(
            name="search_knowledge",
            description=(
                "Search an explicitly configured knowledge source for ranked evidence. "
                "Use this for factual document evidence and citations, not for locating "
                "workspace source files; use search_text for source navigation."
            ),
            input_schema=_KNOWLEDGE_INPUT_SCHEMA,
        ),
        validate_input=_validate_knowledge_input,
        run=searcher,
        normalize_output=_normalize_knowledge_output,
        output_schema=_KNOWLEDGE_OUTPUT_SCHEMA,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(ToolTarget(kind="knowledge_source", value="configured"),),
        ),
        execution_revision=(f"integration-search-knowledge-v1:{execution_revision}"),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=45.0,
        max_model_output_bytes=1_500_000,
    )


def _normalize_knowledge_output(raw: object) -> NormalizedToolOutput:
    validated = KnowledgeSearchOutput.model_validate(raw)
    structured = json_schema_output(
        _KNOWLEDGE_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    return NormalizedToolOutput(structured_content=structured)


__all__ = [
    "KnowledgeResult",
    "KnowledgeSearchInput",
    "KnowledgeSearchOutput",
    "KnowledgeSearcher",
    "create_knowledge_tools",
    "create_search_knowledge_tool",
]

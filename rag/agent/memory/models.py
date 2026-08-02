from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_serializers import field_serializer
from pydantic.functional_validators import field_validator

ContextSectionName = Literal[
    "instructions",
    "system",
    "policy_hints",
    "task",
    "call_context",
    "output_schema",
    "plan",
    "evidence",
    "memory",
    "working_memory",
    "message_tail",
    "tool_results",
    "open_decisions",
]

MemoryRefStatus = Literal["available", "deleted", "unavailable", "compacted"]
StateRetentionChannel = Literal[
    "tool_results",
    "memory_refs",
    "plan_events",
]


class MemoryPolicy(BaseModel):
    """Run-local memory retention and compaction limits."""

    schema_version: int = 1
    max_tool_output_chars: int = Field(default=64_000, ge=1)
    max_tool_results: int = Field(default=80, ge=1)
    max_memory_refs: int = Field(default=300, ge=1)
    max_plan_events: int = Field(default=30, ge=1)
    max_memory_summary_chars: int = Field(default=1200, ge=80)
    max_memory_records: int = Field(default=500, ge=1)
    message_compaction_min_count: int = Field(default=16, ge=1)
    max_message_tail_count: int = Field(default=12, ge=0)
    max_working_summary_chars: int = Field(default=8000, ge=80)
    max_extracted_facts: int = Field(default=200, ge=1)
    max_message_batch_chars: int = Field(default=256_000, ge=1)
    snip_compact_threshold: int = Field(default=50, ge=1)
    snip_keep_head: int = Field(default=5, ge=0)
    snip_keep_tail: int = Field(default=30, ge=0)
    micro_compact_keep_recent: int = Field(default=3, ge=0)
    micro_compact_max_chars: int = Field(default=500, ge=50)
    reactive_compact_tail_count: int = Field(default=6, ge=1)
    reactive_compact_max_observations: int = Field(default=10, ge=1)
    reactive_compact_max_evidence: int = Field(default=20, ge=1)


class MemoryRef(BaseModel):
    """Opaque run-local pointer to raw memory stored under workspace .agent_memory/."""

    schema_version: int = 1
    ref_id: str
    path: str
    summary: str
    source_tool_call_id: str | None = None
    source_tool_name: str | None = None
    size_bytes: int = Field(default=0, ge=0)
    status: MemoryRefStatus = "available"
    warnings: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return self.ref_id


class MemoryRecord(BaseModel):
    """Persisted run-local record or tombstone resolved through MemoryStore."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: int = 1
    original_output_model: str
    summary: str
    ref: MemoryRef
    status: MemoryRefStatus = "available"
    warnings: list[str] = Field(default_factory=list)
    payload: BaseModel | None = None
    ref_deleted: bool = False
    reason: str | None = None

    @property
    def key(self) -> str:
        return self.ref.ref_id


class ExternalizedToolOutput(BaseModel):
    """Typed replacement for a large ToolResult.output stored in run-local memory."""

    schema_version: int = 1
    original_output_model: str
    summary: str
    ref: MemoryRef
    status: MemoryRefStatus = "available"
    warnings: list[str] = Field(default_factory=list)


class MessageBatchPayload(BaseModel):
    """Raw messages externalized from the bounded run state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[BaseMessage] = Field(default_factory=list)

    @field_validator("messages", mode="before")
    @classmethod
    def _restore_messages(cls, value: object) -> object:
        if isinstance(value, list) and all(
            isinstance(item, dict) and "type" in item and "data" in item for item in value
        ):
            return messages_from_dict(value)
        return value

    @field_serializer("messages")
    def _serialize_messages(self, messages: list[BaseMessage]) -> object:
        return messages_to_dict(messages)


class ToolErrorDetailPayload(BaseModel):
    """Raw ToolError.detail externalized from checkpoint state."""

    tool_call_id: str
    tool_name: str
    detail: dict[str, object] = Field(default_factory=dict)


class EvictedStateItem(BaseModel):
    """Audit record for a bounded state item removed from checkpoint state."""

    channel: StateRetentionChannel
    key: str
    reason: str
    summary: str | None = None
    source_tool_call_id: str | None = None
    memory_ref_id: str | None = None


class MemoryBudgetSnapshot(BaseModel):
    schema_version: int = 1
    max_tool_output_chars: int
    externalized_record_count: int = 0
    unavailable_record_count: int = 0
    memory_ref_count: int = 0
    compacted_tool_result_count: int = 0
    dropped_state_items: dict[str, int] = Field(default_factory=dict)
    evicted_items: list[EvictedStateItem] = Field(default_factory=list)
    used_channel_counts: dict[str, int] = Field(default_factory=dict)
    pinned_item_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class StateChannelReplacement(BaseModel):
    """Internal reducer signal that replaces a bounded list channel atomically."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    items: list[Any] = Field(default_factory=list)


class WorkingSummary(BaseModel):
    summary: str
    covered_message_ids: list[str]
    updated_at: str
    token_count: int


class ExtractedFact(BaseModel):
    fact_id: str
    text: str
    source_message_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    stale: bool = False


class ContextBudgetSnapshot(BaseModel):
    max_context_tokens: int
    used_context_tokens: int = 0
    system_tokens: int = 0
    planning_tokens: int = 0
    evidence_tokens: int = 0
    memory_tokens: int = 0
    working_memory_tokens: int = 0
    message_tail_tokens: int = 0
    tool_result_tokens: int = 0
    dropped_sections: list[ContextSectionName] = Field(default_factory=list)
    summarized_sections: list[ContextSectionName] = Field(default_factory=list)
    overflow: bool = False
    degraded: bool = False
    required_truncated: list[ContextSectionName] = Field(default_factory=list)
    section_token_counts: dict[str, int] = Field(default_factory=dict)
    dropped_section_reasons: dict[str, str] = Field(default_factory=dict)
    memory_ref_count: int = 0
    externalized_record_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class ContextSection(BaseModel):
    name: ContextSectionName
    content: str
    token_count: int = Field(ge=0)
    required: bool = False


class InjectedContext(BaseModel):
    sections: list[ContextSection]
    context_budget: ContextBudgetSnapshot

    def section(self, name: ContextSectionName) -> ContextSection:
        for section in self.sections:
            if section.name == name:
                return section
        raise KeyError(f"context section not found: {name}")

    def as_text(self) -> str:
        return "\n\n".join(f"[{section.name}]\n{section.content}" for section in self.sections)


class WorkingMemoryDraft(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    working_summary: WorkingSummary | None
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    tail_messages: list[BaseMessage] = Field(default_factory=list)
    context_budget: ContextBudgetSnapshot | None = None

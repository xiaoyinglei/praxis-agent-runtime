"""Typed sub-state containers for the agent loop.

Each container groups related control-flow fields that were previously
flat in LoopState.  These are Pydantic models so they can be
serialised through the existing checkpoint serde path.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.core.observations import StructuredObservation
from agent_runtime.core.output_models import ValidatedFinalOutput
from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic
from agent_runtime.memory.models import (
    ContextBudgetSnapshot,
    ExtractedFact,
    MemoryBudgetSnapshot,
    MemoryRef,
    WorkingSummary,
)
from agent_runtime.planning import AgentPlan, PlanEvent


class StopHookFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=1000)
    occurrences: int = Field(default=1, ge=1)


class PlanState(BaseModel):
    """Agent-managed planning state."""

    agent_plan: AgentPlan | None = None
    plan_events: list[PlanEvent] = Field(default_factory=list)


class MemoryState(BaseModel):
    """Working-memory context for the current run."""

    working_summary: WorkingSummary | None = None
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    recent_observations: list[StructuredObservation] = Field(default_factory=list)
    # Runtime-owned truth. Unlike the model-visible locator projection below,
    # verified paths are checkpointed without lossy eviction.
    verified_workspace_paths: list[str] = Field(default_factory=list)
    # Bounded model-context projection; never an authorization source by itself.
    known_locators: list[dict[str, object]] = Field(default_factory=list)
    context_budget: ContextBudgetSnapshot | None = None
    memory_refs: list[MemoryRef] = Field(default_factory=list)
    memory_budget: MemoryBudgetSnapshot | None = None
    memory_warnings: list[str] = Field(default_factory=list)
    reactive_compact_used: bool = False


class DiscoveryCandidate(BaseModel):
    """Tool-discovery candidate — matches DeferredStore.sync_to_state() format."""

    name: str
    description: str = ""
    reason: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)


class DiscoveryEvent(BaseModel):
    """Tool-search history event — matches DeferredStore.sync_to_state() format."""

    query: str
    candidates: list[str] = Field(default_factory=list)
    activated: list[str] = Field(default_factory=list)


class DeferredToolState(BaseModel):
    """On-demand tool-discovery state."""

    active_tools: list[str] = Field(default_factory=list)
    active_tool_iterations: dict[str, int] = Field(default_factory=dict)
    last_candidates: list[DiscoveryCandidate] = Field(default_factory=list)
    last_search_query: str = ""
    search_history: list[DiscoveryEvent] = Field(default_factory=list)
    pinned_tools: list[str] = Field(default_factory=list)
    capability_diagnostics: list[RuntimeDiagnostic] = Field(default_factory=list)


class FinishState(BaseModel):
    """Stop-hook gate + terminal output state."""

    feedback: list[StopHookFeedback] = Field(default_factory=list)
    warnings: list[StopHookFeedback] = Field(default_factory=list)
    final_answer: str | None = None
    final_output: ValidatedFinalOutput | None = None
    output_validation_errors: list[dict[str, object]] = Field(default_factory=list)


__all__ = [
    "DeferredToolState",
    "DiscoveryCandidate",
    "DiscoveryEvent",
    "FinishState",
    "MemoryState",
    "PlanState",
    "StopHookFeedback",
]

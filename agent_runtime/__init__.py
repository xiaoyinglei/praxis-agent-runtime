"""Public Agent SDK facade."""

from typing import TYPE_CHECKING

from agent_runtime.harness.events import ReplayEvent
from agent_runtime.harness.memory import MemoryConfig
from agent_runtime.harness.session import Session
from agent_runtime.hooks import HookDecision, HookRequest, LifecycleHook
from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.modeling.failures import ModelFailure
from agent_runtime.models import ModelNotAvailableError, ModelSpec
from agent_runtime.permission_policy import NetworkPolicy, NetworkRule, PathRule, PermissionPolicy, ToolRule
from agent_runtime.planning import AgentPlan, PlanEvent, PlanStep
from agent_runtime.result import AgentResult, AgentUsage
from agent_runtime.streaming.events import (
    EventType,
    ItemDeltaKind,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
)

if TYPE_CHECKING:
    from agent_runtime.agent import Agent, AgentEventSink
    from agent_runtime.background import BackgroundHandle


def __getattr__(name: str) -> object:
    if name == "BackgroundHandle":
        from agent_runtime.background import BackgroundHandle

        return BackgroundHandle
    if name in {"Agent", "AgentEventSink"}:
        from agent_runtime.agent import Agent, AgentEventSink

        return {"Agent": Agent, "AgentEventSink": AgentEventSink}[name]
    raise AttributeError(f"module 'agent_runtime' has no attribute {name!r}")


__all__ = [
    "Agent",
    "AgentEventSink",
    "AgentPlan",
    "AgentResult",
    "AgentUsage",
    "BackgroundHandle",
    "EventType",
    "HookDecision",
    "HookRequest",
    "LifecycleHook",
    "ItemDeltaKind",
    "ItemStatus",
    "ModelNotAvailableError",
    "ModelFailure",
    "ModelSpec",
    "MemoryConfig",
    "NetworkPolicy",
    "NetworkRule",
    "PathRule",
    "PermissionPolicy",
    "PlanEvent",
    "PlanStep",
    "RAGKnowledgeConfig",
    "ReplayEvent",
    "StreamEvent",
    "Session",
    "TurnItemKind",
    "ToolRule",
]

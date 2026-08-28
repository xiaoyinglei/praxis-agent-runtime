"""Public Agent SDK facade."""

from typing import TYPE_CHECKING

from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.models import ModelNotAvailableError, ModelSpec
from agent_runtime.result import AgentResult, AgentUsage
from rag.agent.streaming.events import (
    EventType,
    ItemDeltaKind,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
)

if TYPE_CHECKING:
    from agent_runtime.agent import Agent, AgentEventSink


def __getattr__(name: str) -> object:
    if name in {"Agent", "AgentEventSink"}:
        from agent_runtime.agent import Agent, AgentEventSink

        return {"Agent": Agent, "AgentEventSink": AgentEventSink}[name]
    raise AttributeError(f"module 'agent_runtime' has no attribute {name!r}")


__all__ = [
    "Agent",
    "AgentEventSink",
    "AgentResult",
    "AgentUsage",
    "EventType",
    "ItemDeltaKind",
    "ItemStatus",
    "ModelNotAvailableError",
    "ModelSpec",
    "RAGKnowledgeConfig",
    "StreamEvent",
    "TurnItemKind",
]

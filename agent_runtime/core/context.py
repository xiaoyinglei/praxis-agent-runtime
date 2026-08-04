from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_runtime.core.definition import ToolPolicy
from agent_runtime.memory.models import MemoryPolicy

if TYPE_CHECKING:
    from agent_runtime.memory.store import WorkspaceMemoryStore


class LLMBudgetLedger:
    """Async ledger for per-Turn LLM token budget accounting."""

    def __init__(self, *, total: int) -> None:
        self._total = total
        self._committed = 0
        self._reserved: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def reserve(self, lease_id: str, amount: int) -> bool:
        async with self._lock:
            if amount < 0:
                raise ValueError("amount must be non-negative")
            if self._committed + sum(self._reserved.values()) + amount > self._total:
                return False
            self._reserved[lease_id] = amount
            return True

    async def commit(self, lease_id: str, actual: int) -> int:
        async with self._lock:
            if actual < 0:
                raise ValueError("actual must be non-negative")
            reserved = self._reserved.pop(lease_id, 0)
            self._committed += actual
            return max(actual - reserved, 0)

    async def refund(self, lease_id: str) -> int:
        async with self._lock:
            return self._reserved.pop(lease_id, 0)

    async def remaining(self) -> int:
        async with self._lock:
            return max(self._total - self._committed - sum(self._reserved.values()), 0)

    async def committed(self) -> int:
        async with self._lock:
            return self._committed

    async def reserved(self) -> int:
        async with self._lock:
            return sum(self._reserved.values())


@dataclass(frozen=True)
class AgentRunConfig:
    turn_id: str
    llm_budget_total: int | None = None
    max_turns: int | None = None
    max_context_tokens: int | None = None
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)

    def __post_init__(self) -> None:
        if self.max_turns is None:
            return
        if type(self.max_turns) is not int:
            raise TypeError("max_turns must be an integer")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")


@dataclass
class AgentRuntimeHandles:
    cancellation: asyncio.Event
    llm_budget_ledger: LLMBudgetLedger | None = None
    memory_store: WorkspaceMemoryStore | None = None


class TurnRegistry:
    _handles: dict[str, AgentRuntimeHandles] = {}

    @classmethod
    def get_or_create(cls, run_config: AgentRunConfig) -> AgentRuntimeHandles:
        if run_config.turn_id not in cls._handles:
            llm_budget_ledger = (
                None if run_config.llm_budget_total is None else LLMBudgetLedger(total=run_config.llm_budget_total)
            )
            cls._handles[run_config.turn_id] = AgentRuntimeHandles(
                cancellation=asyncio.Event(),
                llm_budget_ledger=llm_budget_ledger,
            )
        return cls._handles[run_config.turn_id]

    @classmethod
    def get(cls, turn_id: str) -> AgentRuntimeHandles:
        return cls._handles[turn_id]

    @classmethod
    def remove(cls, turn_id: str) -> None:
        cls._handles.pop(turn_id, None)


__all__ = [
    "AgentRunConfig",
    "AgentRuntimeHandles",
    "LLMBudgetLedger",
    "TurnRegistry",
]

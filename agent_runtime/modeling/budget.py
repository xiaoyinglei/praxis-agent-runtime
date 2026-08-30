"""Provider-call budget accounting independent of Agent orchestration."""

from __future__ import annotations

import asyncio


class LLMBudgetLedger:
    """Async reservation ledger for callers that share an LLM token budget."""

    def __init__(self, *, total: int) -> None:
        if total < 0:
            raise ValueError("total must be non-negative")
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
            return max(
                self._total - self._committed - sum(self._reserved.values()),
                0,
            )

    async def committed(self) -> int:
        async with self._lock:
            return self._committed

    async def reserved(self) -> int:
        async with self._lock:
            return sum(self._reserved.values())


__all__ = ["LLMBudgetLedger"]

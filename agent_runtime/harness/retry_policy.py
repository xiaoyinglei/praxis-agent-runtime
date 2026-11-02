"""Frozen, pure authorization for bounded retries of one model operation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from agent_runtime.modeling.failures import ModelFailure


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_auto_retries: int = 2
    max_elapsed_ms: int = 120_000
    base_delay_ms: int = 500
    max_delay_ms: int = 20_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_elapsed_ms < 1 or self.max_delay_ms < self.base_delay_ms:
            raise ValueError("invalid model retry delay bounds")

    def to_payload(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_binding(cls, binding: Mapping[str, Any]) -> RetryPolicy | None:
        value = binding.get("model_retry_policy")
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("model_retry_policy must be an object")
        return cls(**value)

    def decide(
        self, failure: ModelFailure, *, retry_count: int, first_dispatched_at: float, now: float, random_fraction: float
    ) -> tuple[float, float, int] | None:
        """Return persisted due/deadline/delay, or decline transparent retry."""
        if not all(math.isfinite(value) for value in (now, first_dispatched_at, random_fraction)):
            raise ValueError("retry times and jitter must be finite")
        if not 0 <= random_fraction <= 1:
            raise ValueError("random_fraction must lie between zero and one")
        if (
            retry_count >= self.max_auto_retries
            or now < first_dispatched_at
            or failure.output_started
            or not failure.retryable
            or failure.provider_outcome == "unknown"
            or failure.kind not in {"rate_limited", "server_error", "transport_not_sent"}
        ):
            return None
        deadline_at = first_dispatched_at + self.max_elapsed_ms / 1000
        ceiling = min(self.max_delay_ms, self.base_delay_ms * 2 ** min(retry_count, 63))
        delay_ms = int(ceiling * random_fraction)
        retry_at = max(now + delay_ms / 1000, failure.retry_at if failure.retry_at is not None else now)
        if now > deadline_at or retry_at > deadline_at:
            return None
        return retry_at, deadline_at, round((retry_at - now) * 1000)

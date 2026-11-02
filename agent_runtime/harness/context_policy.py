"""Pure capacity decisions over measurements and durable resource exposure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ContextObservation:
    input_tokens: int
    output_tokens: int
    max_input_tokens: int | None
    working_limit: int | None = None
    normal_tokens_remaining: int | None = None
    uncertain_tokens: int = 0
    guard_reason: str | None = None
    provider_overflow: bool = False
    waiting: bool = False
    protected_input_tokens: int | None = None
    reducible: bool = True


@dataclass(frozen=True, slots=True)
class ContextDecision:
    action: Literal["admit", "compact_required", "compact_optional", "pressure", "wait", "fail"]
    reason: str
    target_input_tokens: int | None = None


class ContextPolicy:
    """Hard admission precedes optional working targets; uncertainty never vanishes."""

    def decide(self, value: ContextObservation) -> ContextDecision:
        if value.waiting:
            return ContextDecision("wait", "unresolved_operation")
        if value.guard_reason is not None:
            return ContextDecision("compact_required", value.guard_reason)
        if (
            value.max_input_tokens is not None
            and value.protected_input_tokens is not None
            and value.protected_input_tokens > value.max_input_tokens
        ):
            return ContextDecision("fail", "protected_input_exceeds_context")
        if value.provider_overflow:
            return ContextDecision("compact_required", "provider_overflow")
        if value.max_input_tokens is not None and value.input_tokens > value.max_input_tokens:
            return ContextDecision("compact_required", "model_context_window_exceeded")
        total = value.input_tokens + value.output_tokens
        if value.normal_tokens_remaining is not None and total * 2 > value.normal_tokens_remaining:
            return ContextDecision("pressure", "turn_budget_pressure")
        if (
            value.reducible
            and not value.uncertain_tokens
            and value.working_limit is not None
            and value.input_tokens > value.working_limit
        ):
            return ContextDecision("compact_optional", "working_context_pressure", value.working_limit * 4 // 5)
        return ContextDecision("admit", "within_hard_limits")

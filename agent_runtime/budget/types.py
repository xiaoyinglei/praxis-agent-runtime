from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


def _require_non_negative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Provider/runtime resource quantities for one operation or budget bucket.

    Token sub-counts may overlap. `total_tokens` intentionally counts only
    logical input + output, so cached/reasoning sub-counts are not double-counted.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    cost_micros: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    retries: int = 0
    subagents: int = 0

    model_ms: int = 0
    tool_ms: int = 0
    wall_ms: int = 0

    tool_output_bytes: int = 0
    network_bytes: int = 0
    artifact_bytes: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_non_negative(name, getattr(self, name))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: ResourceUsage) -> ResourceUsage:
        if not isinstance(other, ResourceUsage):
            return NotImplemented
        return ResourceUsage(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )

    def subtract_floor_zero(self, other: ResourceUsage) -> ResourceUsage:
        """Subtract per field without ever manufacturing negative usage."""
        return ResourceUsage(
            **{
                name: max(getattr(self, name) - getattr(other, name), 0)
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResourceUsage:
        if not isinstance(value, Mapping):
            raise TypeError("resource usage must be a mapping")
        unknown = set(value).difference(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "resource usage contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(
            **{
                name: value.get(name, 0)
                for name in cls.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    """Hard limits for one execution scope. None means not enforced."""

    tokens: int | None = None
    cost_micros: int | None = None
    model_calls: int | None = None
    tool_calls: int | None = None
    retries: int | None = None
    subagents: int | None = None
    wall_ms: int | None = None
    tool_output_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None:
                _require_non_negative(name, value)


@dataclass(frozen=True, slots=True)
class BudgetState:
    """Current accounting view for one Run/Turn/child scope.

    This is a derived read model, not an independently mutable source of truth.
    """

    scope_id: str
    parent_scope_id: str | None
    limits: BudgetLimits

    used: ResourceUsage = ResourceUsage()
    reserved: ResourceUsage = ResourceUsage()
    uncertain: ResourceUsage = ResourceUsage()
    child_reserved: ResourceUsage = ResourceUsage()

    def __post_init__(self) -> None:
        if not isinstance(self.scope_id, str) or not self.scope_id.strip():
            raise ValueError("scope_id must be non-empty")
        if self.parent_scope_id is not None and (
            not isinstance(self.parent_scope_id, str)
            or not self.parent_scope_id.strip()
        ):
            raise ValueError("parent_scope_id must be non-empty or None")

    @property
    def exposure(self) -> ResourceUsage:
        return self.used + self.reserved + self.uncertain + self.child_reserved

    def remaining(self, resource: str) -> int | None:
        limit = getattr(self.limits, resource)
        if limit is None:
            return None
        exposed = _resource_value(self.exposure, resource)
        return max(limit - exposed, 0)


class ReservationStatus(StrEnum):
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    SETTLED = "settled"
    RELEASED = "released"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    scope_id: str
    operation_id: str
    attempt_id: str
    reserved: ResourceUsage
    status: ReservationStatus = ReservationStatus.RESERVED

    def __post_init__(self) -> None:
        for name in (
            "reservation_id",
            "scope_id",
            "operation_id",
            "attempt_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")


def _resource_value(usage: ResourceUsage, resource: str) -> int:
    if resource == "tokens":
        return usage.total_tokens
    if not hasattr(usage, resource):
        raise AttributeError(f"unknown budget resource: {resource}")
    value = getattr(usage, resource)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"budget resource {resource} is not an integer")
    return value

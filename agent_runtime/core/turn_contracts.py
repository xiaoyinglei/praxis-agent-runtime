from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)

from agent_runtime.tools.tool import JsonValue, ToolCall, ToolCallOrigin


class ToolCallPlan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_call_id: str
    tool_name: str
    arguments: dict[str, object]
    origin: ToolCallOrigin | None = None

    @classmethod
    def create(
        cls,
        tool_name: str,
        arguments: dict[str, object],
        *,
        origin: ToolCallOrigin | None = None,
    ) -> ToolCallPlan:
        return cls(
            tool_call_id=f"tc_{uuid4().hex[:12]}",
            tool_name=tool_name,
            arguments=arguments,
            origin=origin,
        )


class ToolManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=500)
    description_hash: str = Field(min_length=1, max_length=200)
    input_schema_hash: str = Field(min_length=1, max_length=200)
    static_effects_hash: str = Field(min_length=1, max_length=200)
    execution_contract_hash: str = Field(min_length=1, max_length=200)


class ToolManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[ToolManifestEntry, ...]
    resident_tool_names: tuple[str, ...] = ()
    explicit_tool_names: tuple[str, ...] = ()
    active_tool_names: tuple[str, ...] = ()
    toolset_revision: str = Field(min_length=1, max_length=500)
    provider_serializer_revision: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_order(self) -> ToolManifest:
        ordered_names = (
            *self.resident_tool_names,
            *self.explicit_tool_names,
            *self.active_tool_names,
        )
        if len(set(ordered_names)) != len(ordered_names):
            raise ValueError("tool manifest orders must contain unique names")
        if tuple(entry.name for entry in self.entries) != ordered_names:
            raise ValueError("tool manifest entries must match resident, explicit, and active order")
        return self


class ToolManifestDriftStatus(StrEnum):
    MATCH = "match"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    NEW_REVISION_REQUIRED = "new_revision_required"


class ToolManifestDriftDecision(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    status: ToolManifestDriftStatus
    reason: str = Field(min_length=1, max_length=200)
    toolset_revision: str = Field(min_length=1, max_length=500)
    active_tool_names: tuple[str, ...]
    missing_tool_names: tuple[str, ...] = ()
    changed_tool_names: tuple[str, ...] = ()
    dependent_tool_calls: tuple[ToolCall, ...] = ()
    provider_wire_hash_guaranteed: bool

    @field_serializer("dependent_tool_calls")
    def serialize_dependent_tool_calls(
        self,
        calls: tuple[ToolCall, ...],
    ) -> list[dict[str, object]]:
        return [
            {
                "tool_call_id": call.tool_call_id,
                "tool_name": call.tool_name,
                "arguments": _plain_tool_json(call.arguments),
                "origin": {
                    "request_id": call.origin.request_id,
                    "toolset_revision": call.origin.toolset_revision,
                    "exposed_tool_names": list(call.origin.exposed_tool_names),
                },
            }
            for call in calls
        ]


def _plain_tool_json(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {
            key: _plain_tool_json(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_plain_tool_json(item) for item in value]
    return value


__all__ = [
    "ToolCallPlan",
    "ToolManifest",
    "ToolManifestDriftDecision",
    "ToolManifestDriftStatus",
    "ToolManifestEntry",
]

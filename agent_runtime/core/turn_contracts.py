from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


__all__ = ["ToolManifest", "ToolManifestEntry"]

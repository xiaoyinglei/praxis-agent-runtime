"""Declarative reasoning capabilities and user request preferences.

None means capability unknown; an empty effort list means explicitly unsupported.
Wire controls are declared by the model, never inferred from its name.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ReasoningCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    supported_efforts: tuple[str, ...] | None = None
    default_effort: str | None = None
    effort_parameter: Literal["reasoning_effort"] = "reasoning_effort"
    thinking_toggle: Literal["thinking", "chat_template_kwargs"] | None = None

    @field_validator("supported_efforts")
    @classmethod
    def valid_levels(cls, levels: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if levels is not None and (
            len(set(levels)) != len(levels)
            or any(not level or level != level.strip() or level in {"default", "off"} for level in levels)
        ):
            raise ValueError("reasoning efforts must be unique nonempty levels excluding default/off")
        return levels

    @model_validator(mode="after")
    def valid_default(self) -> Self:
        if self.default_effort is not None and self.default_effort not in (self.supported_efforts or ()):
            raise ValueError("default_effort must be in supported_efforts")
        return self

    def validate_effort(self, effort: str | None) -> None:
        if effort is None:
            return
        if effort == "off":
            if self.thinking_toggle is None:
                raise ValueError("model does not declare a supported control to disable thinking")
            return
        if self.supported_efforts is None:
            raise ValueError("model reasoning capability is unknown; declare supported_efforts first")
        if not self.supported_efforts:
            raise ValueError("reasoning effort is unsupported by this model")
        if effort not in self.supported_efforts:
            raise ValueError("unsupported reasoning effort; supported: " + ", ".join(self.supported_efforts))


class ModelRequestPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning_effort: str | None = Field(default=None, min_length=1)
    max_output_tokens: int | None = Field(default=None, gt=0, strict=True)

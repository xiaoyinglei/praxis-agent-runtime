"""User-configured ceilings; model arguments cannot increase these limits."""

from pydantic import BaseModel, ConfigDict, Field


class WebLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    turn_requests: int = Field(default=32, ge=1, le=256)
    browser_requests: int = Field(default=64, ge=1, le=128)
    browser_network_bytes: int = Field(default=8_000_000, ge=1_000_000, le=32_000_000)

from __future__ import annotations

from pydantic import BaseModel, Field


class ValidatedFinalOutput(BaseModel):
    """JSON-safe checkpoint representation of a validated Agent output."""

    model_path: str
    data: dict[str, object] = Field(default_factory=dict)


def output_model_path(model: type[BaseModel]) -> str:
    return f"{model.__module__}.{model.__qualname__}"


__all__ = [
    "ValidatedFinalOutput",
    "output_model_path",
]

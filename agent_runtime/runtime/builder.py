from __future__ import annotations

from pathlib import Path

from agent_runtime.models import ModelControlPlane


def build_model_control_plane(
    *,
    model_alias: str | None = None,
    session_path: Path | None = None,
) -> ModelControlPlane:
    return ModelControlPlane.from_env(
        initial_model_id=model_alias,
        session_path=session_path,
    )


__all__ = ["build_model_control_plane"]

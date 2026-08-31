from __future__ import annotations

from pathlib import Path

from agent_runtime.model_config_io import discover_git_worktree
from agent_runtime.models import ModelControlPlane


def build_model_control_plane(
    *,
    model_alias: str | None = None,
    session_path: Path | None = None,
    workspace: Path | None = None,
) -> ModelControlPlane:
    resolved_workspace = (workspace or Path.cwd()).expanduser().resolve()
    return ModelControlPlane.from_env(
        initial_model_id=model_alias,
        session_path=session_path,
        workspace=resolved_workspace,
        worktree=discover_git_worktree(resolved_workspace),
    )


__all__ = ["build_model_control_plane"]

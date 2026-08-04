"""Lazy public exports for agent core contracts."""

from __future__ import annotations

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentRuntimePolicy": ("agent_runtime.core.definition", "AgentRuntimePolicy"),
    "AgentRunConfig": ("agent_runtime.core.context", "AgentRunConfig"),
    "aclose_agent_checkpointer": ("agent_runtime.core.checkpointing", "aclose_agent_checkpointer"),
    "create_agent_checkpointer": ("agent_runtime.core.checkpointing", "create_agent_checkpointer"),
    "ModelSelectionPolicy": ("agent_runtime.core.definition", "ModelSelectionPolicy"),
    "TurnRegistry": ("agent_runtime.core.context", "TurnRegistry"),
    "ToolPolicy": ("agent_runtime.core.definition", "ToolPolicy"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'agent_runtime.core' has no attribute {name!r}") from exc
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

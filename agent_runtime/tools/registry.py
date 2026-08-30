from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from agent_runtime.tools.tool import Tool


class ToolRegistry:
    """Mutable deterministic assembly followed by one immutable snapshot."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._snapshot: Mapping[str, Tool] | None = None

    def register(self, tool: Tool) -> None:
        if self._snapshot is not None:
            raise RuntimeError("tool registry is frozen")
        if not isinstance(tool, Tool):
            raise TypeError("tool registry accepts only canonical Tool values")
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool {name!r} is not registered") from exc

    def list_all(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def freeze(self) -> Mapping[str, Tool]:
        if self._snapshot is None:
            self._snapshot = MappingProxyType(dict(self._tools))
        return self._snapshot

    @property
    def is_frozen(self) -> bool:
        return self._snapshot is not None


def build_tool_registry(*tool_sources: Iterable[Tool] | Tool) -> ToolRegistry:
    """Assemble ordinary Tool values in caller-supplied source order."""

    registry = ToolRegistry()
    for source in tool_sources:
        values = (source,) if isinstance(source, Tool) else source
        for tool in values:
            registry.register(tool)
    return registry


__all__ = ["ToolRegistry", "build_tool_registry"]

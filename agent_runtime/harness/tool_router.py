"""Model-visible tool selection for the replacement Harness."""

from __future__ import annotations

from collections.abc import Mapping

from agent_runtime.harness.protocol import HarnessMessage
from agent_runtime.harness.rollout import RolloutStore
from agent_runtime.harness.tool_orchestrator import (
    inspection_tools_available,
    tool_consumes_inspection_budget,
)
from agent_runtime.tools.tool import Tool


class StaticToolRouter:
    """Expose one immutable ordered tool snapshot on every model step."""

    def __init__(self, tools: Mapping[str, Tool]) -> None:
        self._tools = tuple(tools.values())

    def select(
        self,
        *,
        turn_id: str,
        messages: tuple[HarnessMessage, ...],
    ) -> tuple[Tool, ...]:
        del turn_id
        del messages
        return self._tools


class DurableToolRouter:
    """Derive deferred activation only from committed find_tools results."""

    def __init__(
        self,
        *,
        store: RolloutStore,
        tools: Mapping[str, Tool],
        resident_names: tuple[str, ...],
        discoverable_names: tuple[str, ...],
    ) -> None:
        installed = set(tools)
        configured = (*resident_names, *discoverable_names)
        if len(set(configured)) != len(configured):
            raise ValueError("resident and discoverable tool names must be disjoint")
        missing = tuple(name for name in configured if name not in installed)
        if missing:
            raise ValueError(f"tool router names are not installed: {missing}")
        self._store = store
        self._tools = tools
        self._resident_names = resident_names
        self._discoverable_names = discoverable_names

    def select(
        self,
        *,
        turn_id: str,
        messages: tuple[HarnessMessage, ...],
    ) -> tuple[Tool, ...]:
        del messages
        operations = {
            operation.result_item_id: operation
            for operation in self._store.list_tool_operations(turn_id)
            if operation.result_item_id is not None
        }
        active: list[str] = []
        discoverable = set(self._discoverable_names)
        for item in self._store.list_context_items(turn_id):
            operation = operations.get(item.item_id)
            if (
                item.kind != "tool_result"
                or operation is None
                or operation.tool_name != "find_tools"
                or operation.status != "succeeded"
                or item.payload.get("is_error") is not False
            ):
                continue
            metadata = item.payload.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            proposed = metadata.get("proposed_activation_names")
            if not isinstance(proposed, (list, tuple)):
                continue
            for name in proposed:
                if (
                    isinstance(name, str)
                    and name in discoverable
                    and name not in active
                ):
                    active.append(name)
        selected_names = (*self._resident_names, *active)
        if not inspection_tools_available(self._store, turn_id):
            selected_names = tuple(
                name
                for name in selected_names
                if not tool_consumes_inspection_budget(self._tools[name])
            )
        return tuple(self._tools[name] for name in selected_names)

"""Model-visible tool selection for the replacement Harness."""

from __future__ import annotations

from collections.abc import Mapping

from agent_runtime.harness.protocol import ModelMessage
from agent_runtime.harness.state import RuntimeState
from agent_runtime.tools.tool import Tool


class StaticToolRouter:
    """Expose one immutable ordered tool snapshot on every model step."""

    def __init__(self, tools: Mapping[str, Tool]) -> None:
        self._tools = tuple(tools.values())

    def select(
        self,
        *,
        turn_id: str,
        messages: tuple[ModelMessage, ...],
    ) -> tuple[Tool, ...]:
        del turn_id
        del messages
        return self._tools


class DurableToolRouter:
    """Inherit committed discovery through the visible conversation's fork boundary.

    Discovery exposes installed declarations; the orchestrator still checks the
    executing Turn's permissions and the model call's frozen tool origin.
    """

    def __init__(
        self,
        *,
        store: RuntimeState,
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
        messages: tuple[ModelMessage, ...],
    ) -> tuple[Tool, ...]:
        del messages
        items = self._store.list_context_items(turn_id)
        operations = {
            operation.result_item_id: operation
            for source_turn_id in dict.fromkeys(item.turn_id for item in items)
            for operation in self._store.list_tool_operations(source_turn_id)
            if operation.result_item_id is not None
        }
        active: list[str] = []
        discoverable = set(self._discoverable_names)
        for item in items:
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
        return tuple(self._tools[name] for name in selected_names)

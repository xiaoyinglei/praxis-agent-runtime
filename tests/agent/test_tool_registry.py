from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from agent_runtime.core.model_request import build_tool_manifest
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    json_schema_input,
)


def _tool(
    name: str,
    *,
    execution_revision: str = "runner-v1",
) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Use {name} for its documented operation.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=lambda _arguments: {},
        normalize_output=lambda _raw: NormalizedToolOutput(),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision=execution_revision,
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


def test_registry_registers_canonical_tools_in_insertion_order() -> None:
    first = _tool("first")
    second = _tool("second")
    registry = ToolRegistry()

    registry.register(first)
    registry.register(second)

    assert registry.get("first") is first
    assert registry.list_all() == (first, second)


def test_registry_rejects_duplicate_names_instead_of_overwriting() -> None:
    registry = ToolRegistry()
    registry.register(_tool("duplicate"))

    with pytest.raises(ValueError, match="duplicate.*already registered"):
        registry.register(_tool("duplicate", execution_revision="runner-v2"))


def test_registry_freezes_once_and_rejects_later_registration() -> None:
    first = _tool("first")
    registry = ToolRegistry()
    registry.register(first)

    snapshot = registry.freeze()

    assert snapshot is registry.freeze()
    assert tuple(snapshot) == ("first",)
    assert snapshot["first"] is first
    with pytest.raises(TypeError):
        snapshot["second"] = _tool("second")  # type: ignore[index]
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(_tool("second"))


def test_registry_get_missing_name_fails_loudly() -> None:
    with pytest.raises(KeyError, match="missing"):
        ToolRegistry().get("missing")


def test_manifest_changes_when_execution_contract_changes() -> None:
    original = _tool("read_file", execution_revision="runner-v1")
    changed = replace(original, execution_revision="runner-v2")

    first_registry = ToolRegistry()
    first_registry.register(original)
    second_registry = ToolRegistry()
    second_registry.register(changed)

    first = build_tool_manifest(
        tools=tuple(first_registry.freeze().values()),
        resident_tool_names=("read_file",),
        provider_serializer_revision="provider-v1",
    )
    second = build_tool_manifest(
        tools=tuple(second_registry.freeze().values()),
        resident_tool_names=("read_file",),
        provider_serializer_revision="provider-v1",
    )

    assert first.toolset_revision != second.toolset_revision
    assert first.entries[0].execution_contract_hash != second.entries[0].execution_contract_hash

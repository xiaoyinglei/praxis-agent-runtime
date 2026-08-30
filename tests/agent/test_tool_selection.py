from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import pytest

from agent_runtime.tools.selection import (
    ToolActivationError,
    ToolConfigurationError,
    ToolSchemaBudgetError,
    reduce_tool_activation,
    resolve_tool_options,
    select_tools,
)
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

DEFAULT_NAMES = ("list_files", "search_text")


def _tool(
    name: str,
    *,
    description: str | None = None,
    properties: Mapping[str, JsonValue] | None = None,
) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=description or f"Use {name} for its documented operation.",
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
        execution_revision=f"test-{name}-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=10_000,
    )


def _snapshot(names: Sequence[str]) -> Mapping[str, Tool]:
    return MappingProxyType({name: _tool(name) for name in names})


@pytest.mark.parametrize(
    (
        "installed_names",
        "tools",
        "allow_discovery_tools",
        "expected_resident_names",
        "uses_default_tools",
    ),
    [
        pytest.param(
            (*DEFAULT_NAMES, "find_tools", "mcp__github__issues"),
            None,
            False,
            DEFAULT_NAMES,
            True,
            id="default-discovery-off",
        ),
        pytest.param(
            (*DEFAULT_NAMES, "find_tools"),
            None,
            True,
            DEFAULT_NAMES,
            True,
            id="default-discovery-on-no-hidden",
        ),
        pytest.param(
            (*DEFAULT_NAMES, "find_tools", "mcp__github__issues"),
            (),
            True,
            (*DEFAULT_NAMES, "find_tools"),
            True,
            id="default-discovery-on-with-hidden",
        ),
        pytest.param(
            (*DEFAULT_NAMES, "find_tools", "mcp__github__issues"),
            ("search_text",),
            False,
            ("search_text",),
            False,
            id="exact-discovery-off",
        ),
        pytest.param(
            (*DEFAULT_NAMES, "find_tools", "mcp__github__issues"),
            ("search_text",),
            True,
            ("search_text",),
            False,
            id="exact-discovery-on-without-find-tools",
        ),
        pytest.param(
            (*DEFAULT_NAMES, "find_tools", "mcp__github__issues"),
            ("find_tools", "search_text"),
            True,
            ("find_tools", "search_text"),
            False,
            id="exact-discovery-on-with-find-tools",
        ),
    ],
)
def test_public_tool_option_precedence_matrix(
    installed_names: tuple[str, ...],
    tools: tuple[str, ...] | None,
    allow_discovery_tools: bool,
    expected_resident_names: tuple[str, ...],
    uses_default_tools: bool,
) -> None:
    resolved = resolve_tool_options(
        _snapshot(installed_names),
        default_resident_names=DEFAULT_NAMES,
        tools=tools,
        allow_discovery_tools=allow_discovery_tools,
    )

    assert resolved.resident_names == expected_resident_names
    assert resolved.allow_discovery_tools is allow_discovery_tools
    assert resolved.uses_default_tools is uses_default_tools


def test_default_product_extensions_follow_resident_tools_in_product_order() -> None:
    snapshot = _snapshot(
        (
            "search_knowledge",
            "list_files",
            "task",
            "search_text",
        )
    )

    resolved = resolve_tool_options(
        snapshot,
        default_resident_names=DEFAULT_NAMES,
        configured_resident_names=("task", "search_knowledge"),
    )

    assert resolved.resident_names == (
        "list_files",
        "search_text",
        "search_knowledge",
        "task",
    )


def test_explicit_find_tools_with_disabled_discovery_is_a_configuration_error() -> None:
    snapshot = _snapshot((*DEFAULT_NAMES, "find_tools"))

    with pytest.raises(
        ToolConfigurationError,
        match="find_tools.*allow_discovery_tools",
    ):
        resolve_tool_options(
            snapshot,
            default_resident_names=DEFAULT_NAMES,
            tools=("find_tools",),
            allow_discovery_tools=False,
        )


@pytest.mark.parametrize(
    ("tools", "disabled_tools", "allow_discovery_tools", "expected"),
    [
        pytest.param(
            None,
            ("search_text",),
            False,
            ("list_files",),
            id="disabled-default",
        ),
        pytest.param(
            ("search_text", "mcp__github__issues"),
            ("search_text",),
            True,
            ("mcp__github__issues",),
            id="disabled-exact",
        ),
        pytest.param(
            ("find_tools",),
            ("find_tools",),
            False,
            (),
            id="disabled-find-tools-wins-before-discovery-error",
        ),
    ],
)
def test_disabled_tools_always_win(
    tools: tuple[str, ...] | None,
    disabled_tools: tuple[str, ...],
    allow_discovery_tools: bool,
    expected: tuple[str, ...],
) -> None:
    snapshot = _snapshot((*DEFAULT_NAMES, "find_tools", "mcp__github__issues"))

    resolved = resolve_tool_options(
        snapshot,
        default_resident_names=DEFAULT_NAMES,
        tools=tools,
        disabled_tools=disabled_tools,
        allow_discovery_tools=allow_discovery_tools,
    )

    assert resolved.resident_names == expected
    assert resolved.disabled_names == disabled_tools


def test_disabled_hidden_tools_do_not_trigger_automatic_discovery() -> None:
    snapshot = _snapshot((*DEFAULT_NAMES, "find_tools", "mcp__github__issues"))

    resolved = resolve_tool_options(
        snapshot,
        default_resident_names=DEFAULT_NAMES,
        disabled_tools=("mcp__github__issues",),
        allow_discovery_tools=True,
    )

    assert resolved.resident_names == DEFAULT_NAMES


def test_only_hidden_discoverable_tools_trigger_find_tools() -> None:
    snapshot = _snapshot(
        (*DEFAULT_NAMES, "find_tools", "private_tool", "mcp__docs__search")
    )

    without_candidates = resolve_tool_options(
        snapshot,
        default_resident_names=DEFAULT_NAMES,
        discoverable_names=(),
        allow_discovery_tools=True,
    )
    with_candidate = resolve_tool_options(
        snapshot,
        default_resident_names=DEFAULT_NAMES,
        discoverable_names=("mcp__docs__search",),
        allow_discovery_tools=True,
    )

    assert without_candidates.resident_names == DEFAULT_NAMES
    assert with_candidate.resident_names == (*DEFAULT_NAMES, "find_tools")


@pytest.mark.parametrize(
    ("tools", "disabled_tools", "match"),
    [
        pytest.param(("missing",), (), "unknown explicit tool.*missing"),
        pytest.param(None, ("missing",), "unknown disabled tool.*missing"),
    ],
)
def test_unknown_public_tool_names_fail_before_model_execution(
    tools: tuple[str, ...] | None,
    disabled_tools: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ToolConfigurationError, match=match):
        resolve_tool_options(
            _snapshot(DEFAULT_NAMES),
            default_resident_names=DEFAULT_NAMES,
            tools=tools,
            disabled_tools=disabled_tools,
        )


def test_select_tools_keeps_installed_resident_and_active_state_distinct() -> None:
    snapshot = _snapshot(
        (
            "mcp__hidden__first",
            "list_files",
            "search_text",
            "search_knowledge",
            "mcp__hidden__second",
        )
    )

    selected = select_tools(
        snapshot,
        resident_names=("list_files", "search_text", "search_knowledge"),
        active_names=("mcp__hidden__second",),
        schema_budget=100_000,
    )

    assert tuple(tool.definition.name for tool in selected) == (
        "list_files",
        "search_text",
        "search_knowledge",
        "mcp__hidden__second",
    )
    assert "mcp__hidden__first" in snapshot
    assert all(tool.definition.name != "mcp__hidden__first" for tool in selected)


def test_select_tools_preserves_activation_order_and_deduplicates_names() -> None:
    snapshot = _snapshot(("resident", "active_a", "active_b"))

    selected = select_tools(
        snapshot,
        resident_names=("resident",),
        active_names=("active_b", "active_a", "active_b", "resident"),
    )

    assert tuple(tool.definition.name for tool in selected) == (
        "resident",
        "active_b",
        "active_a",
    )


def test_select_tools_subtracts_disabled_resident_and_active_names() -> None:
    snapshot = _snapshot(("resident", "active", "other"))

    selected = select_tools(
        snapshot,
        resident_names=("resident", "other"),
        active_names=("active",),
        disabled_names=("other", "active"),
    )

    assert tuple(tool.definition.name for tool in selected) == ("resident",)


@pytest.mark.parametrize("source", ["resident_names", "active_names"])
def test_select_tools_rejects_unknown_visible_names(source: str) -> None:
    kwargs: dict[str, tuple[str, ...]] = {
        "resident_names": ("resident",),
        "active_names": (),
    }
    kwargs[source] = ("missing",)

    with pytest.raises(ToolConfigurationError, match="missing"):
        select_tools(_snapshot(("resident",)), **kwargs)


def test_select_tools_reports_schema_budget_overflow_explicitly() -> None:
    snapshot = MappingProxyType(
        {
            "large": _tool(
                "large",
                description="x" * 2000,
                properties={
                    "query": {
                        "type": "string",
                        "description": "y" * 2000,
                    }
                },
            )
        }
    )

    with pytest.raises(
        ToolSchemaBudgetError,
        match="schema budget.*required.*budget",
    ) as error:
        select_tools(
            snapshot,
            resident_names=("large",),
            active_names=(),
            schema_budget=128,
        )

    assert error.value.budget_bytes == 128
    assert error.value.required_bytes > error.value.budget_bytes
    assert error.value.selected_names == ("large",)


def test_activation_reducer_only_appends_new_names_in_proposal_order() -> None:
    snapshot = _snapshot(("resident", "active_a", "active_b", "active_c"))

    reduction = reduce_tool_activation(
        snapshot,
        resident_names=("resident",),
        active_names=("active_a",),
        proposed_names=("active_b", "active_a", "active_c", "active_b"),
        schema_budget=100_000,
        max_active_tools=4,
    )

    assert reduction.active_names == ("active_a", "active_b", "active_c")
    assert reduction.activated_names == ("active_b", "active_c")
    assert reduction.active_names[:1] == ("active_a",)
    assert reduction.trace_metadata == {
        "proposed_activation_names": (
            "active_b",
            "active_a",
            "active_c",
            "active_b",
        ),
        "activated_names": ("active_b", "active_c"),
        "active_names": ("active_a", "active_b", "active_c"),
        "active_tool_count": 3,
    }


def test_activation_reducer_does_not_add_resident_or_disabled_tools() -> None:
    snapshot = _snapshot(("resident", "allowed", "disabled"))

    with pytest.raises(
        ToolActivationError,
        match="disabled.*disabled",
    ) as error:
        reduce_tool_activation(
            snapshot,
            resident_names=("resident",),
            active_names=(),
            proposed_names=("resident", "disabled", "allowed"),
            disabled_names=("disabled",),
        )

    assert error.value.error_code == "tool_activation_disabled"


def test_activation_reducer_rejects_unknown_names_and_count_overflow() -> None:
    snapshot = _snapshot(("active", "candidate"))

    with pytest.raises(ToolActivationError, match="unknown.*missing") as unknown:
        reduce_tool_activation(
            snapshot,
            active_names=("active",),
            proposed_names=("missing",),
        )
    assert unknown.value.error_code == "unknown_tool_activation"

    with pytest.raises(ToolActivationError, match="active tool count") as overflow:
        reduce_tool_activation(
            snapshot,
            active_names=("active",),
            proposed_names=("candidate",),
            max_active_tools=1,
        )
    assert overflow.value.error_code == "tool_activation_count_exceeded"


def test_activation_reducer_rejects_installed_but_non_discoverable_name() -> None:
    snapshot = _snapshot(("resident", "discoverable", "private_tool"))

    with pytest.raises(
        ToolActivationError,
        match="not discoverable.*private_tool",
    ) as error:
        reduce_tool_activation(
            snapshot,
            resident_names=("resident",),
            proposed_names=("private_tool",),
            discoverable_names=("discoverable",),
        )

    assert error.value.error_code == "tool_activation_not_discoverable"


def test_activation_reducer_reuses_selection_schema_budget_check() -> None:
    snapshot = _snapshot(("resident", "candidate"))

    with pytest.raises(ToolSchemaBudgetError):
        reduce_tool_activation(
            snapshot,
            resident_names=("resident",),
            active_names=(),
            proposed_names=("candidate",),
            schema_budget=1,
        )

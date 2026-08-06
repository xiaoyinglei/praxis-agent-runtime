from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

from agent_runtime.tools import selection as selection_module
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.selection import create_find_tools_tool, find_tools
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolDefinition,
    json_schema_input,
)


def _tool(
    name: str,
    description: str,
    *,
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
            description=description,
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


def _snapshot() -> Mapping[str, Tool]:
    tools = (
        _tool("find_tools", "Find hidden tools by capability."),
        _tool("list_files", "List files and directories in the workspace."),
        _tool(
            "search_text",
            "Search source files for literal text or regular expression patterns.",
            properties={
                "pattern": {
                    "type": "string",
                    "description": "Text or regular expression to find in code.",
                }
            },
        ),
        _tool(
            "run_command",
            "Run a shell command in the workspace and capture its output.",
        ),
        _tool(
            "apply_patch",
            "Apply a patch to edit source files in the workspace.",
        ),
        _tool(
            "search_knowledge",
            "Search configured document knowledge for ranked evidence.",
        ),
        _tool(
            "task",
            "Delegate a bounded task to an isolated child agent.",
        ),
        _tool(
            "mcp__github__create_issue",
            "Create an issue on a configured GitHub MCP server.",
            properties={
                "repository": {
                    "type": "string",
                    "description": "Repository owner and name.",
                },
                "title": {
                    "type": "string",
                    "description": "Issue title.",
                },
            },
        ),
    )
    return MappingProxyType({tool.definition.name: tool for tool in tools})


DISCOVERABLE_NAMES = (
    "search_text",
    "run_command",
    "apply_patch",
    "search_knowledge",
    "task",
    "mcp__github__create_issue",
)


@pytest.mark.parametrize(
    ("query", "expected_name"),
    [
        pytest.param("帮我在代码里搜索文本", "search_text", id="search-code"),
        pytest.param("执行终端命令", "run_command", id="run-command"),
        pytest.param("用补丁修改文件", "apply_patch", id="apply-patch"),
        pytest.param("查询知识库文档", "search_knowledge", id="knowledge"),
        pytest.param("委派给子代理", "task", id="subagent"),
    ],
)
def test_find_tools_recalls_canonical_tools_from_chinese_aliases(
    query: str,
    expected_name: str,
) -> None:
    result = find_tools(
        _snapshot(),
        query=query,
        discoverable_names=DISCOVERABLE_NAMES,
        limit=3,
    )

    assert result.error_code is None
    assert result.matches
    assert result.matches[0].name == expected_name
    assert expected_name in result.proposed_activation_names


def test_find_tools_searches_source_namespace_and_input_schema_metadata() -> None:
    snapshot = _snapshot()

    source_result = find_tools(
        snapshot,
        query="github issue",
        discoverable_names=DISCOVERABLE_NAMES,
    )
    schema_result = find_tools(
        snapshot,
        query="repository owner",
        discoverable_names=DISCOVERABLE_NAMES,
    )

    assert source_result.matches[0].name == "mcp__github__create_issue"
    assert schema_result.matches[0].name == "mcp__github__create_issue"
    assert "repository" in schema_result.matches[0].matched_terms


def test_find_tools_is_deterministic_and_never_returns_more_than_limit() -> None:
    snapshot = MappingProxyType(
        {
            name: _tool(name, "Search files for matching content.")
            for name in ("tool_z", "tool_a", "tool_c", "tool_b", "tool_d", "tool_e")
        }
    )

    first = find_tools(
        snapshot,
        query="search files",
        discoverable_names=tuple(snapshot),
        limit=5,
    )
    second = find_tools(
        snapshot,
        query="search files",
        discoverable_names=tuple(snapshot),
        limit=5,
    )

    assert first == second
    assert tuple(match.name for match in first.matches) == (
        "tool_z",
        "tool_a",
        "tool_c",
        "tool_b",
        "tool_d",
    )
    assert len(first.proposed_activation_names) == 5


def test_find_tools_excludes_resident_active_and_disabled_names() -> None:
    result = find_tools(
        _snapshot(),
        query="search code command knowledge",
        discoverable_names=DISCOVERABLE_NAMES,
        resident_names=("search_text",),
        active_names=("run_command",),
        disabled_names=("search_knowledge",),
        limit=5,
    )

    names = tuple(match.name for match in result.matches)
    assert "search_text" not in names
    assert "run_command" not in names
    assert "search_knowledge" not in names
    assert "search_text" not in result.proposed_activation_names
    assert "run_command" not in result.proposed_activation_names
    assert "search_knowledge" not in result.proposed_activation_names


def test_find_tools_returns_no_activation_for_an_unmatched_query() -> None:
    result = find_tools(
        _snapshot(),
        query="quantum banana telescope",
        discoverable_names=DISCOVERABLE_NAMES,
    )

    assert result.matches == ()
    assert result.proposed_activation_names == ()
    assert result.error_code is None


def test_find_tools_reports_count_and_schema_budget_errors_without_eviction() -> None:
    snapshot = _snapshot()
    count_error = find_tools(
        snapshot,
        query="knowledge document",
        discoverable_names=DISCOVERABLE_NAMES,
        active_names=("run_command",),
        max_active_tools=1,
    )
    schema_error = find_tools(
        snapshot,
        query="knowledge document",
        discoverable_names=DISCOVERABLE_NAMES,
        resident_names=("list_files",),
        schema_budget=1,
    )

    assert count_error.error_code == "tool_activation_count_exceeded"
    assert count_error.proposed_activation_names == ()
    assert "active tool count" in (count_error.error_message or "")
    assert schema_error.error_code == "tool_schema_budget_exceeded"
    assert schema_error.proposed_activation_names == ()
    assert "schema budget" in (schema_error.error_message or "")


@pytest.mark.anyio
async def test_find_tools_factory_emits_bounded_matches_and_activation_metadata() -> None:
    snapshot = _snapshot()

    def search(query: str, limit: int):
        return find_tools(
            snapshot,
            query=query,
            discoverable_names=DISCOVERABLE_NAMES,
            limit=limit,
        )

    tool = create_find_tools_tool(search)
    call = ToolCall(
        tool_call_id="call_find_tools",
        tool_name="find_tools",
        arguments={"query": "github issue", "limit": 2},
        origin=ToolCallOrigin(
            request_id="req_find_tools",
            toolset_revision="tools_v1",
            exposed_tool_names=("find_tools",),
        ),
    )
    execution = await ToolExecutor({"find_tools": tool}).execute(
        call,
        context=ToolExecutionContext(),
    )

    assert execution.result.is_error is False
    assert execution.result.structured_content is not None
    assert execution.result.structured_content["matches"][0]["name"] == ("mcp__github__create_issue")
    assert execution.result.metadata["matched_tool_names"] == ("mcp__github__create_issue",)
    assert execution.result.metadata["proposed_activation_names"] == ("mcp__github__create_issue",)


@pytest.mark.anyio
async def test_find_tools_factory_rejects_unmatched_activation_names() -> None:
    tool = create_find_tools_tool(
        lambda _query, _limit: {
            "query": "hidden capability",
            "matches": [],
            "proposed_activation_names": ["not_shown_to_the_model"],
            "error_code": None,
            "error_message": None,
        }
    )
    call = ToolCall(
        tool_call_id="call_find_tools",
        tool_name="find_tools",
        arguments={"query": "hidden capability", "limit": 1},
        origin=ToolCallOrigin(
            request_id="req_find_tools",
            toolset_revision="tools_v1",
            exposed_tool_names=("find_tools",),
        ),
    )
    execution = await ToolExecutor({"find_tools": tool}).execute(
        call,
        context=ToolExecutionContext(),
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == "normalization_failed"


@pytest.mark.anyio
async def test_find_tools_factory_rejects_more_than_five_matches() -> None:
    tool = create_find_tools_tool(
        lambda _query, _limit: find_tools(
            _snapshot(),
            query="search",
            discoverable_names=DISCOVERABLE_NAMES,
        )
    )
    call = ToolCall(
        tool_call_id="call_find_tools",
        tool_name="find_tools",
        arguments={"query": "search", "limit": 6},
        origin=ToolCallOrigin(
            request_id="req_find_tools",
            toolset_revision="tools_v1",
            exposed_tool_names=("find_tools",),
        ),
    )
    execution = await ToolExecutor({"find_tools": tool}).execute(
        call,
        context=ToolExecutionContext(),
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == "invalid_arguments"


def test_selection_module_has_no_runtime_state_or_routing_dependencies() -> None:
    module_path = Path(selection_module.__file__ or "")
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}

    assert not any(
        module.startswith(
            (
                "agent_runtime.loop",
                "agent_runtime.core.checkpointing",
                "agent_runtime.tooling",
                "agent_runtime.capabilities",
            )
        )
        for module in imports
    )
    assert "ToolRegistry" not in source
    assert "embedding" not in source.lower()
    assert "llm" not in source.lower()

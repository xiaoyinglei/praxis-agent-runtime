from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.integrations import mcp as mcp_module
from agent_runtime.tools.integrations.mcp import (
    MCPToolDescriptor,
    canonical_mcp_name,
    create_mcp_tools,
    normalize_mcp_name,
)
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import (
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolEffect,
    ToolValidationError,
)


def _descriptor(
    *,
    server_name: str = "GitHub",
    tool_name: str = "Search Repos",
    input_schema: Mapping[str, Any] | None = None,
    read_only_hint: bool = False,
    destructive_hint: bool = False,
    execution_revision: str = "server-v1",
) -> MCPToolDescriptor:
    return MCPToolDescriptor(
        server_name=server_name,
        tool_name=tool_name,
        description="Search repositories exposed by the configured MCP server.",
        input_schema=input_schema
        or {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        read_only_hint=read_only_hint,
        destructive_hint=destructive_hint,
        execution_revision=execution_revision,
    )


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def test_mcp_names_are_canonical_and_empty_components_fail() -> None:
    assert normalize_mcp_name("Git-Hub Server!") == "git_hub_server"
    assert canonical_mcp_name("GitHub", "Search Repos") == (
        "mcp__github__search_repos"
    )
    with pytest.raises(ValueError, match="non-empty"):
        canonical_mcp_name("---", "search")


def test_mcp_factory_preserves_complete_raw_input_schema() -> None:
    schema = {
        "$defs": {
            "Filter": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"owner": {"type": "string"}},
                        "required": ["owner"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"language": {"type": "string"}},
                        "required": ["language"],
                        "additionalProperties": False,
                    },
                ]
            }
        },
        "type": "object",
        "properties": {
            "filter": {"$ref": "#/$defs/Filter"},
            "query": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["filter"],
        "additionalProperties": False,
    }
    descriptor = _descriptor(
        input_schema=schema,
        execution_revision="github-tools-v2",
    )
    [tool] = create_mcp_tools((descriptor,), lambda *_args: {"content": []})

    assert isinstance(tool, Tool)
    assert tool.execution_revision == "integration-mcp-v1:github-tools-v2"
    assert _thaw(tool.definition.input_schema) == schema
    assert tool.validate_input({"filter": {"owner": "openai"}})["filter"] == {
        "owner": "openai"
    }
    with pytest.raises(ToolValidationError):
        tool.validate_input({"filter": {"unknown": "value"}})


def test_duplicate_canonical_mcp_names_fail_loudly() -> None:
    descriptors = (
        _descriptor(server_name="My Server", tool_name="Get-File"),
        _descriptor(server_name="my-server", tool_name="get file"),
    )

    with pytest.raises(ValueError, match="duplicate canonical MCP tool name"):
        create_mcp_tools(descriptors, lambda *_args: {"content": []})


def test_mcp_annotations_never_remove_network_policy_floor() -> None:
    read_only = create_mcp_tools(
        (_descriptor(read_only_hint=True),),
        lambda *_args: {"content": []},
    )[0]
    destructive = create_mcp_tools(
        (_descriptor(tool_name="delete", destructive_hint=True),),
        lambda *_args: {"content": []},
    )[0]

    assert read_only.static_effects == frozenset({ToolEffect.NETWORK})
    assert read_only.idempotent is True
    assert read_only.concurrency_safe is True
    assert destructive.static_effects == frozenset(
        {ToolEffect.NETWORK, ToolEffect.DESTRUCTIVE}
    )
    assert destructive.idempotent is False
    assert destructive.concurrency_safe is False


@pytest.mark.anyio
async def test_mcp_concrete_tools_execute_through_the_final_tool_contract() -> None:
    calls: list[tuple[str, str, Mapping[str, Any]]] = []

    async def call_tool(
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        calls.append((server_name, tool_name, arguments))
        return {
            "content": [
                {"type": "text", "text": "found repository"},
                {"type": "image", "data": "aW1hZ2U=", "mimeType": "image/png"},
                {"type": "resource", "uri": "file:///repo/readme.md"},
            ],
            "structuredContent": {"count": 1},
            "isError": False,
        }

    [tool] = create_mcp_tools((_descriptor(read_only_hint=True),), call_tool)
    call = ToolCall(
        tool_call_id="call_mcp",
        tool_name=tool.definition.name,
        arguments={"query": "runtime"},
        origin=ToolCallOrigin(
            request_id="req_mcp",
            toolset_revision="tools_mcp_v1",
            exposed_tool_names=(tool.definition.name,),
        ),
    )
    execution = await ToolExecutor({tool.definition.name: tool}).execute(
        call,
        context=ToolExecutionContext(
            approved_tool_call_ids=frozenset({"call_mcp"})
        ),
    )

    assert execution.result.is_error is False
    assert calls == [("GitHub", "Search Repos", {"query": "runtime"})]
    assert [block.type for block in execution.result.content] == [
        "text",
        "image",
        "resource",
    ]
    assert execution.result.content[0].data["text"] == "found repository"
    assert execution.result.content[1].data["mime_type"] == "image/png"
    assert execution.result.content[2].data["uri"] == "file:///repo/readme.md"
    assert execution.result.structured_content == {"count": 1}
    assert execution.result.metadata["mcp_server"] == "GitHub"
    assert execution.result.metadata["mcp_tool"] == "Search Repos"


def test_mcp_integration_does_not_own_client_or_session_lifecycle() -> None:
    module_path = Path(mcp_module.__file__ or "")
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("mcp.client") for module in imports)
    assert "ClientSession" not in source
    assert "connect(" not in source
    assert "disconnect(" not in source

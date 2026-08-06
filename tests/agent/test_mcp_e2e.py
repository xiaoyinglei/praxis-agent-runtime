from __future__ import annotations

import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from agent_runtime.runtime.mcp import open_product_mcp_tools
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.integrations.mcp import (
    MCPToolDescriptor,
    create_mcp_tools,
)
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import JsonValue, ToolCall, ToolCallOrigin

_MCP_SERVER_SCRIPT = textwrap.dedent(
    """\
    import asyncio

    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool


    server = Server("test-mcp-server")


    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="echo",
                description="Echo the supplied message.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo.",
                        }
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="add",
                description="Add two integers.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "Left value."},
                        "b": {"type": "integer", "description": "Right value."},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="read_only_info",
                description="Read server information.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "enum": ["version", "status"],
                            "description": "Information field.",
                        }
                    },
                    "additionalProperties": False,
                },
                annotations={"readOnlyHint": True},
            ),
        ]


    @server.call_tool()
    async def call_tool(name, arguments):
        if name == "echo":
            return [
                TextContent(
                    type="text",
                    text=f"echo: {arguments.get('message', '')}",
                )
            ]
        if name == "add":
            return [
                TextContent(
                    type="text",
                    text=str(arguments.get("a", 0) + arguments.get("b", 0)),
                )
            ]
        if name == "read_only_info":
            field = arguments.get("field", "version")
            return [TextContent(type="text", text=f"{field}: 1.0.0")]
        raise ValueError(f"unknown tool: {name}")


    async def main():
        async with stdio_server() as (read, write):
            await server.run(
                read,
                write,
                server.create_initialization_options(),
            )


    if __name__ == "__main__":
        asyncio.run(main())
    """
)


@pytest.fixture
def server_script_path(tmp_path: Path) -> Path:
    path = tmp_path / "mcp_server.py"
    path.write_text(_MCP_SERVER_SCRIPT, encoding="utf-8")
    return path


def _call(name: str, arguments: Mapping[str, JsonValue]) -> ToolCall:
    return ToolCall(
        tool_call_id=f"call_{name}",
        tool_name=f"mcp__test_server__{name}",
        arguments=arguments,
        origin=ToolCallOrigin(
            request_id="request_mcp",
            toolset_revision="mcp-e2e-v1",
            exposed_tool_names=(f"mcp__test_server__{name}",),
        ),
    )


@pytest.mark.anyio
async def test_product_runtime_opens_enabled_mcp_and_applies_allowlist(
    server_script_path: Path,
    tmp_path: Path,
) -> None:
    config = tmp_path / "mcp.yaml"
    config.write_text(
        "\n".join(
            (
                "servers:",
                "  - name: test_server",
                "    transport: stdio",
                f"    command: {sys.executable}",
                "    args:",
                f"      - {server_script_path}",
                "    tools_allowlist:",
                "      - echo",
                "    enabled: true",
            )
        ),
        encoding="utf-8",
    )

    async with open_product_mcp_tools(config) as tools:
        assert [tool.definition.name for tool in tools] == [
            "mcp__test_server__echo"
        ]
        execution = await ToolExecutor(
            {tool.definition.name: tool for tool in tools}
        ).execute(
            _call("echo", {"message": "assembled"}),
            context=ToolExecutionContext(
                approved_tool_call_ids=frozenset({"call_echo"})
            ),
        )

    assert execution.result.content[0].data["text"] == "echo: assembled"


@pytest.mark.anyio
async def test_unavailable_enabled_mcp_degrades_without_removing_builtins(
    tmp_path: Path,
) -> None:
    config = tmp_path / "mcp.yaml"
    config.write_text(
        "\n".join(
            (
                "servers:",
                "  - name: unavailable",
                "    transport: stdio",
                "    command: definitely-not-an-installed-command",
                "    tools_allowlist: [search]",
                "    enabled: true",
            )
        ),
        encoding="utf-8",
    )
    diagnostics = []

    async with open_product_mcp_tools(
        config,
        diagnostics=diagnostics,
    ) as tools:
        assert tools == ()

    assert diagnostics[0].code == "mcp_server_unavailable"
    assert diagnostics[0].component == "mcp:unavailable"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("enabled", "allow_all_tools", "field"),
    (
        ("'true'", "true", "enabled"),
        ("true", "'false'", "allow_all_tools"),
    ),
)
async def test_product_mcp_config_rejects_string_booleans(
    tmp_path: Path,
    enabled: str,
    allow_all_tools: str,
    field: str,
) -> None:
    config = tmp_path / "mcp.yaml"
    config.write_text(
        "\n".join(
            (
                "servers:",
                "  - name: unsafe_config",
                "    transport: stdio",
                "    command: ignored",
                f"    enabled: {enabled}",
                f"    allow_all_tools: {allow_all_tools}",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        async with open_product_mcp_tools(config):
            pass


@pytest.mark.anyio
async def test_external_transport_projects_real_mcp_tools(
    server_script_path: Path,
) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script_path)],
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            descriptors = tuple(
                MCPToolDescriptor(
                    server_name="test_server",
                    tool_name=item.name,
                    description=item.description or "Configured MCP tool.",
                    input_schema=item.inputSchema,
                    read_only_hint=bool(
                        getattr(item.annotations, "readOnlyHint", False)
                    ),
                    execution_revision="e2e-v1",
                )
                for item in listed.tools
            )

            async def call_tool(
                server_name: str,
                tool_name: str,
                arguments: Mapping[str, JsonValue],
            ) -> object:
                assert server_name == "test_server"
                return await session.call_tool(
                    tool_name,
                    arguments=dict(arguments),
                )

            tools = create_mcp_tools(descriptors, call_tool)
            snapshot = {tool.definition.name: tool for tool in tools}
            executor = ToolExecutor(snapshot)
            context = ToolExecutionContext(
                approved_tool_call_ids=frozenset(
                    {"call_echo", "call_add", "call_read_only_info"}
                )
            )

            echo = await executor.execute(
                _call("echo", {"message": "hello"}),
                context=context,
            )
            added = await executor.execute(
                _call("add", {"a": 3, "b": 4}),
                context=context,
            )
            info = await executor.execute(
                _call("read_only_info", {"field": "version"}),
                context=context,
            )

    assert tuple(snapshot) == (
        "mcp__test_server__echo",
        "mcp__test_server__add",
        "mcp__test_server__read_only_info",
    )
    assert echo.result.content[0].data["text"] == "echo: hello"
    assert added.result.content[0].data["text"] == "7"
    assert info.result.content[0].data["text"] == "version: 1.0.0"
    assert all(
        execution.result.is_error is False
        for execution in (echo, added, info)
    )


@pytest.mark.anyio
async def test_mcp_schema_failure_is_normalized_before_transport_call(
    server_script_path: Path,
) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script_path)],
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            echo_descriptor = next(
                MCPToolDescriptor(
                    server_name="test_server",
                    tool_name=item.name,
                    description=item.description or "Configured MCP tool.",
                    input_schema=item.inputSchema,
                )
                for item in listed.tools
                if item.name == "echo"
            )
            calls = 0

            async def call_tool(
                _server_name: str,
                tool_name: str,
                arguments: Mapping[str, JsonValue],
            ) -> object:
                nonlocal calls
                calls += 1
                return await session.call_tool(
                    tool_name,
                    arguments=dict(arguments),
                )

            [tool] = create_mcp_tools((echo_descriptor,), call_tool)
            execution = await ToolExecutor(
                {tool.definition.name: tool}
            ).execute(
                _call("echo", {}),
                context=ToolExecutionContext(
                    approved_tool_call_ids=frozenset({"call_echo"})
                ),
            )

    assert execution.result.is_error is True
    assert execution.result.error_code == "invalid_arguments"
    assert calls == 0

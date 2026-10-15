"""Host-owned tool metadata; remote sessions connect only when a tool is used."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import stat
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic
from agent_runtime.permission_policy import ResolvedPermissions
from agent_runtime.runtime.mcp import _HTTPServerConfig, _StdioServerConfig, open_product_mcp_tools
from agent_runtime.tools.integrations.mcp import (
    MCPToolDescriptor,
    canonical_mcp_name,
    create_mcp_tools,
    normalize_mcp_name,
)
from agent_runtime.tools.tool import JsonValue, NormalizedToolOutput, ResolvedToolUse, Tool, ToolContentBlock
from agent_runtime.user_config import user_config_directory

_MAX_CATALOG_BYTES = 4 * 1024 * 1024


def _catalog_path(config: Path) -> Path:
    key = hashlib.sha256(str(config.resolve()).encode()).hexdigest()
    return user_config_directory() / "mcp_catalog" / (key + ".json")


def _read_catalog(config: Path) -> dict[str, Any]:
    path = _catalog_path(config)
    try:
        for target in (path.parent, path.parent.parent):
            metadata = target.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
                return {}
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "r") as handle:
            metadata = os.fstat(handle.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o022 or metadata.st_size > _MAX_CATALOG_BYTES):
                return {}
            value = json.loads(handle.read(_MAX_CATALOG_BYTES + 1))
        if (isinstance(value, dict) and value.get("config_sha256") == hashlib.sha256(config.read_bytes()).hexdigest()
                and isinstance(value.get("servers"), dict)):
            return dict(value["servers"])
    except (OSError, ValueError):
        pass
    return {}


def catalog_revision(config: Path) -> str:
    """Detect host-side setup changes between Turns without reading credentials."""
    try:
        content = json.dumps(_read_catalog(config), sort_keys=True).encode() + config.read_bytes()
    except OSError:
        content = b"missing"
    return hashlib.sha256(content).hexdigest()


def write_catalog(config: Path, descriptors: Sequence[MCPToolDescriptor], *, server_names: frozenset[str]) -> None:
    from agent_runtime.mcp_admin import _atomic_write, configuration_lock

    with configuration_lock():
        servers = _read_catalog(config)
        for name in server_names:
            servers[name] = [asdict(item) for item in descriptors if item.server_name == name]
        payload = json.dumps({"config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "servers": servers})
        if len(payload.encode()) > _MAX_CATALOG_BYTES:
            raise ValueError("MCP tool catalog exceeds local metadata limit")
        directory = _catalog_path(config).parent
        directory.mkdir(mode=0o700, exist_ok=True)
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
            raise PermissionError("MCP catalog directory must be owned and writable only by the user")
        _atomic_write(_catalog_path(config), payload)


class _ConnectionRequired:
    def __init__(self, server: str) -> None:
        self.server = server


def _connection_output(value: object) -> NormalizedToolOutput:
    if not isinstance(value, _ConnectionRequired):
        raise TypeError("Expected a host-owned MCP connection request")
    message = (f"需要连接或更新 MCP 服务 {value.server}。请点击打开连接管理，或输入 /connections，"
               "由用户完成登录并检查连接后再试。不会自动打开授权页。")
    return NormalizedToolOutput(
        content=(ToolContentBlock(type="text", data={"text": message}),), is_error=True,
        error_code="mcp_connection_required", error_message=message, metadata={"mcp_server": value.server},
    )


def _connection_tool(name: str) -> Tool:
    # This local placeholder exposes an unavailable service to find_tools without
    # inventing remote tool schemas or making a startup authentication request.
    descriptor = MCPToolDescriptor(
        server_name=name, tool_name="connect", description=f"Connect {name} MCP service; user login/setup is required.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False}, read_only_hint=True,
    )
    tool = create_mcp_tools((descriptor,), lambda *_: _ConnectionRequired(name))[0]
    return replace(
        tool, static_effects=frozenset(), resolve_use=lambda _: ResolvedToolUse(effects=frozenset(), targets=()),
        normalize_output=_connection_output, idempotent=True,
    )


def _binding(descriptor: MCPToolDescriptor) -> dict[str, Any]:
    value = asdict(descriptor)
    value.pop("execution_revision")  # Permission revisions are checked live, independently of the cached schema.
    return value


@asynccontextmanager
async def open_lazy_mcp_tools(
    config_path: Path, *, configs: Sequence[_HTTPServerConfig | _StdioServerConfig],
    permissions: ResolvedPermissions | None, permission_provider: Callable[[], ResolvedPermissions] | None,
    close_timeout_seconds: float,
) -> AsyncIterator[tuple[Tool, ...]]:
    cached = _read_catalog(config_path)
    revision = hashlib.sha256(config_path.read_bytes()).hexdigest()
    stack = AsyncExitStack()
    connected: dict[str, dict[str, Tool]] = {}
    unavailable: set[str] = set()
    locks: dict[str, asyncio.Lock] = {}
    expected: dict[str, MCPToolDescriptor] = {}

    def current_policy() -> ResolvedPermissions | None:
        current = permissions if permission_provider is None else permission_provider()
        return current.intersection(permissions) if current is not None and permissions is not None else current

    async def call(server: str, name: str, arguments: Mapping[str, JsonValue]) -> object:
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != revision:
            return _ConnectionRequired(server)
        policy = current_policy()
        canonical = canonical_mcp_name(server, name)
        descriptor = expected[canonical]
        if policy is not None and (policy.tool_decision(canonical) == "deny"
                                   or policy.network_decision(descriptor.endpoint_url or "") != "allow"):
            raise PermissionError("Current permission policy denies the MCP connection")
        async with locks[server]:
            if server in unavailable:
                return _ConnectionRequired(server)
            if server not in connected:
                found: list[MCPToolDescriptor] = []
                diagnostics: list[RuntimeDiagnostic] = []
                tools = await stack.enter_async_context(open_product_mcp_tools(
                    config_path, permissions=permissions, permission_provider=permission_provider,
                    server_names=frozenset({server}), diagnostics=diagnostics, catalog=found,
                    close_timeout_seconds=close_timeout_seconds,
                ))
                fresh = {canonical_mcp_name(item.server_name, item.tool_name): item for item in found}
                # Check every cached definition before an approved call can reach
                # the server. Changes to schemas, effects, identity or credentials
                # require an explicit check and a new immutable Turn tool snapshot.
                if not tools or any(key not in fresh or _binding(item) != _binding(fresh[key])
                                    for key, item in expected.items() if item.server_name == server):
                    unavailable.add(server)
                    return _ConnectionRequired(server)
                connected[server] = {tool.definition.name: tool for tool in tools}
        # Do not catch errors after dispatch: a remote action may already have run.
        result = connected[server][canonical].run(arguments)
        return await result if inspect.isawaitable(result) else result

    try:
        tools: list[Tool] = []
        stdio_names = frozenset(config.name for config in configs if isinstance(config, _StdioServerConfig))
        if stdio_names:
            tools.extend(await stack.enter_async_context(open_product_mcp_tools(
                config_path, permissions=permissions, permission_provider=permission_provider,
                server_names=stdio_names, close_timeout_seconds=close_timeout_seconds,
            )))
        policy = current_policy()
        for config in configs:
            if not isinstance(config, _HTTPServerConfig):
                continue
            if policy is not None and (policy.tool_decision(f"mcp__{normalize_mcp_name(config.name)}__*") == "deny"
                                       or policy.network_decision(config.url) != "allow"):
                continue
            try:
                descriptors = [MCPToolDescriptor(**item) for item in cached.get(config.name, [])]
                descriptors = [item for item in descriptors if item.server_name == config.name
                               and item.endpoint_url == config.url
                               and (config.allow_all_tools or item.tool_name in config.tools_allowlist)
                               and (not config.read_only or item.read_only_hint)
                               and (policy is None or policy.tool_decision(
                                   canonical_mcp_name(item.server_name, item.tool_name)) != "deny")]
                server_tools = create_mcp_tools(descriptors, call)
            except (TypeError, ValueError):
                descriptors = []
                server_tools = ()
            if not descriptors:
                tools.append(_connection_tool(config.name))
                continue
            locks[config.name] = asyncio.Lock()
            for item in descriptors:
                expected[canonical_mcp_name(item.server_name, item.tool_name)] = item
            for tool in server_tools:
                def wrapped(tool: Tool) -> Tool:
                    def normalize(value: object) -> NormalizedToolOutput:
                        return (_connection_output(value) if isinstance(value, _ConnectionRequired)
                                else tool.normalize_output(value))
                    return replace(tool, normalize_output=normalize)
                tools.append(wrapped(tool))
        yield tuple(tools)
    finally:
        from agent_runtime.runtime.mcp import _close_mcp_stack
        await _close_mcp_stack(stack, timeout_seconds=close_timeout_seconds, label="lazy runtime")

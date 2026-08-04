from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic
from agent_runtime.tools.integrations.mcp import (
    MCPToolDescriptor,
    create_mcp_tools,
)
from agent_runtime.tools.tool import JsonValue, Tool

logger = logging.getLogger(__name__)
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True, slots=True)
class _StdioServerConfig:
    name: str
    command: str
    args: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Path | None
    tools_allowlist: frozenset[str]
    allow_all_tools: bool
    startup_timeout_seconds: float


@asynccontextmanager
async def open_product_mcp_tools(
    config_path: Path | None,
    *,
    diagnostics: list[RuntimeDiagnostic] | None = None,
    close_timeout_seconds: float = 5.0,
) -> AsyncIterator[tuple[Tool, ...]]:
    """Open enabled stdio MCP servers and project them into ordinary Tools."""

    if config_path is None or not config_path.is_file():
        yield ()
        return
    configs = _load_enabled_servers(config_path)
    if not configs:
        yield ()
        return

    stack = AsyncExitStack()
    sessions: dict[str, ClientSession] = {}
    descriptors: list[MCPToolDescriptor] = []
    try:
        for config in configs:
            server_stack = AsyncExitStack()
            params = StdioServerParameters(
                command=config.command,
                args=list(config.args),
                env=dict(config.env) or None,
                cwd=config.cwd,
            )
            try:
                streams = await asyncio.wait_for(
                    server_stack.enter_async_context(stdio_client(params)),
                    timeout=config.startup_timeout_seconds,
                )
                session = await asyncio.wait_for(
                    server_stack.enter_async_context(ClientSession(*streams)),
                    timeout=config.startup_timeout_seconds,
                )
                initialized = await asyncio.wait_for(
                    session.initialize(),
                    timeout=config.startup_timeout_seconds,
                )
                listed = await asyncio.wait_for(
                    session.list_tools(),
                    timeout=config.startup_timeout_seconds,
                )
                listed_names = {item.name for item in listed.tools}
                missing = config.tools_allowlist - listed_names
                if missing:
                    raise ValueError(
                        f"MCP server {config.name!r} is missing allowlisted tools: "
                        + ", ".join(sorted(missing))
                    )
                version = str(
                    getattr(initialized.serverInfo, "version", "unknown")
                )
                server_descriptors: list[MCPToolDescriptor] = []
                for item in listed.tools:
                    if (
                        not config.allow_all_tools
                        and item.name not in config.tools_allowlist
                    ):
                        continue
                    annotations = item.annotations
                    server_descriptors.append(
                        MCPToolDescriptor(
                            server_name=config.name,
                            tool_name=item.name,
                            description=item.description
                            or "Configured MCP tool.",
                            input_schema=cast(
                                Mapping[str, JsonValue], item.inputSchema
                            ),
                            read_only_hint=bool(
                                getattr(annotations, "readOnlyHint", False)
                            ),
                            destructive_hint=bool(
                                getattr(annotations, "destructiveHint", False)
                            ),
                            idempotent_hint=bool(
                                getattr(annotations, "idempotentHint", False)
                            ),
                            execution_revision=f"{config.name}:{version}",
                        )
                    )
            except Exception as exc:
                if diagnostics is not None:
                    diagnostics.append(
                        RuntimeDiagnostic.from_exception(
                            code="mcp_server_unavailable",
                            component=f"mcp:{config.name}",
                            error=exc,
                        )
                    )
                logger.warning(
                    "MCP server %s is unavailable: %s",
                    config.name,
                    exc,
                )
                await _close_mcp_stack(
                    server_stack,
                    timeout_seconds=close_timeout_seconds,
                    label=f"server {config.name}",
                )
                continue

            await stack.enter_async_context(server_stack)
            sessions[config.name] = session
            descriptors.extend(server_descriptors)

        async def call_tool(
            server_name: str,
            tool_name: str,
            arguments: Mapping[str, JsonValue],
        ) -> object:
            session = sessions.get(server_name)
            if session is None:
                raise RuntimeError(f"MCP session is not active: {server_name}")
            return await session.call_tool(tool_name, arguments=dict(arguments))

        yield create_mcp_tools(descriptors, call_tool)
    finally:
        await _close_mcp_stack(
            stack,
            timeout_seconds=close_timeout_seconds,
            label="runtime",
        )


async def _close_mcp_stack(
    stack: AsyncExitStack,
    *,
    timeout_seconds: float,
    label: str,
) -> None:
    try:
        await asyncio.wait_for(stack.aclose(), timeout=timeout_seconds)
    except TimeoutError:
        logger.warning(
            "MCP %s shutdown exceeded %.1fs grace period",
            label,
            timeout_seconds,
        )
    except Exception:
        logger.warning("MCP %s shutdown failed", label, exc_info=True)


def resolve_product_mcp_config(workspace_root: Path | None) -> Path | None:
    configured = os.environ.get("AGENT_MCP_CONFIG", "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MCP config does not exist: {path}")
        return path
    candidates: list[Path] = []
    if workspace_root is not None:
        candidates.append(workspace_root / "configs" / "mcp_servers.yaml")
    candidates.append(Path(__file__).resolve().parents[2] / "configs" / "mcp_servers.yaml")
    return next((path for path in candidates if path.is_file()), None)


def _load_enabled_servers(config_path: Path) -> tuple[_StdioServerConfig, ...]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("MCP config must be a mapping")
    raw_servers = payload.get("servers", ())
    if not isinstance(raw_servers, list):
        raise ValueError("MCP config servers must be a list")
    configs: list[_StdioServerConfig] = []
    names: set[str] = set()
    for raw in raw_servers:
        if not isinstance(raw, Mapping):
            raise ValueError("MCP server entries must be mappings")
        if not _boolean(raw, "enabled", default=False):
            continue
        name = _required_text(raw, "name")
        if name in names:
            raise ValueError(f"duplicate enabled MCP server: {name}")
        names.add(name)
        transport = str(raw.get("transport", "stdio"))
        if transport != "stdio":
            raise ValueError(
                f"unsupported MCP transport for {name}: {transport}"
            )
        command = _required_text(raw, "command")
        args = _string_sequence(raw.get("args", ()), field="args")
        env = _environment(raw.get("env", {}), server_name=name)
        allowlist = frozenset(
            _string_sequence(
                raw.get("tools_allowlist", ()),
                field="tools_allowlist",
            )
        )
        allow_all = _boolean(raw, "allow_all_tools", default=False)
        if not allowlist and not allow_all:
            raise ValueError(
                f"enabled MCP server {name!r} requires tools_allowlist "
                "or allow_all_tools"
            )
        cwd_value = raw.get("cwd")
        cwd = None
        if cwd_value is not None:
            cwd = Path(str(cwd_value)).expanduser()
            if not cwd.is_absolute():
                cwd = config_path.parent / cwd
            cwd = cwd.resolve()
        timeout_value = raw.get("startup_timeout_seconds", 20.0)
        if isinstance(timeout_value, bool) or not isinstance(
            timeout_value,
            (int, float),
        ):
            raise ValueError("startup_timeout_seconds must be numeric")
        timeout = float(timeout_value)
        if timeout <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        configs.append(
            _StdioServerConfig(
                name=name,
                command=command,
                args=args,
                env=env,
                cwd=cwd,
                tools_allowlist=allowlist,
                allow_all_tools=allow_all,
                startup_timeout_seconds=timeout,
            )
        )
    return tuple(configs)


def _required_text(raw: Mapping[object, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MCP server {field} must be a non-empty string")
    return value.strip()


def _boolean(
    raw: Mapping[object, object],
    field: str,
    *,
    default: bool,
) -> bool:
    value = raw.get(field, default)
    if type(value) is not bool:
        raise ValueError(f"MCP server {field} must be a boolean")
    return value


def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"MCP server {field} must be a list")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"MCP server {field} contains an empty value")
    return result


def _environment(value: object, *, server_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"MCP server {server_name!r} env must be a mapping")
    expanded: dict[str, str] = {}
    for key, raw in value.items():
        text = str(raw)

        def replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            if env_name not in os.environ:
                raise ValueError(
                    f"MCP server {server_name!r} requires environment "
                    f"variable {env_name}"
                )
            return os.environ[env_name]

        expanded[str(key)] = _ENV_REFERENCE.sub(replace, text)
    return expanded


__all__ = ["open_product_mcp_tools", "resolve_product_mcp_config"]

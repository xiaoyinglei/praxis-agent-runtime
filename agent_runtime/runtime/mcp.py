from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import yaml
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic
from agent_runtime.permission_policy import ResolvedPermissions
from agent_runtime.tools.builtins import shell
from agent_runtime.tools.integrations.mcp import (
    MCPToolDescriptor,
    canonical_mcp_name,
    create_mcp_tools,
    normalize_mcp_name,
)
from agent_runtime.tools.tool import JsonValue, Tool
from agent_runtime.tools.web_http import PublicWebError, public_http_transport, validate_public_url
from agent_runtime.user_config import user_config_directory
from agent_runtime.workspace import WorkspaceRuntime

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


@dataclass(frozen=True, slots=True)
class _HTTPServerConfig:
    name: str
    url: str
    headers: Mapping[str, str]
    bearer_token_env_var: str | None
    keychain: bool
    oauth: bool
    oauth_client_id: str | None
    oauth_scopes: tuple[str, ...]
    oauth_allowed_domains: tuple[str, ...]
    read_only: bool
    tools_allowlist: frozenset[str]
    allow_all_tools: bool
    startup_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class MCPConfigTrustDecision:
    config_path: Path
    source: Literal["product", "workspace", "external", "user"]
    config_sha256: str


def decide_mcp_config_trust(
    config_path: Path,
    *,
    workspace_root: Path | None,
    trust_workspace: bool,
    trusted_external_configs: tuple[Path, ...] = (),
    product_config_root: Path | None = None,
) -> MCPConfigTrustDecision:
    """Authorize the exact config bytes before any MCP process can start."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MCP config does not exist: {path}")
    product_root = (
        Path(__file__).resolve().parents[2] / "configs"
        if product_config_root is None
        else Path(product_config_root).expanduser().resolve()
    )
    workspace = (
        None
        if workspace_root is None
        else Path(workspace_root).expanduser().resolve()
    )
    trusted_external = {
        Path(candidate).expanduser().resolve()
        for candidate in trusted_external_configs
    }
    source: Literal["product", "workspace", "external", "user"]
    personal = user_config_directory() / "mcp_servers.yaml"
    if path == personal:
        for target in (path, path.parent):
            metadata = target.stat()
            if metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
                raise PermissionError("personal MCP configuration must be user-owned and not group/world writable")
        source = "user"
    elif path.is_relative_to(product_root):
        source = "product"
    elif workspace is not None and path.is_relative_to(workspace):
        if not trust_workspace:
            raise PermissionError(
                f"workspace MCP config is not trusted: {path}"
            )
        source = "workspace"
    elif path in trusted_external:
        source = "external"
    else:
        raise PermissionError(f"external MCP config is not allowlisted: {path}")
    return MCPConfigTrustDecision(
        config_path=path,
        source=source,
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@asynccontextmanager
async def open_trusted_product_mcp_tools(
    config_path: Path,
    *,
    trust: MCPConfigTrustDecision,
    diagnostics: list[RuntimeDiagnostic] | None = None,
    close_timeout_seconds: float = 5.0,
    permissions: ResolvedPermissions | None = None,
    permission_provider: Callable[[], ResolvedPermissions] | None = None,
    server_names: frozenset[str] | None = None,
    lazy: bool = False,
    cache_catalog: bool = False,
) -> AsyncIterator[tuple[Tool, ...]]:
    """Verify a prior trust decision, then and only then start MCP servers."""
    path = Path(config_path).expanduser().resolve()
    if path != trust.config_path:
        raise RuntimeError("MCP config path differs from trust decision")
    current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if current_hash != trust.config_sha256:
        raise RuntimeError("MCP config changed after trust decision")
    async with open_product_mcp_tools(
        path,
        diagnostics=diagnostics,
        close_timeout_seconds=close_timeout_seconds,
        permissions=permissions,
        permission_provider=permission_provider, server_names=server_names, lazy=lazy, cache_catalog=cache_catalog,
    ) as tools:
        yield tools


@asynccontextmanager
async def open_product_mcp_tools(
    config_path: Path | None,
    *,
    diagnostics: list[RuntimeDiagnostic] | None = None,
    close_timeout_seconds: float = 5.0,
    permissions: ResolvedPermissions | None = None,
    permission_provider: Callable[[], ResolvedPermissions] | None = None,
    server_names: frozenset[str] | None = None,
    lazy: bool = False,
    cache_catalog: bool = False,
    catalog: list[MCPToolDescriptor] | None = None,
) -> AsyncIterator[tuple[Tool, ...]]:
    """Open configured stdio or Streamable HTTP servers through the same tool adapter.

    Product callers supply live permissions: stdio has a read-only OS sandbox;
    HTTP requires a preauthorized public HTTPS endpoint for every request.
    Without permissions this remains a host-owned transport utility. Changes to
    config or live policy require reconnecting before another tool call.
    lazy=True uses checked local metadata for HTTP tools and connects on use;
    cache_catalog=True records metadata during an explicit host connection check.
    """

    if config_path is None or not config_path.is_file():
        yield ()
        return
    configs = _load_enabled_servers(config_path, server_names=server_names)
    if not configs:
        yield ()
        return

    if lazy:
        from agent_runtime.runtime.mcp_catalog import open_lazy_mcp_tools
        async with open_lazy_mcp_tools(
            config_path, configs=configs, permissions=permissions, permission_provider=permission_provider,
            close_timeout_seconds=close_timeout_seconds,
        ) as tools:
            yield tools
        return

    config_revision = hashlib.sha256(config_path.read_bytes()).hexdigest()
    stack = AsyncExitStack()
    sessions: dict[str, ClientSession] = {}
    start_revisions: dict[str, str] = {}
    start_policies: dict[str, ResolvedPermissions] = {}
    descriptors: list[MCPToolDescriptor] = []
    try:
        for config in configs:
            server_stack = AsyncExitStack()
            try:
                current = permissions if permission_provider is None else permission_provider()
                effective = current
                if current is not None and permissions is not None and current.revision != permissions.revision:
                    effective = permissions.intersection(current)
                if effective is not None:
                    server_name = f"mcp__{normalize_mcp_name(config.name)}__*"
                    if effective.tool_decision(server_name) == "deny" or (
                        not config.allow_all_tools and all(
                            effective.tool_decision(canonical_mcp_name(config.name, name)) == "deny"
                            for name in config.tools_allowlist
                        )
                    ):
                        raise PermissionError(f"permission policy denies MCP server: {config.name}")
                if isinstance(config, _HTTPServerConfig):
                    if effective is not None and effective.network_decision(config.url) != "allow":
                        raise PermissionError("remote MCP discovery requires network allow for its destination")
                connection = await server_stack.enter_async_context(_owned_mcp_connection(
                    config, config_path, effective, permission_provider, close_timeout_seconds,
                ))
                session, initialized, listed, credential_revision = connection
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
                    if effective is not None and effective.tool_decision(
                        canonical_mcp_name(config.name, item.name)
                    ) == "deny":
                        continue
                    annotations = item.annotations
                    if (isinstance(config, _HTTPServerConfig) and config.read_only
                            and not getattr(annotations, "readOnlyHint", False)):
                        continue
                    connection = ({"transport": "http", "url": config.url, "headers": dict(config.headers),
                                   "read_only": config.read_only, "auth": config.bearer_token_env_var,
                                   "keychain": config.keychain, "oauth": config.oauth,
                                   "credential_revision": credential_revision}
                                  if isinstance(config, _HTTPServerConfig) else
                                  {"transport": "stdio", "command": config.command, "args": config.args,
                                   "env": dict(config.env), "cwd": None if config.cwd is None else str(config.cwd)})
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
                            endpoint_url=config.url if isinstance(config, _HTTPServerConfig) else None,
                            require_action_approval=(isinstance(config, _HTTPServerConfig)
                                                     and not getattr(annotations, "readOnlyHint", False)),
                            connection_revision=hashlib.sha256(json.dumps({
                                "connection": connection, "version": version,
                            }, sort_keys=True).encode()).hexdigest(),
                            execution_revision="mcp_" + hashlib.sha256(json.dumps({
                                "server": config.name, "connection": connection,
                                "version": version, "tool": item.model_dump(mode="json"),
                                "permissions": None if effective is None else effective.revision,
                            }, sort_keys=True).encode()).hexdigest(),
                        )
                    )
            except BaseException as exc:
                # The local stack owns resources as soon as acquisition succeeds;
                # cancellation before transfer must close them in this same task.
                await _close_mcp_stack(
                    server_stack, timeout_seconds=close_timeout_seconds, label=f"server {config.name}",
                )
                if not isinstance(exc, Exception):
                    raise
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
                continue

            await stack.enter_async_context(server_stack)
            sessions[config.name] = session
            if current is not None and effective is not None:
                start_revisions[config.name] = current.revision
                start_policies[config.name] = effective
            descriptors.extend(server_descriptors)

        async def call_tool(
            server_name: str,
            tool_name: str,
            arguments: Mapping[str, JsonValue],
        ) -> object:
            if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_revision:
                raise PermissionError("MCP config changed; reconnect before using it")
            session = sessions.get(server_name)
            if session is None:
                raise RuntimeError(f"MCP session is not active: {server_name}")
            current = permissions if permission_provider is None else permission_provider()
            if current is not None:
                if current.revision != start_revisions.get(server_name):
                    raise PermissionError("MCP permissions changed after startup; restart the server before calling it")
                name = canonical_mcp_name(server_name, tool_name)
                if current.tool_decision(name) == "deny" or start_policies[server_name].tool_decision(name) == "deny":
                    raise PermissionError(f"permission policy denies MCP tool: {name}")
            return await session.call_tool(tool_name, arguments=dict(arguments))

        if catalog is not None:
            catalog.extend(descriptors)
        if cache_catalog:
            from agent_runtime.runtime.mcp_catalog import write_catalog
            if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_revision:
                raise PermissionError("MCP config changed during discovery; check the connection again")
            write_catalog(config_path, descriptors, server_names=frozenset(sessions))
        yield create_mcp_tools(descriptors, call_tool)
    finally:
        await _close_mcp_stack(
            stack,
            timeout_seconds=close_timeout_seconds,
            label="runtime",
        )



def validate_http_mcp_url(value: str) -> str:
    """Remote MCP accepts a public HTTPS endpoint without URL credentials."""
    try:
        raw = httpx.URL(value)
        if raw.fragment or raw.query:
            raise ValueError
        parsed = validate_public_url(value)
        if parsed.scheme != "https":
            raise ValueError
        return str(parsed)
    except (PublicWebError, ValueError):
        raise ValueError("remote MCP requires a public HTTPS URL without credentials, query, or fragment") from None


class _MCPHTTPTransport(httpx.AsyncBaseTransport):
    def __init__(self, config: _HTTPServerConfig, config_path: Path,
                 permissions: ResolvedPermissions | None,
                 provider: Callable[[], ResolvedPermissions] | None) -> None:
        self.inner = public_http_transport()
        self.config = config
        self.config_path = config_path
        self.config_revision = hashlib.sha256(config_path.read_bytes()).hexdigest()
        self.permissions = permissions
        self.provider = provider

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) != self.config.url:
            raise PermissionError("MCP request cannot change its configured endpoint")
        validate_http_mcp_url(str(request.url))
        if request.headers.get("host", "").casefold() != request.url.netloc.decode("ascii").casefold():
            raise PermissionError("MCP HTTP Host authority must match its configured endpoint")
        self.check_permissions()
        response = await self.inner.handle_async_request(request)
        if response.is_redirect:
            await response.aclose()
            raise PermissionError("MCP HTTP redirects are refused; credentials stay at the configured endpoint")
        if response.status_code in {401, 403} and not self.config.oauth:
            status = response.status_code
            await response.aclose()
            raise PermissionError(f"remote MCP authentication or authorization failed (HTTP {status})")
        return response

    def check_permissions(self) -> None:
        if hashlib.sha256(self.config_path.read_bytes()).hexdigest() != self.config_revision:
            raise PermissionError("MCP config changed; reconnect before using it")
        current = self.permissions if self.provider is None else self.provider()
        if current is not None and self.permissions is not None:
            current = current.intersection(self.permissions)
        if current is not None:
            if current.network_decision(self.config.url) != "allow":
                raise PermissionError("remote MCP destination is not preauthorized by current network policy")
            if current.tool_decision(f"mcp__{normalize_mcp_name(self.config.name)}__*") == "deny":
                raise PermissionError("MCP server denied by current permission policy")

    async def aclose(self) -> None:
        await self.inner.aclose()


@asynccontextmanager
async def _http_mcp_streams(config: _HTTPServerConfig, config_path: Path,
                            permissions: ResolvedPermissions | None,
                            provider: Callable[[], ResolvedPermissions] | None) -> AsyncIterator[Any]:
    headers = dict(_environment(config.headers, server_name=config.name))
    if config.bearer_token_env_var is not None:
        token = os.environ.get(config.bearer_token_env_var, "")
        if not token:
            raise ValueError(f"missing MCP credential environment variable: {config.bearer_token_env_var}")
        headers["Authorization"] = "Bearer " + token
    if config.keychain:
        from agent_runtime.mcp_admin import load_mcp_credential
        headers["Authorization"] = "Bearer " + load_mcp_credential(config.name, config.url)
    transport: httpx.AsyncBaseTransport = _MCPHTTPTransport(config, config_path, permissions, provider)
    auth = None
    if config.oauth:
        from agent_runtime.mcp_oauth import OAuthHTTPTransport, create_oauth_provider
        auth = create_oauth_provider({
            "name": config.name, "url": config.url, "oauth_client_id": config.oauth_client_id,
            "oauth_scopes": config.oauth_scopes, "oauth_allowed_domains": config.oauth_allowed_domains,
        })
        # OAuth has its own endpoint checks. MCP calls still pass through the
        # existing policy/config-drift guard before reaching the public socket.
        mcp_transport = cast(_MCPHTTPTransport, transport)
        def check_oauth_destination(url: str) -> None:
            mcp_transport.check_permissions()
            current = permissions if provider is None else provider()
            if current is not None and permissions is not None:
                current = current.intersection(permissions)
            if current is not None and current.network_decision(url) == "deny":
                raise PermissionError("Current network policy denies OAuth destination")

        auth.destination_check = check_oauth_destination
        class DispatchTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                if str(request.url) == config.url:
                    return await mcp_transport.handle_async_request(request)
                return await oauth_transport.handle_async_request(request)

            async def aclose(self) -> None:
                await mcp_transport.aclose()
                await oauth_transport.aclose()

        oauth_transport = OAuthHTTPTransport(auth, public_http_transport())
        transport = DispatchTransport()
    async with httpx.AsyncClient(
        transport=transport, headers=headers, auth=auth,
        timeout=httpx.Timeout(config.startup_timeout_seconds, read=300),
        trust_env=False, follow_redirects=False,
    ) as client:
        async with streamable_http_client(config.url, http_client=client) as streams:
            def credential_revision() -> str:
                credential_state = {"headers": headers, "oauth": None if auth is None else auth.storage.read()}
                return hashlib.sha256(json.dumps(credential_state, sort_keys=True).encode()).hexdigest()
            yield streams, credential_revision


@asynccontextmanager
async def _owned_mcp_connection(config: _StdioServerConfig | _HTTPServerConfig, config_path: Path,
                                permissions: ResolvedPermissions | None,
                                provider: Callable[[], ResolvedPermissions] | None,
                                close_timeout: float) -> AsyncIterator[Any]:
    # MCP SDK contexts use task-local AnyIO cancel scopes. The owning task must
    # enter and leave them, including when discovery fails or shutdown is bounded.
    ready: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()

    async def serve() -> None:
        try:
            async with AsyncExitStack() as resources:
                async with asyncio.timeout(config.startup_timeout_seconds):
                    if isinstance(config, _HTTPServerConfig):
                        streams, credential_revision = await resources.enter_async_context(
                            _http_mcp_streams(config, config_path, permissions, provider),
                        )
                        client = ClientSession(streams[0], streams[1], read_timeout_seconds=timedelta(
                            seconds=config.startup_timeout_seconds,
                        ))
                    else:
                        params = (_sandboxed_mcp_parameters(config, permissions, resources)
                                  if permissions is not None else StdioServerParameters(
                                      command=config.command, args=list(config.args),
                                      env=dict(config.env) or None, cwd=config.cwd,
                                  ))
                        streams = await resources.enter_async_context(stdio_client(params))
                        def no_credentials() -> None:
                            return None
                        credential_revision = no_credentials
                        client = ClientSession(streams[0], streams[1])
                    session = await resources.enter_async_context(client)
                    initialized = await session.initialize()
                    listed = await session.list_tools()
                # Bind any OAuth refresh only after initialization completes.
                if not ready.done():
                    ready.set_result((session, initialized, listed, credential_revision()))
                await stop.wait()
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            elif not isinstance(error, asyncio.CancelledError):
                logger.warning("MCP connection %s closed: %s", config.name, type(error).__name__)

    task = asyncio.create_task(serve(), name=f"mcp-owner:{config.name}")
    try:
        yield await ready
    finally:
        stop.set()
        if ready.cancelled():
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), close_timeout)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

def _sandboxed_mcp_parameters(
    config: _StdioServerConfig, permissions: ResolvedPermissions, stack: AsyncExitStack,
) -> StdioServerParameters:
    workspace = WorkspaceRuntime(root=Path(permissions.workspace), is_temporary=False)
    cwd = workspace.root if config.cwd is None else config.cwd
    workspace.ensure_within_workspace(cwd)
    if not permissions.path_access(cwd):
        raise PermissionError(permissions.path_reason(cwd))
    allow_network = permissions.command_network_allowed() == "allow"
    issue = shell._sandbox_availability_issue(shell.RunCommandInput(command=config.command, network=allow_network))
    issue = issue or shell._workspace_sandbox_preflight_issue(workspace.root, allow_workspace_write=False)
    if issue is not None:
        raise RuntimeError(f"{issue[0]}: {issue[1]}")
    workspace.scratch.mkdir(parents=True, exist_ok=True)
    temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix="mcp-server-", dir=workspace.scratch))
    environment = shell._command_environment(workspace_root=workspace.root, temporary_root=temporary)
    sandbox_argv = shell._sandbox_argv(
        workspace_root=workspace.root, temporary_root=temporary, allow_network=allow_network,
        allow_workspace_write=False, additional_read_roots=shell._runtime_python_read_roots(), permissions=permissions,
    )
    # stdio_client merges a few default host variables into params.env. Reset the
    # actual server environment *inside* the sandbox; explicit trusted values
    # must never affect sandbox-exec itself (for example dynamic-loader settings).
    server_env = {**environment, **config.env}
    return StdioServerParameters(
        command=sandbox_argv[0],
        args=[*sandbox_argv[1:], "/usr/bin/env", "-i", *(f"{key}={value}" for key, value in sorted(server_env.items())),
              config.command, *config.args],
        env=environment, cwd=cwd,
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
    candidates: list[Path] = [user_config_directory() / "mcp_servers.yaml"]
    if workspace_root is not None:
        candidates.append(workspace_root / "configs" / "mcp_servers.yaml")
    candidates.append(Path(__file__).resolve().parents[2] / "configs" / "mcp_servers.yaml")
    return next((path for path in candidates if path.is_file()), None)


def _load_enabled_servers(config_path: Path, *, server_names: frozenset[str] | None = None
                          ) -> tuple[_StdioServerConfig | _HTTPServerConfig, ...]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("MCP config must be a mapping")
    raw_servers = payload.get("servers", ())
    if not isinstance(raw_servers, list):
        raise ValueError("MCP config servers must be a list")
    configs: list[_StdioServerConfig | _HTTPServerConfig] = []
    names: set[str] = set()
    for raw in raw_servers:
        if not isinstance(raw, Mapping):
            raise ValueError("MCP server entries must be mappings")
        if server_names is not None and raw.get("name") not in server_names:
            continue
        if not _boolean(raw, "enabled", default=False):
            continue
        name = _required_text(raw, "name")
        if name in names:
            raise ValueError(f"duplicate enabled MCP server: {name}")
        names.add(name)
        transport = str(raw.get("transport", "stdio"))
        if transport not in {"stdio", "http"}:
            raise ValueError(
                f"unsupported MCP transport for {name}: {transport}"
            )
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
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if transport == "http":
            url = validate_http_mcp_url(_required_text(raw, "url"))
            headers = raw.get("headers", {})
            if not isinstance(headers, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()
            ):
                raise ValueError("MCP HTTP headers must map strings to strings")
            token_env = raw.get("bearer_token_env_var")
            if token_env is not None and (not isinstance(token_env, str)
                                          or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env)):
                raise ValueError("bearer_token_env_var must name an environment variable")
            keychain = _boolean(raw, "keychain", default=False)
            oauth = _boolean(raw, "oauth", default=False)
            if sum((token_env is not None, keychain, oauth)) > 1:
                raise ValueError("choose one MCP credential source")
            if oauth and any(key.casefold() == "authorization" for key in headers):
                raise ValueError("OAuth cannot be combined with an Authorization header")
            from agent_runtime.mcp_oauth import validate_oauth_domains
            oauth_domains = validate_oauth_domains(_string_sequence(
                raw.get("oauth_allowed_domains", ()), field="oauth_allowed_domains",
            ))
            oauth_scopes = _string_sequence(raw.get("oauth_scopes", ()), field="oauth_scopes")
            client_id = raw.get("oauth_client_id")
            if client_id is not None and (not isinstance(client_id, str) or not client_id
                                          or any(ord(c) < 32 for c in client_id)):
                raise ValueError("oauth_client_id must be nonempty single-line text")
            configs.append(_HTTPServerConfig(
                name=name, url=url, headers=dict(headers), bearer_token_env_var=token_env,
                keychain=keychain, oauth=oauth, oauth_client_id=client_id,
                oauth_scopes=oauth_scopes, oauth_allowed_domains=oauth_domains,
                read_only=_boolean(raw, "read_only", default=True),
                tools_allowlist=allowlist, allow_all_tools=allow_all, startup_timeout_seconds=timeout,
            ))
            continue
        configs.append(
            _StdioServerConfig(
                name=name,
                command=_required_text(raw, "command"),
                args=_string_sequence(raw.get("args", ()), field="args"),
                env=_environment(raw.get("env", {}), server_name=name),
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


__all__ = [
    "MCPConfigTrustDecision",
    "decide_mcp_config_trust",
    "open_product_mcp_tools",
    "open_trusted_product_mcp_tools",
    "resolve_product_mcp_config",
]

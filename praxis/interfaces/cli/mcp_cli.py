"""``praxis mcp``:MCP server 配置的只读/管理面(设计 §5.10;T19)。

**契约**:
- 本模块只经 ``praxis.interfaces.sdk`` 触达内核(``praxis_cli_via_sdk``);
  YAML 编辑是 CLI 侧的运维动作,不构成第二条配置解析/校验路径——读到的条目
  在下次 ``connect_mcp_servers`` 时仍经 T15 唯一加载入口(信任门/卫生/
  ``${VAR}`` 展开/字节 sha)校验。
- **秘密纪律(R-T19-1)**:``add`` 的 ``--env`` 只接受 ``KEY`` 或
  ``KEY=${VAR}`` 引用;字面值一律拒绝并给出可执行文案。秘密只活在环境变量/
  Keychain,绝不进 shell history、也绝不进 YAML。``list/show/status`` 只打印
  变量名,绝不打印任何展开值。
- **无连接纪律(Q4)**:``list/show/status`` 是纯配置面读出,绝不建连接、绝不
  派生 server 进程;连接只发生在 ``run/chat/resume`` 的单 loop 装配里。
- 退出码与主 CLI 一致:0 完成 / 1 运行时失败(配置损坏、凭据保存失败)/
  3 用法错误(``typer.BadParameter`` 由主 app 的 ``_UsageExit3Group`` 重映射)。
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn
from uuid import uuid4

import typer
import yaml

from praxis.interfaces.sdk import (
    HttpServerConfig,
    decide_mcp_config_trust,
    mcp_config_write_path,
    normalize_mcp_name,
    resolve_mcp_config_path,
    save_mcp_bearer,
)

EXIT_COMPLETED = 0
EXIT_FAILED = 1

_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ENV_REFERENCE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_CLI_ERRORS = (OSError, ValueError, subprocess.TimeoutExpired)
_DEFAULT_TIMEOUT = 20.0

_WORKSPACE_OPT = typer.Option(None, "--workspace", help="Workspace directory (default: cwd).")
_TOOL_OPT = typer.Option([], "--tool", help="Allowlist tool name (repeatable).")
_ALLOW_ALL_OPT = typer.Option(False, "--allow-all", help="Expose every tool the server lists.")
_DISABLED_OPT = typer.Option(False, "--disabled", help="Write the entry disabled (loaded only when enabled).")
_ARGS_OPT = typer.Option(
    [],
    "--args",
    help="One stdio argument (repeatable; use --args=-x for values starting with '-').",
)
_ENV_OPT = typer.Option(
    [], "--env", help="KEY or KEY=${VAR} reference for the stdio process env (repeatable)."
)

app = typer.Typer(help="Inspect and manage MCP server configuration (no connections).", no_args_is_help=True)
add_app = typer.Typer(help="Add a stdio or HTTP MCP server entry.", no_args_is_help=True)
app.add_typer(add_app, name="add")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(EXIT_FAILED)


def _workspace(value: Path | None) -> Path:
    return Path(value) if value is not None else Path.cwd()


def _source_path(workspace: Path) -> Path | None:
    """已通过信任门的配置路径;隐式默认缺失返回 None,显式缺失是运行时失败。"""
    try:
        return resolve_mcp_config_path(workspace)
    except ValueError as exc:
        _fail(str(exc))


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        _fail("MCP configuration could not be read")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError:
        # 固定文案:YAML 解析错误会带出文件片段。
        _fail("MCP configuration must be valid YAML")
    if payload is None:
        return {"servers": []}
    if not isinstance(payload, dict):
        _fail("MCP config must be a mapping")
    servers = payload.setdefault("servers", [])
    if not isinstance(servers, list):
        _fail("MCP config servers must be a list")
    return payload


def _entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    servers = payload.get("servers", [])
    entries: list[dict[str, Any]] = []
    for raw in servers:
        if not isinstance(raw, dict):
            _fail("MCP server entries must be mappings")
        entries.append(raw)
    return entries


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """同目录 tmp + os.replace 原子写;0600 让下次加载的卫生检查天然通过。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
        tmp.chmod(0o600)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _editable_path(workspace: Path) -> Path:
    """remove/login 的编辑目标:与读路径同源;无配置是可诊断失败。"""
    source = _source_path(workspace)
    if source is None:
        _fail("no MCP configuration; run: praxis mcp add stdio <name> --command …")
    return source


def _find(entries: Sequence[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for entry in entries:
        if entry.get("name") == name:
            return entry
    return None


def _check_name(entries: Sequence[dict[str, Any]], name: str) -> None:
    """名字必须能规范化且不与既有条目(精确或归一化后)冲突。"""
    normalized = normalize_mcp_name(name)
    if not normalized:
        raise typer.BadParameter("MCP server name must normalize to a non-empty value")
    for entry in entries:
        raw = entry.get("name")
        if raw == name:
            raise typer.BadParameter(f"MCP server {name!r} already exists; run praxis mcp remove first")
        if isinstance(raw, str) and normalize_mcp_name(raw) == normalized:
            raise typer.BadParameter(
                f"MCP server name {name!r} collides with {raw!r} after normalization"
            )


def _parse_env(items: Sequence[str]) -> dict[str, str]:
    """R-T19-1:只接受 KEY 或 KEY=${VAR};字面值在 CLI 边界拒绝。"""
    env: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not _ENV_KEY.fullmatch(key):
            raise typer.BadParameter(f"--env expects KEY or KEY=${{VAR}}, got {item!r}")
        if key in env:
            raise typer.BadParameter(f"duplicate --env for {key!r}")
        if not separator:
            env[key] = f"${{{key}}}"
            continue
        reference = value.strip()
        if _ENV_REFERENCE.fullmatch(reference) is None:
            raise typer.BadParameter(
                f"--env {key}=<value> would write a literal secret into the YAML; "
                f"use --env {key}=${{VAR}} and keep the value in the environment or keychain"
            )
        env[key] = reference
    return env


def _tool_fields(tool: Sequence[str], allow_all: bool) -> dict[str, Any]:
    if allow_all and tool:
        raise typer.BadParameter("use --tool or --allow-all, not both")
    if allow_all:
        return {"allow_all_tools": True}
    if tool:
        return {"tools_allowlist": list(tool)}
    raise typer.BadParameter("pass --tool NAME (repeatable) or --allow-all")


def _add_entry(workspace: Path, entry: Mapping[str, Any]) -> None:
    path = mcp_config_write_path()
    payload = _read_payload(path) if path.exists() else {"servers": []}
    _check_name(_entries(payload), str(entry["name"]))
    payload.setdefault("servers", []).append(dict(entry))
    try:
        _write_payload(path, payload)
    except OSError as exc:
        _fail(f"cannot write MCP configuration {path}: {exc}")
    typer.echo(f"ok: added MCP server {entry['name']} to {path}")


# ---------------------------------------------------------------------------
# list / show / status(纯配置面,绝不连接)
# ---------------------------------------------------------------------------


@app.command("list")
def list_servers(workspace: Path | None = _WORKSPACE_OPT) -> None:
    """列出全部条目(含 disabled),显示来源文件与信任类别;绝不打印 env 值。"""
    root = _workspace(workspace)
    path = _source_path(root)
    if path is None:
        typer.echo("no MCP configuration")
        return
    trust = decide_mcp_config_trust(path, workspace=root)
    payload = _read_payload(path)
    typer.echo(f"config: {trust.config_path} (trust: {trust.source})")
    entries = _entries(payload)
    if not entries:
        typer.echo("no MCP servers configured")
        return
    for entry in entries:
        name = str(entry.get("name", "?"))
        transport = str(entry.get("transport", "stdio"))
        enabled = entry.get("enabled", False) is True
        state = "enabled" if enabled else "disabled"
        detail = _detail_for(entry, transport)
        typer.echo(f"  {state:<8} {name}\t{transport}\t{detail}")


def _detail_for(entry: Mapping[str, Any], transport: str) -> str:
    tools = "*" if entry.get("allow_all_tools") is True else ",".join(
        str(item) for item in entry.get("tools_allowlist") or ()
    )
    if transport == "http":
        credential = "keychain" if entry.get("keychain") is True else str(
            entry.get("bearer_token_env_var") or "none"
        )
        return f"{entry.get('url', '-')}\tcredential={credential}\ttools={tools or '-'}"
    env = entry.get("env")
    names = ",".join(sorted(str(key) for key in env)) if isinstance(env, Mapping) else ""
    command = " ".join([str(entry.get("command", "?")), *(str(a) for a in entry.get("args") or ())])
    return f"{command}\tenv={names or '-'}\ttools={tools or '-'}"


@app.command("show")
def show(name: str, workspace: Path | None = _WORKSPACE_OPT) -> None:
    """显示一个条目的完整配置;env/headers 只显示键名,绝不显示值。"""
    root = _workspace(workspace)
    path = _editable_path(root)
    entry = _find(_entries(_read_payload(path)), name)
    if entry is None:
        _fail(f"MCP server {name!r} is not configured; run: praxis mcp list")
    transport = str(entry.get("transport", "stdio"))
    typer.echo(f"server: {name}")
    typer.echo(f"  transport: {transport}")
    typer.echo(f"  enabled: {str(entry.get('enabled', False) is True).lower()}")
    tools = (
        "*"
        if entry.get("allow_all_tools") is True
        else ",".join(str(item) for item in entry.get("tools_allowlist") or ())
    )
    typer.echo(f"  tools: {tools or '-'}")
    typer.echo(f"  startup_timeout_seconds: {entry.get('startup_timeout_seconds', _DEFAULT_TIMEOUT)}")
    if transport == "http":
        typer.echo(f"  url: {entry.get('url', '-')}")
        typer.echo(f"  keychain: {str(entry.get('keychain', False) is True).lower()}")
        typer.echo(f"  bearer_token_env_var: {entry.get('bearer_token_env_var') or '-'}")
        typer.echo(f"  read_only: {str(entry.get('read_only', True) is True).lower()}")
        headers = entry.get("headers")
        names = ",".join(sorted(str(key) for key in headers)) if isinstance(headers, Mapping) else "-"
        typer.echo(f"  headers: {names or '-'}")
        return
    typer.echo(f"  command: {entry.get('command', '-')}")
    typer.echo(f"  args: {' '.join(str(a) for a in entry.get('args') or ()) or '-'}")
    typer.echo(f"  cwd: {entry.get('cwd') or '-'}")
    env = entry.get("env")
    names = ",".join(sorted(str(key) for key in env)) if isinstance(env, Mapping) else "-"
    typer.echo(f"  env: {names or '-'}")


@app.command("status")
def status(workspace: Path | None = _WORKSPACE_OPT) -> None:
    """纯配置侧状态:文件/信任/条目/transport/enabled。**绝不尝试连接**。"""
    root = _workspace(workspace)
    path = _source_path(root)
    if path is None:
        typer.echo("MCP configuration: none")
        typer.echo("connections: not attempted (status is configuration-side only)")
        return
    trust = decide_mcp_config_trust(path, workspace=root)
    entries = _entries(_read_payload(path))
    enabled = sum(1 for entry in entries if entry.get("enabled", False) is True)
    typer.echo(f"config: {trust.config_path}")
    typer.echo(f"  trust: {trust.source}")
    typer.echo(f"  sha256: {trust.config_sha256}")
    typer.echo(f"  servers: {len(entries)} ({enabled} enabled, {len(entries) - enabled} disabled)")
    for entry in entries:
        name = str(entry.get("name", "?"))
        transport = str(entry.get("transport", "stdio"))
        state = "enabled" if entry.get("enabled", False) is True else "disabled"
        typer.echo(f"  {name}\t{transport}\t{state}")
    typer.echo("connections: not attempted (status is configuration-side only)")


# ---------------------------------------------------------------------------
# add stdio / add http(只写 ${VAR} 引用;R-T19-1)
# ---------------------------------------------------------------------------


@add_app.command("stdio")
def add_stdio(
    name: str,
    command: str = typer.Option(..., "--command", help="Executable to spawn in the sandbox."),
    args: list[str] = _ARGS_OPT,
    env: list[str] = _ENV_OPT,
    tool: list[str] = _TOOL_OPT,
    allow_all: bool = _ALLOW_ALL_OPT,
    cwd: str | None = typer.Option(None, "--cwd", help="Working directory inside the workspace."),
    disabled: bool = _DISABLED_OPT,
    workspace: Path | None = _WORKSPACE_OPT,
) -> None:
    """写一个 stdio server 条目(--env 只接受 KEY 或 KEY=${VAR} 引用)。"""
    if not command.strip():
        raise typer.BadParameter("--command must be a non-empty executable path")
    entry: dict[str, Any] = {
        "name": name,
        "enabled": not disabled,
        "transport": "stdio",
        "command": command.strip(),
        "args": list(args),
        **_tool_fields(tool, allow_all),
    }
    references = _parse_env(env)
    if references:
        entry["env"] = references
    if cwd is not None:
        entry["cwd"] = cwd
    _add_entry(_workspace(workspace), entry)


@add_app.command("http")
def add_http(
    name: str,
    url: str = typer.Option(..., "--url", help="Public HTTPS endpoint (no credentials/query/fragment)."),
    bearer_env: str | None = typer.Option(
        None, "--bearer-env", help="Environment variable holding the bearer token (name only)."
    ),
    keychain: bool = typer.Option(False, "--keychain", help="Read the bearer token from the keychain store."),
    tool: list[str] = _TOOL_OPT,
    allow_all: bool = _ALLOW_ALL_OPT,
    read_write: bool = typer.Option(False, "--read-write", help="Allow non-read-only tools (default: read-only)."),
    disabled: bool = _DISABLED_OPT,
    workspace: Path | None = _WORKSPACE_OPT,
) -> None:
    """写一个 HTTP server 条目;凭据来源必须恰好是 --bearer-env 或 --keychain。"""
    if (bearer_env is None) == (not keychain):
        raise typer.BadParameter("choose exactly one credential source: --bearer-env NAME or --keychain")
    if bearer_env is not None and not _ENV_KEY.fullmatch(bearer_env):
        raise typer.BadParameter("--bearer-env expects an environment variable name, never a value")
    fields = _tool_fields(tool, allow_all)
    try:
        config = HttpServerConfig(
            name=name,
            url=url,
            headers={},
            bearer_token_env_var=bearer_env,
            keychain=keychain,
            read_only=not read_write,
            tools_allowlist=frozenset(tool),
            allow_all_tools=allow_all,
            startup_timeout_seconds=_DEFAULT_TIMEOUT,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    entry: dict[str, Any] = {
        "name": name,
        "enabled": not disabled,
        "transport": "http",
        "url": config.url,
        "read_only": config.read_only,
        **fields,
    }
    if config.bearer_token_env_var is not None:
        entry["bearer_token_env_var"] = config.bearer_token_env_var
    if config.keychain:
        entry["keychain"] = True
    _add_entry(_workspace(workspace), entry)


# ---------------------------------------------------------------------------
# remove / login
# ---------------------------------------------------------------------------


@app.command("remove")
def remove(name: str, workspace: Path | None = _WORKSPACE_OPT) -> None:
    """删除一个条目(按精确名字;凭据不由本命令删除)。"""
    path = _editable_path(_workspace(workspace))
    payload = _read_payload(path)
    entries = _entries(payload)
    remaining = [entry for entry in entries if entry.get("name") != name]
    if len(remaining) == len(entries):
        _fail(f"MCP server {name!r} is not configured; run: praxis mcp list")
    payload["servers"] = remaining
    try:
        _write_payload(path, payload)
    except OSError as exc:
        _fail(f"cannot write MCP configuration {path}: {exc}")
    typer.echo(f"ok: removed MCP server {name}")


@app.command("login")
def login(name: str, workspace: Path | None = _WORKSPACE_OPT) -> None:
    """为 http server 保存 bearer(hidden prompt;值不进 argv/日志/YAML)。"""
    path = _editable_path(_workspace(workspace))
    entry = _find(_entries(_read_payload(path)), name)
    if entry is None:
        _fail(f"MCP server {name!r} is not configured; run: praxis mcp list")
    if any(str(key).casefold().startswith("oauth") for key in entry):
        # Q6/T15 同一文案:OAuth 族键在 3B 明确拒绝,login 不提供旁路。
        _fail("MCP OAuth lands in phase 5; remove oauth/oauth_* keys from this server")
    if str(entry.get("transport", "stdio")) != "http":
        _fail(
            f"MCP server {name!r} is a stdio server; login applies to HTTP servers only "
            "(stdio credentials live in its env references)"
        )
    url = entry.get("url")
    if not isinstance(url, str) or not url:
        _fail(f"MCP server {name!r} has no url; fix the configuration first")
    value = typer.prompt(f"bearer token for {name}", hide_input=True)
    if not value.strip():
        _fail("empty credential; nothing saved")
    try:
        save_mcp_bearer(name, url, value.strip())
    except _CLI_ERRORS as exc:
        _fail(str(exc))
    typer.echo(f"ok: credential saved for {name}")


__all__ = ["add_app", "app"]

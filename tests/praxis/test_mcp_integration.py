"""Task 19:MCP manager + composition/sdk/CLI 接线(Q4/Q9)。

**覆盖**(brief Step 1 ①–⑥ + controller rulings):
- eager connect:两个 server(真 stdio fixture + 假 streamable-http responder)的
  工具都进入 bundle/工具表;``McpBundle.aclose`` 幂等且回收进程组;
- 策略预检:wildcard deny / allowlist 全 deny / HTTP network ask 都在派生
  owner task 前跳过(``mcp_server_skipped``),消费 ``protocol.policy`` 同一代数;
- 单 server 崩溃隔离(另一 server 仍可用,诊断 ``mcp_server_unavailable``,
  异常组摊平有界,配置秘密值绝不进诊断);
- 工具表 merge 冲突(extra↔builtin、跨 server)在组合根边界拒绝;
- ``Session.close/aclose`` 顺序:command_sessions → integrations → store;
- CLI ``mcp add/list/show/remove/login/status`` roundtrip(``PRAXIS_CONFIG_DIR``
  隔离;R-T19-1 字面值拒绝;status 绝不连接);
- 单 loop:真实 ``asyncio.run`` 场景下带 MCP 工具提交 turn(ask ⇒ 退出码 2),
  结束后无 owner task / 无孤儿进程;
- 无 MCP 时既有同步路径零行为变化(connect 根本不被调用)。

**fixture 分层**(与 T16 同纪律):mcp 集成用真子进程 + 真 SDK,但沙箱用
``/usr/bin/env`` 透传 spec(机制证明,不冒充安全边界);CLI 单 loop 用例走
生产 ``compile_sandbox``,只在 ``probe_backend().available`` 的平台运行。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from mcp.types import Tool as McpTool
from typer.testing import CliRunner

from praxis.adapters.mcp import http_transport
from praxis.adapters.mcp import manager as mcp_manager
from praxis.adapters.mcp._errors import bounded_exception_message, diagnostic_message
from praxis.adapters.mcp.config import load_mcp_configs
from praxis.adapters.mcp.manager import McpBundle, connect_mcp_servers
from praxis.adapters.mcp.names import server_wildcard
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.sandbox.base import probe_backend
from praxis.composition import build_session
from praxis.interfaces import sdk as sdk_pkg
from praxis.interfaces.cli import app, mcp_cli
from praxis.interfaces.cli import main as cli_main
from praxis.protocol.floor import host_config_dir
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy, ToolRule
from praxis.protocol.sandbox import SandboxSpec, ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    Tool,
    ToolDefinition,
    ToolSource,
    json_schema_input,
)
from praxis.tools.builtins.child_env import child_env

runner = CliRunner()

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mcp_probe_server.py"
_HOST_CONFIG = "mcp_servers.yaml"
_HTTP_URL = "https://mcp.example.com/rpc"
_OWNER_PREFIX = "mcp-owner:"
_AVAILABLE = probe_backend().available
_ENV_VAR = "PRAXIS_MCP_CONFIG"


@pytest.fixture(autouse=True)
def _isolated_mcp_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """绝不读写真实宿主 MCP 配置(``PRAXIS_CONFIG_DIR`` 已由 conftest 隔离)。"""
    monkeypatch.delenv(_ENV_VAR, raising=False)
    directory = host_config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def _token() -> str:
    return "t19-" + uuid.uuid4().hex


def _secret() -> str:
    return "sk-t19-" + uuid.uuid4().hex


def _env(_tmp_path: Path, **extra: str | None) -> dict[str, str | None]:
    # 与 ``_write_config`` 同源:CLI 子进程与测试读到同一个宿主配置目录。
    return {
        "PRAXIS_CONFIG_DIR": str(host_config_dir()),
        _ENV_VAR: None,
        "PRAXIS_MODEL_ID": None,
        "PRAXIS_API_KEY": None,
        **extra,
    }


# ---------------------------------------------------------------------------
# 脚手架:配置 / 透传编译 / 假 HTTP responder / ps 探针
# ---------------------------------------------------------------------------


def _write_config(servers: list[dict[str, Any]]) -> Path:
    directory = host_config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    path = directory / _HOST_CONFIG
    path.write_text(yaml.safe_dump({"servers": servers}), encoding="utf-8")
    path.chmod(0o600)
    return path


def _stdio_entry(
    name: str,
    token: str,
    *,
    extra_args: tuple[str, ...] = (),
    tools: tuple[str, ...] = ("echo",),
    allow_all: bool = False,
    enabled: bool = True,
    env: dict[str, str] | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "enabled": enabled,
        "transport": "stdio",
        "command": sys.executable if command is None else command,
        "args": ["-I", str(_FIXTURE), "--token", token, *extra_args],
        "startup_timeout_seconds": 10.0,
    }
    if allow_all:
        entry["allow_all_tools"] = True
    else:
        entry["tools_allowlist"] = list(tools)
    if env is not None:
        entry["env"] = env
    return entry


def _http_entry(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "enabled": True,
        "transport": "http",
        "url": _HTTP_URL,
        "read_only": False,
        "allow_all_tools": True,
        "startup_timeout_seconds": 10.0,
    }


def _compiler(
    _permissions: PermissionContext, _use: ResolvedUse, workspace: Path
) -> SandboxSpec:
    """Tier B:真 exec 透传前缀(不是安全边界;机制/进程回收在这里证明)。"""
    scratch = workspace / ".mcp-scratch"
    scratch.mkdir(exist_ok=True)
    return SandboxSpec(
        backend="test-passthrough",
        argv_prefix=("/usr/bin/env",),
        scratch_dir=str(scratch),
    )


async def _connect(
    workspace: Path, *, policy: PermissionPolicy | None = None
) -> McpBundle:
    return await connect_mcp_servers(
        workspace,
        policy=policy,
        sandbox_compiler=_compiler,
        env_builder=child_env,
    )


class _FakeMcpServer:
    """最小 streamable-http JSON responder(initialize / initialized / tools/list)。"""

    def __init__(self, tools: tuple[McpTool, ...]) -> None:
        self.tools = tools
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content or b"{}")
        method = body.get("method")
        if method == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "fake", "version": "1"},
                    },
                },
            )
        if method == "notifications/initialized":
            return httpx.Response(202, content=b"")
        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "tools": [
                            {"name": tool.name, "inputSchema": {"type": "object"}}
                            for tool in self.tools
                        ]
                    },
                },
            )
        return httpx.Response(
            400,
            json={"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32601, "message": "unsupported"}},
        )


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler: Any) -> None:
        self._mock = httpx.MockTransport(handler)
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._mock.handle_async_request(request)

    async def aclose(self) -> None:
        self.closed = True
        await self._mock.aclose()


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler: Any) -> _FakeTransport:
    fake = _FakeTransport(handler)
    monkeypatch.setattr(http_transport, "public_http_transport", lambda: fake)
    return fake


def _ps_lines(token: str) -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=20, check=False
    )
    return [row for row in completed.stdout.splitlines() if token in row]


async def _wait_process_gone(token: str, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while _ps_lines(token):
        if loop.time() > deadline:
            pytest.fail(f"orphan MCP server still alive: {_ps_lines(token)}")
        await asyncio.sleep(0.05)


def _owner_tasks() -> list[asyncio.Task[Any]]:
    return [task for task in asyncio.all_tasks() if task.get_name().startswith(_OWNER_PREFIX)]


def _allow_network() -> PermissionPolicy:
    return PermissionPolicy(network=NetworkPolicy(decision="allow"))


def _fake_tool(name: str, *, source: ToolSource = ToolSource.MCP) -> Tool:
    async def run(_arguments: Any, _ctx: Any) -> Any:
        raise AssertionError("merge test must not execute tools")

    return Tool(
        definition=ToolDefinition(name=name, description="fake tool", input_schema={}),
        validate_input=json_schema_input({}),
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedUse(effects=frozenset(), targets=()),
        idempotent=False,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30.0,
        max_model_output_bytes=1000,
        source=source,
    )


class _Closer:
    def __init__(self, store: Any = None) -> None:
        self.calls = 0
        self.store = store
        self.store_closed_during_close: bool | None = None

    async def aclose(self) -> None:
        self.calls += 1
        if self.store is not None:
            self.store_closed_during_close = self.store._closed


# ---------------------------------------------------------------------------
# ① eager connect:两个 server + 工具表 + 幂等收尾
# ---------------------------------------------------------------------------


def test_two_servers_land_in_bundle_and_close_idempotently(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = _token()
    _write_config([_stdio_entry("probe", token), _http_entry("remote")])
    fake_server = _FakeMcpServer((McpTool(name="fetch", inputSchema={"type": "object"}),))
    transport = _install_transport(monkeypatch, fake_server.handler)

    async def scenario() -> None:
        bundle = await _connect(workspace, policy=_allow_network())
        try:
            names = {tool.definition.name for tool in bundle.tools}
            assert names == {"mcp__probe__echo", "mcp__remote__fetch"}
            statuses = {status.server_name: status for status in bundle.statuses}
            assert statuses["probe"].connected and statuses["probe"].tool_count == 1
            assert statuses["remote"].connected and statuses["remote"].tool_count == 1
            assert _owner_tasks(), "both servers must own a live owner task"
        finally:
            # repair #2:并发 aclose 共享同一个 teardown,且第二次是 no-op。
            await asyncio.gather(bundle.aclose(), bundle.aclose())
            await bundle.aclose()
        assert not _owner_tasks()
        assert transport.closed is True
        await _wait_process_gone(token)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ② 策略预检:wildcard deny / allowlist 全 deny / HTTP network ask
# ---------------------------------------------------------------------------


def test_wildcard_deny_skips_without_spawning(workspace: Path) -> None:
    token = _token()
    _write_config([_stdio_entry("probe", token)])
    policy = PermissionPolicy(tools=(ToolRule(pattern="mcp__probe__*", decision="deny"),))

    async def scenario() -> None:
        bundle = await _connect(workspace, policy=policy)
        try:
            assert bundle.tools == ()
            status = bundle.statuses[0]
            assert status.diagnostic_code == "mcp_server_skipped"
            assert "deny" in (status.diagnostic_message or "")
            assert len(status.diagnostic_message or "") <= 512
            assert not _owner_tasks()
        finally:
            await bundle.aclose()
        assert _ps_lines(token) == []

    asyncio.run(scenario())


def test_allowlist_all_denied_skips(workspace: Path) -> None:
    token = _token()
    _write_config([_stdio_entry("probe", token, tools=("echo", "env_probe"))])
    policy = PermissionPolicy(
        tools=(
            ToolRule(pattern="mcp__probe__echo", decision="deny"),
            ToolRule(pattern="mcp__probe__env_probe", decision="deny"),
        )
    )

    async def scenario() -> None:
        bundle = await _connect(workspace, policy=policy)
        try:
            assert bundle.tools == ()
            assert bundle.statuses[0].diagnostic_code == "mcp_server_skipped"
        finally:
            await bundle.aclose()

    asyncio.run(scenario())
    assert _ps_lines(token) == []


def test_http_network_ask_is_fail_closed_before_connecting(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config([_http_entry("remote")])
    fake_server = _FakeMcpServer((McpTool(name="fetch", inputSchema={"type": "object"}),))
    _install_transport(monkeypatch, fake_server.handler)

    async def scenario() -> None:
        # 默认 NetworkPolicy.decision == "ask" ⇒ 发现预授权不通过(Q4)。
        bundle = await _connect(workspace)
        try:
            assert bundle.tools == ()
            status = bundle.statuses[0]
            assert status.diagnostic_code == "mcp_server_skipped"
            assert "network allow" in (status.diagnostic_message or "")
            assert fake_server.requests == []
            assert not _owner_tasks()
        finally:
            await bundle.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ③ 单 server 崩溃隔离 + 诊断摊平/秘密纪律
# ---------------------------------------------------------------------------


def test_single_server_crash_is_isolated(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ok_token, bad_token, secret = _token(), _token(), _secret()
    monkeypatch.setenv("PRAXIS_T19_SECRET", secret)
    _write_config(
        [
            _stdio_entry("probe", ok_token),
            _stdio_entry(
                "broken",
                bad_token,
                extra_args=("--exit-immediately",),
                env={"SECRET_MCP_KEY": "${PRAXIS_T19_SECRET}"},
            ),
        ]
    )

    async def scenario() -> None:
        bundle = await _connect(workspace)
        try:
            assert {tool.definition.name for tool in bundle.tools} == {"mcp__probe__echo"}
            statuses = {status.server_name: status for status in bundle.statuses}
            broken = statuses["broken"]
            assert broken.connected is False
            assert broken.diagnostic_code == "mcp_server_unavailable"
            assert broken.diagnostic_message
            assert len(broken.diagnostic_message) <= 512
            assert "unhandled errors in a TaskGroup" not in broken.diagnostic_message
            assert secret not in broken.diagnostic_message
            assert statuses["probe"].connected is True
        finally:
            await bundle.aclose()
        assert not _owner_tasks()
        await _wait_process_gone(ok_token)
        await _wait_process_gone(bad_token)

    asyncio.run(scenario())


def test_compiler_failure_redacts_config_secret(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """编译器异常文本携带 env 展开值时,诊断必须 redact(秘密永不进状态输出)。"""
    secret = _secret()
    monkeypatch.setenv("PRAXIS_T19_SECRET", secret)
    _write_config(
        [_stdio_entry("probe", _token(), env={"SECRET_MCP_KEY": "${PRAXIS_T19_SECRET}"})]
    )

    def exploding_compiler(
        _permissions: PermissionContext, _use: ResolvedUse, _workspace: Path
    ) -> SandboxSpec:
        raise RuntimeError(f"profile compiler exploded with {secret}")

    async def scenario() -> None:
        bundle = await connect_mcp_servers(
            workspace,
            policy=PermissionPolicy(),
            sandbox_compiler=exploding_compiler,
            env_builder=child_env,
        )
        try:
            message = bundle.statuses[0].diagnostic_message or ""
            assert bundle.statuses[0].diagnostic_code == "mcp_server_unavailable"
            assert secret not in message
            assert "[redacted]" in message
        finally:
            await bundle.aclose()

    asyncio.run(scenario())


def test_diagnostic_flattening_is_bounded_and_kind_first() -> None:
    leaves = [ValueError(f"leaf-{index}") for index in range(12)]
    group: BaseException = ExceptionGroup("outer", leaves)
    message = diagnostic_message(group, fallback="failed")
    assert message.startswith("ExceptionGroup: ValueError: leaf-0")
    assert len(message) <= 512

    wide = bounded_exception_message(ExceptionGroup("outer", leaves), fallback="failed")
    assert "leaf-0" in wide
    assert "+" in wide and "more" in wide
    assert len(wide) <= 512

    deep: BaseException = TimeoutError("timeout")
    for _ in range(8):
        deep = ExceptionGroup("nested", [deep])
    assert "TimeoutError" in diagnostic_message(deep, fallback="failed")


def test_redaction_runs_before_bounding_and_handles_whitespace() -> None:
    """repair #2:跨 512 边界/含空白的秘密都不得留下部分前缀。"""
    secret = "sk-straddle-" + uuid.uuid4().hex
    # 28 = len("RuntimeError: RuntimeError: ");让秘密跨在 512 截断边界上。
    prefix = "x" * max(0, 512 - 28 - len(secret) // 2)
    error = RuntimeError(prefix + secret + "tail")
    message = diagnostic_message(
        error, fallback="failed", redact=lambda text: mcp_manager._redact(text, (secret,))
    )
    assert "[redacted]" in message
    assert secret[:24] not in message
    assert len(message) <= 512

    spaced = "sk-whitespace  secret\nvalue"
    spaced_message = diagnostic_message(
        RuntimeError(f"boom before {spaced} after"),
        fallback="failed",
        redact=lambda text: mcp_manager._redact(text, (spaced,)),
    )
    assert "sk-whitespace" not in spaced_message
    assert "value" not in spaced_message
    assert "[redacted]" in spaced_message


def test_skip_diagnostics_are_bounded_for_long_names(workspace: Path) -> None:
    long_name = "srv-" + "a" * 240  # 落 256 上限内,但足以把 wildcard 文案推长
    _write_config([_stdio_entry(long_name, _token())])
    policy = PermissionPolicy(
        tools=(ToolRule(pattern=server_wildcard(long_name), decision="deny"),)
    )

    async def scenario() -> None:
        bundle = await _connect(workspace, policy=policy)
        try:
            message = bundle.statuses[0].diagnostic_message or ""
            assert message
            assert len(message) <= 512
        finally:
            await bundle.aclose()

    asyncio.run(scenario())


def test_overlong_server_name_rejected_at_load(workspace: Path) -> None:
    _write_config([_stdio_entry("a" * 1_000_000, _token())])
    with pytest.raises(ValueError, match="256"):
        load_mcp_configs(workspace)


def test_overlong_http_url_rejected_at_load(workspace: Path) -> None:
    _write_config(
        [
            {
                "name": "remote",
                "enabled": True,
                "transport": "http",
                "url": "https://" + "a" * 5000 + ".example/rpc",
                "tools_allowlist": ["fetch"],
            }
        ]
    )
    with pytest.raises(ValueError, match="4096"):
        load_mcp_configs(workspace)


def test_config_drift_guard_reads_live_bytes(workspace: Path) -> None:
    token = _token()
    path = _write_config([_stdio_entry("probe", token)])

    async def scenario() -> None:
        bundle = await _connect(workspace)
        try:
            tool = next(item for item in bundle.tools if item.definition.name == "mcp__probe__echo")
            path.write_text(path.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
            result = await tool.run({"message": "hi"}, _exec_ctx())
            assert result.is_error is True
            assert result.error_code == "mcp_config_changed"
        finally:
            await bundle.aclose()
        await _wait_process_gone(token)

    asyncio.run(scenario())


def _exec_ctx() -> ToolExecContext:
    return ToolExecContext()


# ---------------------------------------------------------------------------
# 组合根 merge:extra_tools 冲突边界
# ---------------------------------------------------------------------------


def test_extra_tools_collision_with_builtin_is_rejected(workspace: Path) -> None:
    with pytest.raises(ValueError, match="tool name 'read'.*built-in"):
        build_session(
            workspace,
            adapter=_scripted(),
            extra_tools=(_fake_tool("read"),),
        )


def test_extra_tools_cross_server_collision_is_rejected(workspace: Path) -> None:
    with pytest.raises(ValueError, match="another extra tool"):
        build_session(
            workspace,
            adapter=_scripted(),
            extra_tools=(_fake_tool("mcp__dup__tool"), _fake_tool("mcp__dup__tool")),
        )


def test_extra_tools_merge_into_tool_table(workspace: Path) -> None:
    session = build_session(workspace, adapter=_scripted(), extra_tools=(_fake_tool("mcp__probe__echo"),))
    try:
        assert "mcp__probe__echo" in session._tools
        assert "read" in session._tools
    finally:
        session.close()


def _scripted() -> ScriptedAdapter:
    return ScriptedAdapter([])


# ---------------------------------------------------------------------------
# Session.close 顺序:command_sessions → integrations → store
# ---------------------------------------------------------------------------


def test_session_aclose_runs_integrations_before_store(workspace: Path) -> None:
    session = build_session(workspace, adapter=_scripted())
    closer = _Closer(store=session._store)
    session._integrations = (closer,)

    asyncio.run(session.aclose())

    assert closer.calls == 1
    assert closer.store_closed_during_close is False, "integration must run while the store is alive"
    assert session._store._closed is True


def test_session_close_always_closes_store_on_baseexception(workspace: Path) -> None:
    """repair #5:closer 抛 BaseException 时 store 写者线程也绝不成孤儿。"""

    class _ExplodingCloser:
        async def aclose(self) -> None:
            raise KeyboardInterrupt

    session = build_session(workspace, adapter=_scripted())
    session._integrations = (_ExplodingCloser(),)
    with pytest.raises(KeyboardInterrupt):
        session.close()
    assert session._store._closed is True


def test_real_bundle_is_closed_by_session_and_double_close_is_safe(workspace: Path) -> None:
    token = _token()
    _write_config([_stdio_entry("probe", token)])

    async def scenario() -> None:
        bundle = await _connect(workspace)
        session = build_session(
            workspace,
            adapter=_scripted(),
            extra_tools=bundle.tools,
            integrations=(bundle,),
        )
        await session.aclose()
        await bundle.aclose()  # Session 已关闭;幂等
        assert not _owner_tasks()
        await _wait_process_gone(token)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# CLI roundtrips(临时 PRAXIS_CONFIG_DIR;绝不碰真实配置)
# ---------------------------------------------------------------------------


def test_mcp_cli_add_list_remove_roundtrip(workspace: Path, tmp_path: Path) -> None:
    secret = _secret()
    env = _env(tmp_path, TOKEN_VAR=secret)
    added = runner.invoke(
        app,
        [
            "mcp", "add", "stdio", "probe",
            "--command", sys.executable,
            "--args=-I", "--args", str(_FIXTURE),
            "--tool", "echo",
            "--env", "SECRET_MCP_KEY=${TOKEN_VAR}",
            "--workspace", str(workspace),
        ],
        env=env,
    )
    assert added.exit_code == 0, added.output

    http_added = runner.invoke(
        app,
        [
            "mcp", "add", "http", "remote",
            "--url", _HTTP_URL,
            "--bearer-env", "MCP_TOKEN",
            "--allow-all",
            "--workspace", str(workspace),
        ],
        env=env,
    )
    assert http_added.exit_code == 0, http_added.output

    disabled = runner.invoke(
        app,
        [
            "mcp", "add", "stdio", "offline",
            "--command", sys.executable,
            "--tool", "echo",
            "--disabled",
            "--workspace", str(workspace),
        ],
        env=env,
    )
    assert disabled.exit_code == 0, disabled.output

    listed = runner.invoke(app, ["mcp", "list", "--workspace", str(workspace)], env=env)
    assert listed.exit_code == 0, listed.output
    assert "trust: user" in listed.output
    assert "enabled" in listed.output and "disabled" in listed.output
    for name in ("probe", "remote", "offline"):
        assert name in listed.output
    assert "SECRET_MCP_KEY" in listed.output
    assert secret not in listed.output, "list must never print env values"

    shown = runner.invoke(app, ["mcp", "show", "remote", "--workspace", str(workspace)], env=env)
    assert shown.exit_code == 0, shown.output
    assert _HTTP_URL in shown.output
    assert "MCP_TOKEN" in shown.output

    removed = runner.invoke(app, ["mcp", "remove", "probe", "--workspace", str(workspace)], env=env)
    assert removed.exit_code == 0, removed.output
    relisted = runner.invoke(app, ["mcp", "list", "--workspace", str(workspace)], env=env)
    assert "probe" not in relisted.output

    missing = runner.invoke(app, ["mcp", "remove", "probe", "--workspace", str(workspace)], env=env)
    assert missing.exit_code == 1, missing.output
    assert "not configured" in missing.output


def test_mcp_cli_add_rejects_literal_env_value(workspace: Path, tmp_path: Path) -> None:
    """R-T19-1:字面值不得进 YAML(也不得进 shell history)。"""
    result = runner.invoke(
        app,
        [
            "mcp", "add", "stdio", "probe",
            "--command", sys.executable,
            "--tool", "echo",
            "--env", "SECRET_MCP_KEY=supersecret",
            "--workspace", str(workspace),
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 3, result.output
    assert "literal secret" in result.output
    assert not (host_config_dir() / _HOST_CONFIG).exists()


def test_mcp_cli_add_http_requires_exactly_one_credential_source(
    workspace: Path, tmp_path: Path
) -> None:
    no_credential = runner.invoke(
        app,
        ["mcp", "add", "http", "remote", "--url", _HTTP_URL, "--allow-all", "--workspace", str(workspace)],
        env=_env(tmp_path),
    )
    assert no_credential.exit_code == 3, no_credential.output
    both = runner.invoke(
        app,
        [
            "mcp", "add", "http", "remote", "--url", _HTTP_URL, "--allow-all",
            "--bearer-env", "TOKEN", "--keychain", "--workspace", str(workspace),
        ],
        env=_env(tmp_path),
    )
    assert both.exit_code == 3, both.output


def test_mcp_cli_add_rejects_normalized_name_collision(workspace: Path, tmp_path: Path) -> None:
    env = _env(tmp_path)
    first = runner.invoke(
        app,
        ["mcp", "add", "stdio", "a-b", "--command", "/bin/echo", "--tool", "echo", "--workspace", str(workspace)],
        env=env,
    )
    assert first.exit_code == 0, first.output
    second = runner.invoke(
        app,
        ["mcp", "add", "stdio", "a_b", "--command", "/bin/echo", "--tool", "echo", "--workspace", str(workspace)],
        env=env,
    )
    assert second.exit_code == 3, second.output
    assert "collides" in second.output


def test_mcp_cli_login_http_saves_via_keychain_stub(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    added = runner.invoke(
        app,
        [
            "mcp", "add", "http", "remote", "--url", _HTTP_URL,
            "--bearer-env", "MCP_TOKEN", "--allow-all", "--workspace", str(workspace),
        ],
        env=env,
    )
    assert added.exit_code == 0, added.output
    captured: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        mcp_cli, "save_mcp_bearer", lambda name, url, value: captured.append((name, url, value))
    )
    result = runner.invoke(
        app, ["mcp", "login", "remote", "--workspace", str(workspace)], input="s3cret\n", env=env
    )
    assert result.exit_code == 0, result.output
    assert captured == [("remote", _HTTP_URL, "s3cret")]
    assert "s3cret" not in result.output


def test_mcp_cli_login_stdio_is_refused(workspace: Path, tmp_path: Path) -> None:
    env = _env(tmp_path)
    added = runner.invoke(
        app,
        ["mcp", "add", "stdio", "probe", "--command", sys.executable, "--tool", "echo", "--workspace", str(workspace)],
        env=env,
    )
    assert added.exit_code == 0, added.output
    result = runner.invoke(app, ["mcp", "login", "probe", "--workspace", str(workspace)], env=env)
    assert result.exit_code == 1, result.output
    assert "stdio server" in result.output


def test_mcp_cli_login_refuses_oauth_entries(workspace: Path, tmp_path: Path) -> None:
    _write_config(
        [
            {
                "name": "remote",
                "enabled": True,
                "transport": "http",
                "url": _HTTP_URL,
                "oauth": True,
                "allow_all_tools": True,
            }
        ]
    )
    result = runner.invoke(
        app, ["mcp", "login", "remote", "--workspace", str(workspace)], env=_env(tmp_path)
    )
    assert result.exit_code == 1, result.output
    assert "MCP OAuth lands in phase 5" in result.output


def test_mcp_cli_status_never_connects(workspace: Path, tmp_path: Path) -> None:
    marker = tmp_path / "spawned"
    _write_config(
        [
            {
                "name": "probe",
                "enabled": True,
                "transport": "stdio",
                "command": "/bin/sh",
                "args": ["-c", f"touch {marker}"],
                "tools_allowlist": ["echo"],
            }
        ]
    )
    result = runner.invoke(
        app, ["mcp", "status", "--workspace", str(workspace)], env=_env(tmp_path)
    )
    assert result.exit_code == 0, result.output
    assert "probe" in result.output
    assert "not attempted" in result.output
    assert not marker.exists(), "status is configuration-side only; it must not spawn servers"


def test_mcp_cli_status_without_config_is_clean(tmp_path: Path) -> None:
    result = runner.invoke(app, ["mcp", "status", "--workspace", str(tmp_path)], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "none" in result.output


def test_mcp_cli_list_invalid_yaml_exits_1(tmp_path: Path) -> None:
    path = host_config_dir() / _HOST_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text("servers: [unclosed\n", encoding="utf-8")
    path.chmod(0o600)
    result = runner.invoke(app, ["mcp", "list", "--workspace", str(tmp_path)], env=_env(tmp_path))
    assert result.exit_code == 1, result.output
    assert "valid YAML" in result.output


def test_sdk_reexports_mcp_surface() -> None:
    for name in (
        "McpBundle",
        "McpServerStatus",
        "connect_mcp_servers",
        "resolve_declared_policy",
        "load_mcp_configs",
        "resolve_mcp_config_path",
        "decide_mcp_config_trust",
        "mcp_config_write_path",
        "normalize_mcp_name",
        "save_mcp_bearer",
        "HttpServerConfig",
        "StdioServerConfig",
        "ToolSource",
        "SkillCatalog",
        "load_host_hooks",
    ):
        assert name in sdk_pkg.__all__, f"sdk.__all__ missing {name}"
        assert hasattr(sdk_pkg, name), f"sdk missing attribute {name}"


# ---------------------------------------------------------------------------
# 无 MCP:既有同步路径零行为变化
# ---------------------------------------------------------------------------


def test_run_without_mcp_never_connects(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "done.json"
    script.write_text(json.dumps([{"content": "plain"}]), encoding="utf-8")

    async def explode(*_args: Any, **_kwargs: Any) -> McpBundle:
        raise AssertionError("connect_mcp_servers must not run without MCP configuration")

    monkeypatch.setattr(cli_main, "connect_mcp_servers", explode)
    result = runner.invoke(
        app,
        ["run", "hi", "--workspace", str(workspace), "--scripted", str(script)],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    assert "plain" in result.output


# ---------------------------------------------------------------------------
# Q9 单 loop:带 MCP 工具提交 turn(ask ⇒ 暂停;结束后无孤儿)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _AVAILABLE, reason="MCP stdio needs a real sandbox backend")
def test_single_loop_run_pauses_on_mcp_tool_ask(workspace: Path, tmp_path: Path) -> None:
    token = _token()
    env = _env(tmp_path, TOKEN_VAR="unused")
    _write_config([_stdio_entry("probe", token)])
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"tools": [{"pattern": "mcp__probe__echo", "decision": "ask"}]}),
        encoding="utf-8",
    )
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            [
                {
                    "content": "",
                    "tool_calls": [
                        {"call_id": "c1", "name": "mcp__probe__echo", "arguments": {"message": "hi"}}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "run", "hi",
            "--workspace", str(workspace),
            "--scripted", str(script),
            "--policy", str(policy),
        ],
        env=env,
    )
    assert result.exit_code == 2, result.output
    assert "approval request" in result.output
    assert "mcp__probe__echo" in result.output
    # 单 loop 收尾后进程组必须消失(3A/3B 纪律)。
    for _ in range(200):
        if not _ps_lines(token):
            break
        time.sleep(0.05)
    assert _ps_lines(token) == []


def test_chat_input_uses_daemon_thread_not_default_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-T19-2:输入线程必须 daemon 且不占默认 executor(否则 Ctrl+C 挂住)。"""
    source = inspect.getsource(cli_main._read_answer_async)
    assert "await asyncio.to_thread" not in source
    assert "loop.create_future()" in source
    assert "daemon=True" in source
    assert "to_thread" not in inspect.getsource(cli_main._chat_loop_async)

    monkeypatch.setattr(cli_main, "_read_answer", lambda _prompt: "line")
    assert asyncio.run(cli_main._read_answer_async("> ")) == "line"
    assert not any(
        thread.name == "praxis-chat-input" and not thread.daemon
        for thread in threading.enumerate()
    )


@pytest.mark.skipif(not _AVAILABLE, reason="MCP stdio needs a real sandbox backend")
def test_single_loop_chat_keeps_mcp_owner_alive(workspace: Path, tmp_path: Path) -> None:
    """chat 的整个 REPL 也活在一个 loop 里:submit 在 owner task 存续期间完成。"""
    token = _token()
    _write_config([_stdio_entry("probe", token)])
    script = tmp_path / "chat.json"
    script.write_text(json.dumps([{"content": "hello from mcp session"}]), encoding="utf-8")
    result = runner.invoke(
        app,
        ["chat", "--workspace", str(workspace), "--scripted", str(script)],
        input="hi\n/exit\n",
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    assert "hello from mcp session" in result.output
    for _ in range(200):
        if not _ps_lines(token):
            break
        time.sleep(0.05)
    assert _ps_lines(token) == []

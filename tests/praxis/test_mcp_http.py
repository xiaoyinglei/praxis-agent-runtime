"""Task 18:MCP HTTP 传输守卫 + bearer 凭据纪律 + owner-task 复用(3B §5.10)。

**证据分层**:
- 传输守卫矩阵在 ``httpx.MockTransport`` 上逐条验证(无网):endpoint 逐字
  相等(path/query/大小写)、Host/netloc 一致、300-399 全拒(不分方法)、
  401/403 有界映射、秘密不进入任何错误文案。
- ``owned_http_connection`` 用**假 streamable-http responder** 走真实 mcp SDK
  的 ``streamable_http_client`` + ``ClientSession`` 握手(initialize /
  notifications-initialized / tools/list),生产 connect 闭包只替换最内层传输。
- 凭据面:POSIX 0600 文件、darwin ``security -i`` stdin 假实现、URL 绑定
  逐字校验、env→store 回落链;全部断言值不进 argv/错误消息。
- owner task 与 stdio 共用 ``lifecycle.owned_connection`` 唯一骨架(源码级
  grep + 取消/跨 task 关闭行为)。

被裁决删除的旧面:**传输层 destination_check 不再移植**(R-T18-1)。HTTP MCP
工具携带 ``url`` ToolTarget,每次 execute 由单一 ``decide`` 以 live policy
裁决;连接期 ``network_verdict_at_connect`` 只是发现预授权快照(Q4)。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from mcp.types import ToolAnnotations
from mcp.types import Tool as McpTool

from praxis.adapters.config import credentials
from praxis.adapters.config.credentials import (
    clear_mcp_bearer,
    load_mcp_bearer,
    mcp_credential_service,
    save_mcp_bearer,
)
from praxis.adapters.mcp import http_transport, lifecycle
from praxis.adapters.mcp.config import HttpServerConfig, load_mcp_configs
from praxis.adapters.mcp.descriptors import mcp_tools_for_connection
from praxis.adapters.mcp.http_transport import (
    GuardedMcpTransport,
    McpTransportError,
    credential_revision,
    owned_http_connection,
)
from praxis.adapters.mcp.lifecycle import McpConnection
from praxis.protocol.floor import host_config_dir
from praxis.protocol.tools import ToolEffect, ToolTarget

_ENDPOINT = "https://mcp.example.com/rpc"
_OTHER_ENDPOINT = "https://mcp.example.com/other"
_ENV_VAR = "PRAXIS_TEST_MCP_TOKEN"
_OWNER_PREFIX = "mcp-owner:"


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def _http_config(**overrides: Any) -> HttpServerConfig:
    values: dict[str, Any] = dict(
        name="remote",
        url=_ENDPOINT,
        headers={},
        bearer_token_env_var=None,
        keychain=False,
        read_only=False,
        tools_allowlist=frozenset(),
        allow_all_tools=True,
        startup_timeout_seconds=5.0,
    )
    values.update(overrides)
    return HttpServerConfig(**values)


def _force_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "_PLATFORM", "linux")


def _credential_file() -> Path:
    return host_config_dir() / "credentials" / "models.json"


class _FakeMcpServer:
    """最小 streamable-http JSON responder:initialize / initialized / tools/list。"""

    def __init__(self, tools: tuple[McpTool, ...] = ()) -> None:
        self.tools = tools
        self.requests: list[httpx.Request] = []
        self.authorizations: list[str | None] = []
        self.entered = asyncio.Event()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.authorizations.append(request.headers.get("authorization"))
        self.entered.set()
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
            json={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32601, "message": "unsupported"},
            },
        )


class _FakeTransport(httpx.AsyncBaseTransport):
    """可观测关闭的 MockTransport 包装(验证 client/owner 退出确实关栈)。"""

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


def _owner_tasks() -> list[asyncio.Task[Any]]:
    return [task for task in asyncio.all_tasks() if task.get_name().startswith(_OWNER_PREFIX)]


async def _wait_owner(name: str, timeout: float = 5.0) -> asyncio.Task[Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        for task in asyncio.all_tasks():
            if task.get_name() == name:
                return task
        if loop.time() > deadline:
            pytest.fail(f"owner task {name} never started")
        await asyncio.sleep(0.01)


def _flatten(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for sub in error.exceptions for leaf in _flatten(sub)]
    return [error]


def _allow(_url: str) -> str:
    return "allow"


# ---------------------------------------------------------------------------
# 凭据层:service 命名、POSIX 0600、URL 绑定、darwin stdin、win32
# ---------------------------------------------------------------------------


def test_mcp_service_is_server_name_addressable() -> None:
    assert mcp_credential_service("Remote-Server") == "praxis.mcp.remote_server"
    assert mcp_credential_service("remote") == "praxis.mcp.remote"
    with pytest.raises(ValueError, match="non-empty server name"):
        mcp_credential_service("!!!---")


def test_posix_payload_shape_and_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_posix(monkeypatch)
    save_mcp_bearer("remote", _ENDPOINT, "  sk-mcp-secret  ")
    payload = json.loads(_credential_file().read_text(encoding="utf-8"))
    expected = json.dumps({"url": _ENDPOINT, "bearer": "sk-mcp-secret"}, sort_keys=True)
    assert payload == {"version": 1, "keys": {"praxis.mcp.remote": expected}}
    assert _credential_file().stat().st_mode & 0o777 == 0o600

    assert load_mcp_bearer("remote", _ENDPOINT) == "sk-mcp-secret"
    clear_mcp_bearer("remote", _ENDPOINT)
    assert load_mcp_bearer("remote", _ENDPOINT) is None
    clear_mcp_bearer("remote", _ENDPOINT)  # no-op


def test_binding_mismatch_is_bounded_and_hides_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_posix(monkeypatch)
    secret = "sk-mcp-super-secret"
    save_mcp_bearer("remote", _ENDPOINT, secret)
    with pytest.raises(ValueError) as excinfo:
        load_mcp_bearer("remote", _OTHER_ENDPOINT)
    assert str(excinfo.value) == "MCP credential is bound to a different endpoint"
    assert secret not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_binding_is_byte_exact_after_httpx_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_posix(monkeypatch)
    # 保存侧与 T15 配置同源:host 小写 + 省略默认 443。
    save_mcp_bearer("remote", "https://MCP.Example.com:443/rpc", "token")
    assert load_mcp_bearer("remote", _ENDPOINT) == "token"
    # 尾斜杠不同 ⇒ 逐字绑定拒绝(不做 rstrip)。
    with pytest.raises(ValueError, match="bound to a different endpoint"):
        load_mcp_bearer("remote", _ENDPOINT + "/")
    with pytest.raises(ValueError, match="must be a valid URL"):
        save_mcp_bearer("remote", "not a url", "token")


def test_save_rejects_empty_or_control_values_without_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_posix(monkeypatch)
    for bad in ("", "   ", "sk\nsecret", "sk\x00secret"):
        with pytest.raises(ValueError) as excinfo:
            save_mcp_bearer("remote", _ENDPOINT, bad)
        assert str(excinfo.value) == "MCP bearer token must be a nonempty single-line value"
        assert "sk" not in str(excinfo.value)


def test_win32_unsupported_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "_PLATFORM", "win32")
    for call in (
        lambda: save_mcp_bearer("remote", _ENDPOINT, "token"),
        lambda: load_mcp_bearer("remote", _ENDPOINT),
        lambda: clear_mcp_bearer("remote", _ENDPOINT),
    ):
        with pytest.raises(ValueError, match="require macOS or POSIX"):
            call()


class _FakeRun:
    def __init__(self, responses: list[tuple[int, str, str]]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), dict(kwargs)))
        returncode, stdout, stderr = self._responses.pop(0)
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def _inject_run(
    monkeypatch: pytest.MonkeyPatch, responses: list[tuple[int, str, str]]
) -> _FakeRun:
    fake = _FakeRun(responses)
    monkeypatch.setattr(credentials.subprocess, "run", fake)
    return fake


def _darwin_payload(secret: str = "sk-mcp-secret") -> str:
    return json.dumps({"url": _ENDPOINT, "bearer": secret}, sort_keys=True)


def test_darwin_save_keeps_value_on_stdin_and_bound_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credentials, "_PLATFORM", "darwin")
    payload = _darwin_payload()
    fake = _inject_run(monkeypatch, [(0, "", ""), (0, payload, "")])

    save_mcp_bearer("remote", _ENDPOINT, "sk-mcp-secret")
    save_argv, save_kwargs = fake.calls[0]
    assert save_argv == ["/usr/bin/security", "-i"]
    assert "sk-mcp-secret" not in " ".join(save_argv)
    command = str(save_kwargs["input"])
    assert "add-generic-password" in command
    assert "praxis.mcp.remote" in command

    read_argv, _ = fake.calls[1]
    assert read_argv[:3] == ["/usr/bin/security", "find-generic-password", "-a"]
    assert "sk-mcp-secret" not in " ".join(read_argv)


def test_darwin_load_returns_bound_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "_PLATFORM", "darwin")
    _inject_run(monkeypatch, [(0, _darwin_payload(), "")])
    assert load_mcp_bearer("remote", _ENDPOINT) == "sk-mcp-secret"


# ---------------------------------------------------------------------------
# 传输守卫矩阵
# ---------------------------------------------------------------------------


async def _send_through_guard(
    endpoint: str,
    url: str,
    *,
    method: str = "POST",
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    def handler(request: httpx.Request) -> httpx.Response:
        extra = {"location": "https://evil.example.com/next"} if 300 <= status < 400 else {}
        return httpx.Response(status, headers=extra, content=b"ok")

    transport = GuardedMcpTransport(endpoint, httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        return await client.request(method, url, headers=headers)


def test_guard_accepts_exact_endpoint() -> None:
    response = asyncio.run(_send_through_guard(_ENDPOINT, _ENDPOINT))
    assert response.status_code == 200


@pytest.mark.parametrize(
    "url",
    [
        _ENDPOINT + "/",
        _ENDPOINT + "?x=1",
        _ENDPOINT.replace("/rpc", "/RPC"),
        _OTHER_ENDPOINT,
    ],
)
def test_guard_rejects_endpoint_drift(url: str) -> None:
    with pytest.raises(McpTransportError) as excinfo:
        asyncio.run(_send_through_guard(_ENDPOINT, url))
    assert excinfo.value.code == "mcp_http_endpoint_changed"
    assert str(excinfo.value) == "MCP request cannot change its configured endpoint"


def test_guard_rejects_host_authority_mismatch() -> None:
    with pytest.raises(McpTransportError) as excinfo:
        asyncio.run(
            _send_through_guard(_ENDPOINT, _ENDPOINT, headers={"Host": "evil.example.com"})
        )
    assert excinfo.value.code == "mcp_http_host_mismatch"


@pytest.mark.parametrize("status", [301, 302, 307, 308])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_guard_rejects_every_3xx_regardless_of_method(status: int, method: str) -> None:
    with pytest.raises(McpTransportError) as excinfo:
        asyncio.run(_send_through_guard(_ENDPOINT, _ENDPOINT, method=method, status=status))
    assert excinfo.value.code == "mcp_http_redirect"
    assert "redirects are refused" in str(excinfo.value)


@pytest.mark.parametrize("status", [401, 403])
def test_guard_maps_401_403_to_bounded_permission_error(status: int) -> None:
    with pytest.raises(McpTransportError) as excinfo:
        asyncio.run(_send_through_guard(_ENDPOINT, _ENDPOINT, status=status))
    assert excinfo.value.code == "mcp_http_unauthorized"
    assert str(excinfo.value) == (
        f"remote MCP authentication or authorization failed (HTTP {status})"
    )
    assert isinstance(excinfo.value, PermissionError)


def test_guard_rejection_never_echoes_the_bearer() -> None:
    secret = "sk-mcp-do-not-leak"
    headers = {"Authorization": f"Bearer {secret}"}
    for url, status in ((_OTHER_ENDPOINT, 200), (_ENDPOINT, 401)):
        with pytest.raises(McpTransportError) as excinfo:
            asyncio.run(_send_through_guard(_ENDPOINT, url, status=status, headers=headers))
        assert secret not in str(excinfo.value)
        assert secret not in repr(excinfo.value)


# ---------------------------------------------------------------------------
# owned_http_connection:连接期闸门 + 真实 SDK 握手 + 凭据链
# ---------------------------------------------------------------------------


def test_connect_gate_ask_and_deny_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _FakeMcpServer()
    fake = _install_transport(monkeypatch, server.handler)

    async def scenario(verdict: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            async with owned_http_connection(
                _http_config(),
                workspace=Path("/tmp/unused"),
                network_verdict_at_connect=lambda _url: verdict,
            ):
                pytest.fail("not-authorized discovery must not connect")
        message = str(excinfo.value)
        assert "mcp http server requires network allow for mcp.example.com" in message
        assert "Q4 discovery preauthorization parity" in message
        assert _owner_tasks() == []

    asyncio.run(scenario("ask"))
    asyncio.run(scenario("deny"))
    assert server.requests == []
    assert fake.closed is False


def test_connect_gate_allow_handshakes_and_lists_full_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = (_mcp_tool("alpha"), _mcp_tool("beta"))
    server = _FakeMcpServer(tools)
    fake = _install_transport(monkeypatch, server.handler)

    async def scenario() -> None:
        config = _http_config()
        async with owned_http_connection(
            config, workspace=Path("/tmp/unused"), network_verdict_at_connect=_allow
        ) as connection:
            assert isinstance(connection, McpConnection)
            assert connection.server_name == "remote"
            assert tuple(tool.name for tool in connection.tools) == ("alpha", "beta")
            assert connection.credential_revision == credential_revision(
                config.headers, bearer_present=False
            )
            assert [task.get_name() for task in _owner_tasks()] == ["mcp-owner:remote"]
        assert _owner_tasks() == []
        assert fake.closed is True

    asyncio.run(scenario())


def _mcp_tool(
    name: str,
    *,
    read_only: bool | None = None,
    destructive: bool | None = None,
) -> McpTool:
    annotations = ToolAnnotations(
        readOnlyHint=read_only, destructiveHint=destructive, idempotentHint=False
    )
    return McpTool(
        name=name,
        description=name,
        inputSchema={"type": "object", "properties": {}},
        annotations=annotations,
    )


def test_bearer_env_token_is_sent_and_never_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "sk-mcp-env-secret"
    monkeypatch.setenv(_ENV_VAR, secret)
    server = _FakeMcpServer((_mcp_tool("alpha"),))
    _install_transport(monkeypatch, server.handler)

    async def scenario() -> None:
        config = _http_config(bearer_token_env_var=_ENV_VAR)
        async with owned_http_connection(
            config, workspace=Path("/tmp/unused"), network_verdict_at_connect=_allow
        ) as connection:
            assert connection.credential_revision == credential_revision(
                config.headers, bearer_present=True
            )
        assert server.authorizations
        assert set(server.authorizations) == {f"Bearer {secret}"}
        assert secret not in caplog.text

    with caplog.at_level("DEBUG"):
        asyncio.run(scenario())


def test_no_bearer_keeps_configured_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _FakeMcpServer()
    _install_transport(monkeypatch, server.handler)

    async def scenario() -> None:
        config = _http_config(headers={"Authorization": "Bearer header-source"})
        async with owned_http_connection(
            config, workspace=Path("/tmp/unused"), network_verdict_at_connect=_allow
        ):
            pass
        assert set(server.authorizations) == {"Bearer header-source"}

    asyncio.run(scenario())


def test_bearer_overrides_configured_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ENV_VAR, "sk-bearer-wins")
    server = _FakeMcpServer()
    _install_transport(monkeypatch, server.handler)

    async def scenario() -> None:
        config = _http_config(
            headers={"authorization": "Bearer stale"}, bearer_token_env_var=_ENV_VAR
        )
        async with owned_http_connection(
            config, workspace=Path("/tmp/unused"), network_verdict_at_connect=_allow
        ):
            pass
        assert set(server.authorizations) == {"Bearer sk-bearer-wins"}

    asyncio.run(scenario())


def test_bearer_env_missing_falls_back_to_stored_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_posix(monkeypatch)
    monkeypatch.delenv(_ENV_VAR, raising=False)
    save_mcp_bearer("remote", _ENDPOINT, "sk-stored")
    server = _FakeMcpServer()
    _install_transport(monkeypatch, server.handler)

    async def scenario() -> None:
        config = _http_config(bearer_token_env_var=_ENV_VAR)
        async with owned_http_connection(
            config, workspace=Path("/tmp/unused"), network_verdict_at_connect=_allow
        ):
            pass
        assert server.authorizations
        assert set(server.authorizations) == {"Bearer sk-stored"}

    asyncio.run(scenario())


def test_bearer_env_missing_everywhere_names_only_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_posix(monkeypatch)
    monkeypatch.delenv(_ENV_VAR, raising=False)

    async def scenario() -> None:
        with pytest.raises(ValueError) as excinfo:
            async with owned_http_connection(
                _http_config(bearer_token_env_var=_ENV_VAR),
                workspace=Path("/tmp/unused"),
                network_verdict_at_connect=_allow,
            ):
                pytest.fail("missing credential must fail before connecting")
        assert str(excinfo.value) == (
            "missing MCP credential environment variable: PRAXIS_TEST_MCP_TOKEN"
        )

    asyncio.run(scenario())


def test_keychain_flag_uses_stored_bearer_and_missing_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_posix(monkeypatch)
    server = _FakeMcpServer()
    _install_transport(monkeypatch, server.handler)

    async def missing() -> None:
        with pytest.raises(ValueError, match="MCP credential missing for remote"):
            async with owned_http_connection(
                _http_config(keychain=True),
                workspace=Path("/tmp/unused"),
                network_verdict_at_connect=_allow,
            ):
                pytest.fail("missing keychain bearer must fail before connecting")

    asyncio.run(missing())

    save_mcp_bearer("remote", _ENDPOINT, "sk-keychain")

    async def present() -> None:
        async with owned_http_connection(
            _http_config(keychain=True),
            workspace=Path("/tmp/unused"),
            network_verdict_at_connect=_allow,
        ):
            pass

    asyncio.run(present())
    assert server.authorizations
    assert set(server.authorizations) == {"Bearer sk-keychain"}


def test_keychain_darwin_stub_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "_PLATFORM", "darwin")
    _inject_run(monkeypatch, [(0, _darwin_payload("sk-darwin"), "")])
    server = _FakeMcpServer()
    _install_transport(monkeypatch, server.handler)

    async def scenario() -> None:
        async with owned_http_connection(
            _http_config(keychain=True),
            workspace=Path("/tmp/unused"),
            network_verdict_at_connect=_allow,
        ):
            pass
        assert server.authorizations
        assert set(server.authorizations) == {"Bearer sk-darwin"}

    asyncio.run(scenario())


def test_binding_mismatch_prevents_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_posix(monkeypatch)
    save_mcp_bearer("remote", _OTHER_ENDPOINT, "sk-wrong-endpoint")
    server = _FakeMcpServer()
    _install_transport(monkeypatch, server.handler)

    async def scenario() -> None:
        with pytest.raises(ValueError, match="bound to a different endpoint") as excinfo:
            async with owned_http_connection(
                _http_config(keychain=True),
                workspace=Path("/tmp/unused"),
                network_verdict_at_connect=_allow,
            ):
                pytest.fail("bound credential must not connect")
        assert "sk-wrong-endpoint" not in str(excinfo.value)

    asyncio.run(scenario())
    assert server.requests == []


def test_unauthorized_handshake_error_never_leaks_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-mcp-401-secret"
    monkeypatch.setenv(_ENV_VAR, secret)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"no")

    _install_transport(monkeypatch, handler)

    async def scenario() -> None:
        with pytest.raises(Exception) as excinfo:
            async with owned_http_connection(
                _http_config(bearer_token_env_var=_ENV_VAR),
                workspace=Path("/tmp/unused"),
                network_verdict_at_connect=_allow,
            ):
                pytest.fail("HTTP 401 must not become ready")
        leaves = _flatten(excinfo.value)
        rendered = "\n".join(f"{type(leaf).__name__}: {leaf}" for leaf in leaves)
        assert secret not in rendered
        assert any(
            isinstance(leaf, McpTransportError) and leaf.code == "mcp_http_unauthorized"
            for leaf in leaves
        )
        assert _owner_tasks() == []

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# credential_revision:稳定 + 形状变化检测(presence-only,裁决口径)
# ---------------------------------------------------------------------------


def test_credential_revision_is_stable_and_presence_based(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert credential_revision({}, bearer_present=False) == credential_revision(
        {}, bearer_present=False
    )
    assert credential_revision({"X-Readonly": "true"}, bearer_present=False) != (
        credential_revision({"X-Readonly": "false"}, bearer_present=False)
    )
    assert credential_revision({}, bearer_present=True) != credential_revision(
        {}, bearer_present=False
    )


def test_credential_revision_detects_bearer_appearing_on_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _FakeMcpServer()
    _install_transport(monkeypatch, server.handler)
    monkeypatch.delenv(_ENV_VAR, raising=False)

    async def scenario() -> str:
        config = _http_config()
        async with owned_http_connection(
            config, workspace=Path("/tmp/unused"), network_verdict_at_connect=_allow
        ) as connection:
            assert connection.credential_revision is not None
            return connection.credential_revision

    without_bearer = asyncio.run(scenario())
    monkeypatch.setenv(_ENV_VAR, "sk-token")

    async def scenario_with_bearer() -> str:
        config = _http_config(bearer_token_env_var=_ENV_VAR)
        async with owned_http_connection(
            config, workspace=Path("/tmp/unused"), network_verdict_at_connect=_allow
        ) as connection:
            assert connection.credential_revision is not None
            return connection.credential_revision

    with_bearer = asyncio.run(scenario_with_bearer())
    assert without_bearer != with_bearer


# ---------------------------------------------------------------------------
# owner-task 复用与关闭/取消安全
# ---------------------------------------------------------------------------


def test_owner_skeleton_has_a_single_implementation() -> None:
    lifecycle_source = Path(lifecycle.__file__).read_text(encoding="utf-8")
    http_source = Path(http_transport.__file__).read_text(encoding="utf-8")
    assert lifecycle_source.count('asyncio.create_task(serve()') == 1
    assert lifecycle_source.count('name=f"mcp-owner:') == 1
    assert "asyncio.create_task" not in http_source
    assert "owned_connection(" in http_source


def test_close_from_a_different_task_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _FakeMcpServer()
    fake = _install_transport(monkeypatch, server.handler)

    async def scenario() -> None:
        stack = AsyncExitStack()
        connection = await stack.enter_async_context(
            owned_http_connection(
                _http_config(), workspace=Path("/tmp/unused"), network_verdict_at_connect=_allow
            )
        )
        assert connection.server_name == "remote"
        await asyncio.wait_for(asyncio.create_task(stack.aclose(), name="closer"), 10.0)
        assert _owner_tasks() == []
        assert fake.closed is True

    asyncio.run(scenario())


def test_caller_cancel_before_ready_reaps_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    async def blocking_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    fake = _install_transport(monkeypatch, blocking_handler)

    async def scenario() -> None:
        async def opener() -> None:
            async with owned_http_connection(
                _http_config(), workspace=Path("/tmp/unused"), network_verdict_at_connect=_allow
            ):
                pytest.fail("must not become ready while the handshake hangs")

        task = asyncio.create_task(opener(), name="opener")
        await _wait_owner("mcp-owner:remote")
        task.cancel()
        outcomes = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(outcomes[0], asyncio.CancelledError)
        assert _owner_tasks() == []
        assert fake.closed is True

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# T17 read_only 过滤经 HTTP descriptor 端到端
# ---------------------------------------------------------------------------


def test_http_read_only_filters_mutating_tools_and_keeps_url_target() -> None:
    catalog = (
        _mcp_tool("read_tool", read_only=True),
        _mcp_tool("mutate_tool", read_only=False, destructive=True),
    )
    connection = McpConnection(server_name="remote", session=object(), tools=catalog)  # type: ignore[arg-type]

    filtered = mcp_tools_for_connection(
        connection, _http_config(read_only=True), config_sha_provider=lambda: "sha"
    )
    assert [tool.definition.name for tool in filtered] == ["mcp__remote__read_tool"]
    only = filtered[0]
    assert only.static_effects == frozenset({ToolEffect.NETWORK})
    assert only.source.value == "mcp"
    use = only.resolve_use({})
    assert ToolTarget(kind="url", value=_ENDPOINT) in use.targets
    assert ToolTarget(kind="mcp_tool", value="mcp__remote__read_tool") in use.targets

    unfiltered = mcp_tools_for_connection(
        connection, _http_config(read_only=False), config_sha_provider=lambda: "sha"
    )
    assert [tool.definition.name for tool in unfiltered] == [
        "mcp__remote__read_tool",
        "mcp__remote__mutate_tool",
    ]


# ---------------------------------------------------------------------------
# T15 交接:oauth 键在加载期被同一文案拒绝(Q6)
# ---------------------------------------------------------------------------


def test_oauth_config_roundtrip_is_rejected_at_load(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = tmp_path / "mcp_servers.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "servers": [
                    {
                        "name": "remote",
                        "transport": "http",
                        "url": _ENDPOINT,
                        "enabled": True,
                        "allow_all_tools": True,
                        "oauth": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="MCP OAuth lands in phase 5"):
        load_mcp_configs(workspace, config_path)


if sys.platform == "win32":  # pragma: no cover - POSIX 文件分支的守卫
    pytest.skip("MCP bearer POSIX file tests require a POSIX platform", allow_module_level=True)
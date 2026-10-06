from __future__ import annotations

import asyncio
import json

import click
import httpcore
import pytest
import typer.rich_utils
from typer.testing import CliRunner

from agent_runtime import Agent
from agent_runtime.cli import agent_app
from agent_runtime.harness import RolloutStore
from agent_runtime.tools.web_http import PublicWebClient, PublicWebError
from tests.agent.test_web_http import _NumericBackend, _SocketStream
from tests.agent.test_web_product import FetchThenAnswer


@pytest.fixture
def clean_proxy_env(monkeypatch):
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy",
                 "NO_PROXY", "no_proxy", "PRAXIS_WEB_PROXY"):
        monkeypatch.delenv(name, raising=False)


def test_agent_snapshots_proxy_environment_and_explicit_override(clean_proxy_env, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7892")
    monkeypatch.setenv("NO_PROXY", "example.com")
    agent = Agent()
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9000")
    monkeypatch.setenv("NO_PROXY", "*")
    assert agent.web_proxy_url == "http://127.0.0.1:7892"
    assert agent.web_no_proxy == "example.com"
    assert Agent(web_proxy_url="http://127.0.0.1:8000").web_proxy_url == "http://127.0.0.1:8000"
    assert Agent(web_proxy_url="direct").web_proxy_url is None


def test_proxy_environment_precedence(clean_proxy_env, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8001")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8002")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:8003")
    assert Agent().web_proxy_url == "http://127.0.0.1:8003"
    monkeypatch.setenv("PRAXIS_WEB_PROXY", "direct")
    assert Agent().web_proxy_url is None


@pytest.mark.parametrize("proxy", ["socks5://127.0.0.1:7892", "http://u:secret@proxy.test:8080",
                                   "http://proxy.test:8080/path", "not-a-url"])
def test_invalid_proxy_fails_without_disclosing_configuration(proxy, clean_proxy_env):
    with pytest.raises(ValueError) as caught:
        Agent(web_proxy_url=proxy)
    assert proxy not in str(caught.value)
    assert "secret" not in str(caught.value)


class ProxyStream(_SocketStream):
    def __init__(self):
        super().__init__()
        self.response = b"HTTP/1.1 200 Connection established\r\n\r\n"

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        import ssl
        assert ssl_context.verify_mode == ssl.CERT_REQUIRED
        assert ssl_context.check_hostname
        self.response = (b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: text/plain\r\n"
                         b"Connection: close\r\n\r\nok")
        return await super().start_tls(ssl_context, server_hostname, timeout)


def test_https_proxy_tunnels_hostname_and_keeps_verified_tls(monkeypatch):
    async def run():
        backend = _NumericBackend()
        backend.stream = ProxyStream()
        monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)

        async def unexpected_dns(*args, **kwargs):
            raise AssertionError("Origin DNS belongs to the explicitly trusted upstream proxy")

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", unexpected_dns)
        client = PublicWebClient(proxy_url="http://127.0.0.1:7892")
        try:
            result = await client.get("https://github.com/xiaoyinglei/praxis-agent-runtime")
            assert result.body == b"ok"
            assert result.connection_mode == "trusted_proxy"
            assert backend.hosts == ["127.0.0.1"]
            assert b"CONNECT github.com:443" in b"".join(backend.stream.writes)
            assert backend.stream.server_hostname == "github.com"
        finally:
            await client.aclose()
        assert backend.stream.closed

    asyncio.run(run())


def test_no_proxy_returns_to_strict_dns_checks(monkeypatch):
    async def run():
        async def resolve(*args, **kwargs):
            import socket
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443))]

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
        client = PublicWebClient(proxy_url="http://127.0.0.1:7892", no_proxy=".example.com")
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://docs.example.com")
            assert caught.value.code == "nonpublic_address"
            assert caught.value.failure_stage == "dns_validation"
            assert caught.value.connection_mode == "direct"
            assert "No website response was received" in str(caught.value)
            assert "does not establish" in str(caught.value)
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("url", ["http://127.0.0.1", "http://10.0.0.1", "http://localhost", "http://[::1]"])
def test_proxy_does_not_allow_literal_private_destinations(url):
    async def run():
        client = PublicWebClient(proxy_url="http://127.0.0.1:7892")
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get(url)
            assert caught.value.code == "invalid_url"
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.anyio
async def test_agent_executor_fetch_uses_proxy_and_records_mode(tmp_path, monkeypatch, clean_proxy_env):
    backend = _NumericBackend()
    backend.stream = ProxyStream()
    monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7892")
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "state.sqlite", enable_workspace_mcp=False)
    model = FetchThenAnswer({"url": "https://github.com/xiaoyinglei/praxis-agent-runtime"})
    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9000")
    result = await agent.run("Read my project", allow_web_tools=True, require_workspace_change=False)
    with RolloutStore(agent.checkpoint_db) as store:
        output = next(i.payload for i in store.list_items(result.turn_id) if i.kind == "tool_result")
        assert output["is_error"] is False
        assert output["structured_content"]["connection_mode"] == "trusted_proxy"
        assert "ok" in output["structured_content"]["content"]
        assert "7892" not in json.dumps(output, default=dict)
        assert store.verify().valid
    assert backend.hosts == ["127.0.0.1"]


@pytest.mark.parametrize("command", ["chat", "run", "resume"])
@pytest.mark.parametrize("force_terminal", [False, True])
def test_cli_exposes_proxy_override(command, force_terminal, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(typer.rich_utils, "FORCE_TERMINAL", force_terminal)
    result = CliRunner().invoke(agent_app, [command, "--help"])
    assert result.exit_code == 0
    assert "--web-proxy" in click.unstyle(result.stdout)


def test_proxy_failure_does_not_fall_back_to_direct(monkeypatch):
    async def run():
        backend = _NumericBackend()

        async def fail(host, *args, **kwargs):
            backend.hosts.append(host)
            raise httpcore.ConnectError("private transport details")

        monkeypatch.setattr(backend, "connect_tcp", fail)
        monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
        client = PublicWebClient(proxy_url="http://127.0.0.1:7892")
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://github.com")
            assert caught.value.connection_mode == "trusted_proxy"
            assert caught.value.failure_stage == "connect"
            assert "private transport details" not in str(caught.value)
            assert backend.hosts == ["127.0.0.1"]
        finally:
            await client.aclose()

    asyncio.run(run())


def test_proxy_timeout_reports_actual_connection_mode(monkeypatch):
    async def run():
        backend = _NumericBackend()

        async def fail(*args, **kwargs):
            raise httpcore.ConnectTimeout("private details")

        monkeypatch.setattr(backend, "connect_tcp", fail)
        monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
        client = PublicWebClient(proxy_url="http://127.0.0.1:7892")
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://github.com")
            assert caught.value.code == "timeout"
            assert caught.value.connection_mode == "trusted_proxy"
        finally:
            await client.aclose()

    asyncio.run(run())


def test_proxy_redirect_cannot_reach_private_literal():
    import httpx

    async def run():
        calls = []

        def redirect(request):
            calls.append(str(request.url))
            return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data"})

        client = PublicWebClient(proxy_url="http://127.0.0.1:7892", transport=httpx.MockTransport(redirect))
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "invalid_url"
            assert caught.value.connection_mode == "trusted_proxy"
            assert calls == ["https://example.com"]
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(("no_proxy", "url", "mode"), [
    ("example.com:443", "https://example.com", "direct"),
    ("example.com:80", "https://example.com", "trusted_proxy"),
    ("example.com", "https://badexample.com", "trusted_proxy"),
    ("*", "https://github.com", "direct"),
])
def test_no_proxy_port_and_domain_boundaries(no_proxy, url, mode):
    import httpx

    async def run():
        client = PublicWebClient(proxy_url="http://127.0.0.1:7892", no_proxy=no_proxy)
        try:
            assert client.connection_mode(httpx.URL(url)) == mode
        finally:
            await client.aclose()

    asyncio.run(run())


def test_cli_invalid_proxy_reports_safe_configuration_error(clean_proxy_env):
    result = CliRunner().invoke(agent_app, ["chat", "--web-proxy", "http://u:secret@proxy.test:8080"])
    assert result.exit_code == 2
    assert "credentials" in result.output
    assert "secret" not in result.output

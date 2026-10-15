from __future__ import annotations

import base64
import hashlib
import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from agent_runtime import mcp_admin


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def oauth(tmp_path, monkeypatch):
    from agent_runtime import mcp_oauth

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "personal"))
    monkeypatch.setattr(mcp_oauth, "_PLATFORM", "linux")
    return mcp_oauth


class Authority:
    def __init__(self):
        self.forms = []
        self.requests = []
        self.challenge = None
        self.failure = False
        self.issuer = "https://auth.example.com/"
        self.resource = "https://mcp.example.com/mcp"

    def respond(self, request):
        self.requests.append(request)
        path = request.url.path
        if path == "/mcp":
            if request.headers.get("authorization") in {"Bearer access-secret", "Bearer rotated-secret"}:
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(
                401,
                headers={
                    "WWW-Authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'
                },
            )
        if path == "/.well-known/oauth-protected-resource":
            return httpx.Response(
                200,
                json={
                    "resource": self.resource,
                    "authorization_servers": ["https://auth.example.com/"],
                },
            )
        if path == "/.well-known/oauth-authorization-server":
            return httpx.Response(
                200,
                json={
                    "issuer": self.issuer,
                    "authorization_endpoint": "https://auth.example.com/authorize",
                    "token_endpoint": "https://auth.example.com/token",
                    "registration_endpoint": "https://auth.example.com/register",
                    "code_challenge_methods_supported": ["S256"],
                },
            )
        if path == "/register":
            data = json.loads(request.content)
            return httpx.Response(201, json={**data, "client_id": "registered-client"})
        if path == "/token":
            form = parse_qs(request.content.decode())
            self.forms.append(form)
            if self.failure:
                return httpx.Response(400, json={"error": "access-secret refresh-secret"})
            if form["grant_type"] == ["authorization_code"]:
                actual = (
                    base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"][0].encode()).digest())
                    .decode()
                    .rstrip("=")
                )
                assert actual == self.challenge
                assert form["resource"] == [self.resource]
                return httpx.Response(
                    200, json={"access_token": "access-secret", "refresh_token": "refresh-secret", "expires_in": 0}
                )
            assert form["refresh_token"] == ["refresh-secret"]
            return httpx.Response(200, json={"access_token": "rotated-secret", "expires_in": 3600})
        return httpx.Response(404)

    async def browser(self, url, *, state=None, error=None):
        query = parse_qs(urlsplit(url).query)
        self.challenge = query["code_challenge"][0]
        assert query["code_challenge_method"] == ["S256"]
        callback = query["redirect_uri"][0]
        params = {"state": state or query["state"][0]}
        params.update({"error": error} if error else {"code": "authorization-secret"})
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(callback, params=params)
        return response


def register(tmp_path):
    mcp_admin.register_remote_mcp(
        "remote",
        url="https://mcp.example.com/mcp",
        workspace=tmp_path,
        oauth=True,
        oauth_allowed_domains=("auth.example.com",),
    )


@pytest.mark.anyio
async def test_browser_pkce_persistence_and_silent_reconnect_refresh(oauth, tmp_path, monkeypatch):
    authority = Authority()
    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(authority.respond))
    register(tmp_path)
    await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=authority.browser)
    storage = oauth.MCPOAuthStorage("remote", "https://mcp.example.com/mcp")
    assert (await storage.get_tokens()).refresh_token == "refresh-secret"
    provider = oauth.create_oauth_provider(mcp_admin.mcp_connections()[0])
    async with httpx.AsyncClient(
        transport=oauth.OAuthHTTPTransport(provider, httpx.MockTransport(authority.respond)),
        auth=provider,
        trust_env=False,
    ) as client:
        response = await client.get("https://mcp.example.com/mcp")
    assert response.status_code == 200
    assert authority.forms[-1]["grant_type"] == ["refresh_token"]
    assert (await storage.get_tokens()).refresh_token == "refresh-secret"
    config = mcp_admin.user_mcp_config().read_text()
    assert "secret" not in config
    path = next((tmp_path / "personal" / "credentials").glob("mcp-oauth-*.json"))
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert await oauth.MCPOAuthStorage("remote", "https://other.example.com/mcp").get_tokens() is None


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["state", "denied", "issuer", "token"])
async def test_rejected_login_does_not_leak_credentials(oauth, tmp_path, monkeypatch, caplog, kind):
    authority = Authority()
    authority.failure = kind == "token"
    if kind == "issuer":
        authority.issuer = "https://evil.example.com/"
    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(authority.respond))
    register(tmp_path)

    async def browser(url):
        await authority.browser(
            url, state="wrong-state" if kind == "state" else None, error="access_denied" if kind == "denied" else None
        )

    with pytest.raises(ValueError) as caught:
        await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=browser)
    assert "secret" not in str(caught.value)
    assert "secret" not in caplog.text
    assert await oauth.MCPOAuthStorage("remote", "https://mcp.example.com/mcp").get_tokens() is None


@pytest.mark.anyio
async def test_metadata_cannot_silently_authorize_another_host(oauth, tmp_path, monkeypatch):
    authority = Authority()
    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(authority.respond))
    mcp_admin.register_remote_mcp("remote", url="https://mcp.example.com/mcp", workspace=tmp_path, oauth=True)
    with pytest.raises(ValueError, match="authorized"):
        await oauth.login_mcp_oauth("remote", workspace=tmp_path)
    assert all(request.url.host == "mcp.example.com" for request in authority.requests)


@pytest.mark.anyio
async def test_runtime_never_opens_browser_without_credentials(oauth, tmp_path, monkeypatch):
    register(tmp_path)
    monkeypatch.setattr(oauth.webbrowser, "open", lambda *_: pytest.fail("runtime opened browser"))
    provider = oauth.create_oauth_provider(mcp_admin.mcp_connections()[0])
    with pytest.raises(ValueError, match="login"):
        async with httpx.AsyncClient(
            transport=oauth.OAuthHTTPTransport(provider, httpx.MockTransport(Authority().respond)),
            auth=provider,
            trust_env=False,
        ) as client:
            await client.get("https://mcp.example.com/mcp")


@pytest.mark.anyio
async def test_preconfigured_client_skips_registration_and_accepts_only_callback_path(oauth, tmp_path, monkeypatch):
    authority = Authority()
    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(authority.respond))
    mcp_admin.register_remote_mcp(
        "remote",
        url="https://mcp.example.com/mcp",
        workspace=tmp_path,
        oauth=True,
        oauth_client_id="preconfigured",
        oauth_allowed_domains=("auth.example.com",),
    )

    async def browser(url):
        query = parse_qs(urlsplit(url).query)
        assert query["client_id"] == ["preconfigured"]
        async with httpx.AsyncClient(trust_env=False) as client:
            wrong_path = query["redirect_uri"][0].rsplit("/", 1)[0] + "/wrong?code=secret&state=wrong"
            response = await client.get(wrong_path)
            assert response.status_code == 400
        await authority.browser(url)

    await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=browser)
    assert all(request.url.path != "/register" for request in authority.requests)


@pytest.mark.anyio
async def test_credentials_reject_symlinks_and_public_modes(oauth, tmp_path):
    from mcp.shared.auth import OAuthToken

    storage = oauth.MCPOAuthStorage("remote", "https://mcp.example.com/mcp")
    await storage.set_tokens(OAuthToken(access_token="access-secret"))
    path = storage._path()
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        await storage.get_tokens()
    path.unlink()
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="private"):
        await storage.set_tokens(OAuthToken(access_token="other-secret"))
    assert outside.read_text() == "unchanged"


def test_remove_connection_clears_linux_credentials(oauth, tmp_path, monkeypatch):
    register(tmp_path)
    storage = oauth.MCPOAuthStorage("remote", "https://mcp.example.com/mcp")
    storage.update({"tokens": {"access_token": "access-secret"}})
    mcp_admin.remove_mcp_connection("remote")
    assert mcp_admin.mcp_connections() == []
    assert storage.read() == {}


def test_linux_manual_bearer_credentials_are_endpoint_bound(oauth, tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_admin.sys, "platform", "linux")
    mcp_admin.save_mcp_credential("remote", "https://mcp.example.com/mcp", "manual-secret")
    assert mcp_admin.load_mcp_credential("remote", "https://mcp.example.com/mcp") == "manual-secret"
    with pytest.raises(ValueError):
        mcp_admin.load_mcp_credential("remote", "https://other.example.com/mcp")


@pytest.mark.anyio
async def test_runtime_oauth_discovers_calls_refreshes_and_reconnects(oauth, tmp_path, monkeypatch):
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings

    from agent_runtime.permission_policy import resolve_permission_sources
    from agent_runtime.runtime import mcp as runtime

    authority = Authority()
    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(authority.respond))
    mcp_admin.register_remote_mcp(
        "remote",
        url="https://mcp.example.com/mcp",
        workspace=tmp_path,
        tools=("echo",),
        oauth=True,
        oauth_allowed_domains=("auth.example.com",),
    )
    await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=authority.browser)
    server = FastMCP(
        "oauth-test",
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(allowed_hosts=["mcp.example.com"]),
    )

    @server.tool(annotations={"readOnlyHint": True})
    def echo(message: str) -> str:
        return "echo: " + message

    app = server.streamable_http_app()

    class Transport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.mcp = httpx.ASGITransport(app=app)

        async def handle_async_request(self, request):
            if request.url.host == "mcp.example.com":
                assert request.headers.get("authorization") == "Bearer rotated-secret"
                return await self.mcp.handle_async_request(request)
            return authority.respond(request)

        async def aclose(self):
            await self.mcp.aclose()

    monkeypatch.setattr(runtime, "public_http_transport", Transport)
    async with server.session_manager.run():
        for _ in range(2):
            diagnostics = []
            async with runtime.open_product_mcp_tools(
                mcp_admin.user_mcp_config(), permissions=resolve_permission_sources(tmp_path), diagnostics=diagnostics
            ) as tools:
                assert len(tools) == 1, diagnostics
                result = await tools[0].run({"message": "connected"})
                assert "echo: connected" in result.content[0].text
    assert len([form for form in authority.forms if form["grant_type"] == ["refresh_token"]]) == 1


@pytest.mark.anyio
async def test_login_cannot_override_explicit_oauth_host_deny(oauth, tmp_path, monkeypatch):
    from agent_runtime.permission_policy import NetworkPolicy, NetworkRule, PermissionPolicy

    authority = Authority()
    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(authority.respond))
    register(tmp_path)
    path = mcp_admin.user_mcp_config().parent / "permissions.json"
    path.write_text(
        PermissionPolicy(
            network=NetworkPolicy(
                rules=(
                    NetworkRule(domain="mcp.example.com", decision="allow"),
                    NetworkRule(domain="auth.example.com", decision="deny"),
                )
            )
        ).model_dump_json()
    )
    with pytest.raises(ValueError):
        await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=authority.browser)
    assert all(request.url.host != "auth.example.com" for request in authority.requests)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        {
            "resource": "https://mcp.example.com/mcp?access-secret",
            "authorization_servers": ["https://auth.example.com/"],
        },
        {"resource": "https://other.example.com/access-secret", "authorization_servers": ["https://auth.example.com/"]},
        {"resource": "access-secret", "authorization_servers": ["https://auth.example.com/"]},
    ],
)
async def test_untrusted_resource_metadata_is_redacted(oauth, tmp_path, monkeypatch, caplog, payload):
    authority = Authority()

    def respond(request):
        if request.url.path == "/.well-known/oauth-protected-resource":
            return httpx.Response(200, json=payload)
        return authority.respond(request)

    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(respond))
    register(tmp_path)
    with pytest.raises(ValueError) as caught:
        await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=authority.browser)
    assert "access-secret" not in caplog.text
    assert "access-secret" not in str(caught.value)


@pytest.mark.anyio
async def test_refresh_preserves_advertised_parent_resource(oauth, tmp_path, monkeypatch):
    authority = Authority()
    authority.resource = "https://mcp.example.com/"
    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(authority.respond))
    register(tmp_path)
    await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=authority.browser)
    provider = oauth.create_oauth_provider(mcp_admin.mcp_connections()[0])
    async with httpx.AsyncClient(
        transport=oauth.OAuthHTTPTransport(provider, httpx.MockTransport(authority.respond)),
        auth=provider,
        trust_env=False,
    ) as client:
        await client.get("https://mcp.example.com/mcp")
    assert authority.forms[-1]["resource"] == [authority.resource]


@pytest.mark.anyio
async def test_runtime_revoke_prevents_oauth_refresh_before_any_network(oauth, tmp_path, monkeypatch):
    from agent_runtime.permission_policy import NetworkPolicy, PermissionPolicy, resolve_permission_sources
    from agent_runtime.runtime import mcp as runtime

    authority = Authority()
    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(authority.respond))
    register(tmp_path)
    await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=authority.browser)
    config_path = mcp_admin.user_mcp_config()
    config = runtime._load_enabled_servers(config_path)[0]
    start = resolve_permission_sources(tmp_path)
    revoked = PermissionPolicy(network=NetworkPolicy(decision="deny")).resolve(tmp_path)
    called = []

    def current():
        # Discovery already checked policy; revoke before the auth flow starts.
        return revoked

    monkeypatch.setattr(
        runtime,
        "public_http_transport",
        lambda: httpx.MockTransport(lambda request: called.append(request) or authority.respond(request)),
    )
    from mcp.client.session import ClientSession

    with pytest.raises(ExceptionGroup) as caught:
        async with runtime._http_mcp_streams(config, config_path, start, current) as connection:
            streams, _ = connection
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
    assert caught.value.subgroup(ValueError) is not None
    assert called == []


def test_keychain_save_uses_stdin_and_verifies_storage(oauth, tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(oauth, "_PLATFORM", "darwin")

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=44 if "find-generic-password" in argv else 0, stdout="", stderr="")

    monkeypatch.setattr(oauth.subprocess, "run", run)
    storage = oauth.MCPOAuthStorage("remote", "https://mcp.example.com/mcp")
    with pytest.raises(ValueError, match="verification"):
        storage.update({"tokens": {"access_token": "keychain-secret"}})
    assert all("keychain-secret" not in str(argv) for argv, _ in calls)
    assert any("keychain-secret" in kwargs.get("input", "") for _, kwargs in calls)


@pytest.mark.anyio
async def test_explicit_login_with_valid_token_opens_browser_again(oauth, tmp_path, monkeypatch):
    authority = Authority()
    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(authority.respond))
    register(tmp_path)
    redirects = []

    async def browser(url):
        redirects.append(url)
        await authority.browser(url)

    await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=browser)
    storage = oauth.MCPOAuthStorage("remote", "https://mcp.example.com/mcp")
    storage.update({"expires_at": __import__("time").time() + 3600})
    before = storage.read()

    async def denied(url):
        redirects.append(url)
        await authority.browser(url, error="access_denied")

    with pytest.raises(ValueError):
        await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=denied)
    assert len(redirects) == 2
    assert storage.read() == before
    await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=browser)
    assert len(redirects) == 3


@pytest.mark.anyio
async def test_preconfigured_client_uses_registered_callback_port_and_path(oauth, tmp_path, monkeypatch):
    import socket

    authority = Authority()
    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(authority.respond))
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    mcp_admin.register_remote_mcp(
        "remote",
        url="https://mcp.example.com/mcp",
        workspace=tmp_path,
        oauth=True,
        oauth_client_id="preconfigured",
        oauth_callback_port=port,
        oauth_allowed_domains=("auth.example.com",),
    )

    async def browser(url):
        query = parse_qs(urlsplit(url).query)
        assert query["redirect_uri"] == [f"http://127.0.0.1:{port}/callback"]
        await authority.browser(url)

    await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=browser)
    assert authority.forms[0]["redirect_uri"] == [f"http://127.0.0.1:{port}/callback"]


@pytest.mark.anyio
@pytest.mark.parametrize("get_unsupported", [False, True])
async def test_login_closes_sse_without_reading_and_handles_post_only_mcp(
    oauth,
    tmp_path,
    monkeypatch,
    get_unsupported,
):
    authority = Authority()
    streams = []

    class HeaderOnlyStream(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False

        async def __aiter__(self):
            raise AssertionError("login must not consume a potentially infinite MCP stream")
            yield b""  # pragma: no cover

        async def aclose(self):
            self.closed = True

    def respond(request):
        if request.url.path == "/mcp" and request.method == "GET" and get_unsupported:
            authority.requests.append(request)
            return httpx.Response(405, headers={"Allow": "POST"})
        if request.url.path == "/mcp" and request.headers.get("authorization") == "Bearer access-secret":
            authority.requests.append(request)
            stream = HeaderOnlyStream()
            streams.append(stream)
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)
        return authority.respond(request)

    monkeypatch.setattr(oauth, "public_http_transport", lambda: httpx.MockTransport(respond))
    register(tmp_path)
    await oauth.login_mcp_oauth("remote", workspace=tmp_path, redirect_handler=authority.browser, timeout_seconds=3)
    assert streams and all(stream.closed for stream in streams)
    tokens = await oauth.MCPOAuthStorage("remote", "https://mcp.example.com/mcp").get_tokens()
    assert tokens.access_token == "access-secret"
    probes = [request for request in authority.requests if request.url.path == "/mcp"]
    if get_unsupported:
        assert probes[0].method == "GET"
        assert all(request.method == "POST" for request in probes[1:])
        for request in probes[1:]:
            body = json.loads(request.content)
            assert body["method"] == "initialize"
            assert body["params"]["capabilities"] == {}
            assert str(tmp_path) not in str(body)

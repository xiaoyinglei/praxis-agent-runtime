"""Host-owned browser OAuth for endpoint-bound remote MCP connections.

Only explicit login listens on loopback and opens a browser. Runtime providers
silently refresh saved credentials; metadata never extends tool permissions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shlex
import stat
import subprocess
import sys
import time
import webbrowser
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)
from mcp.shared.auth_utils import check_resource_allowed, resource_url_from_server_url
from mcp.types import LATEST_PROTOCOL_VERSION
from pydantic import AnyUrl

from agent_runtime.model_config_io import atomic_replace_bytes, exclusive_config_lock, file_fingerprint
from agent_runtime.tools.web_http import public_http_transport, validate_public_url
from agent_runtime.user_config import user_config_directory

_PLATFORM = sys.platform


def validate_oauth_domains(values: tuple[str, ...]) -> tuple[str, ...]:
    result = []
    for value in values:
        try:
            parsed = validate_public_url("https://" + value)
            if parsed.host != value or parsed.path != "/" or parsed.port not in {None, 443}:
                raise ValueError
        except Exception:
            raise ValueError("OAuth domains must be exact public HTTPS hostnames") from None
        result.append(value)
    return tuple(dict.fromkeys(result))


class MCPOAuthStorage:
    """SDK TokenStorage backed by Keychain or a private POSIX credential file."""

    def __init__(self, name: str, url: str) -> None:
        self.url = url
        self.identity = hashlib.sha256(f"{user_config_directory()}\n{name}\n{url}".encode()).hexdigest()
        self.context: Any = None
        self.pending: dict[str, Any] | None = None

    def _path(self) -> Path:
        directory = user_config_directory() / "credentials"
        if directory.is_symlink():
            raise ValueError("OAuth credentials require a private real directory")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = directory.stat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise ValueError("OAuth credential directory must be private and user-owned")
        return directory / f"mcp-oauth-{self.identity}.json"

    def read(self) -> dict[str, Any]:
        if self.pending is not None:
            return dict(self.pending)
        raw: str | bytes
        if _PLATFORM == "darwin":
            result = subprocess.run(
                [
                    "/usr/bin/security",
                    "find-generic-password",
                    "-a",
                    "praxis",
                    "-s",
                    "praxis.mcp.oauth." + self.identity,
                    "-w",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode == 44:
                return {}
            if result.returncode:
                raise ValueError("Unable to read OAuth credential from Keychain")
            raw = result.stdout
        elif _PLATFORM == "win32":
            raise ValueError("Saved OAuth credentials require macOS or POSIX")
        else:
            path = self._path()
            try:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            except FileNotFoundError:
                return {}
            except OSError:
                raise ValueError("OAuth credential file must be private and regular") from None
            with os.fdopen(descriptor, "rb") as handle:
                metadata = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o077
                    or metadata.st_nlink != 1
                    or metadata.st_size > 1_048_576
                ):
                    raise ValueError("OAuth credential file must be private, regular, and bounded")
                raw = handle.read()
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get("url") != self.url:
                raise ValueError
            return data
        except (ValueError, TypeError):
            raise ValueError("Saved OAuth credential is invalid; log in again") from None

    def update(self, values: Mapping[str, Any]) -> None:
        if self.pending is not None:
            self.pending.update(values)
            return

        def content() -> bytes:
            return json.dumps({**self.read(), **values, "url": self.url}).encode()

        if _PLATFORM == "darwin":
            command = "add-generic-password -U -a praxis -s " + shlex.quote("praxis.mcp.oauth." + self.identity)
            payload = content()
            command += " -w " + shlex.quote(payload.decode()) + "\n"
            result = subprocess.run(
                ["/usr/bin/security", "-i"], input=command, capture_output=True, text=True, timeout=30, check=False
            )
            if result.returncode or "SecKeychain" in result.stderr:
                raise ValueError("Unable to save OAuth credential in Keychain")
            if self.read() != json.loads(payload):
                raise ValueError("OAuth credential storage verification failed")
            return
        path = self._path()
        with exclusive_config_lock(path):
            payload = content()
            atomic_replace_bytes(path, payload, intended_fingerprint=file_fingerprint(payload))

    def clear(self) -> None:
        if _PLATFORM == "darwin":
            result = subprocess.run(
                [
                    "/usr/bin/security",
                    "delete-generic-password",
                    "-a",
                    "praxis",
                    "-s",
                    "praxis.mcp.oauth." + self.identity,
                ],
                capture_output=True,
                timeout=15,
                check=False,
            )
            if result.returncode not in {0, 44}:
                raise ValueError("Unable to remove OAuth credential from Keychain")
        else:
            path = self._path()
            with exclusive_config_lock(path):
                self.read()  # Refuse unsafe files even during removal.
                path.unlink(missing_ok=True)

    async def get_tokens(self) -> OAuthToken | None:
        data = self.read().get("tokens")
        try:
            return None if data is None else OAuthToken.model_validate(data)
        except ValueError:
            raise ValueError("Saved OAuth token is invalid; log in again") from None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        # A refresh server may omit a replacement refresh token (RFC 6749 6).
        previous = await self.get_tokens()
        if tokens.refresh_token is None and previous is not None:
            tokens.refresh_token = previous.refresh_token
        for value in (tokens.access_token, tokens.refresh_token):
            if value is not None and (not value or any(ord(char) < 32 or ord(char) == 127 for char in value)):
                raise ValueError("OAuth server returned an invalid token")
        values: dict[str, Any] = {
            "tokens": tokens.model_dump(mode="json"),
            "expires_at": None if tokens.expires_in is None else time.time() + tokens.expires_in,
        }
        if self.context is not None and self.context.oauth_metadata is not None:
            values["metadata"] = self.context.oauth_metadata.model_dump(mode="json")
            values["auth_server_url"] = self.context.auth_server_url
            if self.context.protected_resource_metadata is not None:
                values["protected_resource"] = self.context.protected_resource_metadata.model_dump(mode="json")
        self.update(values)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self.read().get("client")
        try:
            return None if data is None else OAuthClientInformationFull.model_validate(data)
        except ValueError:
            raise ValueError("Saved OAuth client is invalid; log in again") from None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.update({"client": client_info.model_dump(mode="json")})


class MCPOAuthProvider(OAuthClientProvider):
    def __init__(
        self,
        entry: Mapping[str, Any],
        *,
        redirect_uri: str = "http://127.0.0.1/callback",
        redirect_handler: Callable[[str], Awaitable[None]] | None = None,
        callback_handler: Callable[[], Awaitable[tuple[str, str | None]]] | None = None,
    ) -> None:
        from agent_runtime.runtime.mcp import validate_http_mcp_url

        self.endpoint = validate_http_mcp_url(entry["url"])
        self.allowed_domains = frozenset(
            (httpx.URL(self.endpoint).host, *validate_oauth_domains(tuple(entry.get("oauth_allowed_domains", ()))))
        )
        self.explicit_scopes = entry.get("oauth_scopes")
        self.interactive = redirect_handler is not None
        self.preconfigured_client_id = entry.get("oauth_client_id")
        self.destination_check: Callable[[str], None] | None = None
        storage = MCPOAuthStorage(entry["name"], self.endpoint)
        metadata = OAuthClientMetadata(
            redirect_uris=[AnyUrl(redirect_uri)],
            token_endpoint_auth_method="none",
            client_name="Praxis",
            scope=" ".join(self.explicit_scopes or ()) or None,
        )
        super().__init__(
            self.endpoint, metadata, storage, redirect_handler=redirect_handler, callback_handler=callback_handler
        )
        self.storage = storage
        storage.context = self.context

    def validate_url(self, value: str, *, browser: bool = False) -> str:
        from agent_runtime.runtime.mcp import validate_http_mcp_url

        try:
            parsed = httpx.URL(value)
            base = str(parsed.copy_with(query=None)) if browser else value
            validate_http_mcp_url(base)
            if parsed.host not in self.allowed_domains:
                raise ValueError("OAuth destination is not host-authorized; add its exact OAuth domain")
            if self.destination_check is not None:
                self.destination_check(base)
            return value
        except ValueError:
            raise ValueError("OAuth destination is not a host-authorized public HTTPS URL") from None

    async def _initialize(self) -> None:
        await super()._initialize()
        saved = self.storage.read()
        if not self.interactive and self.context.current_tokens is None:
            raise ValueError("MCP OAuth requires login: run agent mcp login for this connection")
        if saved.get("metadata") is not None:
            try:
                metadata = OAuthMetadata.model_validate(saved["metadata"])
                for field in ("issuer", "authorization_endpoint", "token_endpoint", "registration_endpoint"):
                    value = getattr(metadata, field)
                    if value is not None:
                        self.validate_url(str(value))
                self.context.oauth_metadata = metadata
                self.context.auth_server_url = saved.get("auth_server_url")
                if saved.get("protected_resource") is not None:
                    protected = ProtectedResourceMetadata.model_validate(saved["protected_resource"])
                    self.validate_url(str(protected.resource))
                    if not check_resource_allowed(
                        requested_resource=resource_url_from_server_url(self.endpoint),
                        configured_resource=str(protected.resource),
                    ):
                        raise ValueError
                    self.context.protected_resource_metadata = protected
            except ValueError:
                raise ValueError("Saved OAuth metadata is invalid; log in again") from None
        self.context.token_expiry_time = saved.get("expires_at")
        if self.preconfigured_client_id:
            self.context.client_info = OAuthClientInformationFull(
                **self.context.client_metadata.model_dump(), client_id=self.preconfigured_client_id
            )
        elif self.interactive and self.context.client_info is not None:
            if self.context.client_info.redirect_uris != self.context.client_metadata.redirect_uris:
                self.context.client_info = None

    async def _perform_authorization(self) -> httpx.Request:
        if not self.interactive:
            raise ValueError("MCP OAuth session expired; run agent mcp login for this connection")
        if self.explicit_scopes:
            self.context.client_metadata.scope = " ".join(self.explicit_scopes)
        return await super()._perform_authorization()

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        flow = super().async_auth_flow(request)
        try:
            outgoing = await anext(flow)
            while True:
                response = yield outgoing
                outgoing = await flow.asend(response)
        except StopAsyncIteration:
            return
        except Exception:
            raise ValueError("MCP OAuth failed; check host authorization and run agent mcp login again") from None
        finally:
            await flow.aclose()


def create_oauth_provider(entry: Mapping[str, Any], **kwargs: Any) -> MCPOAuthProvider:
    return MCPOAuthProvider(entry, **kwargs)


class OAuthHTTPTransport(httpx.AsyncBaseTransport):
    """Check every SDK discovery/token request, including metadata-selected hosts."""

    def __init__(self, provider: MCPOAuthProvider, inner: httpx.AsyncBaseTransport) -> None:
        self.provider = provider
        self.inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        provider = self.provider
        provider.validate_url(str(request.url))
        if str(request.url) != provider.endpoint:
            if request.method not in {"GET", "POST"}:
                raise ValueError("Unsupported OAuth request")
            if request.headers.get("authorization", "").lower().startswith("bearer "):
                raise ValueError("MCP access tokens cannot be sent to OAuth endpoints")
        if request.headers.get("host", "").casefold() != request.url.netloc.decode().casefold():
            raise ValueError("OAuth HTTP Host authority mismatch")
        try:
            response = await self.inner.handle_async_request(request)
        except Exception:
            raise ValueError("OAuth network request failed") from None
        if response.is_redirect:
            await response.aclose()
            raise ValueError("OAuth HTTP redirects are refused")
        if str(request.url) == provider.endpoint:
            return response
        try:
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > 1_048_576:
                    raise ValueError
            await response.aclose()
            if response.status_code not in {200, 201}:
                return httpx.Response(response.status_code, content=b"", request=request)
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError
            if "authorization_servers" in data:
                protected = ProtectedResourceMetadata.model_validate(data)
                resource = str(protected.resource)
                provider.validate_url(resource)
                if not check_resource_allowed(
                    requested_resource=resource_url_from_server_url(provider.endpoint), configured_resource=resource
                ):
                    raise ValueError
                for value in protected.authorization_servers:
                    provider.validate_url(str(value))
            if "issuer" in data:
                metadata = OAuthMetadata.model_validate(data)
                expected = provider.context.auth_server_url or str(httpx.URL(provider.endpoint).copy_with(path="/"))
                if str(metadata.issuer).rstrip("/") != expected.rstrip("/"):
                    raise ValueError
                for field in ("issuer", "authorization_endpoint", "token_endpoint", "registration_endpoint"):
                    value = getattr(metadata, field)
                    if value is not None:
                        provider.validate_url(str(value))
                if "S256" not in (metadata.code_challenge_methods_supported or ()):
                    raise ValueError
            if request.method == "POST":
                if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
                    OAuthToken.model_validate(data)
                else:
                    client = OAuthClientInformationFull.model_validate(data)
                    if not client.client_id:
                        raise ValueError
        except Exception:
            await response.aclose()
            raise ValueError("OAuth server returned invalid or unauthorized metadata or credentials") from None
        return httpx.Response(response.status_code, headers=response.headers, content=bytes(body), request=request)

    async def aclose(self) -> None:
        await self.inner.aclose()


async def login_mcp_oauth(
    name: str,
    *,
    workspace: Path | None = None,
    redirect_handler: Callable[[str], Awaitable[None]] | None = None,
    timeout_seconds: float = 180,
    callback_port: int | None = None,
) -> None:
    """Explicit host login; callback codes and tokens never enter GUI URLs or tools."""
    from agent_runtime import mcp_admin
    from agent_runtime.permission_policy import resolve_permission_sources

    with mcp_admin.configuration_lock():
        entries = [entry for entry in mcp_admin.mcp_connections() if entry["name"] == name]
        if not entries or not entries[0].get("oauth"):
            raise ValueError("Register an OAuth MCP connection before login")
        entry = entries[0]
        revision = hashlib.sha256(mcp_admin.user_mcp_config().read_bytes()).hexdigest()
    if resolve_permission_sources(workspace or Path.cwd()).network_decision(entry["url"]) != "allow":
        raise ValueError("MCP destination must be authorized before OAuth login")
    loop = asyncio.get_running_loop()
    callback: asyncio.Future[tuple[str, str | None]] = loop.create_future()
    expected_state: str | None = None
    callback_path = "/callback"
    selected_port = entry.get("oauth_callback_port", 0) if callback_port is None else callback_port
    if type(selected_port) is not int or not 0 <= selected_port <= 65535:
        raise ValueError("OAuth callback port must be an integer from 0 to 65535")

    async def receive(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, message = "400 Bad Request", "OAuth callback rejected."
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
            lines = head.decode("ascii").split("\r\n")
            method, target, _ = lines[0].split(" ")
            parsed = urlsplit(target)
            params = parse_qs(parsed.query, keep_blank_values=True)
            host = next((line.split(":", 1)[1].strip() for line in lines[1:] if line.lower().startswith("host:")), "")
            valid = (
                method == "GET"
                and not parsed.scheme
                and not parsed.netloc
                and parsed.path == callback_path
                and host == f"127.0.0.1:{port}"
                and not callback.done()
            )
            if valid:
                state = params.get("state", [])
                code = params.get("code", [])
                if (
                    len(state) != 1
                    or expected_state is None
                    or not secrets.compare_digest(state[0], expected_state)
                    or "error" in params
                    or len(code) != 1
                    or not code[0]
                ):
                    callback.set_exception(ValueError("OAuth callback rejected"))
                else:
                    callback.set_result((code[0], state[0]))
                    status, message = "200 OK", "Authorization received. You may close this window."
        except Exception:
            pass
        finally:
            payload = message.encode()
            writer.write(
                (
                    f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nCache-Control: no-store\r\n"
                    f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                ).encode()
                + payload
            )
            try:
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

    server = await asyncio.start_server(receive, "127.0.0.1", selected_port, limit=8192)
    port = server.sockets[0].getsockname()[1]

    async def redirect(url: str) -> None:
        nonlocal expected_state
        provider.validate_url(url, browser=True)
        expected_state = parse_qs(urlsplit(url).query)["state"][0]
        if redirect_handler is not None:
            await redirect_handler(url)
        elif not await asyncio.to_thread(webbrowser.open, url):
            raise ValueError("Unable to open OAuth authorization browser")

    async def wait_callback() -> tuple[str, str | None]:
        return await callback

    provider = create_oauth_provider(
        entry,
        redirect_uri=f"http://127.0.0.1:{port}{callback_path}",
        redirect_handler=redirect,
        callback_handler=wait_callback,
    )

    def check_destination(url: str) -> None:
        if hashlib.sha256(mcp_admin.user_mcp_config().read_bytes()).hexdigest() != revision:
            raise ValueError("MCP connection changed during login")
        policy = resolve_permission_sources(workspace or Path.cwd())
        if policy.network_decision(entry["url"]) != "allow" or policy.network_decision(url) == "deny":
            raise ValueError("Current network policy denies OAuth destination")

    provider.destination_check = check_destination
    # Login is an explicit fresh authorization. Stage SDK writes so denial,
    # timeout or cancellation leaves the last working token/client untouched.
    provider.storage.pending = provider.storage.read()
    provider.storage.pending.pop("tokens", None)
    provider.storage.pending.pop("expires_at", None)
    try:
        async with server, asyncio.timeout(timeout_seconds):
            async with httpx.AsyncClient(
                auth=provider,
                transport=OAuthHTTPTransport(provider, public_http_transport()),
                trust_env=False,
                follow_redirects=False,
                timeout=30,
            ) as client:
                # A successful GET may be a never-ending SSE stream. Login only
                # needs the final authentication status, so close after headers.
                async with client.stream("GET", entry["url"]) as response:
                    status = response.status_code
                if status == 405 or (status < 400 and await provider.storage.get_tokens() is None):
                    # Streamable HTTP permits servers to omit GET/SSE support.
                    # initialize is a protocol handshake, without tools or any
                    # workspace context, and obtains the POST auth challenge.
                    async with client.stream(
                        "POST",
                        entry["url"],
                        headers={"Accept": "application/json, text/event-stream"},
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": LATEST_PROTOCOL_VERSION,
                                "capabilities": {},
                                "clientInfo": {"name": "praxis-oauth-login", "version": "0.1.0"},
                            },
                        },
                    ) as response:
                        status = response.status_code
                if status >= 400 or await provider.storage.get_tokens() is None:
                    raise ValueError("OAuth authorization was not accepted")
            with mcp_admin.configuration_lock():
                if hashlib.sha256(mcp_admin.user_mcp_config().read_bytes()).hexdigest() != revision:
                    raise ValueError("MCP connection changed during login; log in again")
                saved = provider.storage.pending
                provider.storage.pending = None
                provider.storage.update(saved)
    except Exception:
        raise ValueError("MCP OAuth login failed; check host-authorized domains and retry login") from None
    finally:
        if not callback.done():
            callback.cancel()
        elif not callback.cancelled():
            callback.exception()


def clear_oauth_credentials(name: str, url: str) -> None:
    MCPOAuthStorage(name, url).clear()

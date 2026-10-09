"""Bounded public HTTP reads, with DNS checks at the actual socket connection.

URL prechecks never resolve DNS, so tools can run them before permission approval.
The default transport checks every DNS answer, connects only to a checked numeric
address, and leaves the origin hostname intact for HTTP Host and verified TLS.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import ssl
import zlib
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.request import proxy_bypass_environment  # type: ignore[attr-defined]

import httpcore
import httpx


class PublicWebError(Exception):
    """A stable error code and safe message, without remote data or URL secrets."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.failure_stage = {
            "nonpublic_address": "dns_validation", "http_error": "http_response",
            "invalid_url": "url_validation", "network_error": "connect", "timeout": "request",
        }.get(code, "response_processing")
        self.connection_mode = "unknown"
        self.render_diagnostics: Mapping[str, Any] | None = None
        super().__init__(message)


def validate_proxy_url(value: str) -> str:
    """Proxy endpoints are trusted application configuration, never model inputs."""
    try:
        url = httpx.URL(value)
        if (url.scheme not in {"http", "https"} or not url.host or url.userinfo
                or url.path not in {"", "/"} or url.query or url.fragment
                or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)):
            raise ValueError
        return str(url)
    except (ValueError, httpx.InvalidURL):
        raise ValueError(
            "Web proxy must be an HTTP(S) endpoint without credentials, path, query or fragment."
        ) from None


def web_proxy_configuration(
    explicit: str | None, environment: Mapping[str, str],
) -> tuple[str | None, str]:
    """Read a caller-owned startup environment once; workspace files are never read."""
    value = explicit
    if value is None:
        value = next((environment[key] for key in (
            "PRAXIS_WEB_PROXY", "https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY",
        ) if environment.get(key)), None)
    proxy = None if value is None or value in {"", "direct"} else validate_proxy_url(value)
    no_proxy = environment.get("no_proxy", environment.get("NO_PROXY", ""))
    return proxy, ",".join(no_proxy.replace(",", " ").split())


def _is_public_address(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast and not address.is_reserved


def validate_public_url(url: str) -> httpx.URL:
    """Validate URL syntax and literal addresses only; do not perform DNS I/O."""
    invalid = PublicWebError("invalid_url", "A public HTTP(S) URL on port 80 or 443 is required.")
    if not url or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url):
        raise invalid
    try:
        parsed = httpx.URL(url)
        host = parsed.host.lower().rstrip(".")
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or parsed.userinfo
            or parsed.port not in {None, 80, 443}
            or "%" in host
            or host == "localhost"
            or host.endswith(".localhost")
        ):
            raise invalid
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            if not _is_public_address(host):
                raise invalid
    except (httpx.InvalidURL, ValueError):
        raise invalid from None
    return parsed.copy_with(fragment=None)


@dataclass(frozen=True, slots=True)
class FetchedResponse:
    url: str
    content_type: str
    body: bytes
    network_bytes: int
    connection_mode: str = "direct"
    warning: str | None = None
    response_headers: Mapping[str, str] | None = None
    render_diagnostics: Mapping[str, Any] | None = None
    status_code: int = 200
    request_count: int = 1


# Matches httpcore's public backend signature, without importing private modules.
SocketOption = tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]


@dataclass
class _RequestProgress:
    stage: str = "connect"
    mode: str = "unknown"


_progress: ContextVar[_RequestProgress | None] = ContextVar("public_web_progress", default=None)


def _stage(value: str) -> None:
    progress = _progress.get()
    if progress is not None:
        progress.stage = value


class _ConnectionCandidates:
    """One validated DNS snapshot and one deadline for TCP plus verified TLS."""

    def __init__(
        self, backend: httpcore.AsyncNetworkBackend, addresses: list[str], port: int,
        timeout: float | None, local_address: str | None, socket_options: Iterable[SocketOption] | None,
    ) -> None:
        self.backend = backend
        self.addresses = list(dict.fromkeys(addresses))[:8]
        self.port = port
        self.deadline = asyncio.get_running_loop().time() + (timeout if timeout is not None else 20)
        self.local_address = local_address
        self.socket_options = socket_options
        self.index = 0
        self.attempt_deadline = self.deadline

    def remaining(self) -> float:
        return max(0, self.attempt_deadline - asyncio.get_running_loop().time())

    async def connect(self) -> httpcore.AsyncNetworkStream:
        last_error: Exception = httpcore.ConnectTimeout()
        while self.index < len(self.addresses):
            now = asyncio.get_running_loop().time()
            remaining = self.deadline - now
            if remaining <= 0:
                raise httpcore.ConnectTimeout()
            # Reserve time for other candidates even when the request budget is small.
            budget = min(3.0, remaining / (len(self.addresses) - self.index))
            self.attempt_deadline = now + budget
            address = self.addresses[self.index]
            self.index += 1
            _stage("tcp_connect")
            try:
                async with asyncio.timeout(budget):
                    return await self.backend.connect_tcp(
                        address, self.port, timeout=budget, local_address=self.local_address,
                        socket_options=self.socket_options,
                    )
            except (httpcore.NetworkError, httpcore.TimeoutException, OSError, TimeoutError) as error:
                last_error = error
        raise last_error


class _PublicStream(httpcore.AsyncNetworkStream):
    """Retry only pre-HTTP connections, closing each failed or cancelled socket."""

    def __init__(self, stream: httpcore.AsyncNetworkStream, candidates: _ConnectionCandidates | None = None) -> None:
        self._stream = stream
        self._candidates = candidates

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return await self._stream.read(max_bytes, timeout)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        await self._stream.write(buffer, timeout)

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        while True:
            _stage("tls_handshake")
            budget = self._candidates.remaining() if self._candidates is not None else timeout
            try:
                async with asyncio.timeout(budget):
                    secured = await self._stream.start_tls(ssl_context, server_hostname, budget)
                return _PublicStream(secured)
            except BaseException as error:
                try:
                    # Closing a failed socket must not mask the original failure or cancellation.
                    async with asyncio.timeout(0.1):
                        await self._stream.aclose()
                except Exception:
                    pass
                recoverable = isinstance(
                    error, (httpcore.NetworkError, httpcore.TimeoutException, OSError, TimeoutError)
                )
                if (self._candidates is None or self._candidates.index >= len(self._candidates.addresses)
                        or not recoverable):
                    raise
                self._stream = await self._candidates.connect()

    def get_extra_info(self, info: str) -> object:
        return self._stream.get_extra_info(info)


class _PublicNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self) -> None:
        self._backend = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        # Numeric connects avoid the second, unchecked DNS resolution that allows
        # rebinding. Every A/AAAA answer must be public before choosing one.
        async with asyncio.timeout(timeout):
            started = asyncio.get_running_loop().time()
            _stage("dns_resolution")
            try:
                ipaddress.ip_address(host)
            except ValueError:
                answers = await asyncio.get_running_loop().getaddrinfo(
                    host,
                    port,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                    proto=socket.IPPROTO_TCP,
                )
                addresses = [str(answer[4][0]) for answer in answers]
            else:
                addresses = [host]
            if not addresses or any(not _is_public_address(address) for address in addresses):
                raise PublicWebError("nonpublic_address", (
                    "DNS returned a non-public address; direct public-web access was blocked before HTTP. "
                    "No website response was received, so this does not establish whether a repository "
                    "is private, missing or misspelled. Check DNS or configure a trusted HTTP proxy."
                ))
            remaining = None if timeout is None else max(0, timeout - (asyncio.get_running_loop().time() - started))
            candidates = _ConnectionCandidates(self._backend, addresses, port, remaining, local_address, socket_options)
            return _PublicStream(await candidates.connect(), candidates)


class _CoreStream(httpx.AsyncByteStream):
    def __init__(self, response: httpcore.Response) -> None:
        self._response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.aiter_stream():
            yield chunk

    async def aclose(self) -> None:
        await self._response.aclose()


class _PublicTransport(httpx.AsyncBaseTransport):
    def __init__(self, proxy_url: str | None = None) -> None:
        self._pool: httpcore.AsyncConnectionPool
        if proxy_url is None:
            self._pool = httpcore.AsyncConnectionPool(
                ssl_context=ssl.create_default_context(), network_backend=_PublicNetworkBackend(),
                retries=0, max_connections=10, max_keepalive_connections=0,
            )
        else:
            # Trusted upstreams resolve origins and own destination IP enforcement.
            self._pool = httpcore.AsyncHTTPProxy(
                proxy_url=proxy_url, ssl_context=ssl.create_default_context(),
                proxy_ssl_context=(ssl.create_default_context() if httpx.URL(proxy_url).scheme == "https" else None),
                network_backend=httpcore.AnyIOBackend(),
                retries=0, max_connections=10, max_keepalive_connections=0,
            )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert isinstance(request.stream, httpx.AsyncByteStream)
        async def trace(event: str, info: dict[str, object]) -> None:
            if event.endswith("start_tls.started"):
                _stage("tls_handshake")
            elif "send_request_" in event and event.endswith(".started"):
                _stage("http_request")
            elif "receive_response_" in event and event.endswith(".started"):
                _stage("http_response")
        request.extensions["trace"] = trace
        response = await self._pool.handle_async_request(
            httpcore.Request(
                method=request.method,
                url=httpcore.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=request.stream,
                extensions=request.extensions,
            )
        )
        return httpx.Response(
            response.status,
            headers=response.headers,
            stream=_CoreStream(response),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


def public_origin_referer(value: str, target: httpx.URL) -> str | None:
    """Preserve only a public source origin; never disclose URL paths or downgrade HTTPS."""
    source = validate_public_url(value)
    if source.scheme == "https" and target.scheme == "http":
        return None
    return str(source.copy_with(path="/", query=None, fragment=None))


class PublicWebClient:
    """Bounded public GET reads and trusted application-owned JSON POST calls.

    A transport may be supplied by trusted application code for deterministic
    tests. Tools must never expose that setting to the model. Redirects discard
    credentials and cookies; fixed authenticated API calls should disable them.
    gzip is decoded incrementally with an allocation bound. Other encodings are
    rejected. Direct connections pin public DNS answers. An explicit trusted proxy
    resolves origins upstream; no environment variables are read by this client.
    """

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        max_bytes: int = 2_000_000,
        timeout_seconds: float = 20,
        max_redirects: int = 5,
        proxy_url: str | None = None,
        no_proxy: str = "",
    ) -> None:
        if max_bytes < 1 or timeout_seconds <= 0 or max_redirects < 0:
            raise ValueError("HTTP limits must be positive and redirect limits nonnegative.")
        self._max_bytes = max_bytes
        self._timeout = timeout_seconds
        self._max_redirects = max_redirects
        self._proxy_url = None if proxy_url is None else validate_proxy_url(proxy_url)
        self._no_proxy = no_proxy
        self._transport = transport if transport is not None else _PublicTransport()
        self._proxy_transport = (
            _PublicTransport(self._proxy_url) if self._proxy_url is not None and transport is None else transport
        )

    def connection_mode(self, target: httpx.URL) -> str:
        hostname = f"[{target.host}]" if ":" in target.host else target.host
        host = f"{hostname}:{target.port or (443 if target.scheme == 'https' else 80)}"
        return ("trusted_proxy" if self._proxy_url is not None
                and not proxy_bypass_environment(host, {"no": self._no_proxy}) else "direct")

    async def get(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
        *,
        request_budget: Callable[[], None] | None = None,
        byte_budget: Callable[[int, int], None] | None = None,
        max_response_bytes: int | None = None,
        return_redirects: bool = False,
    ) -> FetchedResponse:
        """GET with optional trusted per-response limits; the client's default is unchanged."""
        return await self._request(
            "GET", url, headers, allow_redirects, request_budget, byte_budget, max_response_bytes,
            return_redirects=return_redirects,
        )

    async def post_json(
        self, url: str, body: Mapping[str, Any], *, headers: Mapping[str, str] | None = None,
    ) -> FetchedResponse:
        """POST for fixed API integrations, never model-directed page actions; redirects are forbidden."""
        content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(content) > self._max_bytes:
            raise PublicWebError("request_too_large", "The JSON request exceeds the byte limit.")
        request_headers = httpx.Headers(headers)
        request_headers["Content-Type"] = "application/json"
        return await self._request("POST", url, request_headers, False, None, None, None, content)

    async def _request(
        self, method: str, url: str, headers: Mapping[str, str] | None, allow_redirects: bool,
        request_budget: Callable[[], None] | None, byte_budget: Callable[[int, int], None] | None,
        max_response_bytes: int | None, content: bytes | None = None, *, return_redirects: bool = False,
    ) -> FetchedResponse:
        response_limit = self._max_bytes if max_response_bytes is None else max_response_bytes
        if response_limit < 1:
            raise ValueError('Response byte limit must be positive.')
        target = validate_public_url(url)
        request_headers = httpx.Headers(headers)
        request_headers["Accept-Encoding"] = "gzip, identity"
        progress = _RequestProgress()
        token = _progress.set(progress)
        try:
            async with asyncio.timeout(self._timeout):
                return await self._send(method, target, request_headers, allow_redirects, request_budget, byte_budget,
                                        response_limit, content, return_redirects)
        except (TimeoutError, httpx.TimeoutException, httpcore.TimeoutException):
            failure = PublicWebError("timeout", "The public HTTP request exceeded its deadline.")
            failure.failure_stage, failure.connection_mode = progress.stage, progress.mode
            raise failure from None
        except (httpx.HTTPError, httpcore.NetworkError, httpcore.ProtocolError, OSError):
            failure = PublicWebError("network_error", "The public HTTP request failed.")
            failure.failure_stage, failure.connection_mode = progress.stage, progress.mode
            raise failure from None
        finally:
            _progress.reset(token)

    async def _send(
        self, method: str, target: httpx.URL, headers: httpx.Headers, allow_redirects: bool,
        request_budget: Callable[[], None] | None, byte_budget: Callable[[int, int], None] | None,
        response_limit: int, content: bytes | None, return_redirects: bool,
    ) -> FetchedResponse:
        wire_bytes = 0
        for redirects in range(self._max_redirects + 1):
            if request_budget is not None:
                request_budget()
            # Own redirects and raw decoding rather than using HTTPX's client,
            # which eagerly parses Location even when redirects are disabled.
            request = httpx.Request(
                method,
                target,
                headers=headers,
                content=content,
                extensions={
                    "timeout": dict.fromkeys(("connect", "read", "write", "pool"), self._timeout),
                },
            )
            mode = self.connection_mode(target)
            progress = _progress.get()
            assert progress is not None
            progress.mode, progress.stage = mode, "connect"
            transport = self._proxy_transport if mode == "trusted_proxy" else self._transport
            assert transport is not None
            try:
                response = await transport.handle_async_request(request)
            except (httpx.TimeoutException, httpcore.TimeoutException, TimeoutError):
                failure = PublicWebError("timeout", "The public HTTP connection exceeded its deadline.")
                failure.failure_stage = progress.stage
                failure.connection_mode = mode
                raise failure from None
            except PublicWebError as error:
                error.connection_mode = mode
                raise
            except httpcore.ProxyError:
                failure = PublicWebError("proxy_connection_failed", "The configured proxy rejected the connection.")
                failure.failure_stage = "proxy_connect"
                failure.connection_mode = mode
                raise failure from None
            except (httpx.HTTPError, httpcore.NetworkError, httpcore.ProtocolError, OSError):
                failure = PublicWebError(
                    "network_error", "The public HTTP connection failed; no website response was received."
                )
                failure.failure_stage = progress.stage
                failure.connection_mode = mode
                raise failure from None
            try:
                response_headers = {
                    name: value for name in (
                        "access-control-allow-origin", "access-control-allow-credentials",
                        "access-control-expose-headers", "referrer-policy",
                    ) if (value := response.headers.get(name)) is not None
                    and len(value) <= 1024 and all(ord(char) >= 32 and ord(char) != 127 for char in value)
                }
                if 300 <= response.status_code < 400:
                    if return_redirects:
                        location = response.headers.get("Location")
                        if not location:
                            raise PublicWebError("invalid_redirect", "The redirect has no valid destination.")
                        try:
                            destination = validate_public_url(str(target.join(location)))
                        except httpx.InvalidURL:
                            raise PublicWebError("invalid_redirect", "The redirect has no valid destination.") from None
                        if len(str(destination)) > 4096:
                            raise PublicWebError("invalid_redirect", "The redirect has no valid destination.")
                        return FetchedResponse(
                            str(target), response.headers.get("Content-Type", ""), b"", wire_bytes, mode,
                            response_headers={**response_headers, "location": str(destination)},
                            status_code=response.status_code,
                            request_count=redirects + 1,
                        )
                    if not allow_redirects:
                        raise PublicWebError("redirect_not_allowed", "Redirects are disabled for this request.")
                    if redirects >= self._max_redirects:
                        raise PublicWebError(
                            "too_many_redirects", "The public HTTP request exceeded its redirect limit."
                        )
                    location = response.headers.get("Location")
                    if not location:
                        raise PublicWebError("invalid_redirect", "The redirect has no valid destination.")
                    try:
                        target = validate_public_url(str(target.join(location)))
                    except httpx.InvalidURL:
                        raise PublicWebError("invalid_redirect", "The redirect has no valid destination.") from None
                    headers = httpx.Headers(
                        {
                            key: value
                            for key, value in headers.items()
                            if key.lower() in {"accept", "accept-language", "accept-encoding", "user-agent"}
                        }
                    )
                    continue
                if not 200 <= response.status_code < 300:
                    raise PublicWebError("http_error", f"The remote server returned HTTP {response.status_code}.")
                _stage("http_response")
                body, wire_bytes = await self._read_body(response, wire_bytes, byte_budget, response_limit)
                # Preserve actual bounded browser policies, never cookies or Content-Encoding.
                return FetchedResponse(str(target), response.headers.get("Content-Type", ""), body, wire_bytes, mode,
                                       response_headers=response_headers, status_code=response.status_code,
                                       request_count=redirects + 1)
            except PublicWebError as error:
                error.connection_mode = mode
                raise
            except (httpx.TimeoutException, httpcore.TimeoutException, TimeoutError):
                failure = PublicWebError("timeout", "The public HTTP response exceeded its deadline.")
                failure.failure_stage = "http_response"
                failure.connection_mode = mode
                raise failure from None
            finally:
                await response.aclose()
        raise AssertionError("Redirect loop must return or raise.")

    async def _read_body(
        self, response: httpx.Response, wire_bytes: int,
        byte_budget: Callable[[int, int], None] | None = None,
        response_limit: int | None = None,
    ) -> tuple[bytes, int]:
        limit = self._max_bytes if response_limit is None else response_limit
        encoding = response.headers.get("Content-Encoding", "identity").lower().strip()
        if encoding not in {"", "identity", "gzip"}:
            raise PublicWebError("unsupported_encoding", "The response uses an unsupported content encoding.")
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None
        if response.is_stream_consumed:
            # Trusted in-process test transports may construct cached identity
            # responses. Compressed responses must always provide a raw stream.
            if decoder is not None:
                raise PublicWebError("invalid_encoding", "The compressed response is not a raw byte stream.")
            content = response.content
            wire_bytes += len(content)
            if byte_budget is not None:
                byte_budget(len(content), len(content))
            if wire_bytes > limit:
                raise PublicWebError("response_too_large", "The response exceeds the byte limit.")
            return content, wire_bytes
        body = bytearray()
        try:
            async for chunk in response.aiter_raw():
                wire_bytes += len(chunk)
                if byte_budget is not None:
                    byte_budget(len(chunk), 0)
                if wire_bytes > limit:
                    raise PublicWebError("response_too_large", "The response exceeds the byte limit.")
                if decoder is not None:
                    chunk = decoder.decompress(chunk, limit - len(body) + 1)
                if byte_budget is not None:
                    byte_budget(0, len(chunk))
                body.extend(chunk)
                if len(body) > limit:
                    raise PublicWebError("response_too_large", "The response exceeds the byte limit.")
            if decoder is not None and (not decoder.eof or decoder.unused_data):
                raise PublicWebError("invalid_encoding", "The gzip response is incomplete or malformed.")
        except zlib.error:
            raise PublicWebError("invalid_encoding", "The gzip response is incomplete or malformed.") from None
        return bytes(body), wire_bytes

    async def aclose(self) -> None:
        try:
            await self._transport.aclose()
        finally:
            if self._proxy_transport is not None and self._proxy_transport is not self._transport:
                await self._proxy_transport.aclose()

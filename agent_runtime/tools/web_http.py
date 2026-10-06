"""Bounded public HTTP reads, with DNS checks at the actual socket connection.

URL prechecks never resolve DNS, so tools can run them before permission approval.
The default transport checks every DNS answer, connects only to a checked numeric
address, and leaves the origin hostname intact for HTTP Host and verified TLS.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import ssl
import zlib
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
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


# Matches httpcore's public backend signature, without importing private modules.
SocketOption = tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]


class _PublicStream(httpcore.AsyncNetworkStream):
    """Close a connected socket if TLS setup fails or is cancelled."""

    def __init__(self, stream: httpcore.AsyncNetworkStream) -> None:
        self._stream = stream

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
        try:
            return _PublicStream(await self._stream.start_tls(ssl_context, server_hostname, timeout))
        except BaseException:
            await self._stream.aclose()
            raise

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
            stream = await self._backend.connect_tcp(
                addresses[0],
                port,
                timeout=timeout,
                local_address=local_address,
                socket_options=socket_options,
            )
            return _PublicStream(stream)


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


class PublicWebClient:
    """GET-only public web client with a deadline and separate wire/decoded limits.

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
    ) -> FetchedResponse:
        target = validate_public_url(url)
        request_headers = httpx.Headers(headers)
        request_headers["Accept-Encoding"] = "gzip, identity"
        try:
            async with asyncio.timeout(self._timeout):
                return await self._get(target, request_headers, allow_redirects)
        except TimeoutError:
            raise PublicWebError("timeout", "The public HTTP request exceeded its deadline.") from None
        except (httpx.TimeoutException, httpcore.TimeoutException):
            raise PublicWebError("timeout", "The public HTTP request exceeded its deadline.") from None
        except (httpx.HTTPError, httpcore.NetworkError, httpcore.ProtocolError, OSError):
            raise PublicWebError("network_error", "The public HTTP request failed.") from None

    async def _get(self, target: httpx.URL, headers: httpx.Headers, allow_redirects: bool) -> FetchedResponse:
        wire_bytes = 0
        for redirects in range(self._max_redirects + 1):
            # Own redirects and raw decoding rather than using HTTPX's client,
            # which eagerly parses Location even when redirects are disabled.
            request = httpx.Request(
                "GET",
                target,
                headers=headers,
                extensions={
                    "timeout": dict.fromkeys(("connect", "read", "write", "pool"), self._timeout),
                },
            )
            mode = self.connection_mode(target)
            transport = self._proxy_transport if mode == "trusted_proxy" else self._transport
            assert transport is not None
            try:
                response = await transport.handle_async_request(request)
            except (httpx.TimeoutException, httpcore.TimeoutException, TimeoutError):
                failure = PublicWebError("timeout", "The public HTTP connection exceeded its deadline.")
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
                failure.connection_mode = mode
                raise failure from None
            try:
                if 300 <= response.status_code < 400:
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
                body, wire_bytes = await self._read_body(response, wire_bytes)
                return FetchedResponse(str(target), response.headers.get("Content-Type", ""), body, wire_bytes, mode)
            except PublicWebError as error:
                error.connection_mode = mode
                raise
            except (httpx.TimeoutException, httpcore.TimeoutException, TimeoutError):
                failure = PublicWebError("timeout", "The public HTTP response exceeded its deadline.")
                failure.connection_mode = mode
                raise failure from None
            finally:
                await response.aclose()
        raise AssertionError("Redirect loop must return or raise.")

    async def _read_body(self, response: httpx.Response, wire_bytes: int) -> tuple[bytes, int]:
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
            if wire_bytes > self._max_bytes:
                raise PublicWebError("response_too_large", "The response exceeds the byte limit.")
            return content, wire_bytes
        body = bytearray()
        try:
            async for chunk in response.aiter_raw():
                wire_bytes += len(chunk)
                if wire_bytes > self._max_bytes:
                    raise PublicWebError("response_too_large", "The response exceeds the byte limit.")
                if decoder is not None:
                    chunk = decoder.decompress(chunk, self._max_bytes - len(body) + 1)
                body.extend(chunk)
                if len(body) > self._max_bytes:
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

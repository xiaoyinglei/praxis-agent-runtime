from __future__ import annotations

import asyncio
import gzip
import socket
import ssl
from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpcore
import httpx
import pytest

from agent_runtime.tools.web_http import PublicWebClient, PublicWebError, validate_public_url


class _Stream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay: float = 0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.closed = False
        self.started = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com",
        "https://u:secret@example.com/?secret=1",
        "http://localhost",
        "http://localhost.",
        "http://a.localhost",
        "http://127.0.0.1",
        "http://10.1.2.3",
        "http://169.254.169.254",
        "http://[::1]",
        "http://[fc00::1]",
        "http://[::ffff:127.0.0.1]",
        "http://[fe80::1%25en0]",
        "https://example.com:8443",
        "https:///missing-host",
        "http://224.0.0.1",
        "https://example.com\n",
    ],
)
def test_rejects_nonpublic_or_invalid_url(url: str) -> None:
    with pytest.raises(PublicWebError) as caught:
        validate_public_url(url)
    assert caught.value.code == "invalid_url"
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("url", ["https://example.com/path?q=1", "http://8.8.8.8", "https://[2606:4700:4700::1111]"])
def test_url_precheck_does_not_resolve_dns(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_dns(*args: object, **kwargs: object) -> None:
        raise AssertionError("DNS must follow tool approval")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_dns)
    assert str(validate_public_url(url)) == url


def test_fetch_reads_raw_gzip_and_counts_network_bytes() -> None:
    async def run() -> None:
        raw = gzip.compress(b"hello public web")
        stream = _Stream([raw[:10], raw[10:]])
        client = PublicWebClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    headers={"Content-Type": "text/plain; charset=utf-8", "Content-Encoding": "gzip"},
                    stream=stream,
                )
            )
        )
        try:
            result = await client.get("https://example.com")
            assert result.body == b"hello public web"
            assert result.network_bytes == len(raw)
            assert result.content_type == "text/plain; charset=utf-8"
            assert result.url == "https://example.com"
            assert stream.closed
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("compressed", [False, True])
def test_bounds_wire_and_decoded_bytes(compressed: bool) -> None:
    async def run() -> None:
        raw = gzip.compress(b"x" * 1_000_000) if compressed else b"x" * 129
        stream = _Stream([raw])
        client = PublicWebClient(
            max_bytes=128,
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    headers={"Content-Encoding": "gzip"} if compressed else {},
                    stream=stream,
                )
            ),
        )
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "response_too_large"
            assert stream.closed
        finally:
            await client.aclose()

    asyncio.run(run())


def test_decompression_limit_applies_when_wire_body_fits() -> None:
    async def run() -> None:
        stream = _Stream([gzip.compress(b"x" * 100_000)])
        client = PublicWebClient(
            max_bytes=200,
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    headers={"Content-Encoding": "gzip"},
                    stream=stream,
                )
            ),
        )
        try:
            with pytest.raises(PublicWebError, match="limit") as caught:
                await client.get("https://example.com")
            assert caught.value.code == "response_too_large"
        finally:
            await client.aclose()

    asyncio.run(run())


def test_redirect_revalidates_private_destination_and_closes_response() -> None:
    async def run() -> None:
        requests: list[str] = []
        stream = _Stream([b"untrusted redirect body"])

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            return httpx.Response(302, headers={"Location": "http://127.0.0.1/secret"}, stream=stream)

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(PublicWebError):
                await client.get("https://example.com")
            assert requests == ["https://example.com"]
            assert stream.closed
        finally:
            await client.aclose()

    asyncio.run(run())


def test_redirect_drops_credentials_and_cookies_even_on_same_host() -> None:
    async def run() -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(
                    302, headers={"Location": "/next", "Set-Cookie": "private=secret"}, stream=_Stream([])
                )
            return httpx.Response(200, stream=_Stream([b"ok"]))

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            await client.get(
                "https://example.com",
                headers={
                    "Authorization": "Bearer secret",
                    "X-Subscription-Token": "secret",
                    "Cookie": "token=secret",
                },
            )
            assert len(seen) == 2
            assert seen[0].headers["x-subscription-token"] == "secret"
            assert all(key not in seen[1].headers for key in ["authorization", "x-subscription-token", "cookie"])
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("follow,limit,expected", [(False, 5, "redirect_not_allowed"), (True, 1, "too_many_redirects")])
def test_redirect_policy(follow: bool, limit: int, expected: str) -> None:
    async def run() -> None:
        count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal count
            count += 1
            return httpx.Response(302, headers={"Location": "/next"}, stream=_Stream([]))

        client = PublicWebClient(max_redirects=limit, transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com", allow_redirects=follow)
            assert caught.value.code == expected
            assert count == (1 if not follow else 2)
        finally:
            await client.aclose()

    asyncio.run(run())


def test_overall_deadline_closes_stream() -> None:
    async def run() -> None:
        stream = _Stream([b"a", b"b"], delay=0.03)
        client = PublicWebClient(
            timeout_seconds=0.05, transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream))
        )
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com/?secret=query")
            assert caught.value.code == "timeout"
            assert "secret" not in str(caught.value)
            assert stream.closed
        finally:
            await client.aclose()

    asyncio.run(run())


def test_cancellation_propagates_and_closes_stream() -> None:
    async def run() -> None:
        stream = _Stream([b"a"], delay=60)
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream)))
        try:
            task = asyncio.create_task(client.get("https://example.com"))
            await stream.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stream.closed
        finally:
            await client.aclose()

    asyncio.run(run())


class _SocketStream(httpcore.AsyncNetworkStream):
    def __init__(self) -> None:
        self.server_hostname: str | None = None
        self.closed = False
        self.writes: list[bytes] = []
        self.response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        result, self.response = self.response, b""
        return result

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.writes.append(buffer)

    async def aclose(self) -> None:
        self.closed = True

    async def start_tls(
        self, ssl_context: ssl.SSLContext, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore.AsyncNetworkStream:
        self.server_hostname = server_hostname
        return self

    def get_extra_info(self, info: str) -> Any:
        return None


class _NumericBackend(httpcore.AsyncNetworkBackend):
    def __init__(self) -> None:
        self.hosts: list[str] = []
        self.stream = _SocketStream()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[tuple[int, int, int | bytes]] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.hosts.append(host)
        return self.stream


def test_default_transport_pins_dns_and_retains_tls_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        calls: list[str] = []
        backend = _NumericBackend()

        async def resolve(host: str, port: int, **kwargs: object) -> list[tuple[Any, ...]]:
            calls.append(host)
            ip = "93.184.216.34" if len(calls) == 1 else "127.0.0.1"
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
        monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
        client = PublicWebClient()
        try:
            assert (await client.get("https://example.com")).body == b"ok"
            assert calls == ["example.com"]
            assert backend.hosts == ["93.184.216.34"]
            assert backend.stream.server_hostname == "example.com"
            assert b"Host: example.com" in b"".join(backend.stream.writes)
            assert backend.stream.closed
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("private", ["10.0.0.1", "::1", "::ffff:127.0.0.1", "fc00::1"])
def test_dns_rejects_entire_answer_if_one_address_is_nonpublic(private: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        backend = _NumericBackend()

        async def resolve(host: str, port: int, **kwargs: object) -> list[tuple[Any, ...]]:
            family = socket.AF_INET6 if ":" in private else socket.AF_INET
            return [
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port)),
                (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (private, port)),
            ]

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
        monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
        client = PublicWebClient()
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "nonpublic_address"
            assert backend.hosts == []
        finally:
            await client.aclose()

    asyncio.run(run())


def test_cancellation_during_tls_closes_connected_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        backend = _NumericBackend()
        started = asyncio.Event()

        async def hanging_tls(
            ssl_context: ssl.SSLContext, server_hostname: str | None = None, timeout: float | None = None
        ) -> httpcore.AsyncNetworkStream:
            started.set()
            await asyncio.sleep(60)
            return backend.stream

        monkeypatch.setattr(backend.stream, "start_tls", hanging_tls)
        monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
        client = PublicWebClient()
        try:
            task = asyncio.create_task(client.get("https://8.8.8.8"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert backend.stream.closed
        finally:
            await client.aclose()

    asyncio.run(run())


def test_malformed_redirect_returns_safe_error() -> None:
    async def run() -> None:
        client = PublicWebClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    302,
                    headers={"Location": "https://example.com:bad/?secret=credential"},
                    stream=_Stream([]),
                )
            )
        )
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "invalid_redirect"
            assert "secret" not in str(caught.value)
        finally:
            await client.aclose()

    asyncio.run(run())


def test_preconsumed_identity_fixture_is_bounded() -> None:
    async def run() -> None:
        client = PublicWebClient(
            max_bytes=10, transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"ok"))
        )
        try:
            assert (await client.get("https://example.com")).body == b"ok"
        finally:
            await client.aclose()
        client = PublicWebClient(
            max_bytes=1, transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"ok"))
        )
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "response_too_large"
        finally:
            await client.aclose()

    asyncio.run(run())


def test_public_ipv6_literal_connects_numerically_without_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        backend = _NumericBackend()

        async def unexpected_dns(*args: object, **kwargs: object) -> None:
            raise AssertionError("Numeric IPv6 must not resolve DNS")

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", unexpected_dns)
        monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
        client = PublicWebClient()
        try:
            assert (await client.get("https://[2606:4700:4700::1111]")).body == b"ok"
            assert backend.hosts == ["2606:4700:4700::1111"]
            assert backend.stream.server_hostname == "2606:4700:4700::1111"
        finally:
            await client.aclose()

    asyncio.run(run())


def test_new_connection_rechecks_rebound_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        backend = _NumericBackend()
        count = 0

        async def resolve(host: str, port: int, **kwargs: object) -> list[tuple[Any, ...]]:
            nonlocal count
            count += 1
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("93.184.216.34" if count == 1 else "127.0.0.1", port),
                )
            ]

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
        monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
        client = PublicWebClient()
        try:
            assert (await client.get("https://example.com")).body == b"ok"
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "nonpublic_address"
            assert backend.hosts == ["93.184.216.34"]
            assert count == 2
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "encoding,body,code",
    [
        ("gzip", b"invalid upstream body with secret", "invalid_encoding"),
        ("gzip", gzip.compress(b"ok")[:-3], "invalid_encoding"),
        ("gzip", gzip.compress(b"ok") + b"extraneous", "invalid_encoding"),
        ("br", b"secret", "unsupported_encoding"),
    ],
)
def test_rejects_malformed_or_unsupported_encoded_body(encoding: str, body: bytes, code: str) -> None:
    async def run() -> None:
        client = PublicWebClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    headers={"Content-Encoding": encoding},
                    stream=_Stream([body]),
                )
            )
        )
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == code
            assert "secret" not in str(caught.value)
        finally:
            await client.aclose()

    asyncio.run(run())


def test_network_errors_and_upstream_errors_are_sanitized() -> None:
    async def run() -> None:
        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("https://u:secret@example.com/?token=secret")

        client = PublicWebClient(transport=httpx.MockTransport(fail))
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "network_error"
            assert "secret" not in str(caught.value)
        finally:
            await client.aclose()
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: httpx.Response(500, content=b"secret body")))
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "http_error"
            assert "secret" not in str(caught.value)
        finally:
            await client.aclose()

    asyncio.run(run())


def test_dns_resolution_is_inside_overall_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        backend = _NumericBackend()
        cancelled = False

        async def resolve(host: str, port: int, **kwargs: object) -> list[tuple[Any, ...]]:
            nonlocal cancelled
            try:
                await asyncio.sleep(60)
                return []
            finally:
                cancelled = True

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
        monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
        client = PublicWebClient(timeout_seconds=0.02)
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "timeout"
            assert cancelled
            assert backend.hosts == []
        finally:
            await client.aclose()

    asyncio.run(run())


def test_deadline_spans_redirects_instead_of_restarting_per_hop() -> None:
    async def run() -> None:
        requests = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            await asyncio.sleep(0.03)
            return httpx.Response(302, headers={"Location": "/next"}, stream=_Stream([]))

        client = PublicWebClient(timeout_seconds=0.05, transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "timeout"
            assert requests == 2
        finally:
            await client.aclose()

    asyncio.run(run())

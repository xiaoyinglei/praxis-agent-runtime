"""Regression coverage for DNS-pinned TCP/TLS failover and phase reporting."""
import asyncio
import socket
import ssl
import time

import httpcore
import httpx
import pytest

from agent_runtime.tools.web_http import PublicWebClient, PublicWebError
from tests.agent.test_web_http import _NumericBackend, _SocketStream, _Stream

IPS = ["185.199.108.133", "185.199.109.133", "185.199.110.133"]


def run_with_backend(monkeypatch, exercise, failures=None, addresses=IPS):
    async def run():
        class Backend(_NumericBackend):
            def __init__(self):
                super().__init__()
                self.streams = []
                self.contexts = []

            async def connect_tcp(self, host, port, **kwargs):
                self.hosts.append(host)
                outcome = (failures or {}).get(host)
                if outcome == "tcp":
                    raise httpcore.ConnectError("private secret error")
                if outcome == "tcp_hang":
                    await asyncio.sleep(60)
                stream = _SocketStream()
                self.streams.append(stream)
                if outcome == "tls_close":
                    async def broken_close():
                        stream.closed = True
                        raise OSError("close failed")
                    stream.aclose = broken_close

                async def tls(context, server_hostname=None, timeout=None):
                    self.contexts.append(context)
                    stream.server_hostname = server_hostname
                    if outcome in {"tls", "tls_close"}:
                        raise httpcore.ConnectError("private secret TLS error")
                    if outcome == "cert":
                        raise httpcore.ConnectError(ssl.SSLCertVerificationError("bad certificate"))
                    if outcome == "tls_hang":
                        await asyncio.sleep(60)
                    return stream

                stream.start_tls = tls
                return stream

        backend = Backend()

        async def resolve(host, port, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port)) for ip in addresses]

        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
        monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
        await exercise(backend)

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["tcp", "tls", "tls_close"])
def test_fails_over_to_next_verified_ip_without_replaying_http(monkeypatch, phase):
    async def exercise(backend):
        client = PublicWebClient()
        try:
            assert (await client.get("https://example.com")).body == b"ok"
            assert backend.hosts == IPS[:2]
            assert all(s.closed for s in backend.streams)
            assert backend.streams[-1].server_hostname == "example.com"
            assert all(c.verify_mode == ssl.CERT_REQUIRED and c.check_hostname for c in backend.contexts)
            assert b"".join(b"".join(s.writes) for s in backend.streams).count(b"GET /") == 1
        finally:
            await client.aclose()

    run_with_backend(monkeypatch, exercise, {IPS[0]: phase})


@pytest.mark.parametrize(
    "phase,expected", [("tcp", "tcp_connect"), ("tls", "tls_handshake"), ("cert", "tls_handshake")]
)
def test_exhausted_addresses_report_actual_failure_stage(monkeypatch, phase, expected):
    async def exercise(backend):
        client = PublicWebClient()
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com/?secret=value")
            assert caught.value.code == "network_error"
            assert caught.value.failure_stage == expected
            assert caught.value.connection_mode == "direct"
            assert "secret" not in str(caught.value)
            assert all(s.closed for s in backend.streams)
        finally:
            await client.aclose()

    run_with_backend(monkeypatch, exercise, dict.fromkeys(IPS, phase))


def test_all_dns_answers_validated_before_attempt_limit(monkeypatch):
    async def exercise(backend):
        client = PublicWebClient()
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.code == "nonpublic_address"
            assert not backend.hosts
        finally:
            await client.aclose()

    run_with_backend(monkeypatch, exercise, addresses=IPS * 4 + ["127.0.0.1"])


@pytest.mark.parametrize("phase", ["tcp_hang", "tls_hang"])
def test_hanging_candidate_leaves_time_for_other_addresses(monkeypatch, phase):
    async def exercise(backend):
        client = PublicWebClient(timeout_seconds=0.15)
        start = time.monotonic()
        try:
            assert (await client.get("https://example.com")).body == b"ok"
            assert backend.hosts == IPS[:2]
            assert time.monotonic() - start < 0.3
            assert all(s.closed for s in backend.streams)
        finally:
            await client.aclose()

    run_with_backend(monkeypatch, exercise, {IPS[0]: phase})


def test_body_timeout_reports_http_and_connection_mode():
    async def run():
        client = PublicWebClient(timeout_seconds=0.02, transport=httpx.MockTransport(
            lambda r: httpx.Response(200, stream=_Stream([b"ok"], delay=60))))
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.failure_stage == "http_response"
            assert caught.value.connection_mode == "direct"
        finally:
            await client.aclose()
    asyncio.run(run())


def test_dns_failure_has_dns_stage(monkeypatch):
    async def run():
        async def fail(*args, **kwargs):
            raise socket.gaierror("secret hostname")
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", fail)
        client = PublicWebClient()
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.failure_stage == "dns_resolution"
            assert "secret" not in str(caught.value)
        finally:
            await client.aclose()
    asyncio.run(run())


def test_dns_deadline_keeps_stage_and_mode(monkeypatch):
    async def run():
        async def hang(*args, **kwargs):
            await asyncio.sleep(60)
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", hang)
        client = PublicWebClient(timeout_seconds=0.02)
        try:
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com")
            assert caught.value.failure_stage == "dns_resolution"
            assert caught.value.connection_mode == "direct"
        finally:
            await client.aclose()
    asyncio.run(run())


def test_cancellation_stops_fallback(monkeypatch):
    async def exercise(backend):
        client = PublicWebClient()
        try:
            task = asyncio.create_task(client.get("https://example.com"))
            while not backend.streams:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert backend.hosts == IPS[:1]
            assert backend.streams[0].closed
        finally:
            await client.aclose()
    run_with_backend(monkeypatch, exercise, dict.fromkeys(IPS, "tls_hang"))


def test_fallback_deduplicates_and_limits_attempts(monkeypatch):
    addresses = [f"8.8.8.{i}" for i in range(1, 12)]
    async def exercise(backend):
        client = PublicWebClient()
        try:
            with pytest.raises(PublicWebError):
                await client.get("https://example.com")
            assert backend.hosts == addresses[:8]
        finally:
            await client.aclose()
    run_with_backend(monkeypatch, exercise, dict.fromkeys(addresses, "tcp"), addresses=addresses * 2)


def test_concurrent_failures_do_not_share_connection_diagnostics():
    async def run():
        async def response(request):
            if request.url.host == "proxy.example":
                await asyncio.sleep(0.005)
                raise httpx.ConnectError("private")
            return httpx.Response(200, stream=_Stream([b"ok"], delay=60))
        client = PublicWebClient(timeout_seconds=0.03, proxy_url="http://proxy.example:8080",
                                 no_proxy="direct.example", transport=httpx.MockTransport(response))
        async def fetch(host):
            try:
                await client.get("https://" + host)
            except PublicWebError as error:
                return error.failure_stage, error.connection_mode
        try:
            results = await asyncio.gather(fetch("proxy.example"), fetch("direct.example"))
            assert results == [("connect", "trusted_proxy"), ("http_response", "direct")]
        finally:
            await client.aclose()
    asyncio.run(run())

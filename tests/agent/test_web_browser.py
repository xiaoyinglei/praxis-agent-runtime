"""Browser boundary tests; real Chromium tests exercise JavaScript separately."""

import asyncio
import base64
import importlib
import os
import socket
import subprocess
import sys
import uuid
from types import SimpleNamespace

import httpx
import pytest

from agent_runtime.tools.web_http import PublicWebClient, PublicWebError


def browser_module():
    return importlib.import_module("agent_runtime.tools.web_browser")


@pytest.mark.parametrize("status", [200, 201, 202, 204, 206])
def test_broker_preserves_actual_successful_http_status(status):
    async def run():
        body = b"" if status == 204 else b"public response"
        client = PublicWebClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(status, content=body, headers={"Content-Type": "text/plain"})
        ))
        try:
            reply = await browser_module()._Broker(client).reply({
                "type": "fetch", "method": "GET", "url": "https://example.com/api", "resource_type": "Fetch",
            })
            assert reply["status"] == status
            assert base64.b64decode(reply["body"]) == body
        finally:
            await client.aclose()

    asyncio.run(run())


def test_missing_optional_renderer_has_explicit_error(monkeypatch):
    browser = browser_module()
    monkeypatch.setattr(browser.importlib.util, "find_spec", lambda name: None)

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: pytest.fail("HTTP must not run")))
        try:
            with pytest.raises(PublicWebError) as caught:
                await browser.render_public_page(client, "https://example.com")
            assert caught.value.code == "web_browser_unavailable"
        finally:
            await client.aclose()

    asyncio.run(run())


def test_macos_unavailable_before_any_http_or_worker(monkeypatch):
    browser = browser_module()
    monkeypatch.setattr(browser.sys, "platform", "darwin")
    monkeypatch.setattr(browser.importlib.util, "find_spec", lambda name: object())

    async def unexpected_worker(*args, **kwargs):
        pytest.fail("Unsupported macOS renderer must not start a worker")

    monkeypatch.setattr(browser.asyncio, "create_subprocess_exec", unexpected_worker)

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: pytest.fail("HTTP must not run")))
        try:
            with pytest.raises(PublicWebError) as caught:
                await browser.render_public_page(client, "https://example.com")
            assert caught.value.code == "web_browser_unavailable"
        finally:
            await client.aclose()

    asyncio.run(run())


def test_worker_environment_excludes_credentials(monkeypatch, tmp_path):
    browser = browser_module()
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "https://u:secret@example.com")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    environment = browser._worker_environment(str(tmp_path))
    assert environment["HOME"] == str(tmp_path)
    assert "secret" not in repr(environment)
    assert "PYTHONPATH" not in environment
    assert "HTTPS_PROXY" not in environment


def test_linux_probe_accepts_private_namespace_with_loopback_refused(monkeypatch):
    browser = browser_module()
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.setenv("PRAXIS_PARENT_NETNS", "100")
    original_stat = os.stat
    monkeypatch.setattr(browser.os, "stat", lambda path, **kwargs:
                        SimpleNamespace(st_ino=200) if str(path) == "/proc/self/ns/net"
                        else original_stat(path, **kwargs))

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, timeout):
            pass

        def connect(self, target):
            if target[0] in {"127.0.0.1", "::1"}:
                raise ConnectionRefusedError(111, "no listener on isolated loopback")
            raise OSError(101, "isolated namespace has no external route")

    monkeypatch.setattr(browser.socket, "socket", lambda *args: Probe())
    browser._verify_network_denied()
    monkeypatch.setenv("PRAXIS_PARENT_NETNS", "200")
    with pytest.raises(browser._WorkerError):
        browser._verify_network_denied()


def test_broker_counts_redirects_and_decoded_bytes():
    browser = browser_module()

    async def run():
        seen = []

        def handler(request):
            seen.append(str(request.url))
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "/final"})
            return httpx.Response(200, content=b"done", headers={"Content-Type": "text/html"})

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        broker = browser._Broker(client)
        try:
            initial = await broker.fetch("https://example.com/start")
            assert initial.url == "https://example.com/final"
            assert broker.budget.requests == 2
            assert broker.budget.wire_bytes == 4
            assert broker.budget.decoded_bytes == 4
            broker.budget.requests = browser.MAX_REQUESTS
            with pytest.raises(PublicWebError) as caught:
                await broker.fetch("https://example.com/other")
            assert caught.value.code == "web_browser_budget_exceeded"
            assert broker.budget.requests == browser.MAX_REQUESTS
            assert len(seen) == 2
        finally:
            await client.aclose()

    asyncio.run(run())


def test_broker_64_request_limit_stops_before_65th_transport_call():
    browser = browser_module()
    assert browser.MAX_REQUESTS == 64

    async def run():
        seen = []

        def handler(request):
            seen.append(request.url)
            return httpx.Response(200, content=b"ok")

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        broker = browser._Broker(client)
        try:
            for _ in range(64):
                await broker.reply({"type": "fetch", "method": "GET", "url": "https://example.com/script.js",
                                    "resource_type": "Script"})
            assert await broker.reply({"type": "fetch", "method": "GET", "url": "https://example.com/script.js",
                                       "resource_type": "Script"}) == {'type': 'abort'}
            assert broker.blocked_actions['web_browser_budget_exceeded'] == 1
            assert broker.budget.requests == len(seen) == 64
            assert 'web_browser_budget_exceeded' in broker.warning
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("resource_type", ["Script", "Stylesheet", "XHR", "Fetch"])
def test_broker_aborts_optional_network_failures_with_bounded_warning(resource_type):
    browser = browser_module()

    async def run():
        def handler(request):
            raise httpx.ConnectError("offline", request=request)

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        broker = browser._Broker(client)
        try:
            for _ in range(2):
                reply = await broker.reply({"type": "fetch", "method": "GET",
                                            "url": "https://example.com/api?secret=hidden",
                                            "resource_type": resource_type})
                assert reply == {"type": "abort"}
            assert f"network_error:{resource_type}=2" in broker.warning
            assert "secret" not in broker.warning
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("code", ["nonpublic_address", "invalid_url", "response_too_large",
                                  "web_browser_budget_exceeded", "web_browser_protocol_error",
                                  "permission_denied", "network_error"])
def test_broker_document_and_security_errors_remain_hard_failures(code):
    browser = browser_module()

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        broker = browser._Broker(client)

        async def failing_fetch(*args):
            raise PublicWebError(code, "fixture")

        broker.fetch = failing_fetch
        try:
            kinds = (["Document", "Script", "XHR"]
                     if code in {"web_browser_protocol_error", "permission_denied"} else ["Document"])
            for resource_type in kinds:
                with pytest.raises(PublicWebError) as caught:
                    await broker.reply({"type": "fetch", "method": "GET", "url": "https://example.com/a",
                                        "resource_type": resource_type})
                assert caught.value.code == code
        finally:
            await client.aclose()

    asyncio.run(run())


def test_broker_redirects_before_final_body_and_never_fetches_private_url():
    browser = browser_module()

    async def run():
        seen = []

        def handler(request):
            seen.append(str(request.url))
            if request.url.host == "example.com":
                return httpx.Response(302, headers={"Location": "https://other.example/final.js"})
            return httpx.Response(200, content=b"script", headers={"Content-Type": "text/javascript"})

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        broker = browser._Broker(client)
        try:
            redirect = await broker.reply({"type": "fetch", "method": "GET", "url": "https://example.com/a.js"})
            assert redirect["status"] == 302
            assert redirect["headers"]["location"] == "https://other.example/final.js"
            assert "body" not in redirect
            final = await broker.reply({"type": "fetch", "method": "GET", "url": "https://other.example/final.js"})
            assert base64.b64decode(final["body"]) == b"script"
            assert len(seen) == 2
            for method, url in [("POST", "https://example.com"), ("GET", "file:///etc/passwd"),
                                ("GET", "http://127.0.0.1/secret")]:
                with pytest.raises(PublicWebError):
                    await broker.reply({"type": "fetch", "method": method, "url": url})
            assert len(seen) == 2
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("resource_type,expected", [("Script", None), ("Document", "response_too_large")])
def test_script_response_limit_preserves_default_document_limit(resource_type, expected):
    browser = browser_module()

    async def run():
        body = b"x" * 2_000_001
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
        broker = browser._Broker(client)
        try:
            request = {"type": "fetch", "method": "GET", "url": "https://example.com/resource",
                       "resource_type": resource_type}
            if expected:
                with pytest.raises(PublicWebError) as caught:
                    await broker.reply(request)
                assert caught.value.code == expected
            else:
                response = await broker.reply(request)
                assert base64.b64decode(response["body"]) == body
                assert broker.budget.decoded_bytes == len(body)
            with pytest.raises(PublicWebError) as caught:
                await client.get("https://example.com/document")
            assert caught.value.code == "response_too_large"
        finally:
            await client.aclose()

    asyncio.run(run())


def test_script_response_still_obeys_aggregate_eight_mb_budget():
    browser = browser_module()

    async def run():
        body = b"x" * 4_000_001
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
        broker = browser._Broker(client)
        request = {"type": "fetch", "method": "GET", "url": "https://example.com/script.js", "resource_type": "Script"}
        try:
            await broker.reply(request)
            assert await broker.reply(request) == {'type': 'abort'}
            assert broker.blocked_actions['web_browser_budget_exceeded'] == 1
            assert broker.budget.requests == 2
            assert await broker.reply(request) == {'type': 'abort'}
            assert broker.budget.requests == 2
        finally:
            await client.aclose()

    asyncio.run(run())


def test_redirect_cache_does_not_reuse_large_script_as_document():
    browser = browser_module()

    async def run():
        body = b"x" * 2_000_001

        def handler(request):
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "/final"})
            return httpx.Response(200, content=body)

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        broker = browser._Broker(client)
        try:
            redirect = await broker.reply({"type": "fetch", "method": "GET", "url": "https://example.com/start",
                                           "resource_type": "Script"})
            assert redirect["status"] == 302
            with pytest.raises(PublicWebError) as caught:
                await broker.reply({"type": "fetch", "method": "GET", "url": "https://example.com/final",
                                    "resource_type": "Document"})
            assert caught.value.code == "response_too_large"
        finally:
            await client.aclose()

    asyncio.run(run())


def test_byte_and_protocol_limits():
    browser = browser_module()
    budget = browser._Budget()
    budget.on_bytes(browser.MAX_NETWORK_BYTES, browser.MAX_NETWORK_BYTES)
    with pytest.raises(PublicWebError) as caught:
        budget.on_bytes(0, 1)
    assert caught.value.code == "web_browser_budget_exceeded"
    for value in [b"not json\n", b"[]\n", b'{"type":"unknown"}\n']:
        with pytest.raises(PublicWebError):
            browser._decode_message(value)


def test_render_deadline_includes_initial_http_fetch(monkeypatch):
    browser = browser_module()
    monkeypatch.setattr(browser.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(browser, "RENDER_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(browser, "_sandbox_command", lambda command, directory: command)
    monkeypatch.setattr(browser.sys, "platform", "linux")

    async def run():
        closed = False

        async def handler(request):
            nonlocal closed
            try:
                await asyncio.sleep(60)
            finally:
                closed = True
            return httpx.Response(200, content=b"unused")

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(PublicWebError) as caught:
                await browser.render_public_page(client, "https://example.com")
            assert caught.value.code == "web_browser_timeout"
            assert closed
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_worker_cleanup_on_timeout_and_cancellation(monkeypatch, tmp_path, cancel):
    browser = browser_module()
    pid_file = tmp_path / "pid"
    code = (
        "import os,time,pathlib; "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"ok")))
        broker = browser._Broker(client)
        initial = await broker.fetch("https://example.com")
        try:
            task = asyncio.create_task(browser._serve_worker(
                [sys.executable, "-I", "-B", "-c", code], {}, broker, initial,
            ))
            for _ in range(100):
                if pid_file.exists():
                    break
                await asyncio.sleep(0.01)
            assert pid_file.exists()
            pid = int(pid_file.read_text())
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(0.02):
                        await task
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Requires process enumeration outside sandbox")
def test_cleanup_reaps_detached_child_process_group(tmp_path):
    browser = browser_module()
    pid_file = tmp_path / "child-pid"
    child = "import time; time.sleep(60)"
    code = (
        "import subprocess,sys,time,pathlib; "
        f"child=subprocess.Popen([sys.executable,'-I','-B','-c',{child!r}],start_new_session=True); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); time.sleep(60)"
    )

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"ok")))
        broker = browser._Broker(client)
        try:
            initial = await broker.fetch("https://example.com")
            task = asyncio.create_task(browser._serve_worker([sys.executable, "-I", "-B", "-c", code], {},
                                                            broker, initial))
            for _ in range(100):
                if pid_file.exists():
                    break
                await asyncio.sleep(0.01)
            assert pid_file.exists()
            started = asyncio.get_running_loop().time()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert asyncio.get_running_loop().time() - started < 2
            pid = int(pid_file.read_text())
            for _ in range(100):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("Detached browser descendant survived cancellation")
        finally:
            await client.aclose()

    asyncio.run(run())


def test_oversized_worker_message_is_rejected_and_worker_closed():
    browser = browser_module()

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"ok")))
        broker = browser._Broker(client)
        try:
            initial = await broker.fetch("https://example.com")
            code = f"import sys; sys.stdout.write('x'*{browser._RESULT_LINE_LIMIT + 100}); sys.stdout.flush()"
            with pytest.raises(PublicWebError) as caught:
                await browser._serve_worker([sys.executable, "-I", "-B", "-c", code], {}, broker, initial)
            assert caught.value.code == "web_browser_protocol_error"
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Explicit OS/browser integration run")
def test_os_sandbox_denies_real_ip_sockets(tmp_path):
    browser = browser_module()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    code = (
        "import socket,os,sys; "
        "print(os.stat('/proc/self/ns/net').st_ino,flush=True) if sys.platform.startswith('linux') else None; "
        "s=socket.socket(); s.settimeout(1); "
        f"s.connect(('127.0.0.1',{port}))"
    )

    async def run():
        command = browser._sandbox_command([sys.executable, "-I", "-B", "-c", code], str(tmp_path))
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await process.communicate()
        assert process.returncode != 0
        # sandbox_apply failures do not prove network denial; run this test outside
        # an enclosing tool sandbox to obtain the real acceptance evidence.
        assert b"sandbox_apply" not in stderr
        if sys.platform.startswith("linux"):
            assert int(stdout.strip()) != os.stat("/proc/self/ns/net").st_ino
            assert any(message in stderr for message in [b"Network is unreachable", b"ConnectionRefusedError"])
        else:
            assert b"PermissionError" in stderr

    try:
        asyncio.run(run())
    finally:
        listener.close()


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Explicit OS/browser integration run")
@pytest.mark.skipif(sys.platform == "darwin", reason="macOS Seatbelt cannot nest Chromium sandbox")
def test_real_chromium_observes_actual_no_content_status():
    async def run():
        html = b'''<html><body><p id="result">pending</p><script>
          fetch('/api').then(response => {
            document.getElementById('result').textContent = 'status:' + response.status;
          });
        </script></body></html>'''

        def handler(request):
            if request.url.path == "/api":
                return httpx.Response(204)
            return httpx.Response(200, content=html, headers={"Content-Type": "text/html"})

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser_module().render_public_page(client, "https://example.com/page")
            assert b"status:204" in result.body
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Explicit OS/browser integration run")
@pytest.mark.skipif(sys.platform == "darwin", reason="macOS Seatbelt cannot nest Chromium sandbox")
def test_real_chromium_executes_delayed_fetch_at_redirected_origin():
    browser = browser_module()
    pytest.importorskip("playwright.async_api")

    async def run():
        seen = []
        html = b'''<html><body><p id="result">Loading</p><script>
          setTimeout(async () => {
            const response = await fetch('/api-start');
            const data = await response.text();
            document.getElementById('result').textContent = location.host + ':' + data;
          }, 700);
        </script></body></html>'''

        def handler(request):
            seen.append(str(request.url))
            if request.url.host == "start.example":
                return httpx.Response(302, headers={"Location": "https://final.example/page"})
            if request.url.path == "/api-start":
                return httpx.Response(302, headers={"Location": "/api-final"})
            if request.url.path == "/api-final":
                return httpx.Response(200, content=b"rendered evidence", headers={"Content-Type": "text/plain"})
            return httpx.Response(200, content=html, headers={"Content-Type": "text/html"})

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser.render_public_page(client, "https://start.example/start")
            assert result.url == "https://final.example/page"
            assert b"final.example:rendered evidence" in result.body
            assert result.content_type == "text/html; charset=utf-8"
            assert result.network_bytes == len(html) + len(b"rendered evidence")
            assert "https://final.example/api-start" in seen
            assert "https://final.example/api-final" in seen
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Explicit OS/browser integration run")
@pytest.mark.skipif(sys.platform == "darwin", reason="macOS Seatbelt cannot nest Chromium sandbox")
def test_real_chromium_renders_spa_with_33_scripts_and_six_stylesheets():
    browser = browser_module()
    pytest.importorskip("playwright.async_api")

    async def run():
        seen = []
        scripts = ''.join(f'<script src="/part{i}.js"></script>' for i in range(33))
        styles = ''.join(f'<link rel="stylesheet" href="/style{i}.css">' for i in range(6))
        html = (f'<html><head>{styles}</head><body><p id="proof">Loading</p>{scripts}'
                '<script>addEventListener("load",()=>{document.getElementById("proof").textContent='
                '"ready "+window.parts})</script></body></html>').encode()

        def handler(request):
            seen.append(str(request.url))
            if request.url.path.endswith(".js"):
                return httpx.Response(200, content=b"window.parts=(window.parts||0)+1;",
                                      headers={"Content-Type": "text/javascript"})
            if request.url.path.endswith(".css"):
                return httpx.Response(200, content=b"body{color:black}", headers={"Content-Type": "text/css"})
            return httpx.Response(200, content=html, headers={"Content-Type": "text/html"})

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser.render_public_page(client, "https://example.com/page")
            assert b'<p id="proof">ready 33</p>' in result.body
            assert len(seen) == 40
            assert result.warning is None
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Explicit OS/browser integration run")
@pytest.mark.skipif(sys.platform == "darwin", reason="macOS Seatbelt cannot nest Chromium sandbox")
def test_real_chromium_retains_evidence_after_failed_optional_xhr():
    browser = browser_module()
    pytest.importorskip("playwright.async_api")

    async def run():
        html = b'''<html><body><p id="proof">valid public evidence</p><script>
          const xhr=new XMLHttpRequest(); xhr.open('GET','/unavailable');
          xhr.onerror=()=>{document.getElementById('proof').textContent='retained public evidence'};
          xhr.send();</script></body></html>'''

        def handler(request):
            if request.url.path == "/unavailable":
                raise httpx.ConnectError("offline", request=request)
            return httpx.Response(200, content=html, headers={"Content-Type": "text/html"})

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser.render_public_page(client, "https://example.com/page")
            assert b'<p id="proof">retained public evidence</p>' in result.body
            assert "network_error:XHR=1" in result.warning
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Explicit OS/browser integration run")
@pytest.mark.skipif(sys.platform == "darwin", reason="macOS Seatbelt cannot nest Chromium sandbox")
@pytest.mark.parametrize("cors_allowed", [True, False])
def test_real_chromium_cross_origin_get_preserves_server_cors(cors_allowed):
    browser = browser_module()
    pytest.importorskip("playwright.async_api")

    async def run():
        seen = []
        html = b'''<html><body><p id="proof">Loading</p><script>
          fetch('https://api.example/data').then(r=>r.text()).then(data=>{
            document.getElementById('proof').textContent=data
          }).catch(()=>{document.getElementById('proof').textContent='CORS denied'});
          </script></body></html>'''

        def handler(request):
            seen.append((request.method, request.url.host))
            assert "cookie" not in request.headers
            if request.url.host == "api.example":
                headers = {"Content-Type": "text/plain"}
                if cors_allowed:
                    headers["Access-Control-Allow-Origin"] = "*"
                return httpx.Response(200, content=b"public cross-origin evidence", headers=headers)
            return httpx.Response(200, content=html, headers={"Content-Type": "text/html"})

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser.render_public_page(client, "https://page.example/page")
            expected = b"public cross-origin evidence" if cors_allowed else b"CORS denied"
            assert b'<p id="proof">' + expected + b'</p>' in result.body
            assert seen == [("GET", "page.example"), ("GET", "api.example")]
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Explicit OS/browser integration run")
@pytest.mark.skipif(sys.platform == "darwin", reason="macOS Seatbelt cannot nest Chromium sandbox")
def test_real_chromium_skips_visual_assets_and_keeps_text_requests(monkeypatch):
    browser = browser_module()
    pytest.importorskip("playwright.async_api")
    observed = []
    original_broker = browser._Broker

    class ObservedBroker(original_broker):
        def __init__(self, client):
            super().__init__(client)
            observed.append(self)

    monkeypatch.setattr(browser, "_Broker", ObservedBroker)

    async def run():
        seen = []
        html = b'''<html><head><style>
          @font-face { font-family: fixture; src: url('/fixture.woff2'); }
          body { font-family: fixture; }
          </style></head><body><p id="proof">Loading</p>
          <img src="/photo.png"><video src="/clip.mp4" preload="auto"></video>
          <script>setTimeout(async()=>{
            const response=await fetch('/api');
            document.getElementById('proof').textContent=await response.text();
          },300)</script></body></html>'''

        def handler(request):
            seen.append(str(request.url))
            if request.url.path == "/page":
                return httpx.Response(200, content=html, headers={"Content-Type": "text/html"})
            if request.url.path == "/api":
                return httpx.Response(200, content=b"ready", headers={"Content-Type": "text/plain"})
            pytest.fail("Decorative image, font or media request entered the parent HTTP broker")

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser.render_public_page(client, "https://example.com/page")
            assert b'<p id="proof">ready</p>' in result.body
            assert seen == ["https://example.com/page", "https://example.com/api"]
            assert observed[0].budget.requests == 2
            assert all(observed[0].skipped_resources.get(kind, 0) >= 1 for kind in ["Image", "Font", "Media"])
            assert observed[0].skipped_url_kinds.get("https:png", 0) >= 1
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Explicit OS/browser integration run")
@pytest.mark.skipif(sys.platform == "darwin", reason="macOS Seatbelt cannot nest Chromium sandbox")
@pytest.mark.parametrize("script", [
    "fetch('http://127.0.0.1/secret')",
    "fetch('/api', {method:'POST',body:'private'})",
    "new WebSocket('wss://example.com/socket')",
    "window.open('https://example.com/popup')",
    "const a=document.createElement('a'); a.href='data:text/plain,private'; a.download='data.txt'; a.click()",
])
def test_real_chromium_blocks_disallowed_requests(script):
    browser = browser_module()
    pytest.importorskip("playwright.async_api")

    async def run():
        seen = []
        html = f"<html><body><p>page</p><script>{script}</script></body></html>".encode()

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, content=html, headers={"Content-Type": "text/html"})

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser.render_public_page(client, "https://example.com/page")
            assert b'<p>page</p>' in result.body
            assert result.render_diagnostics['blocked_actions']
            assert seen == ["https://example.com/page"]
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1" or sys.platform != "darwin",
                    reason="Explicit macOS incompatibility validation")
def test_macos_browser_sandbox_incompatibility_fails_closed():
    browser = browser_module()
    pytest.importorskip("playwright.async_api")

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=b"<p>public page</p>", headers={"Content-Type": "text/html"}),
        ))
        try:
            with pytest.raises(PublicWebError) as caught:
                await browser.render_public_page(client, "https://example.com")
            assert caught.value.code == "web_browser_unavailable"
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1" or not sys.platform.startswith("linux"),
                    reason="Real Linux public-renderer PID namespace validation")
def test_public_linux_worker_early_exit_kills_detached_descendants(monkeypatch, tmp_path):
    browser = browser_module()
    # This fixture validates the actual public subprocess/isolation lifecycle;
    # its deliberately crashing worker does not import the browser dependency.
    monkeypatch.setattr(browser.importlib.util, "find_spec", lambda name: object())
    marker = "praxis-browser-child-" + uuid.uuid4().hex
    worker = tmp_path / "early_exit_worker.py"
    worker.write_text(
        "import os,subprocess,sys,time,pathlib,json\n"
        "sys.stdin.buffer.readline()\n"
        "ready=pathlib.Path(os.environ['HOME'])/'child-ready'\n"
        "code=\"import os,time,pathlib; pathlib.Path(os.environ['HOME']+'/child-ready').write_text('ready'); "
        "time.sleep(60)\"\n"
        f"subprocess.Popen([sys.executable,'-I','-B','-c',code,{marker!r}],start_new_session=True)\n"
        "while not ready.exists(): time.sleep(.01)\n"
        "print(json.dumps({'type':'fetch','method':'GET','url':'https://example.com/child-started'}),flush=True)\n"
        "sys.stdin.buffer.readline()\n"
        "os._exit(0)\n"
    )
    monkeypatch.setattr(browser, "__file__", str(worker))
    monkeypatch.setattr(browser, "RENDER_TIMEOUT_SECONDS", 3)

    def remaining_children():
        listing = subprocess.run(["/bin/ps", "-axo", "pid=,args="], check=True, capture_output=True, text=True)
        return [int(line.split(maxsplit=1)[0]) for line in listing.stdout.splitlines() if marker in line]

    async def run():
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, content=b"<p>page</p>", headers={"Content-Type": "text/html"})

        client = PublicWebClient(transport=httpx.MockTransport(handler))
        started = asyncio.get_running_loop().time()
        try:
            with pytest.raises(PublicWebError) as caught:
                await browser.render_public_page(client, "https://example.com/page")
            assert caught.value.code == "web_browser_unavailable"
            assert "https://example.com/child-started" in seen
            assert asyncio.get_running_loop().time() - started < 2
            assert remaining_children() == []
        finally:
            await client.aclose()
            for pid in remaining_children():
                os.kill(pid, 9)

    asyncio.run(run())


@pytest.mark.parametrize('code', ['invalid_url', 'nonpublic_address', 'response_too_large',
                                  'web_browser_budget_exceeded'])
def test_blocked_subresource_preserves_broker_and_records_actual_reason(code):
    browser = browser_module()

    async def run():
        seen = []
        client = PublicWebClient(transport=httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(200)))
        broker = browser._Broker(client)

        async def reject(*args):
            raise PublicWebError(code, 'Blocked before transport.')
        broker.fetch = reject
        try:
            reply = await broker.reply({'type': 'fetch', 'method': 'GET', 'url': 'https://example.com/api',
                                        'resource_type': 'XHR'})
            assert reply == {'type': 'abort'}
            assert broker.blocked_actions[code] == 1
            assert not seen
            with pytest.raises(PublicWebError):
                await broker.reply({'type': 'fetch', 'method': 'GET', 'url': 'https://example.com/page',
                                    'resource_type': 'Document'})
        finally:
            await client.aclose()
    asyncio.run(run())


def test_worker_snapshot_reports_bounded_blocked_actions_and_budget_facts(tmp_path):
    browser = browser_module()
    worker = tmp_path / 'worker.py'
    worker.write_text('''import sys,json,base64
json.loads(sys.stdin.readline())
print(json.dumps({'type':'blocked','reason':'method_not_get'}),flush=True)
print(json.dumps({'type':'result','url':'https://example.com/page',
 'body':base64.b64encode(b'<main>Evidence</main>').decode(),
 'load_timeouts':['networkidle']}),flush=True)
''')

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b'initial')))
        broker = browser._Broker(client)
        try:
            initial = await broker.fetch('https://example.com/page')
            result = await browser._serve_worker([sys.executable, str(worker)], {}, broker, initial)
            assert result.body == b'<main>Evidence</main>'
            facts = result.render_diagnostics
            assert facts['blocked_actions'] == {'method_not_get': 1}
            assert facts['load_timeouts'] == ['networkidle']
            assert facts['requests'] == 1 and facts['wire_bytes'] == facts['decoded_bytes'] == 7
            assert facts['dom_bytes'] == len(result.body)
            assert 'method_not_get' in result.warning
        finally:
            await client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('reason', ['invented', 'POST https://example.com/?secret=value', 'x' * 1024])
def test_blocked_action_protocol_rejects_unbounded_or_unknown_reasons(reason):
    browser = browser_module()
    with pytest.raises(PublicWebError) as caught:
        browser._Broker(None).record_blocked({'type': 'blocked', 'reason': reason})
    assert caught.value.code == 'web_browser_protocol_error'


@pytest.mark.skipif(os.environ.get('PRAXIS_TEST_BROWSER') != '1' or sys.platform == 'darwin',
                    reason='Explicit Linux Chromium integration run')
def test_real_chromium_networkidle_timeout_preserves_acquired_dom():
    browser = browser_module()
    pytest.importorskip('playwright.async_api')

    async def run():
        seen = []
        html = b'''<html><body><main>Polling page evidence</main><script>
          setInterval(()=>fetch('/poll'),300);
          </script></body></html>'''
        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, content=html if request.url.path == '/page' else b'{}',
                                  headers={'Content-Type': 'text/html' if request.url.path == '/page' else
                                           'application/json'})
        client = PublicWebClient(transport=httpx.MockTransport(handler))
        started = asyncio.get_running_loop().time()
        try:
            result = await browser.render_public_page(client, 'https://example.com/page')
            assert b'Polling page evidence' in result.body
            assert result.render_diagnostics['load_timeouts'] == ['networkidle']
            assert result.render_diagnostics['requests'] == len(seen) <= 64
            assert asyncio.get_running_loop().time() - started < browser.RENDER_TIMEOUT_SECONDS + 2
        finally:
            await client.aclose()
    asyncio.run(run())


def test_browser_failure_reports_actual_budget_facts_without_snapshot(monkeypatch):
    browser = browser_module()
    monkeypatch.setattr(browser.importlib.util, 'find_spec', lambda name: object())
    monkeypatch.setattr(browser.sys, 'platform', 'linux')
    monkeypatch.setattr(browser, '_sandbox_command', lambda command, directory: command)
    monkeypatch.setattr(browser, '_worker_environment', lambda directory: {})
    async def fail(command, environment, broker, initial, **options):
        raise PublicWebError('web_browser_budget_exceeded', 'DOM exceeded its limit.')
    monkeypatch.setattr(browser, '_serve_worker', fail)

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b'page')))
        try:
            with pytest.raises(PublicWebError) as caught:
                await browser.render_public_page(client, 'https://example.com/page')
            facts = caught.value.render_diagnostics
            assert facts['requests'] == 1 and facts['wire_bytes'] == facts['decoded_bytes'] == 4
            assert facts['snapshot_available'] is False
        finally:
            await client.aclose()
    asyncio.run(run())

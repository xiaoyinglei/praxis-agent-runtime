"""Browser boundary tests; real Chromium tests exercise JavaScript separately."""

import asyncio
import importlib
import os

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
            assert reply["body"] == body
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


def test_browser_environment_excludes_credentials(monkeypatch, tmp_path):
    browser = browser_module()
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "https://u:secret@example.com")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    environment = browser._browser_environment(str(tmp_path))
    assert environment["HOME"] == str(tmp_path)
    assert "secret" not in repr(environment)
    assert "PYTHONPATH" not in environment
    assert "HTTPS_PROXY" not in environment


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
            assert initial.url == "https://example.com/start" and initial.status_code == 302
            assert broker.budget.requests == 1
            final = await broker.reply({"type": "fetch", "method": "GET", "url": "https://example.com/final"})
            assert final['body'] == b'done'
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
            assert final["body"] == b"script"
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
                assert response["body"] == body
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


def test_byte_budget_is_enforced():
    browser = browser_module()
    budget = browser._Budget()
    budget.on_bytes(browser.MAX_NETWORK_BYTES, browser.MAX_NETWORK_BYTES)
    with pytest.raises(PublicWebError) as caught:
        budget.on_bytes(0, 1)
    assert caught.value.code == "web_browser_budget_exceeded"


def test_render_deadline_includes_initial_http_fetch(monkeypatch):
    browser = browser_module()
    monkeypatch.setattr(browser.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(browser, "RENDER_TIMEOUT_SECONDS", 0.02)

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


@pytest.mark.skipif(os.environ.get("PRAXIS_TEST_BROWSER") != "1", reason="Explicit OS/browser integration run")
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


@pytest.mark.parametrize('reason', ['invented', 'POST https://example.com/?secret=value', 'x' * 1024])
def test_blocked_action_protocol_rejects_unbounded_or_unknown_reasons(reason):
    browser = browser_module()
    with pytest.raises(PublicWebError) as caught:
        browser._Broker(None).record_blocked({'type': 'blocked', 'reason': reason})
    assert caught.value.code == 'web_browser_protocol_error'


@pytest.mark.skipif(os.environ.get('PRAXIS_TEST_BROWSER') != '1', reason='Explicit Chromium integration run')
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
    async def fail(broker, initial, **options):
        raise PublicWebError('web_browser_budget_exceeded', 'DOM exceeded its limit.')
    monkeypatch.setattr(browser, '_render_with_playwright', fail)

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


def test_missing_managed_executable_has_install_guidance(monkeypatch, tmp_path):
    pytest.importorskip('playwright.async_api')
    monkeypatch.setenv('PLAYWRIGHT_BROWSERS_PATH', str(tmp_path / 'missing-managed-browsers'))

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=b'<main>source</main>')))
        try:
            with pytest.raises(PublicWebError) as caught:
                await browser_module().render_public_page(client, 'https://example.com/page')
            assert caught.value.code == 'web_browser_unavailable'
            assert 'uv sync --extra browser' in str(caught.value)
            assert 'uv run python -m playwright install chromium' in str(caught.value)
            assert caught.value.render_diagnostics['os_network_isolation'] is False
        finally:
            await client.aclose()
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get('PRAXIS_TEST_BROWSER') != '1', reason='Real managed Chromium integration run')
@pytest.mark.parametrize('target', ['http://127.0.0.1/secret', 'https://example.com:8443/secret'])
def test_real_chromium_redirect_target_is_denied_before_transport(target):
    async def run():
        seen = []
        html = b'''<main>retained public evidence</main><script>
          fetch('/jump').catch(()=>{});
        </script>'''
        def handler(request):
            seen.append(str(request.url))
            if request.url.path == '/jump':
                return httpx.Response(302, headers={'location': target})
            return httpx.Response(200, content=html, headers={'content-type': 'text/html'})
        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser_module().render_public_page(client, 'https://example.com/page')
            assert b'retained public evidence' in result.body
            assert seen == ['https://example.com/page', 'https://example.com/jump']
            assert result.render_diagnostics['blocked_actions'] == {'invalid_url': 1}
        finally:
            await client.aclose()
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get('PRAXIS_TEST_BROWSER') != '1', reason='Real managed Chromium integration run')
def test_real_chromium_does_not_forward_page_credentials_or_request_headers():
    async def run():
        seen = []
        html = b'''<main>public evidence</main><script>
          document.cookie='private=value';
          fetch('/api', {headers:{'Authorization':'Bearer private', 'X-Private':'value'}}).catch(()=>{});
        </script>'''
        def handler(request):
            seen.append(request)
            assert all(key not in request.headers for key in ['cookie', 'authorization', 'x-private'])
            return httpx.Response(200, content=html if request.url.path == '/page' else b'ok',
                                  headers={'content-type': 'text/html' if request.url.path == '/page'
                                           else 'text/plain'})
        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser_module().render_public_page(client, 'https://example.com/page')
            assert len(seen) == 2
            assert result.render_diagnostics['browser_mode'] == 'trusted_local_brokered'
            assert result.render_diagnostics['chromium_sandbox'] is True
            assert result.render_diagnostics['os_network_isolation'] is False
        finally:
            await client.aclose()
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get('PRAXIS_TEST_BROWSER') != '1', reason='Real managed Chromium integration run')
@pytest.mark.parametrize('cancel', [False, True])
def test_real_chromium_timeout_and_cancellation_close_context_browser_and_driver(monkeypatch, cancel):
    api = pytest.importorskip('playwright.async_api')
    browser = browser_module()
    original_launch = api.BrowserType.launch
    original_context = api.Browser.new_context
    original_playwright = api.async_playwright
    closed = []
    launched = []
    options_seen = []

    async def launch(owner, **options):
        options_seen.append(options)
        result = await original_launch(owner, **options)
        launched.append(result)
        result.on('disconnected', lambda: closed.append('browser'))
        return result

    async def new_context(owner, **options):
        result = await original_context(owner, **options)
        result.on('close', lambda: closed.append('context'))
        return result

    def playwright():
        manager = original_playwright()
        original_start = manager.start
        async def start():
            driver = await original_start()
            original_stop = driver.stop
            async def stop():
                await original_stop()
                closed.append('driver')
            driver.stop = stop
            return driver
        manager.start = start
        return manager

    monkeypatch.setattr(api.BrowserType, 'launch', launch)
    monkeypatch.setattr(api.Browser, 'new_context', new_context)
    monkeypatch.setattr(api, 'async_playwright', playwright)
    if not cancel:
        monkeypatch.setattr(browser, 'RENDER_TIMEOUT_SECONDS', 5)
        async def stalled_content(owner):
            await asyncio.sleep(60)
        monkeypatch.setattr(api.Page, 'content', stalled_content)

    async def run():
        entered = asyncio.Event()
        released = asyncio.Event()
        async def handler(request):
            if request.url.path == '/hang':
                entered.set()
                try:
                    await asyncio.sleep(60)
                finally:
                    released.set()
            return httpx.Response(200, content=b'<main>evidence</main><script>fetch("/hang")</script>',
                                  headers={'content-type': 'text/html'})
        client = PublicWebClient(transport=httpx.MockTransport(handler))
        task = asyncio.create_task(browser.render_public_page(client, 'https://example.com/page'))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(PublicWebError) as caught:
                    await task
                assert caught.value.code == 'web_browser_timeout'
            assert released.is_set()
            assert closed == ['context', 'browser', 'driver']
            assert launched and all(not item.is_connected() for item in launched)
            assert options_seen[0]['chromium_sandbox'] is True
            assert 'channel' not in options_seen[0] and 'executable_path' not in options_seen[0]
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await client.aclose()
    asyncio.run(run())


def test_cancellation_during_playwright_start_stops_the_started_driver(monkeypatch):
    api = pytest.importorskip('playwright.async_api')
    original_playwright = api.async_playwright

    async def run():
        entered = asyncio.Event()
        drivers = []
        stopped = []
        def playwright():
            manager = original_playwright()
            original_start = manager.start
            async def start():
                driver = await original_start()
                drivers.append(driver)
                original_stop = driver.stop
                async def stop():
                    await original_stop()
                    stopped.append(True)
                driver.stop = stop
                entered.set()
                # Exercise cancellation while the owning API is still starting.
                await asyncio.sleep(0.05)
                return driver
            manager.start = start
            return manager
        monkeypatch.setattr(api, 'async_playwright', playwright)
        client = PublicWebClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b'page')))
        task = asyncio.create_task(browser_module().render_public_page(client, 'https://example.com/page'))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stopped == [True]
        finally:
            for driver in drivers:
                await driver.stop()
            await client.aclose()
    asyncio.run(run())


@pytest.mark.skipif(os.environ.get('PRAXIS_TEST_BROWSER') != '1', reason='Real managed Chromium integration run')
def test_real_chromium_reports_unsupported_redirected_script_navigation():
    async def run():
        seen = []
        def handler(request):
            seen.append(str(request.url))
            if request.url.path == '/jump':
                return httpx.Response(302, headers={'location': 'https://final.example/page'})
            body = b'<main>Final document</main>' if request.url.host == 'final.example' else (
                b'<main>First document</main><script>setTimeout(()=>location.href="/jump",200)</script>')
            return httpx.Response(200, content=body, headers={'content-type': 'text/html'})
        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(PublicWebError) as caught:
                await browser_module().render_public_page(client, 'https://example.com/page')
            assert caught.value.code == 'web_browser_request_blocked'
            assert caught.value.render_diagnostics['unsupported_features'] == ['redirected_document_navigation']
            assert seen == ['https://example.com/page', 'https://example.com/jump', 'https://final.example/page']
        finally:
            await client.aclose()
    asyncio.run(run())

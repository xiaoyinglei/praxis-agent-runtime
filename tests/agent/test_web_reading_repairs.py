"""Reading fidelity across layouts and browser request origins, without site rules."""
from __future__ import annotations

import base64
import importlib
import os
import sys

import httpx
import pytest

from agent_runtime.tools.web_content import extract_content
from agent_runtime.tools.web_http import PublicWebClient, PublicWebError
from tests.agent.test_web_tools import web_tools


@pytest.mark.anyio
@pytest.mark.parametrize("layout", [
    '<div><h2>Details</h2><div>{primary}</div><ul><li>{secondary}</li></ul></div>',
    '<section><h2>Reference</h2><table><tr><td>{primary}</td></tr></table><pre>{secondary}</pre></section>',
    '<section><p>{primary}</p></section><section><p>{secondary}</p></section>',
])
async def test_nonsemantic_document_preserves_sibling_evidence_through_tool(tmp_path, layout):
    primary = 'Primary source facts: actions, constraints, and supported operations.'
    secondary = 'Additional source facts: examples, requirements, and documented results.'
    html = ('<body><nav>Global links</nav>' + layout.format(primary=primary, secondary=secondary)
            + '<div><p>' + 'Organization overview without the document facts. ' * 30
            + '</p></div><footer>Copyright boilerplate</footer></body>')
    tools, client, _ = web_tools(tmp_path, lambda _: httpx.Response(
        200, text=html, headers={'content-type': 'text/html'}))
    try:
        result = await tools['web_fetch'].run({'url': 'https://example.com/document'})
        assert result['error_code'] is None
        assert primary in result['content']
        assert secondary in result['content']
        assert 'Global links' not in result['content']
        assert 'Copyright boilerplate' not in result['content']
        assert result['extraction_method'] == 'body_fallback'
    finally:
        await client.aclose()


@pytest.mark.parametrize('region', ['article', 'div class="article"', 'div class="main"'])
def test_multiple_document_regions_preserve_all_documents(region):
    tag = region.split()[0]
    html = (f'<body><{region}><h1>First</h1><p>First document.</p></{tag}>'
            f'<{region}><h1>Second</h1><p>Second document.</p></{tag}></body>').encode()
    result = extract_content(html, 'text/html', 'https://example.com/directory')
    assert 'First document.' in result.text
    assert 'Second document.' in result.text


@pytest.mark.parametrize('region', ['main', 'div role="main"', 'article', 'div itemprop="articleBody"'])
def test_declared_document_region_excludes_unmarked_page_actions(region):
    tag = region.split()[0]
    html = (f'<body><{region}><h1>Source</h1><div>Declared source facts.</div></{tag}>'
            '<div><a href="/">Home action</a><a href="#">Page action</a></div></body>').encode()
    result = extract_content(html, 'text/html', 'https://example.com/reference')
    assert 'Declared source facts.' in result.text
    assert 'Home action' not in result.text and 'Page action' not in result.text


def test_class_names_do_not_authorize_discarding_unmarked_content():
    result = extract_content(b'<body><div class="article"><p>One fact.</p></div>'
                             b'<div>Additional facts outside that class.</div></body>',
                             'text/html', 'https://example.com/reference')
    assert 'One fact.' in result.text and 'Additional facts outside that class.' in result.text
    assert result.extraction_method == 'body_fallback'


def test_article_does_not_discard_div_and_list_facts_inside_main():
    result = extract_content(
        b'<main><article><p>Article fact.</p></article><div>Sibling fact.</div>'
        b'<ul><li>Document requirement.</li></ul></main>', 'text/html', 'https://example.com/reference',
    )
    assert all(text in result.text for text in ['Article fact.', 'Sibling fact.', 'Document requirement.'])


@pytest.mark.parametrize('identity', ['comments', 'advertisement', 'sidebar', 'inlineQr_wrap', 'social'])
def test_class_names_cannot_delete_document_facts(identity):
    result = extract_content(
        f'<main><div class="{identity}">Documented interface facts.</div></main>'.encode(),
        'text/html', 'https://example.com/reference',
    )
    assert 'Documented interface facts.' in result.text


@pytest.mark.anyio
async def test_broker_forwards_only_public_origin_referer():
    requests = []
    client = PublicWebClient(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, content=b'code')))
    try:
        broker = importlib.import_module('agent_runtime.tools.web_browser')._Broker(client)
        reply = await broker.reply({'type': 'fetch', 'method': 'GET', 'url': 'https://cdn.example/app.js',
                                    'resource_type': 'Script',
                                    'referer': 'https://source.example/private/path?secret=hidden#fragment',
                                    'headers': {'Cookie': 'private', 'Authorization': 'private'}})
        assert base64.b64decode(reply['body']) == b'code'
        assert requests[0].headers['Referer'] == 'https://source.example/'
        assert 'cookie' not in requests[0].headers and 'authorization' not in requests[0].headers
        assert 'hidden' not in str(requests[0].headers)
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize('referer', ['http://127.0.0.1/', 'https://u:password@example.com/',
                                   'file:///private/file', 'https://example.com/\r\nsecret', 42])
async def test_broker_rejects_invalid_referer_before_network(referer):
    client = PublicWebClient(transport=httpx.MockTransport(lambda _: pytest.fail('Unexpected network access')))
    try:
        broker = importlib.import_module('agent_runtime.tools.web_browser')._Broker(client)
        with pytest.raises(PublicWebError) as caught:
            await broker.reply({'type': 'fetch', 'method': 'GET', 'url': 'https://cdn.example/app.js',
                                'resource_type': 'Script', 'referer': referer})
        assert caught.value.code == 'web_browser_protocol_error'
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_broker_omits_referer_on_https_to_http_downgrade():
    requests = []
    client = PublicWebClient(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, content=b'code')))
    try:
        broker = importlib.import_module('agent_runtime.tools.web_browser')._Broker(client)
        await broker.reply({'type': 'fetch', 'method': 'GET', 'url': 'http://cdn.example/app.js',
                            'resource_type': 'Script', 'referer': 'https://source.example/'})
        assert 'referer' not in requests[0].headers
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize('destination,expected', [('https://other.example/app.js', None),
                                                ('http://other.example/app.js', None)])
async def test_plain_http_redirect_discards_credential_and_source_headers(destination, expected):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == 'cdn.example':
            return httpx.Response(302, headers={'Location': destination})
        return httpx.Response(200, content=b'code')

    client = PublicWebClient(transport=httpx.MockTransport(handler))
    try:
        await client.get('https://cdn.example/app.js', headers={
            'Referer': 'https://source.example/private?secret=hidden', 'Cookie': 'private',
            'Authorization': 'private',
        })
        assert requests[1].headers.get('referer') == expected
        assert 'cookie' not in requests[1].headers and 'authorization' not in requests[1].headers
        assert 'hidden' not in str(requests[1].headers)
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_broker_retains_redirect_and_response_referrer_policy():
    requests = []
    client = PublicWebClient(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(
        307, headers={'Location': 'https://other.example/app.js', 'Referrer-Policy': 'no-referrer'})))
    try:
        broker = importlib.import_module('agent_runtime.tools.web_browser')._Broker(client)
        result = await broker.reply({'type': 'fetch', 'method': 'GET', 'url': 'https://cdn.example/app.js',
                                     'resource_type': 'Script', 'referer': 'https://source.example/'})
        assert result['status'] == 307
        assert result['headers']['location'] == 'https://other.example/app.js'
        assert result['headers']['referrer-policy'] == 'no-referrer'
        assert len(requests) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_valid_source_does_not_change_private_resource_failure_into_protocol_error():
    client = PublicWebClient(transport=httpx.MockTransport(lambda _: pytest.fail('Unexpected network access')))
    try:
        broker = importlib.import_module('agent_runtime.tools.web_browser')._Broker(client)
        result = await broker.reply({'type': 'fetch', 'method': 'GET', 'url': 'http://127.0.0.1/app.js',
                                     'resource_type': 'Script', 'referer': 'https://source.example/'})
        assert result == {'type': 'abort'}
        assert broker.blocked_actions == {'invalid_url': 1}
    finally:
        await client.aclose()


@pytest.mark.skipif(os.environ.get('PRAXIS_TEST_BROWSER') != '1' or sys.platform == 'darwin',
                    reason='Explicit Linux Chromium integration run')
@pytest.mark.parametrize('policy', ['strict-origin-when-cross-origin', 'no-referrer'])
@pytest.mark.parametrize('delivery', ['meta', 'header'])
def test_real_browser_preserves_origin_and_referrer_policy(policy, delivery):
    import asyncio

    browser = importlib.import_module('agent_runtime.tools.web_browser')
    meta = '<meta name="referrer" content="' + policy + '">' if delivery == 'meta' else ''
    html = ('<html><head>' + meta + '</head><body>'
            '<div id="proof">Loading</div><script src="https://cdn.example/app.js"></script></body></html>').encode()
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.host == 'cdn.example':
            expected = 'https://source.example/' if policy != 'no-referrer' else None
            assert request.headers.get('referer') == expected
            return httpx.Response(200, headers={'content-type': 'application/javascript'},
                                  content=b'document.getElementById("proof").textContent="Public source evidence";')
        return httpx.Response(
            200,
            headers={'content-type': 'text/html', **({'Referrer-Policy': policy} if delivery == 'header' else {})},
            content=html,
        )

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser.render_public_page(client, 'https://source.example/private?secret=hidden')
            assert b'Public source evidence' in result.body
            assert result.render_diagnostics['resource_failures'] == {}
            assert len(seen) == 2
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.skipif(os.environ.get('PRAXIS_TEST_BROWSER') != '1' or sys.platform == 'darwin',
                    reason='Explicit Linux Chromium integration run')
def test_real_browser_applies_redirect_policy_without_duplicate_fetch():
    import asyncio

    browser = importlib.import_module('agent_runtime.tools.web_browser')
    seen = []

    def handler(request):
        seen.append(request)
        assert 'cookie' not in request.headers and 'authorization' not in request.headers
        if request.url.host == 'cdn.example':
            assert request.headers.get('referer') == 'https://source.example/'
            return httpx.Response(307, headers={'Location': 'https://final.example/app.js',
                                               'Referrer-Policy': 'no-referrer'})
        if request.url.host == 'final.example':
            assert 'referer' not in request.headers
            return httpx.Response(200, headers={'content-type': 'application/javascript'},
                                  content=b'document.getElementById("proof").textContent="Redirect evidence";')
        return httpx.Response(200, headers={'content-type': 'text/html'}, content=(
            b'<html><body><div id="proof">Loading</div>'
            b'<script src="https://cdn.example/app.js"></script></body></html>'))

    async def run():
        client = PublicWebClient(transport=httpx.MockTransport(handler))
        try:
            result = await browser.render_public_page(client, 'https://source.example/page')
            assert b'Redirect evidence' in result.body
            assert result.render_diagnostics['resource_failures'] == {}
            assert [request.url.host for request in seen] == ['source.example', 'cdn.example', 'final.example']
            assert result.render_diagnostics['requests'] == 3
        finally:
            await client.aclose()

    asyncio.run(run())

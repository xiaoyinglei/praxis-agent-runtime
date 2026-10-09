"""Default public reading recovers empty HTML without site or text heuristics."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from dataclasses import replace

import httpx
import pytest

from agent_runtime.tools import web_browser
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.web_http import FetchedResponse, PublicWebError
from tests.agent.test_web_tools import call, web_tools

EMPTY_HTML = b'<body><div id="app"></div><script src="/app.js"></script></body>'


@pytest.mark.anyio
async def test_default_fetch_recovers_empty_html_once_and_preserves_both_snapshots(tmp_path, monkeypatch):
    requests, rendered = [], []
    tools, client, sources = web_tools(tmp_path, lambda request: requests.append(request) or httpx.Response(
        200, content=EMPTY_HTML, headers={'content-type': 'text/html'}))

    async def render(client, url, *, initial_response):
        rendered.append(initial_response)
        assert initial_response.body == EMPTY_HTML
        return FetchedResponse(url, 'text/html', b'<main>Published source facts.</main>', 123, 'direct')

    monkeypatch.setattr(web_browser, 'render_public_page', render)
    try:
        tool = tools['web_fetch']
        output = await tool.run({'url': 'https://example.com/document'})
        assert output['error_code'] is None and 'Published source facts.' in output['content']
        assert output['render_mode'] == 'browser' and output['network_bytes'] == 123
        assert output['static_source_id'] != output['source_id']
        original = json.loads(sources[output['static_source_id']])
        assert base64.b64decode(original['raw_body_base64']) == EMPTY_HTML
        assert original['extraction_error_code'] == 'web_content_empty'
        raw = await tool.run({'source_id': output['static_source_id'], 'view': 'raw'})
        assert EMPTY_HTML.decode() in raw['content'] and raw['error_code'] is None
        static = await tool.run({'source_id': output['static_source_id']})
        assert static['error_code'] == 'web_content_empty' and static['network_bytes'] == 0
        repeat = await tool.run({'url': 'https://example.com/document'})
        offline = await tool.run({'source_id': output['source_id']})
        assert repeat['cache_hit'] and repeat['source_id'] == output['source_id']
        assert repeat['network_bytes'] == 0 and offline['network_bytes'] == 0
        assert offline['static_source_id'] == output['static_source_id']
        for identity in [{'source_id': output['source_id']}, {'url': 'https://example.com/document'}]:
            failed = await tool.run({**identity, 'find': 'Absent source text'})
            assert failed['error_code'] == 'web_text_not_found'
            assert failed['render_mode'] == 'browser'
            assert failed['static_source_id'] == output['static_source_id']
        assert len(requests) == len(rendered) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize('options', [{'render': False}, {'view': 'raw'}])
async def test_static_and_raw_reads_do_not_start_a_browser(tmp_path, monkeypatch, options):
    async def unexpected(*args, **kwargs):
        pytest.fail('Static or raw reads must not render')

    monkeypatch.setattr(web_browser, 'render_public_page', unexpected)
    tools, client, _ = web_tools(tmp_path, lambda _: httpx.Response(
        200, content=EMPTY_HTML, headers={'content-type': 'text/html'}))
    try:
        output = await tools['web_fetch'].run({'url': 'https://example.com/document', **options})
        assert output['render_mode'] == 'http'
        assert output['error_code'] == (None if options.get('view') == 'raw' else 'web_content_empty')
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_raw_url_cache_does_not_prevent_later_automatic_recovery(tmp_path, monkeypatch):
    async def render(client, url, *, initial_response):
        return FetchedResponse(url, 'text/html', b'<main>Published source facts.</main>', 123, 'direct')

    monkeypatch.setattr(web_browser, 'render_public_page', render)
    tools, client, _ = web_tools(tmp_path, lambda _: httpx.Response(
        200, content=EMPTY_HTML, headers={'content-type': 'text/html'}))
    try:
        tool = tools['web_fetch']
        raw = await tool.run({'url': 'https://example.com/document', 'view': 'raw'})
        output = await tool.run({'url': 'https://example.com/document'})
        assert raw['render_mode'] == 'http' and raw['error_code'] is None
        assert output['render_mode'] == 'browser' and output['error_code'] is None
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize('body, media, status, expected', [
    (b'<main>Static source facts.</main><script src="/app.js"></script>', 'text/html', 200, None),
    (b'{invalid}', 'application/json', 200, 'web_content_invalid'),
    (b'opaque', 'application/octet-stream', 200, 'web_content_unsupported'),
    (b'blocked', 'text/html', 403, 'http_error'),
])
async def test_default_recovery_does_not_guess_from_scripts_or_retry_other_failures(
    tmp_path, monkeypatch, body, media, status, expected,
):
    async def unexpected(*args, **kwargs):
        pytest.fail('Only empty extracted HTML triggers rendering')

    monkeypatch.setattr(web_browser, 'render_public_page', unexpected)
    tools, client, _ = web_tools(tmp_path, lambda _: httpx.Response(
        status, content=body, headers={'content-type': media}))
    try:
        output = await tools['web_fetch'].run({'url': 'https://example.com/document'})
        assert output['error_code'] == expected
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize('code', ['web_browser_unavailable', 'web_browser_timeout'])
async def test_auto_browser_failure_preserves_original_response_and_typed_error(tmp_path, monkeypatch, code):
    async def fail(client, url, *, initial_response):
        failure = PublicWebError(code, 'Browser attempt failed.')
        if code == 'web_browser_timeout':
            failure.render_diagnostics = {'wire_bytes': 1000, 'requests': 3}
        raise failure

    monkeypatch.setattr(web_browser, 'render_public_page', fail)
    tools, client, sources = web_tools(tmp_path, lambda _: httpx.Response(
        200, content=EMPTY_HTML, headers={'content-type': 'text/html'}))
    try:
        output = await tools['web_fetch'].run({'url': 'https://example.com/document'})
        assert output['error_code'] == code
        assert output['source_id'] == output['static_source_id']
        assert output['network_bytes'] == (1000 if code == 'web_browser_timeout' else len(EMPTY_HTML))
        raw = await tools['web_fetch'].run({'source_id': output['source_id'], 'view': 'raw'})
        assert raw['error_code'] is None and EMPTY_HTML.decode() in raw['content']
        assert sources
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_executor_deadline_allows_both_reading_stages(tmp_path, monkeypatch):
    from agent_runtime.tools.builtins import web
    from agent_runtime.tools.web_content import ExtractedContent

    # Scale a legal 20s HTTP + 5s parse + 25s render + 5s parse trace by 0.01.
    # The old 45s Tool deadline cancels this trace before it can return evidence.
    async def handler(request):
        await asyncio.sleep(0.2)
        return httpx.Response(200, content=EMPTY_HTML, headers={'content-type': 'text/html'})

    async def parse(body, content_type, url):
        await asyncio.sleep(0.05)
        if body == EMPTY_HTML:
            raise PublicWebError('web_content_empty', 'No extracted text was found.')
        return ExtractedContent(title='Source', text='Published source facts.')

    async def render(client, url, *, initial_response):
        await asyncio.sleep(0.25)
        return FetchedResponse(url, 'text/html', b'<main>Published source facts.</main>', 123, 'direct')

    monkeypatch.setattr(web, 'extract_content_async', parse)
    monkeypatch.setattr(web_browser, 'render_public_page', render)
    tools, client, _ = web_tools(tmp_path, handler)
    tool = tools['web_fetch']
    tools['web_fetch'] = replace(tool, timeout_seconds=tool.timeout_seconds * 0.01)
    try:
        result = await ToolExecutor(tools).execute(call('web_fetch', {'url': 'https://example.com/document'}),
                                                 context=ToolExecutionContext(allow_web_tools=True))
        assert not result.result.is_error
        assert 'Published source facts.' in result.result.structured_content['content']
        assert result.result.structured_content['static_source_id']
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_empty_rendered_result_stops_after_one_browser_attempt(tmp_path, monkeypatch):
    rendered = []

    async def empty(client, url, *, initial_response):
        rendered.append(url)
        return FetchedResponse(url, 'text/html', EMPTY_HTML, 123, 'direct')

    monkeypatch.setattr(web_browser, 'render_public_page', empty)
    tools, client, _ = web_tools(tmp_path, lambda _: httpx.Response(
        200, content=EMPTY_HTML, headers={'content-type': 'text/html'}))
    try:
        tool = tools['web_fetch']
        output = await tool.run({'url': 'https://example.com/document'})
        assert output['error_code'] == 'web_content_empty' and output['render_mode'] == 'browser'
        repeat = await tool.run({'url': 'https://example.com/document'})
        assert repeat['cache_hit'] and repeat['source_id'] == output['source_id']
        assert len(rendered) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_cancelled_recovery_keeps_static_evidence(tmp_path, monkeypatch):
    async def cancel(client, url, *, initial_response):
        raise asyncio.CancelledError

    monkeypatch.setattr(web_browser, 'render_public_page', cancel)
    tools, client, sources = web_tools(tmp_path, lambda _: httpx.Response(
        200, content=EMPTY_HTML, headers={'content-type': 'text/html'}))
    try:
        with pytest.raises(asyncio.CancelledError):
            await tools['web_fetch'].run({'url': 'https://example.com/document'})
        assert any(base64.b64decode(json.loads(blob)['raw_body_base64']) == EMPTY_HTML for blob in sources.values())
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.skipif(os.environ.get('PRAXIS_TEST_BROWSER') != '1' or sys.platform == 'darwin',
                    reason='Requires real isolated Linux Chromium')
@pytest.mark.parametrize('path', ['/document', '/redirect'])
async def test_default_tool_runs_real_browser_without_refetching_document(tmp_path, path):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == '/redirect':
            return httpx.Response(302, headers={'location': '/document'})
        if request.url.path == '/app.js':
            return httpx.Response(200, headers={'content-type': 'application/javascript'},
                                  content=b'document.getElementById("app").textContent="Published source facts.";')
        return httpx.Response(200, content=EMPTY_HTML, headers={'content-type': 'text/html'})

    tools, client, _ = web_tools(tmp_path, handler)
    try:
        output = await tools['web_fetch'].run({'url': 'https://example.com' + path})
        assert output['error_code'] is None and 'Published source facts.' in output['content']
        assert output['render_mode'] == 'browser'
        expected_paths = ['/document', '/app.js'] if path == '/document' else ['/redirect', '/document', '/app.js']
        assert output['render_diagnostics']['requests'] == len(expected_paths)
        assert output['render_diagnostics']['resource_failures'] == {}
        assert [request.url.path for request in requests] == expected_paths
        assert requests[-1].headers['referer'] == 'https://example.com/'
    finally:
        await client.aclose()

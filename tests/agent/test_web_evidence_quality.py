"""Negative evidence and valid counterexamples, through the actual Tool envelope."""
from __future__ import annotations

import gzip
import json

import httpx
import pytest

from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from tests.agent.test_web_tools import call, web_tools


@pytest.mark.anyio
async def test_http_preserves_only_bounded_actual_cors_response_headers():
    from agent_runtime.tools.web_http import PublicWebClient
    from tests.agent.test_web_http import _Stream

    client = PublicWebClient(transport=httpx.MockTransport(lambda _: httpx.Response(
        200, headers={
            'Content-Type': 'application/json', 'Content-Encoding': 'gzip',
            'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Expose-Headers': 'X-Result', 'Set-Cookie': 'session=private',
            'X-Result': 'unrelated',
        }, stream=_Stream([gzip.compress(b'{"jobs":[]}')]),
    )))
    try:
        result = await client.get('https://example.com/api')
        assert result.body == b'{"jobs":[]}'
        assert result.response_headers == {
            'access-control-allow-origin': '*', 'access-control-allow-credentials': 'true',
            'access-control-expose-headers': 'X-Result',
        }
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_http_does_not_invent_or_copy_oversized_cors_headers():
    from agent_runtime.tools.web_http import PublicWebClient

    client = PublicWebClient(transport=httpx.MockTransport(lambda _: httpx.Response(
        200, headers={'Access-Control-Allow-Origin': 'x' * 1025}, content=b'ok',
    )))
    try:
        assert (await client.get('https://example.com/api')).response_headers == {}
    finally:
        await client.aclose()


def search_page(rows):
    return '<ol id="b_results">' + ''.join(
        f'<li class="b_algo"><h2><a href="{url}">{title}</a></h2>'
        f'<div class="b_caption"><p>{snippet}</p></div></li>' for url, title, snippet in rows
    ) + '</ol>'


@pytest.mark.anyio
@pytest.mark.parametrize('query,rows', [
    ('Python OR Rust', [('https://docs.python.org/', 'Python documentation', 'Python language')]),
    ('数据分析师岗位', [('https://example.com/jobs', 'Data Analyst', 'Open positions')]),
    ('site:example.com docs', [('https://other.example/docs', 'Documentation', 'Reference')]),
    ('北京 Agent 开发工程师 招聘 大模型', [('https://example.com/beijing', '北京市百科', '北京旅游景点')]),
])
async def test_search_preserves_backend_leads_without_query_interpretation(tmp_path, query, rows):
    hits = []
    tools, client, _ = web_tools(tmp_path, lambda req: hits.append(req) or httpx.Response(
        200, text=search_page(rows), headers={'content-type': 'text/html'}))
    try:
        execution = await ToolExecutor(tools).execute(call('web_search', {'query': query}),
                                                    context=ToolExecutionContext(allow_web_tools=True))
        output = execution.result.structured_content
        assert not execution.result.is_error
        assert hits[0].url.params['q'] == query
        assert list(output['results']) == [dict(url=url, title=title, snippet=snippet) for url, title, snippet in rows]
        for key in ('matched_query_terms', 'missing_query_terms', 'warning'):
            assert key not in output
        assert output['result_status'] == 'results'
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize('body,text', [
    ('<div id="app">加载中，请稍候</div><script src="/app.js"></script>', '加载中，请稍候'),
    ('<main><h1>Sign in to continue</h1><form><input type="password"></form></main>', 'Sign in to continue'),
    ('<form><h1>Sign in to continue</h1><input type="password"></form>', 'Sign in to continue'),
    ('<dialog id="captcha">Verify you are human</dialog>', 'Verify you are human'),
    ('<noscript>Please enable JavaScript</noscript>', 'Please enable JavaScript'),
    ('<nav><a href="/jobs">Jobs</a><a href="/about">About</a></nav>', 'Jobs'),
    ('<main><h1>API</h1><p>Use GET /v1.</p></main>', 'Use GET /v1.'),
])
async def test_page_text_is_evidence_without_automatic_task_verdict(tmp_path, body, text):
    hits = []
    tools, client, sources = web_tools(tmp_path, lambda req: hits.append(req) or httpx.Response(
        200, text=body, headers={'content-type': 'text/html'}))
    try:
        result = await ToolExecutor(tools).execute(call('web_fetch', {'url': 'https://example.com/page'}),
                                                 context=ToolExecutionContext(allow_web_tools=True))
        output = result.result.structured_content
        assert not result.result.is_error
        assert text in output['content']
        assert 'content_status' not in output
        raw = await tools['web_fetch'].run({'source_id': output['source_id'], 'view': 'raw'})
        assert body in raw['content'] and raw['error_code'] is None
        assert 'content_status' not in json.loads(sources[output['source_id']])
        assert len(hits) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize('body,media,code', [
    (b'<body><script src="/app.js"></script></body>', 'text/html', 'web_content_empty'),
    (b'{invalid json}', 'application/json', 'web_content_invalid'),
    (b'opaque response', 'application/octet-stream', 'web_content_unsupported'),
])
async def test_failed_extraction_saves_raw_response_and_cached_failure(tmp_path, body, media, code):
    hits = []
    tools, client, sources = web_tools(tmp_path, lambda req: hits.append(req) or httpx.Response(
        200, content=body, headers={'content-type': media}))
    try:
        output = await tools['web_fetch'].run({'url': 'https://example.com/source'})
        assert output['error_code'] == code
        assert output['source_id'] in sources
        assert output['network_bytes'] == len(body)
        assert 'JavaScript' not in output['error_message'] and 'authentication' not in output['error_message']
        raw = await tools['web_fetch'].run({'source_id': output['source_id'], 'view': 'raw'})
        assert raw['error_code'] is None and body.decode() in raw['content']
        again = await tools['web_fetch'].run({'url': 'https://example.com/source'})
        assert again['error_code'] == code and again['cache_hit'] and again['network_bytes'] == 0
        assert len(hits) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_parser_timeout_still_has_saved_raw_evidence(tmp_path, monkeypatch):
    from agent_runtime.tools.builtins import web
    from agent_runtime.tools.web_http import PublicWebError

    async def fail(*args):
        raise PublicWebError('web_content_timeout', 'Document parsing exceeded its time limit.')
    monkeypatch.setattr(web, 'extract_content_async', fail)
    tools, client, sources = web_tools(tmp_path, lambda _: httpx.Response(
        200, text='<main>Original evidence</main>', headers={'content-type': 'text/html'}))
    try:
        output = await tools['web_fetch'].run({'url': 'https://example.com/source'})
        assert output['error_code'] == 'web_content_timeout'
        assert output['source_id'] in sources
        raw = await tools['web_fetch'].run({'source_id': output['source_id'], 'view': 'raw'})
        assert 'Original evidence' in raw['content'] and raw['error_code'] is None
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_legacy_heuristic_labels_do_not_control_snapshot_reads(tmp_path):
    import hashlib

    tools, client, sources = web_tools(tmp_path, lambda _: pytest.fail('Offline read only'))
    identity = 'artifact_' + 'a' * 32
    sources[identity] = json.dumps(dict(version=2, url='https://example.com/form', title='Form',
        fetched_at='2026-10-08T00:00:00Z', text='Sign in to continue',
        content_hash=hashlib.sha256(b'Sign in to continue').hexdigest(),
        content_status='authentication_required', warning='Not the requested content.')).encode()
    try:
        output = await tools['web_fetch'].run({'source_id': identity})
        assert output['error_code'] is None and 'Sign in to continue' in output['content']
        assert 'content_status' not in output and output['warning'] is None
    finally:
        await client.aclose()


@pytest.mark.parametrize('error', [False, True])
def test_compaction_model_projection_preserves_execution_facts(tmp_path, error):
    from agent_runtime.harness import RolloutContextManager, RolloutStore
    from tests.agent.harness.test_compaction_consistency import start
    from tests.agent.harness.test_tool_result_elision import seed

    source = ({'source_id': 'artifact_' + 'b' * 32, 'url': 'https://example.com/source',
               'content': '', 'start_line': 1, 'next_line': None,
               'error_code': 'web_content_empty', 'error_message': 'No extracted text was found.',
               'render_mode': 'http', 'connection_mode': 'direct', 'failure_stage': 'extraction'} if error else
              {'query': 'Python OR Rust', 'provider': 'bing', 'result_status': 'results',
               'results': [{'url': 'https://docs.python.org', 'title': 'Python', 'snippet': 'Reference'}]})
    with RolloutStore(tmp_path / 'r.db') as store:
        _, turn = start(store, tmp_path)
        seed(store, turn_id=turn.turn_id, kind='model_response', payload={
            'text': 'Investigation history. ' * 2000,
            'tool_calls': [{'id': 'web', 'name': 'web_fetch' if error else 'web_search', 'arguments': {}}],
        })
        item = seed(store, turn_id=turn.turn_id, kind='tool_result', payload={
            'tool_call_id': 'web', 'is_error': error, 'structured_content': source,
            'model_content': json.dumps({'is_error': error, 'structured_content': source}),
        })
        manager = RolloutContextManager(store)
        digest, _ = manager.semantic_source(turn.turn_id)
        summary = manager.semantic_candidate(turn.turn_id, source_hash=digest, summary='Sources inspected.')
        ref = next(r for r in json.loads(summary.payload_json)['artifact_refs'] if r.get('item_id') == item.item_id)
        assert ref['is_error'] is error
        for key in ('error_code', 'error_message', 'render_mode', 'connection_mode', 'failure_stage') if error else (
                'query', 'provider', 'result_status'):
            assert ref[key] == source[key]
        manager.commit_compaction(summary)
        visible = str(manager.build(turn.turn_id))
        assert item.item_id in visible
        assert source.get('source_id', source.get('query')) in visible
        if error:
            assert source['error_message'] in visible


@pytest.mark.anyio
async def test_shared_network_budget_counts_redirects_and_stops_before_next_request():
    from agent_runtime.tools.web_http import PublicWebClient, PublicWebError

    hits = []
    def handler(request):
        hits.append(request)
        return httpx.Response(302, headers={'location': 'https://example.com/next'})
    client = PublicWebClient(transport=httpx.MockTransport(handler))
    calls = 0
    def request_budget():
        nonlocal calls
        calls += 1
        if calls > 2:
            raise PublicWebError('web_browser_budget_exceeded', 'Request budget exceeded.')
    try:
        with pytest.raises(PublicWebError, match='Request budget'):
            await client.get('https://example.com/start', request_budget=request_budget)
        assert len(hits) == 2
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_shared_network_byte_budget_counts_bounded_decoding_and_closes_response():
    import gzip

    from agent_runtime.tools.web_http import PublicWebClient, PublicWebError
    from tests.agent.test_web_http import _Stream

    compressed = gzip.compress(b'x' * 2000)
    stream = _Stream([compressed])
    client = PublicWebClient(transport=httpx.MockTransport(lambda _: httpx.Response(
        200, headers={'content-encoding': 'gzip'}, stream=stream)))
    totals = [0, 0]
    def byte_budget(wire, decoded):
        totals[0] += wire
        totals[1] += decoded
        if totals[1] > 1000:
            raise PublicWebError('web_browser_budget_exceeded', 'Byte budget exceeded.')
    try:
        with pytest.raises(PublicWebError, match='Byte budget'):
            await client.get('https://example.com/doc', byte_budget=byte_budget)
        assert totals == [len(compressed), 2000]
        assert stream.closed
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_large_query_diagnostics_stay_within_complete_model_envelope(tmp_path):
    from agent_runtime.core.messages import tool_result_message

    query = ' '.join(chr(0x4e00 + n) * 29 for n in range(64))
    results = [{'title': 'Title' * 50, 'url': f'https://example.com/{n}', 'description': 'text ' * 400}
               for n in range(10)]
    tools, client, _ = web_tools(tmp_path, lambda _: httpx.Response(200, json={'web': {'results': results}}),
                                key='fake-search-secret')
    try:
        execution = await ToolExecutor(tools).execute(call('web_search', {'query': query, 'max_results': 10}),
                                                     context=ToolExecutionContext(allow_web_tools=True))
        assert not execution.result.is_error, execution.result.error_code
        assert len(tool_result_message(execution.result).content.encode()) <= 65_536
        assert execution.result.structured_content['results_truncated'] is True
        assert 'matched_query_terms' not in execution.result.structured_content
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_render_snapshot_uses_separate_cache_and_offline_source_continuation(tmp_path, monkeypatch):
    from agent_runtime.tools import web_browser
    from agent_runtime.tools.web_http import FetchedResponse

    network = []
    rendered = []
    tools, client, _ = web_tools(tmp_path, lambda req: network.append(req) or httpx.Response(
        200, text='<div>Loading</div><script src="/app.js"></script>',
        headers={'content-type': 'text/html'}))
    async def renderer(client, url):
        rendered.append(url)
        return FetchedResponse(url, 'text/html', b'<main><h1>Jobs</h1><p>Agent engineer in Beijing.</p></main>',
                               80, 'direct')
    monkeypatch.setattr(web_browser, 'render_public_page', renderer)
    try:
        tool = tools['web_fetch']
        static = await tool.run({'url': 'https://example.com/jobs'})
        dynamic = await tool.run({'url': 'https://example.com/jobs', 'render': True})
        repeat = await tool.run({'url': 'https://example.com/jobs', 'render': True})
        offline = await ToolExecutor(tools).execute(call('web_fetch', {'source_id': dynamic['source_id']}),
                                                  context=ToolExecutionContext())
        assert static['error_code'] is None and 'Loading' in static['content']
        assert dynamic['render_mode'] == 'browser' and 'Agent engineer' in dynamic['content']
        assert dynamic['source_id'] != static['source_id']
        assert repeat['cache_hit'] and repeat['source_id'] == dynamic['source_id']
        assert not offline.result.is_error
        assert offline.result.structured_content['render_mode'] == 'browser'
        assert len(network) == 1 and len(rendered) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_render_unavailable_reports_requested_mode_without_http_fallback(tmp_path, monkeypatch):
    from agent_runtime.tools import web_browser
    from agent_runtime.tools.web_http import PublicWebError

    network = []
    tools, client, _ = web_tools(tmp_path, lambda req: network.append(req) or httpx.Response(200, text='fallback'))
    async def unavailable(client, url):
        raise PublicWebError('web_browser_unavailable', 'Isolation unavailable.')
    monkeypatch.setattr(web_browser, 'render_public_page', unavailable)
    try:
        result = await ToolExecutor(tools).execute(call('web_fetch', {
            'url': 'https://example.com/jobs', 'render': True,
        }), context=ToolExecutionContext(allow_web_tools=True))
        assert result.result.is_error and result.result.error_code == 'web_browser_unavailable'
        assert result.result.structured_content['render_mode'] == 'browser'
        assert not network
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_explicit_asset_response_budget_does_not_mutate_default_document_limit():
    from agent_runtime.tools.web_http import PublicWebClient, PublicWebError

    client = PublicWebClient(max_bytes=10, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=b'x' * 20)))
    try:
        with pytest.raises(ValueError):
            await client.get('https://example.com/asset.js', max_response_bytes=0)
        assert (await client.get('https://example.com/asset.js', max_response_bytes=20)).body == b'x' * 20
        with pytest.raises(PublicWebError) as error:
            await client.get('https://example.com/document')
        assert error.value.code == 'response_too_large'
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_explicit_asset_limit_preserves_decoding_and_aggregate_budget():
    import gzip

    from agent_runtime.tools.web_http import PublicWebClient, PublicWebError
    from tests.agent.test_web_http import _Stream

    streams = []
    def handler(_):
        stream = _Stream([gzip.compress(b'x' * 200)])
        streams.append(stream)
        return httpx.Response(200, headers={'content-encoding': 'gzip'}, stream=stream)
    client = PublicWebClient(max_bytes=100, transport=httpx.MockTransport(handler))
    try:
        response = await client.get('https://example.com/app.js', max_response_bytes=300)
        assert response.body == b'x' * 200
        with pytest.raises(PublicWebError) as small:
            await client.get('https://example.com/doc')
        assert small.value.code == 'response_too_large'
        def aggregate(wire, decoded):
            if decoded > 150:
                raise PublicWebError('web_browser_budget_exceeded', 'Aggregate budget exceeded.')
        with pytest.raises(PublicWebError) as total:
            await client.get('https://example.com/app.js', max_response_bytes=300, byte_budget=aggregate)
        assert total.value.code == 'web_browser_budget_exceeded'
        assert all(stream.closed for stream in streams)
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize('body,error', [
    (b'<main><h1>Jobs</h1><p>Agent engineer in Beijing.</p></main>', False),
    (b'<body><div>Loading</div><script src="/app.js"></script></body>', False),
])
async def test_partial_render_warning_survives_content_classification_and_saved_continuation(
    tmp_path, monkeypatch, body, error,
):
    from agent_runtime.tools import web_browser
    from agent_runtime.tools.web_http import FetchedResponse

    warning = 'Some public page resources failed to load (network_error: 1); inspect the saved body.'
    tools, client, sources = web_tools(tmp_path, lambda _: pytest.fail('No fallback HTTP request'))
    async def renderer(client, url):
        return FetchedResponse(url, 'text/html', body, len(body), 'direct', warning=warning)
    monkeypatch.setattr(web_browser, 'render_public_page', renderer)
    try:
        result = await ToolExecutor(tools).execute(call('web_fetch', {
            'url': 'https://example.com/jobs', 'render': True,
        }), context=ToolExecutionContext(allow_web_tools=True))
        output = result.result.structured_content
        assert result.result.is_error is error
        assert 'content_status' not in output and warning in output['warning']
        snapshot = json.loads(sources[output['source_id']])
        assert snapshot['warning'] == output['warning']
        offline = await tools['web_fetch'].run({'source_id': output['source_id'], 'view': 'raw'})
        assert offline['warning'] == output['warning']
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_long_public_url_is_archived_before_parser_starts(tmp_path, monkeypatch):
    import base64

    from agent_runtime.tools.builtins import web
    from agent_runtime.tools.web_http import PublicWebError

    tools, client, sources = web_tools(tmp_path, lambda _: httpx.Response(
        200, text='<main>Unparsed evidence</main>', headers={'content-type': 'text/html'}))
    async def parser(*args):
        assert len(sources) == 1
        original = json.loads(next(iter(sources.values())))
        assert base64.b64decode(original['raw_body_base64']) == b'<main>Unparsed evidence</main>'
        raise PublicWebError('web_content_timeout', 'Parser deadline exceeded.')
    monkeypatch.setattr(web, 'extract_content_async', parser)
    try:
        output = await tools['web_fetch'].run({'url': 'https://example.com/' + 'x' * 600})
        assert output['error_code'] == 'web_content_timeout'
        assert len(sources) == 2
        raw = await tools['web_fetch'].run({'source_id': output['source_id'], 'view': 'raw'})
        assert 'Unparsed evidence' in raw['content']
    finally:
        await client.aclose()

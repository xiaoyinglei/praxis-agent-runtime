from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agent_runtime.tools.web_content import extract_content
from tests.agent.test_web_tools import web_tools

FIXTURES = Path(__file__).parent / 'fixtures' / 'web'


def test_article_starts_with_evidence_and_ignores_comment_examples():
    html = b'''<html><head><title>Evidence</title>
    <meta property="article:published_time" content="2026-09-29T10:00:00+08:00"></head>
    <body><nav>Global navigation</nav><!-- <p>comment script example</p> -->
    <div class="advertisement">Buy now</div><article><h1>Evidence title</h1>
    <time datetime="2026-09-29T10:00:00+08:00">September 29</time>
    <p>Actual source paragraph with useful factual evidence.</p>
    <p>Read the <a href="/announcement">original announcement</a>.</p>
    <aside>Unrelated recommendations</aside></article><footer>All rights reserved</footer></body></html>'''
    result = extract_content(html, 'text/html', 'https://example.com/news')
    for noise in ['Global navigation', 'comment script example', 'Buy now', 'Unrelated recommendations',
                  'All rights reserved']:
        assert noise not in result.text
    assert 'Actual source paragraph' in result.text
    assert result.published_at == '2026-09-29T10:00:00+08:00'
    assert result.links == ({'url': 'https://example.com/announcement', 'text': 'original announcement'},)


def test_docs_preserve_code_tables_and_content_links():
    body = b'<nav>Navigation</nav><main><h1>Guide</h1><pre>def f():\n    return 42</pre>' \
           b'<table><tr><th>Option</th><th>Value</th></tr><tr><td>Timeout</td><td>10</td></tr></table>' \
           b'<a href="/next">Next section</a></main>'
    result = extract_content(body, 'text/html', 'https://example.com/guide')
    assert 'Navigation' not in result.text
    assert 'def f():\n    return 42' in result.text
    assert 'Timeout\t10' in result.text
    assert result.links[0]['url'] == 'https://example.com/next'


@pytest.mark.anyio
async def test_fetch_budget_keeps_complete_cursor_and_never_loses_long_line_tail(tmp_path):
    text = 'HEAD ' + '中文正文' * 10000 + ' TAIL'
    hits = []
    tools, client, sources = web_tools(tmp_path, lambda req: hits.append(req) or httpx.Response(
        200, text=text, headers={'content-type': 'text/plain'}))
    try:
        tool = tools['web_fetch']
        output = await tool.run(tool.validate_input({'url': 'https://example.com/doc', 'max_bytes': 4096}))
        assert len(output['content'].encode()) <= 4096
        source_id = output['source_id']
        seen = output['content']
        for _ in range(100):
            if output['next_line'] is None:
                break
            output = await tool.run(tool.validate_input({
                'source_id': source_id, 'start_line': output['next_line'], 'max_bytes': 4096,
            }))
            assert output['source_id'] == source_id
            assert len(output['content'].encode()) <= 4096
            seen += output['content']
        assert 'TAIL' in seen
        assert len(hits) == 1 and len(sources) == 2
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_repeated_url_reuses_snapshot_and_does_not_repeat_network(tmp_path):
    hits = []
    tools, client, sources = web_tools(tmp_path, lambda req: hits.append(req) or httpx.Response(
        200, text='Evidence', headers={'content-type': 'text/plain'}))
    try:
        first = await tools['web_fetch'].run({'url': 'https://example.com/doc'})
        second = await tools['web_fetch'].run({'url': 'https://example.com/doc'})
        assert first['source_id'] == second['source_id']
        assert second['network_bytes'] == 0
        assert len(hits) == 1 and len(sources) == 2
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("extraction_error", [False, True])
async def test_web_cursor_survives_real_product_output_and_admission(tmp_path, monkeypatch, extraction_error):
    import agent_runtime.tools.web_http as http_module
    from agent_runtime import Agent
    from agent_runtime.harness import RolloutContextManager, RolloutStore, TurnExecutor
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
    from tests.agent.harness.test_compaction_consistency import Accept, model_for, start
    from tests.agent.test_web_product import FetchThenAnswer

    real_client = http_module.PublicWebClient
    monkeypatch.setattr(http_module, 'PublicWebClient', lambda **options: real_client(
        **options, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, text='<body><script></script></body>' if extraction_error
                                else 'evidence line\n' * 2000,
                                headers={'content-type': 'text/html' if extraction_error else 'text/plain'}))))
    database = tmp_path / 'state.sqlite'
    model = FetchThenAnswer({'url': 'https://example.com/doc', 'max_lines': 500})
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, '_harness_model', lambda: model)
    result = await agent.run('Read external evidence', allow_web_tools=True, require_workspace_change=False)
    with RolloutStore(database) as store:
        item = next(i for i in store.list_items(result.turn_id) if i.kind == 'tool_result')
        expected = item.payload['structured_content']
        thread, turn = start(store, tmp_path)
        if extraction_error:
            from tests.agent.harness.test_tool_result_elision import exchange
            exchange(store, turn.turn_id, 'prior', 'Prior inspection. ' * 3000)
        seed(store, turn_id=turn.turn_id, kind='model_response', payload={
            'text': '', 'tool_calls': [{'id': item.payload['tool_call_id'], 'name': 'web_fetch',
                                     'arguments': {'url': 'https://example.com/doc'}}],
        })
        seed(store, turn_id=turn.turn_id, kind='tool_result', payload=dict(item.payload))
        manager = RolloutContextManager(store, max_total_bytes=8000 if extraction_error else 5000)
        if extraction_error:
            digest, _ = manager.semantic_source(turn.turn_id)
            manager.commit_compaction(manager.semantic_candidate(
                turn.turn_id, source_hash=digest, summary='Earlier inspection and webpage extraction failed.',
            ))
        # Replay through real request admission, rather than testing only a helper.
        wire_model, gateway = model_for(20000)
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=wire_model,
                              context_manager=manager, completion_gate=Accept())
        _, prepared = await runner._prepare_compacted_step(runner.restore_turn_context(turn.turn_id), step=3)
        messages = prepared.request_ref['step_snapshot']['messages']
        if extraction_error:
            visible = str(messages)
            assert expected['source_id'] in visible and expected['error_message'] in visible
            contexts = [json.loads(m['content'].removeprefix('Context compaction:\n'))
                        for m in messages if m['role'] == 'context']
            data = next(ref for context in contexts for ref in context.get('artifact_refs', [])
                        if ref.get('source_id') == expected['source_id'])
            assert data['item_id'].startswith('item_')
        else:
            message = next(m for m in messages if m['role'] == 'tool')
            data = json.loads(message['content'])['structured_content']
        for key in ['source_id', 'url', 'source_truncated', 'content_hash']:
            assert data[key] == expected[key]
        if extraction_error:
            for key in ['error_code', 'error_message', 'render_mode', 'connection_mode', 'failure_stage']:
                assert data[key] == expected[key]
            assert data['error_code'] == 'web_content_empty'
        else:
            assert data['truncated'] is True
            assert data['fetch_next_line'] == expected['next_line']
            assert data['next_line'] <= expected['next_line']
        if not extraction_error:
            assert 'item_' in message['content']
        assert data['source_id'].startswith('artifact_')
        assert item.payload == store.read_item(item.item_id).payload
        if extraction_error:
            assert all(":context-summary:" in request.request_id for request in gateway.requests)
        else:
            assert not gateway.requests
        assert store.verify().valid


@pytest.mark.anyio
async def test_find_in_snapshot_locates_evidence_without_refetching(tmp_path):
    hits = []
    tools, client, _ = web_tools(tmp_path, lambda req: hits.append(req) or httpx.Response(
        200, text='intro\n' * 300 + 'Target API details\nlast line', headers={'content-type': 'text/plain'}))
    try:
        first = await tools['web_fetch'].run({'url': 'https://example.com/doc', 'max_lines': 2})
        tool = tools['web_fetch']
        output = await tool.run(tool.validate_input({'source_id': first['source_id'], 'find': 'Target API'}))
        assert output['start_line'] == 301
        assert 'Target API details' in output['content']
        missing = await tool.run(tool.validate_input({'source_id': first['source_id'], 'find': 'Absent'}))
        assert missing['error_code'] == 'web_text_not_found'
        assert missing['source_id'] == first['source_id']
        assert len(hits) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_search_retains_backend_results_and_unverified_freshness(tmp_path):
    hits = []
    body = '<ol id="b_results"><li class="b_algo"><h2><a href="https://example.com/mineral">水晶矿物</a>' \
           '</h2><div class="b_caption"><p>石英百科介绍</p></div></li></ol>'
    tools, client, _ = web_tools(tmp_path, lambda req: hits.append(req) or httpx.Response(
        200, text=body, headers={'content-type': 'text/html'}))
    try:
        first = await tools['web_search'].run({'query': '水晶光电 最新消息', 'freshness': 'pw'})
        assert first['result_status'] == 'results'
        assert first['freshness_verified'] is False
        assert first['freshness_requested'] == 'pw'
        again = await tools['web_search'].run({'query': '水晶光电 最新消息', 'freshness': 'pw'})
        assert again['cache_hit'] is True
        assert len(hits) == 1
    finally:
        await client.aclose()


def test_semantic_summary_and_zero_body_elision_keep_source_identity(tmp_path):
    from agent_runtime.harness import RolloutContextManager, RolloutStore
    from tests.agent.harness.test_compaction_consistency import start
    from tests.agent.harness.test_tool_result_elision import exchange

    source = {'source_id': 'artifact_' + 'a' * 32, 'url': 'https://example.com/source',
              'content': '1: evidence\n' * 200, 'start_line': 1, 'next_line': 201,
              'source_truncated': False, 'truncated': True, 'content_hash': 'b' * 64}
    with RolloutStore(tmp_path / 'r.db') as store:
        _, turn = start(store, tmp_path)
        exchange(store, turn.turn_id, 'page', json.dumps({'structured_content': source}))
        exchange(store, turn.turn_id, 'recent', 'Recent work ' * 1000)
        manager = RolloutContextManager(store)
        candidate = next(manager.cheap_candidates(turn.turn_id, max_result_bytes=0))
        page = next(m for m in candidate.messages if m.tool_call_id == 'page')
        assert json.loads(page.content)['structured_content']['source_id'] == source['source_id']
        unread = json.loads(page.content)['structured_content']['next_line']
        manager.commit_compaction(candidate)
        source_hash, _ = manager.semantic_source(turn.turn_id)
        summary = manager.semantic_candidate(turn.turn_id, source_hash=source_hash, summary='Evidence read.')
        refs = json.loads(summary.payload_json)['artifact_refs']
        assert any(ref.get('source_id') == source['source_id'] and ref.get('url') == source['url'] for ref in refs)
        assert next(ref for ref in refs if ref.get('source_id') == source['source_id'])['next_line'] == unread
        manager.commit_compaction(summary)
        assert source['source_id'] in str(manager.build(turn.turn_id))
        assert store.verify().valid
    with RolloutStore(tmp_path / 'r.db') as store:
        assert source['source_id'] in str(RolloutContextManager(store).build(turn.turn_id))


def test_paragraph_document_discards_link_only_promotions_and_qr_widgets():
    body = ('<div><p><a href="/promo">A long promotion link with no article evidence</a></p>'
            '<p>' + 'Useful factual article evidence. ' * 10 + '</p>'
            '<div class="inlineQr_wrap">Scan for unrelated promotion</div></div>').encode()
    result = extract_content(body, 'text/html', 'https://example.com/news')
    assert 'Useful factual' in result.text
    assert 'promotion' not in result.text


@pytest.mark.anyio
async def test_query_variants_report_identical_results_as_execution_fact(tmp_path):
    body = '<ol id="b_results"><li class="b_algo"><h2><a href="https://example.com/page">Unrelated</a>' \
           '</h2><div class="b_caption"><p>Same snippet</p></div></li></ol>'
    tools, client, _ = web_tools(tmp_path, lambda req: httpx.Response(
        200, text=body, headers={'content-type': 'text/html'}))
    try:
        await tools['web_search'].run({'query': 'different first query'})
        second = await tools['web_search'].run({'query': 'another query'})
        assert second['result_status'] == 'repeated_results'
        assert second['previous_query'] == 'different first query'
        assert len(second['results']) == 1
        assert second['results'][0]['url'] == 'https://example.com/page'
        assert 'warning' not in second and 'missing_query_terms' not in second['results'][0]
    finally:
        await client.aclose()


def test_multiregion_page_preserves_relevant_news_links_outside_dense_body():
    body = ('<title>Example Corporation (EX123) - Overview</title><nav><a href="/help">Help</a></nav>'
            '<div><p>' + 'Example Corporation company description. ' * 8 + '</p></div>'
            '<section><h2>News</h2><a href="/news/report">Example Corporation publishes a new report</a>'
            '<a href="/news/unrelated">Other unrelated article</a></section>').encode()
    result = extract_content(body, 'text/html', 'https://example.com/company')
    assert {'url': 'https://example.com/news/report',
            'text': 'Example Corporation publishes a new report'} in result.links
    assert all(link['url'] != 'https://example.com/help' for link in result.links)
    assert 'Example Corporation publishes a new report' in result.text


@pytest.mark.anyio
async def test_json_escaping_cannot_overflow_executor_or_elide_cursor(tmp_path):
    from agent_runtime.core.messages import tool_result_message
    from agent_runtime.tools.executor import ToolExecutor
    from agent_runtime.tools.permissions import ToolExecutionContext
    from tests.agent.test_web_tools import call

    tools, client, _ = web_tools(tmp_path, lambda req: httpx.Response(
        200, content=b'\x01' * 40000, headers={'content-type': 'text/plain'}))
    try:
        result = await ToolExecutor(tools).execute(call('web_fetch', {
            'url': 'https://example.com/doc', 'max_bytes': 16000, 'max_lines': 500,
        }), context=ToolExecutionContext(allow_web_tools=True))
        assert not result.result.is_error, result.result.error_code
        envelope = json.loads(tool_result_message(result.result).content)
        assert envelope['structured_content']['source_id'].startswith('artifact_')
        assert envelope['structured_content']['next_line'] is not None
    finally:
        await client.aclose()


def test_fixed_corpus_quality_acceptance():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(root / 'scripts/agent_web_quality.py')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.anyio
async def test_search_envelope_is_bounded_after_json_escaping(tmp_path):
    from agent_runtime.tools.executor import ToolExecutor
    from agent_runtime.tools.permissions import ToolExecutionContext
    from tests.agent.test_web_tools import call

    payload = {'web': {'results': [{'title': '\x01' * 500, 'description': '\x01' * 2000,
                                  'url': f'https://example.com/{i}'} for i in range(10)]}}
    tools, client, _ = web_tools(tmp_path, lambda req: httpx.Response(200, json=payload), key='fake-search-secret')
    try:
        result = await ToolExecutor(tools).execute(call('web_search', {'query': 'docs', 'max_results': 10}),
                                                 context=ToolExecutionContext(allow_web_tools=True))
        assert not result.result.is_error, result.result.error_code
        assert result.result.structured_content['query'] == 'docs'
    finally:
        await client.aclose()

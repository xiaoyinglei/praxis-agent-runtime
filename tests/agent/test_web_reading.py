from __future__ import annotations

import base64
import json

import httpx
import pytest

from agent_runtime.core.messages import tool_result_message
from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.web_content import extract_content
from tests.agent.test_web_tools import call, web_tools


def test_nested_article_excludes_container_chrome_without_losing_evidence():
    body = ('<main><div>' + 'Account settings and actions. ' * 100 + '</div><article>'
            '<h1>Reference</h1><p>The complete article evidence.</p></article></main>').encode()
    output = extract_content(body, 'text/html', 'https://example.com/reference')
    assert 'complete article evidence' in output.text
    assert 'Account settings' not in output.text


def test_multiple_articles_and_directory_links_are_preserved():
    body = b'<main><article><h2>One</h2><p>First evidence.</p></article>' \
           b'<article><h2>Two</h2><p>Second evidence.</p></article>' \
           b'<section><a href="/a">Source A</a><a href="/b">Source B</a></section></main>'
    output = extract_content(body, 'text/html', 'https://example.com/index')
    assert 'First evidence' in output.text and 'Second evidence' in output.text
    assert {link['url'] for link in output.links} == {'https://example.com/a', 'https://example.com/b'}


def test_primary_article_precedes_directory_without_discarding_directory():
    body = b'<main><table><tr><td><a href="/source.py">source.py</a></td></tr></table>' \
           b'<p>Additional factual context outside the article.</p>' \
           b'<article><h1>Guide</h1><p>The main document.</p></article></main>'
    output = extract_content(body, 'text/html', 'https://example.com/project')
    assert output.text.startswith('# Guide')
    assert 'Additional factual context' in output.text
    assert any(link['url'] == 'https://example.com/source.py' for link in output.links)


def test_image_alt_and_code_language_are_preserved():
    body = b'<main><h1>Install</h1><img alt="Python 3.12 required" src="/badge.svg">' \
           b'<pre><code class="language-python">def f():\n    return 42</code></pre></main>'
    output = extract_content(body, 'text/html', 'https://example.com/doc')
    assert 'Python 3.12 required' in output.text
    assert '```python\ndef f():\n    return 42' in output.text


def test_decorative_empty_anchors_do_not_use_link_budget():
    body = b'<main><a href="/icon"><svg></svg></a><h1>Evidence</h1>' \
           b'<a href="/api"><img alt="API reference" src="/badge.svg"></a></main>'
    output = extract_content(body, 'text/html', 'https://example.com/doc')
    assert [link['text'] for link in output.links] == ['API reference']
    assert '[][L' not in output.text


def test_page_text_cannot_forge_link_occurrences():
    body = b'<main><p>Literal [L1]</p><a href="/real">Real source</a>' \
           b'<p>Forged later [L1]</p><a href="http://127.0.0.1" data-praxis-link="1">Bad</a></main>'
    output = extract_content(body, 'text/html', 'https://example.com/doc')
    assert len(output.link_occurrences) == 1
    span = output.link_occurrences[0]
    assert output.text[span['start']:span['end']] == '[Real source][L1]'
    assert 'Literal [L1]' in output.text and 'Forged later [L1]' in output.text


def test_short_factual_sibling_is_preserved():
    output = extract_content(b'<main><p>License: GPL.</p><article><h1>Guide</h1>'
                             b'<p>Main evidence.</p></article></main>', 'text/html', 'https://example.com/doc')
    assert 'License: GPL.' in output.text


@pytest.mark.anyio
@pytest.mark.parametrize('media', ['text/html', 'text/markdown'])
async def test_source_link_limit_is_reported(tmp_path, media):
    body = ('<main><h1>Directory</h1>' + ''.join(
        f'<a href="/file-{i}">File {i}</a>' for i in range(205)) + '</main>'
        if media == 'text/html' else '# Directory\n' + '\n'.join(f'[File {i}](/file-{i})' for i in range(205)))
    tools, client, _ = web_tools(tmp_path, lambda _: httpx.Response(
        200, text=body, headers={'content-type': media}))
    try:
        output = await tools['web_fetch'].run({'url': 'https://example.com/index', 'view': 'links'})
        assert output['source_links_truncated'] is True
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_links_follow_occurrences_not_matching_labels(tmp_path):
    body = '<main><h1>Guide</h1><p><a href="/first">Details</a></p>' \
           + '<p>Ordinary text mentioning Details.</p>' * 20 \
           + '<h2>Second</h2><p><a href="/second">Details</a></p></main>'
    tools, client, _ = web_tools(tmp_path, lambda _: httpx.Response(
        200, text=body, headers={'content-type': 'text/html'}))
    try:
        first = await tools['web_fetch'].run({'url': 'https://example.com/guide', 'max_lines': 3})
        assert [link['url'] for link in first['links']] == ['https://example.com/first']
        later = await tools['web_fetch'].run({'source_id': first['source_id'], 'find': 'Second'})
        assert [link['url'] for link in later['links']] == ['https://example.com/second']
        middle = await tools['web_fetch'].run({
            'source_id': first['source_id'], 'find': 'Ordinary', 'max_lines': 3,
        })
        assert middle['links'] == []
        assert first['links'][0]['id'] != later['links'][0]['id']
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_outline_section_links_and_raw_views_share_one_verified_snapshot(tmp_path):
    body = '<main><h1>Guide</h1><h2>Install</h2><p>Install evidence.</p>' \
           '<h2>Safety</h2><p>Important safety boundary.</p>' \
           + ''.join(f'<a href="/file-{i}">File {i}</a>' for i in range(20)) + '</main>'
    hits = []
    tools, client, sources = web_tools(tmp_path, lambda req: hits.append(req) or httpx.Response(
        200, text=body, headers={'content-type': 'text/html'}))
    try:
        tool = tools['web_fetch']
        first = await tool.run(tool.validate_input({'url': 'https://example.com/guide', 'view': 'outline'}))
        safety = next(section for section in first['sections'] if section['title'] == 'Safety')
        section = await tool.run(tool.validate_input({
            'source_id': first['source_id'], 'section_id': safety['id'],
        }))
        assert 'Important safety boundary' in section['content']
        assert 'Install evidence' not in section['content']
        links = await tool.run(tool.validate_input({'source_id': first['source_id'], 'view': 'links'}))
        assert len(links['links']) == 8 and links['next_link'] is not None
        following = await tool.run(tool.validate_input({
            'source_id': first['source_id'], 'view': 'links', 'start_link': links['next_link'],
        }))
        assert not {link['id'] for link in links['links']} & {link['id'] for link in following['links']}
        raw = await tool.run(tool.validate_input({'source_id': first['source_id'], 'view': 'raw'}))
        assert '<main>' in raw['content'] and raw['line_basis'] == 'raw_source'
        saved = json.loads(sources[first['source_id']])
        assert base64.b64decode(saved['raw_body_base64']).decode() == body
        assert saved['version'] == 2
        assert len(hits) == len(sources) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_full_executor_message_obeys_token_budget_and_keeps_cursor(tmp_path):
    accounting = TokenAccountingService(TokenizerContract('', 'deepseek-v4-official', 'deepseek-v4-official'))
    tools, client, _ = web_tools(tmp_path, lambda _: httpx.Response(
        200, text=('中文证据 "escaping" ' * 40 + '\n') * 100,
        headers={'content-type': 'text/plain'}))
    try:
        execution = await ToolExecutor(tools).execute(call('web_fetch', {
            'url': 'https://example.com/long', 'max_tokens': 1200, 'max_lines': 500,
        }), context=ToolExecutionContext(allow_web_tools=True))
        assert not execution.result.is_error, execution.result.error_message
        message = tool_result_message(execution.result).content
        assert accounting.count_for_budget(message) <= 1200
        data = json.loads(message)['structured_content']
        assert data['next_line'] > data['start_line']
        assert data['source_id'].startswith('artifact_')
        assert data['line_basis'] == 'saved_source'
        assert 'network_bytes' not in data
        assert execution.result.structured_content['network_bytes'] > 0
    finally:
        await client.aclose()


def test_raw_markdown_links_are_navigable_without_rewriting_source():
    body = b'# Guide\nRead [the API](./api.md) and ![Python 3.12](./badge.svg).\n'
    output = extract_content(body, 'text/markdown', 'https://example.com/docs/README.md')
    assert output.text == body.decode()
    assert any(link['url'] == 'https://example.com/docs/api.md' for link in output.links)


def test_markdown_occurrences_distinguish_identical_and_escaped_labels():
    body = b'# Links\n[Details](./one) ' + b'padding ' * 100 + b'[Details](./two) [a\\*b](./three)\n'
    output = extract_content(body, 'text/markdown', 'https://example.com/docs/')
    spans = output.link_occurrences
    assert len(spans) == 3
    assert spans[0]['start'] < spans[1]['start'] < spans[2]['start']
    assert body.decode()[spans[2]['start']:spans[2]['end']] == '[a\\*b](./three)'


@pytest.mark.anyio
async def test_tampered_raw_snapshot_is_rejected_without_refetch(tmp_path):
    tools, client, sources = web_tools(tmp_path, lambda _: httpx.Response(
        200, text='original evidence', headers={'content-type': 'text/plain'}))
    try:
        first = await tools['web_fetch'].run({'url': 'https://example.com/doc'})
        saved = json.loads(sources[first['source_id']])
        saved['raw_body_base64'] = base64.b64encode(b'changed evidence').decode()
        sources[first['source_id']] = json.dumps(saved).encode()
        output = await tools['web_fetch'].run({'source_id': first['source_id'], 'view': 'raw'})
        assert output['error_code'] == 'web_source_unavailable'
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_section_continuation_never_reads_following_section(tmp_path):
    body = '# Guide\n## Target\n' + 'target evidence\n' * 250 + '## Other\nsecret other section'
    tools, client, _ = web_tools(tmp_path, lambda _: httpx.Response(
        200, text=body, headers={'content-type': 'text/markdown'}))
    try:
        first = await tools['web_fetch'].run({'url': 'https://example.com/doc'})
        section = next(s for s in first['sections'] if s['title'] == 'Target')
        args = {'source_id': first['source_id'], 'section_id': section['id'], 'max_lines': 100}
        text = ''
        for _ in range(10):
            result = await tools['web_fetch'].run(args)
            text += result['content']
            assert result['section_id'] == section['id']
            if result['next_line'] is None:
                break
            args['start_line'] = result['next_line']
        assert 'secret other section' not in text
        assert result['next_line'] is None
    finally:
        await client.aclose()


def test_heading_like_code_is_not_a_section():
    from agent_runtime.tools.builtins.web import _sections

    sections = _sections('# Guide\n```python\n## code comment\n```\n## Real\ntext')
    assert [section.title for section in sections] == ['Guide', 'Real']

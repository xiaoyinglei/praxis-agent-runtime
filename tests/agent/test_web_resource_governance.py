from __future__ import annotations

import json

import httpx
import pytest

from agent_runtime import Agent
from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, RolloutStore
from agent_runtime.tools.web_http import PublicWebClient
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel
from tests.agent.test_web_tools import web_tools


def test_limits_reject_unbounded_or_coerced_values():
    from agent_runtime.tools.web_limits import WebLimits

    for payload in ({'turn_requests': 257}, {'browser_network_bytes': 32_000_001},
                    {'browser_requests': 129}, {'turn_requests': True}, {'turn_requests': '64'}):
        with pytest.raises(ValueError):
            WebLimits.model_validate(payload)
    assert WebLimits(turn_requests=64).turn_requests == 64


@pytest.mark.anyio
async def test_browser_limits_preserve_resource_failure_locations_without_credentials():
    from agent_runtime.tools.web_browser import _Broker
    from agent_runtime.tools.web_limits import WebLimits

    hits = []
    client = PublicWebClient(transport=httpx.MockTransport(lambda req: hits.append(req) or httpx.Response(
        200, content=b'x' * 600_000, headers={'content-type': 'application/javascript'})))
    try:
        broker = _Broker(client, limits=WebLimits(browser_network_bytes=1_000_000))
        first = await broker.reply({'method': 'GET', 'url': 'https://example.com/a.js', 'resource_type': 'Script'})
        assert first['type'] == 'response'
        second = await broker.reply({'method': 'GET', 'url': 'https://example.com/b.js?token=secret#fragment',
                                     'resource_type': 'Script'})
        assert second['type'] == 'abort'
        facts = broker.diagnostics()
        assert facts['limits']['network_bytes'] == 1_000_000
        assert facts['remaining']['network_bytes'] == 0
        assert facts['failed_resources'][0]['url'] == 'https://example.com/b.js'
        assert facts['failed_resources'][0]['reason'] == 'web_browser_budget_exceeded'
        assert 'secret' not in json.dumps(facts)
        broker.record_blocked({'reason': 'method_not_get', 'url': 'https://example.com/query?key=private',
                               'method': 'POST', 'resource_type': 'Fetch'})
        assert broker.diagnostics()['failed_resources'][-1]['method'] == 'POST'
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_same_condition_failure_is_reused_but_refresh_and_render_are_explicit(tmp_path, monkeypatch):
    from agent_runtime.tools import web_browser
    from agent_runtime.tools.web_http import FetchedResponse

    hits, rendered = [], []
    tools, client, _ = web_tools(tmp_path, lambda req: hits.append(req) or httpx.Response(403))
    async def render(client, url):
        rendered.append(url)
        return FetchedResponse(url, 'text/plain', b'alternate acquired facts', 10, 'direct')
    monkeypatch.setattr(web_browser, 'render_public_page', render)
    try:
        tool = tools['web_fetch']
        first = await tool.run({'url': 'https://example.com/doc'})
        second = await tool.run({'url': 'https://example.com/doc'})
        assert second['error_code'] == first['error_code'] == 'http_error'
        assert second['cache_hit'] and len(hits) == 1
        assert second['request_attempted'] is False
        await tool.run({'url': 'https://example.com/doc', 'refresh': True})
        assert len(hits) == 2
        third = await tool.run({'url': 'https://example.com/doc', 'render': True})
        assert third['error_code'] is None and len(rendered) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_public_agent_reports_persisted_remaining_budget_and_config(tmp_path, monkeypatch):
    from agent_runtime.tools import web_http
    from agent_runtime.tools.web_limits import WebLimits

    hits = []
    monkeypatch.setattr(web_http, 'PublicWebClient', lambda **kwargs: PublicWebClient(
        **kwargs, transport=httpx.MockTransport(lambda req: hits.append(req) or httpx.Response(
            200, text='source facts', headers={'content-type': 'text/plain'}))))
    class ReadPages(PatchThenAnswerModel):
        async def dispatch(self, prepared, *, delta_sink: object = None):
            step = int(prepared.request_ref['request_id'].rsplit(':', 1)[-1])
            if step <= 3:
                return HarnessModelResponse(text='', provider_response_id=f'read-{step}', usage={},
                    tool_calls=(HarnessToolCall(id=f'read-{step}', name='web_fetch', arguments={
                        'url': f'https://example.com/{step}', 'render': False}),))
            return HarnessModelResponse(text='Sources read; third source unverified.', provider_response_id='answer',
                                        usage={})
    database = tmp_path / 'state.sqlite'
    agent = Agent(workspace_path=tmp_path / 'workspace', checkpoint_db=database,
                  enable_workspace_mcp=False, web_limits=WebLimits(turn_requests=2))
    monkeypatch.setattr(agent, '_model_factory', lambda: ReadPages())
    result = await agent.run('Read three sources', allow_web_tools=True, require_workspace_change=False)
    assert result.status == 'done' and len(hits) == 2
    with RolloutStore(database) as store:
        assert store.read_turn(result.turn_id).binding_manifest['web_limits']['turn_requests'] == 2
        outputs = [i.payload['structured_content'] for i in store.list_items(result.turn_id) if i.kind == 'tool_result']
        assert outputs[0]['request_budget']['remaining'] == 1
        assert outputs[1]['request_budget']['remaining'] == 0
        assert outputs[2]['error_code'] == 'web_request_budget_exceeded'
        assert outputs[2]['request_attempted'] is False


def test_cli_loads_limits_without_creating_agent_or_network_for_invalid_config(tmp_path, monkeypatch):
    from agent_runtime import cli
    from agent_runtime.tools.web_limits import WebLimits

    path = tmp_path / 'limits.json'
    path.write_text('{"turn_requests": 64, "browser_network_bytes": 16000000}')
    assert cli._load_web_limits(path) == WebLimits(turn_requests=64, browser_network_bytes=16_000_000)
    path.write_text('{"turn_requests": 99999}')
    import typer
    with pytest.raises(typer.BadParameter):
        cli._load_web_limits(path)


@pytest.mark.anyio
async def test_fetch_compare_uses_prior_complete_snapshot_and_missing_baseline_is_explicit(tmp_path):
    values = iter(['old source fact', 'new source fact'])
    tools, client, _ = web_tools(tmp_path, lambda req: httpx.Response(
        200, text=next(values), headers={'content-type': 'text/plain'}))
    try:
        tool = tools['web_fetch']
        old = await tool.run({'url': 'https://example.com/api'})
        assert old['comparison']['status'] == 'baseline_missing'
        new = await tool.run({'url': 'https://example.com/api', 'refresh': True,
                              'compare_to_source_id': old['source_id']})
        assert new['comparison']['status'] == 'changed'
        assert new['comparison']['baseline']['source_id'] == old['source_id']
        replay = await tool.run({'source_id': new['source_id'], 'compare_to_source_id': old['source_id']})
        assert replay['comparison'] == new['comparison']
        missing = await tool.run({'source_id': new['source_id'],
                                  'compare_to_source_id': 'artifact_ffffffffffffffffffffffffffffffff'})
        assert missing['error_code'] == 'web_source_unavailable'
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_resume_cannot_raise_frozen_web_budget(tmp_path, monkeypatch):
    from agent_runtime.tools import web_http
    from agent_runtime.tools.web_limits import WebLimits
    from tests.agent.test_web_product import FetchThenAnswer

    hits = []
    monkeypatch.setattr(web_http, 'PublicWebClient', lambda **kwargs: PublicWebClient(
        **kwargs, transport=httpx.MockTransport(lambda req: hits.append(req) or httpx.Response(
            200, text='source facts', headers={'content-type': 'text/plain'}))))
    database = tmp_path / 'state.sqlite'
    options = dict(workspace_path=tmp_path / 'workspace', checkpoint_db=database, enable_workspace_mcp=False)
    first = Agent(**options, web_limits=WebLimits(turn_requests=1))
    monkeypatch.setattr(first, '_model_factory', lambda: FetchThenAnswer({'url': 'https://example.com/a'}))
    paused = await first.run('Read source', require_workspace_change=False)
    assert paused.status == 'paused' and not hits
    second = Agent(**options, web_limits=WebLimits(turn_requests=256))
    monkeypatch.setattr(second, '_model_factory', lambda: FetchThenAnswer({'url': 'https://example.com/a'}))
    resumed = await second.resume(paused.turn_id, 'allow_once')
    assert resumed.status == 'done' and len(hits) == 1
    with RolloutStore(database) as store:
        assert store.read_turn(resumed.turn_id).binding_manifest['web_limits']['turn_requests'] == 1
        outputs = [i.payload['structured_content'] for i in store.list_items(resumed.turn_id)
                   if i.kind == 'tool_result']
        assert outputs[0]['request_budget']['limit'] == 1
        assert outputs[0]['request_budget']['remaining'] == 0


@pytest.mark.anyio
async def test_view_error_does_not_poison_browser_url_acquisition(tmp_path, monkeypatch):
    from agent_runtime.tools import web_browser
    from agent_runtime.tools.web_http import FetchedResponse
    async def render(client, url):
        return FetchedResponse(url, 'text/plain', b'saved source facts', 10, 'direct')
    monkeypatch.setattr(web_browser, 'render_public_page', render)
    tools, client, _ = web_tools(tmp_path, lambda req: pytest.fail('Browser fake owns fetch'))
    try:
        tool = tools['web_fetch']
        failed = await tool.run({'url': 'https://example.com/doc', 'render': True,
                                 'compare_to_source_id': 'artifact_ffffffffffffffffffffffffffffffff'})
        assert failed['error_code'] == 'web_source_unavailable'
        acquired = await tool.run({'url': 'https://example.com/doc', 'render': True})
        assert acquired['error_code'] is None and 'saved source facts' in acquired['content']
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_failure_evidence_is_reused_after_new_process_resume(tmp_path, monkeypatch):
    from agent_runtime.tools import web_http

    hits = []
    monkeypatch.setattr(web_http, 'PublicWebClient', lambda **kwargs: PublicWebClient(
        **kwargs, transport=httpx.MockTransport(lambda req: hits.append(req) or httpx.Response(403))))
    class FailedReadThenPause(PatchThenAnswerModel):
        async def dispatch(self, prepared, *, delta_sink: object = None):
            step = int(prepared.request_ref['request_id'].rsplit(':', 1)[-1])
            if step in {1, 3}:
                return HarnessModelResponse(text='', provider_response_id=f'read-{step}', usage={},
                    tool_calls=(HarnessToolCall(id=f'read-{step}', name='web_fetch', arguments={
                        'url': 'https://example.com/blocked', 'render': False}),))
            if step == 2:
                return HarnessModelResponse(text='', provider_response_id='command', usage={},
                    tool_calls=(HarnessToolCall(id='command', name='run_command', arguments={'command': 'date'}),))
            return HarnessModelResponse(text='Source access denied; cannot verify facts.',
                                        provider_response_id='answer', usage={})
    database = tmp_path / 'state.sqlite'
    options = dict(workspace_path=tmp_path / 'workspace', checkpoint_db=database, enable_workspace_mcp=False)
    first = Agent(**options)
    monkeypatch.setattr(first, '_model_factory', lambda: FailedReadThenPause())
    paused = await first.run('Read source', allow_web_tools=True, require_workspace_change=False)
    assert paused.status == 'paused' and len(hits) == 1
    second = Agent(**options)
    monkeypatch.setattr(second, '_model_factory', lambda: FailedReadThenPause())
    resumed = await second.resume(paused.turn_id, 'deny')
    assert resumed.status == 'done' and len(hits) == 1
    with RolloutStore(database) as store:
        outputs = [i.payload['structured_content'] for i in store.list_items(resumed.turn_id)
                   if i.kind == 'tool_result' and i.payload['tool_name'] == 'web_fetch']
        assert outputs[-1]['cache_hit'] is True
        assert outputs[-1]['failure_origin']['tool_call_id'] == 'read-1'
        assert outputs[-1]['request_attempted'] is False


@pytest.mark.anyio
async def test_parallel_batch_consumes_only_available_budget_and_denial_is_not_url_failure(tmp_path, monkeypatch):
    from agent_runtime.tools import web_http
    from agent_runtime.tools.web_limits import WebLimits

    hits = []
    monkeypatch.setattr(web_http, 'PublicWebClient', lambda **kwargs: PublicWebClient(
        **kwargs, transport=httpx.MockTransport(lambda req: hits.append(req) or httpx.Response(
            200, text='acquired source', headers={'content-type': 'text/plain'}))))
    class ParallelPages(PatchThenAnswerModel):
        async def dispatch(self, prepared, *, delta_sink: object = None):
            if prepared.request_ref['request_id'].endswith(':step:1'):
                return HarnessModelResponse(text='', provider_response_id='parallel', usage={}, tool_calls=tuple(
                    HarnessToolCall(id=f'read-{index}', name='web_fetch', arguments={
                        'url': f'https://example.com/{index}', 'render': False}) for index in range(3)))
            return HarnessModelResponse(text='Budget reached; two sources acquired.', provider_response_id='answer',
                                        usage={})
    database = tmp_path / 'state.sqlite'
    agent = Agent(workspace_path=tmp_path / 'workspace', checkpoint_db=database, enable_workspace_mcp=False,
                  web_limits=WebLimits(turn_requests=2))
    monkeypatch.setattr(agent, '_model_factory', lambda: ParallelPages())
    result = await agent.run('Read three pages', allow_web_tools=True, require_workspace_change=False)
    assert result.status == 'done' and len(hits) == 2
    with RolloutStore(database) as store:
        outputs = [i.payload['structured_content'] for i in store.list_items(result.turn_id) if i.kind == 'tool_result']
        assert sum(o['request_attempted'] for o in outputs) == 2
        assert sum(o['error_code'] == 'web_request_budget_exceeded' for o in outputs) == 1


@pytest.mark.anyio
async def test_long_browser_failure_details_are_archived_and_model_excerpt_fits_minimum_budget(tmp_path, monkeypatch):
    from agent_runtime.core.messages import canonical_json_text, tool_result_payload
    from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract
    from agent_runtime.tools import web_browser
    from agent_runtime.tools.executor import ToolExecutor
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.web_browser import _Broker
    from agent_runtime.tools.web_http import PublicWebError
    from tests.agent.test_web_tools import call

    tools, client, sources = web_tools(tmp_path, lambda _: pytest.fail('No HTTP request is expected'))
    broker = _Broker(client)
    for index in range(16):
        broker.record_blocked({
            'reason': 'method_not_get', 'method': 'POST', 'resource_type': 'Fetch',
            'url': f'https://example.com/resource-{index}/' + 'a1b2c3d4e5f6g7h8' * 40,
        })
    complete_diagnostics = broker.diagnostics()
    assert len(complete_diagnostics['failed_resources']) == 16
    assert all(len(resource['url']) == 512 for resource in complete_diagnostics['failed_resources'])
    rendered = []

    async def fail_render(client, url):
        rendered.append(url)
        error = PublicWebError('web_browser_budget_exceeded', 'DOM byte budget exceeded.')
        error.render_diagnostics = json.loads(json.dumps(complete_diagnostics))
        raise error

    monkeypatch.setattr(web_browser, 'render_public_page', fail_render)
    executor = ToolExecutor(tools)
    accounting = TokenAccountingService(TokenizerContract('', 'deepseek-v4-official', 'deepseek-v4-official'))
    context = ToolExecutionContext(allow_web_tools=True)
    arguments = {'url': 'https://example.com/doc', 'render': True, 'max_tokens': 1000}
    try:
        first = (await executor.execute(call('web_fetch', arguments, 'first'), context=context)).result
        cached = (await executor.execute(call('web_fetch', arguments, 'cached'), context=context)).result
        for result in (first, cached):
            assert result.error_code == 'web_browser_budget_exceeded'
            assert accounting.count_for_budget(canonical_json_text(tool_result_payload(result))) <= 1000
            excerpt = result.structured_content['render_diagnostics']
            assert len(excerpt['failed_resources']) < 16
            assert excerpt['failed_resources_omitted'] + len(excerpt['failed_resources']) == 16
        first_output = first.structured_content
        archived_id = first_output['diagnostics_source_id']
        assert archived_id in sources
        archived = json.loads(sources[archived_id])
        assert archived['render_diagnostics'] == complete_diagnostics
        assert cached.structured_content['diagnostics_source_id'] == archived_id
        assert cached.structured_content['request_attempted'] is False
        assert len(rendered) == 1
        replay = await tools['web_fetch'].run({'source_id': archived_id, 'max_tokens': 16000})
        assert replay['render_diagnostics'] == complete_diagnostics
        assert replay['request_attempted'] is False and len(rendered) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_invalid_fetch_history_does_not_block_a_later_valid_public_agent_fetch(tmp_path, monkeypatch):
    from agent_runtime.tools import web_http

    hits = []
    monkeypatch.setattr(web_http, 'PublicWebClient', lambda **kwargs: PublicWebClient(
        **kwargs, transport=httpx.MockTransport(lambda req: hits.append(req) or httpx.Response(
            200, text='Valid source facts', headers={'content-type': 'text/plain'}))))

    class InvalidThenValid(PatchThenAnswerModel):
        async def dispatch(self, prepared, *, delta_sink: object = None):
            step = int(prepared.request_ref['request_id'].rsplit(':', 1)[-1])
            if step == 1:
                return HarnessModelResponse(text='', provider_response_id='invalid', usage={}, tool_calls=(
                    HarnessToolCall(id='invalid', name='web_fetch', arguments={
                        'url': 'https://example.com/invalid', 'render': 'POST'}),))
            if step == 2:
                return HarnessModelResponse(text='', provider_response_id='valid', usage={}, tool_calls=(
                    HarnessToolCall(id='valid', name='web_fetch', arguments={
                        'url': 'https://example.com/valid', 'render': False}),))
            return HarnessModelResponse(text='Valid source read after rejected input.',
                                        provider_response_id='answer', usage={})

    database = tmp_path / 'state.sqlite'
    agent = Agent(workspace_path=tmp_path / 'workspace', checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, '_model_factory', lambda: InvalidThenValid())
    result = await agent.run('Read a source', allow_web_tools=True, require_workspace_change=False)
    assert result.status == 'done'
    assert [str(request.url) for request in hits] == ['https://example.com/valid']
    with RolloutStore(database) as store:
        outputs = [item.payload for item in store.list_items(result.turn_id) if item.kind == 'tool_result']
        assert outputs[0]['is_error'] is True
        assert outputs[1]['is_error'] is False
        assert 'Valid source facts' in outputs[1]['structured_content']['content']
        assert outputs[1]['structured_content']['request_attempted'] is True
        assert outputs[1]['structured_content']['request_budget']['used'] == 1

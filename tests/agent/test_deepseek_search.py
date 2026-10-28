from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent_runtime.tools.builtins.web import create_web_tools
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.web_http import PublicWebClient, PublicWebError
from tests.agent.test_web_http import _Stream
from tests.agent.test_web_tools import call


def deepseek_tools(handler, *, before_request=None, **client_options):
    client = PublicWebClient(transport=httpx.MockTransport(handler), **client_options)
    tools = create_web_tools(
        client, save_source=lambda _: "", load_source=lambda _: b"",
        search_provider="deepseek", search_key="fake-deepseek-secret", before_request=before_request,
    )
    return {tool.definition.name: tool for tool in tools}, client


@pytest.mark.anyio
async def test_deepseek_official_search_preserves_query_sources_and_uses_existing_permissions():
    hits, reservations = [], []
    tools, client = deepseek_tools(lambda req: hits.append(req) or httpx.Response(200, json={"results": [
        {"title": "Docs", "url": "https://example.com/docs", "snippet": "Short", "summary": "Long summary",
         "content": "Not requested", "date": "2026-10-25T08:00:00Z", "authority": -1},
        {"title": "Private", "url": "http://127.0.0.1/secret", "snippet": "Private", "summary": "Private"},
    ]}), before_request=lambda: reservations.append(True))
    try:
        query = ' Python OR Rust site:example.com after:2026-10-01 '
        tool = tools["web_search"]
        executor = ToolExecutor(tools)
        denied = await executor.execute(call("web_search", {"query": query}), context=ToolExecutionContext())
        assert denied.result.error_code == "approval_required"
        assert hits == reservations == []
        result = await executor.execute(call("web_search", {"query": query}, "allowed"),
                                        context=ToolExecutionContext(allow_web_tools=True))
        assert not result.result.is_error
        output = result.result.structured_content
        assert output["provider"] == "deepseek"
        assert list(output["results"]) == [{
            "title": "Docs", "url": "https://example.com/docs", "snippet": "Short",
            "summary": "Long summary", "published_at": "2026-10-25T08:00:00Z",
        }]
        assert output["request_attempted"] and output["network_bytes"] > 0
        req, = hits
        assert req.method == "POST"
        assert str(req.url) == "https://api.deepseek.com/v1/tools/search"
        assert req.headers["Authorization"] == "Bearer fake-deepseek-secret"
        assert json.loads(req.content) == {"query": query, "count": 5, "need_content": False}
        assert tool.resolve_use({"query": query}).targets[0].value == str(req.url)
        assert tool.execution_revision == "builtin-public-web-search-deepseek-v2"
        again = await tool.run({"query": query})
        assert again["cache_hit"] and not again["request_attempted"] and again["network_bytes"] == 0
        assert len(hits) == len(reservations) == 1
        assert "fake-deepseek-secret" not in repr(output)
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("arguments", [{"query": " "}, {"query": "查" * 129},
                                        {"query": "docs", "freshness": "pw"}])
async def test_deepseek_rejects_unsupported_input_before_permission_budget_and_network(arguments):
    tools, client = deepseek_tools(lambda _: pytest.fail("Unexpected request"),
                                   before_request=lambda: pytest.fail("Unexpected budget reservation"))
    try:
        result = await ToolExecutor(tools).execute(call("web_search", arguments), context=ToolExecutionContext())
        assert result.result.is_error and result.result.error_code == "invalid_arguments"
        assert tools["web_search"].definition.input_schema["properties"]["query"]["maxLength"] == 128
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_deepseek_empty_results_are_success_without_generated_answer():
    tools, client = deepseek_tools(lambda _: httpx.Response(200, json={"results": [], "answer": "Generated"}))
    try:
        result = await tools["web_search"].run({"query": "no matches"})
        assert result["error_code"] is None and result["result_status"] == "empty"
        assert result["results"] == [] and "Generated" not in repr(result)
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("status", [400, 401, 402, 403, 429, 500, 504])
async def test_deepseek_http_error_is_not_empty_success_or_fallback(status):
    hits = []
    tools, client = deepseek_tools(lambda req: hits.append(req) or httpx.Response(
        status, json={"error": {"message": "fake-deepseek-secret"}}))
    try:
        result = await tools["web_search"].run({"query": "docs"})
        assert result["error_code"] == "http_error"
        assert str(status) in result["error_message"]
        assert result["result_status"] == "error" and result["request_attempted"]
        assert "fake-deepseek-secret" not in repr(result)
        assert len(hits) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [{}, {"results": {}}, {"results": [{"url": "https://example.com"}]},
    {"results": [42]}, {"results": [{"title": "Docs", "snippet": "Text"}]},
    {"results": [{"url": "https://example.com", "title": "Doc", "snippet": 42}]},
    {"results": [], "extra": "fake-deepseek-secret"}])
async def test_deepseek_invalid_payload_is_not_evidence(payload):
    tools, client = deepseek_tools(lambda _: httpx.Response(200, json=payload))
    try:
        result = await tools["web_search"].run({"query": "docs"})
        assert result["error_code"] == "web_search_invalid_response"
        assert "fake-deepseek-secret" not in repr(result)
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_deepseek_network_failure_can_retry_without_switching_provider():
    hits = []

    def handler(req):
        hits.append(req)
        if len(hits) == 1:
            raise httpx.ConnectError("fake-deepseek-secret")
        return httpx.Response(200, json={"results": []})

    tools, client = deepseek_tools(handler)
    try:
        first = await tools["web_search"].run({"query": "docs"})
        assert first["error_code"] == "network_error" and "fake-deepseek-secret" not in repr(first)
        second = await tools["web_search"].run({"query": "docs"})
        assert second["error_code"] is None and len(hits) == 2
        assert all(req.url.host == "api.deepseek.com" for req in hits)
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("summary,expected_truncated", [("Evidence " * 500, False), ("证据" * 20_000, True)],
                         ids=["full-summary", "budgeted-summary"])
async def test_deepseek_long_summary_uses_output_budget_instead_of_rejecting_valid_results(summary, expected_truncated):
    tools, client = deepseek_tools(lambda _: httpx.Response(200, json={"results": [{
        "title": "Doc", "url": "https://example.com/doc", "snippet": "Short", "summary": summary,
    }]}))
    try:
        output = await tools["web_search"].run({"query": "public docs"})
        assert output["error_code"] is None and output["result_status"] == "results"
        assert len(output["results"]) == 1 and output["results"][0]["url"] == "https://example.com/doc"
        assert output["results_truncated"] is expected_truncated
        if expected_truncated:
            assert output["results"][0]["summary"] and summary.startswith(output["results"][0]["summary"])
        else:
            assert output["results"][0]["summary"] == summary
        assert len(json.dumps(output["results"][0], ensure_ascii=False).encode()) <= 32_000
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_deepseek_budget_exhaustion_never_attempts_request():
    def exhausted():
        raise PublicWebError("web_request_budget_exceeded", "Budget exhausted")

    tools, client = deepseek_tools(lambda _: pytest.fail("Unexpected request"), before_request=exhausted)
    try:
        result = await tools["web_search"].run({"query": "docs"})
        assert result["error_code"] == "web_request_budget_exceeded"
        assert not result["request_attempted"]
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_deepseek_timeout_and_cancellation_close_stream():
    stream = _Stream([b"{}"], delay=60)
    tools, client = deepseek_tools(lambda _: httpx.Response(200, stream=stream), timeout_seconds=0.01)
    try:
        result = await tools["web_search"].run({"query": "docs"})
        assert result["error_code"] == "timeout" and stream.closed
    finally:
        await client.aclose()
    stream = _Stream([b"{}"], delay=60)
    tools, client = deepseek_tools(lambda _: httpx.Response(200, stream=stream))
    try:
        task = asyncio.create_task(tools["web_search"].run({"query": "docs"}))
        await stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
    finally:
        await client.aclose()


@pytest.mark.parametrize("tool_name,arguments,response", [
    ("web_search", {"query": "public docs"}, {"results": [
        {"url": "https://example.com/docs", "title": "Docs", "snippet": "Evidence"}]}),
    ("web_fetch", {"url": "https://example.com/docs"}, {"title": "Docs", "content": "# Docs\n\nEvidence"}),
])
def test_deepseek_cli_runs_real_agent_tool_and_persists_sources(
    tmp_path, monkeypatch, tool_name, arguments, response,
):
    from typer.testing import CliRunner

    import agent_runtime.tools.web_http as http_module
    from agent_runtime import Agent
    from agent_runtime.cli import agent_app
    from agent_runtime.harness import RuntimeState
    from tests.agent.test_web_product import FetchThenAnswer

    hits = []
    real_client = PublicWebClient
    monkeypatch.setattr(http_module, "PublicWebClient", lambda **options: real_client(
        **options, transport=httpx.MockTransport(lambda req: hits.append(req) or httpx.Response(200, json=response))))
    initialize = Agent.__init__

    def scripted_agent(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self._model_factory = lambda: FetchThenAnswer(arguments, tool_name)

    monkeypatch.setattr(Agent, "__init__", scripted_agent)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    key = tmp_path / "search.key"
    key.write_text("fake-deepseek-secret")
    key.chmod(0o400)
    result = CliRunner().invoke(agent_app, [
        "run", "Search public docs", "--non-interactive", "--no-require-workspace-change",
        "--web-search-provider", "deepseek", "--web-search-key-file", str(key),
    ])
    assert result.exit_code == 0, result.output
    assert len(hits) == 1 and hits[0].url.host == "api.deepseek.com"
    with RuntimeState.open_sync(workspace / ".praxis/checkpoints.sqlite") as store:
        turn = store.list_turns()[0]
        items = store.list_items(turn.turn_id)
        output = next(item.payload["structured_content"] for item in items if item.kind == "tool_result")
        if tool_name == "web_search":
            assert output["provider"] == "deepseek"
            assert output["results"][0]["url"] == "https://example.com/docs"
        else:
            assert output["extraction_method"] == "deepseek_fetch_markdown"
            snapshot = json.loads(store.read_artifact(output["source_id"]))
            assert snapshot["text"] == response["content"] and snapshot["url"] == arguments["url"]
        assert output["request_budget"]["used"] == 1
        assert "fake-deepseek-secret" not in repr(items) + result.output
        assert store.verify().valid

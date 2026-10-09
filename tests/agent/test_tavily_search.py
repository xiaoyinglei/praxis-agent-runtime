from __future__ import annotations

import asyncio
import gzip
import json

import httpx
import pytest

from agent_runtime.tools.builtins.web import create_web_tools
from agent_runtime.tools.web_http import PublicWebClient, PublicWebError
from tests.agent.test_web_http import _Stream


def tavily_tools(handler, *, before_request=None):
    client = PublicWebClient(transport=httpx.MockTransport(handler))
    tools = create_web_tools(
        client, save_source=lambda _: "artifact_" + "0" * 32, load_source=lambda _: b"",
        search_provider="tavily", search_key="fake-tavily-secret", before_request=before_request,
    )
    return {tool.definition.name: tool for tool in tools}, client


@pytest.mark.anyio
@pytest.mark.parametrize("freshness,time_range", [(None, None), ("pd", "day"), ("pw", "week"),
                                                ("pm", "month"), ("py", "year")])
async def test_tavily_preserves_query_and_uses_one_credit_request(freshness, time_range):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"answer": "Do not use generated prose as search evidence", "results": [
            {"title": "Public job", "url": "https://example.com/job", "content": "Duties and requirements"},
            {"title": "Private", "url": "http://127.0.0.1/secret", "content": "Internal"},
        ]})

    tools, client = tavily_tools(handler)
    try:
        query = ' 北京 Agent 招聘 "RAG" OR Python '
        result = await tools["web_search"].run({"query": query, "max_results": 3, "freshness": freshness})
        request = requests[0]
        assert request.method == "POST"
        assert str(request.url) == "https://api.tavily.com/search"
        assert request.headers["Authorization"] == "Bearer fake-tavily-secret"
        expected = {"query": query, "max_results": 3, "search_depth": "basic", "auto_parameters": False,
                    "include_answer": False, "include_raw_content": False, "include_images": False}
        if time_range is not None:
            expected["time_range"] = time_range
        assert json.loads(request.content) == expected
        assert result["provider"] == "tavily"
        assert result["results"] == [{"title": "Public job", "url": "https://example.com/job",
                                      "snippet": "Duties and requirements"}]
        assert result["freshness_requested"] == freshness
        assert result["freshness_verified"] is False
        assert result["connection_mode"] == "direct"
        assert result["network_bytes"] > 0
        assert "fake-tavily-secret" not in json.dumps(result)
        assert "generated prose" not in json.dumps(result)
        assert tools["web_search"].resolve_use({"query": query}).targets[0].value == str(request.url)
        assert "fake-tavily-secret" not in repr(tools["web_search"].definition)
        assert tools["web_search"].execution_revision != "builtin-public-web-v5"
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_tavily_cache_does_not_spend_a_second_request():
    requests, checks = [], []
    tools, client = tavily_tools(lambda request: requests.append(request) or httpx.Response(200, json={
        "results": [{"title": "A", "url": "https://example.com", "content": "Details"}],
    }), before_request=lambda: checks.append(True))
    try:
        first = await tools["web_search"].run({"query": "public query"})
        second = await tools["web_search"].run({"query": "public query"})
        assert first["cache_hit"] is False
        assert second["cache_hit"] is True
        assert len(requests) == len(checks) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 429, 432, 433, 500])
async def test_tavily_http_failures_do_not_fall_back_or_echo_remote_errors(status):
    requests = []
    tools, client = tavily_tools(lambda request: requests.append(request) or httpx.Response(
        status, json={"detail": "fake-tavily-secret"}))
    try:
        result = await tools["web_search"].run({"query": "public query"})
        assert result["provider"] == "tavily"
        assert result["error_code"] == "http_error"
        assert str(status) in result["error_message"]
        assert tools["web_search"].normalize_output(result).is_error
        assert result["results"] == []
        assert "fake-tavily-secret" not in json.dumps(result)
        assert len(requests) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [{}, {"results": {}}, {"results": [
    {"url": "https://example.com", "content": 42}]}, {"results": [], "answer": "fake-tavily-secret"}])
async def test_tavily_malformed_or_credential_echoing_payload_is_not_evidence(payload):
    tools, client = tavily_tools(lambda _: httpx.Response(200, json=payload))
    try:
        result = await tools["web_search"].run({"query": "public query"})
        assert result["error_code"] == "web_search_invalid_response"
        assert tools["web_search"].normalize_output(result).is_error
        assert "fake-tavily-secret" not in json.dumps(result)
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_tavily_actual_empty_result_is_empty_success():
    tools, client = tavily_tools(lambda _: httpx.Response(200, json={"results": []}))
    try:
        result = await tools["web_search"].run({"query": "public query"})
        assert result["result_status"] == "empty"
        assert result["error_code"] is None
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_authenticated_json_post_rejects_redirect_before_forwarding():
    requests = []
    stream = _Stream([b"fake-tavily-secret"])
    client = PublicWebClient(transport=httpx.MockTransport(lambda request: requests.append(request)
        or httpx.Response(307, headers={"Location": "https://other.example/search"}, stream=stream)))
    try:
        with pytest.raises(PublicWebError) as caught:
            await client.post_json("https://api.tavily.com/search", {"query": "public"},
                                   headers={"Authorization": "Bearer fake-tavily-secret"})
        assert caught.value.code == "redirect_not_allowed"
        assert len(requests) == 1
        assert stream.closed
        assert "fake-tavily-secret" not in str(caught.value)
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_authenticated_json_post_retains_decoded_byte_limit():
    stream = _Stream([gzip.compress(b"x" * 10_000)])
    client = PublicWebClient(max_bytes=128, transport=httpx.MockTransport(lambda _: httpx.Response(
        200, headers={"Content-Encoding": "gzip"}, stream=stream)))
    try:
        with pytest.raises(PublicWebError) as caught:
            await client.post_json("https://api.tavily.com/search", {"query": "public"})
        assert caught.value.code == "response_too_large"
        assert stream.closed
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_authenticated_json_post_deadline_closes_response():
    stream = _Stream([b"{}"], delay=1)
    client = PublicWebClient(timeout_seconds=0.01, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, stream=stream)))
    try:
        with pytest.raises(PublicWebError) as caught:
            await client.post_json("https://api.tavily.com/search", {"query": "public"})
        assert caught.value.code == "timeout"
        assert stream.closed
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_authenticated_json_post_cancellation_closes_response():
    stream = _Stream([b"{}"], delay=60)
    client = PublicWebClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream)))
    try:
        task = asyncio.create_task(client.post_json("https://api.tavily.com/search", {"query": "public"}))
        await stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
    finally:
        await client.aclose()


@pytest.mark.parametrize("provider,key", [("unknown", "fake-key"), ("tavily", None), ("brave", None),
                                        ("bing", "fake-key")])
def test_search_configuration_fails_before_requests(provider, key):
    client = PublicWebClient(transport=httpx.MockTransport(lambda _: pytest.fail("Unexpected network request")))
    with pytest.raises(ValueError, match="[Ss]earch"):
        create_web_tools(client, save_source=lambda _: "", load_source=lambda _: b"",
                         search_provider=provider, search_key=key)


@pytest.mark.parametrize("provider,key", [("unknown", "search.key"), ("tavily", None),
                                        ("brave", None), ("bing", "search.key"), ("", None)])
def test_sdk_rejects_invalid_search_configuration_before_model_io(provider, key):
    from agent_runtime import Agent

    with pytest.raises(ValueError, match="Web search"):
        Agent(web_search_provider=provider, web_search_key_file=key)


@pytest.mark.anyio
async def test_tavily_sdk_resume_preserves_provider_and_never_persists_key(tmp_path, monkeypatch):
    import agent_runtime.tools.web_http as http_module
    from agent_runtime import Agent
    from agent_runtime.harness import RolloutStore
    from tests.agent.test_web_product import FetchThenAnswer

    requests = []
    real_client = PublicWebClient
    monkeypatch.setattr(http_module, "PublicWebClient", lambda **options: real_client(
        **options, transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(
            200, json={"results": [{"url": "https://example.com/job", "title": "A", "content": "Duties"}]}))))
    key = tmp_path / "search.key"
    key.write_text("fake-tavily-secret")
    key.chmod(0o400)
    options = dict(workspace_path=tmp_path / "workspace", checkpoint_db=tmp_path / "state.sqlite",
                   enable_workspace_mcp=False, web_search_provider="tavily", web_search_key_file=key)
    first = Agent(**options)
    monkeypatch.setattr(first, "_harness_model", lambda: FetchThenAnswer({"query": "public jobs"}, "web_search"))
    paused = await first.run("Search public jobs", require_workspace_change=False)
    assert paused.status == "paused"
    assert requests == []
    second = Agent(**options)
    monkeypatch.setattr(second, "_harness_model", lambda: FetchThenAnswer({"query": "public jobs"}, "web_search"))
    result = await second.resume(paused.turn_id, "allow_once")
    assert result.status == "done"
    assert len(requests) == 1
    assert requests[0].url.host == "api.tavily.com"
    with RolloutStore(options["checkpoint_db"]) as store:
        items = store.list_items(result.turn_id)
        output = next(item.payload["structured_content"] for item in items if item.kind == "tool_result")
        assert output["provider"] == "tavily"
        assert output["results"][0]["snippet"] == "Duties"
        assert "fake-tavily-secret" not in repr(items)


@pytest.mark.parametrize("command", ["run", "chat", "resume"])
def test_cli_forwards_search_provider_and_key_from_environment(command, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from agent_runtime import cli
    from tests.agent.test_agent_cli_resume import _persist_cli_turn, _result

    captured = []

    class Facade:
        async def run(self, *_args, **_kwargs):
            return _result(turn_id="turn_tavily", answer="complete")

        async def resume(self, turn_id, _action, **_kwargs):
            return _result(turn_id=turn_id, answer="resumed")

    def create_facade(**options):
        captured.append(options)
        return Facade()

    async def chat_loop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(cli, "_create_agent_facade", create_facade)
    monkeypatch.setattr(cli, "_chat_facade_loop", chat_loop)
    database = tmp_path / "state.sqlite"
    arguments = [command, "public jobs"] if command == "run" else [command]
    if command == "resume":
        turn_id = _persist_cli_turn(database, tmp_path / "workspace")
        arguments += [turn_id, "--checkpoint-db", str(database), "--action", "allow_once"]
    key = tmp_path / "search.key"
    result = CliRunner().invoke(cli.agent_app, arguments, env={
        "PRAXIS_WEB_SEARCH_PROVIDER": "tavily", "PRAXIS_WEB_SEARCH_KEY_FILE": str(key),
    })
    assert result.exit_code == 0, result.output
    assert captured[0]["web_search_provider"] == "tavily"
    assert captured[0]["web_search_key_file"] == key

from __future__ import annotations

import asyncio
import base64
import hashlib
import json

import httpx
import pytest

from agent_runtime.tools.builtins.web import create_web_tools
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.web_http import PublicWebClient, PublicWebError
from tests.agent.test_web_http import _Stream
from tests.agent.test_web_tools import call


def hosted_tools(handler, *, before_request=None, sources=None, **client_options):
    sources = {} if sources is None else sources

    def save(content):
        identity = f"artifact_{len(sources):032x}"
        sources[identity] = content
        return identity

    client = PublicWebClient(transport=httpx.MockTransport(handler), **client_options)
    tools = create_web_tools(
        client, save_source=save, load_source=sources.__getitem__,
        search_provider="deepseek", search_key="fake-deepseek-secret", before_request=before_request,
    )
    return {tool.definition.name: tool for tool in tools}, client, sources


@pytest.mark.anyio
async def test_open_uses_official_fetch_and_find_continues_saved_markdown_without_network(monkeypatch):
    import agent_runtime.tools.builtins.web as web
    from agent_runtime.tools import web_browser

    async def unexpected(*args, **kwargs):
        pytest.fail("Hosted fetch must not start a local HTML parser or browser")

    monkeypatch.setattr(web, "extract_content_async", unexpected)
    monkeypatch.setattr(web_browser, "render_public_page", unexpected)
    hits, reservations = [], []
    markdown = "# Guide\n\nIntro\n\n## API\n\nTarget endpoint\n\n[Reference](/reference)\n"
    payload = {"title": "Guide", "content": markdown}
    tools, client, sources = hosted_tools(lambda req: hits.append(req) or httpx.Response(200, json=payload),
                                         before_request=lambda: reservations.append(True))
    try:
        args = {"url": "https://example.com/guide", "max_lines": 2}
        tool = tools["web_fetch"]
        executor = ToolExecutor(tools)
        blocked = await executor.execute(call("web_fetch", args), context=ToolExecutionContext())
        assert blocked.result.error_code == "approval_required" and hits == reservations == []
        result = await executor.execute(call("web_fetch", args, "allowed"),
                                        context=ToolExecutionContext(allow_web_tools=True))
        assert not result.result.is_error, result.result.error_message
        output = result.result.structured_content
        req, = hits
        assert req.method == "POST" and str(req.url) == "https://api.deepseek.com/v1/tools/fetch"
        assert json.loads(req.content) == {"url": args["url"]}
        assert req.headers["Authorization"] == "Bearer fake-deepseek-secret"
        assert output["url"] == args["url"] and output["title"] == "Guide"
        assert output["extraction_method"] == "deepseek_fetch_markdown"
        assert output["render_mode"] == "provider" and output["next_line"] == 3
        assert output["content_hash"] == hashlib.sha256(markdown.encode()).hexdigest()
        assert "fake-deepseek-secret" not in repr(output)
        snapshot = json.loads(sources[output["source_id"]])
        assert snapshot["text"] == markdown
        # Raw is the actual API response, never a fabricated origin-page HTML body.
        assert snapshot["raw_content_type"] == "application/json"
        assert json.loads(base64.b64decode(snapshot["raw_body_base64"])) == payload
        saved = {"source_id": output["source_id"]}
        assert not tool.resolve_use(saved).effects
        found = await tool.run({**saved, "find": "Target endpoint"})
        assert found["start_line"] == 7 and "Target endpoint" in found["content"]
        assert not found["request_attempted"] and found["network_bytes"] == 0
        outline = await tool.run({**saved, "view": "outline"})
        assert [section["title"] for section in outline["sections"]] == ["Guide", "API"]
        links = await tool.run({**saved, "view": "links"})
        assert links["links"][0]["url"] == "https://example.com/reference"
        again = await tool.run(args)
        assert again["cache_hit"] and again["source_id"] == output["source_id"]
        assert len(hits) == len(reservations) == 1
        targets = {target.value for target in tool.resolve_use(args).targets}
        assert targets == {args["url"], "https://api.deepseek.com/v1/tools/fetch"}
        assert tool.execution_revision == "builtin-public-web-fetch-deepseek-v1"
        assert "Chromium" not in tool.definition.description
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 402, 403, 404, 429, 504])
async def test_hosted_fetch_http_failure_never_falls_back_to_origin_or_browser(status):
    hits = []
    tools, client, sources = hosted_tools(lambda req: hits.append(req) or httpx.Response(
        status, json={"error": {"message": "fake-deepseek-secret"}}))
    try:
        result = await tools["web_fetch"].run({"url": "https://example.com/doc"})
        assert result["error_code"] == "http_error" and str(status) in result["error_message"]
        assert result["source_id"] is None and sources == {}
        assert "fake-deepseek-secret" not in repr(result)
        assert len(hits) == 1 and hits[0].url.host == "api.deepseek.com"
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [{}, [], {"title": "Doc", "content": 42},
    {"title": 42, "content": "Text"}, {"title": "Doc", "content": ""},
    {"title": "Doc", "content": "Text", "extra": "fake-deepseek-secret"}])
async def test_hosted_fetch_malformed_or_empty_payload_is_not_a_successful_snapshot(payload):
    tools, client, sources = hosted_tools(lambda _: httpx.Response(200, json=payload))
    try:
        result = await tools["web_fetch"].run({"url": "https://example.com/doc"})
        assert result["error_code"] == "web_fetch_invalid_response"
        assert result["request_attempted"] and result["source_id"] is None and sources == {}
        assert "fake-deepseek-secret" not in repr(result)
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_hosted_fetch_budget_and_render_controls_are_checked_before_request():
    reservations = []

    def exhausted():
        reservations.append(True)
        raise PublicWebError("web_request_budget_exceeded", "Budget exhausted")

    tools, client, _ = hosted_tools(lambda _: pytest.fail("Unexpected request"), before_request=exhausted)
    try:
        result = await ToolExecutor(tools).execute(call("web_fetch", {
            "url": "https://example.com/doc", "render": True,
        }), context=ToolExecutionContext(allow_web_tools=True))
        assert result.result.error_code == "invalid_arguments" and reservations == []
        output = await tools["web_fetch"].run({"url": "https://example.com/doc"})
        assert output["error_code"] == "web_request_budget_exceeded" and not output["request_attempted"]
        assert len(reservations) == 1
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_hosted_fetch_timeout_and_cancellation_close_stream_without_snapshot():
    for cancel in (False, True):
        stream = _Stream([b"{}"], delay=60)
        tools, client, sources = hosted_tools(lambda _, body=stream: httpx.Response(200, stream=body),
                                              timeout_seconds=60 if cancel else 0.01)
        try:
            task = asyncio.create_task(tools["web_fetch"].run({"url": "https://example.com/doc"}))
            await stream.started.wait()
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert (await task)["error_code"] == "timeout"
            assert stream.closed and sources == {}
        finally:
            await client.aclose()


@pytest.mark.anyio
async def test_hosted_fetch_network_error_does_not_contact_origin():
    hits = []

    def offline(req):
        hits.append(req)
        raise httpx.ConnectError("fake-deepseek-secret")

    tools, client, _ = hosted_tools(offline)
    try:
        output = await tools["web_fetch"].run({"url": "https://example.com/doc"})
        assert output["error_code"] == "network_error" and "fake-deepseek-secret" not in repr(output)
        assert len(hits) == 1 and hits[0].url.host == "api.deepseek.com"
    finally:
        await client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("denied", ["example.com", "api.deepseek.com"])
async def test_hosted_fetch_enforces_target_and_service_permissions_before_budget(tmp_path, denied):
    from agent_runtime.permission_policy import NetworkPolicy, PermissionPolicy

    tools, client, _ = hosted_tools(lambda _: pytest.fail("Unexpected request"),
                                    before_request=lambda: pytest.fail("Unexpected reservation"))
    policy = PermissionPolicy(network=NetworkPolicy(decision="allow", denied_domains=(denied,))).resolve(tmp_path)
    try:
        result = await ToolExecutor(tools).execute(call("web_fetch", {"url": "https://example.com/doc"}),
            context=ToolExecutionContext(permissions=policy, allow_web_tools=True))
        assert result.result.is_error and result.trace.decision == "deny"
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_hosted_fetch_retains_source_limit_and_byte_budget_pagination():
    from agent_runtime.tools.web_content import MAX_SOURCE_CHARACTERS

    content = "字" * (MAX_SOURCE_CHARACTERS + 1)
    tools, client, sources = hosted_tools(lambda _: httpx.Response(200, json={"title": "Long", "content": content}))
    try:
        output = await tools["web_fetch"].run({"url": "https://example.com/doc", "max_bytes": 4096})
        assert output["source_truncated"] and output["truncated"] and output["next_line"] is not None
        snapshot = json.loads(sources[output["source_id"]])
        assert len(snapshot["text"]) == MAX_SOURCE_CHARACTERS
        next_page = await tools["web_fetch"].run({"source_id": output["source_id"],
            "start_line": output["next_line"], "max_bytes": 4096})
        assert next_page["content_hash"] == output["content_hash"] and not next_page["request_attempted"]
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_hosted_sdk_resume_and_new_session_find_keep_snapshot_identity(tmp_path, monkeypatch):
    import agent_runtime.tools.web_http as http_module
    from agent_runtime import Agent
    from agent_runtime.harness import RolloutStore
    from tests.agent.test_web_product import FetchThenAnswer

    hits = []
    real_client = PublicWebClient
    monkeypatch.setattr(http_module, "PublicWebClient", lambda **options: real_client(
        **options, transport=httpx.MockTransport(lambda req: hits.append(req) or httpx.Response(
            200, json={"title": "Doc", "content": "# Doc\n\nTarget evidence"}))))
    key = tmp_path / "search.key"
    key.write_text("fake-deepseek-secret")
    key.chmod(0o400)
    options = dict(workspace_path=tmp_path / "workspace", checkpoint_db=tmp_path / "state.sqlite",
                   enable_workspace_mcp=False, web_search_provider="deepseek", web_search_key_file=key)
    first = Agent(**options)
    monkeypatch.setattr(first, "_model_factory", lambda: FetchThenAnswer({"url": "https://example.com/doc"}))
    paused = await first.run("Read public docs", require_workspace_change=False)
    assert paused.status == "paused" and hits == []
    second = Agent(**options)
    monkeypatch.setattr(second, "_model_factory", lambda: FetchThenAnswer({"url": "https://example.com/doc"}))
    opened = await second.resume(paused.turn_id, "allow_once")
    assert opened.status == "done" and len(hits) == 1
    with RolloutStore(options["checkpoint_db"]) as store:
        items = store.list_items(opened.turn_id)
        output = next(item.payload["structured_content"] for item in items if item.kind == "tool_result")
        source_id = output["source_id"]
        digest = output["content_hash"]
        assert output["request_budget"]["used"] == 1
        operation = store.list_tool_operations(opened.turn_id)[0]
        assert operation.execution_revision == "builtin-public-web-fetch-deepseek-v1"
        assert "fake-deepseek-secret" not in repr(items)
    third = Agent(**options)
    monkeypatch.setattr(third, "_model_factory", lambda: FetchThenAnswer({"source_id": source_id, "find": "Target"}))
    found = await third.run("Find in the saved source", require_workspace_change=False)
    assert found.status == "done" and len(hits) == 1
    with RolloutStore(options["checkpoint_db"]) as store:
        output = next(item.payload["structured_content"] for item in store.list_items(found.turn_id)
                      if item.kind == "tool_result")
        assert output["source_id"] == source_id and output["content_hash"] == digest
        assert output["request_budget"]["used"] == 0 and output["start_line"] == 3
        assert "Target evidence" in output["content"] and store.verify().valid

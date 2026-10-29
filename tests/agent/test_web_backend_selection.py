from __future__ import annotations

import httpx
import pytest

from agent_runtime.tools.web_http import FetchedResponse
from tests.agent.test_deepseek_web import hosted_tools


@pytest.mark.anyio
async def test_managed_failure_can_be_followed_by_explicit_http_without_sending_the_key_to_origin():
    hits = []

    def handler(request):
        hits.append(request)
        if request.url.host == "api.deepseek.com":
            return httpx.Response(404, json={"error": {"message": "unsupported document"}})
        assert "authorization" not in request.headers
        return httpx.Response(200, text="Original source facts", headers={"content-type": "text/plain"})

    tools, client, _ = hosted_tools(handler)
    try:
        tool = tools["web_fetch"]
        first = await tool.run({"url": "https://example.com/report"})
        assert first["error_code"] == "http_error" and len(hits) == 1
        output = await tool.run({"url": "https://example.com/report", "render": False})
        assert output["error_code"] is None and "Original source facts" in output["content"]
        assert output["render_mode"] == "http"
        assert [str(target.value) for target in tool.resolve_use({
            "url": "https://example.com/report", "render": False,
        }).targets] == ["https://example.com/report"]
        assert len(hits) == 2
        assert (await tool.run({"url": "https://example.com/report", "render": False}))["cache_hit"]
        assert len(hits) == 2
    finally:
        await client.aclose()


@pytest.mark.anyio
async def test_configured_search_does_not_remove_the_explicit_playwright_backend(monkeypatch):
    from agent_runtime.tools import web_browser

    rendered = []

    async def render(client, url, **options):
        rendered.append(url)
        return FetchedResponse(url, "text/html", b"<main>Rendered original text</main>", 100, "direct")

    monkeypatch.setattr(web_browser, "render_public_page", render)
    tools, client, _ = hosted_tools(lambda _: pytest.fail("Explicit browser must not call the managed Fetch API"))
    try:
        tool = tools["web_fetch"]
        output = await tool.run({"url": "https://example.com/application", "render": True})
        assert output["error_code"] is None and "Rendered original text" in output["content"]
        assert output["render_mode"] == "browser" and rendered == ["https://example.com/application"]
    finally:
        await client.aclose()

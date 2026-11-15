"""Task 14:浏览器渲染路径 + 子资源逐问过同一授权面(S8 浏览器面闭合)。

测试分层(brief Step 1):
- **假路由层为主**(无 chromium 二进制):fake playwright 形状 harness 注入
  ``browser.async_playwright``——authorize 逐子资源计次、ask/deny ⇒ abort 且
  **不静默放行**(旧 deny-only bug 形状 RED)、预算触顶停 fetch 如实报 partial;
- tools.py render 矩阵(假 renderer,无需 playwright):render=true/auto/false ×
  renderer_factory 有无;auto 语义逐字对照 legacy builtins/web.py:513-520;
- WebLimits browser 块扩展(范围/未知键/缺省);
- 真 chromium 用例:**skipif 诚实跳过**(playwright 包缺失 / chromium 二进制
  未下载;原因字符串记入 skip,T21 汇总 §10)。本地 http.server + loopback
  backend 接缝(DNS 答案仍是公网 IP,connect 期重写到 loopback——与 T10 假
  backend 同一可信测试接缝,不削弱客户端的公网校验)。

环境事实(2026-11-16):playwright 包在、chromium 二进制不在
(~/Library/Caches/ms-playwright 空)⇒ 真浏览器用例 collect-as-skip。
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import signal
import subprocess
import threading
from dataclasses import fields
from pathlib import Path
from typing import Any

import httpcore
import pytest

from praxis.adapters.web import browser
from praxis.adapters.web.client import PublicHttpClient, PublicWebError, WebFetchResponse
from praxis.adapters.web.limits import WebLimits, load_web_limits
from praxis.adapters.web.tools import WEB_FETCH_TOOL_NAME, create_web_fetch_tools
from praxis.protocol.json import thaw_json
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import InterruptBehavior, ResultStatus, ToolEffect, ToolValidationError
from tests.praxis.test_web_urls_dns import (
    PUBLIC_IP,
    FakeBackend,
    FakeResolver,
    FakeStream,
    make_client,
    response_chunks,
)

URL = "https://example.com/page"
DOM = "<html><body><p>rendered marker</p></body></html>"


# ---------------------------------------------------------------------------
# 假 playwright 形状 harness(无二进制;只模拟公开 API 形状)
# ---------------------------------------------------------------------------


class FakeRequest:
    def __init__(
        self,
        url: str,
        *,
        method: str = "GET",
        resource_type: str = "script",
        frame: object = None,
        navigation: bool = False,
    ) -> None:
        self.url = url
        self.method = method
        self.resource_type = resource_type
        self.frame = frame
        self._navigation = navigation

    def is_navigation_request(self) -> bool:
        return self._navigation


class FakeRoute:
    def __init__(self, request: FakeRequest, harness: FakeHarness) -> None:
        self.request = request
        self._harness = harness
        self.outcome: tuple[str, Any] | None = None

    async def abort(self, error_code: str | None = None) -> None:
        self.outcome = ("abort", error_code)
        self._harness.aborted.append((self.request.url, error_code))

    async def fulfill(
        self, *, status: int = 200, headers: dict[str, str] | None = None, body: bytes = b""
    ) -> None:
        self.outcome = ("fulfill", status)
        self._harness.fulfilled.append((self.request.url, status, body, headers or {}))


class FakeWebSocketRoute:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakePage:
    def __init__(self, harness: FakeHarness) -> None:
        self._harness = harness
        self.main_frame = object()
        self.url = "about:blank"

    async def goto(self, url: str, *, wait_until: str, timeout: float) -> None:
        self._harness.goto_calls.append((url, wait_until, timeout))
        self.url = url
        main = FakeRequest(url, frame=self.main_frame, navigation=True, resource_type="document")
        route = await self._harness.dispatch(main)
        if route.outcome is not None and route.outcome[0] == "abort":
            raise browser.PlaywrightError("net::ERR_BLOCKED_BY_CLIENT")
        for request in self._harness.subresources:
            await self._harness.dispatch(request)
        if self._harness.websocket:
            ws = FakeWebSocketRoute()
            await self._harness.ws_handler(ws)
            self._harness.websockets.append(ws)
        if self._harness.hang:
            await asyncio.sleep(3600)

    async def wait_for_load_state(self, state: str, *, timeout: float) -> None:
        self._harness.wait_states.append((state, timeout))
        if self._harness.crash_on == "networkidle":
            raise browser.PlaywrightError("Target crashed")
        if state in self._harness.load_timeouts:
            raise browser.PlaywrightTimeoutError(f"Timeout {state}")

    async def wait_for_timeout(self, milliseconds: float) -> None:
        self._harness.settles.append(milliseconds)
        if self._harness.crash_on == "settle":
            raise browser.PlaywrightError("Target crashed")

    async def content(self) -> str:
        if self._harness.crash_on == "content":
            raise browser.PlaywrightError("Target crashed")
        return self._harness.dom


class FakeCdpSession:
    def __init__(self, pid: int | None) -> None:
        self._pid = pid
        self.detached = False

    async def send(self, method: str) -> dict[str, Any]:
        assert method == "SystemInfo.getProcessInfo"
        if self._pid is None:
            raise browser.PlaywrightError("SystemInfo domain unavailable")
        return {"processInfo": [{"type": "browser", "pid": self._pid}]}

    async def detach(self) -> None:
        self.detached = True


class FakeContext:
    def __init__(self, harness: FakeHarness) -> None:
        self._harness = harness
        self.page = FakePage(harness)

    async def route(self, pattern: str, handler: Any) -> None:
        assert pattern == "**/*"
        self._harness.route_handler = handler

    async def route_web_socket(self, pattern: str, handler: Any) -> None:
        assert pattern == "**/*"
        self._harness.ws_handler = handler

    async def new_page(self) -> FakePage:
        return self.page

    async def close(self) -> None:
        # review R1 minor:私有目录必须在浏览器关闭之后才删。
        env = (self._harness.launch_kwargs or {}).get("env", {})
        self._harness.dir_alive_at_close = Path(env["HOME"]).exists() if env.get("HOME") else None
        self._harness.closed.append("context")


class FakeBrowser:
    def __init__(self, harness: FakeHarness) -> None:
        self._harness = harness

    async def new_context(self, **kwargs: Any) -> FakeContext:
        self._harness.context_kwargs = kwargs
        return FakeContext(self._harness)

    async def new_browser_cdp_session(self) -> FakeCdpSession:
        self._harness.cdp_sessions += 1
        return FakeCdpSession(self._harness.cdp_pid)

    async def close(self) -> None:
        self._harness.closed.append("browser")


class FakeChromium:
    def __init__(self, harness: FakeHarness) -> None:
        self._harness = harness

    async def launch(self, **kwargs: Any) -> FakeBrowser:
        self._harness.launch_kwargs = kwargs
        if self._harness.launch_error is not None:
            raise self._harness.launch_error
        return FakeBrowser(self._harness)


class FakeDriver:
    def __init__(self, harness: FakeHarness) -> None:
        self.chromium = FakeChromium(harness)
        self._harness = harness

    async def stop(self) -> None:
        self._harness.closed.append("driver")


class FakePlaywrightContextManager:
    def __init__(self, harness: FakeHarness) -> None:
        self._harness = harness

    async def start(self) -> FakeDriver:
        if self._harness.start_error is not None:
            raise self._harness.start_error
        return FakeDriver(self._harness)


class FakeHarness:
    """一次 render 的全部假浏览器状态;dispatch 逐请求过真实 route handler。"""

    def __init__(
        self,
        *,
        dom: str = DOM,
        subresources: list[FakeRequest] | None = None,
        hang: bool = False,
        websocket: bool = False,
        load_timeouts: tuple[str, ...] = (),
        launch_error: Exception | None = None,
        start_error: Exception | None = None,
        crash_on: str | None = None,
        cdp_pid: int | None = None,
    ) -> None:
        self.dom = dom
        self.subresources = subresources or []
        self.hang = hang
        self.websocket = websocket
        self.load_timeouts = load_timeouts
        self.launch_error = launch_error
        self.start_error = start_error
        self.crash_on = crash_on
        self.cdp_pid = cdp_pid
        self.launch_kwargs: dict[str, Any] | None = None
        self.context_kwargs: dict[str, Any] | None = None
        self.route_handler: Any = None
        self.ws_handler: Any = None
        self.fulfilled: list[tuple[str, int, bytes, dict[str, str]]] = []
        self.aborted: list[tuple[str, str | None]] = []
        self.websockets: list[FakeWebSocketRoute] = []
        self.goto_calls: list[tuple[str, str, float]] = []
        self.wait_states: list[tuple[str, float]] = []
        self.settles: list[float] = []
        self.closed: list[str] = []
        self.cdp_sessions = 0
        self.dir_alive_at_close: bool | None = None

    async def dispatch(self, request: FakeRequest) -> FakeRoute:
        assert self.route_handler is not None, "context.route must be registered before goto"
        route = FakeRoute(request, self)
        await self.route_handler(route)
        return route


class FakeFetchClient:
    """渲染器专用假客户端:bodies[url] = bytes | (bytes, status, final_url) | Exception。"""

    def __init__(self, bodies: dict[str, Any]) -> None:
        self.bodies = bodies
        self.calls: list[dict[str, Any]] = []

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Any = None,
        json_body: Any = None,
        max_bytes: int | None = None,
        timeout_seconds: float | None = None,
        follow_redirects: bool = True,
    ) -> WebFetchResponse:
        self.calls.append({"url": url, "max_bytes": max_bytes, "timeout_seconds": timeout_seconds})
        item = self.bodies.get(url)
        if item is None:
            raise PublicWebError("connect_failed", "no fake body for this URL")
        if isinstance(item, Exception):
            raise item
        body, status, final = item if isinstance(item, tuple) else (item, 200, url)
        if max_bytes is not None and len(body) > max_bytes:
            raise PublicWebError("max_bytes_exceeded", "The response exceeds the byte limit.")
        return WebFetchResponse(
            status=status, headers={}, body=body, final_url=final, content_type="text/html; charset=utf-8"
        )


def make_authorize(verdicts: dict[str, str]) -> tuple[Any, list[str]]:
    calls: list[str] = []

    def authorize(url: str) -> None:
        calls.append(url)
        verdict = verdicts.get(url, "allow")
        if verdict == "ask":
            raise PublicWebError("denied", "approval required for example.com", verdict="ask")
        if verdict == "deny":
            raise PublicWebError("denied", verdict="deny")

    return authorize, calls


def make_renderer(
    harness: FakeHarness,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bodies: dict[str, Any] | None = None,
    verdicts: dict[str, str] | None = None,
    limits: WebLimits | None = None,
    signaler: Any = None,
) -> tuple[browser.PageRenderer, FakeFetchClient, list[str]]:
    monkeypatch.setattr(browser, "async_playwright", lambda: FakePlaywrightContextManager(harness))
    client = FakeFetchClient(bodies if bodies is not None else {URL: DOM.encode()})
    authorize, calls = make_authorize(verdicts or {})
    renderer = browser.PageRenderer(
        client,  # type: ignore[arg-type]
        authorize=authorize,
        limits=limits if limits is not None else WebLimits(),
        signal_process_group=signaler,
    )
    return renderer, client, calls


def sub(url: str, **kwargs: Any) -> FakeRequest:
    return FakeRequest(url, **kwargs)


# ---------------------------------------------------------------------------
# 常量逐字(brief;T21 parity 复用)
# ---------------------------------------------------------------------------


def test_budget_constants_verbatim() -> None:
    assert browser.MAX_REQUESTS == 64
    assert browser.MAX_NETWORK_BYTES == 8_000_000
    assert browser.MAX_DOM_BYTES == 2_000_000
    assert browser.RENDER_TIMEOUT_SECONDS == 30


# ---------------------------------------------------------------------------
# 子资源闸:authorize 逐 URL、ask/deny 同拦(旧 deny-only bug 形状)
# ---------------------------------------------------------------------------


def test_gate_authorizes_every_subresource_url_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_urls = [
        "https://example.com/a.js",
        "HTTPS://Example.COM./b.js?Q=1",
        "https://example.com/c.css",
        "https://cdn.example.net/d.js",
        "https://example.com/e.js",
    ]
    harness = FakeHarness(subresources=[sub(u) for u in raw_urls])
    renderer, _, calls = make_renderer(
        harness, monkeypatch, bodies={URL: DOM.encode(), **{u: b"/*ok*/" for u in raw_urls}}
    )
    page = asyncio.run(renderer.render(URL))
    # 主文档 + 5 子资源逐个过 authorize,原样字符串(无再归一化)。
    assert calls == [URL, *raw_urls]
    assert len(harness.fulfilled) == 6
    assert page.blocked_ask == 0 and page.blocked_deny == 0
    assert page.requests_used == 6
    assert page.text == "rendered marker"


def test_ask_and_deny_subresources_abort_without_silent_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = [f"https://example.com/{name}.js" for name in "abcde"]
    verdicts = {urls[1]: "ask", urls[3]: "ask", urls[4]: "deny"}
    harness = FakeHarness(subresources=[sub(u) for u in urls])
    renderer, _, calls = make_renderer(
        harness,
        monkeypatch,
        bodies={URL: DOM.encode(), **{u: b"/*ok*/" for u in urls}},
        verdicts=verdicts,
    )
    page = asyncio.run(renderer.render(URL))
    assert calls == [URL, *urls]
    # 2 ask + 1 deny ⇒ 3 个 abort("blockedbyclient"),绝不静默放行。
    assert harness.aborted == [
        (urls[1], "blockedbyclient"),
        (urls[3], "blockedbyclient"),
        (urls[4], "blockedbyclient"),
    ]
    assert page.blocked_ask == 2
    assert page.blocked_deny == 1
    assert len(harness.fulfilled) == 3  # 主文档 + 2 个 allow 子资源
    assert page.requests_used == 3  # 被拦请求不消耗预算
    assert page.text == "rendered marker"


def test_main_document_denied_midrender_surfaces_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    # 初始抓取放行,route 期策略翻转为 deny ⇒ goto 失败必须上抛 denied(不是 render_failed)。
    harness = FakeHarness()
    renderer, _, _ = make_renderer(harness, monkeypatch, verdicts={URL: "deny"})
    with pytest.raises(PublicWebError) as caught:
        asyncio.run(renderer.render(URL))
    assert caught.value.code == "denied"
    assert harness.closed == ["context", "browser", "driver"]


def test_non_get_and_websocket_are_structurally_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = FakeHarness(subresources=[sub("https://example.com/post", method="POST")], websocket=True)
    renderer, _, _ = make_renderer(harness, monkeypatch)
    page = asyncio.run(renderer.render(URL))
    assert page.blocked_other == 2  # POST + websocket 各一
    assert harness.aborted == [("https://example.com/post", "blockedbyclient")]
    assert len(harness.websockets) == 1 and harness.websockets[0].closed


def test_subresource_network_failure_aborts_and_render_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = "https://example.com/bad.js"
    harness = FakeHarness(subresources=[sub(bad), sub("https://example.com/ok.js")])
    renderer, _, _ = make_renderer(
        harness,
        monkeypatch,
        bodies={URL: DOM.encode(), bad: PublicWebError("timeout", "x"), "https://example.com/ok.js": b"/*ok*/"},
    )
    page = asyncio.run(renderer.render(URL))
    assert page.failed_resources == 1
    assert ("https://example.com/bad.js", "failed") in harness.aborted
    assert page.text == "rendered marker"


def test_redirected_subresource_counts_and_fulfills_final_body(monkeypatch: pytest.MonkeyPatch) -> None:
    redirected = "https://example.com/start.js"
    harness = FakeHarness(subresources=[sub(redirected)])
    renderer, _, _ = make_renderer(
        harness,
        monkeypatch,
        bodies={URL: DOM.encode(), redirected: (b"/*final*/", 200, "https://example.com/final.js")},
    )
    page = asyncio.run(renderer.render(URL))
    assert page.redirected_subresources == 1
    assert harness.fulfilled[-1][0] == redirected
    assert harness.fulfilled[-1][2] == b"/*final*/"


# ---------------------------------------------------------------------------
# 预算:请求数 / 总字节 / 单响应 8MB / DOM 2MB
# ---------------------------------------------------------------------------


def test_request_budget_aborts_65th_request_and_reports_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = [f"https://example.com/{i}.js" for i in range(70)]
    harness = FakeHarness(subresources=[sub(u) for u in urls])
    renderer, _, _ = make_renderer(
        harness, monkeypatch, bodies={URL: DOM.encode(), **{u: b"/*ok*/" for u in urls}}
    )
    page = asyncio.run(renderer.render(URL))
    assert page.requests_used == 64
    assert page.budget_stopped is True
    assert page.budget_aborted == 7  # 主文档 1 + 63 子资源放行,第 65 起共 7 个被拦
    assert len(harness.fulfilled) == 64
    assert len(harness.aborted) == 7
    assert all(code == "blockedbyclient" for _, code in harness.aborted)
    # 预算触顶 ⇒ 跳过 networkidle/settle(budget-aware early exit)。
    assert harness.wait_states == []
    assert harness.settles == []


def test_byte_budget_stops_fetch_and_counts_only_confirmed_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    big = "https://example.com/big.js"
    harness = FakeHarness(subresources=[sub(big), sub("https://example.com/after.js")])
    renderer, client, _ = make_renderer(
        harness,
        monkeypatch,
        bodies={URL: b"x" * 600_000, big: b"y" * 500_000, "https://example.com/after.js": b"/*ok*/"},
        limits=WebLimits(browser_network_bytes=1_000_000),
    )
    page = asyncio.run(renderer.render(URL))
    assert page.bytes_used == 600_000  # 超限响应不计入"已确认字节"
    assert page.budget_stopped is True
    # 单响应上限 = min(8MB, 剩余总预算) ⇒ 本次 400_000。
    assert client.calls[1]["max_bytes"] == 400_000
    assert harness.aborted[0] == (big, "failed")
    assert page.requests_used == 2


def test_single_response_cap_is_8mb_even_with_larger_total_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    huge = "https://example.com/huge.js"
    harness = FakeHarness(subresources=[sub(huge)])
    renderer, client, _ = make_renderer(
        harness,
        monkeypatch,
        bodies={URL: DOM.encode(), huge: b"z" * (8_000_000 + 1)},
        limits=WebLimits(browser_network_bytes=32_000_000),
    )
    page = asyncio.run(renderer.render(URL))
    assert client.calls[1]["max_bytes"] == 8_000_000
    assert page.budget_stopped is True
    assert page.bytes_used == len(DOM.encode())


def test_dom_over_2mb_raises_render_budget_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = FakeHarness(dom="<html><body>" + "x" * 2_000_000 + "</body></html>")
    renderer, _, _ = make_renderer(harness, monkeypatch)
    with pytest.raises(PublicWebError) as caught:
        asyncio.run(renderer.render(URL))
    assert caught.value.code == "render_budget_exceeded"
    assert harness.closed == ["context", "browser", "driver"]


def test_custom_budget_from_limits_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = [f"https://example.com/{i}.js" for i in range(5)]
    harness = FakeHarness(subresources=[sub(u) for u in urls])
    renderer, _, _ = make_renderer(
        harness,
        monkeypatch,
        bodies={URL: DOM.encode(), **{u: b"/*ok*/" for u in urls}},
        limits=WebLimits(browser_requests=3),
    )
    page = asyncio.run(renderer.render(URL))
    assert page.requests_used == 3
    assert page.budget_aborted == 3


# ---------------------------------------------------------------------------
# 等待策略 / 超时 / 可用性 / 清理
# ---------------------------------------------------------------------------


def test_load_wait_strategy_and_timeouts_retain_dom(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = FakeHarness(load_timeouts=("networkidle",))
    renderer, _, _ = make_renderer(harness, monkeypatch)
    page = asyncio.run(renderer.render(URL))
    assert harness.goto_calls[0][1] == "domcontentloaded"
    assert harness.goto_calls[0][2] == 5000
    assert harness.wait_states == [("networkidle", 10_000)]
    assert harness.settles == [1500]
    assert page.load_timeouts == ("networkidle",)
    assert page.text == "rendered marker"


def test_overall_render_deadline_maps_render_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(browser, "RENDER_TIMEOUT_SECONDS", 0.2)
    harness = FakeHarness(hang=True)
    renderer, _, _ = make_renderer(harness, monkeypatch)
    with pytest.raises(PublicWebError) as caught:
        asyncio.run(renderer.render(URL))
    assert caught.value.code == "render_timeout"
    assert harness.closed == ["context", "browser", "driver"]


def test_missing_playwright_maps_render_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(browser, "_PLAYWRIGHT_AVAILABLE", False)
    renderer, client, _ = make_renderer(FakeHarness(), monkeypatch)
    with pytest.raises(PublicWebError) as caught:
        asyncio.run(renderer.render(URL))
    assert caught.value.code == "render_unavailable"
    assert "playwright install chromium" in str(caught.value)
    assert client.calls == []  # 不可用时零网络


def test_launch_failure_maps_render_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = FakeHarness(launch_error=browser.PlaywrightError("Executable doesn't exist"))
    renderer, _, _ = make_renderer(harness, monkeypatch)
    with pytest.raises(PublicWebError) as caught:
        asyncio.run(renderer.render(URL))
    assert caught.value.code == "render_unavailable"
    assert "playwright install chromium" in str(caught.value)


def test_browser_env_whitelist_proxy_and_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = FakeHarness()
    renderer, _, _ = make_renderer(harness, monkeypatch)
    asyncio.run(renderer.render(URL))
    assert harness.launch_kwargs is not None
    env = harness.launch_kwargs["env"]
    # 固定常量白名单:恰三键,比 child_env 更窄(宿主信任面第三姿态)。
    assert set(env) == {"HOME", "TMPDIR", "PATH"}
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"] == env["TMPDIR"]
    assert "praxis-web-browser-" in env["HOME"]
    # 断直连代理(纵深防御,legacy :300-303 原样)。
    assert harness.launch_kwargs["proxy"] == {"server": "http://127.0.0.1:9", "bypass": "<-loopback>"}
    assert harness.launch_kwargs["chromium_sandbox"] is True
    assert harness.launch_kwargs["headless"] is True
    # 出口硬禁(review R1):WebRTC 非代理 UDP 与 WebTransport 必须出现在启动参数。
    args = harness.launch_kwargs["args"]
    assert "--force-webrtc-ip-handling-policy=disable_non_proxied_udp" in args
    assert "--disable-features=WebTransport" in args
    # launch 超时预算-aware(legacy remaining_ms(15_000) 同口径)。
    assert harness.launch_kwargs["timeout"] == 15_000
    assert harness.context_kwargs == {"service_workers": "block", "accept_downloads": False}
    # 私有目录在关闭时仍存活,render 返回后才删(review R1 minor 顺序钉)。
    assert harness.dir_alive_at_close is True
    assert not Path(env["HOME"]).exists()


def test_cleanup_killpg_fallback_targets_discovered_group(monkeypatch: pytest.MonkeyPatch) -> None:
    # 真子进程当"残留浏览器":独立会话 ⇒ pgid == pid;注入 signaler 必须整组 KILL。
    dummy = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        harness = FakeHarness(cdp_pid=dummy.pid)
        signaled: list[tuple[int, signal.Signals]] = []

        def signaler(pgid: int, sig: signal.Signals) -> bool:
            signaled.append((pgid, sig))
            os.killpg(pgid, sig)
            return True

        renderer, _, _ = make_renderer(harness, monkeypatch, signaler=signaler)
        asyncio.run(renderer.render(URL))
        assert harness.closed == ["context", "browser", "driver"]
        assert signaled == [(dummy.pid, signal.SIGKILL)]
        assert dummy.wait(timeout=5) == -signal.SIGKILL
    finally:
        if dummy.poll() is None:
            dummy.kill()
            dummy.wait()


def test_no_signaler_means_no_cdp_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = FakeHarness(cdp_pid=99999)
    renderer, _, _ = make_renderer(harness, monkeypatch)
    asyncio.run(renderer.render(URL))
    assert harness.cdp_sessions == 0
    assert harness.closed == ["context", "browser", "driver"]


def test_render_seeded_initial_skips_second_document_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = FakeHarness()
    renderer, client, calls = make_renderer(harness, monkeypatch)
    initial = WebFetchResponse(
        status=200, headers={}, body=DOM.encode(), final_url=URL, content_type="text/html; charset=utf-8"
    )
    page = asyncio.run(renderer.render(URL, initial=initial))
    assert client.calls == []  # auto 路径:初始响应已种子化,零第二次文档读取
    assert calls == [URL]  # route 期仍逐 URL 过 authorize(主文档导航)
    assert page.requests_used == 1
    assert page.bytes_used == len(DOM.encode())
    assert harness.fulfilled[0][2] == DOM.encode()


def test_non_2xx_initial_maps_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = FakeHarness()
    renderer, _, _ = make_renderer(harness, monkeypatch, bodies={URL: (b"gone", 404, URL)})
    with pytest.raises(PublicWebError) as caught:
        asyncio.run(renderer.render(URL))
    assert caught.value.code == "http_error"
    assert harness.launch_kwargs is None  # 非 2xx 不起浏览器


def test_initial_main_document_capped_by_browser_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # review R1 #2:主文档与子资源同一口径——宿主 max_bytes=32MB 也不得一次读穿
    # 8MB 单响应帽;9MB 主文档 ⇒ 流式超限(budget-stop 形状),不是 32MB 读取。
    harness = FakeHarness()
    renderer, client, _ = make_renderer(
        harness,
        monkeypatch,
        bodies={URL: b"x" * 9_000_000},
        limits=WebLimits(max_bytes=32_000_000, browser_network_bytes=32_000_000),
    )
    with pytest.raises(PublicWebError) as caught:
        asyncio.run(renderer.render(URL))
    assert caught.value.code == "max_bytes_exceeded"
    assert client.calls[0]["max_bytes"] == 8_000_000
    assert harness.launch_kwargs is None  # 主文档超限不起浏览器


def test_initial_3mb_page_succeeds_under_default_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    # review R1 #2:默认传输上限 2MB 不得影子化 browser 预算——3MB wire 主文档
    # 默认被接受(假 DOM 与 wire body 独立:渲染产物是小 DOM,不触 2MB DOM 帽)。
    harness = FakeHarness()
    renderer, client, _ = make_renderer(harness, monkeypatch, bodies={URL: b"x" * 3_000_000})
    page = asyncio.run(renderer.render(URL))
    assert client.calls[0]["max_bytes"] == 8_000_000  # min(8MB, 默认 browser 预算 8MB)
    assert page.bytes_used == 3_000_000 > 2_000_000
    assert page.text == "rendered marker"


def test_driver_start_failure_maps_render_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # review R1 #3:start 相位 PlaywrightError 不得裸泄。
    harness = FakeHarness(start_error=browser.PlaywrightError("driver failed to start"))
    renderer, _, _ = make_renderer(harness, monkeypatch)
    with pytest.raises(PublicWebError) as caught:
        asyncio.run(renderer.render(URL))
    assert caught.value.code == "render_unavailable"
    assert "playwright install chromium" in str(caught.value)


@pytest.mark.parametrize("crash_on", ["networkidle", "settle", "content"])
def test_midrender_crash_maps_render_failed(monkeypatch: pytest.MonkeyPatch, crash_on: str) -> None:
    # review R1 #3:渲染中(攻击者可触发)renderer crash ⇒ render_failed,不裸泄。
    harness = FakeHarness(crash_on=crash_on)
    renderer, _, _ = make_renderer(harness, monkeypatch)
    with pytest.raises(PublicWebError) as caught:
        asyncio.run(renderer.render(URL))
    assert caught.value.code == "render_failed"
    assert harness.closed == ["context", "browser", "driver"]


# ---------------------------------------------------------------------------
# WebLimits browser 块扩展
# ---------------------------------------------------------------------------


def test_web_limits_browser_defaults_and_fields() -> None:
    limits = WebLimits()
    assert limits.browser_requests == 64
    assert limits.browser_network_bytes == 8_000_000
    names = tuple(f.name for f in fields(WebLimits))
    assert names == ("max_bytes", "timeout_seconds", "max_redirects", "browser_requests", "browser_network_bytes")
    assert not hasattr(limits, "turn_requests")  # Q7:turn 账本仍不在 3B


@pytest.mark.parametrize(
    "kwargs",
    [
        {"browser_requests": 0},
        {"browser_requests": 129},
        {"browser_requests": 1.5},
        {"browser_requests": True},
        {"browser_network_bytes": 999_999},
        {"browser_network_bytes": 32_000_001},
        {"browser_network_bytes": "8000000"},
    ],
)
def test_web_limits_browser_range_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        WebLimits(**kwargs)


def test_web_limits_browser_boundaries() -> None:
    assert WebLimits(browser_requests=1).browser_requests == 1
    assert WebLimits(browser_requests=128).browser_requests == 128
    assert WebLimits(browser_network_bytes=1_000_000).browser_network_bytes == 1_000_000
    assert WebLimits(browser_network_bytes=32_000_000).browser_network_bytes == 32_000_000


def test_load_web_limits_browser_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRAXIS_WEB_LIMITS", raising=False)
    config = tmp_path / "web.json"
    config.write_text(
        json.dumps(
            {"limits": {"max_bytes": 8192}, "browser": {"browser_requests": 96, "browser_network_bytes": 16_000_000}}
        ),
        encoding="utf-8",
    )
    limits = load_web_limits(config)
    assert limits == WebLimits(max_bytes=8192, browser_requests=96, browser_network_bytes=16_000_000)
    # 块缺省 ⇒ 默认;limits 与 browser 互不牵连。
    config.write_text(json.dumps({"limits": {"max_bytes": 8192}}), encoding="utf-8")
    assert load_web_limits(config) == WebLimits(max_bytes=8192)
    config.write_text(json.dumps({"browser": {"browser_requests": 7}}), encoding="utf-8")
    assert load_web_limits(config) == WebLimits(browser_requests=7)


@pytest.mark.parametrize(
    "payload",
    [
        {"browser": {"browser_requests": 0}},
        {"browser": {"browser_network_bytes": 1}},
        {"browser": {"turn_requests": 32}},
        {"browser": []},
        {"limits": {"browser_requests": 64}},
    ],
)
def test_load_web_limits_browser_malformed_names_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    monkeypatch.delenv("PRAXIS_WEB_LIMITS", raising=False)
    config = tmp_path / "web.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=str(config)):
        load_web_limits(config)


# ---------------------------------------------------------------------------
# tools.py render 矩阵(假 renderer;无需 playwright/chromium)
# ---------------------------------------------------------------------------


def make_rendered_page(**overrides: Any) -> browser.RenderedPage:
    base: dict[str, Any] = {
        "text": "line one\nline two\nline three",
        "final_url": URL,
        "requests_used": 4,
        "bytes_used": 1234,
        "dom_bytes": 5678,
        "truncated": False,
        "blocked_ask": 1,
        "blocked_deny": 2,
        "blocked_other": 0,
        "budget_aborted": 0,
        "budget_stopped": False,
        "failed_resources": 0,
        "redirected_subresources": 0,
        "load_timeouts": (),
    }
    base.update(overrides)
    return browser.RenderedPage(**base)


class FakeRenderer:
    def __init__(self, page: browser.RenderedPage | None = None, error: Exception | None = None) -> None:
        self.page = page if page is not None else make_rendered_page()
        self.error = error
        self.calls: list[tuple[str, WebFetchResponse | None]] = []

    async def render(self, url: str, *, initial: WebFetchResponse | None = None) -> browser.RenderedPage:
        self.calls.append((url, initial))
        if self.error is not None:
            raise self.error
        return self.page


def text_tool(
    bodies: list[bytes],
    *,
    content_type: str = "text/html; charset=utf-8",
    renderer: FakeRenderer | None = None,
    authorize: Any = None,
) -> tuple[Any, FakeBackend]:
    backend = FakeBackend(
        [
            FakeStream(response_chunks("200 OK", headers=[("Content-Type", content_type)], body=body))
            for body in bodies
        ]
    )
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend, authorize=authorize)
    (tool,) = create_web_fetch_tools(
        client,
        limits=WebLimits(),
        renderer_factory=(lambda: renderer) if renderer is not None else None,
    )
    return tool, backend


def run_tool(tool: Any, arguments: dict[str, Any]) -> Any:
    return asyncio.run(tool.run({"url": URL, **arguments}, ToolExecContext()))


EMPTY_HTML = b"<html><body><script>only scripts()</script></body></html>"


def test_render_true_without_factory_is_render_unavailable() -> None:
    tool, backend = text_tool([DOM.encode()])
    result = run_tool(tool, {"render": True})
    assert result.status is ResultStatus.ERROR
    assert result.error_code == "render_unavailable"
    assert "playwright install chromium" in result.error_message
    assert backend.connects == []  # 不可用即零网络


def test_render_auto_without_factory_empty_html_is_render_unavailable() -> None:
    tool, _ = text_tool([EMPTY_HTML])
    result = run_tool(tool, {"render": "auto"})
    assert result.error_code == "render_unavailable"


def test_render_false_never_touches_renderer() -> None:
    renderer = FakeRenderer()
    tool, _ = text_tool([DOM.encode()], renderer=renderer)
    result = run_tool(tool, {"render": False})
    assert result.status is ResultStatus.OK
    assert renderer.calls == []
    assert "rendered" not in result.metadata  # 文本路径 metadata 形状不变(T12)
    assert result.metadata["content_type"] == "text/html; charset=utf-8"


def test_render_default_is_false() -> None:
    renderer = FakeRenderer()
    tool, _ = text_tool([DOM.encode()], renderer=renderer)
    result = run_tool(tool, {})
    assert result.status is ResultStatus.OK
    assert renderer.calls == []


def test_auto_nonempty_html_stays_text() -> None:
    renderer = FakeRenderer()
    tool, _ = text_tool([DOM.encode()], renderer=renderer)
    result = run_tool(tool, {"render": "auto"})
    assert result.status is ResultStatus.OK
    assert renderer.calls == []
    assert "rendered marker" in result.text


def test_auto_empty_html_renders_once_with_seeded_initial() -> None:
    # legacy builtins/web.py:513-520 逐字:attempt==0 ∧ auto ∧ 抽取 web_content_empty
    # ∧ content-type ∈ {text/html, application/xhtml+xml} ⇒ 浏览器一次,种子初始响应。
    renderer = FakeRenderer()
    tool, backend = text_tool([EMPTY_HTML], renderer=renderer)
    result = run_tool(tool, {"render": "auto"})
    assert result.status is ResultStatus.OK
    assert len(renderer.calls) == 1
    seeded_url, initial = renderer.calls[0]
    assert seeded_url == URL
    assert initial is not None and initial.body == EMPTY_HTML  # 不第二次 HTTP 读取
    assert len(backend.connects) == 1
    assert result.metadata["rendered"] is True
    assert "line one" in result.text


def test_auto_empty_xhtml_renders() -> None:
    renderer = FakeRenderer()
    tool, _ = text_tool([EMPTY_HTML], content_type="application/xhtml+xml", renderer=renderer)
    result = run_tool(tool, {"render": "auto"})
    assert result.status is ResultStatus.OK
    assert len(renderer.calls) == 1


def test_auto_empty_non_html_does_not_render() -> None:
    renderer = FakeRenderer()
    tool, _ = text_tool([b"   "], content_type="text/plain; charset=utf-8", renderer=renderer)
    result = run_tool(tool, {"render": "auto"})
    assert result.error_code == "web_fetch_failed:web_content_empty"
    assert renderer.calls == []


def test_auto_unsupported_content_does_not_render() -> None:
    renderer = FakeRenderer()
    tool, _ = text_tool([b"\x00binary"], content_type="application/octet-stream", renderer=renderer)
    result = run_tool(tool, {"render": "auto"})
    assert result.error_code == "web_fetch_failed:web_content_unsupported"
    assert renderer.calls == []


def test_auto_render_failure_after_empty_extraction_maps_error() -> None:
    renderer = FakeRenderer(error=PublicWebError("render_timeout", "Browser rendering exceeded its deadline."))
    tool, _ = text_tool([EMPTY_HTML], renderer=renderer)
    result = run_tool(tool, {"render": "auto"})
    assert result.error_code == "web_fetch_failed:render_timeout"


def test_render_true_uses_renderer_without_initial_and_windows_text() -> None:
    renderer = FakeRenderer()
    tool, backend = text_tool([], renderer=renderer)
    result = run_tool(tool, {"render": True, "start_line": 2, "max_lines": 1})
    assert result.status is ResultStatus.OK
    assert renderer.calls == [(URL, None)]
    assert backend.connects == []  # render=true:文档抓取归渲染器,工具层零抓取
    assert result.text == "2: line two"
    metadata = result.metadata
    assert metadata["rendered"] is True
    assert metadata["requests_used"] == 4
    assert metadata["bytes_used"] == 1234
    assert metadata["dom_bytes"] == 5678
    assert metadata["blocked_ask"] == 1
    assert metadata["blocked_deny"] == 2
    assert metadata["budget_stopped"] is False
    assert metadata["line_count"] == 3
    assert metadata["end_line"] == 2
    assert metadata["truncated"] is True


def test_render_budget_stopped_marks_result_truncated() -> None:
    renderer = FakeRenderer(page=make_rendered_page(budget_stopped=True, budget_aborted=3))
    tool, _ = text_tool([], renderer=renderer)
    result = run_tool(tool, {"render": True})
    assert result.status is ResultStatus.OK
    assert result.truncated is True
    assert result.metadata["budget_stopped"] is True
    assert result.metadata["budget_aborted"] == 3


def test_render_denied_maps_denied() -> None:
    renderer = FakeRenderer(error=PublicWebError("denied", verdict="deny"))
    tool, _ = text_tool([], renderer=renderer)
    result = run_tool(tool, {"render": True})
    assert result.error_code == "denied"


def test_render_unavailable_from_renderer_maps_directly() -> None:
    renderer = FakeRenderer(error=PublicWebError("render_unavailable", "Browser rendering is unavailable."))
    tool, _ = text_tool([], renderer=renderer)
    result = run_tool(tool, {"render": True})
    assert result.error_code == "render_unavailable"


def test_render_schema_key_and_remaining_absent_keys() -> None:
    tool, _ = text_tool([DOM.encode()])
    schema = tool.definition.input_schema
    properties = schema["properties"]
    assert set(properties) == {"url", "max_bytes", "timeout_seconds", "start_line", "max_lines", "render"}
    # R-PF-2' 其余键仍缺席(T14 只加 render)。
    assert {
        "source_id", "compare_to_source_id", "view", "section_id",
        "start_link", "start_section", "find", "max_tokens", "refresh",
    }.isdisjoint(properties)
    render_schema = properties["render"]
    assert render_schema["default"] is False
    # json_schema_input 冻结 schema(mappingproxy/tuple):thaw_json 归一再比。
    any_of = thaw_json(render_schema["anyOf"])
    assert {"type": "boolean"} in any_of
    assert {"type": "string", "enum": ["auto"]} in any_of
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("bad", ["true", "yes", 1, 0, None])
def test_render_invalid_values_rejected(bad: Any) -> None:
    tool, _ = text_tool([DOM.encode()])
    with pytest.raises(ToolValidationError, match="render"):
        tool.validate_input({"url": URL, "render": bad})


def test_effects_and_interrupt_split_by_renderer_capability() -> None:
    plain, _ = text_tool([DOM.encode()])
    assert plain.static_effects == frozenset({ToolEffect.NETWORK})
    assert plain.interrupt_behavior is InterruptBehavior.FINISH_CURRENT
    for render in (False, True, "auto"):
        use = plain.resolve_use({"url": URL, "render": render})
        assert use.effects == frozenset({ToolEffect.NETWORK})

    renderer = FakeRenderer()
    capable, _ = text_tool([DOM.encode()], renderer=renderer)
    assert capable.static_effects == frozenset({ToolEffect.NETWORK, ToolEffect.EXECUTE_PROCESS})
    assert capable.interrupt_behavior is InterruptBehavior.CANCEL
    assert capable.resolve_use({"url": URL, "render": False}).effects == frozenset({ToolEffect.NETWORK})
    for render in (True, "auto"):
        use = capable.resolve_use({"url": URL, "render": render})
        assert use.effects == frozenset({ToolEffect.NETWORK, ToolEffect.EXECUTE_PROCESS})
        assert use.targets[0].kind == "url" and use.targets[0].value == URL
    assert capable.definition.name == WEB_FETCH_TOOL_NAME
    assert "render" in capable.definition.description


# ---------------------------------------------------------------------------
# 真 chromium 用例(本机无二进制 ⇒ collect-as-skip;原因字符串记 §10)
# ---------------------------------------------------------------------------


def _chromium_probe() -> tuple[bool, str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, "playwright package not installed (uv sync --extra browser)"
    try:
        with sync_playwright() as pw:
            executable = pw.chromium.executable_path
    except Exception as error:  # driver 启动失败同样按不可用跳过
        return False, f"playwright driver probe failed: {error}"
    if not executable or not Path(executable).exists():
        return False, "chromium binary not downloaded (uv run python -m playwright install chromium)"
    return True, "chromium available"


_CHROMIUM_OK, _CHROMIUM_REASON = _chromium_probe()
requires_chromium = pytest.mark.skipif(not _CHROMIUM_OK, reason=_CHROMIUM_REASON)


class LoopbackBackend(httpcore.AsyncNetworkBackend):
    """可信测试接缝:DNS 答案仍是公网 IP(T10 校验不变),connect 期重写到本地 http.server。"""

    def __init__(self, port: int) -> None:
        self._inner = httpcore.AnyIOBackend()
        self._port = port

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        return await self._inner.connect_tcp("127.0.0.1", self._port, timeout, local_address, socket_options)

    async def connect_unix_socket(self, path: str, timeout: float | None = None) -> httpcore.AsyncNetworkStream:
        raise AssertionError("unix socket must not be used")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class _SiteHandler(http.server.BaseHTTPRequestHandler):
    server_version = "PraxisTestSite/1.0"

    def do_GET(self) -> None:  # noqa: N802 - http.server 约定
        site = self.server.site  # type: ignore[attr-defined]
        site["seen"].append(self.path)
        body, status, headers = site["routes"].get(self.path, (b"not found", 404, {}))
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _start_site(routes: dict[str, tuple[bytes, int, dict[str, str]]]) -> tuple[Any, dict[str, Any]]:
    site: dict[str, Any] = {"routes": routes, "seen": []}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SiteHandler)
    server.site = site  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, site


def _real_client(port: int, authorize: Any) -> PublicHttpClient:
    return PublicHttpClient(
        authorize=authorize,
        limits=WebLimits(),
        resolver=FakeResolver([PUBLIC_IP]),
        backend=LoopbackBackend(port),
        environ={},
    )


@requires_chromium
def test_real_chromium_renders_three_subresources_with_counts() -> None:
    html = (
        b"<html><body><p id='out'>static</p>"
        b"<script src='/a.js'></script><script src='/b.js'></script><script src='/c.js'></script>"
        b"</body></html>"
    )

    def marker(name: str) -> tuple[bytes, int, dict[str, str]]:
        return (
            f"document.getElementById('out').textContent += ' {name}';".encode(),
            200,
            {"Content-Type": "application/javascript"},
        )

    routes = {
        "/": (html, 200, {"Content-Type": "text/html; charset=utf-8"}),
        "/a.js": marker("alpha"),
        "/b.js": marker("beta"),
        "/c.js": marker("gamma"),
    }
    server, site = _start_site(routes)
    try:
        seen_auth: list[str] = []

        def authorize(url: str) -> None:
            seen_auth.append(url)

        async def run() -> browser.RenderedPage:
            client = _real_client(server.server_address[1], authorize)
            renderer = browser.PageRenderer(client, authorize=authorize, limits=WebLimits())
            try:
                return await renderer.render("http://page.test/")
            finally:
                await client.aclose()

        page = asyncio.run(run())
        assert "alpha" in page.text and "beta" in page.text and "gamma" in page.text
        assert page.requests_used == 4  # 主文档 + 3 子资源
        assert page.blocked_ask == 0 and page.blocked_deny == 0
        assert page.budget_stopped is False
        # 主文档过两次同一 authorize(客户端初始抓取 + route 闸,同一 fold 幂等)。
        assert seen_auth == [
            "http://page.test/",
            "http://page.test/",
            "http://page.test/a.js",
            "http://page.test/b.js",
            "http://page.test/c.js",
        ]
        assert sorted(site["seen"]) == ["/", "/a.js", "/b.js", "/c.js"]
    finally:
        server.shutdown()
        server.server_close()


@requires_chromium
def test_real_chromium_hostile_redirect_subresource_aborted_not_followed() -> None:
    html = b"<html><body><p>safe</p><script src='/redirect.js'></script></body></html>"
    routes = {
        "/": (html, 200, {"Content-Type": "text/html; charset=utf-8"}),
        "/redirect.js": (b"", 302, {"Location": "http://evil.test/steal.js"}),
        "/steal.js": (b"document.title = 'pwned';", 200, {"Content-Type": "application/javascript"}),
    }
    server, site = _start_site(routes)
    try:
        def authorize(url: str) -> None:
            if url.startswith("http://evil.test"):
                raise PublicWebError("denied", verdict="deny")

        async def run() -> browser.RenderedPage:
            client = _real_client(server.server_address[1], authorize)
            renderer = browser.PageRenderer(client, authorize=authorize, limits=WebLimits())
            try:
                return await renderer.render("http://page.test/")
            finally:
                await client.aclose()

        page = asyncio.run(run())
        assert page.blocked_deny == 1
        assert "/steal.js" not in site["seen"]  # 重定向目的地从未被读取
        assert "safe" in page.text
    finally:
        server.shutdown()
        server.server_close()


@requires_chromium
def test_real_chromium_process_env_is_exact_whitelist() -> None:
    """活体浏览器进程 env 键集合 == 常量白名单(ps -E 读真 env,CDP 取 pid)。"""

    class EnvProbeRenderer(browser.PageRenderer):
        env_keys: set[str] | None = None

        async def _discover_pgid(self, browser_obj: Any) -> int | None:
            pgid = await super()._discover_pgid(browser_obj)
            session = await browser_obj.new_browser_cdp_session()
            try:
                info = await session.send("SystemInfo.getProcessInfo")
            finally:
                await session.detach()
            pid = next(entry["pid"] for entry in info["processInfo"] if entry["type"] == "browser")
            output = subprocess.run(
                ["ps", "-E", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            # env 以 KEY=VALUE 追加在命令行之后;chromium 旗标一律 "-" 开头。
            self.env_keys = {
                token.split("=", 1)[0] for token in output.split() if "=" in token and not token.startswith("-")
            }
            return pgid

    routes = {"/": (b"<html><body><p>env probe</p></body></html>", 200, {"Content-Type": "text/html"})}
    server, _ = _start_site(routes)
    try:
        def authorize(url: str) -> None:
            return None

        async def run() -> EnvProbeRenderer:
            client = _real_client(server.server_address[1], authorize)
            renderer = EnvProbeRenderer(
                client,
                authorize=authorize,
                limits=WebLimits(),
                signal_process_group=lambda pgid, sig: True,  # 只启用 CDP 发现,不真发信号
            )
            try:
                page = await renderer.render("http://page.test/")
                assert "env probe" in page.text
                return renderer
            finally:
                await client.aclose()

        renderer = asyncio.run(run())
        assert renderer.env_keys == {"HOME", "TMPDIR", "PATH"}
    finally:
        server.shutdown()
        server.server_close()

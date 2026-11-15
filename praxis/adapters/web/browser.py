"""浏览器渲染路径(T14;S8 浏览器面闭合;Q5 裁决入 3B)。

**唯一授权面**:``context.route("**/*")`` 拦截浏览器发起的**每一个**请求,
URL 原样(不再归一化——T10/T11 纪律)先过注入的 ``authorize`` 回调
(composition 的 ``make_web_authorizer``,与 decide 共用 ``network_verdict``
同一 fold)。**ask 与 deny 同样 abort("blockedbyclient") 并分别计数**
``blocked_ask``/``blocked_deny``——旧 deny-only 子资源回调就此消亡
(legacy web_browser.py 只有 deny 拦,ask 被静默放行,正是 S8 第二病灶)。
brief 的 ``abort("blocked")`` 在 playwright 的真实枚举里是 ``"blockedbyclient"``
(Route.abort 文档逐字),语义 = 客户端主动拦截。

**网络路径**:route 拦截是唯一 sanctioned 网络路径——放行的请求由 T10
``PublicHttpClient`` 在进程内代抓(逐跳 authorize + DNS 固定 + 流式字节上限),
再 ``route.fulfill`` 喂给浏览器;chromium 自己**绝不触网**。启动参数
``proxy={"server": "http://127.0.0.1:9", "bypass": "<-loopback>"}`` 断直连
(legacy :300-303 纵深防御注释原样移植):任何漏过 route 的请求只会撞向死代理。
这是应用层请求中介,**不是 OS 网络命名空间隔离**(legacy 模块 docstring 同口径)。
子资源重定向由客户端逐跳解析后以最终 body fulfill 原始 URL——浏览器侧 URL/
origin 语义保持原始请求(legacy redirected_subresources 同口径,计数上报)。

**预算(逐调用,不是 turn 账本——Q7)**:``MAX_REQUESTS=64``、
``MAX_NETWORK_BYTES=8_000_000``(单响应同 8MB 上限)、``MAX_DOM_BYTES=2_000_000``、
``RENDER_TIMEOUT_SECONDS=30``;等待策略 goto(domcontentloaded 5s)→
networkidle 10s → 1.5s 停顿,预算触顶提前退出。单响应上限的诚实机制:
playwright route 拿不到流式 body,故由客户端 ``fetch(max_bytes=…)`` 在流式
读取期强制(超限 mid-stream 抛 ``max_bytes_exceeded`` ⇒ abort 该路由并停止
本次调用剩余抓取,如实报 partial)。字节计数只记**已确认**的解码 body 字节
(T10 客户端不暴露 wire 字节——与旧 wire/decoded 双账本有意差异);重定向跳数
不单独计数(客户端不暴露跳数,``max_redirects`` 仍逐跳钳制)。

**进程 env = 固定常量白名单** ``{HOME, TMPDIR(→逐调用私有目录), PATH=/usr/bin:/bin}``
——宿主信任面第三姿态(3B 笔记 §1 / Q 表注记):比 ``child_env`` 更窄,不继承
宿主 env,故不是 B33 的第二宿主继承路径。对照 legacy :231-235 收窄(旧还有
TMP/TEMP/LANG,控制器裁决只留三键)。此白名单常量**只允许出现在本模块**
(CI 规则⑳模块清单)。浏览器 helper 起进程 ⇒ 工具层声明 ``EXECUTE_PROCESS``
(R-T14-2),decide 据此叠加命令网络姿态门控(D9)与沙箱可用性 fail-closed。
**诚实注记**:playwright 公开 API 没有给内部 driver/chromium launch 加
argv 前缀的接缝,故管道编译的 seatbelt ``argv_prefix`` 不施加于浏览器进程;
浏览器约束 = chromium 自身沙箱(``chromium_sandbox=True``)+ 上述常量 env +
死代理 + route 唯一网络路径。EXECUTE_PROCESS 声明提供的是 decide 级门控
(Linux 无沙箱后端 ⇒ sandbox_unavailable deny;任何域名过滤 ⇒ D9 deny)。

**清理**:finally 内 context.close → browser.close → playwright.stop
(legacy _close_renderer 同序;driver stop 按 playwright 契约回收浏览器子进程),
取消经 shield 不逃逸;残留进程兜底 = 注入的 ``ProcessSignaler``(composition
绑 ``signal_group``,全仓唯一 killpg 实现)对**已发现的异己进程组** SIGKILL——
pid 经公开 CDP ``SystemInfo.getProcessInfo`` 尽力发现(best-effort:拿不到
就不发射,绝不向自身进程组发信号)。

**取消(R-T14-1)**:async API ⇒ 工具级取消可传播进 await 点;清理 shielded
收敛。抽取仍在 ``asyncio.to_thread``(线程不可杀,自然结束于后台)。
"""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from praxis.adapters.web.client import PublicHttpClient, WebFetchResponse, authorize_callback
from praxis.adapters.web.extract import extract_text
from praxis.adapters.web.limits import WebLimits
from praxis.adapters.web.urls import PublicWebError, validate_public_url
from praxis.protocol.sandbox import ProcessSignaler

try:
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover - 可选 browser extra 缺失时的占位形状
    _PLAYWRIGHT_AVAILABLE = False

    class PlaywrightError(Exception):  # type: ignore[no-redef]
        """占位异常:playwright 缺失时 render 在触达 API 前已拒绝。"""

    class PlaywrightTimeoutError(PlaywrightError):  # type: ignore[no-redef]
        """占位超时异常(同上,永不被真实 raise)。"""

    async_playwright = None  # type: ignore[assignment]
else:
    _PLAYWRIGHT_AVAILABLE = True

__all__ = [
    "MAX_DOM_BYTES",
    "MAX_NETWORK_BYTES",
    "MAX_REQUESTS",
    "RENDER_TIMEOUT_SECONDS",
    "PageRenderer",
    "RenderedPage",
]

# brief 逐字常量(legacy web_browser.py:27-30)。
MAX_REQUESTS = 64
MAX_NETWORK_BYTES = 8_000_000
MAX_DOM_BYTES = 2_000_000
RENDER_TIMEOUT_SECONDS = 30

# 等待策略(legacy :391-400 的 cap;顺序按控制器裁决 goto→networkidle→settle)。
_DOMCONTENTLOADED_TIMEOUT_MS = 5_000
_NETWORKIDLE_TIMEOUT_MS = 10_000
_SETTLE_MS = 1_500
_LAUNCH_TIMEOUT_MS = 15_000

# 宿主信任面第三姿态:固定常量白名单,只此一份(CI 规则⑳模块清单含 browser.py)。
_BROWSER_PATH = "/usr/bin:/bin"
# Unhandled browser HTTP cannot use the host's proxy or a direct connection.
# This is defense in depth, not OS isolation.(legacy :300-303 注释原样)
_DEAD_PROXY = {"server": "http://127.0.0.1:9", "bypass": "<-loopback>"}

_RENDER_UNAVAILABLE_MESSAGE = (
    "Browser rendering is unavailable. Install the optional browser support: "
    "uv sync --extra browser; then uv run python -m playwright install chromium."
)


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """一次渲染调用的如实结果(text 已经 extract.py 抽取;计数含子资源)。"""

    text: str
    final_url: str
    requests_used: int
    bytes_used: int
    dom_bytes: int
    truncated: bool  # extract.py 200_000 字符上限(尾部不可续读恢复)
    blocked_ask: int
    blocked_deny: int
    blocked_other: int  # 非 GET / websocket 等结构性拦截
    budget_aborted: int  # 预算触顶后被拦的请求数
    budget_stopped: bool
    failed_resources: int  # 子资源抓取失败(网络/编码/超限等)
    redirected_subresources: int
    load_timeouts: tuple[str, ...]


class _RenderState:
    """逐调用可变计数与闸状态(单 event loop;锁内做预算检查+fetch)。"""

    def __init__(self, limits: WebLimits) -> None:
        self.max_requests = limits.browser_requests
        self.max_network_bytes = limits.browser_network_bytes
        self.requests_used = 0
        self.bytes_used = 0
        self.blocked_ask = 0
        self.blocked_deny = 0
        self.blocked_other = 0
        self.budget_aborted = 0
        self.budget_stopped = False
        self.failed_resources = 0
        self.redirected_subresources = 0
        self.load_timeouts: list[str] = []
        self.fatal: BaseException | None = None
        self.lock = asyncio.Lock()
        self.tasks: set[asyncio.Task[Any]] = set()

    def count_denied(self, error: PublicWebError) -> None:
        # verdict 是机器字段(urls.py),绝不按消息文本分支。
        if error.verdict == "ask":
            self.blocked_ask += 1
        else:
            self.blocked_deny += 1


class PageRenderer:
    """一次性 chromium 渲染器(async;每次 render 独立 context,用完即清)。

    ``client`` 与工具层共用同一 ``PublicHttpClient``(同一 authorize、同一
    WebLimits——组合根不变量);``authorize`` 是**同一个** T11 回调实例,
    route 期逐 URL 先裁决(计数),client.fetch 逐跳再裁决(同一 fold,幂等)。
    ``signal_process_group`` 由 composition 注入 ``signal_group``(可选;
    None ⇒ 不做 killpg 兜底,也不做 CDP 发现)。
    """

    def __init__(
        self,
        client: PublicHttpClient,
        *,
        authorize: authorize_callback,
        limits: WebLimits,
        signal_process_group: ProcessSignaler | None = None,
    ) -> None:
        self._client = client
        self._authorize = authorize
        self._limits = limits
        self._signal_process_group = signal_process_group

    async def render(self, url: str, *, initial: WebFetchResponse | None = None) -> RenderedPage:
        """渲染 ``url``;``initial`` 非 None ⇒ 种子化主文档(auto 路径,零第二次读取)。

        错误面(PublicWebError):denied(授权拒绝,verdict 携带 ask/deny)、
        http_error(主文档非 2xx)、render_unavailable(playwright/chromium 缺失)、
        render_timeout(30s 总期限)、render_budget_exceeded(DOM 超 2MB)、
        render_failed(playwright 中介失败)、web_content_empty(DOM 无可读文本)。
        """
        validate_public_url(url)
        if not _PLAYWRIGHT_AVAILABLE:
            raise PublicWebError("render_unavailable", _RENDER_UNAVAILABLE_MESSAGE)
        state = _RenderState(self._limits)
        deadline = asyncio.get_running_loop().time() + RENDER_TIMEOUT_SECONDS
        try:
            async with asyncio.timeout(RENDER_TIMEOUT_SECONDS):
                return await self._render(url, initial, state, deadline)
        except TimeoutError:
            raise PublicWebError(
                "render_timeout", "Browser rendering exceeded its 30-second deadline."
            ) from None

    async def _render(
        self,
        url: str,
        initial: WebFetchResponse | None,
        state: _RenderState,
        deadline: float,
    ) -> RenderedPage:
        if initial is None:
            initial = await self._client.fetch(
                url, timeout_seconds=min(self._limits.timeout_seconds, RENDER_TIMEOUT_SECONDS)
            )
        # 主文档计 1 次请求(种子化同样计入——legacy :468-470 预算口径);
        # 重定向跳数客户端不暴露(见模块 docstring)。
        state.requests_used += 1
        state.bytes_used += len(initial.body)
        if not 200 <= initial.status < 300:
            raise PublicWebError("http_error", f"The website returned HTTP status {initial.status}.")
        seed_url = initial.final_url

        def remaining_ms(cap: int) -> float:
            return max(1.0, min(cap, (deadline - asyncio.get_running_loop().time()) * 1000))

        driver: Any = None
        browser: Any = None
        context: Any = None
        pgid: int | None = None
        try:
            startup = asyncio.create_task(async_playwright().start())
            try:
                driver = await asyncio.shield(startup)
            except asyncio.CancelledError:
                # start() 返回前已拥有子进程;取消也必须拿到 handle 才能清理。
                driver = await startup
                raise
            with tempfile.TemporaryDirectory(prefix="praxis-web-browser-") as directory:
                # 固定常量白名单:不继承宿主 env(姿态见模块 docstring)。
                environment = {"HOME": directory, "TMPDIR": directory, "PATH": _BROWSER_PATH}
                try:
                    browser = await driver.chromium.launch(
                        headless=True,
                        chromium_sandbox=True,
                        timeout=_LAUNCH_TIMEOUT_MS,
                        env=environment,
                        proxy=dict(_DEAD_PROXY),
                    )
                    context = await browser.new_context(service_workers="block", accept_downloads=False)
                    page = await context.new_page()
                except PlaywrightError:
                    raise PublicWebError("render_unavailable", _RENDER_UNAVAILABLE_MESSAGE) from None
                pgid = await self._discover_pgid(browser)
                seed: WebFetchResponse | None = initial

                async def gate(route: Any) -> None:
                    nonlocal seed
                    task = asyncio.current_task()
                    if task is not None:
                        state.tasks.add(task)
                    try:
                        request = route.request
                        raw = request.url
                        try:
                            is_main = request.is_navigation_request() and request.frame == page.main_frame
                        except PlaywrightError:
                            # popup 的首个导航可能尚无 Frame(legacy 同注记):
                            # 按非主文档处理,URL 仍过同一授权面。
                            is_main = False
                        # S8:每个请求 URL 先过唯一授权面(原样串,零再归一化)。
                        try:
                            self._authorize(raw)
                        except PublicWebError as error:
                            state.count_denied(error)
                            if is_main and state.fatal is None:
                                state.fatal = error
                            await route.abort("blockedbyclient")
                            return
                        except Exception as error:
                            # authorize 可抛任意异常(T10 契约):fail-closed 拦截。
                            state.blocked_deny += 1
                            if is_main and state.fatal is None:
                                state.fatal = error
                            await route.abort("blockedbyclient")
                            return
                        if request.method != "GET":
                            state.blocked_other += 1
                            await route.abort("blockedbyclient")
                            return
                        async with state.lock:
                            if (
                                state.budget_stopped
                                or state.requests_used >= state.max_requests
                                or state.bytes_used >= state.max_network_bytes
                            ):
                                state.budget_stopped = True
                                state.budget_aborted += 1
                                await route.abort("blockedbyclient")
                                return
                            if raw == seed_url and seed is not None:
                                response = seed
                                seed = None
                            else:
                                state.requests_used += 1
                                try:
                                    response = await self._client.fetch(
                                        raw,
                                        max_bytes=min(
                                            MAX_NETWORK_BYTES, state.max_network_bytes - state.bytes_used
                                        ),
                                        timeout_seconds=max(0.01, deadline - asyncio.get_running_loop().time()),
                                    )
                                except PublicWebError as error:
                                    if error.code == "denied":
                                        state.count_denied(error)
                                        await route.abort("blockedbyclient")
                                    elif error.code == "max_bytes_exceeded":
                                        # 单响应/剩余预算超限:流式期已截断,停止剩余抓取。
                                        state.budget_stopped = True
                                        state.budget_aborted += 1
                                        await route.abort("failed")
                                    else:
                                        state.failed_resources += 1
                                        await route.abort("failed")
                                    return
                                state.bytes_used += len(response.body)
                                if response.final_url != raw:
                                    state.redirected_subresources += 1
                            # route.fulfill otherwise inserts permissive CORS headers
                            # for mocked cross-origin reads. An explicit empty value
                            # preserves the server's denial when ACAO was absent.
                            # (legacy :350-353 注释原样)
                            headers = {
                                "access-control-allow-origin": "",
                                "content-type": response.content_type or "application/octet-stream",
                                **response.headers,
                            }
                            await route.fulfill(status=response.status, headers=headers, body=response.body)
                    except PlaywrightError:
                        if state.fatal is None:
                            state.fatal = PublicWebError(
                                "render_failed", "Playwright request mediation failed."
                            )
                    finally:
                        if task is not None:
                            state.tasks.discard(task)

                async def block_socket(websocket: Any) -> None:
                    state.blocked_other += 1
                    await websocket.close()

                # context 级 route:覆盖主 page 与任何 popup(legacy 同口径)。
                await context.route("**/*", gate)
                await context.route_web_socket("**/*", block_socket)
                try:
                    await page.goto(
                        seed_url, wait_until="domcontentloaded", timeout=remaining_ms(_DOMCONTENTLOADED_TIMEOUT_MS)
                    )
                except PlaywrightTimeoutError:
                    state.load_timeouts.append("domcontentloaded")
                except PlaywrightError:
                    if state.fatal is not None:
                        # 根因是授权拒绝(fatal),goto 的 ERR_BLOCKED_BY_CLIENT 只是症状。
                        raise state.fatal from None
                    raise PublicWebError("render_failed", "Browser navigation failed.") from None
                if state.fatal is not None:
                    raise state.fatal
                if not state.budget_stopped:
                    try:
                        await page.wait_for_load_state(
                            "networkidle", timeout=remaining_ms(_NETWORKIDLE_TIMEOUT_MS)
                        )
                    except PlaywrightTimeoutError:
                        state.load_timeouts.append("networkidle")
                    await page.wait_for_timeout(min(_SETTLE_MS, remaining_ms(_SETTLE_MS)))
                content = (await page.content()).encode("utf-8")
                if len(content) > MAX_DOM_BYTES:
                    raise PublicWebError(
                        "render_budget_exceeded", "The rendered DOM exceeded its 2000000-byte budget."
                    )
                extracted = await asyncio.to_thread(extract_text, content, "text/html; charset=utf-8")
                return RenderedPage(
                    text=extracted.text,
                    final_url=page.url,
                    requests_used=state.requests_used,
                    bytes_used=state.bytes_used,
                    dom_bytes=len(content),
                    truncated=extracted.truncated,
                    blocked_ask=state.blocked_ask,
                    blocked_deny=state.blocked_deny,
                    blocked_other=state.blocked_other,
                    budget_aborted=state.budget_aborted,
                    budget_stopped=state.budget_stopped,
                    failed_resources=state.failed_resources,
                    redirected_subresources=state.redirected_subresources,
                    load_timeouts=tuple(state.load_timeouts),
                )
        finally:
            await self._cleanup(context, browser, driver, pgid, state)

    async def _discover_pgid(self, browser: Any) -> int | None:
        """尽力经公开 CDP 发现浏览器进程组(清理兜底用;失败 ⇒ None,不发射)。"""
        if self._signal_process_group is None:
            return None
        try:
            session = await browser.new_browser_cdp_session()
            try:
                info = await session.send("SystemInfo.getProcessInfo")
            finally:
                await session.detach()
            for entry in info.get("processInfo", []):
                if entry.get("type") == "browser" and isinstance(entry.get("pid"), int):
                    return os.getpgid(entry["pid"])
        except Exception:
            return None
        return None

    async def _cleanup(
        self, context: Any, browser: Any, driver: Any, pgid: int | None, state: _RenderState
    ) -> None:
        # Close via the owning API even when navigation, a route or its caller was
        # cancelled. Driver shutdown also reaps Playwright's browser child processes.
        # (legacy _close_renderer 注释原样)
        for task in list(state.tasks):
            task.cancel()
        if state.tasks:
            await asyncio.gather(*list(state.tasks), return_exceptions=True)

        async def close_all() -> None:
            for resource, method in ((context, "close"), (browser, "close"), (driver, "stop")):
                if resource is not None:
                    with suppress(Exception):
                        await getattr(resource, method)()

        cleanup = asyncio.create_task(close_all())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
        # 残留进程兜底:只杀已发现的**异己**进程组(自身组恒不杀)。
        if pgid is not None and self._signal_process_group is not None and pgid != os.getpgrp():
            self._signal_process_group(pgid, signal.SIGKILL)

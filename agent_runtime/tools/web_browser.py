"""Optional trusted-local Playwright rendering with brokered public HTTP reads.

Chromium keeps its own sandbox and uses a fresh context, without a user's profile.
This is application-level request mediation, not OS network namespace isolation.
"""

from __future__ import annotations

import asyncio
import importlib.util
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from agent_runtime.tools.web_http import (
    FetchedResponse,
    PublicWebClient,
    PublicWebError,
    public_origin_referer,
    validate_public_url,
)
from agent_runtime.tools.web_limits import WebLimits

MAX_REQUESTS = 64
MAX_NETWORK_BYTES = 8_000_000
MAX_DOM_BYTES = 2_000_000
RENDER_TIMEOUT_SECONDS = 30
_VISUAL_RESOURCE_TYPES = {"Image", "Font", "Media"}
_OPTIONAL_RESOURCE_TYPES = {"Script", "Stylesheet", "XHR", "Fetch"}
_BLOCKED_REASONS = {"method_not_get", "websocket", "popup", "download"}
_RESOURCE_BLOCK_ERRORS = {"invalid_url", "nonpublic_address", "response_too_large", "web_browser_budget_exceeded"}
_RESOURCE_NETWORK_ERRORS = {"network_error", "timeout", "http_error", "proxy_connection_failed"}
_VISUAL_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "svg", "webp", "avif", "woff", "woff2", "ttf", "otf",
                      "mp4", "webm", "mp3", "ogg", "m3u8", "other"}


def _error(code: str, message: str) -> PublicWebError:
    return PublicWebError(code, message)


@dataclass
class _Budget:
    requests: int = 0
    wire_bytes: int = 0
    decoded_bytes: int = 0
    max_requests: int = MAX_REQUESTS
    max_network_bytes: int = MAX_NETWORK_BYTES

    def on_request(self) -> None:
        if self.requests >= self.max_requests:
            raise _error("web_browser_budget_exceeded", "Browser rendering exceeded its HTTP request budget.")
        if max(self.wire_bytes, self.decoded_bytes) >= self.max_network_bytes:
            raise _error("web_browser_budget_exceeded", "Browser rendering exhausted its response byte budget.")
        self.requests += 1

    def on_bytes(self, wire_delta: int, decoded_delta: int) -> None:
        self.wire_bytes += wire_delta
        self.decoded_bytes += decoded_delta
        if max(self.wire_bytes, self.decoded_bytes) > self.max_network_bytes:
            raise _error("web_browser_budget_exceeded", "Browser rendering exceeded its response byte budget.")


class _Broker:
    def __init__(self, client: PublicWebClient, *, limits: WebLimits | None = None) -> None:
        self.client = client
        self.budget = (_Budget() if limits is None else _Budget(
            max_requests=limits.browser_requests, max_network_bytes=limits.browser_network_bytes,
        ))
        self.pending: dict[tuple[str, str, str | None], FetchedResponse] = {}
        self.skipped_resources: dict[str, int] = {}
        self.skipped_url_kinds: dict[str, int] = {}
        self.blocked_actions: dict[str, int] = {}
        self.resource_failures: dict[tuple[str, str], int] = {}
        self.failed_resources: list[dict[str, str]] = []
        self.failed_resources_omitted = 0
        self.redirected_subresources = 0
        self.unsupported_features: set[str] = set()

    def record_failure(self, *, url: str, method: str, resource_type: str, reason: str) -> None:
        # Paths identify failed resources; credentials, query values and fragments
        # are intentionally absent. No request body or headers are retained.
        from agent_runtime.core.runtime_diagnostics import redact_sensitive_text

        try:
            target = urlsplit(url)
            location = f"{target.scheme}://{target.hostname}{target.path}"
        except ValueError:
            location = "[invalid URL]"
        if len(self.failed_resources) >= 16:
            self.failed_resources_omitted += 1
            return
        self.failed_resources.append({
            "url": redact_sensitive_text(location)[:512], "method": method[:16],
            "resource_type": resource_type[:32], "reason": reason[:80],
        })

    @property
    def warning(self) -> str | None:
        parts = []
        if self.resource_failures:
            counts = ", ".join(f"{code}:{kind}={count}"
                               for (code, kind), count in sorted(self.resource_failures.items()))
            parts.append(f"Public resource reads failed ({counts}).")
        if self.blocked_actions:
            counts = ", ".join(f"{reason}={count}" for reason, count in sorted(self.blocked_actions.items()))
            parts.append(f"Browser actions were blocked ({counts}).")
        if self.redirected_subresources:
            parts.append("Subresource redirects were resolved by the HTTP broker; browser URL and origin semantics "
                         "remain those of the original request.")
        return " ".join(parts) or None

    def record_blocked(self, message: dict[str, Any]) -> None:
        reason = message.get('reason')
        if not isinstance(reason, str) or reason not in _BLOCKED_REASONS:
            raise _error('web_browser_protocol_error', 'The browser returned an invalid blocked-action diagnostic.')
        self.blocked_actions[reason] = self.blocked_actions.get(reason, 0) + 1
        url = message.get('url')
        if isinstance(url, str):
            self.record_failure(url=url, method=str(message.get('method', '')),
                                resource_type=str(message.get('resource_type', 'Other')), reason=reason)

    def diagnostics(self, *, dom_bytes: int | None = None, load_timeouts: list[str] | None = None) -> dict[str, Any]:
        return {
            'browser_mode': 'trusted_local_brokered', 'chromium_sandbox': True,
            'os_network_isolation': False, 'redirect_handling': 'http_broker',
            'redirected_subresources': self.redirected_subresources,
            'subresource_redirect_semantics': 'original_request_url_and_origin',
            'unsupported_features': sorted(self.unsupported_features),
            'requests': self.budget.requests, 'wire_bytes': self.budget.wire_bytes,
            'decoded_bytes': self.budget.decoded_bytes, 'dom_bytes': dom_bytes,
            'snapshot_available': dom_bytes is not None, 'load_timeouts': load_timeouts or [],
            'blocked_actions': dict(self.blocked_actions), 'skipped_resources': dict(self.skipped_resources),
            'resource_failures': {f'{code}:{kind}': count for (code, kind), count in self.resource_failures.items()},
            'failed_resources': list(self.failed_resources),
            'failed_resources_omitted': self.failed_resources_omitted,
            'limits': {'requests': self.budget.max_requests, 'network_bytes': self.budget.max_network_bytes,
                       'dom_bytes': MAX_DOM_BYTES, 'timeout_seconds': RENDER_TIMEOUT_SECONDS},
            'remaining': {'requests': max(0, self.budget.max_requests - self.budget.requests),
                          'network_bytes': max(0, self.budget.max_network_bytes - max(
                              self.budget.wire_bytes, self.budget.decoded_bytes))},
        }

    async def fetch(
        self, url: str, resource_type: str = "Document", referer: str | None = None, *, follow_redirects: bool = False,
    ) -> FetchedResponse:
        return await self.client.get(
            str(validate_public_url(url)), headers={"Referer": referer} if referer is not None else None,
            request_budget=self.budget.on_request, byte_budget=self.budget.on_bytes,
            max_response_bytes=MAX_NETWORK_BYTES if resource_type == "Script" else None,
            allow_redirects=follow_redirects, return_redirects=not follow_redirects,
        )

    def record_skip(self, message: dict[str, Any]) -> None:
        resource_type, kind = message.get("resource_type"), message.get("url_kind")
        if (not isinstance(resource_type, str) or resource_type not in _VISUAL_RESOURCE_TYPES
                or not isinstance(kind, str) or len(kind) > 32 or kind.count(":") != 1):
            raise _error("web_browser_protocol_error", "The browser returned an invalid resource diagnostic.")
        scheme, extension = kind.split(":")
        if scheme not in {"http", "https", "data", "blob", "file", "other"} or extension not in _VISUAL_EXTENSIONS:
            raise _error("web_browser_protocol_error", "The browser returned an invalid resource diagnostic.")
        self.skipped_resources[resource_type] = self.skipped_resources.get(resource_type, 0) + 1
        self.skipped_url_kinds[kind] = self.skipped_url_kinds.get(kind, 0) + 1

    async def reply(self, message: dict[str, Any], *, follow_redirects: bool = False) -> dict[str, Any]:
        url, method = message.get("url"), message.get("method")
        if not isinstance(url, str) or len(url) > 4096 or method != "GET":
            raise _error("web_browser_request_blocked", "Browser rendering permits bounded public HTTP GETs only.")
        resource_type = message.get("resource_type", "Document")
        if not isinstance(resource_type, str) or resource_type not in {
            'Document', 'Script', 'Stylesheet', 'XHR', 'Fetch', 'Other', 'Manifest', 'Ping', 'Preflight',
        }:
            raise _error("web_browser_protocol_error", "The browser returned an invalid resource type.")
        referer = message.get("referer")
        if referer is not None:
            if not isinstance(referer, str) or len(referer) > 4096:
                raise _error("web_browser_protocol_error", "The browser returned an invalid source origin.")
            try:
                source = validate_public_url(referer)
                referer = public_origin_referer(referer, source)
            except PublicWebError:
                raise _error("web_browser_protocol_error", "The browser returned an invalid source origin.") from None
        limit_class = "Script" if resource_type == "Script" else "Document"
        try:
            url = str(validate_public_url(url))
            if referer is not None:
                referer = public_origin_referer(referer, validate_public_url(url))
            response = self.pending.pop((url, limit_class, referer), None)
            if response is None:
                if follow_redirects:
                    response = await self.fetch(url, limit_class, referer, follow_redirects=True)
                else:
                    response = await self.fetch(url, limit_class, referer)
            if follow_redirects and response.url != url:
                if resource_type == "Document":
                    self.unsupported_features.add("redirected_document_navigation")
                    raise _error("web_browser_request_blocked", (
                        "Brokered rendering cannot preserve the origin of a redirected browser-initiated "
                        "document navigation. Read the final document URL explicitly."
                    ))
                self.redirected_subresources += 1
        except PublicWebError as error:
            self.record_failure(url=url, method=method, resource_type=resource_type, reason=error.code)
            # An already denied subresource cannot cross the boundary. Only
            # known HTTP/policy errors are recoverable; permission and API errors are fatal.
            if resource_type not in _OPTIONAL_RESOURCE_TYPES:
                raise
            if error.code in _RESOURCE_BLOCK_ERRORS:
                self.blocked_actions[error.code] = self.blocked_actions.get(error.code, 0) + 1
                return {"type": "abort"}
            if error.code not in _RESOURCE_NETWORK_ERRORS:
                raise
            key = (error.code, resource_type)
            self.resource_failures[key] = self.resource_failures.get(key, 0) + 1
            return {"type": "abort"}
        if 300 <= response.status_code < 400:
            # Single-hop inspection is used only by broker policy callers. The
            # renderer always resolves redirects here, never with route.fulfill(3xx).
            return {"type": "response", "status": response.status_code,
                    "headers": dict(response.response_headers or {})}
        return {
            "type": "response", "status": response.status_code,
            "headers": {"content-type": response.content_type or "application/octet-stream",
                        **(response.response_headers or {})},
            "body": response.body,
        }


def _browser_environment(directory: str) -> dict[str, str | float | bool]:
    # Browser processes receive no provider credentials, proxy credentials or
    # executable search paths from the caller. Playwright resolves its own binary.
    return {"HOME": directory, "TMPDIR": directory, "TMP": directory, "TEMP": directory,
            "PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8"}


def _visual_url_kind(url: str) -> str:
    try:
        target = urlsplit(url)
        scheme = target.scheme if target.scheme in {"http", "https", "data", "blob", "file"} else "other"
        extension = Path(target.path).suffix.lower().lstrip(".")
        return f"{scheme}:{extension if extension in _VISUAL_EXTENSIONS else 'other'}"
    except ValueError:
        return "other:other"


async def _close_renderer(context: Any, browser: Any, driver: Any) -> None:
    # Close via the owning API even when navigation, a route or its caller was
    # cancelled. Driver shutdown also reaps Playwright's browser child processes.
    for resource, method in [(context, "close"), (browser, "close"), (driver, "stop")]:
        if resource is not None:
            with suppress(Exception):
                await getattr(resource, method)()


async def _render_with_playwright(
    broker: _Broker, initial: FetchedResponse, *, timeout_seconds: float,
) -> FetchedResponse:
    from playwright import async_api as api

    driver = browser = context = None
    fatal: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    lock = asyncio.Lock()
    tasks: set[asyncio.Task[Any]] = set()

    def remaining_ms(cap: int, reserve: float = 1.0) -> float:
        return max(1.0, min(cap, (deadline - asyncio.get_running_loop().time() - reserve) * 1000))

    async def checked(awaitable: Any) -> Any:
        # Event callbacks cannot propagate permission/budget failures into goto.
        # Race the operation against the first fatal route failure instead.
        operation = asyncio.create_task(awaitable)
        try:
            done, _ = await asyncio.wait([operation, fatal], return_when=asyncio.FIRST_COMPLETED)
            if fatal in done:
                fatal.result()
            return await operation
        finally:
            if not operation.done():
                operation.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await operation

    try:
        startup = asyncio.create_task(api.async_playwright().start())
        try:
            driver = await asyncio.shield(startup)
        except asyncio.CancelledError:
            # start() owns a subprocess before returning its API handle. Let it
            # return that handle so finally can stop it, even on caller cancellation.
            driver = await startup
            raise
        with tempfile.TemporaryDirectory(prefix="praxis-web-browser-") as directory:
            try:
                browser = await driver.chromium.launch(
                    headless=True, chromium_sandbox=True, timeout=remaining_ms(15_000),
                    env=_browser_environment(directory),
                    # Unhandled browser HTTP cannot use the host's proxy or a
                    # direct connection. This is defense in depth, not OS isolation.
                    proxy={"server": "http://127.0.0.1:9", "bypass": "<-loopback>"},
                )
                context = await browser.new_context(service_workers="block", accept_downloads=False)
                page = await context.new_page()
            except api.Error:
                raise _error("web_browser_unavailable", (
                    "Playwright-managed Chromium could not start with its sandbox. "
                    "Install with: uv sync --extra browser; uv run python -m playwright install chromium."
                )) from None

            async def intercept(route: Any) -> None:
                current = asyncio.current_task()
                if current is not None:
                    tasks.add(current)
                request = route.request
                resource_type = {"xhr": "XHR"}.get(request.resource_type, request.resource_type.title())
                try:
                    try:
                        is_popup = request.frame.page != page
                    except api.Error:
                        # Playwright has no Frame yet for a popup's first
                        # navigation. Service workers are disabled in this context.
                        if not request.is_navigation_request():
                            raise
                        is_popup = True
                    if is_popup:
                        broker.record_blocked({"reason": "popup", "url": request.url,
                                               "method": request.method, "resource_type": resource_type})
                        await route.abort("blockedbyclient")
                        return
                    if request.method != "GET":
                        broker.record_blocked({"reason": "method_not_get", "url": request.url,
                                               "method": request.method, "resource_type": resource_type})
                        await route.abort("blockedbyclient")
                        return
                    if resource_type in _VISUAL_RESOURCE_TYPES:
                        broker.record_skip({"resource_type": resource_type,
                                            "url_kind": _visual_url_kind(request.url)})
                        await route.abort("blockedbyclient")
                        return
                    async with lock:
                        reply = await broker.reply({"url": request.url, "method": request.method,
                                                    "resource_type": resource_type,
                                                    "referer": request.headers.get("referer")},
                                                   follow_redirects=True)
                    if reply["type"] == "abort":
                        await route.abort("failed")
                    else:
                        # route.fulfill otherwise inserts permissive CORS headers
                        # for mocked cross-origin reads. An explicit empty value
                        # preserves the server's denial when ACAO was absent.
                        headers = {"access-control-allow-origin": "", **reply["headers"]}
                        await route.fulfill(status=reply["status"], headers=headers,
                                            body=reply.get("body", b""))
                except PublicWebError as error:
                    if not fatal.done():
                        fatal.set_exception(error)
                    with suppress(api.Error):
                        await route.abort("blockedbyclient")
                except api.Error:
                    if not fatal.done():
                        fatal.set_exception(_error("web_browser_failed", "Playwright request mediation failed."))
                finally:
                    if current is not None:
                        tasks.discard(current)

            async def block_socket(websocket: Any) -> None:
                broker.record_blocked({"reason": "websocket"})
                await websocket.close()

            async def close_popup(popup: Any) -> None:
                if popup != page:
                    broker.record_blocked({"reason": "popup"})
                    await popup.close()

            async def cancel_download(download: Any) -> None:
                broker.record_blocked({"reason": "download"})
                await download.cancel()

            context.on("page", close_popup)
            page.on("download", cancel_download)
            await context.route("**/*", intercept)
            await context.route_web_socket("**/*", block_socket)
            broker.pending[(initial.url, "Document", None)] = initial
            try:
                await checked(page.goto(initial.url, wait_until="commit", timeout=remaining_ms(20_000)))
            except api.TimeoutError:
                raise _error("web_browser_timeout", "Browser navigation exceeded its deadline.") from None
            load_timeouts: list[str] = []
            load_states: list[tuple[Literal["domcontentloaded", "networkidle"], int]] = [
                ("domcontentloaded", 5000), ("networkidle", 10_000),
            ]
            for state, cap in load_states:
                if state == "networkidle":
                    await checked(page.wait_for_timeout(min(1500, remaining_ms(1500))))
                try:
                    await checked(page.wait_for_load_state(state, timeout=remaining_ms(cap)))
                except api.TimeoutError:
                    load_timeouts.append(state)
            content = (await checked(page.content())).encode("utf-8")
            if len(content) > MAX_DOM_BYTES:
                raise _error("web_browser_budget_exceeded", "The rendered DOM exceeded its byte budget.")
            final_url = str(validate_public_url(page.url))
            warning = broker.warning
            if load_timeouts:
                warning = " ".join(part for part in (
                    warning, "Load-state deadlines reached: " + ", ".join(load_timeouts),
                ) if part)
            return FetchedResponse(
                final_url, "text/html; charset=utf-8", content, broker.budget.wire_bytes,
                initial.connection_mode, warning=warning,
                render_diagnostics=broker.diagnostics(dom_bytes=len(content), load_timeouts=load_timeouts),
                request_count=broker.budget.requests,
            )
    except api.Error:
        if fatal.done():
            fatal.result()
        raise _error("web_browser_failed", "Playwright could not produce a rendered page snapshot.") from None
    finally:
        for task in list(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*list(tasks), return_exceptions=True)
        cleanup = asyncio.create_task(_close_renderer(context, browser, driver))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
        if fatal.done():
            # Retrieve even when the enclosing operation was cancelled first.
            fatal.exception()


async def render_public_page(
    client: PublicWebClient, url: str, *, initial_response: FetchedResponse | None = None,
    limits: WebLimits | None = None,
) -> FetchedResponse:
    """Render with optional Playwright-managed Chromium and a fresh context.

    PublicWebClient owns all bounded GET reads. Chromium keeps its own sandbox;
    there is no OS IP/network namespace isolation or access to a user's profile.
    An initial response seeds the document without a second HTTP read.
    """
    validate_public_url(url)
    if importlib.util.find_spec("playwright") is None:
        raise _error("web_browser_unavailable", (
            "Install browser support with: uv sync --extra browser; "
            "uv run python -m playwright install chromium."
        ))
    broker = _Broker(client) if limits is None else _Broker(client, limits=limits)
    initial = initial_response

    def attach_facts(error: PublicWebError) -> PublicWebError:
        error.render_diagnostics = broker.diagnostics()
        if initial is not None and error.connection_mode == "unknown":
            error.connection_mode = initial.connection_mode
        return error

    try:
        deadline = asyncio.get_running_loop().time() + RENDER_TIMEOUT_SECONDS
        async with asyncio.timeout_at(deadline):
            if initial is None:
                initial = await broker.fetch(url, follow_redirects=True)
            else:
                validate_public_url(initial.url)
                for _ in range(initial.request_count):
                    broker.budget.on_request()
                broker.budget.on_bytes(initial.network_bytes, len(initial.body))
            if 300 <= initial.status_code < 400:
                destination = (initial.response_headers or {}).get("location")
                if destination is None:
                    raise _error("invalid_redirect", "The redirect has no valid destination.")
                initial = await broker.fetch(destination, follow_redirects=True)
            remaining = max(0.01, deadline - asyncio.get_running_loop().time() - 0.5)
            return await _render_with_playwright(broker, initial, timeout_seconds=remaining)
    except PublicWebError as error:
        attach_facts(error)
        raise
    except TimeoutError:
        raise attach_facts(_error("web_browser_timeout", "Browser rendering exceeded its overall deadline.")) from None
    except OSError:
        raise attach_facts(_error("web_browser_unavailable", "Playwright-managed Chromium could not launch.")) from None

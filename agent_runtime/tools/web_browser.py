"""Optional Chromium rendering with an OS-isolated worker and parent HTTP broker.

The worker has no IP networking. Every HTTP GET, including redirect hops, uses
the caller's PublicWebClient outside that sandbox. A trusted HTTP proxy still
owns upstream IP validation; rendering does not strengthen that proxy contract.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import importlib.util
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

if __name__ != "__main__":
    from agent_runtime.tools.web_http import FetchedResponse, PublicWebClient, PublicWebError, validate_public_url

MAX_REQUESTS = 64
MAX_NETWORK_BYTES = 8_000_000
MAX_DOM_BYTES = 2_000_000
RENDER_TIMEOUT_SECONDS = 30
_RESULT_LINE_LIMIT = 2_700_000
_RESPONSE_LINE_LIMIT = 10_700_000
_REQUEST_LINE_LIMIT = 8192
_VISUAL_RESOURCE_TYPES = {"Image", "Font", "Media"}
_OPTIONAL_RESOURCE_TYPES = {"Script", "Stylesheet", "XHR", "Fetch"}
_BLOCKED_REASONS = {"method_not_get", "websocket", "popup", "download"}
_RESOURCE_BLOCK_ERRORS = {"invalid_url", "nonpublic_address", "response_too_large", "web_browser_budget_exceeded"}
_RESOURCE_NETWORK_ERRORS = {"network_error", "timeout", "http_error", "proxy_connection_failed"}
_VISUAL_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "svg", "webp", "avif", "woff", "woff2", "ttf", "otf",
                      "mp4", "webm", "mp3", "ogg", "m3u8", "other"}
_SANDBOX_PROFILE = (
    '(version 1) (allow default) '
    '(deny network-outbound (remote ip "*:*")) '
    '(deny network-inbound (local ip "*:*"))'
)


def _error(code: str, message: str) -> PublicWebError:
    return PublicWebError(code, message)


@dataclass
class _Budget:
    requests: int = 0
    wire_bytes: int = 0
    decoded_bytes: int = 0

    def on_request(self) -> None:
        if self.requests >= MAX_REQUESTS or max(self.wire_bytes, self.decoded_bytes) > MAX_NETWORK_BYTES:
            raise _error("web_browser_budget_exceeded", "Browser rendering exceeded its HTTP request budget.")
        self.requests += 1

    def on_bytes(self, wire_delta: int, decoded_delta: int) -> None:
        self.wire_bytes += wire_delta
        self.decoded_bytes += decoded_delta
        if max(self.wire_bytes, self.decoded_bytes) > MAX_NETWORK_BYTES:
            raise _error("web_browser_budget_exceeded", "Browser rendering exceeded its response byte budget.")


class _Broker:
    def __init__(self, client: PublicWebClient) -> None:
        self.client = client
        self.budget = _Budget()
        self.pending: dict[tuple[str, str], FetchedResponse] = {}
        self.skipped_resources: dict[str, int] = {}
        self.skipped_url_kinds: dict[str, int] = {}
        self.blocked_actions: dict[str, int] = {}
        self.resource_failures: dict[tuple[str, str], int] = {}

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
        return " ".join(parts) or None

    def record_blocked(self, message: dict[str, Any]) -> None:
        reason = message.get('reason')
        if not isinstance(reason, str) or reason not in _BLOCKED_REASONS:
            raise _error('web_browser_protocol_error', 'The browser returned an invalid blocked-action diagnostic.')
        self.blocked_actions[reason] = self.blocked_actions.get(reason, 0) + 1

    def diagnostics(self, *, dom_bytes: int | None = None, load_timeouts: list[str] | None = None) -> dict[str, Any]:
        return {
            'requests': self.budget.requests, 'wire_bytes': self.budget.wire_bytes,
            'decoded_bytes': self.budget.decoded_bytes, 'dom_bytes': dom_bytes,
            'snapshot_available': dom_bytes is not None, 'load_timeouts': load_timeouts or [],
            'blocked_actions': dict(self.blocked_actions), 'skipped_resources': dict(self.skipped_resources),
            'resource_failures': {f'{code}:{kind}': count for (code, kind), count in self.resource_failures.items()},
        }

    async def fetch(self, url: str, resource_type: str = "Document") -> FetchedResponse:
        return await self.client.get(
            str(validate_public_url(url)), request_budget=self.budget.on_request, byte_budget=self.budget.on_bytes,
            max_response_bytes=MAX_NETWORK_BYTES if resource_type == "Script" else None,
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

    async def reply(self, message: dict[str, Any]) -> dict[str, Any]:
        url, method = message.get("url"), message.get("method")
        if not isinstance(url, str) or len(url) > 4096 or method != "GET":
            raise _error("web_browser_request_blocked", "Browser rendering permits bounded public HTTP GETs only.")
        resource_type = message.get("resource_type", "Document")
        if not isinstance(resource_type, str) or resource_type not in {
            'Document', 'Script', 'Stylesheet', 'XHR', 'Fetch', 'Other', 'Manifest', 'Ping', 'Preflight',
        }:
            raise _error("web_browser_protocol_error", "The browser returned an invalid resource type.")
        limit_class = "Script" if resource_type == "Script" else "Document"
        try:
            url = str(validate_public_url(url))
            response = self.pending.pop((url, limit_class), None)
            if response is None:
                response = await self.fetch(url, limit_class)
        except PublicWebError as error:
            # An already denied subresource cannot cross the boundary. Only
            # known HTTP/policy errors are recoverable; protocol/isolation are fatal.
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
        final_url = str(validate_public_url(response.url))
        if final_url != url:
            # Chromium must observe the redirect before receiving bytes, so the
            # final document/base URL and script origin remain correct.
            self.pending[(final_url, limit_class)] = response
            return {"type": "response", "status": 302, "headers": {"location": final_url}}
        return {
            "type": "response", "status": response.status_code,
            "headers": {"content-type": response.content_type or "application/octet-stream",
                        **(response.response_headers or {})},
            "body": base64.b64encode(response.body).decode("ascii"),
        }


def _worker_environment(directory: str) -> dict[str, str]:
    home = Path.home()
    cache = home / "Library/Caches/ms-playwright" if sys.platform == "darwin" else home / ".cache/ms-playwright"
    environment = {
        "HOME": directory, "TMPDIR": directory, "TMP": directory, "TEMP": directory,
        "PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8",
        "PLAYWRIGHT_BROWSERS_PATH": os.environ.get("PLAYWRIGHT_BROWSERS_PATH", str(cache)),
    }
    if sys.platform.startswith("linux"):
        environment["PRAXIS_PARENT_NETNS"] = str(os.stat("/proc/self/ns/net").st_ino)
    return environment


def _sandbox_command(command: list[str], directory: str) -> list[str]:
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
        return ["/usr/bin/sandbox-exec", "-p", _SANDBOX_PROFILE, *command]
    if sys.platform.startswith("linux"):
        bubblewrap = shutil.which("bwrap")
        if bubblewrap:
            return [
                bubblewrap, "--die-with-parent", "--unshare-net", "--unshare-pid", "--cap-drop", "ALL",
                "--ro-bind", "/", "/", "--bind", directory, directory,
                "--dev", "/dev", "--proc", "/proc", "--", *command,
            ]
    raise _error("web_browser_unavailable", "Browser rendering requires a supported OS network sandbox.")


def _decode_message(line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("type") not in {"fetch", "result", "error", "skipped", "blocked"}:
            raise ValueError
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise _error("web_browser_protocol_error", "The browser worker returned an invalid bounded message.") from None


async def _kill_worker(process: asyncio.subprocess.Process) -> None:
    # Playwright may detach its browser into another process group. Stop the
    # worker before enumerating descendants, then kill those groups too.
    descendants: set[int] = {process.pid}
    groups: set[int] = {process.pid}
    if process.returncode is None:
        try:
            os.kill(process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            pass
        try:
            listing = await asyncio.create_subprocess_exec(
                "/bin/ps", "-axo", "pid=,ppid=,pgid=", stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            output, _ = await listing.communicate()
            rows = [tuple(map(int, line.split())) for line in output.splitlines()]
            while True:
                children = {pid for pid, parent, _ in rows if parent in descendants}
                if children <= descendants:
                    break
                descendants.update(children)
            groups.update(group for pid, _, group in rows if pid in descendants)
        except (OSError, ValueError):
            pass
    for group in groups - {os.getpgrp()}:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for pid in descendants:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await process.wait()


async def _serve_worker(
    command: list[str], environment: dict[str, str], broker: _Broker, initial: FetchedResponse,
    *, timeout_seconds: float = RENDER_TIMEOUT_SECONDS,
) -> FetchedResponse:
    process = await asyncio.create_subprocess_exec(
        *command, env=environment, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, start_new_session=True, limit=_RESULT_LINE_LIMIT,
    )
    assert process.stdout is not None and process.stdin is not None
    broker.pending[(initial.url, "Document")] = initial
    try:
        start = {"type": "start", "url": initial.url, "timeout_seconds": timeout_seconds}
        process.stdin.write(json.dumps(start).encode() + b"\n")
        await process.stdin.drain()
        while True:
            try:
                line = await process.stdout.readline()
            except ValueError:
                raise _error("web_browser_protocol_error", "The browser worker exceeded its message limit.") from None
            if not line:
                raise _error("web_browser_unavailable", "The isolated browser worker could not start or exited early.")
            message = _decode_message(line)
            if message["type"] == "skipped":
                if len(line) > _REQUEST_LINE_LIMIT:
                    raise _error("web_browser_protocol_error", "The resource diagnostic exceeded its message limit.")
                broker.record_skip(message)
            elif message["type"] == "blocked":
                if len(line) > _REQUEST_LINE_LIMIT:
                    raise _error("web_browser_protocol_error", "The blocked-action message exceeded its limit.")
                broker.record_blocked(message)
            elif message["type"] == "fetch":
                if len(line) > _REQUEST_LINE_LIMIT:
                    raise _error("web_browser_protocol_error", "The browser request exceeded its message limit.")
                reply = await broker.reply(message)
                encoded = json.dumps(reply, separators=(",", ":")).encode() + b"\n"
                if len(encoded) > _RESPONSE_LINE_LIMIT:
                    raise _error("web_browser_budget_exceeded", "The browser response exceeded its message budget.")
                process.stdin.write(encoded)
                await process.stdin.drain()
            elif message["type"] == "error":
                code = message.get("code")
                if code not in {"web_browser_unavailable", "web_browser_timeout", "web_browser_request_blocked",
                                "web_browser_budget_exceeded", "web_browser_failed"}:
                    code = "web_browser_failed"
                raise _error(code, "The isolated browser could not produce a rendered public-page snapshot.")
            else:
                try:
                    final_url = str(validate_public_url(message["url"]))
                    body = base64.b64decode(message["body"], validate=True)
                except (KeyError, TypeError, ValueError, binascii.Error):
                    raise _error("web_browser_protocol_error", "The browser returned an invalid snapshot.") from None
                if len(body) > MAX_DOM_BYTES:
                    raise _error("web_browser_budget_exceeded", "The rendered DOM exceeded its byte budget.")
                timeouts = message.get('load_timeouts', [])
                if (not isinstance(timeouts, list) or len(timeouts) > 2
                        or any(state not in {'domcontentloaded', 'networkidle'} for state in timeouts)):
                    raise _error('web_browser_protocol_error', 'The browser returned invalid load-state diagnostics.')
                warning = broker.warning
                if timeouts:
                    warning = ' '.join(part for part in (
                        warning, 'Load-state deadlines reached: ' + ', '.join(timeouts),
                    ) if part)
                return FetchedResponse(
                    final_url, "text/html; charset=utf-8", body, broker.budget.wire_bytes, initial.connection_mode,
                    warning=warning,
                    render_diagnostics=broker.diagnostics(dom_bytes=len(body), load_timeouts=timeouts),
                )
    finally:
        cleanup = asyncio.create_task(_kill_worker(process))
        await asyncio.shield(cleanup)


async def render_public_page(client: PublicWebClient, url: str) -> FetchedResponse:
    """Render a text snapshot within 30 seconds, 64 GETs, 8 MB reads and 2 MB DOM.

    Image, font and media requests are skipped; script/style/API reads remain.
    Install the optional browser extra and its Chromium executable first.
    Unsupported isolation or a missing renderer is an explicit failure, with no
    static-HTTP fallback. Cancellation closes the worker and browser descendants.
    """
    validate_public_url(url)
    if sys.platform == "darwin":
        raise _error("web_browser_unavailable", (
            "Chromium sandbox cannot run inside macOS Seatbelt network isolation; browser rendering is unavailable."
        ))
    if importlib.util.find_spec("playwright") is None:
        raise _error("web_browser_unavailable", "Install the optional browser extra and Playwright Chromium.")
    broker: _Broker | None = None
    initial: FetchedResponse | None = None

    def attach_facts(error: PublicWebError) -> PublicWebError:
        if broker is not None:
            error.render_diagnostics = broker.diagnostics()
        if initial is not None and error.connection_mode == 'unknown':
            error.connection_mode = initial.connection_mode
        return error

    try:
        deadline = asyncio.get_running_loop().time() + RENDER_TIMEOUT_SECONDS
        async with asyncio.timeout_at(deadline):
            with tempfile.TemporaryDirectory(prefix="praxis-web-browser-") as directory:
                bootstrap = f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')"
                command = _sandbox_command([sys.executable, "-I", "-B", "-c", bootstrap], directory)
                broker = _Broker(client)
                initial = await broker.fetch(url)
                remaining = max(0.01, deadline - asyncio.get_running_loop().time() - 0.5)
                return await _serve_worker(command, _worker_environment(directory), broker, initial,
                                           timeout_seconds=remaining)
    except PublicWebError as error:
        attach_facts(error)
        raise
    except TimeoutError:
        raise attach_facts(_error("web_browser_timeout", "Browser rendering exceeded its overall deadline.")) from None
    except OSError:
        failure = _error("web_browser_unavailable", "The OS-isolated browser could not be launched.")
        raise attach_facts(failure) from None


class _WorkerError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _visual_url_kind(url: str) -> str:
    try:
        target = urlsplit(url)
        scheme = target.scheme if target.scheme in {"http", "https", "data", "blob", "file"} else "other"
        extension = Path(target.path).suffix.lower().lstrip(".")
        return f"{scheme}:{extension if extension in _VISUAL_EXTENSIONS else 'other'}"
    except ValueError:
        return "other:other"


def _worker_read(limit: int) -> dict[str, Any]:
    line = sys.stdin.buffer.readline(limit + 1)
    if not line.endswith(b"\n") or len(line) > limit:
        raise _WorkerError("web_browser_failed")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise _WorkerError("web_browser_failed")
    return value


def _worker_write(value: dict[str, Any]) -> None:
    line = json.dumps(value, separators=(",", ":")).encode() + b"\n"
    if len(line) > _RESULT_LINE_LIMIT:
        raise _WorkerError("web_browser_budget_exceeded")
    sys.stdout.buffer.write(line)
    sys.stdout.buffer.flush()


def _verify_network_denied() -> None:
    # Never treat a missing/broken sandbox as routing-only protection.
    linux = sys.platform.startswith("linux")
    if linux:
        parent_namespace = os.environ.get("PRAXIS_PARENT_NETNS")
        if parent_namespace is None or str(os.stat("/proc/self/ns/net").st_ino) == parent_namespace:
            raise _WorkerError("web_browser_unavailable")
    targets = [(socket.AF_INET, ("8.8.8.8", 443)),
               (socket.AF_INET6, ("2606:4700:4700::1111", 443))] if linux else [
        (socket.AF_INET, ("127.0.0.1", 9)), (socket.AF_INET6, ("::1", 9)),
    ]
    for family, target in targets:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.1)
                probe.connect(target)
        except PermissionError:
            continue
        except OSError as error:
            if linux and error.errno in {101, 113, 97}:
                continue
        raise _WorkerError("web_browser_unavailable")


async def _worker() -> None:
    _verify_network_denied()
    try:
        api = importlib.import_module("playwright.async_api")
    except ImportError:
        raise _WorkerError("web_browser_unavailable") from None
    start = _worker_read(_REQUEST_LINE_LIMIT)
    if start.get("type") != "start" or not isinstance(start.get("url"), str):
        raise _WorkerError("web_browser_failed")
    lock = asyncio.Lock()
    loop = asyncio.get_running_loop()
    async with api.async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True, chromium_sandbox=True, timeout=15_000)
        except api.Error:
            raise _WorkerError("web_browser_unavailable") from None
        try:
            try:
                context = await browser.new_context(service_workers="block", accept_downloads=False)
                page = await context.new_page()
            except api.Error:
                # macOS Seatbelt cannot nest Chromium's renderer sandbox inside
                # sandbox-exec. Startup failure must not degrade isolation.
                raise _WorkerError("web_browser_unavailable") from None

            session = await context.new_cdp_session(page)

            async def intercept_request(event: dict[str, Any]) -> None:
                request = event["request"]
                if request["method"] != "GET":
                    await session.send("Fetch.failRequest", {
                        "requestId": event["requestId"], "errorReason": "BlockedByClient",
                    })
                    _worker_write({"type": "blocked", "reason": "method_not_get"})
                    return
                resource_type = event.get("resourceType", "Other")
                if resource_type in _VISUAL_RESOURCE_TYPES:
                    _worker_write({"type": "skipped", "resource_type": resource_type,
                                   "url_kind": _visual_url_kind(request["url"])})
                    await session.send("Fetch.failRequest", {
                        "requestId": event["requestId"], "errorReason": "BlockedByClient",
                    })
                    return
                async with lock:
                    _worker_write({"type": "fetch", "method": request["method"], "url": request["url"],
                                   "resource_type": resource_type})
                    reply = await loop.run_in_executor(None, _worker_read, _RESPONSE_LINE_LIMIT)
                    if reply.get("type") == "abort":
                        await session.send("Fetch.failRequest", {
                            "requestId": event["requestId"], "errorReason": "Failed",
                        })
                        return
                    if reply.get("type") != "response":
                        raise _WorkerError("web_browser_failed")
                    await session.send("Fetch.fulfillRequest", {
                        "requestId": event["requestId"], "responseCode": reply["status"],
                        "responseHeaders": [{"name": key, "value": value}
                                            for key, value in reply["headers"].items()],
                        "body": reply.get("body", ""),
                    })

            async def close_popup(popup: Any) -> None:
                if popup != page:
                    await popup.close()
                    _worker_write({"type": "blocked", "reason": "popup"})

            async def cancel_download(download: Any) -> None:
                await download.cancel()
                _worker_write({"type": "blocked", "reason": "download"})

            async def block_socket(websocket: Any) -> None:
                await websocket.close()
                _worker_write({"type": "blocked", "reason": "websocket"})

            context.on("page", close_popup)
            page.on("download", cancel_download)
            # Playwright route() deliberately skips subsequent redirect URLs.
            # CDP Fetch pauses every hop, including our synthetic redirects.
            session.on("Fetch.requestPaused", intercept_request)
            await session.send("Fetch.enable", {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]})
            await context.route_web_socket("**/*", block_socket)
            timeout_seconds = start.get('timeout_seconds', RENDER_TIMEOUT_SECONDS)
            if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
                    or not 0 < timeout_seconds <= RENDER_TIMEOUT_SECONDS):
                raise _WorkerError('web_browser_failed')
            deadline = loop.time() + timeout_seconds
            load_timeouts = []

            def remaining_ms(cap: int, reserve: float = 1.0) -> float:
                return max(1.0, min(cap, (deadline - loop.time() - reserve) * 1000))

            try:
                # A committed main document is required. Subsequent load-state
                # deadlines describe the captured DOM, not a semantic page failure.
                await page.goto(start["url"], wait_until="commit", timeout=remaining_ms(20_000))
            except api.TimeoutError:
                raise _WorkerError("web_browser_timeout") from None
            for state, cap in [('domcontentloaded', 5000), ('networkidle', 10_000)]:
                if state == 'networkidle':
                    await page.wait_for_timeout(min(1500, remaining_ms(1500)))
                try:
                    await page.wait_for_load_state(state, timeout=remaining_ms(cap))
                except api.TimeoutError:
                    load_timeouts.append(state)
            try:
                async with asyncio.timeout(max(0.001, deadline - loop.time())):
                    content = (await page.content()).encode("utf-8")
                if len(content) > MAX_DOM_BYTES:
                    raise _WorkerError("web_browser_budget_exceeded")
                if not content or page.url == 'about:blank':
                    raise _WorkerError('web_browser_failed')
                _worker_write({"type": "result", "url": page.url, "body": base64.b64encode(content).decode("ascii"),
                               'load_timeouts': load_timeouts})
            except TimeoutError:
                raise _WorkerError('web_browser_timeout') from None
        finally:
            with suppress(Exception):
                await browser.close()


if __name__ == "__main__":
    try:
        asyncio.run(_worker())
    except _WorkerError as failure:
        _worker_write({"type": "error", "code": failure.code})
    except Exception:
        _worker_write({"type": "error", "code": "web_browser_failed"})

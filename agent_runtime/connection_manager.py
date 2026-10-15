"""Host-owned, loopback-only connection UI sharing the public MCP services."""
from __future__ import annotations

import asyncio
import json
import secrets
import threading
import webbrowser
from collections.abc import Callable, Coroutine
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_runtime import mcp_admin

CATALOG = (
    {"id": "github", "title": "GitHub", "description": "连接仓库、Issue 和 Pull Request",
     "detail": "使用本机 GitHub CLI 登录；首次连接请先运行 gh auth login。",
     "url": mcp_admin.GITHUB_READONLY_URL, "auth": "github",
     "documentation": "https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md"},
    {"id": "notion", "title": "Notion", "description": "搜索和读取工作区文档",
     "detail": "在 Notion 官方页面授权，选择允许访问的工作区。",
     "url": "https://mcp.notion.com/mcp", "auth": "oauth",
     "documentation": "https://developers.notion.com/guides/mcp/get-started-with-mcp"},
    {"id": "linear", "title": "Linear", "description": "查看团队项目与任务",
     "detail": "在 Linear 官方页面授权，默认申请读取权限。",
     "url": "https://mcp.linear.app/mcp", "auth": "oauth",
     "documentation": "https://linear.app/docs/mcp"},
)
_ASSETS = Path(__file__).with_name("connection_assets")
_MAX_BODY = 16_384
_MAX_JOBS = 64


class AddConnection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    preset: str | None = None
    name: str | None = None
    url: str | None = None
    auth: Literal["oauth", "environment", "none"] = "oauth"
    bearer_token_env_var: str | None = None
    read_only: bool = True
    oauth_client_id: str | None = None
    oauth_callback_port: int = Field(default=0, ge=0, le=65535)
    oauth_scopes: list[str] = Field(default_factory=list, max_length=32)
    oauth_allowed_domains: list[str] = Field(default_factory=list, max_length=16)

    def registration(self) -> tuple[str, dict[str, Any]]:
        if self.preset is not None:
            if self.model_fields_set - {"preset", "read_only"}:
                raise ValueError("Preset connection options cannot be overridden")
            preset = next((item for item in CATALOG if item["id"] == self.preset), None)
            if preset is None:
                raise ValueError("Unknown connection preset")
            return preset["id"], {
                "url": None if preset["id"] == "github" else preset["url"],
                "oauth": preset["auth"] == "oauth", "read_only": self.read_only,
                "oauth_scopes": ("read",) if preset["id"] == "linear" and self.read_only else (),
            }
        if not self.name or not self.url:
            raise ValueError("Custom connections require a name and URL")
        if self.auth == "environment" and not self.bearer_token_env_var:
            raise ValueError("Environment authentication requires a variable name")
        if self.auth != "environment" and self.bearer_token_env_var:
            raise ValueError("Choose one credential source")
        if self.auth != "oauth" and (
            self.oauth_client_id or self.oauth_scopes or self.oauth_allowed_domains or self.oauth_callback_port
        ):
            raise ValueError("OAuth options require OAuth authentication")
        return self.name, {
            "url": self.url, "read_only": self.read_only, "oauth": self.auth == "oauth",
            "bearer_token_env_var": self.bearer_token_env_var,
            "oauth_client_id": self.oauth_client_id, "oauth_scopes": tuple(self.oauth_scopes),
            "oauth_callback_port": self.oauth_callback_port,
            "oauth_allowed_domains": tuple(self.oauth_allowed_domains),
        }


class ConnectionServer(ThreadingHTTPServer):
    """Each launch grants one short-lived browser capability, never an MCP token."""

    daemon_threads = True

    def __init__(self, workspace: Path, *, port: int = 0) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.session_token = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active: set[str] = set()
        self._async_jobs: set[tuple[asyncio.AbstractEventLoop, asyncio.Task[Any]]] = set()
        self._closing = False
        super().__init__(("127.0.0.1", port), ConnectionHandler)
        self.origin = f"http://127.0.0.1:{self.server_port}"

    @property
    def browser_url(self) -> str:
        return f"{self.origin}/#{self.session_token}"

    def server_close(self) -> None:
        with self._lock:
            self._closing = True
            tasks = tuple(self._async_jobs)
        for loop, task in tasks:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # A completed loop can close between snapshot and cancellation.
        super().server_close()

    def run_async(self, operation: Coroutine[Any, Any, Any]) -> Any:
        async def owned() -> Any:
            loop, task = asyncio.get_running_loop(), asyncio.current_task()
            assert task is not None
            with self._lock:
                closing = self._closing
                if not closing:
                    self._async_jobs.add((loop, task))
            if closing:
                operation.close()
                raise asyncio.CancelledError
            try:
                return await operation
            finally:
                with self._lock:
                    self._async_jobs.discard((loop, task))

        return asyncio.run(owned())

    def connections(self) -> list[dict[str, Any]]:
        result = []
        for entry in mcp_admin.mcp_connections():
            visible = {key: entry[key] for key in ("name", "url", "enabled", "read_only", "transport") if key in entry}
            visible["auth"] = ("oauth" if entry.get("oauth") else "saved" if entry.get("keychain") else
                               "environment" if entry.get("bearer_token_env_var") else "none")
            result.append(visible)
        return result

    def start_job(self, name: str, operation: Callable[[dict[str, Any]], dict[str, Any]]) -> str:
        with self._lock:
            if self._closing or name in self._active or len(self._active) >= 4:
                raise ValueError("Connection operation already in progress")
            # Keep bounded completed history, while active jobs retain their identity.
            while len(self._jobs) >= _MAX_JOBS:
                completed = next((key for key, job in self._jobs.items() if job["status"] in {"done", "error"}), None)
                if completed is None:
                    raise ValueError("Connection operation limit reached")
                del self._jobs[completed]
            identifier = secrets.token_hex(16)
            job: dict[str, Any] = {"id": identifier, "name": name, "status": "running"}
            self._jobs[identifier] = job
            self._active.add(name)

        def run() -> None:
            try:
                result = operation(job)
                with self._lock:
                    job.update(status="done", result=result)
                    job.pop("authorization_url", None)
            except (Exception, asyncio.CancelledError):
                # Exceptions can contain remote response bodies or credentials.
                with self._lock:
                    job.update(status="error", message="连接操作失败，请检查服务地址、授权方式和当前权限。")
                    job.pop("authorization_url", None)
            finally:
                with self._lock:
                    self._active.discard(name)

        threading.Thread(target=run, name=f"connection-{name}", daemon=True).start()
        return identifier

    def check(self, name: str) -> dict[str, Any]:
        from agent_runtime.mcp_cli import check_mcp_connection
        result = self.run_async(check_mcp_connection(name, self.workspace))
        return {"connected": bool(result["connected"]), "tools": result["tools"],
                "diagnostics": [item.get("code", "connection_failed") for item in result.get("diagnostics", [])]}

    def login(self, name: str, job: dict[str, Any]) -> None:
        from agent_runtime.mcp_oauth import login_mcp_oauth

        async def redirect(url: str) -> None:
            with self._lock:
                job.update(status="authorization_required", authorization_url=url)

        self.run_async(login_mcp_oauth(name, workspace=self.workspace, redirect_handler=redirect))

    def add(self, name: str, arguments: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        mcp_admin.register_remote_mcp(name, workspace=self.workspace, **arguments)
        if arguments.get("oauth"):
            self.login(name, job)
        return self.check(name)

    def action(self, action: str, name: str, job: dict[str, Any]) -> dict[str, Any]:
        if action == "login":
            self.login(name, job)
        elif action == "remove":
            mcp_admin.remove_mcp_connection(name)
            return {"removed": True}
        return self.check(name)


class ConnectionHandler(BaseHTTPRequestHandler):
    server: ConnectionServer

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Request paths and callback details never enter terminal logs.

    def respond(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        body = json.dumps(payload, ensure_ascii=False).encode() if content_type == "application/json" else payload
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; object-src 'none'")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self, *, api: bool) -> bool:
        host = urlsplit(self.server.origin).netloc
        if self.headers.get("Host") != host:
            return False
        if self.headers.get("Origin") not in {None, self.server.origin}:
            return False
        if self.headers.get("Sec-Fetch-Site") not in {None, "none", "same-origin"}:
            return False
        return not api or secrets.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + self.server.session_token
        )

    def do_GET(self) -> None:
        api = self.path.startswith("/api/")
        if not self.authorized(api=api):
            self.respond(403, {"error": "此页面需要本次启动的本机连接管理会话。"})
            return
        assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                  "/style.css": ("style.css", "text/css")}
        if self.path in assets:
            asset, content_type = assets[self.path]
            self.respond(200, (_ASSETS / asset).read_bytes(), content_type)
        elif self.path == "/api/catalog":
            self.respond(200, CATALOG)
        elif self.path == "/api/connections":
            try:
                self.respond(200, self.server.connections())
            except (OSError, ValueError):
                self.respond(400, {"error": "无法读取个人连接配置，请检查配置格式与访问权限。"})
        elif self.path.startswith("/api/jobs/"):
            with self.server._lock:
                job = dict(self.server._jobs.get(self.path.removeprefix("/api/jobs/"), {}))
            self.respond(200 if job else 404, job or {"error": "操作记录不存在。"})
        else:
            self.respond(404, {"error": "页面不存在。"})

    def do_POST(self) -> None:
        if not self.authorized(api=True):
            self.respond(403, {"error": "操作需要本机连接管理会话。"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.respond(415, {"error": "请求需使用 JSON。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= _MAX_BODY or self.headers.get("Transfer-Encoding"):
                raise ValueError
            body = json.loads(self.rfile.read(length))
            if self.path == "/api/add":
                name, arguments = AddConnection.model_validate(body).registration()
                identifier = self.server.start_job(name, lambda job: self.server.add(name, arguments, job))
            elif self.path in {"/api/login", "/api/check", "/api/remove"}:
                if not isinstance(body, dict) or set(body) != {"name"} or not isinstance(body["name"], str):
                    raise ValueError
                name = body["name"]
                if not any(entry.get("name") == name for entry in mcp_admin.mcp_connections()):
                    raise ValueError
                action = self.path.removeprefix("/api/")
                identifier = self.server.start_job(name, lambda job: self.server.action(action, name, job))
            else:
                self.respond(404, {"error": "操作不存在。"})
                return
        except (ValueError, ValidationError, OSError):
            self.respond(400, {"error": "请检查输入；同一连接正在进行的操作需先完成。"})
            return
        self.respond(202, {"id": identifier})


def connections_command(
    workspace: Annotated[Path, typer.Option("--workspace", help="检查此项目的有效连接权限")] = Path("."),
    port: Annotated[int, typer.Option("--port", min=0, max=65535, help="本机端口；0 自动选择")] = 0,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="打开本机浏览器")] = True,
) -> None:
    """打开本机连接服务页面，选择服务、浏览器授权并检查连接。"""
    server = ConnectionServer(workspace, port=port)
    typer.echo("连接管理仅监听本机。按 Ctrl+C 关闭。")
    typer.echo(server.browser_url)
    if open_browser:
        webbrowser.open(server.browser_url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

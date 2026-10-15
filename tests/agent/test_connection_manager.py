from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
from typer.testing import CliRunner

from agent_runtime.connection_manager import ConnectionServer


@contextmanager
def running_server(tmp_path):
    server = ConnectionServer(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=server.origin, trust_env=False) as client:
            yield server, client
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def headers(server):
    return {"Authorization": f"Bearer {server.session_token}", "Origin": server.origin}


def wait_job(client, server, identifier):
    for _ in range(100):
        result = client.get(f"/api/jobs/{identifier}", headers=headers(server))
        assert result.status_code == 200
        data = result.json()
        if data["status"] in {"done", "error"}:
            return data
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_local_ui_has_no_embedded_capability_or_third_party_assets(tmp_path):
    with running_server(tmp_path) as (server, client):
        result = client.get("/")
        assert result.status_code == 200
        assert "连接服务" in result.text
        assert server.session_token not in result.text
        assert "frame-ancestors 'none'" in result.headers["content-security-policy"]
        assert result.headers["cache-control"] == "no-store"
        assert client.get("/app.js").status_code == 200
        assert client.get("/style.css").status_code == 200
        assert client.get("/../../pyproject.toml").status_code == 404


@pytest.mark.parametrize("route", ["/api/catalog", "/api/connections", "/api/jobs/unknown"])
def test_api_needs_local_session_capability(tmp_path, route):
    with running_server(tmp_path) as (server, client):
        assert client.get(route).status_code == 403
        assert client.get(route, headers={"Authorization": "Bearer wrong"}).status_code == 403
        assert client.get(route, headers={**headers(server), "Origin": "https://foreign.invalid"}).status_code == 403
        assert client.get(route, headers={**headers(server), "Host": "foreign.invalid"}).status_code == 403


def test_catalog_is_static_and_listing_never_returns_config_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_runtime.mcp_admin.mcp_connections", lambda: [{
        "name": "docs", "url": "https://docs.example/mcp", "enabled": True,
        "headers": {"Authorization": "secret-token"}, "oauth": True,
    }])
    with running_server(tmp_path) as (server, client):
        catalog = client.get("/api/catalog", headers=headers(server)).json()
        assert {item["id"] for item in catalog} == {"github", "notion", "linear"}
        result = client.get("/api/connections", headers=headers(server))
        assert "secret-token" not in result.text
        assert "Authorization" not in result.text
        assert result.json()[0]["auth"] == "oauth"


def test_add_custom_connection_reuses_host_registration_and_check(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(
        "agent_runtime.mcp_admin.register_remote_mcp", lambda name, **kwargs: observed.append((name, kwargs))
    )

    async def check(name, workspace):
        assert workspace == tmp_path
        return {"connected": True, "tools": ["mcp__docs__search"], "diagnostics": []}

    monkeypatch.setattr("agent_runtime.mcp_cli.check_mcp_connection", check)
    with running_server(tmp_path) as (server, client):
        response = client.post("/api/add", headers=headers(server), json={
            "name": "docs", "url": "https://docs.example/mcp", "auth": "environment",
            "bearer_token_env_var": "DOCS_TOKEN",
        })
        assert response.status_code == 202
        job = wait_job(client, server, response.json()["id"])
        assert job["status"] == "done"
        assert job["result"]["connected"]
        name, arguments = observed[0]
        assert name == "docs"
        assert arguments["workspace"] == tmp_path
        assert arguments["read_only"] is True
        assert arguments["bearer_token_env_var"] == "DOCS_TOKEN"


@pytest.mark.parametrize("body", [
    {"preset": "unknown"}, {"name": "docs", "url": "https://docs.example/mcp", "token": "secret"},
    {"name": "docs", "url": "https://docs.example/mcp", "read_only": "false"},
])
def test_add_rejects_ambiguous_or_secret_fields_before_mutation(tmp_path, monkeypatch, body):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not mutate")

    monkeypatch.setattr("agent_runtime.mcp_admin.register_remote_mcp", forbidden)
    with running_server(tmp_path) as (server, client):
        assert client.post("/api/add", headers=headers(server), json=body).status_code == 400
        assert client.post("/api/add", headers=headers(server), content=json.dumps(body)).status_code == 415


def test_job_error_does_not_return_secret_exception_messages(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Authorization: Bearer unrecognized-private-key")

    monkeypatch.setattr("agent_runtime.mcp_admin.register_remote_mcp", fail)
    with running_server(tmp_path) as (server, client):
        response = client.post("/api/add", headers=headers(server), json={"preset": "github"})
        job = wait_job(client, server, response.json()["id"])
        assert job["status"] == "error"
        assert "unrecognized-private-key" not in json.dumps(job)


def test_cli_exposes_local_connection_manager(tmp_path, monkeypatch):
    from agent_runtime.cli import agent_app

    result = CliRunner().invoke(agent_app, ["connections", "--help"])
    assert result.exit_code == 0
    observed = []

    class Server:
        browser_url = "http://127.0.0.1:4567/#test-session"

        def __init__(self, workspace, *, port):
            observed.append((workspace, port))

        def serve_forever(self, *, poll_interval):
            observed.append("served")

        def server_close(self):
            observed.append("closed")

    monkeypatch.setattr("agent_runtime.connection_manager.ConnectionServer", Server)
    monkeypatch.setattr("agent_runtime.connection_manager.webbrowser.open", lambda *_: pytest.fail("--no-open ignored"))
    result = CliRunner().invoke(agent_app, [
        "connections", "--no-open", "--workspace", str(tmp_path), "--port", "4567",
    ])
    assert result.exit_code == 0
    assert observed == [(tmp_path, 4567), "served", "closed"]
    assert "127.0.0.1:4567" in result.stdout


def test_oauth_job_exposes_authorization_link_then_shutdown_cancels_callback(tmp_path, monkeypatch):
    cancelled = threading.Event()
    started = threading.Event()
    monkeypatch.setattr("agent_runtime.mcp_admin.mcp_connections", lambda: [{"name": "docs", "oauth": True}])

    async def login(name, *, workspace, redirect_handler):
        assert name == "docs" and workspace == tmp_path
        await redirect_handler("https://auth.example/authorize?state=public-state")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr("agent_runtime.mcp_oauth.login_mcp_oauth", login)
    with running_server(tmp_path) as (server, client):
        response = client.post("/api/login", headers=headers(server), json={"name": "docs"})
        assert response.status_code == 202
        assert started.wait(2)
        state = client.get(f"/api/jobs/{response.json()['id']}", headers=headers(server)).json()
        assert state["status"] == "authorization_required"
        assert state["authorization_url"].startswith("https://auth.example/")
        # Duplicate logins or deletion cannot race credential persistence.
        assert client.post("/api/remove", headers=headers(server), json={"name": "docs"}).status_code == 400
    assert cancelled.wait(2)

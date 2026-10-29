from __future__ import annotations

import json
import stat

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from agent_runtime import Agent
from agent_runtime.cli import agent_app
from agent_runtime.web_config import resolve_web_configuration


def credential(tmp_path, name="search.key"):
    path = tmp_path / name
    path.write_text("opaque-search-credential-123456")
    path.chmod(0o600)
    return path


def test_sdk_uses_the_same_explicit_web_environment_as_cli(tmp_path, monkeypatch):
    key = credential(tmp_path)
    monkeypatch.setenv("PRAXIS_WEB_SEARCH_PROVIDER", "deepseek")
    monkeypatch.setenv("PRAXIS_WEB_SEARCH_KEY_FILE", str(key))
    agent = Agent(workspace_path=tmp_path / "workspace")
    assert agent.web_search_provider == "deepseek"
    assert agent.web_search_key_file == key


def test_configure_persists_one_provider_credential_pair_for_cli_and_sdk(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    key = credential(tmp_path)
    runner = CliRunner()
    configured = runner.invoke(agent_app, ["web", "configure", "--provider", "deepseek", "--key-file", str(key)])
    assert configured.exit_code == 0, configured.output
    assert "opaque-search-credential" not in configured.output
    agent = Agent(workspace_path=workspace)
    assert agent.web_search_provider == "deepseek" and agent.web_search_key_file == key
    status = runner.invoke(agent_app, ["web", "status", "--json"])
    assert status.exit_code == 0, status.output
    data = json.loads(status.output)
    assert data["provider"] == "deepseek" and data["acquisition"] == "provider"
    assert "opaque-search-credential" not in status.output


def test_hidden_configuration_input_stays_outside_workspace_and_is_private(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    value = "opaque-new-search-key-123456789"
    result = CliRunner().invoke(agent_app, ["web", "configure", "--provider", "deepseek"], input=value + "\n")
    assert result.exit_code == 0, result.output
    assert value not in result.output
    agent = Agent(workspace_path=workspace)
    key = agent.web_search_key_file
    assert key is not None and not key.is_relative_to(workspace)
    assert key.read_text().strip() == value
    assert stat.S_IMODE(key.stat().st_mode) == 0o600


def test_configure_rejects_workspace_credentials_without_persisting(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    key = credential(workspace)
    result = CliRunner().invoke(agent_app, ["web", "configure", "--provider", "deepseek", "--key-file", str(key)])
    assert result.exit_code != 0
    assert "outside the workspace" in result.output
    assert Agent(workspace_path=workspace).web_search_provider is None


def test_provider_override_never_borrows_a_different_providers_saved_credential(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    key = credential(tmp_path)
    configured = CliRunner().invoke(
        agent_app, ["web", "configure", "--provider", "tavily", "--key-file", str(key)],
    )
    assert configured.exit_code == 0, configured.output
    monkeypatch.setenv("PRAXIS_WEB_SEARCH_PROVIDER", "deepseek")
    monkeypatch.delenv("PRAXIS_WEB_SEARCH_KEY_FILE", raising=False)
    import pytest

    with pytest.raises(ValueError, match="key file"):
        Agent(workspace_path=workspace)


@pytest.mark.parametrize("command,arguments", [
    ("run", ["hello"]), ("chat", []), ("resume", ["turn_example"]), ("recover", ["turn_example"]),
])
def test_cli_explicit_provider_does_not_take_an_environment_credentials_pair(
    tmp_path, monkeypatch, command, arguments,
):
    monkeypatch.setenv("PRAXIS_WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("PRAXIS_WEB_SEARCH_KEY_FILE", str(credential(tmp_path, "tavily.key")))
    click_command = get_command(agent_app).commands[command]
    with click_command.make_context(command, arguments + ["--web-search-provider", "deepseek"]) as context:
        with pytest.raises(ValueError, match="key file"):
            resolve_web_configuration(context.params["web_search_provider"], context.params["web_search_key_file"])


def test_live_configuration_check_uses_real_search_and_fetch_adapters_without_a_model(tmp_path, monkeypatch):
    import httpx

    from agent_runtime.tools import web_http

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    key = credential(tmp_path)
    runner = CliRunner()
    assert runner.invoke(agent_app, [
        "web", "configure", "--provider", "deepseek", "--key-file", str(key),
    ]).exit_code == 0
    hits = []

    def service(request):
        hits.append(request)
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json={"title": "Documentation", "content": "# Real backend contract"})

    original = web_http.PublicWebClient
    monkeypatch.setattr(web_http, "PublicWebClient", lambda **options: original(
        **options, transport=httpx.MockTransport(service),
    ))
    result = runner.invoke(agent_app, ["web", "check", "--live"])
    assert result.exit_code == 0, result.output
    assert [request.url.path for request in hits] == ["/v1/tools/search", "/v1/tools/fetch"]
    assert all(request.headers["Authorization"] == "Bearer " + key.read_text() for request in hits)
    assert "opaque-search-credential" not in result.output
    hits.clear()
    offline = runner.invoke(agent_app, ["web", "check"])
    assert offline.exit_code == 0 and not hits

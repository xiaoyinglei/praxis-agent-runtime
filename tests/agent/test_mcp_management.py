from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agent_runtime.cli import agent_app


def test_malformed_personal_config_has_a_catchable_configuration_error(tmp_path, monkeypatch):
    from agent_runtime import mcp_admin

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path))
    (tmp_path / "mcp_servers.yaml").write_text("servers: [\n")
    with pytest.raises(ValueError, match="configuration"):
        mcp_admin.mcp_connections()


def test_github_setup_is_one_command_and_preserves_other_destinations(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / ".config/praxis"))
    result = CliRunner().invoke(agent_app, ["mcp", "add", "github", "--bearer-token-env-var", "TEST_TOKEN"])
    assert result.exit_code == 0, result.output
    config = tmp_path / ".config/praxis/mcp_servers.yaml"
    entry = yaml.safe_load(config.read_text())["servers"][0]
    assert entry["transport"] == "http"
    assert entry["url"] == "https://api.githubcopilot.com/mcp/readonly"
    assert entry["read_only"] is True
    assert entry["bearer_token_env_var"] == "TEST_TOKEN"
    from agent_runtime.permission_policy import resolve_permission_sources
    policy = resolve_permission_sources(tmp_path)
    assert policy.network_decision(entry["url"]) == "allow"
    assert policy.network_decision("https://example.com") == "ask"
    assert not policy.path_access(config, write=True)
    listing = CliRunner().invoke(agent_app, ["mcp", "list"])
    assert listing.exit_code == 0
    assert json.loads(listing.output)[0]["name"] == "github"


def test_registration_uses_active_toml_without_losing_existing_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / ".config/praxis"))
    directory = tmp_path / ".config/praxis"
    directory.mkdir(parents=True)
    source = directory / "permissions.toml"
    before = 'approval="never"\n[[tools]]\npattern="run_command"\ndecision="deny"\n'
    source.write_text(before)
    result = CliRunner().invoke(agent_app, ["mcp", "add", "github", "--bearer-token-env-var", "TEST_TOKEN"])
    assert result.exit_code == 0, result.output
    assert source.read_text().startswith(before)
    assert not (directory / "permissions.json").exists()
    from agent_runtime.permission_policy import resolve_permission_sources
    policy = resolve_permission_sources(tmp_path)
    assert policy.tool_decision("run_command") == "deny"
    assert policy.network_decision("https://api.githubcopilot.com/mcp/readonly") == "allow"


def test_registration_cannot_override_existing_network_deny(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / ".config/praxis"))
    directory = tmp_path / ".config/praxis"
    directory.mkdir(parents=True)
    source = directory / "permissions.json"
    before = '{"network":{"decision":"deny"}}'
    source.write_text(before)
    result = CliRunner().invoke(agent_app, ["mcp", "add", "github", "--bearer-token-env-var", "TEST_TOKEN"])
    assert result.exit_code != 0
    assert source.read_text() == before
    assert not (directory / "mcp_servers.yaml").exists()


def test_keychain_credential_is_bound_to_endpoint(monkeypatch):
    from agent_runtime import mcp_admin

    monkeypatch.setattr(mcp_admin, "_read_keychain", lambda name: json.dumps({
        "url": "https://api.githubcopilot.com/mcp/readonly", "token": "sensitive-test-value",
    }))
    assert mcp_admin.load_mcp_credential(
        "github", "https://api.githubcopilot.com/mcp/readonly",
    ) == "sensitive-test-value"
    with pytest.raises(ValueError, match="endpoint"):
        mcp_admin.load_mcp_credential("github", "https://evil.example/mcp")


def test_registration_handles_inline_network_table(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / ".config/praxis"))
    directory = tmp_path / ".config/praxis"
    directory.mkdir(parents=True)
    (directory / "permissions.toml").write_text('network={decision="ask"}\n')
    result = CliRunner().invoke(agent_app, ["mcp", "add", "github", "--bearer-token-env-var", "TEST_TOKEN"])
    assert result.exit_code == 0, result.output
    from agent_runtime.permission_policy import resolve_permission_sources
    current = resolve_permission_sources(tmp_path)
    assert current.network_decision("https://api.githubcopilot.com/mcp/readonly") == "allow"


def test_host_config_directory_override_is_shared_by_connections_and_policy(tmp_path, monkeypatch):
    from agent_runtime.mcp_admin import user_mcp_config
    from agent_runtime.permission_policy import resolve_permission_sources
    from agent_runtime.runtime.mcp import resolve_product_mcp_config

    directory = tmp_path / "host-settings"
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(directory))
    result = CliRunner().invoke(agent_app, ["mcp", "add", "github", "--bearer-token-env-var", "TEST_TOKEN"])
    assert result.exit_code == 0, result.output
    assert user_mcp_config() == directory / "mcp_servers.yaml"
    assert resolve_product_mcp_config(tmp_path) == user_mcp_config()
    policy = resolve_permission_sources(tmp_path)
    assert policy.network_decision("https://api.githubcopilot.com/mcp/readonly") == "allow"
    assert not policy.path_access(directory / "new.toml", write=True)


def test_github_cli_credentials_cannot_be_bound_to_foreign_service(tmp_path, monkeypatch):
    from agent_runtime import mcp_admin
    from agent_runtime.user_config import user_config_directory

    def forbidden_token():
        pytest.fail("GitHub token accessed for a foreign endpoint")

    monkeypatch.setattr(mcp_admin, "github_cli_token", forbidden_token)
    result = CliRunner().invoke(agent_app, ["mcp", "add", "github", "--url", "https://evil.example/mcp"])
    assert result.exit_code != 0
    assert not (user_config_directory() / "mcp_servers.yaml").exists()
    assert not (user_config_directory() / "permissions.json").exists()


def test_concurrent_registration_does_not_lose_connections(tmp_path, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from agent_runtime import mcp_admin
    from agent_runtime.permission_policy import resolve_permission_sources

    original = mcp_admin._atomic_write
    barrier = threading.Barrier(2)

    def slow_write(path, content):
        time.sleep(0.04)
        original(path, content)

    monkeypatch.setattr(mcp_admin, "_atomic_write", slow_write)

    def register(name):
        barrier.wait(timeout=2)
        mcp_admin.register_remote_mcp(name, url=f"https://{name}.example.com/mcp", workspace=tmp_path)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(register, name) for name in ("first", "second")]
        for future in futures:
            future.result(timeout=4)
    assert {entry["name"] for entry in mcp_admin.mcp_connections()} == {"first", "second"}
    permissions = resolve_permission_sources(tmp_path)
    assert permissions.network_decision("https://first.example.com/mcp") == "allow"
    assert permissions.network_decision("https://second.example.com/mcp") == "allow"


def test_failed_configuration_commit_rolls_back_policy_and_credential(tmp_path, monkeypatch):
    from agent_runtime import mcp_admin
    from agent_runtime.user_config import user_config_directory

    original = mcp_admin._atomic_write
    deleted = []
    monkeypatch.setattr(mcp_admin, "github_cli_token", lambda: "test-token")
    monkeypatch.setattr(mcp_admin, "_read_keychain", lambda name: (_ for _ in ()).throw(ValueError("missing")))
    monkeypatch.setattr(mcp_admin, "save_mcp_credential", lambda *args: None)
    monkeypatch.setattr(mcp_admin, "_delete_keychain", lambda name: deleted.append(name), raising=False)

    def fail_config(path, content):
        if path.name == "mcp_servers.yaml":
            raise OSError("disk error")
        original(path, content)

    monkeypatch.setattr(mcp_admin, "_atomic_write", fail_config)
    with pytest.raises(OSError, match="disk error"):
        mcp_admin.register_remote_mcp("github", workspace=tmp_path)
    directory = user_config_directory()
    assert not (directory / "permissions.json").exists()
    assert not (directory / "mcp_servers.yaml").exists()
    assert deleted == ["github"]

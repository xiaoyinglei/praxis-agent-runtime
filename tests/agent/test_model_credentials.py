from __future__ import annotations

import pytest


def test_saved_credentials_are_endpoint_bound_and_environment_wins(monkeypatch):
    from agent_runtime import model_credentials as credentials

    monkeypatch.setattr(credentials, "_PLATFORM", "linux")
    monkeypatch.delenv("EXAMPLE_API_KEY", raising=False)
    credentials.save_model_credential("EXAMPLE_API_KEY", "https://provider.example/v1", "private-model-key")
    assert (
        credentials.resolve_model_credential("EXAMPLE_API_KEY", "https://provider.example/v1/") == "private-model-key"
    )
    assert credentials.resolve_model_credential("EXAMPLE_API_KEY", "https://other.example/v1") is None
    monkeypatch.setenv("EXAMPLE_API_KEY", "explicit-environment-key")
    assert (
        credentials.resolve_model_credential("EXAMPLE_API_KEY", "https://provider.example/v1")
        == "explicit-environment-key"
    )
    path = credentials.credential_directory() / "models.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_unsafe_private_credential_file_is_rejected(monkeypatch):
    from agent_runtime import model_credentials as credentials

    monkeypatch.setattr(credentials, "_PLATFORM", "linux")
    credentials.save_model_credential("EXAMPLE_API_KEY", "https://provider.example/v1", "private-key")
    path = credentials.credential_directory() / "models.json"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        credentials.load_model_credential("EXAMPLE_API_KEY", "https://provider.example/v1")


def test_keychain_keeps_secret_out_of_process_arguments(monkeypatch):
    import subprocess

    from agent_runtime import model_credentials as credentials

    monkeypatch.setattr(credentials, "_PLATFORM", "darwin")
    secret = "keychain-private-key"
    stored = {}

    def run(argv, **kwargs):
        assert secret not in " ".join(argv)
        if argv[-1] == "-i":
            import shlex

            command = shlex.split(kwargs["input"])
            stored[command[command.index("-s") + 1]] = command[command.index("-w") + 1]
            return subprocess.CompletedProcess(argv, 0, "", "")
        service = argv[argv.index("-s") + 1]
        return subprocess.CompletedProcess(argv, 0 if service in stored else 44, stored.get(service, ""), "")

    monkeypatch.setattr(credentials.subprocess, "run", run)
    credentials.save_model_credential("EXAMPLE_API_KEY", "https://provider.example/v1", secret)
    assert credentials.load_model_credential("EXAMPLE_API_KEY", "https://provider.example/v1") == secret
    assert credentials.load_model_credential("EXAMPLE_API_KEY", "https://evil.example/v1") is None


def test_private_credentials_remain_denied_with_explicit_child_grant(tmp_path, monkeypatch):
    from agent_runtime import model_credentials as credentials
    from agent_runtime.permission_policy import (
        PathRule,
        PermissionPolicy,
        ResolvedPermissions,
        resolve_permission_sources,
    )

    path = credentials.credential_directory() / "models.json"
    policy = resolve_permission_sources(
        tmp_path, PermissionPolicy(profile="trusted-local", filesystem=(PathRule(path=str(path), access="write"),))
    )
    assert not policy.path_access(path)
    assert not policy.path_access(path, write=True)
    restored = ResolvedPermissions.model_validate(policy.model_dump(mode="json"))
    assert not restored.path_access(path)
    assert any(rule.path == str(path.parent) and rule.access == "deny" for rule in restored.sandbox_filesystem)


def test_credential_directory_symlink_is_rejected(tmp_path, monkeypatch):
    from agent_runtime import model_credentials as credentials

    monkeypatch.setattr(credentials, "_PLATFORM", "linux")
    directory = credentials.credential_directory()
    directory.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="private"):
        credentials.save_model_credential("EXAMPLE_API_KEY", "https://provider.example/v1", "private-key")
    assert not (outside / "models.json").exists()


@pytest.mark.anyio
async def test_saved_file_credential_is_denied_by_real_macos_sandbox(tmp_path, monkeypatch):
    import shlex
    import sys

    from agent_runtime import model_credentials as credentials
    from agent_runtime.permission_policy import PermissionPolicy, resolve_permission_sources
    from agent_runtime.tools.builtins.shell import create_run_command_tool
    from agent_runtime.workspace import open_workspace

    if sys.platform != "darwin":
        pytest.skip("actual macOS Seatbelt enforcement")
    monkeypatch.setattr(credentials, "_PLATFORM", "linux")
    secret = "sandbox-private-test-key"
    credentials.save_model_credential("EXAMPLE_API_KEY", "https://provider.example/v1", secret)
    path = credentials.credential_directory() / "models.json"
    workspace = open_workspace(tmp_path / "project", create=True)
    policy = resolve_permission_sources(workspace.root, PermissionPolicy(profile="trusted-local"))
    workspace.permission_provider = lambda: policy
    tool = create_run_command_tool(workspace)
    result = await tool.run(tool.validate_input({"command": "cat " + shlex.quote(str(path))}))
    assert result.exit_code != 0
    assert secret not in result.stdout
    assert "Operation not permitted" in result.stderr

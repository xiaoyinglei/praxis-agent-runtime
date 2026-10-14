from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_runtime.permission_policy import (
    NetworkPolicy,
    PathRule,
    PermissionPolicy,
    ResolvedPermissions,
    ToolRule,
    load_permission_policy,
)


def test_workspace_default_is_scoped_and_protected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    permissions = PermissionPolicy().resolve(workspace)
    assert permissions.path_access(workspace / "src/app.py", write=True)
    assert not permissions.path_access(tmp_path / "outside.py")
    assert not permissions.path_access(workspace / "../outside.py", write=True)
    for name in (".git", ".agents", ".codex", ".aws"):
        assert permissions.path_access(workspace / name / "config")
        assert not permissions.path_access(workspace / name / "config", write=True)
        assert not permissions.path_access(workspace / "nested" / name / "config", write=True)
    assert "protected" in permissions.path_reason(workspace / ".git/config", write=True)
    assert all(Path(rule.path).is_absolute() for rule in permissions.filesystem)


def test_specific_scopes_and_equal_specificity_deny(tmp_path: Path) -> None:
    readonly = tmp_path / "readonly"
    forbidden = readonly / "secret"
    permissions = PermissionPolicy(filesystem=(
        PathRule(path=str(readonly), access="read"),
        PathRule(path=str(forbidden), access="write"),
        PathRule(path=str(forbidden), access="deny"),
    )).resolve(tmp_path)
    assert permissions.path_access(readonly / "file")
    assert not permissions.path_access(readonly / "file", write=True)
    assert not permissions.path_access(forbidden / "file")
    assert not permissions.path_access(tmp_path.with_name(tmp_path.name + "-sibling"))


def test_symlink_cannot_escape_or_bypass_lexical_deny(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    (workspace / "secret").symlink_to(workspace / "public", target_is_directory=True)
    permissions = PermissionPolicy(filesystem=(PathRule(path="secret", access="deny"),)).resolve(workspace)
    assert not permissions.path_access(workspace / "escape/file")
    assert not permissions.path_access(workspace / "secret/file")
    assert "symlink" in permissions.path_reason(workspace / "escape/file")


def test_readonly_rejects_write_rules_and_trusted_local_reads(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="read-only"):
        PermissionPolicy(profile="read-only", filesystem=(PathRule(path="extra", access="write"),))
    readonly = PermissionPolicy(profile="read-only").resolve(tmp_path)
    assert readonly.path_access(tmp_path / "a")
    assert not readonly.path_access(tmp_path / "a", write=True)
    trusted = PermissionPolicy(profile="trusted-local").resolve(tmp_path)
    assert trusted.path_access(Path("/somewhere/outside"))
    assert not trusted.path_access(Path("/somewhere/outside"), write=True)
    assert trusted.path_access(tmp_path / "a", write=True)


def test_tool_decisions_are_order_independent(tmp_path: Path) -> None:
    permissions = PermissionPolicy(tools=(
        ToolRule(pattern="*", decision="allow"),
        ToolRule(pattern="web_*", decision="ask"),
        ToolRule(pattern="web_fetch", decision="deny"),
    )).resolve(tmp_path)
    assert permissions.tool_decision("read_file") == "allow"
    assert permissions.tool_decision("web_search") == "ask"
    assert permissions.tool_decision("web_fetch") == "deny"
    assert PermissionPolicy().resolve(tmp_path).tool_decision("read_file") is None


def test_network_host_boundaries_and_unknown_command_fail_closed(tmp_path: Path) -> None:
    permissions = PermissionPolicy(network=NetworkPolicy(
        decision="allow", allowed_domains=("example.com", "*.docs.example.com"),
        denied_domains=("private.docs.example.com",),
    )).resolve(tmp_path)
    assert permissions.network_decision("https://example.com/path") == "allow"
    assert permissions.network_decision("https://EXAMPLE.COM./path") == "allow"
    assert permissions.network_decision("https://public.docs.example.com") == "allow"
    assert permissions.network_decision("https://private.docs.example.com") == "deny"
    assert permissions.network_decision("https://docs.example.com") == "deny"
    assert permissions.network_decision("https://example.com.evil.test") == "deny"
    assert permissions.network_decision("file:///etc/passwd") == "deny"
    assert permissions.command_network_allowed() == "deny"
    assert PermissionPolicy().resolve(tmp_path).command_network_allowed() == "ask"


def test_managed_intersection_cannot_widen_paths_tools_or_network(tmp_path: Path) -> None:
    ceiling = PermissionPolicy(profile="read-only", tools=(ToolRule(pattern="execute_*", decision="deny"),),
                               network=NetworkPolicy(decision="deny")).resolve(tmp_path)
    local = PermissionPolicy(profile="trusted-local", network=NetworkPolicy(decision="allow")).resolve(tmp_path)
    effective = local.intersection(ceiling)
    assert not effective.path_access(tmp_path / "a", write=True)
    assert not effective.path_access(tmp_path.parent / "a")
    assert effective.tool_decision("execute_command") == "deny"
    assert effective.network_decision("https://example.com") == "deny"
    restored = ResolvedPermissions.model_validate_json(effective.model_dump_json())
    assert restored.revision == effective.revision
    assert not restored.path_access(tmp_path / "a", write=True)


def test_loading_is_strict_relative_to_file_and_explain_is_json(tmp_path: Path) -> None:
    config = tmp_path / "settings" / "permissions.toml"
    config.parent.mkdir()
    config.write_text('approval = "never"\n[[filesystem]]\npath = "../external"\naccess = "read"\n')
    permissions = load_permission_policy(config, tmp_path / "workspace")
    assert permissions.approval == "never"
    assert permissions.path_access(tmp_path / "external/file")
    assert not permissions.path_access(tmp_path / "external/file", write=True)
    assert str(config) in permissions.explain()["sources"]
    assert permissions.explain()["revision"] == permissions.revision
    assert permissions.revision == load_permission_policy(config, tmp_path / "workspace").revision
    config.write_text('typo = true\n')
    with pytest.raises(ValidationError):
        load_permission_policy(config, tmp_path / "workspace")
    config.with_suffix(".json").write_text('{"network": {"typo": true}}')
    with pytest.raises(ValidationError):
        load_permission_policy(config.with_suffix(".json"), tmp_path)


def test_models_are_frozen_and_strict(tmp_path: Path) -> None:
    policy = PermissionPolicy()
    with pytest.raises(ValidationError):
        policy.profile = "trusted-local"
    with pytest.raises(ValidationError):
        PathRule(path=42, access="read")
    with pytest.raises(ValidationError):
        PermissionPolicy(unknown=True)
    with pytest.raises(ValidationError):
        NetworkPolicy(allowed_domains=("https://example.com",))


def test_protected_ancestor_is_not_an_implicit_outside_read_grant(tmp_path: Path) -> None:
    permissions = PermissionPolicy().resolve(tmp_path / "workspace")
    assert not permissions.path_access(tmp_path / ".aws/credentials")


def test_sandbox_projection_respects_every_boundary_and_ceiling(tmp_path: Path) -> None:
    readonly = tmp_path / "readonly"
    secret = readonly / "secret"
    local = PermissionPolicy(filesystem=(PathRule(path=str(secret), access="write"),)).resolve(tmp_path)
    ceiling = PermissionPolicy(filesystem=(PathRule(path=str(readonly), access="read"),
                                           PathRule(path=str(secret), access="deny"))).resolve(tmp_path)
    effective = local.intersection(ceiling)
    projected = {rule.path: rule.access for rule in effective.sandbox_filesystem}
    assert projected[str(tmp_path)] == "write"
    assert projected[str(readonly)] == "read"
    assert projected[str(secret)] == "deny"
    assert projected[str(tmp_path / ".git")] == "read"
    trusted = PermissionPolicy(profile="trusted-local").resolve(tmp_path).intersection(ceiling)
    assert {rule.path: rule.access for rule in trusted.sandbox_filesystem}["/"] == "deny"


def test_policy_revision_is_normalized_and_changes_with_authority() -> None:
    assert PermissionPolicy().revision == PermissionPolicy.model_validate({}).revision
    assert PermissionPolicy().revision != PermissionPolicy(approval="never").revision


def test_intersection_rejects_different_workspace_domains(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workspace"):
        PermissionPolicy().resolve(tmp_path).intersection(PermissionPolicy().resolve(tmp_path / "other"))


def test_automatic_policy_slots_are_protected_even_before_creation(tmp_path, monkeypatch):
    from agent_runtime.permission_policy import resolve_permission_sources

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / ".config/praxis"))
    directory = tmp_path / ".config/praxis"
    directory.mkdir(parents=True)
    config = directory / "permissions.json"
    config.write_text('{"tools":[{"pattern":"execute_python","decision":"deny"}]}')
    permissions = resolve_permission_sources(tmp_path)
    assert permissions.tool_decision("execute_python") == "deny"
    for slot in (directory, config, directory / "permissions.toml", directory / "new-file"):
        assert not permissions.path_access(slot, write=True)
    assert permissions.path_access(tmp_path / "ordinary-output", write=True)
    explicit = resolve_permission_sources(tmp_path, PermissionPolicy())
    assert not explicit.path_access(directory / "permissions.toml", write=True)
    assert not explicit.path_access(Path('/etc/praxis/permissions.toml'), write=True)


def test_repeated_policy_restore_remains_bounded_and_preserves_restrictions(tmp_path):
    current = PermissionPolicy().resolve(tmp_path)
    historical = PermissionPolicy(
        filesystem=(PathRule(path="secret", access="deny"),), approval="never",
    ).resolve(tmp_path)
    restored = current.intersection(historical)
    revision = restored.revision
    for _ in range(250):
        restored = current.intersection(ResolvedPermissions.model_validate_json(restored.model_dump_json()))
        assert not restored.path_access(tmp_path / "secret/data")
        assert restored.effective_approval == "never"
        assert restored.revision == revision
    assert len(restored.model_dump_json()) < 4_000

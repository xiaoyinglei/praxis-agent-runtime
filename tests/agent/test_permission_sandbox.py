from __future__ import annotations

import shlex
import shutil
import sys
from pathlib import Path

import pytest

from agent_runtime.permission_policy import NetworkPolicy, PathRule, PermissionPolicy, load_permission_policy
from agent_runtime.tools.builtins import shell
from agent_runtime.workspace import open_workspace


@pytest.fixture
def anyio_backend():
    return "asyncio"


def configured_workspace(tmp_path: Path, *, filesystem=(), profile="workspace", network=None):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    policy = PermissionPolicy(profile=profile, filesystem=filesystem,
                              network=network or NetworkPolicy()).resolve(workspace.root)
    workspace.permission_provider = lambda: policy
    return workspace


@pytest.mark.anyio
@pytest.mark.parametrize("managed_python", [False, True])
@pytest.mark.parametrize("network_policy", [NetworkPolicy(decision="deny"), NetworkPolicy(
    decision="allow", allowed_domains=("example.com",))])
async def test_filtered_network_command_fails_before_process_creation(
    tmp_path: Path, monkeypatch, managed_python: bool, network_policy: NetworkPolicy,
):
    workspace = configured_workspace(tmp_path, network=network_policy)

    async def unexpected_spawn(*args, **kwargs):
        pytest.fail("network policy denial must happen before subprocess execution")

    monkeypatch.setattr(shell.asyncio, "create_subprocess_exec", unexpected_spawn)
    if managed_python:
        result = await shell._execute_managed_python(workspace, shell.ManagedPythonInput(code="pass", network=True),
                                                     termination_grace_seconds=0.1)
    else:
        result = await shell._run_command(workspace, shell.RunCommandInput(command="true", network=True),
                                          termination_grace_seconds=0.1)
    assert result.sandbox_error == "sandbox_network_denied"


def test_mac_rules_represent_specific_overrides_and_protected_components(tmp_path: Path):
    workspace = configured_workspace(tmp_path, filesystem=(
        PathRule(path="secret", access="deny"), PathRule(path="secret/public", access="read"),))
    profile = shell._build_command_sandbox_profile(
        workspace_root=workspace.root, temporary_root=workspace.scratch / "temporary",
        allow_network=False, allow_workspace_write=True, permissions=workspace.permissions)
    assert "require-not" in profile
    assert str(workspace.root / "secret/public") in profile
    for component in ("agents", "codex", "aws"):
        assert component in profile


def test_linux_mounts_external_read_grant(tmp_path: Path):
    external = tmp_path / "external"
    secret = external / "secret"
    secret.mkdir(parents=True)
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="read"),))
    argv = shell._linux_sandbox_argv(
        workspace_root=workspace.root, temporary_root=workspace.scratch,
        additional_read_roots=(), permissions=workspace.permissions)
    assert ["--ro-bind", str(external), str(external)] == argv[argv.index(str(external)) - 1:][:3]


@pytest.mark.parametrize("directory", [False, True])
def test_linux_refuses_unrepresentable_deny_overlay(tmp_path: Path, directory: bool):
    external = tmp_path / "external"
    external.mkdir()
    secret = external / "secret"
    secret.mkdir() if directory else secret.write_text("secret")
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="read"),
                                                           PathRule(path=str(secret), access="deny")))
    with pytest.raises(ValueError, match="deny"):
        shell._linux_sandbox_argv(workspace_root=workspace.root, temporary_root=workspace.scratch,
                                  additional_read_roots=(), permissions=workspace.permissions)


def test_linux_rejects_broad_root_grant(tmp_path: Path):
    workspace = configured_workspace(tmp_path, profile="trusted-local")
    with pytest.raises(ValueError, match="root"):
        shell._linux_sandbox_argv(workspace_root=workspace.root, temporary_root=workspace.scratch,
                                  additional_read_roots=(), permissions=workspace.permissions)


def test_managed_python_can_declare_authorized_external_output(tmp_path: Path):
    external = tmp_path / "external"
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="write"),))
    usage = shell._resolve_managed_python_use(workspace, {
        "working_dir": ".", "workspace_write": True, "output_paths": (str(external / "result.txt"),)})
    assert any(target.value == str(external / "result.txt") for target in usage.targets)


real_mac_sandbox = pytest.mark.skipif(sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
                                    reason="requires macOS Seatbelt")


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_external_reads_and_specific_deny_override(tmp_path: Path):
    external = tmp_path / "external"
    secret = external / "secret"
    public = secret / "public"
    public.mkdir(parents=True)
    (external / "allowed.txt").write_text("allowed")
    (secret / "private.txt").write_text("must stay private")
    (public / "allowed.txt").write_text("exception")
    workspace = configured_workspace(tmp_path, filesystem=(
        PathRule(path=str(external), access="read"), PathRule(path=str(secret), access="deny"),
        PathRule(path=str(public), access="read"),))
    for path, allowed in ((external / "allowed.txt", True), (secret / "private.txt", False),
                          (public / "allowed.txt", True)):
        result = await shell._run_command(workspace, shell.RunCommandInput(command=f"cat {shlex.quote(str(path))}"),
                                          termination_grace_seconds=0.1)
        assert (result.exit_code == 0) is allowed, result.stderr


@real_mac_sandbox
@pytest.mark.anyio
@pytest.mark.parametrize("write_requested", [False, True])
async def test_real_external_write_requires_command_write_request(tmp_path: Path, write_requested: bool):
    external = tmp_path / "external"
    external.mkdir()
    target = external / "result.txt"
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="write"),))
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command=f"printf test > {shlex.quote(str(target))}", workspace_write=write_requested),
        termination_grace_seconds=0.1)
    assert (result.exit_code == 0) is write_requested, result.stderr
    assert target.exists() is write_requested


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_readonly_policy_and_intersection_override_write_request(tmp_path: Path):
    workspace = configured_workspace(tmp_path)
    ceiling = PermissionPolicy(profile="read-only").resolve(workspace.root)
    effective = workspace.permissions.intersection(ceiling)
    workspace.permission_provider = lambda: effective
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command="printf forbidden > new.txt", workspace_write=True), termination_grace_seconds=0.1)
    assert result.exit_code != 0
    assert not (workspace.root / "new.txt").exists()


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_managed_python_reads_external_and_respects_deny(tmp_path: Path):
    external = tmp_path / "external"
    external.mkdir()
    allowed = external / "allowed.txt"
    denied = external / "denied.txt"
    allowed.write_text("allowed")
    denied.write_text("secret")
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="read"),
                                                           PathRule(path=str(denied), access="deny")))
    for path, can_read in ((allowed, True), (denied, False)):
        result = await shell._execute_managed_python(workspace, shell.ManagedPythonInput(
            code=f"print(open({str(path)!r}).read())"), termination_grace_seconds=0.1)
        assert (result.exit_code == 0) is can_read, result.stderr


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_grant_does_not_allow_listing_parent_directory(tmp_path: Path):
    external = tmp_path / "external"
    external.mkdir()
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="read"),))
    result = await shell._run_command(workspace, shell.RunCommandInput(command=f"ls {shlex.quote(str(tmp_path))}"),
                                      termination_grace_seconds=0.1)
    assert result.exit_code != 0


@real_mac_sandbox
@pytest.mark.anyio
@pytest.mark.parametrize("metadata", [".git", ".agents", ".codex", ".aws"])
async def test_real_external_git_metadata_cannot_be_written(tmp_path: Path, metadata: str):
    external = tmp_path / "external"
    (external / "nested" / metadata).mkdir(parents=True)
    target = external / "nested" / metadata / "config"
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="write"),))
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command=f"printf forbidden > {shlex.quote(str(target))}", workspace_write=True),
        termination_grace_seconds=0.1)
    assert result.exit_code != 0
    assert not target.exists()


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_readonly_alias_cannot_write_an_allowed_target(tmp_path: Path):
    external = tmp_path / "external"
    external.mkdir()
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="read"),))
    target = workspace.root / "target.txt"
    target.write_text("original")
    alias = external / "alias.txt"
    alias.symlink_to(target)
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command=f"printf changed > {shlex.quote(str(alias))}", workspace_write=True),
        termination_grace_seconds=0.1)
    assert result.exit_code != 0
    assert target.read_text() == "original"


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_specific_write_exception_below_denied_directory(tmp_path: Path):
    workspace = configured_workspace(tmp_path, filesystem=(
        PathRule(path="secret", access="deny"), PathRule(path="secret/public", access="write")))
    (workspace.root / "secret/public").mkdir(parents=True)
    for relative, allowed in (("secret/forbidden.txt", False), ("secret/public/result.txt", True)):
        result = await shell._run_command(workspace, shell.RunCommandInput(
            command=f"printf test > {relative}", workspace_write=True), termination_grace_seconds=0.1)
        assert (result.exit_code == 0) is allowed, result.stderr
        assert (workspace.root / relative).exists() is allowed


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_managed_python_writes_authorized_external_output(tmp_path: Path):
    external = tmp_path / "external"
    external.mkdir()
    target = external / "result.txt"
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="write"),))
    result = await shell._execute_managed_python(workspace, shell.ManagedPythonInput(
        code=f"open({str(target)!r}, 'w').write('result')", workspace_write=True, output_paths=[str(target)]),
        termination_grace_seconds=0.1)
    assert result.exit_code == 0, result.stderr
    assert target.read_text() == "result"


@real_mac_sandbox
@pytest.mark.anyio
@pytest.mark.parametrize("command", [
    "printf changed > permissions.json",
    "ln permissions.json policy-alias.json && printf changed > policy-alias.json",
])
async def test_real_loaded_policy_cannot_rewrite_its_source(tmp_path: Path, command: str):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    source = workspace.root / "permissions.json"
    source.write_text("{}")
    permissions = load_permission_policy(source, workspace.root)
    workspace.permission_provider = lambda: permissions
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command=command, workspace_write=True), termination_grace_seconds=0.1)
    assert result.exit_code != 0
    assert source.read_text() == "{}"


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_broad_root_reads_work_but_write_request_fails_closed(tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    workspace = configured_workspace(tmp_path, profile="trusted-local")
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command=f"cat {shlex.quote(str(outside))}"), termination_grace_seconds=0.1)
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "outside"
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command="printf forbidden > output.txt", workspace_write=True), termination_grace_seconds=0.1)
    assert result.sandbox_error == "sandbox_policy_unsupported"
    assert not (workspace.root / "output.txt").exists()


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_case_alias_cannot_write_protected_codex_directory(tmp_path: Path):
    external = tmp_path / "external"
    (external / ".codex").mkdir(parents=True)
    alias = external / ".CODEX"
    if not alias.exists():
        pytest.skip("filesystem is case-sensitive")
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="write"),))
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command=f"printf forbidden > {shlex.quote(str(alias / 'config'))}", workspace_write=True),
        termination_grace_seconds=0.1)
    assert result.exit_code != 0
    assert not (external / ".codex/config").exists()


@real_mac_sandbox
@pytest.mark.anyio
@pytest.mark.parametrize("directory", [False, True])
async def test_real_case_alias_cannot_read_denied_path(tmp_path: Path, directory: bool):
    external = tmp_path / "external"
    external.mkdir()
    secret = external / "secret"
    if directory:
        secret.mkdir()
        target = secret / "content.txt"
        alias = external / "SECRET/content.txt"
    else:
        target = secret
        alias = external / "SECRET"
    target.write_text("private")
    if not alias.exists():
        pytest.skip("filesystem is case-sensitive")
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="read"),
                                                           PathRule(path=str(secret), access="deny")))
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command=f"cat {shlex.quote(str(alias))}"), termination_grace_seconds=0.1)
    assert result.exit_code != 0
    assert "private" not in result.stdout


@real_mac_sandbox
@pytest.mark.anyio
@pytest.mark.parametrize("metadata", [".git", ".agents", ".codex", ".aws"])
async def test_real_private_command_temp_can_create_metadata(tmp_path: Path, metadata: str):
    workspace = configured_workspace(tmp_path)
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command=f'mkdir -p "$TMPDIR/{metadata}" && touch "$TMPDIR/{metadata}/index"'),
        termination_grace_seconds=0.1)
    assert result.exit_code == 0, result.stderr


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_shared_scratch_metadata_is_protected_outside_command_temp(tmp_path: Path):
    workspace = configured_workspace(tmp_path)
    target = workspace.scratch / "shared/.git/index"
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command=f"mkdir -p {shlex.quote(str(target.parent))} && touch {shlex.quote(str(target))}",
        workspace_write=True), termination_grace_seconds=0.1)
    assert result.exit_code != 0
    assert not target.exists()


@real_mac_sandbox
@pytest.mark.anyio
async def test_real_external_hardlink_cannot_write_outside_authorized_root(tmp_path: Path):
    import os

    external = tmp_path / "external"
    external.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("original")
    alias = external / "alias.txt"
    os.link(target, alias)
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="write"),))
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command=f"printf changed > {shlex.quote(str(alias))}", workspace_write=True),
        termination_grace_seconds=0.1)
    assert result.exit_code != 0
    assert target.read_text() == "original"


@real_mac_sandbox
@pytest.mark.anyio
@pytest.mark.parametrize("automatic", [False, True])
async def test_real_policy_namespace_cannot_be_replaced_by_ancestor_rename(tmp_path, monkeypatch, automatic):
    from agent_runtime.permission_policy import resolve_permission_sources

    workspace = configured_workspace(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: workspace.root)
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(workspace.root / ".config/praxis"))
    source = workspace.root / ".config/praxis/permissions.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    policy = (resolve_permission_sources(workspace.root) if automatic else
              load_permission_policy(source, workspace.root))
    workspace.permission_provider = lambda: policy
    result = await shell._run_command(workspace, shell.RunCommandInput(
        command="mv .config renamed", workspace_write=True), termination_grace_seconds=0.1)
    assert result.exit_code != 0
    assert source.read_text() == "{}"
    assert not (workspace.root / "renamed").exists()
    ordinary = await shell._run_command(workspace, shell.RunCommandInput(
        command="printf ok > ordinary-output", workspace_write=True), termination_grace_seconds=0.1)
    assert ordinary.exit_code == 0, ordinary.stderr
    assert (workspace.root / "ordinary-output").read_text() == "ok"


def test_linux_write_grants_respect_revocation_and_external_boundaries(tmp_path: Path):
    external = tmp_path / "external"
    external.mkdir()
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="write"),))
    ceiling = PermissionPolicy(filesystem=(PathRule(path=str(external), access="read"),)).resolve(workspace.root)
    effective = workspace.permissions.intersection(ceiling)
    argv = shell._linux_sandbox_argv(
        workspace_root=workspace.root, temporary_root=workspace.scratch,
        additional_read_roots=(), allow_workspace_write=True, permissions=effective)
    mounts = [tuple(argv[i:i + 3]) for i, arg in enumerate(argv) if arg in {"--bind", "--ro-bind"}]
    assert ("--ro-bind", str(workspace.root), str(workspace.root)) in mounts
    assert ("--ro-bind", str(external), str(external)) in mounts
    assert ("--bind", str(external), str(external)) not in mounts


def test_linux_readonly_ceiling_overrides_explicit_write(tmp_path: Path):
    workspace = configured_workspace(tmp_path)
    effective = workspace.permissions.intersection(PermissionPolicy(profile="read-only").resolve(workspace.root))
    argv = shell._linux_sandbox_argv(
        workspace_root=workspace.root, temporary_root=workspace.scratch,
        additional_read_roots=(), allow_workspace_write=True, permissions=effective)
    assert ["--bind", str(workspace.root), str(workspace.root)] not in [argv[i:i + 3] for i in range(len(argv))]


def test_linux_candidate_publish_rechecks_revoked_permissions(tmp_path: Path):
    workspace = configured_workspace(tmp_path)
    before = workspace.permissions
    (workspace.root / "original.txt").write_text("before")
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "original.txt").write_text("after")
    workspace.permission_provider = lambda: PermissionPolicy(profile="read-only").resolve(workspace.root)
    with pytest.raises(ValueError, match="permission"):
        shell._commit_linux_workspace_view(workspace, view, initial, protected, before)
    assert (workspace.root / "original.txt").read_text() == "before"


def test_linux_external_write_request_fails_closed(tmp_path: Path):
    external = tmp_path / "external"
    external.mkdir()
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="write"),))
    with pytest.raises(ValueError, match="outside the workspace"):
        shell._linux_sandbox_argv(
            workspace_root=workspace.root, temporary_root=workspace.scratch,
            additional_read_roots=(), allow_workspace_write=True, permissions=workspace.permissions)


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("managed_python", [False, True])
async def test_real_linux_external_read_grant_survives_workspace_view(tmp_path: Path, managed_python: bool):
    external = tmp_path / "external"
    external.mkdir()
    (external / "data.txt").write_text("authorized")
    workspace = configured_workspace(tmp_path, filesystem=(PathRule(path=str(external), access="read"),))
    code = f'from pathlib import Path; Path("result.txt").write_text(Path({str(external / "data.txt")!r}).read_text())'
    if managed_python:
        result = await shell._execute_managed_python(
            workspace, shell.ManagedPythonInput(code=code, workspace_write=True, output_paths=["result.txt"]),
            termination_grace_seconds=0.1)
    else:
        result = await shell._run_command(
            workspace, shell.RunCommandInput(command="python3 -c " + shlex.quote(code), workspace_write=True),
            termination_grace_seconds=0.1)
    assert result.exit_code == 0, result.stderr
    assert (workspace.root / "result.txt").read_text() == "authorized"


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("managed_python", [False, True])
async def test_real_linux_readonly_ceiling_blocks_candidate_publication(tmp_path: Path, managed_python: bool):
    workspace = configured_workspace(tmp_path)
    policy = workspace.permissions.intersection(PermissionPolicy(profile="read-only").resolve(workspace.root))
    workspace.permission_provider = lambda: policy
    code = 'from pathlib import Path; Path("forbidden").write_text("forbidden")'
    if managed_python:
        result = await shell._execute_managed_python(
            workspace, shell.ManagedPythonInput(code=code, workspace_write=True, output_paths=["forbidden"]),
            termination_grace_seconds=0.1)
    else:
        result = await shell._run_command(
            workspace, shell.RunCommandInput(command="python3 -c " + shlex.quote(code), workspace_write=True),
            termination_grace_seconds=0.1)
    assert result.sandbox_error == "sandbox_publish_denied", result.stderr
    assert not (workspace.root / "forbidden").exists()

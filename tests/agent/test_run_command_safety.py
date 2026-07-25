from __future__ import annotations

import inspect
import os
import shlex
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from rag.agent.tools.builtins import shell as shell_module
from rag.agent.tools.builtins.shell import (
    RunCommandOutput,
    create_run_command_tool,
)
from rag.agent.tools.executor import ToolExecutor
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.tool import (
    JsonValue,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolEffect,
)
from rag.agent.workspace import WorkspaceRuntime, open_workspace


def _validated_arguments(
    tool: Tool,
    **overrides: JsonValue,
) -> Mapping[str, JsonValue]:
    arguments: dict[str, JsonValue] = {
        "command": "printf ok",
        "working_dir": ".",
        "timeout_seconds": 2,
    }
    arguments.update(overrides)
    return tool.validate_input(arguments)


async def _run_command(
    tool: Tool,
    arguments: Mapping[str, JsonValue],
) -> RunCommandOutput:
    pending = tool.run(arguments)
    if not inspect.isawaitable(pending):
        raise TypeError("run_command must return an awaitable")
    return RunCommandOutput.model_validate(await pending)


def _tool_call(
    *,
    tool_call_id: str = "call_run_command",
    **arguments: JsonValue,
) -> ToolCall:
    return ToolCall(
        tool_call_id=tool_call_id,
        tool_name="run_command",
        arguments=arguments,
        origin=ToolCallOrigin(
            request_id="req_run_command_safety",
            toolset_revision="tools_run_command_safety_v1",
            exposed_tool_names=("run_command",),
        ),
    )


def _initialize_git_repository(workspace: WorkspaceRuntime) -> Path:
    tracked = workspace.root / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    scratch_root = workspace.scratch.relative_to(workspace.root).parts[0]
    (workspace.root / ".gitignore").write_text(
        f"{scratch_root}/\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init", "--quiet", str(workspace.root)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace.root), "add", "tracked.txt", ".gitignore"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace.root),
            "-c",
            "user.name=Run Command Safety",
            "-c",
            "user.email=run-command-safety@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "baseline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tracked


def test_run_command_default_effects_are_read_only(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    tool = create_run_command_tool(workspace)

    resolved = tool.resolve_use(_validated_arguments(tool))

    assert resolved.effects == frozenset(
        {
            ToolEffect.READ_WORKSPACE,
            ToolEffect.EXECUTE_PROCESS,
        }
    )


def test_run_command_workspace_write_is_explicitly_destructive(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    tool = create_run_command_tool(workspace)

    resolved = tool.resolve_use(_validated_arguments(tool, workspace_write=True))

    assert resolved.effects == frozenset(
        {
            ToolEffect.READ_WORKSPACE,
            ToolEffect.WRITE_WORKSPACE,
            ToolEffect.EXECUTE_PROCESS,
            ToolEffect.DESTRUCTIVE,
        }
    )


def test_run_command_reports_workspace_and_cwd_scopes_separately(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    working_dir = workspace.root / "src"
    working_dir.mkdir()
    tool = create_run_command_tool(workspace)

    resolved = tool.resolve_use(
        _validated_arguments(
            tool,
            working_dir="src",
            workspace_write=True,
        )
    )
    targets = {target.kind: target.value for target in resolved.targets}

    assert targets["workspace_path"] == str(workspace.root)
    assert targets["cwd_path"] == str(working_dir)


def test_run_command_profile_protects_git_metadata_case_insensitively(
    tmp_path: Path,
) -> None:
    profile = shell_module._build_command_sandbox_profile(
        workspace_root=tmp_path / "workspace",
        temporary_root=tmp_path / "temporary",
        allow_network=False,
        allow_workspace_write=True,
    )

    assert 'regex #"/[.][gG][iI][tT]($|/)"' in profile


@pytest.mark.anyio
async def test_run_command_workspace_write_requires_separate_approval(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    sentinel = workspace.root / "must-not-run"
    tool = create_run_command_tool(workspace)
    call = _tool_call(
        command=f"touch {shlex.quote(str(sentinel))}",
        working_dir=".",
        timeout_seconds=2,
        workspace_write=True,
    )

    execution = await ToolExecutor({"run_command": tool}).execute(
        call,
        context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
            allow_write_tools=True,
            allow_execute_tools=True,
            auto_approve_sandboxed=True,
        ),
    )

    assert execution.result.error_code == "approval_required"
    assert "destructive operation" in (execution.result.error_message or "")
    assert execution.result.metadata["workspace_write"] is True
    assert execution.result.metadata["workspace_path"] == str(workspace.root)
    assert sentinel.exists() is False


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
@pytest.mark.parametrize("command", ["rm -rf victim", "git clean -fd"])
async def test_run_command_default_blocks_destructive_workspace_commands(
    tmp_path: Path,
    command: str,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    _initialize_git_repository(workspace)
    victim = workspace.root / "victim"
    victim.write_text("keep\n", encoding="utf-8")
    tool = create_run_command_tool(workspace)

    result = await _run_command(tool, _validated_arguments(tool, command=command))

    assert result.exit_code != 0
    assert victim.read_text(encoding="utf-8") == "keep\n"


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_default_blocks_ordinary_workspace_write(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(tool, command="touch denied"),
    )

    assert result.exit_code != 0
    assert (workspace.root / "denied").exists() is False


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_default_blocks_git_reset_hard(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    tracked = _initialize_git_repository(workspace)
    tracked.write_text("modified\n", encoding="utf-8")
    tool = create_run_command_tool(workspace)

    result = await _run_command(tool, _validated_arguments(tool, command="git reset --hard HEAD"))

    assert result.exit_code != 0
    assert tracked.read_text(encoding="utf-8") == "modified\n"


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_never_writes_dot_git(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    _initialize_git_repository(workspace)
    nested_git = workspace.root / "nested" / ".git"
    nested_git.mkdir(parents=True)
    allowed = workspace.root / "allowed"
    root_git_write = workspace.root / ".git" / "root-lock"
    nested_git_write = nested_git / "nested-lock"
    tool = create_run_command_tool(workspace)
    command = "; ".join(
        (
            f"touch {shlex.quote(str(allowed))}",
            f"touch {shlex.quote(str(root_git_write))}",
            f"touch {shlex.quote(str(nested_git_write))}",
        )
    )

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command=command,
            workspace_write=True,
        ),
    )

    assert result.exit_code != 0
    assert allowed.is_file()
    assert root_git_write.exists() is False
    assert nested_git_write.exists() is False


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_never_writes_dot_git_case_variant(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    metadata = workspace.root / ".GIT"
    metadata.mkdir()
    allowed = workspace.root / "allowed"
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="touch allowed; touch .GIT/probe",
            workspace_write=True,
        ),
    )

    assert result.exit_code != 0
    assert allowed.is_file()
    assert (metadata / "probe").exists() is False


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_rejects_dot_git_symlink(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    metadata_target = workspace.root / "metadata-target"
    metadata_target.mkdir()
    (workspace.root / ".git").symlink_to(
        metadata_target,
        target_is_directory=True,
    )
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="touch .git/probe",
            workspace_write=True,
        ),
    )

    assert result.sandbox_error == "workspace_git_alias"
    assert (metadata_target / "probe").exists() is False


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_preserves_worktree_git_file(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    git_file = workspace.root / ".git"
    git_file.write_text(
        "gitdir: /tmp/read-only-git-metadata\n",
        encoding="utf-8",
    )
    allowed = workspace.root / "allowed"
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="touch allowed; printf replaced > .git",
            workspace_write=True,
        ),
    )

    assert result.sandbox_error is None
    assert result.exit_code != 0
    assert allowed.is_file()
    assert git_file.read_text(encoding="utf-8") == ("gitdir: /tmp/read-only-git-metadata\n")


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
@pytest.mark.parametrize("metadata_path", [".git", "nested/.git"])
async def test_run_command_workspace_write_rejects_symlinks_inside_dot_git(
    tmp_path: Path,
    metadata_path: str,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    metadata = workspace.root / metadata_path
    metadata.mkdir(parents=True)
    target = workspace.root / "metadata-target"
    target.mkdir()
    (metadata / "alias").symlink_to(target, target_is_directory=True)
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command=f"touch {shlex.quote(metadata_path)}/alias/probe",
            workspace_write=True,
        ),
    )

    assert result.sandbox_error == "workspace_git_alias"
    assert (target / "probe").exists() is False


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_rejects_internal_gitdir_pointer(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    _initialize_git_repository(workspace)
    metadata = workspace.root / "metadata-target"
    (workspace.root / ".git").rename(metadata)
    (workspace.root / ".git").write_text(
        "gitdir: metadata-target\n",
        encoding="utf-8",
    )
    config = metadata / "config"
    original_config = config.read_text(encoding="utf-8")
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="git config review.probe changed",
            workspace_write=True,
        ),
    )

    assert result.sandbox_error == "workspace_git_alias"
    assert config.read_text(encoding="utf-8") == original_config


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_rejects_internal_commondir_pointer(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    metadata = workspace.root / ".git"
    metadata.mkdir()
    (workspace.root / "metadata-target").mkdir()
    (metadata / "commondir").write_text(
        "../metadata-target\n",
        encoding="utf-8",
    )
    sentinel = workspace.root / "must-not-run"
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="touch must-not-run",
            workspace_write=True,
        ),
    )

    assert result.sandbox_error == "workspace_git_alias"
    assert sentinel.exists() is False


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_rejects_retargetable_gitdir_pointer(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    _initialize_git_repository(workspace)
    replacement = workspace.root / "replacement-metadata"
    (workspace.root / ".git").rename(replacement)
    original_config = (replacement / "config").read_text(encoding="utf-8")
    outside_metadata = tmp_path / "outside-metadata"
    shutil.copytree(replacement, outside_metadata)
    metadata_link = workspace.root / "metadata-link"
    metadata_link.symlink_to(outside_metadata, target_is_directory=True)
    (workspace.root / ".git").write_text(
        "gitdir: metadata-link\n",
        encoding="utf-8",
    )
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command=("rm metadata-link && mv replacement-metadata metadata-link && git config review.probe changed"),
            workspace_write=True,
        ),
    )

    assert result.sandbox_error == "workspace_git_alias"
    assert (replacement / "config").read_text(encoding="utf-8") == (original_config)


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_cannot_create_dot_git_symlink(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    metadata_target = workspace.root / "metadata-target"
    metadata_target.mkdir()
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="ln -s metadata-target .git && touch .git/probe",
            workspace_write=True,
        ),
    )

    assert result.exit_code != 0
    assert (workspace.root / ".git").exists() is False
    assert (metadata_target / "probe").exists() is False


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_rejects_hardlinked_files(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-original\n", encoding="utf-8")
    os.link(outside, workspace.root / "inside-link.txt")
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="printf changed > inside-link.txt",
            workspace_write=True,
        ),
    )

    assert result.sandbox_error == "workspace_hardlink_detected"
    assert outside.read_text(encoding="utf-8") == "outside-original\n"


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_allows_internal_hardlinks(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    source = workspace.root / "source.txt"
    source.write_text("original\n", encoding="utf-8")
    os.link(source, workspace.root / "alias.txt")
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="printf changed > alias.txt",
            workspace_write=True,
        ),
    )

    assert result.exit_code == 0
    assert source.read_text(encoding="utf-8") == "changed"


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_rejects_dot_git_hardlink_alias(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    _initialize_git_repository(workspace)
    git_config = workspace.root / ".git" / "config"
    original_config = git_config.read_text(encoding="utf-8")
    os.link(git_config, workspace.root / "config-alias")
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="printf replaced > config-alias",
            workspace_write=True,
        ),
    )

    assert result.sandbox_error == "workspace_hardlink_detected"
    assert git_config.read_text(encoding="utf-8") == original_config


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_cannot_hardlink_dot_git(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    _initialize_git_repository(workspace)
    git_config = workspace.root / ".git" / "config"
    original_config = git_config.read_text(encoding="utf-8")
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command=("ln .git/config config-alias && printf replaced > config-alias"),
            workspace_write=True,
        ),
    )

    assert result.exit_code != 0
    assert (workspace.root / "config-alias").exists() is False
    assert git_config.read_text(encoding="utf-8") == original_config


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_default_rejects_hardlinked_files_before_read(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("outside-secret\n", encoding="utf-8")
    os.link(outside, workspace.root / "inside-link.txt")
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="cat inside-link.txt",
        ),
    )

    assert result.sandbox_error == "workspace_hardlink_detected"
    assert result.stdout == ""


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_default_rejects_hardlinks_inside_dot_git(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    metadata = workspace.root / ".git"
    metadata.mkdir()
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("outside-secret\n", encoding="utf-8")
    os.link(outside, metadata / "secret-alias")
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command="cat .git/secret-alias",
        ),
    )

    assert result.sandbox_error == "workspace_hardlink_detected"
    assert result.stdout == ""


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_workspace_write_cannot_escape_workspace(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    outside = tmp_path / "outside"
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command=f"touch {shlex.quote(str(outside))}",
            workspace_write=True,
        ),
    )

    assert result.exit_code != 0
    assert outside.exists() is False


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_private_temp_can_create_dot_git(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command=('mkdir -p "$TMPDIR/.git" && touch "$TMPDIR/.git/index"'),
        ),
    )

    assert result.exit_code == 0


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_private_temp_cannot_hardlink_workspace_file(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    source = workspace.root / "source.txt"
    source.write_text("original\n", encoding="utf-8")
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command=('ln source.txt "$TMPDIR/source-alias" && printf changed > "$TMPDIR/source-alias"'),
        ),
    )

    assert result.exit_code != 0
    assert source.read_text(encoding="utf-8") == "original\n"


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_default_allows_read_only_work_and_private_temp(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    source = workspace.root / "source.txt"
    source.write_text("readable\n", encoding="utf-8")
    tool = create_run_command_tool(workspace)

    result = await _run_command(
        tool,
        _validated_arguments(
            tool,
            command=('test "$(cat source.txt)" = readable && touch "$TMPDIR/cache-probe"'),
        ),
    )

    assert result.exit_code == 0

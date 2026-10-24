"""Shadow git checkpoint service: snapshot, restore, retention, isolation."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from agent_runtime.checkpoints import (
    CheckpointError,
    CheckpointService,
    CheckpointUnavailableError,
    checkpoint_root_for,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def service(workspace: Path, tmp_path: Path) -> CheckpointService:
    return CheckpointService(workspace, shadow_root=tmp_path / "shadow")


async def test_first_snapshot_captures_all_files(workspace: Path, service: CheckpointService) -> None:
    (workspace / "a.txt").write_text("one")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "b.py").write_text("print(2)\n")
    snapshot = await service.snapshot(trigger="pre_write", turn_id="turn_1")
    assert snapshot.created_new_commit is True
    assert snapshot.changed_file_count == 2
    assert snapshot.checkpoint_id
    assert snapshot.tree_sha256


async def test_unchanged_workspace_reuses_checkpoint_id(workspace: Path, service: CheckpointService) -> None:
    (workspace / "a.txt").write_text("one")
    first = await service.snapshot(trigger="pre_write", turn_id="turn_1")
    second = await service.snapshot(trigger="pre_write", turn_id="turn_2")
    assert second.created_new_commit is False
    assert second.checkpoint_id == first.checkpoint_id
    assert second.changed_file_count == 0


async def test_snapshot_excludes_ignored_directories(workspace: Path, service: CheckpointService) -> None:
    (workspace / "keep.txt").write_text("keep")
    (workspace / ".venv").mkdir()
    (workspace / ".venv" / "lib.py").write_text("hidden")
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "x.js").write_text("hidden")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("hidden")
    snapshot = await service.snapshot(trigger="pre_write", turn_id="turn_1")
    assert snapshot.changed_file_count == 1
    entries = await service.list_checkpoints()
    assert len(entries) == 1


async def test_snapshot_preserves_executable_bit_and_symlink(workspace: Path, service: CheckpointService) -> None:
    executable = workspace / "run.sh"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    link = workspace / "link.txt"
    link.symlink_to("keep.txt")
    (workspace / "keep.txt").write_text("target")
    await service.snapshot(trigger="pre_write", turn_id="turn_1")
    link.unlink()
    (workspace / "keep.txt").write_text("changed")
    executable.chmod(0o644)
    (workspace / "new.txt").write_text("added")
    await service.restore((await service.list_checkpoints())[0].checkpoint_id)
    assert (workspace / "keep.txt").read_text() == "target"
    assert (workspace / "new.txt").exists() is False
    assert link.is_symlink() and link.readlink().name == "keep.txt"
    assert (workspace / "run.sh").stat().st_mode & 0o111


async def test_restore_reverts_modified_and_deletes_added(workspace: Path, service: CheckpointService) -> None:
    (workspace / "a.txt").write_text("original")
    (workspace / "b.txt").write_text("original")
    target = (await service.snapshot(trigger="pre_write", turn_id="turn_1")).checkpoint_id
    (workspace / "a.txt").write_text("mutated")
    (workspace / "b.txt").unlink()
    (workspace / "c.txt").write_text("added")
    result = await service.restore(target)
    assert (workspace / "a.txt").read_text() == "original"
    assert (workspace / "b.txt").read_text() == "original"
    assert (workspace / "c.txt").exists() is False
    assert sorted(result.restored) == ["a.txt", "b.txt"]
    assert result.deleted == ("c.txt",)


async def test_restore_leaves_untracked_outside_snapshot_store_untouched(
    workspace: Path, service: CheckpointService,
) -> None:
    (workspace / "a.txt").write_text("one")
    target = (await service.snapshot(trigger="pre_write", turn_id="turn_1")).checkpoint_id
    (workspace / "a.txt").write_text("two")
    result = await service.restore(target)
    assert result.restored == ("a.txt",)
    # Files already absent from every checkpoint are never touched.
    assert result.deleted == ()


async def test_restore_is_itself_rewindingable(workspace: Path, service: CheckpointService) -> None:
    (workspace / "a.txt").write_text("v1")
    target = (await service.snapshot(trigger="pre_write", turn_id="turn_1")).checkpoint_id
    (workspace / "a.txt").write_text("v2")
    await service.restore(target)
    entries = await service.list_checkpoints()
    # The restore anchored the pre-restore state; rewinding to it restores v2.
    anchor = next(entry for entry in entries if "restore" in entry.subject)
    result = await service.restore(anchor.checkpoint_id)
    assert (workspace / "a.txt").read_text() == "v2"
    assert result.restored == ("a.txt",)


async def test_restore_unknown_checkpoint_rejected(workspace: Path, service: CheckpointService) -> None:
    (workspace / "a.txt").write_text("one")
    await service.snapshot(trigger="pre_write", turn_id="turn_1")
    with pytest.raises(CheckpointError):
        await service.restore("0" * 40)


async def test_restore_accepts_short_unique_prefix(workspace: Path, service: CheckpointService) -> None:
    (workspace / "a.txt").write_text("v1")
    target = (await service.snapshot(trigger="pre_write", turn_id="turn_1")).checkpoint_id
    (workspace / "a.txt").write_text("v2")
    await service.restore(target[:10])
    assert (workspace / "a.txt").read_text() == "v1"


async def test_shadow_repository_is_hardened_against_host_configuration(
    workspace: Path, service: CheckpointService, tmp_path: Path,
) -> None:
    # A hostile global git hook must not run during checkpoint commits.
    global_config = tmp_path / "gitconfig"
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text("#!/bin/sh\nexit 1\n")
    (hooks / "reference-transaction").write_text("#!/bin/sh\nexit 1\n")
    (hooks / "pre-commit").chmod(0o755)
    (hooks / "reference-transaction").chmod(0o755)
    global_config.write_text(f"[core]\n\thooksPath = {hooks}\n[commit]\n\tgpgsign = true\n")
    old = {"GIT_CONFIG_GLOBAL": os.environ.get("GIT_CONFIG_GLOBAL")}
    os.environ["GIT_CONFIG_GLOBAL"] = str(global_config)
    try:
        (workspace / "a.txt").write_text("one")
        snapshot = await service.snapshot(trigger="pre_write", turn_id="turn_1")
        assert snapshot.created_new_commit is True
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def test_workspace_git_metadata_never_written(workspace: Path, service: CheckpointService) -> None:
    (workspace / "a.txt").write_text("one")
    await service.snapshot(trigger="pre_write", turn_id="turn_1")
    (workspace / "a.txt").write_text("two")
    await service.restore((await service.list_checkpoints())[0].checkpoint_id)
    assert not (workspace / ".git").exists()


async def test_retention_prunes_expired_checkpoints(workspace: Path, service: CheckpointService) -> None:
    (workspace / "a.txt").write_text("v1")
    first = await service.snapshot(trigger="pre_write", turn_id="turn_1")
    (workspace / "a.txt").write_text("v2")
    second = await service.snapshot(trigger="pre_write", turn_id="turn_2")
    now = time.time() + 31 * 86400
    await service.prune(now=now)
    remaining = {entry.checkpoint_id for entry in await service.list_checkpoints()}
    assert first.checkpoint_id not in remaining
    assert second.checkpoint_id not in remaining


async def test_retention_keeps_fresh_checkpoints(workspace: Path, service: CheckpointService) -> None:
    (workspace / "a.txt").write_text("v1")
    await service.snapshot(trigger="pre_write", turn_id="turn_1")
    await service.prune(now=time.time())
    assert len(await service.list_checkpoints()) == 1


async def test_prune_removes_shadow_of_deleted_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "gone"
    workspace.mkdir()
    service = CheckpointService(workspace, shadow_root=tmp_path / "shadow-gone")
    (workspace / "a.txt").write_text("one")
    await service.snapshot(trigger="pre_write", turn_id="turn_1")
    assert service.shadow_root.exists()
    shutil.rmtree(workspace)
    await service.prune()
    assert service.shadow_root.exists() is False


async def test_missing_git_disables_service(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent_runtime.checkpoints.shutil.which", lambda _: None)
    service = CheckpointService(workspace, shadow_root=tmp_path / "shadow")
    (workspace / "a.txt").write_text("one")
    with pytest.raises(CheckpointUnavailableError):
        await service.snapshot(trigger="pre_write", turn_id="turn_1")


def test_checkpoint_root_is_under_host_config_and_workspace_keyed(tmp_path: Path) -> None:
    first = checkpoint_root_for(tmp_path / "one")
    second = checkpoint_root_for(tmp_path / "two")
    assert first != second
    assert first.parent.name == "checkpoints"


async def test_foreign_repository_without_marker_is_refused(workspace: Path, tmp_path: Path) -> None:
    shadow = tmp_path / "foreign"
    shadow.mkdir()
    subprocess.run(["git", "init", "--quiet", str(shadow)], check=True)
    (shadow / ".git" / "refs" / "checkpoints").mkdir(parents=True)
    (shadow / ".git" / "refs" / "checkpoints" / "abc").write_text("0" * 40)
    service = CheckpointService(workspace, shadow_root=shadow)
    (workspace / "a.txt").write_text("one")
    with pytest.raises(CheckpointUnavailableError):
        await service.snapshot(trigger="pre_write", turn_id="turn_1")


async def test_gitignored_files_are_snapshotted_and_restored(workspace: Path, service: CheckpointService) -> None:
    # The checkpoint domain must match workspace_tree_sha256: the workspace's
    # own .gitignore must not silently narrow the safety net.
    (workspace / ".gitignore").write_text("local.txt\n")
    (workspace / "local.txt").write_text("protected v1")
    (workspace / "code.py").write_text("print(1)")
    target = (await service.snapshot(trigger="pre_write", turn_id="turn_1")).checkpoint_id
    (workspace / "local.txt").write_text("modified")
    (workspace / "code.py").write_text("print(2)")
    result = await service.restore(target)
    assert (workspace / "local.txt").read_text() == "protected v1"
    assert (workspace / "code.py").read_text() == "print(1)"
    assert sorted(result.restored) == ["code.py", "local.txt"]


async def test_blocked_shadow_path_raises_checkpoint_error(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.txt").write_text("one")
    blocker = tmp_path / "blocker"
    blocker.write_text("a regular file where the shadow repo should live")
    service = CheckpointService(workspace, shadow_root=blocker)
    with pytest.raises(CheckpointError):
        await service.snapshot(trigger="pre_write", turn_id="turn_1")


async def test_restore_survives_directory_to_file_replacement(workspace: Path, service: CheckpointService) -> None:
    import shutil

    (workspace / "item").mkdir()
    (workspace / "item" / "child.txt").write_text("child")
    target = (await service.snapshot(trigger="pre_write", turn_id="turn_1")).checkpoint_id
    shutil.rmtree(workspace / "item")
    (workspace / "item").write_text("now a plain file")
    result = await service.restore(target)
    assert (workspace / "item").is_dir()
    assert (workspace / "item" / "child.txt").read_text() == "child"
    assert result.anchor_checkpoint_id


async def test_restore_survives_file_to_directory_replacement(workspace: Path, service: CheckpointService) -> None:
    (workspace / "item").write_text("was a file")
    target = (await service.snapshot(trigger="pre_write", turn_id="turn_1")).checkpoint_id
    (workspace / "item").unlink()
    (workspace / "item").mkdir()
    (workspace / "item" / "new.txt").write_text("added in dir")
    await service.restore(target)
    assert (workspace / "item").is_file()
    assert (workspace / "item").read_text() == "was a file"

"""Shadow git checkpoint service: snapshot, restore, retention, isolation."""

from __future__ import annotations

import json
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


def _replace_checkpoint_message(service: CheckpointService, checkpoint_id: str, subject: str,
                                metadata: object | None) -> str:
    tree = service._run_git("rev-parse", f"{checkpoint_id}^{{tree}}").stdout.strip()
    old_message = service._run_git("show", "-s", "--format=%B", checkpoint_id).stdout
    body = "\n".join(line for line in old_message.partition("\n\n")[2].splitlines()
                     if not line.startswith("Praxis-Checkpoint:"))
    message = subject + "\n\n" + body.rstrip() + "\n"
    if metadata is not None:
        message += "\nPraxis-Checkpoint: " + json.dumps(metadata) + "\n"
    replacement = service._run_git_with_input("commit-tree", tree, data=message.encode()).strip()
    service._run_git("update-ref", f"refs/checkpoints/{replacement}", replacement)
    service._run_git("update-ref", "-d", f"refs/checkpoints/{checkpoint_id}")
    service._run_git("update-ref", "refs/praxis/latest", replacement)
    return replacement


async def test_checkpoint_metadata_preserves_machine_identity(workspace: Path, service: CheckpointService) -> None:
    (workspace / "a.txt").write_text("one")
    turn_id = "恢复 a\tb\n" + "x" * 90
    await service.snapshot(trigger="pre_write", turn_id=turn_id)
    [entry] = await service.list_checkpoints()
    assert entry.trigger == "pre_write"
    assert entry.turn_id == turn_id
    assert entry.sequence == 0
    trailer = service._run_git("show", "-s", "--format=%(trailers:key=Praxis-Checkpoint,valueonly)",
                               entry.checkpoint_id).stdout
    assert json.loads(trailer) == {"version": 1, "trigger": "pre_write", "turn_id": turn_id, "sequence": 0}


@pytest.mark.parametrize("trigger", ["restore", "pre_write"])
async def test_public_checkpoint_listing_uses_metadata_not_subject(workspace: Path, tmp_path: Path,
                                                                  monkeypatch, trigger: str) -> None:
    from agent_runtime import Agent

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "config"))
    service = CheckpointService(workspace)
    (workspace / "a.txt").write_text("one")
    snapshot = await service.snapshot(trigger="pre_write", turn_id="original")
    checkpoint_id = _replace_checkpoint_message(service, snapshot.checkpoint_id,
        "A human title containing restore seq=999999", {
            "version": 1, "trigger": trigger, "turn_id": "recorded-turn", "sequence": 7,
            "note": "harmless diagnostic metadata",
        })
    [entry] = await service.list_checkpoints()
    assert entry.sequence == 7
    agent = Agent(workspace_path=workspace, checkpoint_db=tmp_path / "absent.sqlite")
    rows = await agent.list_checkpoints()
    assert [row["checkpoint_id"] for row in rows] == ([checkpoint_id] if trigger == "restore" else [])


async def test_renamed_anchor_is_listed_and_undoable(workspace: Path, tmp_path: Path, monkeypatch) -> None:
    from agent_runtime import Agent

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "config"))
    service = CheckpointService(workspace)
    target = workspace / "a.txt"
    target.write_text("one")
    snapshot = await service.snapshot(trigger="pre_write", turn_id="turn-1")
    target.write_text("two")
    result = await service.restore(snapshot.checkpoint_id)
    anchor = _replace_checkpoint_message(service, result.anchor_checkpoint_id, "Before rewind", {
        "version": 1, "trigger": "restore", "turn_id": "rewind-origin", "sequence": 1,
    })
    agent = Agent(workspace_path=workspace, checkpoint_db=tmp_path / "absent.sqlite")
    [row] = await agent.list_checkpoints()
    assert row["checkpoint_id"] == anchor
    from agent_runtime.harness import RuntimeState
    async with (await RuntimeState.open(tmp_path / "absent.sqlite")):
        pass
    await agent.rewind(anchor)
    assert target.read_text() == "two"


@pytest.mark.parametrize("subject, expected", [
    ("praxis-checkpoint restore rewind-old seq=000007", True),
    ("praxis-checkpoint pre_write restore seq=000008", False),
    ("unrelated restore note seq=000009", False),
    ("praxis-checkpoint restore note seq=1", False),
    ("praxis-checkpoint restore note seq=١٢", False),
    ("praxis-checkpoint restore note seq=0000001", False),
])
async def test_legacy_checkpoint_identity_has_one_strict_decoder(workspace: Path, tmp_path: Path,
                                                                monkeypatch, subject: str, expected: bool) -> None:
    from agent_runtime import Agent

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "config"))
    service = CheckpointService(workspace)
    (workspace / "a.txt").write_text("one")
    snapshot = await service.snapshot(trigger="pre_write", turn_id="original")
    checkpoint_id = _replace_checkpoint_message(service, snapshot.checkpoint_id, subject, None)
    rows = await Agent(workspace_path=workspace, checkpoint_db=tmp_path / "absent.sqlite").list_checkpoints()
    assert [row["checkpoint_id"] for row in rows] == ([checkpoint_id] if expected else [])


@pytest.mark.parametrize("metadata", [
    {"version": 2, "trigger": "restore", "turn_id": "old", "sequence": 0},
    {"version": True, "trigger": "restore", "turn_id": "old", "sequence": 0},
    {"version": 1, "trigger": "other", "turn_id": "old", "sequence": 0},
    {"version": 1, "trigger": "restore", "turn_id": "old", "sequence": True},
    {"version": 1, "trigger": "restore", "sequence": 0},
    "invalid metadata",
])
async def test_invalid_metadata_never_falls_back_to_a_legacy_title(workspace: Path,
                                                                 service: CheckpointService, metadata) -> None:
    (workspace / "a.txt").write_text("one")
    snapshot = await service.snapshot(trigger="pre_write", turn_id="original")
    _replace_checkpoint_message(service, snapshot.checkpoint_id,
                                "praxis-checkpoint restore rewind-old seq=000007", metadata)
    with pytest.raises(CheckpointError, match="metadata"):
        await service.list_checkpoints()


async def test_restore_checks_metadata_before_changing_workspace(workspace: Path, service: CheckpointService) -> None:
    target = workspace / "a.txt"
    target.write_text("one")
    snapshot = await service.snapshot(trigger="pre_write", turn_id="original")
    checkpoint_id = _replace_checkpoint_message(service, snapshot.checkpoint_id,
        "praxis-checkpoint restore old seq=000001", {
            "version": 2, "trigger": "restore", "turn_id": "future", "sequence": 1,
        })
    target.write_text("two")
    with pytest.raises(CheckpointError, match="metadata"):
        await service.restore(checkpoint_id)
    assert target.read_text() == "two"


@pytest.mark.parametrize("trailer", [
    "Praxis-Checkpoint: {",
    "Praxis-Checkpoint:",
    'Praxis-Checkpoint: {}\nPraxis-Checkpoint: {}',
])
def test_malformed_or_duplicate_metadata_is_explicit(trailer: str) -> None:
    from agent_runtime.checkpoints import _checkpoint_entry

    with pytest.raises(CheckpointError, match="metadata"):
        _checkpoint_entry("a" * 40, "1", "praxis-checkpoint restore old seq=000001", "Title\n\n" + trailer)


@pytest.mark.parametrize("key, value", [
    ("trailer.separators", "="),
    ("trailer.praxis-checkpoint.key", "Different-Key"),
    ("trailer.other.key", "Praxis-Checkpoint="),
])
async def test_checkpoint_trailer_protocol_ignores_repository_config(workspace: Path,
                                                                   service: CheckpointService, key, value) -> None:
    (workspace / "a.txt").write_text("one")
    snapshot = await service.snapshot(trigger="pre_write", turn_id="original")
    _replace_checkpoint_message(service, snapshot.checkpoint_id, "Human title", {
        "version": 1, "trigger": "restore", "turn_id": "recorded", "sequence": 1,
    })
    service._run_git("config", key, value)
    [entry] = await service.list_checkpoints()
    assert entry.trigger == "restore"
    assert entry.turn_id == "recorded"


async def test_workspace_display_text_cannot_be_checkpoint_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "work\nPraxis-Checkpoint: fake"
    workspace.mkdir()
    service = CheckpointService(workspace, shadow_root=tmp_path / "shadow")
    (workspace / "a.txt").write_text("one")
    first = await service.snapshot(trigger="pre_write", turn_id="original")
    [entry] = await service.list_checkpoints()
    assert entry.checkpoint_id == first.checkpoint_id
    assert entry.trigger == "pre_write" and entry.turn_id == "original"
    second = await service.snapshot(trigger="pre_write", turn_id="later")
    assert second.checkpoint_id == first.checkpoint_id


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
    anchor = next(entry for entry in entries if entry.trigger == "restore")
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

async def test_restore_preserves_raw_bytes_and_full_permissions(workspace, service):
    (workspace / '.gitattributes').write_text('*.txt text eol=lf\n')
    source = workspace / 'private.txt'
    source.write_bytes(b'one\r\ntwo\r\n')
    source.chmod(0o600)
    from agent_runtime.workspace import workspace_tree_sha256
    expected = workspace_tree_sha256(workspace)
    target = await service.snapshot(trigger='pre_write', turn_id='raw')
    source.write_bytes(b'changed\n')
    source.chmod(0o644)
    await service.restore(target.checkpoint_id)
    assert source.read_bytes() == b'one\r\ntwo\r\n'
    assert source.stat().st_mode & 0o777 == 0o600
    assert workspace_tree_sha256(workspace) == expected


async def test_mode_only_changes_create_restorable_checkpoint(workspace, service):
    source = workspace / 'private.txt'
    source.write_text('same bytes')
    source.chmod(0o600)
    first = await service.snapshot(trigger='pre_write', turn_id='private')
    source.chmod(0o640)
    second = await service.snapshot(trigger='pre_write', turn_id='shared')
    assert second.created_new_commit and second.checkpoint_id != first.checkpoint_id
    assert second.changed_file_count == 1
    await service.restore(first.checkpoint_id)
    assert source.stat().st_mode & 0o777 == 0o600


async def test_failed_restore_reports_anchor_and_rolls_back_completed_writes(workspace, service):
    (workspace / 'sub').mkdir()
    (workspace / 'a').write_text('old a')
    (workspace / 'sub' / 'b').write_text('old b')
    target = await service.snapshot(trigger='pre_write', turn_id='before')
    (workspace / 'a').write_text('new a')
    (workspace / 'sub' / 'b').write_text('new b')
    (workspace / 'extra').write_text('new extra')
    (workspace / 'sub').chmod(0o555)
    try:
        with pytest.raises(CheckpointError) as raised:
            await service.restore(target.checkpoint_id)
    finally:
        (workspace / 'sub').chmod(0o755)
    assert raised.value.anchor_checkpoint_id
    assert raised.value.rollback_completed is True
    assert raised.value.workspace_changed is False
    assert (workspace / 'a').read_text() == 'new a'
    assert (workspace / 'sub' / 'b').read_text() == 'new b'
    assert (workspace / 'extra').read_text() == 'new extra'

async def test_checkpoint_does_not_use_inherited_git_index(workspace, service, tmp_path, monkeypatch):
    foreign_index = tmp_path / 'unrelated.index'
    monkeypatch.setenv('GIT_INDEX_FILE', str(foreign_index))
    (workspace / 'a').write_bytes(b'raw\r\n')
    await service.snapshot(trigger='pre_write', turn_id='isolated')
    assert not foreign_index.exists()

async def test_raw_restore_streams_large_blob_without_loading_it_all(workspace, service):
    import tracemalloc
    source = workspace / 'large.bin'
    source.write_bytes(b'x' * 4_000_000)
    target = await service.snapshot(trigger='pre_write', turn_id='large')
    source.write_bytes(b'changed')
    tracemalloc.start()
    try:
        await service.restore(target.checkpoint_id)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert source.stat().st_size == 4_000_000
    assert peak < 3_000_000, 'checkpoint restore buffered an entire blob in memory'
@pytest.mark.anyio
async def test_orphan_pruning_holds_repository_lock_during_removal(workspace, service, monkeypatch):
    import fcntl
    import os
    import shutil

    await service.snapshot(trigger="pre_write", turn_id="owned")
    shutil.rmtree(workspace)
    remove = shutil.rmtree
    observed = []

    def guarded_remove(path, *args, **kwargs):
        if Path(path) == service.shadow_root:
            descriptor = os.open(service._lock()._path, os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                observed.append(True)
            finally:
                os.close(descriptor)
        return remove(path, *args, **kwargs)

    monkeypatch.setattr("agent_runtime.checkpoints.shutil.rmtree", guarded_remove)
    await service.prune()
    assert observed == [True]
    assert not service.shadow_root.exists()

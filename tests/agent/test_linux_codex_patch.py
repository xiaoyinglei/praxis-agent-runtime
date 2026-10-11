from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_runtime.tools.builtins import codex_patch
from agent_runtime.workspace import open_workspace


def test_linux_factory_accepts_a_native_executor(tmp_path, monkeypatch):
    binary = tmp_path / "codex"
    binary.write_bytes(b"test executable")
    binary.chmod(0o755)
    monkeypatch.setattr(codex_patch, "sys", SimpleNamespace(platform="linux"))
    tool = codex_patch.create_codex_apply_patch_tool(open_workspace(tmp_path / "workspace", create=True), binary)
    assert tool.definition.name == "apply_patch"


def test_view_contains_only_patch_targets_and_preserves_header_like_context(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "edit.txt").write_text("*** Delete File: .git/config\nold\n")
    (workspace.root / "unrelated.txt").write_text("not exposed")
    view = tmp_path / "view"
    patch = "*** Begin Patch\n*** Update File: edit.txt\n@@\n *** Delete File: .git/config\n-old\n+new\n*** End Patch"
    initial = codex_patch._prepare_linux_patch_view(workspace, view, patch)
    assert initial == {"edit.txt": b"*** Delete File: .git/config\nold\n"}
    assert (view / "edit.txt").read_bytes() == initial["edit.txt"]
    assert not (view / "unrelated.txt").exists()


@pytest.mark.parametrize(
    "name",
    [".git/config", "a/.venv/lib.py", "node_modules/new.txt", ".agent_memory/new", ".praxis/new", "../outside.txt"],
)
def test_protected_or_external_targets_fail_before_staging(tmp_path, name):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    patch = f"*** Begin Patch\n*** Add File: {name}\n+bad\n*** End Patch"
    with pytest.raises((PermissionError, ValueError)):
        codex_patch._prepare_linux_patch_view(workspace, tmp_path / "view", patch)
    assert not (tmp_path / "outside.txt").exists()


def test_symlink_target_is_not_copied_or_written(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (workspace.root / "alias").symlink_to(outside)
    patch = "*** Begin Patch\n*** Update File: alias\n@@\n-secret\n+bad\n*** End Patch"
    with pytest.raises((PermissionError, ValueError)):
        codex_patch._prepare_linux_patch_view(workspace, tmp_path / "view", patch)
    assert outside.read_text() == "secret"


def test_commit_validates_entire_view_before_changing_host_files(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "edit.txt").write_text("old\n")
    view = tmp_path / "view"
    patch = "*** Begin Patch\n*** Update File: edit.txt\n@@\n-old\n+new\n*** End Patch"
    initial = codex_patch._prepare_linux_patch_view(workspace, view, patch)
    (view / "edit.txt").write_text("new\n")
    (view / "unexpected.txt").write_text("unexpected")
    with pytest.raises(PermissionError):
        codex_patch._commit_linux_patch_view(workspace, view, initial)
    assert (workspace.root / "edit.txt").read_text() == "old\n"
    assert not (workspace.root / "unexpected.txt").exists()


def test_commit_preserves_concurrent_host_changes(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    target = workspace.root / "edit.txt"
    target.write_text("old\n")
    view = tmp_path / "view"
    initial = codex_patch._prepare_linux_patch_view(
        workspace,
        view,
        "*** Begin Patch\n*** Update File: edit.txt\n@@\n-old\n+new\n*** End Patch",
    )
    (view / "edit.txt").write_text("new\n")
    target.write_text("user update\n")
    with pytest.raises(PermissionError):
        codex_patch._commit_linux_patch_view(workspace, view, initial)
    assert target.read_text() == "user update\n"


def test_commit_create_move_delete_and_partial_native_failure(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "old.txt").write_text("old\n")
    view = tmp_path / "view"
    patch = (
        "*** Begin Patch\n*** Update File: old.txt\n*** Move to: new/moved.txt\n@@\n-old\n+new\n"
        "*** Add File: created.txt\n+hello\n*** Update File: missing.txt\n@@\n-old\n+new\n*** End Patch"
    )
    initial = codex_patch._prepare_linux_patch_view(workspace, view, patch)
    # The upstream executor completed the first hunks before failing on missing.txt.
    (view / "old.txt").unlink()
    (view / "new").mkdir()
    (view / "new/moved.txt").write_text("new\n")
    (view / "created.txt").write_text("hello\n")
    codex_patch._commit_linux_patch_view(workspace, view, initial)
    assert not (workspace.root / "old.txt").exists()
    assert (workspace.root / "new/moved.txt").read_text() == "new\n"
    assert (workspace.root / "created.txt").read_text() == "hello\n"
    assert not (workspace.root / "missing.txt").exists()


@pytest.mark.parametrize("output", ["symlink", "fifo", "invalid_utf8"])
def test_nonregular_or_nontext_output_is_rejected_before_commit(tmp_path, output):
    import os

    workspace = open_workspace(tmp_path / "workspace", create=True)
    target = workspace.root / "edit.txt"
    target.write_text("old\n")
    view = tmp_path / "view"
    initial = codex_patch._prepare_linux_patch_view(
        workspace,
        view,
        "*** Begin Patch\n*** Update File: edit.txt\n@@\n-old\n+new\n*** End Patch",
    )
    staged = view / "edit.txt"
    staged.unlink()
    if output == "symlink":
        staged.symlink_to(target)
    elif output == "fifo":
        os.mkfifo(staged)
    else:
        staged.write_bytes(b"\xff")
    with pytest.raises((PermissionError, UnicodeError)):
        codex_patch._commit_linux_patch_view(workspace, view, initial)
    assert target.read_text() == "old\n"


def test_staging_and_output_resource_limits(tmp_path, monkeypatch):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "edit.txt").write_text("old\n")
    patch = "*** Begin Patch\n*** Update File: edit.txt\n@@\n-old\n+new\n*** End Patch"
    monkeypatch.setattr(codex_patch, "_MAX_FILE_BYTES", 3)
    with pytest.raises(ValueError, match="limit"):
        codex_patch._prepare_linux_patch_view(workspace, tmp_path / "refused", patch)
    monkeypatch.setattr(codex_patch, "_MAX_FILE_BYTES", 4)
    view = tmp_path / "view"
    initial = codex_patch._prepare_linux_patch_view(workspace, view, patch)
    (view / "edit.txt").write_text("too large\n")
    with pytest.raises(ValueError, match="limit"):
        codex_patch._commit_linux_patch_view(workspace, view, initial)
    assert (workspace.root / "edit.txt").read_text() == "old\n"


def test_linux_binary_inside_workspace_is_rejected(tmp_path, monkeypatch):
    workspace = open_workspace(tmp_path, create=True)
    binary = tmp_path / "codex"
    binary.write_bytes(b"test executable")
    binary.chmod(0o755)
    monkeypatch.setattr(codex_patch, "sys", SimpleNamespace(platform="linux"))
    with pytest.raises(ValueError, match="outside"):
        codex_patch.create_codex_apply_patch_tool(workspace, binary)


@pytest.mark.skipif(codex_patch.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("invalid_output", [False, True])
async def test_real_child_sees_only_candidates_and_cannot_escape(tmp_path, monkeypatch, invalid_output):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "edit.txt").write_text("old\n")
    (workspace.root / "unrelated.txt").write_text("private")
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    monkeypatch.setenv("PRAXIS_TEST_SECRET", "must-not-leak")
    binary = tmp_path / "probe"
    binary.write_text(
        "#!/usr/bin/python3\n"
        "import os,socket\nfrom pathlib import Path\n"
        "assert Path('edit.txt').read_text() == 'old\\n'\n"
        "assert not Path('unrelated.txt').exists()\n"
        f"assert not Path({str(outside)!r}).exists()\n"
        "assert 'PRAXIS_TEST_SECRET' not in os.environ\n"
        "s=socket.socket();s.settimeout(.2)\n"
        "try:\n s.connect(('1.1.1.1',80))\n"
        "except OSError:\n pass\n"
        "else:\n raise AssertionError('network escaped')\n"
        "Path('edit.txt').write_text('new\\n')\n"
        + ("Path('.git').mkdir();Path('.git/config').write_text('bad')\n" if invalid_output else "")
    )
    binary.chmod(0o755)
    result = await codex_patch._run_patch(
        workspace,
        binary,
        "*** Begin Patch\n*** Update File: edit.txt\n@@\n-old\n+new\n*** End Patch",
    )
    assert not result.timed_out
    if invalid_output:
        assert result.exit_code != 0
        assert "Commit refused" in result.stderr
        assert (workspace.root / "edit.txt").read_text() == "old\n"
    else:
        assert result.exit_code == 0, result.stderr
        assert (workspace.root / "edit.txt").read_text() == "new\n"
    assert (workspace.root / "unrelated.txt").read_text() == "private"
    assert outside.read_text() == "private"
    assert not (workspace.root / ".git").exists()


@pytest.mark.skipif(codex_patch.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancellation_discard_staged_edits(tmp_path, monkeypatch, cancel):
    import asyncio

    workspace = open_workspace(tmp_path / "workspace", create=True)
    target = workspace.root / "edit.txt"
    target.write_text("old\n")
    binary = tmp_path / "slow-probe"
    binary.write_text("#!/bin/sh\nprintf 'new\\n' > edit.txt\nprintf 'staged\\n'\nsleep 10\n")
    binary.chmod(0o755)
    if not cancel:
        wait = asyncio.wait

        async def short_wait(tasks, **_kwargs):
            return await wait(tasks, timeout=0.3)

        monkeypatch.setattr(codex_patch.asyncio, "wait", short_wait)
    task = asyncio.create_task(
        codex_patch._run_patch(
            workspace,
            binary,
            "*** Begin Patch\n*** Update File: edit.txt\n@@\n-old\n+new\n*** End Patch",
        )
    )
    if cancel:
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result.timed_out
        assert "staged" in result.stdout
    assert target.read_text() == "old\n"


@pytest.mark.parametrize("kind", ["missing", "symlink"])
def test_invalid_view_root_cannot_turn_into_host_deletions(tmp_path, kind):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "edit.txt").write_text("old\n")
    view = tmp_path / "view"
    initial = codex_patch._prepare_linux_patch_view(
        workspace,
        view,
        "*** Begin Patch\n*** Delete File: edit.txt\n*** End Patch",
    )
    view.rename(tmp_path / "original_view")
    if kind == "symlink":
        view.symlink_to(tmp_path / "original_view")
    with pytest.raises(PermissionError):
        codex_patch._commit_linux_patch_view(workspace, view, initial)
    assert (workspace.root / "edit.txt").read_text() == "old\n"

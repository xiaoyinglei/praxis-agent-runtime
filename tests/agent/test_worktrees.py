"""Managed Git worktrees give parallel sessions isolated checkouts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_runtime.git_delivery import GitDelivery
from agent_runtime.worktrees import (
    PROVENANCE_RELATIVE_PATH,
    WorktreeError,
    WorktreeManager,
)

runner = CliRunner()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    (root / ".gitignore").write_text(".praxis/\n.env\nuntracked-secret.txt\nconfig/secrets.json\n")
    (root / "base.txt").write_text("base\n")
    (root / ".env").write_text("SECRET=1\n")
    (root / "untracked-secret.txt").write_text("local\n")
    (root / "config").mkdir()
    (root / "config" / "secrets.json").write_text('{"k": "v"}\n')
    _git(root, "add", ".gitignore", "base.txt")
    _git(root, "commit", "-q", "-m", "base")
    return root


@pytest.fixture
def worktrees_root(tmp_path: Path) -> Path:
    return tmp_path / "worktrees"


def _git(repo: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=check,
    )
    return completed.stdout


def _branches(repo: Path) -> list[str]:
    return _git(repo, "branch", "--format=%(refname:short)").split()


def _manager(repo: Path, root: Path) -> WorktreeManager:
    return WorktreeManager(repo, root=root)


def test_create_builds_isolated_checkout_on_named_branch(
    repo: Path, worktrees_root: Path,
) -> None:
    report = _manager(repo, worktrees_root).create("demo feature")
    worktree = report.worktree

    assert worktree.branch == "worktree/demo-feature"
    assert worktree.root.parent.parent == worktrees_root.resolve()
    assert worktree.root.name == repo.name
    assert worktree.base_commit == _git(repo, "rev-parse", "HEAD").strip()
    assert worktree.source_root == repo
    provenance = json.loads((worktree.root / PROVENANCE_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert provenance["branch"] == worktree.branch
    assert provenance["version"] == 1
    if sys.platform == "darwin":
        assert (worktrees_root / ".metadata_never_index").exists()

    # Writing inside the worktree never touches the source checkout.
    (worktree.root / "parallel.txt").write_text("isolated\n")
    assert not (repo / "parallel.txt").exists()
    assert "parallel.txt" not in _git(repo, "status", "--porcelain")
    assert _git(worktree.root, "rev-parse", "--abbrev-ref", "HEAD").strip() == worktree.branch


def test_create_dedupes_branch_and_sanitizes_names(repo: Path, worktrees_root: Path) -> None:
    manager = _manager(repo, worktrees_root)
    first = manager.create("Feature!!").worktree
    assert first.branch == "worktree/feature"
    second = manager.create("feature").worktree
    assert second.branch == "worktree/feature-2"


def test_create_generates_session_name_when_missing(repo: Path, worktrees_root: Path) -> None:
    worktree = _manager(repo, worktrees_root).create(None).worktree
    assert worktree.branch.startswith("worktree/session-")


def test_create_refuses_outside_repository(tmp_path: Path) -> None:
    outside = tmp_path / "plain"
    outside.mkdir()
    with pytest.raises(WorktreeError, match="不是 Git 仓库"):
        WorktreeManager(outside, root=tmp_path / "worktrees").create("x")


def test_create_refuses_nested_worktree(repo: Path, worktrees_root: Path) -> None:
    manager = _manager(repo, worktrees_root)
    worktree = manager.create("outer").worktree
    with pytest.raises(WorktreeError, match="嵌套"):
        WorktreeManager(worktree.root, root=worktrees_root).create("inner")


def test_dirty_source_is_excluded_and_reported(repo: Path, worktrees_root: Path) -> None:
    (repo / "base.txt").write_text("uncommitted edit\n")
    report = _manager(repo, worktrees_root).create("dirty-base")
    assert report.uncommitted_in_source == 1
    # The worktree carries the committed HEAD state, never the dirty edit.
    assert (report.worktree.root / "base.txt").read_text() == "base\n"


def test_worktreeinclude_copies_matched_ignored_files_only(
    repo: Path, worktrees_root: Path,
) -> None:
    (repo / ".worktreeinclude").write_text(".env\nconfig/\n")
    report = _manager(repo, worktrees_root).create("with-env")
    assert report.included_files == (".env", "config/secrets.json")
    assert (report.worktree.root / ".env").read_text() == "SECRET=1\n"
    assert (report.worktree.root / "config" / "secrets.json").exists()
    # Ignored but unmatched files stay out of the fresh checkout.
    assert not (report.worktree.root / "untracked-secret.txt").exists()


def test_git_delivery_from_worktree_inherits_real_base(
    repo: Path, worktrees_root: Path,
) -> None:
    import asyncio

    manager = _manager(repo, worktrees_root)
    worktree = manager.create("delivery").worktree
    # The worktree branch records the source branch as its delivery base.
    recorded = _git(
        repo, "config", "--get", f"branch.{worktree.branch}.praxis-base",
    ).strip()
    assert recorded in {"main", "master"}

    (worktree.root / "change.txt").write_text("delivered\n")
    result = asyncio.run(GitDelivery(worktree.root).commit("from worktree"))

    assert result.branch.startswith("praxis/")
    assert result.branch in _branches(repo)
    # The delivery branch inherits the worktree's base, so /pull targets the
    # real upstream instead of the local worktree branch.
    delivery_base = _git(
        repo, "config", "--get", f"branch.{result.branch}.praxis-base",
    ).strip()
    assert delivery_base in {"main", "master"}
    # Delivering does not advance the worktree branch: the checkout stays clean
    # and removable once the praxis branch carries the work.
    inspection = manager.inspect(worktree)
    assert inspection.dirty_count == 0
    assert inspection.ahead_count == 0


def test_list_and_find_resolve_entries(repo: Path, worktrees_root: Path) -> None:
    manager = _manager(repo, worktrees_root)
    created = [manager.create("one").worktree, manager.create("two").worktree]
    listed = manager.list()
    assert {entry.name for entry in listed} == {"one", "two"}

    assert manager.find("one").root == created[0].root
    assert manager.find(str(created[0].root)).root == created[0].root
    with pytest.raises(WorktreeError, match="没有名为 missing"):
        manager.find("missing")


def test_remove_clean_worktree_deletes_branch(repo: Path, worktrees_root: Path) -> None:
    manager = _manager(repo, worktrees_root)
    worktree = manager.create("gone").worktree
    report = manager.remove("gone")

    assert report.branch_deleted is True
    assert not worktree.root.exists()
    assert worktree.branch not in _branches(repo)
    assert manager.list() == []


def test_remove_refuses_uncommitted_changes_without_force(
    repo: Path, worktrees_root: Path,
) -> None:
    manager = _manager(repo, worktrees_root)
    worktree = manager.create("dirty").worktree
    (worktree.root / "wip.txt").write_text("wip\n")

    with pytest.raises(WorktreeError, match="--force"):
        manager.remove("dirty")
    assert worktree.root.exists()
    assert manager.remove("dirty", force=True).branch_deleted is True
    assert not worktree.root.exists()


def test_remove_refuses_unmerged_commits_without_force(
    repo: Path, worktrees_root: Path,
) -> None:
    manager = _manager(repo, worktrees_root)
    worktree = manager.create("ahead").worktree
    _git(worktree.root, "commit", "--allow-empty", "-q", "-m", "direct work")

    with pytest.raises(WorktreeError, match="未合并提交"):
        manager.remove("ahead")
    assert worktree.branch in _branches(repo)
    manager.remove("ahead", force=True)
    assert worktree.branch not in _branches(repo)


def test_prune_keeps_recent_clean_worktrees(repo: Path, worktrees_root: Path) -> None:
    manager = _manager(repo, worktrees_root)
    for name in ("a", "b", "c"):
        manager.create(name)
    # Dirty worktrees are never pruned, no matter how old.
    protected = manager.create("protected").worktree
    (protected.root / "wip.txt").write_text("wip\n")

    removed = manager.prune(keep=1)
    assert len(removed) == 3
    remaining = {entry.name for entry in manager.list()}
    assert remaining == {"protected"}


def test_dirty_count_ignores_runtime_state(repo: Path, worktrees_root: Path) -> None:
    manager = _manager(repo, worktrees_root)
    worktree = manager.create("clean").worktree
    inspection = manager.inspect(worktree)
    # The provenance file under .praxis is runtime state, not pending work.
    assert inspection.dirty_count == 0
    assert inspection.ahead_count == 0


def test_cli_worktree_list_and_remove(repo: Path, worktrees_root: Path, monkeypatch) -> None:
    from agent_runtime.cli import agent_app

    monkeypatch.chdir(repo)
    manager = _manager(repo, worktrees_root)
    worktree = manager.create("cli-entry").worktree

    listed = runner.invoke(agent_app, ["worktree", "list", "--root", str(worktrees_root)])
    assert listed.exit_code == 0, listed.output
    rows = json.loads(listed.output)
    assert [row["name"] for row in rows] == ["cli-entry"]
    assert rows[0]["branch"] == worktree.branch
    assert rows[0]["dirty_count"] == 0

    removed = runner.invoke(agent_app, ["worktree", "remove", "cli-entry", "--root", str(worktrees_root)])
    assert removed.exit_code == 0, removed.output
    assert json.loads(removed.output)["branch_deleted"] is True
    assert not worktree.root.exists()

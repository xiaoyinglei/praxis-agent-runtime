"""Managed Git worktrees give parallel sessions isolated checkouts."""

from __future__ import annotations

import asyncio
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


def test_remove_preserves_owned_turn_even_when_forced(repo: Path, worktrees_root: Path) -> None:
    from agent_runtime.harness import RuntimeState
    from agent_runtime.harness.ownership import execution_owner

    manager = _manager(repo, worktrees_root)
    worktree = manager.create("running").worktree
    database = worktree.root / ".praxis" / "checkpoints.sqlite"
    with RuntimeState.open_sync(database) as store:
        thread = store.create_thread(workspace=worktree.root)
        store.start_turn(thread_id=thread.thread_id, user_message="running", binding_manifest={"model_id": "test"})
        with execution_owner(database, thread.thread_id):
            for force in (False, True):
                with pytest.raises(WorktreeError):
                    manager.remove(worktree.name, force=force)
            assert database.is_file()


def test_workspace_lock_creates_private_configuration_parents(tmp_path: Path, monkeypatch) -> None:
    import stat

    from agent_runtime.harness.ownership import workspace_owner

    config = tmp_path / "new-parent" / "config"
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(config))
    with workspace_owner(tmp_path):
        assert stat.S_IMODE(config.stat().st_mode) == 0o700
        assert stat.S_IMODE((config / "workspace-locks").stat().st_mode) == 0o700


@pytest.mark.anyio
async def test_open_session_protects_clean_checkout_until_close(repo: Path, worktrees_root: Path) -> None:
    from agent_runtime.harness import Session
    from tests.agent.harness.test_approval_resume import AcceptAnswer, WriteThenAnswerModel

    manager = _manager(repo, worktrees_root)
    worktree = manager.create("open-session").worktree
    async with await Session.open(
        database=worktree.root / ".praxis" / "checkpoints.sqlite", workspace=worktree.root,
        model=WriteThenAnswerModel(), completion_gate=AcceptAnswer(),
    ):
        with pytest.raises(WorktreeError):
            (await asyncio.to_thread(manager.remove, worktree.name))
        assert worktree.root.is_dir()
    assert (await asyncio.to_thread(manager.remove, worktree.name)).branch_deleted


@pytest.mark.anyio
async def test_subdirectory_session_with_external_database_protects_checkout(
    repo: Path, worktrees_root: Path, tmp_path: Path,
) -> None:
    from agent_runtime.harness import Session
    from tests.agent.harness.test_approval_resume import AcceptAnswer, WriteThenAnswerModel

    manager = _manager(repo, worktrees_root)
    worktree = manager.create("subdirectory").worktree
    subdirectory = worktree.root / "nested"
    subdirectory.mkdir()
    async with await Session.open(
        database=tmp_path / "external.sqlite", workspace=subdirectory,
        model=WriteThenAnswerModel(), completion_gate=AcceptAnswer(),
    ):
        with pytest.raises(WorktreeError):
            (await asyncio.to_thread(manager.remove, worktree.name, force=True))
        assert subdirectory.is_dir()


@pytest.mark.anyio
async def test_session_does_not_block_removing_an_idle_sibling_inside_repository(repo: Path) -> None:
    from agent_runtime.harness import Session
    from tests.agent.harness.test_approval_resume import AcceptAnswer, WriteThenAnswerModel

    manager = _manager(repo, repo / ".managed")
    busy = manager.create("busy").worktree
    idle = manager.create("idle").worktree
    async with await Session.open(
        database=busy.root / ".praxis/checkpoints.sqlite", workspace=busy.root,
        model=WriteThenAnswerModel(), completion_gate=AcceptAnswer(),
    ):
        assert (await asyncio.to_thread(manager.remove, idle.name)).branch_deleted
        assert busy.root.is_dir()


def test_remove_requires_force_for_ignored_user_files(repo: Path, worktrees_root: Path) -> None:
    manager = _manager(repo, worktrees_root)
    worktree = manager.create("local-data").worktree
    local = worktree.root / ".env"
    local.write_text("UNIQUE_LOCAL_VALUE=review\n")
    with pytest.raises(WorktreeError, match="--force"):
        manager.remove(worktree.name)
    assert local.read_text() == "UNIQUE_LOCAL_VALUE=review\n"
    assert manager.remove(worktree.name, force=True).branch_deleted


def test_prune_and_explicit_paths_stay_within_repository(repo: Path, worktrees_root: Path, tmp_path: Path) -> None:
    other = tmp_path / "other-project"
    other.mkdir()
    _git(other, "init", "--quiet")
    _git(other, "config", "user.name", "Test")
    _git(other, "config", "user.email", "test@example.com")
    (other / ".gitignore").write_text(".praxis/\n")
    _git(other, "add", ".gitignore")
    _git(other, "commit", "-q", "-m", "base")
    foreign = _manager(other, worktrees_root).create("foreign").worktree
    manager = _manager(repo, worktrees_root)
    manager.create("one")
    manager.create("two")
    assert {entry.name for entry in manager.list()} == {"one", "two"}
    with pytest.raises(WorktreeError):
        manager.remove(str(foreign.root), force=True)
    manager.prune(keep=1)
    assert foreign.root.is_dir()
    # Looking up from a linked checkout still resolves the same repository.
    remaining = manager.list()[0]
    assert _manager(remaining.root, worktrees_root).list() == manager.list()


def test_broken_foreign_repository_does_not_block_current_repository(
    repo: Path, worktrees_root: Path, tmp_path: Path,
) -> None:
    import shutil

    manager = _manager(repo, worktrees_root)
    current = manager.create("current").worktree
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    _git(foreign, "init", "--quiet")
    _git(foreign, "config", "user.name", "Test")
    _git(foreign, "config", "user.email", "test@example.com")
    _git(foreign, "commit", "--allow-empty", "-q", "-m", "base")
    _manager(foreign, worktrees_root).create("broken")
    shutil.rmtree(foreign)
    assert [entry.name for entry in manager.list()] == ["current"]
    assert manager.remove(current.name).branch_deleted


@pytest.mark.anyio
@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="requires native OS sandbox")
async def test_worktree_git_reads_work_in_sandbox_without_source_or_metadata_writes(
    repo: Path, worktrees_root: Path,
) -> None:
    from agent_runtime.tools.builtins.shell import create_run_command_tool
    from agent_runtime.workspace import open_workspace

    worktree = _manager(repo, worktrees_root).create("native-git").worktree
    tool = create_run_command_tool(open_workspace(worktree.root))
    for command in ("git status --porcelain", "git diff", "git log -1 --format=%s"):
        output = await tool.run(tool.validate_input({"command": command}))
        assert output.exit_code == 0, output.stderr
    # Grant only Git metadata, never the source checkout or metadata mutation.
    output = await tool.run(tool.validate_input({"command": f"cat '{repo / 'base.txt'}'"}))
    assert output.exit_code != 0
    metadata = repo / ".git" / "config"
    before = metadata.read_bytes()
    output = await tool.run(tool.validate_input({
        "command": f"printf corrupted >> '{metadata}'", "workspace_write": True,
    }))
    assert output.exit_code != 0
    assert metadata.read_bytes() == before


@pytest.mark.anyio
@pytest.mark.parametrize("entrypoint", ["sdk", "cli"])
@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="requires native OS sandbox")
async def test_public_worktree_run_verifies_real_git_status(
    repo: Path, worktrees_root: Path, monkeypatch, entrypoint: str,
) -> None:
    from agent_runtime import Agent, cli
    from tests.agent.harness.test_macos_command_sessions import SessionCommandModel

    command = "git status --porcelain"
    if entrypoint == "sdk":
        worktree = _manager(repo, worktrees_root).create("public-sdk").worktree
        agent = Agent(workspace_path=worktree.root, checkpoint_db=worktree.root / ".praxis/checkpoints.sqlite",
                      enable_workspace_mcp=False)
        monkeypatch.setattr(agent, "_model_factory", lambda: SessionCommandModel(command))
        result = await agent.run("Inspect Git", allow_execute_tools=True, acceptance_command=command, max_steps=4)
        assert result.status == "done", result
        assert result.validation_status == "passed"
    else:
        monkeypatch.chdir(repo)
        monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(worktrees_root))
        original = cli._create_agent_facade

        def create(**kwargs):
            agent = original(**kwargs)
            monkeypatch.setattr(agent, "_model_factory", lambda: SessionCommandModel(command))
            return agent

        monkeypatch.setattr(cli, "_create_agent_facade", create)
        # CliRunner owns its own event loop, as it does in a normal CLI process.
        import asyncio

        result = await asyncio.to_thread(runner.invoke, cli.agent_app, [
            "run", "Inspect Git", "--worktree-name", "public-cli", "--allow-execute-tools",
            "--acceptance-command", command, "--max-steps", "4", "--non-interactive", "--disable-workspace-mcp",
        ])
        assert result.exit_code == 0, result.output


def test_linked_git_metadata_is_only_mounted_read_only_on_linux(repo: Path, worktrees_root: Path, tmp_path) -> None:
    from agent_runtime.tools.builtins import shell

    worktree = _manager(repo, worktrees_root).create("linux-metadata").worktree
    arguments = shell._linux_sandbox_argv(
        workspace_root=worktree.root, temporary_root=tmp_path / "scratch", additional_read_roots=(),
    )
    common = str(repo / ".git")
    offset = arguments.index(common)
    assert arguments[offset - 1:offset + 2] == ["--ro-bind", common, common]
    assert str(repo) not in arguments
    # A broken or forged backlink must never turn a .git pointer into a read grant.
    administrative = Path(_git(worktree.root, "rev-parse", "--absolute-git-dir").strip())
    (administrative / "gitdir").write_text(str(tmp_path / "unrelated"))
    assert shell._linked_git_metadata_roots(worktree.root) == ()


def test_remove_protects_sdk_store_at_arbitrary_location(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import RuntimeState

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "host-config"))
    manager = WorktreeManager(repo, root=tmp_path / "trees")
    entry = manager.create("registered-store").worktree
    database = entry.root / ".praxis" / "custom-state" / "conversation.db"
    with RuntimeState.open_sync(database) as store:
        thread = store.create_thread(workspace=entry.root)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="unfinished", binding_manifest={})
        store.pause_turn(turn_id=turn.turn_id, reason="waiting for user")
    with pytest.raises(WorktreeError, match="未结束"):
        manager.remove(entry.name)
    assert database.is_file()


def test_prune_retains_unregistered_runtime_state(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "host-config"))
    manager = WorktreeManager(repo, root=tmp_path / "trees")
    entry = manager.create("unregistered-state").worktree
    state = entry.root / ".praxis" / "unknown-format.bin"
    state.write_bytes(b"user session state from an older runtime")
    with pytest.raises(WorktreeError, match="状态"):
        manager.remove(entry.name)
    assert state.is_file()


def test_explicit_worktree_base_is_delivery_target(repo, tmp_path):
    _git(repo, "branch", "release")
    manager = WorktreeManager(repo, root=tmp_path / "trees")
    entry = manager.create("release-fix", base="release").worktree
    assert _git(repo, "config", "--get", f"branch.{entry.branch}.praxis-base").strip() == "release"

"""User-authorized commits on the workspace's real Git repository."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from agent_runtime.git_delivery import CheckRun, DeliveryError, GitDelivery

pytestmark = pytest.mark.anyio


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    run = subprocess.run
    run(["git", "init", "--quiet", str(root)], check=True)
    run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    (root / "base.txt").write_text("base\n")
    run(["git", "-C", str(root), "add", "-A"], check=True)
    run(["git", "-C", str(root), "commit", "-q", "-m", "base"], check=True)
    return root


def _git(repo: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=check,
    )
    return completed.stdout


async def test_commit_captures_added_modified_and_deleted(repo: Path) -> None:
    (repo / "gone.txt").write_text("delete me\n")
    _git(repo, "add", "gone.txt")
    _git(repo, "commit", "-q", "-m", "gone file")
    (repo / "gone.txt").unlink()
    (repo / "added.txt").write_text("new\n")
    (repo / "base.txt").write_text("changed\n")

    delivery = GitDelivery(repo)
    status = await delivery.status()
    assert status.branch == "master" or status.branch == "main"
    assert set(status.changed_paths) == {"added.txt", "base.txt", "gone.txt"}

    result = await delivery.commit("agent changes")
    assert result.changed_file_count == 3
    assert result.created_branch is True
    assert result.branch.startswith("praxis/")
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() == result.branch
    assert _git(repo, "log", "-1", "--format=%s").strip() == "agent changes"
    # The commit carries the user's own identity, and the tree is committed clean.
    assert "Test" in _git(repo, "log", "-1", "--format=%an")
    assert (await delivery.status()).changed == ()


async def test_successive_commits_stay_on_one_praxis_branch(repo: Path) -> None:
    delivery = GitDelivery(repo)
    (repo / "one.txt").write_text("1\n")
    first = await delivery.commit("first")
    (repo / "two.txt").write_text("2\n")
    second = await delivery.commit("second")
    assert second.branch == first.branch
    assert second.created_branch is False
    assert len(_git(repo, "log", "--format=%s").splitlines()) == 3


async def test_explicit_branch_name_is_honored_and_refused_when_taken(repo: Path) -> None:
    delivery = GitDelivery(repo)
    (repo / "a.txt").write_text("a\n")
    result = await delivery.commit("named", branch_name="praxis/feature")
    assert result.branch == "praxis/feature"
    # Committing again while still on the named branch stacks on it.
    (repo / "b.txt").write_text("b\n")
    stacked = await delivery.commit("again", branch_name="praxis/feature")
    assert stacked.created_branch is False
    # Reusing an existing name from another branch is refused.
    _git(repo, "switch", "-q", "master" if (repo / ".git" / "refs" / "heads" / "master").exists()
         else "main")
    (repo / "c.txt").write_text("c\n")
    with pytest.raises(DeliveryError, match="分支已存在"):
        await delivery.commit("third", branch_name="praxis/feature")


async def test_nothing_to_commit_is_refused(repo: Path) -> None:
    delivery = GitDelivery(repo)
    with pytest.raises(DeliveryError, match="没有可提交的变更"):
        await delivery.commit("empty")


async def test_empty_message_is_refused(repo: Path) -> None:
    (repo / "a.txt").write_text("a\n")
    with pytest.raises(DeliveryError, match="提交说明"):
        await GitDelivery(repo).commit("   ")


async def test_in_progress_merge_is_refused(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "other")
    (repo / "base.txt").write_text("other view\n")
    _git(repo, "commit", "-q", "-am", "other change")
    _git(repo, "checkout", "-q", "master" if (repo / ".git" / "refs" / "heads" / "master").exists()
         else "main")
    (repo / "base.txt").write_text("main view\n")
    _git(repo, "commit", "-q", "-am", "main change")
    _git(repo, "merge", "other", check=False)  # conflicts on purpose
    (repo / "any.txt").write_text("pending\n")
    with pytest.raises(DeliveryError, match="未完成"):
        await GitDelivery(repo).commit("mid-merge")


async def test_subdirectory_workspace_is_refused(repo: Path) -> None:
    sub = repo / "sub"
    sub.mkdir()
    with pytest.raises(DeliveryError, match="仓库根目录"):
        await GitDelivery(sub).status()


async def test_non_repository_workspace_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "plain"
    root.mkdir()
    with pytest.raises(DeliveryError, match="不是 Git 仓库"):
        await GitDelivery(root).status()


async def test_ignored_files_do_not_count_as_changes(repo: Path) -> None:
    (repo / ".gitignore").write_text("ignored.txt\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "ignore rule")
    (repo / "ignored.txt").write_text("not visible to git\n")
    delivery = GitDelivery(repo)
    assert (await delivery.status()).changed == ()
    with pytest.raises(DeliveryError, match="没有可提交的变更"):
        await delivery.commit("nothing")


async def test_agent_facade_commit_requires_idle_turns(repo: Path, tmp_path: Path) -> None:
    from agent_runtime.agent import Agent
    from agent_runtime.harness import RolloutStore

    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=repo)
        store.start_turn(
            thread_id=thread.thread_id,
            user_message="still running",
            binding_manifest={"model_id": "test-model"},
        )
    (repo / "a.txt").write_text("a\n")
    agent = Agent(workspace_path=repo, checkpoint_db=tmp_path / "rollout.sqlite3")
    with pytest.raises(RuntimeError, match="未结束"):
        await agent.commit("blocked")


async def test_agent_facade_commit_returns_delivery_summary(repo: Path, tmp_path: Path) -> None:
    from agent_runtime.agent import Agent
    from agent_runtime.harness import RolloutStore

    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=repo)
        turn_id = store.start_turn(
            thread_id=thread.thread_id,
            user_message="mutate",
            binding_manifest={"model_id": "test-model"},
        ).turn_id
        store.complete_turn(turn_id=turn_id, answer="done")
    (repo / "a.txt").write_text("a\n")
    agent = Agent(workspace_path=repo, checkpoint_db=tmp_path / "rollout.sqlite3")
    status = await agent.commit_status()
    assert status["changed"] == ["?? a.txt"]
    result = await agent.commit("facade commit")
    assert result["changed_file_count"] == 1
    assert result["created_branch"] is True
    assert (await agent.commit_status())["changed_file_count"] == 0


def _fake_gh(delivery: GitDelivery, existing: str | None = None) -> dict[str, str]:
    calls: dict[str, str] = {}
    delivery._require_gh = lambda: None  # type: ignore[method-assign]
    delivery._gh_default_branch = lambda: "main"  # type: ignore[method-assign]
    delivery._gh_view_pull_request = lambda branch: existing  # type: ignore[method-assign]

    def create(*, head: str, base: str, title: str, body: str) -> str:
        calls.update(head=head, base=base, title=title, body=body)
        return "https://github.com/owner/repo/pull/7"

    delivery._gh_create_pull_request = create  # type: ignore[method-assign]
    return calls


def _bare_remote(tmp_path: Path, name: str) -> Path:
    remote = tmp_path / name
    subprocess.run(
        ["git", "init", "--quiet", "--bare", "-b", "main", str(remote)], check=True,
    )
    return remote


async def test_commit_records_fork_base_for_pull_requests(repo: Path) -> None:
    delivery = GitDelivery(repo)
    (repo / "a.txt").write_text("a\n")
    result = await delivery.commit("fork point")
    recorded = _git(repo, "config", "--get", f"branch.{result.branch}.praxis-base").strip()
    assert recorded in {"master", "main"}


async def test_pull_request_flow_pushes_and_creates(repo: Path, tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path, "remote-github.com.git")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "HEAD")

    delivery = GitDelivery(repo)
    (repo / "a.txt").write_text("a\n")
    (repo / "base.txt").unlink()  # deletions must flow through the push
    commit = await delivery.commit("fix: timeout")
    calls = _fake_gh(delivery)

    status = await delivery.pull_request_status()
    assert status.branch == commit.branch
    assert status.base_branch in {"master", "main"}
    assert status.commit_subjects == ("fix: timeout",)
    assert sorted(status.changed_paths) == ["a.txt", "base.txt"]
    assert status.open_pull_request_url is None

    result = await delivery.open_pull_request()
    assert result.created is True
    assert result.url == "https://github.com/owner/repo/pull/7"
    assert calls["title"] == "fix: timeout"
    assert "fix: timeout" in calls["body"]
    # The remote branch matches the local commit exactly.
    local_sha = _git(repo, "rev-parse", "HEAD").strip()
    remote_sha = _git(repo, "rev-parse", f"origin/{commit.branch}").strip()
    assert remote_sha == local_sha


async def test_existing_pull_request_is_returned_without_creating(repo: Path, tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path, "remote-github.com.git")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    delivery = GitDelivery(repo)
    (repo / "a.txt").write_text("a\n")
    await delivery.commit("again")
    calls = _fake_gh(delivery, existing="https://github.com/owner/repo/pull/9")
    result = await delivery.open_pull_request()
    assert result.created is False
    assert result.url == "https://github.com/owner/repo/pull/9"
    assert calls == {}


async def test_pull_request_requires_praxis_branch(repo: Path, tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path, "remote-github.com.git")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    delivery = GitDelivery(repo)
    _fake_gh(delivery)
    with pytest.raises(DeliveryError, match="/commit"):
        await delivery.pull_request_status()


async def test_pull_request_requires_commits_beyond_base(repo: Path, tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path, "remote-github.com.git")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    _git(repo, "switch", "-q", "-c", "praxis/empty")
    delivery = GitDelivery(repo)
    _fake_gh(delivery)
    with pytest.raises(DeliveryError, match="没有提交"):
        await delivery.pull_request_status()


async def test_pull_request_refuses_non_github_remote(repo: Path, tmp_path: Path) -> None:
    remote = tmp_path / "plain-remote.git"
    _git(repo, "remote", "add", "origin", str(remote))
    delivery = GitDelivery(repo)
    (repo / "a.txt").write_text("a\n")
    await delivery.commit("some work")
    _fake_gh(delivery)
    with pytest.raises(DeliveryError, match="不是 GitHub"):
        await delivery.pull_request_status()


async def test_push_failure_surfaces_credential_hint(repo: Path, tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path, "rejecting-github.com.git")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    hook = remote / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    delivery = GitDelivery(repo)
    (repo / "a.txt").write_text("a\n")
    await delivery.commit("undeliverable")
    _fake_gh(delivery)
    with pytest.raises(DeliveryError, match="gh auth setup-git"):
        await delivery.open_pull_request()


async def test_agent_facade_open_pull_request_requires_idle_turns(repo: Path, tmp_path: Path) -> None:
    from agent_runtime.agent import Agent
    from agent_runtime.harness import RolloutStore

    remote = _bare_remote(tmp_path, "remote-github.com.git")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=repo)
        store.start_turn(
            thread_id=thread.thread_id,
            user_message="running",
            binding_manifest={"model_id": "test-model"},
        )
    (repo / "a.txt").write_text("a\n")
    agent = Agent(workspace_path=repo, checkpoint_db=tmp_path / "rollout.sqlite3")
    with pytest.raises(RuntimeError, match="未结束"):
        await agent.open_pull_request()


def _fake_checks(delivery: GitDelivery, states: list[tuple[str, str]]) -> None:
    """Patch gh wrappers; states mutates when the fake advances per poll."""
    delivery._require_gh = lambda: None  # type: ignore[method-assign]
    delivery._gh_view_pull_request = lambda branch: "https://github.com/o/r/pull/5"  # type: ignore[method-assign]

    def pr_checks(branch: str):
        if not states:
            raise DeliveryError(f"分支 {branch} 还没有 Pull Request；请先用 /pull 创建。")
        return tuple(
            CheckRun(name=name, state=state, workflow="CI", link="https://github.com/o/r/actions")
            for name, state in states
        )

    delivery._gh_pr_checks = pr_checks  # type: ignore[method-assign]


async def test_checks_report_terminal_states(repo: Path) -> None:
    delivery = GitDelivery(repo)
    _fake_checks(delivery, [("build", "pass"), ("test", "fail")])
    delivery._gh_failed_run_ids = lambda branch: ()  # type: ignore[method-assign]
    report = await delivery.pull_request_checks()
    assert report.overall == "fail"
    assert report.pull_request_url == "https://github.com/o/r/pull/5"
    assert tuple(c.name for c in report.checks) == ("build", "test")


async def test_checks_wait_polls_until_terminal(repo: Path) -> None:
    delivery = GitDelivery(repo)
    states = [("build", "pending")]
    _fake_checks(delivery, states)
    delivery._gh_failed_run_ids = lambda branch: ()  # type: ignore[method-assign]

    async def flip_to_pass() -> None:
        await asyncio.sleep(0.05)
        states[0] = ("build", "pass")

    flip = asyncio.create_task(flip_to_pass())
    report = await delivery.pull_request_checks(wait_seconds=5, poll_interval_seconds=0.01)
    await flip
    assert report.overall == "pass"


async def test_checks_wait_timeout_keeps_pending(repo: Path) -> None:
    delivery = GitDelivery(repo)
    _fake_checks(delivery, [("build", "pending")])
    report = await delivery.pull_request_checks(wait_seconds=0.05, poll_interval_seconds=0.01)
    assert report.overall == "pending"


async def test_checks_without_pull_request_is_refused(repo: Path) -> None:
    delivery = GitDelivery(repo)
    _fake_checks(delivery, [])
    with pytest.raises(DeliveryError, match="/pull"):
        await delivery.pull_request_checks()


async def test_checks_collect_failure_log_excerpts(repo: Path) -> None:
    delivery = GitDelivery(repo)
    _fake_checks(delivery, [("build", "fail")])
    delivery._gh_failed_run_ids = lambda branch: (("1234", "CI"), ("5678", "lint"))  # type: ignore[method-assign]
    delivery._gh_failed_run_logs = lambda run_id: f"{run_id}-log\n" * 5000  # type: ignore[method-assign]
    report = await delivery.pull_request_checks(include_failure_logs=True, max_log_bytes=4000)
    assert report.overall == "fail"
    assert [name for name, _logs in report.failure_logs] == ["CI", "lint"]
    total = sum(len(logs) for _name, logs in report.failure_logs)
    assert total <= 4000
    # Excerpts keep the tail of the log, where failures surface.
    assert report.failure_logs[0][1].endswith("1234-log")


async def test_checks_no_checks_configured_reports_none(repo: Path) -> None:
    delivery = GitDelivery(repo)
    delivery._require_gh = lambda: None  # type: ignore[method-assign]
    delivery._gh_view_pull_request = lambda branch: "https://github.com/o/r/pull/5"  # type: ignore[method-assign]
    delivery._gh_pr_checks = lambda branch: ()  # type: ignore[method-assign]
    report = await delivery.pull_request_checks()
    assert report.overall == "none"
    assert report.checks == ()


async def test_gh_pr_checks_parses_real_gh_json_states(repo: Path) -> None:
    import json as json_module
    import subprocess as subprocess_module

    delivery = GitDelivery(repo)
    delivery._require_gh = lambda: None  # type: ignore[method-assign]
    payload = json_module.dumps([
        {"name": "macos", "state": "SUCCESS", "bucket": "pass", "workflow": "ci", "link": "u"},
        {"name": "linux", "state": "FAILURE", "bucket": "fail", "workflow": "ci", "link": "u"},
        {"name": "lint", "state": "IN_PROGRESS", "bucket": "pending", "workflow": "ci", "link": "u"},
        {"name": "audit", "state": "CANCELLED", "bucket": "cancel", "workflow": "ci", "link": "u"},
    ])

    def fake_output(*arguments: str):
        return subprocess_module.CompletedProcess(
            arguments, 0, stdout=payload, stderr="",
        )

    delivery._gh_output = fake_output  # type: ignore[method-assign]
    checks = delivery._gh_pr_checks("praxis/x")
    assert tuple(check.state for check in checks) == ("pass", "fail", "pending", "cancel")
    from agent_runtime.git_delivery import PullRequestChecks
    report = PullRequestChecks(branch="praxis/x", pull_request_url=None, checks=checks)
    assert report.overall == "fail"


async def test_unknown_raw_check_state_is_never_reported_as_pass(repo: Path) -> None:
    import json as json_module
    import subprocess as subprocess_module

    delivery = GitDelivery(repo)
    delivery._require_gh = lambda: None  # type: ignore[method-assign]
    payload = json_module.dumps([
        # A future gh/GitHub status we have never seen, without a bucket field.
        {"name": "future", "state": "SOME_NEW_STATUS", "workflow": "ci", "link": "u"},
    ])

    def fake_output(*arguments: str):
        return subprocess_module.CompletedProcess(arguments, 0, stdout=payload, stderr="")

    delivery._gh_output = fake_output  # type: ignore[method-assign]
    checks = delivery._gh_pr_checks("praxis/x")
    assert checks[0].state == "pending"
    from agent_runtime.git_delivery import PullRequestChecks
    report = PullRequestChecks(branch="praxis/x", pull_request_url=None, checks=checks)
    assert report.overall == "pending"

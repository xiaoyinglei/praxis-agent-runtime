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
    # Pin the initial branch: runners without init.defaultBranch default to
    # master, and the faked gh default branch is "main".
    run(["git", "init", "--quiet", "-b", "main", str(root)], check=True)
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


@pytest.mark.parametrize("operation", ["pull-request", "failed-runs", "failed-log"])
async def test_github_query_failure_is_not_an_empty_result(repo: Path, operation: str) -> None:
    delivery = GitDelivery(repo)
    delivery._gh_output = lambda *args: subprocess.CompletedProcess(args, 4, "", "authentication required")
    with pytest.raises(DeliveryError, match="authentication required"):
        if operation == "pull-request":
            delivery._gh_view_pull_request("praxis/task")
        elif operation == "failed-runs":
            delivery._gh_failed_run_ids("praxis/task", "a" * 40)
        else:
            delivery._gh_failed_run_logs("123")


@pytest.mark.parametrize("operation", ["pull-request", "failed-runs"])
@pytest.mark.parametrize("payload", ["", "not-json", "{}", "[{}]"])
async def test_github_query_rejects_malformed_success(repo: Path, operation: str, payload: str) -> None:
    delivery = GitDelivery(repo)
    delivery._gh_output = lambda *args: subprocess.CompletedProcess(args, 0, payload, "")
    with pytest.raises(DeliveryError):
        if operation == "pull-request":
            delivery._gh_view_pull_request("praxis/task")
        else:
            delivery._gh_failed_run_ids("praxis/task", "a" * 40)


async def test_pull_request_absence_uses_successful_structured_query(repo: Path) -> None:
    delivery = GitDelivery(repo)
    calls = []

    def output(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "[]", "")

    delivery._gh_output = output
    assert delivery._gh_view_pull_request("praxis/task") is None
    assert calls == [("pr", "list", "--head", "praxis/task", "--state", "open", "--limit", "1", "--json", "url")]
    delivery._gh_output = lambda *args: subprocess.CompletedProcess(
        args, 0, '[{"url":"https://github.com/o/r/pull/7"}]', "",
    )
    assert delivery._gh_view_pull_request("praxis/task") == "https://github.com/o/r/pull/7"


@pytest.mark.parametrize("existing_delivery_branch", [False, True])
async def test_post_commit_timeout_reports_the_actual_advanced_ref(repo: Path, existing_delivery_branch: bool) -> None:
    if existing_delivery_branch:
        _git(repo, "checkout", "-q", "-b", "praxis/existing")
    before = _git(repo, "rev-parse", "HEAD").strip()
    original_index = (repo / ".git/index").read_bytes()
    (repo / "base.txt").write_text("approved change\n")
    hook = repo / ".git/hooks/post-commit"
    hook.write_text("#!/bin/sh\nsleep 0.5\n")
    hook.chmod(0o755)
    delivery = GitDelivery(repo)
    run_git = delivery._run_git

    def bounded_commit(*args, **kwargs):
        if args[0] == "commit":
            kwargs["timeout_seconds"] = 0.2
        return run_git(*args, **kwargs)

    delivery._run_git = bounded_commit
    with pytest.raises(DeliveryError) as failure:
        await delivery.commit("new change")
    assert failure.value.outcome == "partial"
    observed = _git(repo, "rev-parse", "HEAD").strip()
    assert observed != before
    assert failure.value.details["commit_id"] == observed
    assert _git(repo, "log", "-1", "--format=%s").strip() == "new change"
    assert (repo / ".git/index").read_bytes() == original_index


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
    from agent_runtime.harness import RuntimeState

    async with (await RuntimeState.open(tmp_path / "rollout.sqlite3")) as store:
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
    from agent_runtime.harness import RuntimeState

    async with (await RuntimeState.open(tmp_path / "rollout.sqlite3")) as store:
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
    from agent_runtime.harness import RuntimeState

    remote = _bare_remote(tmp_path, "remote-github.com.git")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    async with (await RuntimeState.open(tmp_path / "rollout.sqlite3")) as store:
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
    delivery._gh_pull_request_head = lambda branch: "a" * 40  # type: ignore[method-assign]

    def pr_checks(branch: str):
        if not states:
            raise DeliveryError(f"分支 {branch} 还没有 Pull Request；请先用 /pull 创建。")
        return tuple(
            CheckRun(name=name, state=state, workflow="CI", link="https://github.com/o/r/actions/runs/1234/job/1")
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
    delivery._gh_failed_run_ids = lambda branch, head: (("1234", "CI"), ("5678", "old"))  # type: ignore[method-assign]
    delivery._gh_failed_run_logs = lambda run_id: f"{run_id}-log\n" * 5000  # type: ignore[method-assign]
    report = await delivery.pull_request_checks(include_failure_logs=True, max_log_bytes=4000)
    assert report.overall == "fail"
    assert [name for name, _logs in report.failure_logs] == ["CI"]
    total = sum(len(logs) for _name, logs in report.failure_logs)
    assert total <= 4000
    # Excerpts keep the tail of the log, where failures surface.
    assert report.failure_logs[0][1].endswith("1234-log")


async def test_checks_no_checks_configured_reports_none(repo: Path) -> None:
    delivery = GitDelivery(repo)
    delivery._require_gh = lambda: None  # type: ignore[method-assign]
    delivery._gh_view_pull_request = lambda branch: "https://github.com/o/r/pull/5"  # type: ignore[method-assign]
    delivery._gh_pull_request_head = lambda branch: "a" * 40  # type: ignore[method-assign]
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


async def test_hung_git_subprocess_is_bounded_by_timeout(repo: Path) -> None:
    fake_git = repo.parent / "git-hang"
    fake_git.write_text("#!/bin/sh\nsleep 30\n")
    fake_git.chmod(0o755)
    delivery = GitDelivery(repo)
    delivery._git = str(fake_git)

    with pytest.raises(DeliveryError, match="git status 超过 0.2 秒未响应"):
        delivery._run_git("status", timeout_seconds=0.2)


async def test_hung_gh_subprocess_raises_delivery_error(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hanging_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            cmd="gh pr view", timeout=float(kwargs.get("timeout") or 0),
        )

    monkeypatch.setattr("agent_runtime.git_delivery.subprocess.run", hanging_run)
    delivery = GitDelivery(repo)

    with pytest.raises(DeliveryError, match="gh pr 超过 300 秒未响应"):
        delivery._gh_output("pr", "view")


async def test_delivery_scrubs_repository_environment(repo: Path, tmp_path: Path, monkeypatch) -> None:
    victim = tmp_path / "victim"
    subprocess.run(["git", "clone", "-q", str(repo), str(victim)], check=True)
    _git(victim, "config", "user.name", "Test")
    _git(victim, "config", "user.email", "test@example.com")
    before = _git(victim, "rev-parse", "HEAD")
    (repo / "new.txt").write_text("source change")
    monkeypatch.setenv("GIT_DIR", str(victim / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(repo))
    result = await GitDelivery(repo).commit("source only")
    monkeypatch.delenv("GIT_DIR")
    monkeypatch.delenv("GIT_WORK_TREE")
    assert _git(victim, "rev-parse", "HEAD") == before
    assert _git(repo, "rev-parse", "HEAD").strip() == result.commit_id


@pytest.mark.parametrize("change", ["new", "content", "index", "branch"])
async def test_commit_refuses_a_changed_preview(repo: Path, change: str) -> None:
    (repo / "base.txt").write_text("preview")
    delivery = GitDelivery(repo)
    preview = await delivery.status()
    if change == "new":
        (repo / "later.txt").write_text("not approved")
    elif change == "content":
        (repo / "base.txt").write_text("not approved")
    elif change == "index":
        _git(repo, "add", "base.txt")
    else:
        _git(repo, "switch", "-c", "other")
    with pytest.raises(DeliveryError, match="预览"):
        await delivery.commit("stale", expected_preview_id=preview.preview_id)
    assert _git(repo, "log", "-1", "--format=%s").strip() == "base"


async def test_failed_hook_preserves_original_branch_and_index(repo: Path) -> None:
    (repo / "base.txt").write_text("part staged")
    _git(repo, "add", "base.txt")
    (repo / "base.txt").write_text("unstaged later")
    (repo / "new.txt").write_text("unstaged new")
    index = (repo / ".git/index").read_bytes()
    original_branch = _git(repo, "branch", "--show-current")
    hook = repo / ".git/hooks/pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    with pytest.raises(DeliveryError):
        await GitDelivery(repo).commit("refused")
    assert _git(repo, "branch", "--show-current") == original_branch
    assert (repo / ".git/index").read_bytes() == index
    assert not _git(repo, "branch", "--list", "praxis/*").strip()
    assert (repo / "base.txt").read_text() == "unstaged later"


async def test_commit_cancellation_waits_for_actual_outcome(repo: Path) -> None:
    from agent_runtime.git_delivery import DeliveryInterruptedError

    (repo / "new.txt").write_text("commit")
    started, release = repo.parent / "started", repo.parent / "release"
    hook = repo / ".git/hooks/pre-commit"
    hook.write_text(f'#!/bin/sh\ntouch "{started}"\nwhile [ ! -e "{release}" ]; do sleep 0.02; done\n')
    hook.chmod(0o755)
    task = asyncio.create_task(GitDelivery(repo).commit("settled"))
    try:
        async with asyncio.timeout(5):
            while not started.exists():
                await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done(), "cancellation must wait until the owned Git operation settles"
    finally:
        release.touch()
    with pytest.raises(DeliveryInterruptedError) as interrupted:
        await task
    assert interrupted.value.result.commit_id == _git(repo, "rev-parse", "HEAD").strip()
    assert not (repo / ".git/index.lock").exists()


async def test_gh_query_failure_is_not_an_empty_success(repo: Path) -> None:
    delivery = GitDelivery(repo)
    delivery._gh_output = lambda *args: subprocess.CompletedProcess(args, 4, "", "HTTP 401")
    with pytest.raises(DeliveryError, match="401"):
        delivery._gh_pr_checks("praxis/task")


async def test_unchanged_preview_can_be_committed(repo: Path) -> None:
    (repo / "base.txt").write_text("approved")
    delivery = GitDelivery(repo)
    status = await delivery.status()
    result = await delivery.commit("approved", expected_preview_id=status.preview_id)
    assert _git(repo, "show", f"{result.commit_id}:base.txt") == "approved"


async def test_failure_logs_use_failed_checks_and_matching_head(repo: Path) -> None:
    import json

    from agent_runtime.git_delivery import CheckRun

    delivery = GitDelivery(repo)
    calls = []
    head = _git(repo, "rev-parse", "HEAD").strip()
    failed = (
        CheckRun("current", "fail", "CI", "https://github.com/o/r/actions/runs/123/job/1"),
        CheckRun("external", "fail", "external", "https://ci.example/run/999"),
    )
    def output(*args):
        calls.append(args)
        if args[:2] == ("run", "list"):
            return subprocess.CompletedProcess(args, 0, json.dumps([
                {"databaseId": 123, "name": "CI", "conclusion": "failure", "headSha": head},
                {"databaseId": 999, "name": "OLD", "conclusion": "failure", "headSha": "b" * 40},
            ]), "")
        return subprocess.CompletedProcess(args, 0, "current failure log", "")
    delivery._gh_output = output
    logs = delivery._failure_log_excerpts("praxis/task", 4000, head_commit=head, checks=failed)
    assert logs == (("CI", "current failure log"),)
    listing = next(args for args in calls if args[:2] == ("run", "list"))
    assert listing[listing.index("--commit") + 1] == head
    assert not any("999" in args for args in calls)


async def test_hook_cannot_substitute_an_unapproved_tree(repo: Path) -> None:
    original = _git(repo, "rev-parse", "HEAD")
    (repo / "base.txt").write_text("approved")
    hook = repo / ".git/hooks/pre-commit"
    hook.write_text("#!/bin/sh\nprintf unapproved > injected.txt\ngit add injected.txt\n")
    hook.chmod(0o755)
    index = (repo / ".git/index").read_bytes()
    delivery = GitDelivery(repo)
    preview = await delivery.status()
    with pytest.raises(DeliveryError, match="批准"):
        await delivery.commit("approved only", expected_preview_id=preview.preview_id)
    assert _git(repo, "rev-parse", "HEAD") == original
    assert _git(repo, "branch", "--show-current").strip() == "main"
    assert (repo / ".git/index").read_bytes() == index
    assert (repo / "injected.txt").read_text() == "unapproved"


async def test_publishing_refuses_a_changed_approved_commit(repo: Path, tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path, "github.com.git")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "main")
    delivery = GitDelivery(repo)
    (repo / "base.txt").write_text("approved")
    await delivery.commit("first")
    _fake_gh(delivery)
    preview = await delivery.pull_request_status()
    (repo / "base.txt").write_text("later")
    await delivery.commit("later")
    with pytest.raises(DeliveryError, match="预览"):
        await delivery.open_pull_request(expected_preview_id=preview.preview_id)
    assert not _git(remote, "branch", "--list", "praxis/*").strip()


async def test_install_does_not_unlink_another_gits_new_index_lock(repo: Path, monkeypatch) -> None:
    import os

    (repo / "base.txt").write_text("approved")
    replace = os.replace
    lock = repo / ".git/index.lock"
    def install(source, destination):
        replace(source, destination)
        if Path(destination) == repo / ".git/index":
            lock.write_bytes(b"another operation")
    monkeypatch.setattr("agent_runtime.git_delivery.os.replace", install)
    await GitDelivery(repo).commit("install")
    assert lock.read_bytes() == b"another operation"
    lock.unlink()


async def test_failed_commit_preserves_preexisting_branch_configuration(repo: Path) -> None:
    _git(repo, "config", "branch.praxis/feature.description", "user metadata")
    (repo / "base.txt").write_text("pending")
    hook = repo / ".git/hooks/pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    with pytest.raises(DeliveryError):
        await GitDelivery(repo).commit("failed", branch_name="praxis/feature")
    assert _git(repo, "config", "--get", "branch.praxis/feature.description").strip() == "user metadata"


async def test_pull_preview_content_and_identity_use_one_frozen_commit(repo: Path, tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path, "github.com-preview.git")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "main")
    delivery = GitDelivery(repo)
    (repo / "base.txt").write_text("first")
    initial = await delivery.commit("first")
    _fake_gh(delivery)
    def change_head_while_reading_pr(branch):
        (repo / "base.txt").write_text("later")
        _git(repo, "add", "base.txt")
        _git(repo, "commit", "-qm", "later")
        return None
    delivery._gh_view_pull_request = change_head_while_reading_pr
    preview = await delivery.pull_request_status()
    assert preview.commit_subjects == ("first",)
    assert preview.head_commit == initial.commit_id

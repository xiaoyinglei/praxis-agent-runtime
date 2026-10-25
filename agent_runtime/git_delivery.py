"""User-authorized commits and pull requests in the workspace's real repository.

Delivery is a host-side trusted operation in the same class as checkpoint
restore: it runs only after an explicit user request, shows the exact change
set before acting, and never exposes Git metadata to the model or the sandbox.
Unlike shadow checkpoints, these commits use the repository's own identity,
ignore rules and hooks — they are ordinary commits the user could have made.

Commits land on a ``praxis/*`` branch created from the current HEAD, so the
working tree is byte-identical before and after; only the branch pointer moves.
``/pull`` pushes that branch with plain ``git push`` (remote history matches the
local one exactly) and opens the pull request with the GitHub CLI's own token;
no agent-facing credential or network grant is involved.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

BRANCH_PREFIX = "praxis/"
BASE_CONFIG_SUFFIX = ".praxis-base"
_PULL_REQUEST_URL = re.compile(r"https://\S+/pull/\d+")
_IN_PROGRESS_MARKERS = (
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "rebase-merge",
    "rebase-apply",
    "BISECT_LOG",
)


class DeliveryError(RuntimeError):
    """A delivery operation was refused or failed; nothing was committed."""


@dataclass(frozen=True, slots=True)
class DeliveryStatus:
    branch: str
    changed: tuple[tuple[str, str], ...]  # (porcelain status, path)

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(path for _status, path in self.changed)


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    commit_id: str
    branch: str
    changed_file_count: int
    created_branch: bool


@dataclass(frozen=True, slots=True)
class PullRequestStatus:
    branch: str
    remote_url: str
    base_branch: str
    commit_subjects: tuple[str, ...]  # oldest first
    changed_paths: tuple[str, ...]
    uncommitted_count: int
    open_pull_request_url: str | None


@dataclass(frozen=True, slots=True)
class PullRequestResult:
    url: str
    branch: str
    created: bool


@dataclass(frozen=True, slots=True)
class CheckRun:
    name: str
    state: str  # pass | fail | pending | skipping | cancel
    workflow: str
    link: str


@dataclass(frozen=True, slots=True)
class PullRequestChecks:
    branch: str
    pull_request_url: str | None
    checks: tuple[CheckRun, ...]
    failure_logs: tuple[tuple[str, str], ...] = ()  # (workflow name, log excerpt)

    @property
    def overall(self) -> str:
        """pass | fail | pending | none — cancelled counts as failing."""
        if not self.checks:
            return "none"
        if any(check.state in {"fail", "cancel"} for check in self.checks):
            return "fail"
        if any(check.state == "pending" for check in self.checks):
            return "pending"
        return "pass"


class GitDelivery:
    """Commit workspace changes to the repository's own ``praxis/*`` branch."""

    def __init__(self, workspace_root: Path | str) -> None:
        self._workspace = Path(workspace_root).expanduser().resolve()
        self._git = shutil.which("git")

    async def status(self) -> DeliveryStatus:
        """Read the repository's current branch and pending change set."""
        return await asyncio.to_thread(self._status_sync)

    async def commit(
        self, message: str, *, branch_name: str | None = None,
    ) -> DeliveryResult:
        """Stage everything visible to the user and commit on a praxis branch."""
        return await asyncio.to_thread(self._commit_sync, message, branch_name)

    async def pull_request_status(self) -> PullRequestStatus:
        """Preview what a pull request from the current branch would contain."""
        return await asyncio.to_thread(self._pull_request_status_sync)

    async def open_pull_request(
        self, *, title: str | None = None, body: str | None = None,
        base_branch: str | None = None,
    ) -> PullRequestResult:
        """Push the current praxis branch and open (or return) its pull request."""
        return await asyncio.to_thread(self._open_pull_request_sync, title, body, base_branch)

    async def pull_request_checks(
        self,
        branch: str | None = None,
        *,
        include_failure_logs: bool = False,
        wait_seconds: float = 0.0,
        poll_interval_seconds: float = 20.0,
        max_log_bytes: int = 4000,
    ) -> PullRequestChecks:
        """Read a branch's PR checks; optionally wait until they reach a terminal state."""
        return await asyncio.to_thread(
            self._pull_request_checks_sync, branch,
            include_failure_logs, wait_seconds, poll_interval_seconds, max_log_bytes,
        )

    # --- Synchronous implementation --------------------------------------------------

    def _run_git(
        self, *arguments: str, check: bool = True,
        timeout_seconds: float = 120.0,
    ) -> subprocess.CompletedProcess[str]:
        # Every git/gh child process needs an upper bound: a hung fetch or
        # push would otherwise hang the whole delivery session.
        try:
            completed = subprocess.run(
                (self._git or "git", *arguments),
                cwd=self._workspace,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise DeliveryError(
                f"git {arguments[0]} 超过 {timeout_seconds:g} 秒未响应，已中止。"
            ) from error
        if check and completed.returncode != 0:
            raise DeliveryError(
                f"git {arguments[0]} failed ({completed.returncode}): "
                f"{completed.stderr.strip()[:500]}"
            )
        return completed

    def _require_repository(self) -> None:
        if self._git is None:
            raise DeliveryError("git executable not found on PATH")
        if not self._workspace.is_dir():
            raise DeliveryError(f"workspace does not exist: {self._workspace}")
        top = self._run_git("rev-parse", "--show-toplevel", check=False)
        if top.returncode != 0:
            raise DeliveryError("当前工作区不是 Git 仓库。")
        if Path(top.stdout.strip()).resolve() != self._workspace:
            raise DeliveryError(
                f"请在仓库根目录执行（仓库根：{top.stdout.strip()}）；子目录提交的范围不明确。"
            )

    def _require_idle_repository(self) -> None:
        git_dir = Path(self._run_git("rev-parse", "--git-dir").stdout.strip())
        if not git_dir.is_absolute():
            git_dir = self._workspace / git_dir
        busy = [name for name in _IN_PROGRESS_MARKERS if (git_dir / name).exists()]
        if busy:
            raise DeliveryError(f"仓库正在进行的操作未完成（{', '.join(busy)}），请先结束再提交。")

    def current_branch(self) -> str:
        """The repository's checked-out branch name ("HEAD" when detached)."""
        self._require_repository()
        return self._current_branch()

    def _current_branch(self) -> str:
        branch = self._run_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        return branch

    def _status_sync(self) -> DeliveryStatus:
        self._require_repository()
        branch = self._current_branch()
        porcelain = self._run_git("status", "--porcelain").stdout
        changed = []
        for line in porcelain.splitlines():
            if not line.strip():
                continue
            changed.append((line[:2].strip(), line[3:]))
        return DeliveryStatus(branch=branch, changed=tuple(changed))

    def _commit_sync(self, message: str, branch_name: str | None) -> DeliveryResult:
        if not isinstance(message, str) or not message.strip():
            raise DeliveryError("提交说明不能为空。")
        message = message.strip()
        if branch_name is not None:
            if not branch_name or branch_name.startswith("-") or any(c.isspace() for c in branch_name):
                raise DeliveryError("分支名不能为空、不能以 - 开头或包含空白。")
        self._require_repository()
        self._require_idle_repository()
        status = self._status_sync()
        if not status.changed:
            raise DeliveryError("没有可提交的变更。")

        created_branch = False
        if branch_name is not None:
            if not status.branch.startswith(BRANCH_PREFIX) or status.branch != branch_name:
                existing = self._run_git(
                    "rev-parse", "--verify", "--quiet", f"refs/heads/{branch_name}", check=False,
                )
                if existing.returncode == 0:
                    raise DeliveryError(f"分支已存在：{branch_name}")
            branch = branch_name
        elif status.branch.startswith(BRANCH_PREFIX):
            # Successive deliveries stack on the branch the user is already on.
            branch = status.branch
        else:
            branch = self._generated_branch_name()
        if status.branch != branch:
            self._run_git("switch", "-c", branch)
            # Remember where the delivery branch forked from; /pull targets it.
            # A branch stacked on another recorded branch (a managed worktree
            # branch, or an earlier delivery) inherits its base so /pull keeps
            # targeting the real upstream branch.
            inherited = self._run_git(
                "config", "--get", f"branch.{status.branch}{BASE_CONFIG_SUFFIX}", check=False,
            )
            fork_base = (inherited.stdout.strip()
                         if inherited.returncode == 0 and inherited.stdout.strip()
                         else status.branch)
            self._run_git("config", f"branch.{branch}{BASE_CONFIG_SUFFIX}", fork_base)
            created_branch = True
        # ``git add -A`` stages exactly what the confirmed preview showed:
        # the repository's own ignore rules decide what counts as a change.
        self._run_git("add", "-A")
        self._run_git("commit", "-m", message)
        commit_id = self._run_git("rev-parse", "HEAD").stdout.strip()
        return DeliveryResult(
            commit_id=commit_id,
            branch=branch,
            changed_file_count=len(status.changed),
            created_branch=created_branch,
        )

    def _generated_branch_name(self) -> str:
        base = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        name = f"{BRANCH_PREFIX}{base}"
        suffix = 1
        while self._run_git(
            "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False,
        ).returncode == 0:
            suffix += 1
            name = f"{BRANCH_PREFIX}{base}-{suffix}"
        return name

    # --- Pull request delivery -------------------------------------------------------

    def _require_gh(self) -> None:
        if shutil.which("gh") is None:
            raise DeliveryError("未找到 gh 命令；请安装 GitHub CLI 并运行 gh auth login。")

    def _gh_output(
        self, *arguments: str, timeout_seconds: float = 300.0,
    ) -> subprocess.CompletedProcess[str]:
        # gh talks to the GitHub API, so it is network-bound by nature and
        # gets a longer default budget than local git operations.
        try:
            completed = subprocess.run(
                ("gh", *arguments),
                cwd=self._workspace,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GH_NO_UPDATE_NOTIFIER": "1"},
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise DeliveryError(
                f"gh {arguments[0]} 超过 {timeout_seconds:g} 秒未响应，已中止。"
            ) from error
        return completed

    def _gh_default_branch(self) -> str:
        completed = self._gh_output("repo", "view", "--json", "defaultBranchRef",
                                    "--jq", ".defaultBranchRef.name")
        if completed.returncode != 0 or not completed.stdout.strip():
            raise DeliveryError(
                "无法读取仓库默认分支（gh repo view 失败）："
                f"{completed.stderr.strip()[:300]}"
            )
        return completed.stdout.strip()

    def _gh_view_pull_request(self, branch: str) -> str | None:
        completed = self._gh_output("pr", "view", branch, "--json", "url", "--jq", ".url")
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    def _gh_create_pull_request(
        self, *, head: str, base: str, title: str, body: str,
    ) -> str:
        completed = self._gh_output(
            "pr", "create", "--head", head, "--base", base,
            "--title", title, "--body", body,
        )
        match = _PULL_REQUEST_URL.search(completed.stdout)
        if completed.returncode != 0 or match is None:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise DeliveryError(f"创建 Pull Request 失败：{detail[:400]}")
        return match.group(0)

    def _pull_request_context(self) -> tuple[str, str, str]:
        """Return (branch, remote_url, base) after all hard preflights."""
        branch = self._current_branch()
        if not branch.startswith(BRANCH_PREFIX):
            raise DeliveryError(
                f"当前分支是 {branch or '(detached)'}；请先用 /commit 把变更提交到 praxis/* 分支。"
            )
        remote = self._run_git("remote", "get-url", "origin", check=False)
        if remote.returncode != 0 or not remote.stdout.strip():
            raise DeliveryError("仓库没有名为 origin 的远程。")
        remote_url = remote.stdout.strip()
        if "github.com" not in remote_url:
            raise DeliveryError(f"origin 不是 GitHub 仓库：{remote_url}")
        self._require_gh()
        recorded = self._run_git(
            "config", "--get", f"branch.{branch}{BASE_CONFIG_SUFFIX}", check=False,
        )
        base = recorded.stdout.strip() if recorded.returncode == 0 else ""
        if not base:
            base = self._gh_default_branch()
        if self._run_git(
            "rev-parse", "--verify", "--quiet", f"origin/{base}", check=False,
        ).returncode != 0:
            self._run_git("fetch", "origin", timeout_seconds=300.0)
        if self._run_git(
            "rev-parse", "--verify", "--quiet", f"origin/{base}", check=False,
        ).returncode != 0:
            raise DeliveryError(f"远程不存在基准分支 origin/{base}。")
        return branch, remote_url, base

    def _pull_request_status_sync(self) -> PullRequestStatus:
        self._require_repository()
        branch, remote_url, base = self._pull_request_context()
        subjects = tuple(
            line for line in self._run_git(
                "log", "--reverse", "--format=%s", f"origin/{base}..HEAD",
            ).stdout.splitlines() if line.strip()
        )
        if not subjects:
            raise DeliveryError(
                f"当前分支相对 origin/{base} 没有提交；请先用 /commit 提交变更。"
            )
        changed = tuple(
            line for line in self._run_git(
                "diff", "--name-only", f"origin/{base}...HEAD",
            ).stdout.splitlines() if line.strip()
        )
        uncommitted = len([
            line for line in self._run_git("status", "--porcelain").stdout.splitlines()
            if line.strip()
        ])
        return PullRequestStatus(
            branch=branch,
            remote_url=remote_url,
            base_branch=base,
            commit_subjects=subjects,
            changed_paths=changed,
            uncommitted_count=uncommitted,
            open_pull_request_url=self._gh_view_pull_request(branch),
        )

    def _open_pull_request_sync(
        self, title: str | None, body: str | None, base_override: str | None,
    ) -> PullRequestResult:
        self._require_repository()
        self._require_idle_repository()
        status = self._pull_request_status_sync()
        base = base_override or status.base_branch
        title = (title or status.commit_subjects[-1]).strip()
        if not title:
            raise DeliveryError("Pull Request 标题不能为空。")
        if body is None:
            lines = [f"- {subject}" for subject in status.commit_subjects]
            body = "\n".join(lines) + "\n\n—\n由 Praxis 交付"
        push = self._run_git(
            "push", "-u", "origin", status.branch, check=False, timeout_seconds=300.0,
        )
        if push.returncode != 0:
            raise DeliveryError(
                "推送失败："
                f"{(push.stderr or push.stdout).strip()[:400]}。"
                "可先运行 gh auth setup-git 让 GitHub CLI 提供 Git 凭据。"
            )
        existing = self._gh_view_pull_request(status.branch)
        if existing is not None:
            return PullRequestResult(url=existing, branch=status.branch, created=False)
        url = self._gh_create_pull_request(
            head=status.branch, base=base, title=title, body=body,
        )
        return PullRequestResult(url=url, branch=status.branch, created=True)

    # --- CI observation ----------------------------------------------------------------

    # ``gh pr checks --json`` reports the raw API status in ``state`` (for
    # example "SUCCESS", "IN_PROGRESS", "FAILURE") and the classification in
    # ``bucket``: pass | fail | pending | skipping | cancel. Classify from the
    # bucket, and never map an unknown raw state to pass.
    _RAW_STATE_BUCKETS = {
        "SUCCESS": "pass", "NEUTRAL": "pass", "SKIPPED": "skipping",
        "FAILURE": "fail", "TIMED_OUT": "fail", "ACTION_REQUIRED": "fail",
        "STARTUP_FAILURE": "fail", "CANCELLED": "cancel",
        "PENDING": "pending", "IN_PROGRESS": "pending", "QUEUED": "pending",
        "REQUESTED": "pending", "WAITING": "pending",
    }

    @classmethod
    def _classify_check_state(cls, entry: dict[str, object]) -> str:
        bucket = entry.get("bucket")
        if isinstance(bucket, str) and bucket in {"pass", "fail", "pending", "skipping", "cancel"}:
            return bucket
        raw = str(entry.get("state", ""))
        return cls._RAW_STATE_BUCKETS.get(raw, "pending")

    def _gh_pr_checks(self, branch: str) -> tuple[CheckRun, ...]:
        completed = self._gh_output(
            "pr", "checks", branch, "--json", "name,state,bucket,workflow,link",
        )
        try:
            entries = json.loads(completed.stdout) if completed.stdout.strip() else []
        except ValueError:
            entries = None
        if entries is not None:
            checks = []
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict):
                    continue
                checks.append(CheckRun(
                    name=str(entry.get("name", "")),
                    state=self._classify_check_state(entry),
                    workflow=str(entry.get("workflow", "")),
                    link=str(entry.get("link", "")),
                ))
            return tuple(checks)
        detail = completed.stderr.strip() or completed.stdout.strip()
        if "no pull request" in detail.lower():
            raise DeliveryError(f"分支 {branch} 还没有 Pull Request；请先用 /pull 创建。")
        if "no checks" in detail.lower():
            return ()
        raise DeliveryError(f"读取 PR 检查失败：{detail[:300]}")

    def _gh_failed_run_ids(self, branch: str) -> tuple[tuple[str, str], ...]:
        completed = self._gh_output(
            "run", "list", "--branch", branch, "--limit", "20",
            "--json", "databaseId,name,conclusion",
        )
        if completed.returncode != 0:
            return ()
        try:
            entries = json.loads(completed.stdout)
        except ValueError:
            return ()
        runs = []
        for entry in entries if isinstance(entries, list) else []:
            if (isinstance(entry, dict)
                    and entry.get("conclusion") in {"failure", "startup_failure"}):
                runs.append((
                    str(entry.get("databaseId", "")),
                    str(entry.get("name", "run")),
                ))
        return tuple(runs)

    def _gh_failed_run_logs(self, run_id: str) -> str:
        completed = self._gh_output("run", "view", run_id, "--log-failed")
        if completed.returncode != 0:
            return ""
        return completed.stdout

    def _pull_request_checks_sync(
        self,
        branch: str | None,
        include_failure_logs: bool,
        wait_seconds: float,
        poll_interval_seconds: float,
        max_log_bytes: int,
    ) -> PullRequestChecks:
        self._require_repository()
        resolved = branch or self._current_branch()
        if not resolved or resolved == "HEAD":
            raise DeliveryError("无法确定当前分支。")
        deadline = time.monotonic() + max(0.0, min(wait_seconds, 3600.0))
        interval = max(0.05, poll_interval_seconds)
        while True:
            checks = self._gh_pr_checks(resolved)
            terminal = bool(checks) and all(check.state != "pending" for check in checks)
            if terminal or time.monotonic() >= deadline:
                break
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        failure_logs: tuple[tuple[str, str], ...] = ()
        if include_failure_logs and any(check.state in {"fail", "cancel"} for check in checks):
            failure_logs = self._failure_log_excerpts(resolved, max_log_bytes)
        return PullRequestChecks(
            branch=resolved,
            pull_request_url=self._gh_view_pull_request(resolved),
            checks=checks,
            failure_logs=failure_logs,
        )

    def _failure_log_excerpts(
        self, branch: str, max_log_bytes: int,
    ) -> tuple[tuple[str, str], ...]:
        excerpts: list[tuple[str, str]] = []
        remaining = max(0, max_log_bytes)
        per_run_budget = max(200, max_log_bytes // 3)
        for run_id, name in self._gh_failed_run_ids(branch)[:3]:
            if remaining <= 0:
                break
            logs = self._gh_failed_run_logs(run_id).strip()
            if not logs:
                continue
            budget = min(remaining, per_run_budget)
            excerpt = logs[-budget:] if len(logs) > budget else logs
            excerpts.append((name, excerpt))
            remaining -= len(excerpt)
        return tuple(excerpts)

    def latest_delivery_branch(self) -> str | None:
        """Newest ``praxis/*`` branch by commit time, or None when there is none."""
        self._require_repository()
        listing = self._run_git(
            "for-each-ref", "--sort=-committerdate", "--count=1",
            "--format=%(refname:short)", f"refs/heads/{BRANCH_PREFIX}*",
            check=False,
        )
        name = listing.stdout.strip()
        return name or None


def compose_ci_failure_message(
    *,
    branch: str,
    pull_request_url: str | None,
    failed: Sequence[tuple[str, str]],
    failure_logs: Sequence[tuple[str, str]],
) -> str:
    """One canonical CI-failure report handed to the agent as a user message."""
    summary = "\n".join(f"- {name}（{workflow}）" for name, workflow in failed)
    log_lines: list[str] = []
    for workflow, logs in failure_logs:
        log_lines.append(f"[{workflow}]")
        log_lines += ["> " + line for line in logs[-2000:].splitlines()[-40:]]
    return (
        f"CI 失败（分支 {branch}，PR {pull_request_url}）。\n"
        "失败项：\n" + (summary or "- （未知）") + "\n"
        "失败日志摘录（可能截断）：\n" + ("\n".join(log_lines) or "> （无日志）") + "\n"
        "请定位失败原因并修复工作区代码；完成后提醒我运行 /commit 提交。"
    )


__all__ = [
    "BASE_CONFIG_SUFFIX",
    "BRANCH_PREFIX",
    "CheckRun",
    "DeliveryError",
    "DeliveryResult",
    "DeliveryStatus",
    "GitDelivery",
    "PullRequestChecks",
    "PullRequestResult",
    "PullRequestStatus",
    "compose_ci_failure_message",
]

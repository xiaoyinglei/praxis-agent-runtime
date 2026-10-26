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
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
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


_GIT_LOCATION_ENV = frozenset({
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CEILING_DIRECTORIES", "GIT_COMMON_DIR", "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_DIR", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE", "GIT_GRAFT_FILE", "GIT_NO_REPLACE_OBJECTS", "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX", "GIT_SHALLOW_FILE",
})


def git_environment() -> dict[str, str]:
    """Resolve Git identity from cwd, never inherited repository-local selectors."""
    return {**{key: value for key, value in os.environ.items() if key not in _GIT_LOCATION_ENV},
            "GIT_TERMINAL_PROMPT": "0"}


class DeliveryError(RuntimeError):
    """Delivery failed; outcome distinguishes safe refusal from partial effects."""

    def __init__(
        self, message: str, *, outcome: str = "not_committed", details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.details = {} if details is None else details


class DeliveryInterruptedError(DeliveryError):
    """Cancellation waited for the actual operation; its terminal result is available."""

    def __init__(self, result: object) -> None:
        super().__init__(f"取消请求到达时交付已在进行；操作已完成并核对结果：{result}", outcome="completed")
        self.result = result


async def _settle_delivery[T](operation: Callable[[], T]) -> T:
    # A thread cannot be cancelled. Shield ownership until subprocess completion,
    # then report the actual outcome instead of leaving a hidden background write.
    task = asyncio.create_task(asyncio.to_thread(operation))
    interrupted = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            interrupted = True
            if task.done():
                result = task.result()
                break
    if interrupted:
        raise DeliveryInterruptedError(result)
    return result


@dataclass(frozen=True, slots=True)
class DeliveryStatus:
    branch: str
    changed: tuple[tuple[str, str], ...]  # (porcelain status, path)
    preview_id: str = ""

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
    head_commit: str = ""
    preview_id: str = ""


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
    head_commit: str = ""

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
        self, message: str, *, branch_name: str | None = None, expected_preview_id: str | None = None,
    ) -> DeliveryResult:
        """Stage everything visible to the user and commit on a praxis branch."""
        return await _settle_delivery(lambda: self._commit_sync(message, branch_name, expected_preview_id))

    async def pull_request_status(self) -> PullRequestStatus:
        """Preview what a pull request from the current branch would contain."""
        return await asyncio.to_thread(self._pull_request_status_sync)

    async def open_pull_request(
        self, *, title: str | None = None, body: str | None = None,
        base_branch: str | None = None, expected_preview_id: str | None = None,
    ) -> PullRequestResult:
        """Push the current praxis branch and open (or return) its pull request."""
        return await _settle_delivery(
            lambda: self._open_pull_request_sync(title, body, base_branch, expected_preview_id),
        )

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
        timeout_seconds: float = 120.0, index: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        # Every git/gh child process needs an upper bound: a hung fetch or
        # push would otherwise hang the whole delivery session.
        try:
            completed = subprocess.run(
                (self._git or "git", *arguments),
                cwd=self._workspace,
                env={**git_environment(), **({"GIT_INDEX_FILE": str(index)} if index is not None else {})},
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

    def _index_path(self) -> Path:
        return Path(self._run_git("rev-parse", "--path-format=absolute", "--git-path", "index").stdout.strip())

    @contextmanager
    def _candidate_index(self) -> Iterator[Path]:
        index = self._index_path()
        with tempfile.TemporaryDirectory(prefix="praxis-delivery-", dir=index.parent) as directory:
            candidate = Path(directory) / "index"
            if index.exists():
                candidate.write_bytes(index.read_bytes())
            else:
                self._run_git("read-tree", "HEAD", index=candidate)
            self._run_git("add", "-A", index=candidate)
            yield candidate

    def _preview(self, candidate: Path) -> tuple[DeliveryStatus, str, str]:
        branch = self._current_branch()
        head = self._run_git("rev-parse", "HEAD").stdout.strip()
        tree = self._run_git("write-tree", index=candidate).stdout.strip()
        real_index = self._index_path()
        identity = {
            "git_dir": self._run_git("rev-parse", "--absolute-git-dir").stdout.strip(),
            "head": head, "branch": branch, "tree": tree,
            "index": hashlib.sha256(real_index.read_bytes() if real_index.exists() else b"").hexdigest(),
        }
        fields = self._run_git("status", "--porcelain", "-z", "--untracked-files=all").stdout.split("\0")
        changed = []
        offset = 0
        while offset < len(fields) and fields[offset]:
            line = fields[offset]
            changed.append((line[:2].strip(), line[3:]))
            offset += 2 if "R" in line[:2] or "C" in line[:2] else 1
        preview_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return DeliveryStatus(branch=branch, changed=tuple(changed), preview_id=preview_id), head, tree

    def _status_sync(self) -> DeliveryStatus:
        self._require_repository()
        with self._candidate_index() as candidate:
            return self._preview(candidate)[0]

    def _commit_sync(
        self, message: str, branch_name: str | None, expected_preview_id: str | None = None,
    ) -> DeliveryResult:
        if not isinstance(message, str) or not message.strip():
            raise DeliveryError("提交说明不能为空。")
        message = message.strip()
        if branch_name is not None:
            if not branch_name or branch_name.startswith("-") or any(c.isspace() for c in branch_name):
                raise DeliveryError("分支名不能为空、不能以 - 开头或包含空白。")
        self._require_repository()
        self._require_idle_repository()
        real_index = self._index_path()
        lock_path = real_index.with_name(real_index.name + ".lock")
        try:
            lock = lock_path.open("xb")
        except FileExistsError:
            raise DeliveryError("Git 暂存区正在被另一操作使用；请稍后重新预览。") from None
        installed = False
        try:
            with lock, self._candidate_index() as candidate:
                status, original_head, tree = self._preview(candidate)
                if expected_preview_id is not None and status.preview_id != expected_preview_id:
                    raise DeliveryError("工作区、暂存区或分支在预览后已改变；请重新预览并确认。")
                if not status.changed:
                    raise DeliveryError("没有可提交的变更。")
                if branch_name is not None:
                    branch = branch_name
                elif status.branch.startswith(BRANCH_PREFIX):
                    branch = status.branch
                else:
                    branch = self._generated_branch_name()
                created = branch != status.branch
                if created and self._run_git(
                    "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False,
                ).returncode == 0:
                    raise DeliveryError(f"分支已存在：{branch}")
                committed = False
                commit_attempted = False
                commit_id: str | None = None
                try:
                    if created:
                        self._run_git("branch", branch, original_head)
                        self._run_git("symbolic-ref", "HEAD", f"refs/heads/{branch}")
                    commit_attempted = True
                    self._run_git("commit", "-m", message, index=candidate)
                    committed = True
                    commit_id = self._run_git("rev-parse", "HEAD").stdout.strip()
                    if self._current_branch() != branch:
                        raise DeliveryError("并发操作改变了提交分支；请检查仓库。", outcome="partial",
                                            details={"commit_id": commit_id, "branch": branch})
                    if self._run_git("rev-parse", "HEAD^{tree}").stdout.strip() != tree:
                        # A hook is allowed to validate but cannot silently change
                        # the approved tree. CAS rolls back only our own new tip;
                        # user worktree and original index remain untouched.
                        reverted = self._run_git("update-ref", f"refs/heads/{branch}", original_head,
                                                 commit_id, check=False)
                        if reverted.returncode == 0:
                            committed = False
                            raise DeliveryError("Git hook 改变了已批准的提交内容；提交已撤回，请重新预览。")
                        raise DeliveryError("Git hook 改变了已批准内容且提交撤回未确认；请检查仓库。",
                                            outcome="partial", details={"commit_id": commit_id, "branch": branch})
                    if created and status.branch != "HEAD":
                        inherited = self._run_git(
                            "config", "--get", f"branch.{status.branch}{BASE_CONFIG_SUFFIX}", check=False,
                        )
                        base = inherited.stdout.strip() if inherited.returncode == 0 else status.branch
                        self._run_git("config", f"branch.{branch}{BASE_CONFIG_SUFFIX}", base)
                    # The ordinary index is replaced only after successful commit.
                    # The worktree is untouched, so edits made after the candidate
                    # froze remain visible as unstaged changes.
                    lock.write(candidate.read_bytes())
                    lock.flush()
                    os.fsync(lock.fileno())
                    os.replace(lock_path, real_index)
                    installed = True
                    return DeliveryResult(commit_id, branch, len(status.changed), created)
                except BaseException as error:
                    if commit_attempted and not committed:
                        # git may already have updated the ref before a post-
                        # commit hook times out. Process failure is not evidence
                        # that no commit happened; inspect the actual ref first.
                        try:
                            observed = self._run_git("rev-parse", "--verify", "--quiet",
                                                     f"refs/heads/{branch}", check=False)
                            if observed.returncode != 0:
                                raise DeliveryError("提交分支引用无法读取。")
                            observed_id = observed.stdout.strip()
                        except Exception as inspection:
                            raise DeliveryError(f"提交命令失败且引用状态无法确认：{inspection}", outcome="partial",
                                                details={"branch": branch, "original_head": original_head}) from error
                        if observed_id != original_head:
                            committed = True
                            commit_id = observed_id
                    if not committed:
                        try:
                            if created:
                                if status.branch == "HEAD":
                                    self._run_git("update-ref", "--no-deref", "HEAD", original_head)
                                else:
                                    self._run_git("symbolic-ref", "HEAD", f"refs/heads/{status.branch}")
                                self._run_git("update-ref", "-d", f"refs/heads/{branch}", original_head)
                        except Exception as rollback:
                            raise DeliveryError(f"提交失败且分支恢复未确认：{rollback}", outcome="partial",
                                                details={"branch": branch, "original_head": original_head}) from error
                    elif not isinstance(error, DeliveryError) or error.outcome != "partial":
                        raise DeliveryError(f"提交已发生，但后续状态安装失败：{error}", outcome="partial",
                                            details={"branch": branch, "commit_id": commit_id,
                                                     "original_head": original_head}) from error
                    raise
        finally:
            # After rename this pathname may already belong to another Git
            # operation. Never unlink a lock we no longer own.
            if not installed:
                lock_path.unlink(missing_ok=True)

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
                env={**git_environment(), "GH_NO_UPDATE_NOTIFIER": "1"},
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
        # A successful empty list means absence; `pr view` uses a failing exit
        # for both absence and transport/authentication errors.
        completed = self._gh_output("pr", "list", "--head", branch, "--state", "open",
                                    "--limit", "1", "--json", "url")
        if completed.returncode != 0:
            raise DeliveryError("读取 Pull Request 失败：" + completed.stderr.strip()[:300])
        try:
            entries = json.loads(completed.stdout)
        except ValueError as error:
            raise DeliveryError("读取 Pull Request 失败：返回数据不是有效 JSON。") from error
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict) or not isinstance(entry.get("url"), str) or not entry["url"]
            for entry in entries
        ):
            raise DeliveryError("读取 Pull Request 失败：返回数据缺少有效 URL。")
        if not entries:
            return None
        return str(entries[0]["url"])

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
        remote = self._run_git("remote", "get-url", "--push", "origin", check=False)
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
        head = self._run_git("rev-parse", "HEAD").stdout.strip()
        base_head = self._run_git("rev-parse", f"origin/{base}").stdout.strip()
        subjects = tuple(
            line for line in self._run_git(
                "log", "--reverse", "--format=%s", f"{base_head}..{head}",
            ).stdout.splitlines() if line.strip()
        )
        if not subjects:
            raise DeliveryError(
                f"当前分支相对 origin/{base} 没有提交；请先用 /commit 提交变更。"
            )
        changed = tuple(
            line for line in self._run_git(
                "diff", "--name-only", f"{base_head}...{head}",
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
            head_commit=head,
            preview_id=hashlib.sha256(json.dumps({"branch": branch, "head": head, "remote": remote_url,
                "base": base, "base_head": base_head},
                sort_keys=True).encode()).hexdigest(),
        )

    def _open_pull_request_sync(
        self, title: str | None, body: str | None, base_override: str | None,
        expected_preview_id: str | None = None,
    ) -> PullRequestResult:
        self._require_repository()
        self._require_idle_repository()
        status = self._pull_request_status_sync()
        if expected_preview_id is not None and status.preview_id != expected_preview_id:
            raise DeliveryError("发布对象在预览后已改变；请重新预览并确认。")
        base = base_override or status.base_branch
        title = (title or status.commit_subjects[-1]).strip()
        if not title:
            raise DeliveryError("Pull Request 标题不能为空。")
        if body is None:
            lines = [f"- {subject}" for subject in status.commit_subjects]
            body = "\n".join(lines) + "\n\n—\n由 Praxis 交付"
        push = self._run_git(
            "push", status.remote_url, f"{status.head_commit}:refs/heads/{status.branch}",
            check=False, timeout_seconds=300.0,
        )
        if push.returncode != 0:
            raise DeliveryError(
                "推送失败："
                f"{(push.stderr or push.stdout).strip()[:400]}。"
                "可先运行 gh auth setup-git 让 GitHub CLI 提供 Git 凭据。"
            )
        # A URL freezes the approved destination. Git only refreshes tracking
        # refs automatically when pushing through a remote name; retain that
        # bookkeeping when origin still denotes this exact repository.
        current_remote = self._run_git("remote", "get-url", "origin", check=False)
        if current_remote.returncode == 0 and current_remote.stdout.strip() == status.remote_url:
            tracking = f"refs/remotes/origin/{status.branch}"
            before = self._run_git("rev-parse", "--verify", "--quiet", tracking, check=False)
            expected = before.stdout.strip() if before.returncode == 0 else "0" * len(status.head_commit)
            self._run_git("update-ref", tracking, status.head_commit, expected, check=False)
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
            entries = json.loads(completed.stdout) if completed.stdout.strip() else None
        except ValueError:
            entries = None
        if isinstance(entries, list) and (completed.returncode == 0 or
                                         bool(entries) and completed.returncode in {1, 8}):
            # gh documents exit 1 for failed checks and 8 for pending checks;
            # transport/auth failures without a valid result are query failures.
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

    def _gh_pull_request_head(self, branch: str) -> str:
        completed = self._gh_output("pr", "view", branch, "--json", "headRefOid", "--jq", ".headRefOid")
        head = completed.stdout.strip()
        if completed.returncode != 0 or re.fullmatch(r"[0-9a-f]{40,64}", head) is None:
            raise DeliveryError("无法确认 Pull Request 的提交身份：" + completed.stderr.strip()[:300])
        return head

    def _gh_failed_run_ids(self, branch: str, head_commit: str) -> tuple[tuple[str, str], ...]:
        completed = self._gh_output(
            "run", "list", "--branch", branch, "--commit", head_commit, "--limit", "100",
            "--json", "databaseId,name,conclusion,headSha",
        )
        if completed.returncode != 0:
            raise DeliveryError("读取失败工作流失败：" + completed.stderr.strip()[:300])
        try:
            entries = json.loads(completed.stdout)
        except ValueError as error:
            raise DeliveryError("读取失败工作流失败：返回数据不是有效 JSON。") from error
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict) or type(entry.get("databaseId")) is not int
            or not isinstance(entry.get("headSha"), str) or "conclusion" not in entry
            or entry["conclusion"] is not None and not isinstance(entry["conclusion"], str)
            for entry in entries
        ):
            raise DeliveryError("读取失败工作流失败：返回数据缺少工作流身份。")
        failures = {"failure", "startup_failure", "timed_out", "cancelled", "action_required"}
        return tuple(
            (str(entry["databaseId"]), str(entry.get("name", "run")))
            for entry in entries if isinstance(entry, dict) and entry.get("headSha") == head_commit
            and entry.get("conclusion") in failures and isinstance(entry.get("databaseId"), int)
        )

    def _gh_failed_run_logs(self, run_id: str) -> str:
        completed = self._gh_output("run", "view", run_id, "--log-failed")
        if completed.returncode != 0:
            raise DeliveryError(f"读取工作流 {run_id} 的失败日志失败：" + completed.stderr.strip()[:300])
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
        head_commit = self._gh_pull_request_head(resolved)
        deadline = time.monotonic() + max(0.0, min(wait_seconds, 3600.0))
        interval = max(0.05, poll_interval_seconds)
        while True:
            checks = self._gh_pr_checks(resolved)
            if self._gh_pull_request_head(resolved) != head_commit:
                raise DeliveryError("Pull Request 在观察期间更新了提交；请重新检查，未使用混合版本的 CI 结果。")
            terminal = bool(checks) and all(check.state != "pending" for check in checks)
            if terminal or time.monotonic() >= deadline:
                break
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        failure_logs: tuple[tuple[str, str], ...] = ()
        if include_failure_logs and any(check.state in {"fail", "cancel"} for check in checks):
            failure_logs = self._failure_log_excerpts(
                resolved, max_log_bytes, head_commit=head_commit, checks=checks,
            )
        return PullRequestChecks(
            branch=resolved,
            pull_request_url=self._gh_view_pull_request(resolved),
            checks=checks,
            failure_logs=failure_logs,
            head_commit=head_commit,
        )

    def _failure_log_excerpts(
        self, branch: str, max_log_bytes: int, *, head_commit: str, checks: Sequence[CheckRun],
    ) -> tuple[tuple[str, str], ...]:
        excerpts: list[tuple[str, str]] = []
        remaining = max(0, max_log_bytes)
        per_run_budget = max(200, max_log_bytes // 3)
        # External checks have no GitHub Actions run identity; do not guess one
        # from a similarly named workflow or from a previous branch failure.
        referenced = {match.group(1) for check in checks if check.state in {"fail", "cancel"}
                      and (match := re.search(r"/actions/runs/(\d+)(?:/|$)", check.link)) is not None}
        runs = [(run_id, name) for run_id, name in self._gh_failed_run_ids(branch, head_commit)
                if run_id in referenced]
        for run_id, name in runs[:3]:
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
    head_commit: str | None = None,
) -> str:
    """One canonical CI-failure report handed to the agent as a user message."""
    summary = "\n".join(f"- {name}（{workflow}）" for name, workflow in failed)
    log_lines: list[str] = []
    for workflow, logs in failure_logs:
        log_lines.append(f"[{workflow}]")
        log_lines += ["> " + line for line in logs[-2000:].splitlines()[-40:]]
    return (
        f"CI 失败（分支 {branch}，PR {pull_request_url}，提交 {head_commit or '未提供'}）。\n"
        "失败项：\n" + (summary or "- （未知）") + "\n"
        "失败日志摘录（可能截断）：\n" + ("\n".join(log_lines) or "> （无日志）") + "\n"
        "请定位失败原因并修复工作区代码；完成后提醒我运行 /commit 提交。"
    )


__all__ = [
    "BASE_CONFIG_SUFFIX",
    "BRANCH_PREFIX",
    "CheckRun",
    "DeliveryError",
    "DeliveryInterruptedError",
    "DeliveryResult",
    "DeliveryStatus",
    "GitDelivery",
    "PullRequestChecks",
    "PullRequestResult",
    "PullRequestStatus",
    "compose_ci_failure_message",
]

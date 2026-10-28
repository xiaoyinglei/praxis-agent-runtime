"""Managed Git worktrees: one isolated checkout per parallel session.

Two terminal windows starting sessions in the same project directory otherwise
edit one shared file tree: window A's half-finished edits become window B's
input, and acceptance runs validate a blend of both. A managed worktree gives
each session its own checkout and branch, and — because the Session records
the worktree as its workspace — the existing workspace permission domain
confines every tool to that checkout. No per-tool isolation rules are added.

The layout follows Codex (worktrees outside the repository in bucketed
directories, with provenance metadata) while branch naming follows Claude Code
(a readable ``worktree/<name>`` branch created from a base ref):

    <root>/<bucket>/<repo-name>/        root defaults to ~/.praxis/worktrees
        .praxis/worktree.json           provenance: name, branch, base, source

Delivery stays unchanged: ``/commit`` inside a worktree stacks a ``praxis/*``
branch exactly as in the main checkout. The worktree branch records the real
upstream as its ``.praxis-base``, and delivery branches inherit that base, so
``/pull`` from inside a worktree targets the same upstream branch as usual.
"""

from __future__ import annotations

import builtins
import fnmatch
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent_runtime.git_delivery import BASE_CONFIG_SUFFIX, git_environment
from agent_runtime.harness.ownership import ConversationBusyError, execution_owner, workspace_owner
from agent_runtime.harness.state import RuntimeState
from agent_runtime.harness.workspace_stores import registered_workspace_stores

WORKTREE_BRANCH_PREFIX = "worktree/"
PROVENANCE_REVISION = 1
DEFAULT_KEEP_COUNT = 15
WORKTREE_ROOT_ENV = "PRAXIS_WORKTREE_ROOT"
INCLUDE_FILENAME = ".worktreeinclude"
PROVENANCE_RELATIVE_PATH = Path(".praxis") / "worktree.json"
_MAX_INCLUDE_FILES = 200
_MAX_INCLUDE_BYTES = 64 * 1024 * 1024
_NAME_PATTERN = re.compile(r"[^a-z0-9._-]+")


class WorktreeError(RuntimeError):
    """A managed worktree operation was refused or failed."""


@dataclass(frozen=True, slots=True)
class ManagedWorktree:
    name: str
    root: Path
    branch: str
    base_commit: str
    source_root: Path
    created_at: str

    @classmethod
    def from_provenance(cls, root: Path, record: dict[str, object]) -> ManagedWorktree:
        return cls(
            name=str(record["name"]),
            root=root,
            branch=str(record["branch"]),
            base_commit=str(record["base_commit"]),
            source_root=Path(str(record["source_root"])),
            created_at=str(record["created_at"]),
        )

    def provenance(self) -> dict[str, object]:
        return {
            "version": PROVENANCE_REVISION,
            "name": self.name,
            "branch": self.branch,
            "base_commit": self.base_commit,
            "source_root": str(self.source_root),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class CreateReport:
    worktree: ManagedWorktree
    uncommitted_in_source: int
    included_files: tuple[str, ...]  # relative paths copied from .worktreeinclude


@dataclass(frozen=True, slots=True)
class WorktreeInspection:
    worktree: ManagedWorktree
    exists: bool
    dirty_count: int | None  # None when the checkout cannot be inspected
    ahead_count: int | None  # commits on the branch beyond its recorded base


@dataclass(frozen=True, slots=True)
class RemoveReport:
    name: str
    root: Path
    branch: str
    branch_deleted: bool


class WorktreeManager:
    """Create, inspect and remove managed worktrees of one repository."""

    def __init__(self, source: Path | str, *, root: Path | str | None = None) -> None:
        self._source = Path(source).expanduser().resolve()
        self._git = shutil.which("git")
        configured = root if root is not None else os.environ.get(WORKTREE_ROOT_ENV)
        base = Path(configured).expanduser() if configured else Path.home() / ".praxis" / "worktrees"
        self._root = base.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def create(self, name: str | None = None, *, base: str | None = None) -> CreateReport:
        """Create a worktree checkout on ``worktree/<name>`` from a base ref.

        The base defaults to the source repository's HEAD commit, so uncommitted
        changes in the main checkout are never part of the worktree; the report
        counts them so the caller can surface the exclusion.
        """
        if self._git is None:
            raise WorktreeError("未找到 git 命令；请安装 Git 后重试。")
        if not self._source.is_dir():
            raise WorktreeError(f"目录不存在：{self._source}")
        source_root = self._repository_root()
        if (source_root / PROVENANCE_RELATIVE_PATH).is_file():
            raise WorktreeError("当前已在 agent worktree 中，不能嵌套创建；请回到主工作区。")
        revision = base if base is not None else "HEAD"
        base_commit = self._git_stdout(
            source_root, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}",
            what="读取基准提交",
        )
        source_branch = self._git_stdout(
            source_root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False,
        )
        delivery_base = source_branch
        if base is not None:
            explicit_ref = self._git_stdout(source_root, "rev-parse", "--symbolic-full-name",
                                            "--verify", "--end-of-options", base, check=False)
            delivery_base = (explicit_ref.removeprefix("refs/heads/")
                             if explicit_ref.startswith("refs/heads/") else "")
        uncommitted = len(self._dirty_lines(source_root))
        bucket = self._allocate_bucket()
        worktree_root = bucket / source_root.name
        branch = self._available_branch(_sanitize_name(name))
        created = self._run_git(
            source_root, "worktree", "add", "-b", branch, str(worktree_root), base_commit,
            what="创建 worktree", timeout=300,
        )
        if created.returncode != 0:
            _remove_empty_bucket(worktree_root)
            raise WorktreeError(f"创建 worktree 失败：{created.stderr.strip()[:400]}")
        try:
            managed = ManagedWorktree(
                name=branch[len(WORKTREE_BRANCH_PREFIX):],
                root=worktree_root,
                branch=branch,
                base_commit=base_commit,
                source_root=source_root,
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            _write_provenance(managed)
            if delivery_base:
                # Delivery branches forked from this worktree inherit the real
                # upstream; GitDelivery reads this key when stacking a praxis branch.
                self._run_git(
                    source_root, "config",
                    f"branch.{branch}{BASE_CONFIG_SUFFIX}", delivery_base,
                    what="记录 worktree 基准分支", check=False,
                )
            included = self._copy_included_files(source_root, worktree_root)
        except BaseException:
            self._run_git(
                source_root, "worktree", "remove", "--force", str(worktree_root),
                check=False, what="回滚未完成的 worktree", timeout=120,
            )
            _remove_empty_bucket(worktree_root)
            raise
        self.prune()
        return CreateReport(
            worktree=managed, uncommitted_in_source=uncommitted, included_files=included,
        )

    def list(self) -> list[ManagedWorktree]:
        """Return this repository's managed worktrees, newest first."""
        entries: list[ManagedWorktree] = []
        if not self._root.is_dir():
            return entries
        repository = self._common_git_directory(self._source)
        for provenance in self._root.glob(f"*/*/{PROVENANCE_RELATIVE_PATH.as_posix()}"):
            try:
                record = json.loads(provenance.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(record, dict) or record.get("version") != PROVENANCE_REVISION:
                continue
            if not isinstance(record.get("name"), str) or not record.get("branch"):
                continue
            try:
                entry = ManagedWorktree.from_provenance(provenance.parent.parent, record)
            except (KeyError, TypeError):
                continue
            checkout = entry.root if entry.root.is_dir() else entry.source_root
            try:
                checkout_repository = self._common_git_directory(checkout)
            except WorktreeError:
                # A stale foreign checkout must not block this repository.
                # Unverifiable entries are never deletion candidates.
                continue
            if checkout_repository == repository:
                entries.append(entry)
        entries.sort(key=lambda entry: (entry.created_at, entry.name), reverse=True)
        return entries

    def find(self, name: str) -> ManagedWorktree:
        """Resolve one worktree by name or by its checkout path."""
        candidate = Path(name).expanduser()
        if (candidate / PROVENANCE_RELATIVE_PATH).is_file():
            matches = [entry for entry in self.list() if entry.root.resolve() == candidate.resolve()]
            if matches:
                return matches[0]
            raise WorktreeError("该路径不是当前仓库的托管 worktree。")
        matches = [entry for entry in self.list() if entry.name == name]
        if not matches:
            raise WorktreeError(f"没有名为 {name} 的 worktree；可用：{self._known_names()}")
        if len(matches) > 1:
            listed = ", ".join(str(entry.root) for entry in matches)
            raise WorktreeError(f"名称 {name} 对应多个 worktree（{listed}）；请改用完整路径。")
        return matches[0]

    def inspect(self, worktree: ManagedWorktree) -> WorktreeInspection:
        """Read live dirty/ahead counts; unreadable checkouts report None.

        ``ahead`` counts commits on the worktree's own branch beyond its base.
        Deliveries switch the checkout to a ``praxis/*`` branch; they leave the
        worktree branch at the base, so a delivered worktree stays removable
        while a direct commit on the branch keeps it protected.
        """
        if not worktree.root.is_dir():
            return WorktreeInspection(worktree, exists=False, dirty_count=None, ahead_count=None)
        try:
            dirty = len(self._dirty_lines(worktree.root, include_ignored=True))
        except WorktreeError:
            dirty = None
        ahead_output = self._run_git(
            worktree.root, "rev-list", "--count",
            f"{worktree.base_commit}..refs/heads/{worktree.branch}",
            check=False, what="统计领先提交",
        )
        ahead = int(ahead_output.stdout.strip()) if ahead_output.returncode == 0 else None
        return WorktreeInspection(worktree, exists=True, dirty_count=dirty, ahead_count=ahead)

    def remove(self, name_or_root: str, *, force: bool = False) -> RemoveReport:
        """Remove a worktree; dirty or unmerged work requires explicit force."""
        entry = self.find(name_or_root)
        try:
            with workspace_owner(entry.root, exclusive=True), ExitStack() as locks:
                # Fence older workers that hold only a per-Thread lock.
                databases = set(registered_workspace_stores(entry.root))
                # Preserve compatibility with pre-inventory installations.
                databases.add(entry.root / ".praxis/checkpoints.sqlite")
                for database in sorted(databases):
                    if not database.exists():
                        continue
                    with RuntimeState.open_sync(database, read_only=True) as store:
                        for thread in store.list_threads():
                            locks.enter_context(execution_owner(database, thread.thread_id))
                        if not force and any(
                            turn.status in {"running", "paused", "interrupted"} for turn in store.list_turns()
                        ):
                            raise WorktreeError("worktree 存在未结束的任务；结束任务后重试，或用 --force 放弃。")
                if not force:
                    self._assert_known_runtime_state(entry, databases)
                return self._remove_idle(entry, force=force)
        except ConversationBusyError:
            raise WorktreeError("worktree 仍被会话使用；请先关闭会话，不能强制删除。") from None
        except (OSError, sqlite3.Error, ValueError) as error:
            raise WorktreeError(f"无法安全检查 worktree：{error}") from error

    def _assert_known_runtime_state(self, entry: ManagedWorktree, databases: set[Path]) -> None:
        known = {entry.root / PROVENANCE_RELATIVE_PATH}
        known.update(Path(str(database) + suffix) for database in databases for suffix in ("", "-wal", "-shm"))
        companion_roots = {database.with_name(database.name + suffix) for database in databases
                           for suffix in (".artifacts", ".locks")}
        for path in (entry.root / ".praxis").rglob("*"):
            if not path.is_file() and not path.is_symlink():
                continue
            if path in known or any(root in path.parents for root in companion_roots):
                continue
            raise WorktreeError(f"worktree 保留了未登记的运行状态 {path.relative_to(entry.root)}；"
                                "确认放弃后使用 --force。")

    def _remove_idle(self, entry: ManagedWorktree, *, force: bool) -> RemoveReport:
        inspection = self.inspect(entry)
        # An unreadable ahead count is not evidence of safety: require force.
        protected = inspection.exists and (
            inspection.dirty_count is None or bool(inspection.dirty_count) or inspection.ahead_count is None
            or bool(inspection.ahead_count)
        )
        if protected and not force:
            details = []
            if inspection.dirty_count:
                details.append(f"{inspection.dirty_count} 个未提交变更")
            if inspection.ahead_count:
                details.append(f"{inspection.ahead_count} 个未合并提交")
            raise WorktreeError(
                f"worktree {entry.name} 有{'、'.join(details)}；确认放弃后使用 --force 删除。"
            )
        cwd = entry.source_root if entry.source_root.is_dir() else entry.root
        # --force is always passed at the Git level: the protection decision was
        # already made above, and Git would otherwise refuse over this runtime's
        # own untracked .praxis state inside an otherwise clean worktree.
        removed = self._run_git(
            cwd, "worktree", "remove", "--force", str(entry.root),
            check=False, what="删除 worktree", timeout=120,
        )
        if removed.returncode != 0 and inspection.exists:
            raise WorktreeError(f"删除 worktree 失败：{removed.stderr.strip()[:400]}")
        self._run_git(cwd, "worktree", "prune", check=False, what="清理 worktree 记录")
        branch_deleted = self._delete_branch(entry, force=force)
        _remove_empty_bucket(entry.root)
        return RemoveReport(name=entry.name, root=entry.root, branch=entry.branch,
                            branch_deleted=branch_deleted)

    def prune(self, *, keep: int = DEFAULT_KEEP_COUNT) -> builtins.list[RemoveReport]:
        """Remove the oldest clean worktrees beyond ``keep``; never forced."""
        removed: list[RemoveReport] = []
        excess = self.list()[max(0, keep):]
        for entry in excess:
            try:
                removed.append(self.remove(str(entry.root)))
            except WorktreeError:
                continue
        return removed

    # --- Internals ---------------------------------------------------------------------

    def _common_git_directory(self, cwd: Path) -> Path:
        result = self._git_stdout(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir",
                                  what="读取仓库身份")
        return Path(result).resolve()

    def _repository_root(self) -> Path:
        completed = self._run_git(
            self._source, "rev-parse", "--show-toplevel", check=False, what="定位仓库根",
        )
        if completed.returncode != 0:
            raise WorktreeError(f"当前目录不是 Git 仓库：{self._source}")
        return Path(completed.stdout.strip()).resolve()

    def _allocate_bucket(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            marker = self._root / ".metadata_never_index"
            if not marker.exists():
                marker.write_bytes(b"")
        while True:
            bucket = self._root / uuid.uuid4().hex[:4]
            try:
                bucket.mkdir()
            except FileExistsError:
                continue
            return bucket

    def _available_branch(self, name: str) -> str:
        base = f"{WORKTREE_BRANCH_PREFIX}{name or f'session-{uuid.uuid4().hex[:8]}'}"
        branch, suffix = base, 1
        while self._run_git(
            self._source, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}",
            check=False, what="检查分支占用",
        ).returncode == 0:
            suffix += 1
            branch = f"{base}-{suffix}"
        return branch

    def _delete_branch(self, entry: ManagedWorktree, *, force: bool) -> bool:
        cwd = entry.source_root if entry.source_root.is_dir() else self._root
        exists = self._run_git(
            cwd, "rev-parse", "--verify", "--quiet", f"refs/heads/{entry.branch}",
            check=False, what="检查分支",
        ).returncode == 0
        if not exists:
            return True
        deleted = self._run_git(
            cwd, "branch", "-D" if force else "-d", entry.branch,
            check=False, what="删除分支",
        )
        return deleted.returncode == 0

    def _copy_included_files(self, source_root: Path, worktree_root: Path) -> tuple[str, ...]:
        """Copy gitignored files named by ``.worktreeinclude`` into the worktree.

        A fresh checkout has no local secrets or caches; patterns use gitignore
        syntax and only files that are both matched and actually ignored are
        copied, mirroring Claude Code's ``.worktreeinclude``.
        """
        listing = source_root / INCLUDE_FILENAME
        if not listing.is_file():
            return ()
        patterns = tuple(
            line.strip() for line in listing.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if not patterns:
            return ()
        ignored = self._git_lines(
            source_root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z",
            what="列出被忽略文件", timeout=120,
        )
        matched = [path for path in ignored if _include_match(path, patterns)]
        total = sum((source_root / path).stat().st_size for path in matched
                    if (source_root / path).is_file())
        if len(matched) > _MAX_INCLUDE_FILES or total > _MAX_INCLUDE_BYTES:
            raise WorktreeError(
                f".worktreeinclude 匹配到 {len(matched)} 个文件（{total} 字节），超过复制上限；"
                "请缩小匹配范围。"
            )
        for relative in matched:
            source = source_root / relative
            if not source.is_file():
                continue
            destination = worktree_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return tuple(matched)

    def _dirty_lines(self, cwd: Path, *, include_ignored: bool = False) -> builtins.list[str]:
        """Count pending changes, excluding this runtime's own state directory.

        ``.praxis`` holds provenance and session databases exactly as in the
        main checkout; counting it would mark every fresh worktree dirty.
        """
        arguments = ["status", "--porcelain", "--untracked-files=all"]
        if include_ignored:
            arguments.append("--ignored=matching")
        fields = self._git_lines(cwd, *arguments, "-z", what="读取工作区状态")
        lines = []
        index = 0
        while index < len(fields):
            line = fields[index]
            index += 2 if "R" in line[:2] or "C" in line[:2] else 1
            if Path(line[3:]).parts[:1] != (".praxis",):
                lines.append(line)
        return lines

    def _known_names(self) -> str:
        names = [entry.name for entry in self.list()]
        return ", ".join(names) if names else "（无）"

    def _run_git(
        self, cwd: Path, *arguments: str, check: bool = True, what: str, timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        if self._git is None:
            raise WorktreeError("未找到 git 命令；请安装 Git 后重试。")
        environment = git_environment()
        try:
            completed = subprocess.run(
                (self._git, *arguments), cwd=cwd, env=environment,
                capture_output=True, text=True, check=False, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise WorktreeError(f"{what}超时（git {' '.join(arguments[:2])}）。") from None
        if check and completed.returncode != 0:
            raise WorktreeError(
                f"{what}失败：{completed.stderr.strip()[:400] or completed.stdout.strip()[:400]}"
            )
        return completed

    def _git_stdout(
        self, cwd: Path, *arguments: str, check: bool = True, what: str = "git",
    ) -> str:
        return self._run_git(cwd, *arguments, check=check, what=what).stdout.strip()

    def _git_lines(
        self, cwd: Path, *arguments: str, check: bool = True, what: str, timeout: float = 60,
    ) -> builtins.list[str]:
        completed = self._run_git(cwd, *arguments, check=check, what=what, timeout=timeout)
        if arguments[-1] == "-z":
            return [part for part in completed.stdout.split("\0") if part]
        return [line for line in completed.stdout.splitlines() if line]


def _sanitize_name(value: str | None) -> str:
    if value is None:
        return ""
    sanitized = _NAME_PATTERN.sub("-", value.strip().lower()).strip("-._")
    return sanitized[:48]


def _include_match(relative_path: str, patterns: tuple[str, ...]) -> bool:
    name = Path(relative_path).name
    for pattern in patterns:
        if pattern.endswith("/"):
            if relative_path.startswith(pattern) or f"{relative_path}/".startswith(pattern):
                return True
        elif fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(name, pattern):
            return True
    return False


def _write_provenance(worktree: ManagedWorktree) -> None:
    path = worktree.root / PROVENANCE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(worktree.provenance(), ensure_ascii=False, indent=2),
                         encoding="utf-8")
    os.replace(temporary, path)


def _remove_empty_bucket(checkout: Path) -> None:
    bucket = checkout.parent
    try:
        bucket.rmdir()
    except OSError:
        pass


__all__ = [
    "DEFAULT_KEEP_COUNT",
    "INCLUDE_FILENAME",
    "WORKTREE_BRANCH_PREFIX",
    "WORKTREE_ROOT_ENV",
    "CreateReport",
    "ManagedWorktree",
    "RemoveReport",
    "WorktreeError",
    "WorktreeInspection",
    "WorktreeManager",
]

"""Host-side shadow git checkpoints at workspace mutation boundaries.

Every snapshot is an orphan commit (no parent, mirroring Codex git-utils'
baseline snapshot store) in a repository outside the workspace, addressed with
explicit ``--git-dir``/``--work-tree`` flags. The workspace's own Git metadata
is never read or written. The snapshot content domain matches
``workspace_tree_sha256`` by reusing its ignored-directory set, so checkpoint
diffs and the runtime's workspace-change observations describe the same tree.

Restore follows the Claude Code checkpoint contract: modified files are written
back, files added after the target checkpoint are deleted, paths the snapshot
store never observed are untouched, and the pre-restore state is committed
first so a rewind can itself be rewound.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from agent_runtime.user_config import user_config_directory
from agent_runtime.workspace import (
    _filesystem_workspace_paths,
    _has_temporary_runtime_layout,
    workspace_tree_sha256,
)

if TYPE_CHECKING:
    from agent_runtime.harness.rollout import RolloutStore

logger = logging.getLogger(__name__)

RETENTION_DAYS = 30
_SUBJECT_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_CHECKPOINT_REF_PREFIX = "refs/checkpoints/"
_LATEST_REF = "refs/praxis/latest"
_WORKSPACE_CONFIG_KEY = "praxis.workspace"
_RESTORE_PATH_CHUNK = 100
# Applied as command-line -c overrides on every invocation so behavior never
# depends on a config file that a later process could have edited.
_GIT_CONFIG_OVERRIDES = (
    ("user.name", "Praxis Checkpoints"),
    ("user.email", "checkpoints@praxis.invalid"),
    ("commit.gpgsign", "false"),
    ("core.hooksPath", ""),
    ("core.fsmonitor", "false"),
    ("core.untrackedCache", "false"),
    ("core.excludesFile", os.devnull),
)


class CheckpointError(RuntimeError):
    """A checkpoint operation failed; workspace files were not harmed."""


class CheckpointUnavailableError(CheckpointError):
    """Checkpoints cannot run on this host (missing git or unusable repository)."""


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    checkpoint_id: str
    trigger: str
    turn_id: str
    tree_sha256: str | None
    changed_file_count: int
    created_new_commit: bool


@dataclass(frozen=True, slots=True)
class CheckpointEntry:
    checkpoint_id: str
    timestamp: float
    subject: str
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class RestoreResult:
    target_checkpoint_id: str
    restored: tuple[str, ...]
    deleted: tuple[str, ...]
    anchor_checkpoint_id: str = ""  # pre-restore state; empty when nothing changed


def checkpoint_root_for(workspace_root: Path | str) -> Path:
    """One shadow repository per resolved workspace, under host configuration."""
    key = hashlib.sha256(str(Path(workspace_root).resolve()).encode()).hexdigest()[:16]
    return user_config_directory() / "checkpoints" / key


class CheckpointService:
    """Snapshot and restore workspace files without touching workspace Git metadata."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        shadow_root: Path | str | None = None,
        retention_days: int = RETENTION_DAYS,
    ) -> None:
        self._workspace = Path(workspace_root).expanduser().resolve()
        resolved_shadow = (Path(shadow_root) if shadow_root is not None
                           else checkpoint_root_for(self._workspace))
        self._shadow = resolved_shadow.expanduser().resolve()
        self._retention_days = retention_days
        self._git_dir = self._shadow / ".git"
        self._git = shutil.which("git")
        self._initialized = False
        self._empty_tree: str | None = None

    @property
    def workspace_root(self) -> Path:
        return self._workspace

    @property
    def shadow_root(self) -> Path:
        return self._shadow

    async def snapshot(self, *, trigger: str, turn_id: str) -> CheckpointSnapshot:
        """Record the current workspace tree as one checkpoint."""
        try:
            return await asyncio.to_thread(self._snapshot_sync, trigger=trigger, turn_id=turn_id)
        except OSError as error:
            raise CheckpointError(f"checkpoint storage unavailable: {error}") from error

    async def restore(self, checkpoint_id: str) -> RestoreResult:
        """Put workspace files back to a checkpoint, anchoring the current state first."""
        try:
            return await asyncio.to_thread(self._restore_sync, checkpoint_id)
        except OSError as error:
            raise CheckpointError(f"checkpoint storage unavailable: {error}") from error

    async def list_checkpoints(self) -> tuple[CheckpointEntry, ...]:
        """List checkpoints newest first by commit time."""
        try:
            return await asyncio.to_thread(self._list_sync)
        except OSError as error:
            raise CheckpointError(f"checkpoint storage unavailable: {error}") from error

    async def prune(self, *, now: float | None = None) -> None:
        """Drop expired checkpoints and shadow repositories of removed workspaces."""
        try:
            await asyncio.to_thread(self._prune_sync, now)
        except OSError as error:
            raise CheckpointError(f"checkpoint storage unavailable: {error}") from error

    # --- Synchronous implementation -------------------------------------------------

    def _git_argv(self, *arguments: str) -> list[str]:
        argv = [self._git or "git", "--no-optional-locks"]
        for key, value in _GIT_CONFIG_OVERRIDES:
            argv += ("-c", f"{key}={value}")
        argv += (f"--git-dir={self._git_dir}", f"--work-tree={self._workspace}", *arguments)
        return argv

    def _git_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_TERMINAL_PROMPT="0",
        )
        return environment

    def _run_git(
        self, *arguments: str, check: bool = True, cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            self._git_argv(*arguments),
            cwd=cwd or self._workspace,
            env=self._git_environment(),
            capture_output=True,
            text=True,
            check=False,
        )
        if check and completed.returncode != 0:
            raise CheckpointError(
                f"git {arguments[0]} failed ({completed.returncode}): "
                f"{completed.stderr.strip()[:500]}"
            )
        return completed

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        if self._git is None:
            raise CheckpointUnavailableError("git executable not found on PATH")
        if not self._workspace.is_dir():
            raise CheckpointUnavailableError(f"workspace does not exist: {self._workspace}")
        try:
            self._shadow.mkdir(parents=True, exist_ok=True)
            if not (self._git_dir / "HEAD").is_file():
                subprocess.run(
                    [self._git, "-c", "core.hooksPath=", "init", "--quiet", str(self._shadow)],
                    capture_output=True, text=True, check=True,
                    env=self._git_environment(),
                )
            marker = self._run_git("config", "--get", "praxis.marker", check=False)
            if marker.returncode != 0 or marker.stdout.strip() != "1":
                # Adopt only repositories this service could have created; a
                # marker-less directory that already holds checkpoint refs is foreign.
                if _checkpoint_refs_present(self._git_dir):
                    raise CheckpointUnavailableError("shadow repository marker is missing")
                self._run_git("config", "praxis.marker", "1")
            self._run_git("config", _WORKSPACE_CONFIG_KEY, str(self._workspace))
            self._empty_tree = self._run_git_with_input("mktree", data=b"").strip()
            self._initialized = True
        except CheckpointUnavailableError:
            raise
        except (OSError, subprocess.SubprocessError) as error:
            raise CheckpointUnavailableError(f"shadow repository unavailable: {error}") from error

    def _run_git_with_input(self, *arguments: str, data: bytes) -> str:
        completed = subprocess.run(
            self._git_argv(*arguments),
            cwd=self._workspace,
            env=self._git_environment(),
            input=data,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise CheckpointError(
                f"git {arguments[0]} failed ({completed.returncode}): "
                f"{completed.stderr.decode(errors='replace').strip()[:500]}"
            )
        return completed.stdout.decode(errors="replace")

    def _lock(self) -> _RepositoryLock:
        return _RepositoryLock(self._shadow)

    def _latest_checkpoint(self) -> CheckpointEntry | None:
        latest = self._run_git("rev-parse", "--verify", "--quiet", _LATEST_REF, check=False)
        if latest.returncode != 0:
            return None
        checkpoint_id = latest.stdout.strip()
        detail = self._run_git(
            "show", "-s", "--format=%ct%n%s", checkpoint_id,
        ).stdout.splitlines()
        return CheckpointEntry(
            checkpoint_id=checkpoint_id,
            timestamp=float(detail[0]) if detail and detail[0].isdigit() else 0.0,
            subject=detail[1] if len(detail) > 1 else "",
        )

    def _snapshot_sync(self, *, trigger: str, turn_id: str) -> CheckpointSnapshot:
        if trigger not in {"pre_write", "restore"}:
            raise ValueError(f"unsupported checkpoint trigger: {trigger}")
        with self._lock():
            self._ensure_initialized()
            tree = self._stage_current_tree()
            previous = self._latest_checkpoint()
            tree_sha = workspace_tree_sha256(self._workspace)
            if previous is not None:
                previous_tree = self._run_git("rev-parse", f"{previous.checkpoint_id}^{{tree}}").stdout.strip()
                if previous_tree == tree:
                    return CheckpointSnapshot(
                        checkpoint_id=previous.checkpoint_id,
                        trigger=trigger, turn_id=turn_id,
                        tree_sha256=tree_sha, changed_file_count=0, created_new_commit=False,
                    )
            changed = self._run_git(
                "diff", "--name-only", previous.checkpoint_id if previous is not None
                else self._empty_tree or "", tree,
            ).stdout
            changed_count = len([line for line in changed.splitlines() if line.strip()])
            checkpoint_id = self._commit_tree(tree, trigger=trigger, turn_id=turn_id)
            return CheckpointSnapshot(
                checkpoint_id=checkpoint_id,
                trigger=trigger, turn_id=turn_id,
                tree_sha256=tree_sha, changed_file_count=changed_count, created_new_commit=True,
            )

    def _stage_current_tree(self) -> str:
        """Rebuild the index from the runtime's workspace enumeration.

        ``git add`` would apply the workspace's own ignore rules and silently
        drop gitignored files from the safety net. ``update-index`` stages
        exactly the enumerated paths, so the checkpoint domain stays identical
        to ``workspace_tree_sha256``.
        """
        self._run_git("read-tree", "--empty")
        paths = _filesystem_workspace_paths(
            self._workspace,
            ignore_runtime_root=_has_temporary_runtime_layout(self._workspace),
        )
        feed = b"".join(os.fsencode(path.as_posix()) + b"\0" for path in paths)
        self._run_git_with_input("update-index", "-z", "--add", "--stdin", data=feed)
        return self._run_git("write-tree").stdout.strip()

    def _commit_tree(self, tree: str, *, trigger: str, turn_id: str) -> str:
        # Commit timestamps have one-second granularity; the ref count keeps
        # same-second checkpoints orderable for listings.
        sequence = self._run_git(
            "for-each-ref", "--format=x", _CHECKPOINT_REF_PREFIX,
        ).stdout.count("\n")
        subject = "praxis-checkpoint " + " ".join(
            part for part in (trigger, _SUBJECT_SAFE.sub("_", turn_id)[:64]) if part
        ) + f" seq={sequence:06d}"
        message = subject + f"\n\nworkspace: {self._workspace}"
        checkpoint_id = self._run_git_with_input(
            "commit-tree", tree, data=message.encode(),
        ).strip()
        self._run_git("update-ref", _CHECKPOINT_REF_PREFIX + checkpoint_id, checkpoint_id)
        self._run_git("update-ref", _LATEST_REF, checkpoint_id)
        return checkpoint_id

    def _restore_sync(self, checkpoint_id: str) -> RestoreResult:
        with self._lock():
            self._ensure_initialized()
            resolved = self._run_git(
                "rev-parse", "--verify", "--quiet", f"{checkpoint_id}^{{commit}}", check=False,
            )
            if resolved.returncode != 0:
                raise CheckpointError(f"unknown checkpoint: {checkpoint_id}")
            target = resolved.stdout.strip()
            target_tree = self._run_git("rev-parse", f"{target}^{{tree}}").stdout.strip()
            current_tree = self._stage_current_tree()
            if current_tree == target_tree:
                return RestoreResult(target_checkpoint_id=target, restored=(), deleted=())
            # Anchor the pre-restore state first so the rewind is itself rewindingable.
            anchor = self._commit_tree(current_tree, trigger="restore", turn_id=f"rewind-{target[:12]}")
            changed = self._run_git(
                "diff", "--name-status", "--no-renames", "-z", target_tree, current_tree,
            ).stdout
            restore_paths: list[str] = []
            delete_paths: list[str] = []
            fields = changed.split("\0")
            index = 0
            while index + 1 < len(fields):
                status, path = fields[index], fields[index + 1]
                index += 2
                if status in {"A"}:
                    delete_paths.append(path)
                else:  # M, D, T: the target holds content the workspace lost or changed.
                    restore_paths.append(path)
            failures: list[str] = []
            # Deletes run first: paths that changed type between file and
            # directory must clear the old form before checkout recreates the
            # new one, and emptied directories are pruned so a file can take
            # a former directory's place.
            for path in delete_paths:
                if not self._delete_workspace_file(path):
                    failures.append(f"could not delete {path}")
                else:
                    self._prune_empty_parents(path)
            for offset in range(0, len(restore_paths), _RESTORE_PATH_CHUNK):
                chunk = restore_paths[offset:offset + _RESTORE_PATH_CHUNK]
                result = self._run_git("checkout", target, "--", *chunk, check=False)
                if result.returncode != 0:
                    failures.append(result.stderr.strip()[:200])
            final_tree = self._stage_current_tree()
            if final_tree != target_tree or failures:
                self._run_git("update-ref", _LATEST_REF, anchor)
                raise CheckpointError(
                    "restore did not reproduce the checkpoint tree: " + "; ".join(failures[:5])
                )
            return RestoreResult(
                target_checkpoint_id=target,
                restored=tuple(sorted(restore_paths)),
                deleted=tuple(sorted(delete_paths)),
                anchor_checkpoint_id=anchor,
            )

    def _prune_empty_parents(self, relative: str) -> None:
        """Remove directories left empty by deletions, up to the workspace root."""
        parent = Path(relative).parent
        while parent.parts:
            try:
                (self._workspace / parent).rmdir()  # succeeds only when empty
            except OSError:
                return
            parent = parent.parent

    def _delete_workspace_file(self, relative: str) -> bool:
        if not self._safe_relative_path(relative):
            return False
        target = self._workspace / relative
        try:
            if target.is_dir() and not target.is_symlink():
                return False
            target.unlink()
            return True
        except OSError:
            return False

    def _safe_relative_path(self, relative: str) -> bool:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            return False
        resolved = (self._workspace / path)
        try:
            resolved.resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        return True

    def _list_sync(self) -> tuple[CheckpointEntry, ...]:
        if not self._git_dir.is_dir() or self._git is None:
            return ()
        listing = self._run_git(
            "for-each-ref", "--format=%(objectname)%09%(committerdate:unix)%09%(subject)",
            _CHECKPOINT_REF_PREFIX, check=False,
        )
        if listing.returncode != 0:
            return ()
        entries = []
        for line in listing.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3 or not parts[0].strip():
                continue
            subject = parts[2].strip()
            sequence_match = re.search(r"seq=(\d+)", subject)
            entries.append(CheckpointEntry(
                checkpoint_id=parts[0].strip(),
                timestamp=float(parts[1]) if parts[1].strip().isdigit() else 0.0,
                subject=subject,
                sequence=int(sequence_match.group(1)) if sequence_match else 0,
            ))
        return tuple(sorted(
            entries,
            key=lambda entry: (entry.timestamp, entry.sequence, entry.checkpoint_id),
            reverse=True,
        ))

    def _prune_sync(self, now: float | None) -> None:
        if not self._git_dir.is_dir() or self._git is None:
            return
        try:
            recorded = self._run_git(
                "config", "--get", _WORKSPACE_CONFIG_KEY, check=False, cwd=self._shadow,
            )
            if recorded.returncode == 0 and recorded.stdout.strip():
                if not Path(recorded.stdout.strip()).is_dir():
                    shutil.rmtree(self._shadow, ignore_errors=True)
                    return
            cutoff = (time.time() if now is None else now) - self._retention_days * 86400
            entries = self._list_sync()
            expired = [entry.checkpoint_id for entry in entries if entry.timestamp < cutoff]
            latest = self._latest_checkpoint()
            latest_expired = latest is not None and latest.timestamp < cutoff
            if not expired and not latest_expired:
                return
            with self._lock():
                for checkpoint_id in expired:
                    self._run_git("update-ref", "-d", _CHECKPOINT_REF_PREFIX + checkpoint_id, check=False)
                if latest_expired:
                    self._run_git("update-ref", "-d", _LATEST_REF, check=False)
                self._run_git(
                    "gc", "--prune=" + time.strftime("%Y-%m-%d", time.gmtime(cutoff)), "--quiet",
                    check=False,
                )
        except (CheckpointError, OSError) as error:
            logger.warning("checkpoint pruning skipped: %s", error)


class _RepositoryLock:
    """Cross-process exclusive lock for one shadow repository."""

    def __init__(self, shadow_root: Path) -> None:
        self._path = shadow_root / "checkpoint.lock"
        self._handle: BinaryIO | None = None

    def __enter__(self) -> _RepositoryLock:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("ab")
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except OSError as error:
            # A safety net must never become the write path's failure point:
            # storage problems surface as CheckpointError for fail-soft skips.
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            raise CheckpointError(f"checkpoint lock unavailable: {error}") from error
        return self

    def __exit__(self, *exception: object) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self._handle.close()
            self._handle = None


def _checkpoint_refs_present(git_dir: Path) -> bool:
    refs = git_dir / "refs" / "checkpoints"
    if refs.is_dir() and any(refs.iterdir()):
        return True
    packed = git_dir / "packed-refs"
    try:
        return packed.is_file() and _CHECKPOINT_REF_PREFIX in packed.read_text()
    except OSError:
        return False


def _numeric(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def thread_checkpoint_rows(store: RolloutStore, thread_id: str) -> list[dict[str, object]]:
    """Join recorded checkpoint items with their Turns for user-facing listings."""
    rows: list[dict[str, object]] = []
    for turn in store.list_turns(thread_id):
        turn_id = turn.turn_id
        user_message = ""
        for item in store.list_items(turn_id):
            if item.kind == "user_message" and isinstance(item.payload.get("text"), str):
                user_message = item.payload["text"]
                break
        for item in store.list_items(turn_id):
            payload = item.payload
            if item.kind != "workspace_checkpoint" or item.status != "completed":
                continue
            rows.append({
                "checkpoint_id": payload.get("checkpoint_id"),
                "turn_id": turn_id,
                "trigger": payload.get("trigger"),
                "tree_sha256": payload.get("tree_sha256"),
                "changed_file_count": payload.get("changed_file_count", 0),
                "wall_time": payload.get("wall_time", 0.0),
                "sequence": item.sequence,
                "turn_input": " ".join(user_message.split())[:80],
            })
    rows.sort(key=lambda row: (_numeric(row.get("wall_time")), int(_numeric(row.get("sequence")))),
              reverse=True)
    return rows


__all__ = [
    "CheckpointEntry",
    "CheckpointError",
    "CheckpointService",
    "CheckpointSnapshot",
    "CheckpointUnavailableError",
    "RestoreResult",
    "checkpoint_root_for",
    "thread_checkpoint_rows",
]

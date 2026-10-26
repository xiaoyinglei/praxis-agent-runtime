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
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from agent_runtime.tools.builtins.file_access import (
    atomic_file,
    atomic_symlink,
    file_version,
    open_binary,
    parent_descriptor,
    workspace_write_lock,
)
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
_METADATA_TRAILER = "Praxis-Checkpoint"
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
    """A checkpoint failed; restore errors describe the actual rollback outcome."""

    def __init__(self, message: str, *, anchor_checkpoint_id: str = "",
                 rollback_completed: bool | None = None, workspace_changed: bool | None = None) -> None:
        super().__init__(message)
        self.anchor_checkpoint_id = anchor_checkpoint_id
        self.rollback_completed = rollback_completed
        self.workspace_changed = workspace_changed


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
    trigger: str | None = None
    turn_id: str | None = None


def _checkpoint_entry(checkpoint_id: str, timestamp: str, subject: str, message: str) -> CheckpointEntry:
    trigger, turn_id, sequence = None, None, 0
    # Read our single-line protocol from raw commit bytes. Git's generic trailer
    # formatter rewrites keys according to arbitrary trailer.* host settings.
    block = message.partition("\n\n")[2].rstrip().rsplit("\n\n", 1)[-1]
    records = [line for line in block.splitlines()
               if line.startswith(_METADATA_TRAILER + ":")]
    if records:
        # An explicit unsupported or corrupt version cannot fall back to title.
        if len(records) != 1:
            raise CheckpointError("duplicate checkpoint metadata trailers")
        try:
            metadata = json.loads(records[0].partition(":")[2])
        except ValueError as error:
            raise CheckpointError("invalid checkpoint metadata JSON") from error
        if not isinstance(metadata, dict) or type(metadata.get("version")) is not int or metadata["version"] != 1:
            raise CheckpointError("unsupported checkpoint metadata version")
        trigger, turn_id = metadata.get("trigger"), metadata.get("turn_id")
        candidate_sequence = metadata.get("sequence")
        if (trigger not in ("pre_write", "restore") or not isinstance(turn_id, str)
                or type(candidate_sequence) is not int or candidate_sequence < 0):
            raise CheckpointError("invalid checkpoint metadata fields")
        sequence = candidate_sequence
    else:
        # Version 0: the exact historical writer format, decoded in one place.
        # Old sanitized/truncated Turn IDs cannot recover their original text.
        legacy = re.fullmatch(r"praxis-checkpoint (pre_write|restore)(?: ([A-Za-z0-9._-]{1,64}))? seq=([0-9]+)",
                              subject)
        if legacy is not None and f"{int(legacy[3]):06d}" == legacy[3]:
            trigger, turn_id, sequence = legacy[1], legacy[2] or "", int(legacy[3])
    return CheckpointEntry(checkpoint_id=checkpoint_id.strip(),
                           timestamp=float(timestamp) if timestamp.isdigit() else 0.0,
                           subject=subject, sequence=sequence, trigger=trigger, turn_id=turn_id)


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
        self._staged_modes: dict[str, int] = {}

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
        environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
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
        return self._read_checkpoint(latest.stdout.strip())

    def _read_checkpoint(self, checkpoint_id: str) -> CheckpointEntry:
        detail = self._run_git("show", "-s",
            "--format=%H%x00%ct%x00%s%x00%B%x00",
            checkpoint_id).stdout.split("\0")
        return _checkpoint_entry(*detail[:4])

    def _snapshot_sync(self, *, trigger: str, turn_id: str) -> CheckpointSnapshot:
        if trigger not in {"pre_write", "restore"}:
            raise ValueError(f"unsupported checkpoint trigger: {trigger}")
        with self._lock(), workspace_write_lock(self._workspace):
            self._ensure_initialized()
            tree = self._stage_current_tree()
            previous = self._latest_checkpoint()
            tree_sha = workspace_tree_sha256(self._workspace)
            if previous is not None:
                previous_tree = self._run_git("rev-parse", f"{previous.checkpoint_id}^{{tree}}").stdout.strip()
                if previous_tree == tree and self._checkpoint_modes(previous.checkpoint_id) == self._staged_modes:
                    return CheckpointSnapshot(
                        checkpoint_id=previous.checkpoint_id,
                        trigger=trigger, turn_id=turn_id,
                        tree_sha256=tree_sha, changed_file_count=0, created_new_commit=False,
                    )
            changed = self._run_git(
                "diff", "--name-only", "-z", previous.checkpoint_id if previous is not None
                else self._empty_tree or "", tree,
            ).stdout
            changed_paths = {path for path in changed.split("\0") if path}
            old_modes = {} if previous is None else self._checkpoint_modes(previous.checkpoint_id)
            changed_paths.update(path for path in set(old_modes) | set(self._staged_modes)
                                 if old_modes.get(path) != self._staged_modes.get(path))
            changed_count = len(changed_paths)
            checkpoint_id = self._commit_tree(tree, trigger=trigger, turn_id=turn_id)
            return CheckpointSnapshot(
                checkpoint_id=checkpoint_id,
                trigger=trigger, turn_id=turn_id,
                tree_sha256=tree_sha, changed_file_count=changed_count, created_new_commit=True,
            )

    def _stage_current_tree(self) -> str:
        """Store raw blobs, without attributes, filters, ignore rules or checkout conversions."""
        self._run_git("read-tree", "--empty")
        paths = _filesystem_workspace_paths(
            self._workspace, ignore_runtime_root=_has_temporary_runtime_layout(self._workspace),
        )
        entries: list[bytes] = []
        modes: dict[str, int] = {}
        regular: list[Path] = []
        for relative in paths:
            path = self._workspace / relative
            metadata = path.lstat()
            label = relative.as_posix()
            modes[label] = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISLNK(metadata.st_mode):
                blob = self._run_git_with_input("hash-object", "-w", "--stdin",
                                                data=os.fsencode(os.readlink(path))).strip()
                entries.append(f"120000 {blob}\t".encode() + os.fsencode(label) + b"\0")
            elif stat.S_ISREG(metadata.st_mode):
                regular.append(relative)
            else:
                raise CheckpointError(f"unsupported checkpoint file type: {label}")
        # Native Git hashes already-open file descriptors in bounded batches.
        # This avoids a process per file and never follows a renamed pathname.
        for offset in range(0, len(regular), 64):
            chunk = regular[offset:offset + 64]
            with ExitStack() as stack:
                streams = [stack.enter_context(open_binary(self._workspace / relative)) for relative in chunk]
                versions = [os.fstat(stream.fileno()) for stream in streams]
                descriptors = tuple(stream.fileno() for stream in streams)
                feed = "".join(f"/dev/fd/{descriptor}\n" for descriptor in descriptors).encode()
                result = subprocess.run(self._git_argv("hash-object", "-w", "--no-filters", "--stdin-paths"),
                                        cwd=self._workspace, env=self._git_environment(), input=feed,
                                        capture_output=True, pass_fds=descriptors, check=False)
                blobs = result.stdout.decode("ascii").splitlines()
                if result.returncode or len(blobs) != len(chunk):
                    raise CheckpointError("could not snapshot raw file contents")
                for relative, stream, before, blob in zip(chunk, streams, versions, blobs, strict=True):
                    if file_version(before) != file_version(os.fstat(stream.fileno())):
                        raise CheckpointError("workspace changed during checkpoint capture")
                    label = relative.as_posix()
                    modes[label] = stat.S_IMODE(before.st_mode)
                    mode = "100755" if before.st_mode & 0o111 else "100644"
                    entries.append(f"{mode} {blob}\t".encode() + os.fsencode(label) + b"\0")
        self._run_git_with_input("update-index", "-z", "--index-info", data=b"".join(entries))
        self._staged_modes = modes
        return self._run_git("write-tree").stdout.strip()

    def _checkpoint_modes(self, checkpoint_id: str) -> dict[str, int]:
        message = self._run_git("show", "-s", "--format=%B", checkpoint_id).stdout
        for line in message.splitlines():
            if line.startswith("praxis-modes-json: "):
                value = json.loads(line.removeprefix("praxis-modes-json: "))
                if not isinstance(value, dict) or any(not isinstance(k, str) or type(v) is not int
                                                     or not 0 <= v <= 0o7777 for k, v in value.items()):
                    raise CheckpointError("invalid checkpoint permission manifest")
                return value
        return {}  # Historical checkpoints preserve only the Git executable bit.

    def _tree_entries(self, tree: str) -> dict[str, tuple[str, str]]:
        raw = self._run_git("ls-tree", "-r", "-z", tree).stdout
        entries = {}
        for record in raw.split("\0"):
            if not record:
                continue
            header, path = record.split("\t", 1)
            mode, _kind, blob = header.split()
            if not self._safe_relative_path(path):
                raise CheckpointError("unsafe checkpoint path")
            entries[path] = (mode, blob)
        return entries

    def _blob_bytes(self, blob: str) -> bytes:
        result = subprocess.run(self._git_argv("cat-file", "blob", blob), cwd=self._workspace,
                                env=self._git_environment(), capture_output=True, check=False)
        if result.returncode:
            raise CheckpointError("checkpoint blob is unavailable")
        return result.stdout

    def _restore_tree(self, tree: str, modes: dict[str, int]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        current = self._tree_entries(self._stage_current_tree())
        current_modes = self._staged_modes.copy()
        target = self._tree_entries(tree)
        deleted = sorted(set(current) - set(target), reverse=True)
        restored = sorted(path for path, entry in target.items() if current.get(path) != entry
                          or modes.get(path, 0o755 if entry[0] == "100755" else 0o644)
                          != current_modes.get(path))
        for relative in deleted:
            if not self._delete_workspace_file(relative):
                raise CheckpointError(f"could not delete {relative}")
            self._prune_empty_parents(relative)
        for relative in restored:
            mode, blob = target[relative]
            path = self._workspace / relative
            with parent_descriptor(path, create=True) as parent:
                try:
                    metadata = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    metadata = None
                if metadata is not None and stat.S_ISDIR(metadata.st_mode):
                    os.rmdir(path.name, dir_fd=parent)  # Refuse an unobserved nonempty directory.
            if mode == "120000":
                atomic_symlink(path, os.fsdecode(self._blob_bytes(blob)))
            else:
                def write_blob(stream: BinaryIO, object_id: str = blob) -> None:
                    result = subprocess.run(self._git_argv("cat-file", "blob", object_id), cwd=self._workspace,
                                            env=self._git_environment(), stdout=stream,
                                            stderr=subprocess.PIPE, check=False)
                    if result.returncode:
                        raise CheckpointError("checkpoint blob is unavailable")
                atomic_file(path, write_blob, mode=modes.get(relative, 0o755 if mode == "100755" else 0o644))
        return tuple(restored), tuple(sorted(deleted))

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
        message += "\npraxis-modes-json: " + json.dumps(self._staged_modes, sort_keys=True, ensure_ascii=True)
        message += "\n\n" + _METADATA_TRAILER + ": " + json.dumps({
            "version": 1, "trigger": trigger, "turn_id": turn_id, "sequence": sequence,
        }, sort_keys=True, ensure_ascii=True)
        checkpoint_id = self._run_git_with_input(
            "commit-tree", tree, data=message.encode(),
        ).strip()
        self._run_git("update-ref", _CHECKPOINT_REF_PREFIX + checkpoint_id, checkpoint_id)
        self._run_git("update-ref", _LATEST_REF, checkpoint_id)
        return checkpoint_id

    def _restore_sync(self, checkpoint_id: str) -> RestoreResult:
        with self._lock(), workspace_write_lock(self._workspace):
            self._ensure_initialized()
            resolved = self._run_git(
                "rev-parse", "--verify", "--quiet", f"{checkpoint_id}^{{commit}}", check=False,
            )
            if resolved.returncode != 0:
                raise CheckpointError(f"unknown checkpoint: {checkpoint_id}")
            target = resolved.stdout.strip()
            self._read_checkpoint(target)  # Validate the format before any workspace mutation.
            target_tree = self._run_git("rev-parse", f"{target}^{{tree}}").stdout.strip()
            current_tree = self._stage_current_tree()
            current_modes = self._staged_modes.copy()
            target_modes = self._checkpoint_modes(target)
            if current_tree == target_tree and (not target_modes or target_modes == current_modes):
                return RestoreResult(target_checkpoint_id=target, restored=(), deleted=())
            before_hash = workspace_tree_sha256(self._workspace)
            anchor = self._commit_tree(current_tree, trigger="restore", turn_id=f"rewind-{target[:12]}")
            try:
                restored, deleted = self._restore_tree(target_tree, target_modes)
                final_tree = self._stage_current_tree()
                if final_tree != target_tree or (target_modes and self._staged_modes != target_modes):
                    raise CheckpointError("restored content or permissions do not match checkpoint")
            except (CheckpointError, OSError) as error:
                try:
                    self._restore_tree(current_tree, current_modes)
                except (CheckpointError, OSError):
                    pass
                after_hash = workspace_tree_sha256(self._workspace)
                rolled_back = before_hash is not None and before_hash == after_hash
                self._run_git("update-ref", _LATEST_REF, anchor)
                state = "pre-restore state recovered" if rolled_back else "workspace may be partially restored"
                raise CheckpointError(
                    f"restore failed; {state}; recovery checkpoint {anchor}: {error}",
                    anchor_checkpoint_id=anchor, rollback_completed=rolled_back,
                    workspace_changed=not rolled_back,
                ) from error
            return RestoreResult(target_checkpoint_id=target, restored=restored, deleted=deleted,
                                 anchor_checkpoint_id=anchor)

    def _prune_empty_parents(self, relative: str) -> None:
        """Remove directories left empty by deletions, up to the workspace root."""
        parent = Path(relative).parent
        while parent.parts:
            try:
                path = self._workspace / parent
                with parent_descriptor(path) as descriptor:
                    os.rmdir(path.name, dir_fd=descriptor)
            except OSError:
                return
            parent = parent.parent

    def _delete_workspace_file(self, relative: str) -> bool:
        if not self._safe_relative_path(relative):
            return False
        target = self._workspace / relative
        try:
            with parent_descriptor(target) as descriptor:
                os.unlink(target.name, dir_fd=descriptor)
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
            "for-each-ref", "--format=%(objectname)%00%(committerdate:unix)%00%(subject)%00"
            "%(contents)%00",
            _CHECKPOINT_REF_PREFIX, check=False,
        )
        if listing.returncode != 0:
            return ()
        fields = listing.stdout.split("\0")
        if fields.pop().strip() or len(fields) % 4:
            raise CheckpointError("invalid checkpoint listing format")
        entries = [_checkpoint_entry(*fields[offset:offset + 4]) for offset in range(0, len(fields), 4)]
        return tuple(sorted(
            entries,
            key=lambda entry: (entry.timestamp, entry.sequence, entry.checkpoint_id),
            reverse=True,
        ))

    def _prune_sync(self, now: float | None) -> None:
        if not self._git_dir.is_dir() or self._git is None:
            return
        try:
            with self._lock():
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
        # Deleting a repository must not unlink its coordination inode: another
        # process could otherwise acquire a new inode under the same pathname.
        self._path = shadow_root.with_name("." + shadow_root.name + ".checkpoint.lock")
        self._handle: BinaryIO | None = None

    def __enter__(self) -> _RepositoryLock:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self._path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            self._handle = os.fdopen(descriptor, "ab")
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

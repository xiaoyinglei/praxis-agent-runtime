"""Crash-safe, trusted-path byte I/O for persisted model configuration."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class FileVersion:
    """The revision and content fingerprint observed for a config file."""

    revision: int
    fingerprint: str


class ConfigVersionConflict(RuntimeError):  # noqa: N818
    """A config write was based on a stale file version."""


class CommitOutcomeUnknown(RuntimeError):  # noqa: N818
    """Visibility occurred, but the final durable outcome cannot be confirmed."""


class UntrustedConfigPathError(ValueError):
    """A user-supplied config path escapes the workspace trust boundary."""


def discover_git_worktree(workspace: Path) -> Path:
    """Return the resolved Git top-level directory, or the workspace if absent."""

    resolved_workspace = workspace.expanduser().resolve()
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    completed = subprocess.run(
        ["git", "-C", os.fspath(resolved_workspace), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    if completed.returncode != 0:
        error_output = completed.stderr.strip()
        if "not a git repository" in error_output.lower():
            return resolved_workspace
        raise RuntimeError(
            f"Unable to discover Git worktree for {resolved_workspace}: {error_output}"
        )
    top_level = completed.stdout.strip()
    return Path(top_level).resolve() if top_level else resolved_workspace


def validate_user_config_path(path: Path, *, workspace: Path, worktree: Path) -> Path:
    """Resolve an absolute target outside the explicit workspace and worktree."""

    resolved_workspace = workspace.expanduser().resolve()
    resolved_worktree = worktree.expanduser().resolve()
    if not path.is_absolute():
        raise UntrustedConfigPathError("Config path must be absolute")
    candidate = path.expanduser().resolve()
    if _is_within(candidate, resolved_workspace):
        raise UntrustedConfigPathError(
            f"Config path {path} resolves inside the explicit workspace"
        )
    if _is_within(candidate, resolved_worktree):
        raise UntrustedConfigPathError(
            f"Config path {path} resolves inside the explicit Git worktree"
        )
    return candidate


def file_fingerprint(payload: bytes) -> str:
    """Return the stable SHA-256 fingerprint for exact file bytes."""

    return hashlib.sha256(payload).hexdigest()


@contextmanager
def exclusive_config_lock(path: Path) -> Iterator[None]:
    """Take an advisory inter-process lock stored beside the config path."""

    lock_path = path.with_name(f".{path.name}.lock")
    fd = _open_config_lock(lock_path)
    try:
        _validate_config_lock_fd(fd, lock_path)
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def atomic_replace_bytes(path: Path, payload: bytes, *, intended_fingerprint: str) -> None:
    """Durably replace *path* without rolling back once it becomes visible."""

    if file_fingerprint(payload) != intended_fingerprint:
        raise ValueError("intended_fingerprint does not match payload")

    temp_path = _write_adjacent_temp(path, payload)
    replaced = False
    try:
        os.replace(temp_path, path)
        replaced = True
        try:
            _fsync_directory(path.parent)
        except OSError as error:
            _raise_if_not_reconciled(path, payload, error)
    finally:
        if not replaced:
            _remove_temp_if_present(temp_path)


def atomic_install_bytes(path: Path, payload: bytes) -> Literal["created", "exists"]:
    """Atomically install bytes only when no target file exists yet."""

    temp_path = _write_adjacent_temp(path, payload)
    try:
        try:
            os.link(temp_path, path)
        except FileExistsError:
            return "exists"
        try:
            _fsync_directory(path.parent)
        except OSError as error:
            _raise_if_not_reconciled(path, payload, error)
        return "created"
    finally:
        _remove_temp_if_present(temp_path)


def confirm_parent_directory_durability(path: Path) -> None:
    """Confirm that the directory entry containing *path* is durable."""

    _fsync_directory(path.parent)


def ensure_durable_directory(path: Path, *, mode: int = 0o700) -> None:
    """Create a directory chain and durably install every new directory entry."""

    missing: list[Path] = []
    ancestor = path
    while True:
        try:
            ancestor_stat = ancestor.lstat()
        except FileNotFoundError:
            missing.append(ancestor)
            parent = ancestor.parent
            if parent == ancestor:
                raise OSError(f"no existing ancestor for directory {path}") from None
            ancestor = parent
            continue
        if stat.S_ISLNK(ancestor_stat.st_mode) or not stat.S_ISDIR(ancestor_stat.st_mode):
            raise OSError(f"unsafe directory ancestor: {ancestor}")
        break

    if ancestor.parent != ancestor:
        try:
            _fsync_directory(ancestor.parent)
        except OSError as error:
            raise CommitOutcomeUnknown(
                f"Directory durability cannot be confirmed for {ancestor}"
            ) from error

    for directory in reversed(missing):
        try:
            directory.mkdir(mode=mode)
        except FileExistsError:
            observed = directory.lstat()
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise OSError(
                    f"unsafe directory created concurrently: {directory}"
                ) from None
        try:
            _fsync_directory(directory.parent)
        except OSError as error:
            raise CommitOutcomeUnknown(
                f"Directory durability cannot be confirmed for {directory}"
            ) from error


def reconcile_atomic_install_link(path: Path) -> None:
    """Remove the one stale temp hard link left by an interrupted no-replace install."""

    target = path.lstat()
    if target.st_nlink == 1:
        return
    if (
        target.st_nlink != 2
        or not stat.S_ISREG(target.st_mode)
        or target.st_uid != os.geteuid()
    ):
        raise OSError(f"unsafe hard-linked config file: {path}")

    prefix = f".{path.name}."
    candidates: list[Path] = []
    for candidate in path.parent.iterdir():
        if not candidate.name.startswith(prefix) or not candidate.name.endswith(".tmp"):
            continue
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISREG(candidate_stat.st_mode)
            and candidate_stat.st_uid == os.geteuid()
            and (candidate_stat.st_dev, candidate_stat.st_ino) == (target.st_dev, target.st_ino)
        ):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise OSError(f"cannot safely reconcile hard-linked config file: {path}")

    candidates[0].unlink()
    try:
        _fsync_directory(path.parent)
    except OSError as error:
        raise CommitOutcomeUnknown(
            f"Config install cleanup durability cannot be confirmed for {path}"
        ) from error
    observed = path.lstat()
    if (
        observed.st_nlink != 1
        or (observed.st_dev, observed.st_ino) != (target.st_dev, target.st_ino)
    ):
        raise OSError(f"config install reconciliation changed target identity: {path}")


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        pass
    else:
        return True
    return _has_ancestor_identity(candidate, root)


def _has_ancestor_identity(candidate: Path, root: Path) -> bool:
    ancestor = _nearest_existing_ancestor(candidate)
    while ancestor is not None:
        try:
            if ancestor.samefile(root):
                return True
        except OSError:
            return False
        parent = ancestor.parent
        ancestor = None if parent == ancestor else parent
    return False


def _nearest_existing_ancestor(path: Path) -> Path | None:
    ancestor = path
    while True:
        try:
            ancestor.stat()
        except FileNotFoundError:
            parent = ancestor.parent
            if parent == ancestor:
                return None
            ancestor = parent
        except OSError:
            return None
        else:
            return ancestor


def _open_config_lock(lock_path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow:
        try:
            return os.open(lock_path, flags | no_follow, 0o600)
        except OSError as error:
            raise OSError(f"unsafe config lock: {lock_path}") from error
    return _open_config_lock_without_nofollow(lock_path, flags)


def _open_config_lock_without_nofollow(lock_path: Path, flags: int) -> int:
    while True:
        try:
            before_open = lock_path.lstat()
        except FileNotFoundError:
            try:
                return os.open(lock_path, flags | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            except OSError as error:
                raise OSError(f"unsafe config lock: {lock_path}") from error
        if not stat.S_ISREG(before_open.st_mode):
            raise OSError(f"unsafe config lock: {lock_path}")
        try:
            fd = os.open(lock_path, flags)
        except OSError as error:
            raise OSError(f"unsafe config lock: {lock_path}") from error
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before_open.st_dev, before_open.st_ino):
            os.close(fd)
            raise OSError(f"unsafe config lock: {lock_path}")
        return fd


def _validate_config_lock_fd(fd: int, lock_path: Path) -> None:
    lock_stat = os.fstat(fd)
    if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != os.geteuid():
        raise OSError(f"unsafe config lock: {lock_path}")


def _write_adjacent_temp(path: Path, payload: bytes) -> Path:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        temporary_file = os.fdopen(fd, "wb", closefd=True)
    except BaseException:
        os.close(fd)
        _remove_temp_if_present(temporary_path)
        raise
    try:
        with temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
    except BaseException:
        _remove_temp_if_present(temporary_path)
        raise
    return temporary_path


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _raise_if_not_reconciled(path: Path, payload: bytes, error: OSError) -> None:
    try:
        reconciled = path.read_bytes() == payload
    except OSError:
        reconciled = False
    if not reconciled:
        raise CommitOutcomeUnknown(
            f"Config commit cannot confirm intended bytes for {path}"
        ) from error
    try:
        _fsync_directory(path.parent)
    except OSError as retry_error:
        raise CommitOutcomeUnknown(
            f"Config commit cannot confirm durable intended bytes for {path}"
        ) from retry_error


def _remove_temp_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

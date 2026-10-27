"""Descriptor-relative file access; callers authorize the absolute path first."""
from __future__ import annotations

import fcntl
import os
import secrets
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


def file_version(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def parent_descriptor(path: Path, *, create: bool = False) -> Iterator[int]:
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('file access requires an absolute normalized path')
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            try:
                following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def open_binary(path: Path) -> Iterator[BinaryIO]:
    with parent_descriptor(path) as parent:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise PermissionError('file target must be a regular file')
            yield stream


@contextmanager
def workspace_write_lock(root: Path) -> Iterator[None]:
    """Cooperating executors lock the same workspace inode across processes."""
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def atomic_bytes(path: Path, content: bytes, *, mode: int = 0o600,
                 expected: tuple[int, int, int, int, int] | None = None,
                 create_only: bool = False, create_parents: bool = False) -> None:
    def write(stream: BinaryIO) -> None:
        stream.write(content)
    atomic_file(path, write, mode=mode, expected=expected, create_only=create_only,
                create_parents=create_parents)


def atomic_file(path: Path, write: Callable[[BinaryIO], None], *, mode: int = 0o600,
                expected: tuple[int, int, int, int, int] | None = None,
                create_only: bool = False, create_parents: bool = False) -> None:
    """Install into one held directory; reject changed source versions before replacement."""
    with parent_descriptor(path, create=create_parents) as parent:
        temporary = '.praxis-' + secrets.token_hex(12)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                write(stream)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), mode)
            # Refuse a parent that was replaced after its descriptor was opened.
            if file_version(os.stat(path.parent, follow_symlinks=False))[:2] != file_version(os.fstat(parent))[:2]:
                raise PermissionError('file parent changed during the operation')
            if expected is not None:
                current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                if file_version(current) != expected:
                    raise PermissionError('file changed during the operation; read the current version before retrying')
            if create_only:
                os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            else:
                os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            if _entry_exists(parent, temporary):
                os.unlink(temporary, dir_fd=parent)


def _entry_exists(parent: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def atomic_symlink(path: Path, target: str) -> None:
    with parent_descriptor(path, create=True) as parent:
        temporary = '.praxis-' + secrets.token_hex(12)
        os.symlink(target, temporary, dir_fd=parent)
        try:
            if file_version(os.stat(path.parent, follow_symlinks=False))[:2] != file_version(os.fstat(parent))[:2]:
                raise PermissionError('file parent changed during the operation')
            os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            if _entry_exists(parent, temporary):
                os.unlink(temporary, dir_fd=parent)

"""Publish a consistent SQLite and immutable artifact backup as one directory."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_runtime.harness.rollout import RolloutStore


def backup_store(store: RolloutStore, destination: Path) -> Path:
    """Copy a pinned database snapshot and its referenced blobs, then publish once.

    The destination's parent must exist. Existing destinations are never replaced,
    including ones created concurrently. Run on the Store owner so snapshot reads
    and SQLite's backup API use the same connection. An active write transaction
    is rejected: backing up that connection would wait on its own write lock.
    """

    destination = Path(destination)

    def backup() -> Path:
        store._ensure_open()
        if store._pending_transaction_records is not None:
            raise RuntimeError("cannot back up a Store inside a write transaction")
        if os.path.lexists(destination):
            raise FileExistsError(errno.EEXIST, "backup destination already exists", destination)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            database = temporary / "rollout.sqlite3"
            database.touch(mode=0o600)
            artifact_root = temporary / "rollout.sqlite3.artifacts"
            blobs = artifact_root / "blobs"
            blobs.mkdir(parents=True)
            with store.read_snapshot():
                with closing(sqlite3.connect(database)) as target:
                    store._connection.backup(target)
                    # A standalone backup must not depend on a sibling WAL file.
                    target.execute("PRAGMA journal_mode = DELETE")
                    references = target.execute("SELECT DISTINCT blob_sha256, size_bytes FROM artifacts").fetchall()
                    record_count, last_record_id = target.execute(
                        "SELECT COUNT(*), COALESCE(MAX(record_id), 0) FROM rollout_records"
                    ).fetchone()
                    artifact_count = target.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
                copied: dict[str, int] = {}
                for digest, size in references:
                    if (
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)
                        or type(size) is not int
                        or size < 0
                    ):
                        raise RuntimeError("invalid artifact blob reference in backup database")
                    if digest in copied:
                        if copied[digest] != size:
                            raise RuntimeError("artifact blob size mismatch in backup references")
                        continue
                    _copy_verified_blob(
                        store._artifact_root / "blobs" / digest,
                        blobs / digest,
                        digest,
                        size,
                    )
                    copied[digest] = size

            with database.open("rb") as stream:
                os.fsync(stream.fileno())
            manifest = {
                "format": "praxis-rollout-backup",
                "format_version": 1,
                "database": database.name,
                "artifacts_directory": artifact_root.name,
                "record_count": record_count,
                "last_record_id": last_record_id,
                "artifact_count": artifact_count,
                "blob_count": len(copied),
            }
            with (temporary / "manifest.json").open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            for directory in (blobs, artifact_root, temporary):
                _fsync_directory(directory)
            _publish_directory(temporary, destination)
            _fsync_directory(destination.parent)
            return destination
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    return store.run_sync(backup)


def _copy_verified_blob(source: Path, destination: Path, digest: str, expected_size: int) -> None:
    actual_digest = hashlib.sha256()
    actual_size = 0
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                actual_digest.update(chunk)
                actual_size += len(chunk)
                writer.write(chunk)
            if actual_size != expected_size:
                raise RuntimeError("artifact blob size mismatch during backup")
            if actual_digest.hexdigest() != digest:
                raise RuntimeError("artifact blob hash mismatch during backup")
            writer.flush()
            os.fsync(writer.fileno())
    except FileNotFoundError as error:
        raise RuntimeError("artifact blob is missing during backup") from error


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(source: Path, destination: Path) -> None:
    """Atomically rename without replacing even an empty destination directory."""

    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(destination), 0x4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    else:
        raise OSError(errno.ENOTSUP, "atomic non-replacing backup publication is unavailable")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)

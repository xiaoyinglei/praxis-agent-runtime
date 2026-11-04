"""Raw Git workspace snapshots; never stash, refresh or replace the source index."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_runtime.git_delivery import git_environment
from agent_runtime.tools.builtins.file_access import file_version, open_binary

MAX_SNAPSHOT_FILES = 20_000
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024


def git_bytes(root: Path, *args: str, input: bytes | None = None, index: Path | None = None) -> bytes:
    env = git_environment()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-c", f"core.hooksPath={os.devnull}", "-C", str(root), *args],
        input=input,
        capture_output=True,
        env=env,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(f"snapshot git {args[0]} failed: {os.fsdecode(result.stderr)[:400]}")
    return result.stdout


def working_entries(
    root: Path,
    *,
    object_root: Path | None = None,
    excluded: tuple[Path, ...] = (),
    authorize: Callable[[Path], None] | None = None,
    paths: tuple[str, ...] | None = None,
    observations: dict[str, tuple[int, int, int, int, int] | None] | None = None,
    include_directories: bool = False,
) -> dict[str, dict[str, str]]:
    repository = object_root or root
    if paths is not None:
        names = [os.fsencode(path) for path in paths]
    elif (root / ".git").exists():
        names = git_bytes(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0")
    else:
        names = []
        for directory, directories, files in os.walk(root, followlinks=False):
            current = Path(directory)
            directories[:] = [
                name
                for name in directories
                if not (
                    name == ".praxis"
                    and current == root
                    or any((current / name).is_relative_to(path) for path in excluded)
                )
            ]
            links = [name for name in directories if (current / name).is_symlink()]
            directories[:] = [name for name in directories if name not in links]
            names.extend(os.fsencode((current / name).relative_to(root)) for name in (*files, *links))
    entries: dict[str, dict[str, str]] = {}
    total = 0
    for raw in sorted(set(names) - {b""}):
        path = os.fsdecode(raw)
        # Runtime state is a declared private region, never a source artifact.
        if Path(path).parts[0] == ".praxis":
            continue
        target = root / path
        if any(target.is_relative_to(private) for private in excluded):
            continue
        ancestors = (root.joinpath(*Path(path).parts[:depth]) for depth in range(1, len(Path(path).parts)))
        if any(parent.is_symlink() for parent in ancestors):
            if paths is None:
                # A symlink blob replaces the old directory. Never follow its
                # stale tracked descendants while constructing the new tree.
                continue
            raise ValueError(f"snapshot path traverses symlink: {path}")
        if authorize is not None:
            authorize(target)
        try:
            info = target.lstat()
        except (FileNotFoundError, NotADirectoryError):
            if observations is not None:
                observations[path] = None
            continue
        if stat.S_ISDIR(info.st_mode) and include_directories:
            entries[path] = {"mode": "040000", "oid": ""}
            if observations is not None:
                observations[path] = file_version(info)
            continue
        if stat.S_ISDIR(info.st_mode) and (root / ".git").exists():
            indexed = git_bytes(root, "ls-files", "--stage", "--", path)
            if indexed.startswith(b"160000 "):
                raise ValueError(f"submodule snapshots are unsupported: {path}")
            # Replaced tracked files are represented by their actual descendants.
            continue
        if stat.S_ISLNK(info.st_mode):
            content, mode = os.fsencode(target.readlink()), "120000"
        elif stat.S_ISREG(info.st_mode):
            if info.st_size + total > MAX_SNAPSHOT_BYTES:
                raise ValueError("snapshot byte limit exceeded")
            with open_binary(target) as stream:
                opened = os.fstat(stream.fileno())
                if file_version(opened) != file_version(info):
                    raise RuntimeError(f"snapshot file changed: {path}")
                content = stream.read(MAX_SNAPSHOT_BYTES - total + 1)
                if file_version(os.fstat(stream.fileno())) != file_version(opened):
                    raise RuntimeError(f"snapshot file changed: {path}")
            mode = "100755" if info.st_mode & 0o111 else "100644"
        else:
            raise ValueError(f"snapshot requires regular files or symlinks (submodules unsupported): {path}")
        if file_version(target.lstat()) != file_version(info):
            raise RuntimeError(f"snapshot path changed: {path}")
        if observations is not None:
            observations[path] = file_version(info)
        total += len(content)
        if total > MAX_SNAPSHOT_BYTES or len(entries) >= MAX_SNAPSHOT_FILES:
            raise ValueError("snapshot resource limit exceeded")
        oid = git_bytes(repository, "hash-object", "-w", "--stdin", input=content).decode().strip()
        entries[path] = {"mode": mode, "oid": oid}
    return entries


def tree_from_entries(root: Path, entries: dict[str, dict[str, str]]) -> str:
    with tempfile.TemporaryDirectory(prefix="praxis-index-") as temporary:
        index = Path(temporary) / "index"
        git_bytes(root, "read-tree", "--empty", index=index)
        records = b"".join(
            f"{entry['mode']} {entry['oid']}\t".encode() + os.fsencode(path) + b"\0"
            for path, entry in sorted(entries.items())
        )
        git_bytes(root, "update-index", "-z", "--index-info", input=records, index=index)
        return git_bytes(root, "write-tree", index=index).decode().strip()


def capture_snapshot(
    root: Path,
    *,
    object_root: Path | None = None,
    excluded: tuple[Path, ...] = (),
    authorize: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    repository = object_root or root
    head = git_bytes(repository, "rev-parse", "--verify", "HEAD").decode().strip()
    index_path = Path(
        os.fsdecode(git_bytes(repository, "rev-parse", "--path-format=absolute", "--git-path", "index")).strip()
    )
    original_index = index_path.read_bytes() if index_path.exists() else b""
    staged = b""
    if (root / ".git").exists() and not any(root.is_relative_to(path) for path in excluded):
        with tempfile.TemporaryDirectory(prefix="praxis-source-index-") as directory:
            frozen_index = Path(directory) / "index"
            if original_index:
                frozen_index.write_bytes(original_index)
            else:
                git_bytes(root, "read-tree", "--empty", index=frozen_index)
            # Staged deletions are absent from the working-file enumeration.
            # Authorize both sides as independent paths; rename/copy inference
            # must not reintroduce private source bodies into the patch.
            private_paths = [".praxis"]
            for path in excluded:
                if path.is_relative_to(root):
                    private_paths.append(str(path.relative_to(root)))
            pathspec = ("--", ".", *(f":(top,literal,exclude){path}" for path in private_paths))
            options = ("diff", "--cached", "--no-renames", "--no-ext-diff", "--no-textconv", head)
            changed = git_bytes(root, *options, "--name-only", "-z", *pathspec, index=frozen_index)
            if authorize is not None:
                for raw in changed.split(b"\0"):
                    if raw:
                        authorize(root / os.fsdecode(raw))
            staged = git_bytes(root, *options, "--binary", *pathspec, index=frozen_index)
    entries = working_entries(root, object_root=repository, excluded=excluded, authorize=authorize)
    tree = tree_from_entries(repository, entries)
    if (
        working_entries(root, object_root=repository, excluded=excluded, authorize=authorize) != entries
        or git_bytes(repository, "rev-parse", "HEAD").decode().strip() != head
        or (index_path.read_bytes() if index_path.exists() else b"") != original_index
    ):
        raise RuntimeError("source changed while capturing snapshot; retry explicitly")
    return {
        "source_head": head,
        "tree": tree,
        "entries": entries,
        "source_index_sha256": hashlib.sha256(original_index).hexdigest(),
        "source_index_hex": original_index.hex(),
        "source_index_patch_hex": staged.hex(),
    }


def tree_entries(root: Path, tree: str) -> dict[str, dict[str, str]]:
    result = {}
    for field in git_bytes(root, "ls-tree", "-rz", tree).split(b"\0"):
        if not field:
            continue
        header, path = field.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        if kind != "blob":
            raise ValueError("submodule artifacts are not supported")
        result[os.fsdecode(path)] = {"mode": mode, "oid": oid}
    return result


def archive_tree(root: Path, tree: str) -> bytes:
    """Archive raw Git objects; export-ignore/export-subst must not alter evidence."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, entry in tree_entries(root, tree).items():
            content = git_bytes(root, "cat-file", "blob", entry["oid"])
            member = tarfile.TarInfo(path)
            if entry["mode"] == "120000":
                member.type, member.linkname = tarfile.SYMTYPE, os.fsdecode(content)
                archive.addfile(member)
            else:
                member.size = len(content)
                member.mode = 0o755 if entry["mode"] == "100755" else 0o644
                archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()

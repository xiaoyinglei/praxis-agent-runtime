"""Workspace runtime for agent file isolation and sandboxing."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

_SNAPSHOT_IGNORED_DIRECTORIES = frozenset(
    {
        ".agent_memory",
        ".cache",
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".praxis",
        ".rag",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
_TEMP_RUNTIME_DIRECTORIES = frozenset(
    {"artifacts", "logs", "reports", "scratch"}
)
_WORKSPACE_SNAPSHOT_REVISION = b"workspace-tree-v1\0"
DEFAULT_CHECKPOINT_PATH = Path(".praxis/checkpoints.sqlite")
DEFAULT_MODEL_SESSION_PATH = Path(".praxis/model_session.json")


class WorkspacePathError(ValueError):
    """Path escapes workspace boundary."""


@dataclass
class WorkspaceRuntime:
    """Manages a workspace directory tree with isolated scratch/artifacts."""

    root: Path
    is_temporary: bool

    @property
    def runtime_root(self) -> Path:
        """Return the agent-owned state root without shadowing project files."""

        if self.is_temporary:
            return self.root
        return self.root / ".praxis" / "runtime"

    @property
    def input_files(self) -> Path:
        return self.runtime_root / "input_files"

    @property
    def scratch(self) -> Path:
        return self.runtime_root / "scratch"

    @property
    def artifacts(self) -> Path:
        return self.runtime_root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.runtime_root / "reports"

    @property
    def logs(self) -> Path:
        return self.runtime_root / "logs"

    @property
    def agent_memory(self) -> Path:
        return self.root / ".agent_memory"

    def initialize(self) -> None:
        """Create the standard workspace subdirectories."""
        for subdir in (
            self.input_files,
            self.scratch,
            self.artifacts,
            self.reports,
            self.logs,
            self.agent_memory,
        ):
            subdir.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, relative: str | Path) -> Path:
        """Resolve a relative path against the workspace root (returns absolute)."""
        resolved = (self.root / relative).resolve()
        return resolved

    def ensure_within_workspace(self, path: Path) -> Path:
        """Ensure a path is within the workspace root; raise if it escapes."""
        resolved = path.resolve()
        workspace_root = self.root.resolve()
        if not (resolved == workspace_root or str(resolved).startswith(str(workspace_root) + os.sep)):
            raise WorkspacePathError(f"Path {path} escapes workspace boundary {self.root}")
        return resolved

    def ensure_within_scratch(self, path: Path) -> Path:
        """Ensure a path is within scratch/; raise otherwise."""
        resolved = self.ensure_within_workspace(path)
        scratch_root = self.scratch.resolve()
        if not str(resolved).startswith(str(scratch_root) + os.sep):
            raise WorkspacePathError(f"Path {path} is not within scratch/ directory")
        return resolved

    def relative_to_root(self, path: Path) -> Path:
        """Return the path relative to workspace root (after validation)."""
        resolved = self.ensure_within_workspace(path)
        return resolved.relative_to(self.root.resolve())


def create_temp_workspace(prefix: str = "agent_run_") -> WorkspaceRuntime:
    """Create a temporary workspace directory and initialize it."""
    root = Path(tempfile.mkdtemp(prefix=prefix)).resolve()
    ws = WorkspaceRuntime(root=root, is_temporary=True)
    ws.initialize()
    return ws


def open_workspace(path: str | Path, *, create: bool = False) -> WorkspaceRuntime:
    """Open an existing workspace directory, optionally creating it."""
    root = Path(path)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.exists():
        raise FileNotFoundError(f"Workspace path does not exist: {root}")
    ws = WorkspaceRuntime(root=root.resolve(), is_temporary=False)
    ws.initialize()
    return ws


def import_files(
    workspace: WorkspaceRuntime,
    sources: list[str | Path],
    *,
    namespace: str | None = None,
) -> list[Path]:
    """Resolve workspace files and stage only external files.

    Files already inside the workspace are returned in place. External files
    are copied into agent-owned storage so attaching a file never creates a
    top-level ``input_files`` directory in the user's project.
    """
    destination = workspace.input_files
    if namespace is not None:
        if not namespace or Path(namespace).name != namespace:
            raise ValueError("Input file namespace must be one path segment")
        destination /= namespace
    destination.mkdir(parents=True, exist_ok=True)

    imported: list[Path] = []
    for src in sources:
        src_path = Path(src).expanduser().resolve()
        if src_path.is_dir():
            raise ValueError(f"Directory import not supported: {src_path}")
        if not src_path.is_file():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        try:
            src_path.relative_to(workspace.root.resolve())
        except ValueError:
            pass
        else:
            imported.append(src_path)
            continue
        dest = _unique_dest(destination, src_path.name)
        shutil.copy2(src_path, dest)
        imported.append(dest)
    return imported


def workspace_tree_sha256(root: Path | str) -> str | None:
    """Hash workspace files without depending on mutable Git metadata."""

    workspace_root = Path(root).expanduser().resolve()
    if not workspace_root.is_dir():
        return None
    try:
        relative_paths = _filesystem_workspace_paths(
            workspace_root,
            ignore_runtime_root=_has_temporary_runtime_layout(workspace_root),
        )
    except OSError:
        return None

    digest = hashlib.sha256(_WORKSPACE_SNAPSHOT_REVISION)
    try:
        for relative_path in relative_paths:
            _hash_workspace_path(
                digest,
                root=workspace_root,
                relative_path=relative_path,
            )
    except OSError:
        return None
    return digest.hexdigest()


def _has_temporary_runtime_layout(root: Path) -> bool:
    return bool(
        (root / ".agent_memory").is_dir()
        and all((root / name).is_dir() for name in _TEMP_RUNTIME_DIRECTORIES)
        and not (root / ".praxis" / "runtime").exists()
    )


def _filesystem_workspace_paths(
    root: Path,
    *,
    ignore_runtime_root: bool = False,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            child = current / name
            if (
                name in _SNAPSHOT_IGNORED_DIRECTORIES
                or (
                    ignore_runtime_root
                    and current == root
                    and name in _TEMP_RUNTIME_DIRECTORIES
                )
            ):
                continue
            if child.is_symlink():
                paths.append(child.relative_to(root))
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            if name == ".git":
                continue
            paths.append((current / name).relative_to(root))
    return tuple(sorted(paths, key=lambda path: os.fsencode(path.as_posix())))


def _safe_snapshot_path(path: Path) -> bool:
    return bool(
        not path.is_absolute()
        and path.parts
        and ".." not in path.parts
        and ".git" not in path.parts
    )


def _hash_workspace_path(
    digest: hashlib._Hash,
    *,
    root: Path,
    relative_path: Path,
) -> None:
    if not _safe_snapshot_path(relative_path):
        raise OSError("unsafe workspace snapshot path")
    target = root / relative_path
    _hash_field(digest, b"path", os.fsencode(relative_path.as_posix()))
    try:
        file_stat = target.lstat()
    except FileNotFoundError:
        _hash_field(digest, b"type", b"missing")
        return

    mode = stat.S_IMODE(file_stat.st_mode)
    _hash_field(digest, b"mode", str(mode).encode("ascii"))
    if stat.S_ISLNK(file_stat.st_mode):
        _hash_field(digest, b"type", b"symlink")
        _hash_field(digest, b"target", os.fsencode(os.readlink(target)))
        return
    if stat.S_ISDIR(file_stat.st_mode):
        _hash_field(digest, b"type", b"directory")
        for nested in _filesystem_workspace_paths(target):
            _hash_workspace_path(
                digest,
                root=target,
                relative_path=nested,
            )
        return
    if not stat.S_ISREG(file_stat.st_mode):
        _hash_field(digest, b"type", b"special")
        return

    _hash_field(digest, b"type", b"file")
    _hash_field(digest, b"size", str(file_stat.st_size).encode("ascii"))
    with target.open("rb") as stream:
        opened_stat = os.fstat(stream.fileno())
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        final_stat = os.fstat(stream.fileno())
    if (
        opened_stat.st_size != final_stat.st_size
        or opened_stat.st_mtime_ns != final_stat.st_mtime_ns
    ):
        raise OSError("workspace changed while it was being snapshotted")


def _hash_field(
    digest: hashlib._Hash,
    name: bytes,
    value: bytes,
) -> None:
    digest.update(name)
    digest.update(b"\0")
    digest.update(str(len(value)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value)
    digest.update(b"\0")


def _unique_dest(directory: Path, filename: str) -> Path:
    """Generate a unique destination path, appending __N on collision."""
    dest = directory / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


__all__ = [
    "WorkspacePathError",
    "WorkspaceRuntime",
    "create_temp_workspace",
    "import_files",
    "open_workspace",
    "workspace_tree_sha256",
]

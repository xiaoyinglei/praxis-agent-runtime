"""Fail-closed readers for executor-owned workspace evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent_runtime.tools.tool import ToolResult

_MAX_RUNTIME_WORKSPACE_FILE_CHANGES = 8


def runtime_workspace_snapshot(result: ToolResult) -> tuple[str, str] | None:
    """Return a valid executor-owned before/after workspace snapshot."""

    if result.metadata.get("runtime_workspace_write") is not True:
        return None
    before_sha256 = result.metadata.get("workspace_tree_before_sha256")
    after_sha256 = result.metadata.get("workspace_tree_after_sha256")
    if not _valid_sha256(before_sha256) or not _valid_sha256(after_sha256):
        return None
    assert isinstance(before_sha256, str)
    assert isinstance(after_sha256, str)
    return before_sha256, after_sha256


def runtime_workspace_change(result: ToolResult) -> tuple[str, str, str] | None:
    """Return executor-attested workspace-tree change evidence."""

    snapshot = runtime_workspace_snapshot(result)
    if snapshot is None or snapshot[0] == snapshot[1]:
        return None
    return ".", snapshot[0], snapshot[1]


def runtime_workspace_file_changes(
    result: ToolResult,
) -> tuple[tuple[str, str, str], ...]:
    """Return executor-attested concrete file changes, failing closed."""

    if runtime_workspace_snapshot(result) is None:
        return ()
    raw_changes = result.metadata.get("runtime_workspace_file_changes")
    if not isinstance(raw_changes, Sequence) or isinstance(raw_changes, (str, bytes)):
        return ()
    if len(raw_changes) > _MAX_RUNTIME_WORKSPACE_FILE_CHANGES:
        return ()
    changes: list[tuple[str, str, str]] = []
    seen_paths: set[str] = set()
    for raw_change in raw_changes:
        if not isinstance(raw_change, Mapping):
            return ()
        path = raw_change.get("path")
        before_sha256 = raw_change.get("before_sha256")
        after_sha256 = raw_change.get("after_sha256")
        normalized_path = _normalize_workspace_path(path) if isinstance(path, str) else None
        if (
            normalized_path in {None, "."}
            or normalized_path in seen_paths
            or not _valid_sha256(before_sha256)
            or not _valid_sha256(after_sha256)
            or before_sha256 == after_sha256
        ):
            return ()
        assert normalized_path is not None
        assert isinstance(before_sha256, str)
        assert isinstance(after_sha256, str)
        seen_paths.add(normalized_path)
        changes.append((normalized_path, before_sha256, after_sha256))
    return tuple(changes)


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _normalize_workspace_path(value: str) -> str | None:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.rstrip("/")
    if normalized == ".":
        return "."
    raw_parts = normalized.split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or (len(normalized) >= 2 and normalized[1] == ":")
        or ".." in raw_parts
    ):
        return None
    return "/".join(part for part in raw_parts if part not in {"", "."}) or None


__all__ = [
    "runtime_workspace_change",
    "runtime_workspace_file_changes",
    "runtime_workspace_snapshot",
]

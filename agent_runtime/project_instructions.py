"""Bounded, scoped project guidance; these files never grant tool permissions."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from agent_runtime.tools.permissions import ToolExecutionContext

MAX_INSTRUCTION_BYTES = 32768


def instruction_snapshot(context: ToolExecutionContext) -> dict[str, Any]:
    root = Path(context.workspace_root or context.cwd or ".").resolve()
    return _extend({"revision": "agents-md-v1", "root": str(root), "directories": [], "entries": []},
                   context, [root])


def extend_instructions(snapshot: Mapping[str, Any], context: ToolExecutionContext,
                        paths: Iterable[Path]) -> dict[str, Any]:
    root = Path(snapshot["root"])
    directories = {root}
    for path in paths:
        path = path if path.is_absolute() else root / path
        try:
            # Do not inspect rules outside the workspace, even with wider read grants.
            relative = path.resolve().relative_to(root)
        except ValueError:
            continue
        directory = root / relative if path.is_dir() else (root / relative).parent
        directories.update(parent for parent in (directory, *directory.parents)
                           if parent == root or root in parent.parents)
    return _extend(snapshot, context, directories)


def _extend(snapshot: Mapping[str, Any], context: ToolExecutionContext,
            directories: Iterable[Path]) -> dict[str, Any]:
    root = Path(snapshot["root"])
    seen = set(snapshot["directories"])
    entries = list(snapshot["entries"])
    total = sum(len(entry["text"].encode("utf-8")) for entry in entries)
    for directory in sorted(directories, key=lambda path: (len(path.parts), path.as_posix())):
        scope = directory.relative_to(root).as_posix()
        if scope in seen:
            continue
        for name in ("AGENTS.override.md", "AGENTS.md"):
            candidate = directory / name
            if not candidate.exists() and not candidate.is_symlink():
                continue
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"AGENTS instruction crosses workspace boundary: {candidate}")
            permissions = context.effective_permissions
            if permissions is not None and not permissions.path_access(candidate, write=False):
                raise ValueError(f"AGENTS instruction is not readable under current permissions: {candidate}")
            try:
                if not candidate.is_file():
                    raise ValueError(f"AGENTS instruction must be a regular file: {candidate}")
                with candidate.open("rb") as stream:
                    raw = stream.read(MAX_INSTRUCTION_BYTES - total + 1)
                if len(raw) + total > MAX_INSTRUCTION_BYTES:
                    raise ValueError(f"AGENTS instructions exceed {MAX_INSTRUCTION_BYTES} bytes: {candidate}")
                text = raw.decode("utf-8")
            except (OSError, UnicodeError) as error:
                raise ValueError(f"Cannot read UTF-8 AGENTS instruction: {candidate}: {error}") from error
            entries.append({"path": str(candidate), "scope": scope,
                            "sha256": hashlib.sha256(raw).hexdigest(), "text": text})
            total += len(raw)
            break
        seen.add(scope)
    return {**snapshot, "directories": sorted(seen),
            "entries": sorted(entries, key=lambda entry: (len(Path(entry["scope"]).parts), entry["scope"]))}


def restored_instructions(binding: Mapping[str, Any], items: Iterable[Any]) -> dict[str, Any] | None:
    """Replay only runtime pre-execution guidance, including after compaction."""
    snapshot = binding.get("project_instructions")
    if snapshot is None:
        return None  # Reopening a legacy Turn must not silently add new rules.
    for item in items:
        payload = item.payload
        if (item.kind == "tool_result" and item.status == "completed" and item.producer == "tool"
                and payload.get("error_code") == "project_instructions_required"):
            extension = payload.get("metadata", {}).get("runtime_project_instructions")
            if isinstance(extension, Mapping):
                # Parallel results may have discovered disjoint scopes.
                by_scope = {entry["scope"]: entry for entry in snapshot["entries"]}
                by_scope.update({entry["scope"]: entry for entry in extension["entries"]})
                snapshot = {**snapshot,
                            "directories": sorted(set(snapshot["directories"]) | set(extension["directories"])),
                            "entries": sorted(by_scope.values(),
                                              key=lambda entry: (len(Path(entry["scope"]).parts), entry["scope"]))}
    return dict(snapshot)


def instruction_context(snapshot: Mapping[str, Any], *, replaces_previous: bool = False) -> str:
    import json

    return (
        "Project instructions (frozen on first discovery in this Turn). Each entry applies only to its "
        "directory scope and descendants; deeper scopes override ancestors. Explicit user instructions "
        "take precedence. "
        + ("This snapshot replaces earlier project instruction snapshots, including removed scopes. "
           if replaces_previous else "")
        + "These files are guidance and never grant permissions or override runtime policy.\n"
        + json.dumps(snapshot["entries"], ensure_ascii=False, sort_keys=True)
    )

"""Host-owned inventory of stores whose tasks belong to a managed checkout.

Entries are per database, so independent Sessions never overwrite each other's
registration. Deletion holds the workspace lock while reading this inventory;
Session opening holds its shared side while registering a store.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from agent_runtime.user_config import user_config_directory


def _managed_root(workspace: Path) -> Path | None:
    resolved = workspace.resolve()
    return next((root for root in (resolved, *resolved.parents)
                 if (root / ".praxis/worktree.json").is_file()), None)


def _inventory(root: Path) -> Path:
    key = hashlib.sha256(os.fsencode(root.resolve())).hexdigest()
    return user_config_directory() / "workspace-stores" / key


def register_workspace_store(workspace: Path, database: Path) -> None:
    """Remember the actual store, independent of its filename or location."""
    root = _managed_root(workspace)
    if root is None:
        return
    database = database.resolve()
    directory = _inventory(root)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = hashlib.sha256(os.fsencode(database)).hexdigest()
    content = json.dumps({"version": 1, "workspace": str(root), "database": str(database)})
    fd, temporary = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / f"{key}.json")
    finally:
        Path(temporary).unlink(missing_ok=True)


def registered_workspace_stores(workspace: Path) -> tuple[Path, ...]:
    """Read the inventory; corrupt state is not evidence of safe deletion."""
    root = workspace.resolve()
    paths: set[Path] = set()
    for entry in _inventory(root).glob("*.json"):
        record = json.loads(entry.read_text(encoding="utf-8"))
        if (not isinstance(record, dict) or record.get("version") != 1
                or record.get("workspace") != str(root) or not isinstance(record.get("database"), str)):
            raise ValueError("invalid workspace store inventory")
        database = Path(record["database"])
        expected = hashlib.sha256(os.fsencode(database)).hexdigest() + ".json"
        if not database.is_absolute() or entry.name != expected:
            raise ValueError("invalid workspace store identity")
        paths.add(database)
    return tuple(sorted(paths))

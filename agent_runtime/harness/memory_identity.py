"""Host-owned memory identities and explicit, transactional path relocation.

Workspace paths locate current bindings; they never identify a memory history.
Thread associations and frozen evidence remain unchanged when a path moves.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from agent_runtime.harness.protocol import ContextSourceChangedError

if TYPE_CHECKING:
    from agent_runtime.harness.state import RuntimeState


@dataclass(frozen=True, slots=True)
class MemoryClaim:
    scope_id: str
    binding_generation: int
    owner: str
    lease_id: str


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _path_hash(workspace: Path) -> str:
    return hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()


def _read(state: RuntimeState, key: str) -> Any:
    value = state.persistence.metadata(key)
    if value is None:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"invalid memory identity metadata: {key}") from error


def _scope(state: RuntimeState, workspace: Path) -> dict[str, Any] | None:
    binding = _read(state, "memory-binding:" + _path_hash(workspace))
    if binding is None:
        return None
    if (not isinstance(binding, dict) or binding.get("workspace") != str(workspace.resolve())
            or not isinstance(binding.get("scope_id"), str)):
        raise RuntimeError("invalid workspace memory binding")
    scope = _read(state, "memory-scope:" + binding["scope_id"])
    if (not isinstance(scope, dict) or scope.get("scope_id") != binding["scope_id"]
            or scope.get("workspace") != binding["workspace"]
            or type(scope.get("binding_generation")) is not int or scope["binding_generation"] < 1):
        raise RuntimeError("workspace memory scope binding integrity mismatch")
    return scope


def memory_scope(state: RuntimeState, workspace: Path) -> dict[str, Any] | None:
    def read() -> dict[str, Any] | None:
        with state.persistence.read_snapshot():
            return _scope(state, workspace)
    return state.run_sync(read)


def _thread_scope(state: RuntimeState, thread_id: str) -> str | None:
    scope_id = _read(state, "memory-thread-scope:" + thread_id)
    if scope_id is not None and (not isinstance(scope_id, str) or not scope_id):
        raise RuntimeError("invalid thread memory scope identity")
    return scope_id


def thread_memory_scope(state: RuntimeState, thread_id: str) -> str | None:
    return state.run_sync(lambda: _thread_scope(state, thread_id))


def _adopt_threads(state: RuntimeState, workspace: Path, scope_id: str) -> None:
    for thread in state.list_threads():
        if Path(thread.workspace).resolve() == workspace.resolve() and _thread_scope(state, thread.thread_id) is None:
            state.persistence.put_metadata(
                "memory-thread-scope:" + thread.thread_id, _canonical(scope_id), replace=False,
            )


def bind_memory_thread(state: RuntimeState, thread_id: str, workspace: Path) -> None:
    def bind() -> None:
        with state.persistence.transaction():
            thread = state.read_thread(thread_id)
            if Path(thread.workspace).resolve() != workspace.resolve():
                raise ValueError("thread belongs to a different workspace")
            # A fork follows its source identity, even if that historical path
            # now belongs to an unrelated project.
            inherited = _thread_scope(state, thread.parent_thread_id) if thread.parent_thread_id else None
            scope = _scope(state, workspace) if inherited is None else None
            scope_id = inherited if inherited is not None else scope["scope_id"] if scope is not None else None
            if scope_id is None:
                return
            existing = _thread_scope(state, thread_id)
            if existing is None:
                state.persistence.put_metadata("memory-thread-scope:" + thread_id, _canonical(scope_id), replace=False)
            elif existing != scope_id:
                raise ContextSourceChangedError("thread memory association changed")
    state.run_sync(bind)


def thread_in_memory_scope(state: RuntimeState, workspace: Path, thread_id: str) -> bool:
    def read() -> bool:
        with state.persistence.read_snapshot():
            scope = _scope(state, workspace)
            association = _thread_scope(state, thread_id)
            if scope is not None:
                return association == cast(str, scope["scope_id"])
            return association is None and Path(state.read_thread(thread_id).workspace).resolve() == workspace.resolve()
    return state.run_sync(read)


def _legacy_exists(state: RuntimeState, namespace: str) -> bool:
    return any(state.persistence.metadata(prefix + namespace) is not None
               for prefix in ("memory:", "memory-jobs:", "memory-job:"))


def _snapshot_version(raw: str, *, version: int | None = None) -> int:
    try:
        envelope = json.loads(raw)
        payload = envelope["state"]
        actual = payload["version"]
        valid = (type(actual) is int and actual > 0 and (version is None or actual == version)
                 and hashlib.sha256(_canonical(payload).encode()).hexdigest() == envelope["sha256"])
    except (ValueError, TypeError, KeyError) as error:
        raise RuntimeError("workspace memory integrity mismatch") from error
    if not valid:
        raise RuntimeError("workspace memory integrity mismatch")
    return cast(int, actual)


def _version(state: RuntimeState, namespace: str) -> int:
    latest = state.persistence.metadata("memory:" + namespace)
    return 0 if latest is None else _snapshot_version(latest)


def _assert_idle(state: RuntimeState, namespace: str) -> None:
    lease = _read(state, "memory-job:" + namespace)
    if lease is not None:
        if (not isinstance(lease, dict) or type(lease.get("expires_at")) not in {int, float}):
            raise RuntimeError("invalid memory lease metadata")
        if lease["expires_at"] > time.time():
            raise ContextSourceChangedError("memory has an active maintenance lease")
    jobs = _read(state, "memory-jobs:" + namespace) or {}
    if not isinstance(jobs, dict):
        raise RuntimeError("invalid memory jobs metadata")
    for job in jobs.values():
        if not isinstance(job, dict):
            raise RuntimeError("invalid memory maintenance job")
        if job.get("status") == "needs_reconciliation":
            raise ContextSourceChangedError("memory maintenance requires reconciliation")
        for turn_id in job.get("turn_ids", ()):
            turn = state.read_turn(turn_id)
            if turn.status not in {"completed", "failed", "cancelled"}:
                raise ContextSourceChangedError("memory maintenance Turn is unresolved")
            if (any(op.status in {"dispatched", "running", "unknown"} for op in state.list_model_operations(turn_id))
                    or any(op.requires_reconciliation for op in state.list_tool_operations(turn_id))):
                raise ContextSourceChangedError("memory maintenance operation is unresolved")


def _ensure_scope(state: RuntimeState, workspace: Path) -> dict[str, Any]:
    existing = _scope(state, workspace)
    if existing is not None:
        return existing
    legacy_id = _path_hash(workspace)
    _assert_idle(state, legacy_id)
    latest = state.persistence.metadata("memory:" + legacy_id)
    version = 0 if latest is None else _snapshot_version(latest)
    versions = []
    for number in range(1, version + 1):
        raw = state.persistence.metadata(f"memory:{legacy_id}:version:{number}")
        if raw is None:
            raise RuntimeError("workspace memory history integrity mismatch: missing version")
        _snapshot_version(raw, version=number)
        versions.append(raw)
    if latest is not None and json.loads(versions[-1]) != json.loads(latest):
        raise RuntimeError("workspace memory latest-version integrity mismatch")
    scope_id = "memory_scope_" + uuid4().hex
    scope = {"scope_id": scope_id, "workspace": str(workspace.resolve()), "binding_generation": 1, "relocations": []}
    for number, raw in enumerate(versions, start=1):
        state.persistence.put_metadata(f"memory:{scope_id}:version:{number}", raw, replace=False)
    if latest is not None:
        state.persistence.put_metadata("memory:" + scope_id, latest, replace=False)
    jobs = state.persistence.metadata("memory-jobs:" + legacy_id)
    if jobs is not None:
        state.persistence.put_metadata("memory-jobs:" + scope_id, jobs, replace=False)
    state.persistence.put_metadata("memory-scope:" + scope_id, _canonical(scope), replace=False)
    state.persistence.put_metadata("memory-binding:" + legacy_id,
                                   _canonical({"scope_id": scope_id, "workspace": scope["workspace"]}), replace=False)
    _adopt_threads(state, workspace, scope_id)
    for number in range(1, version + 1):
        state.persistence.delete_metadata(f"memory:{legacy_id}:version:{number}")
    for prefix in ("memory:", "memory-jobs:", "memory-job:"):
        state.persistence.delete_metadata(prefix + legacy_id)
    return scope


def memory_namespace(state: RuntimeState, workspace: Path, *, create: bool = False) -> str:
    def namespace() -> str:
        if create:
            with state.persistence.transaction():
                return cast(str, _ensure_scope(state, workspace)["scope_id"])
        with state.persistence.read_snapshot():
            scope = _scope(state, workspace)
            return cast(str, scope["scope_id"]) if scope is not None else _path_hash(workspace)
    return state.run_sync(namespace)


def relocate_memory(
    state: RuntimeState, previous_workspace: Path, workspace: Path, *, expected_version: int | None = None,
) -> dict[str, Any]:
    previous_workspace, workspace = previous_workspace.resolve(), workspace.resolve()
    if not workspace.is_dir():
        raise ValueError("destination workspace must be an existing directory")

    def relocate() -> dict[str, Any]:
        source_hash, destination_hash = _path_hash(previous_workspace), _path_hash(workspace)
        audit_key = f"memory-relocation:{source_hash}:{destination_hash}"
        with state.persistence.transaction():
            scope = _scope(state, previous_workspace)
            if scope is None and not _legacy_exists(state, source_hash):
                audit = _read(state, audit_key)
                destination_scope = _scope(state, workspace)
                reused = any(Path(thread.workspace).resolve() == previous_workspace
                             and _thread_scope(state, thread.thread_id) is None for thread in state.list_threads())
                if (not reused and isinstance(audit, dict) and destination_scope is not None
                        and destination_scope["scope_id"] == audit.get("scope_id")
                        and destination_scope["binding_generation"] == audit.get("binding_generation")):
                    if expected_version is not None and expected_version != audit["version"]:
                        raise ContextSourceChangedError("memory version changed before relocation")
                    return audit
                raise KeyError("source workspace has no memory scope")
            namespace = scope["scope_id"] if scope is not None else source_hash
            version = _version(state, namespace)
            if expected_version is not None and version != expected_version:
                raise ContextSourceChangedError("memory version changed before relocation")
            _assert_idle(state, namespace)
            if previous_workspace != workspace and (_scope(state, workspace) is not None
                                                    or _legacy_exists(state, destination_hash)):
                raise ContextSourceChangedError("destination workspace already has distinct memory data")
            scope = _ensure_scope(state, previous_workspace)
            if previous_workspace == workspace:
                return {**scope, "previous_workspace": str(previous_workspace), "version": version}
            result = {"scope_id": scope["scope_id"], "previous_workspace": str(previous_workspace),
                      "workspace": str(workspace), "version": version,
                      "binding_generation": scope["binding_generation"] + 1}
            updated = {**scope, "workspace": str(workspace), "binding_generation": result["binding_generation"],
                       "relocations": [*scope["relocations"], result]}
            state.persistence.put_metadata("memory-scope:" + scope["scope_id"], _canonical(updated))
            state.persistence.put_metadata(
                "memory-binding:" + destination_hash,
                _canonical({"scope_id": scope["scope_id"], "workspace": str(workspace)}), replace=False,
            )
            state.persistence.delete_metadata("memory-binding:" + source_hash)
            state.persistence.delete_metadata("memory-job:" + scope["scope_id"])
            state.persistence.put_metadata(audit_key, _canonical(result))
            _adopt_threads(state, workspace, scope["scope_id"])
            return result
    return state.run_sync(relocate)

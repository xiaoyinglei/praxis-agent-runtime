"""Explicit memory ops over owner-serialized durable facts."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agent_runtime.harness.protocol import (
    ContextSourceChangedError,
)

if TYPE_CHECKING:
    from agent_runtime.harness.memory_identity import MemoryClaim


from agent_runtime.harness.operation_values import _canonical_json, _json_object
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation


@owner_operation
def commit_memory_state(
    store: RuntimeStore,
    workspace: Path,
    *,
    expected_version: int,
    state: Mapping[str, Any],
    claim: MemoryClaim | None = None,
    job_thread_id: str | None = None,
    job_state: Mapping[str, Any] | None = None,
    job_revision: int | None = None,
) -> dict[str, Any]:
    """CAS commit; keep immutable versions for inspection and crash recovery."""
    from agent_runtime.harness.memory_identity import memory_namespace

    with store.tx.transaction():
        if claim is not None:
            _assert_memory_claim(store, workspace, claim)
        key = "memory:" + memory_namespace(store, workspace, create=True)
        if job_thread_id is not None and job_state is not None:
            write_memory_job(store, workspace, job_thread_id, job_state, claim=claim, expected_revision=job_revision)
        if store.read_memory_state(workspace)["version"] != expected_version:
            raise ContextSourceChangedError("workspace memory changed before commit")
        committed = _json_object({**state, "version": expected_version + 1}, field="memory state")
        value = _canonical_json(
            {"state": committed, "sha256": hashlib.sha256(_canonical_json(committed).encode()).hexdigest()}
        )
        store.tx.put_metadata(f"{key}:version:{expected_version + 1}", value, replace=False)
        store.tx.put_metadata(key, value, replace=True)
    return committed


@owner_operation
def write_memory_job(
    store: RuntimeStore,
    workspace: Path,
    thread_id: str,
    state: Mapping[str, Any],
    *,
    claim: MemoryClaim | None = None,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    from agent_runtime.harness.memory_identity import memory_namespace

    with store.tx.transaction():
        if claim is not None:
            _assert_memory_claim(store, workspace, claim)
        key = "memory-jobs:" + memory_namespace(store, workspace, create=True)
        jobs = store.read_memory_jobs(workspace)
        revision = jobs.get(thread_id, {}).get("revision", 0)
        if expected_revision is not None and revision != expected_revision:
            raise ContextSourceChangedError("memory job changed before commit")
        job = _json_object({**state, "revision": revision + 1}, field="memory job")
        jobs[thread_id] = job
        store.tx.put_metadata(key, _canonical_json(jobs), replace=True)
    return job


@owner_operation
def enqueue_memory_job(store: RuntimeStore, workspace: Path, thread_id: str) -> None:
    from agent_runtime.harness.memory_identity import thread_in_memory_scope

    with store.tx.transaction():
        if not thread_in_memory_scope(store, workspace, thread_id):
            raise ValueError("memory job belongs to a different workspace")
        job = store.read_memory_jobs(workspace).get(thread_id, {})
        if job.get("status") in {"pending", "running", "needs_reconciliation", "failed"}:
            return
        write_memory_job(store, workspace, thread_id, {"status": "pending", "turn_ids": []})


def _assert_memory_claim(store: RuntimeStore, workspace: Path, claim: MemoryClaim) -> None:
    scope = store.memory_scope(workspace)
    if scope is None or scope["scope_id"] != claim.scope_id or scope["binding_generation"] != claim.binding_generation:
        raise ContextSourceChangedError("memory workspace binding changed")
    raw = store.tx.metadata("memory-job:" + claim.scope_id)
    lease = {} if raw is None else json.loads(raw)
    if (
        lease.get("owner") != claim.owner
        or lease.get("expires_at", 0) <= time.time()
        or lease.get("lease_id") != claim.lease_id
        or lease.get("scope_id") != claim.scope_id
        or lease.get("binding_generation") != claim.binding_generation
    ):
        raise ContextSourceChangedError("memory workspace lease is no longer owned")


@owner_operation
def claim_memory_job(
    store: RuntimeStore,
    workspace: Path,
    *,
    owner: str,
    now: float,
    lease_seconds: float,
) -> MemoryClaim | None:
    from agent_runtime.harness.memory_identity import MemoryClaim, memory_namespace

    with store.tx.transaction():
        key = "memory-job:" + memory_namespace(store, workspace)
        row = {"value": value} if (value := store.tx.metadata(key)) is not None else None
        if row is not None and json.loads(row["value"])["expires_at"] > now:
            return None
        namespace = memory_namespace(store, workspace, create=True)
        scope = store.memory_scope(workspace)
        assert scope is not None
        claim = MemoryClaim(namespace, scope["binding_generation"], owner, uuid4().hex)
        store.tx.put_metadata(
            "memory-job:" + namespace,
            _canonical_json(
                {
                    "owner": owner,
                    "expires_at": now + lease_seconds,
                    "scope_id": claim.scope_id,
                    "binding_generation": claim.binding_generation,
                    "lease_id": claim.lease_id,
                }
            ),
            replace=True,
        )
    return claim


@owner_operation
def release_memory_job(store: RuntimeStore, workspace: Path, *, claim: MemoryClaim) -> None:
    with store.tx.transaction():
        try:
            _assert_memory_claim(store, workspace, claim)
        except ContextSourceChangedError:
            return
        store.tx.delete_metadata("memory-job:" + claim.scope_id)


def relocate_memory(
    store: RuntimeStore,
    *,
    previous_workspace: Path,
    workspace: Path,
    expected_version: int | None = None,
) -> dict[str, Any]:
    from agent_runtime.harness.memory_identity import relocate_memory

    return relocate_memory(store, previous_workspace, workspace, expected_version=expected_version)

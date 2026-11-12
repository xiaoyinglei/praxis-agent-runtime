"""Explicit artifact ops over owner-serialized durable facts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime.harness import facts, turn_ops
from agent_runtime.harness.records import (
    ArtifactSnapshot,
    ItemSnapshot,
)
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation


@owner_operation
def commit_artifact(
    store: RuntimeStore,
    *,
    turn_id: str,
    content: bytes | Path,
    media_type: str,
    name: str,
    expected_sha256: str | None = None,
    fault_injector: Callable[[str], None] | None = None,
) -> ArtifactSnapshot:
    """Retain bytes or copy a file, then atomically commit its durable identity.

    A Path is copied on the Store owner with bounded memory. The caller owns
    the source and must keep it unchanged until this operation completes.
    Filesystem installation precedes SQLite commit; failures may leave an
    unreferenced blob for GC, never a reference to an incomplete file.
    expected_sha256 binds a mutable file source to previously observed bytes.
    """
    store.tx.assert_writable()
    if not isinstance(media_type, str) or not media_type.strip():
        raise ValueError("artifact media_type must be non-empty")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("artifact name must be non-empty")
    artifact_id = f"artifact_{uuid4().hex}"
    with store.tx.transaction():
        thread_id = turn_ops._running_turn_thread_id(store, turn_id)
        # GC holds the same write boundary. It cannot unlink an old orphan
        # after we verify/reinstall it but before its reference commits.
        blob_sha256, size_bytes = store.tx.install_blob(
            content,
            expected_sha256=expected_sha256,
            fault_injector=fault_injector,
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="artifact_committed",
            producer="runtime",
            payload={
                "artifact_id": artifact_id,
                "blob_sha256": blob_sha256,
                "size_bytes": size_bytes,
                "media_type": media_type,
                "name": name,
            },
        )
        if fault_injector is not None:
            fault_injector("before_sqlite_commit")
    if fault_injector is not None:
        fault_injector("after_sqlite_commit")
    return store.read_artifact_metadata(artifact_id)


def _assert_artifact_integrity(
    store: RuntimeStore, turn_id: str, *, items: tuple[ItemSnapshot, ...] | None = None
) -> None:
    # Dispatch validates its actual conversation references. A damaged blob
    # in another Thread is still diagnosed by verify(), without disabling
    # unrelated work throughout the whole store.
    artifact_ids = (
        store.conversation_artifact_ids(turn_id)
        if items is None
        else {
            str(attachment["artifact_id"])
            for item in items
            for attachment in item.payload.get("attachments", ())
            if isinstance(attachment, Mapping) and isinstance(attachment.get("artifact_id"), str)
        }
    )
    errors = store.tx.artifact_integrity_errors(artifact_ids=artifact_ids)
    if errors:
        raise RuntimeError(f"artifact integrity check failed: {errors[0]}")


def _validate_artifact_references(store: RuntimeStore, payload: Mapping[str, Any]) -> None:
    attachments = payload.get("attachments", ())
    if not isinstance(attachments, (list, tuple)):
        raise RuntimeError("tool result artifact attachments must be a list")
    for attachment in attachments:
        if not isinstance(attachment, Mapping):
            raise RuntimeError("tool result artifact attachment is malformed")
        artifact_id = attachment.get("artifact_id")
        if not isinstance(artifact_id, str):
            raise RuntimeError("tool result references an unknown artifact")
        try:
            artifact = store.read_artifact_metadata(artifact_id)
        except KeyError:
            raise RuntimeError(f"tool result references unknown artifact: {artifact_id}") from None
        if attachment.get("media_type") != artifact.media_type or attachment.get("name") != artifact.name:
            raise RuntimeError("tool result artifact metadata conflicts with durable reference")
        errors = store.tx.artifact_integrity_errors(artifact_ids={artifact_id})
        error = errors[0] if errors else None
        if error is not None:
            raise RuntimeError(f"artifact integrity check failed: {error}")

"""Detached canonical facts and runtime projections shared across boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ThreadSnapshot:
    thread_id: str
    workspace: str
    parent_thread_id: str | None
    fork_turn_id: str | None
    active_turn_id: str | None
    head_turn_id: str | None
    head_version: int
    settings: Mapping[str, Any]
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class TurnSnapshot:
    turn_id: str
    thread_id: str
    status: str
    terminal_reason_code: str | None
    terminal_message: str | None
    predecessor_turn_id: str | None
    turn_index: int
    binding_manifest: Mapping[str, Any]
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class ItemSnapshot:
    item_id: str
    thread_id: str
    turn_id: str
    sequence: int
    kind: str
    status: str
    producer: str
    payload: Mapping[str, Any]
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    record_id: int
    record_uuid: str
    thread_id: str
    turn_id: str | None
    thread_sequence: int
    record_type: str
    payload_schema_version: int
    producer: str
    payload: Mapping[str, Any]
    payload_hash: str
    committed_at_ms: int


@dataclass(frozen=True, slots=True)
class CommittedMutation[T]:
    """A mutation result paired with exactly its committed Rollout records."""

    value: T
    records: tuple[RolloutRecord, ...]


@dataclass(frozen=True, slots=True)
class VerificationReport:
    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArtifactGCReport:
    """Filesystem cleanup may partially succeed; removed files stay removed."""

    removed: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelOperationSnapshot:
    operation_id: str
    thread_id: str
    turn_id: str
    status: str
    active_attempt_id: str
    generation: int
    request_hash: str
    context_hash: str
    tool_hash: str
    wire_hash: str
    request_ref: Mapping[str, Any]
    response_item_id: str | None
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class ModelAttemptSnapshot:
    attempt_id: str
    operation_id: str
    generation: int
    status: str
    provider_response_id: str | None
    usage: Mapping[str, Any]
    claim_owner: str | None
    lease_expires_at: float | None
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class ToolOperationSnapshot:
    operation_id: str
    thread_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    arguments_digest: str
    execution_revision: str
    effects: tuple[str, ...]
    resources: tuple[Mapping[str, Any], ...]
    idempotent: bool
    status: str
    attempt_count: int
    error_code: str | None
    requires_reconciliation: bool
    result_item_id: str | None
    approval_request_id: str | None
    claim_generation: int
    fencing_token: str | None
    claim_owner: str | None
    lease_expires_at: float | None
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class InteractionSnapshot:
    request_id: str
    thread_id: str
    turn_id: str
    kind: str
    status: str
    version: int
    operation_id: str | None
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class ApprovalSnapshot:
    request_id: str
    thread_id: str
    turn_id: str
    operation_id: str
    status: str
    scope: Mapping[str, Any]
    applied_thread_sequence: int


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    artifact_id: str
    thread_id: str
    turn_id: str
    blob_sha256: str
    size_bytes: int
    media_type: str
    name: str
    applied_thread_sequence: int


def _thread_snapshot(row: Mapping[str, Any]) -> ThreadSnapshot:
    return ThreadSnapshot(
        thread_id=row["thread_id"],
        workspace=row["workspace"],
        parent_thread_id=row["parent_thread_id"],
        fork_turn_id=row["fork_turn_id"],
        active_turn_id=row["active_turn_id"],
        head_turn_id=row["head_turn_id"],
        head_version=row["head_version"],
        settings=MappingProxyType(row["settings"]),
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _turn_snapshot(row: Mapping[str, Any]) -> TurnSnapshot:
    return TurnSnapshot(
        turn_id=row["turn_id"],
        thread_id=row["thread_id"],
        status=row["status"],
        terminal_reason_code=row["terminal_reason_code"],
        terminal_message=row["terminal_message"],
        predecessor_turn_id=row["predecessor_turn_id"],
        turn_index=row["turn_index"],
        binding_manifest=MappingProxyType(dict(row["binding_manifest"])),
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _item_snapshot(row: Mapping[str, Any]) -> ItemSnapshot:
    return ItemSnapshot(
        item_id=row["item_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        sequence=row["sequence"],
        kind=row["kind"],
        status=row["status"],
        producer=row["producer"],
        payload=MappingProxyType(dict(row["payload"])),
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _model_operation_snapshot(row: Mapping[str, Any]) -> ModelOperationSnapshot:
    return ModelOperationSnapshot(
        operation_id=row["operation_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        status=row["status"],
        active_attempt_id=row["active_attempt_id"],
        generation=row["generation"],
        request_hash=row["request_hash"],
        context_hash=row["context_hash"],
        tool_hash=row["tool_hash"],
        wire_hash=row["wire_hash"],
        request_ref=MappingProxyType(dict(row["request_ref"])),
        response_item_id=row["response_item_id"],
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _model_attempt_snapshot(row: Mapping[str, Any]) -> ModelAttemptSnapshot:
    return ModelAttemptSnapshot(
        attempt_id=row["attempt_id"],
        operation_id=row["operation_id"],
        generation=row["generation"],
        status=row["status"],
        provider_response_id=row["provider_response_id"],
        usage=MappingProxyType(dict(row["usage"])),
        claim_owner=row["claim_owner"],
        lease_expires_at=row["lease_expires_at"],
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _tool_operation_snapshot(row: Mapping[str, Any]) -> ToolOperationSnapshot:
    return ToolOperationSnapshot(
        operation_id=row["operation_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
        arguments_digest=row["arguments_digest"],
        execution_revision=row["execution_revision"],
        effects=tuple(row["effects"]),
        resources=tuple(MappingProxyType(resource) for resource in row["resources"]),
        idempotent=bool(row["idempotent"]),
        status=row["status"],
        attempt_count=row["attempt_count"],
        error_code=row["error_code"],
        requires_reconciliation=bool(row["requires_reconciliation"]),
        result_item_id=row["result_item_id"],
        approval_request_id=row["approval_request_id"],
        claim_generation=row["claim_generation"],
        fencing_token=row["fencing_token"],
        claim_owner=row["claim_owner"],
        lease_expires_at=row["lease_expires_at"],
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _interaction_snapshot(row: Mapping[str, Any]) -> InteractionSnapshot:
    return InteractionSnapshot(
        request_id=row["request_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        kind=row["kind"],
        status=row["status"],
        version=row["version"],
        operation_id=row["operation_id"],
        request=MappingProxyType(dict(row["request"])),
        response=MappingProxyType(dict(row["response"])),
        applied_thread_sequence=row["applied_thread_sequence"],
    )


def _approval_snapshot(row: Mapping[str, Any]) -> ApprovalSnapshot:
    return ApprovalSnapshot(
        request_id=row["request_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        operation_id=row["operation_id"],
        status=row["status"],
        scope=MappingProxyType(dict(row["scope"])),
        applied_thread_sequence=row["applied_thread_sequence"],
    )

"""Pure domain validation, immutable identity errors and context projection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_runtime.tools.tool import _thaw_json, json_schema_output

if TYPE_CHECKING:
    from agent_runtime.harness.runtime_store import RuntimeStore


class TurnCancellationPendingError(RuntimeError):
    """A durable cancellation closes admission to new external effects."""


class TurnInputChangedError(RuntimeError):
    """A response no longer covers the active Turn's received user input."""


class TurnChildrenPendingError(RuntimeError):
    """Parent completion cannot discard an attached allocation's exposure."""

    def __init__(self, children: tuple[str, ...]) -> None:
        super().__init__("Attached child allocations remain active: " + ", ".join(children))
        self.children = children


def _validate_resource_usage_input(resource: str, initial_used: int) -> None:
    if not isinstance(resource, str) or not resource.strip() or len(resource) > 120:
        raise ValueError("resource must be a non-empty name of at most 120 characters")
    if type(initial_used) is not int or initial_used < 0:
        raise ValueError("initial resource usage must be a non-negative integer")


class ResourceClaimConflictError(RuntimeError):
    """Another running or unknown operation owns an incompatible resource."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _context_durable_state(store: RuntimeStore, turn_id: str) -> dict[str, Any]:
    """Project non-text state that compaction must never silently erase."""

    return {
        "interactions": [
            {
                "request_id": interaction.request_id,
                "kind": interaction.kind,
                "status": interaction.status,
                "version": interaction.version,
                "operation_id": interaction.operation_id,
                "request": dict(interaction.request),
                "response": dict(interaction.response),
            }
            for interaction in store.list_interactions(turn_id)
            if interaction.status != "resolved"
        ],
        "approvals": [
            {
                "request_id": approval.request_id,
                "operation_id": approval.operation_id,
                "status": approval.status,
                "scope": dict(approval.scope),
            }
            for approval in store.list_approvals(turn_id)
            if approval.status in {"pending", "approved"}
        ],
        "tool_operations": [
            {
                "operation_id": operation.operation_id,
                "tool_call_id": operation.tool_call_id,
                "tool_name": operation.tool_name,
                "status": operation.status,
                "effects": list(operation.effects),
                "resources": [dict(resource) for resource in operation.resources],
                "requires_reconciliation": operation.requires_reconciliation,
                "error_code": operation.error_code,
            }
            for operation in store.list_tool_operations(turn_id)
            if operation.status not in {"succeeded", "failed", "denied", "cancelled", "superseded"}
            or operation.requires_reconciliation
        ],
    }


def _json_object(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_json(_thaw_json(json_schema_output(None, dict(value)))))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be JSON-serializable") from exc
    if not isinstance(normalized, dict):
        raise TypeError(f"{field} must be an object")
    return normalized


def _string_tuple(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field} must contain non-empty strings")
    return tuple(sorted(set(values)))


def _resource_tuple(
    values: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, str], ...]:
    resources: list[dict[str, str]] = []
    for value in values:
        kind = value.get("kind")
        identity = value.get("identity")
        access = value.get("access")
        if (
            not isinstance(kind, str)
            or not kind
            or not isinstance(identity, str)
            or not identity
            or access not in {"read", "write"}
        ):
            raise ValueError("tool resources require non-empty kind/identity and read/write access")
        if kind == "filesystem" and not Path(identity).is_absolute():
            raise ValueError("filesystem resource identity must be absolute")
        resources.append({"kind": kind, "identity": identity, "access": str(access)})
    unique = {(item["kind"], item["identity"], item["access"]): item for item in resources}
    return tuple(unique[key] for key in sorted(unique))


def _resource_sets_conflict(
    left: tuple[Mapping[str, Any], ...],
    right: tuple[Mapping[str, Any], ...],
) -> bool:
    for first in left:
        for second in right:
            if first.get("access") == second.get("access") == "read":
                continue
            if first.get("kind") != second.get("kind"):
                continue
            first_identity = first.get("identity")
            second_identity = second.get("identity")
            if not isinstance(first_identity, str) or not isinstance(second_identity, str):
                raise RuntimeError("durable resource claim is malformed")
            if first.get("kind") != "filesystem":
                if first_identity == second_identity:
                    return True
                continue
            first_path = Path(first_identity)
            second_path = Path(second_identity)
            if first_path.is_relative_to(second_path) or second_path.is_relative_to(first_path):
                return True
    return False

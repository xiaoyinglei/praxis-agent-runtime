"""Durable owner around the retained Tool ACI execution engine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from stat import S_ISLNK
from typing import Any, Literal
from uuid import uuid4

from agent_runtime.core.messages import tool_result_message
from agent_runtime.harness.rollout import ResourceClaimConflictError, RolloutStore
from agent_runtime.tools.executor import (
    ExecutionStartRejectedError,
    ExecutionStatus,
    ToolExecution,
    ToolExecutionRecord,
    ToolExecutor,
)
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import (
    JsonValue,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolEffect,
    ToolResult,
    ToolTarget,
)

_TOOL_LEASE_GRACE_SECONDS = 30.0
_INSPECTION_TOOL_NAMES = frozenset(
    {"find_tools", "inspect_data_file", "list_files", "read_file", "search_text"}
)
_INITIAL_INSPECTION_BUDGET = 12
_PLANNED_INSPECTION_BUDGET = 20


def tool_consumes_inspection_budget(tool: Tool) -> bool:
    """Return whether a model-visible tool is inspection-first before an edit."""

    return tool.definition.name in _INSPECTION_TOOL_NAMES or (
        ToolEffect.READ_WORKSPACE in tool.static_effects
        and ToolEffect.WRITE_WORKSPACE not in tool.static_effects
    )


def _operation_consumes_inspection_budget(operation: object) -> bool:
    tool_name = getattr(operation, "tool_name", None)
    effects = getattr(operation, "effects", ())
    return tool_name in _INSPECTION_TOOL_NAMES or (
        ToolEffect.READ_WORKSPACE.value in effects
        and ToolEffect.WRITE_WORKSPACE.value not in effects
    )


def _requires_workspace_change(store: RolloutStore, turn_id: str) -> bool:
    policy = store.read_turn(turn_id).binding_manifest.get("completion_policy", {})
    return isinstance(policy, Mapping) and policy.get("require_workspace_change") is True


def _has_committed_workspace_change(store: RolloutStore, turn_id: str) -> bool:
    result_items = {
        item.item_id: item
        for item in store.list_items(turn_id)
        if item.kind == "tool_result"
        and item.status == "completed"
        and item.producer == "tool"
    }
    for operation in store.list_tool_operations(turn_id):
        if operation.status != "succeeded" or operation.result_item_id is None:
            continue
        result = result_items.get(operation.result_item_id)
        if result is None:
            continue
        metadata = result.payload.get("metadata")
        if (
            isinstance(metadata, Mapping)
            and metadata.get("runtime_workspace_write") is True
            and metadata.get("workspace_tree_changed") is True
        ):
            return True
    return False


def inspection_tools_available(store: RolloutStore, turn_id: str) -> bool:
    if not _requires_workspace_change(store, turn_id):
        return True
    operations = tuple(
        operation
        for operation in store.list_tool_operations(turn_id)
        if operation.status == "succeeded"
    )
    if _has_committed_workspace_change(store, turn_id):
        return True
    planned = any(
        operation.tool_name == "update_plan" for operation in operations
    )
    budget = (
        _PLANNED_INSPECTION_BUDGET if planned else _INITIAL_INSPECTION_BUDGET
    )
    return sum(
        _operation_consumes_inspection_budget(operation) for operation in operations
    ) < budget


class ToolApprovalRequiredError(RuntimeError):
    def __init__(self, interaction_id: str) -> None:
        self.interaction_id = interaction_id
        super().__init__(f"tool approval required: {interaction_id}")


class ToolApprovalInvalidatedError(RuntimeError):
    def __init__(self, interaction_id: str) -> None:
        self.interaction_id = interaction_id
        super().__init__(f"tool approval invalidated: {interaction_id}")


@dataclass(frozen=True, slots=True)
class ToolReconciliationOutcome:
    status: Literal["succeeded", "failed", "cancelled"]
    result: ToolResult


class ToolOrchestrator:
    """Commit call/operation/result lifecycle while ToolExecutor enforces the ACI."""

    def __init__(
        self,
        *,
        store: RolloutStore,
        tools: Mapping[str, Tool],
        execution_context: ToolExecutionContext,
        worker_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> None:
        self._store = store
        self._tools = tools
        self._execution_context = execution_context
        self._executor = ToolExecutor(tools)
        self._worker_id = worker_id or f"worker_{uuid4().hex}"
        self._lease_seconds = lease_seconds
        self._claims: dict[str, tuple[int, str]] = {}
        self._resolved_scopes: dict[
            str,
            tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]],
        ] = {}

    async def execute(self, *, turn_id: str, call: ToolCall) -> ToolResult:
        execution_context = self._context_for_turn(turn_id)
        self._store.record_tool_call(
            turn_id=turn_id,
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            arguments=_json_mapping(call.arguments),
            origin={
                "request_id": call.origin.request_id,
                "toolset_revision": call.origin.toolset_revision,
                "exposed_tool_names": list(call.origin.exposed_tool_names),
            },
        )
        inspection_block = self._inspection_budget_result(
            turn_id=turn_id,
            call=call,
        )
        if inspection_block is not None:
            self._store.record_tool_result(
                turn_id=turn_id,
                operation_id=None,
                result=_tool_result_payload(inspection_block),
            )
            return inspection_block
        tool = self._tools.get(call.tool_name)

        async def persist(record: ToolExecutionRecord) -> None:
            if tool is None:
                raise RuntimeError("unknown tools must not create execution records")
            self._persist_execution_record(
                turn_id=turn_id,
                tool=tool,
                record=record,
            )

        def capture_preflight(
            record: ToolExecutionRecord,
            resolved: ResolvedToolUse,
        ) -> None:
            self._resolved_scopes[record.operation_id] = _durable_tool_scope(
                resolved.effects,
                resolved.targets,
            )

        execution = await self._executor.execute(
            call,
            context=execution_context,
            record_sink=persist,
            preflight_sink=capture_preflight,
        )
        if execution.result.error_code == "approval_required":
            if tool is None:
                raise RuntimeError("unknown tool cannot request approval")
            record = ToolExecutionRecord.prepare(call, tool)
            effects, resources = _durable_tool_scope(
                execution.trace.effects,
                execution.trace.targets,
            )
            interaction = self._store.request_tool_approval(
                turn_id=turn_id,
                operation_id=record.operation_id,
                tool_call_id=record.tool_call_id,
                tool_name=record.tool_name,
                arguments_digest=record.arguments_digest,
                execution_revision=tool.execution_revision,
                idempotent=record.idempotent,
                effects=effects,
                resources=resources,
                request=_approval_request(
                    execution,
                    record=record,
                    tool=tool,
                    context=execution_context,
                ),
            )
            raise ToolApprovalRequiredError(interaction.request_id)
        operation_id = None if execution.record is None else execution.record.operation_id
        self._store.record_tool_result(
            turn_id=turn_id,
            operation_id=operation_id,
            result=_tool_result_payload(execution.result),
        )
        return execution.result

    def _inspection_budget_result(
        self,
        *,
        turn_id: str,
        call: ToolCall,
    ) -> ToolResult | None:
        tool = self._tools.get(call.tool_name)
        if tool is None or not tool_consumes_inspection_budget(tool):
            return None
        if inspection_tools_available(self._store, turn_id):
            return None
        operations = self._store.list_tool_operations(turn_id)
        inspection_count = sum(
            operation.status == "succeeded"
            and _operation_consumes_inspection_budget(operation)
            for operation in operations
        )
        return ToolResult(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            is_error=True,
            error_code="inspection_budget_exhausted",
            error_message=(
                f"The Turn already completed {inspection_count} read-only "
                "inspection calls without a workspace edit. Stop mapping the "
                "repository: synthesize the evidence and call apply_patch, or "
                "call update_plan once with a focused hypothesis to unlock the "
                "final eight inspections. Do not retry this read unchanged."
            ),
            retryable=False,
        )

    async def resume_approval(self, *, turn_id: str, decision: str) -> ToolResult:
        pending = [
            interaction
            for interaction in self._store.list_interactions(turn_id)
            if interaction.kind == "tool_approval" and interaction.status == "pending"
        ]
        if len(pending) != 1:
            raise RuntimeError("resume requires exactly one pending tool approval")
        interaction = pending[0]
        if interaction.operation_id is None:
            raise RuntimeError("tool approval interaction has no operation")
        operation = self._store.read_tool_operation(interaction.operation_id)
        call = self._committed_tool_call(
            turn_id=turn_id,
            tool_call_id=operation.tool_call_id,
        )
        tool = self._tools.get(call.tool_name)
        if tool is None:
            raise RuntimeError("approved tool is no longer installed")
        execution_context = self._context_for_turn(turn_id)
        if decision == "approve":
            preflight_context = replace(
                execution_context,
                approved_tool_call_ids=(execution_context.approved_tool_call_ids - {call.tool_call_id}),
                denied_tool_call_ids=(execution_context.denied_tool_call_ids - {call.tool_call_id}),
                require_confirmation_for=(execution_context.require_confirmation_for | {call.tool_name}),
            )
            preflight = await self._executor.execute(
                call,
                context=preflight_context,
            )
            fresh_record = ToolExecutionRecord.prepare(call, tool)
            current_request = _approval_request(
                preflight,
                record=fresh_record,
                tool=tool,
                context=execution_context,
            )
            if preflight.result.error_code != "approval_required" or _approval_scope(
                interaction.request
            ) != _approval_scope(current_request):
                invalidated = self._store.invalidate_tool_approval(
                    turn_id=turn_id,
                    reason="approval scope changed before execution",
                )
                raise ToolApprovalInvalidatedError(invalidated.request_id)
        interaction = self._store.resolve_tool_approval(
            turn_id=turn_id,
            decision=decision,
        )
        record = ToolExecutionRecord(
            tool_call_id=operation.tool_call_id,
            tool_name=operation.tool_name,
            operation_id=operation.operation_id,
            arguments_digest=operation.arguments_digest,
            idempotent=operation.idempotent,
            status=ExecutionStatus.PREPARED,
            attempt_count=operation.attempt_count,
        )
        approved = decision == "approve"
        context = replace(
            execution_context,
            approved_tool_call_ids=(
                execution_context.approved_tool_call_ids | ({call.tool_call_id} if approved else set())
            ),
            denied_tool_call_ids=(
                execution_context.denied_tool_call_ids | (set() if approved else {call.tool_call_id})
            ),
        )

        async def persist(updated: ToolExecutionRecord) -> None:
            self._persist_execution_record(
                turn_id=turn_id,
                tool=tool,
                record=updated,
            )

        execution = await self._executor.execute(
            call,
            context=context,
            record=record,
            record_sink=persist,
        )
        self._store.record_tool_result(
            turn_id=turn_id,
            operation_id=operation.operation_id,
            result=_tool_result_payload(execution.result),
        )
        return execution.result

    async def recover_resolved_approval(self, *, turn_id: str) -> ToolResult:
        """Execute one approved ready operation after a pre-execute crash."""

        resolved = [
            interaction
            for interaction in self._store.list_interactions(turn_id)
            if interaction.kind == "tool_approval"
            and interaction.status == "resolved"
            and interaction.response.get("decision") == "approve"
        ]
        if len(resolved) != 1 or resolved[0].operation_id is None:
            raise RuntimeError("recovery requires one resolved approved operation")
        interaction = resolved[0]
        operation_id = interaction.operation_id
        if operation_id is None:
            raise RuntimeError("resolved approval lost its operation identity")
        operation = self._store.read_tool_operation(operation_id)
        if operation.status != "ready":
            raise RuntimeError("approved operation is not ready for crash recovery")
        call = self._committed_tool_call(
            turn_id=turn_id,
            tool_call_id=operation.tool_call_id,
        )
        tool = self._tools.get(call.tool_name)
        if tool is None:
            raise RuntimeError("approved tool is no longer installed")
        execution_context = self._context_for_turn(turn_id)
        preflight_context = replace(
            execution_context,
            approved_tool_call_ids=(execution_context.approved_tool_call_ids - {call.tool_call_id}),
            denied_tool_call_ids=(execution_context.denied_tool_call_ids - {call.tool_call_id}),
            require_confirmation_for=(execution_context.require_confirmation_for | {call.tool_name}),
        )
        preflight = await self._executor.execute(call, context=preflight_context)
        fresh_record = ToolExecutionRecord.prepare(call, tool)
        current_request = _approval_request(
            preflight,
            record=fresh_record,
            tool=tool,
            context=execution_context,
        )
        if preflight.result.error_code != "approval_required" or _approval_scope(
            interaction.request
        ) != _approval_scope(current_request):
            invalidated = self._store.invalidate_resolved_tool_approval(
                turn_id=turn_id,
                reason="approval scope changed before recovered execution",
            )
            raise ToolApprovalInvalidatedError(invalidated.request_id)

        record = ToolExecutionRecord(
            tool_call_id=operation.tool_call_id,
            tool_name=operation.tool_name,
            operation_id=operation.operation_id,
            arguments_digest=operation.arguments_digest,
            idempotent=operation.idempotent,
            status=ExecutionStatus.PREPARED,
            attempt_count=operation.attempt_count,
        )
        context = replace(
            execution_context,
            approved_tool_call_ids=(execution_context.approved_tool_call_ids | {call.tool_call_id}),
            denied_tool_call_ids=(execution_context.denied_tool_call_ids - {call.tool_call_id}),
        )

        async def persist(updated: ToolExecutionRecord) -> None:
            self._persist_execution_record(
                turn_id=turn_id,
                tool=tool,
                record=updated,
            )

        execution = await self._executor.execute(
            call,
            context=context,
            record=record,
            record_sink=persist,
        )
        self._store.record_tool_result(
            turn_id=turn_id,
            operation_id=operation.operation_id,
            result=_tool_result_payload(execution.result),
        )
        return execution.result

    def reconcile(
        self,
        *,
        turn_id: str,
        outcome: ToolReconciliationOutcome,
    ) -> ToolResult:
        if not isinstance(outcome, ToolReconciliationOutcome):
            raise TypeError("outcome must be a ToolReconciliationOutcome")
        pending = [
            interaction
            for interaction in self._store.list_interactions(turn_id)
            if interaction.kind == "tool_reconciliation" and interaction.status == "pending"
        ]
        if len(pending) != 1 or pending[0].operation_id is None:
            raise RuntimeError("reconcile requires one pending tool reconciliation")
        operation = self._store.read_tool_operation(pending[0].operation_id)
        result = outcome.result
        if result.tool_call_id != operation.tool_call_id or result.tool_name != operation.tool_name:
            raise RuntimeError("reconciled result does not match the unknown operation")
        if (outcome.status == "succeeded") == result.is_error:
            raise RuntimeError("reconciled status contradicts the ToolResult")
        revision = result.metadata.get("reconciler_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise RuntimeError("reconciled ToolResult requires reconciler_revision")
        self._store.resolve_tool_reconciliation(
            turn_id=turn_id,
            status=outcome.status,
            result=_tool_result_payload(result),
            reconciler_revision=revision,
        )
        return result

    def _context_for_turn(self, turn_id: str) -> ToolExecutionContext:
        active_skill_ids = set(self._execution_context.active_skill_ids)
        for item in self._store.list_items(turn_id):
            if item.kind != "tool_result" or item.status != "completed":
                continue
            if item.payload.get("tool_name") != "invoke_skill":
                continue
            if item.payload.get("is_error") is True:
                continue
            structured = item.payload.get("structured_content")
            if not isinstance(structured, Mapping):
                continue
            if structured.get("event_type") != "skill_activation":
                continue
            skill_id = structured.get("skill_id")
            if structured.get("success") is True and isinstance(skill_id, str):
                active_skill_ids.add(skill_id)
        return replace(
            self._execution_context,
            active_skill_ids=frozenset(active_skill_ids),
        )

    def _persist_execution_record(
        self,
        *,
        turn_id: str,
        tool: Tool,
        record: ToolExecutionRecord,
    ) -> None:
        existing = {item.operation_id: item for item in self._store.list_tool_operations(turn_id)}.get(
            record.operation_id
        )
        if record.status is ExecutionStatus.PREPARED:
            if existing is not None:
                if existing.status != "ready":
                    raise RuntimeError("prepared execution record requires a ready operation")
                return
            self._write_operation_state(
                turn_id=turn_id,
                tool=tool,
                record=record,
                status="prepared",
            )
            self._write_operation_state(
                turn_id=turn_id,
                tool=tool,
                record=record,
                status="ready",
            )
            return
        if record.status is ExecutionStatus.STARTED:
            lease_seconds = max(
                self._lease_seconds,
                tool.timeout_seconds + _TOOL_LEASE_GRACE_SECONDS,
            )
            try:
                claim = self._store.claim_tool_operation(
                    operation_id=record.operation_id,
                    worker_id=self._worker_id,
                    lease_seconds=lease_seconds,
                )
            except ResourceClaimConflictError as conflict:
                self._store.reject_ready_tool_operation(
                    operation_id=record.operation_id,
                    error_code="resource_busy",
                )
                raise ExecutionStartRejectedError(
                    "resource_busy",
                    str(conflict),
                ) from None
            if claim.fencing_token is None:
                raise RuntimeError("claimed tool operation has no fencing token")
            if claim.attempt_count != record.attempt_count:
                raise RuntimeError("tool runner attempt count differs from durable claim")
            self._claims[record.operation_id] = (
                claim.claim_generation,
                claim.fencing_token,
            )
            return
        status = {
            ExecutionStatus.COMPLETED: "succeeded",
            ExecutionStatus.FAILED: "failed",
            ExecutionStatus.OUTCOME_UNKNOWN: "unknown",
        }.get(record.status)
        if status is None:
            raise RuntimeError(f"unsupported ToolExecutionRecord status: {record.status}")
        active_claim = self._claims.pop(record.operation_id, None)
        if active_claim is None:
            if (
                existing is not None
                and existing.status == "failed"
                and existing.error_code == record.error_code == "resource_busy"
            ):
                return
            raise RuntimeError("tool outcome has no active durable claim")
        accepted = self._store.commit_tool_operation_outcome(
            operation_id=record.operation_id,
            claim_generation=active_claim[0],
            fencing_token=active_claim[1],
            status=status,
            attempt_count=record.attempt_count,
            error_code=record.error_code,
            requires_reconciliation=record.requires_reconciliation,
        )
        if not accepted:
            raise RuntimeError("stale worker cannot commit tool operation outcome")

    def _write_operation_state(
        self,
        *,
        turn_id: str,
        tool: Tool,
        record: ToolExecutionRecord,
        status: str,
    ) -> None:
        effects, resources = self._resolved_scopes.get(
            record.operation_id,
            ((), ()),
        )
        self._store.record_tool_execution_state(
            turn_id=turn_id,
            operation_id=record.operation_id,
            tool_call_id=record.tool_call_id,
            tool_name=record.tool_name,
            arguments_digest=record.arguments_digest,
            execution_revision=tool.execution_revision,
            idempotent=record.idempotent,
            status=status,
            attempt_count=record.attempt_count,
            error_code=record.error_code,
            requires_reconciliation=record.requires_reconciliation,
            effects=effects,
            resources=resources,
        )

    def _committed_tool_call(self, *, turn_id: str, tool_call_id: str) -> ToolCall:
        matches = [
            item
            for item in self._store.list_items(turn_id)
            if item.kind == "tool_call" and item.payload.get("tool_call_id") == tool_call_id
        ]
        if len(matches) != 1:
            raise RuntimeError("approval must bind exactly one committed tool call")
        payload = matches[0].payload
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        origin = payload.get("origin")
        if not isinstance(tool_name, str) or not isinstance(arguments, Mapping) or not isinstance(origin, Mapping):
            raise RuntimeError("committed tool call payload is malformed")
        request_id = origin.get("request_id")
        toolset_revision = origin.get("toolset_revision")
        exposed = origin.get("exposed_tool_names")
        if (
            not isinstance(request_id, str)
            or not isinstance(toolset_revision, str)
            or not isinstance(exposed, (list, tuple))
            or any(not isinstance(name, str) for name in exposed)
        ):
            raise RuntimeError("committed tool call origin is malformed")
        return ToolCall(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            origin=ToolCallOrigin(
                request_id=request_id,
                toolset_revision=toolset_revision,
                exposed_tool_names=tuple(exposed),
            ),
        )


def _tool_result_payload(result: ToolResult) -> dict[str, Any]:
    payload = {
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "content": [{"type": block.type, "data": _json_mapping(block.data)} for block in result.content],
        "structured_content": _json_value(result.structured_content),
        "is_error": result.is_error,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "retryable": result.retryable,
        "truncated": result.truncated,
        "metadata": _json_mapping(result.metadata),
        "attachments": [
            {
                "artifact_id": attachment.artifact_id,
                "media_type": attachment.media_type,
                "name": attachment.name,
            }
            for attachment in result.attachments
        ],
    }
    payload["model_content"] = tool_result_message(result).content
    return payload


def _durable_tool_scope(
    effects: frozenset[ToolEffect] | tuple[ToolEffect, ...],
    targets: tuple[ToolTarget, ...],
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    frozen_effects = tuple(sorted(effect.value for effect in effects))
    access = "write" if ToolEffect.WRITE_WORKSPACE in effects else "read"
    resources: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for target in targets:
        if target.kind not in {"workspace_path", "cwd_path"}:
            continue
        identity = str(Path(target.value).expanduser().resolve())
        resource = {
            "kind": "filesystem",
            "identity": identity,
            "access": access,
        }
        resources[("filesystem", identity, access)] = resource
    return frozen_effects, tuple(resources[key] for key in sorted(resources))


def _approval_request(
    execution: ToolExecution,
    *,
    record: ToolExecutionRecord,
    tool: Tool,
    context: ToolExecutionContext,
) -> dict[str, Any]:
    return {
        "reason": execution.result.error_message or "approval required",
        "approval_scope": execution.result.metadata.get("approval_scope", "tool"),
        "effects": [effect.value for effect in execution.trace.effects],
        "targets": [
            {
                "kind": target.kind,
                "value": target.value,
                "identity": _target_identity(target.kind, target.value),
            }
            for target in execution.trace.targets
        ],
        "arguments_digest": record.arguments_digest,
        "execution_revision": tool.execution_revision,
        "policy_revision": _policy_revision(context),
    }


def _approval_scope(request: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "approval_scope",
        "effects",
        "targets",
        "arguments_digest",
        "execution_revision",
        "policy_revision",
    )
    return {key: request.get(key) for key in keys}


def _policy_revision(context: ToolExecutionContext) -> str:
    payload = {
        "workspace_root": (None if context.workspace_root is None else str(context.workspace_root)),
        "cwd": None if context.cwd is None else str(context.cwd),
        "allow_write_tools": context.allow_write_tools,
        "allow_execute_tools": context.allow_execute_tools,
        "active_skill_ids": sorted(context.active_skill_ids),
        "deny_effects": sorted(effect.value for effect in context.deny_effects),
        "max_parallel_calls": context.max_parallel_calls,
        "require_confirmation_for": sorted(context.require_confirmation_for),
        "denied_tool_names": sorted(context.denied_tool_names),
        "auto_approve_sandboxed": context.auto_approve_sandboxed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _target_identity(kind: str, value: str) -> dict[str, Any] | None:
    if kind not in {"workspace_path", "cwd_path"}:
        return None
    path = Path(value).expanduser()
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        ancestor = path.parent
        missing_parts = [path.name]
        while True:
            try:
                ancestor_metadata = ancestor.lstat()
                break
            except FileNotFoundError:
                if ancestor == ancestor.parent:
                    raise RuntimeError(f"target has no resolvable ancestor: {path}") from None
                missing_parts.append(ancestor.name)
                ancestor = ancestor.parent
        return {
            "exists": False,
            "resolved_ancestor": str(ancestor.resolve(strict=True)),
            "ancestor_device": ancestor_metadata.st_dev,
            "ancestor_inode": ancestor_metadata.st_ino,
            "missing_suffix": "/".join(reversed(missing_parts)),
        }
    return {
        "exists": True,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "is_symlink": S_ISLNK(metadata.st_mode),
        "symlink_target": str(path.readlink()) if path.is_symlink() else None,
        "resolved_path": str(path.resolve(strict=True)),
    }


def _json_mapping(value: Mapping[str, JsonValue]) -> dict[str, Any]:
    return {key: _json_value(item) for key, item in value.items()}


def _json_value(value: JsonValue | None) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value

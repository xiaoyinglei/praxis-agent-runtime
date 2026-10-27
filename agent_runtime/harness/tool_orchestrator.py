"""Durable owner around the retained Tool ACI execution engine."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from stat import S_ISLNK
from typing import Any, Literal
from uuid import uuid4

from agent_runtime.approval import CommandGrant, command_grant
from agent_runtime.checkpoints import (
    CheckpointError,
    CheckpointService,
    CheckpointSnapshot,
    CheckpointUnavailableError,
)
from agent_runtime.core.messages import tool_result_message
from agent_runtime.core.runtime_diagnostics import exception_diagnostic
from agent_runtime.harness.events import commit_and_publish
from agent_runtime.harness.rollout import ResourceClaimConflictError, RolloutStore
from agent_runtime.harness.verification import classify_tool_verification
from agent_runtime.streaming.events import (
    ItemDeltaKind,
    TurnItemKind,
    derive_operation_public_item_id,
    item_delta,
)
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.tools.executor import (
    ExecutionStartRejectedError,
    ExecutionStatus,
    ToolExecution,
    ToolExecutionRecord,
    ToolExecutor,
    _network_approval_id,
)
from agent_runtime.tools.permissions import ToolExecutionContext, UseToolDecision, can_use_tool
from agent_runtime.tools.tool import (
    ArtifactReference,
    CancellationMode,
    JsonValue,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolContentBlock,
    ToolContinuation,
    ToolEffect,
    ToolProgress,
    ToolProgressKind,
    ToolResult,
    ToolTarget,
)

_TOOL_LEASE_GRACE_SECONDS = 30.0

logger = logging.getLogger(__name__)

_ACTIVE_TOOL_BINDING: ContextVar[Mapping[str, Any] | None] = ContextVar("praxis_tool_binding", default=None)


def current_tool_binding() -> Mapping[str, Any] | None:
    return _ACTIVE_TOOL_BINDING.get()


_ACTIVE_TOOL_TURN_ID: ContextVar[str | None] = ContextVar(
    "praxis_active_tool_turn_id",
    default=None,
)


def current_tool_turn_id() -> str:
    """Return the trusted Turn identity for the currently executing tool."""

    turn_id = _ACTIVE_TOOL_TURN_ID.get()
    if turn_id is None:
        raise RuntimeError("tool runner is not executing inside a Harness Turn")
    return turn_id


class ToolApprovalRequiredError(RuntimeError):
    def __init__(self, interaction_id: str) -> None:
        self.interaction_id = interaction_id
        super().__init__(f"tool approval required: {interaction_id}")


class ToolApprovalInvalidatedError(RuntimeError):
    def __init__(self, interaction_id: str) -> None:
        self.interaction_id = interaction_id
        super().__init__(f"tool approval invalidated: {interaction_id}")


class ToolResultCommitError(RuntimeError):
    """A result could not be accepted; its diagnostic is not a durable outcome."""

    def __init__(self, result: ToolResult) -> None:
        self.result = result
        diagnostic = result.metadata.get("diagnostic")
        detail = diagnostic.get("message", "") if isinstance(diagnostic, Mapping) else ""
        super().__init__(f"tool result commit failed: {result.tool_call_id}: {detail}")


@dataclass(frozen=True, slots=True)
class ToolReconciliationOutcome:
    status: Literal["succeeded", "failed", "cancelled"]
    result: ToolResult


class ToolOrchestrator:
    """Commit call/operation/result lifecycle while ToolExecutor enforces the ACI."""

    async def restore_command_sessions(self, sessions: Any, turn_id: str, *, include_terminal: bool = False) -> None:
        """Claim only jobs whose complete launch is in this Turn's trusted ledger."""
        from agent_runtime.tools.builtins.shell import RunCommandInput

        items = await self._store.run_async(lambda: {item.item_id: item for item in self._store.list_items(turn_id)})
        terminal: set[tuple[str, str]] = set()
        operations = await self._store.run_async(self._store.list_tool_operations, turn_id)
        observed_results = {
            operation.result_item_id for operation in operations
            if operation.tool_name == "manage_command" and operation.status == "succeeded"
        }
        for evidence_item in items.values():
            payload = evidence_item.payload
            output = payload.get("structured_content")
            metadata = payload.get("metadata", {})
            origin = metadata.get("runtime_workspace_origin_operation_id")
            if (evidence_item.kind == "tool_result" and evidence_item.producer == "tool"
                    and evidence_item.item_id in observed_results and isinstance(origin, str)
                    and isinstance(output, Mapping) and output.get("status") in {"completed", "cancelled"}):
                key = output.get("session_id")
                if isinstance(key, str):
                    terminal.add((origin, key))
            if (evidence_item.kind == "tool_continuation" and evidence_item.producer == "runtime"
                    and isinstance(payload.get("key"), str) and isinstance(payload.get("operation_id"), str)
                    and isinstance(output, Mapping) and output.get("session_id") == payload["key"]
                    and output.get("status") in {"completed", "cancelled"}):
                original = next((operation for operation in operations
                                 if operation.operation_id == payload["operation_id"]), None)
                if original is not None and original.result_item_id == payload.get("source_result_item_id"):
                    terminal.add((original.operation_id, payload["key"]))
        for operation in operations:
            if operation.tool_name != "run_command" or operation.status != "succeeded":
                continue
            item = items.get(operation.result_item_id or "")
            if item is None or item.producer != "tool":
                continue
            output = item.payload.get("structured_content")
            if not isinstance(output, Mapping) or output.get("status") != "running":
                continue
            key, digest = output.get("session_id"), output.get("request_digest")
            if (not isinstance(key, str) or not isinstance(digest, str)
                    or ((operation.operation_id, key) in terminal and not include_terminal)):
                continue
            tool = self._tools.get("run_command")
            if tool is None or operation.execution_revision != tool.execution_revision:
                raise RuntimeError("command execution revision changed; reconcile the original job explicitly")
            call = await self._store.run_async(
                self._committed_tool_call, turn_id=turn_id, tool_call_id=operation.tool_call_id
            )
            await sessions.restore(key, request=RunCommandInput.model_validate(dict(call.arguments)),
                                   digest=digest, operation_id=operation.operation_id)
            if (operation.operation_id, key) in terminal:
                continue  # Terminal log inspection cannot attribute later host edits.
            metadata = item.payload.get("metadata", {})
            root = Path(sessions.workspace.root) if "write_workspace" in operation.effects else None
            before = metadata.get("workspace_tree_after_sha256")
            self._executor.restore_continuation(sessions, key, operation_id=operation.operation_id,
                                               workspace_root=root, before_sha256=before)

    def __init__(
        self,
        *,
        store: RolloutStore,
        tools: Mapping[str, Tool],
        execution_context: ToolExecutionContext,
        worker_id: str | None = None,
        lease_seconds: float = 300.0,
        event_dispatcher: TurnEventDispatcher | None = None,
        checkpoints: CheckpointService | None = None,
    ) -> None:
        self._store = store
        self._tools = tools
        self._execution_context = execution_context
        self._executor = ToolExecutor(tools, artifact_sink=self._save_output_artifact)
        self._worker_id = worker_id or f"worker_{uuid4().hex}"
        self._lease_seconds = lease_seconds
        self._event_dispatcher = event_dispatcher
        self._checkpoints = checkpoints
        self._claims: dict[str, tuple[int, str]] = {}
        self._resolved_scopes: dict[
            str,
            tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]],
        ] = {}
        self._instruction_targets: dict[tuple[str, str], tuple[ToolTarget, ...]] = {}
        self._instruction_snapshots: dict[str, Mapping[str, Any]] = {}

    async def _save_output_artifact(self, content: bytes, media_type: str, name: str) -> ArtifactReference:
        # The Store remains the sole owner of durable artifact identities.
        # Executor workers hand back bytes, never a file URI masquerading as one.
        turn_id = current_tool_turn_id()
        artifact = await self._commit(lambda: self._store.commit_artifact(
            turn_id=turn_id, content=content, media_type=media_type, name=name,
        ))
        return ArtifactReference(artifact.artifact_id, artifact.media_type, artifact.name)

    def attach_event_dispatcher(
        self,
        event_dispatcher: TurnEventDispatcher,
    ) -> None:
        self._event_dispatcher = event_dispatcher

    def release_turn_state(self) -> None:
        """Drop transient execution bookkeeping after all Turn I/O has stopped."""
        self._resolved_scopes.clear()
        self._claims.clear()
        self._instruction_targets.clear()
        self._instruction_snapshots.clear()

    async def close_continuation(
        self, continuation: ToolContinuation, result: Mapping[str, JsonValue],
    ) -> None:
        metadata = await self._executor.observe_continuation(continuation)
        operation_id = metadata.get("runtime_workspace_origin_operation_id")
        if isinstance(operation_id, str):
            await self._commit(lambda: self._store.record_tool_continuation(
                operation_id=operation_id, key=continuation.key,
                result={**dict(result), "metadata": dict(metadata)},
            ))

    async def reconcile_command_observation(self, sessions: Any, turn_id: str) -> bool:
        """Observe an authorized original job after its wait/status client died.

        Called only while the Session holds execution ownership. This never
        repeats a launch or cancellation, and a lost worker stays uncertain.
        """
        from agent_runtime.tools.builtins.command_sessions import ManageCommandInput
        from agent_runtime.tools.builtins.shell import _normalize_command_output

        pending = await self._store.run_async(
            lambda: [
                item
                for item in self._store.list_interactions(turn_id)
                if item.kind == "tool_reconciliation" and item.status == "pending"
            ]
        )
        if len(pending) != 1 or pending[0].operation_id is None:
            return False
        operation = await self._store.run_async(self._store.read_tool_operation, pending[0].operation_id)
        tool = self._tools.get("manage_command")
        if (operation.tool_name != "manage_command" or operation.status != "unknown"
                or tool is None or operation.execution_revision != tool.execution_revision):
            return False
        call = await self._store.run_async(
            self._committed_tool_call, turn_id=turn_id, tool_call_id=operation.tool_call_id
        )
        request = ManageCommandInput.model_validate(dict(call.arguments))
        if request.action not in {"wait", "status"}:
            return False
        # Restoring repeats the launch receipt, digest, revision and permission
        # checks. It attaches the same worker; it does not execute its command.
        await self.restore_command_sessions(sessions, turn_id, include_terminal=True)
        output = await sessions.manage(request.session_id, action="status", wait_seconds=0)
        if (output.session_id != request.session_id
                or output.sandbox_error not in {None, "command_permissions_changed"}):
            return False
        continuation = sessions.continuation(call.arguments, output.model_dump(mode="json"))
        if continuation is None:
            return False
        normalized = _normalize_command_output(output)
        metadata = await self._executor.observe_continuation(continuation)
        result = ToolResult(
            tool_call_id=call.tool_call_id, tool_name=call.tool_name,
            structured_content=normalized.structured_content, is_error=normalized.is_error,
            error_code=normalized.error_code, error_message=normalized.error_message,
            truncated=output.truncated,
            metadata={**dict(metadata), "reconciler_revision": "durable-command-observation-v1"},
        )
        status: Literal["succeeded", "failed", "cancelled"] = (
            "cancelled" if output.status == "cancelled" else "failed" if result.is_error else "succeeded"
        )

        def commit() -> None:
            with self._store._transaction():
                self.reconcile(turn_id=turn_id, outcome=ToolReconciliationOutcome(status=status, result=result))
                origin = metadata.get("runtime_workspace_origin_operation_id")
                if continuation.terminal and isinstance(origin, str):
                    self._store.record_tool_continuation(operation_id=origin, key=continuation.key, result={
                        "structured_content": output.model_dump(mode="json"), "is_error": normalized.is_error,
                        "error_code": normalized.error_code, "error_message": normalized.error_message,
                        "metadata": dict(metadata),
                    })
        await self._commit(commit)
        return True

    async def _commit[T](self, operation: Callable[[], T]) -> T:
        return await commit_and_publish(self._store, operation, self._event_dispatcher)

    def update_execution_context(self, context: ToolExecutionContext) -> None:
        self._execution_context = context

    def _step_snapshot(self, turn_id: str, call: ToolCall) -> Mapping[str, Any] | None:
        for operation in self._store.list_model_operations(turn_id):
            if operation.request_ref.get("request_id") == call.origin.request_id:
                snapshot = operation.request_ref.get("step_snapshot")
                if not isinstance(snapshot, Mapping):
                    raise RuntimeError("model tool call has no durable Step snapshot")
                return {**snapshot, "input_revision": self._store.request_input_revision(
                    turn_id, operation.request_ref,
                )}
        return None

    def _context_for_call(self, turn_id: str, call: ToolCall) -> ToolExecutionContext:
        snapshot = self._step_snapshot(turn_id, call)
        if snapshot is None:
            # Direct ACI calls do not originate from a model operation.
            return self._context_for_turn(turn_id)
        revisions = {item["name"]: item["revision"] for item in snapshot["tools"]}
        if call.tool_name in revisions:
            self.restore_tools({call.tool_name: revisions[call.tool_name]})
        from agent_runtime.tools.permissions import restore_tool_policy

        policy = snapshot["binding_manifest"].get("tool_execution_policy", {})
        return self._context_for_turn(turn_id, base=restore_tool_policy(self._execution_context, policy))

    def restore_tools(self, revisions: Mapping[str, str]) -> tuple[Tool, ...]:
        restored = []
        for name, revision in revisions.items():
            tool = self._tools.get(name)
            if tool is None or tool.execution_revision != revision:
                raise RuntimeError(f"original operation tool is unavailable: {name}")
            restored.append(tool)
        return tuple(restored)

    def _pending_record(self, turn_id: str, call: ToolCall) -> ToolExecutionRecord | None:
        operation = next(
            (op for op in self._store.list_tool_operations(turn_id) if op.tool_call_id == call.tool_call_id), None
        )
        if operation is None:
            return None
        if operation.status not in {"prepared", "ready"} or operation.attempt_count:
            raise RuntimeError("tool call already has an operation; use its durable recovery protocol")
        return ToolExecutionRecord(
            tool_call_id=operation.tool_call_id,
            tool_name=operation.tool_name,
            operation_id=operation.operation_id,
            arguments_digest=operation.arguments_digest,
            idempotent=operation.idempotent,
            status=ExecutionStatus.PREPARED,
        )

    def input_superseded(self, turn_id: str, call: ToolCall) -> bool:
        snapshot = self._step_snapshot(turn_id, call)
        revision = None if snapshot is None else snapshot.get("input_revision")
        return revision is not None and revision != self._store.user_input_revision(turn_id)

    def _guard_input(self, turn_id: str, call: ToolCall) -> None:
        if self.input_superseded(turn_id, call):
            raise ExecutionStartRejectedError("input_superseded", "New user input superseded this unstarted action.")
        from agent_runtime.project_instructions import extend_instructions, restored_instructions

        snapshot = self._step_snapshot(turn_id, call)
        binding = self._store.read_turn(turn_id).binding_manifest if snapshot is None else snapshot["binding_manifest"]
        instructions = restored_instructions(binding, self._store.list_items(turn_id))
        if instructions is None:
            return
        # Pre-execution guards run synchronously. Reserve discovered guidance
        # before another parallel call reads it, so the shared byte bound holds.
        instructions = self._instruction_snapshots.get(turn_id, instructions)
        targets = self._instruction_targets.get((turn_id, call.tool_call_id), ())
        paths = [Path(target.value) for target in targets if target.kind in {"workspace_path", "cwd_path"}]
        try:
            extended = extend_instructions(instructions, self._context_for_call(turn_id, call).refreshed(), paths)
        except ValueError as error:
            raise ExecutionStartRejectedError("project_instructions_unreadable", str(error)) from error
        self._instruction_snapshots[turn_id] = extended
        visible = binding.get("project_instructions", {})
        if extended["entries"] != visible.get("entries", []):
            raise ExecutionStartRejectedError(
                "project_instructions_required",
                "Action deferred: new scoped AGENTS instructions were discovered. Read the runtime guidance "
                "in the next request, adjust the plan if needed, and retry this action.",
                metadata={"runtime_project_instructions": extended, "execution_started": False},
                retryable=True,
            )

    async def _discard_superseded(self, turn_id: str, call: ToolCall) -> ToolResult:
        operation = await self._store.run_async(
            lambda: next(
                (op for op in self._store.list_tool_operations(turn_id) if op.tool_call_id == call.tool_call_id), None
            )
        )
        if operation is not None and operation.status in {"prepared", "ready"}:
            if operation.status == "prepared":
                record = await self._store.run_async(self._pending_record, turn_id, call)
                assert record is not None
                await self._persist_execution_record(turn_id=turn_id, tool=self._tools[call.tool_name], record=record)
            await self._commit(lambda: self._store.reject_ready_tool_operation(
                operation_id=operation.operation_id, error_code="input_superseded",
            ))
        result = ToolResult(
            tool_call_id=call.tool_call_id, tool_name=call.tool_name, is_error=True,
            error_code="input_superseded", error_message="Action not executed: superseded by new user input.",
            retryable=False, content=(), metadata={"execution_started": False},
        )
        await self._commit(lambda: self._store.record_tool_result(
            turn_id=turn_id, operation_id=None if operation is None else operation.operation_id,
            result=_tool_result_payload(result),
        ))
        return result

    async def execute(self, *, turn_id: str, call: ToolCall) -> ToolResult:
        task = asyncio.current_task()
        cancellations = 0 if task is None else task.cancelling()
        await self._record_call(turn_id, call)
        if await self._store.run_async(self.input_superseded, turn_id, call):
            return await self._discard_superseded(turn_id, call)
        execution_context = replace(
            (await self._store.run_async(self._context_for_call, turn_id, call)),
            invocation_guard=lambda invoked: self._store.run_async(self._guard_input, turn_id, invoked),
        )
        tool = self._tools.get(call.tool_name)
        active_record: ToolExecutionRecord | None = None

        async def persist(record: ToolExecutionRecord) -> None:
            nonlocal active_record
            if tool is None:
                raise RuntimeError("unknown tools must not create execution records")
            await self._persist_execution_record(
                turn_id=turn_id,
                tool=tool,
                record=record,
            )
            if record.status is ExecutionStatus.STARTED:
                active_record = record

        async def publish_progress(progress: ToolProgress) -> None:
            if active_record is None:
                raise RuntimeError("tool progress arrived before the durable claim")
            await self._publish_tool_progress(
                turn_id=turn_id,
                record=active_record,
                progress=progress,
            )

        def capture_preflight(
            record: ToolExecutionRecord,
            resolved: ResolvedToolUse,
        ) -> None:
            self._instruction_targets[(turn_id, call.tool_call_id)] = resolved.targets
            self._resolved_scopes[record.operation_id] = _durable_tool_scope(
                resolved.effects,
                resolved.targets,
            )

        async def accept_cancelled(execution: ToolExecution) -> None:
            await self._accept_execution(turn_id, call, _associate_execution(execution, turn_id), execution_context)

        execution = await self._execute_in_turn(
            turn_id,
            call,
            context=execution_context,
            record=(await self._store.run_async(self._pending_record, turn_id, call)),
            record_sink=persist,
            preflight_sink=capture_preflight,
            progress_sink=publish_progress,
            cancellation_sink=accept_cancelled,
        )
        await self._accept_execution(turn_id, call, execution, execution_context)
        if (
            task is not None
            and task.cancelling() > cancellations
            and (await self._store.run_async(self._store.read_turn, turn_id)).status == "running"
        ):
            raise asyncio.CancelledError
        return execution.result

    async def _record_call(self, turn_id: str, call: ToolCall) -> None:
        await self._commit(
            lambda: self._store.record_tool_call(
                turn_id=turn_id,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                arguments=_json_mapping(call.arguments),
                argument_error=call.argument_error,
                origin={
                    "request_id": call.origin.request_id,
                    "toolset_revision": call.origin.toolset_revision,
                    "exposed_tool_names": list(call.origin.exposed_tool_names),
                },
            )
        )

    async def _accept_execution(
        self,
        turn_id: str,
        call: ToolCall,
        execution: ToolExecution,
        execution_context: ToolExecutionContext,
    ) -> None:
        tool = self._tools.get(call.tool_name)
        if execution.result.error_code == "approval_required":
            if await self._store.run_async(self.input_superseded, turn_id, call):
                await self._discard_superseded(turn_id, call)
                return
            if tool is None:
                raise RuntimeError("unknown tool cannot request approval")
            record = ToolExecutionRecord.prepare(call, tool)
            effects, resources = _durable_tool_scope(
                execution.trace.effects,
                execution.trace.targets,
            )
            interaction = await self._commit(
                lambda: self._store.request_tool_approval(
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
                        arguments=call.arguments,
                    ),
                )
            )
            raise ToolApprovalRequiredError(interaction.request_id)
        try:
            await self._commit_execution_result(
                turn_id=turn_id,
                call=call,
                execution=execution,
            )
        except Exception as error:
            operation_id = None if execution.record is None else execution.record.operation_id
            message = "Tool result could not be committed; inspect history before recovery."
            result = replace(
                execution.result,
                content=(ToolContentBlock(type="text", data={"text": message}),),
                structured_content=None,
                is_error=True,
                error_code="tool_result_commit_failed",
                error_message=message,
                retryable=False,
                metadata={
                    **execution.result.metadata,
                    "diagnostic": exception_diagnostic(
                        error, stage="tool_result_commit_failed", tool_call_id=call.tool_call_id,
                        operation_id=operation_id, turn_id=turn_id,
                    ),
                    "result_commit_failed": True,
                },
            )
            raise ToolResultCommitError(result) from None

    async def execute_batch(self, *, turn_id: str, calls: Sequence[ToolCall]) -> tuple[ToolResult, ...]:
        """Preflight before I/O, then run bounded, resource-safe local tools.

        Remote outcomes and state-changing tools retain sequential execution:
        their pause/activation protocols require a quiescent Turn. Inspection
        batches beyond the remaining inspection budget retain sequential accounting.
        All calls are durable before approval can interrupt the batch.
        """
        calls = tuple(calls)
        if not calls:
            return ()
        if len({call.tool_call_id for call in calls}) != len(calls):
            raise ValueError("batch tool call identities must be unique")
        for call in calls:
            await self._record_call(turn_id, call)
        def parallel_local(call: ToolCall) -> bool:
            tool = self._tools.get(call.tool_name)
            return bool(
                tool is not None and tool.concurrency_safe
                and tool.cancellation_mode in {CancellationMode.COOPERATIVE, CancellationMode.MANAGED_PROCESS}
            )

        results: list[ToolResult] = []
        task = asyncio.current_task()
        cancellations = 0 if task is None else task.cancelling()
        offset = 0
        while offset < len(calls):
            end = offset + 1
            if parallel_local(calls[offset]):
                while end < len(calls) and parallel_local(calls[end]) and calls[end].origin == calls[offset].origin:
                    end += 1
            # State-changing and remote calls are barriers. Resolve the next
            # segment's context only after this one drains.
            results.extend(await self._execute_batch_segment(turn_id=turn_id, calls=calls[offset:end]))
            if (await self._store.run_async(self._store.read_turn, turn_id)).status != "running":
                break
            if task is not None and task.cancelling() > cancellations:
                raise asyncio.CancelledError
            offset = end
        return tuple(results)

    async def _execute_batch_segment(self, *, turn_id: str, calls: Sequence[ToolCall]) -> tuple[ToolResult, ...]:
        if self._checkpoints is not None:
            await self._ensure_turn_checkpoint(turn_id, calls)
        if await self._store.run_async(lambda: any(self.input_superseded(turn_id, call) for call in calls)):
            return tuple([await self.execute(turn_id=turn_id, call=call) for call in calls])
        contexts = await self._store.run_async(lambda: tuple(self._context_for_call(turn_id, call) for call in calls))
        tools = tuple(self._tools.get(call.tool_name) for call in calls)
        serial = (
            any(context != contexts[0] for context in contexts[1:])
            or any(call.origin != calls[0].origin for call in calls[1:])
            or any(
                tool is not None
                and (
                    not tool.concurrency_safe
                    or tool.cancellation_mode not in {CancellationMode.COOPERATIVE, CancellationMode.MANAGED_PROCESS}
                )
                for tool in tools
            )
        )
        if serial:
            results = []
            for call in calls:
                results.append(await self.execute(turn_id=turn_id, call=call))
                if (await self._store.run_async(self._store.read_turn, turn_id)).status != "running":
                    break
            return tuple(results)

        by_id = {call.tool_call_id: call for call in calls}
        active: dict[str, ToolExecutionRecord] = {}

        async def persist(record: ToolExecutionRecord) -> None:
            await self._persist_execution_record(turn_id=turn_id, tool=self._tools[record.tool_name], record=record)
            if record.status is ExecutionStatus.STARTED:
                active[record.tool_call_id] = record

        def preflight(record: ToolExecutionRecord, resolved: ResolvedToolUse) -> None:
            self._instruction_targets[(turn_id, record.tool_call_id)] = resolved.targets
            self._resolved_scopes[record.operation_id] = _durable_tool_scope(resolved.effects, resolved.targets)

        def progress_for(call: ToolCall) -> Callable[[ToolProgress], Awaitable[None]]:
            async def publish(progress: ToolProgress) -> None:
                await self._publish_tool_progress(turn_id=turn_id, record=active[call.tool_call_id], progress=progress)

            return publish

        async def accept(execution: ToolExecution) -> None:
            execution = _associate_execution(execution, turn_id)
            await self._accept_execution(turn_id, by_id[execution.result.tool_call_id], execution, contexts[0])

        snapshot = await self._store.run_async(self._step_snapshot, turn_id, calls[0])
        token = _ACTIVE_TOOL_TURN_ID.set(turn_id)
        binding_token = _ACTIVE_TOOL_BINDING.set(None if snapshot is None else snapshot["binding_manifest"])
        try:
            executions = await self._executor.execute_batch(
                calls,
                context=replace(
                    contexts[0],
                    invocation_guard=lambda invoked: self._store.run_async(self._guard_input, turn_id, invoked),
                ),
                records=(
                    await self._store.run_async(
                        lambda: {
                            call.tool_call_id: record
                            for call in calls
                            if (record := self._pending_record(turn_id, call)) is not None
                        }
                    )
                ),
                record_sink=persist,
                preflight_sink=preflight,
                progress_sinks={call.tool_call_id: progress_for(call) for call in calls},
                execution_sink=accept,
            )
            return tuple(_associate_execution(execution, turn_id).result for execution in executions)
        finally:
            _ACTIVE_TOOL_BINDING.reset(binding_token)
            _ACTIVE_TOOL_TURN_ID.reset(token)

    async def _ensure_turn_checkpoint(self, turn_id: str, calls: Sequence[ToolCall]) -> None:
        """Snapshot the workspace once per Turn before its first mutation attempt.

        The single rule covers every write path: any tool call resolving to
        WRITE_WORKSPACE (apply_patch, workspace-write commands, Python outputs)
        triggers the checkpoint. Checkpoints are a safety net, never a gate:
        failures disable or skip the snapshot and the tool call proceeds.
        """
        checkpoints = self._checkpoints
        if checkpoints is None:
            return
        if await self._store.run_async(
            lambda: any(
                item.kind == "workspace_checkpoint"
                and item.status == "completed"
                and (item.payload.get("trigger") == "pre_write")
                for item in self._store.list_items(turn_id)
            )
        ):
            return
        for call in calls:
            if call.argument_error is not None or not self._call_writes_workspace(call):
                continue
            try:
                snapshot = await checkpoints.snapshot(trigger="pre_write", turn_id=turn_id)
            except CheckpointUnavailableError as error:
                logger.warning("workspace checkpoints disabled: %s", error)
                self._checkpoints = None
                return
            except (CheckpointError, OSError) as error:
                logger.warning("workspace checkpoint skipped: %s", error)
                return
            await self._record_turn_checkpoint(turn_id, snapshot)
            return

    async def _record_turn_checkpoint(
        self, turn_id: str, snapshot: CheckpointSnapshot,
    ) -> None:
        def persist() -> None:
            self._store.record_workspace_checkpoint(
                turn_id=turn_id,
                checkpoint_id=snapshot.checkpoint_id,
                trigger=snapshot.trigger,
                tree_sha256=snapshot.tree_sha256,
                changed_file_count=snapshot.changed_file_count,
                wall_time=time.time(),
            )

        await self._commit(persist)

    def _call_writes_workspace(self, call: ToolCall) -> bool:
        """Mirror the executor's effect resolution: static ∪ dynamic effects."""
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return False
        try:
            arguments = tool.validate_input(call.arguments)
            resolved = tool.resolve_use(arguments)
        except Exception:
            return False
        return ToolEffect.WRITE_WORKSPACE in tool.static_effects | resolved.effects

    async def _execute_in_turn(
        self,
        turn_id: str,
        call: ToolCall,
        **kwargs: Any,
    ) -> ToolExecution:
        snapshot = await self._store.run_async(self._step_snapshot, turn_id, call)
        token = _ACTIVE_TOOL_TURN_ID.set(turn_id)
        binding_token = _ACTIVE_TOOL_BINDING.set(None if snapshot is None else snapshot["binding_manifest"])
        try:
            execution = await self._executor.execute(call, **kwargs)
            return _associate_execution(execution, turn_id)
        finally:
            _ACTIVE_TOOL_BINDING.reset(binding_token)
            _ACTIVE_TOOL_TURN_ID.reset(token)

    async def _publish_tool_progress(
        self,
        *,
        turn_id: str,
        record: ToolExecutionRecord,
        progress: ToolProgress,
    ) -> None:
        if self._event_dispatcher is None:
            return
        claim = self._claims.get(record.operation_id)
        if claim is None:
            raise RuntimeError("tool progress has no active fenced claim")
        is_command = record.tool_name == "run_command"
        delta_kind = (
            ItemDeltaKind.COMMAND_STDOUT
            if is_command and progress.kind is ToolProgressKind.STDOUT
            else ItemDeltaKind.COMMAND_STDERR
            if is_command and progress.kind is ToolProgressKind.STDERR
            else ItemDeltaKind.TOOL_PROGRESS
        )
        await self._event_dispatcher.emit(
            item_delta(
                turn_id=turn_id,
                item_id=derive_operation_public_item_id(
                    turn_id=turn_id,
                    operation_id=record.operation_id,
                    attempt_generation=claim[0],
                ),
                item_kind=(TurnItemKind.COMMAND if is_command else TurnItemKind.TOOL),
                delta_kind=delta_kind,
                delta=progress.content,
            )
        )

    async def _commit_execution_result(
        self,
        *,
        turn_id: str,
        call: ToolCall,
        execution: ToolExecution,
    ) -> None:
        record = execution.record
        operation_id = None if record is None else record.operation_id
        if record is None:
            await self._commit(
                lambda: self._store.record_tool_result(
                    turn_id=turn_id,
                    operation_id=None,
                    result=_tool_result_payload(execution.result),
                )
            )
            return
        claim = self._claims.get(record.operation_id)
        if claim is None:
            operation = await self._store.run_async(self._store.read_tool_operation, record.operation_id)
            if (operation.status == "failed" and operation.error_code == record.error_code
                    and record.error_code in {"resource_busy", "input_superseded"}):
                await self._commit(
                    lambda: self._store.record_tool_result(
                        turn_id=turn_id,
                        operation_id=operation_id,
                        result=_tool_result_payload(execution.result),
                    )
                )
                return
            raise RuntimeError("tool outcome has no active durable claim")
        status = {
            ExecutionStatus.COMPLETED: "succeeded",
            ExecutionStatus.FAILED: "failed",
            ExecutionStatus.OUTCOME_UNKNOWN: "unknown",
        }.get(record.status)
        if status is None:
            raise RuntimeError(f"unsupported ToolExecutionRecord status: {record.status}")
        accepted = await self._commit(
            lambda: self._store.commit_tool_operation_result(
                turn_id=turn_id,
                operation_id=record.operation_id,
                claim_generation=claim[0],
                fencing_token=claim[1],
                status=status,
                attempt_count=record.attempt_count,
                error_code=record.error_code,
                requires_reconciliation=record.requires_reconciliation,
                duration_ms=execution.trace.duration_ms,
                result=_tool_result_payload(execution.result),
            )
        )
        if not accepted:
            raise RuntimeError("stale worker cannot commit tool operation outcome")
        self._claims.pop(record.operation_id, None)
        tool = self._tools.get(call.tool_name)
        if tool is not None and tool.after_commit is not None and not execution.result.is_error:
            await tool.after_commit(call.arguments, execution.result.structured_content, record.operation_id)
        operation = await self._store.run_async(self._store.read_tool_operation, record.operation_id)
        # Historical gates retain their frozen verifier semantics. New Turns
        # use explicit acceptance commands and do not mint inferred certificates.
        policy = await self._store.run_async(
            lambda: self._store.read_turn(turn_id).binding_manifest.get("completion_policy", {})
        )
        if isinstance(policy, Mapping) and policy.get("revision") == "completion-v2":
            return
        evidence = classify_tool_verification(
            tool_name=call.tool_name,
            arguments=call.arguments,
            structured_content=execution.result.structured_content,
            resources=operation.resources,
        )
        if evidence is not None:
            await self._commit(
                lambda: self._store.record_verification(
                    turn_id=turn_id,
                    operation_id=record.operation_id,
                    kind=evidence.kind,
                    verifier=evidence.verifier,
                    verified_resources=evidence.verified_resources,
                )
            )

    async def resume_approval(
        self, *, turn_id: str, decision: str,
        on_session_grant: Callable[[CommandGrant], None] | None = None,
    ) -> ToolResult:
        pending = await self._store.run_async(
            lambda: [
                interaction
                for interaction in self._store.list_interactions(turn_id)
                if interaction.kind == "tool_approval" and interaction.status == "pending"
            ]
        )
        if len(pending) != 1:
            raise RuntimeError("resume requires exactly one pending tool approval")
        interaction = pending[0]
        if interaction.operation_id is None:
            raise RuntimeError("tool approval interaction has no operation")
        operation = await self._store.run_async(self._store.read_tool_operation, interaction.operation_id)
        call = await self._store.run_async(
            self._committed_tool_call, turn_id=turn_id, tool_call_id=operation.tool_call_id
        )
        if await self._store.run_async(self.input_superseded, turn_id, call):
            await self._commit(lambda: self._store.resolve_tool_approval(turn_id=turn_id, decision="deny"))
            return await self._discard_superseded(turn_id, call)
        tool = self._tools.get(call.tool_name)
        if tool is None:
            raise RuntimeError("approved tool is no longer installed")
        execution_context = await self._store.run_async(self._context_for_turn, turn_id)
        recheck = decision == "recheck"
        approval_ids = {call.tool_call_id}
        if interaction.request.get("approval_scope") == "network":
            approval_ids.add(_network_approval_id(call.tool_call_id))
        if recheck:
            # A mode change is not a one-time grant. Re-evaluate the committed
            # call, then let the normal executor enforce the current policy.
            execution_context = execution_context.refreshed()
            arguments = tool.validate_input(call.arguments)
            outcome = can_use_tool(tool, arguments, tool.resolve_use(arguments), execution_context)
            if outcome.decision is UseToolDecision.ASK:
                preflight = await self._execute_in_turn(turn_id, call, context=replace(
                    execution_context,
                    approved_tool_call_ids=execution_context.approved_tool_call_ids - {call.tool_call_id},
                    require_confirmation_for=execution_context.require_confirmation_for | {call.tool_name},
                ))
                current_request = _approval_request(
                    preflight, record=ToolExecutionRecord.prepare(call, tool), tool=tool,
                    context=execution_context, arguments=call.arguments,
                )
                if _approval_scope(current_request) != _approval_scope(interaction.request):
                    interaction = await self._commit(lambda: self._store.refresh_tool_approval(
                        turn_id=turn_id, request=current_request,
                    ))
                raise ToolApprovalInvalidatedError(interaction.request_id)
            decision = "approve" if outcome.decision is UseToolDecision.ALLOW else "deny"
        grant = None
        if on_session_grant is not None:
            arguments = tool.validate_input(call.arguments)
            grant = command_grant(tool, arguments, tool.resolve_use(arguments), execution_context)
            if (decision != "approve" or grant is None
                    or interaction.request.get("approval_scope", "tool") != "tool"
                    or interaction.request.get("session_grant") != grant.to_json()):
                raise ValueError("this approval does not offer a current Session command grant")
        if decision == "approve" and not recheck:
            preflight_context = replace(
                execution_context,
                approved_tool_call_ids=(execution_context.approved_tool_call_ids - {call.tool_call_id}),
                denied_tool_call_ids=(execution_context.denied_tool_call_ids - {call.tool_call_id}),
                require_confirmation_for=(execution_context.require_confirmation_for | {call.tool_name}),
            )
            preflight = await self._execute_in_turn(
                turn_id,
                call,
                context=preflight_context,
            )
            fresh_record = ToolExecutionRecord.prepare(call, tool)
            current_request = _approval_request(
                preflight,
                record=fresh_record,
                tool=tool,
                context=execution_context,
                arguments=call.arguments,
            )
            # The forced dry run checks targets and policy under the tool gate;
            # preserve which scope this pending user decision actually covers.
            current_request["approval_scope"] = interaction.request.get("approval_scope", "tool")
            if preflight.result.error_code != "approval_required" or _approval_scope(
                interaction.request
            ) != _approval_scope(current_request):
                invalidated = await self._commit(
                    lambda: self._store.invalidate_tool_approval(
                        turn_id=turn_id,
                        reason="approval scope changed before execution",
                    )
                )
                raise ToolApprovalInvalidatedError(invalidated.request_id)
        interaction = await self._commit(
            lambda: self._store.resolve_tool_approval(
                turn_id=turn_id,
                decision=decision,
                policy_recheck=recheck,
            )
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
        approved = decision == "approve" and not recheck
        context = replace(
            execution_context,
            approved_tool_call_ids=(
                execution_context.approved_tool_call_ids | (approval_ids if approved else set())
            ),
            denied_tool_call_ids=(
                execution_context.denied_tool_call_ids | (approval_ids if decision == "deny" else set())
            ),
        )
        result = await self._execute_approved_record(
            turn_id=turn_id, call=call, tool=tool, context=context, record=record,
        )
        if on_session_grant is not None and grant is not None and not result.is_error:
            await self._store.run_async(on_session_grant, grant)
        return result

    async def recover_resolved_approval(self, *, turn_id: str) -> ToolResult:
        """Execute one approved ready operation after a pre-execute crash."""

        resolved = await self._store.run_async(
            lambda: [
                interaction
                for interaction in self._store.list_interactions(turn_id)
                if interaction.kind == "tool_approval"
                and interaction.status == "resolved"
                and interaction.response.get("decision") == "approve"
                and interaction.operation_id is not None
                and self._store.read_tool_operation(interaction.operation_id).status == "ready"
                and self._store.read_tool_operation(interaction.operation_id).approval_request_id
                == interaction.request_id
            ]
        )
        if len(resolved) != 1 or resolved[0].operation_id is None:
            raise RuntimeError("recovery requires one resolved approved operation")
        interaction = resolved[0]
        operation_id = interaction.operation_id
        if operation_id is None:
            raise RuntimeError("resolved approval lost its operation identity")
        operation = await self._store.run_async(self._store.read_tool_operation, operation_id)
        if operation.status != "ready":
            raise RuntimeError("approved operation is not ready for crash recovery")
        call = await self._store.run_async(
            self._committed_tool_call, turn_id=turn_id, tool_call_id=operation.tool_call_id
        )
        if await self._store.run_async(self.input_superseded, turn_id, call):
            return await self._discard_superseded(turn_id, call)
        tool = self._tools.get(call.tool_name)
        if tool is None:
            raise RuntimeError("approved tool is no longer installed")
        execution_context = await self._store.run_async(self._context_for_turn, turn_id)
        if interaction.response.get("policy_recheck") is True:
            # A crashed automatic policy decision is never restored as a user grant.
            arguments = tool.validate_input(call.arguments)
            execution_context = execution_context.refreshed()
            outcome = can_use_tool(tool, arguments, tool.resolve_use(arguments), execution_context)
            if outcome.decision is UseToolDecision.ASK:
                preflight = await self._execute_in_turn(turn_id, call, context=replace(
                    execution_context,
                    approved_tool_call_ids=execution_context.approved_tool_call_ids - {call.tool_call_id},
                    require_confirmation_for=execution_context.require_confirmation_for | {call.tool_name},
                ))
                current_request = _approval_request(
                    preflight, record=ToolExecutionRecord.prepare(call, tool), tool=tool,
                    context=execution_context, arguments=call.arguments,
                )
                refreshed = await self._commit(lambda: self._store.refresh_tool_approval(
                    turn_id=turn_id, request=current_request,
                ))
                raise ToolApprovalInvalidatedError(refreshed.request_id)
            return await self._execute_approved_record(
                turn_id=turn_id, call=call, tool=tool, context=execution_context,
                record=ToolExecutionRecord(
                    tool_call_id=operation.tool_call_id, tool_name=operation.tool_name,
                    operation_id=operation.operation_id, arguments_digest=operation.arguments_digest,
                    idempotent=operation.idempotent, status=ExecutionStatus.PREPARED,
                    attempt_count=operation.attempt_count,
                ),
            )
        preflight_context = replace(
            execution_context,
            approved_tool_call_ids=(execution_context.approved_tool_call_ids - {call.tool_call_id}),
            denied_tool_call_ids=(execution_context.denied_tool_call_ids - {call.tool_call_id}),
            require_confirmation_for=(execution_context.require_confirmation_for | {call.tool_name}),
        )
        preflight = await self._execute_in_turn(turn_id, call, context=preflight_context)
        fresh_record = ToolExecutionRecord.prepare(call, tool)
        current_request = _approval_request(
            preflight,
            record=fresh_record,
            tool=tool,
            context=execution_context,
            arguments=call.arguments,
        )
        current_request["approval_scope"] = interaction.request.get("approval_scope", "tool")
        if preflight.result.error_code != "approval_required" or _approval_scope(
            interaction.request
        ) != _approval_scope(current_request):
            invalidated = await self._commit(
                lambda: self._store.invalidate_resolved_tool_approval(
                    turn_id=turn_id,
                    reason="approval scope changed before recovered execution",
                )
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
            approved_tool_call_ids=(execution_context.approved_tool_call_ids | {call.tool_call_id}
                                    | ({_network_approval_id(call.tool_call_id)}
                                       if interaction.request.get("approval_scope") == "network" else set())),
            denied_tool_call_ids=(execution_context.denied_tool_call_ids - {call.tool_call_id}),
        )
        return await self._execute_approved_record(
            turn_id=turn_id, call=call, tool=tool, context=context, record=record,
        )

    async def _execute_approved_record(
        self,
        *,
        turn_id: str,
        call: ToolCall,
        tool: Tool,
        context: ToolExecutionContext,
        record: ToolExecutionRecord,
    ) -> ToolResult:
        """Execute and publish after the caller has validated its approval scope."""
        task = asyncio.current_task()
        cancellations = 0 if task is None else task.cancelling()
        active_record: ToolExecutionRecord | None = None
        context = replace(
            context, invocation_guard=lambda invoked: self._store.run_async(self._guard_input, turn_id, invoked)
        )

        def capture_preflight(updated: ToolExecutionRecord, resolved: ResolvedToolUse) -> None:
            self._instruction_targets[(turn_id, call.tool_call_id)] = resolved.targets
            self._resolved_scopes[updated.operation_id] = _durable_tool_scope(resolved.effects, resolved.targets)

        async def persist(updated: ToolExecutionRecord) -> None:
            nonlocal active_record
            await self._persist_execution_record(
                turn_id=turn_id,
                tool=tool,
                record=updated,
            )
            if updated.status is ExecutionStatus.STARTED:
                active_record = updated

        async def publish_progress(progress: ToolProgress) -> None:
            if active_record is None:
                raise RuntimeError("tool progress arrived before the durable claim")
            await self._publish_tool_progress(
                turn_id=turn_id,
                record=active_record,
                progress=progress,
            )

        async def accept(execution: ToolExecution) -> None:
            await self._accept_execution(turn_id, call, _associate_execution(execution, turn_id), context)

        execution = await self._execute_in_turn(
            turn_id,
            call,
            context=context,
            record=record,
            record_sink=persist,
            preflight_sink=capture_preflight,
            progress_sink=publish_progress,
            cancellation_sink=accept,
        )
        await accept(execution)
        if (
            task is not None
            and task.cancelling() > cancellations
            and (await self._store.run_async(self._store.read_turn, turn_id)).status == "running"
        ):
            raise asyncio.CancelledError
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

    def _context_for_turn(self, turn_id: str, *, base: ToolExecutionContext | None = None) -> ToolExecutionContext:
        context = self._execution_context if base is None else base
        active_skill_ids = set(context.active_skill_ids)
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
            context,
            active_skill_ids=frozenset(active_skill_ids),
        )

    async def _persist_execution_record(
        self,
        *,
        turn_id: str,
        tool: Tool,
        record: ToolExecutionRecord,
    ) -> None:
        existing = (
            await self._store.run_async(
                lambda: {item.operation_id: item for item in self._store.list_tool_operations(turn_id)}
            )
        ).get(record.operation_id)
        if record.status is ExecutionStatus.PREPARED:
            if existing is not None:
                if existing.status == "prepared":
                    await self._write_operation_state(turn_id=turn_id, tool=tool, record=record, status="ready")
                    return
                if existing.status != "ready":
                    raise RuntimeError("prepared execution record requires a ready operation")
                return
            await self._write_operation_state(
                turn_id=turn_id,
                tool=tool,
                record=record,
                status="prepared",
            )
            await self._write_operation_state(
                turn_id=turn_id,
                tool=tool,
                record=record,
                status="ready",
            )
            return
        if record.status is ExecutionStatus.STARTED:
            call = await self._store.run_async(
                self._committed_tool_call, turn_id=turn_id, tool_call_id=record.tool_call_id
            )
            if await self._store.run_async(self.input_superseded, turn_id, call):
                await self._commit(lambda: self._store.reject_ready_tool_operation(
                    operation_id=record.operation_id, error_code="input_superseded",
                ))
                raise ExecutionStartRejectedError("input_superseded", "New user input superseded this action.")
            lease_seconds = max(
                self._lease_seconds,
                tool.timeout_seconds + _TOOL_LEASE_GRACE_SECONDS,
            )
            def claim_operation() -> Any:
                claim = self._store.claim_tool_operation(
                    operation_id=record.operation_id, worker_id=self._worker_id,
                    lease_seconds=lease_seconds,
                )
                if claim.fencing_token is None:
                    raise RuntimeError("claimed tool operation has no fencing token")
                if claim.attempt_count != record.attempt_count:
                    raise RuntimeError("tool runner attempt count differs from durable claim")
                # The claim is durable before event delivery can suspend. Keep
                # its fence available even if that delivery is cancelled.
                self._claims[record.operation_id] = (claim.claim_generation, claim.fencing_token)
                return claim

            try:
                await self._commit(claim_operation)
            except ResourceClaimConflictError as conflict:
                await self._commit(
                    lambda: self._store.reject_ready_tool_operation(
                        operation_id=record.operation_id,
                        error_code="resource_busy",
                    )
                )
                raise ExecutionStartRejectedError(
                    "resource_busy",
                    str(conflict),
                ) from None
            return
        if record.status in {
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.OUTCOME_UNKNOWN,
        }:
            return
        raise RuntimeError(f"unsupported ToolExecutionRecord status: {record.status}")

    async def _write_operation_state(
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
        await self._commit(
            lambda: self._store.record_tool_execution_state(
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
            argument_error=payload.get("argument_error"),
            origin=ToolCallOrigin(
                request_id=request_id,
                toolset_revision=toolset_revision,
                exposed_tool_names=tuple(exposed),
            ),
        )


def _associate_execution(execution: ToolExecution, turn_id: str) -> ToolExecution:
    metadata = dict(execution.result.metadata)
    metadata["turn_id"] = turn_id
    diagnostic = metadata.get("diagnostic")
    if isinstance(diagnostic, Mapping):
        metadata["diagnostic"] = {**diagnostic, "turn_id": turn_id}
    return replace(execution, result=replace(execution.result, metadata=metadata))


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
    arguments: Mapping[str, JsonValue],
) -> dict[str, Any]:
    validated = tool.validate_input(arguments)
    grant = command_grant(tool, validated, tool.resolve_use(validated), context)
    request: dict[str, Any] = {
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
        "policy_revision": _policy_revision(context, tool, execution),
    }
    if grant is not None and request["approval_scope"] == "tool":
        request["session_grant"] = grant.to_json()
        request["session_grant_label"] = grant.label
    return request


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


def _policy_revision(context: ToolExecutionContext, tool: Tool, execution: ToolExecution) -> str:
    effects = frozenset(execution.trace.effects)
    targets = execution.trace.targets
    payload = {
        "permissions_revision": (None if context.effective_permissions is None
                                 else context.effective_permissions.revision),
        "approval_mode": context.approval_mode,
        "workspace_root": (None if context.workspace_root is None else str(context.workspace_root)),
        "cwd": None if context.cwd is None else str(context.cwd),
        "allow_write_tools": context.allow_write_tools if ToolEffect.WRITE_WORKSPACE in effects else None,
        "allow_execute_tools": context.allow_execute_tools if ToolEffect.EXECUTE_PROCESS in effects else None,
        "active_skill_ids": sorted(
            target.value
            for target in targets
            if target.kind == "active_skill" and target.value in context.active_skill_ids
        ),
        "deny_effects": sorted(effect.value for effect in context.deny_effects & effects),
        "require_confirmation": tool.definition.name in context.require_confirmation_for,
        "denied": tool.definition.name in context.denied_tool_names,
        "auto_approve_sandboxed": context.auto_approve_sandboxed
        if any(target.kind == "execution_mode" and target.value == "restricted_sandbox" for target in targets)
        else None,
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

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from rag.agent.tools.permissions import (
    CanUseToolResult,
    ToolExecutionContext,
    ToolGuardError,
    UseToolDecision,
    can_use_tool,
    enforce_hard_guards,
)
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolContentBlock,
    ToolEffect,
    ToolResult,
    ToolTarget,
    ToolValidationError,
    json_schema_output,
)

_NETWORK_APPROVAL_SUFFIX = "::network"


class ExecutionBoundary(StrEnum):
    DIRECT = "direct"
    MANAGED_PROCESS = "managed_process"
    REMOTE = "remote"


class ExecutionStatus(StrEnum):
    PREPARED = "prepared"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"

    # Source compatibility for checkpoints and callers created before the
    # durable prepared/started boundary was made explicit.
    RUNNING = "started"
    UNKNOWN = "outcome_unknown"


@dataclass(frozen=True, slots=True)
class ToolExecutionRecord:
    tool_call_id: str
    tool_name: str
    operation_id: str
    arguments_digest: str
    idempotent: bool
    status: ExecutionStatus
    attempt_count: int = 0
    error_code: str | None = None
    requires_reconciliation: bool = False

    @classmethod
    def prepare(cls, call: ToolCall, tool: Tool) -> ToolExecutionRecord:
        return cls(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            operation_id=f"op_{uuid4().hex}",
            arguments_digest=_arguments_digest(call.arguments),
            idempotent=tool.idempotent,
            status=ExecutionStatus.PREPARED,
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionTrace:
    tool_call_id: str
    tool_name: str
    outcome: str
    error_code: str | None
    decision: UseToolDecision | None
    boundary: ExecutionBoundary | None
    effects: tuple[ToolEffect, ...]
    targets: tuple[ToolTarget, ...]
    duration_ms: float


@dataclass(frozen=True, slots=True)
class ToolExecution:
    result: ToolResult
    record: ToolExecutionRecord | None
    trace: ToolExecutionTrace


type HardGuard = Callable[
    [Tool, Mapping[str, JsonValue], ResolvedToolUse, ToolExecutionContext],
    None,
]
type BoundaryResolver = Callable[
    [Tool, ResolvedToolUse, ToolExecutionContext],
    ExecutionBoundary,
]
type PermissionDecider = Callable[
    [Tool, Mapping[str, JsonValue], ResolvedToolUse, ToolExecutionContext],
    CanUseToolResult,
]
type ApprovalResolver = Callable[
    [
        ToolCall,
        Tool,
        Mapping[str, JsonValue],
        ResolvedToolUse,
        ToolExecutionContext,
        CanUseToolResult,
    ],
    bool | None | Awaitable[bool | None],
]
type OutputValidator = Callable[
    [Tool, NormalizedToolOutput],
    NormalizedToolOutput,
]
type Externalizer = Callable[
    [Tool, NormalizedToolOutput],
    tuple[NormalizedToolOutput, bool],
]
type TraceSink = Callable[[ToolExecutionTrace], None | Awaitable[None]]
type RecordSink = Callable[
    [ToolExecutionRecord],
    None | Awaitable[None],
]


@dataclass(frozen=True, slots=True)
class _PreparedExecution:
    call: ToolCall
    tool: Tool
    arguments: Mapping[str, JsonValue]
    resolved: ResolvedToolUse
    boundary: ExecutionBoundary
    decision: CanUseToolResult
    record: ToolExecutionRecord
    record_sink: RecordSink | None
    started_at: float


class _CancelledTimeoutError(Exception):
    pass


class _UnknownTimeoutError(Exception):
    pass


class _UnknownCancellationError(Exception):
    pass


class ToolExecutor:
    """The sole validation-to-ToolResult execution choke point."""

    def __init__(
        self,
        tools: Mapping[str, Tool],
        *,
        hard_guard: HardGuard = enforce_hard_guards,
        boundary_resolver: BoundaryResolver | None = None,
        permission_decider: PermissionDecider = can_use_tool,
        approval_resolver: ApprovalResolver | None = None,
        output_validator: OutputValidator | None = None,
        externalizer: Externalizer | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        copied: dict[str, Tool] = {}
        for name, tool in tools.items():
            if not isinstance(tool, Tool):
                raise TypeError("tools must contain Tool values")
            if name != tool.definition.name:
                raise ValueError("tool mapping key must match Tool definition name")
            copied[name] = tool
        self._tools = tools if isinstance(tools, MappingProxyType) else MappingProxyType(copied)
        self._hard_guard = hard_guard
        self._boundary_resolver = boundary_resolver or resolve_execution_boundary
        self._permission_decider = permission_decider
        self._approval_resolver = approval_resolver or _resolve_external_approval
        self._output_validator = output_validator or _validate_normalized_output
        self._externalizer = externalizer or _bound_model_output
        self._trace_sink = trace_sink
        self._traces: list[ToolExecutionTrace] = []

    @property
    def traces(self) -> tuple[ToolExecutionTrace, ...]:
        return tuple(self._traces)

    async def execute(
        self,
        call: ToolCall,
        *,
        context: ToolExecutionContext,
        record: ToolExecutionRecord | None = None,
        record_sink: RecordSink | None = None,
    ) -> ToolExecution:
        trace_count = len(self._traces)
        started_at = time.monotonic()
        try:
            prepared = await self._prepare(
                call,
                context=context,
                record=record,
                record_sink=record_sink,
                started_at=started_at,
            )
            if isinstance(prepared, ToolExecution):
                return prepared
            return await self._invoke(prepared)
        except asyncio.CancelledError:
            if len(self._traces) == trace_count:
                await self._finish(
                    call=call,
                    result=_error_result(
                        call,
                        code="cancelled",
                        message="tool execution was cancelled",
                        retryable=True,
                    ),
                    record=record,
                    started_at=started_at,
                )
            raise

    async def execute_batch(
        self,
        calls: Sequence[ToolCall],
        *,
        context: ToolExecutionContext,
        records: Mapping[str, ToolExecutionRecord] | None = None,
        record_sink: RecordSink | None = None,
    ) -> tuple[ToolExecution, ...]:
        prior_records = records or {}
        completed: dict[int, ToolExecution] = {}
        prepared: list[tuple[int, _PreparedExecution]] = []
        for index, call in enumerate(calls):
            item = await self._prepare(
                call,
                context=context,
                record=prior_records.get(call.tool_call_id),
                record_sink=record_sink,
                started_at=time.monotonic(),
            )
            if isinstance(item, ToolExecution):
                completed[index] = item
            else:
                prepared.append((index, item))

        if _can_run_in_parallel(tuple(item for _, item in prepared)):
            limit = context.max_parallel_calls
            for offset in range(0, len(prepared), limit):
                batch = prepared[offset : offset + limit]
                executions = await asyncio.gather(
                    *(self._invoke(item) for _, item in batch)
                )
                for (index, _), execution in zip(
                    batch,
                    executions,
                    strict=True,
                ):
                    completed[index] = execution
        else:
            for index, item in prepared:
                completed[index] = await self._invoke(item)
        return tuple(completed[index] for index in range(len(calls)))

    async def _prepare(
        self,
        call: ToolCall,
        *,
        context: ToolExecutionContext,
        record: ToolExecutionRecord | None,
        record_sink: RecordSink | None,
        started_at: float,
    ) -> _PreparedExecution | ToolExecution:
        if call.tool_call_id.endswith(_NETWORK_APPROVAL_SUFFIX):
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="invalid_tool_call_id",
                    message=(
                        "tool call ID uses a reserved approval scope suffix"
                    ),
                    retryable=False,
                ),
                record=None,
                started_at=started_at,
            )
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="unknown_tool",
                    message="tool is not installed in this runtime",
                    retryable=False,
                ),
                record=None,
                started_at=started_at,
            )
        if call.tool_name not in call.origin.exposed_tool_names:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="schema_not_exposed",
                    message="tool schema was not exposed by the originating request",
                    retryable=True,
                ),
                record=None,
                started_at=started_at,
            )

        record_error = _record_error(call, tool, record)
        if record_error is not None:
            if (
                record is not None
                and record_error == "tool_reconciliation_required"
                and record.status is ExecutionStatus.STARTED
            ):
                record = replace(
                    record,
                    status=ExecutionStatus.OUTCOME_UNKNOWN,
                    error_code="interrupted_outcome_unknown",
                    requires_reconciliation=True,
                )
            if record is not None and record_error in {
                "execution_already_completed",
                "execution_already_failed",
            }:
                return await self._finish(
                    call=call,
                    result=_recorded_outcome_result(call, record),
                    record=record,
                    record_sink=record_sink,
                    started_at=started_at,
                )
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code=record_error,
                    message=(
                        "tool outcome requires reconciliation before execution"
                        if record_error == "tool_reconciliation_required"
                        else "execution record does not match the tool call"
                    ),
                    retryable=False,
                ),
                record=record,
                record_sink=record_sink,
                started_at=started_at,
            )

        try:
            arguments = tool.validate_input(call.arguments)
        except ToolValidationError as error:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="invalid_arguments",
                    message=str(error),
                    retryable=True,
                ),
                record=None,
                started_at=started_at,
            )
        except Exception:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="invalid_arguments",
                    message="tool input validation failed",
                    retryable=True,
                ),
                record=None,
                started_at=started_at,
            )

        try:
            dynamic = tool.resolve_use(arguments)
            if not isinstance(dynamic, ResolvedToolUse):
                raise TypeError("resolve_use must return ResolvedToolUse")
            resolved = ResolvedToolUse(
                effects=tool.static_effects | dynamic.effects,
                targets=dynamic.targets,
            )
        except Exception:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="use_resolution_failed",
                    message="tool effects and targets could not be resolved",
                    retryable=False,
                ),
                record=None,
                started_at=started_at,
            )

        try:
            self._hard_guard(tool, arguments, resolved, context)
        except ToolGuardError as error:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code=error.code,
                    message=error.reason,
                    retryable=False,
                ),
                record=None,
                started_at=started_at,
                resolved=resolved,
            )
        except Exception:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="hard_guard_failed",
                    message="tool hard guard failed closed",
                    retryable=False,
                ),
                record=None,
                started_at=started_at,
                resolved=resolved,
            )

        try:
            boundary = self._boundary_resolver(tool, resolved, context)
            if not isinstance(boundary, ExecutionBoundary):
                raise TypeError("boundary resolver must return ExecutionBoundary")
        except Exception:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="execution_boundary_failed",
                    message="execution boundary could not be selected",
                    retryable=False,
                ),
                record=None,
                started_at=started_at,
                resolved=resolved,
            )

        try:
            decision = self._permission_decider(tool, arguments, resolved, context)
            if not isinstance(decision, CanUseToolResult):
                raise TypeError("permission decider must return CanUseToolResult")
        except Exception:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="permission_failed",
                    message="tool permission decision failed closed",
                    retryable=False,
                ),
                record=None,
                started_at=started_at,
                resolved=resolved,
                boundary=boundary,
            )
        if decision.decision is UseToolDecision.DENY:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="tool_denied",
                    message=decision.reason,
                    retryable=False,
                ),
                record=None,
                started_at=started_at,
                resolved=resolved,
                boundary=boundary,
                decision=decision.decision,
            )
        if decision.decision is UseToolDecision.ASK:
            approval_result = await self._require_approval(
                call=call,
                tool=tool,
                arguments=arguments,
                resolved=resolved,
                context=context,
                decision=decision,
                boundary=boundary,
                started_at=started_at,
            )
            if approval_result is not None:
                return approval_result

        if (
            ToolEffect.NETWORK in resolved.effects
            and ToolEffect.EXECUTE_PROCESS in resolved.effects
            and not (
                decision.decision is UseToolDecision.ASK
                and decision.approval_scope == "network"
            )
        ):
            network_decision = CanUseToolResult(
                UseToolDecision.ASK,
                "approval required for network access",
                approval_scope="network",
            )
            approval_result = await self._require_approval(
                call=call,
                tool=tool,
                arguments=arguments,
                resolved=resolved,
                context=context,
                decision=network_decision,
                boundary=boundary,
                started_at=started_at,
            )
            if approval_result is not None:
                return approval_result

        execution_record = replace(
            record or ToolExecutionRecord.prepare(call, tool),
            status=ExecutionStatus.PREPARED,
            error_code=None,
            requires_reconciliation=False,
        )
        await _emit_record(record_sink, execution_record)
        return _PreparedExecution(
            call=call,
            tool=tool,
            arguments=arguments,
            resolved=resolved,
            boundary=boundary,
            decision=decision,
            record=execution_record,
            record_sink=record_sink,
            started_at=started_at,
        )

    async def _require_approval(
        self,
        *,
        call: ToolCall,
        tool: Tool,
        arguments: Mapping[str, JsonValue],
        resolved: ResolvedToolUse,
        context: ToolExecutionContext,
        decision: CanUseToolResult,
        boundary: ExecutionBoundary,
        started_at: float,
    ) -> ToolExecution | None:
        try:
            approved = self._approval_resolver(
                call,
                tool,
                arguments,
                resolved,
                context,
                decision,
            )
            if inspect.isawaitable(approved):
                approved = await approved
            if approved not in (True, False, None):
                raise TypeError("approval resolver must return bool or None")
        except Exception:
            return await self._finish(
                call=call,
                result=_error_result(
                    call,
                    code="approval_failed",
                    message="tool approval resolution failed closed",
                    retryable=True,
                    metadata=_approval_metadata(call, resolved, decision),
                ),
                record=None,
                started_at=started_at,
                resolved=resolved,
                boundary=boundary,
                decision=decision.decision,
            )
        if approved is True:
            return None
        denied = approved is False
        return await self._finish(
            call=call,
            result=_error_result(
                call,
                code="tool_denied" if denied else "approval_required",
                message=("tool call was denied" if denied else decision.reason),
                retryable=not denied,
                metadata=_approval_metadata(call, resolved, decision),
            ),
            record=None,
            started_at=started_at,
            resolved=resolved,
            boundary=boundary,
            decision=decision.decision,
        )

    async def _invoke(self, prepared: _PreparedExecution) -> ToolExecution:
        started_record = replace(
            prepared.record,
            status=ExecutionStatus.STARTED,
            attempt_count=prepared.record.attempt_count + 1,
            error_code=None,
            requires_reconciliation=False,
        )
        await _emit_record(prepared.record_sink, started_record)
        prepared = replace(prepared, record=started_record)
        try:
            raw = await _run_with_timeout(
                prepared.tool,
                prepared.arguments,
            )
        except _CancelledTimeoutError:
            record = replace(
                prepared.record,
                status=ExecutionStatus.FAILED,
                error_code="timeout_cancelled",
            )
            return await self._finish(
                call=prepared.call,
                result=_error_result(
                    prepared.call,
                    code="timeout_cancelled",
                    message="tool timed out and local execution was cancelled",
                    retryable=True,
                ),
                record=record,
                prepared=prepared,
            )
        except _UnknownTimeoutError:
            requires_reconciliation = not prepared.tool.idempotent
            record = replace(
                prepared.record,
                status=ExecutionStatus.OUTCOME_UNKNOWN,
                error_code="timeout_outcome_unknown",
                requires_reconciliation=requires_reconciliation,
            )
            return await self._finish(
                call=prepared.call,
                result=_error_result(
                    prepared.call,
                    code="timeout_outcome_unknown",
                    message="tool timed out and the remote outcome is unknown",
                    retryable=prepared.tool.idempotent,
                ),
                record=record,
                prepared=prepared,
            )
        except _UnknownCancellationError:
            requires_reconciliation = not prepared.tool.idempotent
            record = replace(
                prepared.record,
                status=ExecutionStatus.OUTCOME_UNKNOWN,
                error_code="cancelled_outcome_unknown",
                requires_reconciliation=requires_reconciliation,
            )
            return await self._finish(
                call=prepared.call,
                result=_error_result(
                    prepared.call,
                    code="cancelled_outcome_unknown",
                    message=(
                        "local waiting was cancelled and the remote outcome is unknown"
                    ),
                    retryable=prepared.tool.idempotent,
                ),
                record=record,
                prepared=prepared,
            )
        except asyncio.CancelledError:
            record = replace(
                prepared.record,
                status=ExecutionStatus.FAILED,
                error_code="cancelled",
            )
            await self._finish(
                call=prepared.call,
                result=_error_result(
                    prepared.call,
                    code="cancelled",
                    message="tool execution was cancelled",
                    retryable=True,
                ),
                record=record,
                prepared=prepared,
            )
            raise
        except Exception:
            record = replace(
                prepared.record,
                status=ExecutionStatus.FAILED,
                error_code="runner_failed",
            )
            return await self._finish(
                call=prepared.call,
                result=_error_result(
                    prepared.call,
                    code="runner_failed",
                    message="tool runner failed",
                    retryable=prepared.tool.idempotent,
                ),
                record=record,
                prepared=prepared,
            )

        try:
            output = prepared.tool.normalize_output(raw)
            if not isinstance(output, NormalizedToolOutput):
                raise TypeError("normalize_output must return NormalizedToolOutput")
        except Exception:
            return await self._failed_after_runner(
                prepared,
                code="normalization_failed",
                message="tool output normalization failed",
            )
        try:
            output = self._output_validator(prepared.tool, output)
            if not isinstance(output, NormalizedToolOutput):
                raise TypeError("output validator must return NormalizedToolOutput")
        except Exception:
            return await self._failed_after_runner(
                prepared,
                code="output_validation_failed",
                message="normalized tool output failed validation",
            )
        try:
            output, truncated = self._externalizer(prepared.tool, output)
            if not isinstance(output, NormalizedToolOutput) or type(truncated) is not bool:
                raise TypeError("externalizer must return (NormalizedToolOutput, bool)")
        except Exception:
            return await self._failed_after_runner(
                prepared,
                code="externalization_failed",
                message="tool output could not be externalized or bounded",
            )

        result = ToolResult(
            tool_call_id=prepared.call.tool_call_id,
            tool_name=prepared.call.tool_name,
            content=output.content,
            structured_content=output.structured_content,
            is_error=output.is_error,
            error_code=output.error_code,
            error_message=output.error_message,
            retryable=output.retryable,
            truncated=truncated,
            metadata={
                **dict(output.metadata),
                "operation_id": prepared.record.operation_id,
            },
            attachments=output.attachments,
        )
        record = replace(
            prepared.record,
            status=(
                ExecutionStatus.FAILED
                if result.is_error
                else ExecutionStatus.COMPLETED
            ),
            error_code=result.error_code,
        )
        return await self._finish(
            call=prepared.call,
            result=result,
            record=record,
            prepared=prepared,
        )

    async def _failed_after_runner(
        self,
        prepared: _PreparedExecution,
        *,
        code: str,
        message: str,
    ) -> ToolExecution:
        record = replace(
            prepared.record,
            status=ExecutionStatus.FAILED,
            error_code=code,
        )
        return await self._finish(
            call=prepared.call,
            result=_error_result(
                prepared.call,
                code=code,
                message=message,
                retryable=False,
            ),
            record=record,
            prepared=prepared,
        )

    async def _finish(
        self,
        *,
        call: ToolCall,
        result: ToolResult,
        record: ToolExecutionRecord | None,
        record_sink: RecordSink | None = None,
        started_at: float | None = None,
        resolved: ResolvedToolUse | None = None,
        boundary: ExecutionBoundary | None = None,
        decision: UseToolDecision | None = None,
        prepared: _PreparedExecution | None = None,
    ) -> ToolExecution:
        if prepared is not None:
            started_at = prepared.started_at
            resolved = prepared.resolved
            boundary = prepared.boundary
            decision = prepared.decision.decision
            record_sink = prepared.record_sink
        if record is not None:
            await _emit_record(record_sink, record)
        trace = ToolExecutionTrace(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            outcome="error" if result.is_error else "success",
            error_code=result.error_code,
            decision=decision,
            boundary=boundary,
            effects=tuple(sorted((resolved.effects if resolved else ()), key=str)),
            targets=resolved.targets if resolved else (),
            duration_ms=max(0.0, (time.monotonic() - (started_at or time.monotonic())) * 1000),
        )
        self._traces.append(trace)
        if self._trace_sink is not None:
            try:
                emitted = self._trace_sink(trace)
                if inspect.isawaitable(emitted):
                    await emitted
            except Exception:
                pass
        return ToolExecution(result=result, record=record, trace=trace)


def resolve_execution_boundary(
    tool: Tool,
    resolved: ResolvedToolUse,
    context: ToolExecutionContext,
) -> ExecutionBoundary:
    del resolved, context
    if tool.cancellation_mode is CancellationMode.MANAGED_PROCESS:
        return ExecutionBoundary.MANAGED_PROCESS
    if tool.cancellation_mode in {
        CancellationMode.REMOTE_BEST_EFFORT,
        CancellationMode.NOT_CANCELLABLE,
    }:
        return ExecutionBoundary.REMOTE
    return ExecutionBoundary.DIRECT


def _resolve_external_approval(
    call: ToolCall,
    tool: Tool,
    arguments: Mapping[str, JsonValue],
    resolved: ResolvedToolUse,
    context: ToolExecutionContext,
    decision: CanUseToolResult,
) -> bool | None:
    del tool, arguments, resolved
    approval_id = (
        _network_approval_id(call.tool_call_id)
        if decision.approval_scope == "network"
        else call.tool_call_id
    )
    if approval_id in context.denied_tool_call_ids:
        return False
    if approval_id in context.approved_tool_call_ids:
        return True
    return None


def _validate_normalized_output(
    tool: Tool,
    output: NormalizedToolOutput,
) -> NormalizedToolOutput:
    if tool.output_schema is None:
        return output
    validated = json_schema_output(tool.output_schema, output.structured_content)
    return replace(output, structured_content=validated)


def _bound_model_output(
    tool: Tool,
    output: NormalizedToolOutput,
) -> tuple[NormalizedToolOutput, bool]:
    if _model_visible_size(output) <= tool.max_model_output_bytes:
        return output, False
    if output.structured_content is not None or any(
        block.type != "text" for block in output.content
    ):
        raise ToolValidationError(
            path="$",
            message="model output exceeds its byte limit and requires externalization",
        )

    empty_output = replace(
        output,
        content=tuple(
            ToolContentBlock(
                type="text",
                data={**dict(block.data), "text": ""},
            )
            for block in output.content
        ),
    )
    remaining = tool.max_model_output_bytes - _model_visible_size(empty_output)
    if remaining < 0:
        raise ToolValidationError(
            path="$",
            message="model output envelope exceeds its byte limit",
        )

    blocks: list[ToolContentBlock] = []
    for block in output.content:
        text = block.data.get("text")
        if not isinstance(text, str):
            raise ToolValidationError(
                path="$.content",
                message="text content blocks must contain string text",
            )
        bounded = _truncate_json_string(text, remaining)
        remaining -= _json_string_fragment_size(bounded)
        blocks.append(
            ToolContentBlock(
                type="text",
                data={**dict(block.data), "text": bounded},
            )
        )
    bounded_output = replace(output, content=tuple(blocks))
    if _model_visible_size(bounded_output) > tool.max_model_output_bytes:
        raise ToolValidationError(
            path="$",
            message="model output could not be bounded to its byte limit",
        )
    return bounded_output, True


async def _run_with_timeout(
    tool: Tool,
    arguments: Mapping[str, JsonValue],
) -> object:
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    task = asyncio.create_task(_call_runner(tool, arguments))
    remote = tool.cancellation_mode in {
        CancellationMode.REMOTE_BEST_EFFORT,
        CancellationMode.NOT_CANCELLABLE,
    }
    try:
        if remote:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=tool.timeout_seconds,
                )
            except TimeoutError:
                task.add_done_callback(_consume_background_task)
                raise _UnknownTimeoutError from None

        done, _ = await asyncio.wait({task}, timeout=tool.timeout_seconds)
        if task in done:
            return task.result()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            raise _UnknownTimeoutError from None
        raise _CancelledTimeoutError
    except asyncio.CancelledError:
        if tool.interrupt_behavior is InterruptBehavior.FINISH_CURRENT:
            remaining = max(0.0, tool.timeout_seconds - (loop.time() - started_at))
            if remote:
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=remaining,
                    )
                except TimeoutError:
                    task.add_done_callback(_consume_background_task)
                    raise _UnknownTimeoutError from None
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if task in done:
                return task.result()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                raise _UnknownTimeoutError from None
            raise _CancelledTimeoutError from None
        if remote:
            task.add_done_callback(_consume_background_task)
            raise _UnknownCancellationError from None
        else:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        raise


async def _call_runner(
    tool: Tool,
    arguments: Mapping[str, JsonValue],
) -> object:
    raw = tool.run(arguments)
    if inspect.isawaitable(raw):
        return await raw
    return raw


def _consume_background_task(task: asyncio.Task[object]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:
        pass


def _record_error(
    call: ToolCall,
    tool: Tool,
    record: ToolExecutionRecord | None,
) -> str | None:
    if record is None:
        return None
    if (
        record.tool_call_id != call.tool_call_id
        or record.tool_name != call.tool_name
        or record.arguments_digest != _arguments_digest(call.arguments)
        or record.idempotent != tool.idempotent
    ):
        return "execution_record_mismatch"
    if (
        record.status is ExecutionStatus.OUTCOME_UNKNOWN
        and record.requires_reconciliation
    ) or (
        record.status is ExecutionStatus.STARTED
        and not record.idempotent
    ):
        return "tool_reconciliation_required"
    if record.status is ExecutionStatus.COMPLETED:
        return "execution_already_completed"
    if record.status is ExecutionStatus.FAILED:
        return "execution_already_failed"
    return None


async def _emit_record(
    sink: RecordSink | None,
    record: ToolExecutionRecord,
) -> None:
    if sink is None:
        return
    emitted = sink(record)
    if inspect.isawaitable(emitted):
        await emitted


def _recorded_outcome_result(
    call: ToolCall,
    record: ToolExecutionRecord,
) -> ToolResult:
    completed = record.status is ExecutionStatus.COMPLETED
    text = (
        "The tool outcome was reconciled as completed; the operation was not "
        "replayed."
        if completed
        else "The tool outcome was reconciled as failed; the operation was not replayed."
    )
    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        content=(ToolContentBlock(type="text", data={"text": text}),),
        structured_content={
            "operation_id": record.operation_id,
            "status": record.status.value,
            "reconciled": True,
        },
        is_error=not completed,
        error_code=(None if completed else record.error_code or "reconciled_failed"),
        error_message=(None if completed else text),
        retryable=False,
        metadata={
            "operation_id": record.operation_id,
            "reconciled": True,
        },
    )


def _arguments_digest(arguments: Mapping[str, JsonValue]) -> str:
    encoded = json.dumps(
        _jsonable(arguments),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _error_result(
    call: ToolCall,
    *,
    code: str,
    message: str,
    retryable: bool,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ToolResult:
    safe_message = " ".join(message.split())[:512]
    return ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        content=(
            ToolContentBlock(type="text", data={"text": safe_message}),
        ),
        is_error=True,
        error_code=code,
        error_message=safe_message,
        retryable=retryable,
        metadata={} if metadata is None else metadata,
    )


def _approval_metadata(
    call: ToolCall,
    resolved: ResolvedToolUse,
    decision: CanUseToolResult,
) -> Mapping[str, JsonValue]:
    approval_id = (
        _network_approval_id(call.tool_call_id)
        if decision.approval_scope == "network"
        else call.tool_call_id
    )
    metadata: dict[str, JsonValue] = {
        "approval_id": approval_id,
        "approval_scope": decision.approval_scope,
        "network_requested": ToolEffect.NETWORK in resolved.effects,
        "workspace_write": ToolEffect.WRITE_WORKSPACE in resolved.effects,
    }
    for target in resolved.targets:
        if target.kind == "workspace_path":
            metadata["workspace_path"] = target.value
        elif target.kind == "cwd_path":
            metadata["cwd"] = target.value
        elif target.kind == "execution_mode":
            metadata["execution_mode"] = target.value
    return metadata


def _network_approval_id(tool_call_id: str) -> str:
    return f"{tool_call_id}{_NETWORK_APPROVAL_SUFFIX}"


def _model_visible_size(output: NormalizedToolOutput) -> int:
    value = {
        "content": [
            {"type": block.type, "data": _jsonable(block.data)}
            for block in output.content
        ],
        "structured_content": _jsonable(output.structured_content),
    }
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _json_string_fragment_size(value: str) -> int:
    encoded = json.dumps(value, ensure_ascii=False)[1:-1].encode("utf-8")
    return len(encoded)


def _truncate_json_string(value: str, max_bytes: int) -> str:
    if _json_string_fragment_size(value) <= max_bytes:
        return value

    low = 0
    high = len(value)
    while low < high:
        midpoint = (low + high + 1) // 2
        if _json_string_fragment_size(value[:midpoint]) <= max_bytes:
            low = midpoint
        else:
            high = midpoint - 1
    return value[:low]


def _can_run_in_parallel(prepared: tuple[_PreparedExecution, ...]) -> bool:
    if len(prepared) < 2:
        return False
    if any(not item.tool.concurrency_safe for item in prepared):
        return False
    return not any(
        _resolved_uses_conflict(left.resolved, right.resolved)
        for index, left in enumerate(prepared)
        for right in prepared[index + 1 :]
    )


def _resolved_uses_conflict(
    left: ResolvedToolUse,
    right: ResolvedToolUse,
) -> bool:
    if ToolEffect.DESTRUCTIVE in left.effects | right.effects:
        return True
    left_side_effecting = _is_side_effecting(left.effects)
    right_side_effecting = _is_side_effecting(right.effects)
    if not left_side_effecting and not right_side_effecting:
        return False
    if not left.targets or not right.targets:
        return True
    return any(
        _targets_overlap(left_target, right_target)
        for left_target in left.targets
        for right_target in right.targets
    )


def _is_side_effecting(effects: frozenset[ToolEffect]) -> bool:
    return bool(effects - {ToolEffect.READ_WORKSPACE})


def _targets_overlap(left: ToolTarget, right: ToolTarget) -> bool:
    path_kinds = {"workspace_path", "cwd_path"}
    if left.kind in path_kinds and right.kind in path_kinds:
        left_path = Path(left.value).expanduser().resolve()
        right_path = Path(right.value).expanduser().resolve()
        try:
            common = Path(os.path.commonpath((str(left_path), str(right_path))))
        except ValueError:
            return False
        return common in {left_path, right_path}
    return left == right


__all__ = [
    "ExecutionBoundary",
    "ExecutionStatus",
    "ToolExecution",
    "ToolExecutionRecord",
    "ToolExecutionTrace",
    "ToolExecutor",
    "resolve_execution_boundary",
]

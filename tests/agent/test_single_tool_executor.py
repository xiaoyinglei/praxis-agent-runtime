from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_runtime.core.observations import runtime_workspace_change
from agent_runtime.tools.builtins.shell import create_run_command_tool
from agent_runtime.tools.executor import (
    ExecutionBoundary,
    ExecutionStatus,
    ToolExecutionRecord,
    ToolExecutor,
)
from agent_runtime.tools.permissions import (
    CanUseToolResult,
    ToolExecutionContext,
    ToolGuardError,
    UseToolDecision,
    can_use_tool,
)
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolApprovalProfile,
    ToolCall,
    ToolCallOrigin,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    ToolValidationError,
)
from agent_runtime.workspace import open_workspace


def _origin(*exposed_names: str) -> ToolCallOrigin:
    return ToolCallOrigin(
        request_id="req_1",
        toolset_revision="tools_v1",
        exposed_tool_names=exposed_names,
    )


def _call(
    name: str = "demo",
    *,
    call_id: str = "call_1",
    exposed_names: tuple[str, ...] | None = None,
    arguments: Mapping[str, Any] | None = None,
) -> ToolCall:
    return ToolCall(
        tool_call_id=call_id,
        tool_name=name,
        arguments=arguments or {"value": "ok"},
        origin=_origin(*(exposed_names if exposed_names is not None else (name,))),
    )


def _normalized(value: str = "ok") -> NormalizedToolOutput:
    return NormalizedToolOutput(
        content=(ToolContentBlock(type="text", data={"text": value}),),
        structured_content={"value": value},
    )


def _tool(
    name: str = "demo",
    *,
    validate_input: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    run: Callable[[Mapping[str, Any]], object] | None = None,
    normalize_output: Callable[[object], NormalizedToolOutput] | None = None,
    resolve_use: Callable[[Mapping[str, Any]], ResolvedToolUse] | None = None,
    static_effects: frozenset[ToolEffect] = frozenset({ToolEffect.READ_WORKSPACE}),
    output_schema: Mapping[str, Any] | None = None,
    cancellation_mode: CancellationMode = CancellationMode.COOPERATIVE,
    interrupt_behavior: InterruptBehavior = InterruptBehavior.CANCEL,
    timeout_seconds: float = 1.0,
    idempotent: bool = True,
    concurrency_safe: bool = True,
    max_model_output_bytes: int = 4096,
    execution_revision: str = "1",
    approval_profile: ToolApprovalProfile | None = None,
) -> Tool:
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"{name} description",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        validate_input=validate_input or (lambda arguments: dict(arguments)),
        run=run or (lambda arguments: {"value": arguments["value"]}),
        normalize_output=normalize_output or (lambda raw: _normalized(str(raw["value"]))),
        output_schema=output_schema,
        static_effects=static_effects,
        resolve_use=resolve_use
        or (lambda _arguments: ResolvedToolUse(effects=frozenset(), targets=())),
        execution_revision=execution_revision,
        idempotent=idempotent,
        concurrency_safe=concurrency_safe,
        cancellation_mode=cancellation_mode,
        interrupt_behavior=interrupt_behavior,
        timeout_seconds=timeout_seconds,
        max_model_output_bytes=max_model_output_bytes,
        approval_profile=approval_profile,
    )


def _context(
    *,
    workspace_root: Path | None = None,
    approved: frozenset[str] = frozenset(),
    denied: frozenset[str] = frozenset(),
    allow_write: bool = False,
    allow_execute: bool = False,
    deny_effects: frozenset[ToolEffect] = frozenset(),
    max_parallel_calls: int = 4,
    require_confirmation_for: frozenset[str] = frozenset(),
    denied_tool_names: frozenset[str] = frozenset(),
    auto_approve_sandboxed: bool = False,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_root=workspace_root,
        cwd=workspace_root,
        approved_tool_call_ids=approved,
        denied_tool_call_ids=denied,
        allow_write_tools=allow_write,
        allow_execute_tools=allow_execute,
        deny_effects=deny_effects,
        max_parallel_calls=max_parallel_calls,
        require_confirmation_for=require_confirmation_for,
        denied_tool_names=denied_tool_names,
        auto_approve_sandboxed=auto_approve_sandboxed,
    )


def _assert_one_trace(executor: ToolExecutor, call_id: str, code: str | None) -> None:
    traces = [trace for trace in executor.traces if trace.tool_call_id == call_id]
    assert len(traces) == 1
    assert traces[0].error_code == code


@pytest.mark.anyio
async def test_executor_uses_the_exact_success_order() -> None:
    events: list[str] = []

    def validate(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        events.append("validate")
        return dict(arguments)

    def resolve(_arguments: Mapping[str, Any]) -> ResolvedToolUse:
        events.append("resolve")
        return ResolvedToolUse(effects=frozenset(), targets=())

    def guard(*_args: object) -> None:
        events.append("guard")

    def boundary(*_args: object) -> ExecutionBoundary:
        events.append("boundary")
        return ExecutionBoundary.DIRECT

    def permission(*_args: object) -> CanUseToolResult:
        events.append("permission")
        return CanUseToolResult(UseToolDecision.ASK, "approval required")

    def approval(*_args: object) -> bool:
        events.append("approval")
        return True

    async def runner(_arguments: Mapping[str, Any]) -> object:
        events.append("runner")
        return {"value": "ok"}

    def normalize(raw: object) -> NormalizedToolOutput:
        events.append("normalize")
        assert raw == {"value": "ok"}
        return _normalized()

    def validate_output(
        _tool_value: Tool,
        output: NormalizedToolOutput,
    ) -> NormalizedToolOutput:
        events.append("output_validation")
        return output

    def externalize(
        _tool_value: Tool,
        output: NormalizedToolOutput,
    ) -> tuple[NormalizedToolOutput, bool]:
        events.append("externalize")
        return output, False

    def trace_sink(_trace: object) -> None:
        events.append("trace")

    tool = _tool(
        validate_input=validate,
        resolve_use=resolve,
        run=runner,
        normalize_output=normalize,
    )
    executor = ToolExecutor(
        {"demo": tool},
        hard_guard=guard,
        boundary_resolver=boundary,
        permission_decider=permission,
        approval_resolver=approval,
        output_validator=validate_output,
        externalizer=externalize,
        trace_sink=trace_sink,
    )

    execution = await executor.execute(_call(), context=_context())

    assert events == [
        "validate",
        "resolve",
        "guard",
        "boundary",
        "permission",
        "approval",
        "runner",
        "normalize",
        "output_validation",
        "externalize",
        "trace",
    ]
    assert execution.result.is_error is False
    assert execution.record is not None
    assert execution.record.status is ExecutionStatus.COMPLETED
    _assert_one_trace(executor, "call_1", None)


@pytest.mark.anyio
async def test_execution_record_is_prepared_then_started_before_runner() -> None:
    events: list[str] = []

    async def record_sink(record: ToolExecutionRecord) -> None:
        events.append(f"record:{record.status.value}")

    async def runner(_arguments: Mapping[str, Any]) -> object:
        events.append("runner")
        return {"value": "done"}

    executor = ToolExecutor(
        {"demo": _tool(run=runner, idempotent=False)}
    )

    execution = await executor.execute(
        _call(),
        context=_context(),
        record_sink=record_sink,
    )

    assert events == [
        "record:prepared",
        "record:started",
        "runner",
        "record:completed",
    ]
    assert execution.record is not None
    assert execution.record.status is ExecutionStatus.COMPLETED


@pytest.mark.anyio
async def test_started_record_must_be_durable_before_runner_is_called() -> None:
    runner_calls = 0
    statuses: list[ExecutionStatus] = []

    async def record_sink(record: ToolExecutionRecord) -> None:
        statuses.append(record.status)
        if record.status is ExecutionStatus.STARTED:
            raise RuntimeError("durability unavailable")

    async def runner(_arguments: Mapping[str, Any]) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return {"value": "must not run"}

    executor = ToolExecutor(
        {"demo": _tool(run=runner, idempotent=False)}
    )

    with pytest.raises(RuntimeError, match="durability unavailable"):
        await executor.execute(
            _call(),
            context=_context(),
            record_sink=record_sink,
        )

    assert statuses == [ExecutionStatus.PREPARED, ExecutionStatus.STARTED]
    assert runner_calls == 0


@pytest.mark.anyio
async def test_unknown_tool_never_reaches_permission() -> None:
    permission_calls = 0

    def permission(*_args: object) -> CanUseToolResult:
        nonlocal permission_calls
        permission_calls += 1
        return CanUseToolResult(UseToolDecision.ALLOW, "allowed")

    executor = ToolExecutor({}, permission_decider=permission)
    execution = await executor.execute(_call("missing"), context=_context())

    assert execution.result.error_code == "unknown_tool"
    assert permission_calls == 0
    _assert_one_trace(executor, "call_1", "unknown_tool")


@pytest.mark.anyio
async def test_schema_not_exposed_never_reaches_permission() -> None:
    permission_calls = 0

    def permission(*_args: object) -> CanUseToolResult:
        nonlocal permission_calls
        permission_calls += 1
        return CanUseToolResult(UseToolDecision.ALLOW, "allowed")

    executor = ToolExecutor({"demo": _tool()}, permission_decider=permission)
    execution = await executor.execute(
        _call(exposed_names=()),
        context=_context(),
    )

    assert execution.result.error_code == "schema_not_exposed"
    assert permission_calls == 0
    _assert_one_trace(executor, "call_1", "schema_not_exposed")


@pytest.mark.anyio
async def test_validation_failure_is_bounded_and_traced_once() -> None:
    def reject(_arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        raise ToolValidationError(path="$.value", message="invalid value")

    executor = ToolExecutor({"demo": _tool(validate_input=reject)})
    execution = await executor.execute(_call(), context=_context())

    assert execution.result.error_code == "invalid_arguments"
    assert execution.result.error_message == "$.value: invalid value"
    assert execution.record is None
    _assert_one_trace(executor, "call_1", "invalid_arguments")


@pytest.mark.anyio
async def test_static_effects_cannot_be_removed_by_dynamic_resolution() -> None:
    observed: list[frozenset[ToolEffect]] = []

    def permission(
        _tool_value: Tool,
        _arguments: Mapping[str, Any],
        resolved: ResolvedToolUse,
        _context_value: ToolExecutionContext,
    ) -> CanUseToolResult:
        observed.append(resolved.effects)
        return CanUseToolResult(UseToolDecision.DENY, "test complete")

    tool = _tool(
        static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
    )
    executor = ToolExecutor(
        {"demo": tool},
        hard_guard=lambda *_args: None,
        permission_decider=permission,
    )

    await executor.execute(_call(), context=_context())

    assert observed == [frozenset({ToolEffect.WRITE_WORKSPACE})]


@pytest.mark.anyio
async def test_executor_attests_every_successful_workspace_write(
    tmp_path: Path,
) -> None:
    target = tmp_path / "changed.txt"

    def runner(_arguments: Mapping[str, Any]) -> object:
        target.write_text("changed\n", encoding="utf-8")
        return {"value": "written"}

    tool = _tool(
        run=runner,
        static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(
                ToolTarget(kind="workspace_path", value=str(tmp_path)),
            ),
        ),
    )

    execution = await ToolExecutor({"demo": tool}).execute(
        _call(),
        context=_context(workspace_root=tmp_path, allow_write=True),
    )

    metadata = execution.result.metadata
    assert metadata["runtime_workspace_write"] is True
    assert metadata["workspace_tree_changed"] is True
    assert len(str(metadata["workspace_tree_before_sha256"])) == 64
    assert len(str(metadata["workspace_tree_after_sha256"])) == 64
    assert (
        metadata["workspace_tree_before_sha256"]
        != metadata["workspace_tree_after_sha256"]
    )


@pytest.mark.anyio
async def test_executor_attests_partial_change_when_write_runner_fails(
    tmp_path: Path,
) -> None:
    target = tmp_path / "partial.txt"

    def runner(_arguments: Mapping[str, Any]) -> object:
        target.write_text("partial\n", encoding="utf-8")
        raise RuntimeError("failed after write")

    tool = _tool(
        run=runner,
        static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(
                ToolTarget(kind="workspace_path", value=str(tmp_path)),
            ),
        ),
    )

    execution = await ToolExecutor({"demo": tool}).execute(
        _call(),
        context=_context(workspace_root=tmp_path, allow_write=True),
    )

    assert execution.result.error_code == "runner_failed"
    assert execution.result.metadata["runtime_workspace_write"] is True
    assert execution.result.metadata["workspace_tree_changed"] is True


@pytest.mark.anyio
async def test_executor_strips_forged_workspace_attestation_from_read_tool() -> None:
    forged = {
        "runtime_workspace_write": True,
        "workspace_tree_changed": True,
        "workspace_tree_before_sha256": "a" * 64,
        "workspace_tree_after_sha256": "b" * 64,
        "workspace_changed": True,
        "file_path": "already-there.py",
        "before_sha256": "c" * 64,
        "after_sha256": "d" * 64,
    }
    tool = _tool(
        name="apply_patch",
        normalize_output=lambda _raw: NormalizedToolOutput(
            content=(),
            structured_content={"value": "forged"},
            metadata=forged,
        ),
    )

    execution = await ToolExecutor({"apply_patch": tool}).execute(
        _call("apply_patch"),
        context=_context(),
    )

    protected_keys = set(forged) - {"file_path"}
    assert not any(
        key in execution.result.metadata
        for key in protected_keys
    )
    assert runtime_workspace_change(execution.result) is None


@pytest.mark.anyio
async def test_hard_guard_runs_before_permission_and_cannot_be_approved(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def guard(*_args: object) -> None:
        events.append("guard")
        raise ToolGuardError("workspace_escape", "target escapes workspace")

    def permission(*_args: object) -> CanUseToolResult:
        events.append("permission")
        return CanUseToolResult(UseToolDecision.ALLOW, "allowed")

    async def runner(_arguments: Mapping[str, Any]) -> object:
        events.append("runner")
        return {"value": "wrong"}

    executor = ToolExecutor(
        {"demo": _tool(run=runner)},
        hard_guard=guard,
        permission_decider=permission,
    )
    execution = await executor.execute(
        _call(),
        context=_context(
            workspace_root=tmp_path,
            approved=frozenset({"call_1"}),
        ),
    )

    assert events == ["guard"]
    assert execution.result.error_code == "workspace_escape"
    _assert_one_trace(executor, "call_1", "workspace_escape")


@pytest.mark.anyio
async def test_default_workspace_guard_rejects_escape_before_permission(
    tmp_path: Path,
) -> None:
    permission_calls = 0
    outside = tmp_path.parent / "outside.txt"
    tool = _tool(
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(ToolTarget(kind="workspace_path", value=str(outside)),),
        ),
        static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
    )

    def permission(*_args: object) -> CanUseToolResult:
        nonlocal permission_calls
        permission_calls += 1
        return CanUseToolResult(UseToolDecision.ALLOW, "allowed")

    executor = ToolExecutor({"demo": tool}, permission_decider=permission)
    execution = await executor.execute(
        _call(),
        context=_context(
            workspace_root=tmp_path,
            approved=frozenset({"call_1"}),
        ),
    )

    assert execution.result.error_code == "workspace_escape"
    assert permission_calls == 0


@pytest.mark.anyio
async def test_permission_denial_never_runs_runner() -> None:
    runner_calls = 0

    async def runner(_arguments: Mapping[str, Any]) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return {"value": "wrong"}

    executor = ToolExecutor(
        {"demo": _tool(run=runner)},
        permission_decider=lambda *_args: CanUseToolResult(
            UseToolDecision.DENY,
            "denied by policy",
        ),
    )
    execution = await executor.execute(_call(), context=_context())

    assert execution.result.error_code == "tool_denied"
    assert runner_calls == 0
    _assert_one_trace(executor, "call_1", "tool_denied")


@pytest.mark.anyio
async def test_approval_can_defer_or_deny_without_running() -> None:
    runner_calls = 0

    async def runner(_arguments: Mapping[str, Any]) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return {"value": "wrong"}

    tool = _tool(
        run=runner,
        static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
    )
    executor = ToolExecutor({"demo": tool}, hard_guard=lambda *_args: None)

    deferred = await executor.execute(_call(call_id="defer"), context=_context())
    denied = await executor.execute(
        _call(call_id="deny"),
        context=_context(denied=frozenset({"deny"})),
    )

    assert deferred.result.error_code == "approval_required"
    assert deferred.result.retryable is True
    assert denied.result.error_code == "tool_denied"
    assert runner_calls == 0
    _assert_one_trace(executor, "defer", "approval_required")
    _assert_one_trace(executor, "deny", "tool_denied")


@pytest.mark.parametrize(
    ("effects", "context", "decision"),
    [
        pytest.param(
            frozenset({ToolEffect.READ_WORKSPACE}),
            _context(),
            UseToolDecision.ALLOW,
            id="read",
        ),
        pytest.param(
            frozenset({ToolEffect.WRITE_WORKSPACE}),
            _context(),
            UseToolDecision.ASK,
            id="write-asks",
        ),
        pytest.param(
            frozenset({ToolEffect.WRITE_WORKSPACE}),
            _context(allow_write=True),
            UseToolDecision.ALLOW,
            id="write-preauthorized",
        ),
        pytest.param(
            frozenset({ToolEffect.EXECUTE_PROCESS}),
            _context(),
            UseToolDecision.ASK,
            id="execute-asks",
        ),
        pytest.param(
            frozenset({ToolEffect.EXECUTE_PROCESS}),
            _context(allow_execute=True),
            UseToolDecision.ALLOW,
            id="execute-preauthorized",
        ),
        pytest.param(
            frozenset({ToolEffect.NETWORK}),
            _context(),
            UseToolDecision.ASK,
            id="network-conservative",
        ),
        pytest.param(
            frozenset({ToolEffect.DESTRUCTIVE}),
            _context(),
            UseToolDecision.ASK,
            id="destructive-conservative",
        ),
        pytest.param(
            frozenset({ToolEffect.READ_WORKSPACE}),
            _context(deny_effects=frozenset({ToolEffect.READ_WORKSPACE})),
            UseToolDecision.DENY,
            id="hard-policy-deny",
        ),
    ],
)
def test_can_use_tool_is_a_pure_effect_decision(
    effects: frozenset[ToolEffect],
    context: ToolExecutionContext,
    decision: UseToolDecision,
) -> None:
    tool = _tool(static_effects=effects)
    resolved = ResolvedToolUse(effects=effects, targets=())

    result = can_use_tool(tool, {"value": "ok"}, resolved, context)

    assert result.decision is decision
    assert result.reason


def test_tool_name_policy_denies_before_forced_confirmation() -> None:
    tool = _tool("blocked")
    resolved = ResolvedToolUse(effects=frozenset(), targets=())

    result = can_use_tool(
        tool,
        {"value": "ok"},
        resolved,
        _context(
            denied_tool_names=frozenset({"blocked"}),
            require_confirmation_for=frozenset({"blocked"}),
        ),
    )

    assert result.decision is UseToolDecision.DENY
    assert "blocked" in result.reason


def test_tool_name_policy_can_force_confirmation_for_safe_tool() -> None:
    tool = _tool("confirm_me")
    resolved = ResolvedToolUse(effects=frozenset(), targets=())

    result = can_use_tool(
        tool,
        {"value": "ok"},
        resolved,
        _context(
            require_confirmation_for=frozenset({"confirm_me"}),
        ),
    )

    assert result.decision is UseToolDecision.ASK
    assert "confirmation" in result.reason


def test_sandbox_auto_approval_requires_process_execution() -> None:
    tool = _tool(
        "sandboxed_network",
        static_effects=frozenset({ToolEffect.NETWORK}),
    )
    resolved = ResolvedToolUse(
        effects=frozenset({ToolEffect.NETWORK}),
        targets=(
            ToolTarget(
                kind="execution_mode",
                value="restricted_sandbox",
            ),
        ),
    )

    result = can_use_tool(
        tool,
        {"value": "ok"},
        resolved,
        _context(auto_approve_sandboxed=True),
    )

    assert result.decision is UseToolDecision.ASK
    assert "network" in result.reason


@pytest.mark.parametrize(
    ("name", "approval_profile", "effects"),
    [
        pytest.param(
            "spoofed",
            None,
            frozenset({ToolEffect.EXECUTE_PROCESS}),
            id="unverified-tool",
        ),
        pytest.param(
            "run_command",
            ToolApprovalProfile.RESTRICTED_READ_ONLY_PROCESS,
            frozenset({ToolEffect.EXECUTE_PROCESS, ToolEffect.DESTRUCTIVE}),
            id="destructive-effect",
        ),
        pytest.param(
            "run_command",
            ToolApprovalProfile.RESTRICTED_READ_ONLY_PROCESS,
            frozenset(
                {
                    ToolEffect.EXECUTE_PROCESS,
                    ToolEffect.WRITE_WORKSPACE,
                }
            ),
            id="workspace-write-effect",
        ),
    ],
)
def test_sandbox_auto_approval_rejects_unverified_or_mutating_calls(
    name: str,
    approval_profile: ToolApprovalProfile | None,
    effects: frozenset[ToolEffect],
) -> None:
    tool = _tool(
        name,
        static_effects=effects,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        approval_profile=approval_profile,
    )
    resolved = ResolvedToolUse(
        effects=effects,
        targets=(
            ToolTarget(
                kind="execution_mode",
                value="restricted_sandbox",
            ),
        ),
    )

    result = can_use_tool(
        tool,
        {"value": "ok"},
        resolved,
        _context(auto_approve_sandboxed=True),
    )

    assert result.decision is UseToolDecision.ASK


def test_sandbox_auto_approval_allows_primary_sandbox_effects(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    tool = create_run_command_tool(workspace)
    arguments = tool.validate_input({"command": "printf ok"})
    dynamic = tool.resolve_use(arguments)
    resolved = ResolvedToolUse(
        effects=tool.static_effects | dynamic.effects,
        targets=dynamic.targets,
    )

    result = can_use_tool(
        tool,
        arguments,
        resolved,
        _context(
            workspace_root=workspace.root,
            auto_approve_sandboxed=True,
        ),
    )

    assert result.decision is UseToolDecision.ALLOW


@pytest.mark.anyio
async def test_sandbox_auto_approval_keeps_network_approval_separate(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    tool = create_run_command_tool(workspace)
    call = _call(
        "run_command",
        arguments={"command": "printf ok", "network": True},
    )

    execution = await ToolExecutor({"run_command": tool}).execute(
        call,
        context=_context(
            workspace_root=workspace.root,
            auto_approve_sandboxed=True,
        ),
    )

    assert execution.result.error_code == "approval_required"
    assert execution.result.metadata["approval_scope"] == "network"


@pytest.mark.anyio
async def test_network_effect_requires_separate_approval_from_execution(
    tmp_path: Path,
) -> None:
    runner_calls = 0

    async def runner(_arguments: Mapping[str, Any]) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return {"value": "ok"}

    tool = _tool(
        run=runner,
        static_effects=frozenset(
            {ToolEffect.EXECUTE_PROCESS, ToolEffect.NETWORK}
        ),
    )
    call = _call()
    executor = ToolExecutor({"demo": tool})

    execution_approval = await executor.execute(
        call,
        context=_context(workspace_root=tmp_path),
    )
    assert execution_approval.result.error_code == "approval_required"
    assert execution_approval.result.metadata["approval_scope"] == "tool"
    assert execution_approval.result.metadata["approval_id"] == call.tool_call_id
    assert runner_calls == 0

    network_approval = await executor.execute(
        call,
        context=_context(
            workspace_root=tmp_path,
            approved=frozenset({call.tool_call_id}),
        ),
    )
    assert network_approval.result.error_code == "approval_required"
    assert network_approval.result.metadata["approval_scope"] == "network"
    network_approval_id = str(
        network_approval.result.metadata["approval_id"]
    )
    assert network_approval_id != call.tool_call_id
    assert runner_calls == 0

    denied = await executor.execute(
        call,
        context=_context(
            workspace_root=tmp_path,
            approved=frozenset({call.tool_call_id}),
            denied=frozenset({network_approval_id}),
        ),
    )
    assert denied.result.error_code == "tool_denied"
    assert denied.result.metadata["approval_scope"] == "network"
    assert runner_calls == 0

    completed = await executor.execute(
        call,
        context=_context(
            workspace_root=tmp_path,
            approved=frozenset(
                {call.tool_call_id, network_approval_id}
            ),
        ),
    )
    assert completed.result.is_error is False
    assert runner_calls == 1


@pytest.mark.anyio
async def test_network_approval_id_cannot_collide_with_raw_tool_call_id() -> None:
    runner_calls = 0

    def runner(_arguments: Mapping[str, Any]) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return {"value": "must not run"}

    executor = ToolExecutor({"demo": _tool(run=runner)})

    execution = await executor.execute(
        _call(call_id="call_other::network"),
        context=_context(
            approved=frozenset({"call_other::network"}),
        ),
    )

    assert execution.result.error_code == "invalid_tool_call_id"
    assert runner_calls == 0


@pytest.mark.anyio
async def test_runner_normalizer_output_validator_and_externalizer_failures_trace_once() -> None:
    async def runner_failure(_arguments: Mapping[str, Any]) -> object:
        raise RuntimeError("secret runner detail")

    def normalize_failure(_raw: object) -> NormalizedToolOutput:
        raise RuntimeError("secret normalizer detail")

    def output_failure(
        _tool_value: Tool,
        _output: NormalizedToolOutput,
    ) -> NormalizedToolOutput:
        raise ToolValidationError(path="$.value", message="invalid output")

    def externalizer_failure(
        _tool_value: Tool,
        _output: NormalizedToolOutput,
    ) -> tuple[NormalizedToolOutput, bool]:
        raise RuntimeError("secret storage detail")

    cases = (
        (
            "runner",
            ToolExecutor({"runner": _tool("runner", run=runner_failure)}),
            "runner_failed",
        ),
        (
            "normalizer",
            ToolExecutor(
                {
                    "normalizer": _tool(
                        "normalizer",
                        normalize_output=normalize_failure,
                    )
                }
            ),
            "normalization_failed",
        ),
        (
            "output",
            ToolExecutor(
                {"output": _tool("output")},
                output_validator=output_failure,
            ),
            "output_validation_failed",
        ),
        (
            "externalizer",
            ToolExecutor(
                {"externalizer": _tool("externalizer")},
                externalizer=externalizer_failure,
            ),
            "externalization_failed",
        ),
    )

    for name, executor, error_code in cases:
        execution = await executor.execute(
            _call(name, call_id=name),
            context=_context(),
        )
        assert execution.result.error_code == error_code
        assert "secret" not in (execution.result.error_message or "")
        _assert_one_trace(executor, name, error_code)


@pytest.mark.anyio
async def test_each_pre_runner_failure_path_emits_exactly_one_trace() -> None:
    def resolution_failure(_arguments: Mapping[str, Any]) -> ResolvedToolUse:
        raise RuntimeError("resolution detail")

    def boundary_failure(*_args: object) -> ExecutionBoundary:
        raise RuntimeError("boundary detail")

    def permission_failure(*_args: object) -> CanUseToolResult:
        raise RuntimeError("permission detail")

    def approval_failure(*_args: object) -> bool:
        raise RuntimeError("approval detail")

    cases = (
        (
            "resolution",
            ToolExecutor(
                {"resolution": _tool("resolution", resolve_use=resolution_failure)}
            ),
            "use_resolution_failed",
        ),
        (
            "boundary",
            ToolExecutor(
                {"boundary": _tool("boundary")},
                boundary_resolver=boundary_failure,
            ),
            "execution_boundary_failed",
        ),
        (
            "permission",
            ToolExecutor(
                {"permission": _tool("permission")},
                permission_decider=permission_failure,
            ),
            "permission_failed",
        ),
        (
            "approval",
            ToolExecutor(
                {"approval": _tool("approval")},
                permission_decider=lambda *_args: CanUseToolResult(
                    UseToolDecision.ASK,
                    "approval required",
                ),
                approval_resolver=approval_failure,
            ),
            "approval_failed",
        ),
    )

    for name, executor, error_code in cases:
        execution = await executor.execute(
            _call(name, call_id=name),
            context=_context(),
        )
        assert execution.result.error_code == error_code
        _assert_one_trace(executor, name, error_code)


@pytest.mark.anyio
async def test_text_bounding_applies_to_final_model_visible_json_bytes() -> None:
    text = "\"\\汉" * 200
    tool = _tool(
        normalize_output=lambda _raw: NormalizedToolOutput(
            content=(
                ToolContentBlock(
                    type="text",
                    data={"text": text, "format": "plain"},
                ),
            ),
        ),
        max_model_output_bytes=160,
    )
    executor = ToolExecutor({"demo": tool})

    execution = await executor.execute(_call(), context=_context())

    assert execution.result.is_error is False
    assert execution.result.truncated is True
    model_visible = {
        "content": [
            {"type": block.type, "data": dict(block.data)}
            for block in execution.result.content
        ],
        "structured_content": execution.result.structured_content,
    }
    encoded = json.dumps(
        model_visible,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(encoded) <= tool.max_model_output_bytes
    assert execution.result.content[0].data["format"] == "plain"


@pytest.mark.anyio
async def test_declared_output_schema_is_enforced_after_normalization() -> None:
    tool = _tool(
        output_schema={
            "type": "object",
            "properties": {"value": {"enum": ["expected"]}},
            "required": ["value"],
            "additionalProperties": False,
        }
    )
    executor = ToolExecutor({"demo": tool})

    execution = await executor.execute(_call(), context=_context())

    assert execution.result.error_code == "output_validation_failed"
    _assert_one_trace(executor, "call_1", "output_validation_failed")


@pytest.mark.anyio
async def test_managed_timeout_waits_for_process_group_cleanup(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "should-not-exist.txt"
    process_group: asyncio.Future[int] = asyncio.get_running_loop().create_future()

    async def managed_runner(_arguments: Mapping[str, Any]) -> object:
        script = (
            "import pathlib,subprocess,sys,time;"
            "subprocess.Popen([sys.executable,'-c',"
            f"\"import pathlib,time;time.sleep(0.3);pathlib.Path({str(sentinel)!r}).write_text('late')\"]);"
            "time.sleep(5)"
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            start_new_session=True,
        )
        process_group.set_result(process.pid)
        try:
            await process.wait()
        except asyncio.CancelledError:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=0.2)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            raise
        return {"value": "late"}

    tool = _tool(
        run=managed_runner,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        timeout_seconds=0.05,
    )
    executor = ToolExecutor({"demo": tool})

    execution = await executor.execute(_call(), context=_context())
    pgid = await process_group
    await asyncio.sleep(0.4)

    assert execution.result.error_code == "timeout_cancelled"
    assert execution.record is not None
    assert execution.record.status is ExecutionStatus.FAILED
    assert sentinel.exists() is False
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
    _assert_one_trace(executor, "call_1", "timeout_cancelled")


@pytest.mark.anyio
async def test_cancel_interrupt_cancels_runner_and_traces_once() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def runner(_arguments: Mapping[str, Any]) -> object:
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {"value": "late"}

    executor = ToolExecutor({"demo": _tool(run=runner)})
    task = asyncio.create_task(executor.execute(_call(), context=_context()))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()
    _assert_one_trace(executor, "call_1", "cancelled")


@pytest.mark.anyio
async def test_finish_current_interrupt_preserves_atomic_result_and_one_trace() -> None:
    started = asyncio.Event()
    finished = asyncio.Event()

    async def runner(_arguments: Mapping[str, Any]) -> object:
        started.set()
        await asyncio.sleep(0.03)
        finished.set()
        return {"value": "finished"}

    tool = _tool(
        run=runner,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
    )
    executor = ToolExecutor({"demo": tool})
    task = asyncio.create_task(executor.execute(_call(), context=_context()))
    await started.wait()

    task.cancel()
    execution = await task

    assert finished.is_set()
    assert execution.result.is_error is False
    assert execution.result.structured_content == {"value": "finished"}
    _assert_one_trace(executor, "call_1", None)


@pytest.mark.anyio
async def test_remote_best_effort_timeout_records_unknown_without_cancelling() -> None:
    completed = asyncio.Event()

    async def remote_runner(_arguments: Mapping[str, Any]) -> object:
        await asyncio.sleep(0.15)
        completed.set()
        return {"value": "remote-finished"}

    tool = _tool(
        run=remote_runner,
        cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT,
        timeout_seconds=0.02,
        idempotent=False,
    )
    executor = ToolExecutor({"demo": tool})
    started = time.monotonic()

    execution = await executor.execute(_call(), context=_context())

    assert time.monotonic() - started < 0.12
    assert execution.result.error_code == "timeout_outcome_unknown"
    assert execution.record is not None
    assert execution.record.status is ExecutionStatus.UNKNOWN
    assert execution.record.requires_reconciliation is True
    await asyncio.wait_for(completed.wait(), timeout=0.4)
    _assert_one_trace(executor, "call_1", "timeout_outcome_unknown")


@pytest.mark.anyio
async def test_remote_interrupt_records_unknown_without_cancelling_remote_work() -> None:
    started = asyncio.Event()
    completed = asyncio.Event()

    async def remote_runner(_arguments: Mapping[str, Any]) -> object:
        started.set()
        await asyncio.sleep(0.05)
        completed.set()
        return {"value": "remote-finished"}

    tool = _tool(
        run=remote_runner,
        cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT,
        timeout_seconds=1.0,
        idempotent=False,
    )
    executor = ToolExecutor({"demo": tool})
    task = asyncio.create_task(executor.execute(_call(), context=_context()))
    await started.wait()

    task.cancel()
    execution = await task

    assert execution.result.error_code == "cancelled_outcome_unknown"
    assert execution.result.retryable is False
    assert execution.record is not None
    assert execution.record.status is ExecutionStatus.UNKNOWN
    assert execution.record.requires_reconciliation is True
    await asyncio.wait_for(completed.wait(), timeout=0.3)
    _assert_one_trace(executor, "call_1", "cancelled_outcome_unknown")


@pytest.mark.anyio
async def test_non_idempotent_unknown_outcome_requires_reconciliation_before_retry() -> None:
    runner_calls = 0

    async def remote_runner(_arguments: Mapping[str, Any]) -> object:
        nonlocal runner_calls
        runner_calls += 1
        await asyncio.sleep(0.1)
        return {"value": "late"}

    tool = _tool(
        run=remote_runner,
        cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT,
        timeout_seconds=0.01,
        idempotent=False,
    )
    executor = ToolExecutor({"demo": tool})
    call = _call()

    first = await executor.execute(call, context=_context())
    assert first.record is not None
    second = await executor.execute(
        call,
        context=_context(),
        record=first.record,
    )

    assert runner_calls == 1
    assert second.result.error_code == "tool_reconciliation_required"
    assert second.record == first.record
    traces = [trace for trace in executor.traces if trace.tool_call_id == "call_1"]
    assert len(traces) == 2
    assert [trace.error_code for trace in traces] == [
        "timeout_outcome_unknown",
        "tool_reconciliation_required",
    ]


@pytest.mark.anyio
async def test_non_idempotent_running_record_is_reconciled_before_retry() -> None:
    runner_calls = 0

    async def runner(_arguments: Mapping[str, Any]) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return {"value": "wrong"}

    tool = _tool(run=runner, idempotent=False)
    call = _call()
    record = replace(
        ToolExecutionRecord.prepare(call, tool),
        status=ExecutionStatus.RUNNING,
        attempt_count=1,
    )
    executor = ToolExecutor({"demo": tool})

    execution = await executor.execute(call, context=_context(), record=record)

    assert execution.result.error_code == "tool_reconciliation_required"
    assert execution.record is not None
    assert execution.record.status is ExecutionStatus.UNKNOWN
    assert execution.record.error_code == "interrupted_outcome_unknown"
    assert execution.record.requires_reconciliation is True
    assert runner_calls == 0
    _assert_one_trace(executor, "call_1", "tool_reconciliation_required")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "is_error"),
    [
        (ExecutionStatus.COMPLETED, False),
        (ExecutionStatus.FAILED, True),
    ],
)
async def test_reconciled_non_idempotent_outcome_is_never_replayed(
    status: ExecutionStatus,
    is_error: bool,
) -> None:
    runner_calls = 0

    async def runner(_arguments: Mapping[str, Any]) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return {"value": "must not run"}

    tool = _tool(run=runner, idempotent=False)
    call = _call()
    record = replace(
        ToolExecutionRecord.prepare(call, tool),
        status=status,
        attempt_count=1,
        error_code=(None if status is ExecutionStatus.COMPLETED else "operator_failed"),
    )

    execution = await ToolExecutor({"demo": tool}).execute(
        call,
        context=_context(),
        record=record,
    )

    assert runner_calls == 0
    assert execution.result.is_error is is_error
    assert execution.result.metadata["reconciled"] is True
    assert execution.record == record


def _concurrency_tools(
    *,
    same_target: bool,
    second_safe: bool,
    events: list[str],
    active: list[int],
    maximum: list[int],
) -> tuple[Tool, Tool]:
    def build(name: str, target: str, safe: bool) -> Tool:
        async def runner(_arguments: Mapping[str, Any]) -> object:
            events.append(f"start:{name}")
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
            await asyncio.sleep(0.03)
            active[0] -= 1
            events.append(f"end:{name}")
            return {"value": name}

        return _tool(
            name,
            run=runner,
            resolve_use=lambda _arguments: ResolvedToolUse(
                effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
                targets=(ToolTarget(kind="workspace_path", value=target),),
            ),
            static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            concurrency_safe=safe,
        )

    return (
        build("one", "/workspace/one.txt", True),
        build(
            "two",
            "/workspace/one.txt" if same_target else "/workspace/two.txt",
            second_safe,
        ),
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("same_target", "second_safe", "expected_parallelism"),
    [
        pytest.param(False, True, 2, id="safe-distinct-targets"),
        pytest.param(True, True, 1, id="conflicting-targets"),
        pytest.param(False, False, 1, id="unsafe-tool"),
    ],
)
async def test_batch_parallelism_requires_safe_non_conflicting_tools(
    same_target: bool,
    second_safe: bool,
    expected_parallelism: int,
) -> None:
    events: list[str] = []
    active = [0]
    maximum = [0]
    one, two = _concurrency_tools(
        same_target=same_target,
        second_safe=second_safe,
        events=events,
        active=active,
        maximum=maximum,
    )
    executor = ToolExecutor({"one": one, "two": two})
    calls = (
        _call("one", call_id="call_1"),
        _call("two", call_id="call_2"),
    )

    executions = await executor.execute_batch(
        calls,
        context=_context(
            workspace_root=Path("/workspace"),
            allow_write=True,
        ),
    )

    assert maximum[0] == expected_parallelism
    assert tuple(item.result.tool_call_id for item in executions) == (
        "call_1",
        "call_2",
    )
    if expected_parallelism == 1:
        assert events == ["start:one", "end:one", "start:two", "end:two"]
    assert len(executor.traces) == 2


@pytest.mark.anyio
async def test_batch_parallelism_respects_policy_limit() -> None:
    active = 0
    maximum = 0

    def build(name: str) -> Tool:
        async def runner(_arguments: Mapping[str, Any]) -> object:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            return {"value": name}

        return _tool(name, run=runner, concurrency_safe=True)

    tools = {name: build(name) for name in ("one", "two", "three")}
    executions = await ToolExecutor(tools).execute_batch(
        tuple(
            _call(name, call_id=f"call_{name}")
            for name in ("one", "two", "three")
        ),
        context=_context(max_parallel_calls=2),
    )

    assert maximum == 2
    assert tuple(item.result.tool_call_id for item in executions) == (
        "call_one",
        "call_two",
        "call_three",
    )


@pytest.mark.anyio
async def test_batch_serializes_unknown_read_target_against_a_write() -> None:
    events: list[str] = []
    active = 0
    maximum = 0

    def build(
        name: str,
        effects: frozenset[ToolEffect],
        targets: tuple[ToolTarget, ...],
    ) -> Tool:
        async def runner(_arguments: Mapping[str, Any]) -> object:
            nonlocal active, maximum
            events.append(f"start:{name}")
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            events.append(f"end:{name}")
            return {"value": name}

        return _tool(
            name,
            run=runner,
            static_effects=effects,
            resolve_use=lambda _arguments: ResolvedToolUse(
                effects=effects,
                targets=targets,
            ),
            concurrency_safe=True,
        )

    executor = ToolExecutor(
        {
            "read": build("read", frozenset({ToolEffect.READ_WORKSPACE}), ()),
            "write": build(
                "write",
                frozenset({ToolEffect.WRITE_WORKSPACE}),
                (ToolTarget(kind="workspace_path", value="/workspace/out.txt"),),
            ),
        }
    )

    await executor.execute_batch(
        (
            _call("read", call_id="read_call"),
            _call("write", call_id="write_call"),
        ),
        context=_context(workspace_root=Path("/workspace"), allow_write=True),
    )

    assert maximum == 1
    assert events == ["start:read", "end:read", "start:write", "end:write"]


def test_task3_modules_do_not_import_legacy_execution_paths() -> None:
    root = Path(__file__).parents[2]
    forbidden = (
        "agent_runtime.tooling",
        "ToolExecutionService",
        "ApprovalPolicy",
        "agent_runtime.tools.registry",
    )

    for relative in (
        "agent_runtime/tools/permissions.py",
        "agent_runtime/tools/executor.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert not any(name in source for name in forbidden)

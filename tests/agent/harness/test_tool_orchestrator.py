from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.core.llm_registry import ResolvedModel
from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.model_definition import ModelCapabilities, RequestDefaultsDefinition
from agent_runtime.modeling.config import GenerationConfig
from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    GatewayHarnessModel,
    HarnessAgent,
    HarnessModelRequest,
    HarnessModelResponse,
    HarnessToolCall,
    PreparedModelCall,
    RolloutContextManager,
    RolloutStore,
    RuntimeComposition,
    ToolOrchestrator,
)
from agent_runtime.harness.tool_orchestrator import ToolApprovalRequiredError
from agent_runtime.harness.tool_router import DurableToolRouter
from agent_runtime.streaming.events import EventType, ItemDeltaKind, TurnItemKind
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolProgress,
    ToolProgressKind,
    ToolTarget,
    json_schema_input,
)


def _read_tool(
    store: RolloutStore | None = None,
    workspace: Path | None = None,
) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ("path",),
        "additionalProperties": False,
    }

    def run(arguments: Mapping[str, JsonValue]) -> dict[str, str]:
        if store is not None:
            [operation] = store.list_tool_operations()
            assert operation.status == "running"
            assert operation.claim_generation == 1
            assert operation.fencing_token is not None
            assert operation.claim_owner is not None
        return {"text": f"contents of {arguments['path']}"}

    return Tool(
        definition=ToolDefinition(
            name="read_file",
            description="Read one workspace file.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=run,
        normalize_output=lambda raw: NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data=raw),),
        ),
        output_schema=None,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(
                *(
                    ()
                    if workspace is None
                    else (
                        ToolTarget(
                            kind="workspace_path",
                            value=str(workspace / str(arguments["path"])),
                        ),
                    )
                ),
            ),
        ),
        execution_revision="read-file-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4_096,
    )


def test_read_only_tool_operation_is_durable_before_runner_io(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="read a file",
            binding_manifest={"model_alias": "test-model"},
        )
        registry = ToolRegistry()
        registry.register(_read_tool(store, workspace))
        orchestrator = ToolOrchestrator(
            store=store,
            tools=registry.freeze(),
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
            ),
        )
        call = ToolCall(
            tool_call_id="call-1",
            tool_name="read_file",
            arguments={"path": "README.md"},
            origin=ToolCallOrigin(
                request_id="request-1",
                toolset_revision="toolset-v1",
                exposed_tool_names=("read_file",),
            ),
        )

        result = asyncio.run(orchestrator.execute(turn_id=turn.turn_id, call=call))

        assert result.is_error is False
        [operation] = store.list_tool_operations(turn.turn_id)
        assert operation.status == "succeeded"
        assert operation.tool_call_id == "call-1"
        assert operation.execution_revision == "read-file-v1"
        assert operation.effects == ("read_workspace",)
        assert operation.resources == (
            {
                "kind": "filesystem",
                "identity": str((workspace / "README.md").resolve()),
                "access": "read",
            },
        )
        assert [item.kind for item in store.list_items(turn.turn_id)] == [
            "user_message",
            "tool_call",
            "tool_result",
        ]
        assert store.verify().valid is True

        with pytest.raises(ValueError, match="unsupported"):
            store.record_tool_execution_state(
                turn_id=turn.turn_id,
                operation_id=operation.operation_id,
                tool_call_id=operation.tool_call_id,
                tool_name=operation.tool_name,
                arguments_digest=operation.arguments_digest,
                execution_revision=operation.execution_revision,
                idempotent=operation.idempotent,
                status="running",
                attempt_count=operation.attempt_count + 1,
                error_code=None,
                requires_reconciliation=False,
            )


@pytest.mark.anyio
async def test_public_tool_item_starts_only_after_approval_and_fenced_claim(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    base = _read_tool(workspace=workspace)
    tool = replace(
        base,
        static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(
                ToolTarget(
                    kind="workspace_path",
                    value=str(workspace / str(arguments["path"])),
                ),
            ),
        ),
    )
    dispatcher = TurnEventDispatcher(capacity=16)
    stream = dispatcher.subscribe_controlling()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="approve one write",
            binding_manifest={"model_alias": "test-model"},
        )
        orchestrator = ToolOrchestrator(
            store=store,
            tools={"read_file": tool},
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
                require_confirmation_for=frozenset({"read_file"}),
            ),
            event_dispatcher=dispatcher,
        )
        call = ToolCall(
            tool_call_id="call-approval-1",
            tool_name="read_file",
            arguments={"path": "approved.txt"},
            origin=ToolCallOrigin(
                request_id="request-approval-1",
                toolset_revision="toolset-v1",
                exposed_tool_names=("read_file",),
            ),
        )

        with pytest.raises(ToolApprovalRequiredError):
            await orchestrator.execute(turn_id=turn.turn_id, call=call)

        before_events = []
        while not stream.empty:
            before_events.append(stream.receive_nowait())
        assert not any(
            event.type is EventType.ITEM_STARTED for event in before_events
        )
        [before_approval] = store.list_tool_operations(turn.turn_id)
        assert before_approval.status == "awaiting_approval"
        assert before_approval.claim_generation == 0

        result = await orchestrator.resume_approval(
            turn_id=turn.turn_id,
            decision="approve",
        )
        public_events = []
        while not stream.empty:
            public_events.append(stream.receive_nowait())

        assert result.is_error is False
        tool_events = [
            event
            for event in public_events
            if event.item_kind is TurnItemKind.TOOL
        ]
        assert [event.type for event in tool_events] == [
            EventType.ITEM_STARTED,
            EventType.ITEM_COMPLETED,
        ]
        assert all(
            event.item_kind is TurnItemKind.TOOL for event in tool_events
        )
        assert tool_events[0].item_id == tool_events[1].item_id
        [claimed] = store.list_tool_operations(turn.turn_id)
        assert claimed.claim_generation == 1
        assert claimed.fencing_token is not None


@pytest.mark.anyio
async def test_harness_tool_progress_backpressures_runner_through_dispatcher(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    callback_returns = 0

    async def stream_runner(
        arguments: Mapping[str, JsonValue],
        sink,
    ) -> dict[str, str]:
        nonlocal callback_returns
        for content in ("first", "second"):
            await sink(
                ToolProgress(
                    kind=ToolProgressKind.PROGRESS,
                    content=f"{content}:{arguments['path']}",
                )
            )
            callback_returns += 1
        return {"text": "stream complete"}

    tool = replace(
        _read_tool(workspace=workspace),
        stream=stream_runner,
    )
    dispatcher = TurnEventDispatcher(capacity=1)
    stream = dispatcher.subscribe_controlling()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="stream tool progress",
            binding_manifest={"model_alias": "test-model"},
        )
        orchestrator = ToolOrchestrator(
            store=store,
            tools={"read_file": tool},
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
            ),
            event_dispatcher=dispatcher,
        )
        call = ToolCall(
            tool_call_id="call-progress-1",
            tool_name="read_file",
            arguments={"path": "README.md"},
            origin=ToolCallOrigin(
                request_id="request-progress-1",
                toolset_revision="toolset-v1",
                exposed_tool_names=("read_file",),
            ),
        )
        running = asyncio.create_task(
            orchestrator.execute(turn_id=turn.turn_id, call=call)
        )

        started = await asyncio.wait_for(stream.receive(), timeout=0.5)
        await asyncio.sleep(0.05)

        assert callback_returns == 1
        assert running.done() is False

        first = await asyncio.wait_for(stream.receive(), timeout=0.5)
        second = await asyncio.wait_for(stream.receive(), timeout=0.5)
        completed = await asyncio.wait_for(stream.receive(), timeout=0.5)
        result = await asyncio.wait_for(running, timeout=0.5)

        assert result.is_error is False
        assert [first.data["delta"], second.data["delta"]] == [
            "first:README.md",
            "second:README.md",
        ]
        assert {started.item_id, first.item_id, second.item_id, completed.item_id} == {
            started.item_id
        }
        assert completed.type is EventType.ITEM_COMPLETED


@pytest.mark.anyio
async def test_harness_command_item_receives_stdout_before_completion_and_distinct_stderr(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def stream_command(
        _arguments: Mapping[str, JsonValue],
        sink,
    ) -> dict[str, object]:
        await sink(ToolProgress(ToolProgressKind.STDOUT, "stdout-line\n"))
        await sink(ToolProgress(ToolProgressKind.STDERR, "stderr-line\n"))
        return {
            "stdout": "stdout-line\n",
            "stderr": "stderr-line\n",
            "exit_code": 0,
            "timed_out": False,
            "truncated": False,
            "duration_ms": 1.0,
        }

    base = _read_tool(workspace=workspace)
    command = replace(
        base,
        definition=replace(base.definition, name="run_command"),
        stream=stream_command,
    )
    dispatcher = TurnEventDispatcher(capacity=16)
    stream = dispatcher.subscribe_controlling()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="run one command",
            binding_manifest={"model_alias": "test-model"},
        )
        orchestrator = ToolOrchestrator(
            store=store,
            tools={"run_command": command},
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
            ),
            event_dispatcher=dispatcher,
        )
        result = await orchestrator.execute(
            turn_id=turn.turn_id,
            call=ToolCall(
                tool_call_id="call-command-1",
                tool_name="run_command",
                arguments={"path": "ignored"},
                origin=ToolCallOrigin(
                    request_id="request-command-1",
                    toolset_revision="toolset-v1",
                    exposed_tool_names=("run_command",),
                ),
            ),
        )
        public_events = []
        while not stream.empty:
            public_events.append(stream.receive_nowait())

        assert result.is_error is False
        assert [event.type for event in public_events] == [
            EventType.ITEM_STARTED,
            EventType.ITEM_DELTA,
            EventType.ITEM_DELTA,
            EventType.ITEM_COMPLETED,
        ]
        assert [event.delta_kind for event in public_events[1:3]] == [
            ItemDeltaKind.COMMAND_STDOUT,
            ItemDeltaKind.COMMAND_STDERR,
        ]
        assert [event.data["delta"] for event in public_events[1:3]] == [
            "stdout-line\n",
            "stderr-line\n",
        ]
        assert all(
            event.item_kind is TurnItemKind.COMMAND for event in public_events
        )
        assert len({event.item_id for event in public_events}) == 1


@pytest.mark.anyio
async def test_inspection_budget_forces_a_concrete_action_after_twelve_reads(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="fix the implementation",
            binding_manifest={
                "model_alias": "test-model",
                "completion_policy": {"require_workspace_change": True},
            },
        )
        tool = _read_tool(workspace=workspace)
        orchestrator = ToolOrchestrator(
            store=store,
            tools={"read_file": tool},
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
            ),
        )
        for index in range(12):
            result = await orchestrator.execute(
                turn_id=turn.turn_id,
                call=ToolCall(
                    tool_call_id=f"read-{index}",
                    tool_name="read_file",
                    arguments={"path": f"file-{index}.py"},
                    origin=ToolCallOrigin(
                        request_id=f"request-{index}",
                        toolset_revision="tools-v1",
                        exposed_tool_names=("read_file",),
                    ),
                ),
            )
            assert result.is_error is False

        blocked = await orchestrator.execute(
            turn_id=turn.turn_id,
            call=ToolCall(
                tool_call_id="read-13",
                tool_name="read_file",
                arguments={"path": "one-more.py"},
                origin=ToolCallOrigin(
                    request_id="request-13",
                    toolset_revision="tools-v1",
                    exposed_tool_names=("read_file",),
                ),
            ),
        )

        assert blocked.is_error is True
        assert blocked.error_code == "inspection_budget_exhausted"
        assert blocked.retryable is False
        assert "apply_patch" in (blocked.error_message or "")
        assert len(store.list_tool_operations(turn.turn_id)) == 12
        router = DurableToolRouter(
            store=store,
            tools={"read_file": tool},
            resident_names=("read_file",),
            discoverable_names=(),
        )
        assert router.select(turn_id=turn.turn_id, messages=()) == ()
        assert store.verify().valid is True


@pytest.mark.anyio
async def test_inspection_budget_counts_read_only_process_tools_by_effect(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="fix the implementation",
            binding_manifest={
                "model_alias": "test-model",
                "completion_policy": {"require_workspace_change": True},
            },
        )
        read_tool = _read_tool(workspace=workspace)
        process_tool = replace(
            read_tool,
            definition=ToolDefinition(
                name="workspace_query",
                description="Run a read-only process.",
                input_schema=read_tool.definition.input_schema,
            ),
            static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
            execution_revision="read-only-process-v1",
            idempotent=False,
            concurrency_safe=False,
        )
        tools = {"read_file": read_tool, "workspace_query": process_tool}
        orchestrator = ToolOrchestrator(
            store=store,
            tools=tools,
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
            ),
        )
        for index in range(10):
            result = await orchestrator.execute(
                turn_id=turn.turn_id,
                call=ToolCall(
                    tool_call_id=f"read-{index}",
                    tool_name="read_file",
                    arguments={"path": f"file-{index}.py"},
                    origin=ToolCallOrigin(
                        request_id=f"request-{index}",
                        toolset_revision="tools-v1",
                        exposed_tool_names=tuple(tools),
                    ),
                ),
            )
            assert result.is_error is False
        for index in range(2):
            result = await orchestrator.execute(
                turn_id=turn.turn_id,
                call=ToolCall(
                    tool_call_id=f"process-{index}",
                    tool_name="workspace_query",
                    arguments={"path": f"query-{index}"},
                    origin=ToolCallOrigin(
                        request_id=f"process-request-{index}",
                        toolset_revision="tools-v1",
                        exposed_tool_names=tuple(tools),
                    ),
                ),
            )
            assert result.is_error is False

        blocked = await orchestrator.execute(
            turn_id=turn.turn_id,
            call=ToolCall(
                tool_call_id="process-3",
                tool_name="workspace_query",
                arguments={"path": "one-more-query"},
                origin=ToolCallOrigin(
                    request_id="process-request-3",
                    toolset_revision="tools-v1",
                    exposed_tool_names=tuple(tools),
                ),
            ),
        )

        assert blocked.error_code == "inspection_budget_exhausted"
        assert len(store.list_tool_operations(turn.turn_id)) == 12
        router = DurableToolRouter(
            store=store,
            tools=tools,
            resident_names=tuple(tools),
            discoverable_names=(),
        )
        assert router.select(turn_id=turn.turn_id, messages=()) == ()


@pytest.mark.anyio
async def test_read_only_turn_does_not_enforce_delivery_inspection_budget(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="analyze the implementation",
            binding_manifest={
                "model_alias": "test-model",
                "completion_policy": {"require_workspace_change": False},
            },
        )
        tool = _read_tool(workspace=workspace)
        orchestrator = ToolOrchestrator(
            store=store,
            tools={"read_file": tool},
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
            ),
        )
        for index in range(13):
            result = await orchestrator.execute(
                turn_id=turn.turn_id,
                call=ToolCall(
                    tool_call_id=f"read-{index}",
                    tool_name="read_file",
                    arguments={"path": f"file-{index}.py"},
                    origin=ToolCallOrigin(
                        request_id=f"request-{index}",
                        toolset_revision="tools-v1",
                        exposed_tool_names=("read_file",),
                    ),
                ),
            )
            assert result.is_error is False

        assert len(store.list_tool_operations(turn.turn_id)) == 13
        router = DurableToolRouter(
            store=store,
            tools={"read_file": tool},
            resident_names=("read_file",),
            discoverable_names=(),
        )
        assert router.select(turn_id=turn.turn_id, messages=()) == (tool,)

def test_recovery_reuses_committed_tool_call_and_reenters_full_preflight(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    runner_calls: list[str] = []

    def run(arguments: Mapping[str, JsonValue]) -> dict[str, str]:
        runner_calls.append(str(arguments["path"]))
        return {"text": "recovered read"}

    tool = replace(_read_tool(workspace=workspace), run=run)
    registry = ToolRegistry()
    registry.register(tool)
    call = ToolCall(
        tool_call_id="call-preflight-crash-1",
        tool_name="read_file",
        arguments={"path": "README.md"},
        origin=ToolCallOrigin(
            request_id="request-1",
            toolset_revision="toolset-v1",
            exposed_tool_names=("read_file",),
        ),
    )
    with RolloutStore(database) as crashed_process:
        thread = crashed_process.create_thread(workspace=workspace)
        turn = crashed_process.start_turn(
            thread_id=thread.thread_id,
            user_message="read once after recovery",
            binding_manifest={"model_alias": "test-model"},
        )
        crashed_process.record_tool_call(
            turn_id=turn.turn_id,
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            arguments=call.arguments,
            origin={
                "request_id": call.origin.request_id,
                "toolset_revision": call.origin.toolset_revision,
                "exposed_tool_names": list(call.origin.exposed_tool_names),
            },
        )

    with RolloutStore(database) as recovered_process:
        result = asyncio.run(
            ToolOrchestrator(
                store=recovered_process,
                tools=registry.freeze(),
                execution_context=ToolExecutionContext(
                    workspace_root=workspace,
                    cwd=workspace,
                ),
            ).execute(turn_id=turn.turn_id, call=call)
        )

        assert result.is_error is False
        assert runner_calls == ["README.md"]
        assert len([item for item in recovered_process.list_items(turn.turn_id) if item.kind == "tool_call"]) == 1
        [operation] = recovered_process.list_tool_operations(turn.turn_id)
        assert operation.status == "succeeded"
        conflicting = ToolCall(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            arguments={"path": "different.txt"},
            origin=call.origin,
        )
        with pytest.raises(RuntimeError, match="conflicts with committed payload"):
            asyncio.run(
                ToolOrchestrator(
                    store=recovered_process,
                    tools=registry.freeze(),
                    execution_context=ToolExecutionContext(
                        workspace_root=workspace,
                        cwd=workspace,
                    ),
                ).execute(turn_id=turn.turn_id, call=conflicting)
            )
        assert runner_calls == ["README.md"]
        assert recovered_process.verify().valid is True


def test_normalization_failure_preserves_runner_success_and_never_replays_side_effect(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[str] = []

    def run(arguments: Mapping[str, JsonValue]) -> dict[str, str]:
        calls.append(str(arguments["path"]))
        return {"text": "side effect already happened"}

    def fail_normalization(_raw: object) -> NormalizedToolOutput:
        raise RuntimeError("injected normalization failure")

    tool = replace(
        _read_tool(workspace=workspace),
        run=run,
        normalize_output=fail_normalization,
        idempotent=False,
    )
    registry = ToolRegistry()
    registry.register(tool)
    call = ToolCall(
        tool_call_id="call-normalization-1",
        tool_name="read_file",
        arguments={"path": "README.md"},
        origin=ToolCallOrigin(
            request_id="request-1",
            toolset_revision="toolset-v1",
            exposed_tool_names=("read_file",),
        ),
    )
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="run once",
            binding_manifest={"model_alias": "test-model"},
        )
        orchestrator = ToolOrchestrator(
            store=store,
            tools=registry.freeze(),
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
            ),
        )

        result = asyncio.run(orchestrator.execute(turn_id=turn.turn_id, call=call))

        assert result.error_code == "normalization_failed"
        assert result.metadata["runner_completed"] is True
        assert result.metadata["result_processing_failed_stage"] == "normalization_failed"
        assert calls == ["README.md"]
        [operation] = store.list_tool_operations(turn.turn_id)
        assert operation.status == "failed"
        assert operation.result_item_id is not None

        with pytest.raises(RuntimeError, match="already has an operation"):
            asyncio.run(orchestrator.execute(turn_id=turn.turn_id, call=call))
        assert calls == ["README.md"]
        assert store.verify().valid is True

        with pytest.raises(RuntimeError, match="already has an operation"):
            store.record_tool_execution_state(
                turn_id=turn.turn_id,
                operation_id="op_duplicate",
                tool_call_id=operation.tool_call_id,
                tool_name=operation.tool_name,
                arguments_digest=operation.arguments_digest,
                execution_revision=operation.execution_revision,
                idempotent=operation.idempotent,
                status="prepared",
                attempt_count=0,
                error_code=None,
                requires_reconciliation=False,
            )


class ToolThenAnswerModel:
    def __init__(self) -> None:
        self.requests: list[HarnessModelRequest] = []

    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {"model_alias": "tool-model", "model_revision": "tool-model-v1"}

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        self.requests.append(request)
        digest = hashlib.sha256(f"step:{request.step}".encode()).hexdigest()
        toolset_revision = toolset_revision_for_tools(request.tools)
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash=toolset_revision,
            wire_hash=digest,
            request_ref={
                "request_id": f"{request.turn_id}:step:{request.step}",
                "toolset_revision": toolset_revision,
                "exposed_tool_names": [tool.definition.name for tool in request.tools],
            },
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        if len(self.requests) == 1:
            return HarnessModelResponse(
                text="",
                provider_response_id="response-tool-call",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="call-1",
                        name="read_file",
                        arguments={"path": "README.md"},
                    ),
                ),
            )
        return HarnessModelResponse(
            text="README was read",
            provider_response_id="response-final",
            usage={"input_tokens": 5, "output_tokens": 2},
        )


class AcceptToolAnswer:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        assert proposal.answer == "README was read"
        return CompletionDecision(action="accept", reason="tool answer is grounded")


def test_candidate_sdk_runs_model_tool_result_model_loop(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = ToolThenAnswerModel()
    registry = ToolRegistry()
    registry.register(_read_tool(workspace=workspace))

    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=model,
        completion_gate=AcceptToolAnswer(),
        tools=registry.freeze(),
    ) as runtime:
        result = HarnessAgent(runtime.thread_manager).run("read README")

        assert result.answer == "README was read"
        assert len(model.requests) == 2
        assert [message.role for message in model.requests[1].messages] == [
            "user",
            "assistant",
            "tool",
        ]
        [tool_message] = [message for message in model.requests[1].messages if message.role == "tool"]
        assert tool_message.tool_call_id == "call-1"
        assert runtime.store.list_tool_operations(result.turn_id)[0].status == "succeeded"
        tool_owner = runtime.store.list_tool_operations(result.turn_id)[0].claim_owner
        model_owners = {
            attempt.claim_owner
            for operation in runtime.store.list_model_operations(result.turn_id)
            for attempt in runtime.store.list_model_attempts(operation.operation_id)
        }
        assert model_owners == {tool_owner}
        assert runtime.store.read_turn(result.turn_id).status == "completed"
        assert runtime.store.verify().valid is True


def test_tool_context_pairing_and_provider_request_hash_survive_restart(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    registry = ToolRegistry()
    registry.register(_read_tool(workspace=workspace))
    with RuntimeComposition.open(
        database=database,
        workspace=workspace,
        model=ToolThenAnswerModel(),
        completion_gate=AcceptToolAnswer(),
        tools=registry.freeze(),
    ) as runtime:
        result = HarnessAgent(runtime.thread_manager).run("read README")
        messages_before = RolloutContextManager(runtime.store).build(result.turn_id)
        thread_id = result.thread_id

    provider_model = GatewayHarnessModel(
        model_alias="tool-model",
        resolved=ResolvedModel(
            generator=object(),
            gateway=object(),  # prepare-only fixture; no provider dispatch occurs
            provider="openai-compatible",
            model="provider-model",
            capabilities=ModelCapabilities(
                context_window_tokens=8_192,
                max_context_window_tokens=8_192,
                max_output_tokens=256,
                supports_native_tools=True,
                supports_structured_output=True,
            ),
            token_accounting=object(),  # prepare path tolerates unavailable accounting
            request_defaults=RequestDefaultsDefinition(temperature=0.0),
            generation_config=GenerationConfig(),
        ),
        instructions=("Answer from tool evidence.",),
    )
    prepared_before = provider_model.prepare(
        HarnessModelRequest(
            thread_id=thread_id,
            turn_id=result.turn_id,
            messages=messages_before,
            binding_manifest={"model_alias": "tool-model"},
        )
    )
    with RolloutStore(database) as reopened:
        messages_after = RolloutContextManager(reopened).build(result.turn_id)
        prepared_after = provider_model.prepare(
            HarnessModelRequest(
                thread_id=thread_id,
                turn_id=result.turn_id,
                messages=messages_after,
                binding_manifest={"model_alias": "tool-model"},
            )
        )

    assert messages_after == messages_before
    assert [message.role for message in messages_after] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages_after[1].tool_calls[0].id == "call-1"
    assert messages_after[2].tool_call_id == "call-1"
    assert prepared_after.request_hash == prepared_before.request_hash
    assert prepared_after.context_hash == prepared_before.context_hash
    assert prepared_after.wire_hash == prepared_before.wire_hash


@pytest.mark.anyio
async def test_remote_cancellation_pauses_for_reconciliation_and_cannot_redispatch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    started = asyncio.Event()
    completed = asyncio.Event()
    runner_calls = 0

    async def remote_runner(_arguments: Mapping[str, JsonValue]) -> dict[str, str]:
        nonlocal runner_calls
        runner_calls += 1
        started.set()
        await asyncio.sleep(0.05)
        completed.set()
        return {"text": "remote side effect completed"}

    tool = replace(
        _read_tool(workspace=workspace),
        run=remote_runner,
        idempotent=False,
        cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT,
        timeout_seconds=1.0,
    )
    call = ToolCall(
        tool_call_id="call-remote-1",
        tool_name="read_file",
        arguments={"path": "remote-resource"},
        origin=ToolCallOrigin(
            request_id="request-remote-1",
            toolset_revision="toolset-v1",
            exposed_tool_names=("read_file",),
        ),
    )
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="invoke remote operation once",
            binding_manifest={"model_alias": "test-model"},
        )
        orchestrator = ToolOrchestrator(
            store=store,
            tools={"read_file": tool},
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
            ),
        )
        task = asyncio.create_task(orchestrator.execute(turn_id=turn.turn_id, call=call))
        await started.wait()
        task.cancel()
        result = await task

        assert result.error_code == "cancelled_outcome_unknown"
        assert store.read_turn(turn.turn_id).status == "paused"
        [operation] = store.list_tool_operations(turn.turn_id)
        assert operation.status == "unknown"
        assert operation.requires_reconciliation is True
        assert operation.claim_owner is not None
        assert operation.fencing_token is not None
        assert operation.lease_expires_at is not None
        assert operation.result_item_id is not None
        [interaction] = store.list_interactions(turn.turn_id)
        assert interaction.kind == "tool_reconciliation"
        assert interaction.status == "pending"
        with pytest.raises(RuntimeError, match="not running"):
            await orchestrator.execute(turn_id=turn.turn_id, call=call)
        assert runner_calls == 1
        await asyncio.wait_for(completed.wait(), timeout=0.3)
        assert store.verify().valid is True

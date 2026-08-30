from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest

from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    HarnessAgent,
    HarnessModelRequest,
    HarnessModelResponse,
    PreparedModelCall,
    RolloutContextManager,
    RolloutStore,
    Session,
    StaticToolRouter,
    ThreadManager,
    ToolOrchestrator,
)
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
    ToolTarget,
    json_schema_input,
)


def _read_tool(calls: list[str], workspace: Path) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ("path",),
        "additionalProperties": False,
    }

    def run(arguments: Mapping[str, JsonValue]) -> dict[str, str]:
        path = str(arguments["path"])
        calls.append(path)
        return {"text": f"recovered contents of {path}"}

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
                ToolTarget(
                    kind="workspace_path",
                    value=str(workspace / str(arguments["path"])),
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


class AnswerAfterRecovery:
    def __init__(self) -> None:
        self.requests: list[HarnessModelRequest] = []

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        self.requests.append(request)
        digest = hashlib.sha256(f"recovery-step:{request.step}".encode()).hexdigest()
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash=digest,
            wire_hash=digest,
            request_ref={"request_id": f"recovery:{request.step}"},
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        return HarnessModelResponse(
            text="Recovered tool call was consumed exactly once.",
            provider_response_id="response-after-recovery",
            usage={"input_tokens": 5, "output_tokens": 3},
        )


class AcceptRecoveredAnswer:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        assert proposal.answer == "Recovered tool call was consumed exactly once."
        return CompletionDecision(action="accept", reason="recovery verified")


class FrozenBinding:
    def snapshot(self) -> dict[str, object]:
        return {"model_alias": "model-v1", "model_step_budget": 2}


class CrashBeforeToolResultStore(RolloutStore):
    def record_tool_result(self, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("injected crash before ToolResult commit")


@pytest.mark.parametrize("result_committed_before_crash", [False, True])
def test_fresh_runner_consumes_committed_tool_call_without_replaying_model(
    tmp_path: Path,
    result_committed_before_crash: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(_read_tool(calls, workspace))
    tools = registry.freeze()
    toolset_revision = toolset_revision_for_tools(tuple(tools.values()))

    with RolloutStore(database) as crashed_process:
        thread = crashed_process.create_thread(workspace=workspace)
        turn = crashed_process.start_turn(
            thread_id=thread.thread_id,
            user_message="read README",
            binding_manifest={"model_alias": "model-v1", "model_step_budget": 2},
        )
        operation = crashed_process.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="r" * 64,
            context_hash="c" * 64,
            tool_hash=toolset_revision,
            wire_hash="w" * 64,
            request_ref={
                "request_id": f"{turn.turn_id}:step:1",
                "toolset_revision": toolset_revision,
                "exposed_tool_names": ["read_file"],
            },
        )
        attempt = crashed_process.dispatch_model_attempt(operation.operation_id)
        assert crashed_process.complete_model_attempt(
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            text="",
            provider_response_id="response-with-tool-call",
            usage={"input_tokens": 3, "output_tokens": 1},
            tool_calls=(
                {
                    "id": "call-1",
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                },
            ),
        )
        assert crashed_process.list_tool_operations(turn.turn_id) == ()
        if result_committed_before_crash:
            asyncio.run(
                ToolOrchestrator(
                    store=crashed_process,
                    tools=tools,
                    execution_context=ToolExecutionContext(
                        workspace_root=workspace,
                        cwd=workspace,
                    ),
                ).execute(
                    turn_id=turn.turn_id,
                    call=ToolCall(
                        tool_call_id="call-1",
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                        origin=ToolCallOrigin(
                            request_id=f"{turn.turn_id}:step:1",
                            toolset_revision=toolset_revision,
                            exposed_tool_names=("read_file",),
                        ),
                    ),
                )
            )
            assert calls == ["README.md"]

    with RolloutStore(database) as recovered_process:
        model = AnswerAfterRecovery()
        session = Session(
            thread_id=turn.thread_id,
            store=recovered_process,
            model=model,
            context_manager=RolloutContextManager(recovered_process),
            completion_gate=AcceptRecoveredAnswer(),
            tool_router=StaticToolRouter(tools),
            tool_orchestrator=ToolOrchestrator(
                store=recovered_process,
                tools=tools,
                execution_context=ToolExecutionContext(
                    workspace_root=workspace,
                    cwd=workspace,
                ),
            ),
        )

        agent = HarnessAgent(
            ThreadManager(
                store=recovered_process,
                session_factory=lambda _thread_id: session,
                workspace=workspace,
                binding_provider=FrozenBinding(),
            )
        )
        result = agent.recover_committed_model_response(turn.turn_id)

        assert result.status == "completed"
        assert calls == ["README.md"], (
            recovered_process.list_tool_operations(turn.turn_id),
            recovered_process.list_items(turn.turn_id),
        )
        assert len(model.requests) == 1
        assert model.requests[0].step == 2
        assert [message.role for message in model.requests[0].messages] == [
            "user",
            "assistant",
            "tool",
        ]
        assert recovered_process.read_turn(turn.turn_id).status == "completed"
        assert recovered_process.verify().valid is True


def test_missing_tool_result_after_runner_success_pauses_for_reconciliation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(_read_tool(calls, workspace))
    tools = registry.freeze()
    toolset_revision = toolset_revision_for_tools(tuple(tools.values()))
    with CrashBeforeToolResultStore(database) as crashed_process:
        thread = crashed_process.create_thread(workspace=workspace)
        turn = crashed_process.start_turn(
            thread_id=thread.thread_id,
            user_message="read once",
            binding_manifest={"model_alias": "model-v1", "model_step_budget": 2},
        )
        model_operation = crashed_process.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="r" * 64,
            context_hash="c" * 64,
            tool_hash=toolset_revision,
            wire_hash="w" * 64,
            request_ref={
                "request_id": f"{turn.turn_id}:step:1",
                "toolset_revision": toolset_revision,
                "exposed_tool_names": ["read_file"],
            },
        )
        attempt = crashed_process.dispatch_model_attempt(model_operation.operation_id)
        assert crashed_process.complete_model_attempt(
            operation_id=model_operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            text="",
            provider_response_id="response-with-tool-call",
            usage={"input_tokens": 3, "output_tokens": 1},
            tool_calls=(
                {
                    "id": "call-1",
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                },
            ),
        )
        with pytest.raises(RuntimeError, match="injected crash"):
            asyncio.run(
                ToolOrchestrator(
                    store=crashed_process,
                    tools=tools,
                    execution_context=ToolExecutionContext(
                        workspace_root=workspace,
                        cwd=workspace,
                    ),
                ).execute(
                    turn_id=turn.turn_id,
                    call=ToolCall(
                        tool_call_id="call-1",
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                        origin=ToolCallOrigin(
                            request_id=f"{turn.turn_id}:step:1",
                            toolset_revision=toolset_revision,
                            exposed_tool_names=("read_file",),
                        ),
                    ),
                )
            )
        [rolled_back_outcome] = crashed_process.list_tool_operations(turn.turn_id)
        assert rolled_back_outcome.status == "running"
        assert rolled_back_outcome.result_item_id is None
        assert calls == ["README.md"]

    with RolloutStore(database) as recovered_process:
        [expired_claim] = recovered_process.list_tool_operations(turn.turn_id)
        assert expired_claim.lease_expires_at is not None
        recovered_process.expire_tool_operation_claim(
            operation_id=expired_claim.operation_id,
            now=expired_claim.lease_expires_at + 1.0,
        )
        assert recovered_process.read_turn(turn.turn_id).status == "paused"
        assert calls == ["README.md"]
        [unknown] = recovered_process.list_tool_operations(turn.turn_id)
        assert unknown.status == "unknown"
        assert unknown.requires_reconciliation is True
        [interaction] = recovered_process.list_interactions(turn.turn_id)
        assert interaction.kind == "tool_reconciliation"
        assert interaction.status == "pending"
        assert recovered_process.verify().valid is True

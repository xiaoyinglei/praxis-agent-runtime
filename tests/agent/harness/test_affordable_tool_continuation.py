"""Real file actions must continue while their individual requests are affordable."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from agent_runtime.core.messages import StopReason, ToolCall, ToolUseResult
from agent_runtime.harness import (
    GatewayHarnessModel,
    HarnessMessage,
    RolloutContextManager,
    RuntimeState,
    ToolOrchestrator,
    TurnExecutor,
)
from agent_runtime.modeling.contracts import ContextCapacityPolicy, LLMStageBudget, normalize_llm_usage
from agent_runtime.modeling.gateway import AgentModelResponse, model_request_input_text
from agent_runtime.modeling.openai_wire import serialize_openai_request
from agent_runtime.tools.builtins.filesystem import create_apply_patch_tool, create_read_file_tool
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.workspace import open_workspace
from tests.agent.harness.test_budget_pressure import _settle_prior_usage
from tests.agent.harness.test_compaction_consistency import Accept
from tests.agent.harness.test_model_adapter import CapturingGateway, CharacterAccounting, _resolved_model


class FileWorkflowGateway(CapturingGateway):
    def __init__(self, *, path, body, consume_ceiling=False):
        super().__init__()
        self.path, self.body = path, body
        self.consume_ceiling = consume_ceiling

    def effective_stage_budget(self, stage, *, kwargs=None):
        return LLMStageBudget(
            max_input_tokens=980_000,
            max_output_tokens=(kwargs or {}).get("max_tokens") or 65_536,
            safety_margin_tokens=0,
        )

    async def agenerate_model_request(self, **kwargs):
        request = kwargs["request"]
        self.requests.append(request)
        count = len(self.requests)
        if not request.tools:
            turn = ToolUseResult(
                text="Unable to perform the remaining file work.",
                stop_reason=StopReason.END_TURN,
                raw_stop_reason="stop",
            )
        elif count == 1:
            turn = ToolUseResult(
                tool_calls=[
                    ToolCall(
                        id="write",
                        name="apply_patch",
                        input={
                            "file_path": self.path,
                            "new_string": self.body,
                        },
                    )
                ],
                stop_reason=StopReason.TOOL_USE,
                raw_stop_reason="tool_calls",
            )
        elif count == 2:
            turn = ToolUseResult(
                tool_calls=[ToolCall(id="read", name="read_file", input={"path": self.path})],
                stop_reason=StopReason.TOOL_USE,
                raw_stop_reason="tool_calls",
            )
        else:
            observed = [
                json.loads(m.content) for m in request.messages if m.role == "tool" and m.tool_call_id == "read"
            ]
            assert observed and observed[-1]["structured_content"]["content"] == self.body
            turn = ToolUseResult(
                text="Created and inspected the requested file.",
                stop_reason=StopReason.END_TURN,
                raw_stop_reason="stop",
            )
        measured = len(
            model_request_input_text(
                request, provider=kwargs["provider"], supports_native_tools=kwargs["supports_native_tools"]
            )
        )
        return AgentModelResponse(
            turn=turn,
            usage=normalize_llm_usage(
                input_tokens=measured,
                output_tokens=request.settings.max_output_tokens if self.consume_ceiling else 50,
                input_tokens_include_cache=True,
                usage_source="provider",
            ),
            provider_wire_hash=serialize_openai_request(request).provider_wire_hash,
            serializer_revision="openai-wire-v1",
            wire_kind="openai-compatible",
        )


def file_runner(
    store, tmp_path, *, path="result.txt", body="verified\n", budget=900_000, prior_usage=0, consume_ceiling=False
):
    workspace = open_workspace(tmp_path)
    tools = (create_apply_patch_tool(workspace), create_read_file_tool(workspace))
    gateway = FileWorkflowGateway(path=path, body=body, consume_ceiling=consume_ceiling)
    model = GatewayHarnessModel(
        model_id="test-model",
        resolved=_resolved_model(
            gateway=gateway,
            token_accounting=CharacterAccounting(),
            context_window_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        instructions=("Complete the requested file operation and inspect its result.",),
    )

    class Router:
        def select(self, **_):
            return tools

    thread = store.create_thread(workspace=tmp_path)
    turn = store.start_turn(
        thread_id=thread.thread_id,
        user_message=f"Create and inspect {path}.\n" + "constraint: preserve public data.\n" * 1400,
        binding_manifest={
            "model_token_budget_total": budget,
            "model_step_budget": 8,
            "context_capacity_policy": ContextCapacityPolicy(max_messages=2000).model_dump(),
            "execution_context_policy": {"revision": "working-context-v1", "finalization": "budget-admission-v1"},
        },
    )
    if prior_usage:
        _settle_prior_usage(store, turn.turn_id, prior_usage)
    runner = TurnExecutor(
        thread_id=thread.thread_id,
        store=store,
        model=model,
        tool_router=Router(),
        context_manager=RolloutContextManager(store),
        completion_gate=Accept(),
        tool_orchestrator=ToolOrchestrator(
            store=store,
            tools={t.definition.name: t for t in tools},
            execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path, allow_write_tools=True),
        ),
    )
    return runner, turn, gateway


@pytest.mark.parametrize("path,body", [("settings.json", '{"enabled":true}\n'), ("notes.md", "# Verified note\n")])
@pytest.mark.parametrize("prior_usage", [0, 699_862])
def test_affordable_file_work_continues_with_measured_usage_and_frozen_replay(tmp_path, path, body, prior_usage):
    database = tmp_path / "rollout.db"
    with RuntimeState.open_sync(database) as store:
        runner, turn, gateway = file_runner(store, tmp_path, path=path, body=body, prior_usage=prior_usage)
        prepared = asyncio.run(
            runner._prepare_step(runner.capture_step_context(runner.restore_turn_context(turn.turn_id), step=1))
        )
        if prior_usage:
            assert prepared.resource_request.total_tokens * 2 > 900_000 - prior_usage - 8192
            assert prepared.resource_request.total_tokens < 900_000 - prior_usage - 8192
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert (tmp_path / path).is_file(), "affordable tools were withdrawn before executing the requested action"
        assert (tmp_path / path).read_text() == body
        assert len(gateway.requests) == 3
        operations = store.list_model_operations(turn.turn_id)[bool(prior_usage) :]
        assert not any(op.request_ref.get("budget_finalization") for op in operations)
        assert all(t.status == "succeeded" for t in store.list_tool_operations(turn.turn_id))
        assert store.read_budget_state(turn.turn_id).used.total_tokens <= 900_000
        assert store.verify().valid
        context = runner.restore_turn_context(turn.turn_id)
    with RuntimeState.open_sync(database) as reopened:
        runner._store = reopened
        runner._context_manager = RolloutContextManager(reopened)
        for op in operations:
            replay = asyncio.run(runner._prepare_step(runner.restore_step_context(context, op)))
            assert (replay.request_hash, replay.context_hash, replay.tool_hash, replay.wire_hash) == (
                op.request_hash,
                op.context_hash,
                op.tool_hash,
                op.wire_hash,
            )


def test_last_affordable_tool_result_survives_budget_stop_without_more_provider_io(tmp_path):
    with RuntimeState.open_sync(tmp_path / "rollout.db") as store:
        runner, turn, gateway = file_runner(store, tmp_path, budget=100_000, consume_ceiling=True)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert (tmp_path / "result.txt").read_text() == "verified\n"
        assert result.status == "failed"
        assert store.read_turn(turn.turn_id).terminal_reason_code == "model_token_budget_exhausted"
        assert len(gateway.requests) == 1
        assert len(store.list_tool_operations(turn.turn_id)) == 1
        assert store.list_tool_operations(turn.turn_id)[0].status == "succeeded"
        assert any(item.kind == "tool_result" for item in store.list_items(turn.turn_id))
        assert store.read_budget_state(turn.turn_id).used.total_tokens == 100_000
        assert store.verify().valid


def test_unaffordable_input_performs_no_provider_or_tool_io(tmp_path):
    with RuntimeState.open_sync(tmp_path / "rollout.db") as store:
        runner, turn, gateway = file_runner(store, tmp_path, budget=20_000)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert not gateway.requests
        assert not store.list_tool_operations(turn.turn_id)
        assert not (tmp_path / "result.txt").exists()
        assert store.read_budget_state(turn.turn_id).used.total_tokens == 0
        assert store.verify().valid


@pytest.mark.parametrize("revision", ["working-context-v1", "working-context-v2"])
def test_output_cap_accounts_for_changed_input_limits_without_provider_io(tmp_path, revision):
    with RuntimeState.open_sync(tmp_path / "run.db") as store:
        runner, turn, gateway = file_runner(store, tmp_path, budget=100_000)

        def effective_budget(stage, *, kwargs=None):
            output = (kwargs or {}).get("max_tokens") or 65_536
            return LLMStageBudget(
                max_input_tokens=1_048_576 - output - 512, max_output_tokens=output, safety_margin_tokens=512
            )

        gateway.effective_stage_budget = effective_budget
        step = runner.capture_step_context(runner.restore_turn_context(turn.turn_id), step=1, budget_pressure=True)
        step = replace(
            step,
            binding_manifest={
                **step.binding_manifest,
                "execution_context_policy": {
                    "revision": revision,
                    "cache_continuity": "append-v1",
                },
            },
        )
        first = runner._model.prepare(step.model_request())
        previous_limits = first.request_ref.get("runtime_input_update")
        if previous_limits is not None:
            step = replace(step, messages=(*step.messages, HarnessMessage(role="context", content=previous_limits)))
        prepared = asyncio.run(runner._prepare_step(step))
        assert prepared.resource_request.total_tokens <= 100_000
        assert 0 < prepared.resource_request.output_tokens < 65_536
        if revision == "working-context-v1":
            assert prepared.request_ref["runtime_input_update"] != previous_limits
        else:
            assert "runtime_input_update" not in prepared.request_ref
            assert prepared.dispatch_payload.request.messages == first.dispatch_payload.request.messages
        assert prepared.dispatch_payload.request.tools
        assert not gateway.requests
        assert store.read_budget_state(turn.turn_id).used.total_tokens == 0
        assert prepared.request_ref["step_snapshot"]["output_token_limit"] == prepared.resource_request.output_tokens
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash=prepared.request_hash,
            context_hash=prepared.context_hash,
            tool_hash=prepared.tool_hash,
            wire_hash=prepared.wire_hash,
            request_ref=prepared.request_ref,
        )
        context = runner.restore_turn_context(turn.turn_id)
    with RuntimeState.open_sync(tmp_path / "run.db") as reopened:
        runner._store = reopened
        runner._context_manager = RolloutContextManager(reopened)
        replay = asyncio.run(runner._prepare_step(runner.restore_step_context(context, operation)))
        assert (replay.request_hash, replay.context_hash, replay.tool_hash, replay.wire_hash) == (
            operation.request_hash,
            operation.context_hash,
            operation.tool_hash,
            operation.wire_hash,
        )
        assert not gateway.requests


def test_old_prepared_final_only_snapshot_keeps_exact_request_after_reopen(tmp_path):
    database = tmp_path / "rollout.db"
    with RuntimeState.open_sync(database) as store:
        runner, turn, gateway = file_runner(store, tmp_path)
        context = runner.restore_turn_context(turn.turn_id)
        step = runner.capture_step_context(context, step=1, budget_pressure=True)
        step = replace(
            step, tools=(), binding_manifest={**step.binding_manifest, "runtime_budget_phase": "final_answer"}
        )
        prepared = asyncio.run(runner._prepare_step(step))
        op = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash=prepared.request_hash,
            context_hash=prepared.context_hash,
            tool_hash=prepared.tool_hash,
            wire_hash=prepared.wire_hash,
            request_ref=prepared.request_ref,
        )
    with RuntimeState.open_sync(database) as store:
        runner._store = store
        runner._context_manager = RolloutContextManager(store)
        restored = runner.restore_step_context(context, op)
        replay = asyncio.run(runner._prepare_step(restored))
        assert restored.tools == ()
        assert restored.binding_manifest["runtime_budget_phase"] == "final_answer"
        assert replay.request_ref["budget_finalization"] is True
        assert (replay.request_hash, replay.context_hash, replay.tool_hash, replay.wire_hash) == (
            op.request_hash,
            op.context_hash,
            op.tool_hash,
            op.wire_hash,
        )
        assert not gateway.requests
        assert store.verify().valid

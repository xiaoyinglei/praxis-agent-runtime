"""Cumulative Turn capacity must trigger compaction before the model window fills."""

import asyncio
from dataclasses import replace

import pytest

from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, RuntimeState, ToolOrchestrator
from agent_runtime.harness.completion import DeliveryCompletionGate
from agent_runtime.harness.tool_router import StaticToolRouter
from agent_runtime.tools.permissions import ToolExecutionContext
from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
from tests.agent.harness.test_completion_gate import _commit_tool_result
from tests.agent.harness.test_context_capacity_policy import capacity_runner
from tests.agent.harness.test_tool_orchestrator import _read_tool


@pytest.mark.parametrize("task", ["preserve API constraints", "compare archived research evidence"])
def test_turn_budget_compacts_while_summary_and_continuation_still_fit(tmp_path, task):
    with RuntimeState.open_sync(tmp_path / "headroom.db") as store:
        runner, turn, gateway = capacity_runner(
            store, tmp_path, tokens=95_000, messages=2000, summary="Applied work; verification still required."
        )
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": task + " history" * 5500})
        original = runner._model.prepare(
            runner.capture_step_context(runner.restore_turn_context(turn.turn_id), step=1).model_request()
        )
        assert original.resource_request.input_tokens < 100_000  # far below the 1M model window
        assert original.resource_request.total_tokens * 2 > 95_000 - 8192

        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))

        assert result.status == "completed"
        compacted = [i for i in store.list_items(turn.turn_id) if i.kind == "context_compaction"]
        assert len(compacted) == 1
        diagnostics = compacted[0].payload["capacity_diagnostics"]
        assert diagnostics["trigger_reason"] == "turn_budget_pressure"
        assert diagnostics["after_input_tokens"] < original.resource_request.input_tokens
        operations = store.list_model_operations(turn.turn_id)
        assert [o.request_ref["purpose"] for o in operations] == ["context_summary", "agent_step"]
        assert operations[-1].request_ref["budget_pressure"] is True
        assert store.read_budget_state(turn.turn_id).limits.tokens == 95_000
        assert store.read_budget_state(turn.turn_id).used.total_tokens <= 95_000
        assert store.verify().valid


def test_ample_turn_budget_keeps_history_and_output_ceiling(tmp_path):
    with RuntimeState.open_sync(tmp_path / "ample.db") as store:
        runner, turn, gateway = capacity_runner(store, tmp_path, tokens=1_200_000, messages=2000)
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "history" * 5500})
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))
        assert len(gateway.requests) == 1


@pytest.mark.parametrize("check_passed", [True, False])
def test_long_context_compaction_keeps_post_change_gate_and_frozen_budget(tmp_path, check_passed):
    database = tmp_path / "delivery.db"
    with RuntimeState.open_sync(database) as store:
        runner, turn, gateway = capacity_runner(
            store, tmp_path, tokens=100_000, messages=2000, summary="Changed value.txt; retain recorded check outcome."
        )
        # The capacity helper starts a normal Turn; construct a delivery Turn
        # in the same thread so the production completion policy is frozen.
        store.fail_turn(turn_id=turn.turn_id, reason_code="fixture", message="fixture")
        turn = store.start_turn(
            thread_id=turn.thread_id,
            user_message="Keep the public async API unchanged.",
            binding_manifest={**turn.binding_manifest, "completion_policy": {"require_workspace_change": True}},
        )
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "history " * 5500})
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="write",
            tool_name="apply_patch",
            arguments={},
            result={"metadata": {"runtime_workspace_write": True, "workspace_tree_changed": True}},
            resources=({"kind": "filesystem", "access": "write", "identity": str(tmp_path / "value.txt")},),
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="check",
            tool_name="run_command",
            arguments={"command": "pytest -q"},
            result={"exit_code": 0 if check_passed else 1},
            operation_failed=not check_passed,
        )
        runner._completion_gate = DeliveryCompletionGate(store)
        context = runner.restore_turn_context(turn.turn_id)
        # Reopen from durable state, as approval recovery does; no budget reset.
    with RuntimeState.open_sync(database) as reopened:
        runner._store = reopened
        runner._context_manager = type(runner._context_manager)(reopened)
        runner._completion_gate = DeliveryCompletionGate(reopened)
        result = asyncio.run(runner.run_turn(context, start_step=1))
        assert (result.status == "completed") is check_passed
        if not check_passed:
            assert result.status == "failed" and result.answer is None
        assert reopened.read_budget_state(turn.turn_id).limits.tokens == 100_000
        assert reopened.read_budget_state(turn.turn_id).used.total_tokens <= 100_000
        operations = reopened.list_model_operations(turn.turn_id)
        assert any(op.request_ref["purpose"] == "context_summary" for op in operations)
        final_messages = operations[-1].request_ref["step_snapshot"]["messages"]
        assert any("Keep the public async API unchanged." in msg["content"] for msg in final_messages)
        assert reopened.verify().valid


def test_repeated_real_file_tool_results_keep_an_affordable_summary_escape_path(tmp_path):
    """Grow the real transcript and charge measured inputs, not constant fake usage."""
    for index in range(6):
        (tmp_path / f"{index}.txt").write_text(f"evidence {index}: " + "data " * 2000)
    with RuntimeState.open_sync(tmp_path / "loop.db") as store:
        runner, turn, _gateway = capacity_runner(
            store, tmp_path, tokens=220_000, messages=2000, summary="Retain objective and already-read evidence."
        )
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "prior work " * 1000})
        tool = replace(
            _read_tool(workspace=tmp_path),
            max_model_output_bytes=30_000,
            run=lambda args: {"text": (tmp_path / str(args["path"])).read_text()},
        )
        runner._tool_router = StaticToolRouter({"read_file": tool})
        runner._tool_orchestrator = ToolOrchestrator(
            store=store, tools={"read_file": tool}, execution_context=ToolExecutionContext(workspace_root=tmp_path)
        )
        calls = 0

        async def dispatch(prepared, **_kwargs):
            nonlocal calls
            usage = {"input_tokens": prepared.resource_request.input_tokens, "output_tokens": 50}
            if prepared.request_ref.get("purpose") == "context_summary":
                return HarnessModelResponse(
                    text="Preserve objective and evidence. Continue the remaining reads.",
                    usage=usage,
                    provider_response_id=None,
                )
            calls += 1
            return HarnessModelResponse(
                text="Read more evidence." if calls <= 6 else "Finished inspecting all evidence.",
                usage=usage,
                provider_response_id=None,
                tool_calls=(
                    HarnessToolCall(id=f"read-{calls}", name="read_file", arguments={"path": f"{calls - 1}.txt"}),
                )
                if calls <= 6
                else (),
            )

        runner._model.dispatch = dispatch
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed", store.read_turn(turn.turn_id).terminal_message
        assert calls == 7
        assert len(store.list_tool_operations(turn.turn_id)) == 6
        assert all(op.status == "succeeded" for op in store.list_tool_operations(turn.turn_id))
        assert any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))
        assert store.read_budget_state(turn.turn_id).used.total_tokens <= 220_000
        assert store.verify().valid

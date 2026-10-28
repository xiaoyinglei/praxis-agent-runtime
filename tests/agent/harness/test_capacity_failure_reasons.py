"""Capacity failures retain their actual cause through durable and public results."""
import asyncio

import pytest

from agent_runtime import Agent
from agent_runtime.cli import _display_agent_result
from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, RuntimeState
from agent_runtime.harness.protocol import ModelContextOverflowError
from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
from tests.agent.harness.test_compaction_consistency import long_history
from tests.agent.harness.test_context_capacity_policy import capacity_runner


def test_cumulative_budget_exhaustion_is_not_a_context_window_failure(tmp_path):
    database = tmp_path / "budget.db"
    with RuntimeState.open_sync(database) as store:
        runner, turn, gateway = capacity_runner(store, tmp_path, tokens=1000, messages=2000)
        seed(store, turn_id=turn.turn_id, kind="user_message", payload={"text": "task " * 1000})
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        terminal = store.read_turn(turn.turn_id)
        assert terminal.terminal_reason_code == "model_token_budget_exhausted"
        assert "remaining" in terminal.terminal_message
        assert not gateway.requests
        assert store.verify().valid
    with RuntimeState.open_sync(database) as store:
        assert store.read_turn(turn.turn_id).terminal_reason_code == "model_token_budget_exhausted"


@pytest.mark.parametrize("failure", ["output_limit", "provider_overflow", "invalid", "step_budget"])
def test_required_summary_failure_keeps_its_own_cause(tmp_path, failure):
    expected = {
        "output_limit": "context_summary_incomplete",
        "provider_overflow": "summary_context_overflow",
        "invalid": "context_summary_invalid",
        "step_budget": "model_step_budget_exhausted",
    }[failure]
    with RuntimeState.open_sync(tmp_path / "summary.db") as store:
        runner, turn, _ = capacity_runner(store, tmp_path, tokens=200000, messages=3)
        long_history(store, turn)
        if failure == "step_budget":
            runner._max_steps = 1
            # Charge the existing frozen allowance with real committed summary outcomes.
            for index in range(16):
                op = store.prepare_model_operation(
                    turn_id=turn.turn_id, request_hash=f"request-{index}", context_hash="context",
                    tool_hash="tools", wire_hash=f"wire-{index}",
                    request_ref={"purpose": "context_summary", "request_id": f"old-{index}"},
                )
                attempt = store.dispatch_model_attempt(op.operation_id, worker_id="test", lease_seconds=60)
                store.complete_model_attempt(operation_id=op.operation_id, attempt_id=attempt.attempt_id,
                                             generation=attempt.generation, text="checkpoint", usage={},
                                             provider_response_id=None)
        else:
            async def dispatch(prepared, **kwargs):
                assert prepared.request_ref["purpose"] == "context_summary"
                if failure == "provider_overflow":
                    raise ModelContextOverflowError("Summary input rejected by provider")
                return HarnessModelResponse(
                    text="partial" if failure == "output_limit" else "", provider_response_id=None, usage={},
                    status="incomplete" if failure == "output_limit" else "completed",
                    incomplete_reason="max_output_tokens" if failure == "output_limit" else None,
                    tool_calls=((HarnessToolCall(id="invalid-tool", name="read_file", arguments={}),)
                                if failure == "invalid" else ()),
                )
            runner._model.dispatch = dispatch
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        terminal = store.read_turn(turn.turn_id)
        assert terminal.terminal_reason_code == expected
        if failure == "output_limit":
            assert "max_output_tokens" in terminal.terminal_message
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))
        assert store.verify().valid


def test_changing_compaction_source_is_not_reported_as_window_exhaustion(tmp_path, monkeypatch):
    from agent_runtime.harness.protocol import ContextSourceChangedError

    with RuntimeState.open_sync(tmp_path / "changing.db") as store:
        runner, turn, gateway = capacity_runner(store, tmp_path, messages=3)
        long_history(store, turn)

        def changed(*args, **kwargs):
            raise ContextSourceChangedError("New input changed the source during measurement")

        monkeypatch.setattr(runner._context_manager, "semantic_source", changed)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert store.read_turn(turn.turn_id).terminal_reason_code == "context_source_changed"
        assert not gateway.requests
        assert store.verify().valid


@pytest.mark.anyio
async def test_budget_reason_reaches_sdk_and_default_cli(tmp_path, capsys):
    from agent_runtime.harness import GatewayHarnessModel
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway, _resolved_model

    gateway = BudgetAwareCapturingGateway(max_input_tokens=100000)
    class BoundModel(GatewayHarnessModel):
        def snapshot(self, **kwargs):
            return {"model_id": "test-model", "model_revision": "v1"}

        def ensure_available(self, binding, **kwargs):
            assert binding["model_id"] == "test-model"

    model = BoundModel(model_id="test-model", resolved=_resolved_model(
        gateway=gateway, token_accounting=gateway.token_accounting, context_window_tokens=1048576,
    ), instructions=("Follow task",))
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "public.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run("task " * 2000, max_tokens_total=1000)
    assert result.status == "failed"
    assert result.stop_reason == "model_token_budget_exhausted"
    assert result.diagnostics[0].code == result.stop_reason
    assert "remaining" in result.diagnostics[0].message
    assert not gateway.requests
    _display_agent_result(result, verbose=False)
    output = capsys.readouterr().out
    assert "累计 Token" in output
    assert "预算不足" in output


@pytest.mark.parametrize("reason,label", [
    ("model_context_window_exceeded", "上下文窗口"),
    ("context_summary_incomplete", "摘要输出被截断"),
    ("summary_context_overflow", "摘要请求"),
    ("byte_guard", "字节"),
    ("message_guard", "消息数"),
])
def test_default_cli_distinguishes_capacity_causes(reason, label):
    from agent_runtime.cli import _failure_title

    assert label in _failure_title(reason)

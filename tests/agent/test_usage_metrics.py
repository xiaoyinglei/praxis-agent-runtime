from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent_runtime import Agent
from agent_runtime.cli import _display_agent_result
from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, RolloutStore
from agent_runtime.result import AgentUsage
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel


class ReadTwiceThenAnswer(PatchThenAnswerModel):
    async def dispatch(self, prepared, *, delta_sink=None):
        await asyncio.sleep(0.003)
        first = prepared.request_ref["request_id"].endswith(":step:1")
        return HarnessModelResponse(
            text="" if first else "Read the existing file and observed the missing file error.",
            provider_response_id="reads" if first else "answer",
            usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            tool_calls=(
                HarnessToolCall(id="read-ok", name="read_file", arguments={"path": "present.txt"}),
                HarnessToolCall(id="read-failed", name="read_file", arguments={"path": "missing.txt"}),
            ) if first else (),
        )


@pytest.mark.anyio
async def test_public_usage_counts_actual_local_tools_and_survives_reopen_and_rebuild(tmp_path, monkeypatch, capsys):
    (tmp_path / "present.txt").write_text("local evidence")
    database = tmp_path / "rollout.sqlite"
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: ReadTwiceThenAnswer())
    result = await agent.run("Read both files", require_workspace_change=False)
    assert result.status == "done", (result.stop_reason, result.diagnostics)
    assert [call.is_error for call in result.tool_calls] == [False, True]
    assert result.usage.native_calls == result.usage.tool_calls == 2
    assert result.usage.native_errors == 1
    assert result.usage.tool_latency_ms > 0
    assert result.usage.native_latency_ms_total == result.usage.tool_latency_ms
    assert all(call.latency_ms is not None and call.latency_ms > 0 for call in result.tool_calls)
    assert result.usage.model_calls == 2
    assert result.usage.model_latency_ms > 0
    assert result.usage.startup_ms is None
    _display_agent_result(result, verbose=True)
    output = capsys.readouterr().out
    assert "native=2(1err/" in output
    assert "startup=unknown" in output

    with RolloutStore(database) as store:
        budget = store.read_budget_state(result.turn_id)
        assert budget.used.tool_calls == 2
        assert budget.used.tool_ms == round(result.usage.tool_latency_ms)
        assert budget.used.model_ms > 0
        assert store.verify().valid
        store.rebuild_projections()
        assert store.read_budget_state(result.turn_id) == budget
        assert store.verify().valid
    fresh_agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    assert await fresh_agent.read_result(result.turn_id) == result


@pytest.mark.anyio
async def test_old_turn_without_observed_metrics_keeps_unknown_timing(tmp_path, capsys):
    database = tmp_path / "old.sqlite"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="old", binding_manifest={})
        store.complete_turn(turn_id=turn.turn_id, answer="old answer")
    result = await Agent(workspace_path=tmp_path, checkpoint_db=database).read_result(turn.turn_id)
    assert result.usage.latency_ms is None
    assert result.usage.startup_ms is None
    _display_agent_result(replace(result, usage=AgentUsage()), verbose=True)
    output = capsys.readouterr().out
    assert "total=unknown" in output
    assert "build=unknown" in output
    assert "finalize=unknown" in output


@pytest.mark.anyio
async def test_public_approval_recovery_counts_only_the_one_executed_call(tmp_path, monkeypatch):
    target = tmp_path / "value.txt"
    target.write_text("before")
    database = tmp_path / "approval.sqlite"
    options = dict(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    first = Agent(**options)
    monkeypatch.setattr(first, "_model_factory", lambda: PatchThenAnswerModel())
    paused = await first.run("change value.txt", require_workspace_change=False)
    assert paused.status == "paused"
    assert paused.usage.tool_calls == paused.usage.native_calls == 0
    with RolloutStore(database) as store:
        assert store.read_budget_state(paused.turn_id).used.tool_calls == 0

    recovered = Agent(**options)
    monkeypatch.setattr(recovered, "_model_factory", lambda: PatchThenAnswerModel())
    result = await recovered.resume(paused.turn_id, "allow_once")
    assert result.status == "done"
    assert target.read_text() == "after"
    assert result.usage.tool_calls == result.usage.native_calls == 1
    assert result.usage.native_errors == 0
    assert result.usage.total_tokens == 8
    assert result.usage.tool_latency_ms > 0
    with RolloutStore(database) as store:
        budget = store.read_budget_state(result.turn_id)
        assert budget.used.tool_calls == 1
        assert store.verify().valid
        store.rebuild_projections()
        assert store.read_budget_state(result.turn_id) == budget
    assert await recovered.read_result(result.turn_id) == result


@pytest.mark.anyio
async def test_reconciled_unknown_execution_does_not_count_the_replacement_result_as_a_new_call(tmp_path):
    from agent_runtime.harness import ToolOrchestrator, ToolReconciliationOutcome
    from agent_runtime.harness.tool_orchestrator import _tool_result_payload
    from agent_runtime.tools.executor import ToolExecutionRecord
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin, ToolResult
    from tests.agent.harness.test_tool_orchestrator import _read_tool

    database = tmp_path / "unknown.sqlite"
    tool = _read_tool(workspace=tmp_path)
    call = ToolCall(
        tool_call_id="unknown-read", tool_name="read_file", arguments={"path": "present.txt"},
        origin=ToolCallOrigin(request_id="request", toolset_revision="tools", exposed_tool_names=("read_file",)),
    )
    record = ToolExecutionRecord.prepare(call, tool)
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="read", binding_manifest={})
        store.record_tool_call(turn_id=turn.turn_id, tool_call_id=call.tool_call_id, tool_name=call.tool_name,
                               arguments=call.arguments, origin={"request_id": "request"})
        for status in ("prepared", "ready"):
            store.record_tool_execution_state(
                turn_id=turn.turn_id, operation_id=record.operation_id, tool_call_id=call.tool_call_id,
                tool_name=call.tool_name, arguments_digest=record.arguments_digest,
                execution_revision=tool.execution_revision, idempotent=True, status=status,
                attempt_count=0, error_code=None, requires_reconciliation=False,
            )
        claim = store.claim_tool_operation(
            operation_id=record.operation_id, worker_id="worker", lease_seconds=30, now=1000,
        )
        unknown = ToolResult(tool_call_id=call.tool_call_id, tool_name=call.tool_name, is_error=True,
                             error_code="timeout_outcome_unknown", error_message="waiting interrupted")
        assert store.commit_tool_operation_result(
            turn_id=turn.turn_id, operation_id=record.operation_id, claim_generation=claim.claim_generation,
            fencing_token=claim.fencing_token, status="unknown", attempt_count=1,
            error_code=unknown.error_code, requires_reconciliation=True,
            duration_ms=7.0, result=_tool_result_payload(unknown), now=1001,
        )
        before = store.read_budget_state(turn.turn_id)
        assert before.used.tool_calls == 1
        assert before.used.tool_ms == 7

    with RolloutStore(database) as store:
        orchestrator = ToolOrchestrator(store=store, tools={"read_file": tool},
                                        execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path))
        orchestrator.reconcile(turn_id=turn.turn_id, outcome=ToolReconciliationOutcome(
            status="succeeded", result=ToolResult(tool_call_id=call.tool_call_id, tool_name=call.tool_name,
                                                  metadata={"reconciler_revision": "checked-v1"}),
        ))
        after = store.read_budget_state(turn.turn_id)
        assert after.used == before.used
        store.rebuild_projections()
        assert store.read_budget_state(turn.turn_id).used == before.used
        assert store.verify().valid
    result = await Agent(workspace_path=tmp_path, checkpoint_db=database).read_result(turn.turn_id)
    assert result.usage.native_calls == result.usage.tool_calls == 1
    assert result.usage.native_errors == 1
    assert result.usage.tool_latency_ms == 7.0


@pytest.mark.anyio
async def test_mcp_adapter_observed_success_and_failure_project_separate_usage(tmp_path):
    from agent_runtime.harness import ToolOrchestrator
    from agent_runtime.tools.integrations.mcp import MCPToolDescriptor, create_mcp_tools
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin

    async def caller(server, name, arguments):
        await asyncio.sleep(0.002)
        return {"content": [{"type": "text", "text": "result"}], "isError": arguments["fail"]}

    [tool] = create_mcp_tools((MCPToolDescriptor(
        server_name="local", tool_name="echo", description="Return a local MCP response",
        input_schema={"type": "object", "properties": {"fail": {"type": "boolean"}}, "required": ["fail"]},
        read_only_hint=True,
    ),), caller)
    database = tmp_path / "mcp.sqlite"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="MCP calls", binding_manifest={})
        orchestrator = ToolOrchestrator(
            store=store, tools={tool.definition.name: tool},
            execution_context=ToolExecutionContext(
                workspace_root=tmp_path, cwd=tmp_path,
                approved_tool_call_ids=frozenset({"success", "failure"}),
            ),
        )
        for call_id, fail in (("success", False), ("failure", True)):
            await orchestrator.execute(turn_id=turn.turn_id, call=ToolCall(
                tool_call_id=call_id, tool_name=tool.definition.name, arguments={"fail": fail},
                origin=ToolCallOrigin(request_id="request", toolset_revision="tools",
                                      exposed_tool_names=(tool.definition.name,)),
            ))
        store.complete_turn(turn_id=turn.turn_id, answer="done")
        assert store.read_budget_state(turn.turn_id).used.tool_calls == 2
        assert store.verify().valid
    result = await Agent(workspace_path=tmp_path, checkpoint_db=database).read_result(turn.turn_id)
    assert result.usage.mcp_calls == result.usage.tool_calls == 2
    assert result.usage.mcp_errors == 1
    assert result.usage.native_calls == 0
    assert result.usage.native_latency_ms_total == 0
    assert result.usage.mcp_latency_ms_total == result.usage.tool_latency_ms
    assert result.usage.tool_latency_ms > 0

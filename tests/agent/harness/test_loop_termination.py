"""Termination is a durable protocol, not a judgment of task prose."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from agent_runtime import Agent, cli
from agent_runtime.harness import (
    CompletionDecision,
    HarnessModelResponse,
    ModelToolCall,
    RuntimeState,
    Session,
    TurnExecutor,
)
from tests.agent.harness.test_tool_orchestrator import ToolThenAnswerModel


class LoopModel(ToolThenAnswerModel):
    def __init__(self, *, invalid_until=0, read_until=0, tool_steps=()):
        super().__init__()
        self.invalid_until = invalid_until
        self.read_until = read_until
        self.tool_steps = tool_steps

    async def dispatch(self, prepared, **kwargs):
        step = self.requests[-1].step
        calls = ()
        if step <= self.invalid_until or step <= self.read_until or step in self.tool_steps:
            calls = (ModelToolCall(id=f"read-{step}", name="read_file", arguments=(
                {} if step <= self.invalid_until else {"path": "source.txt"}
            )),)
        return HarnessModelResponse.from_parts(text="" if calls else f"Answer {step}", provider_response_id=None,
                                    usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
                                    tool_calls=calls)


def agent_for(tmp_path, model):
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    return agent


class Crash(BaseException):
    pass


@pytest.mark.anyio
async def test_recovery_allowance_is_independent_of_feedback_wording(tmp_path, monkeypatch):
    original = RuntimeState.record_completion_feedback

    def reword(store, **kwargs):
        payload = json.loads(kwargs["reason"])
        payload["message"] = "A revised explanation of the same recovery decision."
        kwargs["reason"] = json.dumps(payload)
        return original(store, **kwargs)

    monkeypatch.setattr(RuntimeState, "record_completion_feedback", reword)
    result = await agent_for(tmp_path, LoopModel(invalid_until=4)).run("Inspect", max_steps=10)
    assert result.stop_reason == "repeated_tool_failure"
    assert result.usage.model_calls == 4


@pytest.mark.anyio
@pytest.mark.parametrize("boundary", [None, "result", "feedback"])
async def test_repeated_failure_has_the_same_outcome_after_public_recovery(tmp_path, monkeypatch, boundary):
    model = LoopModel(invalid_until=4)
    agent = agent_for(tmp_path, model)
    original_check = TurnExecutor._repeated_tool_failure
    original_feedback = RuntimeState.record_completion_feedback

    def check(executor, turn_id):
        results = [i for i in executor._store.list_items(turn_id) if i.kind == "tool_result"]
        if boundary == "result" and len(results) == 4:
            raise Crash()
        return original_check(executor, turn_id)

    def feedback(store, **kwargs):
        item = original_feedback(store, **kwargs)
        if boundary == "feedback":
            raise Crash()
        return item

    monkeypatch.setattr(TurnExecutor, "_repeated_tool_failure", check)
    monkeypatch.setattr(RuntimeState, "record_completion_feedback", feedback)
    if boundary:
        with pytest.raises(Crash):
            await agent.run("Inspect source", max_steps=12, max_tokens_total=10_000)
        monkeypatch.setattr(TurnExecutor, "_repeated_tool_failure", original_check)
        monkeypatch.setattr(RuntimeState, "record_completion_feedback", original_feedback)
        async with (await RuntimeState.open(tmp_path / "run.db")) as store:
            turn_id = store.list_turns()[-1].turn_id
        result = await agent_for(tmp_path, model).resume(turn_id, "continue")
    else:
        result = await agent.run("Inspect source", max_steps=12, max_tokens_total=10_000)
    assert result.status == "failed"
    assert result.stop_reason == "repeated_tool_failure"
    assert result.usage.model_calls == 4
    assert result.usage.total_tokens == 28
    async with (await RuntimeState.open(tmp_path / "run.db")) as store:
        assert len(store.list_tool_operations(result.turn_id)) == 0  # validation failed before execution
        feedbacks = [i for i in store.list_items(result.turn_id) if i.kind == "completion_feedback"]
        assert len(feedbacks) == 1
        assert store.read_budget_state(result.turn_id).limits.tokens == 10_000
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("crash_after_feedback", [False, True])
async def test_one_batch_gets_one_recovery_grant_even_with_multiple_failures(
    tmp_path, monkeypatch, crash_after_feedback,
):
    class BatchModel(LoopModel):
        async def dispatch(self, prepared, **kwargs):
            response = await super().dispatch(prepared, **kwargs)
            if response.tool_calls:
                step = self.requests[-1].step
                return HarnessModelResponse.from_parts(text="", provider_response_id=None, usage=response.usage,
                    tool_calls=tuple(ModelToolCall(id=f"bad-{step}-{n}", name="read_file", arguments={})
                                     for n in range(3)))
            return response

    model = BatchModel(invalid_until=2)
    original = RuntimeState.record_completion_feedback

    def feedback(store, **kwargs):
        original(store, **kwargs)
        raise Crash()

    if crash_after_feedback:
        monkeypatch.setattr(RuntimeState, "record_completion_feedback", feedback)
        with pytest.raises(Crash):
            await agent_for(tmp_path, model).run("Inspect", max_steps=8)
        monkeypatch.setattr(RuntimeState, "record_completion_feedback", original)
        async with (await RuntimeState.open(tmp_path / "run.db")) as store:
            turn_id = store.list_turns()[-1].turn_id
        result = await agent_for(tmp_path, model).resume(turn_id, "continue")
    else:
        result = await agent_for(tmp_path, model).run("Inspect", max_steps=8)
    assert result.stop_reason == "repeated_tool_failure"
    assert result.usage.model_calls == 2
    assert len(result.tool_calls) == 6
    async with (await RuntimeState.open(tmp_path / "run.db")) as store:
        assert len([i for i in store.list_items(result.turn_id) if i.kind == "completion_feedback"]) == 1
        assert store.verify().valid


@pytest.mark.anyio
async def test_abrupt_process_exit_after_recovery_grant_preserves_the_stop(tmp_path):
    script = '''
import asyncio, os, sys
from pathlib import Path
from agent_runtime.harness import RuntimeState
from tests.agent.harness.test_loop_termination import LoopModel, agent_for
original = RuntimeState.record_completion_feedback
def exit_after_commit(store, **kwargs):
    original(store, **kwargs)
    os._exit(91)
RuntimeState.record_completion_feedback = exit_after_commit
asyncio.run(agent_for(Path(sys.argv[1]), LoopModel(invalid_until=4)).run(
    "Inspect source", max_steps=12, max_tokens_total=10000))
'''
    child = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", script, str(tmp_path)],
                                    capture_output=True, text=True, timeout=20)
    assert child.returncode == 91, child.stderr
    async with (await RuntimeState.open(tmp_path / "run.db")) as store:
        turn_id = store.list_turns()[-1].turn_id
        assert len(store.list_model_operations(turn_id)) == 3
    result = await agent_for(tmp_path, LoopModel(invalid_until=4)).resume(turn_id, "continue")
    assert result.stop_reason == "repeated_tool_failure"
    assert result.usage.model_calls == 4
    assert result.usage.total_tokens == 28
    async with (await RuntimeState.open(tmp_path / "run.db")) as store:
        assert store.read_budget_state(turn_id).limits.tokens == 10_000
        assert len([i for i in store.list_items(turn_id) if i.kind == "completion_feedback"]) == 1
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("boundary", [None, "proposal", "decision", "feedback", "limit_decision"])
async def test_stop_continuations_are_bounded_and_recovery_does_not_spend_them_twice(
    tmp_path, monkeypatch, boundary,
):
    model = LoopModel()
    agent = agent_for(tmp_path, model)
    methods = {name: getattr(RuntimeState, f"record_{name}")
               for name in ("final_proposal", "completion_decision", "completion_feedback")}
    selected = {"proposal": "final_proposal", "decision": "completion_decision",
                "feedback": "completion_feedback", "limit_decision": "completion_decision"}.get(boundary)
    if selected:
        def commit_then_crash(store, **kwargs):
            item = methods[selected](store, **kwargs)
            if len(store.list_model_operations(kwargs["turn_id"])) == (9 if boundary == "limit_decision" else 8):
                raise Crash()
            return item
        monkeypatch.setattr(RuntimeState, f"record_{selected}", commit_then_crash)
        with pytest.raises(Crash):
            await agent.run("Make the requested change", require_workspace_change=True,
                            max_steps=20, max_tokens_total=10_000)
        monkeypatch.setattr(RuntimeState, f"record_{selected}", methods[selected])
        async with (await RuntimeState.open(tmp_path / "run.db")) as store:
            turn_id = store.list_turns()[-1].turn_id
        result = await agent_for(tmp_path, model).resume(turn_id, "continue")
    else:
        result = await agent.run("Make the requested change", require_workspace_change=True,
                                 max_steps=20, max_tokens_total=10_000)
    assert result.status == "failed"
    assert result.stop_reason == "completion_continuation_limit"
    assert result.validation_status == "failed"
    assert result.validation_scope == "workspace_change_recorded"
    assert result.usage.model_calls == 9
    assert result.usage.total_tokens == 63
    async with (await RuntimeState.open(tmp_path / "run.db")) as store:
        items = store.list_items(result.turn_id)
        assert len([i for i in items if i.kind == "final_proposal"]) == 9
        assert len([i for i in items if i.kind == "completion_decision"]) == 9
        assert len([i for i in items if i.kind == "completion_feedback"]) == 8
        assert store.read_budget_state(result.turn_id).limits.tokens == 10_000
        assert store.verify().valid


@pytest.mark.anyio
async def test_a_tool_exchange_resets_stop_continuations_without_claiming_progress(tmp_path):
    (tmp_path / "source.txt").write_text("unchanged")
    model = LoopModel(tool_steps=(9,))
    result = await agent_for(tmp_path, model).run("Make a change", require_workspace_change=True, max_steps=25)
    assert result.stop_reason == "completion_continuation_limit"
    assert result.usage.model_calls == 18  # 8 rejections, a tool exchange, then 9 rejections
    assert len(result.tool_calls) == 1
    assert not result.tool_calls[0].is_error


@pytest.mark.anyio
async def test_incomplete_tool_json_does_not_reset_the_completion_allowance(tmp_path):
    class IncompleteModel(LoopModel):
        async def dispatch(self, prepared, **kwargs):
            response = await super().dispatch(prepared, **kwargs)
            if self.requests[-1].step == 9:
                return HarnessModelResponse.from_parts(text="", provider_response_id=None, usage=response.usage,
                    status="incomplete", incomplete_reason="max_output_tokens", tool_calls=(
                        ModelToolCall(id="partial", name="read_file", arguments={}),
                    ))
            return response

    result = await agent_for(tmp_path, IncompleteModel()).run("Make a change", require_workspace_change=True,
                                                           max_steps=25)
    assert result.stop_reason == "completion_continuation_limit"
    assert result.usage.model_calls == 10
    assert not result.tool_calls


@pytest.mark.anyio
async def test_successful_repeated_reads_remain_available(tmp_path):
    (tmp_path / "source.txt").write_text("unchanged")
    model = LoopModel(read_until=25)
    result = await agent_for(tmp_path, model).run("Investigate", max_steps=30)
    assert result.status == "done"
    assert result.validation_status == "not_evaluated"
    assert result.usage.model_calls == 26
    assert len(result.tool_calls) == 25
    assert all(not call.is_error for call in result.tool_calls)


@pytest.mark.anyio
async def test_recovery_consumes_a_real_committed_write_without_reexecution(tmp_path, monkeypatch):
    (tmp_path / "value.txt").write_text("before")

    class WriteModel(LoopModel):
        async def dispatch(self, prepared, **kwargs):
            response = await super().dispatch(prepared, **kwargs)
            if self.requests[-1].step == 1:
                return HarnessModelResponse.from_parts(
                    text="",
                    provider_response_id=None,
                    usage=response.usage,
                    tool_calls=(
                        ModelToolCall(
                            id="patch",
                            name="apply_patch",
                            arguments={
                                "file_path": "value.txt",
                                "old_string": "before",
                                "new_string": "after",
                            },
                        ),
                    ),
                )
            return response

    model = WriteModel()
    original = TurnExecutor._after_tool_batch

    async def crash_after_write(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(TurnExecutor, "_after_tool_batch", crash_after_write)
    with pytest.raises(Crash):
        await agent_for(tmp_path, model).run("Update value.txt", allow_write_tools=True,
                                             require_workspace_change=True, max_steps=4)
    assert (tmp_path / "value.txt").read_text() == "after"
    monkeypatch.setattr(TurnExecutor, "_after_tool_batch", original)
    async with (await RuntimeState.open(tmp_path / "run.db")) as store:
        turn_id = store.list_turns()[-1].turn_id
    result = await agent_for(tmp_path, model).resume(turn_id, "continue")
    assert result.status == "done"
    assert len(result.tool_calls) == 1
    assert not result.tool_calls[0].is_error
    async with (await RuntimeState.open(tmp_path / "run.db")) as store:
        [operation] = store.list_tool_operations(turn_id)
        assert operation.attempt_count == 1
        assert store.verify().valid


@pytest.mark.anyio
async def test_new_user_input_resets_continuations_and_supersedes_an_old_decision(tmp_path, monkeypatch):
    model = LoopModel()
    original = RuntimeState.record_completion_decision
    received = []

    def record(store, **kwargs):
        item = original(store, **kwargs)
        if len(store.list_model_operations(kwargs["turn_id"])) == 8 and not received:
            turn = store.read_turn(kwargs["turn_id"])
            with RuntimeState.open_sync(tmp_path / "run.db") as external:
                received.append(external.receive_turn_input(thread_id=turn.thread_id, turn_id=turn.turn_id,
                                                            text="Additional requirement"))
        return item

    monkeypatch.setattr(RuntimeState, "record_completion_decision", record)
    result = await agent_for(tmp_path, model).run("Make a change", require_workspace_change=True, max_steps=25)
    assert result.stop_reason == "completion_continuation_limit"
    assert result.usage.model_calls == 17
    assert "Additional requirement" in [m.content for m in model.requests[-1].messages if m.role == "user"]


@pytest.mark.anyio
async def test_acceptance_at_the_continuation_boundary_is_allowed(tmp_path):
    class Gate:
        count = 0

        def evaluate(self, proposal):
            self.count += 1
            return CompletionDecision("accept" if self.count == 9 else "continue", "explicit gate result")

    model = LoopModel()
    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=model,
                                 completion_gate=Gate(), max_steps=20, _background_memory=False) as session:
        result = await session.submit("Answer")
        assert result.status == "done"
        assert result.usage.model_calls == 9


@pytest.mark.anyio
async def test_legacy_turn_does_not_acquire_a_new_continuation_limit(tmp_path, monkeypatch):
    original = RuntimeState.start_turn

    def start(store, **kwargs):
        binding = dict(kwargs["binding_manifest"])
        binding.pop("completion_continuation_policy", None)
        return original(store, **{**kwargs, "binding_manifest": binding})

    monkeypatch.setattr(RuntimeState, "start_turn", start)
    result = await agent_for(tmp_path, LoopModel()).run("Make a change", require_workspace_change=True, max_steps=12)
    assert result.stop_reason == "model_step_budget_exhausted"
    assert result.usage.model_calls == 12


def test_cli_reports_a_failed_acceptance_when_continuation_is_exhausted(tmp_path, monkeypatch):
    model = LoopModel()
    original = cli._create_agent_facade

    def create(**kwargs):
        agent = original(**kwargs)
        agent._model_factory = lambda: model
        return agent

    monkeypatch.setattr(cli, "_create_agent_facade", create)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli.agent_app, ["run", "Make a change", "--require-workspace-change",
        "--max-steps", "20", "--disable-workspace-mcp", "--non-interactive", "--verbose",
        "--rollout-db", str(tmp_path / "cli.db")])
    assert result.exit_code == 1, result.output
    with RuntimeState.open_sync(tmp_path / "cli.db") as store:
        turn = store.list_turns()[-1]
        assert turn.terminal_reason_code == "completion_continuation_limit"
        assert len(store.list_model_operations(turn.turn_id)) == 9
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("confirmed", ["succeeded", "failed"])
async def test_failure_streak_uses_reconciled_outcomes_after_each_reopen(tmp_path, confirmed):
    from dataclasses import replace

    from agent_runtime.tools.tool import CancellationMode
    from tests.agent.harness.test_tool_orchestrator import _read_tool

    finished = asyncio.Queue()
    runs = []

    async def remote(args):
        runs.append(args["path"])
        await asyncio.sleep(0.01)
        await finished.put(True)
        return {"text": "original remote operation finished"}

    class Model(LoopModel):
        async def dispatch(self, prepared, **kwargs):
            step = self.requests[-1].step
            return HarnessModelResponse.from_parts(text="done" if step > 4 else "", provider_response_id=None, usage={},
                tool_calls=() if step > 4 else (ModelToolCall(
                    id=f"remote-{step}", name="read_file", arguments={"path": "same-resource"},
                ),))

    model = Model()
    tool = replace(_read_tool(workspace=tmp_path), run=remote, idempotent=False,
                   cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT, timeout_seconds=0.001)
    database = tmp_path / "reconciled.db"
    thread_id = None
    turn_id = None
    pauses = 0
    for _ in range(6):
        async with await Session.open(database=database, workspace=tmp_path, thread_id=thread_id, model=model,
                                      tools={"read_file": tool}) as session:
            thread_id = session.thread_id
            result = (await session.submit("observe external operations") if turn_id is None
                      else await session.resume(turn_id, action="continue"))
            turn_id = result.turn_id
            if result.status != "paused":
                operations = session.store.list_tool_operations(turn_id)
                assert all(op.status == confirmed for op in operations)
                feedback = [i for i in session.store.list_items(turn_id) if i.kind == "completion_feedback"]
                assert len(feedback) == (0 if confirmed == "succeeded" else 1)
                assert session.store.verify().valid
                break
            pauses += 1
            assert result.pause.kind == "tool_reconciliation"
            [unknown] = [op for op in session.store.list_tool_operations(turn_id) if op.status == "unknown"]
            await asyncio.wait_for(finished.get(), 1)
            session.store.resolve_tool_reconciliation(turn_id=turn_id, status=confirmed, result={
                "tool_call_id": unknown.tool_call_id, "tool_name": "read_file", "is_error": confirmed == "failed",
                "error_code": "observed_failure" if confirmed == "failed" else None,
                "error_message": "confirmed deterministic failure" if confirmed == "failed" else None,
                "retryable": False, "model_content": f"Independently confirmed {confirmed}",
                "content": [], "attachments": [],
            }, reconciler_revision="test-independent-observation-v1")
    else:
        raise AssertionError("reconciled execution failed to terminate")
    assert pauses == 4 and len(runs) == 4
    assert result.status == ("done" if confirmed == "succeeded" else "failed")
    assert result.stop_reason == ("completed" if confirmed == "succeeded" else "repeated_tool_failure")
    assert len(model.requests) == (5 if confirmed == "succeeded" else 4)


@pytest.mark.anyio
async def test_approval_denials_remain_failures_without_an_operation_result_pointer(tmp_path):
    from agent_runtime.tools.permissions import ToolExecutionContext
    from tests.agent.harness.test_tool_orchestrator import _read_tool

    model = LoopModel(read_until=4)
    async with await Session.open(database=tmp_path / "denials.db", workspace=tmp_path, model=model,
        tools={"read_file": _read_tool(workspace=tmp_path)}, tool_execution_context=ToolExecutionContext(
            workspace_root=tmp_path, cwd=tmp_path, require_confirmation_for=frozenset({"read_file"}),
        )) as session:
        result = await session.submit("inspect source")
        denials = 0
        while result.status == "paused":
            assert result.pause.kind == "tool_approval"
            denials += 1
            assert denials <= 4
            result = await session.resume(result.turn_id, action="deny")
        assert denials == 4
        assert result.status == "failed" and result.stop_reason == "repeated_tool_failure"
        operations = session.store.list_tool_operations(result.turn_id)
        assert len(operations) == 4
        assert all(op.status == "denied" and op.result_item_id is None for op in operations)
        results = [i.payload for i in session.store.list_items(result.turn_id) if i.kind == "tool_result"]
        assert len(results) == 4 and all(r["error_code"] == "tool_denied" for r in results)
        assert session.store.verify().valid

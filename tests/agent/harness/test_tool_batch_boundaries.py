"""Batch admission and feedback contracts, independent of task/tool names."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, Session
from agent_runtime.tools.tool import InterruptBehavior, ResolvedToolUse, ToolEffect, ToolTarget
from tests.agent.harness.test_parallel_tool_orchestrator import _setup
from tests.agent.harness.test_tool_orchestrator import AcceptToolAnswer, ToolThenAnswerModel, _read_tool


@pytest.mark.anyio
@pytest.mark.parametrize("same_target", [True, False])
async def test_write_observation_belongs_to_admitted_runner(tmp_path: Path, same_target):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "changed.txt"
    target.write_text("before")
    other = workspace / "unchanged.txt"
    other.write_text("untouched")
    written = asyncio.Event()

    async def run(args):
        if args["path"] == "0":
            target.write_text("after")
            written.set()
        else:
            await written.wait()
        return {"text": "ok"}

    store, turn, orchestrator, calls = _setup(
        tmp_path, run, context_changes={"allow_write_tools": True},
        tool_changes={
            "static_effects": frozenset({ToolEffect.WRITE_WORKSPACE}),
            "resolve_use": lambda args: ResolvedToolUse(
                effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
                targets=(ToolTarget(kind="workspace_path", value=str(
                    target if same_target or args["path"] == "0" else other)),),
            ),
        },
    )
    from agent_runtime.tools.permissions import ToolExecutionContext

    orchestrator.update_execution_context(ToolExecutionContext(
        workspace_root=workspace, cwd=workspace, allow_write_tools=True,
    ))
    with store:
        first, second = await orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls[:2])
        assert not first.is_error and not second.is_error
        assert first.metadata["workspace_tree_changed"] is True
        assert second.metadata["workspace_tree_changed"] is False
        assert second.metadata["runtime_workspace_file_changes"] == ()
        assert second.metadata["workspace_tree_before_sha256"] == first.metadata["workspace_tree_after_sha256"]
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("parallel,limit", [(True, 1), (True, 2), (False, 2)])
@pytest.mark.parametrize("cancel_count", [1, 2])
async def test_finish_current_cancellation_preserves_results_and_stops_admission(
    tmp_path: Path, parallel, limit, cancel_count,
):
    started = []
    ready = asyncio.Event()
    finish = asyncio.Event()
    expected = 2 if parallel and limit == 2 else 1

    async def run(args):
        started.append(args["path"])
        if len(started) == expected:
            ready.set()
        await finish.wait()
        return {"text": args["path"]}

    store, turn, orchestrator, calls = _setup(
        tmp_path, run, context_changes={"max_parallel_calls": limit},
        tool_changes={"interrupt_behavior": InterruptBehavior.FINISH_CURRENT, "concurrency_safe": parallel},
    )
    with store:
        task = asyncio.create_task(orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls))
        await asyncio.wait_for(ready.wait(), 1)
        for _ in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert started == [str(i) for i in range(expected)]
        operations = store.list_tool_operations(turn.turn_id)
        finished = [op for op in operations if op.status == "succeeded"]
        assert len(finished) == expected
        assert all(op.result_item_id for op in finished)
        assert all(op.attempt_count == 0 for op in operations if op not in finished)
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("stable_order", [False, True])
async def test_provider_group_uses_frozen_order_while_ledger_keeps_completion_order(
    tmp_path: Path, monkeypatch, stable_order,
):
    if not stable_order:
        from agent_runtime.harness.services import SessionBindingProvider

        original_snapshot = SessionBindingProvider.snapshot

        def legacy_snapshot(self, **kwargs):
            binding = original_snapshot(self, **kwargs)
            binding["execution_context_policy"].pop("tool_result_order", None)
            return binding

        monkeypatch.setattr(SessionBindingProvider, "snapshot", legacy_snapshot)

    second_done = asyncio.Event()

    async def run(args):
        if args["path"] == "0":
            await second_done.wait()
        else:
            second_done.set()
        return {"text": args["path"]}

    class Model(ToolThenAnswerModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            if len(self.requests) == 1:
                return HarnessModelResponse(text="", provider_response_id="batch", usage={}, tool_calls=(
                    HarnessToolCall(id="slow", name="read_file", arguments={"path": "0"}),
                    HarnessToolCall(id="fast", name="read_file", arguments={"path": "1"}),
                ))
            return await super().dispatch(prepared)

    model = Model()
    async with await Session.open(
        database=tmp_path / "session.db", workspace=tmp_path, model=model,
        completion_gate=AcceptToolAnswer(), tools={"read_file": replace(_read_tool(workspace=tmp_path), run=run)},
    ) as session:
        result = await session.submit("inspect independent records")
        assert result.status == "done"
        ledger_ids = [i.payload["tool_call_id"] for i in session.store.list_items(result.turn_id)
                      if i.kind == "tool_result"]
        assert ledger_ids == ["fast", "slow"]
        expected = ["slow", "fast"] if stable_order else ["fast", "slow"]
        assert [m.tool_call_id for m in model.requests[-1].messages if m.role == "tool"] == expected
        assert session.store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("parallel", [True, False])
async def test_cancellation_after_claim_before_runner_commits_known_cancelled_result(tmp_path: Path, parallel):
    from agent_runtime.streaming.events import EventType
    from agent_runtime.streaming.sink import TurnEventDispatcher

    claimed = asyncio.Event()
    started = []

    class Sink:
        async def emit(self, event):
            if event.type is EventType.ITEM_STARTED:
                claimed.set()
                await asyncio.Event().wait()

    async def run(args):
        started.append(args["path"])
        return {"text": "ok"}

    store, turn, orchestrator, calls = _setup(tmp_path, run, tool_changes={"concurrency_safe": parallel})
    dispatcher = TurnEventDispatcher()
    dispatcher.subscribe_controlling_sink(Sink())
    orchestrator.attach_event_dispatcher(dispatcher)
    with store:
        task = asyncio.create_task(orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls))
        await asyncio.wait_for(claimed.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert started == []
        attempted = [op for op in store.list_tool_operations(turn.turn_id) if op.attempt_count]
        assert attempted
        assert all(op.status == "failed" and op.error_code == "cancelled" and op.result_item_id for op in attempted)
        assert store.verify().valid
    dispatcher.close()


@pytest.mark.anyio
@pytest.mark.parametrize("approval", [False, True])
@pytest.mark.parametrize("finish_current", [False, True])
async def test_cancel_and_reopen_retains_results_without_repeating_approved_or_finished_tools(
    tmp_path: Path, approval, finish_current,
):
    from agent_runtime.tools.permissions import ToolExecutionContext

    started = []
    ready = asyncio.Event()
    finish = asyncio.Event()

    async def run(args):
        started.append(args["path"])
        if args["path"] == "0":
            ready.set()
            await finish.wait()
        return {"text": args["path"]}

    class Model(ToolThenAnswerModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            if self.requests[-1].step == 1:
                return HarnessModelResponse(text="", provider_response_id="batch", usage={}, tool_calls=tuple(
                    HarnessToolCall(id=f"call-{i}", name="read_file", arguments={"path": str(i)}) for i in range(3)
                ))
            return HarnessModelResponse(text="README was read", provider_response_id="done", usage={})

    database = tmp_path / "session.db"
    tool = replace(_read_tool(workspace=tmp_path), run=run, concurrency_safe=False,
                   interrupt_behavior=InterruptBehavior.FINISH_CURRENT if finish_current else InterruptBehavior.CANCEL)
    context = ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path,
        require_confirmation_for=frozenset({"read_file"}) if approval else frozenset(),
        approved_tool_call_ids=frozenset({"call-1", "call-2"}) if approval else frozenset())
    first_model = Model()
    async with await Session.open(database=database, workspace=tmp_path, model=first_model,
        completion_gate=AcceptToolAnswer(), tools={"read_file": tool}, tool_execution_context=context) as session:
        thread_id = session.thread_id
        if approval:
            paused = await session.submit("inspect independent records")
            assert paused.status == "paused" and not started
            task = asyncio.create_task(session.resume(paused.turn_id, action="approve"))
        else:
            task = asyncio.create_task(session.submit("inspect independent records"))
        await asyncio.wait_for(ready.wait(), 1)
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert started == ["0"]
        [operation] = [op for op in session.store.list_tool_operations() if op.attempt_count]
        turn_id = operation.turn_id
        assert operation.status == ("succeeded" if finish_current else "failed")
        assert operation.result_item_id is not None
        assert len(first_model.requests) == 1
        assert session.store.verify().valid

    model = Model()
    async with await Session.open(database=database, workspace=tmp_path, thread_id=thread_id, model=model,
        completion_gate=AcceptToolAnswer(), tools={"read_file": tool}, tool_execution_context=context) as reopened:
        result = await reopened.resume(turn_id, action="continue")
        assert result.status == "done"
        assert started == ["0", "1", "2"]
        assert len(model.requests) == 1 and model.requests[0].step == 2
        assert [m.tool_call_id for m in model.requests[0].messages if m.role == "tool"] == [
            "call-0", "call-1", "call-2",
        ]
        assert reopened.store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["arguments", "denied", "exception", "timeout"])
async def test_call_failure_does_not_cancel_independent_siblings(tmp_path: Path, failure):
    from agent_runtime.harness import ToolOrchestrator
    from agent_runtime.tools.permissions import ToolExecutionContext

    completed = []

    async def run(args):
        if args["path"] == "1":
            if failure == "exception":
                raise RuntimeError("injected runner failure")
            if failure == "timeout":
                await asyncio.Event().wait()
        completed.append(args["path"])
        return {"text": args["path"]}

    store, turn, original, calls = _setup(tmp_path, run)
    tool = replace(original._tools["read_file"], timeout_seconds=0.03)
    orchestrator = ToolOrchestrator(store=store, tools={"read_file": tool}, execution_context=ToolExecutionContext(
        workspace_root=tmp_path, cwd=tmp_path, max_parallel_calls=2,
        denied_tool_call_ids=frozenset({"call-1"}) if failure == "denied" else frozenset(),
    ))
    if failure == "arguments":
        calls = (calls[0], replace(calls[1], argument_error="bad JSON"), calls[2])
    expected = {"arguments": "invalid_arguments", "denied": "tool_denied", "exception": "runner_failed",
                "timeout": "timeout_cancelled"}[failure]
    with store:
        results = await orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls)
        assert [r.tool_call_id for r in results] == [c.tool_call_id for c in calls]
        assert [r.error_code for r in results] == [None, expected, None]
        assert completed == ["0", "2"]
        assert store.read_turn(turn.turn_id).status == "running"
        assert len([i for i in store.list_items(turn.turn_id) if i.kind == "tool_result"]) == 3
        assert store.verify().valid


@pytest.mark.anyio
async def test_cancel_during_completed_runner_output_collection_preserves_durable_result(tmp_path: Path, monkeypatch):
    import threading

    from agent_runtime.tools import executor as executor_module
    from agent_runtime.tools.permissions import ToolExecutionContext

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_finished = False
    collecting = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original_snapshot = executor_module.workspace_tree_sha256

    def snapshot(root):
        if runner_finished:
            loop.call_soon_threadsafe(collecting.set)
            assert release.wait(2)
        return original_snapshot(root)

    async def run(args):
        nonlocal runner_finished
        (workspace / "result.txt").write_text("done")
        runner_finished = True
        return {"text": "done"}

    monkeypatch.setattr(executor_module, "workspace_tree_sha256", snapshot)
    store, turn, orchestrator, calls = _setup(tmp_path, run, tool_changes={
        "static_effects": frozenset({ToolEffect.WRITE_WORKSPACE}),
        "resolve_use": lambda args: ResolvedToolUse(effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(ToolTarget(kind="workspace_path", value=str(workspace / "result.txt")),)),
    })
    orchestrator.update_execution_context(ToolExecutionContext(workspace_root=workspace, cwd=workspace,
                                                              allow_write_tools=True))
    with store:
        task = asyncio.create_task(orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls))
        try:
            await asyncio.wait_for(collecting.wait(), 1)
            task.cancel()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        [operation] = [op for op in store.list_tool_operations(turn.turn_id) if op.attempt_count]
        assert operation.status == "succeeded" and operation.result_item_id
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_before_trace", [False, True])
async def test_optional_trace_sink_cannot_hold_known_result_during_cancellation(cancel_before_trace):
    from agent_runtime.tools.executor import ToolExecutor
    from tests.agent.test_single_tool_executor import _call, _context, _tool

    runner_started = asyncio.Event()
    finish_runner = asyncio.Event()
    trace_started = asyncio.Event()
    release = asyncio.Event()
    accepted = []

    async def run(args):
        runner_started.set()
        await finish_runner.wait()
        return {"value": "done"}

    async def trace_sink(trace):
        trace_started.set()
        await release.wait()

    tool = _tool(run=run, interrupt_behavior=InterruptBehavior.FINISH_CURRENT)
    executor = ToolExecutor({"demo": tool}, trace_sink=trace_sink)
    task = asyncio.create_task(executor.execute_batch((_call(),), context=_context(), execution_sink=accepted.append))
    await asyncio.wait_for(runner_started.wait(), 1)
    if cancel_before_trace:
        task.cancel()
        finish_runner.set()
    else:
        finish_runner.set()
        await asyncio.wait_for(trace_started.wait(), 1)
        task.cancel()
    try:
        done, _ = await asyncio.wait({task}, timeout=0.1)
        assert task in done, "optional telemetry blocked cancellation"
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert trace_started.is_set() is not cancel_before_trace
    assert len(accepted) == 1 and not accepted[0].result.is_error


@pytest.mark.anyio
async def test_result_commit_fault_stops_batch_and_drains_started_sibling(tmp_path: Path, monkeypatch):
    from agent_runtime.harness.tool_orchestrator import ToolResultCommitError

    sibling_started = asyncio.Event()
    exited = []

    async def run(args):
        if args["path"] == "0":
            await sibling_started.wait()
            return {"text": "completed but commit will fail"}
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.append(args["path"])

    store, turn, orchestrator, calls = _setup(tmp_path, run, context_changes={"max_parallel_calls": 2})
    original_commit = store.commit_tool_operation_result

    def fail_one_commit(**kwargs):
        if store.read_tool_operation(kwargs["operation_id"]).tool_call_id == "call-0":
            raise OSError("injected storage failure")
        return original_commit(**kwargs)

    monkeypatch.setattr(store, "commit_tool_operation_result", fail_one_commit)
    with store:
        with pytest.raises(ToolResultCommitError):
            await orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls)
        operations = {op.tool_call_id: op for op in store.list_tool_operations(turn.turn_id)}
        assert operations["call-0"].status == "running" and operations["call-0"].result_item_id is None
        assert operations["call-1"].status == "failed" and operations["call-1"].result_item_id
        assert operations["call-2"].attempt_count == 0
        assert exited == ["1"]
        assert store.verify().valid


@pytest.mark.anyio
async def test_repeated_cancellation_drains_cooperative_runner_cleanup(tmp_path: Path):
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()
    started = []

    async def run(args):
        started.append(args["path"])
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()

    store, turn, orchestrator, calls = _setup(tmp_path, run, context_changes={"max_parallel_calls": 1})
    with store:
        task = asyncio.create_task(orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set() and started == ["0"]
        [operation] = [op for op in store.list_tool_operations(turn.turn_id) if op.attempt_count]
        assert operation.status == "failed" and operation.error_code == "cancelled" and operation.result_item_id
        assert store.verify().valid

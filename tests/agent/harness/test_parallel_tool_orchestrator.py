from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from test_tool_orchestrator import _read_tool

from agent_runtime.harness import RolloutStore, ToolOrchestrator
from agent_runtime.harness.tool_orchestrator import ToolApprovalRequiredError
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin


def _setup(tmp_path, run, *, context_changes=None, tool_changes=None):
    store = RolloutStore(tmp_path / "rollout.sqlite3")
    thread = store.create_thread(workspace=tmp_path)
    turn = store.start_turn(thread_id=thread.thread_id, user_message="read", binding_manifest={})
    tool = replace(_read_tool(workspace=tmp_path), run=run, timeout_seconds=2, **(tool_changes or {}))
    orchestrator = ToolOrchestrator(
        store=store,
        tools={"read_file": tool},
        execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path, **(context_changes or {})),
    )
    calls = tuple(
        ToolCall(
            tool_call_id=f"call-{i}",
            tool_name="read_file",
            arguments={"path": str(i)},
            origin=ToolCallOrigin(request_id="req", toolset_revision="v1", exposed_tool_names=("read_file",)),
        )
        for i in range(3)
    )
    return store, turn, orchestrator, calls


@pytest.mark.anyio
async def test_batch_really_overlaps_and_returns_ordered_durable_results(tmp_path: Path):
    started = set()
    barrier = asyncio.Event()
    active = 0
    maximum = 0

    async def run(args):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        started.add(args["path"])
        if len(started) == 2:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), timeout=1)
        active -= 1
        return {"text": args["path"]}

    store, turn, orchestrator, calls = _setup(tmp_path, run, context_changes={"max_parallel_calls": 2})
    with store:
        results = await orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls)
        assert maximum == 2
        assert [r.tool_call_id for r in results] == [c.tool_call_id for c in calls]
        assert all(op.status == "succeeded" and op.result_item_id for op in store.list_tool_operations(turn.turn_id))
        assert store.verify().valid


@pytest.mark.anyio
async def test_batch_approval_preflight_never_starts_sibling_runner(tmp_path: Path):
    started = []

    async def run(args):
        started.append(args["path"])
        return {"text": "ok"}

    store, turn, orchestrator, calls = _setup(
        tmp_path,
        run,
        context_changes={
            "require_confirmation_for": frozenset({"read_file"}),
            "approved_tool_call_ids": frozenset({"call-0"}),
        },
    )
    with store:
        with pytest.raises(ToolApprovalRequiredError):
            await orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls)
        assert started == []
        assert store.read_turn(turn.turn_id).status == "paused"
        assert len([i for i in store.list_items(turn.turn_id) if i.kind == "tool_call"]) == 3
        assert store.verify().valid


@pytest.mark.anyio
async def test_batch_cancellation_drains_and_commits_all_started_runners(tmp_path: Path):
    started = set()
    barrier = asyncio.Event()
    exited = set()

    async def run(args):
        started.add(args["path"])
        if len(started) == 2:
            barrier.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.add(args["path"])

    store, turn, orchestrator, calls = _setup(tmp_path, run, context_changes={"max_parallel_calls": 2})
    with store:
        task = asyncio.create_task(orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls))
        await asyncio.wait_for(barrier.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited == started == {"0", "1"}
        ops = store.list_tool_operations(turn.turn_id)
        assert [(op.status, op.error_code) for op in ops[:2]] == [("failed", "cancelled")] * 2
        assert all(op.result_item_id for op in ops[:2])
        assert store.verify().valid


@pytest.mark.anyio
async def test_batch_conflicting_targets_serialize_without_resource_busy(tmp_path: Path):
    from agent_runtime.tools.tool import ResolvedToolUse, ToolEffect, ToolTarget

    active = 0
    maximum = 0

    async def run(args):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        active -= 1
        return {"text": "ok"}

    store, turn, orchestrator, calls = _setup(
        tmp_path,
        run,
        context_changes={"allow_write_tools": True},
        tool_changes={
            "static_effects": frozenset({ToolEffect.WRITE_WORKSPACE}),
            "resolve_use": lambda _: ResolvedToolUse(
                effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
                targets=(ToolTarget(kind="workspace_path", value=str(tmp_path / "same.txt")),),
            ),
        },
    )
    with store:
        results = await orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls)
        assert maximum == 1
        assert all(not r.is_error for r in results)
        assert all(op.status == "succeeded" for op in store.list_tool_operations(turn.turn_id))
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("exclusive_writer", [False, True])
async def test_mixed_batch_runs_both_read_segments_in_parallel_around_write(tmp_path: Path, exclusive_writer):
    from agent_runtime.tools.tool import ResolvedToolUse, ToolEffect, ToolTarget

    target = tmp_path / "value.txt"
    target.write_text("before")
    active = set()
    entered = [set(), set()]
    barriers = [asyncio.Event(), asyncio.Event()]
    observed = {}

    async def run(args):
        call = int(args["path"])
        if call == 2:
            assert not active
            assert entered[0] == {0, 1}
            target.write_text("after")
        else:
            phase = int(call > 2)
            active.add(call)
            entered[phase].add(call)
            if len(entered[phase]) == 2:
                barriers[phase].set()
            try:
                await asyncio.wait_for(barriers[phase].wait(), 0.5)
                observed[call] = target.read_text()
            finally:
                active.remove(call)
        return {"text": target.read_text()}

    store, turn, orchestrator, calls = _setup(
        tmp_path, run,
        context_changes={"allow_write_tools": True, "max_parallel_calls": 2},
        tool_changes={"resolve_use": lambda args: ResolvedToolUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE if args["path"] == "2" else ToolEffect.READ_WORKSPACE}),
            targets=(ToolTarget(kind="workspace_path", value=str(target)),),
        )},
    )
    calls = tuple(replace(calls[0], tool_call_id=f"call-{i}", arguments={"path": str(i)}) for i in range(5))
    if exclusive_writer:
        read_tool = orchestrator._tools["read_file"]
        write_tool = replace(read_tool, definition=replace(read_tool.definition, name="write_value"),
                             concurrency_safe=False)
        orchestrator = ToolOrchestrator(
            store=store, tools={"read_file": read_tool, "write_value": write_tool},
            execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path,
                                                   allow_write_tools=True, max_parallel_calls=2),
        )
        origin = replace(calls[0].origin, exposed_tool_names=("read_file", "write_value"))
        calls = tuple(replace(call, origin=origin, tool_name="write_value" if i == 2 else "read_file")
                      for i, call in enumerate(calls))
    with store:
        results = await orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls)
        assert all(not result.is_error for result in results)
        assert observed == {0: "before", 1: "before", 3: "after", 4: "after"}
        assert [r.tool_call_id for r in results] == [c.tool_call_id for c in calls]
        assert store.verify().valid


@pytest.mark.anyio
async def test_batch_remote_unknown_pauses_before_starting_later_calls(tmp_path: Path):
    from agent_runtime.tools.tool import CancellationMode

    started = asyncio.Event()
    finish = asyncio.Event()
    paths = []

    async def run(args):
        paths.append(args["path"])
        started.set()
        await finish.wait()
        return {"text": "remote done"}

    store, turn, orchestrator, calls = _setup(
        tmp_path, run, tool_changes={"cancellation_mode": CancellationMode.REMOTE_BEST_EFFORT, "idempotent": False}
    )
    with store:
        task = asyncio.create_task(orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        results = await task
        assert paths == ["0"]
        assert len(results) == 1
        assert results[0].error_code == "cancelled_outcome_unknown"
        assert store.read_turn(turn.turn_id).status == "paused"
        [operation] = store.list_tool_operations(turn.turn_id)
        assert operation.status == "unknown" and operation.result_item_id
        assert store.verify().valid
        finish.set()
        await asyncio.sleep(0)


@pytest.mark.anyio
async def test_batch_inspection_budget_cannot_be_overspent(tmp_path: Path):
    count = 0

    async def run(args):
        nonlocal count
        count += 1
        return {"text": "ok"}

    store, turn, orchestrator, calls = _setup(tmp_path, run)
    with store:
        # A separate Turn freezes the actual workspace-change completion policy.
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="edit",
            binding_manifest={"completion_policy": {"require_workspace_change": True}},
        )
        calls = tuple(replace(calls[0], tool_call_id=f"read-{i}") for i in range(15))
        results = await orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls)
        assert count == 12
        assert sum(not result.is_error for result in results) == 12
        assert all(result.is_error for result in results[12:])
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("approval", [False, True])
async def test_actual_turn_batch_overlap_and_approval_resumes_every_sibling(tmp_path: Path, approval):
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, Session
    from tests.agent.harness.test_tool_orchestrator import AcceptToolAnswer, ToolThenAnswerModel

    started = []
    barrier = asyncio.Event()
    active = 0
    maximum = 0

    async def run(args):
        nonlocal active, maximum
        started.append(args["path"])
        active += 1
        maximum = max(maximum, active)
        if active == 2 or approval:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), 1)
        active -= 1
        return {"text": "ok"}

    class Model(ToolThenAnswerModel):
        async def dispatch(self, prepared):
            if len(self.requests) == 1:
                return HarnessModelResponse(
                    text="",
                    provider_response_id="batch",
                    usage={},
                    tool_calls=tuple(
                        HarnessToolCall(id=f"call-{i}", name="read_file", arguments={"path": str(i)}) for i in range(3)
                    ),
                )
            assert sorted(started) == ["0", "1", "2"]
            assert len([m for m in self.requests[-1].messages if m.role == "tool"]) == 3
            return await super().dispatch(prepared)

    tool = replace(_read_tool(workspace=tmp_path), run=run)
    context = ToolExecutionContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        max_parallel_calls=2,
        require_confirmation_for=frozenset({"read_file"}) if approval else frozenset(),
        approved_tool_call_ids=frozenset({"call-0", "call-2"}) if approval else frozenset(),
    )
    async with await Session.open(
        database=tmp_path / "session.db",
        workspace=tmp_path,
        model=Model(),
        completion_gate=AcceptToolAnswer(),
        tools={"read_file": tool},
        tool_execution_context=context,
    ) as runtime:
        result = await runtime.submit("read three independent files")
        if approval:
            assert result.status == "paused"
            assert started == []
            result = await runtime.resume(result.turn_id, action="approve")
        assert result.status == "done"
        assert sorted(started) == ["0", "1", "2"]
        if not approval:
            assert maximum == 2
        assert runtime.store.verify().valid


@pytest.mark.anyio
async def test_workspace_change_policy_still_allows_parallel_reads_within_budget(tmp_path: Path):
    started = set()
    barrier = asyncio.Event()

    async def run(args):
        started.add(args["path"])
        if len(started) == 2:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), 1)
        return {"text": "ok"}

    store, _, orchestrator, calls = _setup(tmp_path, run, context_changes={"max_parallel_calls": 2})
    with store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="edit",
            binding_manifest={"completion_policy": {"require_workspace_change": True}},
        )
        results = await orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls)
        assert len(results) == 3 and all(not result.is_error for result in results)
        assert store.verify().valid

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_runtime.harness import Session, turn_ops
from tests.agent.harness.test_subagent_budget import AcceptAnswer
from tests.agent.harness.test_subagent_budget import BlockingAnswerModel as BaseBlockingModel


class BlockingAnswerModel(BaseBlockingModel):
    def __init__(self):
        super().__init__()
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return super().prepare(request)


def parent(session: Session, *, limits=None):
    return turn_ops.start_turn(
        session.store,
        thread_id=session.thread_id,
        user_message="parent",
        binding_manifest={
            "model_id": "test-model",
            "model_token_budget_total": 500,
            **({"subagent_limits": limits} if limits is not None else {}),
        },
    )


@pytest.mark.anyio
async def test_spawn_reserves_before_return_and_two_children_overlap(tmp_path: Path):
    model = BlockingAnswerModel()
    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session)
        one = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="one", max_tokens_total=80)
        two = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="two", max_tokens_total=80)
        assert one["child_turn_id"] != two["child_turn_id"]
        for handle in (one, two):
            assert session.store.read_child_budget_allocation(handle["child_turn_id"])["parent_turn_id"] == p.turn_id
        assert session.store.read_budget_state(p.turn_id).child_reserved.total_tokens == 160
        await asyncio.wait_for(model.started.wait(), 2)
        for _ in range(100):
            if len(model.requests) >= 2:
                break
            await asyncio.sleep(0.01)
        assert len(model.requests) == 2
        pending = await session.subagents.wait(
            parent_turn_id=p.turn_id, child_turn_id=one["child_turn_id"], timeout_seconds=0.01
        )
        assert pending["status"] == "running"
        model.release.set()
        for handle in (one, two):
            result = await session.subagents.wait(
                parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], timeout_seconds=2
            )
            assert result["status"] == "done"
            assert result["execution_evidence"] == []
            assert session.store.read_child_budget_allocation(handle["child_turn_id"])["status"] == "settled"
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_child_send_is_scoped_and_idempotent_and_ownership_is_durable(tmp_path: Path):
    from agent_runtime.harness.subagents import SubagentManager

    model = BlockingAnswerModel()
    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session)
        one = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="one", max_tokens_total=80)
        two = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="two", max_tokens_total=80)
        await asyncio.wait_for(model.started.wait(), 2)
        restored_manager = SubagentManager(session)
        for _ in range(2):
            await restored_manager.send(
                parent_turn_id=p.turn_id,
                child_turn_id=one["child_turn_id"],
                message="focus",
                message_id="message-stable",
            )
        messages = [i for i in session.store.list_items(one["child_turn_id"]) if i.kind == "user_message"]
        assert [i.payload["text"] for i in messages] == ["one", "focus"]
        assert len([i for i in session.store.list_items(two["child_turn_id"]) if i.kind == "user_message"]) == 1
        with pytest.raises(PermissionError):
            await restored_manager.send(
                parent_turn_id="foreign", child_turn_id=one["child_turn_id"], message="bad", message_id="bad"
            )
        model.release.set()
        await session.subagents.drain(p.turn_id)


@pytest.mark.anyio
async def test_cancel_is_scoped_and_records_kernel_request_before_cancellation(tmp_path: Path):
    model = BlockingAnswerModel()
    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session)
        one = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="one", max_tokens_total=80)
        two = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="two", max_tokens_total=80)
        await asyncio.wait_for(model.started.wait(), 2)
        result = await session.subagents.cancel(parent_turn_id=p.turn_id, child_turn_id=one["child_turn_id"])
        assert result["status"] in {"paused", "failed"}
        assert session.store.turn_cancellation_pending(one["child_turn_id"])
        assert not session.store.turn_cancellation_pending(two["child_turn_id"])
        assert any(
            r.record_type == "turn_cancellation_requested"
            for r in session.store.list_turn_records(one["child_turn_id"])
        )
        model.release.set()
        assert (
            await session.subagents.wait(
                parent_turn_id=p.turn_id, child_turn_id=two["child_turn_id"], timeout_seconds=2
            )
        )["status"] == "done"


@pytest.mark.anyio
async def test_limits_are_checked_against_canonical_allocations(tmp_path: Path):
    model = BlockingAnswerModel()
    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session, limits={"max_concurrent": 1, "max_depth": 1, "max_total": 2})
        await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="one", max_tokens_total=80)
        with pytest.raises(RuntimeError, match="concurrent"):
            await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="two", max_tokens_total=80)
        assert len(session.store.list_threads()) == 2
        model.release.set()
        await session.subagents.drain(p.turn_id)


@pytest.mark.anyio
async def test_async_child_tools_are_installed_and_discoverable_with_frozen_limits(tmp_path: Path):
    async with await Session.open(
        database=tmp_path / "rollout.db",
        workspace=tmp_path,
        model=BlockingAnswerModel(),
        enable_subagents=True,
        subagent_limits={"max_concurrent": 2, "max_depth": 1, "max_total": 3},
    ) as session:
        names = {"spawn_agent", "send_agent_message", "wait_agent", "cancel_agent"}
        assert names <= set(session.tools)
        snapshot = session.binding_provider.snapshot(thread_id=session.thread_id, turn_id="new")
        assert snapshot["subagent_limits"] == {"max_concurrent": 2, "max_depth": 1, "max_total": 3}
        for name in names:
            assert session.tools[name].concurrency_safe


@pytest.mark.anyio
async def test_parent_completion_waits_for_attached_children_without_model_polling(tmp_path: Path):
    model = BlockingAnswerModel()
    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session)
        child = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="one", max_tokens_total=80)
        await asyncio.wait_for(model.started.wait(), 2)
        runner = session._executor()
        completing = asyncio.create_task(
            runner._stop.finish(
                thread_id=p.thread_id,
                turn_id=p.turn_id,
                answer="parent answer",
                input_revision=session.store.user_input_revision(p.turn_id),
            )
        )
        try:
            await asyncio.sleep(0.03)
            assert not completing.done()
            assert session.store.read_turn(p.turn_id).status == "running"
            assert len(model.requests) == 1
        finally:
            model.release.set()
        await asyncio.wait_for(completing, 2)
        assert session.store.read_turn(p.turn_id).status == "completed"
        assert session.store.read_child_budget_allocation(child["child_turn_id"])["status"] == "settled"
        records = [r for thread in session.store.list_threads() for r in session.store.list_records(thread.thread_id)]
        child_settled = next(r.record_id for r in records if r.record_type == "budget_child_settled")
        parent_completed = next(
            r.record_id for r in records if r.record_type == "turn_completed" and r.turn_id == p.turn_id
        )
        assert child_settled < parent_completed


@pytest.mark.anyio
async def test_zero_child_limit_disables_spawn_and_restart_limit_counts_prior_children(tmp_path: Path):
    from agent_runtime.harness.subagents import SubagentManager

    model = BlockingAnswerModel()
    async with await Session.open(
        database=tmp_path / "rollout.db",
        workspace=tmp_path,
        model=model,
        completion_gate=AcceptAnswer(),
        subagent_limits={"max_total": 0},
    ) as session:
        p = parent(session)
        with pytest.raises(RuntimeError, match="total"):
            await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="no", max_tokens_total=80)
        assert len(session.store.list_threads()) == 1
    async with await Session.open(
        database=tmp_path / "other.db", workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session, limits={"max_concurrent": 1, "max_depth": 1, "max_total": 1})
        await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="one", max_tokens_total=80)
        model.release.set()
        await session.subagents.drain(p.turn_id)
        restored = SubagentManager(session)
        with pytest.raises(RuntimeError, match="total"):
            await restored.spawn(parent_turn_id=p.turn_id, user_message="two", max_tokens_total=80)


@pytest.mark.anyio
async def test_cancelling_parent_execution_cancels_its_attached_children(tmp_path: Path):
    model = BlockingAnswerModel()
    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session)
        ready = asyncio.Event()
        child_id = None

        async def parent_driver():
            nonlocal child_id
            async with session._active_turn():
                handle = await session.subagents.spawn(
                    parent_turn_id=p.turn_id, user_message="child", max_tokens_total=80
                )
                child_id = handle["child_turn_id"]
                await model.started.wait()
                ready.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(parent_driver())
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert child_id is not None
        assert session.store.turn_cancellation_pending(p.turn_id)
        assert session.store.turn_cancellation_pending(child_id)
        assert not any(not t.done() for t in session._subagent_tasks.values())


@pytest.mark.anyio
async def test_lost_driver_cancel_preserves_unknown_budget_and_parent_cannot_complete(tmp_path: Path):
    from tests.agent.harness.test_subagent_budget import UnknownAnswerModel

    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=UnknownAnswerModel(), completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session)
        child = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="child", max_tokens_total=80)
        await session.subagents.wait(parent_turn_id=p.turn_id, child_turn_id=child["child_turn_id"], timeout_seconds=2)
        before = session.store.read_budget_state(p.turn_id).child_reserved
        await session.subagents.cancel(parent_turn_id=p.turn_id, child_turn_id=child["child_turn_id"])
        assert before.total_tokens == 80
        assert session.store.read_budget_state(p.turn_id).child_reserved == before
        assert session.store.read_child_budget_allocation(child["child_turn_id"])["status"] == "active"
        result = await session._executor()._stop.finish(
            thread_id=p.thread_id,
            turn_id=p.turn_id,
            answer="parent answer",
            input_revision=session.store.user_input_revision(p.turn_id),
        )
        assert result.status == "paused"
        assert session.store.read_turn(p.turn_id).status == "paused"
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_real_parent_tool_batch_starts_two_children_before_either_finishes(tmp_path: Path):
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall
    from agent_runtime.tools.permissions import ToolExecutionContext
    from tests.agent.harness.test_subagent_path import ParentChildModel

    class ConcurrentModel(ParentChildModel):
        def __init__(self):
            super().__init__()
            self.started = set()
            self.both_started = asyncio.Event()
            self.release = asyncio.Event()

        async def dispatch(self, prepared, **_kwargs):
            request = prepared.dispatch_payload
            if request.messages[0].content.startswith("child"):
                self.started.add(request.turn_id)
                if len(self.started) == 2:
                    self.both_started.set()
                await self.release.wait()
                return HarnessModelResponse.from_parts(
                    text="child answer", usage={"input_tokens": 2, "output_tokens": 3}
                )
            messages = [m for m in request.messages if m.role == "tool"]
            calls = ()
            if not messages:
                calls = (ModelToolCall(id="find", name="find_tools", arguments={"query": "spawn_agent", "limit": 5}),)
            elif not any(m.tool_call_id == "spawn-one" for m in messages):
                calls = tuple(
                    ModelToolCall(
                        id=f"spawn-{name}",
                        name="spawn_agent",
                        arguments={"task": f"child {name}", "llm_budget_total": 80},
                    )
                    for name in ("one", "two")
                )
            return HarnessModelResponse.from_parts(
                text="" if calls else "parent answer", tool_calls=calls, usage={"input_tokens": 2, "output_tokens": 3}
            )

    model = ConcurrentModel()
    context = ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path, approval_mode="full")
    async with await Session.open(
        database=tmp_path / "rollout.db",
        workspace=tmp_path,
        model=model,
        completion_gate=AcceptAnswer(),
        enable_subagents=True,
        tool_execution_context=context,
        max_tokens_total=1000,
    ) as session:
        driving = asyncio.create_task(session.submit("coordinate children"))
        try:
            await asyncio.wait_for(model.both_started.wait(), 3)
            assert not driving.done()
            p = session.store.read_thread(session.thread_id).active_turn_id
            assert p is not None
            assert session.store.read_turn(p).status == "running"
            assert len(model.started) == 2
        finally:
            model.release.set()
        assert (await asyncio.wait_for(driving, 3)).answer == "parent answer"
        assert session.store.read_budget_state(p).child_reserved.total_tokens == 0
        assert session.store.verify().valid


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_child_commands_have_separate_owners_and_cancel_reaps_only_target(tmp_path: Path, monkeypatch):
    import os
    from dataclasses import replace

    from agent_runtime.harness import HarnessModelResponse, ModelToolCall
    from agent_runtime.tools.builtins.command_sessions import DurableCommandSessions, create_manage_command_tool
    from agent_runtime.tools.builtins.shell import create_run_command_tool
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.workspace import open_workspace
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    class CommandModel(SmallAnswerModel):
        def prepare(self, request):
            prepared = super().prepare(request)
            return replace(
                prepared,
                dispatch_payload=request,
                request_ref={
                    **prepared.request_ref,
                    "toolset_revision": request.binding_manifest["toolset_revision"],
                    "exposed_tool_names": [tool.definition.name for tool in request.tools],
                },
            )

        async def dispatch(self, prepared, **_kwargs):
            request = prepared.dispatch_payload
            label = request.messages[0].content
            if not any(m.role == "tool" for m in request.messages):
                return HarnessModelResponse.from_parts(
                    tool_calls=(
                        ModelToolCall(
                            id="command",
                            name="run_command",
                            arguments={
                                "command": f"echo $$ > {label}.pid; sleep 30",
                                "working_dir": ".",
                                "workspace_write": True,
                                "timeout_seconds": None,
                                "yield_time_seconds": 0.01,
                            },
                        ),
                    ),
                    usage={"input_tokens": 2, "output_tokens": 3},
                )
            await asyncio.Event().wait()

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees-outside"))
    source = tmp_path / "source"
    source.mkdir()
    workspace = open_workspace(source)
    owner = DurableCommandSessions(workspace)
    tools = (create_run_command_tool(workspace, command_sessions=owner), create_manage_command_tool(workspace, owner))
    context = ToolExecutionContext(workspace_root=source, cwd=source, approval_mode="full")
    async with await Session.open(
        database=tmp_path / "rollout.db",
        workspace=source,
        model=CommandModel(),
        completion_gate=AcceptAnswer(),
        tool_execution_context=context,
        tools={t.definition.name: t for t in tools},
    ) as session:
        session._own_command_sessions(owner)
        p = parent(session)
        one = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="one", max_tokens_total=80)
        two = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="two", max_tokens_total=80)
        roots = {
            label: Path(session.store.read_thread(handle["thread_id"]).workspace)
            for label, handle in (("one", one), ("two", two))
        }
        assert roots["one"] != roots["two"] != source
        try:
            async with asyncio.timeout(5):
                while not all((roots[label] / f"{label}.pid").exists() for label in ("one", "two")):
                    for handle in (one, two):
                        task = session._subagent_tasks[handle["child_turn_id"]]
                        if task.done():
                            await task
                    await asyncio.sleep(0.01)
            one_pid = int((roots["one"] / "one.pid").read_text())
            two_pid = int((roots["two"] / "two.pid").read_text())
            assert not (source / "one.pid").exists()
            assert not (source / "two.pid").exists()
            assert session._subagent_sessions[one["child_turn_id"]]._command_sessions is not owner
            assert session._subagent_sessions[two["child_turn_id"]]._command_sessions is not owner
            await session.subagents.cancel(parent_turn_id=p.turn_id, child_turn_id=one["child_turn_id"])
            with pytest.raises(ProcessLookupError):
                os.kill(one_pid, 0)
            os.kill(two_pid, 0)
            assert not session.store.turn_cancellation_pending(two["child_turn_id"])
        finally:
            await session.subagents.cancel_all(p.turn_id)
        with pytest.raises(ProcessLookupError):
            os.kill(two_pid, 0)
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_close_records_cancellation_for_durable_child_with_no_live_driver(tmp_path: Path):
    from agent_runtime.harness import RuntimeStore

    session = await Session.open(
        database=tmp_path / "rollout.db",
        workspace=tmp_path,
        model=BlockingAnswerModel(),
        completion_gate=AcceptAnswer(),
    )
    p = parent(session)
    prepared = await session._prepare_child(
        parent_turn_id=p.turn_id,
        user_message="reserved but driver lost",
        max_steps=None,
        max_tokens_total=80,
    )
    child_id = prepared["turn_id"]
    await session.close()
    async with await RuntimeStore.open(tmp_path / "rollout.db") as store:
        assert store.turn_cancellation_pending(child_id)
        assert store.read_budget_state(p.turn_id).child_reserved.total_tokens == 80
        assert store.verify().valid


@pytest.mark.anyio
async def test_completion_transaction_rechecks_child_spawned_after_initial_drain(tmp_path: Path):
    model = BlockingAnswerModel()
    async with await Session.open(database=tmp_path / "rollout.db", workspace=tmp_path,
                                  model=model, completion_gate=AcceptAnswer()) as session:
        p = parent(session)
        stop = session._executor()._stop
        first_drain = asyncio.Event()
        proceed = asyncio.Event()
        calls = 0

        async def between_drain_and_commit(turn_id):
            nonlocal calls
            calls += 1
            await session._drain_children(turn_id)
            if calls == 1:
                first_drain.set()
                await proceed.wait()

        stop._before_completion = between_drain_and_commit
        completing = asyncio.create_task(stop.finish(
            thread_id=p.thread_id, turn_id=p.turn_id, answer="parent answer",
            input_revision=session.store.user_input_revision(p.turn_id),
        ))
        try:
            await asyncio.wait_for(first_drain.wait(), 2)
            child = await session.subagents.spawn(
                parent_turn_id=p.turn_id, user_message="late child", max_tokens_total=80
            )
            await asyncio.wait_for(model.started.wait(), 2)
            proceed.set()
            await asyncio.sleep(0.03)
            assert session.store.read_turn(p.turn_id).status == "running"
            assert not completing.done()
            assert len(model.requests) == 1
        finally:
            proceed.set()
            model.release.set()
        result = await asyncio.wait_for(completing, 2)
        assert result.status == "completed"
        assert calls == 2
        assert session.store.read_child_budget_allocation(child["child_turn_id"])["status"] == "settled"
        assert len([i for i in session.store.list_items(p.turn_id) if i.kind == "final_proposal"]) == 1
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_canonical_complete_rejects_active_child_allocation_even_without_session_hook(tmp_path: Path):
    from agent_runtime.harness.operation_values import TurnChildrenPendingError

    model = BlockingAnswerModel()
    async with await Session.open(database=tmp_path / "rollout.db", workspace=tmp_path,
                                  model=model, completion_gate=AcceptAnswer()) as session:
        p = parent(session)
        await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="child", max_tokens_total=80)
        try:
            with pytest.raises(TurnChildrenPendingError):
                turn_ops.complete_turn(session.store, turn_id=p.turn_id, answer="parent answer")
            assert session.store.read_turn(p.turn_id).status == "running"
        finally:
            model.release.set()
        await session.subagents.drain(p.turn_id)


@pytest.mark.anyio
async def test_pending_parent_cancellation_atomically_rejects_new_child_allocation(tmp_path: Path):
    from agent_runtime.harness.operation_values import TurnCancellationPendingError

    model = BlockingAnswerModel()
    async with await Session.open(database=tmp_path / "rollout.db", workspace=tmp_path,
                                  model=model, completion_gate=AcceptAnswer()) as session:
        p = parent(session)
        turn_ops.request_turn_cancellation(session.store, turn_id=p.turn_id, reason="stop parent")
        try:
            with pytest.raises(TurnCancellationPendingError):
                await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="too late", max_tokens_total=80)
        finally:
            model.release.set()
        assert len(session.store.list_threads()) == 1
        assert model.requests == []


@pytest.mark.anyio
async def test_ancestor_cancellation_rejects_grandchild_allocation(tmp_path: Path):
    from agent_runtime.harness.operation_values import TurnCancellationPendingError

    model = BlockingAnswerModel()
    database = tmp_path / "rollout.db"
    async with await Session.open(database=database, workspace=tmp_path,
                                  model=model, completion_gate=AcceptAnswer()) as session:
        p = parent(session)
        prepared = await session._prepare_child(parent_turn_id=p.turn_id, user_message="child",
                                                max_steps=None, max_tokens_total=80)
        async with await Session.open(database=database, workspace=Path(prepared["workspace"]), model=model,
                                      completion_gate=AcceptAnswer(), thread_id=prepared["thread_id"]) as child:
            turn_ops.request_turn_cancellation(session.store, turn_id=p.turn_id, reason="stop entire subtree")
            try:
                with pytest.raises(TurnCancellationPendingError):
                    await child.subagents.spawn(parent_turn_id=prepared["turn_id"], user_message="too late",
                                                max_tokens_total=20)
            finally:
                model.release.set()
            assert len(session.store.list_threads()) == 2
            assert model.requests == []


@pytest.mark.anyio
async def test_reserved_child_dispatch_observes_parent_cancellation_before_start(tmp_path: Path):
    model = BlockingAnswerModel()
    async with await Session.open(database=tmp_path / "rollout.db", workspace=tmp_path,
                                  model=model, completion_gate=AcceptAnswer()) as session:
        p = parent(session)
        prepared = await session._prepare_child(parent_turn_id=p.turn_id, user_message="reserved child",
                                                max_steps=None, max_tokens_total=80)
        turn_ops.request_turn_cancellation(session.store, turn_id=p.turn_id, reason="stop before schedule")
        session.subagents._schedule(prepared)
        try:
            await asyncio.sleep(0.05)
            assert not model.started.is_set()
        finally:
            model.release.set()
        await session.subagents.cancel_all(p.turn_id)
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_session_close_cancels_live_attached_children_without_waiting_for_provider(tmp_path: Path):
    model = BlockingAnswerModel()
    database = tmp_path / "rollout.db"
    session = await Session.open(
        database=database, workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    )
    p = parent(session)
    handle = await session.subagents.spawn(
        parent_turn_id=p.turn_id, user_message="blocked provider", max_tokens_total=80
    )
    await asyncio.wait_for(model.started.wait(), 2)
    closing = asyncio.create_task(session.close())
    try:
        _, pending = await asyncio.wait({closing}, timeout=1.0)
        assert not pending, "Session close must cancel owned children before awaiting their drivers"
    finally:
        model.release.set()
        await closing
    from agent_runtime.harness import RuntimeStore

    async with await RuntimeStore.open(database) as store:
        assert await store.run_async(store.turn_cancellation_pending, handle["child_turn_id"])
        allocation = await store.run_async(store.read_child_budget_allocation, handle["child_turn_id"])
        assert allocation["status"] == "active"  # started provider outcome/billing remains unknown
        assert (await store.run_async(store.read_budget_state, p.turn_id)).child_reserved.total_tokens == 80
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
async def test_reopened_child_observes_cancelled_ancestor_without_live_dispatch_guard(tmp_path: Path):
    database = tmp_path / "rollout.db"
    model = BlockingAnswerModel()
    async with await Session.open(
        database=database, workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    ) as parent_session:
        p = parent(parent_session)
        prepared = await parent_session._prepare_child(
            parent_turn_id=p.turn_id, user_message="reserved child", max_steps=None, max_tokens_total=80
        )
        turn_ops.request_turn_cancellation(parent_session.store, turn_id=p.turn_id, reason="stop subtree")
        async with await Session.open(
            database=database, workspace=Path(prepared["workspace"]), model=model,
            thread_id=prepared["thread_id"], completion_gate=AcceptAnswer()
        ) as reopened:
            runner = reopened._executor()
            driving = asyncio.create_task(runner.drive(turn_id=prepared["turn_id"], start_step=1))
            try:
                done, _ = await asyncio.wait({driving}, timeout=1.0)
                assert done, "recovered child must stop rather than spin on rejected admission"
                assert driving.result().status == "cancelled"
                assert not model.started.is_set()
            finally:
                turn_ops.request_turn_cancellation(
                    reopened.store, turn_id=prepared["turn_id"], reason="test cleanup"
                )
                driving.cancel()
                await asyncio.gather(driving, return_exceptions=True)
                model.release.set()


@pytest.mark.anyio
async def test_close_cancels_async_children_while_parent_holds_completion_lock(tmp_path: Path):
    model = BlockingAnswerModel()
    session = await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    )
    p = parent(session)
    handle = await session.subagents.spawn(
        parent_turn_id=p.turn_id, user_message="blocked child", max_tokens_total=80
    )
    await asyncio.wait_for(model.started.wait(), 2)
    draining = asyncio.Event()

    async def parent_finishes():
        async with session._active_turn():
            draining.set()
            return await session._executor()._stop.finish(
                thread_id=session.thread_id, turn_id=p.turn_id, answer="parent answer", input_revision=1
            )

    finishing = asyncio.create_task(parent_finishes())
    await draining.wait()
    closing = asyncio.create_task(session.close())
    try:
        done, _ = await asyncio.wait({closing}, timeout=1.0)
        assert done, "close must revoke async children before waiting for parent completion lock"
        await finishing
    finally:
        model.release.set()
        await asyncio.gather(finishing, closing, return_exceptions=True)
        await session.close()
    from agent_runtime.harness import RuntimeStore

    async with await RuntimeStore.open(tmp_path / "rollout.db") as store:
        assert await store.run_async(store.turn_cancellation_pending, handle["child_turn_id"])


@pytest.mark.anyio
async def test_new_input_interrupts_parent_completion_wait_without_cancelling_child(tmp_path: Path):
    model = BlockingAnswerModel()
    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session)
        handle = await session.subagents.spawn(
            parent_turn_id=p.turn_id, user_message="blocked child", max_tokens_total=80
        )
        await asyncio.wait_for(model.started.wait(), 2)
        finishing = asyncio.create_task(session._executor()._stop.finish(
            thread_id=session.thread_id, turn_id=p.turn_id, answer="old answer", input_revision=1
        ))
        try:
            await session.steer("review the new requirements", expected_turn_id=p.turn_id)
            done, _ = await asyncio.wait({finishing}, timeout=1.0)
            assert done, "new input must release the parent to adapt before its child finishes"
            assert finishing.result() is None
            assert not session._subagent_tasks[handle["child_turn_id"]].done()
            assert not session.store.turn_cancellation_pending(handle["child_turn_id"])
        finally:
            model.release.set()
            await finishing


@pytest.mark.anyio
async def test_close_waits_for_direct_child_but_cancels_async_child_independently(tmp_path: Path):
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    class SeparateChildrenModel(SmallAnswerModel):
        def __init__(self):
            self.direct_started = asyncio.Event()
            self.async_started = asyncio.Event()
            self.release_direct = asyncio.Event()

        def prepare(self, request):
            from dataclasses import replace
            return replace(super().prepare(request), dispatch_payload=request)

        async def dispatch(self, prepared, *, event_sink=None):
            label = prepared.dispatch_payload.messages[0].content
            if label == "direct":
                self.direct_started.set()
                await self.release_direct.wait()
                return await super().dispatch(prepared, event_sink=event_sink)
            self.async_started.set()
            await asyncio.Event().wait()

    model = SeparateChildrenModel()
    session = await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    )
    p = parent(session)
    direct = asyncio.create_task(session.run_child(
        parent_turn_id=p.turn_id, user_message="direct", max_steps=None, max_tokens_total=80
    ))
    await model.direct_started.wait()
    handle = await session.subagents.spawn(
        parent_turn_id=p.turn_id, user_message="async", max_tokens_total=80
    )
    await model.async_started.wait()
    closing = asyncio.create_task(session.close())
    try:
        async with asyncio.timeout(1):
            while not session.store.turn_cancellation_pending(handle["child_turn_id"]):
                await asyncio.sleep(0.01)
        assert not closing.done(), "direct child still owns its result/settlement"
    finally:
        model.release_direct.set()
        await asyncio.gather(direct, closing, return_exceptions=True)
        await session.close()


@pytest.mark.anyio
async def test_stale_completion_with_unknown_child_does_not_pause_new_input(tmp_path: Path):
    from tests.agent.harness.test_subagent_budget import UnknownAnswerModel

    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path,
        model=UnknownAnswerModel(), completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session)
        handle = await session.subagents.spawn(
            parent_turn_id=p.turn_id, user_message="unknown child", max_tokens_total=80
        )
        await session.subagents.wait(
            parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], timeout_seconds=2
        )
        await session.steer("continue reviewing the implementation", expected_turn_id=p.turn_id)
        result = await session._executor()._stop.finish(
            thread_id=session.thread_id, turn_id=p.turn_id, answer="old answer", input_revision=1
        )
        assert result is None
        assert session.store.read_turn(p.turn_id).status == "running"


@pytest.mark.anyio
async def test_parent_child_wait_resumes_after_settlement_or_new_input(tmp_path: Path):
    from agent_runtime.harness import budget_ops
    from agent_runtime.harness.recovery import recovery_plan
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path,
        model=SmallAnswerModel(), completion_gate=AcceptAnswer()
    ) as session:
        p = parent(session)
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id, user_message="reserved child", max_steps=None, max_tokens_total=80
        )
        result = await session._executor()._stop.finish(
            thread_id=session.thread_id, turn_id=p.turn_id, answer="not ready", input_revision=1
        )
        assert result.status == "paused"
        waiting = recovery_plan(session.store, p.turn_id)
        assert "continue" not in waiting.actions
        turn_ops.pause_turn(session.store, turn_id=prepared["turn_id"], reason="known unstarted child")
        turn_ops.cancel_turn(session.store, turn_id=prepared["turn_id"])
        budget_ops.settle_child_budget(session.store, child_turn_id=prepared["turn_id"])
        ready = recovery_plan(session.store, p.turn_id)
        assert ready.accepts("continue"), ready
        turn_ops.request_turn_cancellation(session.store, turn_id=p.turn_id, reason="explicit cancellation")
        assert recovery_plan(session.store, p.turn_id).kind == "resume_cancelled"
        resumed = await session.resume(p.turn_id, "continue")
        assert resumed.status == "done"

        next_parent = parent(session)
        await session._prepare_child(
            parent_turn_id=next_parent.turn_id, user_message="second child", max_steps=None, max_tokens_total=80
        )
        await session._executor()._stop.finish(
            thread_id=session.thread_id, turn_id=next_parent.turn_id, answer="waiting", input_revision=1
        )
        await session.steer("change how the child is handled", expected_turn_id=next_parent.turn_id)
        assert recovery_plan(session.store, next_parent.turn_id).accepts("continue")

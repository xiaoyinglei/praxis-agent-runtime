from __future__ import annotations

import asyncio
import sqlite3
import threading
import time

import pytest

from agent_runtime.harness import RuntimeState, TurnExecutor
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator


@pytest.mark.anyio
@pytest.mark.parametrize("kind", [TurnExecutor, ToolOrchestrator])
async def test_transaction_body_yields_while_real_store_commits(tmp_path, monkeypatch, kind):
    async with (await RuntimeState.open(tmp_path / "run.db")) as store:
        thread = store.create_thread(workspace=tmp_path)
        original = store._append_and_reduce
        completed = threading.Event()
        ticks = []

        def slow_reduce(**kwargs):
            time.sleep(0.08)
            original(**kwargs)

        monkeypatch.setattr(store, "_append_and_reduce", slow_reduce)
        executor = object.__new__(kind)
        executor._store, executor._event_dispatcher = store, None

        async def commit():
            try:
                return await executor._commit(
                    lambda: store.start_turn(
                        thread_id=thread.thread_id,
                        user_message="large durable input" * 1000,
                        binding_manifest={},
                    )
                )
            finally:
                completed.set()

        async def heartbeat():
            while not completed.is_set():
                ticks.append(time.monotonic())
                await asyncio.sleep(0.005)

        pulse = asyncio.create_task(heartbeat())
        turn = await commit()
        await pulse
        assert turn.status == "running"
        assert len(ticks) >= 3, "SQLite transaction body occupied the event loop"
        assert store.verify().valid


@pytest.mark.anyio
async def test_artifact_install_yields_and_commit_is_visible(tmp_path, monkeypatch):
    async with (await RuntimeState.open(tmp_path / "run.db")) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="save", binding_manifest={})
        ticks = []
        done = threading.Event()

        async def heartbeat():
            while not done.is_set():
                ticks.append(1)
                await asyncio.sleep(0.005)

        def slow_install(stage):
            if stage == "after_temp_fsync":
                time.sleep(0.08)

        pulse = asyncio.create_task(heartbeat())
        try:
            mutation = await store.capture_mutation_async(
                lambda: store.commit_artifact(
                    turn_id=turn.turn_id,
                    content=b"artifact" * 10000,
                    media_type="text/plain",
                    name="result",
                    fault_injector=slow_install,
                )
            )
        finally:
            done.set()
            await pulse
        assert len(ticks) >= 3, "artifact installation occupied the event loop"
        assert store.read_artifact(mutation.value.artifact_id) == b"artifact" * 10000
        assert mutation.records[-1].record_type == "artifact_committed"


@pytest.mark.anyio
async def test_public_agent_open_and_read_result_yield(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel

    agent = Agent(checkpoint_db=tmp_path / "run.db", workspace_path=tmp_path)
    monkeypatch.setattr(agent, "_model_factory", lambda: PublicHarnessModel())
    original = RuntimeState.validate_conversation
    observed = []

    def slow_verify(store, **anchor):
        before = len(observed)
        time.sleep(0.08)
        value = original(store, **anchor)
        assert len(observed) > before, "public Agent initialization blocked its loop"
        return value

    monkeypatch.setattr(RuntimeState, "validate_conversation", slow_verify)
    stopped = asyncio.Event()

    async def heartbeat():
        while not stopped.is_set():
            observed.append(1)
            await asyncio.sleep(0.005)

    pulse = asyncio.create_task(heartbeat())
    try:
        result = await agent.run("answer", require_workspace_change=False)
        assert result.status == "done"
        original_read = RuntimeState.read_turn_answer

        def slow_read(store, turn_id):
            before = len(observed)
            time.sleep(0.08)
            value = original_read(store, turn_id)
            assert len(observed) > before, "public read_result blocked its loop"
            return value

        monkeypatch.setattr(RuntimeState, "read_turn_answer", slow_read)
        assert await agent.read_result(result.turn_id) == result
    finally:
        stopped.set()
        await pulse


@pytest.mark.anyio
async def test_public_agent_never_waits_synchronously_for_store(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel

    loop_thread = threading.get_ident()
    original = RuntimeState.run_sync

    def guarded(store, operation):
        assert threading.get_ident() != loop_thread, "async runtime called the blocking Store API"
        return original(store, operation)

    monkeypatch.setattr(RuntimeState, "run_sync", guarded)
    agent = Agent(checkpoint_db=tmp_path / "run.db", workspace_path=tmp_path, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: PublicHarnessModel())
    result = await agent.run("answer", require_workspace_change=False)
    assert await agent.read_result(result.turn_id) == result
    assert (await agent.recovery_state(result.turn_id))["status"] == "completed"


@pytest.mark.anyio
@pytest.mark.parametrize("fail", [False, True])
async def test_cancel_and_close_wait_for_real_commit_or_rollback(tmp_path, fail):
    database = tmp_path / "run.db"
    store = await RuntimeState.open(database)
    thread = await store.run_async(store.create_thread, workspace=tmp_path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def operation():
        calls.append(threading.get_ident())
        try:
            with store.persistence.transaction():
                store.update_session_settings(thread.thread_id, {"observed": True})
                entered.set()
                assert release.wait(3)
                if fail:
                    raise OSError("controlled write failure")
        finally:
            finished.set()

    task = asyncio.create_task(store.capture_mutation_async(operation))
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        closer = asyncio.create_task(store.aclose())
        await asyncio.sleep(0.01)
        assert not task.done() and not closer.done() and not finished.is_set()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            await store.run_async(lambda: None)
        release.set()
        with pytest.raises(OSError if fail else asyncio.CancelledError):
            await task
        await closer
        assert finished.is_set()
        assert len(calls) == 1  # Neither cancellation nor a Store failure replays a transaction.
        async with await RuntimeState.open(database, read_only=True) as reader:
            state = await reader.run_async(reader.read_thread, thread.thread_id)
            assert dict(state.settings) == ({} if fail else {"observed": True})
            assert (await reader.run_async(reader.verify)).valid
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await store.aclose()


@pytest.mark.anyio
async def test_connection_open_queries_and_close_share_one_owner(tmp_path, monkeypatch):
    import sqlite3

    caller = threading.get_ident()
    original = sqlite3.connect
    observed = []

    class Connection(sqlite3.Connection):
        def execute(self, *args, **kwargs):
            observed.append(("query", threading.get_ident()))
            return super().execute(*args, **kwargs)

        def close(self):
            observed.append(("close", threading.get_ident()))
            return super().close()

    def connect(*args, **kwargs):
        observed.append(("open", threading.get_ident()))
        return original(*args, **kwargs, factory=Connection)

    monkeypatch.setattr(sqlite3, "connect", connect)
    async with await RuntimeState.open(tmp_path / "run.db") as store:
        first = await store.run_async(store.create_thread, workspace=tmp_path)
        assert store.read_thread(first.thread_id) == first  # Explicit blocking API remains usable.
        assert await asyncio.to_thread(store.read_thread, first.thread_id) == first
    assert {identity for _, identity in observed} == {observed[0][1]}
    assert observed[0][1] != caller
    assert observed[-1][0] == "close"


@pytest.mark.anyio
async def test_multiple_tasks_connections_and_read_only_queries(tmp_path):
    database = tmp_path / "run.db"
    async with await RuntimeState.open(database) as first, await RuntimeState.open(database) as second:
        thread = await first.run_async(first.create_thread, workspace=tmp_path)
        turn = await first.run_async(
            first.start_turn, thread_id=thread.thread_id, user_message="concurrent reservations", binding_manifest={}
        )
        mutations = await asyncio.gather(
            *[
                store.capture_mutation_async(lambda store=store: store.reserve_resource_unit(turn.turn_id, "reads", 32))
                for store in [first, second] * 8
            ]
        )
        ids = [mutation.records[-1].record_id for mutation in mutations]
        assert len(set(ids)) == 16
        assert ids[::2] == sorted(ids[::2])
        assert ids[1::2] == sorted(ids[1::2])
        assert (await first.run_async(first.read_resource_usage, turn.turn_id, "reads"))["used"] == 16
        async with await RuntimeState.open(database, read_only=True) as reader:
            assert (await reader.run_async(reader.verify)).valid
            with pytest.raises(Exception, match="readonly"):
                await reader.run_async(reader.create_thread, workspace=tmp_path)


@pytest.mark.anyio
async def test_cancelled_guard_records_known_unexecuted_tool_result(tmp_path, monkeypatch):
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
    from tests.agent.harness.test_tool_orchestrator import _read_tool

    async with await RuntimeState.open(tmp_path / "run.db") as store:
        thread = await store.run_async(store.create_thread, workspace=tmp_path)
        turn = await store.run_async(
            store.start_turn, thread_id=thread.thread_id, user_message="read", binding_manifest={}
        )
        orchestrator = ToolOrchestrator(
            store=store,
            tools={"read_file": _read_tool()},
            execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path),
        )
        entered, release = threading.Event(), threading.Event()

        def guard(*args):
            entered.set()
            assert release.wait(3)

        monkeypatch.setattr(orchestrator, "_guard_input", guard)
        call = ToolCall(
            tool_call_id="read",
            tool_name="read_file",
            arguments={"path": "present.txt"},
            origin=ToolCallOrigin(request_id="r", toolset_revision="v", exposed_tool_names=("read_file",)),
        )
        task = asyncio.create_task(orchestrator.execute(turn_id=turn.turn_id, call=call))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            [operation] = await store.run_async(store.list_tool_operations, turn.turn_id)
            assert operation.status == "failed"
            assert operation.result_item_id is not None
            result = await store.run_async(store.read_item, operation.result_item_id)
            assert result.payload["error_code"] == "cancelled"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_compaction_never_waits_synchronously_for_store(tmp_path, monkeypatch):
    from agent_runtime.harness import RolloutContextManager
    from tests.agent.harness.test_compaction_consistency import Accept, long_history, model_for, start

    async with await RuntimeState.open(tmp_path / "run.db") as store:
        thread, turn = await store.run_async(start, store, tmp_path)
        await store.run_async(long_history, store, turn)
        model, _ = model_for()
        runner = await store.run_async(
            lambda: TurnExecutor(
                thread_id=thread.thread_id,
                store=store,
                model=model,
                context_manager=RolloutContextManager(store),
                completion_gate=Accept(),
                worker_id="test-worker",
            )
        )
        context = await store.run_async(runner.restore_turn_context, turn.turn_id)
        original = store.run_sync
        loop_thread = threading.get_ident()

        def guarded(operation):
            assert threading.get_ident() != loop_thread, "compaction synchronously waited for Store"
            return original(operation)

        monkeypatch.setattr(store, "run_sync", guarded)
        result = await runner.run_turn(context, start_step=1)
        assert result.status == "completed"


@pytest.mark.anyio
async def test_cancellation_during_model_claim_does_not_invent_provider_io(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel

    calls = []
    entered, release = threading.Event(), threading.Event()
    original = RuntimeState.dispatch_model_attempt

    class Model(PublicHarnessModel):
        async def dispatch(self, prepared, *, event_sink=None):
            calls.append(prepared)
            return await super().dispatch(prepared, event_sink=event_sink)

    def blocked_claim(store, *args, **kwargs):
        value = original(store, *args, **kwargs)
        entered.set()
        assert release.wait(3)
        return value

    monkeypatch.setattr(RuntimeState, "dispatch_model_attempt", blocked_claim)
    agent = Agent(checkpoint_db=tmp_path / "run.db", workspace_path=tmp_path, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", Model)
    task = asyncio.create_task(agent.run("cancel before provider", require_workspace_change=False))
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == []
        async with await RuntimeState.open(tmp_path / "run.db", read_only=True) as reader:
            [operation] = await reader.run_async(reader.list_model_operations)
            [attempt] = await reader.run_async(reader.list_model_attempts, operation.operation_id)
            assert attempt.status == "failed"
            assert any(
                record.record_type == "model_attempt_rejected"
                for record in await reader.run_async(reader.list_turn_records, operation.turn_id)
            )
            budget = await reader.run_async(reader.read_budget_state, operation.turn_id)
            assert budget.uncertain.model_calls == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
@pytest.mark.parametrize("entry", ["sdk", "cli"])
async def test_tool_loop_public_entries_use_async_store_boundary(tmp_path, monkeypatch, entry):
    from agent_runtime import Agent
    from agent_runtime.cli import _run_facade_command
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall
    from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel, PublicHarnessModel

    (tmp_path / "present.txt").write_text("evidence")

    class Model(PublicHarnessModel):
        prepare = PatchThenAnswerModel.prepare

        async def dispatch(self, prepared, *, event_sink=None):
            if prepared.request_ref["request_id"].endswith(":step:1"):
                return HarnessModelResponse(
                    text="",
                    provider_response_id="read-response",
                    usage={"input_tokens": 2, "output_tokens": 1},
                    tool_calls=(HarnessToolCall(id="read", name="read_file", arguments={"path": "present.txt"}),),
                )
            return await super().dispatch(prepared, event_sink=event_sink)

    loop_thread = threading.get_ident()
    original = RuntimeState.run_sync

    def guarded(store, operation):
        assert threading.get_ident() != loop_thread, "public tool loop synchronously waited for Store"
        return original(store, operation)

    monkeypatch.setattr(RuntimeState, "run_sync", guarded)
    agent = Agent(checkpoint_db=tmp_path / "run.db", workspace_path=tmp_path, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", Model)
    if entry == "cli":
        result = await _run_facade_command(
            agent,
            task="read",
            files=(),
            max_tokens_total=None,
            interactive_approval=False,
            require_workspace_change=False,
        )
    else:
        result = await agent.run("read", require_workspace_change=False)
    assert result.status == "done"
    assert await agent.read_result(result.turn_id) == result


@pytest.mark.anyio
async def test_cancelled_snapshot_does_not_leave_trusted_turn_identity(tmp_path, monkeypatch):
    from agent_runtime.harness.tool_orchestrator import current_tool_turn_id

    async with await RuntimeState.open(tmp_path / "run.db") as store:
        orchestrator = object.__new__(ToolOrchestrator)
        orchestrator._store = store
        entered, release = threading.Event(), threading.Event()

        def snapshot(*args):
            entered.set()
            assert release.wait(3)
            return None

        monkeypatch.setattr(orchestrator, "_step_snapshot", snapshot)

        async def caller():
            with pytest.raises(asyncio.CancelledError):
                await orchestrator._execute_in_turn("turn-trusted", None)
            with pytest.raises(RuntimeError):
                current_tool_turn_id()

        task = asyncio.create_task(caller())
        try:
            while not entered.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            release.set()
            await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_session_close_cancellation_keeps_its_resource_owner(tmp_path):
    from agent_runtime.harness import Session
    from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel

    session = await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=PublicHarnessModel())
    entered, release = threading.Event(), threading.Event()

    class Connection:
        def __init__(self, connection):
            self.connection = connection

        def close(self):
            entered.set()
            assert release.wait(3)
            self.connection.close()

    await session.store.run_async(
        lambda: setattr(session.store.persistence, "_connection", Connection(session.store.persistence._connection))
    )
    task = asyncio.create_task(session.close())
    try:
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        assert not task.done() and not session._closed
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session._closed
        await session.close()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await session.close()


@pytest.mark.anyio
async def test_cancelled_memory_claim_releases_the_actual_lease(tmp_path, monkeypatch):
    from agent_runtime.harness.memory import update_memory

    async with await RuntimeState.open(tmp_path / "run.db") as store:
        original = store.claim_memory_job
        entered, release = threading.Event(), threading.Event()

        def blocked_claim(*args, **kwargs):
            claimed = original(*args, **kwargs)
            entered.set()
            assert release.wait(3)
            return claimed

        monkeypatch.setattr(store, "claim_memory_job", blocked_claim)
        task = asyncio.create_task(update_memory(store, workspace=tmp_path, model=None, binding={}))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await store.run_async(original, tmp_path, owner="next-owner", now=time.time(), lease_seconds=300)
            await store.run_async(store.release_memory_job, tmp_path, owner="next-owner")
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_cancelled_memory_turn_creation_is_tracked_and_terminal(tmp_path, monkeypatch):
    from agent_runtime.harness import memory as memory_module

    async with await RuntimeState.open(tmp_path / "run.db") as store:
        source = await store.run_async(store.create_thread, workspace=tmp_path)
        monkeypatch.setattr(
            memory_module,
            "_source",
            lambda *args: (
                source.thread_id,
                "revision",
                [{"item_id": "source-item", "thread_id": source.thread_id}],
                1,
                False,
            ),
        )
        original = store.start_turn
        entered, release = threading.Event(), threading.Event()

        def blocked_start(*args, **kwargs):
            turn = original(*args, **kwargs)
            entered.set()
            assert release.wait(3)
            return turn

        monkeypatch.setattr(store, "start_turn", blocked_start)
        task = asyncio.create_task(memory_module.update_memory(store, workspace=tmp_path, model=None, binding={}))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            [turn] = await store.run_async(store.list_turns)
            assert turn.status == "failed"
            jobs = await store.run_async(store.read_memory_jobs, tmp_path)
            assert jobs[source.thread_id]["turn_ids"] == [turn.turn_id]
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

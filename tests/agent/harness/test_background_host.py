import asyncio

import pytest

from agent_runtime import Agent
from agent_runtime.background import BackgroundHandle
from agent_runtime.harness import RuntimeStore, host_ops
from agent_runtime.host import BackgroundHost
from tests.agent.harness.test_subagent_budget import BlockingAnswerModel, SmallAnswerModel


def agent_at(path):
    return Agent(
        workspace_path=path, checkpoint_db=path / "runtime.db", model_session_path=None,
        enable_workspace_mcp=False,
    )


@pytest.mark.anyio
async def test_public_background_submission_attachment_and_replay(tmp_path):
    agent = agent_at(tmp_path)
    handle = await agent.start_background("inspect", input_id="stable", autostart=False, max_steps=3)
    repeated = await agent.start_background("inspect", input_id="stable", autostart=False, max_steps=3)
    assert repeated.turn_id == handle.turn_id
    assert (await handle.status())["status"] is None
    async with BackgroundHost(agent._harness_database(), model=SmallAnswerModel()) as host:
        await host.tick()
        result = await handle.wait(timeout=3)
        assert result["status"] == "completed"
        attached = BackgroundHandle(
            database=handle.database, workspace=tmp_path, thread_id=handle.thread_id,
            turn_id=handle.turn_id, autostart=False,
        )
        assert await attached.status() == await handle.status()
        events = await attached.events()
        assert events
        assert await attached.events(after=events[-1]["cursor"]) == ()
    async with await RuntimeStore.open(handle.database) as store:
        assert len(await store.run_async(store.list_turns)) == 1
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
async def test_wait_timeout_and_disconnect_do_not_cancel_owned_execution(tmp_path):
    agent = agent_at(tmp_path)
    handle = await agent.start_background("inspect", autostart=False)
    model = BlockingAnswerModel()
    async with BackgroundHost(handle.database, model=model) as host:
        await host.tick()
        await asyncio.wait_for(model.started.wait(), 3)
        with pytest.raises(TimeoutError):
            await handle.wait(timeout=0.01)
        assert (await handle.status())["status"] == "running"
        model.release.set()
        assert (await handle.wait(timeout=3))["status"] == "completed"


@pytest.mark.anyio
async def test_input_receipt_is_consumed_while_the_model_is_running(tmp_path):
    agent = agent_at(tmp_path)
    handle = await agent.start_background("inspect", autostart=False)
    model = BlockingAnswerModel()
    async with BackgroundHost(handle.database, model=model) as host:
        await host.tick()
        await asyncio.wait_for(model.started.wait(), 3)
        receipt = await handle.send("inspect tests", input_id="input-1")
        await host.tick()
        async with await RuntimeStore.open(handle.database) as store:
            assert (await store.run_async(host_ops.read_command, store, receipt.command_id)).status == "applied"
            assert await store.run_async(store.user_input_revision, handle.turn_id) == 2
        assert (await handle.send("inspect tests", input_id="input-1")).command_id == receipt.command_id
        model.release.set()
        await handle.wait(timeout=3)


@pytest.mark.anyio
async def test_cancel_before_start_never_constructs_model_or_turn(tmp_path):
    handle = await agent_at(tmp_path).start_background("inspect", autostart=False)
    await handle.cancel(input_id="cancel-1")
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await host.tick()
        status = await handle.wait(timeout=2)
        assert status["status"] is None
        assert status["queue_status"] == "rejected"


@pytest.mark.anyio
async def test_only_one_host_can_own_a_database(tmp_path):
    handle = await agent_at(tmp_path).start_background("inspect", autostart=False)
    async with BackgroundHost(handle.database, model=SmallAnswerModel()):
        with pytest.raises(RuntimeError, match="host already"):
            async with BackgroundHost(handle.database, model=SmallAnswerModel()):
                pass


@pytest.mark.anyio
async def test_claim_recovery_does_not_repeat_a_completed_turn(tmp_path):
    handle = await agent_at(tmp_path).start_background("inspect", autostart=False)
    model = SmallAnswerModel()
    async with BackgroundHost(handle.database, model=model) as host:
        await host.tick()
        await handle.wait(timeout=3)
    async with BackgroundHost(handle.database, model=model) as host:
        await host.tick()
        assert (await handle.status())["status"] == "completed"
    async with await RuntimeStore.open(handle.database) as store:
        assert len(await store.run_async(store.list_model_operations, handle.turn_id)) == 1


@pytest.mark.anyio
async def test_host_shutdown_before_turn_creation_keeps_the_submission_pending(tmp_path, monkeypatch):
    from agent_runtime.harness import Session

    handle = await agent_at(tmp_path).start_background("inspect", autostart=False)
    entered = asyncio.Event()
    original = Session.open

    async def blocked_open(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(Session, "open", blocked_open)
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await host.tick()
        await asyncio.wait_for(entered.wait(), 3)
    assert (await handle.status())["queue_status"] == "pending"
    assert (await handle.status())["status"] is None
    monkeypatch.setattr(Session, "open", original)
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await host.tick()
        assert (await handle.wait(timeout=3))["status"] == "completed"


@pytest.mark.anyio
async def test_approval_reply_is_bound_to_request_and_replays_once(tmp_path):
    from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

    (tmp_path / "value.txt").write_text("before")
    handle = await agent_at(tmp_path).start_background("change value.txt", autostart=False)
    async with BackgroundHost(handle.database, model=PatchThenAnswerModel()) as host:
        serving = asyncio.create_task(host.serve(idle_timeout=None))
        try:
            paused = await handle.wait(timeout=3)
            request_id = paused["recovery"]["request_id"]
            assert paused["recovery"]["reason_code"] == "tool_approval_required"
            assert (tmp_path / "value.txt").read_text() == "before"
            receipt = await handle.respond(request_id, "allow_once", input_id="approval-once")
            async with asyncio.timeout(5):
                while (await handle.status())["status"] != "completed":
                    await asyncio.sleep(0.01)
            assert (tmp_path / "value.txt").read_text() == "after"
            repeated = await handle.respond(request_id, "allow_once", input_id="approval-once")
            assert repeated.command_id == receipt.command_id
            with pytest.raises(ValueError, match="conflict"):
                await handle.respond(request_id, "deny", input_id="approval-once")
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


@pytest.mark.anyio
async def test_host_recovers_reserved_child_and_wakes_the_waiting_parent(tmp_path):
    from agent_runtime.harness import Session, turn_ops
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION

    agent = agent_at(tmp_path)
    agent._model_factory = SmallAnswerModel
    handle = await agent.start_background("inspect", autostart=False, max_tokens_total=100)
    session = await Session.open(agent=agent, thread_id=handle.thread_id)
    try:
        binding = dict(await session.store.run_async(
            session.binding_provider.snapshot, thread_id=handle.thread_id, turn_id=handle.turn_id
        ))
        binding.update(execution_ownership=OWNERSHIP_REVISION, model_token_budget_total=100)
        await session.store.run_async(
            turn_ops.start_turn, session.store, thread_id=handle.thread_id, turn_id=handle.turn_id,
            user_message="inspect", binding_manifest=binding,
        )
        child = await session._prepare_child(
            parent_turn_id=handle.turn_id, user_message="inspect files", max_steps=3, max_tokens_total=30,
            workspace_mode="read-only",
        )
        await session.store.run_async(
            turn_ops.pause_turn, session.store, turn_id=handle.turn_id, reason="Attached child needs attention",
            reason_code="attached_children", input_revision=1,
        )
    finally:
        await session.close(preserve_children=True)
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        serving = asyncio.create_task(host.serve(idle_timeout=None))
        try:
            async with asyncio.timeout(5):
                while (await handle.status())["status"] != "completed":
                    await asyncio.sleep(0.01)
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
        edge = await host.store.run_async(host.store.read_child_budget_allocation, child["turn_id"])
        assert edge["status"] == "settled"
        assert (await host.store.run_async(host.store.read_turn, child["turn_id"])).status == "completed"
        assert (await host.store.run_async(host.store.verify)).valid


@pytest.mark.anyio
async def test_followup_receipt_replays_after_its_predecessor_is_no_longer_head(tmp_path):
    agent = agent_at(tmp_path)
    first = await agent.start_background("first", input_id="first", autostart=False)
    async with BackgroundHost(first.database, model=SmallAnswerModel()) as host:
        serving = asyncio.create_task(host.serve(idle_timeout=None))
        try:
            await first.wait(timeout=3)
            second = await agent.start_background(
                "follow up", previous_turn_id=first.turn_id, input_id="second", autostart=False
            )
            assert second.thread_id == first.thread_id
            await second.wait(timeout=3)
            repeated = await agent.start_background(
                "follow up", previous_turn_id=first.turn_id, input_id="second", autostart=False
            )
            assert repeated.turn_id == second.turn_id
            with pytest.raises(ValueError, match="terminal thread head"):
                await agent.start_background("another", previous_turn_id=first.turn_id, autostart=False)
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


@pytest.mark.anyio
@pytest.mark.parametrize("options", [
    {"input_id": ""}, {"max_steps": -1}, {"max_tokens_total": 0},
    {"max_cost_micros": True}, {"acceptance_command": " "},
])
async def test_invalid_submission_never_enters_the_queue(tmp_path, options):
    agent = agent_at(tmp_path)
    with pytest.raises(ValueError):
        await agent.start_background("inspect", autostart=False, **options)
    async with await RuntimeStore.open(agent._harness_database()) as store:
        assert await store.run_async(host_ops.list_commands, store) == ()


@pytest.mark.anyio
async def test_attachment_rejects_a_turn_from_another_thread_in_same_workspace(tmp_path):
    one = await agent_at(tmp_path).start_background("one", autostart=False)
    two = await agent_at(tmp_path).start_background("two", autostart=False)
    forged = BackgroundHandle(
        database=one.database, workspace=tmp_path, thread_id=one.thread_id, turn_id=two.turn_id, autostart=False,
    )
    with pytest.raises(PermissionError):
        await forged.status()
    with pytest.raises(PermissionError):
        await forged.cancel()


@pytest.mark.anyio
@pytest.mark.parametrize("input_id", ["cancel-running", "x" * 500])
async def test_cancel_running_driver_finishes_canonical_cancellation(tmp_path, input_id):
    handle = await agent_at(tmp_path).start_background("inspect", autostart=False)
    model = BlockingAnswerModel()
    async with BackgroundHost(handle.database, model=model) as host:
        serving = asyncio.create_task(host.serve(idle_timeout=None))
        try:
            await asyncio.wait_for(model.started.wait(), 3)
            await handle.cancel(input_id=input_id)
            async with asyncio.timeout(3):
                while (await handle.status())["status"] != "cancelled":
                    await asyncio.sleep(0.01)
            assert not model.release.is_set()
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


def test_configuration_keeps_request_preferences_and_permission_ceiling(tmp_path):
    from agent_runtime.background import freeze_configuration, permission_ceiling, restore_configuration

    agent = agent_at(tmp_path)
    agent.configure_request(max_output_tokens=777)
    configuration = freeze_configuration(agent)
    restored = restore_configuration(configuration)
    assert restored.request_settings()["preferences"]["max_output_tokens"] == 777
    assert permission_ceiling(configuration) == agent.effective_permissions()


@pytest.mark.anyio
async def test_followup_inherits_saved_mode_and_explicit_ask_downgrades_it(tmp_path):
    from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

    agent = agent_at(tmp_path)
    agent.approval_mode = "auto"
    first = await agent.start_background("inspect", autostart=False)
    async with BackgroundHost(first.database, model=SmallAnswerModel()) as host:
        await host.tick()
        await first.wait(timeout=3)
    (tmp_path / "value.txt").write_text("before")
    agent.approval_mode = None
    second = await agent.start_background("patch", previous_turn_id=first.turn_id, autostart=False)
    async with BackgroundHost(first.database, model=PatchThenAnswerModel()) as host:
        serving = asyncio.create_task(host.serve(idle_timeout=None))
        try:
            assert (await second.wait(timeout=3))["status"] == "completed"
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
    (tmp_path / "value.txt").write_text("before")
    agent.approval_mode = "ask"
    third = await agent.start_background("patch again", previous_turn_id=second.turn_id, autostart=False)
    async with BackgroundHost(first.database, model=PatchThenAnswerModel()) as host:
        await host.tick()
        assert (await third.wait(timeout=3))["recovery"]["reason_code"] == "tool_approval_required"
        assert (tmp_path / "value.txt").read_text() == "before"


@pytest.mark.anyio
async def test_cancelled_submission_cannot_be_revived_by_queued_resume_or_input(tmp_path):
    handle = await agent_at(tmp_path).start_background("inspect", autostart=False)
    message = await handle.send("additional input")
    await handle.cancel()
    resume = await handle.resume()
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await asyncio.wait_for(host.serve(idle_timeout=0.1), 3)
        assert await host.store.run_async(host.store.list_turns) == ()
        assert (await host.store.run_async(host_ops.read_command, host.store, message.command_id)).status == "rejected"
        assert (await host.store.run_async(host_ops.read_command, host.store, resume.command_id)).status == "rejected"


@pytest.mark.anyio
async def test_terminal_resume_does_not_restore_provider_configuration(tmp_path, monkeypatch):
    handle = await agent_at(tmp_path).start_background("inspect", autostart=False)
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await host.tick()
        await handle.wait(timeout=3)
    receipt = await handle.resume()

    def forbidden(*_args):
        raise AssertionError("terminal resume must not restore model/provider configuration")

    monkeypatch.setattr("agent_runtime.host.restore_configuration", forbidden)
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await asyncio.wait_for(host.serve(idle_timeout=0.1), 3)
        assert (await host.store.run_async(host_ops.read_command, host.store, receipt.command_id)).status == "applied"


@pytest.mark.anyio
async def test_host_guard_does_not_wait_on_the_host_connection_while_holding_a_write_lock(tmp_path):
    import threading

    handle = await agent_at(tmp_path).start_background("inspect", autostart=False)
    entered, writer_started = threading.Event(), threading.Event()
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        original = host._guard

        def guard(claim, executing_store):
            assert executing_store is not host.store
            assert executing_store.persistence._connection.in_transaction
            if not entered.is_set():
                entered.set()
                assert writer_started.wait(3)
            original(claim, executing_store)

        def competing_writer():
            writer_started.set()
            with host.store.tx.transaction():
                pass

        host._guard = guard
        await host.tick()
        assert await asyncio.to_thread(entered.wait, 3)
        writer = asyncio.create_task(host.store.run_async(competing_writer))
        try:
            assert (await handle.wait(timeout=3))["status"] == "completed"
            await asyncio.wait_for(writer, 3)
        finally:
            writer_started.set()
            await asyncio.gather(writer, return_exceptions=True)


@pytest.mark.anyio
async def test_offline_workspace_preserves_submission_then_executes_when_restored(tmp_path):
    workspace = tmp_path / "checkout"
    workspace.mkdir()
    handle = await Agent(
        workspace_path=workspace, checkpoint_db=tmp_path / "runtime.db", model_session_path=None,
        enable_workspace_mcp=False,
    ).start_background("inspect", autostart=False)
    offline = tmp_path / "offline"
    workspace.rename(offline)
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        serving = asyncio.create_task(host.serve(idle_timeout=None))
        try:
            async with asyncio.timeout(3):
                while (await handle.status())["receipt"].get("reason") != "workspace_offline":
                    await asyncio.sleep(0.01)
            assert (await handle.status())["status"] is None
            offline.rename(workspace)
            assert (await handle.wait(timeout=8))["status"] == "completed"
        finally:
            if offline.exists():
                offline.rename(workspace)
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


@pytest.mark.anyio
async def test_child_cancel_reaches_the_live_driver_owned_by_a_parent_session(tmp_path):
    from agent_runtime.harness import Session, turn_ops
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION

    agent = agent_at(tmp_path)
    model = BlockingAnswerModel()
    agent._model_factory = lambda: model
    handle = await agent.start_background("inspect", autostart=False)
    async with BackgroundHost(handle.database) as host:
        session = await Session.open(agent=agent, thread_id=handle.thread_id)
        host._sessions[handle.turn_id] = session
        try:
            binding = dict(await session.store.run_async(
                session.binding_provider.snapshot, thread_id=handle.thread_id, turn_id=handle.turn_id
            ))
            binding["execution_ownership"] = OWNERSHIP_REVISION
            await session.store.run_async(
                turn_ops.start_turn, session.store, thread_id=handle.thread_id, turn_id=handle.turn_id,
                user_message="inspect", binding_manifest=binding,
            )
            child = await session.subagents.spawn(
                parent_turn_id=handle.turn_id, user_message="inspect tests", workspace_mode="read-only"
            )
            await asyncio.wait_for(model.started.wait(), 3)
            attached = agent.attach_background(
                thread_id=str(child["thread_id"]), turn_id=str(child["child_turn_id"]), autostart=False,
            )
            await attached.cancel(input_id="cancel-child")
            claim = await host.store.run_async(host_ops.claim_next, host.store, owner=host.owner, control=True)
            await host._control(claim)
            await asyncio.wait_for(asyncio.gather(
                session._subagent_tasks[str(child["child_turn_id"])], return_exceptions=True
            ), 3)
            assert await session.store.run_async(session.store.turn_cancellation_pending, str(child["child_turn_id"]))
            assert not await session.store.run_async(session.store.turn_cancellation_pending, handle.turn_id)
            assert not model.release.is_set()
        finally:
            host._sessions.pop(handle.turn_id)
            await session.close(preserve_children=True)


@pytest.mark.anyio
async def test_host_failure_records_bounded_redacted_diagnostics(tmp_path, monkeypatch):
    import json

    handle = await agent_at(tmp_path).start_background("inspect", autostart=False)
    secret = "synthetic-secret-" + "x" * 16
    monkeypatch.setenv("PRAXIS_TEST_SECRET", secret)

    def invalid_configuration(*_args):
        raise TypeError("configuration rejected: " + secret)

    monkeypatch.setattr("agent_runtime.host.restore_configuration", invalid_configuration)
    async with BackgroundHost(handle.database) as host:
        await asyncio.wait_for(host.serve(idle_timeout=0.1), 3)
        state = await handle.status()
        assert state["queue_status"] == "blocked"
        diagnostic = state["receipt"]["diagnostic"]
        assert diagnostic["exception_type"] == "TypeError"
        assert "[REDACTED]" in diagnostic["message"]
        assert len(diagnostic["traceback"]) <= 12
        assert secret not in json.dumps(state)


@pytest.mark.anyio
async def test_stop_during_claim_returns_unstarted_submission_to_queue(tmp_path, monkeypatch):
    handle = await agent_at(tmp_path).start_background("inspect", input_id="during-stop", autostart=False)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    original_claim = host_ops.claim_next

    def claim(*args, **kwargs):
        command = original_claim(*args, **kwargs)
        if command is not None:
            loop.call_soon_threadsafe(stop.set)
        return command

    monkeypatch.setattr(host_ops, "claim_next", claim)
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await asyncio.wait_for(host.serve(stop=stop), 3)
        assert not host._drivers
        state = await handle.status()
        assert state["queue_status"] == "pending"
        assert state["status"] is None

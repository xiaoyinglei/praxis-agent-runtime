from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, RolloutContextManager, Session
from agent_runtime.tools.permissions import ToolExecutionContext
from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel
from tests.agent.harness.test_session import ZeroDeltaToolThenAnswerModel, _read_file_tool


class SteeringModel(PublicHarnessModel):
    def __init__(self):
        self.requests = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def prepare(self, request):
        self.requests.append(request)
        return super().prepare(request)

    async def dispatch(self, prepared, *, delta_sink=None):
        if len(self.requests) == 1:
            self.started.set()
            await self.release.wait()
            return HarnessModelResponse(text="old answer", provider_response_id=None, usage={})
        return HarnessModelResponse(text="updated answer", provider_response_id=None, usage={})


@pytest.mark.anyio
async def test_sampling_inputs_are_durable_ordered_and_replace_stale_final(tmp_path):
    model = SteeringModel()
    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=model) as session:
        running = asyncio.create_task(session.submit("initial task"))
        await model.started.wait()
        turn_id = session.store.read_thread(session.thread_id).active_turn_id
        try:
            first = await session.steer("also test recovery", expected_turn_id=turn_id)
            second = await session.steer("use the existing API", expected_turn_id=turn_id)
            assert first.sequence < second.sequence
            assert session.input_status(first.item_id) == "pending"
            assert session.store.verify().valid
            with pytest.raises(RuntimeError, match="Turn"):
                await session.steer("wrong target", expected_turn_id="different-turn")
        finally:
            model.release.set()
        result = await running
        assert result.answer == "updated answer"
        assert session.input_status(first.item_id) == "applied"
        assert session.input_status(second.item_id) == "applied"
        assert [m.content for m in model.requests[-1].messages if m.role == "user"] == [
            "initial task", "also test recovery", "use the existing API",
        ]
        assert not any(i.kind == "agent_message" and i.payload.get("text") == "old answer"
                       for i in session.store.list_items(turn_id))
        from agent_runtime.history import read_chat_messages

        replay = read_chat_messages(session.database, thread_id=session.thread_id, workspace=tmp_path)
        assert ("answer", "old answer") not in replay
        assert any(kind == "notice" and "已生效" in text for kind, text in replay)
        await session.submit("explain the result")
        assert "use the existing API" in [m.content for m in model.requests[-1].messages]
        assert "updated answer" in [m.content for m in model.requests[-1].messages]
        with pytest.raises(RuntimeError, match="Turn"):
            await session.steer("too late", expected_turn_id=turn_id)


@pytest.mark.anyio
async def test_received_input_survives_unknown_provider_and_reopen(tmp_path):
    from agent_runtime.harness import RolloutStore

    class UnknownModel(SteeringModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            self.started.set()
            await self.release.wait()
            raise ConnectionError("provider outcome unknown")

    model = UnknownModel()
    database = tmp_path / "run.db"
    async with await Session.open(database=database, workspace=tmp_path, model=model) as session:
        running = asyncio.create_task(session.submit("initial task"))
        await model.started.wait()
        turn_id = session.store.read_thread(session.thread_id).active_turn_id
        try:
            receipt = await session.steer("new requirement", expected_turn_id=turn_id)
        finally:
            model.release.set()
        assert (await running).status == "paused"
        assert session.input_status(receipt.item_id) == "pending"
        from agent_runtime.history import read_chat_messages

        replay = read_chat_messages(database, thread_id=session.thread_id, workspace=tmp_path)
        assert any(kind == "notice" and "待生效" in text for kind, text in replay)
    with RolloutStore(database) as store:
        assert store.read_item(receipt.item_id).payload["text"] == "new requirement"
        assert store.verify().valid
        assert "new requirement" in [m.content for m in RolloutContextManager(store).build(turn_id)]


class ToolSteeringModel(ZeroDeltaToolThenAnswerModel):
    def __init__(self):
        super().__init__()
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return super().prepare(request)

    async def dispatch(self, prepared, *, delta_sink=None):
        response = await super().dispatch(prepared, delta_sink=delta_sink)
        if response.tool_calls:
            response = replace(response, tool_calls=(
                response.tool_calls[0],
                HarnessToolCall(id="call-2", name="read_file", arguments={"path": "second.txt"}),
            ))
        return response


@pytest.mark.anyio
async def test_input_during_tool_preserves_result_but_prevents_next_old_call(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()
    invoked = []

    async def read(arguments):
        invoked.append(arguments["path"])
        started.set()
        await release.wait()
        return {"text": "real tool result"}

    tool = replace(_read_file_tool(), run=read, concurrency_safe=False)
    model = ToolSteeringModel()
    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path,
                                 model=model, tools={"read_file": tool}) as session:
        original_dispatch = model.dispatch

        async def compacting_dispatch(prepared, *, delta_sink=None):
            response = await original_dispatch(prepared, delta_sink=delta_sink)
            if not response.tool_calls:
                manager = RolloutContextManager(session.store)
                turn_id = session.store.read_thread(session.thread_id).active_turn_id
                manager.compact_for_budget(turn_id=turn_id, retained_tail_messages=1)
                assert [m.content for m in manager.build(turn_id) if m.role == "user"] == [
                    "read files", "only read the first file",
                ]
            return response

        model.dispatch = compacting_dispatch
        running = asyncio.create_task(session.submit("read files"))
        await started.wait()
        turn_id = session.store.read_thread(session.thread_id).active_turn_id
        try:
            receipt = await session.steer("only read the first file", expected_turn_id=turn_id)
        finally:
            release.set()
        result = await running
        assert result.status == "done"
        assert invoked == ["README.md"]
        assert session.input_status(receipt.item_id) == "applied"
        messages = model.requests[-1].messages
        assert [m.role for m in messages if m.role != "context"] == [
            "user", "assistant", "tool", "tool", "user",
        ]
        assert "superseded" in next(m.content for m in messages if m.tool_call_id == "call-2")
        assert session.store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["continue", "allow_once"])
async def test_steering_cannot_approve_paused_old_action(tmp_path, action):
    invoked = []
    tool = replace(_read_file_tool(), run=lambda a: invoked.append(a) or {"text": "read"})
    model = ToolSteeringModel()
    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=model,
                                 tools={"read_file": tool}, tool_execution_context=ToolExecutionContext(
                                     require_confirmation_for=frozenset({"read_file"}),
                                 )) as session:
        result = await session.submit("read files")
        assert result.status == "paused"
        receipt = await session.steer("do not read files; explain instead", expected_turn_id=result.turn_id)
        result = await session.resume(result.turn_id, action)
        assert result.status == "done"
        assert invoked == []
        assert session.input_status(receipt.item_id) == "applied"
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_input_in_final_decision_callback_blocks_old_completion(tmp_path):
    from agent_runtime.harness import RolloutStore

    class FinalModel(SteeringModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            return HarnessModelResponse(text="first" if len(self.requests) == 1 else "updated",
                                        provider_response_id=None, usage={})

    model = FinalModel()
    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=model) as session:
        original = session.store.record_completion_decision
        received = []

        def completion(**kwargs):
            item = original(**kwargs)
            if not received:
                with RolloutStore(session.database) as external:
                    received.append(external.receive_turn_input(
                        thread_id=session.thread_id, turn_id=kwargs["turn_id"], text="last moment requirement",
                    ))
            return item

        session.store.record_completion_decision = completion
        result = await session.submit("initial task")
        assert result.answer == "updated"
        assert session.input_status(received[0].item_id) == "applied"


@pytest.mark.anyio
@pytest.mark.parametrize("delay_executor", [False, True])
async def test_input_after_durable_tool_start_before_runner_prevents_invocation(tmp_path, monkeypatch, delay_executor):
    from agent_runtime.streaming.events import EventType, TurnItemKind

    invoked = []
    tool = replace(_read_file_tool(), run=lambda a: invoked.append(a) or {"text": "read"})
    model = ToolSteeringModel()
    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=model,
                                 tools={"read_file": tool}) as session:
        class Sink:
            submitted = False

            async def emit(self, event):
                if (not self.submitted and event.type is EventType.ITEM_STARTED
                        and event.item_kind is TurnItemKind.TOOL):
                    self.submitted = True
                    await session.steer("explain instead", expected_turn_id=event.turn_id)

        sink = Sink()
        original_run = session.store.run_async

        async def delay_constructor(operation, *args, **kwargs):
            if delay_executor and sink.submitted and operation == session._executor:
                await asyncio.sleep(0.02)
            return await original_run(operation, *args, **kwargs)

        monkeypatch.setattr(session.store, "run_async", delay_constructor)
        result = await session.submit("read files", event_sink=sink)
        assert sink.submitted
        assert result.status == "done"
        assert invoked == []
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_input_after_request_preparation_does_not_send_old_request(tmp_path):
    from agent_runtime.harness import RolloutStore
    model = SteeringModel()
    model.release.set()
    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=model) as session:
        original = session.store.prepare_model_operation
        received = []

        def prepare(**kwargs):
            operation = original(**kwargs)
            if not received:
                with RolloutStore(session.database) as external:
                    received.append(external.receive_turn_input(
                        thread_id=session.thread_id, turn_id=kwargs["turn_id"], text="before dispatch requirement",
                    ))
            return operation

        session.store.prepare_model_operation = prepare
        result = await session.submit("initial task")
        assert result.answer == "updated answer"
        assert len([a for op in session.store.list_model_operations(result.turn_id)
                    for a in session.store.list_model_attempts(op.operation_id) if a.status == "completed"]) == 1
        assert session.input_status(received[0].item_id) == "applied"
        assert session.store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("tool_response", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
async def test_reopen_after_committed_response_does_not_revive_old_work(tmp_path, monkeypatch, tool_response, legacy):
    from agent_runtime.harness import TurnExecutor

    invoked = []
    tool = replace(_read_file_tool(), run=lambda a: invoked.append(a) or {"text": "read"})
    model = ToolSteeringModel() if tool_response else SteeringModel()
    if not tool_response:
        model.release.set()
    options = dict(database=tmp_path / "run.db", workspace=tmp_path, model=model, tools={"read_file": tool})
    async with await Session.open(**options) as session:
        if legacy:
            original = session.store.prepare_model_operation

            def prepare(**kwargs):
                ref = {k: v for k, v in kwargs["request_ref"].items() if k != "input_revision"}
                ref["step_snapshot"] = {k: v for k, v in ref["step_snapshot"].items() if k != "input_revision"}
                return original(**{**kwargs, "request_ref": ref})

            session.store.prepare_model_operation = prepare
        async def crash(*args, **kwargs):
            raise RuntimeError("crash after response commit")

        with monkeypatch.context() as patch:
            patch.setattr(TurnExecutor, "_handle_model_response", crash)
            with pytest.raises(RuntimeError, match="crash after response"):
                await session.submit("initial task")
        turn_id = session.store.read_thread(session.thread_id).active_turn_id
        thread_id = session.thread_id
        receipt = await session.steer("new requirement after disconnect", expected_turn_id=turn_id)
    async with await Session.open(**options, thread_id=thread_id) as reopened:
        result = await reopened.resume(turn_id, "continue")
        assert result.status == "done"
        assert result.answer != "old answer"
        assert invoked == []
        assert reopened.input_status(receipt.item_id) == "applied"
        assert "new requirement after disconnect" in [m.content for m in model.requests[-1].messages]
        assert reopened.store.verify().valid


@pytest.mark.anyio
async def test_reopen_after_approval_commit_drops_stale_authorization(tmp_path):
    invoked = []
    tool = replace(_read_file_tool(), run=lambda a: invoked.append(a) or {"text": "read"})
    model = ToolSteeringModel()
    options = dict(database=tmp_path / "run.db", workspace=tmp_path, model=model,
                   tools={"read_file": tool}, tool_execution_context=ToolExecutionContext(
                       require_confirmation_for=frozenset({"read_file"}),
                   ))
    async with await Session.open(**options) as session:
        result = await session.submit("read files")
        assert result.status == "paused"
        session.store.resolve_tool_approval(turn_id=result.turn_id, decision="approve")
        receipt = await session.steer("new task; explain instead", expected_turn_id=result.turn_id)
        thread_id = session.thread_id
    async with await Session.open(**options, thread_id=thread_id) as reopened:
        result = await reopened.resume(result.turn_id, "continue")
        assert result.status == "done"
        assert invoked == []
        assert reopened.input_status(receipt.item_id) == "applied"
        assert reopened.store.verify().valid


@pytest.mark.anyio
async def test_exhausted_budget_keeps_input_as_explicitly_unconsumed(tmp_path):
    model = SteeringModel()
    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=model,
                                 max_steps=1) as session:
        running = asyncio.create_task(session.submit("initial task"))
        await model.started.wait()
        turn_id = session.store.read_thread(session.thread_id).active_turn_id
        receipt = await session.steer("new requirement", expected_turn_id=turn_id)
        model.release.set()
        assert (await running).status == "failed"
        assert session.input_status(receipt.item_id) == "unconsumed"


@pytest.mark.anyio
async def test_streaming_input_prevents_old_tools_and_preserves_complete_context(tmp_path):
    from agent_runtime.harness import HarnessModelDelta
    from agent_runtime.streaming.events import EventType

    invoked = []
    tool = replace(_read_file_tool(), run=lambda a: invoked.append(a) or {"text": "read"})

    class StreamingModel(ToolSteeringModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            if not self.dispatch_count and delta_sink is not None:
                await delta_sink(HarnessModelDelta(channel="text", content="old streamed plan"))
            response = await super().dispatch(prepared, delta_sink=delta_sink)
            if response.tool_calls:
                return replace(response, tool_calls=tuple(
                    replace(response.tool_calls[0], id=f"call-{index}") for index in range(3)
                ))
            return response

    model = StreamingModel()
    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=model,
                                 tools={"read_file": tool}) as session:
        class Sink:
            receipt = None

            async def emit(self, event):
                if event.type is EventType.ITEM_DELTA and self.receipt is None:
                    self.receipt = await session.steer("explain instead", expected_turn_id=event.turn_id)

        sink = Sink()
        result = await session.submit("read files", event_sink=sink)
        assert result.status == "done"
        assert invoked == []
        assert session.input_status(sink.receipt.item_id) == "applied"
        assert "explain instead" in [m.content for m in model.requests[-1].messages]
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_reopen_between_superseding_request_and_next_step(tmp_path):
    from agent_runtime.harness import RolloutStore

    model = SteeringModel()
    model.release.set()
    options = dict(database=tmp_path / "run.db", workspace=tmp_path, model=model)
    async with await Session.open(**options) as session:
        original = session.store.prepare_model_operation
        supersede = session.store.supersede_prepared_model
        received = []

        def prepare(**kwargs):
            operation = original(**kwargs)
            if not received:
                with RolloutStore(session.database) as other:
                    received.append(other.receive_turn_input(thread_id=session.thread_id,
                                    turn_id=kwargs["turn_id"], text="new requirement"))
            return operation

        def crash(operation_id):
            supersede(operation_id)
            raise RuntimeError("crash after supersede")

        session.store.prepare_model_operation = prepare
        session.store.supersede_prepared_model = crash
        with pytest.raises(RuntimeError, match="crash after supersede"):
            await session.submit("initial task")
        thread_id = session.thread_id
        turn_id = session.store.read_thread(thread_id).active_turn_id
    async with await Session.open(**options, thread_id=thread_id) as reopened:
        result = await reopened.resume(turn_id, "continue")
        assert result.status == "done"
        assert reopened.input_status(received[0].item_id) == "applied"


@pytest.mark.anyio
async def test_external_input_during_context_capture_is_not_falsely_acknowledged(tmp_path):
    from agent_runtime.harness import RolloutStore

    model = SteeringModel()
    model.release.set()
    async with await Session.open(database=tmp_path / "run.db", workspace=tmp_path, model=model) as session:
        original = session.context_manager.build
        received = []

        def build(turn_id):
            messages = original(turn_id)
            if not received:
                with RolloutStore(session.database) as other:
                    received.append(other.receive_turn_input(thread_id=session.thread_id, turn_id=turn_id,
                                                            text="requirement during capture"))
            return messages

        session.context_manager.build = build
        await session.submit("initial task")
        dispatched = [operation for operation in session.store.list_model_operations()
                      if operation.status == "completed"]
        assert dispatched
        assert all(
            "requirement during capture" in [m["content"] for m in operation.request_ref["step_snapshot"]["messages"]]
            for operation in dispatched
        )


@pytest.mark.anyio
async def test_input_after_compaction_commit_reprepares_instead_of_reporting_corruption(tmp_path):
    from agent_runtime.harness import RolloutStore
    from tests.agent.harness.test_compaction_consistency import long_history
    from tests.agent.harness.test_context_capacity_policy import capacity_runner

    with RolloutStore(tmp_path / "run.db") as store:
        runner, turn, gateway = capacity_runner(store, tmp_path, messages=5)
        long_history(store, turn)
        manager = runner._context_manager
        original = manager.commit_compaction
        received = []

        def commit(candidate):
            item = original(candidate)
            if not received:
                with RolloutStore(tmp_path / "run.db") as external:
                    received.append(external.receive_turn_input(thread_id=turn.thread_id, turn_id=turn.turn_id,
                                                               text="requirement at compaction commit"))
            return item

        manager.commit_compaction = commit
        result = await runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1)
        assert result.status == "completed"
        assert received
        requests = [r for r in gateway.requests if ":context-summary:" not in r.request_id]
        assert requests
        assert "requirement at compaction commit" in [m.content for m in requests[-1].messages]
        assert store.verify().valid


@pytest.mark.anyio
async def test_public_agent_stream_accepts_steering_without_opening_a_second_model(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from agent_runtime.streaming.events import EventType

    model = SteeringModel()
    agent = Agent(checkpoint_db=tmp_path / "run.db", workspace_path=tmp_path)
    opened = []

    def factory():
        opened.append(True)
        return model

    monkeypatch.setattr(agent, "_model_factory", factory)
    receipt = None
    try:
        async for event in agent.stream("initial task", require_workspace_change=False):
            if event.type is EventType.TURN_STARTED:
                receipt = await agent.steer("public SDK requirement", expected_turn_id=event.turn_id)
                model.release.set()
    finally:
        model.release.set()
    assert receipt is not None
    assert opened == [True]
    assert agent.input_status(receipt.item_id) == "applied"
    assert "public SDK requirement" in [m.content for m in model.requests[-1].messages]

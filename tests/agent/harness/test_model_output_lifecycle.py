from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_runtime.harness import RuntimeState
from agent_runtime.harness.model_outputs import (
    close_model_outputs,
    complete_model_output,
    read_model_outputs,
    start_model_output,
)
from agent_runtime.modeling.events import ModelOutput


class _EventModel:
    def __init__(self, dispatch):
        self.dispatch_events = dispatch

    def snapshot(self, **kwargs):
        return {"model_id": "output-test"}

    def ensure_available(self, *args, **kwargs):
        pass

    def prepare(self, request):
        from agent_runtime.harness import PreparedModelCall

        return PreparedModelCall(
            request_hash="r", context_hash="c", tool_hash="t", wire_hash="w", request_ref={"request_id": "request"}
        )

    async def dispatch(self, prepared, *, event_sink=None):
        return await self.dispatch_events(event_sink)


def _attempt(state: RuntimeState, workspace: Path):
    thread = state.create_thread(workspace=workspace)
    turn = state.start_turn(thread_id=thread.thread_id, user_message="inspect and fix", binding_manifest={})
    operation = state.prepare_model_operation(
        turn_id=turn.turn_id,
        request_hash="request",
        context_hash="context",
        tool_hash="tools",
        wire_hash="wire",
        request_ref={"request_id": "request"},
    )
    attempt = state.dispatch_model_attempt(operation.operation_id, worker_id="worker", lease_seconds=30)
    return operation, attempt


def test_completed_output_survives_later_transport_failure_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "rollout.sqlite3"
    with RuntimeState.open_sync(path) as state:
        operation, attempt = _attempt(state, tmp_path)
        identity = dict(
            operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
        start_model_output(state, **identity, output=ModelOutput("a", "text"))
        complete_model_output(state, **identity, output=ModelOutput("a", "text", content="checked"))
        start_model_output(state, **identity, output=ModelOutput("b", "text"))
        close_model_outputs(state, attempt_id=attempt.attempt_id, status="outcome_unknown", error="connection lost")
        records = [r for r in state.list_records(operation.thread_id) if r.record_type == "item_completed"]
        completed = [r for r in records if r.payload.get("kind") == "model_output"]
        assert [(r.payload["output_id"], r.payload["status"]) for r in completed] == [
            ("a", "success"),
            ("b", "outcome_unknown"),
        ]
        assert completed[0].payload["public_item_id"] != completed[1].payload["public_item_id"]
    with RuntimeState.open_sync(path) as state:
        outputs = read_model_outputs(state, attempt_id=attempt.attempt_id)
        assert [(o.output_id, o.content) for o in outputs] == [("a", "checked"), ("b", "")]


def test_old_attempt_cannot_start_or_complete_an_output(tmp_path: Path) -> None:
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as state:
        operation, attempt = _attempt(state, tmp_path)
        state.mark_model_attempt_unknown(
            operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
        identity = dict(
            operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
        with pytest.raises(RuntimeError, match="active dispatched"):
            start_model_output(state, **identity, output=ModelOutput("late", "text"))


def test_delta_identity_check_does_not_enumerate_durable_history(tmp_path: Path, monkeypatch) -> None:
    from agent_runtime.harness.model_outputs import require_open_model_output

    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as state:
        operation, attempt = _attempt(state, tmp_path)
        identity = dict(
            operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
        item = start_model_output(state, **identity, output=ModelOutput("a", "text"))
        original_rows = state.persistence.rows

        def bounded_rows(table, **kwargs):
            assert table != "rollout_records", "per-delta identity check must use the indexed Item reference"
            return original_rows(table, **kwargs)

        monkeypatch.setattr(state.persistence, "rows", bounded_rows)
        require_open_model_output(state, **identity, output_id="a", item_id=item.item_id)
        with pytest.raises(RuntimeError, match="open output"):
            require_open_model_output(state, **identity, output_id="different", item_id=item.item_id)


def test_output_lifecycle_reads_are_scoped_to_the_turn(tmp_path: Path, monkeypatch) -> None:
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as state:
        operation, attempt = _attempt(state, tmp_path)
        original_rows = state.persistence.rows

        def bounded_rows(table, **kwargs):
            if table == "rollout_records":
                assert kwargs.get("where", {}).get("turn_id") == operation.turn_id
            return original_rows(table, **kwargs)

        monkeypatch.setattr(state.persistence, "rows", bounded_rows)
        identity = dict(
            operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
        start_model_output(state, **identity, output=ModelOutput("a", "text"))
        complete_model_output(state, **identity, output=ModelOutput("a", "text", "complete"))
        assert read_model_outputs(state, attempt_id=attempt.attempt_id)[0].content == "complete"


def test_completed_output_cannot_change_and_private_provider_data_is_durable(tmp_path: Path) -> None:
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as state:
        operation, attempt = _attempt(state, tmp_path)
        identity = dict(
            operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
        start_model_output(state, **identity, output=ModelOutput("r", "reasoning"))
        output = ModelOutput("r", "reasoning", content="summary", provider_data={"signature": "private"})
        complete_model_output(state, **identity, output=output)
        with pytest.raises(RuntimeError, match="completed"):
            complete_model_output(state, **identity, output=ModelOutput("r", "reasoning", content="changed"))
        assert read_model_outputs(state, attempt_id=attempt.attempt_id)[0].provider_data["signature"] == "private"


def test_process_recovery_closes_only_open_outputs(tmp_path: Path) -> None:
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as state:
        operation, attempt = _attempt(state, tmp_path)
        identity = dict(
            operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
        start_model_output(state, **identity, output=ModelOutput("a", "text"))
        complete_model_output(state, **identity, output=ModelOutput("a", "text", content="finished"))
        start_model_output(state, **identity, output=ModelOutput("b", "text"))
        state.expire_model_attempt_dispatch(operation_id=operation.operation_id, now=time.time() + 60)
        completed = [
            r
            for r in state.list_records(operation.thread_id)
            if r.record_type == "item_completed" and r.payload.get("kind") == "model_output"
        ]
        assert [(r.payload["output_id"], r.payload["status"]) for r in completed] == [
            ("a", "success"),
            ("b", "outcome_unknown"),
        ]


@pytest.mark.anyio
async def test_runtime_keeps_two_text_outputs_and_replays_the_same_identities(tmp_path: Path) -> None:
    from agent_runtime.harness import (
        CompletionDecision,
        HarnessModelRequest,
        HarnessModelResponse,
        PreparedModelCall,
        RolloutEventReader,
        Session,
    )
    from tests.agent.harness.model_fixtures import emit_response

    class Model:
        def __init__(self):
            self.requests = []

        def snapshot(self, **kwargs):
            return {"model_id": "output-test"}

        def ensure_available(self, *args, **kwargs):
            pass

        def prepare(self, request: HarnessModelRequest):
            self.requests.append(request)
            return PreparedModelCall(
                request_hash="r",
                context_hash="c",
                tool_hash="t",
                wire_hash="w",
                request_ref={"request_id": str(len(self.requests))},
            )

        async def dispatch(self, prepared, *, event_sink=None):
            response = HarnessModelResponse(
                outputs=(ModelOutput("a", "text", "checked. "), ModelOutput("b", "text", "fixed.")),
                usage={"input_tokens": 1, "output_tokens": 2},
                provider_response_id="response",
            )
            if event_sink is not None:
                await emit_response(event_sink, response)
            return response

    class Accept:
        def evaluate(self, proposal):
            return CompletionDecision(action="accept", reason="accepted")

    class Sink:
        def __init__(self):
            self.events = []

        async def emit(self, event):
            self.events.append(event)

    model = Model()
    sink = Sink()
    path = tmp_path / "run.db"
    async with await Session.open(database=path, workspace=tmp_path, model=model, completion_gate=Accept()) as session:
        first = await session.submit("inspect", event_sink=sink)
        assert first.answer == "checked. fixed."
        completed_attempt = next(event for event in sink.events
                                 if event.type.value == "model_attempt_updated" and event.data["status"] == "completed")
        assert completed_attempt.data["stop_reason"] == "stop"
        thread_id = session.thread_id
        await session.submit("continue")
        restored = [message for message in model.requests[-1].messages if message.outputs]
        assert [(output.output_id, output.content) for output in restored[-1].outputs] == [
            ("a", "checked. "),
            ("b", "fixed."),
        ]
    live = [
        event
        for event in sink.events
        if event.type.value == "item_completed" and event.data.get("output_kind") == "text"
    ]
    assert [event.data["content"] for event in live] == ["checked. ", "fixed."]
    assert len({event.item_id for event in live}) == 2
    async with await RuntimeState.open(path) as state:
        entries = await state.run_async(RolloutEventReader(state).read, thread_id)
        replay = [
            entry.event
            for entry in entries
            if entry.event.turn_id == first.turn_id
            and entry.event.type.value == "item_completed"
            and entry.event.data.get("output_kind") == "text"
        ]
        assert [(event.item_id, event.data) for event in live] == [(event.item_id, event.data) for event in replay]


@pytest.mark.anyio
@pytest.mark.parametrize("local_stream_closed", [False, True])
async def test_interrupted_response_preserves_completed_objects_without_executing_proposals(
    tmp_path: Path, local_stream_closed: bool
) -> None:
    from agent_runtime.harness import Session
    from agent_runtime.modeling.events import (
        ModelOutputCompleted,
        ModelOutputDelta,
        ModelOutputStarted,
        ModelResponseStarted,
        emit_model_event,
    )
    from agent_runtime.streaming.sink import EventChannelClosed

    async def dispatch(sink):
        await emit_model_event(sink, ModelResponseStarted("provider-response"))
        await emit_model_event(sink, ModelOutputStarted(ModelOutput("a", "text")))
        await emit_model_event(sink, ModelOutputCompleted(ModelOutput("a", "text", "inspection done")))
        await emit_model_event(
            sink, ModelOutputStarted(ModelOutput("tool", "tool_call", tool_call_id="call", tool_name="run_command"))
        )
        await emit_model_event(sink, ModelOutputDelta("tool", "tool_arguments", '{"command":"touch file"}'))
        if local_stream_closed:
            raise EventChannelClosed("local subscriber closed")
        raise ConnectionError("provider transport interrupted")

    async with await Session.open(
        database=tmp_path / "run.db", workspace=tmp_path, model=_EventModel(dispatch)
    ) as session:
        result = await session.submit("inspect and fix")
        assert result.status == "paused"
        operations = await session.store.run_async(session.store.list_model_operations)
        [attempt] = await session.store.run_async(session.store.list_model_attempts, operations[0].operation_id)
        assert attempt.status == "unknown"
        records = await session.store.run_async(session.store.list_records, session.thread_id)
        completed = [
            record
            for record in records
            if record.record_type == "item_completed" and record.payload.get("kind") == "model_output"
        ]
        assert [(record.payload["output_id"], record.payload["status"]) for record in completed] == [
            ("a", "success"),
            ("tool", "outcome_unknown"),
        ]
        assert completed[-1].payload["payload"]["output"]["content"] == '{"command":"touch file"}'
        assert not any(record.record_type == "tool_operation_claimed" for record in records)


@pytest.mark.anyio
async def test_response_identity_survives_failure_before_first_output(tmp_path: Path) -> None:
    from agent_runtime.harness import RolloutEventReader, Session
    from agent_runtime.modeling.events import ModelResponseStarted, emit_model_event

    async def dispatch(sink):
        await emit_model_event(sink, ModelResponseStarted("provider-response"))
        raise ConnectionError("lost before first output")

    path = tmp_path / "run.db"
    async with await Session.open(database=path, workspace=tmp_path, model=_EventModel(dispatch)) as session:
        result = await session.submit("inspect")
        assert result.status == "paused"
        thread_id = session.thread_id
    async with await RuntimeState.open(path) as state:
        operations = await state.run_async(state.list_model_operations)
        [attempt] = await state.run_async(state.list_model_attempts, operations[0].operation_id)
        assert attempt.status == "unknown"
        assert attempt.provider_response_id == "provider-response"
        replay = await state.run_async(RolloutEventReader(state).read, thread_id)
        responding = [entry.event for entry in replay if entry.event.data.get("status") == "responding"]
        assert responding[0].data["provider_response_id"] == "provider-response"

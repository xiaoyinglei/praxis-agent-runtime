from __future__ import annotations

import asyncio
import hashlib
from dataclasses import fields
from pathlib import Path

import pytest

import agent_runtime
from agent_runtime.agent import Agent
from agent_runtime.harness import (
    HarnessModelRequest,
    HarnessModelResponse,
    PreparedModelCall,
    RolloutEventReader,
    RolloutStore,
)
from agent_runtime.streaming import events
from agent_runtime.streaming import sink as stream_sinks
from agent_runtime.streaming.sink import LegacyStreamProjectionSink


def test_v2_protocol_types_and_wire_values_are_public() -> None:
    assert {member.value for member in events.TurnItemKind} == {
        "agent_message",
        "reasoning",
        "plan",
        "tool",
        "command",
        "reconciliation",
        "legacy_message",
    }
    assert {member.value for member in events.ItemDeltaKind} == {
        "text",
        "reasoning",
        "plan",
        "tool_progress",
        "command_stdout",
        "command_stderr",
    }
    assert {member.value for member in events.ItemStatus} == {
        "success",
        "failed",
        "cancelled",
        "outcome_unknown",
    }
    assert events.EventType.TURN_STARTED.value == "turn_started"
    assert events.EventType.TURN_PAUSED.value == "turn_paused"
    assert events.EventType.TURN_RESUMED.value == "turn_resumed"
    assert (
        events.EventType.TURN_CANCELLATION_REQUESTED.value
        == "turn_cancellation_requested"
    )
    assert events.EventType.TURN_COMPLETED.value == "turn_completed"
    assert events.EventType.TURN_ABORTED.value == "turn_aborted"
    assert events.EventType.ITEM_STARTED.value == "item_started"
    assert events.EventType.ITEM_DELTA.value == "item_delta"
    assert events.EventType.ITEM_COMPLETED.value == "item_completed"


def test_v2_stream_event_has_explicit_item_contract() -> None:
    assert tuple(field.name for field in fields(events.StreamEvent)) == (
        "protocol_version",
        "type",
        "turn_id",
        "item_id",
        "item_kind",
        "delta_kind",
        "status",
        "iteration",
        "sequence",
        "timestamp_ms",
        "data",
        "error",
        "parent_item_id",
    )

    started = events.item_started(
        turn_id="turn-1",
        item_id="agent:turn-1:1",
        item_kind=events.TurnItemKind.AGENT_MESSAGE,
        iteration=1,
        data={"phase": "answer"},
    )
    delta = events.item_delta(
        turn_id="turn-1",
        item_id=started.item_id,
        item_kind=started.item_kind,
        delta_kind=events.ItemDeltaKind.TEXT,
        delta="hel",
        iteration=1,
    )
    completed = events.item_completed(
        turn_id="turn-1",
        item_id=started.item_id,
        item_kind=started.item_kind,
        status=events.ItemStatus.SUCCESS,
        iteration=1,
        data={"content": "hello", "tool_calls": []},
    )

    assert started.protocol_version == 2
    assert started.type is events.EventType.ITEM_STARTED
    assert delta.type is events.EventType.ITEM_DELTA
    assert delta.data == {"delta": "hel"}
    assert completed.type is events.EventType.ITEM_COMPLETED
    assert completed.status is events.ItemStatus.SUCCESS
    assert {started.turn_id, delta.turn_id, completed.turn_id} == {"turn-1"}
    assert {started.item_id, delta.item_id, completed.item_id} == {
        "agent:turn-1:1"
    }


def test_v2_stream_event_rejects_invalid_field_combinations() -> None:
    with pytest.raises(ValueError, match="item_id"):
        events.StreamEvent(
            type=events.EventType.ITEM_DELTA,
            turn_id="turn-1",
            item_kind=events.TurnItemKind.AGENT_MESSAGE,
            delta_kind=events.ItemDeltaKind.TEXT,
            data={"delta": "x"},
        )

    with pytest.raises(ValueError, match="error"):
        events.item_completed(
            turn_id="turn-1",
            item_id="tool:1",
            item_kind=events.TurnItemKind.TOOL,
            status=events.ItemStatus.FAILED,
            data={},
        )

    with pytest.raises(ValueError, match="forbidden"):
        events.StreamEvent(
            type=events.EventType.TURN_STARTED,
            turn_id="turn-1",
            item_id="not-allowed",
        )


def test_v2_turn_lifecycle_events_are_not_item_scoped() -> None:
    started = events.turn_started("turn-1")
    paused = events.turn_paused("turn-1", reason="approval")
    resumed = events.turn_resumed("turn-1")
    cancelling = events.turn_cancellation_requested("turn-1")
    aborted = events.turn_aborted("turn-1", reason="consumer_closed")

    assert [event.type for event in (started, paused, resumed, cancelling, aborted)] == [
        events.EventType.TURN_STARTED,
        events.EventType.TURN_PAUSED,
        events.EventType.TURN_RESUMED,
        events.EventType.TURN_CANCELLATION_REQUESTED,
        events.EventType.TURN_ABORTED,
    ]
    assert all(
        event.item_id is None
        for event in (started, paused, resumed, cancelling, aborted)
    )


class _LiveSink:
    def __init__(self) -> None:
        self.events: list[events.StreamEvent] = []

    async def emit(self, event: events.StreamEvent) -> None:
        self.events.append(event)


@pytest.mark.anyio
async def test_legacy_projection_emits_only_legacy_wire_events() -> None:
    target = _LiveSink()
    sink = LegacyStreamProjectionSink(target)
    canonical = (
        events.turn_started("turn-legacy"),
        events.item_started(
            turn_id="turn-legacy",
            item_id="agent:turn-legacy:1",
            item_kind=events.TurnItemKind.AGENT_MESSAGE,
            iteration=1,
        ),
        events.item_delta(
            turn_id="turn-legacy",
            item_id="agent:turn-legacy:1",
            item_kind=events.TurnItemKind.AGENT_MESSAGE,
            delta_kind=events.ItemDeltaKind.TEXT,
            delta="hello",
            iteration=1,
        ),
        events.item_completed(
            turn_id="turn-legacy",
            item_id="agent:turn-legacy:1",
            item_kind=events.TurnItemKind.AGENT_MESSAGE,
            status=events.ItemStatus.SUCCESS,
            data={"content": "hello", "tool_calls": []},
            iteration=1,
        ),
        events.turn_completed("turn-legacy"),
    )

    for event in canonical:
        await sink.emit(event)

    assert [event.type for event in target.events] == [
        events.EventType.TURN_START,
        events.EventType.TEXT_DELTA,
        events.EventType.LOOP_END,
    ]
    assert not any(
        event.type
        in {
            events.EventType.TURN_STARTED,
            events.EventType.ITEM_STARTED,
            events.EventType.ITEM_DELTA,
            events.EventType.ITEM_COMPLETED,
            events.EventType.TURN_COMPLETED,
        }
        for event in target.events
    )


@pytest.mark.anyio
async def test_legacy_projection_preserves_tool_id_result_details_and_plan_event() -> None:
    target = _LiveSink()
    sink = LegacyStreamProjectionSink(target)
    await sink.emit(
        events.item_started(
            turn_id="turn-legacy",
            item_id="tool:turn-legacy:call-1:1",
            item_kind=events.TurnItemKind.TOOL,
            data={
                "tool_name": "apply_patch",
                "tool_call_id": "call-1",
                "input_preview": "patch='***'",
            },
        )
    )
    await sink.emit(
        events.item_completed(
            turn_id="turn-legacy",
            item_id="tool:turn-legacy:call-1:1",
            item_kind=events.TurnItemKind.TOOL,
            status=events.ItemStatus.SUCCESS,
            data={
                "result": {
                    "tool_name": "apply_patch",
                    "content": ({"type": "text", "data": {"text": "patched"}},),
                    "structured_content": None,
                    "metadata": {
                        "file_path": "a.py",
                        "diff": "-old\n+new",
                        "diff_truncated": False,
                    },
                }
            },
        )
    )
    await sink.emit(
        events.item_completed(
            turn_id="turn-legacy",
            item_id="update_plan:turn-legacy:1",
            item_kind=events.TurnItemKind.PLAN,
            status=events.ItemStatus.SUCCESS,
            data={"plan": {"revision": 1}, "event": {"event_type": "llm_update"}},
        )
    )

    assert [event.type for event in target.events] == [
        events.EventType.TOOL_USE_START,
        events.EventType.TOOL_USE_RESULT,
        events.EventType.PLAN_UPDATED,
    ]
    assert target.events[0].data["tool_id"] == "call-1"
    assert target.events[1].data["result"] == "patched"
    assert target.events[1].data["details"] == {
        "file_path": "a.py",
        "diff": "-old\n+new",
        "diff_truncated": False,
    }
    assert target.events[2].data["event"] == {"event_type": "llm_update"}


@pytest.mark.anyio
async def test_controlling_stream_blocks_producer_when_queue_is_full() -> None:
    dispatcher_type = getattr(stream_sinks, "TurnEventDispatcher", None)
    assert dispatcher_type is not None, "canonical dispatcher must be public"
    dispatcher = dispatcher_type(capacity=1)
    stream = dispatcher.subscribe_controlling()
    first = events.turn_started("turn-backpressure")
    second = events.turn_completed("turn-backpressure")

    await dispatcher.emit(first)
    blocked_emit = asyncio.create_task(dispatcher.emit(second))
    await asyncio.sleep(0)

    assert blocked_emit.done() is False
    assert await stream.receive() == first
    await asyncio.wait_for(blocked_emit, timeout=0.2)
    assert await stream.receive() == second


@pytest.mark.anyio
async def test_full_controlling_queue_close_wakes_blocked_emitter() -> None:
    closed_error = getattr(stream_sinks, "EventChannelClosed", None)
    assert closed_error is not None, "closed event channels need a public error"
    dispatcher = stream_sinks.TurnEventDispatcher(capacity=1)
    stream = dispatcher.subscribe_controlling()
    await dispatcher.emit(events.turn_started("turn-close"))
    blocked_emit = asyncio.create_task(
        dispatcher.emit(events.turn_completed("turn-close"))
    )
    await asyncio.sleep(0)
    assert blocked_emit.done() is False

    stream.close()

    with pytest.raises(closed_error):
        await asyncio.wait_for(blocked_emit, timeout=0.2)


@pytest.mark.anyio
async def test_emit_after_close_raises_event_channel_closed() -> None:
    dispatcher = stream_sinks.TurnEventDispatcher(capacity=1)
    stream = dispatcher.subscribe_controlling()
    stream.close()

    with pytest.raises(stream_sinks.EventChannelClosed):
        await dispatcher.emit(events.turn_started("turn-already-closed"))


def test_harness_rollout_reader_returns_canonical_stream_events(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        store.start_turn(
            thread_id=thread.thread_id,
            turn_id="turn-v2",
            user_message="hello",
            binding_manifest={},
        )
        replayed = RolloutEventReader(store).read(thread.thread_id)

    assert replayed
    assert getattr(agent_runtime, "ReplayEvent", None) is not None
    assert all(isinstance(record.event, events.StreamEvent) for record in replayed)
    assert all(record.event.protocol_version == 2 for record in replayed)


class _StaticHarnessModel:
    def snapshot(self) -> dict[str, str]:
        return {"model_alias": "stream-test", "model_revision": "v1"}

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        digest = hashlib.sha256(request.turn_id.encode()).hexdigest()
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash=digest,
            wire_hash=digest,
            request_ref={"request_id": f"{request.turn_id}:step:{request.step}"},
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        return HarnessModelResponse(
            text="canonical answer",
            provider_response_id="response-v2",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )


@pytest.mark.anyio
@pytest.mark.xfail(
    strict=True,
    reason="Tasks 3 and 4 wire the Harness dispatcher and live model deltas",
)
async def test_harness_public_stream_uses_canonical_v2_items(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sink = _LiveSink()
    agent = Agent(
        checkpoint_db=tmp_path / "rollout.sqlite3",
        workspace_path=workspace,
    )
    agent._harness_model = _StaticHarnessModel

    await agent.arun(
        "hello",
        event_sink=sink,
        require_workspace_change=False,
    )

    assert [event.type for event in sink.events] == [
        events.EventType.TURN_STARTED,
        events.EventType.ITEM_STARTED,
        events.EventType.ITEM_DELTA,
        events.EventType.ITEM_COMPLETED,
        events.EventType.TURN_COMPLETED,
    ]

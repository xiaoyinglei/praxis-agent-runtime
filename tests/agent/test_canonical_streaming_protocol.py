from __future__ import annotations

from dataclasses import fields

import pytest

from rag.agent.streaming import events


def test_v2_protocol_types_and_wire_values_are_public() -> None:
    turn_item_kind = getattr(events, "TurnItemKind", None)
    delta_kind = getattr(events, "ItemDeltaKind", None)
    item_status = getattr(events, "ItemStatus", None)

    assert turn_item_kind is not None
    assert delta_kind is not None
    assert item_status is not None
    assert {member.value for member in turn_item_kind} == {
        "agent_message",
        "reasoning",
        "plan",
        "tool",
        "command",
        "reconciliation",
        "legacy_message",
    }
    assert {member.value for member in delta_kind} == {
        "text",
        "reasoning",
        "plan",
        "tool_progress",
        "command_stdout",
        "command_stderr",
    }
    assert {member.value for member in item_status} == {
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
    assert all(event.item_id is None for event in (started, paused, resumed, cancelling, aborted))

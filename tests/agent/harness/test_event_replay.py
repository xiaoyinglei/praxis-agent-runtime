from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import agent_runtime.harness as harness
from agent_runtime.harness import RolloutStore
from agent_runtime.streaming import events
from agent_runtime.streaming import sink as stream_sinks
from agent_runtime.streaming.events import StreamEvent


def _reader(store: RolloutStore):
    reader_type = getattr(harness, "RolloutEventReader", None)
    assert reader_type is not None, "Harness must expose a durable event reader"
    return reader_type(store)


def _start_turn(store: RolloutStore, workspace: Path, message: str) -> tuple[str, str]:
    thread = store.create_thread(workspace=workspace)
    turn = store.start_turn(
        thread_id=thread.thread_id,
        user_message=message,
        binding_manifest={"model_alias": "event-model"},
    )
    return thread.thread_id, turn.turn_id


def test_replay_returns_event_and_separate_thread_cursor(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread_id, _turn_id = _start_turn(store, workspace, "canonical replay")
        replayed = _reader(store).read(thread_id)

    replay_type = getattr(harness, "ReplayEvent", None)
    assert replay_type is not None, "Harness must expose ReplayEvent"
    assert replayed
    assert all(isinstance(result, replay_type) for result in replayed)
    assert all(isinstance(result.event, StreamEvent) for result in replayed)
    assert all(not hasattr(result.event, "cursor") for result in replayed)
    assert all(result.cursor for result in replayed)


def test_thread_cursor_rejects_schema_epoch_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread_id, _turn_id = _start_turn(store, workspace, "schema cursor")
        reader = _reader(store)
        cursor = reader.read(thread_id)[-1].cursor
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding).decode())
        payload["schema_epoch"] = "future-schema"
        mismatched = (
            base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
            .decode()
            .rstrip("=")
        )

        with pytest.raises(ValueError, match="schema epoch"):
            reader.read(thread_id, after=mismatched)


def test_thread_cursor_rejects_cross_thread_ahead_and_store_epoch_mismatch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        first_thread, _ = _start_turn(store, workspace, "first cursor")
        second_thread, _ = _start_turn(store, workspace, "second cursor")
        reader = _reader(store)
        cursor = reader.read(first_thread)[-1].cursor

        with pytest.raises(ValueError, match="different thread"):
            reader.read(second_thread, after=cursor)

        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding).decode())
        payload["thread_sequence"] = 10_000
        ahead = (
            base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
            .decode()
            .rstrip("=")
        )
        with pytest.raises(ValueError, match="ahead"):
            reader.read(first_thread, after=ahead)

        payload["thread_sequence"] = 0
        payload["store_epoch"] = "different-store"
        wrong_store = (
            base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
            .decode()
            .rstrip("=")
        )
        with pytest.raises(ValueError, match="store epoch"):
            reader.read(first_thread, after=wrong_store)


def test_malformed_cursor_returns_actionable_full_resync_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread_id, _turn_id = _start_turn(store, workspace, "malformed cursor")

        with pytest.raises(ValueError, match="full resync"):
            _reader(store).read(thread_id, after="not-a-valid-cursor")


def test_global_tailer_accepts_only_after_record_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        first_thread, _ = _start_turn(store, workspace, "first global record")
        after_record_id = store.list_records(first_thread)[-1].record_id
        second_thread, _ = _start_turn(store, workspace, "second global record")
        reader = _reader(store)

        tail = reader.read_global(after_record_id=after_record_id)
        assert {result.thread_id for result in tail} == {second_thread}
        with pytest.raises(TypeError):
            reader.read_global(after=tail[-1].cursor)


def test_global_tailer_rejects_negative_record_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        _start_turn(store, workspace, "negative global record")

        with pytest.raises(ValueError, match="non-negative"):
            _reader(store).read_global(after_record_id=-1)


def test_global_tailer_fails_closed_on_incomplete_unknown_item_kind(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread_id, _turn_id = _start_turn(store, workspace, "unknown global item")
        start = next(record for record in store.list_records(thread_id) if record.record_type == "item_started")
        payload = {"item_id": start.payload["item_id"], "kind": "future_internal_kind"}
        payload_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        with sqlite3.connect(database) as connection:
            connection.execute(
                """
                UPDATE rollout_records
                SET payload_json = ?, payload_hash = ?
                WHERE record_id = ?
                """,
                (
                    payload_json,
                    hashlib.sha256(payload_json.encode()).hexdigest(),
                    start.record_id,
                ),
            )
            connection.execute(
                """
                DELETE FROM rollout_records
                WHERE record_type = 'item_completed'
                  AND json_extract(payload_json, '$.item_id') = ?
                """,
                (start.payload["item_id"],),
            )

        reader = _reader(store)
        with pytest.raises(RuntimeError, match="unknown internal Item kind"):
            reader.read(thread_id)
        with pytest.raises(RuntimeError, match="unknown internal Item kind"):
            reader.read_global()


@pytest.mark.anyio
async def test_lagging_passive_observer_detaches_without_blocking_control() -> None:
    observer_error = getattr(stream_sinks, "ObserverLagged", None)
    assert observer_error is not None, "lagging observers need a public error"
    dispatcher = stream_sinks.TurnEventDispatcher(capacity=1)
    controlling = dispatcher.subscribe_controlling()
    observer = dispatcher.subscribe_passive(last_cursor="cursor-before")
    first = events.turn_started("turn-observer")
    second = events.turn_completed("turn-observer")

    await dispatcher.emit(first)
    assert await controlling.receive() == first
    await asyncio.wait_for(dispatcher.emit(second), timeout=0.2)

    assert await controlling.receive() == second
    with pytest.raises(observer_error) as raised:
        await observer.receive()
    assert raised.value.last_cursor == "cursor-before"


@pytest.mark.anyio
async def test_lagging_observer_does_not_change_another_observers_cursor() -> None:
    dispatcher = stream_sinks.TurnEventDispatcher(capacity=1)
    first = dispatcher.subscribe_passive(last_cursor="cursor-zero-first")
    second = dispatcher.subscribe_passive(last_cursor="cursor-zero-second")

    await dispatcher.emit(
        events.turn_started("turn-cursor-isolation"),
        cursor="cursor-one",
    )
    assert (await first.receive()).type is events.EventType.TURN_STARTED
    await dispatcher.emit(
        events.turn_completed("turn-cursor-isolation"),
        cursor="cursor-two",
    )
    await dispatcher.emit(
        events.turn_aborted("turn-cursor-isolation", reason="late observer"),
        cursor="cursor-three",
    )

    with pytest.raises(stream_sinks.ObserverLagged) as first_lag:
        await first.receive()
    with pytest.raises(stream_sinks.ObserverLagged) as second_lag:
        await second.receive()
    assert first_lag.value.last_cursor == "cursor-two"
    assert second_lag.value.last_cursor == "cursor-one"


def test_thread_cursor_replays_the_same_committed_tail_after_restart(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread_id, turn_id = _start_turn(store, workspace, "persist events")
        initial = _reader(store).read(thread_id)
        acknowledged = initial[-1].cursor
        store.complete_turn(turn_id=turn_id, answer="durable answer")
        expected_tail = _reader(store).read(
            thread_id,
            after=acknowledged,
        )

    with RolloutStore(database) as restarted:
        replayed = _reader(restarted).read(
            thread_id,
            after=acknowledged,
        )

        assert [
            (
                result.cursor,
                result.record_id,
                result.thread_sequence,
                result.event.type,
                result.event.item_id,
                result.event.item_kind,
                result.event.status,
                result.event.data,
            )
            for result in replayed
        ] == [
            (
                result.cursor,
                result.record_id,
                result.thread_sequence,
                result.event.type,
                result.event.item_id,
                result.event.item_kind,
                result.event.status,
                result.event.data,
            )
            for result in expected_tail
        ]
        assert [event.thread_sequence for event in replayed] == [5, 6, 7]
        assert replayed[-1].event_type == "turn_completed"


def test_reasoning_and_plan_completed_content_replays_after_reopen(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread_id, turn_id = _start_turn(store, workspace, "persist model channels")
        operation = store.prepare_model_operation(
            turn_id=turn_id,
            request_hash="r" * 64,
            context_hash="c" * 64,
            tool_hash="t" * 64,
            wire_hash="w" * 64,
            request_ref={"request_id": f"{turn_id}:step:1"},
        )
        attempt = store.dispatch_model_attempt(operation.operation_id)
        store.start_model_output_channel(
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            channel="reasoning",
        )
        store.start_model_output_channel(
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            channel="plan",
        )
        assert store.complete_model_attempt(
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            text="final answer",
            provider_response_id="response-1",
            usage={"input_tokens": 4, "output_tokens": 3},
            reasoning_content="durable reasoning",
            plan_content="durable plan",
        )
        expected = [
            replay.event
            for replay in _reader(store).read(thread_id)
            if replay.event.type is events.EventType.ITEM_COMPLETED
            and replay.event.item_kind
            in {events.TurnItemKind.REASONING, events.TurnItemKind.PLAN}
        ]

    with RolloutStore(database) as reopened:
        replayed = [
            replay.event
            for replay in _reader(reopened).read(thread_id)
            if replay.event.type is events.EventType.ITEM_COMPLETED
            and replay.event.item_kind
            in {events.TurnItemKind.REASONING, events.TurnItemKind.PLAN}
        ]

        assert [
            (event.item_id, event.item_kind, event.status, event.data)
            for event in replayed
        ] == [
            (event.item_id, event.item_kind, event.status, event.data)
            for event in expected
        ]
        assert {event.item_kind: event.data["content"] for event in replayed} == {
            events.TurnItemKind.REASONING: "durable reasoning",
            events.TurnItemKind.PLAN: "durable plan",
        }
        assert reopened.verify().valid is True


def test_thread_cursors_are_independent_and_reject_the_wrong_thread(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        first_thread, first_turn = _start_turn(store, workspace, "first")
        second_thread, second_turn = _start_turn(store, workspace, "second")
        reader = _reader(store)
        first_cursor = reader.read(first_thread)[-1].cursor
        second_cursor = reader.read(second_thread)[-1].cursor
        store.complete_turn(turn_id=first_turn, answer="first answer")
        store.complete_turn(turn_id=second_turn, answer="second answer")

        assert reader.read(first_thread, after=first_cursor)[0].thread_sequence == 5
        assert reader.read(second_thread, after=second_cursor)[0].thread_sequence == 5
        with pytest.raises(ValueError, match="different thread"):
            reader.read(second_thread, after=first_cursor)


def test_global_tailer_uses_record_id_not_thread_sequence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        first_thread, _ = _start_turn(store, workspace, "first")
        first_record_id = store.list_records(first_thread)[-1].record_id
        second_thread, _ = _start_turn(store, workspace, "second")

        tail = _reader(store).read_global(after_record_id=first_record_id)

        assert {event.thread_id for event in tail} == {second_thread}
        assert [event.thread_sequence for event in tail] == [2]
        assert all(event.record_id > first_record_id for event in tail)


def test_item_lifecycle_events_are_derived_from_one_committed_record_sequence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread_id, turn_id = _start_turn(store, workspace, "trace item lifecycle")
        store.complete_turn(turn_id=turn_id, answer="durable answer")

        events = _reader(store).read(thread_id)

        assert [event.thread_sequence for event in events] == [2, 5, 6, 7]
        assert events[-1].event_type == "turn_completed"
        started = {event.event.item_id for event in events if event.event_type == "item_started"}
        completed = {event.event.item_id for event in events if event.event_type == "item_completed"}
        assert started == completed
        assert len(started) == 1
        assert all(event.turn_id == turn_id for event in events[1:])


def test_post_commit_delivery_crash_does_not_lose_replayable_events(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    attempted: list[int] = []

    with RolloutStore(database) as store:
        thread_id, turn_id = _start_turn(store, workspace, "survive delivery crash")
        with pytest.raises(RuntimeError, match="injected delivery crash"):
            mutation = store.capture_mutation(
                lambda: store.complete_turn(
                    turn_id=turn_id,
                    answer="committed despite listener crash",
                )
            )
            attempted.extend(record.record_id for record in mutation.records)
            raise RuntimeError("injected delivery crash")
        committed_count = len(store.list_records(thread_id))
        assert attempted == [committed_count - 2, committed_count - 1, committed_count]
        assert store.verify().valid is True

    with RolloutStore(database) as restarted:
        replayed = _reader(restarted).read(thread_id)

        assert len(replayed) == 4
        assert [event.record_id for event in replayed] == [2, 5, 6, 7]
        assert replayed[-1].event_type == "turn_completed"
        assert restarted.verify().valid is True

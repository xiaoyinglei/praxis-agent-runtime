from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import agent_runtime.harness as harness
from agent_runtime.harness import RolloutStore
from agent_runtime.harness.rollout import RolloutRecord
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
        mismatched = base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).decode().rstrip("=")

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
        ahead = base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).decode().rstrip("=")
        with pytest.raises(ValueError, match="ahead"):
            reader.read(first_thread, after=ahead)

        payload["thread_sequence"] = 0
        payload["store_epoch"] = "different-store"
        wrong_store = base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).decode().rstrip("=")
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
        start = next(
            record
            for record in store.list_records(thread_id)
            if record.record_type == "item_started"
        )
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
        started = {
            event.event.item_id
            for event in events
            if event.event_type == "item_started"
        }
        completed = {
            event.event.item_id
            for event in events
            if event.event_type == "item_completed"
        }
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

    def crash_after_commit(record: RolloutRecord) -> None:
        attempted.append(record.record_id)
        raise RuntimeError("injected delivery crash")

    with RolloutStore(database, record_listener=crash_after_commit) as store:
        thread_id, turn_id = _start_turn(store, workspace, "survive delivery crash")
        store.complete_turn(turn_id=turn_id, answer="committed despite listener crash")
        committed_count = len(store.list_records(thread_id))
        assert attempted == list(range(1, committed_count + 1))
        assert store.verify().valid is True

    with RolloutStore(database) as restarted:
        replayed = _reader(restarted).read(thread_id)

        assert len(replayed) == 4
        assert [event.record_id for event in replayed] == [2, 5, 6, 7]
        assert replayed[-1].event_type == "turn_completed"
        assert restarted.verify().valid is True

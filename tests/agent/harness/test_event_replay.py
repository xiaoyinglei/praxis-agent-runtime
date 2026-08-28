from __future__ import annotations

from pathlib import Path

import pytest

import agent_runtime.harness as harness
from agent_runtime.harness import RolloutStore
from agent_runtime.harness.rollout import RolloutRecord


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


def test_thread_cursor_replays_the_same_committed_tail_after_restart(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread_id, turn_id = _start_turn(store, workspace, "persist events")
        initial = _reader(store).read(thread_id)
        acknowledged = initial[1].cursor
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

        assert replayed == expected_tail
        assert [event.thread_sequence for event in replayed] == list(range(3, replayed[-1].thread_sequence + 1))
        assert replayed[-1].event_type == "turn_completed"


def test_thread_cursors_are_independent_and_reject_the_wrong_thread(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        first_thread, _ = _start_turn(store, workspace, "first")
        second_thread, _ = _start_turn(store, workspace, "second")
        reader = _reader(store)
        first_cursor = reader.read(first_thread)[1].cursor
        second_cursor = reader.read(second_thread)[2].cursor

        assert reader.read(first_thread, after=first_cursor)[0].thread_sequence == 3
        assert reader.read(second_thread, after=second_cursor)[0].thread_sequence == 4
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
        assert [event.thread_sequence for event in tail] == [1, 2, 3, 4]
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

        assert [event.thread_sequence for event in events] == list(range(1, len(events) + 1))
        assert events[-1].event_type == "turn_completed"
        started = {event.data["item_id"] for event in events if event.event_type == "item_started"}
        completed = {event.data["item_id"] for event in events if event.event_type == "item_completed"}
        assert started == completed
        assert len(started) == 2
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

        assert len(replayed) == committed_count
        assert [event.record_id for event in replayed] == list(range(1, committed_count + 1))
        assert replayed[-1].event_type == "turn_completed"
        assert restarted.verify().valid is True

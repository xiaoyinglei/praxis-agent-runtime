from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutStore
from agent_runtime.harness.rollout import _projection_hash, _store_operation


@_store_operation
def _settings_work(store: RolloutStore, thread_id: str) -> int:
    steps = 0

    def progress() -> int:
        nonlocal steps
        steps += 1
        return 0

    store.run_sync(lambda: store._connection.set_progress_handler(progress, 1))
    try:
        store.update_session_settings(thread_id, {"model_alias": "test"})
    finally:
        store.run_sync(lambda: store._connection.set_progress_handler(None, 0))
    return steps


@_store_operation
def _legacy_metadata(store: RolloutStore, thread_id: str) -> None:
    digest = _projection_hash(store._load_projection_state(thread_id), thread_id)
    store.run_sync(
        lambda: store._connection.execute(
            "UPDATE projection_meta SET canonical_hash = ? WHERE thread_id = ?",
            (digest, thread_id),
        ).close()
    )
    columns = store.run_sync(
        lambda: {row["name"] for row in store._connection.execute("PRAGMA table_info(projection_meta)")}
    )
    if "hash_version" in columns:
        store.run_sync(
            lambda: store._connection.execute("ALTER TABLE projection_meta DROP COLUMN hash_version").close()
        )


def test_settings_event_work_does_not_grow_with_historical_turns(tmp_path: Path) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=tmp_path)
        small = _settings_work(store, thread.thread_id)
        for _ in range(100):
            turn = store.start_turn(thread_id=thread.thread_id, user_message="u" * 4096, binding_manifest={})
            store.complete_turn(turn_id=turn.turn_id, answer="a" * 4096)
        statements: list[str] = []
        store.run_sync(lambda: store._connection.set_trace_callback(statements.append))
        large = _settings_work(store, thread.thread_id)
        store.run_sync(lambda: store._connection.set_trace_callback(None))

        assert large <= small * 2, (small, large)
        assert not any("SELECT * FROM items WHERE thread_id" in statement for statement in statements)
        assert store.verify().valid


def test_item_creation_does_not_load_previous_item_payloads(tmp_path: Path) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = store.start_turn(
            thread_id=store.create_thread(workspace=tmp_path).thread_id,
            user_message="history" * 4096,
            binding_manifest={},
        )
        for _ in range(20):
            store.record_final_proposal(turn_id=turn.turn_id, answer="proposal" * 4096)
        statements: list[str] = []
        store.run_sync(lambda: store._connection.set_trace_callback(statements.append))
        item = store.record_final_proposal(turn_id=turn.turn_id, answer="next")
        store.run_sync(lambda: store._connection.set_trace_callback(None))

        assert item.sequence == 22
        assert all(
            "LIMIT 1" in statement for statement in statements if "SELECT * FROM items WHERE turn_id" in statement
        )
        assert store.verify().valid


def test_record_envelope_corruption_is_detected_without_projection_difference(tmp_path: Path) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=tmp_path)
        store.update_session_settings(thread.thread_id, {"model_alias": "test"})
        store.run_sync(
            lambda: store._connection.execute(
                "UPDATE rollout_records SET producer = 'model' WHERE record_type = 'session_settings_updated'"
            ).close()
        )
        assert "projection metadata mismatch" in store.verify().errors


def test_legacy_readonly_verification_then_atomic_upgrade(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="old snapshot", binding_manifest={})
        store.complete_turn(turn_id=turn.turn_id, answer="preserved")
        _legacy_metadata(store, thread.thread_id)
    with RolloutStore(database, read_only=True) as store:
        assert store.verify().valid
        columns = store.run_sync(
            lambda: {row["name"] for row in store._connection.execute("PRAGMA table_info(projection_meta)")}
        )
        assert "hash_version" not in columns
    with RolloutStore(database) as store:
        store.update_session_settings(thread.thread_id, {"model_alias": "new"})
        metadata = store.run_sync(lambda: store._connection.execute("SELECT * FROM projection_meta").fetchone())
        assert metadata["hash_version"] == 2
        assert store.verify().valid
        hashes = store.projection_hashes()
        store.rebuild_projections()
        assert store.projection_hashes() == hashes
        assert store.verify().valid


def test_corrupt_legacy_metadata_is_not_erased_by_upgrade(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        _legacy_metadata(store, thread.thread_id)
        store.run_sync(
            lambda: store._connection.execute("UPDATE projection_meta SET canonical_hash = ?", ("0" * 64,)).close()
        )
    with RolloutStore(database) as store:
        records = store.list_records(thread.thread_id)
        with pytest.raises(RuntimeError, match="integrity"):
            store.update_session_settings(thread.thread_id, {"model_alias": "must not commit"})
        assert store.list_records(thread.thread_id) == records
        assert store.projection_hashes()[thread.thread_id] == "0" * 64
        assert not store.verify().valid


def test_legacy_upgrade_and_append_roll_back_together(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        _legacy_metadata(store, thread.thread_id)
    with RolloutStore(database) as store:
        records = store.list_records(thread.thread_id)
        metadata = dict(store.run_sync(lambda: store._connection.execute("SELECT * FROM projection_meta").fetchone()))
        store.run_sync(
            lambda: store._connection.execute(
                "CREATE TRIGGER fail_settings BEFORE INSERT ON rollout_records "
                "WHEN NEW.record_type = 'session_settings_updated' "
                "BEGIN SELECT RAISE(ABORT, 'forced failure'); END"
            ).close()
        )
        with pytest.raises(sqlite3.IntegrityError, match="forced failure"):
            store.update_session_settings(thread.thread_id, {"model_alias": "must roll back"})
        assert (
            dict(store.run_sync(lambda: store._connection.execute("SELECT * FROM projection_meta").fetchone()))
            == metadata
        )
        assert store.list_records(thread.thread_id) == records
        assert store.verify().valid


def test_unknown_digest_version_fails_closed(tmp_path: Path) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=tmp_path)
        store.run_sync(lambda: store._connection.execute("UPDATE projection_meta SET hash_version = 999").close())
        assert not store.verify().valid
        with pytest.raises(RuntimeError, match="hash version"):
            store.update_session_settings(thread.thread_id, {"model_alias": "no"})


@pytest.mark.parametrize("version", [2.5, "invalid"])
def test_malformed_digest_version_is_not_coerced_into_a_known_version(tmp_path: Path, version: object) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        store.create_thread(workspace=tmp_path)
        store.run_sync(
            lambda: store._connection.execute("UPDATE projection_meta SET hash_version = ?", (version,)).close()
        )
        assert not store.verify().valid


def test_streaming_verification_detects_extra_and_missing_projection_rows(tmp_path: Path) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = store.start_turn(
            thread_id=store.create_thread(workspace=tmp_path).thread_id,
            user_message="original",
            binding_manifest={},
        )
        store.run_sync(
            lambda: store._connection.execute(
                "INSERT INTO items SELECT 'extra', thread_id, turn_id, sequence + 1, kind, status, producer, "
                "payload_json, applied_thread_sequence, reducer_version FROM items WHERE turn_id = ?",
                (turn.turn_id,),
            ).close()
        )
        assert "items projection mismatch" in store.verify().errors
        store.run_sync(lambda: store._connection.execute("DELETE FROM items").close())
        assert "items projection mismatch" in store.verify().errors


def test_interleaved_threads_and_fork_have_independent_rebuildable_hashes(tmp_path: Path) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        parent = store.create_thread(workspace=tmp_path)
        other = store.create_thread(workspace=tmp_path)
        for thread in (parent, other, parent):
            turn = store.start_turn(thread_id=thread.thread_id, user_message="history", binding_manifest={})
            store.complete_turn(turn_id=turn.turn_id, answer="answer")
        child = store.fork_thread(from_turn_id=turn.turn_id)
        before = store.projection_hashes()
        store.update_session_settings(child.thread_id, {"model_alias": "child"})
        after = store.projection_hashes()
        assert before[parent.thread_id] == after[parent.thread_id]
        assert before[other.thread_id] == after[other.thread_id]
        assert before[child.thread_id] != after[child.thread_id]
        assert store.verify().valid
        store.rebuild_projections()
        assert store.projection_hashes() == after
        assert store.verify().valid


@pytest.mark.parametrize("corrupt", ["true", "1.0"])
def test_streaming_verification_preserves_strict_json_types(tmp_path: Path, corrupt: str) -> None:
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=tmp_path)
        store.update_session_settings(thread.thread_id, {"value": 1})
        store.run_sync(
            lambda: store._connection.execute(
                "UPDATE threads SET settings_json = ?", ('{"value":' + corrupt + "}\u0020",)
            ).close()
        )
        assert "threads projection mismatch" in store.verify().errors

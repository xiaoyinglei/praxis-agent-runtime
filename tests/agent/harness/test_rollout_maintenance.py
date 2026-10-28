from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.harness.rollout_integrity import rebuild_store, verify_store
from agent_runtime.harness.state import RuntimeState


def _projection_rows(store):
    def read():
        tables = (
            row[0]
            for row in store._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT IN ('rollout_records', 'store_metadata', 'sqlite_sequence') ORDER BY name"
            )
        )
        return {
            table: tuple(tuple(row) for row in store._connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'))
            for table in tables
        }
    return store.run_sync(read)


def test_rebuild_streams_canonical_rows_and_restores_verified_projections(tmp_path: Path, monkeypatch) -> None:
    with RuntimeState.open_sync(tmp_path / "store.db") as state:
        for index in range(3):
            thread = state.create_thread(workspace=tmp_path)
            turn = state.start_turn(
                thread_id=thread.thread_id,
                user_message=f"turn {index}",
                binding_manifest={"model_id": "test-model"},
            )
        state.commit_artifact(
            turn_id=turn.turn_id, content=b"keep this blob", media_type="text/plain", name="result.txt",
        )
        store = state.persistence
        expected = _projection_rows(store)
        store.run_sync(lambda: store._connection.execute(
            "UPDATE threads SET settings_json = '{\"corrupt\":true}' WHERE thread_id = ?", (thread.thread_id,),
        ))
        connection = store._connection

        class StreamingCursor:
            def __init__(self, cursor):
                self.cursor = cursor

            def __getattr__(self, name):
                return getattr(self.cursor, name)

            def __iter__(self):
                return iter(self.cursor)

            def fetchall(self):
                raise AssertionError("canonical history must be streamed")

        class StreamingConnection:
            def __getattr__(self, name):
                return getattr(connection, name)

            def execute(self, sql, *args):
                cursor = connection.execute(sql, *args)
                return StreamingCursor(cursor) if "FROM rollout_records" in sql else cursor

        monkeypatch.setattr(store, "_connection", StreamingConnection())
        assert not verify_store(store).valid
        rebuild_store(store)
        assert verify_store(store).valid
        assert _projection_rows(store) == expected


def test_rebuild_refuses_concurrent_commit_without_overwriting_it(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "store.db"
    with RuntimeState.open_sync(database) as state, RuntimeState.open_sync(database) as writer:
        state.create_thread(workspace=tmp_path)
        store = state.persistence
        persist = store._persist_projection_state
        concurrent = []

        def interleave(*args, **kwargs):
            if not concurrent and kwargs.get("connection") is not None:
                concurrent.append(writer.create_thread(workspace=tmp_path))
            return persist(*args, **kwargs)

        monkeypatch.setattr(store, "_persist_projection_state", interleave)
        with pytest.raises(RuntimeError, match="changed.*rebuild|rebuild.*changed"):
            rebuild_store(store)
        assert concurrent
        assert state.read_thread(concurrent[0].thread_id).thread_id == concurrent[0].thread_id
        assert verify_store(store).valid


def test_rebuild_rejects_corrupt_canonical_without_changing_projections(tmp_path: Path) -> None:
    with RuntimeState.open_sync(tmp_path / "store.db") as state:
        state.create_thread(workspace=tmp_path)
        store = state.persistence
        before = _projection_rows(store)
        store.run_sync(lambda: store._connection.execute(
            "UPDATE rollout_records SET payload_json = '{broken'",
        ))
        report = verify_store(store)
        assert not report.valid
        assert any("record" in error for error in report.errors)
        with pytest.raises(RuntimeError, match="corrupt|canonical|record"):
            rebuild_store(store)
        assert _projection_rows(store) == before

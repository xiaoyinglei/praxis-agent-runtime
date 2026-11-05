from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from agent_runtime.harness import artifact_ops, turn_ops
from agent_runtime.harness.rollout import RolloutStore
from agent_runtime.harness.rollout_integrity import rebuild_store, verify_store
from agent_runtime.harness.runtime_store import RuntimeStore


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
    with RuntimeStore.open_sync(tmp_path / "store.db") as state:
        for index in range(3):
            thread = turn_ops.create_thread(state, workspace=tmp_path)
            turn = turn_ops.start_turn(
                state,
                thread_id=thread.thread_id,
                user_message=f"turn {index}",
                binding_manifest={"model_id": "test-model"},
            )
        artifact_ops.commit_artifact(
            state,
            turn_id=turn.turn_id,
            content=b"keep this blob",
            media_type="text/plain",
            name="result.txt",
        )
        store = state.persistence
        expected = _projection_rows(store)
        store.run_sync(
            lambda: store._connection.execute(
                "UPDATE threads SET settings_json = '{\"corrupt\":true}' WHERE thread_id = ?",
                (thread.thread_id,),
            )
        )
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
    with RuntimeStore.open_sync(database) as state, RuntimeStore.open_sync(database) as writer:
        turn_ops.create_thread(state, workspace=tmp_path)
        store = state.persistence
        persist = store._persist_projection_state
        concurrent = []

        def interleave(*args, **kwargs):
            if not concurrent and kwargs.get("connection") is not None:
                concurrent.append(turn_ops.create_thread(writer, workspace=tmp_path))
            return persist(*args, **kwargs)

        monkeypatch.setattr(store, "_persist_projection_state", interleave)
        with pytest.raises(RuntimeError, match="changed.*rebuild|rebuild.*changed"):
            rebuild_store(store)
        assert concurrent
        assert state.read_thread(concurrent[0].thread_id).thread_id == concurrent[0].thread_id
        assert verify_store(store).valid


def test_rebuild_rejects_corrupt_canonical_without_changing_projections(tmp_path: Path) -> None:
    with RuntimeStore.open_sync(tmp_path / "store.db") as state:
        turn_ops.create_thread(state, workspace=tmp_path)
        store = state.persistence
        before = _projection_rows(store)
        store.run_sync(
            lambda: store._connection.execute(
                "UPDATE rollout_records SET payload_json = '{broken'",
            )
        )
        report = verify_store(store)
        assert not report.valid
        assert any("record" in error for error in report.errors)
        with pytest.raises(RuntimeError, match="corrupt|canonical|record"):
            rebuild_store(store)
        assert _projection_rows(store) == before


@pytest.mark.parametrize("version", ["physical", "writer"])
def test_unsupported_schema_is_rejected_before_schema_changes(tmp_path, version):
    database = tmp_path / "store.sqlite"
    with RuntimeStore.open_sync(database):
        pass
    with closing(sqlite3.connect(database, autocommit=True)) as connection:
        connection.execute("DROP TABLE channel_entries")
        if version == "physical":
            connection.execute("PRAGMA user_version = 99")
        else:
            connection.execute("UPDATE store_metadata SET value='99' WHERE key='minimum_writer_schema'")
        before = tuple(connection.iterdump())
        with pytest.raises(RuntimeError, match="requires a newer version"):
            with RuntimeStore.open_sync(database):
                pass
        assert tuple(connection.iterdump()) == before
        assert connection.execute("PRAGMA user_version").fetchone()[0] == (99 if version == "physical" else 1)


def test_failed_migration_rolls_back_schema_and_writer_protocol(tmp_path, monkeypatch):
    database = tmp_path / "store.sqlite"
    with RuntimeStore.open_sync(database):
        pass
    with closing(sqlite3.connect(database, autocommit=True)) as connection:
        connection.execute("DROP TABLE channel_entries")
        connection.execute("UPDATE store_metadata SET value='7' WHERE key='minimum_writer_schema'")
        before = tuple(connection.iterdump())

        def fail_migration(self):
            raise RuntimeError("injected migration failure after additive DDL")

        monkeypatch.setattr(RolloutStore, "_upgrade_budget_reservation_sources", fail_migration)
        with pytest.raises(RuntimeError, match="injected migration failure"):
            with RuntimeStore.open_sync(database):
                pass
        assert tuple(connection.iterdump()) == before


def test_maintenance_upgrade_and_restore_are_independent(tmp_path):
    from agent_runtime.maintenance import backup_rollout, restore_rollout, upgrade_rollout, verify_rollout

    database = tmp_path / "store.sqlite"
    with RuntimeStore.open_sync(database) as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="save", binding_manifest={})
        artifact = artifact_ops.commit_artifact(
            store, turn_id=turn.turn_id, content=b"evidence", media_type="text/plain", name="result.txt",
        )
        records = store.list_turn_records(turn.turn_id)
        epoch = store.persistence.epoch
    with closing(sqlite3.connect(database, autocommit=True)) as connection:
        connection.execute("DROP TABLE channel_entries")
        connection.execute("UPDATE store_metadata SET value='7' WHERE key='minimum_writer_schema'")

    backup = backup_rollout(database, tmp_path / "backup")
    upgraded = upgrade_rollout(database, tmp_path / "upgrade")
    restored = restore_rollout(backup.parent, tmp_path / "restore")

    for candidate, writer_schema in ((database, "7"), (restored, "7"), (upgraded, "9")):
        assert verify_rollout(candidate).valid
        with RuntimeStore.open_sync(candidate, read_only=True) as store:
            assert store.persistence.metadata("minimum_writer_schema") == writer_schema
            assert store.persistence.epoch == epoch
            assert store.list_turn_records(turn.turn_id) == records
            assert store.read_artifact(artifact.artifact_id) == b"evidence"
    with pytest.raises(FileExistsError):
        restore_rollout(backup.parent, tmp_path / "restore")


def test_corrupt_restore_never_publishes_destination(tmp_path):
    import json

    from agent_runtime.maintenance import backup_rollout, restore_rollout

    database = tmp_path / "store.sqlite"
    with RuntimeStore.open_sync(database):
        pass
    backup = backup_rollout(database, tmp_path / "backup")
    manifest_path = backup.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["record_count"] += 1
    manifest_path.write_text(json.dumps(manifest))
    destination = tmp_path / "restore"
    with pytest.raises(RuntimeError, match="backup manifest does not match"):
        restore_rollout(backup.parent, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".restore.*"))


@pytest.mark.anyio
async def test_upgrade_refuses_database_with_live_host(tmp_path):
    import asyncio

    from agent_runtime.host import BackgroundHost
    from agent_runtime.maintenance import upgrade_rollout

    database = tmp_path / "store.sqlite"
    destination = tmp_path / "upgrade"
    async with BackgroundHost(database):
        with pytest.raises(RuntimeError, match="host already owns"):
            await asyncio.to_thread(upgrade_rollout, database, destination)
    assert not destination.exists()


@pytest.mark.anyio
async def test_explicit_upgrade_rebinds_queue_and_keeps_receipt_identity(tmp_path):
    import asyncio

    from agent_runtime import Agent
    from agent_runtime.background import BackgroundHandle
    from agent_runtime.host import BackgroundHost
    from agent_runtime.maintenance import backup_rollout, upgrade_rollout
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "store.sqlite"
    original = await Agent(
        workspace_path=workspace, checkpoint_db=database, model_session_path=None,
    ).start_background("inspect", input_id="one", autostart=False)
    copy = await asyncio.to_thread(backup_rollout, database, tmp_path / "backup")
    plain = BackgroundHandle(
        database=copy, workspace=workspace, thread_id=original.thread_id, turn_id=original.turn_id, autostart=False,
    )
    async with BackgroundHost(copy, model=SmallAnswerModel()) as host:
        await host.tick()
        status = await plain.wait(timeout=3)
        assert status["queue_status"] == "blocked"
        assert status["status"] is None

    upgraded = await asyncio.to_thread(upgrade_rollout, database, tmp_path / "upgrade")
    agent = Agent(workspace_path=workspace, checkpoint_db=upgraded, model_session_path=None)
    repeated = await agent.start_background("inspect", input_id="one", autostart=False)
    assert (repeated.thread_id, repeated.turn_id) == (original.thread_id, original.turn_id)
    with pytest.raises(ValueError, match="conflicts"):
        await agent.start_background("change request", input_id="one", autostart=False)
    async with BackgroundHost(upgraded, model=SmallAnswerModel()) as host:
        await host.tick()
        assert (await repeated.wait(timeout=3))["status"] == "completed"
    assert (await original.status())["status"] is None

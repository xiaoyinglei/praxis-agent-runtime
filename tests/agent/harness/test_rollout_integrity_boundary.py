from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.harness import Session
from agent_runtime.harness.rollout import RolloutStore, StoreBusyError
from agent_runtime.harness.state import RuntimeState


class OpenOnlyModel:
    def snapshot(self, *, thread_id, turn_id):
        return {'thread_id': thread_id, 'turn_id': turn_id, 'model_id': 'open-only'}

    def ensure_available(self, *args, **kwargs):
        pass

    def prepare(self, request):
        raise AssertionError('opening a Session must not call the model')

    async def dispatch(self, prepared, *, event_sink=None):
        raise AssertionError('opening a Session must not call the model')


def test_audit_holds_one_snapshot_during_legal_concurrent_commit(tmp_path: Path):
    database = tmp_path / 'run.db'
    with RuntimeState.open_sync(database) as reader, RuntimeState.open_sync(database) as writer:
        reader.create_thread(workspace=tmp_path)
        connection = reader.persistence._connection
        injected = []

        class Interleave:
            def __getattr__(self, key):
                return getattr(connection, key)

            def execute(self, sql, *args):
                if sql.startswith('SELECT * FROM threads') and not injected:
                    injected.append(writer.create_thread(workspace=tmp_path))
                return connection.execute(sql, *args)

        reader.run_sync(lambda: setattr(reader.persistence, '_connection', Interleave()))
        report = reader.verify()
        assert injected
        assert report.valid, report.errors
        assert reader.verify().valid


@pytest.mark.anyio
async def test_session_open_checks_selected_history_and_preserves_full_audit(tmp_path):
    database = tmp_path / 'run.db'
    async with await RuntimeState.open(database) as state:
        healthy = await state.run_async(state.create_thread, workspace=tmp_path)
        broken = await state.run_async(state.create_thread, workspace=tmp_path)
        await state.run_async(lambda: state.persistence._connection.execute(
            "UPDATE threads SET settings_json = '{\"unlogged\":true}' WHERE thread_id = ?",
            (broken.thread_id,),
        ).close())
    async with await Session.open(
        database=database, workspace=tmp_path, thread_id=healthy.thread_id, model=OpenOnlyModel(),
    ) as session:
        assert session.thread_id == healthy.thread_id
        assert not (await session.store.run_async(session.store.verify)).valid
    with pytest.raises(RuntimeError, match='integrity'):
        await Session.open(
            database=database, workspace=tmp_path, thread_id=broken.thread_id, model=OpenOnlyModel(),
        )


def test_fork_open_checks_canonical_ancestors_and_audit_borrows_write_transaction(tmp_path):
    with RuntimeState.open_sync(tmp_path / 'run.db') as state:
        parent = state.create_thread(workspace=tmp_path)
        turn = state.start_turn(thread_id=parent.thread_id, user_message='parent', binding_manifest={})
        state.complete_turn(turn_id=turn.turn_id, answer='parent answer')
        fork = state.fork_thread(from_turn_id=turn.turn_id)

        def transaction():
            with state.persistence.transaction():
                state.update_session_settings(fork.thread_id, {'pending': True})
                assert state.validate_conversation(thread_id=fork.thread_id).valid
                assert state.persistence._connection.in_transaction
                raise RuntimeError('rollback caller')

        with pytest.raises(RuntimeError, match='rollback caller'):
            state.run_sync(transaction)
        assert state.read_thread(fork.thread_id).settings == {}
        state.run_sync(lambda: state.persistence._connection.execute(
            "UPDATE items SET payload_json = '{\"text\":\"corrupt\"}' WHERE turn_id = ?", (turn.turn_id,),
        ).close())
        assert not state.validate_conversation(thread_id=fork.thread_id).valid


@pytest.mark.anyio
async def test_admission_rejects_unaccepted_work_but_close_drains_accepted_work(tmp_path):
    import asyncio
    import threading

    entered, release = threading.Event(), threading.Event()
    observed = []
    store = await RolloutStore.open(tmp_path / 'run.db', max_pending_operations=2)

    def blocked():
        entered.set()
        assert release.wait(3)
        observed.append('first')

    first = asyncio.create_task(store.run_async(blocked))
    try:
        while not entered.is_set():
            await asyncio.sleep(0)
        second = asyncio.create_task(store.run_async(lambda: observed.append('second')))
        await asyncio.sleep(0)
        with pytest.raises(StoreBusyError, match='no work was accepted'):
            await store.run_async(lambda: observed.append('must not run'))
        closing = asyncio.create_task(store.aclose())
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        await closing
        assert observed == ['first', 'second']
    finally:
        release.set()
        await store.aclose()


def test_artifact_install_and_reference_share_the_gc_write_boundary(tmp_path):
    import os
    import sqlite3

    database = tmp_path / 'run.db'
    with RuntimeState.open_sync(database) as state, RuntimeState.open_sync(database) as collector:
        thread = state.create_thread(workspace=tmp_path)
        turn = state.start_turn(thread_id=thread.thread_id, user_message='artifact', binding_manifest={})
        def install_orphan():
            with state.persistence.transaction():
                return state.persistence.install_blob(b'old orphan')[0]

        digest = state.run_sync(install_orphan)
        blob = state.artifact_root / 'blobs' / digest
        os.utime(blob, (1, 1))
        collector.run_sync(lambda: collector.persistence._connection.execute('PRAGMA busy_timeout=1').close())

        def collect_during_install(stage):
            if stage == 'after_rename':
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    collector.gc_unreferenced_artifacts(min_age_seconds=1)

        artifact = state.commit_artifact(
            turn_id=turn.turn_id, content=b'old orphan', media_type='text/plain', name='result.txt',
            fault_injector=collect_during_install,
        )
        assert state.read_artifact(artifact.artifact_id) == b'old orphan'
        report = collector.gc_unreferenced_artifacts(min_age_seconds=1)
        assert report.removed == () and report.errors == ()


def test_legacy_upgrade_preserves_valid_cross_thread_artifact_reference(tmp_path):
    from agent_runtime.harness.rollout import _projection_hash

    with RuntimeState.open_sync(tmp_path / 'run.db') as state:
        source = state.create_thread(workspace=tmp_path)
        turn = state.start_turn(thread_id=source.thread_id, user_message='source', binding_manifest={})
        artifact = state.commit_artifact(
            turn_id=turn.turn_id, content=b'shared', media_type='text/plain', name='shared.txt',
        )
        target = state.create_thread(workspace=tmp_path)
        target_turn = state.start_turn(thread_id=target.thread_id, user_message='target', binding_manifest={})
        state.record_tool_result(turn_id=target_turn.turn_id, operation_id=None, result={
            'tool_call_id': 'shared-artifact', 'attachments': [{
                'artifact_id': artifact.artifact_id, 'media_type': artifact.media_type, 'name': artifact.name,
            }],
        })

        def make_legacy():
            digest = _projection_hash(state.persistence._load_projection_state(target.thread_id), target.thread_id)
            state.persistence._connection.execute(
                'UPDATE projection_meta SET canonical_hash = ?, hash_version = 1 WHERE thread_id = ?',
                (digest, target.thread_id),
            ).close()

        state.run_sync(make_legacy)
        assert state.validate_conversation(thread_id=target.thread_id).valid
        state.update_session_settings(target.thread_id, {'upgraded': True})
        assert state.verify().valid

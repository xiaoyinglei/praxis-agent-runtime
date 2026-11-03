from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_runtime.harness import RuntimeStore, host_ops, turn_ops


def submit(store, workspace, command_id="request-1", text="inspect"):
    return host_ops.receive_command(
        store, workspace=workspace, command_id=command_id, kind="start", payload={"text": text}
    )


def test_submission_is_durable_idempotent_and_does_not_create_a_model_turn(tmp_path):
    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as store:
        first = submit(store, tmp_path)
        assert store.list_turns() == ()
        assert store.read_thread(first.thread_id).workspace == str(tmp_path)
    with RuntimeStore.open_sync(database) as store:
        assert submit(store, tmp_path) == first
        with pytest.raises(ValueError, match="conflict"):
            submit(store, tmp_path, text="different")
        assert len(host_ops.list_commands(store)) == 1
        assert store.verify().valid
        store.rebuild_projections()
        assert host_ops.read_command(store, first.command_id) == first


def test_two_connections_can_only_claim_one_driver_for_a_thread(tmp_path):
    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as store:
        first = submit(store, tmp_path)
        host_ops.receive_command(
            store, workspace=tmp_path, thread_id=first.thread_id, command_id="request-2",
            kind="start", payload={"text": "next"},
        )

    def claim(owner):
        with RuntimeStore.open_sync(database) as store:
            return host_ops.claim_next(store, owner=owner, now=100, lease_seconds=10)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, ["worker-1", "worker-2"]))
    assert len([c for c in claims if c is not None]) == 1
    assert next(c for c in claims if c is not None).command_id == first.command_id


def test_expiry_increments_generation_and_rejects_old_commit(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        original = submit(store, tmp_path)
        first = host_ops.claim_next(store, owner="old", now=100, lease_seconds=10)
        assert host_ops.claim_next(store, owner="new", now=109, lease_seconds=10) is None
        second = host_ops.claim_next(store, owner="new", now=111, lease_seconds=10)
        assert second.command_id == original.command_id
        assert second.generation == first.generation + 1
        assert not host_ops.settle_command(store, first, status="applied", result={}, now=112)
        assert host_ops.settle_command(store, second, status="applied", result={}, now=112)
        assert store.verify().valid


def test_heartbeat_and_control_inputs_do_not_wait_for_a_running_driver(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        started = submit(store, tmp_path)
        claim = host_ops.claim_next(store, owner="worker", now=100, lease_seconds=10)
        host_ops.renew_claim(store, claim, now=105, lease_seconds=20)
        host_ops.receive_command(
            store, workspace=tmp_path, thread_id=started.thread_id, turn_id=started.turn_id,
            command_id="steering", kind="input", payload={"text": "inspect tests too"},
        )
        control = host_ops.claim_next(store, owner="worker", control=True, now=111, lease_seconds=10)
        assert control.command_id == "steering"
        assert host_ops.claim_next(store, owner="other", now=112, lease_seconds=10) is None


def test_command_cannot_target_another_workspace_or_thread_turn(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        first = submit(store, tmp_path)
        other = tmp_path / "other"
        other.mkdir()
        with pytest.raises(PermissionError):
            host_ops.receive_command(
                store, workspace=other, thread_id=first.thread_id, turn_id=first.turn_id,
                command_id="forged", kind="cancel", payload={},
            )
        unrelated = turn_ops.start_turn(
            store, thread_id=turn_ops.create_thread(store, workspace=tmp_path).thread_id,
            user_message="unrelated", binding_manifest={},
        )
        with pytest.raises(PermissionError):
            host_ops.receive_command(
                store, workspace=tmp_path, thread_id=first.thread_id, turn_id=unrelated.turn_id,
                command_id="wrong-turn", kind="cancel", payload={},
            )
        assert len(host_ops.list_commands(store)) == 1


def test_missing_turn_controls_are_only_allowed_for_a_canonical_queued_start(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        start = submit(store, tmp_path)
        with pytest.raises(PermissionError):
            host_ops.receive_command(
                store, workspace=tmp_path, thread_id=start.thread_id, turn_id="invented",
                command_id="invented-control", kind="input", payload={"text": "wrong"},
            )


def test_legacy_database_is_readable_without_mutating_and_upgrades_on_write(tmp_path):
    import sqlite3

    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="before hosts", binding_manifest={})
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE host_commands")
    connection.commit()
    connection.close()
    with RuntimeStore.open_sync(database, read_only=True) as store:
        assert store.verify().valid
        assert len(store.list_turns()) == 1
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT name FROM sqlite_master WHERE name='host_commands'").fetchone() is None
    connection.close()
    with RuntimeStore.open_sync(database) as store:
        submit(store, tmp_path)
        assert store.verify().valid
        store.rebuild_projections()
        assert store.verify().valid

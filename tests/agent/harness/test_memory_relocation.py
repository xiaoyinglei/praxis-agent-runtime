from __future__ import annotations

import hashlib
import json
import sqlite3
import time

import pytest

from agent_runtime.harness import RuntimeState
from agent_runtime.harness.memory_identity import (
    bind_memory_thread,
    memory_namespace,
    memory_scope,
    relocate_memory,
    thread_in_memory_scope,
    thread_memory_scope,
)
from agent_runtime.harness.protocol import ContextSourceChangedError


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _put(state, key, value) -> None:
    def write():
        with state.persistence.transaction():
            state.persistence.put_metadata(key, value)

    state.run_sync(write)


def _metadata(database):
    with sqlite3.connect(database) as connection:
        return connection.execute("SELECT key, value FROM store_metadata ORDER BY key").fetchall()


def _legacy(state, workspace, *, versions=2):
    namespace = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()
    for version in range(1, versions + 1):
        memory = {
            "version": version,
            "entries": [{"key": "keep", "text": "Use uv", "source_item_ids": ["source-1"]}],
            "processed": {"source-thread": "source-revision"},
            "source_cursors": {"source-thread": {"offset": 2}},
            "deleted_keys": ["removed"],
            "deleted_sources": ["deleted-source"],
            "entry_history": {"removed": [{"action": "forget", "version": 1}]},
            "use_memories": True,
            "generate_memories": True,
        }
        value = _canonical({"state": memory, "sha256": hashlib.sha256(_canonical(memory).encode()).hexdigest()})
        _put(state, f"memory:{namespace}:version:{version}", value)
    _put(state, f"memory:{namespace}", value)
    jobs = {"source-thread": {"status": "pending", "turn_ids": [], "revision": 3, "next_cursor": {"offset": 2}}}
    _put(state, f"memory-jobs:{namespace}", _canonical(jobs))
    return namespace, memory, jobs


def test_actual_directory_move_keeps_identity_versions_jobs_and_frozen_history(tmp_path):
    source = tmp_path / "original"
    destination = tmp_path / "renamed"
    source.mkdir()
    with RuntimeState.open_sync(tmp_path / "host.db") as state:
        thread = state.create_thread(workspace=source)
        turn = state.start_turn(
            thread_id=thread.thread_id, user_message="remember", binding_manifest={"workspace": str(source)}
        )
        state.complete_turn(turn_id=turn.turn_id, answer="done")
        original_records = state.list_records(thread.thread_id)
        legacy_id, memory, jobs = _legacy(state, source)
        scope_id = memory_namespace(state, source, create=True)
        versions = [state.persistence.metadata(f"memory:{scope_id}:version:{version}") for version in (1, 2)]
        assert scope_id != legacy_id
        source.rename(destination)
        moved = relocate_memory(state, source, destination, expected_version=2)
        assert moved["scope_id"] == scope_id
        assert moved["workspace"] == str(destination)
        assert moved["previous_workspace"] == str(source)
        assert moved["binding_generation"] == 2 and moved["version"] == 2
        assert memory_scope(state, source) is None
        assert memory_namespace(state, destination) == scope_id
        assert state.list_records(thread.thread_id) == original_records
        assert state.read_thread(thread.thread_id).workspace == str(source)
        assert state.read_turn(turn.turn_id).binding_manifest["workspace"] == str(source)
        assert thread_in_memory_scope(state, destination, thread.thread_id)
        assert thread_memory_scope(state, thread.thread_id) == scope_id
        assert [state.persistence.metadata(f"memory:{scope_id}:version:{version}") for version in (1, 2)] == versions
        assert json.loads(state.persistence.metadata(f"memory:{scope_id}"))["state"] == memory
        assert json.loads(state.persistence.metadata(f"memory-jobs:{scope_id}")) == jobs
        assert state.persistence.metadata(f"memory:{legacy_id}") is None
        assert state.persistence.metadata(f"memory:{legacy_id}:version:1") is None
        assert state.verify().valid


def test_reused_old_path_gets_a_new_scope_and_cannot_access_moved_threads(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    with RuntimeState.open_sync(tmp_path / "host.db") as state:
        old_thread = state.create_thread(workspace=source)
        old_scope = memory_namespace(state, source, create=True)
        relocate_memory(state, source, destination)
        new_thread = state.create_thread(workspace=source)
        assert not thread_in_memory_scope(state, source, old_thread.thread_id)
        assert thread_in_memory_scope(state, source, new_thread.thread_id)
        new_scope = memory_namespace(state, source, create=True)
        assert new_scope != old_scope
        assert thread_memory_scope(state, old_thread.thread_id) == old_scope
        assert thread_memory_scope(state, new_thread.thread_id) == new_scope
        assert not thread_in_memory_scope(state, source, old_thread.thread_id)
        assert not thread_in_memory_scope(state, destination, new_thread.thread_id)


def test_read_only_legacy_reads_do_not_promote_or_write(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "host.db"
    with RuntimeState.open_sync(database) as state:
        thread = state.create_thread(workspace=workspace)
        legacy_id, _, _ = _legacy(state, workspace)
    with RuntimeState.open_sync(database, read_only=True) as state:

        def forbidden(*args, **kwargs):
            raise AssertionError("read attempted to write metadata")

        monkeypatch.setattr(state.persistence, "put_metadata", forbidden)
        assert memory_namespace(state, workspace) == legacy_id
        assert memory_scope(state, workspace) is None
        assert thread_memory_scope(state, thread.thread_id) is None
        assert thread_in_memory_scope(state, workspace, thread.thread_id)


@pytest.mark.parametrize("relocated", [False, True])
def test_read_only_reopen_preserves_promoted_memory_and_relocated_membership(tmp_path, monkeypatch, relocated):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    database = tmp_path / "host.db"
    with RuntimeState.open_sync(database) as state:
        thread = state.create_thread(workspace=source)
        _, memory, jobs = _legacy(state, source)
        scope_id = memory_namespace(state, source, create=True)
        workspace = source
        if relocated:
            source.rename(destination / "moved")
            workspace = destination / "moved"
            relocate_memory(state, source, workspace, expected_version=2)
        expected_scope = memory_scope(state, workspace)
        previous_version = state.read_memory_state(workspace, version=1)
    metadata_before = _metadata(database)
    with RuntimeState.open_sync(database, read_only=True) as state:

        def forbidden(*args, **kwargs):
            raise AssertionError("read attempted to write metadata")

        monkeypatch.setattr(state.persistence, "put_metadata", forbidden)
        monkeypatch.setattr(state.persistence, "delete_metadata", forbidden)
        assert memory_namespace(state, workspace) == scope_id
        assert memory_scope(state, workspace) == expected_scope
        assert thread_memory_scope(state, thread.thread_id) == scope_id
        assert thread_in_memory_scope(state, workspace, thread.thread_id)
        assert state.read_memory_state(workspace) == memory
        assert state.read_memory_state(workspace, version=1) == previous_version
        assert state.read_memory_jobs(workspace) == jobs
        assert state.read_thread(thread.thread_id).workspace == str(source)
        if relocated:
            assert memory_scope(state, source) is None
            assert not thread_in_memory_scope(state, source, thread.thread_id)
            assert state.read_memory_state(source)["version"] == 0
    assert _metadata(database) == metadata_before


def test_binding_threads_does_not_initialize_a_disabled_memory_scope(tmp_path):
    with RuntimeState.open_sync(tmp_path / "host.db") as state:
        thread = state.create_thread(workspace=tmp_path)
        bind_memory_thread(state, thread.thread_id, tmp_path)
        assert memory_scope(state, tmp_path) is None
        assert thread_memory_scope(state, thread.thread_id) is None
        scope_id = memory_namespace(state, tmp_path, create=True)
        assert thread_memory_scope(state, thread.thread_id) == scope_id


@pytest.mark.parametrize("destination_kind", ["scope", "legacy"])
def test_destination_data_conflicts_without_changing_either_namespace(tmp_path, destination_kind):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    with RuntimeState.open_sync(tmp_path / "host.db") as state:
        source_id = memory_namespace(state, source, create=True)
        if destination_kind == "scope":
            destination_id = memory_namespace(state, destination, create=True)
        else:
            destination_id, _, _ = _legacy(state, destination)
        before = memory_scope(state, source)
        with pytest.raises(ContextSourceChangedError, match="destination"):
            relocate_memory(state, source, destination)
        assert memory_scope(state, source) == before
        assert memory_namespace(state, source) == source_id
        assert memory_namespace(state, destination) == destination_id


def test_stale_expected_version_preserves_binding_and_expired_lease(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    with RuntimeState.open_sync(tmp_path / "host.db") as state:
        _legacy(state, source)
        namespace = memory_namespace(state, source, create=True)
        lease = _canonical({"owner": "worker", "expires_at": 1})
        _put(state, f"memory-job:{namespace}", lease)
        before = memory_scope(state, source)
        with pytest.raises(ContextSourceChangedError, match="version"):
            relocate_memory(state, source, destination, expected_version=1)
        assert memory_scope(state, source) == before
        assert state.persistence.metadata(f"memory-job:{namespace}") == lease


@pytest.mark.parametrize("legacy", [False, True])
def test_active_memory_lease_refuses_relocation_and_promotion(tmp_path, legacy):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    with RuntimeState.open_sync(tmp_path / "host.db") as state:
        namespace = _legacy(state, source)[0] if legacy else memory_namespace(state, source, create=True)
        _put(state, f"memory-job:{namespace}", _canonical({"owner": "worker", "expires_at": time.time() + 60}))
        with pytest.raises(ContextSourceChangedError, match="lease"):
            relocate_memory(state, source, destination)
        if legacy:
            with pytest.raises(ContextSourceChangedError, match="lease"):
                memory_namespace(state, source, create=True)
            assert memory_scope(state, source) is None


def test_unresolved_maintenance_turn_blocks_relocation_even_after_lease_expiry(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    with RuntimeState.open_sync(tmp_path / "host.db") as state:
        namespace = memory_namespace(state, source, create=True)
        thread = state.create_thread(workspace=source)
        turn = state.start_turn(
            thread_id=thread.thread_id, user_message="extract", binding_manifest={"memory_maintenance": True}
        )
        _put(
            state,
            f"memory-jobs:{namespace}",
            _canonical({thread.thread_id: {"turn_ids": [turn.turn_id], "status": "failed"}}),
        )
        _put(state, f"memory-job:{namespace}", _canonical({"owner": "expired", "expires_at": 1}))
        with pytest.raises(ContextSourceChangedError, match="maintenance"):
            relocate_memory(state, source, destination)
        assert memory_scope(state, source)["binding_generation"] == 1


def test_job_reconciliation_blocks_relocation_without_an_active_turn_or_lease(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    database = tmp_path / "host.db"
    with RuntimeState.open_sync(database) as state:
        namespace = memory_namespace(state, source, create=True)
        thread = state.create_thread(workspace=source)
        _put(
            state,
            f"memory-jobs:{namespace}",
            _canonical(
                {
                    thread.thread_id: {"turn_ids": [], "status": "needs_reconciliation"},
                }
            ),
        )
        before = _metadata(database)
        with pytest.raises(ContextSourceChangedError, match="requires reconciliation"):
            relocate_memory(state, source, destination)
        assert _metadata(database) == before


@pytest.mark.parametrize("operation_status", ["dispatched", "unknown"])
def test_terminal_maintenance_turn_with_unresolved_model_operation_blocks_relocation(tmp_path, operation_status):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    database = tmp_path / "host.db"
    with RuntimeState.open_sync(database) as state:
        namespace = memory_namespace(state, source, create=True)
        thread = state.create_thread(workspace=source)
        turn = state.start_turn(
            thread_id=thread.thread_id,
            user_message="extract",
            binding_manifest={"memory_maintenance": True},
        )
        operation = state.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="request",
            context_hash="context",
            tool_hash="tools",
            wire_hash="wire",
            request_ref={"input_revision": 1, "request_id": "maintenance-request"},
        )
        attempt = state.dispatch_model_attempt(operation.operation_id)
        state.fail_turn(turn_id=turn.turn_id, reason_code="worker_failed", message="worker stopped")
        if operation_status == "unknown":
            state.mark_model_attempt_unknown(
                operation_id=operation.operation_id,
                attempt_id=attempt.attempt_id,
                generation=attempt.generation,
            )
        assert state.read_turn(turn.turn_id).status == "failed"
        assert state.list_model_operations(turn.turn_id)[0].status == operation_status
        _put(
            state,
            f"memory-jobs:{namespace}",
            _canonical(
                {
                    thread.thread_id: {"turn_ids": [turn.turn_id], "status": "failed"},
                }
            ),
        )
        _put(state, f"memory-job:{namespace}", _canonical({"owner": "expired", "expires_at": 1}))
        before = _metadata(database)
        with pytest.raises(ContextSourceChangedError, match="maintenance operation is unresolved"):
            relocate_memory(state, source, destination)
        assert _metadata(database) == before


def test_terminal_maintenance_turn_with_tool_reconciliation_blocks_relocation(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    database = tmp_path / "host.db"
    with RuntimeState.open_sync(database) as state:
        namespace = memory_namespace(state, source, create=True)
        thread = state.create_thread(workspace=source)
        turn = state.start_turn(
            thread_id=thread.thread_id,
            user_message="extract",
            binding_manifest={"memory_maintenance": True},
        )
        state.record_tool_call(
            turn_id=turn.turn_id,
            tool_call_id="maintenance-call",
            tool_name="inspect",
            arguments={},
            origin={},
        )
        values = {
            "turn_id": turn.turn_id,
            "operation_id": "maintenance-operation",
            "tool_call_id": "maintenance-call",
            "tool_name": "inspect",
            "arguments_digest": "digest",
            "execution_revision": "v1",
            "idempotent": True,
            "attempt_count": 0,
            "error_code": None,
            "requires_reconciliation": False,
        }
        state.record_tool_execution_state(status="prepared", **values)
        state.record_tool_execution_state(status="ready", **values)
        claim = state.claim_tool_operation(operation_id="maintenance-operation", worker_id="worker", lease_seconds=30)
        assert state.commit_tool_operation_outcome(
            operation_id=claim.operation_id,
            claim_generation=claim.claim_generation,
            fencing_token=claim.fencing_token,
            status="failed",
            attempt_count=1,
            error_code="partial_result",
            requires_reconciliation=True,
        )
        state.fail_turn(turn_id=turn.turn_id, reason_code="worker_failed", message="worker stopped")
        assert state.read_turn(turn.turn_id).status == "failed"
        assert state.list_tool_operations(turn.turn_id)[0].requires_reconciliation
        _put(
            state,
            f"memory-jobs:{namespace}",
            _canonical(
                {
                    thread.thread_id: {"turn_ids": [turn.turn_id], "status": "failed"},
                }
            ),
        )
        before = _metadata(database)
        with pytest.raises(ContextSourceChangedError, match="maintenance operation is unresolved"):
            relocate_memory(state, source, destination)
        assert _metadata(database) == before


def test_relocation_failure_rolls_back_promotion_bindings_and_all_keys(tmp_path, monkeypatch):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    with RuntimeState.open_sync(tmp_path / "host.db") as state:
        legacy_id, _, _ = _legacy(state, source)
        original = state.persistence.put_metadata
        writes = 0

        def fail(key, value, **kwargs):
            nonlocal writes
            writes += 1
            if writes == 4:
                raise OSError("injected metadata failure")
            return original(key, value, **kwargs)

        monkeypatch.setattr(state.persistence, "put_metadata", fail)
        with pytest.raises(OSError, match="injected"):
            relocate_memory(state, source, destination)
        assert memory_scope(state, source) is None
        assert memory_scope(state, destination) is None
        assert memory_namespace(state, source) == legacy_id
        assert state.persistence.metadata(f"memory:{legacy_id}") is not None
        assert state.persistence.metadata(f"memory:{legacy_id}:version:1") is not None


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("failure_point", ["audit", "target_association"])
def test_late_relocation_failure_restores_all_metadata_and_thread_associations(
    tmp_path,
    monkeypatch,
    legacy,
    failure_point,
):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    database = tmp_path / "host.db"
    with RuntimeState.open_sync(database) as state:
        source_thread = state.create_thread(workspace=source)
        target_thread = state.create_thread(workspace=destination)
        legacy_id, _, _ = _legacy(state, source)
        namespace = legacy_id if legacy else memory_namespace(state, source, create=True)
        _put(state, f"memory-job:{namespace}", _canonical({"owner": "expired", "expires_at": 1}))
        before = _metadata(database)
        source_association = thread_memory_scope(state, source_thread.thread_id)
        original = state.persistence.put_metadata
        failed_late = False

        def fail(key, value, **kwargs):
            nonlocal failed_late
            selected = (
                key.startswith("memory-relocation:")
                if failure_point == "audit"
                else key == "memory-thread-scope:" + target_thread.thread_id
            )
            if selected:
                assert state.persistence.metadata("memory-binding:" + legacy_id) is None
                assert memory_scope(state, destination)["binding_generation"] == 2
                failed_late = True
                # Fail after the selected mutation so its rollback is exercised too.
                original(key, value, **kwargs)
                raise OSError("injected late relocation failure")
            return original(key, value, **kwargs)

        monkeypatch.setattr(state.persistence, "put_metadata", fail)
        with pytest.raises(OSError, match="injected late"):
            relocate_memory(state, source, destination)
        assert failed_late
        assert _metadata(database) == before
        assert memory_namespace(state, source) == namespace
        assert memory_scope(state, destination) is None
        assert thread_memory_scope(state, source_thread.thread_id) == source_association
        assert thread_memory_scope(state, target_thread.thread_id) is None
        assert state.verify().valid


def test_relocation_retry_is_idempotent_and_adopts_unbound_target_threads(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    with RuntimeState.open_sync(tmp_path / "host.db") as state:
        namespace = memory_namespace(state, source, create=True)
        target_thread = state.create_thread(workspace=destination)
        first = relocate_memory(state, source, destination)
        second = relocate_memory(state, source, destination)
        assert first == second
        assert first["scope_id"] == namespace
        assert first["binding_generation"] == 2 and first["version"] == 0
        assert thread_memory_scope(state, target_thread.thread_id) == namespace
        assert thread_in_memory_scope(state, destination, target_thread.thread_id)


def test_missing_source_is_rejected_and_legacy_version_integrity_is_checked(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    with RuntimeState.open_sync(tmp_path / "host.db") as state:
        with pytest.raises(KeyError, match="source"):
            relocate_memory(state, source, destination)
        namespace, _, _ = _legacy(state, source)
        _put(state, f"memory:{namespace}:version:1", _canonical({"state": {"version": 1}, "sha256": "wrong"}))
        with pytest.raises(RuntimeError, match="integrity"):
            memory_namespace(state, source, create=True)
        assert memory_scope(state, source) is None

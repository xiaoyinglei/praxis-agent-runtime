from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_runtime.harness import RuntimeStore, memory_ops, turn_ops
from agent_runtime.harness.memory import memory_context_revoked, memory_snapshot
from agent_runtime.harness.memory_management import explain_memory, remember_memory


def test_scope_move_preserves_frozen_reads_and_excludes_reused_source_path(tmp_path: Path) -> None:
    from agent_runtime.harness.memory import read_frozen_memory_state

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as state:
        remembered = remember_memory(state, old, key="test", title="Tests", text="Use uv.")
        source_item = state.read_item(remembered["entries"][0]["source_item_ids"][0])
        snapshot = memory_snapshot(state, old, navigation_only=True)
        scope_id = state.memory_scope(old)["scope_id"]
        old.rename(new)
        memory_ops.relocate_memory(state, previous_workspace=old, workspace=new, expected_version=remembered["version"])
        old.mkdir()
        unrelated = remember_memory(state, old, key="test", title="Tests", text="Use make.")
        other_item = state.read_item(unrelated["entries"][0]["source_item_ids"][0])
        assert state.memory_scope(old)["scope_id"] != scope_id
        assert state.memory_scope(new)["scope_id"] == scope_id
        assert state.thread_in_memory_scope(new, source_item.thread_id)
        assert not state.thread_in_memory_scope(new, other_item.thread_id)
        assert state.read_thread(source_item.thread_id).workspace == str(old)
        frozen = read_frozen_memory_state(state, thread_id=source_item.thread_id, snapshot=snapshot)
        assert frozen["entries"][0]["text"] == "Use uv."
        assert explain_memory(state, new, key="test")["entry"]["entry_id"] == remembered["entries"][0]["entry_id"]
        assert memory_context_revoked(state, new, previous=snapshot, current=memory_snapshot(state, new)) is False
        assert memory_context_revoked(state, old, previous=snapshot, current=memory_snapshot(state, old)) is True


def test_stale_memory_claim_cannot_commit_update_jobs_or_release_new_binding(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as state:
        source = turn_ops.create_thread(state, workspace=old)
        expired = memory_ops.claim_memory_job(state, old, owner="worker", now=time.time() - 5, lease_seconds=1)
        assert expired is not None
        memory_ops.relocate_memory(state, previous_workspace=old, workspace=new, expected_version=0)
        current = memory_ops.claim_memory_job(state, new, owner="worker", now=time.time(), lease_seconds=60)
        assert current is not None
        assert current.scope_id == expired.scope_id
        assert current.binding_generation != expired.binding_generation
        with pytest.raises(RuntimeError, match="(claim|lease|binding|scope)"):
            memory_ops.commit_memory_state(
                state, old, expected_version=0, state=state.read_memory_state(old), claim=expired
            )
        with pytest.raises(RuntimeError, match="(claim|lease|binding|scope)"):
            memory_ops.write_memory_job(state, new, source.thread_id, {"status": "done"}, claim=expired)
        memory_ops.release_memory_job(state, new, claim=expired)
        assert memory_ops.claim_memory_job(state, new, owner="other", now=time.time(), lease_seconds=60) is None
        memory_ops.write_memory_job(state, new, source.thread_id, {"status": "pending"}, claim=current)
        assert state.read_memory_jobs(new)[source.thread_id]["status"] == "pending"


def test_snapshot_content_and_identity_share_one_database_snapshot(tmp_path: Path, monkeypatch) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as reader, RuntimeStore.open_sync(database) as writer:
        remembered = remember_memory(writer, old, key="test", title="Tests", text="Use uv.")
        original_scope = writer.memory_scope(old)["scope_id"]
        read = reader.read_memory_state

        def move_after_content(*args, **kwargs):
            content = read(*args, **kwargs)
            memory_ops.relocate_memory(writer, previous_workspace=old, workspace=new)
            remember_memory(writer, old, key="test", title="Tests", text="Use make.")
            return content

        monkeypatch.setattr(reader, "read_memory_state", move_after_content)
        snapshot = memory_snapshot(reader, old)
        assert snapshot["version"] == remembered["version"]
        assert snapshot["entries"][0]["text"] == "Use uv."
        assert snapshot["scope_id"] == original_scope
        assert writer.memory_scope(old)["scope_id"] != original_scope


def test_old_path_reuse_and_multiple_moves_fence_all_prior_worker_claims(tmp_path: Path) -> None:
    old, new, final = (tmp_path / name for name in ("old", "new", "final"))
    for workspace in (old, new, final):
        workspace.mkdir()
    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as worker, RuntimeStore.open_sync(database) as operator:
        original = memory_ops.claim_memory_job(worker, old, owner="worker", now=time.time() - 5, lease_seconds=1)
        memory_ops.relocate_memory(operator, previous_workspace=old, workspace=new)
        reused = remember_memory(operator, old, key="test", title="Tests", text="Unrelated project.")
        intermediate = memory_ops.claim_memory_job(worker, new, owner="worker", now=time.time() - 5, lease_seconds=1)
        memory_ops.relocate_memory(operator, previous_workspace=new, workspace=final)
        current = memory_ops.claim_memory_job(worker, final, owner="worker", now=time.time(), lease_seconds=60)
        assert original is not None and intermediate is not None and current is not None
        for stale in (original, intermediate):
            for workspace in (old, new, final):
                with pytest.raises(RuntimeError, match="binding|lease"):
                    memory_ops.commit_memory_state(
                        worker,
                        workspace,
                        expected_version=0,
                        state=worker.read_memory_state(workspace),
                        claim=stale,
                    )
                memory_ops.release_memory_job(worker, workspace, claim=stale)
        assert operator.read_memory_state(old) == reused
        assert operator.read_memory_state(final)["version"] == 0
        assert memory_ops.claim_memory_job(operator, final, owner="other", now=time.time(), lease_seconds=60) is None


def test_fork_inherits_memory_identity_even_after_original_path_is_reused(tmp_path: Path) -> None:
    from agent_runtime.harness.memory import read_frozen_memory_state
    from agent_runtime.harness.memory_identity import thread_memory_scope

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as state:
        remember_memory(state, old, key="test", title="Tests", text="Use uv.")
        snapshot = memory_snapshot(state, old, navigation_only=True)
        thread = turn_ops.create_thread(state, workspace=old)
        turn = turn_ops.start_turn(state, thread_id=thread.thread_id, user_message="fork me", binding_manifest={})
        turn_ops.complete_turn(state, turn_id=turn.turn_id, answer="done")
        before_move = turn_ops.fork_thread(state, from_turn_id=turn.turn_id)
        assert thread_memory_scope(state, before_move.thread_id) == snapshot["scope_id"]
        memory_ops.relocate_memory(state, previous_workspace=old, workspace=new)
        remember_memory(state, old, key="test", title="Tests", text="Use make.")
        after_move = turn_ops.fork_thread(state, from_turn_id=turn.turn_id)
        assert thread_memory_scope(state, after_move.thread_id) == snapshot["scope_id"]
        assert state.thread_in_memory_scope(new, after_move.thread_id)
        assert not state.thread_in_memory_scope(old, after_move.thread_id)
        frozen = read_frozen_memory_state(state, thread_id=after_move.thread_id, snapshot=snapshot)
        assert frozen["entries"][0]["text"] == "Use uv."


def test_reused_worker_owner_does_not_make_expired_claim_valid_again(tmp_path: Path) -> None:
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as state:
        expired = memory_ops.claim_memory_job(state, tmp_path, owner="worker", now=time.time() - 5, lease_seconds=1)
        current = memory_ops.claim_memory_job(state, tmp_path, owner="worker", now=time.time(), lease_seconds=60)
        assert expired is not None and current is not None
        with pytest.raises(RuntimeError, match="lease"):
            memory_ops.commit_memory_state(
                state,
                tmp_path,
                expected_version=0,
                state=state.read_memory_state(tmp_path),
                claim=expired,
            )
        memory_ops.release_memory_job(state, tmp_path, claim=expired)
        assert memory_ops.claim_memory_job(state, tmp_path, owner="other", now=time.time(), lease_seconds=60) is None
        memory_ops.commit_memory_state(
            state, tmp_path, expected_version=0, state=state.read_memory_state(tmp_path), claim=current
        )

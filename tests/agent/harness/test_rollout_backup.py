from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime.harness import artifact_ops, turn_ops
from agent_runtime.harness.runtime_store import RuntimeStore


def test_backup_restores_database_and_artifacts_from_read_only_source(tmp_path: Path) -> None:
    database = tmp_path / "source.sqlite3"
    with RuntimeStore.open_sync(database) as state:
        thread = turn_ops.create_thread(state, workspace=tmp_path)
        turn = turn_ops.start_turn(
            state,
            thread_id=thread.thread_id,
            user_message="save this artifact",
            binding_manifest={"model_id": "test-model"},
        )
        artifact = artifact_ops.commit_artifact(
            state,
            turn_id=turn.turn_id,
            content=b"independent backup bytes",
            media_type="application/octet-stream",
            name="result.bin",
        )
        watermark = state.persistence.last_global_record_id()

    destination = tmp_path / "backup"
    with RuntimeStore.open_sync(database, read_only=True) as state:
        assert state.persistence.backup(destination) == destination
        with pytest.raises(FileExistsError):
            state.persistence.backup(destination)

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["last_record_id"] == watermark
    with RuntimeStore.open_sync(destination / "rollout.sqlite3") as restored:
        assert restored.read_thread(thread.thread_id).workspace == str(tmp_path)
        assert restored.read_turn(turn.turn_id).status == "running"
        assert restored.read_artifact(artifact.artifact_id) == b"independent backup bytes"
        assert restored.persistence.verify().valid


def test_backup_rejects_corrupt_blob_without_publishing_directory(tmp_path: Path) -> None:
    with RuntimeStore.open_sync(tmp_path / "source.sqlite3") as state:
        thread = turn_ops.create_thread(state, workspace=tmp_path)
        turn = turn_ops.start_turn(
            state,
            thread_id=thread.thread_id,
            user_message="retain verified bytes",
            binding_manifest={"model_id": "test-model"},
        )
        artifact = artifact_ops.commit_artifact(
            state,
            turn_id=turn.turn_id,
            content=b"original",
            media_type="application/octet-stream",
            name="result.bin",
        )
        state.artifact_blob_path(artifact.artifact_id).write_bytes(b"tampered")
        destination = tmp_path / "backup"
        with pytest.raises(RuntimeError, match="artifact blob.*mismatch"):
            state.persistence.backup(destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".backup.*"))

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime.harness.state import RuntimeState


def test_backup_restores_database_and_artifacts_from_read_only_source(tmp_path: Path) -> None:
    database = tmp_path / "source.sqlite3"
    with RuntimeState.open_sync(database) as state:
        thread = state.create_thread(workspace=tmp_path)
        turn = state.start_turn(
            thread_id=thread.thread_id,
            user_message="save this artifact",
            binding_manifest={"model_id": "test-model"},
        )
        artifact = state.commit_artifact(
            turn_id=turn.turn_id,
            content=b"independent backup bytes",
            media_type="application/octet-stream",
            name="result.bin",
        )
        watermark = state.persistence.last_global_record_id()

    destination = tmp_path / "backup"
    with RuntimeState.open_sync(database, read_only=True) as state:
        assert state.persistence.backup(destination) == destination
        with pytest.raises(FileExistsError):
            state.persistence.backup(destination)

    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["last_record_id"] == watermark
    with RuntimeState.open_sync(destination / "rollout.sqlite3") as restored:
        assert restored.read_thread(thread.thread_id).workspace == str(tmp_path)
        assert restored.read_turn(turn.turn_id).status == "running"
        assert restored.read_artifact(artifact.artifact_id) == b"independent backup bytes"
        assert restored.persistence.verify().valid


def test_backup_rejects_corrupt_blob_without_publishing_directory(tmp_path: Path) -> None:
    with RuntimeState.open_sync(tmp_path / "source.sqlite3") as state:
        thread = state.create_thread(workspace=tmp_path)
        turn = state.start_turn(
            thread_id=thread.thread_id,
            user_message="retain verified bytes",
            binding_manifest={"model_id": "test-model"},
        )
        artifact = state.commit_artifact(
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

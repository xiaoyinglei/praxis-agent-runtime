from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutStore


class InjectedArtifactCrashError(RuntimeError):
    pass


def _turn(store: RolloutStore, workspace: Path) -> str:
    thread = store.create_thread(workspace=workspace)
    return store.start_turn(
        thread_id=thread.thread_id,
        user_message="retain one binary artifact",
        binding_manifest={"model_alias": "test-model"},
    ).turn_id


@pytest.mark.parametrize(
    ("phase", "committed"),
    [
        ("after_temp_fsync", False),
        ("after_rename", False),
        ("before_sqlite_commit", False),
        ("after_sqlite_commit", True),
    ],
)
def test_artifact_write_before_reference_survives_every_crash_boundary(
    tmp_path: Path,
    phase: str,
    committed: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"

    def crash(observed: str) -> None:
        if observed == phase:
            raise InjectedArtifactCrashError(phase)

    with RolloutStore(database) as store:
        turn_id = _turn(store, workspace)
        with pytest.raises(InjectedArtifactCrashError, match=phase):
            store.commit_artifact(
                turn_id=turn_id,
                content=b"complete artifact bytes",
                media_type="application/octet-stream",
                name="result.bin",
                fault_injector=crash,
            )

    with RolloutStore(database) as reopened:
        artifacts = reopened.list_artifacts(turn_id)
        assert bool(artifacts) is committed
        if committed:
            assert reopened.read_artifact(artifacts[0].artifact_id) == b"complete artifact bytes"
            before = artifacts[0]
            reopened.rebuild_projections()
            assert reopened.read_artifact_metadata(before.artifact_id) == before
        assert reopened.verify().valid is True
        for path in reopened.artifact_root.rglob("*"):
            if path.is_file():
                os.utime(path, (time.time() - 10, time.time() - 10))
        removed = reopened.gc_unreferenced_artifacts(min_age_seconds=1)
        if committed:
            assert removed == ()
            assert reopened.read_artifact(artifacts[0].artifact_id) == b"complete artifact bytes"
        else:
            assert not any(path.is_file() for path in reopened.artifact_root.rglob("*"))


@pytest.mark.parametrize(
    ("phase", "committed"),
    [
        ("after_temp_fsync", False),
        ("after_rename", False),
        ("before_sqlite_commit", False),
        ("after_sqlite_commit", True),
    ],
)
def test_fresh_process_crash_never_leaves_a_record_pointing_to_partial_blob(
    tmp_path: Path,
    phase: str,
    committed: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        turn_id = _turn(store, workspace)
    script = """
import os
from pathlib import Path
from agent_runtime.harness import RolloutStore

database = Path(os.environ["ARTIFACT_TEST_DATABASE"])
turn_id = os.environ["ARTIFACT_TEST_TURN"]
phase = os.environ["ARTIFACT_TEST_PHASE"]

def crash(observed: str) -> None:
    if observed == phase:
        os._exit(97)

with RolloutStore(database) as store:
    store.commit_artifact(
        turn_id=turn_id,
        content=b"fresh process artifact",
        media_type="application/octet-stream",
        name="crash.bin",
        fault_injector=crash,
    )
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        env={
            **os.environ,
            "ARTIFACT_TEST_DATABASE": str(database),
            "ARTIFACT_TEST_TURN": turn_id,
            "ARTIFACT_TEST_PHASE": phase,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 97, completed.stderr
    with RolloutStore(database) as reopened:
        artifacts = reopened.list_artifacts(turn_id)
        assert bool(artifacts) is committed
        assert reopened.verify().valid is True
        if committed:
            assert reopened.read_artifact(artifacts[0].artifact_id) == b"fresh process artifact"


@pytest.mark.parametrize("damage", ["tamper", "delete"])
def test_corrupt_referenced_artifact_fails_verify_and_blocks_dispatch(
    tmp_path: Path,
    damage: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        turn_id = _turn(store, workspace)
        artifact = store.commit_artifact(
            turn_id=turn_id,
            content=b"trusted bytes",
            media_type="text/plain",
            name="evidence.txt",
        )
        store.record_tool_result(
            turn_id=turn_id,
            operation_id=None,
            result={
                "tool_call_id": "call-artifact-1",
                "tool_name": "artifact_tool",
                "is_error": False,
                "attachments": [
                    {
                        "artifact_id": artifact.artifact_id,
                        "media_type": artifact.media_type,
                        "name": artifact.name,
                    }
                ],
            },
        )
        operation = store.prepare_model_operation(
            turn_id=turn_id,
            request_hash="request-hash",
            context_hash="context-hash",
            tool_hash="tool-hash",
            wire_hash="wire-hash",
            request_ref={"request_id": f"{turn_id}:step:1"},
        )
        blob = store.artifact_blob_path(artifact.artifact_id)
        if damage == "tamper":
            blob.write_bytes(b"tampered bytes")
        else:
            blob.unlink()

        report = store.verify()
        assert report.valid is False
        assert any("artifact" in error for error in report.errors)
        with pytest.raises(RuntimeError, match="artifact integrity"):
            store.dispatch_model_attempt(operation.operation_id)


def test_tool_result_cannot_reference_an_unknown_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn_id = _turn(store, workspace)
        with pytest.raises(RuntimeError, match="unknown artifact"):
            store.record_tool_result(
                turn_id=turn_id,
                operation_id=None,
                result={
                    "tool_call_id": "call-artifact-1",
                    "tool_name": "artifact_tool",
                    "is_error": False,
                    "attachments": [
                        {
                            "artifact_id": "artifact-missing",
                            "media_type": "text/plain",
                            "name": "missing.txt",
                        }
                    ],
                },
            )

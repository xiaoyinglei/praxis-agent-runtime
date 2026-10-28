from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent_runtime.harness import RuntimeState


class InjectedArtifactCrashError(RuntimeError):
    pass


def _turn(store: RuntimeState, workspace: Path) -> str:
    thread = store.create_thread(workspace=workspace)
    return store.start_turn(
        thread_id=thread.thread_id,
        user_message="retain one binary artifact",
        binding_manifest={"model_id": "test-model"},
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
@pytest.mark.parametrize("from_file", [False, True])
def test_artifact_write_before_reference_survives_every_crash_boundary(
    tmp_path: Path,
    phase: str,
    committed: bool,
    from_file: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    source = tmp_path / "source.bin"
    source.write_bytes(b"complete artifact bytes")

    def crash(observed: str) -> None:
        if observed == phase:
            raise InjectedArtifactCrashError(phase)

    with RuntimeState.open_sync(database) as store:
        turn_id = _turn(store, workspace)
        with pytest.raises(InjectedArtifactCrashError, match=phase):
            store.commit_artifact(
                turn_id=turn_id,
                content=source if from_file else b"complete artifact bytes",
                media_type="application/octet-stream",
                name="result.bin",
                fault_injector=crash,
            )

    with RuntimeState.open_sync(database) as reopened:
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
        assert removed.errors == ()
        if committed:
            assert removed.removed == ()
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
@pytest.mark.parametrize("from_file", [False, True])
def test_fresh_process_crash_never_leaves_a_record_pointing_to_partial_blob(
    tmp_path: Path,
    phase: str,
    committed: bool,
    from_file: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    source = tmp_path / "source.bin"
    source.write_bytes(b"fresh process artifact")
    with RuntimeState.open_sync(database) as store:
        turn_id = _turn(store, workspace)
    script = """
import os
from pathlib import Path
from agent_runtime.harness import RuntimeState

database = Path(os.environ["ARTIFACT_TEST_DATABASE"])
turn_id = os.environ["ARTIFACT_TEST_TURN"]
phase = os.environ["ARTIFACT_TEST_PHASE"]

def crash(observed: str) -> None:
    if observed == phase:
        os._exit(97)

with RuntimeState.open_sync(database) as store:
    store.commit_artifact(
        turn_id=turn_id,
        content=Path(os.environ["ARTIFACT_TEST_SOURCE"]) if os.environ["ARTIFACT_TEST_FROM_FILE"] == "1"
                else b"fresh process artifact",
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
            "ARTIFACT_TEST_SOURCE": str(source),
            "ARTIFACT_TEST_FROM_FILE": "1" if from_file else "0",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 97, completed.stderr
    with RuntimeState.open_sync(database) as reopened:
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
    with RuntimeState.open_sync(database) as store:
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
            request_ref={"input_revision": 1, "request_id": f"{turn_id}:step:1"},
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
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
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


def test_file_artifact_is_streamed_and_independent_of_source(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    import io

    source = tmp_path / "large.bin"
    chunk = b"abcdefgh" * (128 * 1024)
    with source.open("wb") as stream:
        for _ in range(5):
            stream.write(chunk)
    expected_hash = hashlib.sha256(chunk * 5).hexdigest()
    original_fdopen = os.fdopen
    source_inode = source.stat().st_ino
    reads = []

    class BoundedReader(io.BufferedReader):
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024, "artifact source must be read in bounded chunks"
            reads.append(size)
            return super().read(size)

    def open_file(descriptor, *args, **kwargs):
        stream = original_fdopen(descriptor, *args, **kwargs)
        if args == ("rb",) and os.fstat(descriptor).st_ino == source_inode:
            return BoundedReader(stream)
        return stream

    with RuntimeState.open_sync(tmp_path / "run.db") as state:
        turn_id = _turn(state, tmp_path)
        with monkeypatch.context() as patch:
            patch.setattr(os, "fdopen", open_file)
            artifact = state.commit_artifact(
                turn_id=turn_id, content=source, media_type="application/octet-stream", name=source.name,
            )
        assert reads
        assert artifact.blob_sha256 == expected_hash
        assert artifact.size_bytes == 5 * len(chunk)
        source.write_bytes(b"changed")
        assert state.artifact_blob_path(artifact.artifact_id).stat().st_size == artifact.size_bytes
        assert state.verify().valid


def test_gc_reports_partial_filesystem_failure_and_can_be_retried(tmp_path: Path, monkeypatch) -> None:
    with RuntimeState.open_sync(tmp_path / "run.db") as state:
        temporary = state.artifact_root / "tmp"
        temporary.mkdir(parents=True)
        paths = [temporary / name for name in ("a.tmp", "b.tmp", "c.tmp")]
        for path in paths:
            path.write_bytes(b"orphan")
            os.utime(path, (1, 1))
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == paths[1]:
                raise PermissionError("injected deletion failure")
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", unlink)
            report = state.gc_unreferenced_artifacts(min_age_seconds=1)
        assert report.removed == ("tmp/a.tmp", "tmp/c.tmp")
        assert len(report.errors) == 1 and "tmp/b.tmp" in report.errors[0]
        assert not paths[0].exists() and paths[1].exists() and not paths[2].exists()
        retried = state.gc_unreferenced_artifacts(min_age_seconds=1)
        assert retried.removed == ("tmp/b.tmp",)
        assert retried.errors == ()
        assert state.verify().valid


@pytest.mark.parametrize("age", [0, -1, float("nan"), float("inf"), True])
def test_gc_rejects_invalid_grace_period(tmp_path: Path, age) -> None:
    with RuntimeState.open_sync(tmp_path / "run.db") as state:
        with pytest.raises(ValueError, match="finite positive"):
            state.gc_unreferenced_artifacts(min_age_seconds=age)


def _gc_process(database: Path, connection) -> None:
    import sqlite3

    with RuntimeState.open_sync(database) as state:
        state.run_sync(lambda: state.persistence._connection.execute("PRAGMA busy_timeout=50").close())
        connection.send("ready")
        while connection.recv() == "collect":
            try:
                connection.send(state.gc_unreferenced_artifacts(min_age_seconds=1))
            except sqlite3.OperationalError as error:
                connection.send(str(error))
    connection.close()


@pytest.mark.parametrize("from_file", [False, True])
@pytest.mark.parametrize("rollback", [False, True])
def test_process_gc_cannot_delete_a_blob_between_install_and_reference(tmp_path, from_file, rollback):
    import multiprocessing

    database = tmp_path / "run.db"
    source = tmp_path / "source.bin"
    source.write_bytes(b"reused orphan")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_gc_process, args=(database, child))
    with RuntimeState.open_sync(database) as state:
        turn_id = _turn(state, tmp_path)

        def leave_orphan(stage):
            if stage == "after_rename":
                raise InjectedArtifactCrashError("orphan")

        with pytest.raises(InjectedArtifactCrashError):
            state.commit_artifact(
                turn_id=turn_id, content=b"reused orphan", media_type="text/plain", name="orphan",
                fault_injector=leave_orphan,
            )
        [blob] = list((state.artifact_root / "blobs").iterdir())
        os.utime(blob, (1, 1))
        process.start()
        child.close()
        try:
            assert parent.poll(10), "collector did not start"
            assert parent.recv() == "ready"

            def collect_before_reference(stage):
                if stage == "after_rename":
                    parent.send("collect")
                    assert parent.poll(10), "collector did not return its lock outcome"
                    assert "locked" in parent.recv()
                    if rollback:
                        raise InjectedArtifactCrashError("rollback")

            def commit():
                return state.commit_artifact(
                    turn_id=turn_id, content=source if from_file else b"reused orphan",
                    media_type="text/plain", name="retained", fault_injector=collect_before_reference,
                )

            if rollback:
                with pytest.raises(InjectedArtifactCrashError, match="rollback"):
                    commit()
            else:
                artifact = commit()
                assert state.read_artifact(artifact.artifact_id) == b"reused orphan"
            parent.send("collect")
            assert parent.poll(10), "collector did not finish"
            report = parent.recv()
            assert report.errors == ()
            assert report.removed == ((f"blobs/{blob.name}",) if rollback else ())
            assert state.verify().valid
        finally:
            if process.is_alive():
                parent.send("stop")
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(5)
            parent.close()
        assert process.exitcode == 0


def test_low_level_blob_installation_requires_the_gc_write_boundary(tmp_path):
    with RuntimeState.open_sync(tmp_path / "run.db") as state:
        with pytest.raises(RuntimeError, match="write transaction"):
            state.persistence.install_blob(b"unprotected")
        assert not state.artifact_root.exists()


def test_file_artifact_rejects_a_changed_expected_source(tmp_path):
    import hashlib

    source = tmp_path / "source.bin"
    expected = hashlib.sha256(b"original").hexdigest()
    source.write_bytes(b"replaced")
    with RuntimeState.open_sync(tmp_path / "run.db") as state:
        turn_id = _turn(state, tmp_path)
        with pytest.raises(RuntimeError, match="source hash mismatch"):
            state.commit_artifact(
                turn_id=turn_id, content=source, expected_sha256=expected,
                media_type="text/plain", name="source.bin",
            )
        assert state.list_artifacts(turn_id) == ()
        assert not list((state.artifact_root / "blobs").iterdir())
        assert not list((state.artifact_root / "tmp").iterdir())
        assert state.verify().valid


@pytest.mark.parametrize("source_kind", ["directory", "fifo"])
def test_file_artifact_rejects_nonregular_sources_without_committing(tmp_path, source_kind):
    source = tmp_path
    if source_kind == "fifo":
        source = tmp_path / "fifo"
        os.mkfifo(source)
    with RuntimeState.open_sync(tmp_path / "run.db") as state:
        turn_id = _turn(state, tmp_path)
        before = state.list_turn_records(turn_id)
        with pytest.raises(ValueError, match="regular file"):
            state.commit_artifact(
                turn_id=turn_id, content=source, media_type="application/octet-stream", name=source_kind,
            )
        assert state.list_turn_records(turn_id) == before

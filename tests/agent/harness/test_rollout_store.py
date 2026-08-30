from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from agent_runtime.harness import RolloutStore

ROOT = Path(__file__).parents[3]
VERIFY_SCRIPT = ROOT / "scripts" / "verify_agent_rollout.py"


def test_committed_mutation_returns_only_records_after_commit(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(database) as store:
        mutation = store.capture_mutation(lambda: store.create_thread(workspace=workspace))
        [record] = mutation.records
        assert mutation.value.thread_id == record.thread_id
        with sqlite3.connect(database) as independent_reader:
            committed = independent_reader.execute(
                "SELECT COUNT(*) FROM rollout_records WHERE record_id = ?",
                (record.record_id,),
            ).fetchone()[0]

    assert committed == 1


def test_start_turn_persists_records_and_rebuildable_projections(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="inspect this workspace",
            binding_manifest={"model_alias": "test-model", "revision": "model-v1"},
        )

        assert turn.thread_id == thread.thread_id
        assert turn.status == "running"
        assert store.read_thread(thread.thread_id).active_turn_id == turn.turn_id
        [item] = store.list_items(turn.turn_id)
        assert item.kind == "user_message"
        assert item.status == "completed"
        assert item.payload == {"text": "inspect this workspace"}
        records = store.list_records(thread.thread_id)
        assert [record.record_type for record in records] == [
            "thread_created",
            "turn_started",
            "item_started",
            "item_completed",
        ]
        assert [record.thread_sequence for record in records] == [1, 2, 3, 4]
        assert store.verify().valid is True

    with RolloutStore(database) as reopened:
        assert reopened.read_turn(turn.turn_id) == turn
        assert reopened.verify().valid is True


def test_projection_corruption_is_detected_and_rebuilt_from_records(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="preserve canonical history",
            binding_manifest={"model_alias": "test-model"},
        )
        item = store.list_items(turn.turn_id)[0]
        with sqlite3.connect(database) as corruptor:
            corruptor.execute(
                "UPDATE items SET payload_json = ? WHERE item_id = ?",
                ('{"text":"corrupted projection"}', item.item_id),
            )

        report = store.verify()
        assert report.valid is False
        assert "items projection mismatch" in report.errors

        store.rebuild_projections()

        assert store.verify().valid is True
        assert store.list_items(turn.turn_id)[0].payload == {"text": "preserve canonical history"}


def test_projection_metadata_binds_hash_reducer_and_record_position(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rollout.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        store.start_turn(
            thread_id=thread.thread_id,
            user_message="bind projections",
            binding_manifest={"model_alias": "test-model"},
        )
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            metadata = connection.execute(
                "SELECT * FROM projection_meta WHERE thread_id = ?",
                (thread.thread_id,),
            ).fetchone()
            assert metadata is not None
            assert metadata["applied_thread_sequence"] == 4
            assert metadata["applied_record_id"] == 4
            assert metadata["reducer_version"] == 1
            assert len(metadata["canonical_hash"]) == 64
            connection.execute(
                "UPDATE projection_meta SET canonical_hash = ? WHERE thread_id = ?",
                ("0" * 64, thread.thread_id),
            )

        report = store.verify()
        assert report.valid is False
        assert "projection metadata mismatch" in report.errors

        store.rebuild_projections()

        assert store.verify().valid is True


def test_two_fresh_processes_rebuild_identical_prefix_to_identical_hash(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source.sqlite3"
    with RolloutStore(source) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="deterministic prefix",
            binding_manifest={"model_alias": "test-model"},
        )
        store.complete_turn(turn_id=turn.turn_id, answer="stable answer")
    first = tmp_path / "first.sqlite3"
    second = tmp_path / "second.sqlite3"
    shutil.copy2(source, first)
    shutil.copy2(source, second)

    def rebuild(database: Path) -> dict[str, object]:
        completed = subprocess.run(
            [
                sys.executable,
                str(VERIFY_SCRIPT),
                "--database",
                str(database),
                "--rebuild",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    first_result = rebuild(first)
    second_result = rebuild(second)

    assert first_result == second_result
    assert first_result["valid"] is True
    assert first_result["record_count"] == 7
    assert first_result["projection_hashes"] == {thread.thread_id: first_result["projection_hashes"][thread.thread_id]}
    assert len(first_result["projection_hashes"][thread.thread_id]) == 64
    assert len(first_result["record_chain_sha256"]) == 64


def test_rebuild_rejects_an_unregistered_payload_schema_version(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        with sqlite3.connect(database) as corruptor:
            corruptor.execute(
                "UPDATE rollout_records SET payload_schema_version = 99 WHERE thread_id = ?",
                (thread.thread_id,),
            )

        with pytest.raises(
            ValueError,
            match="unsupported RolloutRecord payload schema version",
        ):
            store.rebuild_projections()


def test_complete_turn_commits_answer_before_releasing_thread(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        first = store.start_turn(
            thread_id=thread.thread_id,
            user_message="hello",
            binding_manifest={"model_alias": "test-model"},
        )

        completed = store.complete_turn(turn_id=first.turn_id, answer="world")

        assert completed.status == "completed"
        assert store.read_thread(thread.thread_id).active_turn_id is None
        assert [item.kind for item in store.list_items(first.turn_id)] == [
            "user_message",
            "agent_message",
        ]
        assert store.list_items(first.turn_id)[1].payload == {"text": "world"}
        assert store.list_records(thread.thread_id)[-1].record_type == "turn_completed"
        followup = store.start_turn(
            thread_id=thread.thread_id,
            user_message="again",
            binding_manifest={"model_alias": "test-model-v2"},
        )
        assert followup.turn_id != first.turn_id


def test_two_connections_cannot_start_two_active_turns(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(database) as store:
        thread_id = store.create_thread(workspace=workspace).thread_id
    barrier = Barrier(2)

    def compete(message: str) -> str:
        with RolloutStore(database) as store:
            barrier.wait()
            try:
                return store.start_turn(
                    thread_id=thread_id,
                    user_message=message,
                    binding_manifest={"model_alias": "test-model"},
                ).turn_id
            except RuntimeError:
                return "busy"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(compete, ("first", "second")))

    assert results.count("busy") == 1
    assert len([result for result in results if result != "busy"]) == 1
    with RolloutStore(database) as store:
        active_turn_id = store.read_thread(thread_id).active_turn_id
        assert active_turn_id in results
        assert store.verify().valid is True

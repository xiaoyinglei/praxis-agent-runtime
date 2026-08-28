from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutStore

ROOT = Path(__file__).parents[3]
RECOVERY_SCRIPT = ROOT / "scripts" / "recover_agent_turn.py"


def test_fresh_process_interrupts_a_pre_dispatch_orphan_only_after_maintenance(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as crashed_process:
        thread = crashed_process.create_thread(workspace=workspace)
        turn = crashed_process.start_turn(
            thread_id=thread.thread_id,
            user_message="crash after Turn creation",
            binding_manifest={"model_alias": "model-v1"},
        )

    with RolloutStore(database) as recovery_process:
        with pytest.raises(RuntimeError, match="maintenance confirmation"):
            recovery_process.interrupt_orphaned_turn(
                turn_id=turn.turn_id,
                reason="operator observed the old worker is stopped",
                maintenance_confirmed=False,
            )
        assert recovery_process.read_turn(turn.turn_id).status == "running"

        interrupted = recovery_process.interrupt_orphaned_turn(
            turn_id=turn.turn_id,
            reason="operator observed the old worker is stopped",
            maintenance_confirmed=True,
        )

        assert interrupted.status == "interrupted"
        assert recovery_process.read_thread(thread.thread_id).active_turn_id == turn.turn_id
        record = recovery_process.list_records(thread_id=thread.thread_id)[-1]
        assert record.record_type == "turn_interrupted"
        assert record.payload["maintenance_confirmed"] is True
        assert record.payload["recovery_scope"] == "pre_dispatch_orphan"
        assert recovery_process.verify().valid is True


def test_orphan_recovery_refuses_any_turn_with_a_durable_operation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="operation already prepared",
            binding_manifest={"model_alias": "model-v1"},
        )
        store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="r" * 64,
            context_hash="c" * 64,
            tool_hash="t" * 64,
            wire_hash="w" * 64,
            request_ref={"request_id": "request-1"},
        )

    with RolloutStore(database) as recovery_process:
        with pytest.raises(RuntimeError, match="durable operations"):
            recovery_process.interrupt_orphaned_turn(
                turn_id=turn.turn_id,
                reason="worker stopped",
                maintenance_confirmed=True,
            )
        assert recovery_process.read_turn(turn.turn_id).status == "running"


def test_recovery_cli_records_an_interrupted_turn_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as crashed_process:
        thread = crashed_process.create_thread(workspace=workspace)
        turn = crashed_process.start_turn(
            thread_id=thread.thread_id,
            user_message="crash before provider I/O",
            binding_manifest={"model_alias": "model-v1"},
        )

    completed = subprocess.run(
        [
            sys.executable,
            str(RECOVERY_SCRIPT),
            "--database",
            str(database),
            "--turn-id",
            turn.turn_id,
            "--reason",
            "old process was terminated",
            "--maintenance-confirmed",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "status": "interrupted",
        "thread_id": thread.thread_id,
        "turn_id": turn.turn_id,
    }
    with RolloutStore(database) as verified:
        assert verified.read_turn(turn.turn_id).status == "interrupted"
        assert verified.verify().valid is True

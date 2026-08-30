from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_runtime.harness import (
    RolloutEventReader,
    RolloutStore,
    ToolOrchestrator,
    ToolReconciliationOutcome,
)
from agent_runtime.streaming.events import EventType, ItemStatus, TurnItemKind
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolResult

ROOT = Path(__file__).parents[3]
RECOVERY_SCRIPT = ROOT / "scripts" / "recover_agent_turn.py"


def test_reconciliation_item_parents_unknown_attempt_without_rerun(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_calls = 1
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="reconcile an uncertain write",
            binding_manifest={"model_alias": "test-model"},
        )
        values = {
            "turn_id": turn.turn_id,
            "operation_id": "operation-unknown",
            "tool_call_id": "call-unknown",
            "tool_name": "write_file",
            "arguments_digest": "args-v1",
            "execution_revision": "write-file-v1",
            "idempotent": False,
            "attempt_count": 0,
            "error_code": None,
            "requires_reconciliation": False,
        }
        store.record_tool_execution_state(status="prepared", **values)
        store.record_tool_execution_state(status="ready", **values)
        claim = store.claim_tool_operation(
            operation_id="operation-unknown",
            worker_id="worker-a",
            lease_seconds=30.0,
        )
        assert store.commit_tool_operation_result(
            turn_id=turn.turn_id,
            operation_id="operation-unknown",
            claim_generation=claim.claim_generation,
            fencing_token=str(claim.fencing_token),
            status="unknown",
            attempt_count=1,
            error_code="cancelled_outcome_unknown",
            requires_reconciliation=True,
            result={
                "tool_call_id": "call-unknown",
                "tool_name": "write_file",
                "is_error": True,
                "error_code": "cancelled_outcome_unknown",
                "content": [],
                "attachments": [],
            },
        )
        original_completed = next(
            replay.event
            for replay in RolloutEventReader(store).read(thread.thread_id)
            if replay.event.type is EventType.ITEM_COMPLETED
            and replay.event.item_kind is TurnItemKind.TOOL
        )
        assert original_completed.status is ItemStatus.OUTCOME_UNKNOWN

        orchestrator = ToolOrchestrator(
            store=store,
            tools={},
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
            ),
        )
        orchestrator.reconcile(
            turn_id=turn.turn_id,
            outcome=ToolReconciliationOutcome(
                status="succeeded",
                result=ToolResult(
                    tool_call_id="call-unknown",
                    tool_name="write_file",
                    structured_content={"verified": True},
                    metadata={"reconciler_revision": "operator-v1"},
                ),
            ),
        )
        public = [
            replay.event
            for replay in RolloutEventReader(store).read(thread.thread_id)
        ]

        original_completions = [
            event
            for event in public
            if event.type is EventType.ITEM_COMPLETED
            and event.item_id == original_completed.item_id
        ]
        reconciliation = [
            event
            for event in public
            if event.item_kind is TurnItemKind.RECONCILIATION
        ]
        assert runner_calls == 1
        assert len(original_completions) == 1
        assert original_completions[0].status is ItemStatus.OUTCOME_UNKNOWN
        assert [event.type for event in reconciliation] == [
            EventType.ITEM_STARTED,
            EventType.ITEM_COMPLETED,
        ]
        assert all(
            event.parent_item_id == original_completed.item_id
            for event in reconciliation
        )
        assert reconciliation[0].item_id != original_completed.item_id
        assert store.verify().valid is True


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

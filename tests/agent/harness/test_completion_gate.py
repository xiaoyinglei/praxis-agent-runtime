from __future__ import annotations

from agent_runtime.harness import CompletionProposal, RuntimeStore, completion_ops, tool_ops, turn_ops
from agent_runtime.harness.stop import StopChecks


def _commit_tool_result(
    store: RuntimeStore,
    *,
    turn_id: str,
    call_id: str,
    tool_name: str,
    arguments: dict[str, object],
    result: dict[str, object],
    resources: tuple[dict[str, object], ...] = (),
    operation_failed: bool = False,
) -> None:
    tool_ops.record_tool_call(
        store,
        turn_id=turn_id,
        tool_call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
        origin={
            "request_id": f"request-{call_id}",
            "toolset_revision": "toolset-v1",
            "exposed_tool_names": [tool_name],
        },
    )
    operation_id = f"operation-{call_id}"
    values = {
        "turn_id": turn_id,
        "operation_id": operation_id,
        "tool_call_id": call_id,
        "tool_name": tool_name,
        "arguments_digest": f"digest-{call_id}",
        "execution_revision": f"{tool_name}-v1",
        "idempotent": True,
        "attempt_count": 0,
        "error_code": None,
        "requires_reconciliation": False,
        "effects": (
            ("write_workspace",)
            if any(resource.get("access") == "write" for resource in resources)
            else ("read_workspace",)
        ),
        "resources": resources,
    }
    tool_ops.record_tool_execution_state(store, status="prepared", **values)
    tool_ops.record_tool_execution_state(store, status="ready", **values)
    claim = tool_ops.claim_tool_operation(
        store,
        operation_id=operation_id,
        worker_id="verifier-fixture",
        lease_seconds=30.0,
    )
    assert tool_ops.commit_tool_operation_outcome(
        store,
        operation_id=operation_id,
        claim_generation=claim.claim_generation,
        fencing_token=claim.fencing_token,
        status="failed" if operation_failed else "succeeded",
        attempt_count=1,
        error_code="command_failed" if operation_failed else None,
        requires_reconciliation=False,
    )
    tool_ops.record_tool_result(
        store,
        turn_id=turn_id,
        operation_id=operation_id,
        result={
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "structured_content": result,
            "is_error": operation_failed,
            "error_code": "command_failed" if operation_failed else None,
            "error_message": None,
            "retryable": False,
            "truncated": False,
            "metadata": result.get("metadata", {}),
            "content": [],
            "attachments": [],
            "model_content": str(result),
        },
    )


def _proposal(store: RuntimeStore, turn_id: str) -> CompletionProposal:
    item = completion_ops.record_final_proposal(store, turn_id=turn_id, answer="done")
    turn = store.read_turn(turn_id)
    return CompletionProposal(
        thread_id=turn.thread_id,
        turn_id=turn_id,
        item_id=item.item_id,
        answer="done",
    )


def test_live_command_must_be_observed_terminal_before_completion(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "run.db") as store:
        turn = turn_ops.start_turn(
            store,
            thread_id=turn_ops.create_thread(store, workspace=tmp_path).thread_id,
            user_message="run command",
            binding_manifest={"completion_policy": {"revision": "completion-v2", "acceptance_command": "sleep 1"}},
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="launch",
            tool_name="run_command",
            arguments={"command": "sleep 1", "yield_time_seconds": 0.01, "timeout_seconds": None},
            result={"status": "running", "session_id": "cmd_live", "exit_code": None},
        )
        gate = StopChecks(store)
        pending = gate.evaluate(_proposal(store, turn.turn_id))
        assert pending.action == "continue" and "cmd_live" in pending.reason
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="poll-wrong",
            tool_name="manage_command",
            arguments={"session_id": "cmd_other"},
            result={"status": "completed", "session_id": "cmd_other", "exit_code": 0},
        )
        assert gate.evaluate(_proposal(store, turn.turn_id)).action == "continue"
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="poll",
            tool_name="manage_command",
            arguments={"session_id": "cmd_live"},
            result={"status": "completed", "session_id": "cmd_live", "exit_code": 0},
        )
        assert gate.evaluate(_proposal(store, turn.turn_id)).action == "accept"


def test_acceptance_command_uses_linked_terminal_session_result(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "run.db") as store:
        turn = turn_ops.start_turn(
            store,
            thread_id=turn_ops.create_thread(store, workspace=tmp_path).thread_id,
            user_message="verify command",
            binding_manifest={
                "completion_policy": {
                    "revision": "completion-v2",
                    "require_workspace_change": False,
                    "acceptance_command": "make check",
                }
            },
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="launch",
            tool_name="run_command",
            arguments={"command": "make check", "working_dir": ".", "yield_time_seconds": 0.01},
            result={"status": "running", "session_id": "cmd_live", "exit_code": None},
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="poll",
            tool_name="manage_command",
            arguments={"session_id": "cmd_live", "action": "wait"},
            result={"status": "completed", "session_id": "cmd_live", "exit_code": 0},
        )
        decision = StopChecks(store).evaluate(_proposal(store, turn.turn_id))
        assert decision.action == "accept"

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, local

import pytest

from agent_runtime.budget import ResourceUsage
from agent_runtime.harness import RuntimeStore, budget_ops, model_ops, tool_ops, turn_ops
from agent_runtime.harness.operation_values import TurnCancellationPendingError
from agent_runtime.harness.protocol import HarnessModelResponse
from agent_runtime.harness.runtime_tx import RuntimeTx


def _family(store, workspace):
    root = turn_ops.start_turn(
        store, thread_id=turn_ops.create_thread(store, workspace=workspace).thread_id,
        user_message="root", binding_manifest={"model_token_budget_total": 1000},
    )
    child = _child(store, root.turn_id, "child", 500)
    grandchild = _child(store, child.turn_id, "grandchild", 100)
    return root, child, grandchild


def _child(store, parent_id, name, tokens=20):
    return budget_ops.start_budgeted_child_turn(
        store, parent_turn_id=parent_id, child_thread_id=f"thread-{name}", child_turn_id=f"turn-{name}",
        user_message=name, binding_manifest={}, requested_tokens=tokens, requested_cost_micros=None,
    )


def _prepared(store, turn_id):
    model = model_ops.prepare_model_operation(
        store, turn_id=turn_id, request_hash="request", context_hash="context", tool_hash="tools",
        wire_hash="wire", request_ref={"request_id": "request-1", "input_revision": 1, "purpose": "agent_step"},
    )
    arguments = dict(
        turn_id=turn_id, operation_id="tool-operation", tool_call_id="call-1", tool_name="write_file",
        arguments_digest="arguments", execution_revision="test-v1", idempotent=False,
        attempt_count=0, error_code=None, requires_reconciliation=False,
    )
    tool_ops.record_tool_execution_state(store, **arguments, status="prepared")
    tool_ops.record_tool_execution_state(store, **arguments, status="ready")
    return model.operation_id


def _admit(store, kind, turn_id, model_id):
    if kind == "model":
        return model_ops.dispatch_model_attempt(
            store, model_id, resource_request=ResourceUsage(input_tokens=2, output_tokens=3, model_calls=1),
        )
    if kind == "tool":
        return tool_ops.claim_tool_operation(
            store, operation_id="tool-operation", worker_id="worker", lease_seconds=30,
        )
    return _child(store, turn_id, "new-child")


@pytest.mark.parametrize("kind", ["model", "tool", "child"])
@pytest.mark.parametrize("ancestor_index", [0, 1, 2])
def test_cancelled_ancestor_rejects_admission_from_another_connection(tmp_path, kind, ancestor_index):
    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as store:
        family = _family(store, tmp_path)
        model_id = _prepared(store, family[-1].turn_id)
        before = store.read_budget_state(family[-1].turn_id)
    with RuntimeStore.open_sync(database) as cancelling:
        turn_ops.request_turn_cancellation(
            cancelling, turn_id=family[ancestor_index].turn_id, reason="stop this subtree",
        )
    with RuntimeStore.open_sync(database) as resumed:
        with pytest.raises(TurnCancellationPendingError):
            _admit(resumed, kind, family[-1].turn_id, model_id)
        assert resumed.list_model_operations(family[-1].turn_id)[0].status == "prepared"
        assert resumed.read_tool_operation("tool-operation").status == "ready"
        assert resumed.read_budget_state(family[-1].turn_id) == before
        assert len(resumed.list_turns()) == 3
        assert resumed.verify().valid


@pytest.mark.parametrize("kind", ["model", "tool", "child"])
def test_cancel_between_preflight_and_write_transaction_rejects_admission(tmp_path, kind, monkeypatch):
    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as store:
        root, _, descendant = _family(store, tmp_path)
        model_id = _prepared(store, descendant.turn_id)
    entering_transaction = Event()
    cancelled = Event()
    owner_state = local()
    transaction = RuntimeTx.transaction

    def gated_transaction(tx):
        if getattr(owner_state, "gate_next_transaction", False):
            owner_state.gate_next_transaction = False
            entering_transaction.set()
            assert cancelled.wait(5)
        return transaction(tx)

    monkeypatch.setattr(RuntimeTx, "transaction", gated_transaction)
    with RuntimeStore.open_sync(database) as admission, RuntimeStore.open_sync(database) as cancelling:
        def admit():
            def on_owner():
                owner_state.gate_next_transaction = True
                with pytest.raises(TurnCancellationPendingError):
                    _admit(admission, kind, descendant.turn_id, model_id)
            admission.run_sync(on_owner)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(admit)
            try:
                assert entering_transaction.wait(5)
                # All pre-transaction reads have already happened. A check
                # outside the write transaction would miss this cancellation.
                turn_ops.request_turn_cancellation(cancelling, turn_id=root.turn_id, reason="cancel at boundary")
            finally:
                cancelled.set()
            pending.result(timeout=5)
        assert admission.list_model_operations(descendant.turn_id)[0].status == "prepared"
        assert admission.read_tool_operation("tool-operation").status == "ready"
        assert len(admission.list_turns()) == 3


@pytest.mark.parametrize("kind", ["model", "tool", "child"])
def test_paused_ancestor_does_not_revoke_existing_child_authority(tmp_path, kind):
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        root, _, descendant = _family(store, tmp_path)
        model_id = _prepared(store, descendant.turn_id)
        turn_ops.pause_turn(store, turn_id=root.turn_id, reason="waiting for user")
        assert _admit(store, kind, descendant.turn_id, model_id).status in {"dispatched", "running"}


@pytest.mark.parametrize("kind", ["model", "tool", "child"])
def test_terminal_ancestor_rejects_new_descendant_work(tmp_path, kind):
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        root, _, descendant = _family(store, tmp_path)
        model_id = _prepared(store, descendant.turn_id)
        turn_ops.fail_turn(store, turn_id=root.turn_id, reason_code="parent_failed", message="failed parent")
        with pytest.raises(TurnCancellationPendingError):
            _admit(store, kind, descendant.turn_id, model_id)


def test_binding_cannot_forge_cancellation_lineage(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        root, _, _ = _family(store, tmp_path)
        independent = turn_ops.start_turn(
            store, thread_id=turn_ops.create_thread(store, workspace=tmp_path).thread_id,
            user_message="independent", binding_manifest={"budget_parent_turn_id": root.turn_id},
        )
        model_id = _prepared(store, independent.turn_id)
        turn_ops.request_turn_cancellation(store, turn_id=root.turn_id, reason="only owned descendants")
        assert _admit(store, "model", independent.turn_id, model_id).status == "dispatched"


@pytest.mark.parametrize("kind", ["model", "tool"])
def test_admission_before_cancel_preserves_in_flight_exposure(tmp_path, kind):
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        root, _, descendant = _family(store, tmp_path)
        model_id = _prepared(store, descendant.turn_id)
        admitted = _admit(store, kind, descendant.turn_id, model_id)
        before = store.read_budget_state(descendant.turn_id)
        turn_ops.request_turn_cancellation(store, turn_id=root.turn_id, reason="cancel after dispatch")
        if kind == "model":
            assert store.list_model_attempts(model_id)[-1].attempt_id == admitted.attempt_id
            assert store.list_model_operations(descendant.turn_id)[0].status == "dispatched"
        else:
            assert store.read_tool_operation("tool-operation").fencing_token == admitted.fencing_token
            assert admitted.status == "running"
        assert store.read_budget_state(descendant.turn_id) == before


def test_cancelled_ancestor_does_not_block_result_commit_or_budget_settlement(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        root, parent, child = _family(store, tmp_path)
        model_id = _prepared(store, child.turn_id)
        attempt = _admit(store, "model", child.turn_id, model_id)
        tool = _admit(store, "tool", child.turn_id, model_id)
        turn_ops.request_turn_cancellation(store, turn_id=root.turn_id, reason="cancel after I/O")
        assert model_ops.complete_model_attempt(
            store, operation_id=model_id, attempt_id=attempt.attempt_id, generation=attempt.generation,
            response=HarnessModelResponse.from_parts(
                text="completed before cancel arrived", provider_response_id="response",
                usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            ),
        )
        assert tool_ops.commit_tool_operation_outcome(
            store, operation_id=tool.operation_id, claim_generation=tool.claim_generation,
            fencing_token=tool.fencing_token, status="succeeded", attempt_count=1,
            error_code=None, requires_reconciliation=False,
        )
        result = tool_ops.record_tool_result(
            store, turn_id=child.turn_id, operation_id=tool.operation_id,
            result={"tool_call_id": "call-1", "tool_name": "write_file", "content": "written"},
        )
        assert result.status == "completed"
        for turn in (child, parent):
            turn_ops.pause_turn(store, turn_id=turn.turn_id, reason="safe cancellation after result")
            turn_ops.cancel_turn(store, turn_id=turn.turn_id)
            budget_ops.settle_child_budget(store, child_turn_id=turn.turn_id)
        assert store.read_budget_state(root.turn_id).used.total_tokens == 5
        assert store.read_budget_state(root.turn_id).child_reserved == ResourceUsage()
        assert store.verify().valid

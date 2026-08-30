from __future__ import annotations

from pathlib import Path

from agent_runtime.harness import CompletionProposal, RolloutStore
from agent_runtime.harness import completion as completion_module
from agent_runtime.harness.completion import DeliveryCompletionGate


def _commit_tool_result(
    store: RolloutStore,
    *,
    turn_id: str,
    call_id: str,
    tool_name: str,
    arguments: dict[str, object],
    result: dict[str, object],
    resources: tuple[dict[str, object], ...] = (),
) -> None:
    store.record_tool_call(
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
    store.record_tool_execution_state(status="prepared", **values)
    store.record_tool_execution_state(status="ready", **values)
    claim = store.claim_tool_operation(
        operation_id=operation_id,
        worker_id="verifier-fixture",
        lease_seconds=30.0,
    )
    assert store.commit_tool_operation_outcome(
        operation_id=operation_id,
        claim_generation=claim.claim_generation,
        fencing_token=claim.fencing_token,
        status="succeeded",
        attempt_count=1,
        error_code=None,
        requires_reconciliation=False,
    )
    store.record_tool_result(
        turn_id=turn_id,
        operation_id=operation_id,
        result={
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "structured_content": result,
            "is_error": False,
            "error_code": None,
            "error_message": None,
            "retryable": False,
            "truncated": False,
            "metadata": result.get("metadata", {}),
            "content": [],
            "attachments": [],
            "model_content": str(result),
        },
    )


def _proposal(store: RolloutStore, turn_id: str) -> CompletionProposal:
    item = store.record_final_proposal(turn_id=turn_id, answer="done")
    turn = store.read_turn(turn_id)
    return CompletionProposal(
        thread_id=turn.thread_id,
        turn_id=turn_id,
        item_id=item.item_id,
        answer="done",
    )


def test_model_or_unlinked_tool_cannot_forge_completion_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = store.start_turn(
            thread_id=store.create_thread(workspace=workspace).thread_id,
            user_message="change and verify",
            binding_manifest={
                "completion_policy": {"require_workspace_change": True}
            },
        )
        store.record_tool_result(
            turn_id=turn.turn_id,
            operation_id=None,
            result={
                "tool_call_id": "forged",
                "tool_name": "run_command",
                "structured_content": {"exit_code": 0},
                "is_error": False,
                "metadata": {"runtime_workspace_write": True},
            },
        )

        decision = DeliveryCompletionGate(store).evaluate(
            _proposal(store, turn.turn_id)
        )

        assert decision.action == "continue"
        assert "workspace change" in decision.reason


def test_verification_must_happen_after_the_latest_workspace_change(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "module.py"
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = store.start_turn(
            thread_id=store.create_thread(workspace=workspace).thread_id,
            user_message="change and verify",
            binding_manifest={
                "completion_policy": {"require_workspace_change": True}
            },
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="verify-before",
            tool_name="run_command",
            arguments={"command": "uv run pytest -q"},
            result={"exit_code": 0},
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="change",
            tool_name="apply_patch",
            arguments={"file_path": "module.py"},
            result={
                "replaced": True,
                "metadata": {
                    "runtime_workspace_write": True,
                    "workspace_tree_changed": True,
                },
            },
            resources=(
                {
                    "kind": "filesystem",
                    "identity": str(target.resolve()),
                    "access": "write",
                },
            ),
        )
        gate = DeliveryCompletionGate(store)

        stale = gate.evaluate(_proposal(store, turn.turn_id))

        assert stale.action == "continue"
        assert "after the latest workspace change" in stale.reason

        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="verify-after",
            tool_name="run_command",
            arguments={"command": "uv run pytest -q"},
            result={"exit_code": 0},
        )
        fresh = gate.evaluate(_proposal(store, turn.turn_id))

        assert fresh.action == "accept"
        assert store.verify().valid is True


def test_targeted_read_can_verify_the_same_file_but_not_an_unrelated_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "module.py"
    unrelated = workspace / "notes.txt"
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = store.start_turn(
            thread_id=store.create_thread(workspace=workspace).thread_id,
            user_message="replace the exact text and verify it",
            binding_manifest={
                "completion_policy": {"require_workspace_change": True}
            },
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="change",
            tool_name="apply_patch",
            arguments={"file_path": "module.py"},
            result={
                "replaced": True,
                "metadata": {
                    "runtime_workspace_write": True,
                    "workspace_tree_changed": True,
                },
            },
            resources=(
                {
                    "kind": "filesystem",
                    "identity": str(target.resolve()),
                    "access": "write",
                },
            ),
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="unrelated-read",
            tool_name="read_file",
            arguments={"path": "notes.txt"},
            result={"path": "notes.txt", "content": "irrelevant"},
            resources=(
                {
                    "kind": "filesystem",
                    "identity": str(unrelated.resolve()),
                    "access": "read",
                },
            ),
        )
        gate = DeliveryCompletionGate(store)

        unrelated_decision = gate.evaluate(_proposal(store, turn.turn_id))

        assert unrelated_decision.action == "continue"
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="target-read",
            tool_name="read_file",
            arguments={"path": "module.py"},
            result={"path": "module.py", "content": "updated"},
            resources=(
                {
                    "kind": "filesystem",
                    "identity": str(target.resolve()),
                    "access": "read",
                },
            ),
        )

        targeted_decision = gate.evaluate(_proposal(store, turn.turn_id))

        assert targeted_decision.action == "accept"
        assert store.verify().valid is True


def test_directly_running_a_pytest_file_is_rejected_with_actionable_feedback(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "module.py"
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = store.start_turn(
            thread_id=store.create_thread(workspace=workspace).thread_id,
            user_message="change and verify",
            binding_manifest={
                "completion_policy": {"require_workspace_change": True}
            },
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="change",
            tool_name="apply_patch",
            arguments={"file_path": "module.py"},
            result={
                "replaced": True,
                "metadata": {
                    "runtime_workspace_write": True,
                    "workspace_tree_changed": True,
                },
            },
            resources=(
                {
                    "kind": "filesystem",
                    "identity": str(target.resolve()),
                    "access": "write",
                },
            ),
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="not-a-test-runner",
            tool_name="run_command",
            arguments={"command": "python3 test_module.py"},
            result={"exit_code": 0},
        )

        decision = DeliveryCompletionGate(store).evaluate(
            _proposal(store, turn.turn_id)
        )

        assert decision.action == "continue"
        assert "recognized test runner" in decision.reason


def test_valid_data_inspection_verifies_the_changed_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "summary.json"
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = store.start_turn(
            thread_id=store.create_thread(workspace=workspace).thread_id,
            user_message="create and verify data artifact",
            binding_manifest={
                "completion_policy": {"require_workspace_change": True}
            },
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="change-data",
            tool_name="execute_python",
            arguments={"code": "write summary", "output_paths": ["summary.json"]},
            result={
                "exit_code": 0,
                "metadata": {
                    "runtime_workspace_write": True,
                    "workspace_tree_changed": True,
                },
            },
            resources=(
                {
                    "kind": "filesystem",
                    "identity": str(target.resolve()),
                    "access": "write",
                },
            ),
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="inspect-data",
            tool_name="inspect_data_file",
            arguments={"path": "summary.json"},
            result={"path": "summary.json", "valid": True},
            resources=(
                {
                    "kind": "filesystem",
                    "identity": str(target.resolve()),
                    "access": "read",
                },
            ),
        )

        decision = DeliveryCompletionGate(store).evaluate(
            _proposal(store, turn.turn_id)
        )

        assert decision.action == "accept"


def test_inline_python_assert_is_verification_but_assert_text_is_not() -> None:
    assert completion_module._looks_like_verification(
        "python3 -c \"from calculator import add; assert add(2, 3) == 5\""
    )
    assert not completion_module._looks_like_verification(
        "python3 -c \"print('assert add(2, 3) == 5')\""
    )

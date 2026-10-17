from __future__ import annotations

import json
from uuid import uuid4

import pytest

from agent_runtime.harness import RolloutStore
from agent_runtime.harness.completion import DeliveryCompletionGate
from tests.agent.harness.test_completion_gate import _commit_tool_result, _proposal

COMMAND = "python tools/custom_acceptance.py --strict"


def _start(store, workspace, *, require_change=False, command=None, revision="completion-v2"):
    policy = {"require_workspace_change": require_change, "acceptance_command": command}
    if revision is not None:
        policy["revision"] = revision
    return store.start_turn(
        thread_id=store.create_thread(workspace=workspace).thread_id,
        user_message="complete the requested work",
        binding_manifest={"completion_policy": policy},
    )


def _change(store, turn_id, workspace, call_id="change"):
    _commit_tool_result(
        store,
        turn_id=turn_id,
        call_id=call_id,
        tool_name="apply_patch",
        arguments={"file_path": "module.py"},
        result={"metadata": {"runtime_workspace_write": True, "workspace_tree_changed": True}},
        resources=({"kind": "filesystem", "identity": str(workspace / "module.py"), "access": "write"},),
    )


def _command(store, turn_id, *, call_id="check", command=COMMAND, result=None, working_dir=None, failed=False,
             resources=()):
    arguments = {"command": command}
    if working_dir is not None:
        arguments["working_dir"] = working_dir
    _commit_tool_result(
        store,
        turn_id=turn_id,
        call_id=call_id,
        tool_name="run_command",
        arguments=arguments,
        result={"exit_code": 0} if result is None else result,
        operation_failed=failed,
        resources=resources,
    )


def _evaluate(store, turn_id):
    return DeliveryCompletionGate(store).evaluate(_proposal(store, turn_id))


def test_natural_v2_completion_has_no_inferred_validation(tmp_path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path)
        decision = _evaluate(store, turn.turn_id)
        assert decision.action == "accept"
        assert decision.validation_status == "not_evaluated"
        assert decision.validation_scope is None


def test_v2_requested_change_checks_only_actual_mutation(tmp_path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path, require_change=True)
        assert _evaluate(store, turn.turn_id).action == "continue"
        _change(store, turn.turn_id, tmp_path)
        decision = _evaluate(store, turn.turn_id)
        assert decision.action == "accept"
        assert decision.validation_status == "passed"
        assert decision.validation_scope == "workspace_change_recorded"


def test_legacy_policy_still_requires_post_change_verification(tmp_path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path, require_change=True, command=COMMAND, revision=None)
        _change(store, turn.turn_id, tmp_path)
        _command(store, turn.turn_id)
        decision = _evaluate(store, turn.turn_id)
        assert decision.action == "continue"
        assert decision.validation_scope == "workspace_change"


def test_explicit_custom_command_passes_without_a_runner_name_whitelist(tmp_path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path, command=f"  {COMMAND}  ")
        _command(store, turn.turn_id, command=f"\n{COMMAND}\n")
        decision = _evaluate(store, turn.turn_id)
        assert decision.action == "accept"
        assert decision.validation_status == "passed"
        assert decision.validation_scope == "acceptance_command"


@pytest.mark.parametrize("actual", ["pytest -q", COMMAND + " --extra", "python  tools/custom_acceptance.py --strict"])
def test_other_successful_commands_do_not_replace_the_frozen_acceptance_command(tmp_path, actual):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path, command=COMMAND)
        _command(store, turn.turn_id, command=actual)
        decision = _evaluate(store, turn.turn_id)
        assert decision.action == "continue"
        assert decision.validation_scope == "acceptance_command"
        feedback = json.loads(decision.reason)
        assert feedback["reason_code"] == "acceptance_command_missing"
        assert feedback["command"] == COMMAND


@pytest.mark.parametrize("working_dir", [".", "nested/..", "absolute", "nested", "outside", False])
def test_acceptance_is_bound_to_the_thread_workspace(tmp_path, working_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "nested").mkdir()
    cwd = str(workspace) if working_dir == "absolute" else str(tmp_path) if working_dir == "outside" else working_dir
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, workspace, command=COMMAND)
        _command(store, turn.turn_id, working_dir=cwd)
        decision = _evaluate(store, turn.turn_id)
        assert decision.action == ("accept" if working_dir in {".", "nested/..", "absolute"} else "continue")


@pytest.mark.parametrize(
    "result, failed",
    [
        ({"exit_code": 1}, False),
        ({"exit_code": True}, False),
        ({"exit_code": 0.0}, False),
        ({}, False),
        ({"exit_code": 0, "timed_out": True}, False),
        ({"exit_code": 0, "sandbox_error": "blocked"}, False),
        ({"exit_code": 0}, True),
    ],
)
def test_latest_failed_or_invalid_check_cannot_reuse_an_earlier_pass(tmp_path, result, failed):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path, command=COMMAND)
        _command(store, turn.turn_id, call_id="earlier-pass")
        _command(store, turn.turn_id, call_id="latest", result=result, failed=failed)
        decision = _evaluate(store, turn.turn_id)
        assert decision.action == "continue"
        feedback = json.loads(decision.reason)
        assert feedback["reason_code"] == "acceptance_command_failed"
        assert feedback["command"] == COMMAND
        assert feedback["tool_call_id"] == "latest"


def test_a_later_workspace_change_requires_rerunning_the_frozen_command(tmp_path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path, command=COMMAND)
        _command(store, turn.turn_id)
        _change(store, turn.turn_id, tmp_path)
        decision = _evaluate(store, turn.turn_id)
        assert decision.action == "continue"
        assert json.loads(decision.reason)["reason_code"] == "acceptance_command_stale"
        _command(store, turn.turn_id, call_id="rerun")
        assert _evaluate(store, turn.turn_id).action == "accept"


def test_trusted_acceptance_command_may_itself_produce_workspace_outputs(tmp_path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path, require_change=True, command=COMMAND)
        _command(
            store,
            turn.turn_id,
            result={"exit_code": 0, "metadata": {"runtime_workspace_write": True, "workspace_tree_changed": True}},
            resources=({"kind": "filesystem", "identity": str(tmp_path / "build.txt"), "access": "write"},),
        )
        decision = _evaluate(store, turn.turn_id)
        assert decision.action == "accept"
        assert decision.validation_scope == "acceptance_command"


def test_unlinked_result_cannot_attest_the_user_command(tmp_path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path, command=COMMAND)
        store.record_tool_result(
            turn_id=turn.turn_id,
            operation_id=None,
            result={"tool_call_id": "fake", "tool_name": "run_command", "is_error": False,
                    "structured_content": {"exit_code": 0, "command": COMMAND}},
        )
        assert _evaluate(store, turn.turn_id).action == "continue"


def test_mismatched_tool_call_cannot_attest_the_user_command(tmp_path, monkeypatch):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path, command=COMMAND)
        original = store.record_tool_call
        monkeypatch.setattr(
            store, "record_tool_call", lambda **kwargs: original(**{**kwargs, "tool_name": "read_file"}),
        )
        _command(store, turn.turn_id)
        assert _evaluate(store, turn.turn_id).action == "continue"


def test_result_error_cannot_be_replaced_by_a_success_exit_status(tmp_path, monkeypatch):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start(store, tmp_path, command=COMMAND)
        original = store.record_tool_result

        def error_result(**kwargs):
            kwargs["result"] = {**kwargs["result"], "is_error": True}
            return original(**kwargs)

        monkeypatch.setattr(store, "record_tool_result", error_result)
        _command(store, turn.turn_id)
        decision = _evaluate(store, turn.turn_id)
        assert decision.action == "continue"
        assert json.loads(decision.reason)["reason_code"] == "acceptance_command_failed"


def _child(store, parent):
    suffix = uuid4().hex
    return store.start_budgeted_child_turn(
        parent_turn_id=parent.turn_id,
        child_thread_id=f"child-thread-{suffix}",
        child_turn_id=f"child-turn-{suffix}",
        user_message="perform delegated work",
        binding_manifest={},
        requested_tokens=None,
        requested_cost_micros=None,
    )


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("rebuild", [False, True])
def test_delegated_change_after_parent_acceptance_requires_a_fresh_parent_check(tmp_path, depth, rebuild):
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        parent = _start(store, tmp_path, command=COMMAND)
        # Parent and child item sequences are unrelated. In particular, a
        # later child's mutation can have a smaller local sequence number.
        for index in range(3):
            _command(store, parent.turn_id, call_id=f"earlier-{index}", command="unrelated command")
        _command(store, parent.turn_id, call_id="parent-check")
        descendant = parent
        for _ in range(depth):
            descendant = _child(store, descendant)
        _change(store, descendant.turn_id, tmp_path, call_id="delegated-change")
        if not rebuild:
            decision = _evaluate(store, parent.turn_id)
            assert decision.action == "continue"
            assert json.loads(decision.reason)["reason_code"] == "acceptance_command_stale"

    with RolloutStore(database) as store:
        if rebuild:
            store.rebuild_projections()
        decision = _evaluate(store, parent.turn_id)
        assert decision.action == "continue"
        assert json.loads(decision.reason)["reason_code"] == "acceptance_command_stale"
        _command(store, parent.turn_id, call_id="parent-rerun")
        assert _evaluate(store, parent.turn_id).action == "accept"
        assert store.verify().valid


@pytest.mark.parametrize("depth", [1, 2])
def test_delegated_change_satisfies_explicit_change_requirement_before_parent_acceptance(tmp_path, depth):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        parent = _start(store, tmp_path, require_change=True, command=COMMAND)
        descendant = parent
        for _ in range(depth):
            descendant = _child(store, descendant)
        _change(store, descendant.turn_id, tmp_path, call_id="delegated-change")
        _command(store, parent.turn_id, call_id="parent-check")
        decision = _evaluate(store, parent.turn_id)
        assert decision.action == "accept"
        assert decision.validation_scope == "acceptance_command"


def test_delegated_change_alone_satisfies_only_the_recorded_mutation_scope(tmp_path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        parent = _start(store, tmp_path, require_change=True)
        child = _child(store, parent)
        _change(store, child.turn_id, tmp_path, call_id="delegated-change")
        decision = _evaluate(store, parent.turn_id)
        assert decision.action == "accept"
        assert decision.validation_scope == "workspace_change_recorded"


@pytest.mark.parametrize("same_workspace", [False, True])
def test_unallocated_turn_cannot_claim_a_parent_relationship_or_stale_its_check(tmp_path, same_workspace):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        parent = _start(store, tmp_path, command=COMMAND)
        _command(store, parent.turn_id, call_id="parent-check")
        other_workspace = tmp_path if same_workspace else tmp_path / "other-workspace"
        other_workspace.mkdir(exist_ok=True)
        unrelated = store.start_turn(
            thread_id=store.create_thread(workspace=other_workspace).thread_id,
            user_message="unrelated work",
            binding_manifest={"budget_parent_turn_id": parent.turn_id},
        )
        _change(store, unrelated.turn_id, other_workspace, call_id="unrelated-change")
        assert _evaluate(store, parent.turn_id).action == "accept"


def test_child_acceptance_command_cannot_masquerade_as_the_parents_acceptance(tmp_path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        parent = _start(store, tmp_path, require_change=True, command=COMMAND)
        child = _child(store, parent)
        _change(store, child.turn_id, tmp_path, call_id="delegated-change")
        _command(store, child.turn_id, call_id="child-check")
        decision = _evaluate(store, parent.turn_id)
        assert decision.action == "continue"
        assert json.loads(decision.reason)["reason_code"] == "acceptance_command_missing"


def test_failed_child_operation_with_attested_partial_writes_stales_parent_acceptance(tmp_path):
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        parent = _start(store, tmp_path, command=COMMAND)
        _command(store, parent.turn_id, call_id="parent-check")
        child = _child(store, parent)
        _command(
            store,
            child.turn_id,
            call_id="partial-child-write",
            result={"exit_code": 1, "metadata": {"runtime_workspace_write": True, "workspace_tree_changed": True}},
            failed=True,
            resources=({"kind": "filesystem", "identity": str(tmp_path / "partial.txt"), "access": "write"},),
        )
        decision = _evaluate(store, parent.turn_id)
        assert decision.action == "continue"
        assert json.loads(decision.reason)["reason_code"] == "acceptance_command_stale"

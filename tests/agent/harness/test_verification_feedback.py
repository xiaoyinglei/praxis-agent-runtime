from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime.harness import RuntimeState
from agent_runtime.harness.completion import DeliveryCompletionGate
from agent_runtime.harness.verification import assess_tool_verification, classify_tool_verification
from tests.agent.harness.test_completion_gate import _commit_tool_result, _proposal


@pytest.mark.parametrize(
    "command",
    [
        "python3 -m unittest -q|tail -5",
        "pytest;true",
        "pytest||true",
        "pytest\ntrue",
        "pytest > /dev/null",
        'pytest "$(echo --help)"',
        "pytest --bad#argument || true",
        "python3 -m unittest --bad#argument; true",
    ],
)
def test_shell_status_cannot_be_mistaken_for_the_test_status(command: str) -> None:
    assert (
        classify_tool_verification(
            tool_name="run_command",
            arguments={"command": command},
            structured_content={"exit_code": 0},
        )
        is None
    )


@pytest.mark.parametrize(
    "result",
    [
        {"exit_code": 0, "timed_out": True},
        {"exit_code": 0, "sandbox_error": "denied"},
        {"exit_code": 0.0},
    ],
)
def test_incomplete_or_invalid_execution_is_not_evidence(result: dict) -> None:
    assert (
        classify_tool_verification(
            tool_name="run_command",
            arguments={"command": "python3 -m unittest -q"},
            structured_content=result,
        )
        is None
    )


def _start_changed_turn(store: RuntimeState, workspace: Path):
    turn = store.start_turn(
        thread_id=store.create_thread(workspace=workspace).thread_id,
        user_message="change and verify",
        binding_manifest={"completion_policy": {"require_workspace_change": True}},
    )
    _commit_tool_result(
        store,
        turn_id=turn.turn_id,
        call_id="change",
        tool_name="apply_patch",
        arguments={"file_path": "module.py"},
        result={"metadata": {"runtime_workspace_write": True, "workspace_tree_changed": True}},
        resources=({"kind": "filesystem", "identity": str(workspace / "module.py"), "access": "write"},),
    )
    return turn


def test_pipeline_feedback_identifies_the_call_and_preserves_the_actual_runner(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="wrapped-test",
            tool_name="run_command",
            arguments={"command": ".venv/bin/python -m unittest -q 2>&1 | tail -5"},
            result={"exit_code": 0, "stdout": "OK"},
        )
        decision = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        assert decision.action == "continue"
        feedback = json.loads(decision.reason)
        assert feedback["reason_code"] == "shell_composition"
        assert feedback["tool_call_id"] == "wrapped-test"
        assert feedback["suggested_command"] == ".venv/bin/python -m unittest -q"
        assert "pytest" not in decision.reason
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="direct-test",
            tool_name="run_command",
            arguments={"command": feedback["suggested_command"]},
            result={"exit_code": 0},
        )
        assert DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id)).action == "accept"
        assert store.verify().valid


@pytest.mark.parametrize(
    "command, code",
    [
        ("if [ -x .venv/bin/python ]; then .venv/bin/python -m unittest -q; fi", "shell_composition"),
        ("python3 test_module.py", "unsupported_command"),
        ("python3 -m unittest --help", "no_tests_executed"),
    ],
)
def test_feedback_reports_the_actual_rejection(tmp_path: Path, command: str, code: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="verify",
            tool_name="run_command",
            arguments={"command": command},
            result={"exit_code": 0},
        )
        decision = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        assert decision.action == "continue"
        assert json.loads(decision.reason)["reason_code"] == code


@pytest.mark.parametrize(
    "command",
    [
        ".venv/bin/python -m unittest -q",
        "uv run pytest -q tests/test_example.py",
        "cargo test",
        "go test ./...",
        "ruff check .",
    ],
)
def test_direct_verification_keeps_working(command: str) -> None:
    assessment = assess_tool_verification(
        tool_name="run_command",
        arguments={"command": command},
        structured_content={"exit_code": 0},
    )
    assert assessment.evidence is not None
    assert assessment.reason_code == "verified"


@pytest.mark.parametrize("exit_code", [1, 0])
def test_failed_operations_are_diagnosed_but_never_accepted(tmp_path: Path, exit_code: int) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="failed-test",
            tool_name="run_command",
            arguments={"command": ".venv/bin/python -m unittest -q"},
            result={"exit_code": exit_code, "stderr": "FAILED"},
            operation_failed=True,
        )
        decision = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        feedback = json.loads(decision.reason)
        assert decision.action == "continue"
        assert feedback["reason_code"] == ("execution_failed" if exit_code else "tool_failed")
        assert feedback["tool_call_id"] == "failed-test"
        assert feedback["exit_code"] == exit_code
        assert feedback["error_code"] == "command_failed"
        assert store.verify().valid


def test_shell_substitution_does_not_produce_a_changed_literal_command() -> None:
    assessment = assess_tool_verification(
        tool_name="run_command",
        arguments={"command": "pytest $TESTS | tail -5"},
        structured_content={"exit_code": 0},
    )
    assert assessment.evidence is None
    assert assessment.suggested_command is None


def test_partial_failed_write_invalidates_earlier_verification(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="verify",
            tool_name="run_command",
            arguments={"command": "python3 -m unittest -q"},
            result={"exit_code": 0},
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="partial-write",
            tool_name="apply_patch",
            arguments={"patch": "partially applied patch"},
            result={"exit_code": 1, "metadata": {"runtime_workspace_write": True, "workspace_tree_changed": True}},
            resources=({"kind": "filesystem", "identity": str(workspace / "module.py"), "access": "write"},),
            operation_failed=True,
        )
        decision = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        assert decision.action == "continue"
        assert json.loads(decision.reason)["reason_code"] == "verification_stale"


def test_inspection_feedback_describes_attested_scope_without_claiming_file_contents(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="inspect-unrelated",
            tool_name="read_file",
            arguments={"path": "notes.txt"},
            result={"content": "notes"},
            resources=({"kind": "filesystem", "identity": str(workspace / "notes.txt"), "access": "read"},),
        )
        decision = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        feedback = json.loads(decision.reason)
        assert decision.action == "continue"
        assert feedback["attested_write_targets"] == [str(workspace / "module.py")]
        assert feedback["inspected_targets"] == [str(workspace / "notes.txt")]
        assert "could not be matched" in feedback["message"]

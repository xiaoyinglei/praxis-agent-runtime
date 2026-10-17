from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutStore
from agent_runtime.harness.completion import DeliveryCompletionGate
from tests.agent.harness.test_completion_gate import _commit_tool_result, _proposal
from tests.agent.harness.test_verification_feedback import _start_changed_turn


@pytest.mark.parametrize(
    "code",
    [
        "assert True",
        "assert open('module.py').read() == 'updated'",
        "assert open('unrelated.txt').read() == 'other'",
    ],
)
def test_assertion_feedback_explains_missing_targets_and_supported_recovery(tmp_path: Path, code: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "module.py").write_text("updated")
    (workspace / "unrelated.txt").write_text("other")
    script = tmp_path / "check.py"
    script.write_text(code)
    report = tmp_path / "observation.json"
    runner = Path(__file__).resolve().parents[3] / "agent_runtime/tools/python_verification.py"
    execution = subprocess.run(
        [sys.executable, str(runner), str(script), str(report)],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert execution.returncode == 0, execution.stderr
    observation = json.loads(report.read_text())
    assert observation == {"assertions_passed": 1, "assertions_failed": 0}

    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="assertion",
            tool_name="execute_python",
            arguments={"code": code},
            result={"exit_code": execution.returncode, "verification": observation},
        )
        decision = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        assert decision.action == "continue"
        assert decision.validation_scope == "workspace_change"
        feedback = json.loads(decision.reason)
        assert feedback["reason_code"] == "assertion_targets_unattested"
        assert "successful assertions" in feedback["message"]
        assert feedback["attested_write_targets"] == [str(workspace / "module.py")]
        assert "read_file" in feedback["next_action"]
        assert "inspect_data_file" in feedback["next_action"]
        assert "test runner" in feedback["next_action"]
        assert "explicit assertion" not in feedback["next_action"]

        # The prescribed file inspection can satisfy this existing gate; the
        # gate still says only workspace_change, not task correctness.
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="inspection",
            tool_name="read_file",
            arguments={"file_path": "module.py"},
            result={"content": "updated"},
            resources=({"kind": "filesystem", "identity": str(workspace / "module.py"), "access": "read"},),
        )
        recovered = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        assert recovered.action == "accept"
        assert recovered.validation_scope == "workspace_change"
        assert store.verify().valid


def test_missing_verification_does_not_recommend_assertions_without_target_attestation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        decision = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        feedback = json.loads(decision.reason)
        assert feedback["reason_code"] == "verification_missing"
        assert "explicit assertion" not in feedback["next_action"]
        assert "read_file" in feedback["next_action"]

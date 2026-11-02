"""Historical certificates cannot replace the user's explicit command."""

import json

import pytest

from agent_runtime.harness import RuntimeStore
from tests.agent.harness.test_completion_gate import _commit_tool_result
from tests.agent.harness.test_completion_policy_v2 import _change, _command, _evaluate, _start


@pytest.mark.parametrize("code", ["assert True", "def not_run():\n    assert False", "assert open('module.py').read()"])
def test_assertions_and_historical_inspection_certificates_do_not_replace_acceptance(tmp_path, code):
    with RuntimeStore.open_sync(tmp_path / "run.db") as store:
        turn = _start(store, tmp_path, command="python check.py")
        _change(store, turn.turn_id, tmp_path)
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="assertion",
            tool_name="execute_python",
            arguments={"code": code},
            result={"exit_code": 0, "verification": {"assertions_passed": 1}},
        )
        decision = _evaluate(store, turn.turn_id)
        assert json.loads(decision.reason)["reason_code"] == "acceptance_command_missing"
        _command(store, turn.turn_id, command="python check.py")
        assert _evaluate(store, turn.turn_id).action == "accept"
        assert store.verify().valid

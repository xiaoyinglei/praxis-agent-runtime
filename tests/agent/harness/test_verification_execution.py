from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_runtime.harness.verification import assess_tool_verification


@pytest.mark.parametrize(
    "code",
    [
        'def never_called():\n    assert False\nprint("no check ran")',
        "if False:\n    assert False",
        "assert True",
    ],
)
def test_code_shape_without_execution_evidence_cannot_verify(code):
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0
    assessment = assess_tool_verification(
        tool_name="execute_python", arguments={"code": code}, structured_content={"exit_code": result.returncode}
    )
    assert assessment.evidence is None


def test_inline_python_shell_assertion_has_no_attested_execution():
    assessment = assess_tool_verification(
        tool_name="run_command",
        arguments={"command": "python3 -c 'def never_called(): assert False'"},
        structured_content={"exit_code": 0},
    )
    assert assessment.evidence is None


@pytest.mark.parametrize(
    "command",
    [
        "uv run --no-sync pytest -q",
        "uv run --offline --project . pytest -q",
        "uv --offline run --no-sync python3 -m unittest -q",
    ],
)
def test_uv_wrapper_flags_are_not_the_verifier(command):
    assessment = assess_tool_verification(
        tool_name="run_command", arguments={"command": command}, structured_content={"exit_code": 0}
    )
    assert assessment.evidence is not None
    assert assessment.evidence.kind == "test"


@pytest.mark.parametrize(
    "code, optimized",
    [
        ("def never_called():\n    assert False", False),
        ("if False:\n    assert False", False),
        ("assert True", True),
        ("assert True", False),
        ("try:\n    assert False\nexcept AssertionError:\n    pass", False),
        (
            "assert True\nAssertionError = ValueError\ntry:\n    assert False\nexcept BaseException:\n    pass",
            False,
        ),
        ("def check():\n    assert True\ncheck()", False),
        ("import sys\nassert sys.modules['__main__'].__file__ == __file__", False),
        ("assert __cached__ is None\nassert __loader__.get_filename() == __file__", False),
    ],
)
def test_managed_runner_preserves_plain_python_semantics_without_certificates(tmp_path, code, optimized):
    runner = Path(__file__).resolve().parents[3] / "agent_runtime/tools/python_verification.py"
    script = tmp_path / "task.py"
    script.write_text(code)
    report = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(runner), str(script), str(report), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    normal = subprocess.run([sys.executable, *(["-O"] if optimized else []), str(script)],
                            capture_output=True, text=True)
    assert result.returncode == normal.returncode
    assert result.stdout == normal.stdout
    assert not report.exists()


@pytest.mark.parametrize("future", [False, True])
def test_managed_runner_preserves_the_source_annotation_semantics(tmp_path, future):
    runner = Path(__file__).resolve().parents[3] / "agent_runtime/tools/python_verification.py"
    script = tmp_path / "annotations.py"
    script.write_text(
        ("from __future__ import annotations\n" if future else "")
        + "def f(value: MissingType):\n    return value\nprint(f.__annotations__)\n"
    )
    normal = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    managed = subprocess.run(
        [sys.executable, "-I", str(runner), str(script), str(tmp_path / "report.json")],
        capture_output=True, text=True,
    )
    assert managed.returncode == normal.returncode
    assert managed.stdout == normal.stdout
    if not future:
        assert "NameError" in normal.stderr and "NameError" in managed.stderr


def test_executed_assertion_does_not_invent_file_coverage(tmp_path):
    runner = Path(__file__).resolve().parents[3] / "agent_runtime/tools/python_verification.py"
    target = tmp_path / "result.txt"
    target.write_text("correct")
    script = tmp_path / "task.py"
    script.write_text("assert open('result.txt').read() == 'correct'\n")
    report = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(runner), str(script), str(report), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not report.exists()
    assessment = assess_tool_verification(
        tool_name="execute_python",
        arguments={"code": script.read_text()},
        structured_content={"exit_code": 0},
    )
    assert assessment.evidence is None


def test_unrelated_assertions_do_not_verify_the_changed_file(tmp_path):
    from agent_runtime.harness import RolloutStore
    from agent_runtime.harness.completion import DeliveryCompletionGate
    from tests.agent.harness.test_completion_gate import _commit_tool_result, _proposal
    from tests.agent.harness.test_verification_feedback import _start_changed_turn

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        for target in ["other.py", "module.py"]:
            _commit_tool_result(
                store,
                turn_id=turn.turn_id,
                call_id=target,
                tool_name="execute_python",
                arguments={"code": "actual managed check"},
                result={
                    "exit_code": 0,
                    "verification": {
                        "assertions_passed": 1,
                        "assertions_failed": 0,
                        "read_resources": [str(workspace / target)],
                    },
                },
            )
            decision = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
            assert decision.action == "continue"
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="inspect",
            tool_name="read_file",
            arguments={"file_path": "module.py"},
            result={"content": "updated"},
            resources=({"kind": "filesystem", "identity": str(workspace / "module.py"), "access": "read"},),
        )
        assert DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id)).action == "accept"
        assert store.verify().valid


@pytest.mark.parametrize("command", ["pytest --co -q", "python3 -m pytest --co", "cargo test --no-run"])
def test_collection_and_build_only_modes_do_not_execute_checks(command):
    result = assess_tool_verification(
        tool_name="run_command",
        arguments={"command": command},
        structured_content={"exit_code": 0},
    )
    assert result.evidence is None
    assert result.reason_code == "no_tests_executed"


def test_managed_runner_does_not_overwrite_task_files(tmp_path):
    runner = Path(__file__).resolve().parents[3] / "agent_runtime/tools/python_verification.py"
    script = tmp_path / "task.py"
    report = tmp_path / "report.json"
    script.write_text(
        "import atexit, json\nfrom pathlib import Path\n"
        "def not_called():\n    assert False\n"
        "atexit.register(lambda: Path(__file__).with_name('report.json').write_text("
        "json.dumps({'assertions_passed': 1, 'assertions_failed': 0})))\n"
    )
    result = subprocess.run([sys.executable, str(runner), str(script), str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text()) == {"assertions_passed": 1, "assertions_failed": 0}


def test_saved_verification_is_reassessed_against_bound_source_result(tmp_path):
    from agent_runtime.harness import RolloutStore
    from agent_runtime.harness.completion import DeliveryCompletionGate
    from tests.agent.harness.test_completion_gate import _commit_tool_result, _proposal
    from tests.agent.harness.test_verification_feedback import _start_changed_turn

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="collected",
            tool_name="run_command",
            arguments={"command": "pytest --co"},
            result={"exit_code": 0},
        )
        # An immutable historical event emitted by the old classifier.
        store.record_verification(
            turn_id=turn.turn_id,
            operation_id="operation-collected",
            kind="test",
            verifier="pytest",
        )
        decision = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        assert decision.action == "continue"
        assert json.loads(decision.reason)["reason_code"] == "no_tests_executed"
        assert store.verify().valid


def test_mismatched_call_and_operation_cannot_attest_saved_verification(tmp_path, monkeypatch):
    from agent_runtime.harness import RolloutStore
    from agent_runtime.harness.completion import DeliveryCompletionGate
    from tests.agent.harness.test_completion_gate import _commit_tool_result, _proposal
    from tests.agent.harness.test_verification_feedback import _start_changed_turn

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        record_call = store.record_tool_call

        def inconsistent_call(**kwargs):
            return record_call(**{**kwargs, "tool_name": "read_file"})

        monkeypatch.setattr(store, "record_tool_call", inconsistent_call)
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="mismatch",
            tool_name="run_command",
            arguments={"command": "pytest -q"},
            result={"exit_code": 0},
        )
        assert store.verify().valid
        assert DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id)).action == "continue"

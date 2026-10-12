from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutStore
from agent_runtime.harness.completion import DeliveryCompletionGate
from agent_runtime.harness.verification import assess_tool_verification
from tests.agent.harness.test_completion_gate import _commit_tool_result, _proposal
from tests.agent.harness.test_verification_feedback import _start_changed_turn


@pytest.mark.parametrize("flags", [["--exit-zero"], ["-e"], ["-qe"], ["--fix-only"], ["--diff"]])
def test_actual_ruff_failure_suppression_cannot_pass_delivery_gate(tmp_path: Path, flags: list[str]) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "module.py"
    target.write_text("def broken(:\n    pass\n")
    ruff = str(Path(sys.executable).parent / "ruff")
    command = [ruff, "check", "--isolated", *flags, "module.py"]
    result = subprocess.run(command, cwd=workspace, capture_output=True, text=True)
    assert result.returncode == 0
    ordinary = subprocess.run([ruff, "check", "--isolated", "module.py"], cwd=workspace, capture_output=True, text=True)
    assert ordinary.returncode == 1

    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn = _start_changed_turn(store, workspace)
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="suppressed-failure",
            tool_name="run_command",
            arguments={"command": shlex.join(command)},
            result={"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
        )
        decision = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        assert decision.action == "continue"
        assert decision.validation_status == "failed"
        assert json.loads(decision.reason)["reason_code"] == "unattested_verifier_mode"

        target.write_text("value = 1\n")
        checked = subprocess.run(
            [ruff, "check", "--isolated", "module.py"], cwd=workspace, capture_output=True, text=True
        )
        assert checked.returncode == 0
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="ordinary-check",
            tool_name="run_command",
            arguments={"command": shlex.join([ruff, "check", "--isolated", "module.py"])},
            result={"exit_code": checked.returncode, "stdout": checked.stdout, "stderr": checked.stderr},
        )
        passed = DeliveryCompletionGate(store).evaluate(_proposal(store, turn.turn_id))
        assert passed.action == "accept"
        assert passed.validation_status == "passed"
        assert passed.validation_scope == "workspace_change"
        assert store.verify().valid


@pytest.mark.parametrize(
    "command, reason",
    [
        ("ruff check --show-files .", "no_tests_executed"),
        ("ruff check --show-settings module.py", "no_tests_executed"),
        ("ruff check --add-noqa=reason module.py", "unattested_verifier_mode"),
        ("ruff check -qw .", "unattested_verifier_mode"),
        ("pytest --setup-plan", "no_tests_executed"),
        ("python3 -m pytest --setup-only", "no_tests_executed"),
        ("pytest --fixtures-per-test", "no_tests_executed"),
        ("pytest --cache-show=lastfailed", "no_tests_executed"),
        ("pytest -qV", "no_tests_executed"),
        ("python3 -m unittest -qh", "no_tests_executed"),
        ("mypy -V", "no_tests_executed"),
        ("cargo test -- --list", "no_tests_executed"),
        ("go test -n ./...", "no_tests_executed"),
        ("go test -h", "no_tests_executed"),
        ("go test -c ./...", "no_tests_executed"),
        ("go test -list=Test ./...", "no_tests_executed"),
        ("go test ./... -args -test.list=Test", "no_tests_executed"),
        ("go test -exec=true ./...", "unattested_verifier_mode"),
        ("npm run test --if-present=true", "unattested_verifier_mode"),
        ("pnpm run test --if-present", "unattested_verifier_mode"),
        ("npm test --script-shell=/usr/bin/true", "unattested_verifier_mode"),
        ("uv run --no-sync ruff check -e module.py", "unattested_verifier_mode"),
    ],
)
def test_nonchecking_or_failure_suppressing_modes_cannot_attest_success(command: str, reason: str) -> None:
    assessment = assess_tool_verification(
        tool_name="run_command", arguments={"command": command}, structured_content={"exit_code": 0}
    )
    assert assessment.evidence is None
    assert assessment.reason_code == reason


@pytest.mark.parametrize(
    "command",
    [
        "ruff check --fix --exit-non-zero-on-fix .",
        "ruff check --no-fix-only --silent --output-format=json .",
        "ruff check -- --exit-zero.py",
        "pytest -q --setup-show -k test_success",
        "python3 -m unittest discover -v",
        "mypy --strict --no-error-summary module.py",
        "cargo test --no-fail-fast -- --nocapture",
        "go test -race -count=1 ./...",
        "go test -shuffle=on -run=TestHealth ./...",
        "npm run test -- --runInBand",
        "pnpm test",
        "yarn run test",
        "uv run --no-sync --offline pytest -q",
    ],
)
def test_ordinary_checks_preserve_the_success_exit_contract(command: str) -> None:
    assessment = assess_tool_verification(
        tool_name="run_command", arguments={"command": command}, structured_content={"exit_code": 0}
    )
    assert assessment.evidence is not None
    assert assessment.reason_code == "verified"

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "agent_public_path_benchmark.py"
MANIFEST = ROOT / "evals" / "harness" / "public_path_scenarios_v1.json"


def _load_module():
    assert SCRIPT.is_file()
    spec = importlib.util.spec_from_file_location("agent_public_path_benchmark", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_freezes_five_distinct_public_path_scenarios() -> None:
    module = _load_module()

    manifest = module.load_manifest(MANIFEST)

    assert manifest.model == "qwen3_5_9b_mlx_4bit"
    assert manifest.repetitions == 3
    assert tuple(scenario.scenario_id for scenario in manifest.scenarios) == (
        "single_file_fix",
        "cross_file_feature",
        "data_artifact",
        "approval_write_command",
        "crash_recovery",
    )
    assert tuple(scenario.mode for scenario in manifest.scenarios) == (
        "direct",
        "direct",
        "direct",
        "approval",
        "crash_after_model_tool_call",
    )


def test_gate_requires_every_scenario_to_pass_three_times() -> None:
    module = _load_module()
    manifest = module.load_manifest(MANIFEST)
    results = [
        {
            "scenario_id": scenario.scenario_id,
            "repetition": repetition,
            "status": "passed",
            "model": manifest.model,
            "side_effect_count": 1,
            "crash_observed": scenario.scenario_id == "crash_recovery",
        }
        for scenario in manifest.scenarios
        for repetition in range(1, 4)
    ]

    summary = module.evaluate_results(manifest, results)

    assert summary["release_ready"] is True
    assert summary["passed"] == 15
    assert summary["reasons"] == []


def test_gate_rejects_missing_failure_duplicate_and_non_exactly_once_effect() -> None:
    module = _load_module()
    manifest = module.load_manifest(MANIFEST)
    results = [
        {
            "scenario_id": scenario.scenario_id,
            "repetition": repetition,
            "status": "passed",
            "model": manifest.model,
            "side_effect_count": 1,
            "crash_observed": scenario.scenario_id == "crash_recovery",
        }
        for scenario in manifest.scenarios
        for repetition in range(1, 4)
    ]
    results.pop()
    results[0]["status"] = "failed"
    crash_result = next(
        result
        for result in results
        if result["scenario_id"] == "crash_recovery"
        and result["repetition"] == 1
    )
    crash_result["side_effect_count"] = 2
    results.append(dict(results[2]))

    with pytest.raises(ValueError, match="duplicate"):
        module.evaluate_results(manifest, results)

    results.pop()
    summary = module.evaluate_results(manifest, results)
    assert summary["release_ready"] is False
    assert any(reason.startswith("failed:") for reason in summary["reasons"])
    assert any(reason.startswith("not_exactly_once:") for reason in summary["reasons"])
    assert any(reason.startswith("missing:") for reason in summary["reasons"])


def test_manifest_rejects_unknown_mode(tmp_path: Path) -> None:
    module = _load_module()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["scenarios"][0]["mode"] = "pretend"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mode"):
        module.load_manifest(path)


def test_direct_scenario_uses_public_cli_and_verifies_the_real_workspace_diff(
    tmp_path: Path,
) -> None:
    module = _load_module()
    fake_agent = tmp_path / "fake_agent.py"
    fake_agent.write_text(
        "from pathlib import Path\n"
        "Path('calculator.py').write_text('def add(left: int, right: int) -> int:\\n    return left + right\\n')\n"
        "print('Turn: turn-public-1')\n"
        "print('状态: done')\n",
        encoding="utf-8",
    )
    scenario = module.PublicPathScenario(
        scenario_id="single_file_fix",
        mode="direct",
        prompt="fix it",
        files={
            "calculator.py": "def add(left: int, right: int) -> int:\n    return left - right\n",
            "test_calculator.py": "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        },
        acceptance_command=("uv", "run", "pytest", "-q", "test_calculator.py"),
        required_changes=("calculator.py",),
    )

    result, result_path = module.run_scenario(
        scenario,
        repetition=1,
        model="real-model",
        agent_command=(sys.executable, str(fake_agent)),
        artifacts_root=tmp_path / "artifacts",
    )

    assert result["status"] == "passed"
    assert result["turn_id"] == "turn-public-1"
    assert result["changed_paths"] == ["calculator.py"]
    assert result_path.is_file()


def test_approval_scenario_must_pause_then_resume_the_same_public_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_approval_is_allowed", lambda **_kwargs: True)
    fake_agent = tmp_path / "fake_approval_agent.py"
    fake_agent.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == 'run':\n"
        "    print('Turn: turn-approval-1')\n"
        "    print('状态: paused')\n"
        "    raise SystemExit(2)\n"
        "assert sys.argv[1:3] == ['resume', 'turn-approval-1']\n"
        "Path('approved.txt').write_text('approved-content\\n')\n"
        "print('Turn: turn-approval-1')\n"
        "print('状态: done')\n",
        encoding="utf-8",
    )
    scenario = module.PublicPathScenario(
        scenario_id="approval_write_command",
        mode="approval",
        prompt="write after approval",
        files={"README.md": "seed\n"},
        acceptance_command=(
            sys.executable,
            "-c",
            "assert open('approved.txt').read() == 'approved-content\\n'",
        ),
        required_changes=("approved.txt",),
    )

    result, _result_path = module.run_scenario(
        scenario,
        repetition=1,
        model="real-model",
        agent_command=(sys.executable, str(fake_agent)),
        artifacts_root=tmp_path / "artifacts",
    )

    assert result["status"] == "passed"
    assert result["turn_id"] == "turn-approval-1"
    assert result["approval_count"] == 1


def test_resume_does_not_approve_a_pause_outside_the_allowed_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    paused = module.subprocess.CompletedProcess(
        args=["agent", "run"],
        returncode=2,
        stdout="Turn: turn-network\n状态: paused\n",
        stderr="",
    )
    monkeypatch.setattr(module, "_approval_is_allowed", lambda **_kwargs: False)

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("disallowed approval must not resume")

    monkeypatch.setattr(module, "_run_process", unexpected_run)

    terminal, attempts = module._resume_until_terminal(
        initial=paused,
        agent_command=("agent",),
        workspace=tmp_path,
        checkpoint=tmp_path / "rollout.sqlite3",
        allowed_approval_effects=frozenset(
            {"read_workspace", "write_workspace", "execute_process", "destructive"}
        ),
    )

    assert terminal is paused
    assert attempts == (paused,)


def test_gate_rejects_crash_scenario_when_no_injected_crash_was_observed() -> None:
    module = _load_module()
    manifest = module.load_manifest(MANIFEST)
    results = [
        {
            "scenario_id": scenario.scenario_id,
            "repetition": repetition,
            "status": "passed",
            "model": manifest.model,
            "side_effect_count": 1,
            "crash_observed": scenario.scenario_id == "crash_recovery",
        }
        for scenario in manifest.scenarios
        for repetition in range(1, 4)
    ]
    crash_result = next(
        result
        for result in results
        if result["scenario_id"] == "crash_recovery"
        and result["repetition"] == 1
    )
    crash_result["crash_observed"] = False

    summary = module.evaluate_results(manifest, results)

    assert summary["release_ready"] is False
    assert "crash_not_observed:crash_recovery:1" in summary["reasons"]

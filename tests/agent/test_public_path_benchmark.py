from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "agent_public_path_benchmark.py"
MANIFEST = ROOT / "tests" / "agent" / "fixtures" / "public_path_scenarios_v1.json"


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

    assert manifest.model == "mlx-community/Qwen3.5-9B-4bit"
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
        acceptance_command=(sys.executable, "-m", "pytest", "-q", "test_calculator.py"),
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
) -> None:
    from agent_runtime.harness import RolloutStore

    module = _load_module()
    fake_agent = tmp_path / "fake_approval_agent.py"
    fake_agent.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from agent_runtime.harness import RolloutStore\n"
        "checkpoint = Path(sys.argv[sys.argv.index('--checkpoint-db') + 1])\n"
        "if sys.argv[1] == 'run':\n"
        "    assert not Path('approved.txt').exists()\n"
        "    with RolloutStore(checkpoint) as store:\n"
        "        thread = store.create_thread(workspace=Path.cwd())\n"
        "        turn = store.start_turn(thread_id=thread.thread_id, user_message='write after approval',\n"
        "                                binding_manifest={}, turn_id='turn-approval-1')\n"
        "        store.request_tool_approval(turn_id=turn.turn_id, operation_id='write-1',\n"
        "            tool_call_id='call-1', tool_name='write_file', arguments_digest='fixture-digest',\n"
        "            execution_revision='fixture-v1', idempotent=False, request={}, effects=('write_workspace',))\n"
        "    print('Turn: turn-approval-1')\n"
        "    print('状态: paused')\n"
        "    raise SystemExit(2)\n"
        "assert sys.argv[1:3] == ['recover', 'turn-approval-1']\n"
        "assert sys.argv[sys.argv.index('--action') + 1] == 'allow_once'\n"
        "assert not Path('approved.txt').exists()\n"
        "with RolloutStore(checkpoint) as store:\n"
        "    assert len(store.list_turns()) == 1\n"
        "    assert store.read_turn(sys.argv[2]).status == 'paused'\n"
        "    store.resolve_tool_approval(turn_id=sys.argv[2], decision='approve')\n"
        "    operation = store.claim_tool_operation(operation_id='write-1', worker_id='fixture', lease_seconds=30)\n"
        "    Path('approved.txt').write_text('approved-content\\n')\n"
        "    assert store.commit_tool_operation_result(turn_id=sys.argv[2], operation_id=operation.operation_id,\n"
        "        claim_generation=operation.claim_generation, fencing_token=operation.fencing_token,\n"
        "        status='succeeded', attempt_count=operation.attempt_count, error_code=None,\n"
        "        requires_reconciliation=False, result={'text': 'written'})\n"
        "    store.complete_turn(turn_id=sys.argv[2], answer='written')\n"
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

    result, result_path = module.run_scenario(
        scenario,
        repetition=1,
        model="real-model",
        agent_command=(sys.executable, str(fake_agent)),
        artifacts_root=tmp_path / "artifacts",
    )

    assert result["status"] == "passed"
    assert result["turn_id"] == "turn-approval-1"
    assert result["approval_count"] == 1
    stdout = (result_path.parent / "agent.stdout").read_text(encoding="utf-8")
    assert stdout.count("Turn: turn-approval-1") == 2
    assert "状态: paused" in stdout
    assert "状态: done" in stdout
    with RolloutStore(result_path.parent / "rollout.sqlite3") as store:
        assert len(store.list_turns()) == 1
        assert store.read_turn("turn-approval-1").status == "completed"
        [approval] = store.list_approvals("turn-approval-1")
        assert approval.status == "approved"
        [operation] = store.list_tool_operations("turn-approval-1")
        assert operation.status == "succeeded"
        assert operation.attempt_count == 1
        assert store.verify().valid


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


def test_candidate_snapshot_binds_dirty_source_and_rejects_drift(tmp_path: Path) -> None:
    module = _load_module()
    repository = tmp_path / "repository"
    repository.mkdir()
    module.subprocess.run(["git", "init", "-q", str(repository)], check=True)
    runtime = repository / "agent_runtime"
    runtime.mkdir()
    (runtime / "agent.py").write_text("original\n")
    module.subprocess.run(["git", "add", "agent_runtime"], cwd=repository, check=True)
    (runtime / "agent.py").write_text("dirty candidate\n")
    (runtime / "new.py").write_text("new candidate\n")
    (repository / ".env").write_text("SECRET=private\n")
    destination = tmp_path / "candidate"

    identity = module.freeze_candidate(repository, destination)

    assert identity["git_dirty"] is True
    assert (destination / "agent_runtime" / "agent.py").read_text() == "dirty candidate\n"
    assert (destination / "agent_runtime" / "new.py").read_text() == "new candidate\n"
    assert not (destination / ".env").exists()
    assert module.verify_candidate(destination) == identity
    (destination / "agent_runtime" / "agent.py").write_text("changed after freeze\n")
    with pytest.raises(ValueError, match="changed"):
        module.verify_candidate(destination)


def test_candidate_freeze_refuses_workspace_destination(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="outside"):
        module.freeze_candidate(tmp_path, tmp_path / "candidate")


def test_coding_fixtures_use_project_tests_and_hidden_acceptance(tmp_path: Path) -> None:
    module = _load_module()
    manifest = module.load_manifest(MANIFEST)
    for scenario in manifest.scenarios[:2]:
        assert "python3 -m unittest" in scenario.prompt
        assert "python3 -c" not in scenario.prompt
        module._seed_workspace(tmp_path / scenario.scenario_id, scenario.files)
        failed = module._run_process(scenario.acceptance_command, cwd=tmp_path / scenario.scenario_id)
        assert failed.returncode != 0
        assert "acceptance" not in scenario.files


def test_live_cli_binds_explicit_budgets_and_rechecks_candidate_before_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import agent_runtime
    from agent_runtime.core.llm_registry import ModelRegistry

    module = _load_module()
    root = tmp_path / "frozen"
    root.mkdir()
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(agent_runtime, "__file__", str(root / "agent_runtime" / "__init__.py"))
    source = {"source_sha256": "candidate"}
    checks: list[Path] = []

    def verify(path: Path):
        checks.append(path)
        return source

    monkeypatch.setattr(module, "verify_candidate", verify)
    definition = SimpleNamespace(provider_name="example", provider=SimpleNamespace(value="example"),
                                 model_id="test-model", definition_revision="test-revision")
    monkeypatch.setattr(
        ModelRegistry, "from_env", lambda *_args: SimpleNamespace(get_model_definition=lambda _m: definition),
    )
    monkeypatch.setattr(module, "_resolve_agent_command", lambda _v: ("agent",))
    calls: list[dict[str, object]] = []

    def run(_manifest, **kwargs):
        calls.append(kwargs)
        return {"release_ready": True}, ()

    monkeypatch.setattr(module, "run_benchmark", run)
    evidence: list[dict[str, object]] = []
    monkeypatch.setattr(module, "_write_release_evidence", lambda **kwargs: evidence.append(kwargs))
    result = module.main([
        "run", "--model", "test", "--max-turns", "7", "--timeout-seconds", "19",
        "--artifacts-root", str(tmp_path / "results"), "--evidence-artifact", "evidence.json",
    ])
    assert result == 0
    assert calls[0]["max_turns"] == 7
    assert calls[0]["timeout_seconds"] == 19
    assert checks == [root, root]
    assert evidence[0]["execution"]["source_unchanged"] is True


def test_process_timeout_is_recordable_and_logged_secrets_are_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    timed = module._run_process([sys.executable, "-c", "import time; time.sleep(5)"], cwd=tmp_path, timeout_seconds=1)
    assert timed.returncode == 124
    secret = "test-only-secret-abcdefghijkl"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    printed = module._run_process(
        [sys.executable, "-c", "import os; print(os.environ['DEEPSEEK_API_KEY'])"], cwd=tmp_path,
    )
    assert printed.returncode == 0
    assert secret not in printed.stdout


def test_crash_injection_uses_public_event_sink_after_durable_model_tool_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import agent_runtime

    module = _load_module()

    class CrashError(Exception):
        pass

    class EventAgent:
        def __init__(self, **kwargs):
            pass

        async def run(self, task, *, max_turns, require_workspace_change, event_sink):
            assert require_workspace_change is True
            await event_sink.emit(SimpleNamespace(
                turn_id="turn-injected", type="item_completed", item_kind="agent_message", data={"tool_calls": []},
            ))
            await event_sink.emit(SimpleNamespace(
                turn_id="turn-injected", type="item_completed", item_kind="agent_message",
                data={"tool_calls": [{"name": "read_file"}]},
            ))

    monkeypatch.setattr(agent_runtime, "Agent", EventAgent)

    def exit_after_commit(status):
        assert status == 97
        assert (tmp_path / "turn.txt").read_text() == "turn-injected"
        raise CrashError

    monkeypatch.setattr(module.os, "_exit", exit_after_commit)
    args = SimpleNamespace(model="fake", workspace=tmp_path, checkpoint_db=tmp_path / "rollout.sqlite3",
                           model_session_path=tmp_path / "model.json", turn_file=tmp_path / "turn.txt",
                           prompt="read task", max_turns=3)
    with pytest.raises(CrashError):
        module._crash_child(args)

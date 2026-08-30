from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "agent_code_benchmark.py"
REAL_MANIFEST_PATH = (
    Path(__file__).parents[2] / "evals" / "code_agent" / "benchmark_v1.json"
)


def _load_benchmark_module():
    assert SCRIPT_PATH.is_file(), "code-agent benchmark entrypoint is missing"
    spec = importlib.util.spec_from_file_location(
        "agent_code_benchmark",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_code_agent_benchmark_entrypoint_exists() -> None:
    module = _load_benchmark_module()

    assert module is not None


def test_architecture_layer_inference_tracks_the_replacement_harness() -> None:
    module = _load_benchmark_module()

    layers = module._infer_architecture_layers(
        (
            "agent_runtime/agent.py",
            "agent_runtime/harness/composition.py",
            "agent_runtime/harness/session.py",
            "agent_runtime/harness/tool_orchestrator.py",
            "agent_runtime/harness/rollout.py",
            "agent_runtime/harness/events.py",
        )
    )

    assert layers == {
        "public_api_cli",
        "service",
        "loop",
        "tool",
        "turn_checkpoint",
        "result_events",
    }


def test_run_task_requires_an_explicit_model_lane() -> None:
    module = _load_benchmark_module()

    with pytest.raises(SystemExit):
        module._build_parser().parse_args(
            [
                "run-task",
                "benchmark.json",
                "task-1",
                "--artifacts-root",
                "artifacts",
            ]
        )


def test_release_run_matrix_contains_every_primary_and_only_control_tasks(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))
    manifest = module.load_manifest(path)

    lanes = module.release_run_matrix(manifest)

    assert lanes == (
        *((task, manifest.primary_model) for task in manifest.tasks),
        *((task, manifest.control_model) for task in manifest.tasks if task.control),
    )
    assert len(lanes) == 36


def test_release_evidence_binds_every_raw_result_and_gate_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))
    manifest = module.load_manifest(path)
    artifacts = tmp_path / "artifacts"
    result = artifacts / "raw" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"outcome":"success"}\n', encoding="utf-8")
    evidence = artifacts / "release" / "evidence.json"
    monkeypatch.setattr(
        module,
        "repository_state_fingerprint",
        lambda _repository: "a" * 64,
    )

    module._write_release_evidence(
        path=Path("release/evidence.json"),
        artifacts_root=artifacts,
        manifest=manifest,
        repository=tmp_path,
        summary={"release_ready": True},
        result_paths=(result,),
    )

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "praxis-code-agent-release-evidence-v1"
    assert payload["runtime_fingerprint"] == "a" * 64
    assert payload["gate"] == {"release_ready": True}
    assert payload["results"] == [
        {
            "path": "raw/result.json",
            "sha256": module.hashlib.sha256(result.read_bytes()).hexdigest(),
        }
    ]


def test_release_evidence_loader_rejects_tampered_raw_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))
    manifest = module.load_manifest(path)
    artifacts = tmp_path / "artifacts"
    result = artifacts / "raw" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text("original\n", encoding="utf-8")
    evidence = artifacts / "release" / "evidence.json"
    monkeypatch.setattr(
        module,
        "repository_state_fingerprint",
        lambda _repository: "a" * 64,
    )
    module._write_release_evidence(
        path=Path("release/evidence.json"),
        artifacts_root=artifacts,
        manifest=manifest,
        repository=tmp_path,
        summary={"release_ready": True},
        result_paths=(result,),
    )
    result.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        module.load_release_evidence(
            evidence,
            artifacts_root=artifacts,
            manifest=manifest,
        )


def test_run_release_cli_requires_external_artifact_and_explicit_agent_command() -> None:
    module = _load_benchmark_module()

    args = module._build_parser().parse_args(
        [
            "run-release",
            "benchmark.json",
            "--evidence-artifact",
            "artifacts/harness/PROD-02/evidence.json",
            "--agent-command",
            "uv run agent",
        ]
    )

    assert args.command == "run-release"
    assert args.agent_command == "uv run agent"
    assert args.artifacts_root is None


def test_current_runtime_agent_command_is_absolute_and_not_task_workspace_relative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    interpreter = tmp_path / "venv" / "bin" / "python"
    entrypoint = interpreter.with_name("agent")
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(module.sys, "executable", str(interpreter))

    command = module._resolve_agent_command("current-runtime")

    assert command == (str(entrypoint.resolve()),)


def test_run_task_cli_resolves_current_runtime_like_run_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    manifest = SimpleNamespace(tasks=(SimpleNamespace(task_id="task-1"),))
    captured: dict[str, object] = {}
    monkeypatch.setattr(module, "load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        module,
        "validate_repository_bindings",
        lambda _manifest, repository: None,
    )
    monkeypatch.setattr(
        module,
        "_resolve_agent_command",
        lambda value: ("/trusted/current/agent", value),
    )

    def fake_run_task(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            task_id="task-1",
            model_alias="model-1",
            outcome=SimpleNamespace(value="passed"),
            turn_id="turn-1",
            artifact_dir=tmp_path / "artifact",
        )

    monkeypatch.setattr(module, "run_task", fake_run_task)

    exit_code = module.main(
        [
            "run-task",
            "benchmark.json",
            "task-1",
            "--artifacts-root",
            str(tmp_path),
            "--model",
            "model-1",
            "--agent-command",
            "current-runtime",
        ]
    )

    assert exit_code == 0
    assert captured["agent_command"] == (
        "/trusted/current/agent",
        "current-runtime",
    )


def test_compare_kernels_cli_freezes_the_old_kernel_commit() -> None:
    module = _load_benchmark_module()

    args = module._build_parser().parse_args(
        [
            "compare-kernels",
            "benchmark.json",
            "--baseline-commit",
            "a" * 40,
            "--evidence-artifact",
            "artifacts/harness/PROD-03/evidence.json",
        ]
    )

    assert args.command == "compare-kernels"
    assert args.baseline_commit == "a" * 40


def _task_payload(
    task_id: str,
    *,
    split: str = "regression",
    category: str = "local",
    layers: list[str] | None = None,
    control: bool = False,
) -> dict[str, object]:
    return {
        "id": task_id,
        "split": split,
        "category": category,
        "layers": layers or ["tool"],
        "control": control,
        "source_commit": "1" * 40,
        "target_commit": "2" * 40,
        "instruction": (
            "This is an implementation task in the current repository. "
            "Modify the code and run focused tests. "
            "Fix the behavior and verify the focused regression."
        ),
        "acceptance_command": [
            "uv",
            "run",
            "pytest",
            "-q",
            "tests/agent/test_example.py",
        ],
        "acceptance_files": ["tests/agent/test_example.py"],
        "setup_command": ["uv", "sync", "--frozen"],
        "allowed_paths": ["agent_runtime/**", "rag/agent/**", "tests/agent/**"],
        "permissions": {
            "write_workspace": True,
            "execute_process": True,
            "network": False,
        },
        "budget": {
            "max_turns": 30,
            "timeout_seconds": 1800,
        },
    }


def _manifest_payload(
    *,
    benchmark_version: str = "code-agent-v1",
    primary_model: str = "qwen3_5_9b_mlx_4bit",
    control_model: str = "groq_gpt_oss_120b",
    diagnostic_model: str = "kimi_cloud",
    tasks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark_version": benchmark_version,
        "models": {
            "primary": primary_model,
            "control": control_model,
            "diagnostic": diagnostic_model,
        },
        "tasks": tasks or [_task_payload("task-1")],
    }


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_manifest_locks_qwen_primary_and_groq_control(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload())

    manifest = module.load_manifest(path)

    assert manifest.primary_model == "qwen3_5_9b_mlx_4bit"
    assert manifest.control_model == "groq_gpt_oss_120b"
    assert manifest.diagnostic_model == "kimi_cloud"
    assert manifest.benchmark_version == "code-agent-v1"
    assert len(manifest.fingerprint) == 64


@pytest.mark.parametrize(
    ("field", "model"),
    [
        ("primary_model", "deepseek_chat"),
        ("control_model", "deepseek_reasoner"),
        ("diagnostic_model", "deepseek_chat"),
    ],
)
def test_manifest_rejects_deepseek_models(
    tmp_path: Path,
    field: str,
    model: str,
) -> None:
    module = _load_benchmark_module()
    kwargs = {field: model}
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(**kwargs))

    with pytest.raises(ValueError, match="model policy"):
        module.load_manifest(path)


def test_cross_layer_task_requires_two_predeclared_layers(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    task = _task_payload(
        "cross-layer",
        category="cross_layer",
        layers=["service"],
    )
    _write_manifest(path, _manifest_payload(tasks=[task]))

    with pytest.raises(ValueError, match="at least two architecture layers"):
        module.load_manifest(path)


def test_task_replacement_requires_new_benchmark_version(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    _write_manifest(
        before_path,
        _manifest_payload(tasks=[_task_payload("task-1")]),
    )
    _write_manifest(
        after_path,
        _manifest_payload(tasks=[_task_payload("task-2")]),
    )
    before = module.load_manifest(before_path)
    after = module.load_manifest(after_path)

    with pytest.raises(ValueError, match="benchmark_version"):
        module.validate_manifest_evolution(before, after)


def _release_tasks() -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    regression_categories = (
        ["local"] * 9
        + ["medium"] * 8
        + ["cross_layer"] * 8
    )
    control_counts = {"local": 0, "medium": 0, "cross_layer": 0}
    for index, category in enumerate(regression_categories, start=1):
        control = control_counts[category] < 2
        if control:
            control_counts[category] += 1
        layers = (
            ["service", "loop"]
            if category == "cross_layer"
            else ["tool"]
        )
        tasks.append(
            _task_payload(
                f"regression-{index:02d}",
                category=category,
                layers=layers,
                control=control,
            )
        )
    holdout_categories = ["local", "medium", "medium", "cross_layer", "cross_layer"]
    for index, category in enumerate(holdout_categories, start=1):
        layers = (
            ["public_api_cli", "service"]
            if category == "cross_layer"
            else ["tool"]
        )
        tasks.append(
            _task_payload(
                f"holdout-{index:02d}",
                split="holdout",
                category=category,
                layers=layers,
            )
        )
    return tasks


def test_release_shape_requires_25_regression_5_holdout_and_6_controls(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))

    manifest = module.load_manifest(path)

    module.validate_release_shape(manifest)
    assert sum(task.control for task in manifest.tasks) == 6
    assert sum(task.category == "cross_layer" for task in manifest.tasks) == 10


def test_release_shape_rejects_missing_holdout_task(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    tasks = _release_tasks()
    tasks.pop()
    _write_manifest(path, _manifest_payload(tasks=tasks))

    manifest = module.load_manifest(path)

    with pytest.raises(ValueError, match="at least 5 holdout"):
        module.validate_release_shape(manifest)


def test_release_shape_rejects_holdout_control(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    tasks = _release_tasks()
    tasks[-1]["control"] = True
    _write_manifest(path, _manifest_payload(tasks=tasks))

    manifest = module.load_manifest(path)

    with pytest.raises(ValueError, match="holdout tasks cannot be controls"):
        module.validate_release_shape(manifest)


def test_release_shape_rejects_ambiguous_implementation_instruction(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    tasks = _release_tasks()
    tasks[0]["instruction"] = "Explain how the behavior could be fixed."
    _write_manifest(path, _manifest_payload(tasks=tasks))

    manifest = module.load_manifest(path)

    with pytest.raises(ValueError, match="implementation task"):
        module.validate_release_shape(manifest)


def test_real_v1_manifest_is_release_shaped_and_bound_to_git_history() -> None:
    module = _load_benchmark_module()

    manifest = module.load_manifest(REAL_MANIFEST_PATH)

    module.validate_release_shape(manifest)
    module.validate_repository_bindings(
        manifest,
        repository=Path(__file__).parents[2],
    )
    assert len(manifest.tasks) == 30
    assert sum(task.split == "regression" for task in manifest.tasks) == 25
    assert sum(task.split == "holdout" for task in manifest.tasks) == 5
    assert sum(task.category == "cross_layer" for task in manifest.tasks) == 10


def _release_result(
    manifest,
    task,
    model_alias: str,
    *,
    outcome: str = "passed",
    diagnosis_primary: str = "unknown",
    diagnosis_evidence: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark_version": manifest.benchmark_version,
        "manifest_fingerprint": manifest.fingerprint,
        "task_id": task.task_id,
        "task_source_commit": task.source_commit,
        "task_target_commit": task.target_commit,
        "runtime_fingerprint": "a" * 64,
        "model_alias": model_alias,
        "outcome": outcome,
        "diagnosis": {
            "primary": diagnosis_primary,
            "secondary": [],
            "confidence": 0.0 if outcome == "passed" else 0.9,
            "evidence": diagnosis_evidence or [],
        },
    }


def _complete_release_results(manifest) -> list[dict[str, object]]:
    results = [
        _release_result(manifest, task, manifest.primary_model)
        for task in manifest.tasks
    ]
    results.extend(
        _release_result(manifest, task, manifest.control_model)
        for task in manifest.tasks
        if task.control
    )
    return results


def _write_result_artifact(
    module,
    directory: Path,
    payload: dict[str, object],
) -> Path:
    directory.mkdir(parents=True)
    evidence: dict[str, object] = {}
    for key, filename in (
        ("agent_stdout", "agent.stdout"),
        ("agent_stderr", "agent.stderr"),
        ("agent_diff", "agent.diff"),
        ("acceptance_stdout", "acceptance.stdout"),
        ("acceptance_stderr", "acceptance.stderr"),
    ):
        value = f"{key}\n"
        (directory / filename).write_text(value, encoding="utf-8")
        evidence[key] = filename
        evidence[f"{key}_sha256"] = module._sha256(value)
    payload["evidence"] = evidence
    result_path = directory / "result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_path


def test_release_gate_computes_primary_and_cross_layer_first_pass_rates(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))
    manifest = module.load_manifest(path)

    summary = module.evaluate_release_results(
        manifest,
        _complete_release_results(manifest),
    )

    assert summary["release_ready"] is True
    assert summary["overall_first_pass_rate"] == 1.0
    assert summary["cross_layer_first_pass_rate"] == 1.0
    assert summary["primary_result_count"] == 30
    assert summary["control_result_count"] == 6
    assert summary["false_completion_count"] == 0
    assert summary["safety_violation_count"] == 0
    assert summary["reasons"] == []


def test_release_gate_blocks_false_completion_from_control_run(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))
    manifest = module.load_manifest(path)
    results = _complete_release_results(manifest)
    control = next(
        result
        for result in results
        if result["model_alias"] == manifest.control_model
    )
    control.update(
        _release_result(
            manifest,
            next(task for task in manifest.tasks if task.task_id == control["task_id"]),
            manifest.control_model,
            outcome="false_completion",
            diagnosis_primary="execution_closure",
            diagnosis_evidence=["workspace_diff:empty"],
        )
    )

    summary = module.evaluate_release_results(manifest, results)

    assert summary["release_ready"] is False
    assert summary["false_completion_count"] == 1
    assert "false_completion_blocker" in summary["reasons"]


def test_release_gate_blocks_any_safety_violation(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))
    manifest = module.load_manifest(path)
    results = _complete_release_results(manifest)
    results[0]["safety_violations"] = ["workspace_escape"]

    summary = module.evaluate_release_results(manifest, results)

    assert summary["release_ready"] is False
    assert summary["safety_violation_count"] == 1
    assert "safety_violation_blocker" in summary["reasons"]


def test_release_gate_requires_complete_unique_runs_and_failure_diagnoses(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))
    manifest = module.load_manifest(path)
    results = _complete_release_results(manifest)
    missing = results.pop()
    failed = results[0]
    failed["outcome"] = "bounded_failure"

    summary = module.evaluate_release_results(manifest, results)

    assert summary["release_ready"] is False
    assert f"missing_result:{missing['task_id']}:{missing['model_alias']}" in summary[
        "reasons"
    ]
    assert f"unknown_diagnosis:{failed['task_id']}:{failed['model_alias']}" in summary[
        "reasons"
    ]

    with pytest.raises(ValueError, match="duplicate result"):
        module.evaluate_release_results(manifest, [*results, results[0]])


def test_release_gate_rejects_results_from_mixed_runtime_states(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))
    manifest = module.load_manifest(path)
    results = _complete_release_results(manifest)
    results[-1]["runtime_fingerprint"] = "b" * 64

    summary = module.evaluate_release_results(manifest, results)

    assert summary["release_ready"] is False
    assert "mixed_runtime_fingerprints" in summary["reasons"]


def test_release_gate_rejects_results_from_a_stale_runtime_state(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))
    manifest = module.load_manifest(path)

    summary = module.evaluate_release_results(
        manifest,
        _complete_release_results(manifest),
        expected_runtime_fingerprint="b" * 64,
    )

    assert summary["release_ready"] is False
    assert "runtime_fingerprint_mismatch" in summary["reasons"]


def test_kernel_comparison_requires_new_harness_to_match_old_pass_count(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    path = tmp_path / "benchmark.json"
    _write_manifest(path, _manifest_payload(tasks=_release_tasks()))
    manifest = module.load_manifest(path)
    baseline = _complete_release_results(manifest)
    candidate = _complete_release_results(manifest)
    for result in baseline:
        result["runtime_fingerprint"] = "b" * 64
    candidate[0]["outcome"] = "bounded_failure"
    candidate[0]["diagnosis"] = {
        "primary": "model_decision",
        "evidence": ["agent_result:failed"],
    }

    comparison = module.evaluate_kernel_comparison(
        manifest,
        candidate,
        baseline,
        candidate_runtime_fingerprint="a" * 64,
        baseline_runtime_fingerprint="b" * 64,
    )

    assert comparison["candidate_primary_passed"] == 29
    assert comparison["baseline_primary_passed"] == 30
    assert comparison["comparison_ready"] is False
    assert comparison["reasons"] == ["candidate_below_baseline"]

    baseline[0]["outcome"] = "bounded_failure"
    baseline[0]["diagnosis"] = {
        "primary": "model_decision",
        "evidence": ["agent_result:failed"],
    }
    matched = module.evaluate_kernel_comparison(
        manifest,
        candidate,
        baseline,
        candidate_runtime_fingerprint="a" * 64,
        baseline_runtime_fingerprint="b" * 64,
    )
    assert matched["comparison_ready"] is True
    assert matched["reasons"] == []


def test_gate_command_verifies_evidence_hashes_and_emits_release_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_benchmark_module()
    manifest_path = tmp_path / "benchmark.json"
    _write_manifest(
        manifest_path,
        _manifest_payload(tasks=_release_tasks()),
    )
    manifest = module.load_manifest(manifest_path)
    result_paths = [
        _write_result_artifact(
            module,
            tmp_path / "results" / str(index),
            payload,
        )
        for index, payload in enumerate(
            _complete_release_results(manifest),
            start=1,
        )
    ]
    monkeypatch.setattr(
        module,
        "validate_repository_bindings",
        lambda _manifest, *, repository: None,
    )
    monkeypatch.setattr(
        module,
        "repository_state_fingerprint",
        lambda _repository: "a" * 64,
    )

    exit_code = module.main(
        [
            "gate",
            str(manifest_path),
            *(str(path) for path in result_paths),
            "--repository",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["release_ready"] is True

    (result_paths[0].parent / "agent.stdout").write_text(
        "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        module.load_result_record(result_paths[0], manifest=manifest)


def test_validate_command_reports_manifest_identity() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "validate",
            str(REAL_MANIFEST_PATH),
            "--repository",
            str(Path(__file__).parents[2]),
            "--release",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["benchmark_version"] == "code-agent-v1.3"
    assert payload["task_count"] == 30
    assert len(payload["manifest_fingerprint"]) == 64


def test_diagnosis_preserves_primary_secondary_confidence_and_evidence() -> None:
    module = _load_benchmark_module()

    diagnosis = module.Diagnosis(
        primary=module.DiagnosisCause.CONTEXT_ASSEMBLY,
        secondary=(module.DiagnosisCause.TOOL_CONTRACT,),
        confidence=0.8,
        evidence=("event:model_request:2", "tool_result:call-7"),
    )

    assert diagnosis.primary.value == "context_assembly"
    assert [cause.value for cause in diagnosis.secondary] == ["tool_contract"]
    assert diagnosis.confidence == 0.8
    assert diagnosis.evidence == ("event:model_request:2", "tool_result:call-7")


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 413 - code='rate_limit_exceeded'",
        "Request too large on tokens per minute (TPM): Limit 8000, Requested 8302",
    ],
)
def test_provider_limit_parser_recognizes_groq_tpm_errors(message: str) -> None:
    module = _load_benchmark_module()

    assert module._provider_error_code(message, "") == "rate_limit"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "GROQ_API_KEY is not set. Export it or add it to .env.",
            "missing_credentials",
        ),
        (
            "httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED]",
            "transport_error",
        ),
        (
            "model_provider_failed: request timed out",
            "request_timeout",
        ),
    ],
)
def test_provider_error_parser_keeps_pre_runtime_failures_out_of_execution_closure(
    message: str,
    expected: str,
) -> None:
    module = _load_benchmark_module()

    assert module._provider_error_code("", message) == expected


def test_provider_limit_diagnosis_preserves_causes_observed_before_limit() -> None:
    module = _load_benchmark_module()
    facts = module.RunFacts(
        model_alias="groq_gpt_oss_120b",
        turn_status="failed",
        valid_diff=False,
        hidden_acceptance_passed=False,
        provider_error_code="rate_limit",
    )

    diagnosis = module.diagnose_run(
        facts,
        module.RunOutcome.PROVIDER_LIMITED,
        agent_stdout=(
            "model_tool_call_rejected: correct_tool_arguments\n"
            "search_text failed: repeated_inspection\n"
            "model_provider_failed: rate_limit\n"
        ),
        acceptance_stdout="",
        acceptance_stderr="",
    )

    assert diagnosis.primary is module.DiagnosisCause.PROVIDER_LIMIT
    assert diagnosis.secondary == (
        module.DiagnosisCause.MODEL_CAPABILITY_GAP,
        module.DiagnosisCause.TOOL_CONTRACT,
    )
    assert diagnosis.confidence == 1.0
    assert "provider_error:rate_limit" in diagnosis.evidence
    assert "pre_limit:repeated_inspection" in diagnosis.evidence
    assert "pre_limit:model_tool_call_rejected" in diagnosis.evidence


def test_delivery_stall_diagnosis_is_separate_from_bounded_outcome() -> None:
    module = _load_benchmark_module()
    facts = module.RunFacts(
        model_alias="groq_gpt_oss_120b",
        turn_status="failed",
        valid_diff=False,
        hidden_acceptance_passed=False,
        stop_reason="delivery_stalled",
    )
    outcome = module.classify_outcome(facts)

    diagnosis = module.diagnose_run(
        facts,
        outcome,
        agent_stdout=(
            "Exploration limit reached after 20 consecutive inspection calls.\n"
            "停止原因: delivery_stalled\n"
        ),
        acceptance_stdout="4 failed, 30 passed",
        acceptance_stderr="",
    )

    assert outcome is module.RunOutcome.BOUNDED_FAILURE
    assert diagnosis.primary is module.DiagnosisCause.MODEL_CAPABILITY_GAP
    assert diagnosis.secondary == ()
    assert diagnosis.confidence >= 0.8
    assert "stop_reason:delivery_stalled" in diagnosis.evidence
    assert "inspection_calls:20" in diagnosis.evidence


def test_incomplete_model_response_is_diagnosed_from_public_harness_evidence() -> None:
    module = _load_benchmark_module()
    facts = module.RunFacts(
        model_alias="deepseek_v4_flash",
        turn_status="failed",
        valid_diff=False,
        hidden_acceptance_passed=False,
    )
    outcome = module.classify_outcome(facts)

    diagnosis = module.diagnose_run(
        facts,
        outcome,
        agent_stdout=(
            "[model] model_response_incomplete: "
            "Model response was incomplete: max_output_tokens.\n"
            "inspection_budget_exhausted\n"
        ),
        acceptance_stdout="1 failed",
        acceptance_stderr="",
    )

    assert outcome is module.RunOutcome.BOUNDED_FAILURE
    assert diagnosis.primary is module.DiagnosisCause.MODEL_CAPABILITY_GAP
    assert diagnosis.secondary == ()
    assert diagnosis.confidence >= 0.9
    assert "model_response:incomplete:max_output_tokens" in diagnosis.evidence


def test_false_completion_diagnosis_names_execution_closure() -> None:
    module = _load_benchmark_module()
    facts = module.RunFacts(
        model_alias="groq_gpt_oss_120b",
        turn_status="done",
        valid_diff=False,
        hidden_acceptance_passed=False,
    )
    outcome = module.classify_outcome(facts)

    diagnosis = module.diagnose_run(
        facts,
        outcome,
        agent_stdout="状态: done",
        acceptance_stdout="1 failed",
        acceptance_stderr="",
    )

    assert outcome is module.RunOutcome.FALSE_COMPLETION
    assert diagnosis.primary is module.DiagnosisCause.EXECUTION_CLOSURE
    assert diagnosis.confidence >= 0.9
    assert "workspace_diff:empty" in diagnosis.evidence
    assert "hidden_acceptance:failed" in diagnosis.evidence


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        (
            {
                "model_alias": "qwen3_5_9b_mlx_4bit",
                "turn_status": "done",
                "valid_diff": True,
                "hidden_acceptance_passed": True,
            },
            "passed",
        ),
        (
            {
                "model_alias": "qwen3_5_9b_mlx_4bit",
                "turn_status": "done",
                "valid_diff": True,
                "hidden_acceptance_passed": False,
            },
            "false_completion",
        ),
        (
            {
                "model_alias": "groq_gpt_oss_120b",
                "turn_status": "failed",
                "valid_diff": False,
                "hidden_acceptance_passed": False,
                "provider_error_code": "rate_limit",
            },
            "provider_limited",
        ),
        (
            {
                "model_alias": "kimi_cloud",
                "turn_status": "failed",
                "valid_diff": False,
                "hidden_acceptance_passed": False,
                "provider_error_code": "rate_limit",
            },
            "provider_limited",
        ),
        (
            {
                "model_alias": "qwen3_5_9b_mlx_4bit",
                "turn_status": "failed",
                "valid_diff": False,
                "hidden_acceptance_passed": False,
                "provider_error_code": "rate_limit",
            },
            "runtime_failed",
        ),
        (
            {
                "model_alias": "qwen3_5_9b_mlx_4bit",
                "turn_status": "done",
                "valid_diff": True,
                "hidden_acceptance_passed": True,
                "safety_violations": ("workspace_escape",),
            },
            "safety_violation",
        ),
    ],
)
def test_outcome_is_derived_from_evidence_with_blockers_first(
    facts: dict[str, object],
    expected: str,
) -> None:
    module = _load_benchmark_module()

    outcome = module.classify_outcome(module.RunFacts(**facts))

    assert outcome.value == expected


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_runtime_fingerprint_captures_tracked_and_untracked_state(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    repo = tmp_path / "runtime"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "benchmark@example.invalid")
    _git(repo, "config", "user.name", "Benchmark Fixture")
    source = repo / "runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "runtime.py")
    _git(repo, "commit", "-qm", "runtime source")

    clean = module.repository_state_fingerprint(repo)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    tracked_change = module.repository_state_fingerprint(repo)
    (repo / "new_runtime.py").write_text("NEW = True\n", encoding="utf-8")
    untracked_change = module.repository_state_fingerprint(repo)

    assert len(clean) == 64
    assert len({clean, tracked_change, untracked_change}) == 3


def test_run_task_uses_public_cli_and_archives_reproducible_evidence(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    repo = tmp_path / "history"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "benchmark@example.invalid")
    _git(repo, "config", "user.name", "Benchmark Fixture")
    (repo / "app.py").write_text("VALUE = 'broken'\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-qm", "fixture source")
    source_commit = _git(repo, "rev-parse", "HEAD")

    (repo / "hidden_test.py").write_text(
        "from pathlib import Path\n"
        "assert \"VALUE = 'fixed'\" in Path('app.py').read_text()\n",
        encoding="utf-8",
    )
    _git(repo, "add", "hidden_test.py")
    _git(repo, "commit", "-qm", "fixture target")
    target_commit = _git(repo, "rev-parse", "HEAD")

    fake_agent = tmp_path / "fake_agent.py"
    fake_agent.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "required = {'--model-session-path', '--checkpoint-db', '--non-interactive'}\n"
        "required.add('--require-workspace-change')\n"
        "required.add('--disable-workspace-mcp')\n"
        "assert required <= set(sys.argv)\n"
        "assert sys.argv[sys.argv.index('--model') + 1] == 'kimi_cloud'\n"
        "assert '--max-tokens-total' not in sys.argv\n"
        "Path('app.py').write_text(\"VALUE = 'fixed'\\n\", encoding='utf-8')\n"
        "print('Turn: fake-turn')\n"
        "print('状态: done')\n",
        encoding="utf-8",
    )

    task = _task_payload("history-task")
    task.update(
        {
            "source_commit": source_commit,
            "target_commit": target_commit,
            "acceptance_command": [sys.executable, "hidden_test.py"],
            "acceptance_files": ["hidden_test.py"],
            "setup_command": [],
            "allowed_paths": ["app.py"],
            "budget": {"max_turns": 12, "timeout_seconds": 30},
        }
    )
    manifest_path = tmp_path / "benchmark.json"
    _write_manifest(manifest_path, _manifest_payload(tasks=[task]))
    manifest = module.load_manifest(manifest_path)

    record = module.run_task(
        repository=repo,
        manifest=manifest,
        task=manifest.tasks[0],
        model_alias="kimi_cloud",
        agent_command=(sys.executable, str(fake_agent)),
        artifacts_root=tmp_path / "artifacts",
    )

    assert record.outcome is module.RunOutcome.PASSED
    assert record.turn_id == "fake-turn"
    assert "VALUE = 'fixed'" in record.diff
    assert record.hidden_acceptance_passed is True
    assert record.safety_violations == ()
    assert (record.artifact_dir / "result.json").is_file()
    assert (record.artifact_dir / "agent.stdout").is_file()
    assert (record.artifact_dir / "acceptance.stdout").is_file()
    payload = json.loads(
        (record.artifact_dir / "result.json").read_text(encoding="utf-8")
    )
    diff = (record.artifact_dir / "agent.diff").read_text(encoding="utf-8")
    assert payload["evidence"]["agent_diff_sha256"] == module._sha256(diff)
    assert payload["max_tokens_total"] is None
    assert payload["runtime_fingerprint"] == record.runtime_fingerprint
    assert len(record.runtime_fingerprint) == 64


def test_run_task_retries_only_a_durable_unknown_model_operation(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    repo = tmp_path / "history"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "benchmark@example.invalid")
    _git(repo, "config", "user.name", "Benchmark Fixture")
    (repo / "app.py").write_text("VALUE = 'broken'\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-qm", "fixture source")
    source_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "hidden_test.py").write_text(
        "from pathlib import Path\n"
        "assert \"VALUE = 'fixed'\" in Path('app.py').read_text()\n",
        encoding="utf-8",
    )
    _git(repo, "add", "hidden_test.py")
    _git(repo, "commit", "-qm", "fixture target")
    target_commit = _git(repo, "rev-parse", "HEAD")

    fake_agent = tmp_path / "retrying_agent.py"
    fake_agent.write_text(
        "from pathlib import Path\n"
        "import sqlite3\n"
        "import sys\n"
        "checkpoint = Path(sys.argv[sys.argv.index('--checkpoint-db') + 1])\n"
        "if sys.argv[1] == 'run':\n"
        "    with sqlite3.connect(checkpoint) as connection:\n"
        "        connection.execute('CREATE TABLE model_operations (operation_id TEXT, turn_id TEXT, status TEXT)')\n"
        "        connection.execute(\"INSERT INTO model_operations VALUES ('modelop-1', 'turn-retry', 'unknown')\")\n"
        "    print('Turn: turn-retry')\n"
        "    print('状态: paused')\n"
        "    raise SystemExit(2)\n"
        "assert sys.argv[1:4] == ['resume', 'turn-retry', '--action']\n"
        "assert sys.argv[4] == 'retry'\n"
        "with sqlite3.connect(checkpoint) as connection:\n"
        "    connection.execute(\"UPDATE model_operations SET status='completed'\")\n"
        "Path('app.py').write_text(\"VALUE = 'fixed'\\n\", encoding='utf-8')\n"
        "print('Turn: turn-retry')\n"
        "print('状态: done')\n",
        encoding="utf-8",
    )
    task = _task_payload("retry-task")
    task.update(
        {
            "source_commit": source_commit,
            "target_commit": target_commit,
            "acceptance_command": [sys.executable, "hidden_test.py"],
            "acceptance_files": ["hidden_test.py"],
            "setup_command": [],
            "allowed_paths": ["app.py"],
            "budget": {"max_turns": 12, "timeout_seconds": 30},
        }
    )
    manifest_path = tmp_path / "benchmark.json"
    _write_manifest(manifest_path, _manifest_payload(tasks=[task]))
    manifest = module.load_manifest(manifest_path)

    record = module.run_task(
        repository=repo,
        manifest=manifest,
        task=manifest.tasks[0],
        model_alias="kimi_cloud",
        agent_command=(sys.executable, str(fake_agent)),
        artifacts_root=tmp_path / "artifacts",
    )

    assert record.outcome is module.RunOutcome.PASSED
    assert record.turn_id == "turn-retry"
    assert record.turn_status == "done"
    assert "VALUE = 'fixed'" in record.diff
    stdout = (record.artifact_dir / "agent.stdout").read_text(encoding="utf-8")
    assert "状态: paused" in stdout
    assert "状态: done" in stdout


def test_command_cleanup_waits_for_background_process_group(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    command = (
        sys.executable,
        "-c",
        (
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(60)'], "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
            "stderr=subprocess.DEVNULL)\n"
            "print('spawned')\n"
        ),
    )

    result = module._run_command(
        command,
        cwd=tmp_path,
        timeout_seconds=10,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "spawned"
    assert result.leftover_processes is False


def test_run_command_streams_output_before_process_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_benchmark_module()
    release_path = tmp_path / "release"
    stdout_path = tmp_path / "agent.stdout"
    stderr_path = tmp_path / "agent.stderr"
    command = (
        sys.executable,
        "-c",
        (
            "import pathlib, sys, time\n"
            "release = pathlib.Path(sys.argv[1])\n"
            "print('first', flush=True)\n"
            "while not release.exists():\n"
            "    time.sleep(0.01)\n"
            "print('second', flush=True)\n"
        ),
        str(release_path),
    )
    result: list[object] = []

    worker = threading.Thread(
        target=lambda: result.append(
            module._run_command(
                command,
                cwd=tmp_path,
                timeout_seconds=10,
                live_log_paths=(stdout_path, stderr_path),
                progress_label="agent",
                heartbeat_seconds=0.05,
            )
        )
    )
    worker.start()
    deadline = time.monotonic() + 3
    while (
        not stdout_path.is_file()
        or "first" not in stdout_path.read_text(encoding="utf-8")
    ):
        assert time.monotonic() < deadline
        time.sleep(0.01)

    assert worker.is_alive()
    time.sleep(0.1)
    release_path.touch()
    worker.join(timeout=3)

    assert not worker.is_alive()
    assert len(result) == 1
    command_result = result[0]
    assert command_result.stdout == "first\nsecond\n"
    assert stdout_path.read_text(encoding="utf-8") == command_result.stdout
    assert stderr_path.read_text(encoding="utf-8") == ""
    terminal = capsys.readouterr()
    assert terminal.out == ""
    assert "[benchmark:agent:stdout] first" in terminal.err
    assert "[benchmark:agent] still running" in terminal.err


def test_run_command_redacts_live_terminal_and_artifact_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_benchmark_module()
    secret = "sk-benchmark-test-secret"
    monkeypatch.setenv("BENCHMARK_TEST_API_KEY", secret)
    stdout_path = tmp_path / "agent.stdout"
    stderr_path = tmp_path / "agent.stderr"

    result = module._run_command(
        (
            sys.executable,
            "-c",
            "import os; print(os.environ['BENCHMARK_TEST_API_KEY'], flush=True)",
        ),
        cwd=tmp_path,
        timeout_seconds=10,
        live_log_paths=(stdout_path, stderr_path),
        progress_label="agent",
    )

    terminal = capsys.readouterr()
    assert result.stdout == f"{secret}\n"
    assert secret not in terminal.err
    assert secret not in stdout_path.read_text(encoding="utf-8")
    assert "[REDACTED]" in terminal.err
    assert stdout_path.read_text(encoding="utf-8") == "[REDACTED]\n"


def test_run_command_interrupt_terminates_process_group(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    process_group_path = tmp_path / "process-group"
    stdout_path = tmp_path / "agent.stdout"
    stderr_path = tmp_path / "agent.stderr"

    def interrupt_when_started() -> None:
        deadline = time.monotonic() + 3
        while not process_group_path.is_file():
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGINT)

    interruptor = threading.Thread(target=interrupt_when_started)
    interruptor.start()
    process_group: int | None = None
    try:
        with pytest.raises(KeyboardInterrupt):
            module._run_command(
                (
                    sys.executable,
                    "-c",
                    (
                        "import os, pathlib, sys, time\n"
                        "pathlib.Path(sys.argv[1]).write_text(str(os.getpgrp()))\n"
                        "time.sleep(60)\n"
                    ),
                    str(process_group_path),
                ),
                cwd=tmp_path,
                timeout_seconds=120,
                live_log_paths=(stdout_path, stderr_path),
                progress_label="agent",
            )
        process_group = int(process_group_path.read_text(encoding="utf-8"))
        assert module._process_group_exists(process_group) is False
    finally:
        interruptor.join(timeout=3)
        if process_group is not None and module._process_group_exists(process_group):
            os.killpg(process_group, signal.SIGKILL)


def test_setup_failure_is_benchmark_invalid_not_runtime_failure(
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    repo = tmp_path / "history"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "benchmark@example.invalid")
    _git(repo, "config", "user.name", "Benchmark Fixture")
    (repo / "app.py").write_text("VALUE = 'broken'\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-qm", "fixture source")
    source_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "hidden_test.py").write_text("assert True\n", encoding="utf-8")
    _git(repo, "add", "hidden_test.py")
    _git(repo, "commit", "-qm", "fixture target")
    target_commit = _git(repo, "rev-parse", "HEAD")

    task = _task_payload("setup-failure")
    task.update(
        {
            "source_commit": source_commit,
            "target_commit": target_commit,
            "setup_command": [sys.executable, "-c", "raise SystemExit(7)"],
            "acceptance_command": [sys.executable, "hidden_test.py"],
            "acceptance_files": ["hidden_test.py"],
            "allowed_paths": ["app.py"],
            "budget": {"max_turns": 12, "timeout_seconds": 30},
        }
    )
    manifest_path = tmp_path / "benchmark.json"
    _write_manifest(manifest_path, _manifest_payload(tasks=[task]))
    manifest = module.load_manifest(manifest_path)

    record = module.run_task(
        repository=repo,
        manifest=manifest,
        task=manifest.tasks[0],
        model_alias="qwen3_5_9b_mlx_4bit",
        agent_command=(sys.executable, "-c", "raise AssertionError('must not run')"),
        artifacts_root=tmp_path / "artifacts",
    )

    assert record.outcome is module.RunOutcome.BENCHMARK_INVALID
    assert (
        record.diagnosis.primary
        is module.DiagnosisCause.BENCHMARK_ENVIRONMENT
    )
    result = json.loads(
        (record.artifact_dir / "result.json").read_text(encoding="utf-8")
    )
    assert result["diagnosis"]["evidence"] == [
        "benchmark_invalid:benchmark_setup_failed"
    ]


def test_secret_written_to_workspace_blocks_run_and_is_redacted_from_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_benchmark_module()
    repo = tmp_path / "history"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "benchmark@example.invalid")
    _git(repo, "config", "user.name", "Benchmark Fixture")
    (repo / "app.py").write_text("VALUE = 'broken'\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-qm", "fixture source")
    source_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "hidden_test.py").write_text("assert True\n", encoding="utf-8")
    _git(repo, "add", "hidden_test.py")
    _git(repo, "commit", "-qm", "fixture target")
    target_commit = _git(repo, "rev-parse", "HEAD")

    secret = "gsk_benchmark_secret_123456"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    fake_agent = tmp_path / "fake_agent.py"
    fake_agent.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path('app.py').write_text(os.environ['GROQ_API_KEY'] + '\\n')\n"
        "print('Turn: fake-turn')\n"
        "print('状态: done')\n",
        encoding="utf-8",
    )
    task = _task_payload("secret-leak")
    task.update(
        {
            "source_commit": source_commit,
            "target_commit": target_commit,
            "setup_command": [],
            "acceptance_command": [sys.executable, "hidden_test.py"],
            "acceptance_files": ["hidden_test.py"],
            "allowed_paths": ["app.py"],
            "budget": {"max_turns": 12, "timeout_seconds": 30},
        }
    )
    manifest_path = tmp_path / "benchmark.json"
    _write_manifest(manifest_path, _manifest_payload(tasks=[task]))
    manifest = module.load_manifest(manifest_path)

    record = module.run_task(
        repository=repo,
        manifest=manifest,
        task=manifest.tasks[0],
        model_alias="qwen3_5_9b_mlx_4bit",
        agent_command=(sys.executable, str(fake_agent)),
        artifacts_root=tmp_path / "artifacts",
    )

    assert record.outcome is module.RunOutcome.SAFETY_VIOLATION
    assert "secret_leak:agent.diff" in record.safety_violations
    assert secret not in record.diff
    assert "[REDACTED]" in record.diff
    for artifact in record.artifact_dir.iterdir():
        if artifact.is_file():
            assert secret.encode() not in artifact.read_bytes()


def test_benchmark_redacts_provider_credential_identifier() -> None:
    module = _load_benchmark_module()
    credential_id = "ak-provider-credential-123456"

    redacted = module._redact_secrets(
        f"rate limit for <{credential_id}>"
    )

    assert credential_id not in redacted
    assert redacted == "rate limit for <[REDACTED]>"

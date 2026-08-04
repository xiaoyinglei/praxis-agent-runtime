from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "render_model_quality_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location(
        "render_model_quality_report",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report_payload(tmp_path: Path) -> dict[str, object]:
    approval_observation = {
        "case_id": "approval_continue",
        "capability": "approval_continuation",
        "status": "done",
        "answer": "Updated approval.txt.",
        "tool_calls": [
            {
                "tool_call_id": "redacted-id-1",
                "tool_name": "read_file",
                "arguments": {"path": "input_files/approval.txt"},
                "is_error": False,
                "error_code": None,
            },
            {
                "tool_call_id": "redacted-id-2",
                "tool_name": "apply_patch",
                "arguments": {
                    "file_path": "input_files/approval.txt",
                    "old_string": "before_gate",
                    "new_string": "after_gate",
                },
                "is_error": False,
                "error_code": None,
            },
        ],
        "model_calls": 3,
        "input_tokens": 100,
        "output_tokens": 20,
        "latency_ms": 125.0,
        "tool_schema_bytes": 400,
        "approval_pause_observed": True,
        "approval_kind": "tool_approval",
        "approval_resumes": 1,
        "workspace_assertions_passed": True,
        "infrastructure_failure": False,
        "error": "",
    }
    inconclusive_observation = {
        "case_id": "provider_down",
        "capability": "failure_recovery",
        "status": "failed",
        "answer": None,
        "tool_calls": [],
        "approval_pause_observed": False,
        "approval_kind": None,
        "approval_resumes": 0,
        "workspace_assertions_passed": False,
        "infrastructure_failure": True,
        "stop_reason": "model_provider_failed",
        "diagnostic_error_types": ["RateLimitError"],
        "error": "",
    }
    return {
        "schema_version": 1,
        "status": "failed",
        "passed": False,
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "dirty": False,
        "suite_id": "agent-model-tool-quality-v1",
        "suite_revision": "suite_1234567890abcdef1234",
        "evaluator_version": "agent_model_quality_gate_v1",
        "measured_at": "2026-08-05T00:00:00+00:00",
        "baseline_path": str(tmp_path / "private" / "baseline.json"),
        "environment": {"OPENAI_API_KEY": "super-secret-value"},
        "headers": {"Authorization": "Bearer super-secret-value"},
        "case_metadata": [
            {
                "case_id": "approval_continue",
                "capability": "approval_continuation",
                "task": "Change before_gate to after_gate.",
                "workspace_before": {
                    "input_files/approval.txt": "before_gate\n"
                },
                "workspace_after": {
                    "input_files/approval.txt": "after_gate\n"
                },
            },
            {
                "case_id": "provider_down",
                "capability": "failure_recovery",
                "task": "Recover after a missing file.",
                "workspace_before": {},
                "workspace_after": {},
            },
        ],
        "models": [
            {
                "model_alias": "groq_gpt_oss_120b",
                "provider_model": "openai/gpt-oss-120b",
                "passed": False,
                "observed": {"task_success_rate": 0.8},
                "thresholds": {
                    "task_success_rate": {"direction": "min", "value": 1.0}
                },
                "failures": [
                    "task_success_rate: observed 0.8 < baseline floor 1.0"
                ],
            }
        ],
        "runs": [
            {
                "model_alias": "groq_gpt_oss_120b",
                "provider": "groq",
                "provider_model": "openai/gpt-oss-120b",
                "trial_count": 1,
                "trial_metrics": [{"task_success_rate": 0.8}],
                "trials": [
                    {
                        "trial": 1,
                        "metrics": {"task_success_rate": 0.8},
                        "cases": [
                            {
                                "observation": approval_observation,
                                "score": {
                                    "case_id": "approval_continue",
                                    "capability": "approval_continuation",
                                    "passed": False,
                                    "core_success": True,
                                    "capability_passed": False,
                                },
                            },
                            {
                                "observation": inconclusive_observation,
                                "score": {
                                    "case_id": "provider_down",
                                    "capability": "failure_recovery",
                                    "passed": False,
                                    "core_success": False,
                                    "capability_passed": False,
                                },
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_renderer_writes_only_reported_metrics_and_expanded_approval_evidence(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    report_path = tmp_path / "raw-report.json"
    benchmark_path = tmp_path / "docs" / "benchmark.md"
    run_record_path = tmp_path / "docs" / "runs" / "groq.md"
    report_path.write_text(
        json.dumps(_report_payload(tmp_path)),
        encoding="utf-8",
    )

    module.render_model_quality_report(
        report_path,
        benchmark_path=benchmark_path,
        run_record_path=run_record_path,
    )

    benchmark = benchmark_path.read_text(encoding="utf-8")
    run_record = run_record_path.read_text(encoding="utf-8")
    rendered = benchmark + run_record
    assert "Overall verdict: **FAILED**" in benchmark
    assert "source_commit: `" + "a" * 40 + "`" in benchmark
    assert "source_tree: `" + "b" * 40 + "`" in benchmark
    assert "dirty: `false`" in benchmark
    assert "suite_1234567890abcdef1234" in benchmark
    assert "agent_model_quality_gate_v1" in benchmark
    assert "task_success_rate" in benchmark
    assert "0.8" in benchmark
    assert "task_success_rate: observed 0.8 < baseline floor 1.0" in benchmark
    assert "provider_down" in benchmark
    assert "INCONCLUSIVE" in benchmark
    assert "accuracy" not in benchmark

    assert "Change before_gate to after_gate." in run_record
    assert "read_file" in run_record
    assert "apply_patch" in run_record
    assert "tool_approval" in run_record
    assert "Approval resumes: `1`" in run_record
    assert "-before_gate" in run_record
    assert "+after_gate" in run_record
    assert "Updated approval.txt." in run_record
    assert "Evaluator verdict: **FAILED**" in run_record

    assert str(tmp_path) not in rendered
    assert "super-secret-value" not in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert "Authorization" not in rendered
    assert "redacted-id" not in rendered


@pytest.mark.parametrize("dirty", [True, None, "false"])
def test_renderer_rejects_any_report_not_explicitly_clean(
    tmp_path: Path,
    dirty: object,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    payload["dirty"] = dirty
    report_path = tmp_path / "raw-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dirty must be false"):
        module.render_model_quality_report(
            report_path,
            benchmark_path=tmp_path / "benchmark.md",
            run_record_path=tmp_path / "run.md",
        )

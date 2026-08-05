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
        "stop_reason": "completed",
        "infrastructure_failure": False,
        "error": "",
    }
    failed_observation = {
        "case_id": "provider_down",
        "capability": "failure_recovery",
        "status": "failed",
        "answer": None,
        "tool_calls": [],
        "model_calls": 2,
        "input_tokens": 80,
        "output_tokens": 10,
        "latency_ms": 75.0,
        "tool_schema_bytes": 400,
        "approval_pause_observed": False,
        "approval_kind": None,
        "approval_resumes": 0,
        "workspace_assertions_passed": False,
        "infrastructure_failure": False,
        "stop_reason": "max_turns",
        "diagnostic_error_types": [],
        "error": "",
    }
    return {
        "schema_version": 1,
        "status": "failed",
        "passed": False,
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "source_unchanged": True,
        "dirty": False,
        "runtime_platform": {
            "os": "Darwin",
            "os_release": "25.6.0",
            "architecture": "arm64",
            "python_version": "3.12.11",
            "python_implementation": "CPython",
        },
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
                "status": "completed",
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
                                "observation": failed_observation,
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
    assert "source_unchanged: `true`" in benchmark
    assert "## Environment" in benchmark
    assert "| `os` | `Darwin` |" in benchmark
    assert "| `os_release` | `25.6.0` |" in benchmark
    assert "| `architecture` | `arm64` |" in benchmark
    assert "| `python_version` | `3.12.11` |" in benchmark
    assert "| `python_implementation` | `CPython` |" in benchmark
    assert "suite_1234567890abcdef1234" in benchmark
    assert "agent_model_quality_gate_v1" in benchmark
    assert "task_success_rate" in benchmark
    assert "0.8" in benchmark
    assert "task_success_rate: observed 0.8 < baseline floor 1.0" in benchmark
    assert "provider_down" in benchmark
    assert "- Provider: `groq`" in benchmark
    assert "- Provider model: `openai/gpt-oss-120b`" in benchmark
    assert "## Per-case usage" in benchmark
    assert "| `groq_gpt_oss_120b` | `1` | `approval_continue` | `2` | `3` | `125.0` | `100` | `20` |" in benchmark
    assert "## 30-task coding-agent protocol" in benchmark
    assert "Manifest status: **validated only**" in benchmark
    assert "not run as this release gate" in benchmark
    assert "INCONCLUSIVE" not in benchmark
    assert "accuracy" not in benchmark

    assert "Change before_gate to after_gate." in run_record
    assert "read_file" in run_record
    assert "apply_patch" in run_record
    assert "tool_approval" in run_record
    assert "Approval resumes: `1`" in run_record
    assert "Stop reason: `\"completed\"`" in run_record
    assert "Verification — workspace assertions passed: `true`" in run_record
    assert "Tool calls: `2`" in run_record
    assert "Model calls: `3`" in run_record
    assert "Latency ms: `125.0`" in run_record
    assert "Input tokens: `100`" in run_record
    assert "Output tokens: `20`" in run_record
    assert "validated fixture before/after assertion contract" in run_record
    assert "not a captured filesystem diff" in run_record
    assert "-before_gate" in run_record
    assert "+after_gate" in run_record
    assert "Updated approval.txt." in run_record
    assert "Evaluator verdict: **FAILED**" in run_record

    assert str(tmp_path) not in rendered
    assert "super-secret-value" not in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert "Authorization" not in rendered
    assert "redacted-id" not in rendered


@pytest.mark.parametrize("missing", ["runtime_platform", "source_unchanged"])
def test_renderer_requires_platform_and_source_consistency_metadata(
    tmp_path: Path,
    missing: str,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    del payload[missing]
    report_path = tmp_path / "incomplete-provenance.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=missing):
        module.render_model_quality_report(
            report_path,
            benchmark_path=tmp_path / "benchmark.md",
            run_record_path=tmp_path / "run.md",
        )


def test_renderer_redacts_sensitive_name_tokens_without_redacting_monkey() -> None:
    module = _load_report_module()
    payload = {
        "monkey": "banana",
        "nested": [
            {
                "apiKey": "api-key-value",
                "x_api_key": "x-api-key-value",
                "access_token": "access-token-value",
                "client_secret": "client-secret-value",
                "auth": "auth-value",
                "cookie": "cookie-value",
                "deeper": [{"monkey": "plantain"}],
            }
        ],
    }

    redacted = module._redacted_mapping(payload)

    assert redacted["monkey"] == "banana"
    nested = redacted["nested"][0]
    assert nested["apiKey"] == "[REDACTED]"
    assert nested["x_api_key"] == "[REDACTED]"
    assert nested["access_token"] == "[REDACTED]"
    assert nested["client_secret"] == "[REDACTED]"
    assert nested["auth"] == "[REDACTED]"
    assert nested["cookie"] == "[REDACTED]"
    assert nested["deeper"][0]["monkey"] == "plantain"


def test_renderer_does_not_turn_partial_approval_infrastructure_evidence_into_success(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    payload["status"] = "inconclusive"
    payload["passed"] = None
    model = payload["models"][0]
    model["status"] = "inconclusive"
    model["passed"] = None
    model["observed"] = {}
    model["thresholds"] = {}
    model["failures"] = []
    infrastructure_failure = {
        "case_id": "approval_continue",
        "stop_reason": "model_provider_failed",
        "diagnostic_error_types": ["RateLimitError"],
    }
    model["infrastructure_failure"] = infrastructure_failure
    payload["runs"][0]["status"] = "inconclusive"
    payload["runs"][0]["infrastructure_failure"] = infrastructure_failure
    trial = payload["runs"][0]["trials"][0]
    approval_case = trial["cases"][0]
    observation = approval_case["observation"]
    observation["status"] = "failed"
    observation["answer"] = "UNTRUSTED_FINAL_SENTINEL"
    observation["tool_calls"] = [
        {
            "tool_call_id": "partial-read",
            "tool_name": "read_file",
            "arguments": {"path": "input_files/approval.txt"},
            "is_error": False,
            "error_code": None,
        }
    ]
    observation["approval_pause_observed"] = False
    observation["approval_kind"] = None
    observation["approval_resumes"] = 0
    observation["workspace_assertions_passed"] = False
    observation["infrastructure_failure"] = True
    observation["stop_reason"] = "model_provider_failed"
    observation["diagnostic_error_types"] = ["RateLimitError"]
    approval_case["score"] = {
        "case_id": "approval_continue",
        "capability": "approval_continuation",
        "passed": None,
        "core_success": None,
        "capability_passed": None,
        "inconclusive": True,
    }
    trial["cases"] = [approval_case]
    payload["runs"][0]["trials"] = [trial]
    report_path = tmp_path / "partial-approval-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"

    module.render_model_quality_report(
        report_path,
        benchmark_path=benchmark_path,
        run_record_path=run_record_path,
    )

    run_record = run_record_path.read_text(encoding="utf-8")
    assert "read_file" in run_record
    assert "Approval was not reached." in run_record
    assert "Approval pause observed: `false`" in run_record
    assert "Approval kind: `null`" in run_record
    assert "Approval resumes: `0`" in run_record
    assert "model_provider_failed" in run_record
    assert "RateLimitError" in run_record
    assert "Evaluator verdict: **INCONCLUSIVE**" in run_record
    assert "-before_gate" not in run_record
    assert "+after_gate" not in run_record
    assert "UNTRUSTED_FINAL_SENTINEL" not in run_record


def test_renderer_preserves_observed_approval_resume_before_infrastructure_failure(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    payload["status"] = "inconclusive"
    payload["passed"] = None
    model = payload["models"][0]
    model["status"] = "inconclusive"
    model["passed"] = None
    model["observed"] = {}
    model["thresholds"] = {}
    model["failures"] = []
    infrastructure_failure = {
        "case_id": "approval_continue",
        "stop_reason": "model_provider_failed",
        "diagnostic_error_types": ["RateLimitError"],
    }
    model["infrastructure_failure"] = infrastructure_failure
    run = payload["runs"][0]
    run["status"] = "inconclusive"
    run["infrastructure_failure"] = infrastructure_failure
    trial = run["trials"][0]
    approval_case = trial["cases"][0]
    observation = approval_case["observation"]
    observation["status"] = "failed"
    observation["answer"] = "UNTRUSTED_FINAL_SENTINEL"
    observation["approval_pause_observed"] = True
    observation["approval_kind"] = "tool_approval"
    observation["approval_resumes"] = 1
    observation["workspace_assertions_passed"] = False
    observation["infrastructure_failure"] = True
    observation["stop_reason"] = "model_provider_failed"
    observation["diagnostic_error_types"] = ["RateLimitError"]
    approval_case["score"] = {
        "case_id": "approval_continue",
        "capability": "approval_continuation",
        "passed": None,
        "core_success": None,
        "capability_passed": None,
        "inconclusive": True,
    }
    trial["cases"] = [approval_case]
    report_path = tmp_path / "resumed-then-infrastructure-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"

    module.render_model_quality_report(
        report_path,
        benchmark_path=benchmark_path,
        run_record_path=run_record_path,
    )

    run_record = run_record_path.read_text(encoding="utf-8")
    assert "Approval pause observed: `true`" in run_record
    assert "Approval kind: `\"tool_approval\"`" in run_record
    assert "Approval resumes: `1`" in run_record
    assert "Approval pause and resume were observed before infrastructure failure." in run_record
    assert "model_provider_failed" in run_record
    assert "RateLimitError" in run_record
    assert "Evaluator verdict: **INCONCLUSIVE**" in run_record
    assert "-before_gate" not in run_record
    assert "+after_gate" not in run_record
    assert "Workspace assertions passed: `false`" in run_record
    assert "### Observed answer before infrastructure failure" in run_record
    assert "UNTRUSTED_FINAL_SENTINEL" in run_record
    assert "### Final answer" not in run_record


def test_renderer_shows_completed_evidence_stages_before_infrastructure_failure(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    payload["status"] = "inconclusive"
    payload["passed"] = None
    infrastructure_failure = {
        "case_id": "approval_continue",
        "stop_reason": "model_provider_failed",
        "diagnostic_error_types": ["RateLimitError"],
    }
    model = payload["models"][0]
    model.update(
        {
            "status": "inconclusive",
            "passed": None,
            "observed": {},
            "thresholds": {},
            "failures": [],
            "infrastructure_failure": infrastructure_failure,
        }
    )
    run = payload["runs"][0]
    run["status"] = "inconclusive"
    run["infrastructure_failure"] = infrastructure_failure
    trial = run["trials"][0]
    approval_case = trial["cases"][0]
    observation = approval_case["observation"]
    observation.update(
        {
            "status": "failed",
            "answer": "ACTUAL_ANSWER_BEFORE_PROVIDER_FAILURE",
            "approval_pause_observed": True,
            "approval_kind": "tool_approval",
            "approval_resumes": 1,
            "workspace_assertions_passed": True,
            "infrastructure_failure": True,
            "stop_reason": "model_provider_failed",
            "diagnostic_error_types": ["RateLimitError"],
        }
    )
    approval_case["score"] = {
        "case_id": "approval_continue",
        "capability": "approval_continuation",
        "passed": None,
        "core_success": None,
        "capability_passed": None,
        "inconclusive": True,
    }
    trial["cases"] = [approval_case]
    report_path = tmp_path / "completed-stages-before-infrastructure.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"

    module.render_model_quality_report(
        report_path,
        benchmark_path=benchmark_path,
        run_record_path=run_record_path,
    )

    run_record = run_record_path.read_text(encoding="utf-8")
    assert "Approval pause and resume were observed before infrastructure failure." in run_record
    assert "### Fixture workspace assertion contract" in run_record
    assert "-before_gate" in run_record
    assert "+after_gate" in run_record
    assert "Workspace assertions passed: `true`" in run_record
    assert "### Observed answer before infrastructure failure" in run_record
    assert "ACTUAL_ANSWER_BEFORE_PROVIDER_FAILURE" in run_record
    assert "### Final answer" not in run_record


@pytest.mark.parametrize(
    "inconsistency",
    ["passed_false", "passed_inconclusive", "alias_mismatch"],
)
def test_renderer_rejects_internally_inconsistent_reports_before_writing(
    tmp_path: Path,
    inconsistency: str,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    if inconsistency == "passed_false":
        payload["status"] = "passed"
    elif inconsistency == "passed_inconclusive":
        payload["status"] = "passed"
        payload["passed"] = True
        infrastructure_failure = {
            "case_id": "approval_continue",
            "stop_reason": "model_provider_failed",
            "diagnostic_error_types": ["RateLimitError"],
        }
        model = payload["models"][0]
        model["status"] = "inconclusive"
        model["passed"] = None
        model["infrastructure_failure"] = infrastructure_failure
        run = payload["runs"][0]
        run["status"] = "inconclusive"
        run["infrastructure_failure"] = infrastructure_failure
    else:
        payload["runs"][0]["model_alias"] = "different_alias"
    report_path = tmp_path / "inconsistent.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"

    with pytest.raises(ValueError, match="inconsistent"):
        module.render_model_quality_report(
            report_path,
            benchmark_path=benchmark_path,
            run_record_path=run_record_path,
        )

    assert not benchmark_path.exists()
    assert not run_record_path.exists()


@pytest.mark.parametrize(
    "inconsistency",
    [
        "duplicate_metadata",
        "score_case_mismatch",
        "completed_infrastructure_case",
        "inconclusive_without_infrastructure_case",
    ],
)
def test_renderer_rejects_internally_inconsistent_case_evidence_before_writing(
    tmp_path: Path,
    inconsistency: str,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    trial = payload["runs"][0]["trials"][0]
    first_case = trial["cases"][0]
    if inconsistency == "duplicate_metadata":
        payload["case_metadata"].append(dict(payload["case_metadata"][0]))
    elif inconsistency == "score_case_mismatch":
        first_case["score"]["case_id"] = "different_case"
    elif inconsistency == "completed_infrastructure_case":
        first_case["observation"]["infrastructure_failure"] = True
        first_case["score"]["passed"] = None
    else:
        payload["status"] = "inconclusive"
        payload["passed"] = None
        infrastructure_failure = {
            "case_id": "approval_continue",
            "stop_reason": "model_provider_failed",
            "diagnostic_error_types": ["RateLimitError"],
        }
        model = payload["models"][0]
        model["status"] = "inconclusive"
        model["passed"] = None
        model["infrastructure_failure"] = infrastructure_failure
        run = payload["runs"][0]
        run["status"] = "inconclusive"
        run["infrastructure_failure"] = infrastructure_failure
    report_path = tmp_path / "inconsistent-case.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"

    with pytest.raises(ValueError, match="inconsistent|duplicate"):
        module.render_model_quality_report(
            report_path,
            benchmark_path=benchmark_path,
            run_record_path=run_record_path,
        )

    assert not benchmark_path.exists()
    assert not run_record_path.exists()


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

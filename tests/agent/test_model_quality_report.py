from __future__ import annotations

import copy
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


def test_renderer_defaults_to_deepseek_v4_flash_run_record() -> None:
    module = _load_report_module()

    assert module.DEFAULT_RUN_RECORD_PATH.name == "deepseek-v4-flash.md"


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
                "structured_output": {"content": "before_gate\n"},
                "is_error": False,
                "error_code": None,
                "error_message": None,
                "retryable": False,
                "truncated": False,
                "latency_ms": None,
            },
            {
                "tool_call_id": "redacted-id-2",
                "tool_name": "apply_patch",
                "arguments": {
                    "file_path": "input_files/approval.txt",
                    "old_string": "before_gate",
                    "new_string": "after_gate",
                },
                "structured_output": {"changed": True},
                "is_error": False,
                "error_code": None,
                "error_message": None,
                "retryable": False,
                "truncated": False,
                "latency_ms": 12.5,
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


def _render_payload(
    module: object,
    payload: dict[str, object],
    tmp_path: Path,
    *,
    name: str,
) -> tuple[str, str]:
    report_path = tmp_path / f"{name}.json"
    benchmark_path = tmp_path / f"{name}-benchmark.md"
    run_record_path = tmp_path / f"{name}-run.md"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    module.render_model_quality_report(
        report_path,
        benchmark_path=benchmark_path,
        run_record_path=run_record_path,
    )
    return (
        benchmark_path.read_text(encoding="utf-8"),
        run_record_path.read_text(encoding="utf-8"),
    )


def _v3_paused_payload(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload = _report_payload(tmp_path)
    payload["evaluator_version"] = "agent_model_quality_gate_v3"
    cases = payload["runs"][0]["trials"][0]["cases"]
    for case in cases:
        case["observation"]["runtime_input_namespace"] = "turn-123"
    approval_case = cases[0]
    observation = approval_case["observation"]
    observation.update(
        status="paused",
        answer=None,
        stop_reason="approval_required",
        final_pause_request_kind="tool_approval",
        final_pause_reason="Allow run_command for verification?",
        final_pause_tool_names=["run_command"],
    )
    approval_case["score"].update(
        passed=False,
        core_success=True,
        capability_passed=False,
    )
    return payload, observation


def test_renderer_accepts_old_v3_artifact_without_final_pause_fields(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    payload["evaluator_version"] = "agent_model_quality_gate_v3"
    for case in payload["runs"][0]["trials"][0]["cases"]:
        case["observation"]["runtime_input_namespace"] = "turn-123"

    _benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name="old-v3-without-final-pause",
    )

    assert "### Final pause" not in run_record


def test_renderer_accepts_empty_final_pause_fields_on_non_paused_observations(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    payload["evaluator_version"] = "agent_model_quality_gate_v3"
    for case in payload["runs"][0]["trials"][0]["cases"]:
        observation = case["observation"]
        observation.update(
            runtime_input_namespace="turn-123",
            final_pause_request_kind=None,
            final_pause_reason=None,
            final_pause_tool_names=[],
        )

    _benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name="current-v3-empty-final-pause",
    )

    assert "### Final pause" not in run_record


def test_renderer_shows_bounded_final_pause_evidence(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload, _observation = _v3_paused_payload(tmp_path)

    _benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name="final-pause-evidence",
    )

    assert "### Final pause" in run_record
    assert '- Request kind: `"tool_approval"`' in run_record
    assert '- Reason: `"Allow run_command for verification?"`' in run_record
    assert '- Pending tools: `["run_command"]`' in run_record
    assert "No answer was reported before final pause." in run_record
    assert "### Final answer" not in run_record


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        (
            "final_pause_request_kind",
            123,
            "final_pause_request_kind must be text or null",
        ),
        (
            "final_pause_reason",
            123,
            "final_pause_reason must be text or null",
        ),
        (
            "final_pause_reason",
            "x" * 2001,
            "final_pause_reason exceeds 2000 characters",
        ),
        (
            "final_pause_tool_names",
            "run_command",
            "final_pause_tool_names must be a sequence",
        ),
        (
            "final_pause_tool_names",
            ["run_command", 123],
            "final_pause_tool_names entries must be text",
        ),
        (
            "final_pause_tool_names",
            [f"tool_{index}" for index in range(33)],
            "final_pause_tool_names exceeds 32 entries",
        ),
    ],
)
def test_renderer_rejects_invalid_final_pause_evidence(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    module = _load_report_module()
    payload, observation = _v3_paused_payload(tmp_path)
    observation[field] = invalid
    report_path = tmp_path / f"invalid-{field}.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.render_model_quality_report(
            report_path,
            benchmark_path=tmp_path / "benchmark.md",
            run_record_path=tmp_path / "run.md",
        )


def test_renderer_rejects_final_pause_evidence_on_non_paused_observation(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload, observation = _v3_paused_payload(tmp_path)
    observation["status"] = "done"
    report_path = tmp_path / "final-pause-on-done.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="final pause evidence requires paused status",
    ):
        module.render_model_quality_report(
            report_path,
            benchmark_path=tmp_path / "benchmark.md",
            run_record_path=tmp_path / "run.md",
        )


def test_renderer_rejects_final_pause_tools_without_request_kind(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload, observation = _v3_paused_payload(tmp_path)
    observation["final_pause_request_kind"] = None
    report_path = tmp_path / "final-pause-tools-without-kind.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="final_pause_tool_names require a request kind",
    ):
        module.render_model_quality_report(
            report_path,
            benchmark_path=tmp_path / "benchmark.md",
            run_record_path=tmp_path / "run.md",
        )


def test_renderer_redacts_absolute_paths_in_final_pause_reason(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload, observation = _v3_paused_payload(tmp_path)
    posix_path = "/Users/private/workspace/approval.txt"
    windows_path = r"C:\private\workspace\approval.txt"
    observation["final_pause_reason"] = (
        f"Inspect {posix_path} and {windows_path} before continuing."
    )

    _benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name="redacted-final-pause",
    )

    assert posix_path not in run_record
    assert windows_path not in run_record
    assert run_record.count("[REDACTED_ABSOLUTE_PATH]") >= 2


def _make_inconclusive_with_completed_prefix(
    payload: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    infrastructure_failure = {
        "case_id": "provider_down",
        "stop_reason": "model_provider_failed",
        "diagnostic_error_types": ["RateLimitError"],
    }
    model = payload["models"][0]
    model.update(
        status="inconclusive",
        passed=None,
        observed={},
        thresholds={},
        failures=[],
        infrastructure_failure=infrastructure_failure,
    )
    run = payload["runs"][0]
    run["status"] = "inconclusive"
    run["infrastructure_failure"] = infrastructure_failure
    prefix_case, infrastructure_case = run["trials"][0]["cases"]
    infrastructure_observation = infrastructure_case["observation"]
    infrastructure_observation.update(
        status="failed",
        infrastructure_failure=True,
        stop_reason="model_provider_failed",
        diagnostic_error_types=["RateLimitError"],
    )
    infrastructure_case["score"].update(
        passed=None,
        core_success=None,
        capability_passed=None,
        inconclusive=True,
    )
    payload["status"] = "inconclusive"
    payload["passed"] = None
    return prefix_case, infrastructure_case


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
    assert "- Infrastructure status: **CONCLUSIVE**" in benchmark
    assert "## Per-case usage" in benchmark
    assert "| `groq_gpt_oss_120b` | `1` | `approval_continue` | `2` | `3` | `125.0` | `100` | `20` |" in benchmark
    assert "## 30-task coding-agent protocol" in benchmark
    assert "Manifest status: **validated only**" in benchmark
    assert "not run as this release gate" in benchmark
    assert "INCONCLUSIVE" not in benchmark
    assert "accuracy" not in benchmark

    assert "Change before_gate to after_gate." in run_record
    assert "## Model identity and infrastructure" in run_record
    assert ("| `groq_gpt_oss_120b` | `groq` | `openai/gpt-oss-120b` | **CONCLUSIVE** |") in run_record
    assert "read_file" in run_record
    assert "apply_patch" in run_record
    assert "Arguments:" in run_record
    assert "Result:" in run_record
    assert '"content":"before_gate\\n"' in run_record
    assert "Error code: `null`" in run_record
    assert "Error message: `null`" in run_record
    assert "Retryable: `false`" in run_record
    assert "Truncated: `false`" in run_record
    assert "Tool latency ms: `12.5`" in run_record
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


def test_completed_resumed_case_with_failed_workspace_assertions_never_renders_expected_diff(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    approval_case = payload["runs"][0]["trials"][0]["cases"][0]
    observation = approval_case["observation"]
    observation["workspace_assertions_passed"] = False
    observation["answer"] = "OBSERVED_FINAL_AFTER_FAILED_ASSERTION"
    approval_case["score"].update(
        passed=False,
        core_success=False,
        capability_passed=False,
    )

    _benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name="failed-workspace-assertion",
    )

    assert "### Workspace assertion evidence" in run_record
    assert "Workspace assertions passed: `false`" in run_record
    assert "expected before/after values are not rendered as observed evidence" in run_record
    assert "### Fixture workspace assertion contract" not in run_record
    assert "validated fixture before/after assertion contract" not in run_record
    assert "-before_gate" not in run_record
    assert "+after_gate" not in run_record
    assert "### Observed final answer" in run_record
    assert "OBSERVED_FINAL_AFTER_FAILED_ASSERTION" in run_record
    assert "Evaluator verdict: **FAILED**" in run_record


@pytest.mark.parametrize(
    ("state", "expected_message"),
    [
        ("approval_not_reached", "Approval was not reached."),
        (
            "resume_not_observed",
            "Approval pause was observed, but approval resume was not observed.",
        ),
    ],
)
def test_incomplete_approval_states_never_render_expected_diff_or_final_answer(
    tmp_path: Path,
    state: str,
    expected_message: str,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    approval_case = payload["runs"][0]["trials"][0]["cases"][0]
    observation = approval_case["observation"]
    observation["answer"] = "UNTRUSTED_INCOMPLETE_APPROVAL_ANSWER"
    observation["workspace_assertions_passed"] = False
    if state == "approval_not_reached":
        observation["approval_pause_observed"] = False
        observation["approval_kind"] = None
    observation["approval_resumes"] = 0
    approval_case["score"].update(
        passed=False,
        core_success=False,
        capability_passed=False,
    )

    _benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name=state,
    )

    assert expected_message in run_record
    assert "### Fixture workspace assertion contract" not in run_record
    assert "-before_gate" not in run_record
    assert "+after_gate" not in run_record
    assert "UNTRUSTED_INCOMPLETE_APPROVAL_ANSWER" not in run_record
    assert "### Final answer" not in run_record
    assert "Evaluator verdict: **FAILED**" in run_record


def test_completed_passing_run_is_conclusive_not_inconclusive(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    payload["status"] = "passed"
    payload["passed"] = True
    model = payload["models"][0]
    model["passed"] = True
    model["observed"] = {"task_success_rate": 1.0}
    model["failures"] = []
    recovered_observation = payload["runs"][0]["trials"][0]["cases"][1][
        "observation"
    ]
    recovered_observation["status"] = "done"
    recovered_observation["workspace_assertions_passed"] = True
    for case in payload["runs"][0]["trials"][0]["cases"]:
        case["score"].update(
            passed=True,
            core_success=True,
            capability_passed=True,
        )

    benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name="passing-conclusive",
    )

    assert "Overall verdict: **PASSED**" in benchmark
    assert "Overall verdict: **PASSED**" in run_record
    assert "Infrastructure status: **CONCLUSIVE**" in benchmark
    assert ("| `groq_gpt_oss_120b` | `groq` | `openai/gpt-oss-120b` | **CONCLUSIVE** |") in run_record
    assert "Infrastructure status: **INCONCLUSIVE**" not in run_record


def test_run_record_lists_every_reported_model_identity_and_infrastructure_status(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    second_model = copy.deepcopy(payload["models"][0])
    second_model["model_alias"] = "qwen3_5_9b_mlx_4bit"
    second_model["provider_model"] = "mlx-community/Qwen3.5-9B-4bit"
    second_run = copy.deepcopy(payload["runs"][0])
    second_run["model_alias"] = "qwen3_5_9b_mlx_4bit"
    second_run["provider"] = "local_mlx_chat_8080"
    second_run["provider_model"] = "mlx-community/Qwen3.5-9B-4bit"
    payload["models"].append(second_model)
    payload["runs"].append(second_run)

    _benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name="multiple-model-identities",
    )

    assert ("| `groq_gpt_oss_120b` | `groq` | `openai/gpt-oss-120b` | **CONCLUSIVE** |") in run_record
    assert (
        "| `qwen3_5_9b_mlx_4bit` | `local_mlx_chat_8080` | `mlx-community/Qwen3.5-9B-4bit` | **CONCLUSIVE** |"
    ) in run_record


def test_renderer_bounds_and_redacts_structured_tool_results_and_errors(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    tool_call = payload["runs"][0]["trials"][0]["cases"][0]["observation"]["tool_calls"][0]
    oversized = "visible-prefix-" + ("x" * 5000)
    tool_call.update(
        structured_output={
            "normal_result": "VISIBLE_NORMAL_RESULT",
            "oversized": oversized,
            "nested": {
                "api_token": "renderer-secret-value",
                "posix": "/Users/private/workspace/result.txt",
                "windows": r"C:\private\workspace\result.txt",
                "unc": r"\\private-server\workspace\result.txt",
            },
        },
        is_error=True,
        error_code="runner_failed",
        error_message=(
            "failure at /Users/private/workspace/result.txt, "
            r"C:\private\workspace\result.txt, "
            r"\\private-server\workspace\result.txt"
        ),
        retryable=True,
        truncated=True,
        latency_ms=None,
    )

    _benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name="bounded-redacted-tool-result",
    )

    assert "VISIBLE_NORMAL_RESULT" in run_record
    assert "[report truncated]" in run_record
    assert oversized not in run_record
    assert "renderer-secret-value" not in run_record
    assert "/Users/private/workspace/result.txt" not in run_record
    assert r"C:\private\workspace\result.txt" not in run_record
    assert r"\\private-server\workspace\result.txt" not in run_record
    assert "runner_failed" in run_record
    assert "Retryable: `true`" in run_record
    assert "Truncated: `true`" in run_record
    assert "Tool latency ms: `null`" in run_record
    assert "redacted-id-1" not in run_record


def test_generated_pages_preserve_the_pending_evidence_contract(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    model = payload["models"][0]
    model["observed"].update(
        approval_continuation_rate=0.0,
        mean_model_calls_per_case=2.5,
    )
    model["thresholds"].update(
        approval_continuation_rate={"direction": "min", "value": 1.0},
        mean_model_calls_per_case={"direction": "max", "value": 3.0},
    )

    benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name="pending-contract-coverage",
    )

    for required in (
        "Overall verdict: **FAILED**",
        "## Evidence provenance",
        "source_commit",
        "source_tree",
        "source_unchanged",
        "dirty",
        "suite_id",
        "suite_revision",
        "evaluator_version",
        "measured_at",
        "Redacted raw report",
        "## Environment",
        "## Model results",
        "Provider model",
        "Infrastructure status: **CONCLUSIVE**",
        "Trials: `1`",
        "task_success_rate",
        "approval_continuation_rate",
        "mean_model_calls_per_case",
        "Reported failures",
        "## Case results",
        "approval_continuation",
        "## Per-case usage",
        "Tool calls",
        "Model calls",
        "Latency ms",
        "Input tokens",
        "Output tokens",
        "Total tokens",
        "## 30-task coding-agent protocol",
        "Manifest status: **validated only**",
        "not run as this release gate",
    ):
        assert required in benchmark

    for required in (
        "Overall verdict: **FAILED**",
        "source_commit",
        "source_tree",
        "source_unchanged",
        "dirty",
        "suite_id",
        "suite_revision",
        "evaluator_version",
        "measured_at",
        "Redacted raw report",
        "## Model identity and infrastructure",
        "groq_gpt_oss_120b",
        "groq",
        "openai/gpt-oss-120b",
        "CONCLUSIVE",
        "## Task",
        "Change before_gate to after_gate.",
        "## `groq_gpt_oss_120b` trial 1",
        "### Tool trace",
        "Arguments:",
        "Result:",
        "Error code:",
        "Error message:",
        "Retryable:",
        "Truncated:",
        "Tool latency ms:",
        "### Runtime observation",
        "Stop reason:",
        "workspace assertions passed",
        "Tool calls:",
        "Model calls:",
        "Latency ms:",
        "Input tokens:",
        "Output tokens:",
        "Total tokens:",
        "### Approval and resume",
        "Approval pause observed: `true`",
        'Approval kind: `"tool_approval"`',
        "Approval resumes: `1`",
        "### Fixture workspace assertion contract",
        "not a captured filesystem diff",
        "### Final answer",
        "Updated approval.txt.",
        "### Evaluator verdict",
        "Evaluator verdict: **FAILED**",
    ):
        assert required in run_record


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("error_message", 123),
        ("retryable", "yes"),
        ("truncated", 0),
        ("latency_ms", -1.0),
    ],
)
def test_renderer_rejects_invalid_bounded_tool_result_evidence(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    tool_call = payload["runs"][0]["trials"][0]["cases"][0]["observation"]["tool_calls"][0]
    tool_call[field] = invalid
    report_path = tmp_path / f"invalid-tool-{field}.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        module.render_model_quality_report(
            report_path,
            benchmark_path=tmp_path / "benchmark.md",
            run_record_path=tmp_path / "run.md",
        )


def test_renderer_rejects_passed_approval_case_without_observed_pause_before_writing(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    approval_case = payload["runs"][0]["trials"][0]["cases"][0]
    observation = approval_case["observation"]
    observation["approval_pause_observed"] = False
    observation["approval_kind"] = None
    approval_case["score"].update(
        passed=True,
        core_success=True,
        capability_passed=True,
    )
    report_path = tmp_path / "passed-without-approval-pause.json"
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        module.render_model_quality_report(
            report_path,
            benchmark_path=benchmark_path,
            run_record_path=run_record_path,
        )

    assert not benchmark_path.exists()
    assert not run_record_path.exists()


def test_renderer_rejects_passed_approval_case_without_resume_before_writing(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    approval_case = payload["runs"][0]["trials"][0]["cases"][0]
    observation = approval_case["observation"]
    observation["approval_resumes"] = 0
    approval_case["score"].update(
        passed=True,
        core_success=True,
        capability_passed=True,
    )
    report_path = tmp_path / "passed-without-approval-resume.json"
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        module.render_model_quality_report(
            report_path,
            benchmark_path=benchmark_path,
            run_record_path=run_record_path,
        )

    assert not benchmark_path.exists()
    assert not run_record_path.exists()


def test_renderer_accepts_multiple_resumes_for_a_passing_approval_chain(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    payload["evaluator_version"] = "agent_model_quality_gate_v2"
    for case in payload["runs"][0]["trials"][0]["cases"]:
        case["observation"]["runtime_input_namespace"] = "turn-123"
    approval_case = payload["runs"][0]["trials"][0]["cases"][0]
    observation = approval_case["observation"]
    observation["approval_resumes"] = 2
    approval_case["score"].update(
        passed=True,
        core_success=True,
        capability_passed=True,
    )

    _benchmark, run_record = _render_payload(
        module,
        payload,
        tmp_path,
        name="multiple-approval-resumes",
    )

    assert "Approval resumes: `2`" in run_record
    assert "Evaluator verdict: **PASSED**" in run_record


@pytest.mark.parametrize(
    ("evaluator_version", "approval_resumes"),
    [
        ("agent_model_quality_gate_v1", 2),
        ("agent_model_quality_gate_v2", 6),
        ("agent_model_quality_gate_v3", 2),
    ],
)
def test_renderer_rejects_approval_resumes_outside_the_evaluator_contract(
    tmp_path: Path,
    evaluator_version: str,
    approval_resumes: int,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    payload["evaluator_version"] = evaluator_version
    approval_case = payload["runs"][0]["trials"][0]["cases"][0]
    observation = approval_case["observation"]
    observation["approval_resumes"] = approval_resumes
    if evaluator_version in {
        "agent_model_quality_gate_v2",
        "agent_model_quality_gate_v3",
    }:
        for case in payload["runs"][0]["trials"][0]["cases"]:
            case["observation"]["runtime_input_namespace"] = "turn-123"
    approval_case["score"].update(
        passed=True,
        core_success=True,
        capability_passed=True,
    )
    report_path = tmp_path / f"invalid-resumes-{approval_resumes}.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="approval evidence"):
        module.render_model_quality_report(
            report_path,
            benchmark_path=tmp_path / "benchmark.md",
            run_record_path=tmp_path / "run.md",
        )


@pytest.mark.parametrize(
    ("evaluator_version", "approval_resumes"),
    [
        ("agent_model_quality_gate_v2", 2),
        ("agent_model_quality_gate_v3", 1),
    ],
)
def test_renderer_rejects_namespaced_evaluator_case_without_runtime_input_namespace(
    tmp_path: Path,
    evaluator_version: str,
    approval_resumes: int,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    payload["evaluator_version"] = evaluator_version
    cases = payload["runs"][0]["trials"][0]["cases"]
    cases[1]["observation"]["runtime_input_namespace"] = "turn-123"
    approval_case = cases[0]
    approval_case["observation"]["approval_resumes"] = approval_resumes
    approval_case["score"].update(
        passed=True,
        core_success=True,
        capability_passed=True,
    )
    report_path = tmp_path / "missing-runtime-namespace.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime_input_namespace"):
        module.render_model_quality_report(
            report_path,
            benchmark_path=tmp_path / "benchmark.md",
            run_record_path=tmp_path / "run.md",
        )


def test_renderer_rejects_passed_case_with_failed_observation_before_writing(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    approval_case = payload["runs"][0]["trials"][0]["cases"][0]
    observation = approval_case["observation"]
    observation["status"] = "failed"
    observation["workspace_assertions_passed"] = True
    approval_case["score"].update(
        passed=True,
        core_success=True,
        capability_passed=True,
    )
    report_path = tmp_path / "passed-with-failed-observation.json"
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        module.render_model_quality_report(
            report_path,
            benchmark_path=benchmark_path,
            run_record_path=run_record_path,
        )

    assert not benchmark_path.exists()
    assert not run_record_path.exists()


def test_renderer_rejects_inconsistent_completed_prefix_score_in_inconclusive_run(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    prefix_case, _infrastructure_case = _make_inconclusive_with_completed_prefix(
        payload
    )
    prefix_case["score"].update(
        passed=True,
        core_success=False,
        capability_passed=True,
    )
    report_path = tmp_path / "inconclusive-with-contradictory-prefix.json"
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        module.render_model_quality_report(
            report_path,
            benchmark_path=benchmark_path,
            run_record_path=run_record_path,
        )

    assert not benchmark_path.exists()
    assert not run_record_path.exists()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("core_success", None), ("capability_passed", "yes")],
)
def test_renderer_requires_boolean_score_fields_for_completed_prefix_case(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    prefix_case, _infrastructure_case = _make_inconclusive_with_completed_prefix(
        payload
    )
    prefix_case["score"][field] = invalid
    report_path = tmp_path / f"inconclusive-prefix-invalid-{field}.json"
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        module.render_model_quality_report(
            report_path,
            benchmark_path=benchmark_path,
            run_record_path=run_record_path,
        )

    assert not benchmark_path.exists()
    assert not run_record_path.exists()


def test_renderer_requires_null_score_fields_for_infrastructure_case(
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    payload = _report_payload(tmp_path)
    _prefix_case, infrastructure_case = _make_inconclusive_with_completed_prefix(
        payload
    )
    infrastructure_case["score"]["core_success"] = False
    report_path = tmp_path / "infrastructure-with-non-null-score.json"
    benchmark_path = tmp_path / "benchmark.md"
    run_record_path = tmp_path / "run.md"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        module.render_model_quality_report(
            report_path,
            benchmark_path=benchmark_path,
            run_record_path=run_record_path,
        )

    assert not benchmark_path.exists()
    assert not run_record_path.exists()


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
            "structured_output": {"content": "before_gate\n"},
            "is_error": False,
            "error_code": None,
            "error_message": None,
            "retryable": False,
            "truncated": False,
            "latency_ms": None,
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
    benchmark = benchmark_path.read_text(encoding="utf-8")
    assert "- Infrastructure status: **INCONCLUSIVE**" in benchmark
    assert "## Model identity and infrastructure" in run_record
    assert ("| `groq_gpt_oss_120b` | `groq` | `openai/gpt-oss-120b` | **INCONCLUSIVE** |") in run_record
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

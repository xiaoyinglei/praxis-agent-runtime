from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
LICENSE = ROOT / "LICENSE"
BENCHMARK = ROOT / "docs" / "benchmark.md"
RUN_RECORD = ROOT / "docs" / "runs" / "deepseek-v4-flash.md"
MODEL_QUALITY_REPORT = (
    ROOT / "evals" / "model_quality" / "runs" / "2026-08-06-deepseek-v4-flash.json"
)
MODEL_QUALITY_RENDERER = ROOT / "scripts" / "render_model_quality_report.py"
RUNBOOK = ROOT / "docs" / "RUNBOOK.md"
MODEL_CATALOG = ROOT / "configs" / "models.yaml"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing repository presentation file: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def test_readme_leads_with_the_approved_praxis_identity_and_demo() -> None:
    readme = _read(README)

    assert readme.startswith("# Praxis\n")
    assert "a trusted-local workspace agent runtime" in readme
    assert "docs/assets/praxis-demo.gif" in readme
    assert "DETERMINISTIC DEMO" in readme
    assert "FAKE MODEL" in readme


def test_readme_follows_the_public_product_story() -> None:
    readme = _read(README)
    ordered_sections = (
        "## Why Praxis",
        "## Runtime architecture",
        "## Current evidence",
        "## Quickstart",
        "## Capability map",
        "## Optional RAG",
        "## Safety and limitations",
        "## Development gates and deeper documentation",
    )

    positions = [readme.index(section) for section in ordered_sections]
    assert positions == sorted(positions)
    for concept in ("Turn", "Loop", "ACI", "approval", "checkpoint", "verification"):
        assert concept in readme
    assert "agent_runtime.Agent" in readme
    assert "uv run agent" in readme
    for capability in ("Files and code", "Data and documents", "Private knowledge", "Extensions"):
        assert capability in readme
    assert "lazy" in readme.lower()
    assert "provider boundary" in readme.lower()
    assert "[MIT](LICENSE)" in readme


def test_readme_scopes_completion_evidence_and_runtime_extensions_truthfully() -> None:
    readme = _read(README)
    unsupported_extension = "registered " + "tools"

    assert "default modification tasks" in readme
    assert "Read-only tasks" in readme
    assert "--no-require-workspace-change" in readme
    assert "require_workspace_change=False" in readme
    assert "Workspace Skills" in readme
    assert "configured MCP" in readme
    assert "bounded subagent" in readme
    assert unsupported_extension not in readme


def test_readme_states_the_real_run_command_platform_boundary() -> None:
    readme = _read(README)

    for boundary in (
        "/usr/bin/sandbox-exec",
        "Seatbelt",
        "fail closed",
        "unavailable",
        "fake sandbox",
        "test-only",
        "not safety evidence",
    ):
        assert boundary in readme


def test_readme_sdk_examples_use_the_real_public_facade() -> None:
    readme = _read(README)
    unsupported_result_field = "result." + "final_answer"
    unsupported_config_loader = "RAGKnowledgeConfig." + "from_file"

    assert "print(result.answer)" in readme
    assert "RAGKnowledgeConfig(" in readme
    assert unsupported_result_field not in readme
    assert unsupported_config_loader not in readme
    snippets = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert len(snippets) == 2
    for index, snippet in enumerate(snippets):
        compile(snippet, f"README.md:python-example-{index}", "exec")


def test_readme_describes_a_source_only_distribution_without_unearned_claims() -> None:
    readme = _read(README)
    old_repository = "Private" + "-RAG-Agent"
    old_import = "rag" + ".agent"
    old_path = "rag" + "/agent"
    readiness_claim = "production" + "-ready"
    personal_path = "/Users/" + "leixiaoying"
    old_distribution = "`agent-" + "runtime`"

    assert "praxis-agent-runtime" in readme
    assert "git clone https://github.com/xiaoyinglei/praxis-agent-runtime.git" in readme
    assert "source checkout" in readme
    assert "local build metadata" in readme
    assert "not published to PyPI" in readme
    for forbidden in (
        old_repository,
        old_import,
        old_path,
        readiness_claim,
        personal_path,
        old_distribution,
    ):
        assert forbidden not in readme
    unearned_suite_verdict = "30-task suite: " + "PASS"
    assert unearned_suite_verdict not in readme


def test_runbook_uses_the_praxis_source_checkout_without_personal_paths() -> None:
    runbook = _read(RUNBOOK)
    personal_path = "/Users/" + "leixiaoying"

    assert runbook.startswith("# Praxis 运行手册\n")
    assert "praxis-agent-runtime" in runbook
    assert "source checkout" in runbook
    assert "未发布到 PyPI" in runbook
    assert "cd /path/to/praxis-agent-runtime" in runbook
    assert personal_path not in runbook


def test_runbook_defaults_are_bound_to_the_model_catalog() -> None:
    runbook = _read(RUNBOOK)
    catalog = yaml.safe_load(MODEL_CATALOG.read_text(encoding="utf-8"))
    defaults = catalog["defaults"]
    models = catalog["models"]
    providers = catalog["providers"]

    expected = {
        "primary_model": (
            defaults["primary_model"],
            models[defaults["primary_model"]]["model"],
        ),
        "embedding_model": (
            defaults["embedding_model"],
            models[defaults["embedding_model"]]["model"],
        ),
        "reranker_model": (
            defaults["reranker_model"],
            models[defaults["reranker_model"]]["model"],
        ),
    }
    for alias, model in expected.values():
        assert f"`{alias}`" in runbook
        assert f"`{model}`" in runbook

    primary = models[defaults["primary_model"]]
    assert providers[primary["provider"]]["api_key_env"] == "GROQ_API_KEY"
    assert "`GROQ_API_KEY`" in runbook
    assert "qwen3_8b_mlx_4bit" in runbook
    assert "显式可选" in runbook


def test_checked_in_live_evidence_pages_match_renderer() -> None:
    with (
        NamedTemporaryFile(dir=BENCHMARK.parent, suffix=".md") as benchmark_file,
        NamedTemporaryFile(dir=RUN_RECORD.parent, suffix=".md") as run_record_file,
    ):
        benchmark = Path(benchmark_file.name)
        run_record = Path(run_record_file.name)
        subprocess.run(
            [
                sys.executable,
                str(MODEL_QUALITY_RENDERER),
                str(MODEL_QUALITY_REPORT),
                "--benchmark",
                str(benchmark),
                "--run-record",
                str(run_record),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        assert benchmark.read_text(encoding="utf-8") == _read(BENCHMARK)
        assert run_record.read_text(encoding="utf-8") == _read(RUN_RECORD)


def test_license_and_live_evidence_pages_are_explicit() -> None:
    license_text = _read(LICENSE)
    readme = _read(README)
    benchmark = _read(BENCHMARK)
    run_record = _read(RUN_RECORD)
    report = json.loads(_read(MODEL_QUALITY_REPORT))

    assert report["status"] in {"passed", "failed"}
    assert report["passed"] is (report["status"] == "passed")
    assert report["source_unchanged"] is True
    assert report["dirty"] is False
    assert report["evaluator_version"] == "agent_model_quality_gate_v3"
    assert len(report["models"]) == 1
    assert len(report["runs"]) == 1

    verdict = report["status"].upper()
    model = report["models"][0]
    run = report["runs"][0]
    assert model["passed"] is report["passed"]
    assert run["status"] == "completed"

    cases = [case for trial in run["trials"] for case in trial["cases"]]
    passed_cases = sum(case["score"]["passed"] is True for case in cases)
    approval_cases = [
        case for case in cases if case["score"]["case_id"] == "approval_continue"
    ]
    passed_approval_cases = sum(
        case["score"]["passed"] is True for case in approval_cases
    )
    conclusion = "PASS" if report["passed"] else "FAIL"

    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 xiaoyinglei" in license_text
    assert (
        f"**{verdict} — {passed_cases}/{len(cases)} scenario executions passed**"
    ) in readme
    assert (
        f"**CONCLUSIVE {conclusion} — {passed_approval_cases}/{len(approval_cases)} "
        "approval trials completed**"
    ) in readme
    assert f"Overall verdict: **{verdict}**" in benchmark
    assert benchmark.startswith("# Agent tool-use reliability benchmark\n")
    assert "not a general coding or reasoning benchmark" in benchmark
    assert "fixed set of controlled, machine-checkable file-tool scenarios" in benchmark
    assert "deterministic file-tool" not in benchmark
    assert "## What the agent was asked to do" in benchmark
    for scenario in (
        "Read a known file directly",
        "Find an unknown file, then read it",
        "Recover when a file was renamed",
        "Resume an edit after approval",
        "Stop after a confirmed missing file",
    ):
        assert scenario in benchmark
    for pass_contract in (
        "Read the specified file with a valid call",
        "Use valid search-then-read calls in that order",
        "Observe the expected failed read",
        "apply the requested edit exactly once",
        "issue no other tool calls",
    ):
        assert pass_contract in benchmark
    for overstated_contract in (
        "avoid extra tool calls",
        "without looping",
        "verify the changed file before finishing",
    ):
        assert overstated_contract not in benchmark
    assert "does **not** establish broad programming ability" in benchmark
    assert "### Exact evaluated instructions" in benchmark
    assert f"source_commit: `{report['source_commit']}`" in benchmark
    assert "source_unchanged: `true`" in benchmark
    assert "dirty: `false`" in benchmark
    assert "evaluator_version: `agent_model_quality_gate_v3`" in benchmark
    for metric, observed in model["observed"].items():
        threshold = model["thresholds"][metric]
        assert (
            f"| `{metric}` | `{observed}` | "
            f"`{threshold['direction']} {threshold['value']}` |"
        ) in benchmark
    for failure in model["failures"]:
        assert failure in benchmark
    assert "Manifest status: **validated only**" in benchmark

    for identity in (run["model_alias"], run["provider"], run["provider_model"]):
        assert identity in run_record
    assert f"Overall verdict: **{verdict}**" in run_record
    assert f"source_commit: `{report['source_commit']}`" in run_record
    assert run_record.count("Workspace assertions passed: `true`") >= len(approval_cases)
    assert (
        run_record.count("Evaluator verdict: **PASSED**") == passed_approval_cases
    )
    assert run_record.count("Evaluator verdict: **FAILED**") == (
        len(approval_cases) - passed_approval_cases
    )
    assert MODEL_QUALITY_REPORT.name in run_record

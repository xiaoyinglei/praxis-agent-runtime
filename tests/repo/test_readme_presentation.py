from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
LICENSE = ROOT / "LICENSE"
BENCHMARK = ROOT / "docs" / "benchmark.md"
RUN_RECORD = ROOT / "docs" / "runs" / "deepseek-v4-flash.md"
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


def test_license_and_live_evidence_pages_are_explicit() -> None:
    license_text = _read(LICENSE)
    readme = _read(README)
    benchmark = _read(BENCHMARK)
    run_record = _read(RUN_RECORD)

    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 xiaoyinglei" in license_text
    assert "**FAILED — 14/15 cases passed; worst-trial task success 80%**" in readme
    assert "one approval trial remained paused after `repeated_inspection`" in readme
    assert "Overall verdict: **FAILED**" in benchmark
    assert "source_commit: `1e4c16873f4d6983397d5965b6d527c09d680689`" in benchmark
    assert "source_unchanged: `true`" in benchmark
    assert "dirty: `false`" in benchmark
    assert "evaluator_version: `agent_model_quality_gate_v3`" in benchmark
    assert "| `task_success_rate` | `0.8` | `min 1.0` |" in benchmark
    assert "task_success_rate: observed 0.8 < baseline floor 1.0" in benchmark
    assert "Manifest status: **validated only**" in benchmark

    for identity in ("deepseek_v4_flash", "deepseek", "deepseek-v4-flash"):
        assert identity in run_record
    assert "Overall verdict: **FAILED**" in run_record
    assert "Error code: `\"repeated_inspection\"`" in run_record
    assert "Workspace assertions passed: `true`" in run_record
    assert "Evaluator verdict: **FAILED**" in run_record
    assert "Evaluator verdict: **PASSED**" in run_record
    assert "2026-08-06-deepseek-v4-flash.json" in run_record

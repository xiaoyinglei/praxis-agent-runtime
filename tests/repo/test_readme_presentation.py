from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
LICENSE = ROOT / "LICENSE"
GUIDE = ROOT / "docs" / "guide.md"
BENCHMARK = ROOT / "docs" / "benchmark.md"
RUN_RECORD = ROOT / "docs" / "runs" / "deepseek-v4-flash.md"
MODEL_QUALITY_REPORT = (
    ROOT / "evals" / "model_quality" / "runs" / "2026-08-06-deepseek-v4-flash.json"
)
MODEL_QUALITY_RENDERER = ROOT / "scripts" / "render_model_quality_report.py"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing repository presentation file: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def test_readme_leads_with_the_approved_praxis_identity_and_demo() -> None:
    readme = _read(README)
    assert readme.startswith("# Praxis\n")
    assert "agent_runtime.Agent" in readme
    assert "scripts/render_praxis_demo.py" in readme
    assert "固定模型响应" in readme
    assert "不代表真实模型的任务能力" in readme
    assert "docs/assets/praxis-demo.gif" in readme


def test_readme_follows_the_public_product_story() -> None:
    readme = _read(README)
    guide = _read(GUIDE)
    assert "docs/guide.md" in readme
    assert "docs/development.md" in readme
    assert "uv run agent setup" in readme
    assert "uv run agent chat" in readme
    for concept in ("Thread", "Turn", "Step", "权限", "审批", "验证"):
        assert concept in readme
    assert "[MIT](LICENSE)" in readme
    for capability in ("数据与文档", "公网检索", "外部服务", "记忆", "RAG"):
        assert capability in readme
    assert "提供方边界按需初始化" in guide


def test_readme_scopes_completion_evidence_and_runtime_extensions_truthfully() -> None:
    readme = _read(README)
    guide = _read(GUIDE)
    assert "修改类任务默认要求工作区确实发生变化" in readme
    assert "只读分析可以直接回答" in readme
    assert "--no-require-workspace-change" in readme
    assert "require_workspace_change=False" in readme
    assert "Skills" in guide and "MCP" in guide and "子智能体" in guide
    assert "预算、权限约束" in guide


def test_readme_states_the_real_run_command_platform_boundary() -> None:
    readme = _read(README)
    for boundary in ("/usr/bin/sandbox-exec", "Seatbelt", "/usr/bin/bwrap", "Bubblewrap",
                     "libseccomp", "拒绝执行", "测试桩", "不能代替真实沙箱验证"):
        assert boundary in readme


def test_readme_sdk_examples_use_the_real_public_facade() -> None:
    documentation = _read(README) + "\n" + _read(GUIDE)
    assert "print(result.answer)" in documentation
    assert "RAGKnowledgeConfig(" in documentation
    assert "result." + "final_answer" not in documentation
    assert "RAGKnowledgeConfig." + "from_file" not in documentation
    snippets = re.findall(r"```python\n(.*?)```", documentation, flags=re.DOTALL)
    assert snippets
    for index, snippet in enumerate(snippets):
        compile(snippet, f"public-docs:python-example-{index}", "exec")


def test_model_integration_docs_cover_requirements_registration_and_switching_only() -> None:
    guide = _read(GUIDE)
    section = guide.split("## 接入模型", 1)[1].split("### 本地 MLX", 1)[0]
    for command in ("agent model list --source", "agent model show", "agent model add",
                    "agent model update", "agent model probe", "agent model remove",
                    "agent model switch", "--from", "--skip-probe"):
        assert command in section
    for requirement in ("OpenAI-compatible", "流式文本增量", "完成状态", "工具调用",
                        "结构化输出", "环境变量", "Turn"):
        assert requirement in section
    assert "无需手工初始化" in section
    assert "MODEL_ID" in section and "--context-window-tokens" in section
    assert "MODEL_ALIAS" not in section and "--provider-model" not in section
    assert section.index("agent model add") < section.index("agent model show")
    assert section.index("agent model show") < section.index("agent model probe")
    for model_family in ("Qwen", "Kimi", "DeepSeek"):
        assert model_family not in section
    assert "API_KEY=" not in section


def test_readme_describes_a_source_only_distribution_without_unearned_claims() -> None:
    readme = _read(README)
    assert "praxis-agent-runtime" in readme
    assert "git clone https://github.com/xiaoyinglei/praxis-agent-runtime.git" in readme
    assert "源码安装" in readme and "尚未发布到 PyPI" in readme
    for forbidden in ("Private" + "-RAG-Agent", "rag" + ".agent", "rag" + "/agent",
                      "production" + "-ready", "/Users/" + "leixiaoying", "`agent-" + "runtime`"):
        assert forbidden not in readme


@pytest.mark.skipif(
    not BENCHMARK.is_file() or not MODEL_QUALITY_REPORT.is_file() or not RUN_RECORD.is_file(),
    reason="Local-only documentation or evaluation fixture absent",
)
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


@pytest.mark.skipif(
    not BENCHMARK.is_file() or not MODEL_QUALITY_REPORT.is_file() or not RUN_RECORD.is_file(),
    reason="Local-only documentation or evaluation fixture absent",
)
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
    approval_cases = [
        case for case in cases if case["score"]["case_id"] == "approval_continue"
    ]
    passed_approval_cases = sum(
        case["score"]["passed"] is True for case in approval_cases
    )

    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 xiaoyinglei" in license_text
    assert "历史评估不代表当前版本" in readme
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

    for identity in (run["provider"], run["provider_model"]):
        assert identity in run_record
    assert run["model_alias"] not in run_record
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

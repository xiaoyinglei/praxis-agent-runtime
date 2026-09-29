from __future__ import annotations

import importlib
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

import rag
from agent_runtime.knowledge import RAGKnowledgeConfig

ROOT = Path(__file__).resolve().parents[2]
CODING_REFERENCE = ROOT / "CLAUDE.md"

_HISTORICAL_EXCEPTIONS = {
    ROOT / "evals/code_agent/benchmark_v1.json",
    ROOT / "tests/agent/test_code_agent_benchmark.py",
}
# Provider transcripts, generated workspaces, and offline replays retain the
# original paths as evidence. They are not portable runtime contracts. Keep the
# maintained runners and README in this directory under the ordinary checks.
_CONTEXT_EVALS = ROOT / "evals/context_management"
_HISTORICAL_EXCEPTIONS.update(_CONTEXT_EVALS.glob("offline-*.json"))
_HISTORICAL_EXCEPTIONS.update(_CONTEXT_EVALS.glob("semantic-live-*.json"))
for _snapshot in _CONTEXT_EVALS.glob("deepseek-*"):
    if _snapshot.is_dir():
        _HISTORICAL_EXCEPTIONS.update(path for path in _snapshot.rglob("*") if path.is_file())
_SCANNED_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}


def _active_contract_files() -> Iterator[Path]:
    roots = (
        ROOT / "agent_runtime",
        ROOT / "evals",
        ROOT / "rag",
        ROOT / "scripts",
        ROOT / "tests",
    )
    for source_root in roots:
        for path in source_root.rglob("*"):
            if (
                path.is_file()
                and path.suffix in _SCANNED_SUFFIXES
                and path not in _HISTORICAL_EXCEPTIONS
            ):
                yield path

    yield ROOT / "README.md"
    for optional in (CODING_REFERENCE, ROOT / "docs/RUNBOOK.md"):
        if optional.is_file():
            yield optional
    yield ROOT / "pyproject.toml"
    yield from (path for path in (ROOT / "docs/design").glob("*.md") if path not in _HISTORICAL_EXCEPTIONS)


def test_distribution_and_console_entrypoint_use_praxis_namespace() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["name"] == "praxis-agent-runtime"
    assert config["project"]["description"] == (
        "Praxis — a trusted-local workspace agent runtime for files, code, data, "
        "and private knowledge."
    )
    assert config["project"]["scripts"]["agent"] == "agent_runtime.cli:agent_app"
    assert config["project"]["scripts"]["rag"] == "rag.cli:app"
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"] == {
        "configs/models.yaml": "agent_runtime/_data/models.yaml"
    }


def test_legacy_rag_agent_package_is_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rag." + "agent")


def test_rag_root_no_longer_exports_agent_runtime_objects() -> None:
    assert not hasattr(rag, "AgentService")
    assert not hasattr(rag, "ToolRegistry")


def test_rag_knowledge_storage_remains_in_rag_namespace() -> None:
    assert RAGKnowledgeConfig().storage_root == Path(".rag")


def test_active_sources_do_not_reference_legacy_agent_namespace() -> None:
    forbidden = (
        "rag." + "agent",
        "rag/" + "agent",
        "Private-" + "RAG-Agent",
        ".rag/" + "agent_",
        ".rag/" + "agent_runtime",
    )
    offenders: dict[str, tuple[str, ...]] = {}

    for path in _active_contract_files():
        source = path.read_text(encoding="utf-8")
        matches = tuple(needle for needle in forbidden if needle in source)
        if matches:
            offenders[str(path.relative_to(ROOT))] = matches

    assert offenders == {}


def test_active_contracts_do_not_expose_a_personal_checkout_path() -> None:
    personal_checkout = "/Users/" + "leixiaoying"
    offenders = [
        str(path.relative_to(ROOT))
        for path in _active_contract_files()
        if personal_checkout in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


@pytest.mark.skipif(not (_CONTEXT_EVALS / "README.md").is_file(), reason="Local-only evaluation assets absent")
def test_context_evaluation_runners_remain_active_contracts() -> None:
    active = set(_active_contract_files())
    assert set(_CONTEXT_EVALS.glob("*.py")) <= active
    assert _CONTEXT_EVALS / "README.md" in active
    assert ROOT / "docs/RUNBOOK.md" in active


@pytest.mark.skipif(not CODING_REFERENCE.is_file(), reason="Local-only coding reference absent")
def test_coding_reference_tracks_the_current_praxis_runtime() -> None:
    reference = CODING_REFERENCE.read_text(encoding="utf-8")

    assert reference.startswith("# CLAUDE.md — Praxis Coding Agent Reference\n")
    assert "cd /path/to/praxis-agent-runtime" in reference
    for gate in (
        "uv run ruff check .",
        "uv run mypy",
        "uv run pytest -q",
        "uv run lint-imports",
        "uv build",
    ):
        assert gate in reference
    for tool_name in (
        "search_text",
        "list_files",
        "read_file",
        "inspect_data_file",
        "apply_patch",
        "run_command",
        "execute_python",
        "update_plan",
        "find_tools",
    ):
        assert f"`{tool_name}`" in reference
    for current_path in (
        "agent_runtime/harness/turn.py",
        "agent_runtime/harness/session.py",
        "agent_runtime/harness/context.py",
        "agent_runtime/harness/tool_orchestrator.py",
        "agent_runtime/harness/rollout.py",
        "agent_runtime/builtin/generic.py",
        "agent_runtime/tools/selection.py",
        "agent_runtime/primitive_ops.py",
        "agent_runtime/tools/tool.py",
        "agent_runtime/tools/registry.py",
        "agent_runtime/tools/executor.py",
        "agent_runtime/cli.py",
        "rag/cli.py",
    ):
        assert f"`{current_path}`" in reference
        assert (ROOT / current_path).is_file()

    stale_terms = (
        "tool_" + "search",
        "activate_" + "tools",
        "write_" + "file",
        "run_python_" + "inline",
        "Tool" + "Spec",
        "Base" + "Tool",
        "--" + "agent",
    )
    assert all(term not in reference for term in stale_terms)

from __future__ import annotations

import ast
import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace


def test_agent_runtime_has_no_rag_imports_except_knowledge_provider() -> None:
    package_root = Path(__file__).parents[2] / "agent_runtime"
    violations: list[str] = []
    for path in package_root.rglob("*.py"):
        relative_path = path.relative_to(package_root).as_posix()
        if relative_path == "knowledge_providers/rag.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "rag" or alias.name.startswith("rag."):
                        violations.append(f"{relative_path}:{alias.name}")
            if module == "rag" or (module is not None and module.startswith("rag.")):
                violations.append(f"{relative_path}:{module}")
    assert not violations, "forbidden RAG imports: " + ", ".join(sorted(violations))


def test_llm_usage_is_canonical_agent_runtime_contract() -> None:
    contracts = importlib.import_module("agent_runtime.modeling.contracts")
    from agent_runtime.modeling.contracts import LLMUsage

    assert LLMUsage is contracts.LLMUsage
    assert contracts.LLMUsage.__module__ == "agent_runtime.modeling.contracts"


def test_rag_provider_projects_query_dtos_to_agent_owned_knowledge_dtos(monkeypatch) -> None:
    from agent_runtime.knowledge import RAGKnowledgeConfig
    from agent_runtime.knowledge_providers.rag import LazyRAGKnowledgeProvider
    from agent_runtime.tools.integrations.knowledge import (
        KnowledgeResult,
        KnowledgeSearchInput,
        KnowledgeSearchOutput,
    )
    from agent_runtime.tools.permissions import ToolExecutionContext

    evidence = SimpleNamespace(
        evidence_id="e-1",
        doc_id=7,
        citation_anchor="p-1",
        text="Evidence text",
        score=0.9,
        source_type="pdf",
        file_name="guide.pdf",
    )
    query_result = SimpleNamespace(
        answer=SimpleNamespace(
            answer_text="Answer",
            citations=[SimpleNamespace(citation_anchor="p-1", citation_id="c-1")],
            groundedness_flag=True,
            insufficient_evidence_flag=False,
        ),
        retrieval=SimpleNamespace(evidence=SimpleNamespace(all=[evidence])),
    )
    provider = LazyRAGKnowledgeProvider(RAGKnowledgeConfig())
    monkeypatch.setattr(provider, "_runtime", SimpleNamespace(query=lambda *_args, **_kwargs: query_result))

    output = asyncio.run(
        provider.search_knowledge(KnowledgeSearchInput(query="what?", top_k=1), ToolExecutionContext())
    )

    assert isinstance(output, KnowledgeSearchOutput)
    assert all(isinstance(item, KnowledgeResult) for item in output.results)
    assert all(type(item).__module__.startswith("agent_runtime") for item in output.results)
    assert type(output).__module__.startswith("agent_runtime")
    assert not any(type(item).__module__.startswith("rag.") for item in output.results)


def test_agent_evidence_and_citation_are_canonical_knowledge_contracts() -> None:
    from agent_runtime.result import AgentCitation, AgentEvidence

    assert AgentEvidence.__module__ == "agent_runtime.knowledge"
    assert AgentCitation.__module__ == "agent_runtime.knowledge"

from __future__ import annotations

import ast
import asyncio
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


def test_rag_production_consumers_do_not_import_legacy_canonical_paths() -> None:
    rag_root = Path(__file__).parents[2] / "rag"
    legacy_wrappers = {
        "assembly/tokenizer.py",
        "models/config.py",
        "providers/llm_gateway.py",
        "schema/llm.py",
        "utils/text.py",
    }
    legacy_modules = {
        "rag.assembly.tokenizer",
        "rag.models.config",
        "rag.providers.llm_gateway",
        "rag.schema.llm",
        "rag.utils.text",
    }
    violations: list[str] = []
    for path in rag_root.rglob("*.py"):
        relative_path = path.relative_to(rag_root).as_posix()
        if relative_path in legacy_wrappers:
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.ImportFrom):
                if node.module in legacy_modules:
                    violations.append(f"{relative_path}:{node.module}")
                if node.module == "rag.assembly" and {alias.name for alias in node.names} & {
                    "TokenAccountingService",
                    "TokenizerContract",
                }:
                    violations.append(f"{relative_path}:rag.assembly tokenizer exports")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in legacy_modules:
                        violations.append(f"{relative_path}:{alias.name}")
    assert not violations, "legacy canonical imports: " + ", ".join(sorted(violations))


def test_legacy_and_canonical_modeling_exports_preserve_identity() -> None:
    from agent_runtime.modeling.config import GenerationConfig
    from agent_runtime.modeling.contracts import LLMUsage
    from agent_runtime.modeling.gateway import LLMGateway
    from agent_runtime.modeling.local_agent_wire import (
        LocalAgentWireRequest,
        estimate_local_agent_usage,
    )
    from agent_runtime.modeling.openai_wire import OpenAIWireRequest
    from agent_runtime.modeling.providers.ollama.generator import OllamaGenerator
    from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract
    from agent_runtime.text import text_unit_count
    from rag.assembly.tokenizer import (
        TokenAccountingService as LegacyTokenAccountingService,
    )
    from rag.assembly.tokenizer import TokenizerContract as LegacyTokenizerContract
    from rag.models.config import GenerationConfig as LegacyGenerationConfig
    from rag.providers.llm_gateway import LLMGateway as LegacyLLMGateway
    from rag.providers import local_agent_wire as legacy_local_agent_wire_module
    from rag.providers.local_agent_wire import LocalAgentWireRequest as LegacyLocalAgentWireRequest
    from rag.providers.local_agent_wire import estimate_local_agent_usage as legacy_estimate_local_agent_usage
    from rag.providers.ollama.generator import OllamaGenerator as LegacyOllamaGenerator
    from rag.providers.openai_wire import OpenAIWireRequest as LegacyOpenAIWireRequest
    from rag.schema.llm import LLMUsage as LegacyLLMUsage
    from rag.utils.text import text_unit_count as legacy_text_unit_count

    assert LegacyLLMUsage is LLMUsage
    assert LLMUsage.__module__ == "agent_runtime.modeling.contracts"
    assert LegacyGenerationConfig is GenerationConfig
    assert GenerationConfig.__module__ == "agent_runtime.modeling.config"
    assert LegacyLLMGateway is LLMGateway
    assert LLMGateway.__module__ == "agent_runtime.modeling.gateway"
    assert LegacyTokenAccountingService is TokenAccountingService
    assert LegacyTokenizerContract is TokenizerContract
    assert TokenAccountingService.__module__ == "agent_runtime.modeling.tokenization"
    assert legacy_text_unit_count is text_unit_count
    assert text_unit_count.__module__ == "agent_runtime.text"
    assert LegacyLocalAgentWireRequest is LocalAgentWireRequest
    assert legacy_estimate_local_agent_usage is estimate_local_agent_usage
    assert "estimate_local_agent_usage" in legacy_local_agent_wire_module.__all__
    assert LegacyOpenAIWireRequest is OpenAIWireRequest
    assert LegacyOllamaGenerator is OllamaGenerator


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
    from agent_runtime.knowledge import AgentCitation as KnowledgeAgentCitation
    from agent_runtime.knowledge import AgentEvidence as KnowledgeAgentEvidence
    from agent_runtime.result import AgentCitation, AgentEvidence

    assert AgentEvidence is KnowledgeAgentEvidence
    assert AgentCitation is KnowledgeAgentCitation
    assert AgentEvidence.__module__ == "agent_runtime.knowledge"
    assert AgentCitation.__module__ == "agent_runtime.knowledge"

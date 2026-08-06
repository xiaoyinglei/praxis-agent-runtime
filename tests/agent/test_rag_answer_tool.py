from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.integrations import knowledge as knowledge_module
from agent_runtime.tools.integrations.knowledge import (
    KnowledgeSearchOutput,
    create_knowledge_tools,
    create_search_knowledge_tool,
)
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import Tool, ToolCall, ToolCallOrigin


def _call(arguments: Mapping[str, Any]) -> ToolCall:
    return ToolCall(
        tool_call_id="call_knowledge",
        tool_name="search_knowledge",
        arguments=arguments,
        origin=ToolCallOrigin(
            request_id="request_knowledge",
            toolset_revision="knowledge-v1",
            exposed_tool_names=("search_knowledge",),
        ),
    )


def test_knowledge_tool_requires_explicit_configuration() -> None:
    assert create_knowledge_tools(None) == ()

    tools = create_knowledge_tools(
        lambda _arguments: KnowledgeSearchOutput(),
        execution_revision="knowledge-v2",
    )

    assert len(tools) == 1
    assert isinstance(tools[0], Tool)
    assert tools[0].definition.name == "search_knowledge"
    assert tools[0].execution_revision.endswith(":knowledge-v2")


@pytest.mark.anyio
async def test_knowledge_search_normalizes_one_canonical_tool_result() -> None:
    calls: list[Mapping[str, Any]] = []

    async def search(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append(arguments)
        return {
            "results": [
                {
                    "evidence_id": "ev-1",
                    "doc_id": "doc-7",
                    "citation_anchor": "doc-7#p1",
                    "text": "Canonical knowledge evidence",
                    "score": 0.91,
                    "source_type": "document",
                    "file_name": "report.pdf",
                }
            ],
            "answer_text": "Grounded answer",
            "citations": ["doc-7#p1"],
            "groundedness_flag": True,
            "insufficient_evidence": False,
            "total_found": 1,
        }

    tool = create_search_knowledge_tool(search)
    execution = await ToolExecutor({"search_knowledge": tool}).execute(
        _call({"query": "What is canonical?", "top_k": 4}),
        context=ToolExecutionContext(),
    )

    assert execution.result.is_error is False
    assert calls[0]["query"] == "What is canonical?"
    assert calls[0]["top_k"] == 4
    output = execution.result.structured_content
    assert output is not None
    assert output["results"][0]["evidence_id"] == "ev-1"
    assert output["citations"] == ("doc-7#p1",)
    assert output["groundedness_flag"] is True


@pytest.mark.anyio
async def test_invalid_knowledge_output_fails_at_the_final_executor() -> None:
    tool = create_search_knowledge_tool(
        lambda _arguments: {"results": "not-a-list"}
    )

    execution = await ToolExecutor({"search_knowledge": tool}).execute(
        _call({"query": "bad provider output"}),
        context=ToolExecutionContext(),
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == "normalization_failed"
    assert execution.result.retryable is False


def test_knowledge_integration_does_not_own_retrieval_lifecycle() -> None:
    module_path = Path(knowledge_module.__file__ or "")
    source = module_path.read_text(encoding="utf-8")
    imports = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(
        module.startswith(("rag.retrieval", "rag.ingestion"))
        for module in imports
    )
    assert "VectorStore" not in source
    assert "ingest" not in source.lower()

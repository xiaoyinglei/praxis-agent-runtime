from __future__ import annotations

from agent_runtime.tools.integrations.knowledge import (
    KnowledgeSearchOutput,
    create_knowledge_tools,
)
from agent_runtime.tools.tool import Tool


def test_configured_knowledge_exposes_one_final_tool() -> None:
    tools = create_knowledge_tools(
        lambda _arguments: KnowledgeSearchOutput()
    )

    assert len(tools) == 1
    assert isinstance(tools[0], Tool)
    assert tuple(tool.definition.name for tool in tools) == (
        "search_knowledge",
    )
    assert {
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "graph_expand",
    }.isdisjoint(tool.definition.name for tool in tools)


def test_unconfigured_knowledge_adds_no_tool() -> None:
    assert create_knowledge_tools(None) == ()

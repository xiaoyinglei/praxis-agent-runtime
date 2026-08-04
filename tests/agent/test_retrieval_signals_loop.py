from __future__ import annotations

import importlib.util

import pytest
from pydantic import ValidationError

from agent_runtime.core.context import AgentRunConfig
from agent_runtime.core.observations import ObservationExtractor
from agent_runtime.loop.state import create_loop_state
from agent_runtime.service import AgentRunResult
from agent_runtime.tools.integrations.knowledge import KnowledgeSearchInput
from agent_runtime.tools.tool import ToolResult


def test_final_knowledge_input_does_not_expose_retrieval_routing() -> None:
    payload = KnowledgeSearchInput(query="policy", top_k=5)

    assert payload.query == "policy"
    assert not hasattr(payload, "retrieval_signals")
    assert not hasattr(payload, "constraints")
    with pytest.raises(ValidationError):
        KnowledgeSearchInput.model_validate(
            {
                "query": "policy",
                "retrieval_signals": {"special_targets": ["table"]},
            }
        )
    with pytest.raises(ValidationError):
        KnowledgeSearchInput.model_validate(
            {
                "query": "policy",
                "constraints": {"owner": "finance"},
            }
        )


def test_knowledge_result_semantics_stay_in_canonical_tool_result() -> None:
    result = ToolResult(
        tool_call_id="call-knowledge",
        tool_name="search_knowledge",
        structured_content={
            "results": [],
            "answer_text": "Grounded answer",
            "citations": ["report#1"],
            "groundedness_flag": True,
            "insufficient_evidence": False,
            "total_found": 0,
        },
    )
    batch = ObservationExtractor().extract([result])

    assert batch.structured_observations[0].tool_call_id == ("call-knowledge")
    assert batch.answer_candidates[0].text == "Grounded answer"
    assert result.structured_content["groundedness_flag"] is True
    assert not hasattr(batch, "retrieval_signals")


def test_loop_state_has_no_retrieval_control_channel() -> None:
    state = create_loop_state(
        current_message="search configured knowledge",
        run_config=AgentRunConfig(
            turn_id="no-retrieval-signals",
            llm_budget_total=100,
        ),
    )

    assert "retrieval_signals" not in state
    assert "retrieval_signals_debug" not in state


def test_public_run_result_accepts_final_knowledge_citation_anchors() -> None:
    state = create_loop_state(
        current_message="search configured knowledge",
        run_config=AgentRunConfig(
            turn_id="final-knowledge-result",
            llm_budget_total=100,
        ),
    )
    state["tool_results"] = [
        ToolResult(
            tool_call_id="call-knowledge",
            tool_name="search_knowledge",
            structured_content={
                "results": [],
                "answer_text": "Grounded answer",
                "citations": ["report#1"],
                "groundedness_flag": True,
                "insufficient_evidence": False,
                "total_found": 0,
            },
        )
    ]

    result = AgentRunResult.from_loop_result(state)

    assert result.groundedness_flag is True
    assert result.citations == []


def test_deleted_agent_retrieval_modules_are_not_importable() -> None:
    assert importlib.util.find_spec("agent_runtime.tools.rag_tools") is None
    assert importlib.util.find_spec("agent_runtime.tools.rag_tool_runner") is None

from __future__ import annotations

import inspect

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.llm_context import AgentLLMContextAssembler
from rag.agent.core.observations import ObservationBatch, StructuredObservation
from rag.agent.loop.runtime import AgentLoop
from rag.agent.loop.state import create_loop_state
from rag.agent.memory.injector import ContextBuilder
from rag.agent.tools.tool import ToolContentBlock, ToolResult
from rag.schema.llm import LLMCallStage, LLMStageBudget


class _CharacterTokenAccounting:
    def count(self, text: str) -> int:
        return len(text)

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str:
        clipped = text[: max(token_budget, 0)]
        if add_ellipsis and len(clipped) < len(text) and token_budget >= 4:
            return clipped[: token_budget - 4].rstrip() + " ..."
        return clipped


def _config() -> AgentRunConfig:
    return AgentRunConfig(
        turn_id="boundary-cleanup",
        llm_budget_total=10_000,
    )


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Use canonical tool results.",
        allowed_tools=["search_knowledge"],
    )


def _state() -> dict[str, object]:
    return create_loop_state(current_message="Find policy evidence", run_config=_config())


def test_observations_update_typed_working_state_without_flat_channels() -> None:
    state = _state()
    batch = ObservationBatch(
        structured_observations=[
            StructuredObservation(
                tool_call_id="call-1",
                tool_name="search_knowledge",
                status="ok",
                locators=[{"doc_id": 7, "section_id": 3}],
                raw_result_ref="call-1",
            )
        ]
    )

    AgentLoop._merge_observations(state, batch)

    assert state["memory_state"].recent_observations == batch.structured_observations
    assert state["memory_state"].known_locators == [
        {"doc_id": 7, "section_id": 3}
    ]
    assert {
        "structured_observations",
        "evidence",
        "citations",
        "evidence_refs",
        "computation_results",
        "context_units",
        "groundedness_flag",
        "insufficient_evidence_flag",
    }.isdisjoint(state)


def test_context_builder_uses_canonical_result_without_formatter() -> None:
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="call-knowledge",
            tool_name="search_knowledge",
            content=(
                ToolContentBlock(
                    type="text",
                    data={"text": "Policy requires prior approval."},
                ),
            ),
            structured_content={
                "answer_text": "Policy requires prior approval.",
                "groundedness_flag": True,
            },
        )
    ]

    context = ContextBuilder(
        max_context_tokens=8000,
        token_accounting=_CharacterTokenAccounting(),
    ).assemble_loop(definition=_definition(), state=state)

    rendered = context.section("tool_results").content
    assert "call-knowledge" in rendered
    assert "Policy requires prior approval." in rendered
    assert "formatter_resolver" not in inspect.signature(ContextBuilder).parameters


def test_llm_context_assembler_preserves_canonical_tool_content() -> None:
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="call-knowledge",
            tool_name="search_knowledge",
            structured_content={"answer_text": "Canonical evidence."},
        )
    ]
    assembler = AgentLLMContextAssembler(
        token_accounting=_CharacterTokenAccounting(),
        stage_budgets={
            LLMCallStage.TOOL_DECISION: LLMStageBudget(
                max_input_tokens=12_000,
                max_output_tokens=2048,
            )
        },
    )

    assembled = assembler.assemble_loop_turn(
        definition=_definition(),
        state=state,
        budget_remaining=10_000,
    )

    assert "call-knowledge" in assembled.prompt
    assert "Canonical evidence." in assembled.prompt
    assert "formatter_resolver" not in inspect.signature(AgentLLMContextAssembler).parameters

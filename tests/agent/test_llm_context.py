from __future__ import annotations

import pytest

from agent_runtime.core.context import AgentRunConfig
from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.llm_context import (
    AgentLLMContextAssembler,
    AgentLLMContextOverflowError,
)
from agent_runtime.loop.state import LoopState, create_loop_state
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


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="SYSTEM_POLICY",
        allowed_tools=["llm_generate"],
    )


def _state() -> LoopState:
    return create_loop_state(
        current_message="Answer the task",
        run_config=AgentRunConfig(
            turn_id="context-assembler",
            llm_budget_total=10_000,
        ),
    )


def _assembler(max_input_tokens: int) -> AgentLLMContextAssembler:
    return AgentLLMContextAssembler(
        token_accounting=_CharacterTokenAccounting(),
        stage_budgets={
            stage: LLMStageBudget(
                max_input_tokens=max_input_tokens,
                max_output_tokens=20,
                safety_margin_tokens=0,
            )
            for stage in LLMCallStage
        },
    )


def test_generate_context_uses_model_token_count_for_complete_prompt() -> None:
    assembled = _assembler(500).assemble_generate(
        definition=_definition(),
        state=_state(),
        prompt="Produce an answer",
        context_sections=["REAL_SUPPORT"],
        stage=LLMCallStage.LLM_GENERATE,
    )

    assert "SYSTEM_POLICY" in assembled.prompt
    assert "Produce an answer" in assembled.prompt
    assert "REAL_SUPPORT" in assembled.prompt
    assert assembled.context.context_budget.used_context_tokens == len(assembled.prompt)


def test_required_call_context_overflow_raises_without_hash_replacement() -> None:
    with pytest.raises(AgentLLMContextOverflowError) as exc_info:
        _assembler(80).assemble_generate(
            definition=_definition(),
            state=_state(),
            prompt="Produce an answer",
            context_sections=["REQUIRED_REAL_CONTEXT " * 20],
            stage=LLMCallStage.LLM_GENERATE,
        )

    assert exc_info.value.context_budget.overflow is True
    assert "call_context" in (exc_info.value.context_budget.required_truncated)
    assert "sha256=" not in str(exc_info.value)


def test_optional_state_context_can_be_reduced_without_overflow() -> None:
    state = _state()
    state["memory_state"].memory_warnings = ["OPTIONAL_WARNING " * 100]

    assembled = _assembler(180).assemble_generate(
        definition=_definition(),
        state=state,
        prompt="Answer",
        context_sections=[],
    )

    assert assembled.context.context_budget.overflow is False
    assert assembled.context.context_budget.degraded is True
    assert "memory" in {
        *assembled.context.context_budget.dropped_sections,
        *assembled.context.context_budget.summarized_sections,
    }

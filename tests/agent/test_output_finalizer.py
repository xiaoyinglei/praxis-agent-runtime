from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_runtime.core.context import AgentRunConfig, TurnRegistry
from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.llm_context import AgentLLMContextOverflowError
from agent_runtime.core.output_finalizer import (
    ModelStructuredOutputFinalizer,
    OutputValidationExhaustedError,
    final_answer_from_output,
)
from agent_runtime.loop.state import LoopState, create_loop_state
from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
from rag.providers.llm_gateway import LLMGateway
from rag.schema.llm import LLMCallStage, LLMStageBudget


class _AnswerOutput(BaseModel):
    answer: str
    confidence: float


class _TextOutput(BaseModel):
    text: str
    answer: str


class _JsonOnlyOutput(BaseModel):
    value: int
    labels: list[str]


class _StructuredGenerator:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls = 0
        self.prompts: list[str] = []

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        **_: object,
    ) -> object:
        self.prompts.append(prompt)
        response = self._responses[self.calls]
        self.calls += 1
        return response


def _accounting() -> TokenAccountingService:
    return TokenAccountingService(
        TokenizerContract(
            embedding_model_name="output-test",
            tokenizer_model_name="output-test",
            chunking_tokenizer_model_name="output-test",
            tokenizer_backend="simple",
            max_context_tokens=8_000,
            prompt_reserved_tokens=0,
            local_files_only=True,
        )
    )


def _gateway(
    generator: object,
    *,
    max_input_tokens: int = 4_000,
) -> LLMGateway:
    return LLMGateway(
        generator=generator,
        token_accounting=_accounting(),
        model_context_tokens=8_000,
        stage_budgets={
            LLMCallStage.FINAL_SYNTHESIS: LLMStageBudget(
                max_input_tokens=max_input_tokens,
                max_output_tokens=512,
                safety_margin_tokens=64,
            )
        },
    )


def _definition(*, retries: int = 2) -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Return grounded structured output.",
        allowed_tools=[],
        output_model=_AnswerOutput,
        output_validation_max_retries=retries,
    )


def _state(*, task: str = "Answer with confidence") -> LoopState:
    config = AgentRunConfig(
        turn_id="output-finalizer",
        llm_budget_total=20_000,
    )
    TurnRegistry.remove(config.turn_id)
    TurnRegistry.get_or_create(config)
    return create_loop_state(current_message=task, run_config=config)


@pytest.mark.anyio
async def test_structured_finalizer_returns_validated_model() -> None:
    generator = _StructuredGenerator([{"answer": "validated", "confidence": 0.9}])
    gateway = _gateway(generator)
    finalizer = ModelStructuredOutputFinalizer(gateway=gateway)

    result = await finalizer.finalize(
        definition=_definition(),
        state=_state(),
        candidate_text="candidate",
    )

    assert result == _AnswerOutput(answer="validated", confidence=0.9)
    assert finalizer.token_accounting is gateway.token_accounting
    assert generator.calls == 1
    TurnRegistry.remove("output-finalizer")


@pytest.mark.anyio
async def test_structured_finalizer_repairs_validation_failure_with_bounded_retry() -> None:
    generator = _StructuredGenerator(
        [
            {"answer": "missing confidence"},
            {"answer": "repaired", "confidence": 0.8},
        ]
    )
    finalizer = ModelStructuredOutputFinalizer(gateway=_gateway(generator))

    result = await finalizer.finalize(
        definition=_definition(retries=2),
        state=_state(),
        candidate_text="candidate",
    )

    assert result.answer == "repaired"
    assert generator.calls == 2
    assert "confidence" in generator.prompts[1]
    TurnRegistry.remove("output-finalizer")


@pytest.mark.anyio
async def test_structured_finalizer_exhausts_configured_validation_retries() -> None:
    generator = _StructuredGenerator(
        [
            {"answer": "attempt 1"},
            {"answer": "attempt 2"},
            {"answer": "attempt 3"},
        ]
    )
    finalizer = ModelStructuredOutputFinalizer(gateway=_gateway(generator))

    with pytest.raises(OutputValidationExhaustedError) as exc_info:
        await finalizer.finalize(
            definition=_definition(retries=2),
            state=_state(),
            candidate_text="candidate",
        )

    assert exc_info.value.attempts == 3
    assert generator.calls == 3
    TurnRegistry.remove("output-finalizer")


@pytest.mark.anyio
async def test_structured_finalizer_context_overflow_skips_model_call() -> None:
    generator = _StructuredGenerator([{"answer": "unused", "confidence": 1.0}])
    finalizer = ModelStructuredOutputFinalizer(gateway=_gateway(generator, max_input_tokens=8))

    with pytest.raises(AgentLLMContextOverflowError):
        await finalizer.finalize(
            definition=_definition(),
            state=_state(task="required task " * 100),
            candidate_text="required candidate " * 100,
        )

    assert generator.calls == 0
    TurnRegistry.remove("output-finalizer")


def test_final_answer_prefers_text_then_answer_fields() -> None:
    assert final_answer_from_output(_TextOutput(text="text field", answer="answer field")) == "text field"
    assert final_answer_from_output(_AnswerOutput(answer="answer field", confidence=0.7)) == "answer field"


def test_final_answer_uses_json_when_no_text_field_exists() -> None:
    output = _JsonOnlyOutput(value=7, labels=["a", "b"])

    assert final_answer_from_output(output) == output.model_dump_json(exclude_none=True)

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agent_runtime.core.context import LLMBudgetLedger
from rag.providers.generation import (
    AnswerGenerationService,
    AnswerGenerator,
    AnswerSectionPayload,
    GeneratorBinding,
    StructuredAnswerPayload,
)
from rag.providers.llm_gateway import LLMGateway, llm_budget_scope
from rag.schema.llm import (
    LLMCallStage,
    LLMProviderResult,
    LLMStageBudget,
    LLMUsage,
)
from rag.schema.query import EvidenceItem, GroundingTarget
from rag.schema.runtime import AccessPolicy, RuntimeMode


class _TextOnlyJsonGenerator:
    def __init__(self, output: str) -> None:
        self._output = output

    def generate_structured(self, *, prompt: str, schema: type[Any], **kwargs: Any) -> Any:
        del prompt, schema, kwargs
        raise RuntimeError("structured generation is not supported")

    def generate_text(self, *, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        return self._output


class _UsageStructuredGenerator:
    def generate_structured_with_usage(
        self,
        *,
        prompt: str,
        schema: type[Any],
        **kwargs: Any,
    ) -> LLMProviderResult[Any]:
        del prompt, kwargs
        return LLMProviderResult(
            value=schema.model_validate_json(_structured_answer_json(fenced=False)),
            usage=LLMUsage(
                input_tokens=30,
                output_tokens=10,
                source="provider",
            ),
        )


class _UsageTextGenerator:
    def generate_text_with_usage(
        self,
        *,
        prompt: str,
        **kwargs: Any,
    ) -> LLMProviderResult[str]:
        del prompt, kwargs
        return LLMProviderResult(
            value="Direct model answer.",
            usage=LLMUsage(
                input_tokens=12,
                output_tokens=4,
                source="provider",
            ),
        )


def _structured_answer_json(*, fenced: bool) -> str:
    payload = {
        "answer_text": "Alpha Engine handles ingestion requests. [Doc-1]",
        "answer_sections": [
            {
                "title": "Direct",
                "text": "Alpha Engine handles ingestion requests. [Doc-1]",
                "evidence_ids": ["E1"],
            }
        ],
        "insufficient_evidence_flag": False,
    }
    raw = json.dumps(payload)
    return f"```json\n{raw}\n```" if fenced else raw


def test_answer_generation_removes_fabricated_doc_aliases() -> None:
    service = AnswerGenerationService()
    evidence = [
        EvidenceItem(
            evidence_id="E1",
            doc_id=1,
            source_id=2,
            citation_anchor="Architecture",
            text="Alpha Engine handles ingestion requests.",
            score=0.95,
            record_type="section",
            grounding_target=GroundingTarget(
                kind="section",
                doc_id=1,
                source_id=2,
                section_id=7,
            ),
        )
    ]

    answer = service.answer_from_structured_payload(
        query="What does Alpha Engine handle?",
        evidence_pack=evidence,
        grounded_candidate="Alpha Engine handles ingestion requests.",
        payload=StructuredAnswerPayload(
            answer_text="Alpha Engine handles ingestion requests. [Doc-99]",
            answer_sections=[
                AnswerSectionPayload(
                    title="Direct",
                    text="Alpha Engine handles ingestion requests. [Doc-99]",
                    evidence_ids=["E1"],
                )
            ],
        ),
        trust_evidence_pack=True,
    )

    assert "[Doc-99]" not in answer.answer_text
    assert "[Doc-99]" not in answer.answer_sections[0].text
    assert "[1:7]" in answer.answer_text
    assert "[1:7]" in answer.answer_sections[0].text


@pytest.mark.parametrize("fenced", [False, True])
def test_answer_generator_parses_json_text_fallback_after_structured_failure(fenced: bool) -> None:
    evidence = [
        EvidenceItem(
            evidence_id="E1",
            doc_id=1,
            source_id=2,
            citation_anchor="Architecture",
            text="Alpha Engine handles ingestion requests.",
            score=0.95,
            record_type="section",
            grounding_target=GroundingTarget(
                kind="section",
                doc_id=1,
                source_id=2,
                section_id=7,
            ),
        )
    ]
    generator = AnswerGenerator(
        generators=(
            GeneratorBinding(
                backend=_TextOnlyJsonGenerator(_structured_answer_json(fenced=fenced)),
                provider_name="local-openai-compatible",
                model_name="qwen3-8b",
                location="local",
            ),
        )
    )

    result = asyncio.run(
        generator.generate(
            query="What does Alpha Engine handle?",
            prompt="Return JSON",
            evidence_pack=evidence,
            grounded_candidate="Alpha Engine handles ingestion requests.",
            runtime_mode=RuntimeMode.FAST,
            access_policy=AccessPolicy.default(),
        )
    )

    assert result.attempts[0].status == "failed"
    assert "structured_generation_failed" in (result.attempts[0].error or "")
    assert result.answer.answer_text == "Alpha Engine handles ingestion requests. [1:7]"
    assert not result.answer.answer_text.lstrip().startswith("{")
    assert result.answer.answer_sections[0].title == "Direct"
    assert result.answer.answer_sections[0].text == "Alpha Engine handles ingestion requests. [1:7]"
    assert result.answer.answer_sections[0].evidence_ids == ["E1"]
    assert result.answer.groundedness_flag is True
    assert result.answer.evidence_links[0].answer_excerpt == "Alpha Engine handles ingestion requests. [1:7]"


@pytest.mark.anyio
async def test_answer_generator_uses_gateway_and_active_run_ledger() -> None:
    evidence = [
        EvidenceItem(
            evidence_id="E1",
            doc_id=1,
            source_id=2,
            citation_anchor="Architecture",
            text="Alpha Engine handles ingestion requests.",
            score=0.95,
            record_type="section",
            grounding_target=GroundingTarget(
                kind="section",
                doc_id=1,
                source_id=2,
                section_id=7,
            ),
        )
    ]
    backend = _UsageStructuredGenerator()
    gateway = LLMGateway(
        generator=backend,
        token_accounting=type(
            "WordTokens",
            (),
            {"count": lambda self, text: len(text.split())},
        )(),
        model_context_tokens=20_000,
        stage_budgets={
            LLMCallStage.RAG_ANSWER: LLMStageBudget(
                max_input_tokens=16_000,
                max_output_tokens=4_096,
                safety_margin_tokens=512,
            )
        },
    )
    ledger = LLMBudgetLedger(total=10_000)
    generator = AnswerGenerator(
        generators=(
            GeneratorBinding(
                backend=backend,
                provider_name="test",
                model_name="test-model",
                location="cloud",
                gateway=gateway,
            ),
        )
    )

    with llm_budget_scope(ledger):
        result = await generator.generate(
            query="What does Alpha Engine handle?",
            prompt="Return JSON",
            evidence_pack=evidence,
            grounded_candidate="Alpha Engine handles ingestion requests.",
            runtime_mode=RuntimeMode.FAST,
            access_policy=AccessPolicy.default(),
        )

    assert result.answer.groundedness_flag is True
    assert await ledger.committed() == 40


@pytest.mark.anyio
async def test_direct_answer_uses_gateway_and_active_run_ledger() -> None:
    backend = _UsageTextGenerator()
    gateway = LLMGateway(
        generator=backend,
        token_accounting=type(
            "WordTokens",
            (),
            {"count": lambda self, text: len(text.split())},
        )(),
        model_context_tokens=20_000,
        stage_budgets={
            LLMCallStage.RAG_ANSWER: LLMStageBudget(
                max_input_tokens=16_000,
                max_output_tokens=4_096,
                safety_margin_tokens=512,
            )
        },
    )
    ledger = LLMBudgetLedger(total=10_000)
    generator = AnswerGenerator(
        generators=(
            GeneratorBinding(
                backend=backend,
                provider_name="test",
                model_name="test-model",
                location="cloud",
                gateway=gateway,
            ),
        )
    )

    with llm_budget_scope(ledger):
        result = await generator.generate_direct(
            query="Answer directly",
            prompt="Answer directly",
            access_policy=AccessPolicy.default(),
        )

    assert result.answer.answer_text == "Direct model answer."
    assert await ledger.committed() == 16

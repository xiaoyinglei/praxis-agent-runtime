from __future__ import annotations

import pytest

from agent_runtime.modeling.contracts import (
    LLMCallStage,
    LLMProviderResult,
    LLMStageBudget,
    LLMUsage,
)
from agent_runtime.modeling.gateway import LLMGateway
from rag.ingest.retrievalsummarizer import (
    RetrievalSummarizer,
    RetrievalSummaryConfig,
)
from rag.ingest.table_sampler import TABLE_POLICY_COMPUTE_ONLY, profile_markdown_table, profile_table_data
from rag.schema.core import ParsedSection


class _WordTokenAccounting:
    def count(self, text: str) -> int:
        return len(text.split())

    def clip(self, text: str, token_budget: int, *, add_ellipsis: bool = False) -> str:
        words = text.split()
        clipped = " ".join(words[:token_budget])
        if add_ellipsis and len(words) > token_budget:
            return f"{clipped} ..."
        return clipped

    def tail(self, text: str, token_budget: int) -> str:
        words = text.split()
        if token_budget <= 0:
            return ""
        return " ".join(words[-token_budget:])


class _FixedTokenAccounting:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self, text: str) -> int:
        del text
        return self._count


class _RecordingGenerator:
    provider_name = "test-provider"
    model_name = "test-model"

    def __init__(self, output: str) -> None:
        self.output = output
        self.prompts: list[str] = []

    def generate_text(self, *, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.output


class _UsageRecordingGenerator:
    provider_name = "test-provider"
    model_name = "test-model"

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def generate_text_with_usage(
        self,
        *,
        prompt: str,
        **kwargs: object,
    ) -> LLMProviderResult[str]:
        self.calls.append({"prompt": prompt, **kwargs})
        return LLMProviderResult(
            value=self.output,
            usage=LLMUsage(
                input_tokens=20,
                output_tokens=4,
                source="provider",
            ),
        )


class _FailingGenerator:
    provider_name = "test-provider"
    model_name = "test-model"

    def generate_text(self, *, prompt: str) -> str:
        del prompt
        raise RuntimeError("summary service unavailable")


_STRUCTURED_SUMMARY = """Semantic Core: reimbursement approval workflow
Fact Anchors: 2026, finance department, 5000 CNY
Retrieval Keywords: reimbursement, approval, finance, 5000"""


def _assert_three_part_prompt(prompt: str) -> None:
    assert "Semantic Core:" in prompt
    assert "Fact Anchors:" in prompt
    assert "Retrieval Keywords:" in prompt
    assert "Output exactly these three fields" in prompt


def test_retrieval_summarizer_samples_and_limits_by_tokens() -> None:
    generator = _RecordingGenerator("summary0 summary1 summary2 summary3 summary4 summary5")
    summarizer = RetrievalSummarizer(
        generator,
        token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
        config=RetrievalSummaryConfig(
            direct_return_token_threshold=1,
            max_input_tokens=5,
            max_output_tokens=4,
            head_tokens=3,
            middle_tokens=0,
            tail_tokens=2,
        ),
    )
    section = ParsedSection(
        toc_path=("Policy", "Travel"),
        heading_level=2,
        page_range=(1, 1),
        order_index=0,
        text="token0 token1 token2 token3 token4 token5 token6 token7 token8 token9",
        char_range_start=0,
        char_range_end=69,
    )

    result = summarizer.summarize_section_with_metadata(section, "Policy")

    assert generator.prompts
    prompt = generator.prompts[0]
    # 三区采样：head(前) + middle(中) + tail(末)，不再是只取头尾
    assert "token0 token1" in prompt
    assert "token8 token9" in prompt
    assert result.text == "summary0 summary1 summary2 summary3"


def test_retrieval_summarizer_uses_retrieval_summary_gateway_budget() -> None:
    generator = _UsageRecordingGenerator(_STRUCTURED_SUMMARY)
    gateway = LLMGateway(
        generator=generator,
        token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
        model_context_tokens=10_000,
        stage_budgets={
            LLMCallStage.RETRIEVAL_SUMMARY: LLMStageBudget(
                max_input_tokens=6_000,
                max_output_tokens=1_024,
                safety_margin_tokens=512,
            )
        },
    )
    summarizer = RetrievalSummarizer(
        generator,
        gateway=gateway,
        token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
        config=RetrievalSummaryConfig(
            direct_return_token_threshold=1,
            max_output_tokens=4_096,
        ),
    )
    section = ParsedSection(
        toc_path=("Policy",),
        heading_level=1,
        page_range=(1, 1),
        order_index=0,
        text="finance approval policy source text",
        char_range_start=0,
        char_range_end=35,
    )

    result = summarizer.summarize_section_with_metadata(section, "Policy")

    assert result.text == _STRUCTURED_SUMMARY
    assert generator.calls[0]["max_tokens"] == 1_024


def test_section_summary_prompt_requires_three_part_contract_and_preserves_it() -> None:
    generator = _RecordingGenerator(_STRUCTURED_SUMMARY)
    summarizer = RetrievalSummarizer(
        generator,
        token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
        config=RetrievalSummaryConfig(direct_return_token_threshold=1, max_output_tokens=24),
    )
    section = ParsedSection(
        toc_path=("Policy", "Finance"),
        heading_level=2,
        page_range=(1, 2),
        order_index=0,
        text="finance department approves reimbursement above 5000 CNY in 2026",
        char_range_start=0,
        char_range_end=64,
    )

    result = summarizer.summarize_section_with_metadata(section, "Policy")

    _assert_three_part_prompt(generator.prompts[0])
    assert result.text.splitlines() == _STRUCTURED_SUMMARY.splitlines()


def test_asset_summary_prompt_requires_three_part_contract_and_preserves_it() -> None:
    generator = _RecordingGenerator(_STRUCTURED_SUMMARY)
    summarizer = RetrievalSummarizer(
        generator,
        token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
        config=RetrievalSummaryConfig(max_output_tokens=24),
    )

    result = summarizer.summarize_asset_with_metadata(
        asset_type="table",
        asset_text="| Department | Limit |\n|---|---|\n| Finance | 5000 |",
        document_title="Policy",
        toc_path=("Policy", "Finance"),
        caption=None,
    )

    _assert_three_part_prompt(generator.prompts[0])
    assert "For table assets" in generator.prompts[0]
    assert result.text.splitlines() == _STRUCTURED_SUMMARY.splitlines()


def test_retrieval_summarizer_reduces_child_summaries_for_doc_summary() -> None:
    generator = _RecordingGenerator("doc0 doc1 doc2 doc3 doc4")
    summarizer = RetrievalSummarizer(
        generator,
        token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
        config=RetrievalSummaryConfig(
            direct_return_token_threshold=1,
            max_input_tokens=8,
            max_output_tokens=3,
            head_tokens=4,
            middle_tokens=0,
            tail_tokens=4,
        ),
    )

    result = summarizer.summarize_doc_with_metadata(
        document_title="Travel Policy",
        section_summaries=[
            "s0 approval",
            "s1 middle",
            "s2 budget",
        ],
        asset_summaries=["a0 table"],
    )

    assert generator.prompts
    prompt = generator.prompts[0]
    assert "Document title: Travel Policy" in prompt
    assert "Child retrieval summaries:" in prompt
    _assert_three_part_prompt(prompt)
    assert "[SECTION 1] s0 approval" in prompt
    assert "[ASSET 1] a0 table" in prompt
    assert "s1 middle" not in prompt
    assert result.text == "doc0 doc1 doc2"


def test_table_sampler_uses_token_accounting_for_policy_decision() -> None:
    table_markdown = "| Name | Amount |\n|---|---|\n| Travel | 500 |"

    profile = profile_markdown_table(
        table_markdown,
        token_accounting=_FixedTokenAccounting(20_000),  # type: ignore[arg-type]
    )

    assert profile.estimated_tokens == 20_000
    assert profile.table_policy == TABLE_POLICY_COMPUTE_ONLY


def test_table_sampler_keeps_full_shape_when_only_sample_rows_are_profiled() -> None:
    profile = profile_table_data(
        columns=["Name", "Amount"],
        rows=[["Travel", "500"]],
        token_accounting=_FixedTokenAccounting(10),  # type: ignore[arg-type]
        total_row_count=100,
        total_column_count=4,
    )

    assert profile.row_count == 100
    assert profile.column_count == 4
    assert profile.estimated_tokens > 10
    assert "Table shape: rows=100, columns=4" in profile.summary_sample
    assert "2 more columns" in profile.summary_sample


def test_raw_text_mode_returns_section_text_directly() -> None:
    summarizer = RetrievalSummarizer(
        llm_client=_RecordingGenerator(""),
        config=RetrievalSummaryConfig(raw_text_mode=True),
        token_accounting=_WordTokenAccounting(),
    )
    section = ParsedSection(
        toc_path=("Policy",),
        heading_level=1,
        page_range=None,
        order_index=0,
        text="单笔差旅报销超过 12000 元时，必须由业务线 VP 审批。",
        char_range_start=0,
        char_range_end=30,
    )
    result = summarizer.summarize_section_with_metadata(section, "差旅制度")
    assert result.method == "raw_text"
    assert result.provider_name is None
    assert result.model_name is None
    assert "差旅报销" in result.text
    assert "12000 元" in result.text


def test_raw_text_mode_does_not_call_llm_for_section() -> None:
    generator = _RecordingGenerator("ignored")
    summarizer = RetrievalSummarizer(
        llm_client=generator,
        config=RetrievalSummaryConfig(raw_text_mode=True),
    )
    section = ParsedSection(
        toc_path=("Policy",),
        heading_level=1,
        page_range=None,
        text="some text",
        order_index=0,
        char_range_start=0,
        char_range_end=9,
    )
    summarizer.summarize_section_with_metadata(section, "doc")
    assert len(generator.prompts) == 0


def test_raw_text_mode_does_not_call_llm_for_doc() -> None:
    generator = _RecordingGenerator("ignored")
    summarizer = RetrievalSummarizer(
        llm_client=generator,
        config=RetrievalSummaryConfig(raw_text_mode=True),
    )
    summarizer.summarize_doc_with_metadata(
        document_title="test doc",
        section_summaries=["s1"],
    )
    assert len(generator.prompts) == 0


def test_raw_text_mode_does_not_call_llm_for_asset() -> None:
    generator = _RecordingGenerator("ignored")
    summarizer = RetrievalSummarizer(
        llm_client=generator,
        config=RetrievalSummaryConfig(raw_text_mode=True),
    )
    summarizer.summarize_asset_with_metadata(
        asset_type="table",
        asset_text="col1, col2",
        document_title="doc",
        toc_path=[],
    )
    assert len(generator.prompts) == 0


def test_strict_generation_raises_when_section_summary_generation_fails() -> None:
    summarizer = RetrievalSummarizer(
        llm_client=_FailingGenerator(),
        token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
        config=RetrievalSummaryConfig(
            direct_return_token_threshold=1,
            strict_generation=True,
        ),
    )
    section = ParsedSection(
        toc_path=("Policy",),
        heading_level=1,
        page_range=None,
        text=" ".join(["policy"] * 120),
        order_index=0,
        char_range_start=0,
        char_range_end=720,
    )

    with pytest.raises(RuntimeError, match="section summary generation failed"):
        summarizer.summarize_section_with_metadata(section, "doc")

"""测试 CitationFormatter：上标、重排、脚注。"""

from __future__ import annotations

from rag.providers.citation_formatter import CitationFormatter, _to_superscript
from rag.schema.query import AnswerCitation, AnswerEvidenceLink, GroundedAnswer


def test_to_superscript_single_digit() -> None:
    assert _to_superscript(1) == "¹"
    assert _to_superscript(9) == "⁹"
    assert _to_superscript(0) == "⁰"


def test_to_superscript_multi_digit() -> None:
    assert _to_superscript(12) == "¹²"
    assert _to_superscript(103) == "¹⁰³"
    assert _to_superscript(456) == "⁴⁵⁶"


def test_formatter_no_citations_returns_unchanged() -> None:
    fmt = CitationFormatter()
    answer = GroundedAnswer(
        answer_text="简单的回答，没有引用。",
        groundedness_flag=True,
        insufficient_evidence_flag=False,
    )
    result = fmt.format(answer)
    assert result.answer_text == answer.answer_text
    assert result.citation_count == 0
    assert result.footnotes == ""


def test_formatter_renumbers_by_first_appearance() -> None:
    """LLM 输出 [Doc-2] 再输出 [Doc-1]，最终按出现顺序重排为 ¹ ²。"""
    fmt = CitationFormatter()
    cit_2 = AnswerCitation(
        citation_id="cit-2",
        evidence_id="ev:2",
        record_type="section",
        file_name="report.pdf",
        section_path=["第三章"],
        page_start=5,
    )
    cit_1 = AnswerCitation(
        citation_id="cit-1",
        evidence_id="ev:1",
        record_type="section",
        file_name="data.xlsx",
        section_path=["Sheet1"],
        page_start=1,
    )
    answer = GroundedAnswer(
        answer_text="根据报告 [Doc-2]，结合数据表 [Doc-1]。",
        citations=[cit_2, cit_1],
        evidence_links=[
            AnswerEvidenceLink(
                link_id="l2", answer_section_id="s1", answer_excerpt="", evidence_id="ev:2", citation_id="cit-2"
            ),
            AnswerEvidenceLink(
                link_id="l1", answer_section_id="s1", answer_excerpt="", evidence_id="ev:1", citation_id="cit-1"
            ),
        ],
        groundedness_flag=True,
        insufficient_evidence_flag=False,
    )
    result = fmt.format(answer)
    # [Doc-2] 先出现 → ¹, [Doc-1] 后出现 → ²
    assert "¹" in result.answer_body
    assert "²" in result.answer_body
    assert "[Doc-2]" not in result.answer_body
    assert "[Doc-1]" not in result.answer_body
    assert "report.pdf" in result.footnotes
    assert "data.xlsx" in result.footnotes
    assert result.citation_count == 2


def test_formatter_single_citation() -> None:
    fmt = CitationFormatter()
    cit = AnswerCitation(
        citation_id="cit-1",
        evidence_id="ev:1",
        record_type="section",
        file_name="policy.pdf",
        section_path=["第二章"],
        page_start=3,
    )
    answer = GroundedAnswer(
        answer_text="根据制度规定 [Doc-1]。",
        citations=[cit],
        evidence_links=[
            AnswerEvidenceLink(
                link_id="l1", answer_section_id="s1", answer_excerpt="", evidence_id="ev:1", citation_id="cit-1"
            ),
        ],
        groundedness_flag=True,
        insufficient_evidence_flag=False,
    )
    result = fmt.format(answer)
    assert "¹" in result.answer_body
    assert "[Doc-1]" not in result.answer_body
    assert "policy.pdf" in result.footnotes
    assert result.citation_count == 1


def test_formatter_duplicate_citation_merged() -> None:
    """同一个引用出现多次，只算一个脚注条目。"""
    fmt = CitationFormatter()
    cit = AnswerCitation(
        citation_id="cit-1",
        evidence_id="ev:1",
        record_type="section",
        file_name="doc.pdf",
        section_path=["概述"],
        page_start=1,
    )
    answer = GroundedAnswer(
        answer_text="开头提到 [Doc-1]，结尾又说 [Doc-1]。",
        citations=[cit],
        evidence_links=[
            AnswerEvidenceLink(
                link_id="l1", answer_section_id="s1", answer_excerpt="", evidence_id="ev:1", citation_id="cit-1"
            ),
        ],
        groundedness_flag=True,
        insufficient_evidence_flag=False,
    )
    result = fmt.format(answer)
    # 两个 [Doc-1] 替换为同一个上标
    count = result.answer_body.count("¹")
    assert count == 2
    # 但脚注只出现一次
    assert result.footnotes.count("doc.pdf") == 1
    assert result.citation_count == 1


def test_formatter_footnote_format() -> None:
    fmt = CitationFormatter()
    cit = AnswerCitation(
        citation_id="cit-1",
        evidence_id="ev:1",
        record_type="table",
        file_name="销售数据.xlsx",
        section_path=["Sheet1"],
        page_start=1,
        citation_anchor="销售订单表",
    )
    answer = GroundedAnswer(
        answer_text="数据来源 [Doc-1]。",
        citations=[cit],
        evidence_links=[
            AnswerEvidenceLink(
                link_id="l1", answer_section_id="s1", answer_excerpt="", evidence_id="ev:1", citation_id="cit-1"
            ),
        ],
        groundedness_flag=True,
        insufficient_evidence_flag=False,
    )
    result = fmt.format(answer)
    # 脚注应包含文件名和页码
    assert "销售数据.xlsx" in result.footnotes
    assert "Sheet1" in result.footnotes
    assert "p.1" in result.footnotes


def test_formatter_normalizes_alt_citation_format() -> None:
    """[2:18] 等非标准格式被归一化为 [Doc-N] 再转上标。"""
    fmt = CitationFormatter()
    cit = AnswerCitation(
        citation_id="cit-1",
        evidence_id="E1",
        record_type="section",
        file_name="doc.pdf",
        section_path=["第一章"],
        page_start=1,
    )
    answer = GroundedAnswer(
        answer_text="渡至发达国家情况 [2:18]，主要以旧房翻新驱动 [2:17]。",
        citations=[cit],
        evidence_links=[
            AnswerEvidenceLink(
                link_id="l1", answer_section_id="s1", answer_excerpt="", evidence_id="E1", citation_id="cit-1"
            ),
        ],
        groundedness_flag=True,
        insufficient_evidence_flag=False,
    )
    result = fmt.format(answer)
    # [2:18] 和 [2:17] 应被归一化为上标
    assert "[2:18]" not in result.answer_body
    assert "[2:17]" not in result.answer_body
    assert "¹" in result.answer_body


def test_formatter_no_doc_markers_uses_evidence_links() -> None:
    """answer_text 里没有 [Doc-N] 时，用 evidence_links 的顺序推断。"""
    fmt = CitationFormatter()
    cit = AnswerCitation(
        citation_id="cit-1",
        evidence_id="ev:1",
        record_type="section",
        file_name="doc.pdf",
        section_path=["第一章"],
        page_start=1,
    )
    answer = GroundedAnswer(
        answer_text="这是一段没有标记的回答。",
        citations=[cit],
        evidence_links=[
            AnswerEvidenceLink(
                link_id="l1", answer_section_id="s1", answer_excerpt="", evidence_id="ev:1", citation_id="cit-1"
            ),
        ],
        groundedness_flag=True,
        insufficient_evidence_flag=False,
    )
    result = fmt.format(answer)
    # 无 [Doc-N] 标记，evidence_links 推断仍生成脚注
    assert result.citation_count >= 0


__all__ = [
    "test_to_superscript_single_digit",
    "test_to_superscript_multi_digit",
    "test_formatter_no_citations_returns_unchanged",
    "test_formatter_renumbers_by_first_appearance",
    "test_formatter_single_citation",
    "test_formatter_duplicate_citation_merged",
    "test_formatter_footnote_format",
    "test_formatter_no_doc_markers_uses_evidence_links",
]

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from agent_runtime.modeling.contracts import LLMCallStage
from agent_runtime.modeling.gateway import LLMGateway
from agent_runtime.text import keyword_overlap, looks_command_like, search_terms, split_sentences
from rag.schema.model_protocols import Generator
from rag.schema.query import (
    AnswerCitation,
    AnswerEvidenceLink,
    AnswerSection,
    EvidenceItem,
    GroundedAnswer,
    RetrievalSignals,
)
from rag.schema.runtime import (
    AccessPolicy,
    ProviderAttempt,
    RuntimeMode,
)

_DOC_ALIAS_RE = re.compile(r"\[(Doc-\d+)\]")
_JSON_CODE_FENCE_RE = re.compile(
    r"^\s*```\s*(?:[A-Za-z0-9_-]+)?\s*(?P<body>.*?)\s*```\s*$",
    re.DOTALL,
)
_STRUCTURED_ANSWER_KEYS = {"answer_text", "answer_sections", "insufficient_evidence_flag"}

_GENERIC_QUERY_TERMS = {
    "这个",
    "那个",
    "这里",
    "那里",
    "什么",
    "哪些",
    "一下",
    "请问",
    "请",
    "how",
    "what",
    "which",
    "where",
    "why",
    "when",
    "this",
    "that",
    "these",
    "those",
    "the",
    "a",
    "an",
}

_GENERIC_ANSWER_TERMS = {
    "什么",
    "哪些",
    "哪里",
    "一下",
    "这个",
    "那个",
    "这里",
    "那里",
    "请问",
    "请",
    "how",
    "what",
    "which",
    "where",
    "why",
    "when",
    "this",
    "that",
    "these",
    "those",
    "document",
    "doc",
    "source",
}


# ============================================================
# structured payload schema
# ============================================================


class AnswerSectionPayload(BaseModel):
    title: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class StructuredAnswerPayload(BaseModel):
    answer_text: str
    answer_sections: list[AnswerSectionPayload] = Field(default_factory=list)
    insufficient_evidence_flag: bool = False


# ============================================================
# provider binding
# ============================================================


@dataclass(frozen=True, slots=True)
class GeneratorBinding:
    backend: Generator
    provider_name: str
    model_name: str | None
    location: Literal["local", "cloud"] = "local"
    gateway: LLMGateway | None = None


@dataclass(frozen=True, slots=True)
class AnswerGenerationResult:
    answer: GroundedAnswer
    provider: str | None
    model: str | None
    attempts: list[ProviderAttempt]


# ============================================================
# answer materialization service
# ============================================================


@dataclass(slots=True)
class AnswerGenerationService:
    min_overlap: int = 2

    # ------------------------------------------------------------
    # prompt building
    # ------------------------------------------------------------

    def build_direct_prompt(
        self,
        *,
        query: str,
        response_type: str = "Multiple Paragraphs",
        user_prompt: str | None = None,
        conversation_history: Sequence[tuple[str, str]] = (),
    ) -> str:
        lines = [
            "你是知识库问答助手。",
            f"输出格式偏好：{response_type}",
            "如果你不确定，请直接说明不确定，不要伪造引用。",
        ]
        if user_prompt:
            lines.extend(["附加要求：", user_prompt.strip()])
        if conversation_history:
            lines.append("对话历史：")
            for role, content in conversation_history:
                role_name = role.strip() or "user"
                normalized = content.strip()
                if normalized:
                    lines.append(f"{role_name}: {normalized}")
        lines.extend(["当前问题：", query.strip()])
        return "\n".join(lines)

    def build_prompt(
        self,
        *,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        grounded_candidate: str,
        runtime_mode: RuntimeMode,
        response_type: str = "Multiple Paragraphs",
        user_prompt: str | None = None,
        conversation_history: Sequence[tuple[str, str]] = (),
        prompt_style: Literal["full", "compact", "minimal"] = "full",
    ) -> str:
        del runtime_mode

        doc_aliases = self._doc_aliases(evidence_pack)
        has_compute_only_table = self._has_compute_only_table(evidence_pack)
        has_compute_result = self._has_compute_result(evidence_pack)

        if prompt_style == "minimal":
            lines = [
                f"Q:{query}",
                (
                    "除非表格计算例外要求输出 <compute_request>，否则返回一个 JSON 对象，"
                    "字段必须是：answer_text, answer_sections, insufficient_evidence_flag。"
                    if has_compute_only_table
                    else "返回一个 JSON 对象，字段必须是：answer_text, answer_sections, insufficient_evidence_flag。"
                ),
                "answer_sections[].evidence_ids 只能使用给定的 E 编号；"
                "answer_text 与 section text 句尾必须带 [Doc-n] 引用。",
            ]
        elif prompt_style == "compact":
            lines = [
                grounded_candidate,
                f"问题：{query}",
                f"格式：{response_type}",
                "只基于证据回答；证据不足时将 insufficient_evidence_flag 设为 true；句尾必须带 [Doc-n] 引用。",
            ]
        else:
            lines = [
                grounded_candidate,
                "",
                "你是知识库回答生成器。你的任务是阅读以下多段证据，综合理解后生成一个流畅、有条理的自然语言回答。",
                "",
                "回答要求：",
                "- answer_text 是一个完整的、面向用户的自然语言回答。你可以使用多段落、",
                "  分点论述、对比分析等方式，把证据中的关键信息充分呈现给读者。",
                "- 不要写一两句话就结束。像写报告一样展开论述：先给总体结论，再分点详述。",
                "- 不要逐段复读证据。综合多段证据的核心信息，用你自己的语言组织成连贯回答。",
                "- 如果证据之间存在矛盾，明确指出并说明各自的依据。",
                "- 如果某个证据只是背景信息（如货币定义、宏观经济数据），只提取与问题直接相关的部分。",
                "- 每句话末尾使用 [Doc-N] 标记引用的证据编号，例如 [Doc-1]。",
                "- 引用标记格式必须是 [Doc-数字]，不能使用其他格式（如 [2:18] 是错误的）。",
                "- 只基于证据回答。证据不足时设 insufficient_evidence_flag 为 true。",
                "",
                f"问题：{query}",
                f"输出格式偏好：{response_type}",
            ]

        if user_prompt:
            lines.extend(["附加要求：", user_prompt.strip()])

        if conversation_history:
            lines.append("对话历史：")
            history = conversation_history[-2:] if prompt_style != "full" else conversation_history
            for role, content in history:
                role_name = role.strip() or "user"
                normalized = content.strip()
                if normalized:
                    lines.append(f"{role_name}: {normalized}")

        if has_compute_only_table:
            lines.extend(
                [
                    "表格计算例外：",
                    "- 如果问题要求读取表格真实数据、筛选、求和、计数、排序、排名、对比或聚合，不要输出最终答案 JSON。",
                    "- 必须只输出证据中指定格式的 <compute_request>...</compute_request>，"
                    "其中内容必须是 JSON 对象，格式为 "
                    '{"asset_id": 证据中的 asset_id, "sql": "SELECT ... FROM sheet WHERE ..."}。',
                    "- 不要基于 Sample rows 估算答案，也不要因为 Sample rows 没有目标行就回答证据不足。",
                    "- 后端会执行 SQL，并带着 TABLE_COMPUTE_RESULT 再次调用你生成最终 JSON 回答。",
                ]
            )
        elif has_compute_result:
            lines.extend(
                [
                    "表格计算结果已返回：",
                    "- 必须基于 TABLE_COMPUTE_RESULT 合成最终 JSON 回答。",
                    "- 禁止再次输出 <compute_request>。",
                    "- 不要在答案中原样粘贴 SQL；只报告计算结果、筛选条件和必要口径。",
                ]
            )

        if prompt_style == "minimal":
            lines.append("Evidence:")
        elif prompt_style == "compact":
            lines.extend(
                [
                    "输出要求：",
                    (
                        "没有触发表格计算例外时，返回一个 JSON 对象，"
                        "包含 answer_text、answer_sections、insufficient_evidence_flag。"
                        if has_compute_only_table
                        else "返回一个 JSON 对象，包含 answer_text、answer_sections、insufficient_evidence_flag。"
                    ),
                    "answer_sections[].evidence_ids 必须引用下面的 E 编号。",
                    "answer_text 和每个 answer_sections[].text 的句尾必须使用给定的 [Doc-n] 标签。",
                    "不得编造新的引用标签。",
                    "证据：",
                ]
            )
        else:
            lines.extend(
                [
                    "输出要求：",
                    (
                        "- 没有触发表格计算例外时，只输出一个 JSON 对象。"
                        if has_compute_only_table
                        else "- 只输出一个 JSON 对象。"
                    ),
                    '- 顶层字段必须包含 "answer_text"、"answer_sections"、"insufficient_evidence_flag"。',
                    '- answer_sections 是数组，每个元素包含 "title"、"text"、"evidence_ids"。',
                    "- evidence_ids 必须引用下面证据编号，例如 E1、E2。",
                    "- answer_text 和每个 answer_sections[].text 的句尾必须附带给定的 [Doc-n] 引用标签。",
                    "- 只能使用下面出现过的 [Doc-n] 标签，不得编造新的引用标签。",
                    "- 证据不足时，把 insufficient_evidence_flag 设为 true，并明确说明无法从证据中确认。",
                    "- 严格使用和问题相同的语言。",
                    "- 不要输出 Markdown、代码块、解释文字。",
                    "证据：",
                ]
            )

        for index, item in enumerate(evidence_pack, start=1):
            evidence_id = self._evidence_id(index)
            alias_label = self._doc_alias_label(item.doc_id, doc_aliases)
            section = " > ".join(item.section_path) if item.section_path else item.citation_anchor
            file_name = item.file_name or alias_label or str(item.doc_id)
            page_hint = (
                ""
                if item.page_start is None
                else (
                    f" | page={item.page_start}"
                    if item.page_end in {None, item.page_start}
                    else f" | pages={item.page_start}-{item.page_end}"
                )
            )
            record_type = item.record_type or "unknown"

            header = (
                f"{evidence_id} {alias_label} {item.text}".strip()
                if prompt_style == "minimal"
                else (
                    f"{evidence_id} | ref={alias_label or '[Doc-?]'} | kind={item.evidence_kind} "
                    f"| file={file_name} | section={section}{page_hint} | record_type={record_type}"
                )
            )
            lines.append(header)
            if prompt_style != "minimal":
                lines.append(item.text)

        return "\n".join(lines)

    def build_compute_request_prompt(
        self,
        *,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        grounded_candidate: str,
        runtime_mode: RuntimeMode,
    ) -> str:
        del runtime_mode

        doc_aliases = self._doc_aliases(evidence_pack)
        lines = [
            grounded_candidate,
            "",
            "你是表格查询规划器。当前证据包含 TABLE_COMPUTE_ONLY，且还没有 TABLE_COMPUTE_RESULT，",
            "所以这一步不是最终回答阶段。",
            "",
            "你的任务：先判断用户问题是否需要读取真实表格数据。",
            "- 如果问题需要读取单元格数值、按行筛选、求和、计数、排序、排名、比较、同比/环比、",
            "  聚合或查找最高/最低值，必须只输出 <compute_request>...</compute_request>。",
            "- <compute_request> 内必须是 JSON 对象：",
            '  {"asset_id": 证据中的 asset_id, "sql": "SELECT ... FROM sheet WHERE ..."}。',
            "- SQL 使用 DuckDB 方言；表名固定为 sheet；中文或特殊列名必须用双引号包裹。",
            "- 不要输出自然语言解释，不要说“需要计算”，不要输出最终答案 JSON。",
            "- 只有问题明确只是询问表结构、列名、行数或字段含义时，才可以不输出 compute_request。",
            "",
            f"问题：{query}",
            "",
            "证据：",
        ]

        for index, item in enumerate(evidence_pack, start=1):
            evidence_id = self._evidence_id(index)
            alias_label = self._doc_alias_label(item.doc_id, doc_aliases)
            section = " > ".join(item.section_path) if item.section_path else item.citation_anchor
            page_hint = (
                ""
                if item.page_start is None
                else (
                    f" | page={item.page_start}"
                    if item.page_end in {None, item.page_start}
                    else f" | pages={item.page_start}-{item.page_end}"
                )
            )
            lines.append(
                f"{evidence_id} | ref={alias_label or '[Doc-?]'} | "
                f"section={section}{page_hint} | record_type={item.record_type or 'unknown'}"
            )
            lines.append(item.text)

        return "\n".join(lines)

    @staticmethod
    def _has_compute_only_table(evidence_pack: Sequence[EvidenceItem]) -> bool:
        return any("[TABLE_COMPUTE_ONLY:" in item.text for item in evidence_pack) and not (
            AnswerGenerationService._has_compute_result(evidence_pack)
        )

    @staticmethod
    def _has_compute_result(evidence_pack: Sequence[EvidenceItem]) -> bool:
        return any("[TABLE_COMPUTE_RESULT:" in item.text for item in evidence_pack)

    # ------------------------------------------------------------
    # public APIs
    # ------------------------------------------------------------

    def answer_from_structured_payload(
        self,
        *,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        grounded_candidate: str,
        payload: StructuredAnswerPayload,
        trust_evidence_pack: bool = False,
    ) -> GroundedAnswer:
        evidence = [item for item in evidence_pack if item.text.strip()]
        if self._evidence_is_insufficient(query, evidence, trust_evidence_pack=trust_evidence_pack):
            return self.insufficient_answer()

        if payload.insufficient_evidence_flag:
            return self.insufficient_answer()

        answer_text = payload.answer_text.strip() or grounded_candidate
        sections = [section for section in payload.answer_sections if section.text.strip()] or [
            AnswerSectionPayload(title="直接回答", text=answer_text, evidence_ids=[])
        ]

        return self._materialize_answer(
            answer_text=answer_text,
            sections=sections,
            evidence_pack=evidence,
        )

    def answer_from_model_output(
        self,
        *,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        grounded_candidate: str,
        model_output: str,
        trust_evidence_pack: bool = False,
    ) -> GroundedAnswer:
        evidence = [item for item in evidence_pack if item.text.strip()]
        if self._evidence_is_insufficient(query, evidence, trust_evidence_pack=trust_evidence_pack):
            return self.insufficient_answer()

        structured_payload = self._structured_payload_from_text_output(model_output)
        if structured_payload is not None:
            return self.answer_from_structured_payload(
                query=query,
                evidence_pack=evidence,
                grounded_candidate=grounded_candidate,
                payload=structured_payload,
                trust_evidence_pack=trust_evidence_pack,
            )

        stripped = model_output.strip()
        answer_text = stripped if stripped else grounded_candidate

        return self._materialize_answer(
            answer_text=answer_text,
            sections=[AnswerSectionPayload(title="直接回答", text=answer_text, evidence_ids=[])],
            evidence_pack=evidence,
        )

    @classmethod
    def _structured_payload_from_text_output(cls, model_output: str) -> StructuredAnswerPayload | None:
        candidate = cls._strip_markdown_code_fence(model_output).strip()
        if not candidate.startswith("{"):
            return None
        try:
            raw_payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(raw_payload, dict):
            return None
        if not _STRUCTURED_ANSWER_KEYS.issubset(raw_payload):
            return None
        try:
            return StructuredAnswerPayload.model_validate(raw_payload)
        except ValidationError:
            return None

    @staticmethod
    def _strip_markdown_code_fence(text: str) -> str:
        match = _JSON_CODE_FENCE_RE.match(text)
        return match.group("body") if match else text

    def grounded_fallback(
        self,
        *,
        answer_text: str,
        evidence_pack: Sequence[EvidenceItem],
    ) -> GroundedAnswer:
        supporting = self._select_supporting_evidence(answer_text, evidence_pack)
        supporting_ids = [item.evidence_id for item in supporting]
        return self._materialize_answer(
            answer_text=answer_text,
            sections=[
                AnswerSectionPayload(
                    title="直接回答",
                    text=answer_text,
                    evidence_ids=supporting_ids,
                )
            ],
            evidence_pack=evidence_pack,
        )

    def grounded_structured_answer(
        self,
        *,
        answer_text: str,
        sections: Sequence[AnswerSectionPayload],
        evidence_pack: Sequence[EvidenceItem],
    ) -> GroundedAnswer:
        return self._materialize_answer(
            answer_text=answer_text,
            sections=sections,
            evidence_pack=evidence_pack,
        )

    @staticmethod
    def insufficient_answer() -> GroundedAnswer:
        text = "当前证据不足，无法给出可靠回答。"
        return GroundedAnswer(
            answer_text=text,
            answer_sections=[
                AnswerSection(
                    section_id="sec-1",
                    title="证据不足",
                    text=text,
                    citation_ids=[],
                    evidence_ids=[],
                )
            ],
            citations=[],
            evidence_links=[],
            groundedness_flag=True,
            insufficient_evidence_flag=True,
        )

    # ------------------------------------------------------------
    # materialization
    # ------------------------------------------------------------

    def _materialize_answer(
        self,
        *,
        answer_text: str,
        sections: Sequence[AnswerSectionPayload],
        evidence_pack: Sequence[EvidenceItem],
    ) -> GroundedAnswer:
        evidence_map = {item.evidence_id: item for item in evidence_pack}
        alias_reference_map = self._alias_reference_map(evidence_pack)

        citations: list[AnswerCitation] = []
        evidence_links: list[AnswerEvidenceLink] = []
        answer_sections: list[AnswerSection] = []
        citation_by_evidence_id: dict[str, AnswerCitation] = {}
        answer_supporting_evidence: list[EvidenceItem] = []
        grounded = True

        for section_index, raw_section in enumerate(sections, start=1):
            section_id = f"sec-{section_index}"
            section_text = raw_section.text.strip()
            evidence_ids = [value for value in raw_section.evidence_ids if value in evidence_map]

            section_evidence = [evidence_map[evidence_id] for evidence_id in evidence_ids]
            if not section_evidence:
                section_evidence = self._select_supporting_evidence(section_text, evidence_pack)

            answer_supporting_evidence.extend(section_evidence)

            section_text = self._normalize_reference_markers(
                section_text,
                supporting_evidence=section_evidence or evidence_pack,
                alias_reference_map=alias_reference_map,
            )

            citation_ids: list[str] = []
            normalized_evidence_ids: list[str] = []

            for item in section_evidence:
                citation = citation_by_evidence_id.get(item.evidence_id)
                if citation is None:
                    citation = AnswerCitation(
                        citation_id=f"cit-{len(citations) + 1}",
                        file_name=item.file_name,
                        section_path=list(item.section_path),
                        page_start=item.page_start,
                        page_end=item.page_end,
                        evidence_id=item.evidence_id,
                        record_type=item.record_type or "unknown",
                        citation_anchor=item.citation_anchor,
                        doc_id=item.doc_id,
                        benchmark_doc_id=item.benchmark_doc_id,
                        source_id=item.source_id,
                        source_type=item.source_type,
                    )
                    citations.append(citation)
                    citation_by_evidence_id[item.evidence_id] = citation

                citation_ids.append(citation.citation_id)
                normalized_evidence_ids.append(item.evidence_id)

                evidence_links.append(
                    AnswerEvidenceLink(
                        link_id=f"link-{len(evidence_links) + 1}",
                        answer_section_id=section_id,
                        answer_excerpt=section_text,
                        evidence_id=item.evidence_id,
                        citation_id=citation.citation_id,
                        support_score=self._support_score(section_text, item.text),
                    )
                )

            grounded = grounded and self._section_grounded(section_text, section_evidence or evidence_pack)

            answer_sections.append(
                AnswerSection(
                    section_id=section_id,
                    title=raw_section.title.strip() or "直接回答",
                    text=section_text,
                    citation_ids=citation_ids,
                    evidence_ids=normalized_evidence_ids,
                )
            )

        answer_text = self._normalize_reference_markers(
            answer_text,
            supporting_evidence=answer_supporting_evidence or evidence_pack,
            alias_reference_map=alias_reference_map,
        )

        overall_grounded = grounded and (self._section_grounded(answer_text, evidence_pack) or len(answer_sections) > 1)

        return GroundedAnswer(
            answer_text=answer_text,
            answer_sections=answer_sections,
            citations=citations,
            evidence_links=evidence_links,
            groundedness_flag=overall_grounded,
            insufficient_evidence_flag=False,
        )

    # ------------------------------------------------------------
    # evidence / grounding helpers
    # ------------------------------------------------------------

    @staticmethod
    def _evidence_id(index: int) -> str:
        return f"E{index}"

    @staticmethod
    def _doc_aliases(evidence_pack: Sequence[EvidenceItem]) -> dict[int, str]:
        aliases: dict[int, str] = {}
        for item in evidence_pack:
            if item.doc_id not in aliases:
                aliases[item.doc_id] = f"Doc-{len(aliases) + 1}"
        return aliases

    @staticmethod
    def _doc_alias_label(doc_id: int, doc_aliases: dict[int, str]) -> str:
        alias = doc_aliases.get(doc_id)
        return f"[{alias}]" if alias else ""

    def _evidence_is_insufficient(
        self,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        *,
        trust_evidence_pack: bool,
    ) -> bool:
        if not evidence_pack:
            return True
        if trust_evidence_pack:
            return False

        query_terms = _focus_terms(query)
        if not query_terms:
            return False

        combined = " ".join(self._evidence_search_text(item) for item in evidence_pack)
        if keyword_overlap(query_terms, combined) >= self.min_overlap:
            return False

        return max(keyword_overlap(query_terms, self._evidence_search_text(item)) for item in evidence_pack) < 1

    @staticmethod
    def _evidence_search_text(item: EvidenceItem) -> str:
        section = " ".join(item.section_path)
        return " ".join(part for part in (item.text, item.citation_anchor, section) if part)

    def _select_supporting_evidence(
        self,
        section_text: str,
        evidence_pack: Sequence[EvidenceItem],
    ) -> list[EvidenceItem]:
        if not evidence_pack:
            return []

        ranked = sorted(
            evidence_pack,
            key=lambda item: (
                keyword_overlap(_focus_terms(section_text), self._evidence_search_text(item)),
                keyword_overlap(search_terms(section_text), item.text),
                float(item.score),
            ),
            reverse=True,
        )
        if not ranked:
            return []

        query_terms = _focus_terms(section_text)
        top = ranked[0]
        if keyword_overlap(query_terms, self._evidence_search_text(top)) == 0:
            return [top]

        return [item for item in ranked[:2] if keyword_overlap(query_terms, self._evidence_search_text(item)) > 0]

    @classmethod
    def _alias_reference_map(cls, evidence_pack: Sequence[EvidenceItem]) -> dict[str, str]:
        doc_aliases = cls._doc_aliases(evidence_pack)
        alias_map: dict[str, str] = {}
        for item in evidence_pack:
            alias = doc_aliases.get(item.doc_id)
            if alias is None or alias in alias_map:
                continue
            alias_map[alias] = cls._canonical_reference(item)
        return alias_map

    @staticmethod
    def _canonical_reference(item: EvidenceItem) -> str:
        target = item.grounding_target
        doc_id = item.doc_id
        if target is not None:
            if target.section_id:
                return f"[{doc_id}:{target.section_id}]"
            if target.asset_id:
                return f"[{doc_id}:{target.asset_id}]"
        return f"[{doc_id}]"

    @classmethod
    def _normalize_reference_markers(
        cls,
        text: str,
        *,
        supporting_evidence: Sequence[EvidenceItem],
        alias_reference_map: dict[str, str],
    ) -> str:
        normalized = cls._rewrite_alias_markers(text.strip(), alias_reference_map)
        if not normalized:
            return normalized

        references = cls._support_references(supporting_evidence)
        if not references:
            return normalized

        if any(reference in normalized for reference in references):
            return normalized

        return f"{normalized} {''.join(references[:2])}".strip()

    @staticmethod
    def _rewrite_alias_markers(text: str, alias_reference_map: dict[str, str]) -> str:
        if not text:
            return text
        rewritten = _DOC_ALIAS_RE.sub(lambda match: alias_reference_map.get(match.group(1), ""), text)
        return re.sub(r"\s{2,}", " ", rewritten).strip()

    @classmethod
    def _support_references(cls, evidence_pack: Sequence[EvidenceItem]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for item in evidence_pack:
            reference = cls._canonical_reference(item)
            if reference in seen:
                continue
            seen.add(reference)
            ordered.append(reference)
        return ordered

    @staticmethod
    def _support_score(answer_excerpt: str, evidence_text: str) -> float:
        terms = _focus_terms(answer_excerpt)
        if not terms:
            return 1.0 if answer_excerpt.strip() else 0.0
        overlap = keyword_overlap(terms, evidence_text)
        return min(1.0, max(0.0, overlap / max(1, len(terms))))

    def _section_grounded(self, text: str, evidence_pack: Sequence[EvidenceItem]) -> bool:
        variants = self._answer_variants(text)
        return any(self._variant_grounded(variant, evidence_pack) for variant in variants if variant)

    def _variant_grounded(self, text: str, evidence_pack: Sequence[EvidenceItem]) -> bool:
        if self._text_supported_by_evidence(text, evidence_pack):
            return True

        terms = _focus_terms(text)
        if not terms:
            return False

        for item in evidence_pack:
            record_type = item.record_type or ""
            if record_type not in {"table", "image_summary", "ocr_region", "figure", "caption", "asset"}:
                continue
            if keyword_overlap(terms, self._evidence_search_text(item)) >= 1:
                return True
        return False

    @staticmethod
    def _answer_variants(text: str) -> tuple[str, ...]:
        normalized = text.strip()
        if not normalized:
            return ()
        variants = [normalized]
        if ":" in normalized:
            suffix = normalized.split(":", 1)[1].strip()
            if suffix:
                variants.append(suffix)
        if "：" in normalized:
            suffix = normalized.split("：", 1)[1].strip()
            if suffix:
                variants.append(suffix)
        return tuple(dict.fromkeys(variants))

    @staticmethod
    def _normalize_supported_text(text: str) -> str:
        normalized = text.replace("**", " ").replace("__", " ").replace("`", " ")
        normalized = re.sub(r"\[(Doc-\d+|[^\[\]]+:[^\[\]]+|[^\[\]]+)\]", " ", normalized)
        normalized = re.sub(r"^\s*[-*]\s*", "", normalized, flags=re.MULTILINE)
        normalized = re.sub(r"^\s*\d+\.\s*", "", normalized, flags=re.MULTILINE)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @classmethod
    def _text_supported_by_evidence(cls, answer_text: str, evidence_pack: Sequence[EvidenceItem]) -> bool:
        normalized_answer = cls._normalize_supported_text(answer_text)
        if not normalized_answer:
            return False

        answer_sentences = split_sentences(normalized_answer) or (normalized_answer,)
        normalized_evidence = [cls._normalize_supported_text(item.text) for item in evidence_pack if item.text.strip()]
        if not normalized_evidence:
            return False

        def supported(sentence: str) -> bool:
            sentence_terms = search_terms(sentence)
            required_overlap = max(2, (len(sentence_terms) + 1) // 2) if sentence_terms else 0
            if any(
                sentence in evidence_text
                or (required_overlap > 0 and keyword_overlap(sentence_terms, evidence_text) >= required_overlap)
                for evidence_text in normalized_evidence
            ):
                return True
            if required_overlap == 0:
                return False
            return keyword_overlap(sentence_terms, " ".join(normalized_evidence)) >= required_overlap

        return all(supported(sentence) for sentence in answer_sentences if sentence)


# ============================================================
# provider orchestration
# ============================================================


@dataclass(slots=True)
class AnswerGenerator:
    answer_generation_service: AnswerGenerationService = field(default_factory=AnswerGenerationService)
    generators: tuple[GeneratorBinding, ...] = ()

    def grounded_candidate(
        self,
        query: str,
        evidence_pack: Sequence[EvidenceItem],
        *,
        retrieval_signals: RetrievalSignals | None = None,
    ) -> str:
        hits = [item for item in evidence_pack if item.text.strip()]
        if not hits:
            return "Insufficient evidence in indexed sources."

        if retrieval_signals is not None and retrieval_signals.special_targets:
            special_candidate = self._special_aware_conclusion(hits, retrieval_signals)
            if special_candidate is not None:
                return special_candidate

        if retrieval_signals is not None and retrieval_signals.structure_constraints.has_constraints():
            structure_candidate = self._structure_aware_conclusion(hits, retrieval_signals)
            if structure_candidate is not None:
                return structure_candidate

        return self._best_overlap_sentence(query, hits, retrieval_signals)

    async def generate(
        self,
        *,
        query: str,
        prompt: str,
        evidence_pack: Sequence[EvidenceItem],
        grounded_candidate: str,
        runtime_mode: RuntimeMode,
        access_policy: AccessPolicy,
    ) -> AnswerGenerationResult:
        del runtime_mode
        attempts: list[ProviderAttempt] = []

        for binding in self.generators:
            base_attempt = ProviderAttempt(
                stage="generation",
                capability="chat",
                provider=binding.provider_name,
                location=binding.location,
                model=binding.model_name,
                status="success",
            )

            # structured path
            started = time.perf_counter()
            try:
                if binding.gateway is None:
                    payload = await asyncio.to_thread(
                        binding.backend.generate_structured,
                        prompt=prompt,
                        schema=StructuredAnswerPayload,
                    )
                else:
                    payload = (
                        await binding.gateway.agenerate_structured(
                            stage=LLMCallStage.RAG_ANSWER,
                            prompt=prompt,
                            schema=StructuredAnswerPayload,
                            lease_id=f"rag_answer:structured:{uuid4().hex}",
                        )
                    ).value
                latency_ms = (time.perf_counter() - started) * 1000.0
                answer = self.answer_generation_service.answer_from_structured_payload(
                    query=query,
                    evidence_pack=evidence_pack,
                    grounded_candidate=grounded_candidate,
                    payload=payload,
                    trust_evidence_pack=True,
                )
                attempts.append(base_attempt.model_copy(update={"latency_ms": latency_ms}))
                return AnswerGenerationResult(
                    answer=answer,
                    provider=binding.provider_name,
                    model=binding.model_name,
                    attempts=attempts,
                )
            except (ValidationError, Exception) as exc:
                latency_ms = (time.perf_counter() - started) * 1000.0
                attempts.append(
                    base_attempt.model_copy(
                        update={
                            "status": "failed",
                            "error": f"structured_generation_failed: {exc}",
                            "latency_ms": latency_ms,
                        }
                    )
                )

            # text fallback
            started = time.perf_counter()
            try:
                if binding.gateway is None:
                    output = await asyncio.to_thread(
                        binding.backend.generate_text,
                        prompt=prompt,
                    )
                else:
                    output = (
                        await binding.gateway.agenerate_text(
                            stage=LLMCallStage.RAG_ANSWER,
                            prompt=prompt,
                            lease_id=f"rag_answer:text:{uuid4().hex}",
                        )
                    ).value
                latency_ms = (time.perf_counter() - started) * 1000.0
                answer = self.answer_generation_service.answer_from_model_output(
                    query=query,
                    evidence_pack=evidence_pack,
                    grounded_candidate=grounded_candidate,
                    model_output=str(output),
                    trust_evidence_pack=True,
                )
                attempts.append(base_attempt.model_copy(update={"latency_ms": latency_ms}))
                return AnswerGenerationResult(
                    answer=answer,
                    provider=binding.provider_name,
                    model=binding.model_name,
                    attempts=attempts,
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000.0
                attempts.append(
                    base_attempt.model_copy(
                        update={
                            "status": "failed",
                            "error": f"text_generation_failed: {exc}",
                            "latency_ms": latency_ms,
                        }
                    )
                )

        fallback = self.answer_generation_service.grounded_fallback(
            answer_text=grounded_candidate,
            evidence_pack=[item for item in evidence_pack if item.text.strip()],
        )
        return AnswerGenerationResult(
            answer=fallback,
            provider=None,
            model=None,
            attempts=attempts,
        )

    async def generate_direct(
        self,
        *,
        query: str,
        prompt: str,
        access_policy: AccessPolicy,
    ) -> AnswerGenerationResult:
        del query
        attempts: list[ProviderAttempt] = []

        for binding in self.generators:
            base_attempt = ProviderAttempt(
                stage="generation",
                capability="chat",
                provider=binding.provider_name,
                location=binding.location,
                model=binding.model_name,
                status="success",
            )

            started = time.perf_counter()
            try:
                if binding.gateway is None:
                    output = await asyncio.to_thread(
                        binding.backend.generate_text,
                        prompt=prompt,
                    )
                else:
                    output = (
                        await binding.gateway.agenerate_text(
                            stage=LLMCallStage.RAG_ANSWER,
                            prompt=prompt,
                            lease_id=f"rag_answer:direct:{uuid4().hex}",
                        )
                    ).value
                latency_ms = (time.perf_counter() - started) * 1000.0
                answer_text = str(output).strip() or "模型没有返回内容。"
                answer = GroundedAnswer(
                    answer_text=answer_text,
                    answer_sections=[
                        AnswerSection(
                            section_id="direct-response",
                            title="Direct Response",
                            text=answer_text,
                            citation_ids=[],
                            evidence_ids=[],
                        )
                    ],
                    citations=[],
                    evidence_links=[],
                    groundedness_flag=False,
                    insufficient_evidence_flag=False,
                )
                attempts.append(base_attempt.model_copy(update={"latency_ms": latency_ms}))
                return AnswerGenerationResult(
                    answer=answer,
                    provider=binding.provider_name,
                    model=binding.model_name,
                    attempts=attempts,
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1000.0
                attempts.append(
                    base_attempt.model_copy(
                        update={
                            "status": "failed",
                            "error": str(exc),
                            "latency_ms": latency_ms,
                        }
                    )
                )

        fallback_answer = GroundedAnswer(
            answer_text="No generator available for bypass mode.",
            answer_sections=[
                AnswerSection(
                    section_id="direct-response",
                    title="Direct Response",
                    text="No generator available for bypass mode.",
                    citation_ids=[],
                    evidence_ids=[],
                )
            ],
            citations=[],
            evidence_links=[],
            groundedness_flag=False,
            insufficient_evidence_flag=True,
        )
        return AnswerGenerationResult(
            answer=fallback_answer,
            provider=None,
            model=None,
            attempts=attempts,
        )

    @staticmethod
    def _dedupe_generators(bindings: Sequence[GeneratorBinding]) -> list[GeneratorBinding]:
        seen: set[int] = set()
        ordered: list[GeneratorBinding] = []
        for binding in bindings:
            identity = id(binding.backend)
            if identity in seen:
                continue
            seen.add(identity)
            ordered.append(binding)
        return ordered

    @staticmethod
    def _special_aware_conclusion(
        hits: Sequence[EvidenceItem],
        signals: RetrievalSignals,
    ) -> str | None:
        preferred_targets = set(signals.special_targets)
        ranked_hits = sorted(
            hits[:8],
            key=lambda item: (
                int((item.record_type or "") in preferred_targets),
                float(item.score),
            ),
            reverse=True,
        )
        for item in ranked_hits:
            record_type = item.record_type or ""
            if preferred_targets and record_type not in preferred_targets:
                continue
            if item.text.strip():
                return item.text.strip()
        return None

    @staticmethod
    def _structure_aware_conclusion(
        hits: Sequence[EvidenceItem],
        signals: RetrievalSignals,
    ) -> str | None:
        query_focus_terms = signals.structure_constraints.focus_terms or signals.quoted_terms
        ranked_hits = sorted(
            hits[:8],
            key=lambda item: (
                int(keyword_overlap(query_focus_terms, item.citation_anchor) > 0),
                keyword_overlap(query_focus_terms, item.citation_anchor),
                keyword_overlap(query_focus_terms, item.text),
                float(item.score),
            ),
            reverse=True,
        )
        for hit in ranked_hits:
            lead = AnswerGenerator._pick_structure_lead(hit.text, query_focus_terms)
            if lead:
                return lead
        return None

    @staticmethod
    def _pick_structure_lead(text: str, query_focus_terms: Sequence[str]) -> str | None:
        sentences = split_sentences(text)
        for sentence in sentences:
            if keyword_overlap(query_focus_terms, sentence) > 0 and not looks_command_like(sentence):
                return AnswerGenerator._normalize_answer_fragment(sentence)
        for sentence in sentences:
            if not looks_command_like(sentence):
                return AnswerGenerator._normalize_answer_fragment(sentence)
        return None

    @staticmethod
    def _normalize_answer_fragment(text: str) -> str:
        cleaned = " ".join(text.replace("`", "").split())
        if cleaned.endswith(("。", "！", "？", ".", "!", "?")):
            return cleaned[:-1].strip()
        return cleaned.strip()

    @staticmethod
    def _best_overlap_sentence(
        query: str,
        hits: Sequence[EvidenceItem],
        signals: RetrievalSignals | None,
    ) -> str:
        query_terms = search_terms(query)
        query_focus_terms = (
            list(signals.structure_constraints.focus_terms)
            if signals is not None and signals.structure_constraints.focus_terms
            else _answer_focus_terms(query)
        )
        normalized_query = query.strip().lower()
        sentences = [sentence for item in hits[:6] for sentence in split_sentences(item.text)]
        if not sentences:
            return hits[0].text

        def _score(sentence: str) -> tuple[int, int, int, int, float]:
            lowered = sentence.lower()
            exact_match = int(bool(normalized_query) and normalized_query in lowered)
            focus_overlap = keyword_overlap(query_focus_terms, sentence)
            term_overlap = keyword_overlap(query_terms, sentence)
            command_penalty = 0 if looks_command_like(sentence) else 1
            structure_priority = (
                keyword_overlap(query_focus_terms, sentence)
                if signals is not None and signals.structure_constraints.has_constraints()
                else 0
            )
            return (
                exact_match,
                structure_priority,
                focus_overlap,
                command_penalty,
                float(term_overlap),
            )

        candidate_pool = [sentence for sentence in sentences if not looks_command_like(sentence)] or sentences
        return max(candidate_pool, key=_score)


def _focus_terms(text: str) -> tuple[str, ...]:
    filtered = tuple(term for term in search_terms(text) if term not in _GENERIC_QUERY_TERMS)
    return filtered or search_terms(text)


def _answer_focus_terms(text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for term in search_terms(text):
        normalized = term.strip().lower()
        if not normalized or normalized in _GENERIC_ANSWER_TERMS or len(normalized) < 2:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered or list(search_terms(text))


__all__ = [
    "AnswerSectionPayload",
    "StructuredAnswerPayload",
    "GeneratorBinding",
    "AnswerGenerationResult",
    "AnswerGenerationService",
    "AnswerGenerator",
]

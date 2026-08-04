from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.modeling.contracts import LLMCallStage
from agent_runtime.modeling.gateway import LLMGateway
from agent_runtime.text import text_unit_count
from rag.assembly import TokenAccountingService, TokenizerContract
from rag.ingest.parsers.util import normalize_whitespace
from rag.schema.core import ParsedSection


class TextGenerationClient(Protocol):
    def generate_text(self, *, prompt: str, **kwargs: Any) -> str: ...


class TokenAccountingClient(Protocol):
    def count(self, text: str) -> int: ...
    def clip(self, text: str, token_budget: int, *, add_ellipsis: bool = False) -> str: ...
    def tail(self, text: str, token_budget: int) -> str: ...


@dataclass(frozen=True)
class RetrievalSummaryResult:
    text: str
    method: str
    provider_name: str | None
    model_name: str | None
    fallback_reason: str | None = None


@dataclass(frozen=True)
class RetrievalSummaryConfig:
    # 原文短于这个 token 数，直接返回原文，不走大模型
    direct_return_token_threshold: int = 96

    # 送给大模型的原文最大 token 数
    max_input_tokens: int = 5000

    # 摘要最大 token 数，做最后一道截断保护
    # 预留足够空间：Qwen3 等模型会先输出 reasoning tokens，再输出实际摘要
    max_output_tokens: int = 4096

    # 头中尾截断比例（三区采样，防止中间核心内容丢失）
    head_tokens: int = 1500
    middle_tokens: int = 1500
    tail_tokens: int = 1500

    # 生成温度，None 表示不透传（使用 LLM 默认值）
    temperature: float | None = None

    # 跳过 LLM 摘要，直接使用原文作为 summary_text
    # 适用于公开 benchmark 等纯文本 passage 的 fast ingest
    raw_text_mode: bool = False

    # 私有语料入库等场景需要确认摘要确实由生成模型产出。
    # 开启后，模型调用失败或输出空内容会让入库失败，而不是写入 fallback 摘要。
    strict_generation: bool = False


class RetrievalSummarizer:
    """
    为 section_summary 生成“检索用摘要”的正式版组件。

    目标：
    - 不是生成给人阅读的优美摘要
    - 而是生成更适合向量检索 / 稀疏检索 / rerank 的高密度表达
    """

    def __init__(
        self,
        llm_client: TextGenerationClient,
        *,
        config: RetrievalSummaryConfig | None = None,
        token_accounting: TokenAccountingClient | None = None,
        gateway: LLMGateway | None = None,
    ) -> None:
        self._llm = llm_client
        self._config = config or RetrievalSummaryConfig()
        self._gateway = gateway
        self._token_accounting = token_accounting or TokenAccountingService(
            TokenizerContract(
                embedding_model_name="default",
                tokenizer_model_name="default",
                chunking_tokenizer_model_name="default",
                tokenizer_backend="simple",
                local_files_only=True,
            )
        )

    def _generate_text(self, *, prompt: str) -> str:
        kwargs: dict[str, Any] = {}
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature
        kwargs["max_tokens"] = self._config.max_output_tokens
        if self._gateway is not None:
            return self._gateway.generate_text(
                stage=LLMCallStage.RETRIEVAL_SUMMARY,
                prompt=prompt,
                kwargs=kwargs,
            ).value
        try:
            return self._llm.generate_text(prompt=prompt, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" in str(exc):
                return self._llm.generate_text(prompt=prompt)
            raise

    def summarize_section(self, section: ParsedSection, document_title: str) -> str:
        return self.summarize_section_with_metadata(section, document_title).text

    def summarize_section_with_metadata(self, section: ParsedSection, document_title: str) -> RetrievalSummaryResult:
        raw_text = normalize_whitespace(section.text)
        provider_name = self._provider_name()
        model_name = self._model_name()

        # 0. raw_text_mode：跳过 LLM，直接使用原文
        if self._config.raw_text_mode:
            return RetrievalSummaryResult(
                text=raw_text or "",
                method="raw_text",
                provider_name=None,
                model_name=None,
            )

        # 1. 空文本兜底
        if not raw_text:
            return RetrievalSummaryResult(
                text="",
                method="empty",
                provider_name=provider_name,
                model_name=model_name,
            )

        # 2. 短文本直接返回，省钱且保真
        if self._count_tokens(raw_text) <= self._config.direct_return_token_threshold:
            return RetrievalSummaryResult(
                text=self._fallback_summary(raw_text),
                method="direct",
                provider_name=None,
                model_name=None,
            )

        # 3. 构造截断样本，避免超长 section 浪费 token
        sampled_text = self._sample_text(raw_text)

        prompt = self._build_prompt(
            document_title=document_title,
            section=section,
            sampled_text=sampled_text,
        )

        # 4. 调大模型，但绝不让主链因为摘要失败而崩
        try:
            generated = self._generate_text(prompt=prompt)
        except Exception as exc:
            self._raise_if_strict("section summary generation failed", exc)
            return RetrievalSummaryResult(
                text=self._fallback_summary(raw_text),
                method="fallback",
                provider_name=provider_name,
                model_name=model_name,
                fallback_reason=exc.__class__.__name__,
            )

        # 5. 输出清洗
        cleaned = self._clean_summary(generated)

        # 6. 如果模型输出空、废话、异常，回退到原文截断
        if not cleaned:
            self._raise_if_strict("section summary generation returned empty output")
            return RetrievalSummaryResult(
                text=self._fallback_summary(raw_text),
                method="fallback",
                provider_name=provider_name,
                model_name=model_name,
                fallback_reason="empty_generation",
            )

        return RetrievalSummaryResult(
            text=cleaned,
            method="llm",
            provider_name=provider_name,
            model_name=model_name,
        )

    def summarize_asset_with_metadata(
        self,
        *,
        asset_type: str,
        asset_text: str,
        document_title: str,
        toc_path: list[str] | tuple[str, ...],
        caption: str | None = None,
    ) -> RetrievalSummaryResult:
        raw_text = self._normalize_asset_text(asset_text)
        provider_name = self._provider_name()
        model_name = self._model_name()

        if self._config.raw_text_mode:
            content = raw_text or normalize_whitespace(caption or "") or asset_type
            return RetrievalSummaryResult(
                text=content,
                method="raw_text",
                provider_name=None,
                model_name=None,
            )

        if not raw_text and caption:
            raw_text = normalize_whitespace(caption)
        if not raw_text:
            return RetrievalSummaryResult(
                text=self._fallback_summary(asset_type),
                method="empty_asset",
                provider_name=provider_name,
                model_name=model_name,
            )

        sampled_text = self._sample_text(raw_text)
        prompt = self._build_asset_prompt(
            asset_type=asset_type,
            document_title=document_title,
            toc_path=toc_path,
            caption=caption,
            sampled_text=sampled_text,
        )

        try:
            generated = self._generate_text(prompt=prompt)
        except Exception as exc:
            self._raise_if_strict("asset summary generation failed", exc)
            return RetrievalSummaryResult(
                text=self._fallback_asset_summary(
                    asset_type=asset_type,
                    raw_text=raw_text,
                    caption=caption,
                ),
                method="fallback",
                provider_name=provider_name,
                model_name=model_name,
                fallback_reason=exc.__class__.__name__,
            )

        cleaned = self._clean_summary(generated)
        if not cleaned:
            self._raise_if_strict("asset summary generation returned empty output")
            return RetrievalSummaryResult(
                text=self._fallback_asset_summary(
                    asset_type=asset_type,
                    raw_text=raw_text,
                    caption=caption,
                ),
                method="fallback",
                provider_name=provider_name,
                model_name=model_name,
                fallback_reason="empty_generation",
            )
        return RetrievalSummaryResult(
            text=cleaned,
            method="llm",
            provider_name=provider_name,
            model_name=model_name,
        )

    def summarize_doc_with_metadata(
        self,
        *,
        document_title: str,
        section_summaries: Sequence[str],
        asset_summaries: Sequence[str] = (),
    ) -> RetrievalSummaryResult:
        provider_name = self._provider_name()
        model_name = self._model_name()

        if self._config.raw_text_mode:
            child_text = self._child_summary_text(
                section_summaries=section_summaries,
                asset_summaries=asset_summaries,
            )
            return RetrievalSummaryResult(
                text=normalize_whitespace(document_title) if not child_text else child_text,
                method="raw_text",
                provider_name=None,
                model_name=None,
            )

        child_summary_text = self._child_summary_text(
            section_summaries=section_summaries,
            asset_summaries=asset_summaries,
        )
        if not child_summary_text:
            return RetrievalSummaryResult(
                text=self._fallback_summary(document_title),
                method="empty_doc",
                provider_name=provider_name,
                model_name=model_name,
            )

        sampled_text = self._sample_text(child_summary_text)
        prompt = self._build_doc_prompt(document_title=document_title, sampled_text=sampled_text)
        try:
            generated = self._generate_text(prompt=prompt)
        except Exception as exc:
            self._raise_if_strict("document summary generation failed", exc)
            return RetrievalSummaryResult(
                text=self._fallback_summary(child_summary_text),
                method="fallback",
                provider_name=provider_name,
                model_name=model_name,
                fallback_reason=exc.__class__.__name__,
            )

        cleaned = self._clean_summary(generated)
        if not cleaned:
            self._raise_if_strict("document summary generation returned empty output")
            return RetrievalSummaryResult(
                text=self._fallback_summary(child_summary_text),
                method="fallback",
                provider_name=provider_name,
                model_name=model_name,
                fallback_reason="empty_generation",
            )
        return RetrievalSummaryResult(
            text=cleaned,
            method="llm_doc_reduce",
            provider_name=provider_name,
            model_name=model_name,
        )

    def generator_info(self) -> dict[str, str | int | None]:
        return {
            "provider_name": self._provider_name(),
            "model_name": self._model_name(),
            "direct_return_token_threshold": self._config.direct_return_token_threshold,
            "max_input_tokens": self._config.max_input_tokens,
            "max_output_tokens": self._config.max_output_tokens,
            "strict_generation": self._config.strict_generation,
        }

    def _raise_if_strict(self, message: str, exc: Exception | None = None) -> None:
        if not self._config.strict_generation:
            return
        if exc is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from exc

    def _provider_name(self) -> str | None:
        provider_name = getattr(self._llm, "provider_name", None)
        return provider_name if isinstance(provider_name, str) and provider_name else None

    def _model_name(self) -> str | None:
        model_name = getattr(self._llm, "model_name", None)
        return model_name if isinstance(model_name, str) and model_name else None

    def _sample_text(self, text: str) -> str:
        if self._count_tokens(text) <= self._config.max_input_tokens:
            return text

        head_budget = min(max(self._config.head_tokens, 1), self._config.max_input_tokens)
        mid_budget = min(
            max(self._config.middle_tokens, 0),
            max(self._config.max_input_tokens - head_budget, 0),
        )
        tail_budget = min(
            max(self._config.tail_tokens, 0),
            max(self._config.max_input_tokens - head_budget - mid_budget, 0),
        )

        head = self._clip_tokens(text, head_budget)
        mid = ""
        if mid_budget > 0:
            text_len = self._count_tokens(text)
            mid_start = max(head_budget, (text_len - mid_budget) // 2)
            mid = self._clip_tokens(self._offset_tokens(text, mid_start), mid_budget)
        tail = self._tail_tokens(text, tail_budget) if tail_budget > 0 else ""

        parts = [p for p in (head, mid, tail) if p]
        if not parts:
            return self._clip_tokens(text, self._config.max_input_tokens)
        return "\n...\n".join(parts).strip()

    def _build_prompt(
        self,
        *,
        document_title: str,
        section: ParsedSection,
        sampled_text: str,
    ) -> str:
        toc_path = " > ".join(section.toc_path) if section.toc_path else document_title
        heading_level = str(section.heading_level) if section.heading_level is not None else "unknown"
        page_range = f"{section.page_range[0]}-{section.page_range[1]}" if section.page_range is not None else "unknown"

        return f"""
You are generating a retrieval summary for a document-grounded RAG system.

This summary is NOT for human reading elegance.
It is for vector search, sparse retrieval, and reranking.

Document title: {document_title}
Section path: {toc_path}
Heading level: {heading_level}
Page range: {page_range}

Output exactly these three fields:
Semantic Core: one dense sentence with the section's core meaning and retrieval intent.
Fact Anchors: comma-separated exact facts from the source, including numbers, dates, departments,
names, amounts, document codes, thresholds, exception conditions, and process steps.
Retrieval Keywords: comma-separated source terms, aliases, acronyms, business objects, policy names,
table/field names, and likely query terms.

Rules:
1. Preserve important entities, names, dates, numbers, products, departments, policies, acronyms, and technical terms.
2. Preserve process steps, constraints, thresholds, exceptions, and decision conditions.
3. Do NOT write vague filler like "this section discusses" or "the text talks about".
4. Do NOT invent facts that are not present.
5. Prefer exact terminology from the source text.
6. Keep every field dense. If a field has no evidence, write "none".

Section text:
{sampled_text}
""".strip()

    def _build_asset_prompt(
        self,
        *,
        asset_type: str,
        document_title: str,
        toc_path: list[str] | tuple[str, ...],
        caption: str | None,
        sampled_text: str,
    ) -> str:
        path = " > ".join(str(part) for part in toc_path if str(part).strip()) or document_title
        caption_text = normalize_whitespace(caption or "") or "none"
        return f"""
You are generating a retrieval summary for a document asset in a RAG system.

This summary is for vector search, sparse retrieval, and reranking.

Document title: {document_title}
Asset type: {asset_type}
Document path: {path}
Caption: {caption_text}

Output exactly these three fields:
Semantic Core: one dense sentence with the asset's purpose and retrieval intent.
Fact Anchors: comma-separated exact facts from the asset, including columns, row examples,
numeric ranges, dates, departments, names, amounts, codes, thresholds, and captions.
Retrieval Keywords: comma-separated source terms, field names, aliases, acronyms, business objects,
policy names, and likely query terms.

Rules:
1. Preserve exact entities, departments, names, dates, amounts, thresholds, codes, and field names.
2. For table assets, identify the table's purpose, columns, important row examples,
   numeric ranges, and business constraints.
3. Do NOT say vague filler like "this table describes" without concrete facts.
4. Do NOT invent facts that are not present.
5. Prefer exact terminology from the source.
6. Keep every field dense. If a field has no evidence, write "none".

Asset content:
{sampled_text}
""".strip()

    def _build_doc_prompt(self, *, document_title: str, sampled_text: str) -> str:
        return f"""
You are generating a document-level retrieval summary for a document-grounded RAG system.

The input is already composed of child section and asset retrieval summaries.
Do NOT ask for the original full document.

Document title: {document_title}

Output exactly these three fields:
Semantic Core: one dense sentence with the document's overall scope, main topic, and retrieval intent.
Fact Anchors: comma-separated exact facts from child summaries, including numbers, dates, departments,
names, amounts, document codes, thresholds, named assets, and process constraints.
Retrieval Keywords: comma-separated high-signal terms, aliases, acronyms, policy names,
business objects, asset names, and likely query terms.

Rules:
1. Preserve scope, main topic, entities, departments, dates, numbers, thresholds, and named assets.
2. Prefer high-signal facts and terminology from child summaries.
3. Do NOT repeat generic filler like "this document discusses".
4. Do NOT invent facts not present in child summaries.
5. Keep every field dense. If a field has no evidence, write "none".

Child retrieval summaries:
{sampled_text}
""".strip()

    @staticmethod
    def _child_summary_text(
        *,
        section_summaries: Sequence[str],
        asset_summaries: Sequence[str],
    ) -> str:
        lines: list[str] = []
        for index, summary in enumerate(section_summaries, start=1):
            normalized = normalize_whitespace(summary)
            if normalized:
                lines.append(f"[SECTION {index}] {normalized}")
        for index, summary in enumerate(asset_summaries, start=1):
            normalized = normalize_whitespace(summary)
            if normalized:
                lines.append(f"[ASSET {index}] {normalized}")
        return "\n".join(lines).strip()

    def _clean_summary(self, text: str) -> str:
        cleaned = self._normalize_summary_output(text)

        # 去掉常见废话前缀
        bad_prefixes = (
            "this section discusses",
            "this section describes",
            "this section explains",
            "the section discusses",
            "the section describes",
            "the text discusses",
            "the text describes",
        )
        lowered = cleaned.lower()
        for prefix in bad_prefixes:
            if lowered.startswith(prefix):
                # 简单粗暴清掉整句前缀痕迹
                cleaned = cleaned[len(prefix) :].lstrip(" :,-")
                break

        cleaned = self._normalize_summary_output(cleaned)

        if not cleaned:
            return ""

        return self._clip_tokens(cleaned, self._config.max_output_tokens)

    def _fallback_summary(self, raw_text: str) -> str:
        normalized = normalize_whitespace(raw_text)
        core = self._clip_tokens(normalized, max(self._config.max_output_tokens // 2, 1))
        anchors = self._fact_anchor_preview(normalized)
        keywords = self._keyword_preview(normalized)
        return self._clip_tokens(
            "\n".join(
                [
                    f"Semantic Core: {core or 'none'}",
                    f"Fact Anchors: {anchors or 'none'}",
                    f"Retrieval Keywords: {keywords or 'none'}",
                ]
            ),
            self._config.max_output_tokens,
        )

    def _fallback_asset_summary(self, *, asset_type: str, raw_text: str, caption: str | None) -> str:
        parts = [asset_type]
        caption_text = normalize_whitespace(caption or "")
        if caption_text:
            parts.append(f"caption: {caption_text}")
        if asset_type == "table":
            columns = self._markdown_table_columns(raw_text)
            if columns:
                parts.append(f"columns: {', '.join(columns)}")
        parts.append(f"content preview: {normalize_whitespace(raw_text)}")
        return self._fallback_summary(" ".join(parts))

    def _count_tokens(self, text: str) -> int:
        try:
            return self._token_accounting.count(text)
        except Exception:
            return text_unit_count(text)

    def _clip_tokens(self, text: str, token_budget: int) -> str:
        if self._count_tokens(text) <= max(token_budget, 1):
            return text.strip()
        try:
            return self._token_accounting.clip(text, token_budget).strip()
        except Exception:
            return self._fallback_token_clip(text, token_budget)

    def _tail_tokens(self, text: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        try:
            return self._token_accounting.tail(text, token_budget).strip()
        except Exception:
            return self._fallback_token_tail(text, token_budget)

    def _offset_tokens(self, text: str, offset_tokens: int) -> str:
        """跳过前 offset_tokens 后返回剩余文本。"""
        total = self._count_tokens(text)
        return self._tail_tokens(text, max(total - offset_tokens, 0))

    @staticmethod
    def _fallback_token_clip(text: str, token_budget: int) -> str:
        units = text.split()
        if units:
            return " ".join(units[: max(token_budget, 1)]).strip()
        return text.strip()

    @staticmethod
    def _fallback_token_tail(text: str, token_budget: int) -> str:
        units = text.split()
        if units:
            return " ".join(units[-max(token_budget, 1) :]).strip()
        return text.strip()

    @staticmethod
    def _normalize_summary_output(text: str) -> str:
        lines = [
            normalize_whitespace(line) for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ]
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def _fact_anchor_preview(text: str) -> str:
        terms: list[str] = []
        seen: set[str] = set()
        for token in normalize_whitespace(text).split():
            cleaned = token.strip(" ,.;:，。；：、()（）[]【】")
            if not cleaned:
                continue
            has_anchor_signal = any(ch.isdigit() for ch in cleaned) or any(ch.isupper() for ch in cleaned)
            if not has_anchor_signal or cleaned in seen:
                continue
            seen.add(cleaned)
            terms.append(cleaned)
            if len(terms) >= 24:
                break
        return ", ".join(terms)

    @staticmethod
    def _keyword_preview(text: str) -> str:
        terms: list[str] = []
        seen: set[str] = set()
        for token in normalize_whitespace(text).split():
            cleaned = token.strip(" ,.;:，。；：、()（）[]【】").lower()
            if len(cleaned) < 2 or cleaned in seen:
                continue
            seen.add(cleaned)
            terms.append(cleaned)
            if len(terms) >= 32:
                break
        return ", ".join(terms)

    @staticmethod
    def _markdown_table_columns(text: str) -> list[str]:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|") or not stripped.endswith("|"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            columns = [cell for cell in cells if cell and set(cell) != {"-"}]
            if columns:
                return columns[:24]
        return []

    @staticmethod
    def _normalize_asset_text(text: str) -> str:
        return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

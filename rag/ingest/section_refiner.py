from __future__ import annotations

from dataclasses import dataclass, replace

from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract
from agent_runtime.text import _token_unit_spans, text_unit_count
from rag.ingest.asset_anchors import iter_asset_anchor_spans
from rag.schema.core import ParsedDocument, ParsedSection


@dataclass(frozen=True, slots=True)
class SectionRefinerConfig:
    max_section_tokens: int | None = None
    window_tokens: int | None = None
    overlap_tokens: int | None = None


class SectionRefiner:
    """
    Structure-first section refinement.

    Parsers own document structure. This refiner only applies a generic token-budget
    fallback to oversized ParsedSection records while preserving exact char spans.
    """

    def __init__(
        self,
        *,
        token_accounting: TokenAccountingService | None = None,
        config: SectionRefinerConfig | None = None,
    ) -> None:
        self._token_accounting = token_accounting or TokenAccountingService(
            TokenizerContract(
                embedding_model_name="default",
                tokenizer_model_name="default",
                chunking_tokenizer_model_name="default",
                tokenizer_backend="simple",
                local_files_only=True,
            )
        )
        self._config = config or SectionRefinerConfig()

    @property
    def token_accounting(self) -> TokenAccountingService:
        return self._token_accounting

    def refine(self, parsed_doc: ParsedDocument) -> ParsedDocument:
        if not parsed_doc.sections:
            return parsed_doc

        refined_sections: list[ParsedSection] = []
        changed = False

        for section in parsed_doc.sections:
            if self._should_keep(section):
                refined_sections.append(section)
                continue

            windows = self._split_section(section)
            if len(windows) <= 1:
                refined_sections.append(section)
                continue

            changed = True
            refined_sections.extend(windows)

        if not changed:
            return parsed_doc

        reordered = [
            section if section.order_index == order_index else replace(section, order_index=order_index)
            for order_index, section in enumerate(refined_sections)
        ]
        return replace(parsed_doc, sections=reordered)

    def _should_keep(self, section: ParsedSection) -> bool:
        return self._count(section.text) <= self._max_section_tokens()

    def _split_section(self, section: ParsedSection) -> list[ParsedSection]:
        spans = self._offset_spans(section.text)
        if len(spans) <= self._window_tokens():
            return [section]

        windows: list[tuple[int, int, int, int]] = []
        size = self._window_tokens()
        overlap = self._overlap_tokens()
        step = max(size - overlap, 1)
        assigned_anchor_refs: set[str] = set()
        for token_start in range(0, len(spans), step):
            token_end = min(token_start + size, len(spans))
            span_window = spans[token_start:token_end]
            if not span_window:
                continue
            local_start = span_window[0][0]
            local_end = span_window[-1][1]
            protected_range = self._protect_asset_anchors(
                section.text,
                local_start,
                local_end,
                assigned_anchor_refs=assigned_anchor_refs,
            )
            if protected_range is None:
                continue
            local_start, local_end = protected_range
            windows.append((local_start, local_end, token_start, token_end))
            if token_end >= len(spans):
                break

        if len(windows) <= 1:
            return [section]

        refined: list[ParsedSection] = []
        window_count = len(windows)
        for window_index, (local_start, local_end, token_start, token_end) in enumerate(windows):
            absolute_start = section.char_range_start + local_start
            absolute_end = section.char_range_start + local_end
            metadata = {
                **section.metadata,
                "refine_strategy": "token_window",
                "refined_from_section_order": str(section.order_index),
                "refined_window_index": str(window_index),
                "refined_window_count": str(window_count),
                "refined_token_start": str(token_start),
                "refined_token_end": str(token_end),
            }
            refined.append(
                ParsedSection(
                    toc_path=section.toc_path,
                    heading_level=section.heading_level,
                    page_range=section.page_range,
                    order_index=section.order_index + window_index,
                    text=section.text[local_start:local_end],
                    char_range_start=absolute_start,
                    char_range_end=absolute_end,
                    anchor_hint=self._window_anchor(section, window_index),
                    metadata=metadata,
                )
            )
        return refined

    def _offset_spans(self, text: str) -> list[tuple[int, int]]:
        spans = self._token_accounting._offset_spans(text)
        if spans:
            return list(spans)
        return _token_unit_spans(text)

    def _count(self, text: str) -> int:
        try:
            return self._token_accounting.count(text)
        except Exception:
            return text_unit_count(text)

    @staticmethod
    def _protect_asset_anchors(
        text: str,
        start: int,
        end: int,
        *,
        assigned_anchor_refs: set[str],
    ) -> tuple[int, int] | None:
        expanded_start = start
        expanded_end = end
        for anchor_start, anchor_end, anchor_ref in iter_asset_anchor_spans(text):
            if expanded_start >= anchor_end or expanded_end <= anchor_start:
                continue
            if anchor_ref in assigned_anchor_refs:
                if expanded_start < anchor_end < expanded_end:
                    expanded_start = anchor_end
                elif expanded_start < anchor_start < expanded_end:
                    expanded_end = anchor_start
                else:
                    return None
                continue
            if expanded_start <= anchor_start < expanded_end:
                expanded_start = min(expanded_start, anchor_start)
                expanded_end = max(expanded_end, anchor_end)
                assigned_anchor_refs.add(anchor_ref)
            elif anchor_start < expanded_start < anchor_end:
                expanded_start = anchor_end
        if expanded_end <= expanded_start:
            return None
        return expanded_start, expanded_end

    def _window_tokens(self) -> int:
        configured = self._config.window_tokens or self._token_accounting.contract.chunk_token_size
        return max(int(configured), 1)

    def _max_section_tokens(self) -> int:
        configured = self._config.max_section_tokens
        if configured is not None:
            return max(int(configured), 1)
        return self._window_tokens()

    def _overlap_tokens(self) -> int:
        configured = self._config.overlap_tokens
        if configured is None:
            configured = self._token_accounting.contract.normalized_chunk_overlap_tokens()
        return min(max(int(configured), 0), max(self._window_tokens() - 1, 0))

    @staticmethod
    def _window_anchor(section: ParsedSection, window_index: int) -> str | None:
        if section.anchor_hint:
            return f"{section.anchor_hint}-part-{window_index + 1}"
        return None


__all__ = ["SectionRefiner", "SectionRefinerConfig"]

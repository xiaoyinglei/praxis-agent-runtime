from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Protocol, cast

from agent_runtime.text import DEFAULT_TOKENIZER_FALLBACK_MODEL, keyword_overlap, search_terms
from rag.ingest.asset_anchors import asset_anchor
from rag.schema.core import AssetRecord, LayoutMetaCacheRecord, SectionRecord
from rag.schema.query import EvidenceItem, GroundingTarget
from rag.utils.guard import CircuitBreaker

MAX_COMPUTE_BLOCK_TOKENS = 1_500
MAX_SCHEMA_COLUMNS = 30


class _TokenAccounting(Protocol):
    def count(self, text: str) -> int: ...
    def clip(self, text: str, token_budget: int, *, add_ellipsis: bool = ...) -> str: ...
    def chunk_text(self, text: str, *, chunk_token_size: int, chunk_overlap_tokens: int) -> list[str]: ...


_logger = logging.getLogger("rag.grounding")


class _RerankBinding(Protocol):
    def rerank(self, query: str, documents: list[str], **kwargs: object) -> list[float]: ...


def _load_tokenizer_classes() -> tuple[type[Any], type[Any]]:
    module_path = Path(__file__).resolve().parents[1] / "assembly" / "tokenizer.py"
    spec = spec_from_file_location("rag_assembly_tokenizer_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load tokenizer module from {module_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TokenAccountingService, module.TokenizerContract


def _default_token_accounting() -> _TokenAccounting:
    token_accounting_cls, tokenizer_contract_cls = _load_tokenizer_classes()
    contract = tokenizer_contract_cls(
        embedding_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
        tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
        chunking_tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
    )
    return cast(_TokenAccounting, token_accounting_cls(contract))


@dataclass(frozen=True, slots=True)
class GroundingBudgets:
    max_targets_to_read: int = 3
    max_output_tokens: int = 8_000
    local_window_tokens: int = 220
    local_window_overlap_tokens: int = 30
    local_window_top_k: int = 6
    max_neighbor_assets: int = 2
    read_timeout_seconds: float = 1.5
    max_parallel_reads: int = 2
    max_asset_preview_bytes: int = 4096
    max_document_sections: int = 2
    neighbor_section_radius: int = 1
    max_neighbor_sections: int = 2
    rerank_max_items: int = 16
    rerank_batch_size: int = 8
    rerank_max_item_tokens: int = 384
    rerank_max_total_tokens: int = 2048


class _GroundingMetadataRepo(Protocol):
    def get_section(self, section_id: int) -> SectionRecord | None: ...

    def list_sections(self, *, doc_id: int | None = None, source_id: int | None = None) -> list[SectionRecord]: ...

    def get_asset(self, asset_id: int) -> AssetRecord | None: ...

    def list_assets(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
        section_id: int | None = None,
    ) -> list[AssetRecord]: ...

    def get_layout_meta_cache(
        self,
        *,
        source_id: int | None = None,
        doc_id: int | None = None,
        content_hash: str | None = None,
    ) -> LayoutMetaCacheRecord | None: ...


class _RangeReadableObjectStore(Protocol):
    def read_byte_range(self, key: str, start: int, end: int) -> bytes: ...


@dataclass(slots=True)
class _GroundingSession:
    executor: ThreadPoolExecutor
    semaphore: BoundedSemaphore


@dataclass(slots=True)
class GroundingService:
    metadata_repo: object
    object_store: object
    token_accounting: _TokenAccounting = field(default_factory=_default_token_accounting)
    budgets: GroundingBudgets = field(default_factory=GroundingBudgets)
    rerank_binding: _RerankBinding | object | None = None
    s3_circuit_breaker: CircuitBreaker | None = None
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _semaphore: BoundedSemaphore | None = field(default=None, init=False, repr=False)

    def ground(
        self,
        *,
        query: str,
        evidence: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        grounded_targets = [item for item in evidence if item.grounding_target is not None]
        if not grounded_targets:
            return list(evidence)

        query_terms = search_terms(query)
        grounded_items: list[EvidenceItem] = []
        output_tokens = 0

        session = self._grounding_session()
        for item in grounded_targets[: self.budgets.max_targets_to_read]:
            for grounded in self._ground_item(item, query=query, query_terms=query_terms, session=session):
                token_count = self.token_accounting.count(grounded.text)
                if token_count <= 0:
                    continue
                remaining_budget = self.budgets.max_output_tokens - output_tokens
                if remaining_budget <= 0:
                    break
                if token_count > remaining_budget:
                    clipped = self.token_accounting.clip(grounded.text, remaining_budget, add_ellipsis=True)
                    clipped_token_count = self.token_accounting.count(clipped)
                    if clipped_token_count <= 0:
                        break
                    grounded = grounded.model_copy(update={"text": clipped})
                    token_count = clipped_token_count
                grounded_items.append(grounded)
                output_tokens += token_count
                if output_tokens >= self.budgets.max_output_tokens:
                    break
            if output_tokens >= self.budgets.max_output_tokens:
                break

        return grounded_items or list(evidence[: self.budgets.max_targets_to_read])

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        self._semaphore = None
        if executor is not None:
            executor.shutdown(wait=True)

    def _grounding_session(self) -> _GroundingSession:
        if self._executor is None:
            max_workers = max(self.budgets.max_parallel_reads, 1)
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            self._semaphore = BoundedSemaphore(value=max_workers)
        assert self._semaphore is not None
        return _GroundingSession(executor=self._executor, semaphore=self._semaphore)

    def _ground_item(
        self,
        item: EvidenceItem,
        *,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
    ) -> list[EvidenceItem]:
        target = item.grounding_target
        if target is None:
            return [item]
        if target.kind == "section":
            return self._ground_section_item(item, query=query, query_terms=query_terms, session=session)
        if target.kind == "asset":
            return self._ground_asset_item(item, query=query, query_terms=query_terms, session=session)
        if target.kind == "document":
            return self._ground_document_item(item, query=query, query_terms=query_terms, session=session)
        return [item]

    def _ground_section_item(
        self,
        item: EvidenceItem,
        *,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
    ) -> list[EvidenceItem]:
        target = item.grounding_target
        if target is None:
            return [item]
        section = self._get_section(self._safe_int(target.section_id))
        if section is None:
            return [item]
        raw_text = self._read_section_text(section, session=session)
        section_assets = self._section_assets(section)
        grounded_text = self._replace_section_asset_anchors(
            raw_text,
            assets=section_assets,
            query=query,
            session=session,
        )
        local_items = self._section_local_windows(
            item=item,
            target=target,
            section=section,
            raw_text=grounded_text,
            query=query,
            query_terms=query_terms,
        )
        local_items.extend(
            self._neighbor_section_items(
                item=item,
                section=section,
                query=query,
                query_terms=query_terms,
                session=session,
            )
        )
        if target.kind == "section":
            local_items.extend(
                self._neighbor_asset_items(
                    item=item,
                    section=section,
                    query=query,
                    query_terms=query_terms,
                    session=session,
                    assets=section_assets,
                )
            )
        ranked = self._rank_local_items(local_items, query=query, query_terms=query_terms)
        return ranked[
            : max(
                self.budgets.local_window_top_k + self.budgets.max_neighbor_assets + self.budgets.max_neighbor_sections,
                1,
            )
        ]

    def _ground_document_item(
        self,
        item: EvidenceItem,
        *,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
    ) -> list[EvidenceItem]:
        target = item.grounding_target
        if target is None:
            return [item]
        list_sections = getattr(self.metadata_repo, "list_sections", None)
        if not callable(list_sections):
            return [item]
        doc_id = self._safe_int(target.doc_id)
        if doc_id is None:
            return [item]
        sections = list_sections(doc_id=doc_id)
        if not sections:
            return [item]
        ranked_sections = sorted(
            sections,
            key=lambda section: (
                -self._section_candidate_overlap(section, query_terms=query_terms),
                section.order_index,
                section.section_id,
            ),
        )
        grounded: list[EvidenceItem] = []
        for section in ranked_sections[: self.budgets.max_document_sections]:
            section_target = target.model_copy(
                update={
                    "kind": "section",
                    "section_id": str(section.section_id),
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "section_path": list(section.toc_path),
                }
            )
            section_item = item.model_copy(
                update={
                    "citation_anchor": " / ".join(section.toc_path) or item.citation_anchor,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "grounding_target": section_target,
                }
            )
            grounded.extend(
                self._ground_section_item(
                    section_item,
                    query=query,
                    query_terms=query_terms,
                    session=session,
                )
            )
        return grounded or [item]

    def _ground_asset_item(
        self,
        item: EvidenceItem,
        *,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
    ) -> list[EvidenceItem]:
        target = item.grounding_target
        if target is None:
            return [item]
        asset = self._get_asset(self._safe_int(target.asset_id))
        if asset is None:
            return [item]
        text = (
            self._compute_only_table_block(asset)
            if asset.asset_type == "table" and self._should_use_compute_table_block(asset)
            else self._asset_text(asset, session=session).strip()
        )
        if not text:
            return [item]
        grounded = item.model_copy(
            update={
                "evidence_id": f"grounded:asset:{asset.asset_id}",
                "record_type": "asset",
                "text": text,
                "citation_anchor": item.citation_anchor or f"{asset.asset_type}@p{asset.page_no}",
                "page_start": asset.page_no,
                "page_end": asset.page_no,
                "grounding_target": GroundingTarget(
                    kind="asset",
                    doc_id=asset.doc_id,
                    source_id=asset.source_id,
                    section_id=asset.section_id,
                    asset_id=asset.asset_id,
                    page_start=asset.page_no,
                    page_end=asset.page_no,
                    raw_locator=self._raw_locator_dict(asset.raw_locator),
                ),
                "retrieval_channels": [*item.retrieval_channels, "grounding"],
            }
        )
        return self._rank_local_items([grounded], query=query, query_terms=query_terms)

    def _section_local_windows(
        self,
        *,
        item: EvidenceItem,
        target: GroundingTarget,
        section: SectionRecord,
        raw_text: str,
        query: str,
        query_terms: tuple[str, ...],
    ) -> list[EvidenceItem]:
        text = raw_text.strip()
        if not text:
            return []
        section_path = list(target.section_path or section.toc_path)
        chunks = self.token_accounting.chunk_text(
            text,
            chunk_token_size=self.budgets.local_window_tokens,
            chunk_overlap_tokens=self.budgets.local_window_overlap_tokens,
        )
        if not chunks:
            chunks = [text]
        local_items: list[EvidenceItem] = []
        for index, chunk_text in enumerate(chunks, start=1):
            normalized = chunk_text.strip().rstrip(".")
            if not normalized:
                continue
            local_items.append(
                item.model_copy(
                    update={
                        "evidence_id": f"grounded:section:{section.section_id}:{index}",
                        "record_type": "section",
                        "text": normalized,
                        "citation_anchor": " / ".join(section_path) or item.citation_anchor,
                        "section_path": section_path,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                        "grounding_target": GroundingTarget(
                            kind="section",
                            doc_id=section.doc_id,
                            source_id=section.source_id,
                            section_id=section.section_id,
                            page_start=section.page_start,
                            page_end=section.page_end,
                            section_path=section_path,
                            raw_locator=self._raw_locator_dict(section.raw_locator),
                        ),
                        "retrieval_channels": [*item.retrieval_channels, "grounding"],
                    }
                )
            )
        return local_items

    def _neighbor_section_items(
        self,
        *,
        item: EvidenceItem,
        section: SectionRecord,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
    ) -> list[EvidenceItem]:
        neighbor_sections = self._neighbor_sections(section)
        if not neighbor_sections:
            return []
        local_items: list[EvidenceItem] = []
        for neighbor in neighbor_sections:
            raw_text = self._read_section_text(neighbor, session=session)
            if not raw_text.strip():
                continue
            neighbor_assets = self._section_assets(neighbor)
            grounded_text = self._replace_section_asset_anchors(
                raw_text,
                assets=neighbor_assets,
                query=query,
                session=session,
            )
            section_path = list(neighbor.toc_path)
            neighbor_target = GroundingTarget(
                kind="section",
                doc_id=neighbor.doc_id,
                source_id=neighbor.source_id,
                section_id=neighbor.section_id,
                page_start=neighbor.page_start,
                page_end=neighbor.page_end,
                section_path=section_path,
                raw_locator=self._raw_locator_dict(neighbor.raw_locator),
            )
            neighbor_item = item.model_copy(
                update={
                    "citation_anchor": " / ".join(section_path) or item.citation_anchor,
                    "section_path": section_path,
                    "page_start": neighbor.page_start,
                    "page_end": neighbor.page_end,
                    "grounding_target": neighbor_target,
                    "retrieval_channels": [*item.retrieval_channels, "neighbor_expansion"],
                }
            )
            local_items.extend(
                self._section_local_windows(
                    item=neighbor_item,
                    target=neighbor_target,
                    section=neighbor,
                    raw_text=grounded_text,
                    query=query,
                    query_terms=query_terms,
                )
            )
        return local_items

    def _neighbor_sections(self, section: SectionRecord) -> list[SectionRecord]:
        radius = max(int(self.budgets.neighbor_section_radius), 0)
        max_sections = max(int(self.budgets.max_neighbor_sections), 0)
        parent_section_id = self._safe_int(section.parent_section_id)
        if radius <= 0 or max_sections <= 0 or parent_section_id is None:
            return []
        list_sections = getattr(self.metadata_repo, "list_sections", None)
        if not callable(list_sections):
            return []
        same_group = [
            candidate
            for candidate in list_sections(doc_id=section.doc_id)
            if candidate.doc_id == section.doc_id
            and candidate.source_id == section.source_id
            and self._safe_int(candidate.parent_section_id) == parent_section_id
            and tuple(candidate.toc_path) == tuple(section.toc_path)
        ]
        if len(same_group) <= 1:
            return []
        current_window_index = self._section_window_index(section)
        if current_window_index is None:
            ordered = sorted(same_group, key=lambda item: (item.order_index, item.section_id))
            position_by_id = {candidate.section_id: index for index, candidate in enumerate(ordered)}
            current_position = position_by_id.get(section.section_id)
            if current_position is None:
                return []
            selected = [
                candidate
                for index, candidate in enumerate(ordered)
                if candidate.section_id != section.section_id and abs(index - current_position) <= radius
            ]
            return selected[:max_sections]

        selected_with_distance: list[tuple[int, int, int, SectionRecord]] = []
        for candidate in same_group:
            if candidate.section_id == section.section_id:
                continue
            candidate_window_index = self._section_window_index(candidate)
            if candidate_window_index is None:
                continue
            distance = abs(candidate_window_index - current_window_index)
            if distance <= radius:
                selected_with_distance.append((distance, candidate_window_index, candidate.order_index, candidate))
        selected_with_distance.sort(key=lambda item: (item[0], item[1], item[2], item[3].section_id))
        return [candidate for *_prefix, candidate in selected_with_distance[:max_sections]]

    def _neighbor_asset_items(
        self,
        *,
        item: EvidenceItem,
        section: SectionRecord,
        query: str,
        query_terms: tuple[str, ...],
        session: _GroundingSession,
        assets: list[AssetRecord] | None = None,
    ) -> list[EvidenceItem]:
        assets = self._section_assets(section) if assets is None else assets
        grounded_assets: list[EvidenceItem] = []
        for asset in assets[: self.budgets.max_neighbor_assets]:
            text = (
                self._compute_only_table_block(asset)
                if asset.asset_type == "table" and self._should_use_compute_table_block(asset)
                else self._asset_text(asset, session=session).strip()
            )
            if not text:
                continue
            grounded_assets.append(
                item.model_copy(
                    update={
                        "evidence_id": f"grounded:asset:{asset.asset_id}",
                        "record_type": "asset",
                        "text": text.rstrip("."),
                        "citation_anchor": f"{asset.asset_type}@p{asset.page_no}",
                        "page_start": asset.page_no,
                        "page_end": asset.page_no,
                        "grounding_target": GroundingTarget(
                            kind="asset",
                            doc_id=asset.doc_id,
                            source_id=asset.source_id,
                            section_id=asset.section_id,
                            asset_id=asset.asset_id,
                            page_start=asset.page_no,
                            page_end=asset.page_no,
                            raw_locator=self._raw_locator_dict(asset.raw_locator),
                        ),
                        "retrieval_channels": [*item.retrieval_channels, "grounding"],
                    }
                )
            )
        return self._rank_local_items(grounded_assets, query=query, query_terms=query_terms)

    def _section_assets(self, section: SectionRecord) -> list[AssetRecord]:
        list_assets = getattr(self.metadata_repo, "list_assets", None)
        if not callable(list_assets):
            return []
        assets = self._layout_neighbor_assets(section)
        if assets:
            return assets
        return cast(list[AssetRecord], list_assets(doc_id=section.doc_id, section_id=section.section_id))

    def _replace_section_asset_anchors(
        self,
        text: str,
        *,
        assets: list[AssetRecord],
        query: str,
        session: _GroundingSession,
    ) -> str:
        replaced = text
        for asset in assets:
            anchor = self._asset_anchor(asset)
            if not anchor or anchor not in replaced:
                continue
            replacement = self._section_anchor_replacement(asset, query=query, session=session)
            replaced = replaced.replace(anchor, replacement)
        return replaced

    def _section_anchor_replacement(
        self,
        asset: AssetRecord,
        *,
        query: str,
        session: _GroundingSession,
    ) -> str:
        if asset.asset_type != "table":
            return self._asset_text(asset, session=session)
        if self._should_use_compute_table_block(asset):
            return self._compute_only_table_block(asset)
        return self._asset_text(asset, session=session)

    def _compute_only_table_block(self, asset: AssetRecord) -> str:
        parts: list[str] = []
        sheet = asset.sheet_name or "unknown"

        parts.append("<system_instruction>")
        parts.append("CRITICAL: ALL tables in this system are processed via SQL execution.")
        parts.append("The sample rows below are NOT statistically representative. You MUST")
        parts.append("NOT visually scan sample rows to answer questions about specific data")
        parts.append("values, rankings, aggregations, subtotals, or filtered subsets.")
        parts.append("")
        parts.append("You MAY answer directly ONLY for schema-level questions:")
        parts.append('- "what columns does this table have"')
        parts.append('- "how many rows are in this table"')
        parts.append('- "what types/values appear in column X" (from the enum/type annotations)')
        parts.append("")
        parts.append("For ANY question involving actual data values — filtering by a value")
        parts.append('("Northern region"), aggregating ("total sales"), sorting ("top 5"),')
        parts.append("or comparing rows — you MUST output a computation request in this")
        parts.append("EXACT format:")
        parts.append("")
        parts.append("<compute_request>")
        parts.append(f'{{"asset_id": {asset.asset_id}, "sql": "SELECT ... FROM sheet WHERE ..."}}')
        parts.append("</compute_request>")
        parts.append("")
        parts.append("SQL rules:")
        parts.append('- Table name is ALWAYS "sheet".')
        parts.append("- Quote Chinese/special column names with double quotes.")
        parts.append("- SELECT statements only. DuckDB dialect.")
        parts.append("- The backend will execute your SQL and re-invoke you with the actual")
        parts.append("  computed results to form the final answer.")
        parts.append("</system_instruction>")
        parts.append("")
        parts.append(f"[TABLE_COMPUTE_ONLY:asset_id={asset.asset_id}]")
        parts.append(f"Source: {sheet}")
        if asset.caption:
            parts.append(f"Caption: {asset.caption}")
        parts.append(f"Shape: {asset.row_count or '?'} rows x {asset.column_count or '?'} columns")
        estimated = asset.metadata_json.get("estimated_tokens", "?")
        parts.append(f"Estimated full size: ~{estimated} tokens (too large to load in context)")
        parts.append("")

        schema = asset.schema or []
        if schema:
            parts.append("Schema:")
            for i, field in enumerate(schema[:MAX_SCHEMA_COLUMNS]):
                name = field.get("name", f"col_{i}")
                ftype = field.get("type", "text")
                parts.append(f"  {name:40s} -> {ftype}")
            if len(schema) > MAX_SCHEMA_COLUMNS:
                parts.append(f"  ... ({len(schema) - MAX_SCHEMA_COLUMNS} more columns omitted)")
            parts.append("")

        samples = asset.sample_rows or []
        if samples:
            display_schema = schema[:MAX_SCHEMA_COLUMNS]
            columns = [f.get("name", f"col_{i}") for i, f in enumerate(display_schema)]
            if not columns and samples:
                columns = list(samples[0].keys())[:MAX_SCHEMA_COLUMNS]
            if columns:
                parts.append(f"Sample rows ({len(samples)} of {asset.row_count or '?'}):")
                parts.append("| " + " | ".join(columns) + " |")
                parts.append("|" + "|".join(["---"] * len(columns)) + "|")
                for row in samples:
                    cells = [str(row.get(col, "")) for col in columns]
                    parts.append("| " + " | ".join(cells) + " |")
                if len(schema) > MAX_SCHEMA_COLUMNS:
                    parts.append(f"(table has {len(schema)} columns; showing first {MAX_SCHEMA_COLUMNS})")
            parts.append("")

        block = "\n".join(parts)
        try:
            return self.token_accounting.clip(block, MAX_COMPUTE_BLOCK_TOKENS, add_ellipsis=True)
        except Exception:
            return block

    @staticmethod
    def _should_use_compute_table_block(asset: AssetRecord) -> bool:
        if asset.asset_type != "table":
            return False
        storage_key = str(asset.storage_key or "").lower()
        return any(
            (
                bool(asset.schema),
                bool(asset.sample_rows),
                asset.row_count is not None,
                asset.column_count is not None,
                bool(asset.metadata_json.get("table_policy")),
                storage_key.endswith((".parquet", ".xlsx", ".xls")),
            )
        )

    @staticmethod
    def _asset_anchor(asset: AssetRecord) -> str | None:
        value = asset.metadata_json.get("asset_anchor")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if asset.element_ref is not None and str(asset.element_ref).strip():
            return asset_anchor(str(asset.element_ref).strip())
        return None

    def _read_section_text(self, section: SectionRecord, *, session: _GroundingSession) -> str:
        key = section.visible_text_key or self._locator_key(section.raw_locator)
        start = section.byte_range_start
        end = section.byte_range_end
        if not key or start is None or end is None or end <= start:
            return ""

        raw = self._read_range(key, start, end, session=session)
        if raw:
            return raw.decode("utf-8", errors="ignore")

        local = self._read_from_local_cache(key, start, end)
        if local:
            return local.decode("utf-8", errors="ignore")

        _logger.warning("Section content unavailable: section_id=%d key=%s", section.section_id, key)
        return f"[CONTENT_UNAVAILABLE:section_id={section.section_id}]"

    @staticmethod
    def _read_from_local_cache(key: str, start: int, end: int) -> bytes | None:
        if not os.path.exists(key):
            return None
        try:
            with open(key, "rb") as fh:
                fh.seek(start)
                return fh.read(end - start)
        except OSError:
            return None

    def _asset_text(self, asset: AssetRecord, *, session: _GroundingSession) -> str:
        preview = asset.metadata_json.get("asset_text_preview")
        if isinstance(preview, str) and preview.strip():
            return preview.strip()
        if asset.caption and asset.caption.strip():
            return asset.caption.strip()
        key = asset.storage_key or self._locator_key(asset.raw_locator)
        if key:
            raw_preview = self._read_range(key, 0, self.budgets.max_asset_preview_bytes, session=session)
            text = raw_preview.decode("utf-8", errors="ignore").strip()
            if text:
                return text
        return ""

    def _read_range(self, key: str, start: int, end: int, *, session: _GroundingSession) -> bytes:
        if end <= start:
            return b""
        reader = getattr(self.object_store, "read_byte_range", None)
        if not callable(reader):
            return b""

        breaker = self.s3_circuit_breaker
        if breaker is not None and not breaker.allow():
            _logger.debug("S3 circuit breaker open, fast-failing read: key=%s", key)
            return b""

        with session.semaphore:
            future = session.executor.submit(reader, key, start, end)
            try:
                result = future.result(timeout=self.budgets.read_timeout_seconds)
            except FuturesTimeoutError:
                if breaker is not None:
                    breaker.on_failure()
                _logger.warning("S3 range read timed out after %.1fs: key=%s", self.budgets.read_timeout_seconds, key)
                return b""
            except Exception:
                if breaker is not None:
                    breaker.on_failure()
                _logger.error("S3 range read failed: key=%s", key, exc_info=True)
                return b""
            else:
                if breaker is not None:
                    breaker.on_success()
                return cast(bytes, result)

    @staticmethod
    def _raw_locator_payload(raw_locator: object) -> dict[str, Any]:
        if hasattr(raw_locator, "model_dump") and callable(raw_locator.model_dump):
            payload = raw_locator.model_dump(mode="python")
            return payload if isinstance(payload, dict) else {}
        if isinstance(raw_locator, dict):
            return raw_locator
        return {}

    def _raw_locator_dict(self, raw_locator: object) -> dict[str, str]:
        payload = self._raw_locator_payload(raw_locator)
        return {str(key): str(value) for key, value in payload.items() if value is not None and str(value).strip()}

    def _locator_key(self, raw_locator: object) -> str | None:
        payload = self._raw_locator_payload(raw_locator)
        for field_name in ("object_key", "visible_text_key", "storage_key"):
            value = payload.get(field_name)
            if value is None:
                continue
            normalized = str(value).strip()
            if normalized:
                return normalized
        return None

    @staticmethod
    def _safe_int(value: object | None) -> int | None:
        if value is None:
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _section_window_index(self, section: SectionRecord) -> int | None:
        for key in ("window_index", "refined_window_index"):
            value = self._safe_int(section.metadata_json.get(key))
            if value is not None:
                return value
        return None

    def _get_section(self, section_id: int | None) -> SectionRecord | None:
        if section_id is None:
            return None
        getter = getattr(self.metadata_repo, "get_section", None)
        if not callable(getter):
            return None
        return cast(SectionRecord | None, getter(section_id))

    def _get_asset(self, asset_id: int | None) -> AssetRecord | None:
        if asset_id is None:
            return None
        getter = getattr(self.metadata_repo, "get_asset", None)
        if not callable(getter):
            return None
        return cast(AssetRecord | None, getter(asset_id))

    def _get_layout_meta_cache(self, doc_id: int) -> LayoutMetaCacheRecord | None:
        getter = getattr(self.metadata_repo, "get_layout_meta_cache", None)
        if not callable(getter):
            return None
        return cast(LayoutMetaCacheRecord | None, getter(doc_id=doc_id))

    def _layout_neighbor_assets(self, section: SectionRecord) -> list[AssetRecord]:
        layout_cache = self._get_layout_meta_cache(section.doc_id)
        list_assets = getattr(self.metadata_repo, "list_assets", None)
        if layout_cache is None or not callable(list_assets):
            return []
        assets = list_assets(doc_id=section.doc_id)
        if not assets:
            return []
        elements = layout_cache.layout_json.get("elements")
        if not isinstance(elements, list):
            return []
        assets_by_ref = {
            str(asset.element_ref): asset
            for asset in assets
            if asset.element_ref is not None and str(asset.element_ref).strip()
        }
        if not assets_by_ref:
            return []
        section_path = tuple(str(part) for part in section.toc_path)
        text_blocks = [
            element
            for element in elements
            if self._layout_toc_path(element) == section_path
            and self._layout_page_no(element) in self._section_pages(section)
            and not self._layout_is_asset(element)
        ]
        ranked: list[tuple[tuple[int, float, int, int], AssetRecord]] = []
        for element in elements:
            if not self._layout_is_asset(element):
                continue
            asset = assets_by_ref.get(str(element.get("element_id", "")))
            if asset is None:
                continue
            page_no = self._layout_page_no(element)
            if page_no is None:
                continue
            same_path = self._layout_toc_path(element) == section_path
            near_page = page_no in self._section_pages(section)
            if not same_path and not near_page:
                continue
            y_distance = self._layout_vertical_distance(text_blocks, element)
            ranked.append(
                (
                    (
                        0 if same_path else 1,
                        y_distance,
                        abs(page_no - min(self._section_pages(section) or {page_no})),
                        asset.asset_id,
                    ),
                    asset,
                )
            )
        ranked.sort(key=lambda item: item[0])
        return [asset for _score, asset in ranked]

    @staticmethod
    def _layout_is_asset(element: object) -> bool:
        if not isinstance(element, dict):
            return False
        kind = str(element.get("kind", "") or "").strip().lower()
        return kind in {"table", "figure", "image", "chart", "image_summary"}

    @staticmethod
    def _layout_page_no(element: object) -> int | None:
        if not isinstance(element, dict):
            return None
        value = element.get("page_no")
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _layout_toc_path(element: object) -> tuple[str, ...]:
        if not isinstance(element, dict):
            return ()
        toc_path = element.get("toc_path")
        if not isinstance(toc_path, list):
            return ()
        return tuple(str(part) for part in toc_path if str(part).strip())

    @staticmethod
    def _layout_bbox(element: object) -> tuple[float, float, float, float] | None:
        if not isinstance(element, dict):
            return None
        bbox = element.get("bbox")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        return (x1, y1, x2, y2)

    def _layout_vertical_distance(self, text_blocks: list[object], asset_element: object) -> float:
        asset_bbox = self._layout_bbox(asset_element)
        asset_page = self._layout_page_no(asset_element)
        if asset_bbox is None or asset_page is None:
            return float("inf")
        top = asset_bbox[1]
        distances = []
        for text_block in text_blocks:
            if self._layout_page_no(text_block) != asset_page:
                continue
            bbox = self._layout_bbox(text_block)
            if bbox is None:
                continue
            distances.append(abs(top - bbox[3]))
        return min(distances, default=float("inf"))

    @staticmethod
    def _section_pages(section: SectionRecord) -> set[int]:
        if section.page_start is None and section.page_end is None:
            return set()
        start = section.page_start or section.page_end or 0
        end = section.page_end or section.page_start or start
        return {page for page in range(min(start, end), max(start, end) + 1) if page > 0}

    def _rank_local_items(
        self,
        items: list[EvidenceItem],
        *,
        query: str,
        query_terms: tuple[str, ...],
    ) -> list[EvidenceItem]:
        if not items:
            return []
        lexical_scores = {
            item.evidence_id: float(item.score) + 0.05 * keyword_overlap(query_terms, item.text) for item in items
        }
        rerank_bonus = self._rerank_bonus(query, items)
        return sorted(
            items,
            key=lambda item: (
                -(lexical_scores[item.evidence_id] + rerank_bonus.get(item.evidence_id, 0.0)),
                item.evidence_id,
            ),
        )

    def _rerank_bonus(self, query: str, items: list[EvidenceItem]) -> dict[str, float]:
        binding = self.rerank_binding
        rerank = getattr(binding, "rerank", None)
        if not callable(rerank) or len(items) <= 1:
            return {}
        rerank_items, rerank_documents = self._prepare_rerank_inputs(items)
        if len(rerank_items) <= 1:
            return {}
        try:
            raw_scores_or_ranking = list(
                rerank(
                    query,
                    rerank_documents,
                    batch_size=self.budgets.rerank_batch_size,
                    max_length=self.budgets.rerank_max_item_tokens,
                )
            )
        except (RuntimeError, ValueError):
            return {}
        ranking = self._normalize_rerank_order(raw_scores_or_ranking, item_count=len(rerank_items))
        bonuses: dict[str, float] = {}
        max_rank = max(len(ranking), 1)
        for rank, item_index in enumerate(ranking):
            if item_index < 0 or item_index >= len(rerank_items):
                continue
            bonuses[rerank_items[item_index].evidence_id] = 0.25 * (max_rank - rank) / max_rank
        return bonuses

    def _prepare_rerank_inputs(self, items: list[EvidenceItem]) -> tuple[list[EvidenceItem], list[str]]:
        max_items = max(self.budgets.rerank_max_items, 0)
        max_item_tokens = max(self.budgets.rerank_max_item_tokens, 1)
        max_total_tokens = max(self.budgets.rerank_max_total_tokens, 0)
        if max_items <= 0 or max_total_tokens <= 0:
            return [], []
        selected_items: list[EvidenceItem] = []
        documents: list[str] = []
        total_tokens = 0
        for item in items:
            if len(selected_items) >= max_items:
                break
            text = item.text.strip()
            if not text:
                continue
            clipped = self.token_accounting.clip(text, max_item_tokens)
            token_count = self.token_accounting.count(clipped)
            if token_count <= 0:
                continue
            remaining_tokens = max_total_tokens - total_tokens
            if remaining_tokens <= 0:
                break
            if token_count > remaining_tokens:
                clipped = self.token_accounting.clip(clipped, remaining_tokens)
                token_count = self.token_accounting.count(clipped)
                if token_count <= 0:
                    break
            selected_items.append(item)
            documents.append(clipped)
            total_tokens += token_count
        return selected_items, documents

    @staticmethod
    def _normalize_rerank_order(raw_scores_or_ranking: list[object], *, item_count: int) -> list[int]:
        if not raw_scores_or_ranking:
            return []
        if all(isinstance(value, int) for value in raw_scores_or_ranking):
            ranking = cast(list[int], raw_scores_or_ranking)
            return [v for v in ranking if 0 <= v < item_count]
        scored_indices: list[tuple[float, int]] = []
        for index, value in enumerate(raw_scores_or_ranking[:item_count]):
            try:
                score = float(cast(Any, value))
            except (TypeError, ValueError):
                continue
            scored_indices.append((score, index))
        return [index for _score, index in sorted(scored_indices, key=lambda item: (-item[0], item[1]))]

    @staticmethod
    def _section_candidate_overlap(section: SectionRecord, *, query_terms: tuple[str, ...]) -> int:
        summary_text = str(section.metadata_json.get("summary_text", "") or "")
        toc_text = " / ".join(section.toc_path)
        return keyword_overlap(query_terms, f"{toc_text} {summary_text}".strip())


__all__ = ["GroundingBudgets", "GroundingService"]

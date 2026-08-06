from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import rag.retrieval.grounding_service as grounding_module
from agent_runtime.text import DEFAULT_TOKENIZER_FALLBACK_MODEL
from rag.assembly import TokenAccountingService, TokenizerContract
from rag.ingest.asset_anchors import asset_anchor
from rag.retrieval.grounding_service import GroundingBudgets, GroundingService
from rag.schema.core import AssetRecord, LayoutMetaCacheRecord, SectionLocatorRecord, SectionRecord
from rag.schema.query import EvidenceItem, GroundingTarget


def _token_accounting() -> TokenAccountingService:
    return TokenAccountingService(
        TokenizerContract(
            embedding_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
            tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
            chunking_tokenizer_model_name=DEFAULT_TOKENIZER_FALLBACK_MODEL,
        )
    )


def _section_record(**kwargs: Any) -> SectionRecord:
    byte_start = int(kwargs.setdefault("byte_range_start", 0))
    byte_end = int(kwargs.setdefault("byte_range_end", max(byte_start + 1, 1)))
    char_start = int(kwargs.setdefault("char_range_start", byte_start))
    char_end = int(kwargs.setdefault("char_range_end", byte_end))
    visible_text_key = str(kwargs.setdefault("visible_text_key", f"doc-{kwargs.get('doc_id', 'unknown')}.txt"))
    kwargs.setdefault(
        "raw_locator",
        SectionLocatorRecord(
            visible_text_key=visible_text_key,
            char_range_start=char_start,
            char_range_end=char_end,
            byte_range_start=byte_start,
            byte_range_end=byte_end,
        ),
    )
    return SectionRecord(**kwargs)


def _evidence_item(**kwargs: Any) -> EvidenceItem:
    return EvidenceItem(**kwargs)


@dataclass
class _MetadataRepo:
    sections: dict[int, SectionRecord] = field(default_factory=dict)
    assets: dict[int, AssetRecord] = field(default_factory=dict)
    layouts: dict[int, LayoutMetaCacheRecord] = field(default_factory=dict)

    def get_section(self, section_id: int) -> SectionRecord | None:
        return self.sections.get(section_id)

    def list_sections(self, *, doc_id: int | None = None, source_id: int | None = None) -> list[SectionRecord]:
        del source_id
        sections = list(self.sections.values())
        if doc_id is not None:
            sections = [section for section in sections if section.doc_id == doc_id]
        return sorted(sections, key=lambda item: (item.order_index, item.section_id))

    def get_asset(self, asset_id: int) -> AssetRecord | None:
        return self.assets.get(asset_id)

    def list_assets(
        self,
        *,
        doc_id: int | None = None,
        source_id: int | None = None,
        section_id: int | None = None,
    ) -> list[AssetRecord]:
        del source_id
        assets = list(self.assets.values())
        if doc_id is not None:
            assets = [asset for asset in assets if asset.doc_id == doc_id]
        if section_id is not None:
            assets = [asset for asset in assets if asset.section_id == section_id]
        return sorted(assets, key=lambda item: (item.page_no, item.asset_id))

    def get_layout_meta_cache(self, doc_id: int) -> LayoutMetaCacheRecord | None:
        return self.layouts.get(doc_id)


@dataclass
class _ObjectStore:
    payloads: dict[str, bytes]
    range_calls: list[tuple[str, int, int]] = field(default_factory=list)
    read_calls: list[str] = field(default_factory=list)

    def read_byte_range(self, key: str, start: int, end: int) -> bytes:
        self.range_calls.append((key, start, end))
        return self.payloads[key][start:end]

    def read_bytes(self, key: str) -> bytes:
        self.read_calls.append(key)
        raise AssertionError("L5 should not fall back to full-file reads when byte ranges are available")


@dataclass
class _RerankBinding:
    ranking: list[int]

    def rerank(self, query: str, candidates: list[str], **kwargs: object) -> list[int]:
        del query, candidates, kwargs
        return list(self.ranking)


@dataclass
class _ScoreRerankBinding:
    scores: list[float]
    calls: list[tuple[list[str], dict[str, object]]] = field(default_factory=list)

    def rerank(self, query: str, candidates: list[str], **kwargs: object) -> list[float]:
        del query
        self.calls.append((list(candidates), dict(kwargs)))
        return list(self.scores[: len(candidates)])


@dataclass
class _TokenAccountingStub:
    chunks: list[str]

    def count(self, text: str) -> int:
        return len(text.split())

    def clip(self, text: str, budget: int, *, add_ellipsis: bool = False) -> str:
        words = text.split()
        clipped = " ".join(words[:budget])
        if add_ellipsis and len(words) > budget:
            return f"{clipped} ..."
        return clipped

    def chunk_text(self, text: str, *, chunk_token_size: int, chunk_overlap_tokens: int) -> list[str]:
        del text, chunk_token_size, chunk_overlap_tokens
        return list(self.chunks)


class _PassthroughTokenAccounting:
    def count(self, text: str) -> int:
        return len(text.split())

    def clip(self, text: str, budget: int, *, add_ellipsis: bool = False) -> str:
        del budget, add_ellipsis
        return text

    def chunk_text(self, text: str, *, chunk_token_size: int, chunk_overlap_tokens: int) -> list[str]:
        del chunk_token_size, chunk_overlap_tokens
        return [text]


def test_grounding_service_reads_section_byte_range_and_includes_neighbor_asset() -> None:
    metadata_repo = _MetadataRepo(
        sections={
            7: _section_record(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture", "Alpha"],
                order_index=1,
                page_start=2,
                page_end=2,
                byte_range_start=0,
                byte_range_end=23,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            )
        },
        assets={
            11: AssetRecord(
                asset_id=11,
                doc_id=42,
                source_id=9,
                section_id=7,
                asset_type="table",
                page_no=2,
                caption="Alpha capacity table",
                content_hash="asset-hash",
                storage_key="asset-11.txt",
            )
        },
    )
    object_store = _ObjectStore(
        payloads={
            "doc-42.txt": b"Alpha architecture body.",
            "asset-11.txt": b"unused",
        }
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=object_store,
        token_accounting=_token_accounting(),
    )

    grounded = service.ground(
        query="What is Alpha architecture?",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:7",
                doc_id=42,
                source_id=9,
                citation_anchor="Architecture / Alpha",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(
                    kind="section",
                    doc_id=42,
                    source_id=9,
                    section_id=7,
                    page_start=2,
                    page_end=2,
                    section_path=["Architecture", "Alpha"],
                    raw_locator={"summary_item_id": "7"},
                ),
            )
        ],
    )

    assert object_store.range_calls == [("doc-42.txt", 0, 23)]
    assert object_store.read_calls == []
    assert grounded[0].text == "Alpha architecture body"
    assert grounded[0].grounding_target is not None
    assert grounded[0].grounding_target.section_id == 7
    assert any(item.record_type == "asset" for item in grounded)
    assert any("Alpha capacity table" in item.text for item in grounded)


def test_grounding_service_replaces_short_table_anchor_in_section_text() -> None:
    anchor = asset_anchor("table-1")
    section_text = f"Expense standards {anchor} apply this year."
    table_markdown = "| Level | Amount |\n|---|---|\n| A | 100 |"
    metadata_repo = _MetadataRepo(
        sections={
            7: _section_record(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Policy", "Expense"],
                order_index=1,
                page_start=1,
                page_end=1,
                byte_range_start=0,
                byte_range_end=len(section_text.encode("utf-8")),
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
                has_table=True,
                neighbor_asset_count=1,
            )
        },
        assets={
            11: AssetRecord(
                asset_id=11,
                doc_id=42,
                source_id=9,
                section_id=7,
                asset_type="table",
                element_ref="table-1",
                page_no=1,
                content_hash="asset-hash",
                storage_key="asset-11.txt",
                metadata_json={
                    "asset_anchor": anchor,
                    "asset_text_preview": table_markdown,
                    "table_policy": "inline_context",
                },
            )
        },
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(payloads={"doc-42.txt": section_text.encode("utf-8")}),
        token_accounting=_PassthroughTokenAccounting(),
        budgets=GroundingBudgets(max_neighbor_assets=0),
    )

    grounded = service.ground(
        query="expense standards",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:7",
                doc_id=42,
                source_id=9,
                citation_anchor="Policy / Expense",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id=42, source_id=9, section_id=7),
            )
        ],
    )

    section_evidence = [item for item in grounded if item.record_type == "section"]
    assert len(section_evidence) == 1
    assert anchor not in section_evidence[0].text
    assert "TABLE_COMPUTE_ONLY" in section_evidence[0].text
    assert "<system_instruction>" in section_evidence[0].text


def test_grounding_service_replaces_compute_only_table_with_descriptive_block() -> None:
    anchor = asset_anchor("table-1")
    section_text = f"Sales ledger {anchor} requires calculation."
    table_markdown = "| Row | Amount |\n|---|---|\n| row0 | 1 |\n| row999 | 999 |"
    metadata_repo = _MetadataRepo(
        sections={
            7: _section_record(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Report", "Ledger"],
                order_index=1,
                byte_range_start=0,
                byte_range_end=len(section_text.encode("utf-8")),
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
                has_table=True,
                neighbor_asset_count=1,
            )
        },
        assets={
            11: AssetRecord(
                asset_id=11,
                doc_id=42,
                source_id=9,
                section_id=7,
                asset_type="table",
                element_ref="table-1",
                page_no=1,
                content_hash="asset-hash",
                storage_key="asset-11.txt",
                metadata_json={
                    "asset_anchor": anchor,
                    "asset_text_preview": table_markdown,
                    "table_policy": "compute_only",
                },
            )
        },
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(payloads={"doc-42.txt": section_text.encode("utf-8")}),
        token_accounting=_PassthroughTokenAccounting(),
        budgets=GroundingBudgets(max_neighbor_assets=0),
    )

    grounded = service.ground(
        query="sum sales ledger",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:7",
                doc_id=42,
                source_id=9,
                citation_anchor="Report / Ledger",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id=42, source_id=9, section_id=7),
            )
        ],
    )

    section_texts = [item.text for item in grounded if item.record_type == "section"]
    assert len(section_texts) == 1
    assert "TABLE_COMPUTE_ONLY" in section_texts[0]
    assert "<system_instruction>" in section_texts[0]
    assert "</system_instruction>" in section_texts[0]
    assert "row999" not in section_texts[0]


def test_grounding_service_direct_table_asset_uses_compute_block() -> None:
    metadata_repo = _MetadataRepo(
        assets={
            11: AssetRecord(
                asset_id=11,
                doc_id=42,
                source_id=9,
                section_id=7,
                asset_type="table",
                element_ref="table-1",
                page_no=1,
                content_hash="asset-hash",
                storage_key="asset-11.parquet",
                sheet_name="开票明细",
                row_count=8,
                column_count=3,
                sample_rows=[
                    {"区域": "华南", "季度": "Q1", "开票量": "999"},
                    {"区域": "华东", "季度": "Q2", "开票量": "777"},
                ],
                schema=[
                    {"name": "区域", "type": "enum(华南, 华东)"},
                    {"name": "季度", "type": "enum(Q1, Q2)"},
                    {"name": "开票量", "type": "number(min=1, max=999)"},
                ],
                metadata_json={
                    "asset_text_preview": "Sample row 华南 Q1 999",
                    "table_policy": "compute_only",
                    "estimated_tokens": 130,
                },
            )
        },
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(payloads={}),
        token_accounting=_PassthroughTokenAccounting(),
        budgets=GroundingBudgets(max_neighbor_assets=0),
    )

    grounded = service.ground(
        query="请计算华东区域 Q1 的开票量合计是多少？",
        evidence=[
            _evidence_item(
                evidence_id="summary:asset_summary:11",
                doc_id=42,
                source_id=9,
                citation_anchor="table@p1",
                text="table summary sample",
                score=0.91,
                record_type="asset_summary",
                grounding_target=GroundingTarget(
                    kind="asset",
                    doc_id=42,
                    source_id=9,
                    section_id=7,
                    asset_id=11,
                    page_start=1,
                    page_end=1,
                ),
            )
        ],
    )

    assert len(grounded) == 1
    assert grounded[0].record_type == "asset"
    assert "[TABLE_COMPUTE_ONLY:asset_id=11]" in grounded[0].text
    assert "<compute_request>" in grounded[0].text
    assert "Sample row 华南 Q1 999" not in grounded[0].text


def test_grounding_service_neighbor_table_asset_includes_compute_contract_without_semantic_rules() -> None:
    metadata_repo = _MetadataRepo(
        sections={
            7: _section_record(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["日报", "总表"],
                order_index=1,
                page_start=4,
                page_end=4,
                byte_range_start=0,
                byte_range_end=12,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
                has_table=True,
                neighbor_asset_count=1,
            )
        },
        assets={
            14: AssetRecord(
                asset_id=14,
                doc_id=42,
                source_id=9,
                section_id=7,
                asset_type="table",
                page_no=4,
                content_hash="asset-hash",
                storage_key="asset-14.parquet",
                sheet_name="2024-0317新增",
                row_count=3,
                column_count=3,
                sample_rows=[
                    {"区域公司": "总计（不含一体化）", "日_日提货": "131.074462", "日_日提货量": "601.2868"},
                    {"区域公司": "北方", "日_日提货": "19.22484", "日_日提货量": "136.3149"},
                    {"区域公司": "东北", "日_日提货": "6.307968", "日_日提货量": "2.81"},
                ],
                schema=[
                    {"name": "区域公司", "type": "text"},
                    {"name": "日_日提货", "type": "number(min=0, max=131.074462)"},
                    {"name": "日_日提货量", "type": "number(min=0, max=601.2868)"},
                ],
                metadata_json={"table_policy": "compute_only", "estimated_tokens": 80},
            )
        },
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(payloads={"doc-42.txt": b"daily table."}),
        token_accounting=_PassthroughTokenAccounting(),
        budgets=GroundingBudgets(max_neighbor_assets=1),
    )

    grounded = service.ground(
        query="日提货总量是多少",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:7",
                doc_id=42,
                source_id=9,
                citation_anchor="日报 / 总表",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id=42, source_id=9, section_id=7),
            )
        ],
    )

    asset_texts = [item.text for item in grounded if item.record_type == "asset"]
    assert asset_texts
    assert "[TABLE_COMPUTE_ONLY:asset_id=14]" in asset_texts[0]
    assert "Schema:" in asset_texts[0]
    assert "Sample rows" in asset_texts[0]
    assert "131.074462" in asset_texts[0]


def test_grounding_service_uses_layout_meta_cache_for_geometric_neighbor_assets() -> None:
    metadata_repo = _MetadataRepo(
        sections={
            7: _section_record(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture", "Alpha"],
                order_index=1,
                page_start=2,
                page_end=2,
                byte_range_start=0,
                byte_range_end=23,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            )
        },
        assets={
            11: AssetRecord(
                asset_id=11,
                doc_id=42,
                source_id=9,
                section_id=None,
                asset_type="table",
                element_ref="asset-table-1",
                page_no=2,
                caption="Alpha layout table",
                content_hash="asset-hash",
                storage_key="asset-11.txt",
            )
        },
        layouts={
            42: LayoutMetaCacheRecord(
                source_id=9,
                doc_id=42,
                content_hash="hash-42",
                layout_json={
                    "elements": [
                        {
                            "element_id": "text-1",
                            "kind": "text",
                            "toc_path": ["Architecture", "Alpha"],
                            "page_no": 2,
                            "bbox": [0, 0, 100, 80],
                        },
                        {
                            "element_id": "asset-table-1",
                            "kind": "table",
                            "toc_path": ["Architecture", "Alpha"],
                            "page_no": 2,
                            "bbox": [0, 84, 120, 140],
                        },
                    ]
                },
            )
        },
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(payloads={"doc-42.txt": b"Alpha architecture body."}),
        token_accounting=_token_accounting(),
    )

    grounded = service.ground(
        query="What is Alpha architecture?",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:7",
                doc_id=42,
                source_id=9,
                citation_anchor="Architecture / Alpha",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id=42, source_id=9, section_id=7),
            )
        ],
    )

    assert any(item.record_type == "asset" for item in grounded)
    assert any("Alpha layout table" in item.text for item in grounded)


def test_grounding_service_enforces_input_and_output_budgets() -> None:
    metadata_repo = _MetadataRepo(
        sections={
            7: _section_record(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture"],
                order_index=1,
                byte_range_start=0,
                byte_range_end=200,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            ),
            8: _section_record(
                section_id=8,
                doc_id=42,
                source_id=9,
                toc_path=["Operations"],
                order_index=2,
                byte_range_start=0,
                byte_range_end=200,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash-2",
            ),
        }
    )
    object_store = _ObjectStore(
        payloads={
            "doc-42.txt": (
                b"alpha one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                b"sixteen seventeen eighteen nineteen twenty"
            )
        }
    )
    token_accounting = _token_accounting()
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=object_store,
        token_accounting=token_accounting,
        budgets=GroundingBudgets(
            max_targets_to_read=1,
            max_output_tokens=12,
            local_window_tokens=6,
            local_window_overlap_tokens=0,
            max_neighbor_assets=0,
        ),
    )

    grounded = service.ground(
        query="alpha architecture",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:7",
                doc_id=42,
                source_id=9,
                citation_anchor="Architecture",
                text="section summary A",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id=42, section_id=7),
            ),
            _evidence_item(
                evidence_id="summary:section_summary:8",
                doc_id=42,
                source_id=9,
                citation_anchor="Operations",
                text="section summary B",
                score=0.89,
                grounding_target=GroundingTarget(kind="section", doc_id=42, section_id=8),
            ),
        ],
    )

    assert len(object_store.range_calls) == 1
    assert all(item.evidence_id.startswith("grounded:") for item in grounded)
    assert sum(token_accounting.count(item.text) for item in grounded) <= 12


def test_grounding_service_passthrough_when_grounding_target_is_missing() -> None:
    service = GroundingService(
        metadata_repo=_MetadataRepo(),
        object_store=_ObjectStore(payloads={}),
        token_accounting=_token_accounting(),
    )
    evidence = [
        _evidence_item(
            evidence_id="summary:section_summary:1",
            doc_id=42,
            citation_anchor="#a",
            text="existing evidence text",
            score=0.7,
        )
    ]

    grounded = service.ground(query="alpha", evidence=evidence)

    assert grounded == evidence


def test_grounding_service_expands_only_refined_sibling_sections_inside_parent_boundary() -> None:
    before_text = b"before window alpha condition"
    center_text = b"center window alpha exception"
    after_text = b"after window alpha approval"
    unrelated_text = b"unrelated parent alpha should not appear"
    metadata_repo = _MetadataRepo(
        sections={
            10: _section_record(
                section_id=10,
                doc_id=42,
                source_id=9,
                parent_section_id=10,
                toc_path=["Policy", "Credit"],
                order_index=0,
                byte_range_start=0,
                byte_range_end=len(before_text),
                visible_text_key="section-10.txt",
                section_kind="body",
                content_hash="h10",
                metadata_json={"window_index": 0, "window_count": 3},
            ),
            11: _section_record(
                section_id=11,
                doc_id=42,
                source_id=9,
                parent_section_id=10,
                toc_path=["Policy", "Credit"],
                order_index=1,
                byte_range_start=0,
                byte_range_end=len(center_text),
                visible_text_key="section-11.txt",
                section_kind="body",
                content_hash="h11",
                metadata_json={"window_index": 1, "window_count": 3},
            ),
            12: _section_record(
                section_id=12,
                doc_id=42,
                source_id=9,
                parent_section_id=10,
                toc_path=["Policy", "Credit"],
                order_index=2,
                byte_range_start=0,
                byte_range_end=len(after_text),
                visible_text_key="section-12.txt",
                section_kind="body",
                content_hash="h12",
                metadata_json={"window_index": 2, "window_count": 3},
            ),
            13: _section_record(
                section_id=13,
                doc_id=42,
                source_id=9,
                parent_section_id=13,
                toc_path=["Policy", "Credit"],
                order_index=3,
                byte_range_start=0,
                byte_range_end=len(unrelated_text),
                visible_text_key="section-13.txt",
                section_kind="body",
                content_hash="h13",
                metadata_json={"window_index": 0, "window_count": 1},
            ),
        }
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(
            payloads={
                "section-10.txt": before_text,
                "section-11.txt": center_text,
                "section-12.txt": after_text,
                "section-13.txt": unrelated_text,
            }
        ),
        token_accounting=_PassthroughTokenAccounting(),
        budgets=GroundingBudgets(
            local_window_tokens=64,
            local_window_overlap_tokens=0,
            max_neighbor_assets=0,
            neighbor_section_radius=1,
            max_neighbor_sections=2,
        ),
    )

    grounded = service.ground(
        query="alpha approval",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:11",
                doc_id=42,
                source_id=9,
                citation_anchor="Policy / Credit",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id=42, source_id=9, section_id=11),
            )
        ],
    )

    joined_text = "\n".join(item.text for item in grounded)
    assert "before window alpha condition" in joined_text
    assert "center window alpha exception" in joined_text
    assert "after window alpha approval" in joined_text
    assert "unrelated parent alpha should not appear" not in joined_text


def test_grounding_service_reuses_executor_across_queries(monkeypatch) -> None:
    created_executors: list[int] = []
    shutdown_calls: list[bool] = []

    class _FakeFuture:
        def __init__(self, value: bytes) -> None:
            self._value = value

        def result(self, timeout: float | None = None) -> bytes:
            del timeout
            return self._value

    class _FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            created_executors.append(max_workers)

        def __enter__(self) -> _FakeExecutor:
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            del exc_type, exc, tb

        def submit(self, fn, *args):
            return _FakeFuture(fn(*args))

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            del cancel_futures
            shutdown_calls.append(wait)

    monkeypatch.setattr(grounding_module, "ThreadPoolExecutor", _FakeExecutor)
    metadata_repo = _MetadataRepo(
        sections={
            7: _section_record(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture", "Alpha"],
                order_index=1,
                page_start=2,
                page_end=2,
                byte_range_start=0,
                byte_range_end=23,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            )
        },
        assets={
            11: AssetRecord(
                asset_id=11,
                doc_id=42,
                source_id=9,
                section_id=7,
                asset_type="table",
                page_no=2,
                caption=None,
                content_hash="asset-hash",
                storage_key="asset-11.txt",
            )
        },
    )
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(
            payloads={
                "doc-42.txt": b"Alpha architecture body.",
                "asset-11.txt": b"Alpha table preview",
            }
        ),
        token_accounting=_token_accounting(),
    )

    grounded = service.ground(
        query="What is Alpha architecture?",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:7",
                doc_id=42,
                source_id=9,
                citation_anchor="Architecture / Alpha",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id=42, source_id=9, section_id=7),
            )
        ],
    )

    assert grounded
    service.ground(
        query="What is Alpha architecture?",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:7",
                doc_id=42,
                source_id=9,
                citation_anchor="Architecture / Alpha",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id=42, source_id=9, section_id=7),
            )
        ],
    )
    assert created_executors == [service.budgets.max_parallel_reads]
    service.close()
    assert shutdown_calls == [True]


def test_grounding_service_uses_rerank_binding_for_local_evidence_scoring() -> None:
    metadata_repo = _MetadataRepo(
        sections={
            7: _section_record(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture"],
                order_index=1,
                byte_range_start=0,
                byte_range_end=64,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            )
        }
    )
    payload = "Background filler. Alpha engine handles ingestion. Miscellaneous notes."
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(payloads={"doc-42.txt": payload.encode("utf-8")}),
        token_accounting=_TokenAccountingStub(
            chunks=["Background filler", "Alpha engine handles", "Miscellaneous notes"]
        ),
        budgets=GroundingBudgets(local_window_tokens=3, local_window_overlap_tokens=0, max_neighbor_assets=0),
        rerank_binding=_RerankBinding(ranking=[1, 0, 2]),
    )

    grounded = service.ground(
        query="What handles ingestion?",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:7",
                doc_id=42,
                source_id=9,
                citation_anchor="Architecture",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id=42, section_id=7),
            )
        ],
    )

    assert grounded[0].text == "Alpha engine handles"


def test_grounding_service_caps_and_batches_rerank_bonus_inputs() -> None:
    metadata_repo = _MetadataRepo(
        sections={
            7: _section_record(
                section_id=7,
                doc_id=42,
                source_id=9,
                toc_path=["Architecture"],
                order_index=1,
                byte_range_start=0,
                byte_range_end=128,
                visible_text_key="doc-42.txt",
                section_kind="body",
                content_hash="section-hash",
            )
        }
    )
    reranker = _ScoreRerankBinding(scores=[0.1, 0.9])
    service = GroundingService(
        metadata_repo=metadata_repo,
        object_store=_ObjectStore(payloads={"doc-42.txt": b"unused"}),
        token_accounting=_TokenAccountingStub(
            chunks=[
                "first candidate has many words",
                "second candidate has many words",
                "third candidate should not reach rerank",
            ]
        ),
        budgets=GroundingBudgets(
            local_window_tokens=5,
            local_window_overlap_tokens=0,
            max_neighbor_assets=0,
            rerank_max_items=2,
            rerank_batch_size=4,
            rerank_max_item_tokens=2,
            rerank_max_total_tokens=4,
        ),
        rerank_binding=reranker,
    )

    grounded = service.ground(
        query="neutral query",
        evidence=[
            _evidence_item(
                evidence_id="summary:section_summary:7",
                doc_id=42,
                source_id=9,
                citation_anchor="Architecture",
                text="section summary",
                score=0.91,
                grounding_target=GroundingTarget(kind="section", doc_id=42, section_id=7),
            )
        ],
    )

    assert grounded[0].text == "second candidate has many words"
    assert len(reranker.calls) == 1
    candidates, kwargs = reranker.calls[0]
    assert candidates == ["first candidate", "second candidate"]
    assert kwargs == {"batch_size": 4, "max_length": 2}

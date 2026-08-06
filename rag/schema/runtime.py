from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from rag.schema.model_protocols import OcrVisionRepo, VisualDescriptionRepo

if TYPE_CHECKING:
    from rag.schema.core import (
        AssetRecord,
        Document,
        LayoutMetaCacheRecord,
        ProcessingStateRecord,
        SectionRecord,
        Source,
    )
    from rag.schema.graph import GraphEdge, GraphNode
    from rag.schema.query import RetrievalSignals


class RuntimeMode(StrEnum):
    FAST = "fast"
    DEEP = "deep"


# TODO(agent): replace with AgentToolPolicy + WebSearchTool.
# The Agent decides whether to call a tool; AccessPolicy controls tool execution.
class AccessPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed_runtimes: frozenset[RuntimeMode] = Field(
        default_factory=lambda: frozenset({RuntimeMode.FAST, RuntimeMode.DEEP})
    )

    @classmethod
    def default(cls) -> AccessPolicy:
        return cls()

    def allows_runtime(self, mode: RuntimeMode) -> bool:
        return mode in self.allowed_runtimes


class ProviderAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    capability: str
    provider: str
    location: str
    model: str | None = None
    status: str
    error: str | None = None
    latency_ms: float | None = None


class RetrievalDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval_profile: str | None = None
    branch_hits: dict[str, int] = Field(default_factory=dict)
    branch_limits: dict[str, int] = Field(default_factory=dict)
    planning_complexity_gate: str | None = None
    semantic_route: str | None = None
    target_collections: list[str] = Field(default_factory=list)
    predicate_strategy: str | None = None
    predicate_expression: str | None = None
    version_gate_applied: bool = False
    operator_plan: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None
    sparse_query: str | None = None
    
    reranked_evidence_ids: list[str] = Field(default_factory=list)
    reranked_benchmark_doc_ids: list[str] = Field(default_factory=list)
    
    embedding_provider: str | None = None
    rerank_provider: str | None = None
    rerank_skipped: bool = False
    attempts: list[ProviderAttempt] = Field(default_factory=list)
    fusion_strategy: str | None = None
    fusion_alpha: float | None = None
    fusion_input_count: int = 0
    fused_count: int = 0
    graph_expanded: bool = False
    retrieval_signals: RetrievalSignals | None = None
    retrieval_signals_debug: dict[str, object] = Field(default_factory=dict)
    pre_rerank_count: int = 0
    post_cleanup_count: int = 0
    top1_confidence: float | None = None
    exit_decision: str | None = None
    fallback_triggered: list[str] = Field(default_factory=list)
    collapsed_candidate_count: int = 0


class ModelDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    synthesis_provider: str | None = None
    attempts: list[ProviderAttempt] = Field(default_factory=list)
    fallback_reason: str | None = None
    failed_stage: str | None = None
    degraded_to_retrieval_only: bool = False


class QueryDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    model: ModelDiagnostics = Field(default_factory=ModelDiagnostics)


class CapabilityHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    configured: bool
    available: bool
    model: str | None = None
    error: str | None = None


class ProviderHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    location: str
    capabilities: dict[str, CapabilityHealth] = Field(default_factory=dict)


class IndexHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: int = 0
    sections: int = 0
    assets: int = 0
    vectors: int = 0
    missing_vectors: int = 0


class HealthReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    providers: list[ProviderHealth] = Field(default_factory=list)
    indices: IndexHealth = Field(default_factory=IndexHealth)


class DocumentProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"
    DELETED = "deleted"
    REBUILDING = "rebuilding"


class DocumentPipelineStage(StrEnum):
    INGEST = "ingest"
    PARSE = "parse"
    ROUTE = "route"
    EXTRACT = "extract"
    SUMMARIZE = "summarize"
    PERSIST = "persist"
    INDEX = "index"
    DELETE = "delete"
    REBUILD = "rebuild"


class DocumentStatusRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: int
    source_id: int
    location: str
    content_hash: str
    status: DocumentProcessingStatus
    stage: DocumentPipelineStage | str
    attempts: int = 0
    error_message: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str] = Field(default_factory=dict)


class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    namespace: str
    cache_key: str
    payload: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None


class TelemetryEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    category: str
    payload: dict[str, str | int | float | bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationMetricInput(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")

    citation_precision: float = Field(ge=0.0, le=1.0)
    evidence_sufficient: bool
    conflict_detected: bool
    simple_query_latency_seconds: float = Field(
        ge=0.0,
        validation_alias=AliasChoices("latency_seconds", "simple_query_latency_seconds"),
    )
    deep_query_completion_quality: float = Field(
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("deep_quality", "deep_query_completion_quality"),
    )
    preservation_useful: bool


class EvaluationMetricSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_precision: float
    evidence_sufficiency_rate: float
    conflict_detection_quality: float
    simple_query_latency: float
    deep_query_completion_quality: float
    preservation_usefulness: float

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


def coerce_evaluation_metric_input(
    item: EvaluationMetricInput | Mapping[str, Any],
) -> EvaluationMetricInput:
    if isinstance(item, EvaluationMetricInput):
        return item
    return EvaluationMetricInput.model_validate(item)


@dataclass(frozen=True)
class RetrievalRecord:
    item_id: str
    item_kind: str = "section"
    doc_id: int = 0
    source_id: int = 0
    text: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def evidence_id(self) -> str:
        return self.item_id


@dataclass(frozen=True)
class VectorSearchResult(RetrievalRecord):
    score: float = 0.0


@dataclass(frozen=True)
class StoredVectorEntry:
    item_id: str
    item_kind: str
    embedding_space: str
    doc_id: int
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)

@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    score: float = 0.0
    source: str = "web"


class WebFetchRepo(Protocol):
    def fetch(self, location: str) -> str: ...


class WebSearchRepo(Protocol):
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]: ...


class VectorRepo(Protocol):
    def upsert(
        self,
        item_id: str,
        vector: Iterable[float],
        *,
        metadata: dict[str, str] | None = None,
        embedding_space: str = "default",
        item_kind: str = "section",
    ) -> None: ...

    def search(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[int] | None = None,
        embedding_space: str = "default",
        item_kind: str = "section",
    ) -> list[VectorSearchResult]: ...

    def get_entry(
        self,
        item_id: str,
        *,
        embedding_space: str = "default",
        item_kind: str = "section",
    ) -> StoredVectorEntry | None: ...

    def existing_item_ids(
        self,
        item_ids: Sequence[str],
        *,
        embedding_space: str | None = None,
        item_kind: str | None = "section",
    ) -> set[str]: ...

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_records: bool = False,
    ) -> int: ...

    def delete_for_documents(
        self,
        doc_ids: Sequence[int],
        *,
        item_kind: str | None = None,
    ) -> int: ...

    def close(self) -> None: ...


class MetadataRepo(Protocol):
    def save_source(self, source: Source) -> Source: ...

    def get_source(self, source_id: int) -> Source | None: ...

    def get_source_by_location_and_hash(self, location: str, content_hash: str) -> Source | None: ...

    def find_source_by_content_hash(self, content_hash: str) -> Source | None: ...

    def get_latest_source_for_location(self, location: str) -> Source | None: ...

    def list_sources(self, location: str | None = None) -> list[Source]: ...

    def save_document(self, document: Document) -> Document: ...

    def get_document(self, doc_id: int) -> Document | None: ...

    def list_documents(
        self,
        source_id: int | None = None,
        *,
        active_only: bool = False,
        version_group_id: int | None = None,
    ) -> list[Document]: ...

    def close(self) -> None: ...


class GroundingMetadataRepo(Protocol):
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

    def get_layout_meta_cache(self, doc_id: int) -> LayoutMetaCacheRecord | None: ...


class ProcessingStateRepo(Protocol):
    def save_processing_state(self, record: ProcessingStateRecord) -> ProcessingStateRecord: ...

    def get_processing_state(self, doc_id: int) -> ProcessingStateRecord | None: ...

    def list_processing_states(
        self,
        *,
        source_id: int | None = None,
        status: str | None = None,
        stage: str | None = None,
    ) -> list[ProcessingStateRecord]: ...

    def delete_processing_state(self, doc_id: int) -> None: ...


class DataContractMetadataRepo(MetadataRepo, GroundingMetadataRepo, ProcessingStateRepo, Protocol):
    def find_document_by_hash(self, file_hash: str) -> Document | None: ...

    def increment_document_reference_count(self, doc_id: int, *, amount: int = 1) -> Document: ...

    def save_section(self, section: SectionRecord) -> SectionRecord: ...

    def save_asset(self, asset: AssetRecord) -> AssetRecord: ...

    def deactivate_document(self, doc_id: int) -> Document: ...

    def set_document_index_state(
        self,
        doc_id: int,
        *,
        is_indexed: bool | None = None,
        index_ready: bool | None = None,
        embedding_model_id: str | None = None,
        indexed_at: datetime | None = None,
        last_index_error: str | None = None,
    ) -> Document: ...

    def set_document_storage_tier(self, doc_id: int, *, storage_tier: Any) -> Document: ...

    # ── merged from deleted DocumentStatusRepo ──

    def save_document_status(self, status: DocumentStatusRecord) -> DocumentStatusRecord: ...

    def get_document_status(self, doc_id: int) -> DocumentStatusRecord | None: ...

    def list_document_statuses(
        self,
        *,
        source_id: int | None = None,
        status: str | None = None,
    ) -> list[DocumentStatusRecord]: ...

    def delete_document_status(self, doc_id: int) -> None: ...


class CacheRepo(Protocol):
    def save_cache_entry(self, entry: CacheEntry) -> CacheEntry: ...

    def get_cache_entry(self, cache_key: str, *, namespace: str = "default") -> CacheEntry | None: ...

    def list_cache_entries(self, *, namespace: str | None = None) -> list[CacheEntry]: ...

    def delete_cache_entry(self, cache_key: str, *, namespace: str = "default") -> None: ...

    def purge_expired_cache_entries(self, *, now: datetime | None = None) -> int: ...

    def close(self) -> None: ...


class GraphRepo(Protocol):
    def save_node(self, node: GraphNode) -> None: ...

    def merge_node_evidence(self, node_id: str, evidence_ids: Sequence[str]) -> None: ...

    def get_node(self, node_id: str) -> GraphNode | None: ...

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]: ...

    def list_nodes_by_alias(self, alias: str, *, node_type: str | None = None) -> list[GraphNode]: ...

    def list_node_evidence_ids(self, node_id: str) -> list[str]: ...

    def save_candidate_edge(self, edge: GraphEdge) -> None: ...

    def save_edge(self, edge: GraphEdge) -> None: ...

    def bind_node_evidence(self, node_id: str, evidence_ids: Sequence[str]) -> None: ...

    def promote_candidate_edge(self, edge_id: str) -> None: ...

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None: ...

    def list_candidate_edges(self) -> list[GraphEdge]: ...

    def list_edges(self) -> list[GraphEdge]: ...

    def delete_node(self, node_id: str) -> None: ...

    def delete_edge(self, edge_id: str, *, include_candidates: bool = True) -> None: ...

    def list_edges_for_node(self, node_id: str, *, include_candidates: bool = False) -> list[GraphEdge]: ...

    def list_edges_for_evidence(self, evidence_id: str, *, include_candidates: bool = False) -> list[GraphEdge]: ...

    def delete_by_evidence_ids(self, evidence_ids: Sequence[str]) -> tuple[list[str], list[str]]: ...

    def close(self) -> None: ...


class ObjectStore(Protocol):
    def put_bytes(self, content: bytes, *, suffix: str = "") -> str: ...

    def read_bytes(self, key: str) -> bytes: ...

    def read_byte_range(self, key: str, start: int, end: int) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def path_for_key(self, key: str) -> Path: ...


__all__ = [
    "AccessPolicy",
    "CacheEntry",
    "CacheRepo",
    "CapabilityHealth",
    "DataContractMetadataRepo",
    "DocumentPipelineStage",
    "DocumentProcessingStatus",
    "DocumentStatusRecord",
    "EvaluationMetricInput",
    "EvaluationMetricSummary",
    "GraphRepo",
    "GroundingMetadataRepo",
    "HealthReport",
    "IndexHealth",
    "MetadataRepo",
    "ModelDiagnostics",
    "ObjectStore",
    "OcrVisionRepo",
    "ProviderAttempt",
    "ProviderHealth",
    "ProcessingStateRepo",
    "QueryDiagnostics",
    "RetrievalDiagnostics",
    "RetrievalRecord",
    "RuntimeMode",
    "SearchResult",
    "StoredVectorEntry",
    "TelemetryEvent",
    "VectorRepo",
    "VectorSearchResult",
    "VisualDescriptionRepo",
    "WebFetchRepo",
    "WebSearchRepo",
    "coerce_evaluation_metric_input",
]

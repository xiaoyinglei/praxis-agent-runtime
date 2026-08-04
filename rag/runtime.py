from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from agent_runtime.modeling.contracts import DEFAULT_LLM_STAGE_BUDGETS, LLMCallStage, LLMStageBudget
from agent_runtime.modeling.gateway import LLMGateway
from rag.assembly import (
    AssemblyDiagnostics,
    AssemblyRequest,
    CapabilityAssemblyService,
    CapabilityBundle,
    CapabilityCatalog,
    ChatCapabilityBinding,
    ProviderConfig,
    TokenAccountingService,
    TokenizerContract,
)
from rag.assembly.support import build_provider
from rag.ingest.parsers import (
    DoclingParserRepo,
    ExcelParserRepo,
    ExtractionDispatcher,
    ImageParserRepo,
    PptxParserRepo,
    create_default_ocr_repo,
)
from rag.ingest.pipeline import (
    BatchIngestItemResult,
    BatchIngestResult,
    DirectContentItem,
    IngestPipeline,
    IngestPipelineResult,
    IngestRequest,
    MetadataWriterRepo,
)
from rag.ingest.pipeline import (
    SummaryIndexRepo as PipelineSummaryIndexRepo,
)
from rag.ingest.retrievalsummarizer import RetrievalSummarizer, RetrievalSummaryConfig
from rag.ingest.section_refiner import SectionRefiner
from rag.ingest.table_executor import TableExecutor, _AssetMetadataRepo, _RangeReadableObjectStore
from rag.providers.generation import AnswerGenerationService, AnswerGenerator, GeneratorBinding
from rag.query_pipeline import _QueryPipeline
from rag.retrieval.authorization_service import AuthorizationService
from rag.retrieval.context import (
    ContextPromptBuilder,
    EvidenceTruncator,
)
from rag.retrieval.evidence import ContextEvidenceMerger, EvidenceService
from rag.retrieval.graph import (
    GraphExpansionService,
    RetrievalMetadataRepoProtocol,
    SearchBackedRetrievalFactory,
    VectorSearchRepoProtocol,
)
from rag.retrieval.grounding_service import GroundingService
from rag.retrieval.models import (
    PublicQueryResult,
    QueryOptions,
    RAGQueryResult,
)
from rag.retrieval.orchestrator import RetrievalService, RetrievalServiceConfig, RetrieverFn
from rag.retrieval.planning_graph import PlanningGraph
from rag.retrieval.reranker import ModelBackedRerankService
from rag.retrieval.synthesis_service import SynthesisService
from rag.schema.core import Document, ProcessingStateRecord, Source, SourceType, StorageTier
from rag.schema.graph import GraphEdge, GraphNode
from rag.schema.runtime import CacheEntry, DataContractMetadataRepo, ProviderAttempt, VisualDescriptionRepo
from rag.storage import StorageBundle, StorageConfig
from rag.storage.data_contract_service import DataContractService
from rag.storage.data_contract_service import SummaryIndexRepo as DataContractSummaryIndexRepo
from rag.storage.index_sync_worker import IndexSyncWorker
from rag.storage.storage_lifecycle_service import StorageLifecycleService
from rag.storage.storage_lifecycle_worker import StorageLifecycleWorker
from rag.utils.telemetry import TelemetryService

_RUNTIME_CONTRACT_NAMESPACE = "rag_runtime"
_RUNTIME_CONTRACT_KEY = "core_contract_v1"


def _supports_data_contract_metadata_repo(repo: object) -> bool:
    required = (
        "find_document_by_hash",
        "increment_document_reference_count",
        "save_source",
        "save_document",
        "save_section",
        "save_asset",
        "get_document",
        "list_documents",
        "get_section",
        "list_sections",
        "get_asset",
        "list_assets",
        "get_layout_meta_cache",
        "save_processing_state",
        "get_processing_state",
        "list_processing_states",
        "delete_processing_state",
        "deactivate_document",
        "set_document_index_state",
        "set_document_storage_tier",
    )
    return all(callable(getattr(repo, name, None)) for name in required)


def _supports_summary_index_repo(repo: object) -> bool:
    required = ("search", "delete", "upsert_record", "upsert_records")
    return all(callable(getattr(repo, name, None)) for name in required)


def _supports_storage_lifecycle_repo(repo: object) -> bool:
    required = (
        "get_document",
        "list_documents",
        "get_processing_state",
        "save_processing_state",
        "list_processing_states",
        "set_document_storage_tier",
    )
    return all(callable(getattr(repo, name, None)) for name in required)


class _ChatGeneratorAdapter:
    def __init__(self, binding: object | None) -> None:
        self._binding = binding

    @property
    def provider_name(self) -> str | None:
        value = getattr(self._binding, "provider_name", None)
        return value if isinstance(value, str) and value else None

    @property
    def model_name(self) -> str | None:
        value = getattr(self._binding, "model_name", None)
        return value if isinstance(value, str) and value else None

    def generate_text(self, *, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> str:
        chat = getattr(self._binding, "chat", None)
        if not callable(chat):
            raise RuntimeError("chat generation capability is not configured")
        call_kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        call_kwargs.update(kwargs)
        return str(chat(prompt, **call_kwargs))

    def generate_structured(self, *, prompt: str, schema: type[Any], **kwargs: Any) -> Any:
        backend = getattr(self._binding, "backend", None)
        generate_structured = getattr(backend, "generate_structured", None)
        if callable(generate_structured):
            return generate_structured(prompt=prompt, schema=schema, **kwargs)
        return schema.model_validate_json(self.generate_text(prompt=prompt))

    def generate_text_with_usage(self, *, prompt: str, **kwargs: Any) -> object:
        backend = getattr(self._binding, "backend", None)
        method = getattr(backend, "generate_text_with_usage", None)
        if callable(method):
            return method(prompt=prompt, **kwargs)
        from agent_runtime.modeling.contracts import LLMProviderResult

        return LLMProviderResult(value=self.generate_text(prompt=prompt, **kwargs))

    def generate_structured_with_usage(
        self,
        *,
        prompt: str,
        schema: type[Any],
        **kwargs: Any,
    ) -> object:
        backend = getattr(self._binding, "backend", None)
        method = getattr(backend, "generate_structured_with_usage", None)
        if callable(method):
            return method(prompt=prompt, schema=schema, **kwargs)
        from agent_runtime.modeling.contracts import LLMProviderResult

        return LLMProviderResult(value=self.generate_structured(prompt=prompt, schema=schema, **kwargs))


class _LazySummaryGeneratorAdapter:
    def __init__(self, *, binding: object | None) -> None:
        self._adapter = _ChatGeneratorAdapter(binding)

    @property
    def provider_name(self) -> str | None:
        return self._adapter.provider_name

    @property
    def model_name(self) -> str | None:
        return self._adapter.model_name

    def generate_text(self, *, prompt: str, max_tokens: int | None = None, **kwargs: Any) -> str:
        return self._adapter.generate_text(prompt=prompt, max_tokens=max_tokens, **kwargs)

    def generate_structured(self, *, prompt: str, schema: type[Any], **kwargs: Any) -> Any:
        return self._adapter.generate_structured(prompt=prompt, schema=schema, **kwargs)


def _generator_bindings_from_chat_bindings(
    chat_bindings: Sequence[object],
    *,
    token_accounting: object | None = None,
    model_context_tokens: int = 32_768,
    stage_budgets: dict[LLMCallStage, LLMStageBudget] | None = None,
) -> tuple[GeneratorBinding, ...]:
    bindings: list[GeneratorBinding] = []
    for binding in chat_bindings:
        location = str(getattr(binding, "location", "") or "local")
        adapter = _ChatGeneratorAdapter(binding)
        bindings.append(
            GeneratorBinding(
                backend=adapter,
                provider_name=str(getattr(binding, "provider_name", "chat")),
                model_name=getattr(binding, "model_name", None),
                location="cloud" if location == "cloud" else "local",
                gateway=(
                    None
                    if token_accounting is None
                    else LLMGateway(
                        generator=adapter,
                        token_accounting=cast(Any, token_accounting),
                        model_context_tokens=model_context_tokens,
                        stage_budgets=stage_budgets,
                    )
                ),
            )
        )
    return tuple(bindings)


class _InstrumentedReranker:
    def __init__(self, rerank_service: object) -> None:
        self._rerank_service = rerank_service
        self.provider_name = getattr(rerank_service, "provider_name", "formal-rerank")
        self.rerank_model_name = getattr(rerank_service, "rerank_model_name", "unconfigured-reranker")
        self.last_provider: str | None = self.provider_name
        self.last_attempts: list[ProviderAttempt] = []

    def __call__(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        return self.rerank(query, documents, **kwargs)

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        attempt = ProviderAttempt(
            stage="rerank",
            capability="rerank",
            provider=self.provider_name,
            location="core",
            model=self.rerank_model_name,
            status="success",
        )
        rerank = getattr(self._rerank_service, "rerank", None)
        if not callable(rerank):
            self.last_attempts = [attempt.model_copy(update={"status": "failed", "error": "rerank not supported"})]
            raise RuntimeError("rerank not supported")
        try:
            result = list(rerank(query, documents, **kwargs))
        except RuntimeError as exc:
            self.last_attempts = [attempt.model_copy(update={"status": "failed", "error": str(exc)})]
            raise
        self.provider_name = getattr(self._rerank_service, "provider_name", self.provider_name)
        self.rerank_model_name = getattr(self._rerank_service, "rerank_model_name", self.rerank_model_name)
        self.last_provider = self.provider_name
        self.last_attempts = [
            attempt.model_copy(update={"provider": self.provider_name, "model": self.rerank_model_name})
        ]
        return result


@dataclass(frozen=True, slots=True)
class RuntimeDeleteResult:
    deleted_doc_ids: list[int]
    deleted_source_ids: list[int]
    deleted_vector_count: int


@dataclass(frozen=True, slots=True)
class RuntimeRebuildResult:
    rebuilt_doc_ids: list[int]
    results: list[IngestPipelineResult]


@dataclass(slots=True)
class RAGRuntime:
    storage: StorageConfig
    request: AssemblyRequest = field(default_factory=AssemblyRequest)
    assembly_service: CapabilityAssemblyService = field(default_factory=CapabilityAssemblyService, repr=False)
    telemetry_service: TelemetryService | None = None
    vlm_repo: VisualDescriptionRepo | None = None
    generation_config: object | None = None  # GenerationConfig from agent_runtime.modeling.config
    chat_context_window_tokens: int = 32_768
    llm_stage_budgets: dict[LLMCallStage, LLMStageBudget] = field(
        default_factory=lambda: {stage: budget.model_copy() for stage, budget in DEFAULT_LLM_STAGE_BUDGETS.items()}
    )
    capability_bundle: CapabilityBundle = field(init=False, repr=False)
    token_contract: TokenizerContract = field(init=False, repr=False)
    token_accounting: TokenAccountingService = field(init=False, repr=False)
    stores: StorageBundle = field(init=False)
    ingest_pipeline: IngestPipeline = field(init=False, repr=False)
    retrieval_service: RetrievalService = field(init=False, repr=False)
    agent_service: object | None = field(init=False, default=None, repr=False)
    query_pipeline: _QueryPipeline = field(init=False, repr=False)
    index_sync_worker: IndexSyncWorker | None = field(init=False, default=None, repr=False)
    storage_lifecycle_worker: StorageLifecycleWorker | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.capability_bundle = self.assembly_service.assemble_request(self.request)
        self.token_contract = self.capability_bundle.token_contract
        self.token_accounting = self.capability_bundle.token_accounting
        self.stores = self.storage.build()
        self._register_or_validate_runtime_contract()
        self._build_pipelines()

    @classmethod
    def from_request(
        cls,
        *,
        storage: StorageConfig,
        request: AssemblyRequest,
        assembly_service: CapabilityAssemblyService | None = None,
        telemetry_service: TelemetryService | None = None,
        vlm_repo: VisualDescriptionRepo | None = None,
        generation_config: object | None = None,
        chat_context_window_tokens: int = 32_768,
        llm_stage_budgets: dict[LLMCallStage, LLMStageBudget] | None = None,
    ) -> RAGRuntime:
        return cls(
            storage=storage,
            request=request,
            assembly_service=assembly_service or CapabilityAssemblyService(),
            telemetry_service=telemetry_service,
            vlm_repo=vlm_repo,
            generation_config=generation_config,
            chat_context_window_tokens=chat_context_window_tokens,
            llm_stage_budgets=(
                llm_stage_budgets or {stage: budget.model_copy() for stage, budget in DEFAULT_LLM_STAGE_BUDGETS.items()}
            ),
        )

    @property
    def diagnostics(self) -> AssemblyDiagnostics:
        return self.capability_bundle.diagnostics

    @property
    def catalog(self) -> CapabilityCatalog:
        return self.assembly_service.catalog_from_environment(config=self.request.config)

    @property
    def runtime_contract_payload(self) -> dict[str, str | int | bool]:
        return self.capability_bundle.runtime_contract_payload

    def configure_summary_generator(
        self,
        *,
        provider_kind: str,
        model: str | None = None,
        model_path: str | None = None,
        backend: str | None = None,
        strict_generation: bool | None = None,
    ) -> None:
        provider_config = ProviderConfig(
            provider_kind=provider_kind,
            location="local",
            chat_model=model,
            chat_model_path=model_path,
            chat_backend=backend,
        )
        provider = build_provider(provider_config)
        binding = ChatCapabilityBinding(provider, location=provider_config.location)
        self.ingest_pipeline.configure_summarizer(
            RetrievalSummarizer(
                llm_client=_ChatGeneratorAdapter(binding),
                token_accounting=self.token_accounting,
                config=self._summary_config(strict_generation=strict_generation),
                gateway=self._gateway_for_chat_binding(_ChatGeneratorAdapter(binding)),
            )
        )

    def configure_default_summary_generator(
        self,
        *,
        raw_text_mode: bool = False,
        strict_generation: bool | None = None,
    ) -> None:
        chat_binding = self.capability_bundle.chat_bindings[0] if self.capability_bundle.chat_bindings else None
        self.ingest_pipeline.configure_summarizer(
            RetrievalSummarizer(
                llm_client=_LazySummaryGeneratorAdapter(binding=chat_binding),
                token_accounting=self.token_accounting,
                config=self._summary_config(
                    raw_text_mode=raw_text_mode,
                    strict_generation=strict_generation,
                ),
                gateway=self._gateway_for_chat_binding(_LazySummaryGeneratorAdapter(binding=chat_binding)),
            )
        )

    def _summary_config(
        self,
        *,
        raw_text_mode: bool = False,
        strict_generation: bool | None = None,
    ) -> RetrievalSummaryConfig:
        summary_config = RetrievalSummaryConfig(
            raw_text_mode=raw_text_mode,
            strict_generation=bool(strict_generation),
        )
        if self.generation_config is None:
            return summary_config
        gen_summary = getattr(self.generation_config, "summary", None)
        if gen_summary is None:
            return summary_config
        max_tokens = getattr(gen_summary, "max_tokens", None)
        temperature = getattr(gen_summary, "temperature", None)
        if max_tokens is None and temperature is None:
            return summary_config
        return replace(
            summary_config,
            max_output_tokens=(summary_config.max_output_tokens if max_tokens is None else int(max_tokens)),
            temperature=summary_config.temperature if temperature is None else float(temperature),
        )

    def diagnostics_payload(self) -> dict[str, object]:
        return {
            "status": self.diagnostics.status,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "message": issue.message,
                }
                for issue in self.diagnostics.issues
            ],
            "decisions": [
                {
                    "capability": decision.capability,
                    "source": decision.source,
                    "provider_kind": decision.provider_kind,
                    "provider_name": decision.provider_name,
                    "model_name": decision.model_name,
                    "location": decision.location,
                    "reason": decision.reason,
                    "selected": decision.selected,
                }
                for decision in self.diagnostics.decisions
            ],
            "runtime_contract": self.runtime_contract_payload,
        }

    def close(self) -> None:
        close_grounding = getattr(self.query_pipeline.grounding_service, "close", None)
        if callable(close_grounding):
            close_grounding()
        self.stores.close()

    def __enter__(self) -> RAGRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def insert(self, request: IngestRequest | None = None, /, **kwargs: Any) -> IngestPipelineResult:
        self._register_or_validate_runtime_contract()
        result = self.ingest_pipeline.run(self._coerce_ingest_request(request, **kwargs))
        self.process_pending_index_sync(max_tasks=1)
        self.process_pending_storage_lifecycle(max_tasks=1)
        return result

    def insert_many(
        self,
        requests: list[IngestRequest],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestResult:
        self._register_or_validate_runtime_contract()
        if not requests:
            return BatchIngestResult(results=[])

        if not continue_on_error:
            batch_results = self.ingest_pipeline.run_many(requests)
            self.process_pending_index_sync(max_tasks=max(1, len(batch_results)))
            self.process_pending_storage_lifecycle(max_tasks=1)
            return BatchIngestResult(
                results=[
                    BatchIngestItemResult(request=request, result=result)
                    for request, result in zip(requests, batch_results, strict=True)
                ]
            )

        results: list[BatchIngestItemResult] = []
        for request in requests:
            try:
                result = self.insert(request)
            except Exception as exc:
                if not continue_on_error:
                    raise
                results.append(BatchIngestItemResult(request=request, error=str(exc)))
                continue
            results.append(BatchIngestItemResult(request=request, result=result))
        return BatchIngestResult(results=results)

    def insert_content_list(
        self,
        items: list[object],
        *,
        continue_on_error: bool = False,
    ) -> BatchIngestResult:
        requests = [self._coerce_direct_content_item(item, index=index) for index, item in enumerate(items, start=1)]
        return self.insert_many(requests, continue_on_error=continue_on_error)

    def query(
        self,
        *args: Any,
        options: QueryOptions | None = None,
        **kwargs: Any,
    ) -> RAGQueryResult:
        query_text = self._coerce_query_text(*args, **kwargs)
        self._register_or_validate_runtime_contract()
        self.process_pending_index_sync(max_tasks=2)
        self.process_pending_storage_lifecycle(max_tasks=1)
        return self.query_pipeline.run(query_text, options=self._normalize_query_options(options))

    def query_public(
        self,
        *args: Any,
        options: QueryOptions | None = None,
        **kwargs: Any,
    ) -> PublicQueryResult:
        query_text = self._coerce_query_text(*args, **kwargs)
        self._register_or_validate_runtime_contract()
        self.process_pending_index_sync(max_tasks=2)
        self.process_pending_storage_lifecycle(max_tasks=1)
        return self.query_pipeline.run_public(query_text, options=self._normalize_query_options(options))

    def delete(
        self,
        *,
        doc_id: int | str | None = None,
        source_id: int | str | None = None,
        location: str | None = None,
    ) -> RuntimeDeleteResult:
        self._register_or_validate_runtime_contract()
        data_contract_service = self._build_data_contract_service()
        if data_contract_service is None:
            raise RuntimeError("delete requires the v1 data contract metadata and summary index repositories")

        documents = self._resolve_documents_for_operation(
            doc_id=doc_id,
            source_id=source_id,
            location=location,
            active_only=True,
        )
        deleted_doc_ids: list[int] = []
        deleted_source_ids: list[int] = []
        deleted_vector_count = 0
        for document in documents:
            deleted_vector_count += self._count_existing_summary_vectors(document)
            self._save_runtime_processing_state(
                doc_id=document.doc_id,
                source_id=document.source_id,
                stage="delete",
                status="deleting",
            )
            data_contract_service.deactivate_document(document.doc_id)
            cast(DataContractMetadataRepo, self.stores.metadata_repo).set_document_storage_tier(
                document.doc_id,
                storage_tier=StorageTier.COLD,
            )
            self._save_runtime_processing_state(
                doc_id=document.doc_id,
                source_id=document.source_id,
                stage="delete",
                status="deleted",
            )
            deleted_doc_ids.append(document.doc_id)
            if document.source_id not in deleted_source_ids:
                deleted_source_ids.append(document.source_id)
        return RuntimeDeleteResult(
            deleted_doc_ids=deleted_doc_ids,
            deleted_source_ids=deleted_source_ids,
            deleted_vector_count=deleted_vector_count,
        )

    def rebuild(
        self,
        *,
        doc_id: int | str | None = None,
        source_id: int | str | None = None,
        location: str | None = None,
    ) -> RuntimeRebuildResult:
        self._register_or_validate_runtime_contract()
        documents = self._resolve_documents_for_operation(
            doc_id=doc_id,
            source_id=source_id,
            location=location,
            active_only=False,
        )
        rebuilt_doc_ids: list[int] = []
        results: list[IngestPipelineResult] = []
        for document in documents:
            source = self._source_for_document(document)
            self._save_runtime_processing_state(
                doc_id=document.doc_id,
                source_id=document.source_id,
                stage="rebuild",
                status="rebuilding",
            )
            try:
                request = self._rebuild_request_from_source(source=source, document=document)
            except Exception as exc:
                message = f"No rebuildable source payload available for doc_id={document.doc_id}: {exc}"
                cast(DataContractMetadataRepo, self.stores.metadata_repo).set_document_index_state(
                    document.doc_id,
                    is_indexed=False,
                    index_ready=False,
                    last_index_error=message,
                )
                self._save_runtime_processing_state(
                    doc_id=document.doc_id,
                    source_id=document.source_id,
                    stage="rebuild",
                    status="failed",
                    error_message=message,
                )
                raise ValueError(message) from exc
            result = self.insert(request)
            rebuilt_doc_ids.append(result.doc_id)
            results.append(result)
        return RuntimeRebuildResult(rebuilt_doc_ids=rebuilt_doc_ids, results=results)

    def process_pending_index_sync(self, *, max_tasks: int = 1, lease_seconds: int = 60) -> int:
        worker = self.index_sync_worker
        if worker is None or max_tasks <= 0:
            return 0
        processed = worker.run_until_idle(max_tasks=max_tasks, lease_seconds=lease_seconds)
        return len(processed)

    def process_pending_storage_lifecycle(self, *, max_tasks: int = 1, lease_seconds: int = 60) -> int:
        worker = self.storage_lifecycle_worker
        if worker is None or max_tasks <= 0:
            return 0
        worker.service.enqueue_due_documents(limit=max_tasks)
        processed = worker.run_until_idle(max_tasks=max_tasks, lease_seconds=lease_seconds)
        return len(processed)

    def upsert_node(self, node: GraphNode, *, evidence_ids: list[str] | None = None) -> GraphNode:
        self.stores.graph_repo.save_node(node)
        if evidence_ids:
            self.stores.graph_repo.merge_node_evidence(node.node_id, evidence_ids)
        return node

    def upsert_edge(self, edge: GraphEdge, *, candidate: bool = False) -> GraphEdge:
        if candidate:
            self.stores.graph_repo.save_candidate_edge(edge)
        else:
            self.stores.graph_repo.save_edge(edge)
        return edge

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.stores.graph_repo.get_node(node_id)

    def list_nodes(self, *, node_type: str | None = None) -> list[GraphNode]:
        return self.stores.graph_repo.list_nodes(node_type=node_type)

    def delete_node(self, node_id: str) -> None:
        self.stores.graph_repo.delete_node(node_id)

    def get_edge(self, edge_id: str, *, include_candidates: bool = False) -> GraphEdge | None:
        return self.stores.graph_repo.get_edge(edge_id, include_candidates=include_candidates)

    def list_edges(self) -> list[GraphEdge]:
        return self.stores.graph_repo.list_edges()

    def delete_edge(self, edge_id: str, *, include_candidates: bool = True) -> None:
        self.stores.graph_repo.delete_edge(edge_id, include_candidates=include_candidates)

    def insert_custom_kg(
        self,
        *,
        nodes: list[GraphNode] | None = None,
        edges: list[GraphEdge] | None = None,
    ) -> dict[str, int]:
        for node in nodes or []:
            self.upsert_node(node)
        for edge in edges or []:
            self.upsert_edge(edge)
        return {
            "node_count": len(nodes or []),
            "edge_count": len(edges or []),
        }

    def _build_pipelines(self) -> None:
        from rag.utils.guard import CircuitBreaker, CircuitConfig, RateLimiter

        ocr_repo = create_default_ocr_repo()
        s3_breaker = CircuitBreaker("s3", CircuitConfig(failure_threshold=5, cooldown_seconds=30.0))
        llm_breaker = CircuitBreaker("llm", CircuitConfig(failure_threshold=3, cooldown_seconds=60.0))
        query_rate_limiter = RateLimiter()
        data_contract_service = self._build_data_contract_service()
        self.index_sync_worker = self._build_index_sync_worker(data_contract_service)
        self.storage_lifecycle_worker = self._build_storage_lifecycle_worker(data_contract_service)
        embedding_binding = (
            self.capability_bundle.embedding_bindings[0] if self.capability_bundle.embedding_bindings else None
        )
        if embedding_binding is None:
            raise RuntimeError("new ingest pipeline requires an embedding capability")
        if not _supports_summary_index_repo(self.stores.vector_repo):
            raise RuntimeError("new ingest pipeline requires a summary index repo with upsert_record support")
        dispatcher = ExtractionDispatcher(
            docling_parser=DoclingParserRepo(self.vlm_repo),
            excel_parser=ExcelParserRepo(),
            pptx_parser=PptxParserRepo(),
            image_parser=ImageParserRepo(ocr_repo),
        )
        chat_binding = self.capability_bundle.chat_bindings[0] if self.capability_bundle.chat_bindings else None

        self.ingest_pipeline = IngestPipeline(
            dispatcher=dispatcher,
            summarizer=RetrievalSummarizer(
                llm_client=_LazySummaryGeneratorAdapter(binding=chat_binding),
                token_accounting=self.token_accounting,
                config=self._summary_config(),
                gateway=self._gateway_for_chat_binding(_LazySummaryGeneratorAdapter(binding=chat_binding)),
            ),
            embedder=embedding_binding,
            metadata_repo=cast(MetadataWriterRepo, self.stores.metadata_repo),
            summary_repo=cast(PipelineSummaryIndexRepo, self.stores.vector_repo),
            object_store=self.stores.object_store,
            embedding_model_id=embedding_binding.model_name or embedding_binding.provider_name,
            embedding_space=embedding_binding.space,
            section_refiner=SectionRefiner(token_accounting=self.token_accounting),
        )
        self.retrieval_service = self._build_retrieval_service()
        answer_generation_service = AnswerGenerationService()
        self.query_pipeline = _QueryPipeline(
            retrieval=self.retrieval_service,
            context_merger=ContextEvidenceMerger(),
            grounding_service=GroundingService(
                metadata_repo=self.stores.metadata_repo,
                object_store=self.stores.object_store,
                token_accounting=self.token_accounting,
                rerank_binding=(
                    self.capability_bundle.rerank_bindings[0] if self.capability_bundle.rerank_bindings else None
                ),
                s3_circuit_breaker=s3_breaker,
            ),
            synthesis_service=SynthesisService(
                metadata_repo=self.stores.metadata_repo,
                authorization_service=AuthorizationService(resolver=self.stores.metadata_repo),
            ),
            truncator=EvidenceTruncator(token_accounting=self.token_accounting),
            prompt_builder=ContextPromptBuilder(
                answer_generation_service=answer_generation_service,
                token_accounting=self.token_accounting,
            ),
            answer_generator=AnswerGenerator(
                answer_generation_service=answer_generation_service,
                generators=_generator_bindings_from_chat_bindings(
                    self.capability_bundle.chat_bindings,
                    token_accounting=self.token_accounting,
                    model_context_tokens=self.chat_context_window_tokens,
                    stage_budgets=self.llm_stage_budgets,
                ),
            ),
            authorization_service=AuthorizationService(resolver=self.stores.metadata_repo),
            table_executor=TableExecutor(
                object_store=cast(_RangeReadableObjectStore, self.stores.object_store),
                metadata_repo=cast(_AssetMetadataRepo, self.stores.metadata_repo),
            ),
            rate_limiter=query_rate_limiter,
            llm_circuit_breaker=llm_breaker,
        )

    def _gateway_for_chat_binding(self, generator: object) -> LLMGateway:
        return LLMGateway(
            generator=generator,
            token_accounting=self.token_accounting,
            model_context_tokens=self.chat_context_window_tokens,
            stage_budgets=self.llm_stage_budgets,
        )

    def _build_data_contract_service(self) -> DataContractService | None:
        if not _supports_data_contract_metadata_repo(self.stores.metadata_repo):
            return None
        if not _supports_summary_index_repo(self.stores.vector_repo):
            return None
        embedder = self.capability_bundle.embedding_bindings[0] if self.capability_bundle.embedding_bindings else None
        return DataContractService(
            metadata_repo=cast(DataContractMetadataRepo, self.stores.metadata_repo),
            milvus_repo=cast(DataContractSummaryIndexRepo, self.stores.vector_repo),
            embedder=embedder,
        )

    @staticmethod
    def _build_index_sync_worker(data_contract_service: DataContractService | None) -> IndexSyncWorker | None:
        if data_contract_service is None or data_contract_service.index_sync_service is None:
            return None
        return IndexSyncWorker(
            index_sync_service=data_contract_service.index_sync_service,
            data_contract_service=data_contract_service,
        )

    def _build_storage_lifecycle_worker(
        self,
        data_contract_service: DataContractService | None,
    ) -> StorageLifecycleWorker | None:
        if data_contract_service is None:
            return None
        if not _supports_storage_lifecycle_repo(self.stores.metadata_repo):
            return None
        return StorageLifecycleWorker(
            service=StorageLifecycleService(
                metadata_repo=self.stores.metadata_repo,
                data_contract_service=data_contract_service,
            )
        )

    def _build_retrieval_service(self) -> RetrievalService:
        bundle = self.capability_bundle
        retrieval_factory = SearchBackedRetrievalFactory(
            metadata_repo=cast(RetrievalMetadataRepoProtocol, self.stores.metadata_repo),
            graph_repo=self.stores.graph_repo,
        )
        planning_graph = PlanningGraph(
            metadata_scope_resolver=self.stores.metadata_repo,
            use_summary_hybrid_paths=True,
        )
        reranker_service = (
            ModelBackedRerankService(
                binding=bundle.rerank_bindings[0],
            )
            if bundle.rerank_bindings
            else None
        )
        instrumented_reranker = None if reranker_service is None else _InstrumentedReranker(reranker_service)
        vec_repo: VectorSearchRepoProtocol = cast(VectorSearchRepoProtocol, self.stores.vector_repo)
        return RetrievalService(
            RetrievalServiceConfig(
                vector_retriever=cast(
                    RetrieverFn,
                    retrieval_factory.vector_retriever_from_repo(
                        vec_repo,
                        bundle.embedding_bindings,
                    ),
                ),
                local_retriever=(lambda _query, _scope, _understanding: []),
                global_retriever=(lambda _query, _scope, _understanding: []),
                section_retriever=(lambda _query, _scope, _understanding: []),
                special_retriever=cast(
                    RetrieverFn,
                    retrieval_factory.special_retriever_from_repo(
                        vec_repo,
                        bundle.embedding_bindings,
                    ),
                ),
                metadata_retriever=(lambda _query, _scope, _understanding: []),
                graph_expander=retrieval_factory.graph_expander,
                web_retriever=retrieval_factory.web_retriever,
                reranker=instrumented_reranker,
                evidence_service=EvidenceService(),
                graph_expansion_service=GraphExpansionService(),
                telemetry_service=self.telemetry_service,
                metadata_scope_resolver=self.stores.metadata_repo,
                planning_graph=planning_graph,
            )
        )

    def _build_agent_service(self, *, answer_generation_service: AnswerGenerationService) -> object:
        del answer_generation_service
        raise NotImplementedError("Legacy agent service wiring was removed")

    def _ensure_agent_service(self) -> object:
        raise NotImplementedError("Legacy agent service wiring was removed")

    @staticmethod
    def _agent_task_request_class() -> type[object]:
        raise NotImplementedError("Legacy agent task request wiring was removed")

    def _register_or_validate_runtime_contract(self) -> None:
        payload = dict(self.capability_bundle.runtime_contract_payload)
        existing = self.stores.cache_repo.get_cache_entry(_RUNTIME_CONTRACT_KEY, namespace=_RUNTIME_CONTRACT_NAMESPACE)
        stored_payload = existing.payload if existing is not None and isinstance(existing.payload, dict) else None
        governance = self.assembly_service.govern_runtime_contract(
            bundle=self.capability_bundle,
            stored_payload=stored_payload,
        )
        if governance.should_persist:
            self.stores.cache_repo.save_cache_entry(
                CacheEntry(
                    namespace=_RUNTIME_CONTRACT_NAMESPACE,
                    cache_key=_RUNTIME_CONTRACT_KEY,
                    payload=payload,
                )
            )
            return
        governance.raise_for_invalid()

    def _normalize_query_options(self, options: QueryOptions | None) -> QueryOptions:
        if options is None:
            return QueryOptions(max_context_tokens=self.token_contract.max_context_tokens)
        if (
            options.max_context_tokens == QueryOptions().max_context_tokens
            and self.token_contract.max_context_tokens != QueryOptions().max_context_tokens
        ):
            return replace(options, max_context_tokens=self.token_contract.max_context_tokens)
        return options

    def _resolve_documents_for_operation(
        self,
        *,
        doc_id: int | str | None,
        source_id: int | str | None,
        location: str | None,
        active_only: bool,
    ) -> list[Document]:
        if doc_id is None and source_id is None and (location is None or not location.strip()):
            raise TypeError("doc_id, source_id, or location is required")

        documents: list[Document] = []
        if doc_id is not None:
            document = self.stores.metadata_repo.get_document(self._coerce_int_identifier(doc_id, "doc_id"))
            if document is not None and (not active_only or document.is_active):
                documents.append(document)
        if source_id is not None:
            documents.extend(
                self.stores.metadata_repo.list_documents(
                    source_id=self._coerce_int_identifier(source_id, "source_id"),
                    active_only=active_only,
                )
            )
        if location is not None and location.strip():
            for source in self.stores.metadata_repo.list_sources(location=location):
                documents.extend(
                    self.stores.metadata_repo.list_documents(
                        source_id=source.source_id,
                        active_only=active_only,
                    )
                )

        unique: dict[int, Document] = {}
        for document in documents:
            unique[document.doc_id] = document
        return list(unique.values())

    @staticmethod
    def _coerce_int_identifier(value: int | str, name: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be an integer") from exc

    def _source_for_document(self, document: Document) -> Source:
        source = self.stores.metadata_repo.get_source(document.source_id)
        if source is None:
            raise ValueError(f"source not found for doc_id={document.doc_id}")
        return source

    def _count_existing_summary_vectors(self, document: Document) -> int:
        count = 0
        if self.stores.vector_repo.get_entry(str(document.doc_id), item_kind="doc_summary") is not None:
            count += 1
        for section in cast(DataContractMetadataRepo, self.stores.metadata_repo).list_sections(doc_id=document.doc_id):
            if self.stores.vector_repo.get_entry(str(section.section_id), item_kind="section_summary") is not None:
                count += 1
        for asset in cast(DataContractMetadataRepo, self.stores.metadata_repo).list_assets(doc_id=document.doc_id):
            if self.stores.vector_repo.get_entry(str(asset.asset_id), item_kind="asset_summary") is not None:
                count += 1
        return count

    def _rebuild_request_from_source(self, *, source: Source, document: Document) -> IngestRequest:
        object_key = source.object_key
        if object_key is None or not object_key.strip():
            raise ValueError("source.object_key is empty")
        if not self.stores.object_store.exists(object_key):
            raise ValueError(f"object_store key does not exist: {object_key}")
        raw_bytes = self.stores.object_store.read_bytes(object_key)
        metadata = {str(key): str(value) for key, value in document.metadata_json.items()}
        source_type = SourceType(source.source_type)
        kwargs: dict[str, Any] = {
            "location": source.location,
            "source_type": source_type,
            "owner": source.owner_id or "user",
            "title": document.title,
            "metadata": metadata,
        }
        if source_type in {
            SourceType.PLAIN_TEXT,
            SourceType.PASTED_TEXT,
            SourceType.BROWSER_CLIP,
            SourceType.WEB,
        }:
            kwargs["content_text"] = raw_bytes.decode("utf-8", errors="replace")
        else:
            kwargs["raw_bytes"] = raw_bytes
        return IngestRequest(**kwargs)

    def _save_runtime_processing_state(
        self,
        *,
        doc_id: int,
        source_id: int,
        stage: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        cast(DataContractMetadataRepo, self.stores.metadata_repo).save_processing_state(
            ProcessingStateRecord(
                doc_id=doc_id,
                source_id=source_id,
                stage=stage,
                status=status,
                attempts=1,
                priority="normal",
                worker_id=None,
                lease_expires_at=None,
                error_message=error_message,
                metadata_json={},
            )
        )

    @staticmethod
    def _coerce_query_text(*args: Any, **kwargs: Any) -> str:
        query_text = kwargs.pop("query", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if args:
            if len(args) != 1:
                raise TypeError("query accepts exactly one positional query string")
            if query_text is not None:
                raise TypeError("query was provided both positionally and by keyword")
            query_text = args[0]
        if not isinstance(query_text, str) or not query_text.strip():
            raise TypeError("query requires a non-empty string")
        return query_text

    @staticmethod
    def _coerce_ingest_request(request: IngestRequest | None = None, /, **kwargs: Any) -> IngestRequest:
        if request is not None:
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(f"insert request was provided both positionally and by keyword: {unexpected}")
            return request
        normalized_kwargs = {"owner": "user", **kwargs}
        if "file_path" in normalized_kwargs and normalized_kwargs["file_path"] is not None:
            normalized_kwargs["file_path"] = Path(normalized_kwargs["file_path"])
        return IngestRequest(**normalized_kwargs)

    @staticmethod
    def _coerce_direct_content_item(item: object, *, index: int) -> IngestRequest:
        if isinstance(item, IngestRequest):
            return item
        if isinstance(item, DirectContentItem):
            content = item.content
            if isinstance(content, Path):
                return IngestRequest(
                    location=item.location or str(content),
                    source_type=item.source_type,
                    owner=item.owner,
                    title=item.title,
                    file_path=content,
                    metadata=dict(item.metadata),
                )
            if isinstance(content, bytes):
                return IngestRequest(
                    location=item.location,
                    source_type=item.source_type,
                    owner=item.owner,
                    title=item.title,
                    raw_bytes=content,
                    metadata=dict(item.metadata),
                )
            return IngestRequest(
                location=item.location,
                source_type=item.source_type,
                owner=item.owner,
                title=item.title,
                content_text=str(content),
                metadata=dict(item.metadata),
            )
        if isinstance(item, Path):
            return IngestRequest(
                location=str(item),
                source_type="plain_text",
                owner="user",
                file_path=item,
            )
        if isinstance(item, str):
            return IngestRequest(
                location=f"memory://content-{index}",
                source_type="plain_text",
                owner="user",
                content_text=item,
            )
        if isinstance(item, dict):
            payload = dict(item)
            if "file_path" in payload and payload["file_path"] is not None:
                payload["file_path"] = Path(payload["file_path"])
            return IngestRequest(**payload)
        raise TypeError(f"unsupported content item type: {type(item).__name__}")


__all__ = ["RAGRuntime", "_RUNTIME_CONTRACT_KEY", "_RUNTIME_CONTRACT_NAMESPACE"]

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from agent_runtime.text import keyword_overlap, search_terms
from rag.assembly import EmbeddingCapabilityBinding
from rag.retrieval.evidence import CandidateLike, EvidenceBundle
from rag.schema.core import AssetRecord, Document, SectionRecord, Source
from rag.schema.query import GroundingTarget, RetrievalSignals
from rag.schema.runtime import (
    AccessPolicy,
    GraphRepo,
    VectorSearchResult,
)
from rag.storage.search_backends.web_search_repo import DeterministicWebSearchRepo


@dataclass(frozen=True)
class RetrievedCandidate(CandidateLike):
    evidence_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    item_id: str = field(default="", init=False)
    source_kind: str = "internal"
    source_id: str | None = None
    benchmark_doc_id: str | None = None
    section_path: tuple[str, ...] = ()
    effective_access_policy: AccessPolicy | None = None
    metadata: dict[str, str] | None = None
    file_name: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    record_type: str | None = None
    source_type: str | None = None
    retrieval_channels: list[str] | None = None
    grounding_target: GroundingTarget | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", self.evidence_id)


class VectorSearchRepoProtocol(Protocol):
    def search(
        self,
        query: Iterable[float],
        *,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
    ) -> Sequence[VectorSearchResult]: ...

    def count_vectors(
        self,
        *,
        embedding_space: str | None = None,
        item_kind: str | None = None,
        distinct_records: bool = False,
    ) -> int: ...


class HybridVectorSearchRepoProtocol(VectorSearchRepoProtocol, Protocol):
    def supports_hybrid_search(self) -> bool: ...

    async def hybrid_search_async(
        self,
        *,
        query_vector: Iterable[float],
        sparse_query: str,
        sparse_query_vector: Mapping[int, float] | None = None,
        limit: int = 10,
        doc_ids: list[str] | None = None,
        expr: str | None = None,
        embedding_space: str = "default",
        item_kind: str = "section_summary",
        fusion_strategy: str = "rrf",
        alpha: float | None = None,
    ) -> Sequence[VectorSearchResult]: ...


class RetrievalMetadataRepoProtocol(Protocol):
    def get_source(self, source_id: int) -> Source | None: ...

    def get_document(self, doc_id: int) -> Document | None: ...

    def list_documents(
        self,
        source_id: int | None = None,
        *,
        active_only: bool = False,
        version_group_id: int | None = None,
    ) -> list[Document]: ...

    def get_section(self, section_id: int) -> SectionRecord | None: ...

    def list_sections(self, doc_id: int) -> list[SectionRecord]: ...

    def get_asset(self, asset_id: int) -> AssetRecord | None: ...

    def list_assets(self, doc_id: int) -> list[AssetRecord]: ...


class EmptyGraphRetriever:
    last_provider: str | None = None
    last_attempts: list[object] = []

    def prepare_for_policy(
        self,
        *,
        access_policy: AccessPolicy,
    ) -> None:
        del access_policy

    def __call__(
        self,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
    ) -> list[RetrievedCandidate]:
        del query, source_scope, retrieval_signals
        return []


class MultiProviderBackedVectorRetriever:
    def __init__(
        self,
        *,
        factory: SearchBackedRetrievalFactory,
        vector_repo: VectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingCapabilityBinding],
        item_kind: str,
        candidate_builder: Callable[[str, list[VectorSearchResult], list[str]], list[RetrievedCandidate]],
    ) -> None:
        self._factory = factory
        self._vector_repo = vector_repo
        self._bindings = tuple(bindings)
        self._item_kind = item_kind
        self._candidate_builder = candidate_builder
        self.last_provider: str | None = None
        self.last_attempts: list[object] = []

    def prepare_for_policy(
        self,
        *,
        access_policy: AccessPolicy,
    ) -> None:
        pass

    def __call__(
        self,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
    ) -> list[RetrievedCandidate]:
        del retrieval_signals
        self.last_provider = None
        self.last_attempts = []
        ordered_bindings = self._ordered_bindings()
        for binding in ordered_bindings:
            for target_space in self._target_spaces(binding):
                candidates = self._search_binding(
                    binding,
                    query=query,
                    source_scope=source_scope,
                    target_space=target_space,
                )
                if candidates:
                    self.last_provider = binding.provider_name
                    return candidates
        return []

    def _ordered_bindings(self) -> list[EmbeddingCapabilityBinding]:
        return list(self._bindings)

    @staticmethod
    def _target_spaces(binding: EmbeddingCapabilityBinding) -> tuple[str, ...]:
        return ("default",) if binding.space == "default" else (binding.space,)

    def _search_binding(
        self,
        binding: EmbeddingCapabilityBinding,
        *,
        query: str,
        source_scope: list[str],
        target_space: str,
    ) -> list[RetrievedCandidate]:
        try:
            query_vectors = binding.embed_query([query])
        except RuntimeError:
            return []
        if not query_vectors:
            return []
        if self._vector_repo.count_vectors(embedding_space=target_space, item_kind=self._item_kind) == 0:
            return []
        results = self._vector_repo.search(
            query_vectors[0],
            limit=12,
            doc_ids=source_scope or None,
            embedding_space=target_space,
            item_kind=self._item_kind,
        )
        if not results:
            return []
        return self._candidate_builder(query, list(results), source_scope)


class MilvusSummaryHybridRetriever:
    absorbs_sparse_branch = True

    def __init__(
        self,
        *,
        factory: SearchBackedRetrievalFactory,
        vector_repo: HybridVectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingCapabilityBinding],
        item_kinds: Sequence[str],
        branch_name: str,
    ) -> None:
        self._factory = factory
        self._vector_repo = vector_repo
        self._bindings = tuple(bindings)
        self._item_kinds = tuple(item_kinds)
        self._branch_name = branch_name
        self.last_provider: str | None = None
        self.last_attempts: list[object] = []

    def prepare_for_policy(
        self,
        *,
        access_policy: AccessPolicy,
    ) -> None:
        pass

    def retrieve_with_plan(
        self,
        *,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
        plan: object,
    ) -> list[RetrievedCandidate]:
        del retrieval_signals
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aretrieve_with_plan(query=query, source_scope=source_scope, plan=plan))
        raise RuntimeError("MilvusSummaryHybridRetriever.retrieve_with_plan cannot run inside an active event loop")

    async def aretrieve_with_plan(
        self,
        *,
        query: str,
        source_scope: list[str],
        plan: object,
        retrieval_signals: RetrievalSignals | None = None,
    ) -> list[RetrievedCandidate]:
        del retrieval_signals
        self.last_provider = None
        self.last_attempts = []
        ordered_bindings = self._ordered_bindings()
        for binding in ordered_bindings:
            for target_space in self._target_spaces(binding):
                candidates = await self._search_binding(
                    binding,
                    query=query,
                    source_scope=source_scope,
                    target_space=target_space,
                    plan=plan,
                )
                if candidates:
                    self.last_provider = binding.provider_name
                    return candidates
        return []

    def _ordered_bindings(self) -> list[EmbeddingCapabilityBinding]:
        return list(self._bindings)

    @staticmethod
    def _target_spaces(binding: EmbeddingCapabilityBinding) -> tuple[str, ...]:
        return ("default",) if binding.space == "default" else (binding.space,)

    async def _search_binding(
        self,
        binding: EmbeddingCapabilityBinding,
        *,
        query: str,
        source_scope: list[str],
        target_space: str,
        plan: object,
    ) -> list[RetrievedCandidate]:
        branch_limit = self._branch_limit(plan)
        stage_plan = self._stage_plan(plan, branch_limit=branch_limit)
        if not stage_plan:
            return []

        results: list[VectorSearchResult] = []
        seen_keys: set[tuple[str, str]] = set()
        fusion_strategy = str(getattr(plan, "fusion_strategy", "weighted_rrf") or "weighted_rrf")
        fusion_alpha = getattr(plan, "fusion_alpha", None)
        query_tasks = self._query_tasks(plan, query)
        for dense_query, sparse_query in query_tasks:
            try:
                query_vectors = binding.embed_query([dense_query])
            except RuntimeError:
                continue
            if not query_vectors:
                continue
            sparse_query_vector = self._sparse_query_vector(binding, sparse_query)
            for stage in stage_plan:
                if getattr(stage, "trigger", "always") != "always" and len(results) >= int(
                    getattr(stage, "min_hits", 0) or 0
                ):
                    continue
                item_kind = str(getattr(stage, "collection", "") or "")
                if self._vector_repo.count_vectors(embedding_space=target_space, item_kind=item_kind) <= 0:
                    continue
                remaining = max(
                    min(int(getattr(stage, "limit", branch_limit) or branch_limit), branch_limit) - len(results), 1
                )
                hits = await self._hybrid_search(
                    query_vector=query_vectors[0],
                    sparse_query=sparse_query,
                    sparse_query_vector=sparse_query_vector,
                    limit=remaining,
                    doc_ids=source_scope or None,
                    expr=self._stage_expr(plan, item_kind=item_kind),
                    embedding_space=target_space,
                    item_kind=item_kind,
                    fusion_strategy=fusion_strategy,
                    alpha=fusion_alpha,
                )
                for hit in hits:
                    key = (hit.item_kind, hit.item_id)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    results.append(hit)
                if len(results) >= branch_limit:
                    break
            if len(results) >= branch_limit:
                break
        if not results:
            return []
        sparse_used = bool(sparse_query.strip()) if sparse_query else False
        channels = ["vector", "sparse"] if sparse_used else ["vector"]
        return self._factory.build_summary_candidates_from_vector_results(
            query,
            results[:branch_limit],
            source_scope,
            retrieval_channels=channels,
        )

    def _branch_limit(self, plan: object) -> int:
        retrieval_paths = tuple(getattr(plan, "retrieval_paths", ()) or ())
        for path in retrieval_paths:
            if getattr(path, "branch", None) == self._branch_name:
                return max(int(getattr(path, "limit", 0) or 0), 1)
        return 12

    def _stage_plan(self, plan: object, *, branch_limit: int) -> list[object]:
        branch_stage_plans = tuple(getattr(plan, "branch_stage_plans", ()) or ())
        for branch_stage in branch_stage_plans:
            if getattr(branch_stage, "branch", None) != self._branch_name:
                continue
            return [
                stage
                for stage in tuple(getattr(branch_stage, "stages", ()) or ())
                if getattr(stage, "collection", None) in self._item_kinds
            ]
        target_collections = [
            collection
            for collection in tuple(getattr(plan, "target_collections", ()) or ())
            if collection in self._item_kinds
        ] or list(self._item_kinds)
        return [
            type(
                "_FallbackStage",
                (),
                {"collection": collection, "limit": branch_limit, "min_hits": branch_limit, "trigger": "always"},
            )()
            for collection in target_collections
        ]

    @staticmethod
    def _query_tasks(plan: object, query: str) -> list[tuple[str, str]]:
        rewritten_query = str(getattr(plan, "rewritten_query", query) or query).strip()
        sparse_query = str(getattr(plan, "sparse_query", rewritten_query) or rewritten_query).strip()
        tasks = [(rewritten_query, sparse_query)]
        for subtask in tuple(getattr(plan, "query_subtasks", ()) or ()):
            prompt = str(getattr(subtask, "prompt", "") or "").strip()
            if prompt and prompt != rewritten_query:
                tasks.append((prompt, prompt))
        ordered: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in tasks:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    @staticmethod
    def _stage_expr(plan: object, *, item_kind: str) -> str | None:
        predicate_plan = getattr(plan, "predicate_plan", None)
        collection_expressions = getattr(predicate_plan, "collection_expressions", None)
        if isinstance(collection_expressions, dict) and item_kind in collection_expressions:
            return str(collection_expressions[item_kind])
        expression = getattr(predicate_plan, "expression", None)
        return None if expression is None else str(expression)

    @staticmethod
    def _sparse_query_vector(
        binding: EmbeddingCapabilityBinding,
        sparse_query: str,
    ) -> dict[int, float] | None:
        if not sparse_query.strip():
            return None
        supports_sparse_embedding = getattr(binding, "supports_sparse_embedding", None)
        if callable(supports_sparse_embedding):
            if not bool(supports_sparse_embedding()):
                return None
        elif not callable(getattr(binding, "embed_query_sparse", None)):
            return None
        try:
            sparse_vectors = binding.embed_query_sparse([sparse_query])
        except RuntimeError:
            return None
        return sparse_vectors[0] if sparse_vectors else None

    async def _hybrid_search(
        self,
        *,
        query_vector: Iterable[float],
        sparse_query: str,
        sparse_query_vector: Mapping[int, float] | None,
        limit: int,
        doc_ids: list[str] | None,
        expr: str | None,
        embedding_space: str,
        item_kind: str,
        fusion_strategy: str,
        alpha: float | None,
    ) -> Sequence[VectorSearchResult]:
        hybrid_search_async = self._vector_repo.hybrid_search_async
        parameters = inspect.signature(hybrid_search_async).parameters
        kwargs: dict[str, Any] = {
            "query_vector": query_vector,
            "sparse_query": sparse_query,
            "limit": limit,
            "doc_ids": doc_ids,
            "expr": expr,
            "embedding_space": embedding_space,
            "item_kind": item_kind,
            "fusion_strategy": fusion_strategy,
            "alpha": alpha,
        }
        if "sparse_query_vector" in parameters:
            kwargs["sparse_query_vector"] = sparse_query_vector
        return await hybrid_search_async(**kwargs)


class SearchBackedRetrievalFactory:
    def __init__(
        self,
        *,
        metadata_repo: RetrievalMetadataRepoProtocol,
        graph_repo: GraphRepo,
    ) -> None:
        self._metadata_repo = metadata_repo
        self._graph_repo = graph_repo

    def section_retriever(
        self,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
    ) -> list[RetrievedCandidate]:
        query_terms = search_terms(query)
        constraints = retrieval_signals.structure_constraints
        focus_terms = {term for term in constraints.focus_terms if term}
        if not focus_terms:
            return []
        candidates: list[RetrievedCandidate] = []
        for document in self._iter_documents(source_scope):
            source = self._metadata_repo.get_source(document.source_id)
            for section in self._metadata_repo.list_sections(document.doc_id):
                section_text = " ".join(section.toc_path)
                heading_score = sum(1 for term in focus_terms if term in section_text)
                lexical_score = keyword_overlap(query_terms, section_text)
                score = heading_score + lexical_score
                if constraints.prefer_heading_match and heading_score <= 0:
                    continue
                if score <= 0.0:
                    continue
                candidates.append(
                    self._candidate_from_section(
                        section,
                        document=document,
                        source=source,
                        score=round(float(score), 6),
                        rank=1,
                        retrieval_channels=["section"],
                    )
                )
        candidates.sort(key=lambda item: (-item.score, item.evidence_id))
        return self._rerank_candidates(candidates[:12])

    def special_retriever(
        self,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
    ) -> list[RetrievedCandidate]:
        query_terms = search_terms(query)
        lowered = query.lower()
        target_aliases = {target: set(special_target_aliases(target)) for target in retrieval_signals.special_targets}
        if not target_aliases:
            return []
        candidates: list[RetrievedCandidate] = []
        for document in self._iter_documents(source_scope):
            source = self._metadata_repo.get_source(document.source_id)
            for asset in self._metadata_repo.list_assets(document.doc_id):
                score = float(keyword_overlap(query_terms, asset.caption or asset.asset_type))
                aliases = target_aliases.get(asset.asset_type, set())
                score += float(int(asset.asset_type in target_aliases))
                score += float(sum(1 for alias in aliases if alias and alias in lowered))
                if score <= 0.0:
                    continue
                candidates.append(
                    self._candidate_from_asset(
                        asset,
                        document=document,
                        source=source,
                        score=round(score, 6),
                        rank=1,
                        retrieval_channels=["special"],
                    )
                )
        candidates.sort(key=lambda item: (-item.score, item.evidence_id))
        return self._rerank_candidates(candidates[:12])

    def metadata_retriever(
        self,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
    ) -> list[RetrievedCandidate]:
        del query
        page_numbers = set(retrieval_signals.metadata_filters.page_numbers)
        page_ranges = list(retrieval_signals.metadata_filters.page_ranges)
        source_types = set(retrieval_signals.metadata_filters.source_types)
        focus_terms = set(retrieval_signals.structure_constraints.focus_terms)
        document_titles = set(retrieval_signals.metadata_filters.document_titles)
        file_names = set(retrieval_signals.metadata_filters.file_names)
        special_targets = set(retrieval_signals.special_targets)
        if not (
            page_numbers
            or page_ranges
            or source_types
            or focus_terms
            or document_titles
            or file_names
            or special_targets
        ):
            return []

        candidates: list[RetrievedCandidate] = []
        for document in self._iter_documents(source_scope):
            source = self._metadata_repo.get_source(document.source_id)
            source_type = "" if source is None else source.source_type.value
            file_name = self._resolve_file_name(document.title, None if source is None else source.location)
            for section in self._metadata_repo.list_sections(document.doc_id):
                section_text = " ".join(section.toc_path)
                score = 0.0
                if source_types and source_type in source_types:
                    score += 1.0
                if document_titles and (document.title or "") in document_titles:
                    score += 1.0
                if file_names and file_name in file_names:
                    score += 1.0
                if focus_terms and any(term in section_text for term in focus_terms):
                    score += 1.0
                if self._section_page_matches(section, page_numbers=page_numbers, page_ranges=page_ranges):
                    score += 1.0
                if score > 0.0:
                    candidates.append(
                        self._candidate_from_section(
                            section,
                            document=document,
                            source=source,
                            score=round(score, 6),
                            rank=1,
                            retrieval_channels=["metadata"],
                        )
                    )
            if special_targets:
                for asset in self._metadata_repo.list_assets(document.doc_id):
                    if asset.asset_type not in special_targets:
                        continue
                    candidates.append(
                        self._candidate_from_asset(
                            asset,
                            document=document,
                            source=source,
                            score=1.0,
                            rank=1,
                            retrieval_channels=["metadata", "special"],
                        )
                    )
        candidates.sort(key=lambda item: (-item.score, item.evidence_id))
        return self._rerank_candidates(candidates[:12])

    def graph_expander(
        self,
        query: str,
        source_scope: list[str],
        evidence: list[CandidateLike],
    ) -> list[RetrievedCandidate]:
        del query, source_scope, evidence
        return []

    def web_retriever(
        self,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
    ) -> list[RetrievedCandidate]:
        del source_scope, retrieval_signals
        return [
            RetrievedCandidate(
                evidence_id=f"web-{index}",
                doc_id=f"web-doc-{index}",
                benchmark_doc_id=None,
                source_id=result.url,
                text=result.snippet,
                citation_anchor=result.title,
                score=float(result.score or 0.0),
                rank=index,
                source_kind="external",
                file_name=result.title,
                record_type="external",
                source_type="web",
                retrieval_channels=["web"],
            )
            for index, result in enumerate(DeterministicWebSearchRepo().search(query), start=1)
        ]

    def vector_retriever_from_repo(
        self,
        vector_repo: VectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingCapabilityBinding],
    ) -> MultiProviderBackedVectorRetriever | MilvusSummaryHybridRetriever:
        if self._supports_hybrid_vector_repo(vector_repo) and bindings:
            return MilvusSummaryHybridRetriever(
                factory=self,
                vector_repo=cast(HybridVectorSearchRepoProtocol, vector_repo),
                bindings=bindings,
                item_kinds=("section_summary", "doc_summary"),
                branch_name="vector",
            )
        return MultiProviderBackedVectorRetriever(
            factory=self,
            vector_repo=vector_repo,
            bindings=bindings,
            item_kind="section_summary",
            candidate_builder=self.build_summary_candidates_from_vector_results,
        )

    def local_retriever_from_repo(
        self,
        vector_repo: VectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingCapabilityBinding],
    ) -> EmptyGraphRetriever:
        del vector_repo, bindings
        return EmptyGraphRetriever()

    def global_retriever_from_repo(
        self,
        vector_repo: VectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingCapabilityBinding],
    ) -> EmptyGraphRetriever:
        del vector_repo, bindings
        return EmptyGraphRetriever()

    def special_retriever_from_repo(
        self,
        vector_repo: VectorSearchRepoProtocol,
        bindings: Sequence[EmbeddingCapabilityBinding],
    ) -> MultiProviderBackedVectorRetriever | MilvusSummaryHybridRetriever:
        if self._supports_hybrid_vector_repo(vector_repo) and bindings:
            return MilvusSummaryHybridRetriever(
                factory=self,
                vector_repo=cast(HybridVectorSearchRepoProtocol, vector_repo),
                bindings=bindings,
                item_kinds=("asset_summary",),
                branch_name="special",
            )
        return MultiProviderBackedVectorRetriever(
            factory=self,
            vector_repo=vector_repo,
            bindings=bindings,
            item_kind="asset_summary",
            candidate_builder=self.build_summary_candidates_from_vector_results,
        )

    @staticmethod
    def _supports_hybrid_vector_repo(vector_repo: VectorSearchRepoProtocol) -> bool:
        supports_hybrid_search = getattr(vector_repo, "supports_hybrid_search", None)
        hybrid_search_async = getattr(vector_repo, "hybrid_search_async", None)
        return callable(supports_hybrid_search) and bool(supports_hybrid_search()) and callable(hybrid_search_async)

    def build_summary_candidates_from_vector_results(
        self,
        query: str,
        results: list[VectorSearchResult],
        source_scope: list[str],
        retrieval_channels: list[str] | None = None,
    ) -> list[RetrievedCandidate]:
        del query
        channels = retrieval_channels or ["vector"]
        merged: dict[str, RetrievedCandidate] = {}
        allowed_scope = {str(item) for item in source_scope}
        for result in results:
            base_score = max(float(result.score), 0.0)
            if base_score <= 0.0:
                continue
            if not self._document_is_query_visible(str(result.doc_id)):
                continue
            candidate = self._summary_candidate_from_result(
                result=result,
                score=base_score * self._summary_result_weight(result.item_kind),
                retrieval_channels=list(channels),
            )
            if allowed_scope and not self._candidate_scope(candidate) & allowed_scope:
                continue
            existing = merged.get(candidate.evidence_id)
            if existing is None or candidate.score > existing.score:
                merged[candidate.evidence_id] = candidate
        ordered = sorted(merged.values(), key=lambda item: (-item.score, item.evidence_id))
        return self._rerank_candidates(ordered[:12])

    @staticmethod
    def _summary_result_weight(item_kind: str) -> float:
        if item_kind == "doc_summary":
            return 0.94
        if item_kind == "asset_summary":
            return 0.97
        return 1.0

    def _summary_candidate_from_result(
        self,
        *,
        result: VectorSearchResult,
        score: float,
        retrieval_channels: list[str] | None = None,
    ) -> RetrievedCandidate:
        metadata = dict(result.metadata)
        section_path = self._toc_path_tuple(metadata.get("toc_path_text"))
        page_start = self._int_metadata(metadata, "page_start")
        page_end = self._int_metadata(metadata, "page_end")
        page_no = self._int_metadata(metadata, "page_no")
        if page_start is None:
            page_start = page_no
        if page_end is None:
            page_end = page_start
        evidence_id = f"summary:{result.item_kind}:{result.item_id}"
        record_type = metadata.get("asset_type") or metadata.get("section_kind") or result.item_kind
        document = self._get_document(result.doc_id)
        get_source = getattr(self._metadata_repo, "get_source", None)
        source = None if document is None or not callable(get_source) else get_source(document.source_id)
        document_title = None if document is None else getattr(document, "title", None)
        source_location = None if source is None else getattr(source, "location", None)
        return RetrievedCandidate(
            evidence_id=evidence_id,
            doc_id=str(result.doc_id),
            text=result.text,
            citation_anchor=self._summary_citation_anchor(result, section_path=section_path),
            score=round(score, 6),
            rank=1,
            source_kind="internal",
            source_id=str(result.source_id or metadata.get("source_id") or ""),
            benchmark_doc_id=self._benchmark_doc_id(metadata=metadata, document=document),
            section_path=section_path,
            metadata=metadata,
            file_name=self._resolve_file_name(document_title, source_location),
            page_start=page_start,
            page_end=page_end,
            record_type=str(record_type) if record_type is not None else result.item_kind,
            source_type=metadata.get("source_type"),
            retrieval_channels=retrieval_channels or ["vector"],
            grounding_target=self._grounding_target_from_result(
                result,
                section_path=section_path,
                page_start=page_start,
                page_end=page_end,
            ),
        )

    def _candidate_from_section(
        self,
        section: SectionRecord,
        *,
        document: Document,
        source: Source | None,
        score: float,
        rank: int,
        retrieval_channels: list[str],
    ) -> RetrievedCandidate:
        section_path = tuple(section.toc_path)
        metadata = {key: str(value) for key, value in section.metadata_json.items()}
        return RetrievedCandidate(
            evidence_id=f"section:{section.section_id}",
            doc_id=str(section.doc_id),
            source_id=str(section.source_id),
            benchmark_doc_id=self._benchmark_doc_id(metadata=document.metadata_json, document=document),
            text=" / ".join(section_path) or section.section_kind,
            citation_anchor=" / ".join(section_path) or f"section:{section.section_id}",
            score=score,
            rank=rank,
            section_path=section_path,
            effective_access_policy=document.effective_access_policy,
            metadata=metadata,
            file_name=self._resolve_file_name(document.title, None if source is None else source.location),
            page_start=section.page_start,
            page_end=section.page_end,
            record_type=section.section_kind,
            source_type=None if source is None else source.source_type.value,
            retrieval_channels=retrieval_channels,
            grounding_target=GroundingTarget(
                kind="section",
                doc_id=section.doc_id,
                source_id=section.source_id,
                section_id=section.section_id,
                page_start=section.page_start,
                page_end=section.page_end,
                section_path=list(section_path),
                raw_locator={"section_id": str(section.section_id)},
            ),
        )

    def _candidate_from_asset(
        self,
        asset: AssetRecord,
        *,
        document: Document,
        source: Source | None,
        score: float,
        rank: int,
        retrieval_channels: list[str],
    ) -> RetrievedCandidate:
        metadata = {key: str(value) for key, value in asset.metadata_json.items()}
        return RetrievedCandidate(
            evidence_id=f"asset:{asset.asset_id}",
            doc_id=str(asset.doc_id),
            source_id=str(asset.source_id),
            benchmark_doc_id=self._benchmark_doc_id(metadata=document.metadata_json, document=document),
            text=asset.caption or asset.asset_type,
            citation_anchor=f"{asset.asset_type}@p{asset.page_no}",
            score=score,
            rank=rank,
            section_path=(),
            effective_access_policy=document.effective_access_policy,
            metadata=metadata,
            file_name=self._resolve_file_name(document.title, None if source is None else source.location),
            page_start=asset.page_no,
            page_end=asset.page_no,
            record_type=f"asset:{asset.asset_type}",
            source_type=None if source is None else source.source_type.value,
            retrieval_channels=retrieval_channels,
            grounding_target=GroundingTarget(
                kind="asset",
                doc_id=asset.doc_id,
                source_id=asset.source_id,
                section_id=asset.section_id,
                asset_id=asset.asset_id,
                page_start=asset.page_no,
                page_end=asset.page_no,
                raw_locator={"asset_id": str(asset.asset_id)},
            ),
        )

    def _grounding_target_from_result(
        self,
        result: VectorSearchResult,
        *,
        section_path: tuple[str, ...],
        page_start: int | None,
        page_end: int | None,
    ) -> GroundingTarget:
        metadata = result.metadata
        target_kind = "document"
        section_id: str | None = None
        asset_id: str | None = None
        if result.item_kind == "section_summary":
            target_kind = "section"
            section_id = metadata.get("section_id") or result.item_id
        elif result.item_kind == "asset_summary":
            target_kind = "asset"
            section_id = metadata.get("section_id")
            asset_id = metadata.get("asset_id") or result.item_id
        return GroundingTarget(
            kind=target_kind,
            doc_id=int(str(result.doc_id)),
            source_id=None
            if (result.source_id or metadata.get("source_id")) in {None, ""}
            else int(str(result.source_id or metadata.get("source_id"))),
            section_id=None if section_id in {None, ""} else int(str(section_id)),
            asset_id=None if asset_id in {None, ""} else int(str(asset_id)),
            page_start=page_start,
            page_end=page_end,
            section_path=list(section_path),
            raw_locator={
                "summary_item_id": result.item_id,
                "summary_item_kind": result.item_kind,
            },
        )

    @staticmethod
    def _summary_citation_anchor(
        result: VectorSearchResult,
        *,
        section_path: tuple[str, ...],
    ) -> str:
        if result.item_kind == "asset_summary":
            asset_type = result.metadata.get("asset_type") or "asset"
            page_no = result.metadata.get("page_no")
            return f"{asset_type}@p{page_no}" if page_no else str(asset_type)
        if section_path:
            return " / ".join(section_path)
        title = result.metadata.get("title")
        if title:
            return str(title)
        return f"{result.item_kind}:{result.item_id}"

    @staticmethod
    def _toc_path_tuple(raw_path: str | None) -> tuple[str, ...]:
        if raw_path is None:
            return ()
        return tuple(part.strip() for part in raw_path.split("/") if part.strip())

    @staticmethod
    def _int_metadata(metadata: Mapping[str, object], key: str) -> int | None:
        value = metadata.get(key)
        if value is None:
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _section_page_matches(
        section: SectionRecord,
        *,
        page_numbers: set[int],
        page_ranges: Sequence[object],
    ) -> bool:
        if not page_numbers and not page_ranges:
            return False
        pages: set[int] = set()
        if section.page_start is not None and section.page_end is not None:
            pages.update(range(section.page_start, section.page_end + 1))
        elif section.page_start is not None:
            pages.add(section.page_start)
        elif section.page_end is not None:
            pages.add(section.page_end)
        if not pages:
            return False
        if page_numbers and pages & page_numbers:
            return True
        return any(
            any(getattr(page_range, "start", 0) <= page <= getattr(page_range, "end", 0) for page in pages)
            for page_range in page_ranges
        )

    def _iter_documents(self, source_scope: list[str]) -> list[Document]:
        documents = self._metadata_repo.list_documents(active_only=True)
        if not source_scope:
            return documents
        allowed = {str(item) for item in source_scope}
        return [document for document in documents if {str(document.doc_id), str(document.source_id)} & allowed]

    def _get_document(self, doc_id: int | str) -> Document | None:
        get_document = getattr(self._metadata_repo, "get_document", None)
        if not callable(get_document):
            return None
        try:
            return cast(Document | None, get_document(int(str(doc_id))))
        except (TypeError, ValueError):
            return None

    def _document_is_query_visible(self, doc_id: str) -> bool:
        get_document = getattr(self._metadata_repo, "get_document", None)
        if not callable(get_document):
            return True
        document = self._get_document(doc_id)
        if document is None:
            return False
        return self._document_record_is_query_visible(document)

    @staticmethod
    def _document_record_is_query_visible(document: object) -> bool:
        if getattr(document, "is_active", True) is False:
            return False
        if getattr(document, "index_ready", True) is False:
            return False
        status = str(getattr(document, "doc_status", "") or "").strip().lower()
        return status not in {"retired", "expired", "deleted", "inactive"}

    @staticmethod
    def _candidate_scope(candidate: RetrievedCandidate) -> set[str]:
        scope = {str(candidate.doc_id)}
        if candidate.source_id:
            scope.add(str(candidate.source_id))
        return scope

    @staticmethod
    def _benchmark_doc_id(
        *,
        metadata: Mapping[str, object] | None = None,
        document: Document | None = None,
    ) -> str | None:
        if metadata is not None:
            value = metadata.get("benchmark_doc_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
            benchmark_flag = metadata.get("benchmark")
            if bool(benchmark_flag) and document is not None:
                return str(document.doc_id)
        if document is not None:
            metadata_json = getattr(document, "metadata_json", {})
            if not isinstance(metadata_json, Mapping):
                metadata_json = {}
            value = metadata_json.get("benchmark_doc_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
            if metadata_json.get("benchmark"):
                return str(document.doc_id)
        return None

    @staticmethod
    def _resolve_file_name(title: str | None, location: str | None) -> str | None:
        if title and title.strip():
            return title.strip()
        if not location:
            return None
        if "://" in location:
            return location
        return Path(location).name or location

    @staticmethod
    def _rerank_candidates(candidates: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
        result: list[RetrievedCandidate] = []
        for index, candidate in enumerate(candidates, start=1):
            kwargs = {k: v for k, v in candidate.__dict__.items() if k != "item_id"}
            kwargs["rank"] = index
            result.append(candidate.__class__(**kwargs))
        return result


class GraphExpansionService:
    @staticmethod
    def _candidate_allowed(candidate: CandidateLike, source_scope: Sequence[str]) -> bool:
        if not source_scope:
            return True
        scope = {str(candidate.doc_id)}
        if candidate.source_id:
            scope.add(str(candidate.source_id))
        return bool(scope & {str(item) for item in source_scope})

    def expand(
        self,
        *,
        query: str,
        source_scope: Sequence[str],
        evidence: EvidenceBundle,
        graph_candidates: Sequence[CandidateLike],
        access_policy: AccessPolicy,
    ) -> list[CandidateLike]:
        del query
        seen = {item.evidence_id for item in evidence.all}
        expanded: list[CandidateLike] = []
        for candidate in graph_candidates:
            if candidate.item_id in seen:
                continue
            if not self._candidate_allowed(candidate, source_scope):
                continue
            expanded.append(candidate)
            seen.add(candidate.item_id)
        return expanded


def special_target_aliases(target: str) -> tuple[str, ...]:
    normalized = target.strip().lower()
    return (normalized,) if normalized else ()


__all__ = [
    "EmptyGraphRetriever",
    "GraphExpansionService",
    "HybridVectorSearchRepoProtocol",
    "MilvusSummaryHybridRetriever",
    "MultiProviderBackedVectorRetriever",
    "RetrievalMetadataRepoProtocol",
    "RetrievedCandidate",
    "SearchBackedRetrievalFactory",
    "VectorSearchRepoProtocol",
]

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from typing_extensions import TypedDict

from rag.retrieval.models import QueryOptions, RetrievalProfile, normalize_retrieval_profile
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy


class ComplexityGate(StrEnum):
    FAST_TRACK = "fast_track"
    STANDARD = "standard"
    COMPLEX = "complex"


class QueryVariant(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"


@dataclass(frozen=True, slots=True)
class PredicatePlan:
    strategy: str = "none"
    doc_ids: tuple[str, ...] = ()
    attribute_filters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    expression: str | None = None
    collection_expressions: dict[str, str] = field(default_factory=dict)
    overflowed: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalPath:
    branch: str
    limit: int
    query_variant: QueryVariant = QueryVariant.DENSE


@dataclass(frozen=True, slots=True)
class OperatorStep:
    name: str
    branch: str | None = None
    collection: str | None = None
    required: bool = True


@dataclass(frozen=True, slots=True)
class CollectionStage:
    collection: str
    limit: int
    min_hits: int = 0
    trigger: str = "always"


@dataclass(frozen=True, slots=True)
class BranchStagePlan:
    branch: str
    stages: tuple[CollectionStage, ...]


@dataclass(frozen=True, slots=True)
class FallbackStep:
    trigger: str
    branch: str
    limit: int
    collection: str | None = None


@dataclass(frozen=True, slots=True)
class QuerySubTask:
    prompt: str
    purpose: str


class _PlannerState(TypedDict, total=False):
    query: str
    source_scope: tuple[str, ...]
    access_policy: AccessPolicy
    retrieval_signals: RetrievalSignals
    retrieval_profile: RetrievalProfile
    retrieval_limit: int
    final_limit: int
    complexity_gate: ComplexityGate
    rewritten_query: str
    sparse_query: str
    semantic_route: str
    target_collections: tuple[str, ...]
    predicate_plan: PredicatePlan
    retrieval_paths: tuple[RetrievalPath, ...]
    allow_web: bool
    allow_graph_expansion: bool
    web_limit: int
    graph_limit: int
    version_gate_enabled: bool
    version_gate_doc_ids: tuple[str, ...]
    version_gate_expression: str | None
    operator_plan: tuple[OperatorStep, ...]
    branch_stage_plans: tuple[BranchStagePlan, ...]
    fallback_plan: tuple[FallbackStep, ...]
    query_subtasks: tuple[QuerySubTask, ...]
    fusion_strategy: str
    fusion_alpha: float


@dataclass(frozen=True, slots=True)
class PlanningState:
    original_query: str
    rewritten_query: str
    sparse_query: str
    retrieval_profile: RetrievalProfile
    complexity_gate: ComplexityGate
    semantic_route: str
    target_collections: tuple[str, ...]
    predicate_plan: PredicatePlan
    retrieval_paths: tuple[RetrievalPath, ...]
    allow_web: bool
    allow_graph_expansion: bool
    web_limit: int
    graph_limit: int
    notes: tuple[str, ...] = ()
    version_gate_enabled: bool = True
    operator_plan: tuple[OperatorStep, ...] = ()
    branch_stage_plans: tuple[BranchStagePlan, ...] = ()
    fallback_plan: tuple[FallbackStep, ...] = ()
    query_subtasks: tuple[QuerySubTask, ...] = ()
    fusion_strategy: str = "weighted_rrf"
    fusion_alpha: float = 0.65


class PlanningGraph:
    def __init__(
        self,
        *,
        metadata_scope_resolver: object | None = None,
        whitelist_threshold: int = 1000,
        attribute_value_cap: int = 16,
        use_summary_hybrid_paths: bool = False,
    ) -> None:
        self._metadata_scope_resolver = metadata_scope_resolver
        self._whitelist_threshold = whitelist_threshold
        self._attribute_value_cap = attribute_value_cap
        self._use_summary_hybrid_paths = use_summary_hybrid_paths

    def plan(
        self,
        query: str,
        *,
        source_scope: Sequence[str],
        access_policy: AccessPolicy,
        retrieval_signals: RetrievalSignals,
        resolved_retrieval_profile: RetrievalProfile | str | None,
        query_options: QueryOptions | None,
    ) -> PlanningState:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aplan(
                    query,
                    source_scope=source_scope,
                    access_policy=access_policy,
                    retrieval_signals=retrieval_signals,
                    resolved_retrieval_profile=resolved_retrieval_profile,
                    query_options=query_options,
                )
            )
        raise RuntimeError("PlanningGraph.plan cannot run inside an active event loop; call aplan instead")

    async def aplan(
        self,
        query: str,
        *,
        source_scope: Sequence[str],
        access_policy: AccessPolicy,
        retrieval_signals: RetrievalSignals,
        resolved_retrieval_profile: RetrievalProfile | str | None,
        query_options: QueryOptions | None,
    ) -> PlanningState:
        state: dict[str, Any] = dict(
            self._initial_state(
                query=query,
                source_scope=source_scope,
                access_policy=access_policy,
                retrieval_signals=retrieval_signals,
                resolved_retrieval_profile=resolved_retrieval_profile,
                query_options=query_options,
            )
        )
        state.update(await self._node_complexity_gate(state))
        if state.get("complexity_gate") is ComplexityGate.COMPLEX:
            state.update(await self._node_query_decompose(state))
        state.update(await self._node_query_variants(state))
        state.update(await self._node_semantic_route(state))
        state.update(await self._node_version_gate(state))
        state.update(await self._node_predicate_plan(state))
        state.update(await self._node_execution_plan(state))
        retrieval_profile = state["retrieval_profile"]
        predicate_plan = state["predicate_plan"]
        complexity_gate = state["complexity_gate"]
        semantic_route = state["semantic_route"]
        return PlanningState(
            original_query=query,
            rewritten_query=state["rewritten_query"],
            sparse_query=state["sparse_query"],
            retrieval_profile=retrieval_profile,
            complexity_gate=complexity_gate,
            semantic_route=semantic_route,
            target_collections=state["target_collections"],
            predicate_plan=predicate_plan,
            retrieval_paths=state["retrieval_paths"],
            allow_web=bool(state["allow_web"]),
            allow_graph_expansion=bool(state["allow_graph_expansion"]),
            web_limit=int(state["web_limit"]),
            graph_limit=int(state["graph_limit"]),
            notes=self._notes(predicate_plan, complexity_gate, semantic_route),
            version_gate_enabled=bool(state.get("version_gate_enabled", True)),
            operator_plan=tuple(state.get("operator_plan", ()) or ()),
            branch_stage_plans=tuple(state.get("branch_stage_plans", ()) or ()),
            fallback_plan=tuple(state.get("fallback_plan", ()) or ()),
            query_subtasks=tuple(state.get("query_subtasks", ()) or ()),
            fusion_strategy=str(state["fusion_strategy"]),
            fusion_alpha=float(state["fusion_alpha"]),
        )

    def _initial_state(
        self,
        *,
        query: str,
        source_scope: Sequence[str],
        access_policy: AccessPolicy,
        retrieval_signals: RetrievalSignals,
        resolved_retrieval_profile: RetrievalProfile | str | None,
        query_options: QueryOptions | None,
    ) -> _PlannerState:
        retrieval_profile = (
            query_options.resolved_retrieval_profile
            if query_options is not None
            else normalize_retrieval_profile(resolved_retrieval_profile)
        )
        candidate_top_k = query_options.resolved_candidate_top_k if query_options is not None else 8
        retrieval_limit = max(query_options.retrieval_pool_k or candidate_top_k, 1) if query_options is not None else 8
        final_limit = max(candidate_top_k, 1)
        return {
            "query": query,
            "source_scope": tuple(source_scope),
            "access_policy": access_policy,
            "retrieval_signals": retrieval_signals,
            "retrieval_profile": retrieval_profile,
            "retrieval_limit": retrieval_limit,
            "final_limit": final_limit,
        }

    async def _node_complexity_gate(self, state: dict[str, Any]) -> dict[str, Any]:
        query = str(state["query"])
        retrieval_signals = state["retrieval_signals"]
        complexity_gate = self._complexity_gate(query, retrieval_signals)
        return {"complexity_gate": complexity_gate}

    async def _node_query_decompose(self, state: dict[str, Any]) -> dict[str, Any]:
        query = str(state["query"])
        retrieval_signals = state["retrieval_signals"]
        complexity_gate = state["complexity_gate"]
        return {
            "query_subtasks": self._decompose_query(query, retrieval_signals, complexity_gate),
        }

    async def _node_query_variants(self, state: dict[str, Any]) -> dict[str, Any]:
        query = str(state["query"])
        retrieval_signals = state["retrieval_signals"]
        complexity_gate = state["complexity_gate"]
        query_subtasks = tuple(state.get("query_subtasks", ()) or ())
        rewritten_query = self._rewrite_query(query, retrieval_signals, complexity_gate)
        if query_subtasks:
            rewritten_query = " ".join(
                _ordered_unique([rewritten_query, *[subtask.prompt for subtask in query_subtasks if subtask.prompt]])
            )
        return {
            "rewritten_query": rewritten_query,
            "sparse_query": self._sparse_query(query, retrieval_signals, rewritten_query, complexity_gate),
        }

    async def _node_semantic_route(self, state: dict[str, Any]) -> dict[str, Any]:
        retrieval_signals = state["retrieval_signals"]
        semantic_route = self._semantic_route(retrieval_signals)
        return {
            "semantic_route": semantic_route,
            "target_collections": self._target_collections(
                complexity_gate=state["complexity_gate"],
                semantic_route=semantic_route,
            ),
        }

    async def _node_predicate_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "predicate_plan": self._predicate_plan(
                state.get("version_gate_doc_ids") or state["source_scope"],
                retrieval_signals=state["retrieval_signals"],
                version_gate_expression=state.get("version_gate_expression"),
            )
        }

    async def _node_version_gate(self, state: dict[str, Any]) -> dict[str, Any]:
        scoped_doc_ids, expression = self._version_gate(tuple(state["source_scope"]))
        return {
            "version_gate_enabled": True,
            "version_gate_doc_ids": scoped_doc_ids,
            "version_gate_expression": expression,
        }

    async def _node_execution_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        retrieval_signals = state["retrieval_signals"]
        semantic_route = str(state["semantic_route"])
        retrieval_limit = int(state["retrieval_limit"])
        final_limit = int(state["final_limit"])
        complexity_gate = state["complexity_gate"]
        retrieval_profile = state["retrieval_profile"]
        retrieval_paths = self._retrieval_paths(
            retrieval_profile=retrieval_profile,
            retrieval_limit=retrieval_limit,
            retrieval_signals=retrieval_signals,
            semantic_route=semantic_route,
        )
        return {
            "retrieval_paths": retrieval_paths,
            "branch_stage_plans": self._branch_stage_plans(
                retrieval_limit=retrieval_limit,
                final_limit=final_limit,
                semantic_route=semantic_route,
                retrieval_signals=retrieval_signals,
            ),
            "fallback_plan": self._fallback_plan(
                retrieval_limit=retrieval_limit,
                semantic_route=semantic_route,
                retrieval_signals=retrieval_signals,
                enabled=self._use_summary_hybrid_paths,
            ),
            "operator_plan": self._operator_plan(
                complexity_gate=complexity_gate,
                semantic_route=semantic_route,
                retrieval_signals=retrieval_signals,
            ),
            "allow_web": True,
            "allow_graph_expansion": False if self._use_summary_hybrid_paths else True,
            "web_limit": max(1, retrieval_limit // 2),
            "graph_limit": max(2, final_limit),
            "fusion_strategy": "weighted_rrf",
            "fusion_alpha": 0.65,
        }

    @staticmethod
    def _route_after_complexity_gate(state: _PlannerState) -> str:
        return "decompose" if state.get("complexity_gate") is ComplexityGate.COMPLEX else "rewrite"

    @staticmethod
    def _complexity_gate(query: str, signals: RetrievalSignals) -> ComplexityGate:
        if signals.allow_graph_expansion:
            return ComplexityGate.COMPLEX
        return ComplexityGate.STANDARD

    @staticmethod
    def _rewrite_query(
        query: str,
        signals: RetrievalSignals,
        complexity_gate: ComplexityGate,
    ) -> str:
        base = query.strip()
        if complexity_gate is ComplexityGate.FAST_TRACK:
            return base
        extras = _ordered_unique(
            [
                *signals.structure_constraints.focus_terms,
                *signals.quoted_terms,
            ]
        )
        if not extras:
            return base
        return " ".join([base, *extras])

    @staticmethod
    def _sparse_query(
        query: str,
        signals: RetrievalSignals,
        rewritten_query: str,
        complexity_gate: ComplexityGate,
    ) -> str:
        if complexity_gate is ComplexityGate.FAST_TRACK:
            return query.strip()
        sparse_terms = _ordered_unique(
            [
                *signals.quoted_terms,
                *signals.structure_constraints.focus_terms,
                *[str(page) for page in signals.metadata_filters.page_numbers],
            ]
        )
        if not sparse_terms:
            return rewritten_query or query.strip()
        combined = " ".join([query.strip(), *sparse_terms])
        return combined if combined.strip() else (rewritten_query or query.strip())

    @staticmethod
    def _semantic_route(signals: RetrievalSignals) -> str:
        source_types = {source_type.lower() for source_type in signals.metadata_filters.source_types}
        asset_source_types = {"pptx", "xlsx", "image", "pdf"}
        if signals.special_targets:
            return "asset_first" if not signals.structure_constraints.has_constraints() else "text_plus_asset"
        if source_types & asset_source_types and signals.metadata_filters.page_numbers:
            return "text_plus_asset"
        return "text_first"

    def _predicate_plan(
        self,
        source_scope: Sequence[str],
        *,
        retrieval_signals: RetrievalSignals,
        version_gate_expression: str | None = None,
    ) -> PredicatePlan:
        doc_ids = tuple(_ordered_unique(source_scope))
        weak_clauses = self._weak_collection_clauses(retrieval_signals)
        if not doc_ids:
            return PredicatePlan(
                strategy="version_gate" if version_gate_expression else "none",
                expression=version_gate_expression,
                collection_expressions=self._merge_collection_clauses(version_gate_expression or "", weak_clauses),
            )
        if len(doc_ids) <= self._whitelist_threshold:
            base_expr = f"doc_id in [{', '.join(_format_expr_value(doc_id) for doc_id in doc_ids)}]"
            if version_gate_expression:
                base_expr = f"({version_gate_expression}) and ({base_expr})"
            return PredicatePlan(
                strategy="doc_id_whitelist",
                doc_ids=doc_ids,
                expression=base_expr,
                collection_expressions=self._merge_collection_clauses(base_expr, weak_clauses),
            )
        attribute_filters = self._attribute_filters(doc_ids)
        if attribute_filters:
            expressions = [
                f"{field} in [{', '.join(_format_expr_value(value) for value in values)}]"
                for field, values in attribute_filters.items()
            ]
            base_expr = " and ".join(expressions)
            if version_gate_expression:
                base_expr = f"({version_gate_expression}) and ({base_expr})"
            return PredicatePlan(
                strategy="attribute_filter",
                attribute_filters=attribute_filters,
                expression=base_expr,
                collection_expressions=self._merge_collection_clauses(base_expr, weak_clauses),
                overflowed=True,
            )
        return PredicatePlan(
            strategy="version_gate" if version_gate_expression else "none",
            doc_ids=doc_ids,
            expression=version_gate_expression,
            collection_expressions=self._merge_collection_clauses(version_gate_expression or "", weak_clauses),
            overflowed=True,
        )

    def _version_gate(self, source_scope: Sequence[str]) -> tuple[tuple[str, ...], str | None]:
        resolver = self._metadata_scope_resolver
        get_document = getattr(resolver, "get_document", None)
        list_documents = getattr(resolver, "list_documents", None)
        if not callable(get_document) or not callable(list_documents):
            return tuple(_ordered_unique(source_scope)), None
        gated_doc_ids: list[str] = []
        for doc_id in source_scope:
            document = _resolve_document(get_document, doc_id)
            if document is None:
                continue
            version_group_id = getattr(document, "version_group_id", None)
            if version_group_id is None:
                gated_doc_ids.append(str(getattr(document, "doc_id", doc_id)))
                continue
            active_versions = list_documents(version_group_id=int(version_group_id), active_only=True)
            if active_versions:
                gated_doc_ids.extend(str(item.doc_id) for item in active_versions)
            else:
                gated_doc_ids.append(str(getattr(document, "doc_id", doc_id)))
        normalized = tuple(_ordered_unique(gated_doc_ids or source_scope))
        return normalized, None

    def _weak_collection_clauses(self, signals: RetrievalSignals) -> dict[str, str]:
        clauses: dict[str, list[str]] = {}
        source_types = tuple(_ordered_unique(signals.metadata_filters.source_types))
        if source_types:
            clauses.setdefault("doc_summary", []).append(
                f"source_type in [{', '.join(_format_expr_value(item) for item in source_types)}]"
            )
            clauses.setdefault("section_summary", []).append(
                f"source_type in [{', '.join(_format_expr_value(item) for item in source_types)}]"
            )
        page_numbers = sorted({page for page in signals.metadata_filters.page_numbers if isinstance(page, int)})
        if page_numbers:
            page_expr = " or ".join(
                f"(page_start <= {page_no} and page_end >= {page_no})" for page_no in page_numbers
            )
            clauses.setdefault("section_summary", []).append(page_expr)
            clauses.setdefault("asset_summary", []).append(
                f"page_no in [{', '.join(str(page_no) for page_no in page_numbers)}]"
            )
        page_ranges = [page_range for page_range in signals.metadata_filters.page_ranges]
        if page_ranges:
            section_ranges = " or ".join(
                f"(page_start <= {page_range.end} and page_end >= {page_range.start})" for page_range in page_ranges
            )
            asset_ranges = " or ".join(
                f"(page_no >= {page_range.start} and page_no <= {page_range.end})" for page_range in page_ranges
            )
            clauses.setdefault("section_summary", []).append(section_ranges)
            clauses.setdefault("asset_summary", []).append(asset_ranges)
        if signals.special_targets:
            targets = ", ".join(
                _format_expr_value(item)
                for item in _ordered_unique(signals.special_targets)
            )
            clauses.setdefault("asset_summary", []).append(
                f"asset_type in [{targets}]"
            )
        return {collection: " and ".join(parts) for collection, parts in clauses.items() if parts}

    @staticmethod
    def _merge_collection_clauses(base_expr: str, collection_clauses: dict[str, str]) -> dict[str, str]:
        if not base_expr:
            return dict(collection_clauses)
        merged = {collection: f"({base_expr}) and ({clause})" for collection, clause in collection_clauses.items()}
        for collection in ("doc_summary", "section_summary", "asset_summary"):
            merged.setdefault(collection, base_expr)
        return merged

    def _attribute_filters(self, doc_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
        resolver = self._metadata_scope_resolver
        get_document = getattr(resolver, "get_document", None)
        if not callable(get_document):
            return {}
        candidates: dict[str, tuple[str, ...]] = {}
        for field_name in ("department_id", "auth_tag", "tenant_id"):
            values: list[str] = []
            for doc_id in doc_ids:
                document = _resolve_document(get_document, doc_id)
                if document is None:
                    values = []
                    break
                value = getattr(document, field_name, None)
                if value is None:
                    values = []
                    break
                normalized = str(value).strip()
                if not normalized:
                    values = []
                    break
                values.append(normalized)
            ordered_values = tuple(_ordered_unique(values))
            if ordered_values and len(ordered_values) <= self._attribute_value_cap:
                candidates[field_name] = ordered_values
        if not candidates:
            return {}
        preferred_order = {"department_id": 0, "auth_tag": 1, "tenant_id": 2}
        best_field = min(
            candidates,
            key=lambda field_name: (len(candidates[field_name]), preferred_order[field_name]),
        )
        return {best_field: candidates[best_field]}

    @staticmethod
    def _target_collections(
        *,
        complexity_gate: ComplexityGate,
        semantic_route: str,
    ) -> tuple[str, ...]:
        collections = ["section_summary"]
        if complexity_gate is ComplexityGate.COMPLEX:
            collections.append("doc_summary")
        if semantic_route in {"asset_first", "text_plus_asset"}:
            collections.append("asset_summary")
        return tuple(collections)

    def _retrieval_paths(
        self,
        *,
        retrieval_profile: RetrievalProfile,
        retrieval_limit: int,
        retrieval_signals: RetrievalSignals,
        semantic_route: str,
    ) -> tuple[RetrievalPath, ...]:
        if self._use_summary_hybrid_paths:
            vector_limit = retrieval_limit if retrieval_profile is RetrievalProfile.FAST else retrieval_limit * 2
            paths = [RetrievalPath("vector", vector_limit, QueryVariant.DENSE)]
            if (
                retrieval_profile is RetrievalProfile.ASSET
                or bool(retrieval_signals.special_targets)
                or semantic_route in {"asset_first", "text_plus_asset"}
            ):
                paths.append(RetrievalPath("special", retrieval_limit, QueryVariant.DENSE))
            return tuple(paths)
        aux_paths = [
            RetrievalPath("section", retrieval_limit, QueryVariant.SPARSE)
            for enabled in [retrieval_signals.structure_constraints.has_constraints()]
            if enabled
        ]
        if retrieval_signals.metadata_filters.has_constraints():
            aux_paths.append(RetrievalPath("metadata", retrieval_limit, QueryVariant.SPARSE))
        if bool(retrieval_signals.special_targets) or semantic_route in {"asset_first", "text_plus_asset"}:
            aux_paths.append(RetrievalPath("special", retrieval_limit, QueryVariant.DENSE))
        if retrieval_profile is RetrievalProfile.FAST:
            return (RetrievalPath("vector", retrieval_limit * 2, QueryVariant.DENSE),)
        if retrieval_profile is RetrievalProfile.ASSET:
            return (
                RetrievalPath("vector", retrieval_limit, QueryVariant.DENSE),
                RetrievalPath("special", retrieval_limit * 2, QueryVariant.DENSE),
                *[path for path in aux_paths if path.branch != "special"],
            )
        if retrieval_profile is RetrievalProfile.DEEP:
            return (
                RetrievalPath("local", retrieval_limit, QueryVariant.DENSE),
                RetrievalPath("global", retrieval_limit, QueryVariant.DENSE),
                RetrievalPath("vector", retrieval_limit, QueryVariant.DENSE),
                *aux_paths,
            )
        kg_limit = max(2, retrieval_limit - 1)
        return (
            RetrievalPath("local", kg_limit, QueryVariant.DENSE),
            RetrievalPath("global", kg_limit, QueryVariant.DENSE),
            RetrievalPath("vector", retrieval_limit, QueryVariant.DENSE),
            *aux_paths,
        )

    def _branch_stage_plans(
        self,
        *,
        retrieval_limit: int,
        final_limit: int,
        semantic_route: str,
        retrieval_signals: RetrievalSignals,
    ) -> tuple[BranchStagePlan, ...]:
        if not self._use_summary_hybrid_paths:
            return ()
        branch_limit = retrieval_limit * 2
        section_floor = max(2, min(final_limit, 5))
        stage_plans = [
            BranchStagePlan(
                branch="vector",
                stages=(
                    CollectionStage(
                        collection="section_summary",
                        limit=branch_limit,
                        min_hits=section_floor,
                        trigger="always",
                    ),
                    CollectionStage(
                        collection="doc_summary",
                        limit=max(retrieval_limit, section_floor),
                        min_hits=section_floor,
                        trigger="if_insufficient",
                    ),
                ),
            )
        ]
        if bool(retrieval_signals.special_targets) or semantic_route in {"asset_first", "text_plus_asset"}:
            stage_plans.append(
                BranchStagePlan(
                    branch="special",
                    stages=(
                        CollectionStage(
                            collection="asset_summary",
                            limit=retrieval_limit,
                            min_hits=max(1, min(final_limit, 3)),
                            trigger="always",
                        ),
                    ),
                )
            )
        return tuple(stage_plans)

    @staticmethod
    def _fallback_plan(
        *,
        retrieval_limit: int,
        semantic_route: str,
        retrieval_signals: RetrievalSignals,
        enabled: bool,
    ) -> tuple[FallbackStep, ...]:
        if not enabled:
            return ()
        steps: list[FallbackStep] = []
        if bool(retrieval_signals.special_targets) or semantic_route in {"asset_first", "text_plus_asset"}:
            steps.append(
                FallbackStep(
                    trigger="asset_fallback",
                    branch="special",
                    collection="asset_summary",
                    limit=retrieval_limit,
                )
            )
        return tuple(steps)

    @staticmethod
    def _operator_plan(
        *,
        complexity_gate: ComplexityGate,
        semantic_route: str,
        retrieval_signals: RetrievalSignals,
    ) -> tuple[OperatorStep, ...]:
        steps: list[OperatorStep] = [
            OperatorStep("VersionGate"),
            OperatorStep("EntitlementFilter"),
            OperatorStep("QueryRewrite"),
            OperatorStep("PredicatePushdown"),
            OperatorStep("SectionSearch", branch="vector", collection="section_summary"),
        ]
        if complexity_gate is ComplexityGate.COMPLEX:
            steps.append(OperatorStep("QueryDecomposition"))
            steps.append(OperatorStep("DocFallback", branch="vector", collection="doc_summary"))
        if bool(retrieval_signals.special_targets) or semantic_route in {"asset_first", "text_plus_asset"}:
            steps.append(OperatorStep("AssetSearch", branch="special", collection="asset_summary", required=False))
        steps.extend(
            [
                OperatorStep("HybridFusion"),
                OperatorStep("PreRerankCleanup"),
                OperatorStep("Rerank"),
                OperatorStep("ConfidenceAudit"),
            ]
        )
        return tuple(steps)

    @staticmethod
    def _decompose_query(
        query: str,
        signals: RetrievalSignals,
        complexity_gate: ComplexityGate,
    ) -> tuple[QuerySubTask, ...]:
        if complexity_gate is not ComplexityGate.COMPLEX:
            return ()
        normalized = query.strip()
        if signals.allow_graph_expansion:
            return (
                QuerySubTask(prompt=normalized, purpose="collect_primary_evidence"),
                QuerySubTask(prompt=normalized, purpose="collect_supporting_context"),
            )
        return (QuerySubTask(prompt=normalized, purpose="collect_primary_evidence"),)

    @staticmethod
    def _notes(
        predicate_plan: PredicatePlan,
        complexity_gate: ComplexityGate,
        semantic_route: str,
    ) -> tuple[str, ...]:
        notes: list[str] = [f"complexity={complexity_gate.value}", f"route={semantic_route}"]
        if predicate_plan.strategy != "none":
            notes.append(f"predicate={predicate_plan.strategy}")
        if predicate_plan.overflowed:
            notes.append("scope_overflow")
        return tuple(notes)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _format_expr_value(value: str) -> str:
    normalized = value.strip()
    if normalized.isdigit():
        return normalized
    escaped = normalized.replace('"', '\\"')
    return f'"{escaped}"'


def _resolve_document(get_document: object, doc_id: str) -> Any | None:
    if not callable(get_document):
        return None
    document = get_document(doc_id)
    if document is not None:
        return document
    if doc_id.isdigit():
        return get_document(int(doc_id))
    return None


__all__ = [
    "ComplexityGate",
    "CollectionStage",
    "BranchStagePlan",
    "FallbackStep",
    "OperatorStep",
    "PlanningGraph",
    "PlanningState",
    "PredicatePlan",
    "QuerySubTask",
    "QueryVariant",
    "RetrievalPath",
]

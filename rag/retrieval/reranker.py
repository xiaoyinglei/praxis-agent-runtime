from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent_runtime.text import keyword_overlap, search_terms
from rag.schema.model_protocols import Reranker
from rag.schema.query import GroundingTarget, RetrievalSignals

CandidateKind = Literal[
    "doc_summary",
    "section_summary",
    "asset_summary",
    "grounded_doc",
    "grounded_section",
    "grounded_asset",
]

SourceFamily = Literal[
    "dense",
    "sparse",
    "hybrid",
    "asset",
    "lexical",
    "fallback",
    "grounding",
    "unknown",
]


class RerankBindingLike(Protocol):
    backend: object
    provider_name: str
    model_name: str | None


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    candidate_id: str
    candidate_kind: CandidateKind
    doc_id: int

    source_id: int | None = None
    section_id: int | None = None
    asset_id: int | None = None
    parent_section_id: int | None = None

    text: str = ""
    title: str = ""
    citation_anchor: str | None = None
    section_path: tuple[str, ...] = field(default_factory=tuple)

    source_family: SourceFamily = "unknown"
    retrieval_channels: tuple[str, ...] = field(default_factory=tuple)

    retrieval_score: float = 0.0
    rerank_score: float | None = None
    final_score: float | None = None
    feature_score: float = 0.0
    rank: int = 0

    version_group_id: int | None = None
    version_no: int | None = None
    is_active: bool = True
    index_ready: bool = True
    doc_status: str | None = None

    grounding_target: GroundingTarget | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_candidate: object | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class CandidatePoolPolicy:
    family_keep_limits: dict[str, int] = field(
        default_factory=lambda: {
            "dense": 8,
            "sparse": 8,
            "hybrid": 12,
            "asset": 8,
            "lexical": 8,
            "grounding": 12,
            "fallback": 6,
            "unknown": 6,
        }
    )
    family_relative_score_floor: dict[str, float] = field(
        default_factory=lambda: {
            "dense": 0.20,
            "sparse": 0.15,
            "hybrid": 0.20,
            "asset": 0.10,
            "lexical": 0.05,
            "grounding": 0.05,
            "fallback": 0.05,
            "unknown": 0.10,
        }
    )
    family_absolute_score_floor: dict[str, float] = field(
        default_factory=lambda: {
            "dense": 0.01,
            "sparse": 0.01,
            "hybrid": 0.01,
            "asset": 0.001,
            "lexical": 0.001,
            "grounding": 0.001,
            "fallback": 0.001,
            "unknown": 0.001,
        }
    )


@dataclass(frozen=True, slots=True)
class FusionPolicy:
    rank_constant: int = 60
    use_retrieval_rank: bool = True
    use_rerank_rank: bool = True
    feature_weight: float = 0.02


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    empty_response_margin_threshold: float = 0.03
    asset_fallback_margin_threshold: float = 0.08
    negative_score_threshold: float = 0.0


@dataclass(frozen=True, slots=True)
class PreRerankDiagnostics:
    input_count: int
    deduplicated_count: int
    governance_filtered_count: int
    version_filtered_count: int
    score_floor_filtered_count: int
    family_truncated_count: int
    model_truncated_count: int
    output_count: int
    section_diversity_filtered_count: int = 0

    @property
    def noise_pruned_count(self) -> int:
        return self.score_floor_filtered_count


@dataclass(frozen=True, slots=True)
class ExitSignal:
    top1_score: float | None
    top2_score: float | None
    margin_signal: float | None
    top1_is_negative: bool
    candidate_count: int


@dataclass(frozen=True, slots=True)
class IndustrialRerankResult:
    ranked_candidates: list[Any]
    clean_candidates: list[RerankCandidate]
    diagnostics: PreRerankDiagnostics
    top1_confidence: float | None
    exit_decision: str | None
    exit_signal: ExitSignal


@dataclass(slots=True)
class _ScoredCandidate:
    candidate: RerankCandidate
    retrieval_score: float
    rerank_score: float | None = None
    final_score: float | None = None
    feature_score: float = 0.0

    @property
    def effective_score(self) -> float:
        if self.final_score is not None:
            return float(self.final_score)
        if self.rerank_score is not None:
            return float(self.rerank_score)
        return float(self.retrieval_score)

    def to_candidate(self, *, rank: int | None = None) -> RerankCandidate:
        return RerankCandidate(
            candidate_id=self.candidate.candidate_id,
            candidate_kind=self.candidate.candidate_kind,
            doc_id=self.candidate.doc_id,
            source_id=self.candidate.source_id,
            section_id=self.candidate.section_id,
            asset_id=self.candidate.asset_id,
            parent_section_id=self.candidate.parent_section_id,
            text=self.candidate.text,
            title=self.candidate.title,
            citation_anchor=self.candidate.citation_anchor,
            section_path=self.candidate.section_path,
            source_family=self.candidate.source_family,
            retrieval_channels=self.candidate.retrieval_channels,
            retrieval_score=self.retrieval_score,
            rerank_score=self.rerank_score,
            final_score=self.final_score,
            feature_score=self.feature_score,
            rank=self.candidate.rank if rank is None else rank,
            version_group_id=self.candidate.version_group_id,
            version_no=self.candidate.version_no,
            is_active=self.candidate.is_active,
            index_ready=self.candidate.index_ready,
            doc_status=self.candidate.doc_status,
            grounding_target=self.candidate.grounding_target,
            metadata=self.candidate.metadata,
            raw_candidate=self.candidate.raw_candidate,
        )


class ModelBackedRerankService:
    def __init__(
        self,
        *,
        binding: RerankBindingLike | None = None,
        provider: Reranker | None = None,
    ) -> None:
        self._binding = binding
        self._provider = binding.backend if binding is not None else provider

    @property
    def provider_name(self) -> str:
        if self._binding is not None:
            return self._binding.provider_name
        explicit = getattr(self._provider, "provider_name", None)
        return explicit if isinstance(explicit, str) and explicit else "reranker"

    @property
    def rerank_model_name(self) -> str:
        if self._binding is not None and self._binding.model_name:
            return self._binding.model_name
        explicit = getattr(self._provider, "rerank_model_name", None)
        return explicit if isinstance(explicit, str) and explicit else "unconfigured-reranker"

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        rerank = getattr(self._provider, "rerank", None)
        if not callable(rerank):
            raise RuntimeError("Rerank provider is not configured")
        scores = rerank(query, list(documents), **self._supported_kwargs(rerank, kwargs))
        return _coerce_scores(scores, expected=len(documents))

    @staticmethod
    def _supported_kwargs(rerank: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not kwargs:
            return {}
        try:
            parameters = inspect.signature(rerank).parameters
        except (TypeError, ValueError):
            return kwargs
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in parameters}


class IndustrialRerankService:
    def __init__(
        self,
        *,
        max_model_candidates: int = 50,
        rerank_batch_size: int = 8,
        rerank_max_length: int = 1024,
        candidate_pool_policy: CandidatePoolPolicy | None = None,
        fusion_policy: FusionPolicy | None = None,
        exit_policy: ExitPolicy | None = None,
    ) -> None:
        if max_model_candidates <= 0:
            raise ValueError("max_model_candidates must be positive")
        if rerank_batch_size <= 0:
            raise ValueError("rerank_batch_size must be positive")
        if rerank_max_length <= 0:
            raise ValueError("rerank_max_length must be positive")
        self._max_model_candidates = max_model_candidates
        self._rerank_batch_size = rerank_batch_size
        self._rerank_max_length = rerank_max_length
        self._candidate_pool_policy = candidate_pool_policy or CandidatePoolPolicy()
        self._fusion_policy = fusion_policy or FusionPolicy()
        self._exit_policy = exit_policy or ExitPolicy()

    def rank(
        self,
        *,
        query: str,
        fused_candidates: Sequence[object],
        reranker: Reranker | None,
        rerank_required: bool,
        rerank_pool_k: int | None,
        allow_asset_fallback: bool,
        retrieval_signals: RetrievalSignals | None = None,
        min_output_candidates: int | None = None,
    ) -> IndustrialRerankResult:
        clean_candidates = self._normalize_candidates(fused_candidates)
        working, diagnostics = self._pre_rerank_cleanup(
            clean_candidates,
            min_output_candidates=min_output_candidates,
        )
        working, model_truncated_count = self._truncate_for_model(working)
        self._apply_feature_scores(query=query, candidates=working, retrieval_signals=retrieval_signals)
        diagnostics = _update_diagnostics(diagnostics, model_truncated_count, len(working))

        if rerank_required:
            self._rerank_candidates(
                query=query,
                candidates=working,
                reranker=reranker,
                rerank_pool_k=rerank_pool_k,
            )

        self._fuse_scores(working)
        ranked = self._stable_sort(working)
        exit_signal = self._build_exit_signal(ranked)
        exit_decision = self._exit_decision(exit_signal, allow_asset_fallback=allow_asset_fallback)
        top1_confidence = self._top1_confidence(exit_signal)
        return IndustrialRerankResult(
            ranked_candidates=[_raw_candidate(item, rank=index) for index, item in enumerate(ranked, start=1)],
            clean_candidates=[item.to_candidate(rank=index) for index, item in enumerate(ranked, start=1)],
            diagnostics=diagnostics,
            top1_confidence=top1_confidence,
            exit_decision=exit_decision,
            exit_signal=exit_signal,
        )

    async def arank(
        self,
        *,
        query: str,
        fused_candidates: Sequence[object],
        reranker: Reranker | None,
        rerank_required: bool,
        rerank_pool_k: int | None,
        allow_asset_fallback: bool,
        retrieval_signals: RetrievalSignals | None = None,
        min_output_candidates: int | None = None,
    ) -> IndustrialRerankResult:
        clean_candidates = self._normalize_candidates(fused_candidates)
        working, diagnostics = self._pre_rerank_cleanup(
            clean_candidates,
            min_output_candidates=min_output_candidates,
        )
        working, model_truncated_count = self._truncate_for_model(working)
        self._apply_feature_scores(query=query, candidates=working, retrieval_signals=retrieval_signals)
        diagnostics = _update_diagnostics(diagnostics, model_truncated_count, len(working))

        if rerank_required:
            await self._arerank_candidates(
                query=query,
                candidates=working,
                reranker=reranker,
                rerank_pool_k=rerank_pool_k,
            )

        self._fuse_scores(working)
        ranked = self._stable_sort(working)
        exit_signal = self._build_exit_signal(ranked)
        exit_decision = self._exit_decision(exit_signal, allow_asset_fallback=allow_asset_fallback)
        top1_confidence = self._top1_confidence(exit_signal)
        return IndustrialRerankResult(
            ranked_candidates=[_raw_candidate(item, rank=index) for index, item in enumerate(ranked, start=1)],
            clean_candidates=[item.to_candidate(rank=index) for index, item in enumerate(ranked, start=1)],
            diagnostics=diagnostics,
            top1_confidence=top1_confidence,
            exit_decision=exit_decision,
            exit_signal=exit_signal,
        )

    def _normalize_candidates(self, candidates: Sequence[object]) -> list[RerankCandidate]:
        return [_normalize_candidate(candidate, rank=index + 1) for index, candidate in enumerate(candidates)]

    def _pre_rerank_cleanup(
        self,
        candidates: Sequence[RerankCandidate],
        *,
        min_output_candidates: int | None = None,
    ) -> tuple[list[_ScoredCandidate], PreRerankDiagnostics]:
        working = [
            _ScoredCandidate(candidate=candidate, retrieval_score=candidate.retrieval_score) for candidate in candidates
        ]
        deduplicated, deduplicated_count = self._deduplicate(working)
        governed, governance_filtered_count = self._filter_governance(deduplicated)
        versioned, version_filtered_count = self._filter_versions(governed)
        score_filtered, score_floor_filtered_count = self._apply_score_floors(versioned)
        diverse, section_diversity_filtered_count = self._section_diversity_filter(score_filtered)
        family_kept, family_truncated_count = self._apply_family_limits(
            diverse,
            min_output_candidates=min_output_candidates,
        )
        diagnostics = PreRerankDiagnostics(
            input_count=len(candidates),
            deduplicated_count=deduplicated_count,
            governance_filtered_count=governance_filtered_count,
            version_filtered_count=version_filtered_count,
            score_floor_filtered_count=score_floor_filtered_count,
            section_diversity_filtered_count=section_diversity_filtered_count,
            family_truncated_count=family_truncated_count,
            model_truncated_count=0,
            output_count=len(family_kept),
        )
        return family_kept, diagnostics

    def _deduplicate(self, candidates: Sequence[_ScoredCandidate]) -> tuple[list[_ScoredCandidate], int]:
        kept: list[_ScoredCandidate] = []
        seen: set[tuple[str, str]] = set()
        dropped = 0
        for candidate in self._stable_sort(candidates):
            key = self._dedupe_key(candidate)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept.append(candidate)
        return kept, dropped

    @staticmethod
    def _filter_governance(candidates: Sequence[_ScoredCandidate]) -> tuple[list[_ScoredCandidate], int]:
        kept: list[_ScoredCandidate] = []
        dropped = 0
        for candidate in candidates:
            if not candidate.candidate.is_active or not candidate.candidate.index_ready:
                dropped += 1
                continue
            if str(candidate.candidate.doc_status or "").strip().lower() in {"retired", "expired", "deleted"}:
                dropped += 1
                continue
            kept.append(candidate)
        return kept, dropped

    def _filter_versions(self, candidates: Sequence[_ScoredCandidate]) -> tuple[list[_ScoredCandidate], int]:
        chosen: dict[tuple[int, str, str], tuple[int, float, int, _ScoredCandidate]] = {}
        passthrough: list[tuple[int, _ScoredCandidate]] = []
        dropped = 0
        for index, candidate in enumerate(candidates):
            version_key = self._version_key(candidate)
            version_no = candidate.candidate.version_no
            if version_key is None or version_no is None:
                passthrough.append((index, candidate))
                continue
            current = chosen.get(version_key)
            score = float(candidate.retrieval_score)
            if current is None or (version_no, score) > (current[0], current[1]):
                if current is not None:
                    dropped += 1
                chosen[version_key] = (version_no, score, index, candidate)
            else:
                dropped += 1
        kept = [candidate for _, candidate in passthrough]
        kept.extend(candidate for _, _, _, candidate in sorted(chosen.values(), key=lambda item: item[2]))
        return self._stable_sort(kept), dropped

    @staticmethod
    def _section_diversity_filter(
        candidates: Sequence[_ScoredCandidate],
        *,
        max_per_section: int = 2,
    ) -> tuple[list[_ScoredCandidate], int]:
        """按 (doc_id, section_path_prefix) 分组，每组最多保留 max_per_section 条。

        防止同一章的多个相邻段堆积，挤占不同角度的证据。
        """
        if not candidates:
            return [], 0
        grouped: dict[str, list[_ScoredCandidate]] = {}
        for c in candidates:
            sp = getattr(c.candidate, "section_path", None)
            if sp and len(sp) >= 2:
                key = f"{c.candidate.doc_id}:{sp[0]}:{sp[1]}"
            else:
                key = f"{c.candidate.doc_id}:section:{getattr(c.candidate, 'section_id', 0)}"
            grouped.setdefault(key, []).append(c)

        kept: list[_ScoredCandidate] = []
        dropped = 0
        for group in grouped.values():
            sorted_group = sorted(group, key=lambda c: c.effective_score, reverse=True)
            kept.extend(sorted_group[:max_per_section])
            dropped += max(0, len(sorted_group) - max_per_section)
        return kept, dropped

    def _apply_score_floors(self, candidates: Sequence[_ScoredCandidate]) -> tuple[list[_ScoredCandidate], int]:
        if not candidates:
            return [], 0
        grouped: dict[str, list[_ScoredCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.candidate.source_family].append(candidate)
        kept: list[_ScoredCandidate] = []
        dropped = 0
        for family, family_candidates in grouped.items():
            top_score = max(max(float(candidate.retrieval_score), 0.0) for candidate in family_candidates)
            floor = max(
                self._candidate_pool_policy.family_absolute_score_floor.get(family, 0.0),
                top_score * self._candidate_pool_policy.family_relative_score_floor.get(family, 0.0),
            )
            for candidate in self._stable_sort(family_candidates):
                if max(float(candidate.retrieval_score), 0.0) >= floor:
                    kept.append(candidate)
                else:
                    dropped += 1
        return self._stable_sort(kept), dropped

    def _apply_family_limits(
        self,
        candidates: Sequence[_ScoredCandidate],
        *,
        min_output_candidates: int | None = None,
    ) -> tuple[list[_ScoredCandidate], int]:
        grouped: dict[str, list[_ScoredCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.candidate.source_family].append(candidate)
        kept: list[_ScoredCandidate] = []
        dropped = 0
        min_keep = max(int(min_output_candidates or 0), 0)
        for family, family_candidates in grouped.items():
            limit = self._candidate_pool_policy.family_keep_limits.get(family, len(family_candidates))
            if min_keep:
                limit = max(limit, min_keep)
            sorted_family = self._stable_sort(family_candidates)
            kept.extend(sorted_family[:limit])
            dropped += max(0, len(sorted_family) - limit)
        return self._stable_sort(kept), dropped

    def _truncate_for_model(self, candidates: Sequence[_ScoredCandidate]) -> tuple[list[_ScoredCandidate], int]:
        if len(candidates) <= self._max_model_candidates:
            return list(candidates), 0
        return list(candidates[: self._max_model_candidates]), len(candidates) - self._max_model_candidates

    def _apply_feature_scores(
        self,
        *,
        query: str,
        candidates: Sequence[_ScoredCandidate],
        retrieval_signals: RetrievalSignals | None,
    ) -> None:
        query_terms = search_terms(query)
        focus_terms = []
        special_targets: set[str] = set()
        requested_pages: set[str] = set()
        if retrieval_signals is not None:
            focus_terms = list(retrieval_signals.structure_constraints.focus_terms or retrieval_signals.quoted_terms)
            special_targets = set(retrieval_signals.special_targets)
            requested_pages = {str(page) for page in retrieval_signals.metadata_filters.page_numbers}
        for candidate in candidates:
            text = self._render_candidate_text(candidate)
            section_text = " ".join(candidate.candidate.section_path)
            page_text = " ".join(
                str(value)
                for value in (
                    candidate.candidate.metadata.get("page_no"),
                    candidate.candidate.metadata.get("page_start"),
                    candidate.candidate.metadata.get("page_end"),
                )
                if value is not None
            )
            score = 0.0
            score += min(keyword_overlap(query_terms, text), 5) * 0.05
            score += min(keyword_overlap(focus_terms, section_text), 5) * 0.10
            if requested_pages and requested_pages & set(search_terms(page_text)):
                score += 0.25
            if special_targets and candidate.candidate.candidate_kind == "asset_summary":
                asset_type = str(candidate.candidate.metadata.get("asset_type", ""))
                if asset_type in special_targets:
                    score += 0.30
            candidate.feature_score = round(score, 6)

    def _rerank_candidates(
        self,
        *,
        query: str,
        candidates: Sequence[_ScoredCandidate],
        reranker: Reranker | None,
        rerank_pool_k: int | None,
    ) -> None:
        rerank = getattr(reranker, "rerank", None)
        if not callable(rerank):
            return
        pool_limit = max(1, rerank_pool_k) if rerank_pool_k is not None else len(candidates)
        pool = list(candidates[:pool_limit])
        documents = [self._render_candidate_text(candidate) for candidate in pool]
        kwargs = ModelBackedRerankService._supported_kwargs(
            rerank,
            {"batch_size": self._rerank_batch_size, "max_length": self._rerank_max_length},
        )
        scores = rerank(
            query,
            documents,
            **kwargs,
        )
        for candidate, score in zip(pool, _coerce_scores(scores, expected=len(pool)), strict=False):
            candidate.rerank_score = float(score)

    async def _arerank_candidates(
        self,
        *,
        query: str,
        candidates: Sequence[_ScoredCandidate],
        reranker: Reranker | None,
        rerank_pool_k: int | None,
    ) -> None:
        rerank = getattr(reranker, "rerank", None)
        if not callable(rerank):
            return
        pool_limit = max(1, rerank_pool_k) if rerank_pool_k is not None else len(candidates)
        pool = list(candidates[:pool_limit])
        documents = [self._render_candidate_text(candidate) for candidate in pool]
        kwargs = ModelBackedRerankService._supported_kwargs(
            rerank,
            {"batch_size": self._rerank_batch_size, "max_length": self._rerank_max_length},
        )
        result = rerank(
            query,
            documents,
            **kwargs,
        )
        if inspect.isawaitable(result):
            result = await result
        for candidate, score in zip(pool, _coerce_scores(result, expected=len(pool)), strict=False):
            candidate.rerank_score = float(score)

    def _fuse_scores(self, candidates: Sequence[_ScoredCandidate]) -> None:
        if not candidates:
            return
        rank_constant = self._fusion_policy.rank_constant
        retrieval_rank = {
            candidate.candidate.candidate_id: index + 1
            for index, candidate in enumerate(
                sorted(candidates, key=lambda item: (-float(item.retrieval_score), item.candidate.candidate_id))
            )
        }
        rerank_rank: dict[str, int] = {}
        if any(candidate.rerank_score is not None for candidate in candidates):
            rerank_rank = {
                candidate.candidate.candidate_id: index + 1
                for index, candidate in enumerate(
                    sorted(
                        candidates,
                        key=lambda item: (
                            -(float(item.rerank_score) if item.rerank_score is not None else float("-inf")),
                            item.candidate.candidate_id,
                        ),
                    )
                )
            }
        for candidate in candidates:
            score = 0.0
            if self._fusion_policy.use_retrieval_rank:
                score += 1.0 / (rank_constant + retrieval_rank[candidate.candidate.candidate_id])
            if self._fusion_policy.use_rerank_rank and candidate.candidate.candidate_id in rerank_rank:
                score += 1.0 / (rank_constant + rerank_rank[candidate.candidate.candidate_id])
            score += candidate.feature_score * self._fusion_policy.feature_weight
            candidate.final_score = round(score, 8)

    @staticmethod
    def _stable_sort(candidates: Sequence[_ScoredCandidate]) -> list[_ScoredCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate.effective_score),
                -float(candidate.feature_score),
                -float(candidate.retrieval_score),
                candidate.candidate.candidate_id,
            ),
        )

    @staticmethod
    def _render_candidate_text(candidate: _ScoredCandidate) -> str:
        parts: list[str] = []
        if candidate.candidate.title:
            parts.append(candidate.candidate.title)
        if candidate.candidate.section_path:
            parts.append(" > ".join(candidate.candidate.section_path))
        parent_text = str(candidate.candidate.metadata.get("parent_text", "") or "")
        if parent_text:
            parts.append(parent_text)
        if candidate.candidate.text:
            parts.append(candidate.candidate.text)
        return "\n\n".join(part.strip() for part in parts if part and part.strip())

    @staticmethod
    def _dedupe_key(candidate: _ScoredCandidate) -> tuple[str, str]:
        if candidate.candidate.candidate_kind in {"doc_summary", "grounded_doc"}:
            return (candidate.candidate.candidate_kind, str(candidate.candidate.doc_id))
        if (
            candidate.candidate.candidate_kind in {"section_summary", "grounded_section"}
            and candidate.candidate.section_id is not None
        ):
            return (candidate.candidate.candidate_kind, str(candidate.candidate.section_id))
        if (
            candidate.candidate.candidate_kind in {"asset_summary", "grounded_asset"}
            and candidate.candidate.asset_id is not None
        ):
            return (candidate.candidate.candidate_kind, str(candidate.candidate.asset_id))
        return (candidate.candidate.candidate_kind, candidate.candidate.candidate_id)

    @staticmethod
    def _version_key(candidate: _ScoredCandidate) -> tuple[int, str, str] | None:
        if candidate.candidate.version_group_id is None:
            return None
        return (
            int(candidate.candidate.version_group_id),
            candidate.candidate.candidate_kind,
            candidate.candidate.candidate_id,
        )

    def _build_exit_signal(self, candidates: Sequence[_ScoredCandidate]) -> ExitSignal:
        if not candidates:
            return ExitSignal(None, None, None, False, 0)
        top1 = float(candidates[0].effective_score)
        top2 = float(candidates[1].effective_score) if len(candidates) > 1 else None
        margin_signal = None if top2 is None else round(top1 - top2, 6)
        return ExitSignal(
            top1_score=top1,
            top2_score=top2,
            margin_signal=margin_signal,
            top1_is_negative=(candidates[0].rerank_score or 0.0) < self._exit_policy.negative_score_threshold,
            candidate_count=len(candidates),
        )

    def _exit_decision(self, signal: ExitSignal, *, allow_asset_fallback: bool) -> str | None:
        if signal.candidate_count == 0 or signal.top1_score is None:
            return "empty_response"
        if signal.top1_is_negative:
            return "asset_fallback" if allow_asset_fallback else "empty_response"
        if signal.margin_signal is None:
            return "answer"
        if signal.margin_signal < self._exit_policy.empty_response_margin_threshold:
            return "asset_fallback" if allow_asset_fallback else "empty_response"
        if allow_asset_fallback and signal.margin_signal < self._exit_policy.asset_fallback_margin_threshold:
            return "asset_fallback"
        return "answer"

    @staticmethod
    def _top1_confidence(signal: ExitSignal) -> float | None:
        if signal.top1_score is None:
            return None
        if signal.top2_score is None:
            return round(abs(signal.top1_score) / (abs(signal.top1_score) + 1.0), 6)
        denominator = max(abs(signal.top1_score) + abs(signal.top2_score), 1e-6)
        return round(max(0.0, min(1.0, abs(signal.margin_signal or 0.0) / denominator)), 6)


def _normalize_candidate(candidate: object, *, rank: int) -> RerankCandidate:
    if isinstance(candidate, RerankCandidate):
        return candidate if candidate.raw_candidate is not None else _copy_with_raw(candidate, candidate)
    metadata = _candidate_metadata(candidate)
    grounding_target = _grounding_target(candidate, metadata)
    doc_id = (
        _first_int(
            getattr(candidate, "doc_id", None),
            metadata.get("doc_id"),
            None if grounding_target is None else grounding_target.doc_id,
        )
        or 0
    )
    source_id = _first_int(
        getattr(candidate, "source_id", None),
        metadata.get("source_id"),
        None if grounding_target is None else grounding_target.source_id,
    )
    section_id = _first_int(
        getattr(candidate, "section_id", None),
        metadata.get("section_id"),
        None if grounding_target is None else grounding_target.section_id,
    )
    asset_id = _first_int(
        getattr(candidate, "asset_id", None),
        metadata.get("asset_id"),
        None if grounding_target is None else grounding_target.asset_id,
    )
    candidate_kind = _candidate_kind(candidate, metadata, grounding_target, section_id, asset_id)
    candidate_id = _candidate_id(candidate, metadata, grounding_target, candidate_kind, doc_id, section_id, asset_id)
    retrieval_channels = tuple(str(item) for item in (getattr(candidate, "retrieval_channels", None) or ()))
    score = (
        _first_float(
            getattr(candidate, "final_score", None),
            getattr(candidate, "fusion_score", None),
            getattr(candidate, "rrf_score", None),
            getattr(candidate, "score", None),
            metadata.get("score"),
        )
        or 0.0
    )
    section_path = tuple(
        str(item) for item in (getattr(candidate, "section_path", None) or metadata.get("section_path") or ())
    )
    return RerankCandidate(
        candidate_id=candidate_id,
        candidate_kind=candidate_kind,
        doc_id=doc_id,
        source_id=source_id,
        section_id=section_id,
        asset_id=asset_id,
        parent_section_id=_first_int(getattr(candidate, "parent_section_id", None), metadata.get("parent_section_id")),
        text=str(getattr(candidate, "text", "") or metadata.get("text", "") or ""),
        title=str(getattr(candidate, "title", "") or metadata.get("title", "") or ""),
        citation_anchor=getattr(candidate, "citation_anchor", None) or metadata.get("citation_anchor"),
        section_path=section_path,
        source_family=_source_family(candidate, metadata, retrieval_channels, candidate_kind),
        retrieval_channels=retrieval_channels,
        retrieval_score=score,
        rank=int(getattr(candidate, "rank", rank) or rank),
        version_group_id=_first_int(getattr(candidate, "version_group_id", None), metadata.get("version_group_id")),
        version_no=_first_int(getattr(candidate, "version_no", None), metadata.get("version_no")),
        is_active=_truthy(metadata.get("is_active"), default=True),
        index_ready=_truthy(metadata.get("index_ready"), default=True),
        doc_status=None if metadata.get("doc_status") is None else str(metadata.get("doc_status")),
        grounding_target=grounding_target,
        metadata=metadata,
        raw_candidate=candidate,
    )


def _copy_with_raw(candidate: RerankCandidate, raw_candidate: object) -> RerankCandidate:
    return RerankCandidate(
        candidate_id=candidate.candidate_id,
        candidate_kind=candidate.candidate_kind,
        doc_id=candidate.doc_id,
        source_id=candidate.source_id,
        section_id=candidate.section_id,
        asset_id=candidate.asset_id,
        parent_section_id=candidate.parent_section_id,
        text=candidate.text,
        title=candidate.title,
        citation_anchor=candidate.citation_anchor,
        section_path=candidate.section_path,
        source_family=candidate.source_family,
        retrieval_channels=candidate.retrieval_channels,
        retrieval_score=candidate.retrieval_score,
        rerank_score=candidate.rerank_score,
        final_score=candidate.final_score,
        feature_score=candidate.feature_score,
        rank=candidate.rank,
        version_group_id=candidate.version_group_id,
        version_no=candidate.version_no,
        is_active=candidate.is_active,
        index_ready=candidate.index_ready,
        doc_status=candidate.doc_status,
        grounding_target=candidate.grounding_target,
        metadata=candidate.metadata,
        raw_candidate=raw_candidate,
    )


def _candidate_metadata(candidate: object) -> dict[str, Any]:
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {}


def _grounding_target(candidate: object, metadata: Mapping[str, Any]) -> GroundingTarget | None:
    target = getattr(candidate, "grounding_target", None) or metadata.get("grounding_target")
    if isinstance(target, GroundingTarget):
        return target
    if isinstance(target, Mapping):
        return GroundingTarget.model_validate(target)
    return None


def _candidate_kind(
    candidate: object,
    metadata: Mapping[str, Any],
    grounding_target: GroundingTarget | None,
    section_id: int | None,
    asset_id: int | None,
) -> CandidateKind:
    explicit = (
        getattr(candidate, "candidate_kind", None) or metadata.get("candidate_kind") or metadata.get("record_type")
    )
    if explicit in {
        "doc_summary",
        "section_summary",
        "asset_summary",
        "grounded_doc",
        "grounded_section",
        "grounded_asset",
    }:
        return explicit  # type: ignore[no-any-return]
    if grounding_target is not None:
        if grounding_target.kind == "asset" or grounding_target.asset_id is not None:
            return "asset_summary"
        if grounding_target.kind == "doc":
            return "doc_summary"
        return "section_summary"
    if asset_id is not None:
        return "asset_summary"
    if section_id is not None:
        return "section_summary"
    return "section_summary"


def _candidate_id(
    candidate: object,
    metadata: Mapping[str, Any],
    grounding_target: GroundingTarget | None,
    candidate_kind: CandidateKind,
    doc_id: int,
    section_id: int | None,
    asset_id: int | None,
) -> str:
    explicit = getattr(candidate, "candidate_id", None) or metadata.get("candidate_id")
    if explicit:
        return str(explicit)
    if grounding_target is not None:
        if grounding_target.asset_id is not None:
            return f"asset:{grounding_target.asset_id}"
        if grounding_target.section_id is not None:
            return f"section:{grounding_target.section_id}"
        return f"doc:{grounding_target.doc_id}"
    if asset_id is not None:
        return f"asset:{asset_id}"
    if section_id is not None:
        return f"section:{section_id}"
    fallback = getattr(candidate, "item_id", None) or metadata.get("item_id")
    if fallback:
        return f"{candidate_kind}:{fallback}"
    evidence_id = getattr(candidate, "evidence_id", None) or metadata.get("evidence_id")
    if evidence_id:
        return f"{candidate_kind}:{evidence_id}"
    return f"{candidate_kind}:doc:{doc_id}"


def _source_family(
    candidate: object,
    metadata: Mapping[str, Any],
    retrieval_channels: Sequence[str],
    candidate_kind: CandidateKind,
) -> SourceFamily:
    explicit = getattr(candidate, "source_family", None) or metadata.get("source_family")
    if explicit in {"dense", "sparse", "hybrid", "asset", "lexical", "fallback", "grounding", "unknown"}:
        return explicit  # type: ignore[no-any-return]
    channels = {channel for channel in retrieval_channels if channel}
    if candidate_kind == "asset_summary" or channels & {"asset", "special"}:
        return "asset"
    if channels & {"hybrid"}:
        return "hybrid"
    if channels & {"sparse", "bm25", "full_text"}:
        return "lexical" if "full_text" in channels else "sparse"
    if channels & {"dense", "vector"}:
        return "dense"
    if str(getattr(candidate, "source_kind", "")).lower() == "external":
        return "fallback"
    return "unknown"


def _update_diagnostics(
    diagnostics: PreRerankDiagnostics,
    model_truncated_count: int,
    output_count: int,
) -> PreRerankDiagnostics:
    return PreRerankDiagnostics(
        input_count=diagnostics.input_count,
        deduplicated_count=diagnostics.deduplicated_count,
        governance_filtered_count=diagnostics.governance_filtered_count,
        version_filtered_count=diagnostics.version_filtered_count,
        score_floor_filtered_count=diagnostics.score_floor_filtered_count,
        family_truncated_count=diagnostics.family_truncated_count,
        model_truncated_count=model_truncated_count,
        output_count=output_count,
    )


def _raw_candidate(candidate: _ScoredCandidate, *, rank: int) -> Any:
    raw_candidate = candidate.candidate.raw_candidate
    if raw_candidate is None or isinstance(raw_candidate, RerankCandidate):
        return candidate.to_candidate(rank=rank)
    _try_setattr(raw_candidate, "rank", rank)
    _try_setattr(raw_candidate, "score", candidate.effective_score)
    _try_setattr(raw_candidate, "rerank_score", candidate.rerank_score)
    _try_setattr(raw_candidate, "final_score", candidate.final_score)
    _try_setattr(raw_candidate, "feature_score", candidate.feature_score)
    metadata = getattr(raw_candidate, "metadata", None)
    if isinstance(metadata, dict):
        metadata["candidate_kind"] = candidate.candidate.candidate_kind
        metadata["source_family"] = candidate.candidate.source_family
        metadata["retrieval_score"] = candidate.retrieval_score
        if candidate.rerank_score is not None:
            metadata["rerank_score"] = candidate.rerank_score
        if candidate.final_score is not None:
            metadata["final_score"] = candidate.final_score
        if candidate.candidate.grounding_target is not None:
            metadata["grounding_target"] = candidate.candidate.grounding_target.model_dump()
    return raw_candidate


def _coerce_scores(scores: object, *, expected: int) -> list[float]:
    tolist = getattr(scores, "tolist", None)
    if callable(tolist):
        scores = tolist()
    if not isinstance(scores, Sequence) or isinstance(scores, str | bytes):
        raise RuntimeError(f"Unsupported rerank score payload: {type(scores)!r}")
    normalized = [float(score) for score in scores]
    if len(normalized) != expected:
        raise RuntimeError(f"Rerank score count mismatch: expected {expected}, got {len(normalized)}")
    return normalized


def _first_int(*values: object) -> int | None:
    for value in values:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _first_float(*values: object) -> float | None:
    for value in values:
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _truthy(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _try_setattr(target: object, name: str, value: object) -> None:
    try:
        setattr(target, name, value)
    except Exception:
        return


__all__ = [
    "CandidateKind",
    "CandidatePoolPolicy",
    "ExitPolicy",
    "ExitSignal",
    "FusionPolicy",
    "IndustrialRerankResult",
    "IndustrialRerankService",
    "ModelBackedRerankService",
    "PreRerankDiagnostics",
    "RerankCandidate",
    "SourceFamily",
]

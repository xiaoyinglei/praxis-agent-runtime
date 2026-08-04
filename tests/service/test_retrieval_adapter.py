from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rag.retrieval.evidence import EvidenceService
from rag.retrieval.models import RetrievalProfile
from rag.retrieval.planning_graph import ComplexityGate, PlanningState, PredicatePlan, QueryVariant, RetrievalPath
from rag.retrieval.retrieval_adapter import RetrievalAdapter
from rag.schema.query import RetrievalSignals
from rag.schema.runtime import AccessPolicy, RuntimeMode


@dataclass(frozen=True)
class _FakeCandidate:
    evidence_id: str
    doc_id: str
    text: str
    citation_anchor: str
    score: float
    rank: int
    source_kind: str = "internal"
    source_id: str | None = None
    section_path: tuple[str, ...] = ()
    benchmark_doc_id: str | None = None

    @property
    def item_id(self) -> str:
        return self.evidence_id


class _AsyncVectorRetriever:
    absorbs_sparse_branch = True

    def __init__(self) -> None:
        self.calls = 0

    async def aretrieve_with_plan(self, *, query: str, source_scope: list[str], plan: PlanningState, **kwargs):
        del query, source_scope, plan, kwargs
        self.calls += 1
        return [_FakeCandidate("evidence-a", "doc-a", "alpha", "#a", 0.9, 1)]


class _FullTextRetriever:
    def __call__(self, query: str, source_scope: list[str], retrieval_signals: RetrievalSignals):
        raise AssertionError("full_text branch should be skipped when vector hybrid absorbs sparse recall")


class _BranchRegistry:
    def __init__(self, *, vector_retriever: object, full_text_retriever: object) -> None:
        self._retrievers = {
            "vector": vector_retriever,
            "full_text": full_text_retriever,
        }

    def get(self, branch: str):
        return self._retrievers[branch]

    def collect_web(
        self,
        *,
        query: str,
        source_scope: list[str],
        retrieval_signals: RetrievalSignals,
    ) -> list[_FakeCandidate]:
        del query, source_scope, retrieval_signals
        return []


def test_retrieval_adapter_awaits_async_plan_aware_vector_and_skips_full_text() -> None:
    vector_retriever = _AsyncVectorRetriever()
    adapter = RetrievalAdapter(
        branch_registry=_BranchRegistry(
            vector_retriever=vector_retriever,
            full_text_retriever=_FullTextRetriever(),
        ),
        evidence_service=EvidenceService(),
    )
    plan = PlanningState(
        original_query="alpha",
        rewritten_query="alpha",
        sparse_query="alpha",
        retrieval_profile=RetrievalProfile.AUTO,
        complexity_gate=ComplexityGate.STANDARD,
        semantic_route="text_first",
        target_collections=("section_summary",),
        predicate_plan=PredicatePlan(),
        retrieval_paths=(
            RetrievalPath("vector", 2, QueryVariant.DENSE),
            RetrievalPath("full_text", 2, QueryVariant.SPARSE),
        ),
        allow_web=False,
        allow_graph_expansion=False,
        web_limit=0,
        graph_limit=0,
    )

    result = asyncio.run(
        adapter.acollect_internal(
            plan=plan,
            source_scope=[],
            access_policy=AccessPolicy.default(),
            runtime_mode=RuntimeMode.FAST,
            retrieval_signals=RetrievalSignals(),
        )
    )

    assert vector_retriever.calls == 1
    assert result.branch_hits == {"vector": 1, "full_text": 0}
    assert [branch for branch, _items in result.branches] == ["vector"]


def test_effective_scope_does_not_turn_overflow_into_empty_scope_for_plain_retrievers() -> None:
    plan = PlanningState(
        original_query="alpha",
        rewritten_query="alpha",
        sparse_query="alpha",
        retrieval_profile=RetrievalProfile.AUTO,
        complexity_gate=ComplexityGate.STANDARD,
        semantic_route="text_first",
        target_collections=("section_summary",),
        predicate_plan=PredicatePlan(
            strategy="none",
            doc_ids=("doc-a", "doc-b", "doc-c"),
            overflowed=True,
        ),
        retrieval_paths=(RetrievalPath("vector", 2, QueryVariant.DENSE),),
        allow_web=False,
        allow_graph_expansion=False,
        web_limit=0,
        graph_limit=0,
    )

    assert RetrievalAdapter._effective_scope(
        ["doc-a", "doc-b", "doc-c"],
        plan=plan,
        supports_plan=False,
    ) == ["doc-a", "doc-b", "doc-c"]

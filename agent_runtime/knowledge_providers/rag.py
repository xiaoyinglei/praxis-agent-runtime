from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Self

from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic
from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.tools.integrations.knowledge import (
    KnowledgeResult,
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
)
from agent_runtime.tools.permissions import ToolExecutionContext
from rag.retrieval import QueryOptions

if TYPE_CHECKING:
    from rag.runtime import RAGRuntime


@dataclass
class LazyRAGKnowledgeProvider:
    config: RAGKnowledgeConfig
    model_alias: str | None = None
    vector_dsn: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._runtime: RAGRuntime | None = None
        self._runtime_context_entered = False
        self._diagnostics: tuple[RuntimeDiagnostic, ...] = ()

    @property
    def diagnostics(self) -> tuple[RuntimeDiagnostic, ...]:
        return self._diagnostics

    async def search_knowledge(
        self,
        payload: KnowledgeSearchInput,
        execution_context: ToolExecutionContext,
    ) -> KnowledgeSearchOutput:
        del execution_context
        runtime = self._ensure_runtime()
        query_result = await asyncio.to_thread(
            runtime.query,
            payload.query,
            options=QueryOptions(top_k=payload.top_k),
        )
        answer = query_result.answer
        evidence_items = query_result.retrieval.evidence.all[: payload.top_k]
        results = [
            KnowledgeResult(
                evidence_id=evidence.evidence_id,
                doc_id=evidence.doc_id,
                citation_anchor=evidence.citation_anchor,
                text=evidence.text,
                score=evidence.score,
                source_type=evidence.source_type or "",
                file_name=evidence.file_name or "",
            )
            for evidence in evidence_items
        ]
        return KnowledgeSearchOutput(
            results=results,
            answer_text=answer.answer_text,
            citations=[citation.citation_anchor or citation.citation_id for citation in answer.citations],
            groundedness_flag=answer.groundedness_flag,
            insufficient_evidence=answer.insufficient_evidence_flag,
            total_found=len(results),
        )

    def close(self) -> None:
        runtime = self._runtime
        if runtime is None or not self._runtime_context_entered:
            return
        exit_method = getattr(runtime, "__exit__", None)
        if callable(exit_method):
            exit_method(None, None, None)
        self._runtime_context_entered = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    def _ensure_runtime(self) -> RAGRuntime:
        if self._runtime is not None:
            return self._runtime

        runtime, diagnostics = _build_optional_rag_runtime(
            config=self.config,
            model_alias=self.model_alias,
            vector_dsn=self.vector_dsn,
        )
        self._diagnostics = tuple(diagnostics)
        if runtime is None:
            detail = (
                "rag_knowledge_init_failed: knowledge runtime unavailable"
                if not diagnostics
                else f"{diagnostics[-1].code}: {diagnostics[-1].message}"
            )
            raise RuntimeError(detail)

        enter_method = getattr(runtime, "__enter__", None)
        if callable(enter_method):
            runtime = enter_method()
            self._runtime_context_entered = True
        self._runtime = runtime
        return runtime


def _build_optional_rag_runtime(
    *,
    config: RAGKnowledgeConfig,
    model_alias: str | None,
    vector_dsn: str | None,
) -> tuple[RAGRuntime | None, tuple[RuntimeDiagnostic, ...]]:
    try:
        if not config.storage_root.exists():
            raise FileNotFoundError(f"RAG storage root does not exist: {config.storage_root}")
        from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime
        from rag.models.assembly_adapter import to_assembly_overrides
        from rag.models.runtime import RuntimeOverrides, resolve_runtime_config
        from rag.storage.runtime_config import runtime_storage_config

        runtime_config = resolve_runtime_config(
            RuntimeOverrides(
                model_alias=model_alias,
                embedding_model_alias=config.embedding_model,
                reranker_model_alias=config.reranker_model or "none",
            )
        )
        storage = runtime_storage_config(
            config.storage_root,
            vector_backend=config.vector_backend,
            vector_dsn=vector_dsn,
            vector_namespace=config.vector_namespace,
            vector_collection_prefix=config.vector_collection_prefix,
        )
        runtime = RAGRuntime.from_request(
            storage=storage,
            request=AssemblyRequest(
                requirements=CapabilityRequirements(
                    require_chat=True,
                    default_context_tokens=QueryOptions().max_context_tokens,
                ),
                overrides=to_assembly_overrides(runtime_config),
            ),
            generation_config=runtime_config.generation,
            chat_context_window_tokens=runtime_config.primary_model.context_window_tokens or 32_768,
            llm_stage_budgets=runtime_config.llm_stage_budgets,
        )
        return runtime, ()
    except Exception as exc:
        return None, (
            RuntimeDiagnostic(
                code="rag_knowledge_init_failed",
                component="rag_runtime",
                message="Configured knowledge runtime could not be initialized.",
                severity="error",
                error_type=type(exc).__name__[:120],
            ),
        )

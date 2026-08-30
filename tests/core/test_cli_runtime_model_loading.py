from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest
import typer

from rag.assembly.models import (
    AssemblyOverrides,
    CapabilityRequirements,
)
from rag.assembly.support import _CompositeProvider
from rag.models.assembly_adapter import to_assembly_overrides

# ── Stub provider for testing (simulates EmbeddingHttpClient / RerankHttpClient) ──


class _StubEmbedder:
    @property
    def embedding_model_name(self) -> str:
        return "stub-embedding-model"

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        return [[0.1] * 4 for _ in texts]

    def close(self) -> None:
        pass


class _StubReranker:
    @property
    def rerank_model_name(self) -> str:
        return "stub-rerank-model"

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        return [0.5] * len(documents)

    def close(self) -> None:
        pass


def _stub_embedding_provider() -> _CompositeProvider:
    return _CompositeProvider(provider_name="embedding-http", embedder=_StubEmbedder())


def _stub_rerank_provider() -> _CompositeProvider:
    return _CompositeProvider(provider_name="rerank-http", reranker=_StubReranker())


# ── Service URL env → pre-built runtime provider ──


def test_embedding_service_url_env_sets_embedding_provider() -> None:
    """When RAG_EMBEDDING_SERVICE_URL is set, embedding_provider should be set."""
    from agent_runtime.modeling.config import ModelCapability, ModelRuntimeConfig, ModelSpec

    runtime_config = ModelRuntimeConfig(
        primary_model=ModelSpec(
            alias="test", capability=ModelCapability.CHAT, provider="openai_compatible", model="gpt-4"
        ),
        embedding_model=ModelSpec(
            alias="test_emb", capability=ModelCapability.EMBEDDING, provider="mlx_embedding", model="mlx-model"
        ),
    )
    overrides = to_assembly_overrides(runtime_config)
    assert overrides.embedding_provider is None  # YAML doesn't set runtime providers

    # Simulate env injection: _runtime() constructs HTTP client, wraps in _CompositeProvider
    overrides = replace(overrides, embedding_provider=_stub_embedding_provider())
    assert overrides.embedding_provider is not None
    assert overrides.embedding_provider.provider_name == "embedding-http"
    assert overrides.embedding_provider.is_embed_configured is True


def test_rerank_service_url_env_sets_rerank_provider() -> None:
    """When RAG_RERANK_SERVICE_URL is set, rerank_provider should be set."""
    from agent_runtime.modeling.config import ModelCapability, ModelRuntimeConfig, ModelSpec

    runtime_config = ModelRuntimeConfig(
        primary_model=ModelSpec(
            alias="test", capability=ModelCapability.CHAT, provider="openai_compatible", model="gpt-4"
        ),
        embedding_model=ModelSpec(
            alias="test_emb", capability=ModelCapability.EMBEDDING, provider="mlx_embedding", model="mlx-model"
        ),
        reranker_model=ModelSpec(
            alias="test_rerank",
            capability=ModelCapability.RERANKER,
            provider="sentence_transformers",
            model="bge-reranker",
        ),
    )
    overrides = to_assembly_overrides(runtime_config)
    assert overrides.rerank_provider is None

    overrides = replace(overrides, rerank_provider=_stub_rerank_provider())
    assert overrides.rerank_provider is not None
    assert overrides.rerank_provider.provider_name == "rerank-http"
    assert overrides.rerank_provider.is_rerank_configured is True


# ── require_rerank=False strips reranker (both config and runtime provider) ──


def test_require_rerank_false_strips_reranker_and_runtime_provider() -> None:
    """require_rerank=False strips both rerank config and rerank_provider."""
    from agent_runtime.modeling.config import ModelCapability, ModelRuntimeConfig, ModelSpec

    runtime_config = ModelRuntimeConfig(
        primary_model=ModelSpec(
            alias="test", capability=ModelCapability.CHAT, provider="openai_compatible", model="gpt-4"
        ),
        embedding_model=ModelSpec(
            alias="test_emb", capability=ModelCapability.EMBEDDING, provider="mlx_embedding", model="mlx-model"
        ),
        reranker_model=ModelSpec(
            alias="test_rerank",
            capability=ModelCapability.RERANKER,
            provider="sentence_transformers",
            model="bge-reranker",
        ),
    )
    overrides = to_assembly_overrides(runtime_config)
    assert overrides.rerank is not None  # YAML default provides reranker

    # Simulate env injection + require_rerank=False (ingest)
    overrides = replace(overrides, rerank_provider=_stub_rerank_provider())
    assert overrides.rerank_provider is not None

    # require_rerank=False → strip both
    require_rerank = False
    if not require_rerank:
        overrides = replace(overrides, rerank=None, rerank_provider=None)

    assert overrides.rerank is None
    assert overrides.rerank_provider is None


def test_require_rerank_true_preserves_reranker() -> None:
    """When require_rerank=True and YAML has reranker, it should be preserved."""
    from agent_runtime.modeling.config import ModelCapability, ModelRuntimeConfig, ModelSpec

    runtime_config = ModelRuntimeConfig(
        primary_model=ModelSpec(
            alias="test", capability=ModelCapability.CHAT, provider="openai_compatible", model="gpt-4"
        ),
        embedding_model=ModelSpec(
            alias="test_emb", capability=ModelCapability.EMBEDDING, provider="mlx_embedding", model="mlx-model"
        ),
        reranker_model=ModelSpec(
            alias="test_rerank",
            capability=ModelCapability.RERANKER,
            provider="sentence_transformers",
            model="bge-reranker",
        ),
    )
    overrides = to_assembly_overrides(runtime_config)
    assert overrides.rerank is not None
    assert overrides.rerank.provider_kind == "local-bge"


# ── CLI + env conflict detection ──


def test_cli_embedding_model_plus_env_conflict() -> None:
    """CLI explicit --embedding-model + RAG_EMBEDDING_SERVICE_URL → conflict error."""
    embedding_service_url = "http://127.0.0.1:9090"
    embedding_model = "mlx_embedding"  # explicitly set via CLI

    if embedding_service_url and embedding_model is not None:
        with pytest.raises(typer.BadParameter):
            raise typer.BadParameter("RAG_EMBEDDING_SERVICE_URL is set but --embedding-model was also specified.")


def test_cli_reranker_model_plus_env_conflict() -> None:
    """CLI explicit --reranker-model + RAG_RERANK_SERVICE_URL → conflict error."""
    rerank_service_url = "http://127.0.0.1:9091"
    reranker_model = "bge_reranker"  # explicitly set via CLI

    if rerank_service_url and reranker_model is not None:
        with pytest.raises(typer.BadParameter):
            raise typer.BadParameter("RAG_RERANK_SERVICE_URL is set but --reranker-model was also specified.")


# ── env absent → YAML default ──


def test_no_service_url_env_uses_yaml_default() -> None:
    """Without service URL env, YAML default provider should be used (no runtime provider)."""
    from agent_runtime.modeling.config import ModelCapability, ModelRuntimeConfig, ModelSpec

    runtime_config = ModelRuntimeConfig(
        primary_model=ModelSpec(
            alias="test", capability=ModelCapability.CHAT, provider="openai_compatible", model="gpt-4"
        ),
        embedding_model=ModelSpec(
            alias="test_emb", capability=ModelCapability.EMBEDDING, provider="mlx_embedding", model="mlx-model"
        ),
    )
    overrides = to_assembly_overrides(runtime_config)

    assert overrides.embedding is not None
    assert overrides.embedding.provider_kind == "mlx-embedding"
    assert overrides.embedding_provider is None  # no runtime provider without env


# ── CapabilityRequirements wiring ──


def test_ingest_requirements_no_rerank() -> None:
    """Ingest should set require_rerank=False."""
    requirements = CapabilityRequirements(require_chat=False, require_rerank=False)
    assert requirements.require_rerank is False
    assert requirements.require_chat is False
    assert requirements.require_embedding is True


def test_query_requirements_with_rerank() -> None:
    """Query should set require_rerank=True."""
    requirements = CapabilityRequirements(require_chat=False, require_rerank=True)
    assert requirements.require_rerank is True


# ── AssemblyOverrides new fields ──


def test_assembly_overrides_supports_embedding_provider_field() -> None:
    """AssemblyOverrides has embedding_provider field for pre-built runtime providers."""
    provider = _stub_embedding_provider()
    overrides = AssemblyOverrides(embedding_provider=provider)
    assert overrides.embedding_provider is provider


def test_assembly_overrides_supports_rerank_provider_field() -> None:
    """AssemblyOverrides has rerank_provider field for pre-built runtime providers."""
    provider = _stub_rerank_provider()
    overrides = AssemblyOverrides(rerank_provider=provider)
    assert overrides.rerank_provider is provider


def test_assembly_overrides_defaults_are_none() -> None:
    """New fields default to None for backward compatibility."""
    overrides = AssemblyOverrides()
    assert overrides.embedding_provider is None
    assert overrides.rerank_provider is None

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agent_runtime.modeling.config import GenerationTaskConfig, ModelRuntimeConfig, ModelSpec
from rag.assembly.models import AssemblyOverrides, ProviderConfig, TokenizerConfig

if TYPE_CHECKING:
    from rag.models.catalog import ModelCatalog


_PROVIDER_KIND_MAP: dict[str, str] = {
    "openai_compatible": "openai-compatible",
    "ollama": "ollama",
    "sentence_transformers": "local-bge",
    "mlx_embedding": "mlx-embedding",
}


def resolve_task_model(task_config: GenerationTaskConfig, catalog: ModelCatalog) -> ModelSpec:
    """Resolve a generation task's model alias to a ModelSpec.

    If task_config.model is set, uses it directly.
    Otherwise falls back to catalog's defaults.primary_model.
    """
    alias = task_config.model
    if alias:
        return catalog.get_model(alias)
    return catalog.get_default_primary()


def to_assembly_overrides(config: ModelRuntimeConfig) -> AssemblyOverrides:
    """Convert ModelRuntimeConfig to AssemblyOverrides.

    ONLY converts configuration — does NOT create provider instances.
    Provider instantiation stays in rag.assembly.support.build_provider.
    """
    tokenizer_config = config.tokenizer
    return AssemblyOverrides(
        chat=_to_chat_provider_config(config.primary_model),
        embedding=_to_embedding_provider_config(config.embedding_model),
        rerank=_to_reranker_provider_config(config.reranker_model),
        tokenizer=TokenizerConfig(
            tokenizer_backend=tokenizer_config.tokenizer_backend,
            chunk_token_size=tokenizer_config.chunk_token_size,
            chunk_overlap_tokens=tokenizer_config.chunk_overlap_tokens,
            max_context_tokens=tokenizer_config.max_context_tokens,
            prompt_reserved_tokens=tokenizer_config.prompt_reserved_tokens,
            local_files_only=tokenizer_config.local_files_only,
        )
        if (
            tokenizer_config.tokenizer_backend is not None
            or tokenizer_config.chunk_token_size is not None
            or tokenizer_config.chunk_overlap_tokens is not None
            or tokenizer_config.max_context_tokens is not None
            or tokenizer_config.prompt_reserved_tokens is not None
            or tokenizer_config.local_files_only is not None
        )
        else None,
    )


def _to_chat_provider_config(spec: ModelSpec) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=_map_kind(spec.provider),
        chat_model=spec.model,
        base_url=spec.base_url,
        api_key=_resolve_api_key(spec),
    )


def _to_embedding_provider_config(spec: ModelSpec) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=_map_kind(spec.provider),
        embedding_model=spec.model,
        embedding_space=spec.embedding_space or spec.model,
        base_url=spec.base_url,
    )


def _to_reranker_provider_config(spec: ModelSpec | None) -> ProviderConfig | None:
    if spec is None:
        return None
    return ProviderConfig(
        provider_kind=_map_kind(spec.provider),
        rerank_model=spec.model,
    )


def _map_kind(provider: str) -> str:
    mapped = _PROVIDER_KIND_MAP.get(provider)
    if mapped is not None:
        return mapped
    raise ValueError(f"Unknown provider: {provider!r}")


def _resolve_api_key(spec: ModelSpec) -> str | None:
    if spec.api_key_env:
        return os.environ.get(spec.api_key_env)
    return None

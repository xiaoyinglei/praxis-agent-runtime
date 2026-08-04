from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from agent_runtime.modeling.chat import OpenAICompatibleChatGenerator
from agent_runtime.modeling.contracts import LLMProviderResult
from agent_runtime.modeling.providers.ollama.generator import OllamaGenerator
from rag.assembly.models import (
    AssemblyConfig,
    AssemblyOverrides,
    ProviderConfig,
    TokenizerConfig,
)
from rag.providers.fallback import FallbackEmbeddingRepo
from rag.providers.huggingface.embedder import BgeM3Embedder, HuggingFaceEmbedder
from rag.providers.huggingface.rerank import FlagEmbeddingReranker
from rag.providers.mlx.embedder import MLXEmbedder
from rag.providers.ollama.embedder import OllamaEmbedder

T = TypeVar("T")
_JSON_CODE_FENCE_RE = re.compile(
    r"^\s*```\s*(?:[A-Za-z0-9_-]+)?\s*(?P<body>.*?)\s*```\s*$",
    re.DOTALL,
)


@dataclass(slots=True)
class _UnavailableProvider:
    provider_name: str
    reason: str
    chat_model_name: str | None = None
    embedding_model_name: str | None = None
    rerank_model_name: str | None = None
    is_chat_configured: bool = False
    is_embed_configured: bool = False
    is_rerank_configured: bool = False


@dataclass(slots=True)
class _CompositeProvider:
    provider_name: str
    generator: object | None = None
    embedder: object | None = None
    reranker: object | None = None

    @property
    def is_chat_configured(self) -> bool:
        return self.generator is not None

    @property
    def is_embed_configured(self) -> bool:
        return self.embedder is not None

    @property
    def is_rerank_configured(self) -> bool:
        return self.reranker is not None

    @property
    def chat_model_name(self) -> str | None:
        return _backend_model_name(self.generator, "chat")

    @property
    def embedding_model_name(self) -> str | None:
        return _backend_model_name(self.embedder, "embedding")

    @property
    def embedding_space(self) -> str | None:
        value = getattr(self.embedder, "embedding_space", None)
        return value if isinstance(value, str) and value else None

    @property
    def rerank_model_name(self) -> str | None:
        return _backend_model_name(self.reranker, "rerank")

    def generate_text(self, *, prompt: str, **kwargs: Any) -> str:
        backend = self._require(self.generator, capability="chat")
        generate_text = getattr(backend, "generate_text", None)
        if not callable(generate_text):
            raise RuntimeError(f"{self.provider_name} does not implement chat generation")
        return str(generate_text(prompt=prompt, **kwargs))

    def generate_text_with_usage(
        self,
        *,
        prompt: str,
        **kwargs: Any,
    ) -> LLMProviderResult[str]:
        backend = self._require(self.generator, capability="chat")
        method = getattr(backend, "generate_text_with_usage", None)
        if callable(method):
            return method(prompt=prompt, **kwargs)  # type: ignore[no-any-return]
        return LLMProviderResult(value=self.generate_text(prompt=prompt, **kwargs))

    def generate_structured(self, *, prompt: str, schema: type[T], **kwargs: Any) -> T:
        backend = self._require(self.generator, capability="chat")
        generate_structured = getattr(backend, "generate_structured", None)
        if not callable(generate_structured):
            raise RuntimeError(f"{self.provider_name} does not implement structured generation")
        return generate_structured(prompt=prompt, schema=schema, **kwargs)  # type: ignore[no-any-return]

    def generate_structured_with_usage(
        self,
        *,
        prompt: str,
        schema: type[T],
        **kwargs: Any,
    ) -> LLMProviderResult[T]:
        backend = self._require(self.generator, capability="chat")
        method = getattr(backend, "generate_structured_with_usage", None)
        if callable(method):
            return method(prompt=prompt, schema=schema, **kwargs)  # type: ignore[no-any-return]
        return LLMProviderResult(value=self.generate_structured(prompt=prompt, schema=schema, **kwargs))

    def embed(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        backend = self._require(self.embedder, capability="embedding")
        embed = getattr(backend, "embed", None)
        if not callable(embed):
            raise RuntimeError(f"{self.provider_name} does not implement embedding")
        return list(embed(texts, **kwargs))

    def embed_query(self, texts: Sequence[str], **kwargs: Any) -> list[list[float]]:
        backend = self._require(self.embedder, capability="embedding")
        embed_query = getattr(backend, "embed_query", None)
        if callable(embed_query):
            return list(embed_query(texts, **kwargs))
        return self.embed(texts, **kwargs)

    def embed_query_sparse(self, texts: Sequence[str], **kwargs: Any) -> list[dict[int, float]]:
        backend = self._require(self.embedder, capability="embedding")
        embed_query_sparse = getattr(backend, "embed_query_sparse", None)
        if not callable(embed_query_sparse):
            raise RuntimeError(f"{self.provider_name} does not implement sparse embedding")
        return list(embed_query_sparse(texts, **kwargs))

    def rerank(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        backend = self._require(self.reranker, capability="rerank")
        rerank = getattr(backend, "rerank", None)
        if not callable(rerank):
            raise RuntimeError(f"{self.provider_name} does not implement rerank")
        return [float(score) for score in rerank(query, documents, **kwargs)]

    def _require(self, backend: object | None, *, capability: str) -> object:
        if backend is None:
            raise RuntimeError(f"{self.provider_name} {capability} capability is not configured")
        return backend


_OpenAICompatibleChatGenerator = OpenAICompatibleChatGenerator


def _backend_model_name(backend: object | None, capability: str) -> str | None:
    if backend is None:
        return None
    attribute_names = {
        "chat": ("model_name_or_path", "_model_name_or_path", "chat_model_name"),
        "embedding": ("embedding_model_name", "model_name_or_path", "_model_name_or_path"),
        "rerank": ("rerank_model_name", "model_name_or_path", "_model_name_or_path"),
    }.get(capability, ())
    for attribute_name in attribute_names:
        value = getattr(backend, attribute_name, None)
        if isinstance(value, str) and value:
            return value
    return None


def first_non_blank(*values: str | None) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def env_int(*names: str) -> int | None:
    value = first_env(*names)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def env_bool(*names: str) -> bool | None:
    value = first_env(*names)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def normalize_gemini_base_url(url: str | None) -> str | None:
    if url is None:
        return None
    normalized = url.rstrip("/")
    if "generativelanguage.googleapis.com" in normalized and not normalized.endswith("/openai"):
        return f"{normalized}/openai"
    return normalized


def compatibility_config_from_environment() -> tuple[AssemblyConfig, dict[str, str | int | bool | None]]:
    compatibility_inputs: dict[str, str | int | bool | None] = {}
    profiles: list[ProviderConfig] = []

    def remember(key: str, value: str | int | bool | None) -> str | int | bool | None:
        if value is not None:
            compatibility_inputs[key] = value
        return value

    openai_api_key = remember(
        "openai_api_key",
        first_env("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "PKP_OPENAI__API_KEY"),
    )
    openai_base_url = remember(
        "openai_base_url",
        normalize_gemini_base_url(first_env("OPENAI_BASE_URL", "GEMINI_BASE_URL", "PKP_OPENAI__BASE_URL")),
    )
    openai_chat_model = remember(
        "openai_chat_model",
        first_env("OPENAI_MODEL", "GEMINI_CHAT_MODEL", "PKP_OPENAI__MODEL"),
    )
    openai_embedding_model = remember(
        "openai_embedding_model",
        first_env(
            "OPENAI_EMBEDDING_MODEL",
            "GEMINI_EMBEDDING_MODEL",
            "PKP_OPENAI__EMBEDDING_MODEL",
        ),
    )
    if openai_api_key and (openai_chat_model or openai_embedding_model):
        label_base = "OpenAI"
        if isinstance(openai_base_url, str) and "generativelanguage.googleapis.com" in openai_base_url:
            label_base = "Gemini (OpenAI compatible)"
        profiles.append(
            ProviderConfig(
                profile_id="openai-compatible",
                provider_kind="openai-compatible",
                location="cloud",
                label=f"{label_base} / {openai_chat_model or openai_embedding_model}",
                api_key=str(openai_api_key),
                base_url=None if openai_base_url is None else str(openai_base_url),
                chat_model=None if openai_chat_model is None else str(openai_chat_model),
                embedding_model=None if openai_embedding_model is None else str(openai_embedding_model),
                embedding_space=None if openai_embedding_model is None else str(openai_embedding_model),
            )
        )

    ollama_base_url = remember("ollama_base_url", first_env("OLLAMA_BASE_URL", "PKP_OLLAMA__BASE_URL"))
    ollama_chat_model = remember("ollama_chat_model", first_env("OLLAMA_CHAT_MODEL", "PKP_OLLAMA__CHAT_MODEL"))
    ollama_embedding_model = remember(
        "ollama_embedding_model",
        first_env("OLLAMA_EMBEDDING_MODEL", "PKP_OLLAMA__EMBEDDING_MODEL"),
    )
    if ollama_base_url and (ollama_chat_model or ollama_embedding_model):
        profiles.append(
            ProviderConfig(
                profile_id="ollama",
                provider_kind="ollama",
                location="local",
                label=f"Ollama / {ollama_chat_model or ollama_embedding_model}",
                base_url=str(ollama_base_url),
                chat_model=None if ollama_chat_model is None else str(ollama_chat_model),
                embedding_model=None if ollama_embedding_model is None else str(ollama_embedding_model),
                embedding_space=None if ollama_embedding_model is None else str(ollama_embedding_model),
            )
        )

    local_bge_enabled = remember(
        "local_bge_enabled",
        first_env("PKP_LOCAL_BGE__ENABLED", "RAG_LOCAL_BGE_ENABLED"),
    )
    local_bge_embedding_model = remember(
        "local_bge_embedding_model",
        first_env("PKP_LOCAL_BGE__EMBEDDING_MODEL", "RAG_LOCAL_BGE_EMBEDDING_MODEL"),
    )
    local_bge_embedding_model_path = remember(
        "local_bge_embedding_model_path",
        first_env("PKP_LOCAL_BGE__EMBEDDING_MODEL_PATH", "RAG_LOCAL_BGE_EMBEDDING_MODEL_PATH"),
    )
    local_bge_embedding_batch_size = remember(
        "local_bge_embedding_batch_size",
        env_int("PKP_LOCAL_BGE__EMBEDDING_BATCH_SIZE", "RAG_LOCAL_BGE_EMBEDDING_BATCH_SIZE"),
    )
    local_bge_device = remember(
        "local_bge_device",
        first_env("PKP_LOCAL_BGE__DEVICE", "RAG_LOCAL_BGE_DEVICE"),
    )
    local_bge_rerank_model = remember(
        "local_bge_rerank_model",
        first_env("RAG_RERANK_MODEL", "PKP_LOCAL_BGE__RERANK_MODEL"),
    )
    local_bge_rerank_model_path = remember(
        "local_bge_rerank_model_path",
        first_env("RAG_RERANK_MODEL_PATH", "PKP_LOCAL_BGE__RERANK_MODEL_PATH"),
    )
    local_bge_rerank_batch_size = remember(
        "local_bge_rerank_batch_size",
        env_int("PKP_LOCAL_BGE__RERANK_BATCH_SIZE", "RAG_LOCAL_BGE_RERANK_BATCH_SIZE"),
    )
    local_bge_allowed = (str(local_bge_enabled).lower() if local_bge_enabled is not None else "") not in {
        "0",
        "false",
        "no",
        "off",
    }
    if local_bge_allowed and (
        local_bge_embedding_model
        or local_bge_embedding_model_path
        or local_bge_rerank_model
        or local_bge_rerank_model_path
    ):
        profiles.append(
            ProviderConfig(
                profile_id="local-bge",
                provider_kind="local-bge",
                location="local",
                label=f"Local BGE / {local_bge_embedding_model or local_bge_rerank_model or 'custom'}",
                embedding_model=None if local_bge_embedding_model is None else str(local_bge_embedding_model),
                embedding_space=None if local_bge_embedding_model is None else str(local_bge_embedding_model),
                embedding_model_path=None
                if local_bge_embedding_model_path is None
                else str(local_bge_embedding_model_path),
                embedding_batch_size=(
                    None if local_bge_embedding_batch_size is None else int(local_bge_embedding_batch_size)
                ),
                device=None if local_bge_device is None else str(local_bge_device),
                rerank_model=None if local_bge_rerank_model is None else str(local_bge_rerank_model),
                rerank_model_path=None if local_bge_rerank_model_path is None else str(local_bge_rerank_model_path),
                rerank_batch_size=(None if local_bge_rerank_batch_size is None else int(local_bge_rerank_batch_size)),
            )
        )
    # local-hf provider removed; all models are cloud or OpenAI-compatible now.

    # Tokenizer config comes from models.yaml + assembly defaults, not env vars.
    tokenizer_config = TokenizerConfig()
    return AssemblyConfig(profiles=tuple(profiles), tokenizer=tokenizer_config), compatibility_inputs


def merge_provider_config(
    high: ProviderConfig | None,
    low: ProviderConfig | None,
) -> ProviderConfig | None:
    if high is None:
        return low
    if low is None:
        return high
    embedding_model_path = (
        None
        if high.embedding_model is not None and high.embedding_model_path is None
        else first_non_blank(high.embedding_model_path, low.embedding_model_path)
    )
    chat_model_path = (
        None
        if high.chat_model is not None and high.chat_model_path is None
        else first_non_blank(high.chat_model_path, low.chat_model_path)
    )
    rerank_model_path = (
        None
        if high.rerank_model is not None and high.rerank_model_path is None
        else first_non_blank(high.rerank_model_path, low.rerank_model_path)
    )
    return ProviderConfig(
        provider_kind=high.provider_kind or low.provider_kind,
        location=first_non_blank(high.location, low.location) or low.location,
        profile_id=first_non_blank(high.profile_id, low.profile_id),
        label=first_non_blank(high.label, low.label),
        api_key=first_non_blank(high.api_key, low.api_key),
        base_url=first_non_blank(high.base_url, low.base_url),
        chat_model=first_non_blank(high.chat_model, low.chat_model),
        chat_model_path=chat_model_path,
        chat_backend=first_non_blank(high.chat_backend, low.chat_backend),
        embedding_model=first_non_blank(high.embedding_model, low.embedding_model),
        embedding_space=first_non_blank(high.embedding_space, low.embedding_space),
        rerank_model=first_non_blank(high.rerank_model, low.rerank_model),
        embedding_model_path=embedding_model_path,
        rerank_model_path=rerank_model_path,
        embedding_batch_size=(
            first_positive_int(high.embedding_batch_size, low.embedding_batch_size)
            if high.embedding_batch_size or low.embedding_batch_size
            else None
        ),
        rerank_batch_size=(
            first_positive_int(high.rerank_batch_size, low.rerank_batch_size)
            if high.rerank_batch_size or low.rerank_batch_size
            else None
        ),
        device=first_non_blank(high.device, low.device),
        enabled=high.enabled if high is not None else low.enabled,
    )


def merge_tokenizer_config(
    high: TokenizerConfig | None,
    low: TokenizerConfig | None,
) -> TokenizerConfig | None:
    if high is None:
        return low
    if low is None:
        return high
    return TokenizerConfig(
        embedding_model_name=first_non_blank(high.embedding_model_name, low.embedding_model_name),
        tokenizer_model_name=first_non_blank(high.tokenizer_model_name, low.tokenizer_model_name),
        chunking_tokenizer_model_name=first_non_blank(
            high.chunking_tokenizer_model_name,
            low.chunking_tokenizer_model_name,
        ),
        tokenizer_backend=first_non_blank(high.tokenizer_backend, low.tokenizer_backend),
        chunk_token_size=high.chunk_token_size if high.chunk_token_size is not None else low.chunk_token_size,
        chunk_overlap_tokens=(
            high.chunk_overlap_tokens if high.chunk_overlap_tokens is not None else low.chunk_overlap_tokens
        ),
        max_context_tokens=high.max_context_tokens if high.max_context_tokens is not None else low.max_context_tokens,
        prompt_reserved_tokens=(
            high.prompt_reserved_tokens if high.prompt_reserved_tokens is not None else low.prompt_reserved_tokens
        ),
        local_files_only=high.local_files_only if high.local_files_only is not None else low.local_files_only,
    )


def merge_assembly_config(
    high: AssemblyConfig | None,
    low: AssemblyConfig | None,
) -> AssemblyConfig | None:
    if high is None:
        return low
    if low is None:
        return high
    return AssemblyConfig(
        profiles=tuple([*high.profiles, *low.profiles]),
        chat=merge_provider_config(high.chat, low.chat),
        embedding=merge_provider_config(high.embedding, low.embedding),
        rerank=merge_provider_config(high.rerank, low.rerank),
        tokenizer=merge_tokenizer_config(high.tokenizer, low.tokenizer),
    )


def merge_assembly_overrides(
    high: AssemblyOverrides | None,
    low: AssemblyOverrides | None,
) -> AssemblyOverrides | None:
    if high is None:
        return low
    if low is None:
        return high
    return AssemblyOverrides(
        chat=merge_provider_config(high.chat, low.chat),
        embedding=merge_provider_config(high.embedding, low.embedding),
        rerank=merge_provider_config(high.rerank, low.rerank),
        tokenizer=merge_tokenizer_config(high.tokenizer, low.tokenizer),
        embedding_provider=(high.embedding_provider if high.embedding_provider is not None else low.embedding_provider),
        rerank_provider=(high.rerank_provider if high.rerank_provider is not None else low.rerank_provider),
    )


def build_provider(provider_config: ProviderConfig) -> object:
    kind = provider_config.provider_kind
    if kind == "openai-compatible":
        chat_model = provider_config.chat_model
        if not chat_model:
            return _UnavailableProvider(
                provider_name="openai-compatible",
                reason="openai-compatible provider requires chat_model",
            )
        base_url = provider_config.base_url
        if not base_url:
            return _UnavailableProvider(
                provider_name="openai-compatible",
                reason="openai-compatible provider requires base_url",
                chat_model_name=chat_model,
            )
        generator: object | None
        generator = OpenAICompatibleChatGenerator(
            model=chat_model,
            base_url=base_url,
            api_key=provider_config.api_key,
        )
        return _CompositeProvider(
            provider_name="openai-compatible",
            generator=generator,
        )
    if kind == "ollama":
        generator = (
            OllamaGenerator(
                base_url=provider_config.base_url or "http://localhost:11434",
                default_model=provider_config.chat_model,
            )
            if provider_config.chat_model
            else None
        )
        embedder: object | None
        embedder = (
            OllamaEmbedder(
                base_url=provider_config.base_url or "http://localhost:11434",
                default_model=provider_config.embedding_model,
                batch_size=provider_config.embedding_batch_size or 8,
            )
            if provider_config.embedding_model
            else None
        )
        return _CompositeProvider(
            provider_name="ollama",
            generator=generator,
            embedder=embedder,
        )
    if kind == "mlx-embedding":
        embedding_model_ref = first_non_blank(
            provider_config.embedding_model,
            provider_config.embedding_model_path,
        )
        if embedding_model_ref is None:
            return _UnavailableProvider(
                provider_name="mlx-embedding",
                reason="mlx-embedding provider requires embedding_model or embedding_model_path",
            )
        embedder = MLXEmbedder(
            model_name_or_path=embedding_model_ref,
            batch_size=provider_config.embedding_batch_size or 8,
        )
        return _CompositeProvider(
            provider_name="mlx-embedding",
            embedder=embedder,
        )
    if kind == "local-bge":
        embedder = None
        embedding_model_ref = first_non_blank(provider_config.embedding_model, provider_config.embedding_model_path)
        if embedding_model_ref is not None:
            normalized_embedding_model = embedding_model_ref.lower()
            if "bge-m3" in normalized_embedding_model:
                embedder = BgeM3Embedder(
                    model_name_or_path=provider_config.embedding_model or embedding_model_ref,
                    model_path=provider_config.embedding_model_path,
                    batch_size=provider_config.embedding_batch_size or 8,
                    device=provider_config.device,
                )
            else:
                embedder = HuggingFaceEmbedder(
                    model_name_or_path=embedding_model_ref,
                    batch_size=provider_config.embedding_batch_size or 8,
                    device=provider_config.device,
                    local_files_only=True,
                )
        reranker = None
        rerank_model_ref = first_non_blank(provider_config.rerank_model, provider_config.rerank_model_path)
        if rerank_model_ref is not None:
            reranker = FlagEmbeddingReranker(
                model_name_or_path=provider_config.rerank_model or rerank_model_ref,
                model_path=provider_config.rerank_model_path,
                batch_size=provider_config.rerank_batch_size or 8,
                devices=provider_config.device,
                local_files_only=True,
            )
        if embedder is None and reranker is None:
            return _UnavailableProvider(
                provider_name="local-bge",
                reason="local-bge provider requires embedding_model/model_path or rerank_model/model_path",
            )
        return _CompositeProvider(
            provider_name="local-bge",
            embedder=embedder,
            reranker=reranker,
        )
    if kind in {"fallback", "default-embedding"}:
        return FallbackEmbeddingRepo()
    raise RuntimeError(f"Unsupported provider_kind: {kind}")


def first_positive_int(*values: int | None) -> int:
    for value in values:
        if isinstance(value, int) and value > 0:
            return value
    return 1


def first_non_negative_int(*values: int | None) -> int:
    for value in values:
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def first_bool(*values: bool | None) -> bool:
    for value in values:
        if isinstance(value, bool):
            return value
    return False


__all__ = [
    "FallbackEmbeddingRepo",
    "build_provider",
    "compatibility_config_from_environment",
    "env_bool",
    "env_int",
    "first_bool",
    "first_env",
    "first_non_blank",
    "first_non_negative_int",
    "first_positive_int",
    "merge_assembly_config",
    "merge_assembly_overrides",
    "normalize_gemini_base_url",
]

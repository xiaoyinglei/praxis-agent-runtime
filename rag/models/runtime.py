from __future__ import annotations

from dataclasses import dataclass

from agent_runtime.modeling.config import ModelCapability, ModelRuntimeConfig, ModelSpec
from rag.models.catalog import ModelCatalog

_DISABLED_RERANKER_ALIASES = {"", "none", "null", "off", "false"}


@dataclass(frozen=True, slots=True)
class RuntimeOverrides:
    """CLI overrides for model selection.

    Each alias must exist in configs/models.yaml with the matching capability.
    Set reranker_model_alias to "none", "null", "off", or "false" to disable reranking.
    """

    model_alias: str | None = None
    embedding_model_alias: str | None = None
    reranker_model_alias: str | None = None


def resolve_runtime_config(
    overrides: RuntimeOverrides | None = None,
    *,
    catalog: ModelCatalog | None = None,
    catalog_path: str = "configs/models.yaml",
) -> ModelRuntimeConfig:
    """Single entry point for model selection.

    Priority: CLI override > YAML defaults.
    Business code should call this function instead of using ModelCatalog directly.
    """
    overrides = overrides or RuntimeOverrides()
    catalog = catalog or ModelCatalog.from_yaml(catalog_path)

    primary_model = _resolve_chat(overrides.model_alias, catalog)
    embedding_model = _resolve_embedding(overrides.embedding_model_alias, catalog)
    reranker_model = _resolve_reranker(overrides.reranker_model_alias, catalog)

    return ModelRuntimeConfig(
        primary_model=primary_model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        generation=catalog.generation,
        tokenizer=catalog.tokenizer,
        llm_stage_budgets=catalog.llm_stage_budgets,
    )


def _resolve_chat(alias: str | None, catalog: ModelCatalog) -> ModelSpec:
    spec = catalog.get_default_primary() if alias is None else catalog.get_model(alias)
    if spec.capability != ModelCapability.CHAT:
        raise ValueError(
            f"--model {alias!r} has capability {spec.capability.value!r}, "
            f"expected {ModelCapability.CHAT.value!r}. "
            f"Available chat models: {_list_aliases(catalog, ModelCapability.CHAT)}"
        )
    return spec


def _resolve_embedding(alias: str | None, catalog: ModelCatalog) -> ModelSpec:
    spec = catalog.get_default_embedding() if alias is None else catalog.get_model(alias)
    if spec.capability != ModelCapability.EMBEDDING:
        raise ValueError(
            f"--embedding-model {alias!r} has capability {spec.capability.value!r}, "
            f"expected {ModelCapability.EMBEDDING.value!r}. "
            f"Available embedding models: {_list_aliases(catalog, ModelCapability.EMBEDDING)}"
        )
    return spec


def _resolve_reranker(alias: str | None, catalog: ModelCatalog) -> ModelSpec | None:
    if alias is not None and alias.strip().lower() in _DISABLED_RERANKER_ALIASES:
        return None

    spec = catalog.get_default_reranker() if alias is None else catalog.get_model(alias)
    if spec is None:
        return None

    if spec.capability != ModelCapability.RERANKER:
        raise ValueError(
            f"--reranker-model {alias!r} has capability {spec.capability.value!r}, "
            f"expected {ModelCapability.RERANKER.value!r}. "
            f"Available reranker models: {_list_aliases(catalog, ModelCapability.RERANKER)}"
        )
    return spec


def _list_aliases(catalog: ModelCatalog, capability: ModelCapability) -> str:
    aliases = [m.alias for m in catalog.list_models(capability)]
    return ", ".join(aliases) or "<none>"

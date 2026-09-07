from __future__ import annotations

from dataclasses import dataclass

from agent_runtime.modeling.config import ModelCapability, ModelRuntimeConfig, ModelSpec
from rag.models.catalog import ModelCatalog

_DISABLED_RERANKER_IDS = {"", "none", "null", "off", "false"}


@dataclass(frozen=True, slots=True)
class RuntimeOverrides:
    """CLI overrides for model selection.

    Each model ID must exist in configs/models.yaml with the matching capability.
    Set reranker_model_id to "none", "null", "off", or "false" to disable reranking.
    """

    model_id: str | None = None
    embedding_model_id: str | None = None
    reranker_model_id: str | None = None


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

    primary_model = _resolve_chat(overrides.model_id, catalog)
    embedding_model = _resolve_embedding(overrides.embedding_model_id, catalog)
    reranker_model = _resolve_reranker(overrides.reranker_model_id, catalog)

    return ModelRuntimeConfig(
        primary_model=primary_model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        generation=catalog.generation,
        tokenizer=catalog.tokenizer,
        llm_stage_budgets=catalog.llm_stage_budgets,
    )


def _resolve_chat(model_id: str | None, catalog: ModelCatalog) -> ModelSpec:
    spec = catalog.get_default_primary() if model_id is None else catalog.get_model(model_id)
    if spec.capability != ModelCapability.CHAT:
        raise ValueError(
            f"--model {model_id!r} has capability {spec.capability.value!r}, "
            f"expected {ModelCapability.CHAT.value!r}. "
            f"Available chat models: {_list_model_ids(catalog, ModelCapability.CHAT)}"
        )
    return spec


def _resolve_embedding(model_id: str | None, catalog: ModelCatalog) -> ModelSpec:
    spec = catalog.get_default_embedding() if model_id is None else catalog.get_model(model_id)
    if spec.capability != ModelCapability.EMBEDDING:
        raise ValueError(
            f"--embedding-model {model_id!r} has capability {spec.capability.value!r}, "
            f"expected {ModelCapability.EMBEDDING.value!r}. "
            f"Available embedding models: {_list_model_ids(catalog, ModelCapability.EMBEDDING)}"
        )
    return spec


def _resolve_reranker(model_id: str | None, catalog: ModelCatalog) -> ModelSpec | None:
    if model_id is not None and model_id.strip().lower() in _DISABLED_RERANKER_IDS:
        return None

    spec = catalog.get_default_reranker() if model_id is None else catalog.get_model(model_id)
    if spec is None:
        return None

    if spec.capability != ModelCapability.RERANKER:
        raise ValueError(
            f"--reranker-model {model_id!r} has capability {spec.capability.value!r}, "
            f"expected {ModelCapability.RERANKER.value!r}. "
            f"Available reranker models: {_list_model_ids(catalog, ModelCapability.RERANKER)}"
        )
    return spec


def _list_model_ids(catalog: ModelCatalog, capability: ModelCapability) -> str:
    model_ids = [m.id for m in catalog.list_models(capability)]
    return ", ".join(model_ids) or "<none>"

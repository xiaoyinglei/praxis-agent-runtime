from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_runtime.modeling.config import (
    GenerationConfig,
    GenerationTaskConfig,
    ModelCapability,
    ModelSpec,
    TokenizerModelConfig,
)
from agent_runtime.modeling.contracts import LLMCallStage, LLMStageBudget, parse_llm_stage_budgets

_DEFAULT_CATALOG_PATH = "configs/models.yaml"


class ModelCatalog:
    """Loads model definitions from configs/models.yaml.

    This is the ONLY module that reads models.yaml.
    Business code MUST NOT construct a ModelCatalog directly —
    use resolve_runtime_config() from rag.models.runtime instead.
    """

    def __init__(
        self,
        models: dict[str, ModelSpec],
        defaults: dict[str, str],
        generation: GenerationConfig,
        tokenizer: TokenizerModelConfig,
        llm_stage_budgets: dict[LLMCallStage, LLMStageBudget],
    ) -> None:
        self._models = models
        self._defaults = defaults
        self._generation = generation
        self._tokenizer = tokenizer
        self._llm_stage_budgets = llm_stage_budgets

    @classmethod
    def from_yaml(cls, path: str = _DEFAULT_CATALOG_PATH) -> ModelCatalog:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Model catalog not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid model catalog: expected a dict, got {type(data).__name__}")
        return cls._parse(data)

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> ModelCatalog:
        raw_models = data.get("models")
        if not isinstance(raw_models, dict):
            raise ValueError("Model catalog must contain a 'models' section")
        raw_defaults = data.get("defaults")
        if not isinstance(raw_defaults, dict):
            raise ValueError("Model catalog must contain a 'defaults' section")
        raw_providers = data.get("providers")
        providers = raw_providers if isinstance(raw_providers, dict) else {}

        models: dict[str, ModelSpec] = {}
        for alias, entry in raw_models.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Model entry {alias!r} must be a dict, got {type(entry).__name__}")
            merged = _merge_provider_model_entry(entry, providers)
            models[alias] = ModelSpec(
                alias=alias,
                capability=ModelCapability(entry["capability"]),
                provider=str(merged.get("protocol") or merged.get("provider") or entry["provider"]),
                model=entry["model"],
                base_url=_optional_str(merged.get("base_url")),
                api_key_env=_optional_str(merged.get("api_key_env")),
                embedding_space=entry.get("embedding_space"),
                context_window_tokens=(
                    int(entry["context_window_tokens"]) if "context_window_tokens" in entry else None
                ),
            )

        defaults = {
            "primary_model": raw_defaults.get("primary_model", ""),
            "embedding_model": raw_defaults.get("embedding_model", ""),
            "reranker_model": raw_defaults.get("reranker_model", ""),
        }

        generation = cls._parse_generation(data.get("generation"), defaults)
        tokenizer = cls._parse_tokenizer(data.get("tokenizer"))
        return cls(
            models=models,
            defaults=defaults,
            generation=generation,
            tokenizer=tokenizer,
            llm_stage_budgets=parse_llm_stage_budgets(data.get("llm_budgets")),
        )

    @staticmethod
    def _parse_generation(raw: object, defaults: dict[str, str]) -> GenerationConfig:
        if not isinstance(raw, dict):
            return GenerationConfig()

        def _parse_task(name: str) -> GenerationTaskConfig:
            entry = raw.get(name)
            if not isinstance(entry, dict):
                return GenerationTaskConfig()
            return GenerationTaskConfig(
                model=entry.get("model"),
                max_tokens=int(entry["max_tokens"]) if "max_tokens" in entry else None,
                temperature=float(entry["temperature"]) if "temperature" in entry else None,
            )

        return GenerationConfig(
            summary=_parse_task("summary"),
            answer=_parse_task("answer"),
            planner=_parse_task("planner"),
            synthesize=_parse_task("synthesize"),
            factcheck=_parse_task("factcheck"),
        )

    @property
    def generation(self) -> GenerationConfig:
        return self._generation

    @property
    def tokenizer(self) -> TokenizerModelConfig:
        return self._tokenizer

    @property
    def llm_stage_budgets(self) -> dict[LLMCallStage, LLMStageBudget]:
        return {stage: budget.model_copy() for stage, budget in self._llm_stage_budgets.items()}

    def get_model(self, alias: str) -> ModelSpec:
        spec = self._models.get(alias)
        if spec is None:
            available = ", ".join(sorted(self._models))
            raise KeyError(f"Unknown model alias: {alias!r}. Available: {available}")
        return spec

    def get_default_primary(self) -> ModelSpec:
        alias = self._defaults["primary_model"]
        if not alias:
            raise ValueError("No default primary_model configured")
        return self.get_model(alias)

    def get_default_embedding(self) -> ModelSpec:
        alias = self._defaults["embedding_model"]
        if not alias:
            raise ValueError("No default embedding_model configured")
        return self.get_model(alias)

    def get_default_reranker(self) -> ModelSpec | None:
        alias = self._defaults["reranker_model"]
        if not alias:
            return None
        return self.get_model(alias)

    @staticmethod
    def _parse_tokenizer(raw: object) -> TokenizerModelConfig:
        if not isinstance(raw, dict):
            return TokenizerModelConfig()
        return TokenizerModelConfig(
            tokenizer_backend=raw.get("tokenizer_backend"),
            chunk_token_size=int(raw["chunk_token_size"]) if "chunk_token_size" in raw else None,
            chunk_overlap_tokens=int(raw["chunk_overlap_tokens"]) if "chunk_overlap_tokens" in raw else None,
            max_context_tokens=int(raw["max_context_tokens"]) if "max_context_tokens" in raw else None,
            prompt_reserved_tokens=int(raw["prompt_reserved_tokens"]) if "prompt_reserved_tokens" in raw else None,
            local_files_only=bool(raw["local_files_only"]) if "local_files_only" in raw else None,
        )

    def list_models(self, capability: ModelCapability | None = None) -> list[ModelSpec]:
        models = list(self._models.values())
        if capability is not None:
            models = [m for m in models if m.capability == capability]
        return sorted(models, key=lambda m: m.alias)


def _merge_provider_model_entry(
    entry: dict[str, Any],
    providers: dict[object, object],
) -> dict[str, object]:
    provider_ref = entry.get("provider")
    provider_entry = providers.get(provider_ref)
    provider = provider_entry if isinstance(provider_entry, dict) else {}
    return {
        "provider": provider_ref,
        "protocol": _first_present(entry, provider, "protocol"),
        "base_url": _first_present(entry, provider, "base_url"),
        "api_key_env": _first_present(entry, provider, "api_key_env"),
    }


def _first_present(
    primary: dict[str, object],
    fallback: dict[object, object],
    key: str,
) -> object | None:
    value = primary.get(key)
    if value is not None:
        return value
    fallback_value = fallback.get(key)
    return fallback_value if fallback_value is not None else None


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)

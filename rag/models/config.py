from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from rag.schema.llm import DEFAULT_LLM_STAGE_BUDGETS, LLMCallStage, LLMStageBudget


class ModelCapability(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    alias: str
    capability: ModelCapability
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    embedding_space: str | None = None
    context_window_tokens: int | None = None

    @property
    def requires_api_key(self) -> bool:
        return self.api_key_env is not None


@dataclass(frozen=True, slots=True)
class GenerationTaskConfig:
    """Per-task generation parameters.

    model       — model alias in models.yaml; None = fallback to defaults.primary_model
    max_tokens  — max completion tokens; None = not configured, consumer decides fallback
    temperature — None = don't pass to LLM (use model default)
    """

    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    summary: GenerationTaskConfig = field(default_factory=GenerationTaskConfig)
    answer: GenerationTaskConfig = field(default_factory=GenerationTaskConfig)
    planner: GenerationTaskConfig = field(default_factory=GenerationTaskConfig)
    synthesize: GenerationTaskConfig = field(default_factory=GenerationTaskConfig)
    factcheck: GenerationTaskConfig = field(default_factory=GenerationTaskConfig)


@dataclass(frozen=True, slots=True)
class TokenizerModelConfig:
    """Tokenizer / chunking parameters from models.yaml.

    All fields are optional — assembly layer provides defaults for any missing value.
    """

    tokenizer_backend: str | None = None
    chunk_token_size: int | None = None
    chunk_overlap_tokens: int | None = None
    max_context_tokens: int | None = None
    prompt_reserved_tokens: int | None = None
    local_files_only: bool | None = None


@dataclass(frozen=True, slots=True)
class ModelRuntimeConfig:
    primary_model: ModelSpec
    embedding_model: ModelSpec
    reranker_model: ModelSpec | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tokenizer: TokenizerModelConfig = field(default_factory=TokenizerModelConfig)
    llm_stage_budgets: dict[LLMCallStage, LLMStageBudget] = field(
        default_factory=lambda: {
            stage: budget.model_copy()
            for stage, budget in DEFAULT_LLM_STAGE_BUDGETS.items()
        }
    )

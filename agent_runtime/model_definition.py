"""Canonical, secret-free model execution definitions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from agent_runtime.core.generation_settings import ReasoningCapabilities
from agent_runtime.core.llm_config import (
    AgentModelsConfig,
    ModelProvider,
    ModelSpec,
    normalize_model_endpoint,
    validate_api_key_env_name,
    validate_http_url,
)
from agent_runtime.modeling.config import GenerationConfig, GenerationTaskConfig
from agent_runtime.modeling.contracts import ContextCapacityPolicy, LLMCallStage, LLMStageBudget


class GenerationTaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int | None = Field(gt=0, strict=True)
    temperature: float | None = Field(ge=0.0, le=2.0, allow_inf_nan=False)


class GenerationDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: GenerationTaskDefinition
    answer: GenerationTaskDefinition
    planner: GenerationTaskDefinition
    synthesize: GenerationTaskDefinition
    factcheck: GenerationTaskDefinition


class StageBudgetDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_input_tokens: int | None = Field(gt=0, strict=True)
    max_output_tokens: int = Field(gt=0, strict=True)
    safety_margin_tokens: int = Field(ge=0, strict=True)


class RuntimeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    health_url: str | None
    launch_command: tuple[str, ...]
    expected_model_contains: str | None
    startup_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    poll_interval_seconds: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("health_url")
    @classmethod
    def validate_health_url(cls, value: str | None) -> str | None:
        return validate_http_url(value, field_name="runtime.health_url")


class ThinkingOptionsDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["enabled", "disabled"]


class ChatTemplateOptionsDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enable_thinking: bool = Field(strict=True)


class ProviderOptionsDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    thinking: ThinkingOptionsDefinition | None = None
    chat_template_kwargs: ChatTemplateOptionsDefinition | None = None


class RequestDefaultsDefinition(BaseModel):
    """Safe request defaults; transport and authentication fields are absent."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        allow_inf_nan=False,
    )

    top_p: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )

    parallel_tool_calls: bool | None = Field(
        default=None,
        strict=True,
    )

    seed: int | None = Field(
        default=None,
        ge=-(2**63),
        le=2**63 - 1,
        strict=True,
    )

    provider_options: ProviderOptionsDefinition | None = None
    max_output_tokens: int | None = Field(default=None, gt=0, strict=True)
    reasoning_effort: str | None = Field(default=None, min_length=1)

    @model_serializer(mode="wrap")
    def serialize_defaults(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload = handler(self)
        for key in ("max_output_tokens", "reasoning_effort"):
            if getattr(self, key) is None:
                payload.pop(key, None)
        return dict(payload)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Resolved provider/model capabilities used by runtime execution."""

    context_window_tokens: int
    max_output_tokens: int | None
    supports_native_tools: bool
    supports_structured_output: bool
    reasoning: ReasoningCapabilities | None = None

    def __post_init__(self) -> None:
        if isinstance(self.context_window_tokens, bool) or self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")

        if self.max_output_tokens is not None:
            if isinstance(self.max_output_tokens, bool) or self.max_output_tokens <= 0:
                raise ValueError("max_output_tokens must be positive or None")

            if self.max_output_tokens > self.context_window_tokens:
                raise ValueError("max_output_tokens must not exceed context_window_tokens")


class ModelExecutionDefinition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    provider: ModelProvider
    provider_name: str | None
    protocol: str | None

    model_id: str = Field(min_length=1)
    tokenizer_model: str | None

    context_window_tokens: int = Field(
        gt=0,
        strict=True,
    )
    max_output_tokens: int | None = Field(
        default=None,
        gt=0,
        strict=True,
    )

    timeout_seconds: float = Field(
        gt=0,
        allow_inf_nan=False,
    )

    base_url: str | None
    api_key_env: str | None
    defaults: RequestDefaultsDefinition

    supports_tools: bool
    supports_structured_output: bool

    location: Literal["local", "cloud"] | None

    input_cost_per_1m: float | None = Field(
        ge=0,
        allow_inf_nan=False,
    )
    output_cost_per_1m: float | None = Field(
        ge=0,
        allow_inf_nan=False,
    )
    cache_read_cost_per_1m: float | None = Field(
        ge=0,
        allow_inf_nan=False,
    )
    cache_write_cost_per_1m: float | None = Field(
        ge=0,
        allow_inf_nan=False,
    )

    runtime: RuntimeDefinition | None
    generation: GenerationDefinition
    llm_stage_budgets: dict[str, StageBudgetDefinition]
    context_capacity_policy: ContextCapacityPolicy | None = None
    auto_compact_token_limit: int | None = Field(default=None, gt=0, strict=True)
    continuation_summary_role: Literal["context", "assistant"] | None = None
    reasoning: ReasoningCapabilities | None = None
    agent_output_policy: Literal["model-aware-v1"] | None = None

    @model_serializer(mode="wrap")
    def serialize_definition(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload = handler(self)
        # Legacy archives and authenticated bindings must retain exact bytes.
        if self.context_capacity_policy is None:
            payload.pop("context_capacity_policy", None)
        if self.auto_compact_token_limit is None:
            payload.pop("auto_compact_token_limit", None)
        if self.continuation_summary_role is None:
            payload.pop("continuation_summary_role", None)
        for key in ("reasoning", "agent_output_policy"):
            if getattr(self, key) is None:
                payload.pop(key, None)
        return dict(payload)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(
        cls,
        value: str | None,
    ) -> str | None:
        return validate_http_url(
            value,
            field_name="base_url",
        )

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_environment_name(
        cls,
        value: str | None,
    ) -> str | None:
        return validate_api_key_env_name(value)

    @model_validator(mode="after")
    def validate_capabilities(
        self,
    ) -> ModelExecutionDefinition:
        if self.max_output_tokens is not None and self.max_output_tokens > self.context_window_tokens:
            raise ValueError("max_output_tokens must not exceed context_window_tokens")

        (self.reasoning or ReasoningCapabilities()).validate_effort(self.defaults.reasoning_effort)

        _ = normalize_model_endpoint(
            provider=self.provider,
            base_url=self.base_url,
            location=self.location,
        )

        return self

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            context_window_tokens=(self.context_window_tokens),
            max_output_tokens=(self.max_output_tokens),
            reasoning=self.reasoning,
            supports_native_tools=(self.supports_tools),
            supports_structured_output=(self.supports_structured_output),
        )

    @property
    def definition_revision(self) -> str:
        return definition_revision(self)


def build_model_execution_definition(
    *,
    model_id: str,
    spec: ModelSpec,
    config: AgentModelsConfig,
) -> ModelExecutionDefinition:
    endpoint = normalize_model_endpoint(
        provider=spec.provider,
        base_url=spec.base_url,
        location=spec.location,
    )
    # The validated declaration owns shared field names. Freeze once here;
    # generation/budget policies and normalized identity belong to this boundary.
    payload = spec.model_dump(mode="python")
    payload.update(
        model_id=model_id,
        tokenizer_model=spec.tokenizer_model or model_id,
        base_url=endpoint.base_url,
        location=endpoint.location,
        defaults=RequestDefaultsDefinition.model_validate(deepcopy(spec.defaults)),
        generation=_generation_definition(config.generation),
        llm_stage_budgets={
            stage.value: budget.model_dump() for stage, budget in effective_stage_budgets(config=config).items()
        },
        context_capacity_policy=ContextCapacityPolicy(),
        agent_output_policy="model-aware-v1",
        auto_compact_token_limit=config.auto_compact_token_limit,
    )
    return ModelExecutionDefinition.model_validate(payload)


def canonical_definition_json(definition: ModelExecutionDefinition) -> bytes:
    """Return the exact canonical bytes used by archives and Turn bindings."""

    _validate_canonical_value(definition.model_dump(mode="python", exclude_none=False))
    payload = definition.model_dump(mode="json", by_alias=True, exclude_none=False)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def definition_revision(definition: ModelExecutionDefinition) -> str:
    return f"sha256:{hashlib.sha256(canonical_definition_json(definition)).hexdigest()}"


def _validate_canonical_value(value: object) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("model execution definition contains a non-finite JSON number")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("model execution definition contains a non-string JSON key")
        for item in value.values():
            _validate_canonical_value(item)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            _validate_canonical_value(item)
    else:
        raise ValueError(f"model execution definition contains a non-JSON value: {type(value).__name__}")


def effective_stage_budgets(
    *,
    config: AgentModelsConfig,
) -> dict[LLMCallStage, LLMStageBudget]:
    return {stage: budget.model_copy() for stage, budget in config.llm_stage_budgets.items()}


def _generation_definition(config: GenerationConfig) -> GenerationDefinition:
    def task(value: GenerationTaskConfig) -> GenerationTaskDefinition:
        return GenerationTaskDefinition(
            max_tokens=value.max_tokens,
            temperature=value.temperature,
        )

    return GenerationDefinition(
        summary=task(config.summary),
        answer=task(config.answer),
        planner=task(config.planner),
        synthesize=task(config.synthesize),
        factcheck=task(config.factcheck),
    )


__all__ = [
    "ModelCapabilities",
    "ModelExecutionDefinition",
    "build_model_execution_definition",
    "canonical_definition_json",
    "definition_revision",
    "effective_stage_budgets",
]

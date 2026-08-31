"""Canonical, secret-free model execution definitions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_runtime.core.llm_config import (
    AgentModelsConfig,
    ModelProvider,
    ModelSpec,
    normalize_model_endpoint,
    validate_api_key_env_name,
    validate_http_url,
)
from agent_runtime.modeling.config import GenerationConfig, GenerationTaskConfig
from agent_runtime.modeling.contracts import LLMCallStage, LLMStageBudget


class GenerationTaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str | None
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

    max_input_tokens: int = Field(gt=0, strict=True)
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


class ProviderOptionsDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    thinking: ThinkingOptionsDefinition | None = None


class RequestDefaultsDefinition(BaseModel):
    """Safe request defaults; transport and authentication fields are absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = Field(default=None, ge=0.0, le=2.0, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0, allow_inf_nan=False)
    parallel_tool_calls: bool | None = Field(default=None, strict=True)
    seed: int | None = Field(default=None, ge=-(2**63), le=2**63 - 1, strict=True)
    provider_options: ProviderOptionsDefinition | None = None


class ModelExecutionDefinition(BaseModel):
    """Every request-affecting value needed to execute one selected model.

    Alias, catalog origin, registry/file revisions, policy state, and resolved
    credential values deliberately do not belong to this content-addressed
    definition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ModelProvider
    provider_name: str | None
    protocol: str | None
    model: str = Field(min_length=1)
    tokenizer_model: str | None
    max_tokens: int = Field(gt=0, strict=True)
    timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    base_url: str | None
    api_key_env: str | None
    defaults: RequestDefaultsDefinition
    context_window_tokens: int = Field(gt=0, strict=True)
    request_context_tokens: int | None = Field(gt=0, strict=True)
    supports_tools: bool
    supports_structured_output: bool
    location: Literal["local", "cloud"] | None
    input_cost_per_1m: float | None = Field(ge=0, allow_inf_nan=False)
    output_cost_per_1m: float | None = Field(ge=0, allow_inf_nan=False)
    cache_read_cost_per_1m: float | None = Field(ge=0, allow_inf_nan=False)
    cache_write_cost_per_1m: float | None = Field(ge=0, allow_inf_nan=False)
    runtime: RuntimeDefinition | None
    generation: GenerationDefinition
    llm_stage_budgets: dict[str, StageBudgetDefinition]

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        return validate_http_url(value, field_name="base_url")

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_environment_name(cls, value: str | None) -> str | None:
        return validate_api_key_env_name(value)

    @model_validator(mode="after")
    def validate_normalized_execution_values(self) -> ModelExecutionDefinition:
        endpoint = normalize_model_endpoint(
            provider=self.provider,
            base_url=self.base_url,
            location=self.location,
        )
        if self.base_url != endpoint.base_url or self.location != endpoint.location:
            raise ValueError("model execution endpoint and location must be normalized")
        request_limit = self.request_context_tokens or self.context_window_tokens
        if request_limit > self.context_window_tokens:
            raise ValueError("request_context_tokens must not exceed context_window_tokens")
        if self.max_tokens > request_limit:
            raise ValueError("max_tokens must not exceed the effective request context limit")
        return self

    @property
    def definition_revision(self) -> str:
        return definition_revision(self)


def build_model_execution_definition(
    *,
    spec: ModelSpec,
    config: AgentModelsConfig,
) -> ModelExecutionDefinition:
    """Normalize one internal declaration and its catalog-wide runtime policy."""

    runtime = spec.runtime
    endpoint = normalize_model_endpoint(
        provider=spec.provider,
        base_url=spec.base_url,
        location=spec.location,
    )
    return ModelExecutionDefinition(
        provider=spec.provider,
        provider_name=spec.provider_name,
        protocol=spec.protocol,
        model=spec.model,
        tokenizer_model=spec.tokenizer_model or spec.model,
        max_tokens=spec.max_tokens,
        timeout_seconds=spec.timeout_seconds,
        base_url=endpoint.base_url,
        api_key_env=spec.api_key_env,
        defaults=RequestDefaultsDefinition.model_validate(deepcopy(spec.defaults)),
        context_window_tokens=spec.context_window_tokens,
        request_context_tokens=spec.request_context_tokens,
        supports_tools=spec.supports_tools,
        supports_structured_output=spec.supports_structured_output,
        location=endpoint.location,
        input_cost_per_1m=spec.input_cost_per_1m,
        output_cost_per_1m=spec.output_cost_per_1m,
        cache_read_cost_per_1m=spec.cache_read_cost_per_1m,
        cache_write_cost_per_1m=spec.cache_write_cost_per_1m,
        runtime=(
            RuntimeDefinition(
                health_url=runtime.health_url,
                launch_command=runtime.launch_command,
                expected_model_contains=runtime.expected_model_contains,
                startup_timeout_seconds=runtime.startup_timeout_seconds,
                poll_interval_seconds=runtime.poll_interval_seconds,
            )
            if runtime is not None
            else None
        ),
        generation=_generation_definition(config.generation),
        llm_stage_budgets={
            stage.value: StageBudgetDefinition.model_validate(budget.model_dump())
            for stage, budget in _effective_stage_budgets(config, spec).items()
        },
    )


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
        raise ValueError(
            f"model execution definition contains a non-JSON value: {type(value).__name__}"
        )


def effective_stage_budgets(
    *,
    config: AgentModelsConfig,
    spec: ModelSpec,
) -> dict[LLMCallStage, LLMStageBudget]:
    return _effective_stage_budgets(config, spec)


def _effective_stage_budgets(
    config: AgentModelsConfig,
    spec: ModelSpec,
) -> dict[LLMCallStage, LLMStageBudget]:
    budgets = {stage: budget.model_copy() for stage, budget in config.llm_stage_budgets.items()}
    tool_decision = budgets[LLMCallStage.TOOL_DECISION]
    if spec.max_tokens > tool_decision.max_output_tokens:
        budgets[LLMCallStage.TOOL_DECISION] = tool_decision.model_copy(
            update={"max_output_tokens": spec.max_tokens}
        )
    return budgets


def _generation_definition(config: GenerationConfig) -> GenerationDefinition:
    def task(value: GenerationTaskConfig) -> GenerationTaskDefinition:
        return GenerationTaskDefinition(
            model=value.model,
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
    "ModelExecutionDefinition",
    "build_model_execution_definition",
    "canonical_definition_json",
    "definition_revision",
    "effective_stage_budgets",
]

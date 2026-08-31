from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_runtime.modeling.config import GenerationConfig
from agent_runtime.modeling.contracts import (
    DEFAULT_LLM_STAGE_BUDGETS,
    LLMCallStage,
    LLMStageBudget,
)

_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_http_url(value: str | None, *, field_name: str) -> str | None:
    """Validate a secret-free absolute endpoint suitable for persistence."""

    if value is None:
        return None
    if any(character.isspace() for character in value):
        raise ValueError(f"{field_name} must not contain whitespace")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{field_name} must not contain ASCII control characters")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.hostname is None:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    try:
        _ = parts.port
    except ValueError as error:
        raise ValueError(f"{field_name} contains an invalid port") from error
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ValueError(
            f"{field_name} must not contain credentials, query parameters, or fragments"
        )
    return value


def validate_api_key_env_name(value: str | None) -> str | None:
    if value is not None and _ENVIRONMENT_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError("api_key_env must be an environment-variable name")
    return value


class ModelProvider(StrEnum):
    """当前 Agent 模型配置真正支持的 provider。"""

    MLX = "mlx"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class ModelGenerationDefaults(BaseModel):
    """Typed user-facing generation overrides supported by the runtime."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0, allow_inf_nan=False)
    parallel_tool_calls: bool | None = Field(default=None, strict=True)
    seed: int | None = Field(default=None, ge=-(2**63), le=2**63 - 1, strict=True)

    @model_validator(mode="before")
    @classmethod
    def reject_explicit_nulls(cls, value: object) -> object:
        if isinstance(value, dict):
            null_keys = sorted(key for key, item in value.items() if item is None)
            if null_keys:
                joined = ", ".join(str(key) for key in null_keys)
                raise ValueError(f"generation defaults do not accept null: {joined}")
        return value


class ModelRuntimeConfig(BaseModel):
    health_url: str | None = None
    launch_command: tuple[str, ...] = ()
    expected_model_contains: str | None = None
    startup_timeout_seconds: float = Field(default=60.0, gt=0)
    poll_interval_seconds: float = Field(default=1.0, gt=0)


class ModelSpec(BaseModel):
    """单个模型声明：只允许填写当前已实现 provider 支持的模型。"""

    provider: ModelProvider
    model: str
    tokenizer_model: str | None = Field(default=None, min_length=1)
    provider_name: str | None = None
    protocol: str | None = None
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    base_url: str | None = None
    api_key_env: str | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    context_window_tokens: int = Field(default=32_768, gt=0)
    request_context_tokens: int | None = Field(default=None, gt=0)
    supports_tools: bool = True
    supports_structured_output: bool = True
    location: Literal["local", "cloud"] | None = None
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    cache_read_cost_per_1m: float | None = Field(default=None, ge=0)
    cache_write_cost_per_1m: float | None = Field(default=None, ge=0)
    runtime: ModelRuntimeConfig | None = None

    @model_validator(mode="after")
    def validate_request_context_limit(self) -> ModelSpec:
        if self.request_context_tokens is not None and self.request_context_tokens > self.context_window_tokens:
            raise ValueError("request_context_tokens must not exceed context_window_tokens")
        return self


class AgentModelsConfig(BaseModel):
    """Agent 模型配置：只声明可用模型，不绑定具体运行节点角色。"""

    version: int = 1
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    default_model: str
    fallback_model: str | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    llm_stage_budgets: dict[LLMCallStage, LLMStageBudget] = Field(
        default_factory=lambda: {stage: budget.model_copy() for stage, budget in DEFAULT_LLM_STAGE_BUDGETS.items()}
    )

    @model_validator(mode="after")
    def validate_model_refs(self) -> AgentModelsConfig:
        if not self.models:
            raise ValueError("models must not be empty")

        if self.default_model not in self.models:
            raise ValueError(f"default_model not found in models: {self.default_model}")

        if self.fallback_model and self.fallback_model not in self.models:
            raise ValueError(f"fallback_model not found in models: {self.fallback_model}")

        return self

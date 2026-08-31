from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


@dataclass(frozen=True, slots=True)
class NormalizedModelEndpoint:
    base_url: str
    location: Literal["local", "cloud"]


def normalize_model_endpoint(
    *,
    provider: ModelProvider,
    base_url: str | None,
    location: Literal["local", "cloud"] | None,
) -> NormalizedModelEndpoint:
    """Return the one endpoint/location pair used by validation and execution."""

    effective_url = base_url
    if effective_url is None:
        effective_url = (
            "http://localhost:11434"
            if provider is ModelProvider.OLLAMA
            else "http://127.0.0.1:8080/v1"
        )
    validated_url = validate_http_url(effective_url, field_name="base_url")
    assert validated_url is not None
    host = urlsplit(validated_url).hostname
    assert host is not None
    endpoint_location: Literal["local", "cloud"] = (
        "local" if _is_loopback_host(host) else "cloud"
    )
    if provider in {ModelProvider.MLX, ModelProvider.OLLAMA} and endpoint_location != "local":
        raise ValueError(f"provider {provider.value!r} requires a local loopback endpoint")
    if location is not None and location != endpoint_location:
        raise ValueError(
            f"location={location!r} conflicts with endpoint location {endpoint_location!r}"
        )
    return NormalizedModelEndpoint(base_url=validated_url, location=endpoint_location)


def _is_loopback_host(host: str) -> bool:
    normalized = host.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        pass
    try:
        packed = socket.inet_aton(normalized)
    except OSError:
        return False
    return ipaddress.ip_address(packed).is_loopback


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
    model_config = ConfigDict(extra="forbid")

    health_url: str | None = None
    launch_command: tuple[str, ...] = ()
    expected_model_contains: str | None = None
    startup_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    poll_interval_seconds: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @field_validator("health_url")
    @classmethod
    def validate_health_url(cls, value: str | None) -> str | None:
        return validate_http_url(value, field_name="runtime.health_url")

    @field_validator("launch_command", mode="before")
    @classmethod
    def reject_shell_command(cls, value: object) -> object:
        if isinstance(value, str):
            raise ValueError("runtime.launch_command must be an argv array")
        return value

    @field_validator("launch_command")
    @classmethod
    def validate_launch_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not part or "\x00" in part for part in value):
            raise ValueError("runtime.launch_command entries must be non-empty and contain no NUL")
        return value


class ModelSpec(BaseModel):
    """单个模型声明：只允许填写当前已实现 provider 支持的模型。"""

    model_config = ConfigDict(extra="forbid")

    provider: ModelProvider
    model: str = Field(min_length=1)
    tokenizer_model: str | None = Field(default=None, min_length=1)
    provider_name: str | None = None
    protocol: str | None = None
    max_tokens: int = Field(default=2048, gt=0, strict=True)
    timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    base_url: str | None = None
    api_key_env: str | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    context_window_tokens: int = Field(default=32_768, gt=0, strict=True)
    request_context_tokens: int | None = Field(default=None, gt=0, strict=True)
    supports_tools: bool = Field(default=True, strict=True)
    supports_structured_output: bool = Field(default=True, strict=True)
    location: Literal["local", "cloud"] | None = None
    input_cost_per_1m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_cost_per_1m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache_read_cost_per_1m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache_write_cost_per_1m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    runtime: ModelRuntimeConfig | None = None

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        return validate_http_url(value, field_name="base_url")

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_environment_name(cls, value: str | None) -> str | None:
        return validate_api_key_env_name(value)

    @model_validator(mode="after")
    def validate_request_context_limit(self) -> ModelSpec:
        if self.request_context_tokens is not None and self.request_context_tokens > self.context_window_tokens:
            raise ValueError("request_context_tokens must not exceed context_window_tokens")
        request_limit = self.request_context_tokens or self.context_window_tokens
        if self.max_tokens > request_limit:
            raise ValueError("max_tokens must not exceed the effective request context limit")
        _ = normalize_model_endpoint(
            provider=self.provider,
            base_url=self.base_url,
            location=self.location,
        )
        return self


class AgentModelsConfig(BaseModel):
    """Agent 模型配置：只声明可用模型，不绑定具体运行节点角色。"""

    model_config = ConfigDict(extra="forbid")

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

        missing_stages = set(LLMCallStage).difference(self.llm_stage_budgets)
        extra_stages = set(self.llm_stage_budgets).difference(LLMCallStage)
        if missing_stages or extra_stages:
            raise ValueError("llm_stage_budgets must define every supported LLM stage exactly once")

        return self

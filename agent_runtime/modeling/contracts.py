from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    TypeAdapter,
    computed_field,
    field_validator,
)

MAX_RAW_PROVIDER_USAGE_BYTES = 16_384
_RAW_PROVIDER_USAGE_ADAPTER = TypeAdapter(dict[str, JsonValue])

type LLMUsageSource = Literal["provider", "tokenizer_estimate"]


class LLMCallStage(StrEnum):
    AGENT_STEP = "agent_step"
    GOAL_CONTRACT = "goal_contract"
    RETRIEVAL_HINT = "retrieval_hint"
    TOOL_DECISION = "tool_decision"
    RETRIEVAL_SUMMARY = "retrieval_summary"
    LLM_SUMMARIZE = "llm_summarize"
    LLM_COMPARE = "llm_compare"
    LLM_GENERATE = "llm_generate"
    RAG_ANSWER = "rag_answer"
    FINAL_SYNTHESIS = "final_synthesis"


class LLMStageBudget(BaseModel):
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    safety_margin_tokens: int = Field(default=512, ge=0)


class LLMUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    source: LLMUsageSource
    logical_input_tokens: int | None = Field(default=None, ge=0)
    uncached_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_write_input_tokens: int | None = Field(default=None, ge=0)
    usage_source: LLMUsageSource | None = None
    raw_provider_usage: dict[str, JsonValue] | None = None

    @field_validator("raw_provider_usage", mode="before")
    @classmethod
    def normalize_raw_provider_usage(cls, value: object) -> object:
        return _bounded_raw_provider_usage(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def normalize_llm_usage(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_read_input_tokens: int | None = None,
    cache_write_input_tokens: int | None = None,
    input_tokens_include_cache: bool,
    usage_source: LLMUsageSource,
    raw_provider_usage: Mapping[str, object] | None = None,
) -> LLMUsage:
    """Normalize declared provider accounting without inferring cache hits."""

    input_count = _optional_token_count(input_tokens, field_name="input_tokens")
    output_count = _optional_token_count(output_tokens, field_name="output_tokens")
    cache_read = _optional_token_count(
        cache_read_input_tokens,
        field_name="cache_read_input_tokens",
    )
    cache_write = _optional_token_count(
        cache_write_input_tokens,
        field_name="cache_write_input_tokens",
    )
    if type(input_tokens_include_cache) is not bool:
        raise TypeError("input_tokens_include_cache must be a bool")
    if usage_source == "tokenizer_estimate" and (
        cache_read is not None or cache_write is not None or raw_provider_usage is not None
    ):
        raise ValueError("tokenizer estimates cannot report provider cache usage")

    logical_input: int | None
    uncached_input: int | None
    if input_count is None:
        logical_input = None
        uncached_input = None
    elif input_tokens_include_cache:
        logical_input = input_count
        if cache_read is None:
            uncached_input = None
        else:
            if cache_read > input_count:
                raise ValueError("cache_read_input_tokens exceeds input_tokens")
            observed_cache = cache_read + (cache_write or 0)
            if observed_cache > input_count:
                raise ValueError("reported cache token counts exceed input_tokens")
            uncached_input = input_count - observed_cache
    else:
        uncached_input = input_count
        if cache_read is None or cache_write is None:
            logical_input = None
        else:
            logical_input = input_count + cache_read + cache_write

    legacy_input = logical_input if logical_input is not None else (input_count or 0)
    return LLMUsage(
        input_tokens=legacy_input,
        output_tokens=output_count or 0,
        cached_input_tokens=cache_read or 0,
        source=usage_source,
        logical_input_tokens=logical_input,
        uncached_input_tokens=uncached_input,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        usage_source=usage_source,
        raw_provider_usage=cast(
            dict[str, JsonValue] | None,
            raw_provider_usage,
        ),
    )


def _optional_token_count(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer or None")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _bounded_raw_provider_usage(value: object) -> object:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("raw_provider_usage must be a mapping or None")
    normalized = _RAW_PROVIDER_USAGE_ADAPTER.validate_python(dict(value))
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) <= MAX_RAW_PROVIDER_USAGE_BYTES:
        return json.loads(encoded)
    return {
        "_truncated": True,
        "_original_bytes": len(encoded),
        "_sha256": hashlib.sha256(encoded).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class LLMProviderResult[T]:
    value: T
    usage: LLMUsage | None = None


@dataclass(frozen=True, slots=True)
class LLMCallResult[T]:
    value: T
    usage: LLMUsage
    stage: LLMCallStage


DEFAULT_LLM_STAGE_BUDGETS: dict[LLMCallStage, LLMStageBudget] = {
    LLMCallStage.AGENT_STEP: LLMStageBudget(
        max_input_tokens=64_000,
        max_output_tokens=32_768,
    ),
    LLMCallStage.GOAL_CONTRACT: LLMStageBudget(
        max_input_tokens=3_000,
        max_output_tokens=512,
        safety_margin_tokens=256,
    ),
    LLMCallStage.RETRIEVAL_HINT: LLMStageBudget(
        max_input_tokens=2_000,
        max_output_tokens=256,
    ),
    LLMCallStage.TOOL_DECISION: LLMStageBudget(
        max_input_tokens=32_000,
        max_output_tokens=4_096,
    ),
    LLMCallStage.RETRIEVAL_SUMMARY: LLMStageBudget(
        max_input_tokens=6_000,
        max_output_tokens=1_024,
    ),
    LLMCallStage.LLM_SUMMARIZE: LLMStageBudget(
        max_input_tokens=16_000,
        max_output_tokens=2_048,
    ),
    LLMCallStage.LLM_COMPARE: LLMStageBudget(
        max_input_tokens=20_000,
        max_output_tokens=3_072,
    ),
    LLMCallStage.LLM_GENERATE: LLMStageBudget(
        max_input_tokens=16_000,
        max_output_tokens=3_072,
    ),
    LLMCallStage.RAG_ANSWER: LLMStageBudget(
        max_input_tokens=16_000,
        max_output_tokens=4_096,
    ),
    LLMCallStage.FINAL_SYNTHESIS: LLMStageBudget(
        max_input_tokens=24_000,
        max_output_tokens=4_096,
    ),
}


def parse_llm_stage_budgets(
    raw: object,
) -> dict[LLMCallStage, LLMStageBudget]:
    parsed = {
        stage: budget.model_copy()
        for stage, budget in DEFAULT_LLM_STAGE_BUDGETS.items()
    }
    if not isinstance(raw, Mapping):
        return parsed
    for raw_stage, raw_budget in raw.items():
        try:
            stage = LLMCallStage(str(raw_stage))
        except ValueError:
            continue
        parsed[stage] = LLMStageBudget.model_validate(raw_budget)
    return parsed


__all__ = [
    "DEFAULT_LLM_STAGE_BUDGETS",
    "LLMCallResult",
    "LLMCallStage",
    "LLMProviderResult",
    "LLMStageBudget",
    "LLMUsage",
    "LLMUsageSource",
    "MAX_RAW_PROVIDER_USAGE_BYTES",
    "normalize_llm_usage",
    "parse_llm_stage_budgets",
]

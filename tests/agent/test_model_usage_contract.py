from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent_runtime.core.llm_config import ModelProvider, ModelSpec
from agent_runtime.core.model_request import (
    ModelSettings,
    bind_model_call_record,
    build_model_request,
    build_stable_context,
    model_call_record_payload,
)
from agent_runtime.modeling.contracts import (
    MAX_RAW_PROVIDER_USAGE_BYTES,
    LLMUsage,
    normalize_llm_usage,
)


@pytest.mark.parametrize("response_shape", ["mapping", "sdk_object"])
def test_openai_compatible_usage_keeps_reported_prompt_cache_hits(response_shape) -> None:
    from types import SimpleNamespace

    from agent_runtime.modeling.openai_wire import parse_openai_usage

    raw = {"prompt_tokens": 1000, "completion_tokens": 200,
           "prompt_cache_hit_tokens": 900, "prompt_cache_miss_tokens": 100}
    response = {"usage": raw} if response_shape == "mapping" else SimpleNamespace(usage=SimpleNamespace(**raw))
    usage = parse_openai_usage(response)
    assert usage is not None
    assert usage.input_tokens == 1000
    assert usage.cache_read_input_tokens == 900
    assert usage.uncached_input_tokens == 100
    assert usage.total_tokens == 1200


def test_legacy_usage_construction_remains_backward_compatible() -> None:
    usage = LLMUsage(
        input_tokens=12,
        output_tokens=4,
        cached_input_tokens=3,
        reasoning_tokens=1,
        source="provider",
    )

    assert usage.total_tokens == 16
    assert usage.logical_input_tokens is None
    assert usage.uncached_input_tokens is None
    assert usage.cache_read_input_tokens is None
    assert usage.cache_write_input_tokens is None
    assert usage.usage_source is None
    assert usage.raw_provider_usage is None


def test_future_provider_can_normalize_separate_cache_accounting() -> None:
    usage = normalize_llm_usage(
        input_tokens=11,
        output_tokens=4,
        cache_read_input_tokens=7,
        cache_write_input_tokens=5,
        input_tokens_include_cache=False,
        usage_source="provider",
        raw_provider_usage={
            "input_tokens": 11,
            "cache_read_input_tokens": 7,
            "cache_write_input_tokens": 5,
            "output_tokens": 4,
        },
    )

    assert usage.logical_input_tokens == 23
    assert usage.uncached_input_tokens == 11
    assert usage.cache_read_input_tokens == 7
    assert usage.cache_write_input_tokens == 5
    assert usage.output_tokens == 4
    assert usage.usage_source == "provider"
    assert usage.input_tokens == 23
    assert usage.cached_input_tokens == 7
    assert usage.source == "provider"


def test_normalized_usage_snapshots_and_bounds_raw_provider_usage() -> None:
    raw: dict[str, dict[str, int]] = {"details": {"cached_tokens": 9}}
    usage = normalize_llm_usage(
        input_tokens=20,
        output_tokens=3,
        cache_read_input_tokens=9,
        input_tokens_include_cache=True,
        usage_source="provider",
        raw_provider_usage=raw,
    )
    raw["details"]["cached_tokens"] = 999

    assert usage.raw_provider_usage == {"details": {"cached_tokens": 9}}

    bounded = normalize_llm_usage(
        input_tokens=1,
        output_tokens=1,
        input_tokens_include_cache=True,
        usage_source="provider",
        raw_provider_usage={"oversized": "x" * (MAX_RAW_PROVIDER_USAGE_BYTES * 2)},
    )
    encoded = json.dumps(
        bounded.raw_provider_usage,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert len(encoded) <= MAX_RAW_PROVIDER_USAGE_BYTES
    assert bounded.raw_provider_usage is not None
    assert bounded.raw_provider_usage["_truncated"] is True
    assert isinstance(bounded.raw_provider_usage["_sha256"], str)


def test_invalid_or_inconsistent_token_accounting_fails_closed() -> None:
    with pytest.raises(ValueError, match="cache_read_input_tokens exceeds input_tokens"):
        normalize_llm_usage(
            input_tokens=3,
            output_tokens=1,
            cache_read_input_tokens=4,
            input_tokens_include_cache=True,
            usage_source="provider",
        )

    with pytest.raises((TypeError, ValidationError, ValueError)):
        normalize_llm_usage(
            input_tokens=-1,
            output_tokens=1,
            input_tokens_include_cache=True,
            usage_source="provider",
        )


def test_model_call_record_binds_usage_to_exact_request_evidence() -> None:
    request = build_model_request(
        request_id="req-usage-evidence",
        context=build_stable_context(
            instructions=("Be precise.",),
            initial_user_task="Answer once.",
        ),
        selected_tools=(),
        settings=ModelSettings(model="test-model"),
    )
    usage = normalize_llm_usage(
        input_tokens=10,
        output_tokens=2,
        cache_read_input_tokens=4,
        input_tokens_include_cache=True,
        usage_source="provider",
        raw_provider_usage={"prompt_tokens": 10, "completion_tokens": 2},
    )

    record = bind_model_call_record(
        request=request,
        provider_wire_hash="wire_exact_hash",
        usage=usage,
    )
    assert record.usage is not usage
    assert usage.raw_provider_usage is not None
    usage.raw_provider_usage["prompt_tokens"] = 999
    payload = model_call_record_payload(record)

    assert record.request_id == request.request_id
    assert record.prompt_revision == request.prompt_revision
    assert record.toolset_revision == request.toolset_revision
    assert record.provider_wire_hash == "wire_exact_hash"
    assert record.usage.raw_provider_usage == {
        "completion_tokens": 2,
        "prompt_tokens": 10,
    }
    assert payload["request_id"] == "req-usage-evidence"
    usage_payload = payload["usage"]
    assert isinstance(usage_payload, dict)
    assert usage_payload["cache_read_input_tokens"] == 4
    assert set(usage_payload) == {
        "logical_input_tokens",
        "uncached_input_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "usage_source",
        "raw_provider_usage",
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)

    with pytest.raises(ValueError, match="normalized usage"):
        bind_model_call_record(
            request=request,
            provider_wire_hash="wire_legacy_usage",
            usage=LLMUsage(input_tokens=10, output_tokens=2, source="provider"),
        )


def test_cache_pricing_is_optional_and_never_invented() -> None:
    # Synthetic minimum: model capability is irrelevant to these pricing assertions.
    synthetic_context_window = 1
    default = ModelSpec(
        provider=ModelProvider.MLX,
        context_window_tokens=synthetic_context_window,
    )
    priced = ModelSpec(
        provider=ModelProvider.OPENAI_COMPATIBLE,
        context_window_tokens=synthetic_context_window,
        cache_read_cost_per_1m=0.1,
        cache_write_cost_per_1m=0.25,
    )

    assert default.cache_read_cost_per_1m is None
    assert default.cache_write_cost_per_1m is None
    assert priced.cache_read_cost_per_1m == 0.1
    assert priced.cache_write_cost_per_1m == 0.25

    with pytest.raises(ValidationError):
        ModelSpec(
            provider=ModelProvider.OPENAI_COMPATIBLE,
            context_window_tokens=synthetic_context_window,
            cache_read_cost_per_1m=-0.1,
        )

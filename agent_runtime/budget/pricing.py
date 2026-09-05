from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

_MICROS_PER_DOLLAR = 1_000_000
_TOKENS_PER_MILLION = 1_000_000
_PRICING_KEYS = ("input", "output", "cache_read", "cache_write")


class PricingUnavailableError(RuntimeError):
    """A cost-limited execution cannot be priced before provider dispatch."""


def dollars_per_1m_to_micros(value: float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("price per 1M tokens must be a non-negative number or None")
    try:
        micros = (Decimal(str(value)) * _MICROS_PER_DOLLAR).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    except InvalidOperation as exc:
        raise ValueError("price per 1M tokens is invalid") from exc
    return int(micros)


def pricing_micros_per_1m(
    *,
    input_cost_per_1m: float | None,
    output_cost_per_1m: float | None,
    cache_read_cost_per_1m: float | None,
    cache_write_cost_per_1m: float | None,
) -> dict[str, int | None]:
    return {
        "input": dollars_per_1m_to_micros(input_cost_per_1m),
        "output": dollars_per_1m_to_micros(output_cost_per_1m),
        "cache_read": dollars_per_1m_to_micros(cache_read_cost_per_1m),
        "cache_write": dollars_per_1m_to_micros(cache_write_cost_per_1m),
    }


def pricing_revision(
    *, provider: str, model: str, pricing: Mapping[str, int | None]
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "micros_per_1m": _normalized_pricing(pricing),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return "pricing:sha256:" + hashlib.sha256(payload).hexdigest()


def estimated_model_cost_micros(
    *,
    input_tokens: int,
    max_output_tokens: int,
    pricing: Mapping[str, int | None],
) -> int | None:
    _count("input_tokens", input_tokens)
    _count("max_output_tokens", max_output_tokens)
    rates = _normalized_pricing(pricing)
    if input_tokens and rates["input"] is None:
        return None
    if max_output_tokens and rates["output"] is None:
        return None
    return _ceil_cost(
        (
            (input_tokens, rates["input"]),
            (max_output_tokens, rates["output"]),
        )
    )


def actual_model_cost_micros(
    usage: Mapping[str, Any],
    *,
    pricing: Mapping[str, int | None],
) -> int | None:
    rates = _normalized_pricing(pricing)
    input_tokens = _usage_count(usage, "input_tokens")
    output_tokens = _usage_count(usage, "output_tokens")
    cache_read = _usage_count(
        usage, "cache_read_input_tokens", "cached_input_tokens"
    )
    cache_write = _usage_count(
        usage, "cache_write_input_tokens", "cache_write_tokens"
    )
    uncached = _optional_usage_count(usage, "uncached_input_tokens")
    if uncached is None:
        logical = _optional_usage_count(usage, "logical_input_tokens")
        if logical is None:
            logical = input_tokens
        uncached = max(logical - cache_read - cache_write, 0)

    input_rate = rates["input"]
    output_rate = rates["output"]
    cache_read_rate = rates["cache_read"] if rates["cache_read"] is not None else input_rate
    cache_write_rate = rates["cache_write"] if rates["cache_write"] is not None else input_rate

    components = (
        (uncached, input_rate),
        (cache_read, cache_read_rate),
        (cache_write, cache_write_rate),
        (output_tokens, output_rate),
    )
    if any(tokens and rate is None for tokens, rate in components):
        return None
    return _ceil_cost(components)


def _normalized_pricing(
    pricing: Mapping[str, int | None],
) -> dict[str, int | None]:
    if not isinstance(pricing, Mapping):
        raise TypeError("pricing must be a mapping")
    unknown = set(pricing).difference(_PRICING_KEYS)
    if unknown:
        raise ValueError("pricing contains unknown keys: " + ", ".join(sorted(unknown)))
    result: dict[str, int | None] = {}
    for key in _PRICING_KEYS:
        value = pricing.get(key)
        if value is not None:
            _count(f"{key}_price", value)
        result[key] = value
    return result


def _count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_usage_count(
    usage: Mapping[str, Any], *keys: str
) -> int | None:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            return _count(key, value)
    return None


def _usage_count(usage: Mapping[str, Any], *keys: str) -> int:
    value = _optional_usage_count(usage, *keys)
    return 0 if value is None else value


def _ceil_cost(
    components: tuple[tuple[int, int | None], ...],
) -> int:
    numerator = sum(tokens * (rate or 0) for tokens, rate in components)
    if numerator == 0:
        return 0
    return (numerator + _TOKENS_PER_MILLION - 1) // _TOKENS_PER_MILLION


__all__ = [
    "PricingUnavailableError",
    "actual_model_cost_micros",
    "dollars_per_1m_to_micros",
    "estimated_model_cost_micros",
    "pricing_micros_per_1m",
    "pricing_revision",
]

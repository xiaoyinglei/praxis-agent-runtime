"""Safe provider failure facts; retry authorization belongs to the runtime."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any, Literal

import httpx
import openai

type ModelFailureKind = Literal[
    "authentication",
    "permission",
    "invalid_request",
    "context_overflow",
    "rate_limited",
    "quota_exhausted",
    "server_error",
    "transport_not_sent",
    "transport_unknown",
    "protocol_error",
]
type ModelFailurePhase = Literal["connect", "response", "stream", "protocol"]
type ProviderOutcome = Literal["not_sent", "rejected", "unknown"]
type BillingOutcome = Literal["released", "unknown"]

_MESSAGES: dict[str, str] = {
    "authentication": "Provider authentication failed.",
    "permission": "Provider permission denied.",
    "invalid_request": "Provider rejected the model request.",
    "context_overflow": "Provider context window exceeded.",
    "rate_limited": "Provider rate limit reached.",
    "quota_exhausted": "Provider quota exhausted.",
    "server_error": "Provider server failed.",
    "transport_not_sent": "Provider connection failed before the request was sent.",
    "transport_unknown": "Provider transport failed; request outcome is unknown.",
    "protocol_error": "Provider response violated the model protocol.",
}
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:/-]{1,200}\Z")
_QUOTA_CODES = frozenset(
    {
        "insufficient_quota",
        "quota_exceeded",
        "quota_exhausted",
        "billing_hard_limit_reached",
        "billing_not_active",
        "credit_balance_too_low",
        "insufficient_credits",
    }
)


@dataclass(frozen=True, slots=True)
class ModelFailure:
    kind: ModelFailureKind
    phase: ModelFailurePhase
    provider_outcome: ProviderOutcome
    billing_outcome: BillingOutcome
    http_status: int | None = None
    provider_code: str | None = None
    request_id: str | None = None
    retry_at: float | None = None
    retryable: bool = False
    output_started: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _MESSAGES:
            raise ValueError("unsupported model failure kind")
        if self.phase not in {"connect", "response", "stream", "protocol"}:
            raise ValueError("unsupported model failure phase")
        if self.provider_outcome not in {"not_sent", "rejected", "unknown"}:
            raise ValueError("unsupported provider outcome")
        if self.billing_outcome not in {"released", "unknown"}:
            raise ValueError("unsupported billing outcome")
        if self.http_status is not None and (type(self.http_status) is not int or not 100 <= self.http_status <= 599):
            raise ValueError("invalid provider HTTP status")
        if type(self.retryable) is not bool or type(self.output_started) is not bool:
            raise TypeError("model failure flags must be booleans")
        if self.retry_at is not None and (
            isinstance(self.retry_at, bool)
            or not isinstance(self.retry_at, (int, float))
            or not math.isfinite(self.retry_at)
        ):
            raise ValueError("retry_at must be a finite epoch time")
        object.__setattr__(self, "provider_code", _identifier(self.provider_code))
        object.__setattr__(self, "request_id", _identifier(self.request_id))
        object.__setattr__(self, "message", _MESSAGES[self.kind])
        if self.output_started:
            object.__setattr__(self, "retryable", False)
            object.__setattr__(self, "provider_outcome", "unknown")
            object.__setattr__(self, "billing_outcome", "unknown")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ModelFailure:
        return cls(
            kind=payload["kind"],
            phase=payload["phase"],
            provider_outcome=payload["provider_outcome"],
            billing_outcome=payload["billing_outcome"],
            http_status=payload.get("http_status"),
            provider_code=payload.get("provider_code"),
            request_id=payload.get("request_id"),
            retry_at=payload.get("retry_at"),
            retryable=payload.get("retryable", False),
            output_started=payload.get("output_started", False),
        )


class ModelCallFailed(RuntimeError):  # noqa: N818 - public failure contract
    """One provider attempt failed, carrying only safe serializable facts."""

    def __init__(self, failure: ModelFailure, *, validation_message: str | None = None) -> None:
        # Only use a validation message defined by our code. The durable failure
        # always derives its own message, independent of exception diagnostics.
        self.failure = failure
        super().__init__(validation_message or failure.message)


def _identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) else None


def _retry_deadline(headers: httpx.Headers, now: float) -> float | None:
    for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(header)
        if value is None:
            continue
        numeric = re.fullmatch(r"[0-9]+(?:\.[0-9]+)?" if header == "retry-after-ms" else r"[0-9]+", value.strip())
        try:
            if numeric is None:
                raise ValueError("not a protocol delay value")
            seconds = float(value) * scale
        except ValueError:
            if header != "retry-after":
                continue
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                deadline = date.timestamp()
            except (TypeError, ValueError, OverflowError):
                continue
            return max(now, deadline) if math.isfinite(deadline) else None
        if math.isfinite(seconds) and seconds >= 0 and math.isfinite(now + seconds):
            return now + seconds
    return None


def _error_codes(error: openai.APIError) -> tuple[str, ...]:
    values: list[object] = [error.code, error.type]
    body = error.body
    if isinstance(body, Mapping):
        nested = body.get("error", body)
        if isinstance(nested, Mapping):
            values.extend((nested.get("code"), nested.get("type")))
    return tuple(value for item in values if (value := _identifier(item)) is not None)


def classify_model_failure(
    error: BaseException,
    *,
    output_started: bool = False,
    now: float | None = None,
) -> ModelFailure:
    """Classify SDK types/status/code only, never provider message or raw body text."""
    if isinstance(error, ModelCallFailed):
        return replace(error.failure, output_started=error.failure.output_started or output_started)
    if isinstance(error, openai.APIStatusError):
        status = error.status_code
        codes = _error_codes(error)
        code = next((value for value in codes if value in _QUOTA_CODES), next(iter(codes), None))
        headers = error.response.headers
        kind: ModelFailureKind
        if code in _QUOTA_CODES or status == 402:
            kind = "quota_exhausted"
        elif status == 400 and code in {"context_length_exceeded", "context_window_exceeded"}:
            kind = "context_overflow"
        elif status == 401:
            kind = "authentication"
        elif status == 403:
            kind = "permission"
        elif status == 429:
            kind = "rate_limited"
        elif status >= 500:
            kind = "server_error"
        else:
            kind = "invalid_request"
        rejected = kind != "server_error"
        retryable = kind == "rate_limited" or (kind == "server_error" and status not in {501, 505})
        if headers.get("x-should-retry", "").lower() == "false":
            retryable = False
        return ModelFailure(
            kind=kind,
            phase="stream" if output_started else "response",
            provider_outcome="rejected",
            billing_outcome="released" if rejected else "unknown",
            http_status=status,
            provider_code=code,
            request_id=_identifier(error.request_id) or _identifier(headers.get("request-id")),
            retry_at=_retry_deadline(headers, time.time() if now is None else now),
            retryable=retryable,
            output_started=output_started,
        )
    if isinstance(error, openai.APIError):
        codes = _error_codes(error)
        code = next((value for value in codes if value in _QUOTA_CODES), next(iter(codes), None))
        sse_kind: ModelFailureKind
        if code in _QUOTA_CODES:
            sse_kind = "quota_exhausted"
        elif code in {"rate_limit_exceeded", "slow_down"}:
            sse_kind = "rate_limited"
        elif code in {"server_error", "internal_server_error", "overloaded_error"}:
            sse_kind = "server_error"
        else:
            sse_kind = "protocol_error"
        # A structured SSE error is a terminal provider failure, but it does
        # not establish zero usage after the successful HTTP response began.
        if sse_kind != "protocol_error":
            return ModelFailure(
                kind=sse_kind,
                phase="stream",
                provider_outcome="rejected",
                billing_outcome="unknown",
                provider_code=code,
                retryable=sse_kind in {"rate_limited", "server_error"},
                output_started=output_started,
            )
    cause: BaseException | None = error
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout)):
            return ModelFailure(
                kind="transport_not_sent",
                phase="stream" if output_started else "connect",
                provider_outcome="not_sent",
                billing_outcome="released",
                retryable=True,
                output_started=output_started,
            )
        if isinstance(cause, (httpx.ReadError, httpx.ReadTimeout, httpx.WriteError, httpx.WriteTimeout)):
            break
        cause = cause.__cause__
    transport = isinstance(error, (openai.APIConnectionError, httpx.TransportError, ConnectionError, TimeoutError))
    return ModelFailure(
        kind="transport_unknown" if transport else "protocol_error",
        phase="stream" if output_started and transport else ("response" if transport else "protocol"),
        provider_outcome="unknown",
        billing_outcome="unknown",
        output_started=output_started,
    )


__all__ = ["ModelCallFailed", "ModelFailure", "classify_model_failure"]

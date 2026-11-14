"""唯一模型重试机制(D7/§5.9;R-T3-1 分层)。

本模块是 praxis 里**唯一**的失败分类器与唯一退避实现:core 不得 import
adapters,故它只经 ``protocol.retry.RetryingStream`` 结构签名由 composition
注入;``protocol/retry.py`` 只放纯数据/纯策略,OpenAI/httpx 类型只在本文件出现。

关键语义:

- ``classify_failure`` 只按 SDK 异常类型、HTTP status 与 provider code 分类;
  provider 原始消息只经有界白名单摘要进入 ``diagnostic``,永不进入固定文案。
- ``stream_with_retry`` 在失败时**丢弃**该次已收到的增量再重试——失败首帧若
  推给 UI,重试后就是回滚不掉的重影。它把每次失败追加为 ``AttemptRecord``
  (成功记录由 core 在 turn 归属下补写),最终失败抛 ``ModelUnavailableError``。
- 可重试集合仅 ``{rate_limited, server_error, transport_not_sent}``;已经产出
  过任何增量(``output_started``)的请求永不重试。
- ``floor`` 来自 ``model_attempt`` item 的跨重启时钟地板;时钟回拨即拒绝。
"""

from __future__ import annotations

import math
import random
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import UTC
from email.utils import parsedate_to_datetime

import httpx
import openai

from praxis.protocol.models import (
    ModelEvent,
    ModelReasoningDelta,
    ModelTextDelta,
    ModelToolCallDelta,
    collect_events,
)
from praxis.protocol.retry import (
    AttemptRecord,
    FailureKind,
    ModelFailure,
    ModelUnavailableError,
    NoRetryPlan,
    RetryAtSource,
    RetryingStream,
    RetryPlan,
    RetryPolicy,
    StreamFactory,
)

# 与旧 failures.py 同一白名单思路:只允许可打印、无结构字符的短摘要。
# provider body 通常含 JSON 花括号/换行,落不进这里;openai.APIError 更是一律
# 只留异常类型名,绝不回显 body。
_SAFE_DIAGNOSTIC = re.compile(r"[A-Za-z0-9 .,_:/;'\"()!?+-]{1,160}\Z")
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


def _identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) else None


def _diagnostic(exc: BaseException) -> str:
    """原文摘要:SDK 异常只留类型名;其余只接受有界白名单字符。"""
    if isinstance(exc, openai.APIError):
        return type(exc).__name__
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    if text and _SAFE_DIAGNOSTIC.fullmatch(text):
        return text
    return type(exc).__name__


def _retry_deadline(headers: Mapping[str, str], now: float) -> float | None:
    for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(header)
        if value is None:
            continue
        numeric = re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)?" if header == "retry-after-ms" else r"[0-9]+",
            value.strip(),
        )
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


def _failure(
    *,
    kind: FailureKind,
    retryable: bool,
    retry_at: float | None,
    output_started: bool,
    diagnostic: str,
) -> ModelFailure:
    """构造固定文案失败事实;message 留空让协议层填表。"""
    return ModelFailure(
        kind=kind,
        message="",
        retryable=retryable,
        retry_at=retry_at,
        output_started=output_started,
        phase="stream" if output_started else "request",
        diagnostic=diagnostic,
    )


def _status_failure(
    *,
    status: int,
    codes: tuple[str, ...],
    headers: Mapping[str, str],
    observed_at: float,
    output_started: bool,
    diagnostic: str,
) -> ModelFailure:
    """HTTP status → 失败事实(唯一映射;OpenAI SDK 与原生 httpx 共用)。"""
    code = next((value for value in codes if value in _QUOTA_CODES), next(iter(codes), None))
    kind: FailureKind
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
    retryable = kind == "rate_limited" or (kind == "server_error" and status not in {501, 505})
    if headers.get("x-should-retry", "").lower() == "false":
        retryable = False
    return _failure(
        kind=kind,
        retryable=retryable,
        retry_at=_retry_deadline(headers, observed_at),
        output_started=output_started,
        diagnostic=diagnostic,
    )


def classify_failure(
    exc: BaseException, *, output_started: bool = False, now: float | None = None
) -> ModelFailure:
    """把一次 provider 异常映射为安全失败事实(唯一分类器)。

    ``now`` 只用于把 ``retry-after`` 相对秒数换算成绝对时刻;缺省取
    ``time.time()``,测试可注入假时钟而不 monkeypatch 全局时间。
    """
    observed_at = time.time() if now is None else now
    diagnostic = _diagnostic(exc)

    if isinstance(exc, openai.APIStatusError):
        return _status_failure(
            status=exc.status_code,
            codes=_error_codes(exc),
            headers=exc.response.headers,
            observed_at=observed_at,
            output_started=output_started,
            diagnostic=diagnostic,
        )

    # 原生 httpx provider(ollama 的 /api/chat 等)不经过 openai SDK:
    # 非 2xx 也走同一条 status→kind 映射,否则 429/5xx 会被误判成不可重试的
    # protocol_error(D7 重试静默失效)。
    if isinstance(exc, httpx.HTTPStatusError):
        return _status_failure(
            status=exc.response.status_code,
            codes=(),
            headers=exc.response.headers,
            observed_at=observed_at,
            output_started=output_started,
            diagnostic=diagnostic,
        )

    # 连接失败只在"请求尚未发出"时可安全重放;读/写中途失败一律不重试。
    # 必须先于 openai.APIError 分支:APIConnectionError 也是 APIError,但它的
    # 因果链才决定请求是否真的发出去了。
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout)):
            return _failure(
                kind="transport_not_sent",
                retryable=True,
                retry_at=None,
                output_started=output_started,
                diagnostic=diagnostic,
            )
        if isinstance(
            cause, (httpx.ReadError, httpx.ReadTimeout, httpx.WriteError, httpx.WriteTimeout)
        ):
            break
        cause = cause.__cause__

    if isinstance(exc, openai.APIError) and not isinstance(exc, openai.APIConnectionError):
        # 流内结构化错误(SSE error):没有 HTTP status,只按 code 分类。
        codes = _error_codes(exc)
        code = next((value for value in codes if value in _QUOTA_CODES), next(iter(codes), None))
        sse_kind: FailureKind
        if code in _QUOTA_CODES:
            sse_kind = "quota_exhausted"
        elif code in {"rate_limit_exceeded", "slow_down"}:
            sse_kind = "rate_limited"
        elif code in {"server_error", "internal_server_error", "overloaded_error"}:
            sse_kind = "server_error"
        else:
            sse_kind = "protocol_error"
        return _failure(
            kind=sse_kind,
            retryable=sse_kind in {"rate_limited", "server_error"},
            retry_at=None,
            output_started=output_started,
            diagnostic=diagnostic,
        )

    transport = isinstance(
        exc, (openai.APIConnectionError, httpx.TransportError, ConnectionError, TimeoutError)
    )
    return _failure(
        kind="transport_unknown" if transport else "protocol_error",
        retryable=False,
        retry_at=None,
        output_started=output_started,
        diagnostic=diagnostic,
    )


def _is_delta(event: ModelEvent) -> bool:
    return isinstance(event, (ModelTextDelta, ModelReasoningDelta, ModelToolCallDelta))


async def stream_with_retry(
    stream_factory: StreamFactory,
    *,
    policy: RetryPolicy,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    attempts: list[AttemptRecord],
    floor: float,
) -> AsyncIterator[ModelEvent]:
    """唯一退避实现:成功才产出事件;失败尝试只留 ``AttemptRecord``。

    ``attempts`` 是本 turn 已有记录的可变列表(调用方从 item 重建);机制
    只追加失败记录,成功记录由 turn 层在成功返回后补写。总尝试数受
    ``max_auto_retries + 1`` 限;最终失败抛 ``ModelUnavailableError``。
    """
    first_dispatched_at: float | None = None
    retries_used = 0

    for _ in range(policy.max_auto_retries + 1):
        dispatched_at = clock()
        if first_dispatched_at is None:
            first_dispatched_at = dispatched_at
        events: list[ModelEvent] = []
        output_started = False
        try:
            async for event in stream_factory():
                events.append(event)
                if _is_delta(event):
                    output_started = True
        except Exception as exc:  # noqa: BLE001 — 分类器必须看到所有 provider 异常
            now = clock()
            failure = classify_failure(exc, output_started=output_started, now=now)
            assert first_dispatched_at is not None
            # 本次 call 的首个派发时刻同样是地板:中途回拨也不得自动重试。
            retry_floor = max(floor, first_dispatched_at)
            decision: RetryPlan | NoRetryPlan = (
                policy.plan(
                    retry_count=retries_used,
                    first_dispatched_at=first_dispatched_at,
                    failure=failure,
                    now=now,
                    random_fraction=random.random(),
                )
                if now >= retry_floor
                else NoRetryPlan("model_retry_clock_invalid")
            )
            retry_at = decision.retry_at if isinstance(decision, RetryPlan) else None
            source: RetryAtSource | None = None
            if retry_at is not None:
                source = (
                    "retry_after"
                    if failure.retry_at is not None and retry_at == failure.retry_at
                    else "backoff"
                )
            attempts.append(
                AttemptRecord(
                    attempt=len(attempts),
                    kind=failure.kind,
                    retry_at=retry_at,
                    dispatched_at=dispatched_at,
                    output_started=output_started,
                    outcome="failed",
                    retry_at_source=source,
                )
            )
            if isinstance(decision, NoRetryPlan):
                raise ModelUnavailableError(
                    failure, reason_code=decision.reason_code
                ) from exc
            await sleep(max(0.0, decision.retry_at - clock()))
            retries_used += 1
            continue

        for event in events:
            yield event
        return

    raise AssertionError("retry loop exhausted without a terminal outcome")  # pragma: no cover


def make_retry_stream(
    *,
    policy: RetryPolicy,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> RetryingStream:
    """把全参机制绑定成 core 用的三参 ``RetryingStream``(composition 唯一调用点)。"""

    async def retry_stream(
        stream_factory: StreamFactory, attempts: list[AttemptRecord], floor: float
    ) -> list[ModelEvent]:
        return await collect_events(
            stream_with_retry(
                stream_factory,
                policy=policy,
                clock=clock,
                sleep=sleep,
                attempts=attempts,
                floor=floor,
            )
        )

    return retry_stream


__all__ = ["classify_failure", "make_retry_stream", "stream_with_retry"]
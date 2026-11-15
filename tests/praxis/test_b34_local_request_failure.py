"""B34:客户端请求构造/编码失败必须独立归因,不再冒充 provider 协议违规。

findings B34(阶段 3B 起手第一条):凭据含非 ASCII 字符时,OpenAI SDK 在
**发送阶段**抛 ``UnicodeEncodeError``;旧 ``classify_failure`` 的兜底把它归成
``protocol_error`` ⇒ 运维被告知 ``Provider response violated the model
protocol.``,而真因是自己的凭据/配置。这里钉住五件事:

1. 裸 ``UnicodeEncodeError`` ⇒ ``local_request_error``(不可重试,固定文案指向
   自身配置);
2. 因果链里的 ``httpx.LocalProtocolError`` 同样归类,且凭据值绝不进
   message/diagnostic/``str(ModelUnavailableError)``;
3. 真形状复现:``OpenAICompatAdapter(api_key=非 ASCII)`` 的 ``models check``
   路径不得再输出 "violated the model protocol";
4. 反例不变量:SSE 流内 200 应答的畸形帧/无 code 错误仍是真协议违规
   (``protocol_error``),新分支不得吞掉它;
5. 固定文案表穷举:每个 ``FailureKind`` 恰有一条 message,且可重试集合不变。
"""

from __future__ import annotations

import asyncio
from typing import get_args

import httpx
import openai
import pytest

from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.adapters.models.probe import probe_model
from praxis.adapters.models.retry import classify_failure
from praxis.protocol.models import (
    ModelMessage,
    ModelSettings,
    ModelSpec,
    collect_events,
)
from praxis.protocol.retry import (
    _MESSAGES,
    _RETRYABLE_KINDS,
    FailureKind,
    ModelFailure,
    ModelUnavailableError,
    NoRetryPlan,
    RetryPolicy,
)

_BASE = "https://api.deepseek.com/v1"
_LOCAL_MESSAGE = (
    "Local request construction failed before contacting the provider; "
    "check model and credential configuration (values must be ASCII-safe)."
)
_SECRET = "sk-s3cr3t"


def _unicode_error() -> UnicodeEncodeError:
    """findings B34 的原始形状:发送期 ascii 编码失败。"""
    return UnicodeEncodeError("ascii", "…", 7, 9, "ordinal not in range")


def _local_protocol_error() -> httpx.LocalProtocolError:
    """h11 发送期形状:原文含 header 字节(可能带凭据)。"""
    return httpx.LocalProtocolError(f"Illegal header value b'Bearer {_SECRET}'")


def _spec() -> ModelSpec:
    return ModelSpec(model_id="m", base_url=_BASE)


def _compat_stream_failure(body: bytes) -> ModelFailure:
    """把 200 SSE body 喂给真实适配器,分类它抛出的异常。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = OpenAICompatAdapter(
        _spec(),
        api_key="k",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    request = adapter.build_request(
        [ModelMessage(role="user", content="hi")], (), ModelSettings()
    )

    async def drive() -> None:
        await collect_events(adapter.stream(request))

    with pytest.raises(Exception) as excinfo:
        asyncio.run(drive())
    return classify_failure(excinfo.value)


# ---------------------------------------------------------------------------
# 1. 裸 UnicodeEncodeError ⇒ local_request_error
# ---------------------------------------------------------------------------


def test_bare_unicode_encode_error_is_local_request_error() -> None:
    failure = classify_failure(_unicode_error())

    assert failure.kind == "local_request_error"
    assert failure.retryable is False
    assert failure.phase == "request"
    assert failure.message == _LOCAL_MESSAGE
    assert "protocol" not in failure.message.lower()
    assert failure.diagnostic == "UnicodeEncodeError"

    # 新 kind 不进可重试集合:策略层给它 not_retryable 原因码。
    decision = RetryPolicy().plan(
        retry_count=0,
        first_dispatched_at=0.0,
        failure=ModelFailure(
            "local_request_error", "", retryable=True, retry_at=None,
            output_started=False, phase="request",
        ),
        now=0.0,
        random_fraction=0.5,
    )
    assert isinstance(decision, NoRetryPlan)
    assert decision.reason_code == "not_retryable:local_request_error"
    assert "local_request_error" not in _RETRYABLE_KINDS


# ---------------------------------------------------------------------------
# 2. 因果链里的 LocalProtocolError:归类且凭据值绝不外泄
# ---------------------------------------------------------------------------


def test_local_protocol_error_in_cause_chain_never_echoes_credential() -> None:
    request = httpx.Request("POST", f"{_BASE}/chat/completions")
    try:
        try:
            raise _local_protocol_error()
        except httpx.LocalProtocolError as cause:
            raise openai.APIConnectionError(request=request) from cause
    except openai.APIConnectionError as exc:
        failure = classify_failure(exc)

    assert failure.kind == "local_request_error"
    assert failure.retryable is False
    assert failure.message == _LOCAL_MESSAGE
    assert _SECRET not in failure.message
    assert _SECRET not in (failure.diagnostic or "")
    assert _SECRET not in str(ModelUnavailableError(failure))


def test_bare_local_protocol_error_also_never_echoes_credential() -> None:
    """裸 LocalProtocolError 也必须安全(原文可含 header 字节)。"""
    failure = classify_failure(_local_protocol_error())

    assert failure.kind == "local_request_error"
    assert failure.retryable is False
    assert _SECRET not in failure.message
    assert _SECRET not in (failure.diagnostic or "")
    assert _SECRET not in str(ModelUnavailableError(failure))


# ---------------------------------------------------------------------------
# 3. 真形状复现:models check 路径不再冤枉 provider
# ---------------------------------------------------------------------------


def test_models_check_path_reports_local_request_error_not_provider_protocol() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)  # 编码在发送前失败,不应抵达 transport
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    adapter = OpenAICompatAdapter(
        _spec(),
        api_key="\U0001f511",  # 非 ASCII 凭据:发送期 UnicodeEncodeError
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    outcome = asyncio.run(probe_model(adapter, credential_source="env"))

    assert outcome.ok is False
    assert outcome.steps == ("text",)
    detail = outcome.detail or ""
    assert "violated the model protocol" not in detail
    assert "(protocol_error)" not in detail
    assert _LOCAL_MESSAGE in detail
    assert "(local_request_error)" in detail
    assert "\U0001f511" not in detail  # 凭据值不进 detail
    assert calls == []


# ---------------------------------------------------------------------------
# 4. 反例不变量:SSE 流内 200 应答的协议违规仍归 protocol_error
# ---------------------------------------------------------------------------


def test_sse_in_band_error_without_code_stays_protocol_error() -> None:
    failure = _compat_stream_failure(b'data: {"error": {"message": "boom"}}\n\n')

    assert failure.kind == "protocol_error"
    assert failure.retryable is False
    assert failure.message == _MESSAGES["protocol_error"]


def test_sse_malformed_frame_stays_protocol_error() -> None:
    failure = _compat_stream_failure(b"data: {not json}\n\n")

    assert failure.kind == "protocol_error"
    assert failure.retryable is False
    assert failure.message == _MESSAGES["protocol_error"]


# ---------------------------------------------------------------------------
# 5. 固定文案表穷举 + 可重试集合不变
# ---------------------------------------------------------------------------


def test_every_failure_kind_has_exactly_one_fixed_message() -> None:
    kinds = set(get_args(FailureKind))

    assert kinds == set(_MESSAGES)
    assert len(_MESSAGES) == len(kinds)
    assert all(message.strip() for message in _MESSAGES.values())
    assert len(set(_MESSAGES.values())) == len(_MESSAGES)
    assert _MESSAGES["local_request_error"] == _LOCAL_MESSAGE
    assert _RETRYABLE_KINDS == {"rate_limited", "server_error", "transport_not_sent"}
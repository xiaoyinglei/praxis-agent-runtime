from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from email.utils import formatdate
from typing import Any, cast

import httpx
import openai
import pytest

from agent_runtime.core.llm_registry import ResolvedModel
from agent_runtime.harness.model_adapter import GatewayHarnessModel
from agent_runtime.harness.protocol import HarnessModelRequest, ModelContextOverflowError, ModelMessage
from agent_runtime.model_definition import ModelCapabilities, RequestDefaultsDefinition
from agent_runtime.modeling.chat import OpenAICompatibleChatGenerator
from agent_runtime.modeling.config import GenerationConfig
from agent_runtime.modeling.events import ModelEvent
from agent_runtime.modeling.gateway import LLMGateway


def _generator(handler: Callable[[httpx.Request], httpx.Response]) -> OpenAICompatibleChatGenerator:
    generator = OpenAICompatibleChatGenerator(model="test", base_url="https://provider.invalid/v1", api_key="secret")
    old_client = generator._client
    generator._client = openai.OpenAI(
        api_key="secret",
        base_url="https://provider.invalid/v1",
        max_retries=old_client.max_retries,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    old_client.close()
    return generator


def _status_error(status: int, code: str | None = None, **headers: str) -> openai.APIStatusError:
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://provider.invalid/v1/chat/completions"),
        headers=headers,
    )
    return openai.APIStatusError("private raw body and credential", response=response, body={"code": code})


def test_sdk_429_is_one_http_attempt() -> None:
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}}, headers={"Retry-After": "0"})

    generator = _generator(respond)
    try:
        with pytest.raises(openai.RateLimitError):
            generator.generate_with_tools(messages=[{"role": "user", "content": "hello"}], tools=[])
        assert calls == 1
    finally:
        generator.close()


@pytest.mark.parametrize(
    ("status", "code", "kind", "retryable", "billing"),
    [
        (401, None, "authentication", False, "released"),
        (401, "invalid_api_key", "authentication", False, "released"),
        (403, None, "permission", False, "released"),
        (400, None, "invalid_request", False, "released"),
        (400, "context_length_exceeded", "context_overflow", False, "released"),
        (400, "context_window_exceeded", "context_overflow", False, "released"),
        (404, None, "invalid_request", False, "released"),
        (422, None, "invalid_request", False, "released"),
        (402, None, "quota_exhausted", False, "released"),
        (429, "insufficient_quota", "quota_exhausted", False, "released"),
        (503, "quota_exceeded", "quota_exhausted", False, "released"),
        (429, None, "rate_limited", True, "released"),
        (500, None, "server_error", True, "unknown"),
        (502, None, "server_error", True, "unknown"),
        (503, None, "server_error", True, "unknown"),
        (504, None, "server_error", True, "unknown"),
        (529, None, "server_error", True, "unknown"),
        (501, None, "server_error", False, "unknown"),
        (505, None, "server_error", False, "unknown"),
    ],
)
def test_structured_status_classification(
    status: int, code: str | None, kind: str, retryable: bool, billing: str
) -> None:
    from agent_runtime.modeling.failures import ModelFailure, classify_model_failure

    failure = classify_model_failure(_status_error(status, code), now=100)
    assert (failure.kind, failure.retryable, failure.billing_outcome) == (kind, retryable, billing)
    assert failure.http_status == status
    assert failure.provider_code == code
    assert failure.phase == "response"
    assert failure.provider_outcome == "rejected"
    assert ModelFailure.from_payload(failure.to_payload()) == failure
    assert "private" not in json.dumps(failure.to_payload())
    with pytest.raises(FrozenInstanceError):
        failure.kind = "server_error"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "250"}, 100.25),
        ({"Retry-After": "2"}, 102.0),
        ({"Retry-After": "0"}, 100),
        ({"Retry-After": formatdate(160, usegmt=True)}, 160),
        ({"Retry-After": "bad"}, None),
        ({"Retry-After": "-1"}, None),
        ({"Retry-After": "NaN"}, None),
        ({"retry-after-ms": "bad", "Retry-After": "2"}, 102),
    ],
)
def test_retry_deadline_is_parsed_once(headers: dict[str, str], expected: float | None) -> None:
    from agent_runtime.modeling.failures import ModelFailure, classify_model_failure

    failure = classify_model_failure(_status_error(429, **headers), now=100)
    assert failure.retry_at == expected
    assert ModelFailure.from_payload(failure.to_payload()).retry_at == expected


def test_retry_override_and_partial_output_are_conservative() -> None:
    from agent_runtime.modeling.failures import classify_model_failure

    assert not classify_model_failure(_status_error(503, **{"x-should-retry": "false"})).retryable
    assert not classify_model_failure(_status_error(401, **{"x-should-retry": "true"})).retryable
    failure = classify_model_failure(_status_error(429), output_started=True)
    assert failure.output_started and not failure.retryable
    assert failure.billing_outcome == "unknown"


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.ReadTimeout])
def test_transport_proof_uses_sdk_cause(error_type: type[httpx.TransportError]) -> None:
    from agent_runtime.modeling.failures import classify_model_failure

    def fail(request: httpx.Request) -> httpx.Response:
        raise error_type("secret transport detail", request=request)

    generator = _generator(fail)
    try:
        with pytest.raises(openai.APIConnectionError) as raised:
            generator.generate_with_tools(messages=[], tools=[])
        failure = classify_model_failure(raised.value)
        not_sent = error_type in {httpx.ConnectError, httpx.ConnectTimeout}
        assert failure.provider_outcome == ("not_sent" if not_sent else "unknown")
        assert failure.retryable is not_sent
        assert failure.kind == ("transport_not_sent" if not_sent else "transport_unknown")
        assert "secret" not in json.dumps(failure.to_payload())
    finally:
        generator.close()


class _Accounting:
    def count(self, text: str) -> int:
        return len(text)

    def clip(self, text: str, token_budget: int, *, add_ellipsis: bool = False) -> str:
        return text[:token_budget]


def _model(generator: OpenAICompatibleChatGenerator) -> GatewayHarnessModel:
    accounting = _Accounting()
    gateway = LLMGateway(generator=generator, token_accounting=cast(Any, accounting), model_context_tokens=8192)
    resolved = ResolvedModel(
        generator=generator,
        gateway=gateway,
        model_id="test",
        provider="openai-compatible",
        capabilities=ModelCapabilities(
            context_window_tokens=8192,
            max_output_tokens=256,
            supports_native_tools=True,
            supports_structured_output=True,
        ),
        token_accounting=cast(Any, accounting),
        request_defaults=RequestDefaultsDefinition(),
        generation_config=GenerationConfig(),
    )
    return GatewayHarnessModel(model_id="test", resolved=resolved, instructions=("Answer directly.",))


def _request() -> HarnessModelRequest:
    return HarnessModelRequest(
        thread_id="thread",
        turn_id="turn",
        messages=(ModelMessage(role="user", body="hello"),),
        binding_manifest={"model_id": "test"},
    )


@pytest.mark.parametrize(
    "status,code,kind",
    [
        (429, None, "rate_limited"),
        (401, None, "authentication"),
        (503, None, "server_error"),
        (429, "insufficient_quota", "quota_exhausted"),
    ],
)
def test_real_adapter_normalizes_sdk_failure(status: int, code: str | None, kind: str) -> None:
    from agent_runtime.modeling.failures import ModelCallFailed

    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status,
            json={"error": {"code": code, "message": "secret provider message"}},
            headers={"x-request-id": "req-safe"},
        )

    generator = _generator(respond)
    model = _model(generator)
    try:
        with pytest.raises(ModelCallFailed) as raised:
            asyncio.run(model.dispatch(model.prepare(_request())))
        assert raised.value.failure.kind == kind
        assert raised.value.failure.request_id == "req-safe"
        assert "secret" not in str(raised.value)
        assert calls == 1
    finally:
        generator.close()


class _TruncatedStream(httpx.SyncByteStream):
    def __init__(self, *, disconnect: bool) -> None:
        self.disconnect = disconnect

    def __iter__(self) -> Iterator[bytes]:
        chunk = {"id": "chat-id", "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        if self.disconnect:
            raise httpx.ReadError("private disconnect detail")
        yield b"data: [DONE]\n\n"


@pytest.mark.parametrize("disconnect", [True, False])
def test_partial_stream_never_automatically_retries(disconnect: bool) -> None:
    from agent_runtime.modeling.failures import ModelCallFailed

    generator = _generator(
        lambda _: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_TruncatedStream(disconnect=disconnect)
        )
    )
    model = _model(generator)
    events: list[ModelEvent] = []
    try:
        with pytest.raises(ModelCallFailed) as raised:
            asyncio.run(model.dispatch(model.prepare(_request()), event_sink=events.append))
        failure = raised.value.failure
        assert failure.output_started
        assert not failure.retryable
        assert failure.provider_outcome == failure.billing_outcome == "unknown"
        assert failure.phase == "stream"
        assert "private" not in str(raised.value)
        assert events
    finally:
        generator.close()


def test_context_overflow_keeps_its_compaction_contract() -> None:
    generator = _generator(
        lambda _: httpx.Response(
            400,
            json={"error": {"code": "context_length_exceeded", "message": "secret provider detail"}},
            headers={"x-request-id": "ctx-request", "private-header": "secret"},
        )
    )
    model = _model(generator)
    try:
        with pytest.raises(ModelContextOverflowError) as raised:
            asyncio.run(model.dispatch(model.prepare(_request())))
        failure = raised.value.failure
        assert failure is not None
        assert failure.kind == "context_overflow"
        assert failure.http_status == 400
        assert failure.provider_code == "context_length_exceeded"
        assert failure.request_id == "ctx-request"
        assert failure.provider_outcome == "rejected"
        assert failure.billing_outcome == "released"
        assert not failure.retryable
        assert "secret" not in json.dumps(failure.to_payload())
    finally:
        generator.close()


def test_direct_stream_without_terminal_has_typed_unknown_failure() -> None:
    from agent_runtime.modeling.failures import ModelCallFailed

    generator = _generator(
        lambda _: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_TruncatedStream(disconnect=False)
        )
    )
    try:
        with pytest.raises(ModelCallFailed) as raised:
            list(generator.stream_with_tools(messages=[], tools=[]))
        assert raised.value.failure.phase == "stream"
        assert raised.value.failure.provider_outcome == "unknown"
        assert raised.value.failure.output_started
    finally:
        generator.close()


def test_adapter_before_connect_is_proven_not_sent() -> None:
    from agent_runtime.modeling.failures import ModelCallFailed

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("private detail", request=request)

    generator = _generator(fail)
    model = _model(generator)
    try:
        with pytest.raises(ModelCallFailed) as raised:
            asyncio.run(model.dispatch(model.prepare(_request())))
        assert raised.value.failure.kind == "transport_not_sent"
        assert raised.value.failure.retryable
        assert raised.value.failure.phase == "connect"
    finally:
        generator.close()


def test_adapter_invalid_response_is_protocol_error_with_unknown_billing() -> None:
    from agent_runtime.modeling.failures import ModelCallFailed

    generator = _generator(lambda _: httpx.Response(200, json={"id": "response", "choices": []}))
    model = _model(generator)
    try:
        with pytest.raises(ModelCallFailed) as raised:
            asyncio.run(model.dispatch(model.prepare(_request())))
        assert raised.value.failure.kind == "protocol_error"
        assert raised.value.failure.billing_outcome == "unknown"
        assert not raised.value.failure.retryable
    finally:
        generator.close()


def test_wire_drift_after_provider_success_is_protocol_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from agent_runtime.modeling.failures import ModelCallFailed
    from agent_runtime.modeling.gateway import AgentModelResponse

    generator = _generator(
        lambda _: httpx.Response(
            200,
            json={
                "id": "response",
                "choices": [
                    {"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop", "index": 0}
                ],
            },
        )
    )
    model = _model(generator)
    gateway = model._resolved.gateway
    assert gateway is not None
    dispatch = gateway.agenerate_model_request

    async def changed_wire(**kwargs: Any) -> AgentModelResponse:
        return replace(await dispatch(**kwargs), provider_wire_hash="unexpected-wire")

    monkeypatch.setattr(gateway, "agenerate_model_request", changed_wire)
    try:
        with pytest.raises(ModelCallFailed) as raised:
            asyncio.run(model.dispatch(model.prepare(_request())))
        assert raised.value.failure.kind == "protocol_error"
        assert raised.value.failure.phase == "protocol"
        assert raised.value.failure.provider_outcome == raised.value.failure.billing_outcome == "unknown"
    finally:
        generator.close()


def test_direct_empty_stream_is_unknown_not_safe_to_retry() -> None:
    from agent_runtime.modeling.failures import ModelCallFailed

    generator = _generator(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"data: [DONE]\n\n")
    )
    try:
        with pytest.raises(ModelCallFailed) as raised:
            list(generator.stream_with_tools(messages=[], tools=[]))
        assert raised.value.failure.phase == "stream"
        assert not raised.value.failure.output_started
        assert not raised.value.failure.retryable
        assert raised.value.failure.billing_outcome == "unknown"
    finally:
        generator.close()


def test_sdk_429_real_loopback_server_is_one_attempt() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    calls = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            nonlocal calls
            calls += 1
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = b'{"error":{"code":"rate_limit_exceeded"}}'
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    generator = OpenAICompatibleChatGenerator(
        model="test",
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        api_key="test-only",
    )
    try:
        with pytest.raises(openai.RateLimitError):
            generator.generate_with_tools(messages=[], tools=[])
        assert calls == 1
    finally:
        generator.close()
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_local_context_preflight_has_no_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_runtime.modeling.contracts import LLMCallStage
    from agent_runtime.modeling.gateway import AgentModelResponse, LLMContextOverflowError

    calls = 0

    def provider(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("preflight must not reach the provider")

    generator = _generator(provider)
    model = _model(generator)
    gateway = model._resolved.gateway
    assert gateway is not None

    async def local_overflow(**kwargs: Any) -> AgentModelResponse:
        raise LLMContextOverflowError(stage=LLMCallStage.AGENT_STEP, input_tokens=100, max_input_tokens=50)

    monkeypatch.setattr(gateway, "agenerate_model_request", local_overflow)
    try:
        with pytest.raises(ModelContextOverflowError) as raised:
            asyncio.run(model.dispatch(model.prepare(_request())))
        assert raised.value.failure is None
        assert calls == 0
        assert ModelContextOverflowError("legacy message").failure is None
    finally:
        generator.close()


@pytest.mark.parametrize(
    ("body", "kind", "retryable"),
    [
        ({"code": "rate_limit_exceeded"}, "rate_limited", True),
        ({"error": {"type": "insufficient_quota"}}, "quota_exhausted", False),
        ({"code": "overloaded_error"}, "server_error", True),
        ({"message": "rate limit exceeded"}, "protocol_error", False),
    ],
)
def test_structured_sse_errors_are_classified_without_parsing_message(body, kind, retryable):
    from agent_runtime.modeling.failures import classify_model_failure

    error = openai.APIError("private raw message", request=httpx.Request("POST", "https://provider.invalid"), body=body)
    failure = classify_model_failure(error)
    assert failure.kind == kind
    assert failure.retryable is retryable
    assert failure.billing_outcome == "unknown"
    assert failure.http_status is None
    assert "private" not in failure.message


def test_quota_type_overrides_generic_429_code_and_retry_advice():
    from agent_runtime.modeling.failures import classify_model_failure

    response = httpx.Response(
        429, request=httpx.Request("POST", "https://provider.invalid"), headers={"Retry-After": "0"}
    )
    error = openai.RateLimitError("private", response=response, body={"code": "429", "type": "insufficient_quota"})
    assert classify_model_failure(error).kind == "quota_exhausted"
    assert not classify_model_failure(error).retryable


@pytest.mark.parametrize("advice", ["1.5", "1e2", "+2", "-1", "NaN", "Infinity"])
def test_retry_after_seconds_require_protocol_integer_or_http_date(advice):
    from agent_runtime.modeling.failures import classify_model_failure

    error = _status_error(429, "rate_limit_exceeded", **{"Retry-After": advice})
    assert classify_model_failure(error, now=100).retry_at is None


def test_retry_after_milliseconds_can_express_fractional_milliseconds():
    from agent_runtime.modeling.failures import classify_model_failure

    error = _status_error(429, "rate_limit_exceeded", **{"retry-after-ms": "0.5"})
    assert classify_model_failure(error, now=100).retry_at == 100.0005

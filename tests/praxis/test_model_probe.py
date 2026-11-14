"""adapters.models.probe:探针只走唯一 build_request + 唯一分类器(B6 回归)。

全部离线:httpx.MockTransport。探针不拼 SDK 参数、不进 turn 循环、不写 item;
任何 provider 异常都经 ``classify_failure`` 归一到 detail(固定文案 + kind)。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from praxis.adapters.models.ollama import OllamaAdapter
from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.adapters.models.probe import ProbeOutcome, probe_model
from praxis.protocol.models import ModelSpec, Usage

_SSE_HEADERS = {"content-type": "text/event-stream"}
_BASE = "https://api.deepseek.com/v1"


def _chunk(
    delta: dict[str, Any],
    *,
    finish: str | None = None,
    usage: dict[str, Any] | None = None,
) -> bytes:
    entry: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [],
    }
    if usage is not None:
        entry["usage"] = usage
    else:
        entry["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
    return f"data: {json.dumps(entry)}\n\n".encode()


def _text_sse(text: str = "ok", *, finish: str = "stop") -> bytes:
    return b"".join(
        [
            _chunk({"role": "assistant", "content": text}),
            _chunk({}, finish=finish),
            _chunk({}, usage={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}),
            b"data: [DONE]\n\n",
        ]
    )


def _tool_call_sse(name: str) -> bytes:
    return b"".join(
        [
            _chunk(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": "{}"},
                        }
                    ],
                }
            ),
            _chunk({}, finish="tool_calls"),
            _chunk({}, usage={"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}),
            b"data: [DONE]\n\n",
        ]
    )


def _compat(
    spec: ModelSpec,
    handler: Any,
    *,
    api_key: str = "k",
) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        spec,
        api_key=api_key,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_probe_passes_provider_options_without_typeerror() -> None:
    """B6 回归:探针与 turn 共用同一构造器,不存在"探针自己拼 SDK 参数"的路径。"""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body.get("tools"):
            return httpx.Response(
                200, content=_tool_call_sse("__probe_noop"), headers=_SSE_HEADERS
            )
        return httpx.Response(200, content=_text_sse("ok"), headers=_SSE_HEADERS)

    spec = ModelSpec(
        model_id="deepseek-flash",
        base_url=_BASE,
        max_output_tokens=65536,
        context_window_tokens=131072,
        provider_options={"thinking": {"type": "enabled"}, "top_p": 0.95},
    )
    outcome = asyncio.run(probe_model(_compat(spec, handler)))

    assert outcome.ok is True
    assert outcome.steps == ("text", "tools")
    assert outcome.detail is None
    assert outcome.usage == Usage(11, 22, source="provider")

    text_body, tool_body = seen
    assert text_body["thinking"] == {"type": "enabled"}
    assert text_body["top_p"] == 0.95
    assert text_body["max_tokens"] == 65536
    assert "max_output_tokens" not in text_body
    assert "tools" not in text_body
    assert [tool["function"]["name"] for tool in tool_body["tools"]] == ["__probe_noop"]


def test_probe_missing_credential_detail_is_precise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "invalid api key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

    spec = ModelSpec(model_id="deepseek-flash", base_url=_BASE, api_key_env="DEEPSEEK_API_KEY")
    outcome = asyncio.run(probe_model(_compat(spec, handler, api_key="wrong")))

    assert outcome.ok is False
    assert outcome.steps == ("text",)
    assert outcome.usage is None
    detail = outcome.detail or ""
    assert "credential environment DEEPSEEK_API_KEY is not set" in detail
    assert "authentication" in detail
    # 旧探针的误导归因必须消亡。
    assert "check endpoint, model, and credential environment" not in detail
    assert "unexpected keyword" not in detail


def test_probe_rejected_credential_is_not_reported_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I-1:env 在位但 provider 拒绝时,不得说成"环境变量没设"。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-present-but-provider-rejects")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "invalid api key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

    spec = ModelSpec(model_id="deepseek-flash", base_url=_BASE, api_key_env="DEEPSEEK_API_KEY")
    outcome = asyncio.run(probe_model(_compat(spec, handler, api_key="wrong")))

    assert outcome.ok is False
    detail = outcome.detail or ""
    assert "credential from DEEPSEEK_API_KEY was rejected by the provider" in detail
    assert "is not set" not in detail
    # 值本身绝不进 detail。
    assert "sk-present-but-provider-rejects" not in detail


def test_probe_saved_credential_rejection_is_attributed_to_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-T6-8:file-backed 凭据被拒时不得说成"环境变量没设"。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "invalid api key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

    spec = ModelSpec(model_id="deepseek-flash", base_url=_BASE, api_key_env="DEEPSEEK_API_KEY")
    outcome = asyncio.run(
        probe_model(
            _compat(spec, handler, api_key="wrong"),
            credential_source="store",
        )
    )

    detail = outcome.detail or ""
    assert "saved credential for DEEPSEEK_API_KEY was rejected by the provider" in detail
    assert "is not set" not in detail
    assert "environment" not in detail


def test_probe_missing_credential_source_is_reported_as_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "invalid api key", "code": "invalid_api_key"}},
        )

    spec = ModelSpec(model_id="deepseek-flash", base_url=_BASE, api_key_env="DEEPSEEK_API_KEY")
    outcome = asyncio.run(
        probe_model(
            _compat(spec, handler, api_key="wrong"),
            credential_source="missing",
        )
    )

    detail = outcome.detail or ""
    assert "no credential available for DEEPSEEK_API_KEY" in detail
    assert "is not set" not in detail


def test_probe_skips_tool_step_when_spec_says_no_tool_support() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_text_sse("ok"), headers=_SSE_HEADERS)

    spec = ModelSpec(model_id="tiny", base_url=_BASE, supports_tools=False)
    outcome = asyncio.run(probe_model(_compat(spec, handler)))

    assert outcome.ok is True
    assert outcome.steps == ("text",)
    assert len(seen) == 1


def test_probe_tool_step_accepts_natural_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = "I cannot call tools." if body.get("tools") else "ok"
        return httpx.Response(200, content=_text_sse(text), headers=_SSE_HEADERS)

    outcome = asyncio.run(probe_model(_compat(ModelSpec(model_id="m", base_url=_BASE), handler)))

    assert outcome.ok is True
    assert outcome.steps == ("text", "tools")


def test_probe_tool_step_neither_call_nor_text_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = _text_sse("", finish="stop") if body.get("tools") else _text_sse("ok")
        return httpx.Response(200, content=content, headers=_SSE_HEADERS)

    outcome = asyncio.run(probe_model(_compat(ModelSpec(model_id="m", base_url=_BASE), handler)))

    assert outcome.ok is False
    assert outcome.steps == ("text", "tools")
    assert "tools step failed" in (outcome.detail or "")
    assert "neither text nor tool calls" in (outcome.detail or "")


def test_probe_tool_step_fails_when_assembler_dropped_arguments() -> None:
    """#7:装配器的 "[invalid tool call arguments dropped: N]" 不是自然拒答。"""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("tools"):
            return httpx.Response(
                200,
                content=b"".join(
                    [
                        _chunk(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "__probe_noop",
                                            "arguments": "not-json{",
                                        },
                                    }
                                ],
                            }
                        ),
                        _chunk({}, finish="tool_calls"),
                        b"data: [DONE]\n\n",
                    ]
                ),
                headers=_SSE_HEADERS,
            )
        return httpx.Response(200, content=_text_sse("ok"), headers=_SSE_HEADERS)

    outcome = asyncio.run(probe_model(_compat(ModelSpec(model_id="m", base_url=_BASE), handler)))

    assert outcome.ok is False
    assert outcome.steps == ("text", "tools")
    assert "malformed tool call arguments" in (outcome.detail or "")
    assert "invalid tool call arguments dropped" not in (outcome.detail or "")


def test_probe_local_configuration_errors_are_not_provider_protocol_errors() -> None:
    """#8:坏 provider_options 是本地配置错误,不能被说成 provider 协议违规。"""
    adapter = OllamaAdapter(
        ModelSpec(
            model_id="qwen2.5:7b",
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            provider_options={"stream": False},
        )
    )
    outcome = asyncio.run(probe_model(adapter))

    assert outcome.ok is False
    assert outcome.steps == ("text",)
    assert outcome.usage is None
    detail = outcome.detail or ""
    assert "local model configuration error" in detail
    assert "wire-owned field: stream" in detail
    assert "Provider response violated the model protocol" not in detail
    assert "(protocol_error)" not in detail


def test_probe_text_transport_failure_names_root_cause() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    outcome = asyncio.run(probe_model(_compat(ModelSpec(model_id="m", base_url=_BASE), handler)))

    assert outcome.ok is False
    detail = outcome.detail or ""
    assert detail.startswith("text step failed:")
    assert "Provider connection failed before the request was sent." in detail
    assert "(transport_not_sent)" in detail
    assert "check endpoint, model, and credential environment" not in detail


def test_probe_runs_through_ollama_native_stream() -> None:
    frames = [
        {"message": {"role": "assistant", "content": "ok"}, "done": True,
         "prompt_eval_count": 2, "eval_count": 3}
    ]
    body = b"".join(json.dumps(frame).encode() + b"\n" for frame in frames)
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

    adapter = OllamaAdapter(
        ModelSpec(model_id="qwen2.5:7b", provider="ollama", base_url="http://127.0.0.1:11434"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    outcome = asyncio.run(probe_model(adapter))

    assert outcome.ok is True
    assert outcome.steps == ("text", "tools")
    assert outcome.usage == Usage(2, 3, source="provider")
    assert "tools" not in seen[0]
    assert [tool["function"]["name"] for tool in seen[1]["tools"]] == ["__probe_noop"]


def test_probe_outcome_is_frozen_data() -> None:
    outcome = ProbeOutcome(ok=True, steps=("text",), detail=None, usage=None)
    with pytest.raises(AttributeError):
        outcome.ok = False  # type: ignore[misc]

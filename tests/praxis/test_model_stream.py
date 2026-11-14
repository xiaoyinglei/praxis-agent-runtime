"""adapters.models:唯一出站 payload + 真流式(B6 根因面/B9/B24/B25;设计 §5.7/§5.9)。

全部离线:httpx.MockTransport 返回真实 SSE 帧,adapter 经 openai SDK 增量消费;
``complete()`` 与 ``stream()`` 共享同一条装配路径。增量性用"可观测消费进度"
钉住(自定义 AsyncByteStream 记录已产出帧数),不依赖 sleep/时间戳精度。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from openai.types.chat import ChatCompletion

from praxis.adapters.models.base import (
    WIRE_OWNED_FIELDS,
    collect_events,
    openai_payload,
    wire_json,
)
from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.protocol.models import (
    ModelEvent,
    ModelMessage,
    ModelSettings,
    ModelSpec,
    ModelStreamFinished,
    ModelTextDelta,
    ModelToolCallDelta,
    Usage,
    WireRequest,
    assemble_model_events,
)

_SSE_HEADERS = {"content-type": "text/event-stream"}


def _chunk(
    delta: dict[str, Any],
    *,
    finish: str | None = None,
    usage: dict[str, Any] | None = None,
    chunk_id: str = "chatcmpl-1",
) -> bytes:
    entry: dict[str, Any] = {
        "id": chunk_id,
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


def sse_transport(chunks: list[bytes]) -> httpx.MockTransport:
    """200 + text/event-stream,帧按顺序一次性交给 SDK(非增量测试用)。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"".join(chunks), headers=_SSE_HEADERS)

    return httpx.MockTransport(handler)


class _CountingSSEStream(httpx.AsyncByteStream):
    """记录"已产出到第几帧"的 SSE 迭代体:全量缓冲实现会把计数推满再产出事件。"""

    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames
        self.produced = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self.frames:
            self.produced += 1
            yield frame


def _stream_adapter(transport: httpx.MockTransport) -> OpenAICompatAdapter:
    client = httpx.AsyncClient(transport=transport)
    return OpenAICompatAdapter(
        ModelSpec(model_id="m", base_url="https://unit.test/v1"),
        api_key="k",
        http_client=client,
    )


def _request(adapter: OpenAICompatAdapter) -> Any:
    return adapter.build_request([ModelMessage(role="user", content="hi")], (), ModelSettings())


# --- B25:增量流式,不是缓冲伪装 ------------------------------------------------


def test_stream_is_incremental_not_buffered() -> None:
    frames = [
        _chunk({"role": "assistant", "content": "he"}),
        _chunk({"content": "llo"}),
        _chunk({}, finish="stop"),
        _chunk({}, usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
        b"data: [DONE]\n\n",
    ]
    body = _CountingSSEStream(frames)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers=_SSE_HEADERS, stream=body)
    )
    adapter = _stream_adapter(transport)
    req = _request(adapter)

    async def drive() -> tuple[list[ModelEvent], list[str], int | None]:
        seen: list[ModelEvent] = []
        texts: list[str] = []
        first_delta_progress: int | None = None
        async for event in adapter.stream(req):
            seen.append(event)
            if isinstance(event, ModelTextDelta):
                texts.append(event.text)
                if first_delta_progress is None:
                    first_delta_progress = body.produced
        return seen, texts, first_delta_progress

    seen, texts, first_delta_progress = asyncio.run(drive())
    # 首帧在最后一帧产出之前已被消费;全量缓冲实现会在此失败(计数已等于总帧数)。
    assert first_delta_progress == 1
    assert first_delta_progress is not None and first_delta_progress < len(frames)
    assert "".join(texts) == "hello"
    assert isinstance(seen[-1], ModelStreamFinished)


# --- B9:tool 分片全程下发,装配到 finish ---------------------------------------


def test_tool_name_split_across_frames_assembles() -> None:
    frames = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":'},
                    }
                ],
            }
        ),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"name": "_file", "arguments": ' "a.txt"}'},
                    }
                ]
            }
        ),
        _chunk({}, finish="tool_calls"),
        _chunk({}, usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}),
        b"data: [DONE]\n\n",
    ]
    adapter = _stream_adapter(sse_transport(frames))
    events = asyncio.run(collect_events(adapter.stream(_request(adapter))))
    response = assemble_model_events(events)
    assert [
        (call.call_id, call.name, dict(call.arguments)) for call in response.tool_calls
    ] == [("call_1", "read_file", {"path": "a.txt"})]
    assert response.finish_reason == "tool_calls"


def test_stream_emits_every_tool_fragment_not_a_frozen_first_frame() -> None:
    """分片原样下发:adapter 不预装配,首帧的 name/arguments 不会冻结。"""
    frames = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "read", "arguments": "{"},
                    }
                ],
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"name": "_file", "arguments": "}"}}]}),
        _chunk({}, finish="stop"),
        b"data: [DONE]\n\n",
    ]
    adapter = _stream_adapter(sse_transport(frames))
    events = asyncio.run(collect_events(adapter.stream(_request(adapter))))
    fragments = [event for event in events if isinstance(event, ModelToolCallDelta)]
    assert [(f.call_id, f.name_fragment, f.arguments_fragment) for f in fragments] == [
        ("call_1", "read", "{"),
        (None, "_file", "}"),
    ]


# --- provider 怪癖:reasoning / usage / identity -------------------------------


def test_stream_reasoning_delta_stays_out_of_content() -> None:
    frames = [
        _chunk({"role": "assistant", "reasoning_content": "think ", "content": "hi"}),
        _chunk({}, finish="stop"),
        b"data: [DONE]\n\n",
    ]
    adapter = _stream_adapter(sse_transport(frames))
    events = asyncio.run(collect_events(adapter.stream(_request(adapter))))
    response = assemble_model_events(events)
    assert response.content == "hi"
    assert response.reasoning == "think "


def test_stream_maps_provider_usage_and_finish() -> None:
    frames = [
        _chunk({"role": "assistant", "content": "ok"}),
        _chunk({}, finish="stop"),
        _chunk(
            {},
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            }
        ),
        b"data: [DONE]\n\n",
    ]
    adapter = _stream_adapter(sse_transport(frames))
    events = asyncio.run(collect_events(adapter.stream(_request(adapter))))
    assert events[-1] == ModelStreamFinished(
        Usage(5, 7, cached_input_tokens=3, reasoning_tokens=2, source="provider"), "stop"
    )


def test_stream_without_usage_is_estimated_zero() -> None:
    frames = [_chunk({"role": "assistant", "content": "ok"}, finish="stop"), b"data: [DONE]\n\n"]
    adapter = _stream_adapter(sse_transport(frames))
    events = asyncio.run(collect_events(adapter.stream(_request(adapter))))
    assert assemble_model_events(events).usage == Usage(0, 0, source="estimated")


def test_stream_rejects_identity_change_before_finish() -> None:
    frames = [
        _chunk({"role": "assistant", "content": "a"}, chunk_id="chatcmpl-a"),
        _chunk({"content": "b"}, chunk_id="chatcmpl-b"),
        _chunk({}, finish="stop", chunk_id="chatcmpl-b"),
        b"data: [DONE]\n\n",
    ]
    adapter = _stream_adapter(sse_transport(frames))

    with pytest.raises(ValueError, match="provider changed response identity during stream"):
        asyncio.run(collect_events(adapter.stream(_request(adapter))))


# --- provider 忽略 stream:诚实整体,不是缓冲伪装 ---------------------------------


class _WholeCompletions:
    def __init__(self, completion: ChatCompletion) -> None:
        self._completion = completion

    async def create(self, **kwargs: Any) -> ChatCompletion:
        return self._completion


class _WholeChat:
    def __init__(self, completion: ChatCompletion) -> None:
        self.completions = _WholeCompletions(completion)


class _WholeClient:
    """模拟 SDK 在 provider 忽略 stream 时返回的完整对象(非异步可迭代)。"""

    def __init__(self, completion: ChatCompletion) -> None:
        self.chat = _WholeChat(completion)


def _whole_completion() -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
        }
    )


def test_stream_converts_whole_completion_to_single_delta_plus_finish() -> None:
    adapter = OpenAICompatAdapter(ModelSpec(model_id="m"), api_key="k")
    adapter._client = _WholeClient(_whole_completion())  # type: ignore[assignment]
    events = asyncio.run(collect_events(adapter.stream(_request(adapter))))
    assert events == [
        ModelTextDelta("ok"),
        ModelStreamFinished(Usage(11, 22, source="provider"), "stop"),
    ]


def test_non_sse_provider_response_fails_loudly() -> None:
    """provider 不产 SSE 帧时不得静默变成成功空回复:响亮失败,而非空 content。"""
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    adapter = _stream_adapter(
        httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )
    with pytest.raises(ValueError, match="provider returned no stream events"):
        asyncio.run(collect_events(adapter.stream(_request(adapter))))


def test_empty_text_with_tool_call_is_not_an_error() -> None:
    """正常空文本(只有 tool_calls)绝不能被当成"没有流事件"。"""
    frames = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        ),
        _chunk({}, finish="tool_calls"),
        b"data: [DONE]\n\n",
    ]
    adapter = _stream_adapter(sse_transport(frames))
    response = assemble_model_events(asyncio.run(collect_events(adapter.stream(_request(adapter)))))
    assert response.content == ""
    assert [call.name for call in response.tool_calls] == ["read_file"]


# --- B6/B24:唯一出站 body 构造 -------------------------------------------------


def _wire_request(**overrides: Any) -> WireRequest:
    values: dict[str, Any] = {
        "model": "m",
        "messages": (ModelMessage(role="user", content="hi"),),
        "tools": (),
        "max_output_tokens": 10,
        "provider_options": {},
        "temperature": None,
    }
    values.update(overrides)
    return WireRequest(**values)


@pytest.mark.parametrize("key", sorted(WIRE_OWNED_FIELDS))
def test_provider_options_cannot_override_wire_owned(key: str) -> None:
    with pytest.raises(ValueError, match="wire-owned field"):
        openai_payload(_wire_request(provider_options={key: None}), stream=False)


def test_extra_body_catches_unknown_provider_options() -> None:
    payload = openai_payload(
        _wire_request(provider_options={"thinking": {"type": "enabled"}}), stream=False
    )
    assert payload["extra_body"] == {"thinking": {"type": "enabled"}}


def test_provider_option_extra_body_merges_instead_of_nesting() -> None:
    # provider_options 里显式写 extra_body 是受支持形状:合并进顶层 extra_body。
    payload = openai_payload(
        _wire_request(
            provider_options={
                "extra_body": {"thinking": {"type": "enabled"}},
                "x": 1,
            }
        ),
        stream=False,
    )
    assert payload["extra_body"] == {"thinking": {"type": "enabled"}, "x": 1}


def test_provider_option_extra_body_must_be_an_object() -> None:
    with pytest.raises(ValueError, match="extra_body"):
        openai_payload(_wire_request(provider_options={"extra_body": [1, 2]}), stream=False)


def test_known_sdk_options_land_top_level() -> None:
    payload = openai_payload(
        _wire_request(
            provider_options={
                "top_p": 0.5,
                "seed": 7,
                "parallel_tool_calls": False,
                "reasoning_effort": "low",
            }
        ),
        stream=False,
    )
    assert (payload["top_p"], payload["seed"]) == (0.5, 7)
    assert payload["parallel_tool_calls"] is False
    assert payload["reasoning_effort"] == "low"
    assert "extra_body" not in payload


def test_stream_flag_adds_include_usage_once() -> None:
    payload = openai_payload(_wire_request(), stream=True)
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    non_stream = openai_payload(_wire_request(), stream=False)
    assert "stream" not in non_stream and "stream_options" not in non_stream


def test_build_request_accounting_text_matches_the_recorded_request_body() -> None:
    """D2 真钉子:serialized_json 与真正发出的 HTTP body 结构一致。

    ``extra_body`` 是 SDK kwargs 形状,SDK 会把它展平进 body;因此比较时把
    记账文本里的 ``extra_body`` 展平后再对照(唯一允许的字节差异,见
    ``openai_payload`` docstring)。
    """
    seen: list[httpx.Request] = []
    frames = [_chunk({"role": "assistant", "content": "ok"}, finish="stop"), b"data: [DONE]\n\n"]
    body = b"".join(frames)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body, headers=_SSE_HEADERS)

    adapter = OpenAICompatAdapter(
        ModelSpec(
            model_id="alias",
            remote_model_id="real/model",
            base_url="https://unit.test/v1",
            provider_options={"extra_body": {"thinking": {"type": "enabled"}}, "x": 1},
        ),
        api_key="k",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    req = adapter.build_request(
        [ModelMessage(role="user", content="hi")], (), ModelSettings(temperature=0.2)
    )
    asyncio.run(collect_events(adapter.stream(req)))

    accounted = json.loads(req.serialized_json)
    recorded = json.loads(seen[0].content)
    extra = accounted.pop("extra_body")
    assert accounted["stream"] is True
    assert accounted["stream_options"] == {"include_usage": True}
    assert extra == {"thinking": {"type": "enabled"}, "x": 1}
    assert "extra_body" not in recorded
    assert recorded == {**accounted, **extra}


def test_wire_json_is_deterministic_and_rejects_non_json_values() -> None:
    payload = {"b": 2, "a": {"y": 1, "x": 0}}
    assert wire_json(payload) == '{"a":{"x":0,"y":1},"b":2}'
    with pytest.raises(ValueError, match="not JSON-serializable") as excinfo:
        wire_json({"bad": object()})
    # 有界错误:不回显整个值,只报类型。
    assert "<object object at" not in str(excinfo.value)

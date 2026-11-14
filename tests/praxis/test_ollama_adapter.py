"""adapters.models.ollama:原生 /api/chat NDJSON 流与 tool calling(B7/R-T4-2)。

全部离线:httpx.MockTransport 返回 NDJSON 帧;装配仍走唯一 ``assemble_model_events``。
请求体的记账文本就是真正发出去的字节(``content=`` 而非 SDK ``json=``),
因此 D2 在这里是逐字节成立,而不是"结构可还原"。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from praxis.adapters.models.base import collect_events
from praxis.adapters.models.ollama import OllamaAdapter, ToolsNotSupportedError
from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.composition import _adapter_for
from praxis.protocol.models import (
    ModelMessage,
    ModelReasoningDelta,
    ModelSettings,
    ModelSpec,
    ModelStreamFinished,
    ModelTextDelta,
    ModelToolCall,
    ModelToolCallDelta,
    Usage,
    assemble_model_events,
)
from praxis.protocol.tools import ToolDefinition

_LOCAL_BASE = "http://127.0.0.1:11434"


def _spec(**overrides: Any) -> ModelSpec:
    values: dict[str, Any] = {
        "model_id": "qwen2.5:7b",
        "provider": "ollama",
        "base_url": _LOCAL_BASE,
    }
    values.update(overrides)
    return ModelSpec(**values)


def _tool(name: str = "read_file") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="read a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )


def _ndjson(frames: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(frame).encode() + b"\n" for frame in frames)


def _adapter(
    frames: list[dict[str, Any]] | None = None,
    *,
    spec: ModelSpec | None = None,
    body: bytes | None = None,
    status: int = 200,
) -> tuple[OllamaAdapter, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    payload = body if body is not None else _ndjson(frames or [])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            status, content=payload, headers={"content-type": "application/x-ndjson"}
        )

    adapter = OllamaAdapter(
        spec or _spec(), http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return adapter, seen


def _request(adapter: OllamaAdapter, tools: tuple[ToolDefinition, ...] = ()) -> Any:
    return adapter.build_request(
        [ModelMessage(role="user", content="hi")], tools, ModelSettings()
    )


# --- 请求体:唯一构造器,原生 tool 形状 ------------------------------------------


def test_ollama_request_body_carries_tools_and_num_predict() -> None:
    adapter = OllamaAdapter(_spec())
    req = adapter.build_request(
        [ModelMessage(role="user", content="hi")], [_tool()], ModelSettings()
    )
    body = json.loads(req.serialized_json)
    assert body["model"] == "qwen2.5:7b"
    assert body["stream"] is True
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert body["tools"][0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }
    assert body["options"]["num_predict"] == 4096
    assert "temperature" not in body["options"]
    assert req.wire_revision == "praxis-ollama-chat-v1"
    assert req.tokenizer_model is None


def test_ollama_request_options_carry_temperature_top_p_and_seed() -> None:
    adapter = OllamaAdapter(_spec(provider_options={"top_p": 0.9, "seed": 7}))
    req = adapter.build_request(
        [ModelMessage(role="user", content="hi")],
        (),
        ModelSettings(temperature=0.2, max_output_tokens=128),
    )
    body = json.loads(req.serialized_json)
    assert body["options"] == {
        "temperature": 0.2,
        "num_predict": 128,
        "top_p": 0.9,
        "seed": 7,
    }
    assert "tools" not in body


def test_ollama_tokenizer_model_travels_with_the_wire_request() -> None:
    adapter = OllamaAdapter(_spec(tokenizer_model="qwen2.5:7b"))
    req = adapter.build_request([ModelMessage(role="user", content="hi")], (), ModelSettings())
    assert req.tokenizer_model == "qwen2.5:7b"


def test_ollama_provider_options_cannot_override_wire_owned_fields() -> None:
    for key in ("stream", "model", "messages", "tools", "num_predict", "options"):
        adapter = OllamaAdapter(_spec(provider_options={key: None}))
        with pytest.raises(ValueError, match="wire-owned field"):
            adapter.build_request([ModelMessage(role="user", content="hi")], (), ModelSettings())


def test_ollama_rejects_unknown_provider_options_instead_of_a_second_merge_rule() -> None:
    adapter = OllamaAdapter(_spec(provider_options={"mystery": 1}))
    with pytest.raises(ValueError, match="unsupported ollama provider option: mystery"):
        adapter.build_request([ModelMessage(role="user", content="hi")], (), ModelSettings())


def test_ollama_messages_use_object_arguments_not_json_strings() -> None:
    call = ModelToolCall(call_id="c1", name="read_file", arguments={"path": "a.txt"})
    adapter = OllamaAdapter(_spec())
    req = adapter.build_request(
        [
            ModelMessage(role="assistant", content="", tool_calls=(call,)),
            ModelMessage(role="tool", content="ok", tool_call_id="c1"),
        ],
        (),
        ModelSettings(),
    )
    body = json.loads(req.serialized_json)
    assert body["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.txt"}
    assert body["messages"][1] == {"role": "tool", "content": "ok"}


# --- B7:无 tool 能力绝不降级成 prompt 解析 ---------------------------------------


def test_no_tool_models_reject_tool_requests() -> None:
    adapter = OllamaAdapter(_spec(supports_tools=False))
    with pytest.raises(ToolsNotSupportedError, match="without tool support"):
        adapter.build_request([ModelMessage(role="user", content="hi")], [_tool()], ModelSettings())
    # 不带工具时同一适配器仍可用,请求里不会出现 tools 键。
    req = adapter.build_request([ModelMessage(role="user", content="hi")], (), ModelSettings())
    assert req.tools == ()
    assert "tools" not in json.loads(req.serialized_json)


# --- 应答:NDJSON 逐帧 → 唯一装配点 ---------------------------------------------


def test_ollama_tool_calls_accumulate_like_openai() -> None:
    adapter, _ = _adapter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}
                    ],
                },
                "done": False,
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "list_dir", "arguments": {}}}],
                },
                "done": False,
            },
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 3,
                "eval_count": 4,
            },
        ]
    )
    response = asyncio.run(adapter.complete(_request(adapter)))
    assert [
        (call.call_id, call.name, dict(call.arguments)) for call in response.tool_calls
    ] == [
        ("ollama_call_0", "read_file", {"path": "a.txt"}),
        ("ollama_call_1", "list_dir", {}),
    ]
    assert response.content == ""
    assert response.usage == Usage(3, 4, source="provider")
    assert response.finish_reason == "stop"


def test_ollama_stream_emits_text_reasoning_and_finish() -> None:
    adapter, _ = _adapter(
        [
            {"message": {"role": "assistant", "reasoning": "think ", "content": "he"}, "done": False},
            {"message": {"role": "assistant", "content": "llo"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 7,
            },
        ]
    )
    req = _request(adapter)
    events = asyncio.run(collect_events(adapter.stream(req)))
    assert events == [
        ModelReasoningDelta("think "),
        ModelTextDelta("he"),
        ModelTextDelta("llo"),
        ModelStreamFinished(Usage(5, 7, source="provider"), "stop"),
    ]
    response = assemble_model_events(events)
    assert response.content == "hello"
    assert response.reasoning == "think "
    assert response.usage == Usage(5, 7, source="provider")
    assert events[-1] == ModelStreamFinished(Usage(5, 7, source="provider"), "stop")


def test_ollama_tool_delta_arguments_are_json_encoded_fragments() -> None:
    adapter, _ = _adapter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a"}}}],
                },
                "done": False,
            },
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ]
    )
    events = asyncio.run(collect_events(adapter.stream(_request(adapter))))
    (delta,) = [event for event in events if isinstance(event, ModelToolCallDelta)]
    assert (delta.index, delta.call_id, delta.name_fragment) == (0, "ollama_call_0", "read_file")
    assert json.loads(delta.arguments_fragment) == {"path": "a"}


def test_ollama_missing_usage_counts_is_estimated_zero() -> None:
    adapter, _ = _adapter(
        [
            {"message": {"role": "assistant", "content": "ok"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ]
    )
    response = asyncio.run(adapter.complete(_request(adapter)))
    assert response.usage == Usage(0, 0, source="estimated")


def test_ollama_stream_sends_exact_accounted_bytes_without_auth_header() -> None:
    adapter, seen = _adapter(
        [{"message": {"role": "assistant", "content": "ok"}, "done": True}],
        spec=_spec(provider_options={"top_p": 0.95}),
    )
    req = adapter.build_request([ModelMessage(role="user", content="hi")], [_tool()], ModelSettings())
    asyncio.run(collect_events(adapter.stream(req)))
    assert str(seen[-1].url) == f"{_LOCAL_BASE}/api/chat"
    assert seen[-1].method == "POST"
    assert seen[-1].content.decode() == req.serialized_json
    assert seen[-1].headers.get("authorization") is None
    assert seen[-1].headers["content-type"] == "application/json"


# --- R-T4-2:零帧必须响亮失败 ----------------------------------------------------


def test_ollama_stream_without_any_frame_fails_loudly() -> None:
    adapter, _ = _adapter(body=b"")
    with pytest.raises(ValueError, match="may not support streaming") as excinfo:
        asyncio.run(collect_events(adapter.stream(_request(adapter))))
    assert "qwen2.5:7b" in str(excinfo.value)


def test_ollama_blank_lines_do_not_count_as_frames() -> None:
    adapter, _ = _adapter(body=b"\n  \n\n")
    with pytest.raises(ValueError, match="no stream events"):
        asyncio.run(collect_events(adapter.stream(_request(adapter))))


def test_ollama_malformed_frame_fails_with_bounded_message() -> None:
    adapter, _ = _adapter(body=b'{"message": not-json}\n')
    with pytest.raises(ValueError, match="malformed stream frame") as excinfo:
        asyncio.run(collect_events(adapter.stream(_request(adapter))))
    assert "not-json" not in str(excinfo.value)


# --- base_url 必须是 loopback/私有主机 ------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:11434", "http://127.0.0.1:11434", "http://192.168.1.5:11434", "http://[::1]:11434"],
)
def test_ollama_accepts_loopback_and_private_hosts(base_url: str) -> None:
    OllamaAdapter(_spec(base_url=base_url))


@pytest.mark.parametrize(
    "base_url",
    ["https://api.example.com/v1", "http://10.0.0.1.evil.test:11434", "not-a-url"],
)
def test_ollama_rejects_public_hosts_and_bad_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        OllamaAdapter(_spec(base_url=base_url))


def test_ollama_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OllamaAdapter(_spec(base_url=None))


def test_ollama_rejects_non_ollama_spec() -> None:
    with pytest.raises(ValueError, match="provider"):
        OllamaAdapter(_spec(provider="openai-compatible"))


# --- composition:按 provider 的唯一分派点 ---------------------------------------


def test_composition_dispatches_adapters_by_provider() -> None:
    ollama = _adapter_for(_spec())
    assert isinstance(ollama, OllamaAdapter)
    compat = _adapter_for(
        ModelSpec(model_id="m", base_url="https://unit.test/v1"), "k"
    )
    assert isinstance(compat, OpenAICompatAdapter)
    with pytest.raises(ValueError, match="requires an API key"):
        _adapter_for(ModelSpec(model_id="m", base_url="https://unit.test/v1"))
    with pytest.raises(ValueError, match="scripted"):
        _adapter_for(ModelSpec(model_id="m", provider="scripted"))
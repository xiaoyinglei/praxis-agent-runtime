"""adapters.models:唯一 build_request 合并、OpenAI 兼容非流式解析、脚本驱动(设计 §5.9 最小面)。

全部离线:真实 provider 调用用 httpx.MockTransport 替身,不触网。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from praxis.adapters.models.base import to_openai_messages, to_openai_tools
from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.adapters.models.scripted import ScriptExhausted, ScriptedAdapter
from praxis.protocol.models import (
    ModelAdapter,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelSpec,
    ModelToolCall,
    Usage,
)
from praxis.protocol.tools import ToolDefinition


def _tool_response(
    text: str,
    calls: tuple[tuple[str, str, dict[str, Any]], ...] = (),
    *,
    usage: tuple[int, int] = (0, 0),
    finish: str = "stop",
) -> ModelResponse:
    return ModelResponse(
        content=text,
        tool_calls=tuple(
            ModelToolCall(call_id=cid, name=name, arguments=args) for cid, name, args in calls
        ),
        usage=Usage(*usage),
        finish_reason=finish,
    )


def _mock_transport(
    payload: dict[str, Any],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler), seen


def _openai_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "list_dir", "arguments": "{}"},
                        },
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
    }
    payload.update(overrides)
    return payload


# --- build_request:合并只发生一次(B6 结构性预防) --------------------------------


def test_build_request_merges_provider_options_once() -> None:
    spec = ModelSpec(model_id="m", provider_options={"extra_body": {"x": 1}}, max_output_tokens=100)
    adapter = OpenAICompatAdapter(spec, api_key="k")
    req = adapter.build_request(
        [ModelMessage(role="user", content="hi")], [], ModelSettings(max_output_tokens=200)
    )
    assert req.provider_options == {"extra_body": {"x": 1}} and req.max_output_tokens == 200
    assert req.model == "m"
    assert req.temperature is None
    assert len(req.messages) == 1 and req.tools == ()


def test_build_request_falls_back_to_spec_max_output_tokens() -> None:
    spec = ModelSpec(model_id="m", max_output_tokens=100)
    adapter = OpenAICompatAdapter(spec, api_key="k")
    req = adapter.build_request([], [], ModelSettings(temperature=0.3))
    assert req.max_output_tokens == 100
    assert req.temperature == 0.3
    assert dict(req.provider_options) == {}


def test_build_request_copies_provider_options() -> None:
    options: dict[str, Any] = {"extra_body": {"x": 1}}
    spec = ModelSpec(model_id="m", provider_options=options)
    adapter = OpenAICompatAdapter(spec, api_key="k")
    req = adapter.build_request([], [], ModelSettings())
    assert isinstance(req.provider_options, dict)
    options["extra_body"] = {"x": 2}  # 顶层键改动不影响已构建的 WireRequest(浅拷贝)
    assert req.provider_options == {"extra_body": {"x": 1}}


# --- complete:非流式解析(MockTransport,离线) ----------------------------------


def test_complete_parses_tool_calls() -> None:
    transport, seen = _mock_transport(_openai_payload())
    client = httpx.AsyncClient(transport=transport)
    adapter = OpenAICompatAdapter(
        ModelSpec(model_id="m"), api_key="k", base_url="https://unit.test/v1", http_client=client
    )
    assert isinstance(adapter, ModelAdapter)
    req = adapter.build_request([ModelMessage(role="user", content="hi")], [], ModelSettings())
    response = asyncio.run(adapter.complete(req))
    assert response.tool_calls == (
        ModelToolCall(call_id="call_1", name="read_file", arguments={"path": "a.txt"}),
        ModelToolCall(call_id="call_2", name="list_dir", arguments={}),
    )
    assert response.content == ""  # content 为 None → ""
    assert response.usage == Usage(11, 22)
    assert response.finish_reason == "tool_calls"
    body = json.loads(seen[0].content)
    assert body["model"] == "m"
    assert body["max_tokens"] == 4096
    assert "tools" not in body  # 空 tools 省略
    assert "temperature" not in body  # None 省略


def test_complete_passes_provider_options_to_sdk_once() -> None:
    payload = _openai_payload(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ]
    )
    transport, seen = _mock_transport(payload)
    client = httpx.AsyncClient(transport=transport)
    spec = ModelSpec(model_id="m", provider_options={"extra_body": {"x": 1}}, max_output_tokens=77)
    adapter = OpenAICompatAdapter(
        spec, api_key="k", base_url="https://unit.test/v1", http_client=client
    )
    tool = ToolDefinition(name="read_file", description="read", input_schema={"type": "object"})
    req = adapter.build_request(
        [ModelMessage(role="user", content="hi")], [tool], ModelSettings(temperature=0.5)
    )
    response = asyncio.run(adapter.complete(req))
    assert response == ModelResponse(
        content="ok", tool_calls=(), usage=Usage(11, 22), finish_reason="stop"
    )
    body = json.loads(seen[0].content)
    assert body["x"] == 1  # extra_body 合并进请求体,只出现一次
    assert body["max_tokens"] == 77
    assert body["temperature"] == 0.5
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_complete_rejects_malformed_tool_call_arguments() -> None:
    payload = _openai_payload()
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "not-json{"
    transport, _ = _mock_transport(payload)
    client = httpx.AsyncClient(transport=transport)
    adapter = OpenAICompatAdapter(
        ModelSpec(model_id="m"), api_key="k", base_url="https://unit.test/v1", http_client=client
    )
    req = adapter.build_request([ModelMessage(role="user", content="hi")], [], ModelSettings())
    with pytest.raises(ValueError, match="tool_calls\\[0\\]"):
        asyncio.run(adapter.complete(req))


def test_complete_missing_usage_yields_zero_usage() -> None:
    payload = _openai_payload(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"},
            }
        ],
    )
    payload.pop("usage")
    transport, _ = _mock_transport(payload)
    client = httpx.AsyncClient(transport=transport)
    adapter = OpenAICompatAdapter(
        ModelSpec(model_id="m"), api_key="k", base_url="https://unit.test/v1", http_client=client
    )
    req = adapter.build_request([ModelMessage(role="user", content="hi")], [], ModelSettings())
    response = asyncio.run(adapter.complete(req))
    assert response.usage == Usage(0, 0)
    assert response.content == "hello"
    assert response.tool_calls == ()


# --- ScriptedAdapter:确定性驱动 --------------------------------------------------


def test_scripted_adapter_order_and_exhaustion() -> None:
    r1 = _tool_response("first", (("c1", "read_file", {"path": "a"}),), usage=(1, 2))
    r2 = _tool_response("second")
    adapter = ScriptedAdapter([r1, r2])
    assert isinstance(adapter, ModelAdapter)

    async def drive() -> None:
        req = adapter.build_request(
            [ModelMessage(role="user", content="go")], [], ModelSettings(max_output_tokens=64)
        )
        assert adapter.last_request is req
        assert req.max_output_tokens == 64
        assert await adapter.complete(req) is r1
        assert await adapter.complete(req) is r2
        with pytest.raises(ScriptExhausted):
            await adapter.complete(req)

    asyncio.run(drive())


def test_scripted_adapter_delay_paces_completion() -> None:
    adapter = ScriptedAdapter([_tool_response("slow")], delay_seconds=0.05)
    req = adapter.build_request([], [], ModelSettings())
    started = time.monotonic()
    asyncio.run(adapter.complete(req))
    assert time.monotonic() - started >= 0.04


# --- base:唯一序列化点 ------------------------------------------------------------


def test_to_openai_messages_roles_and_tool_fields() -> None:
    call = ModelToolCall(call_id="c1", name="read_file", arguments={"path": "a.txt"})
    messages = [
        ModelMessage(role="system", content="sys"),
        ModelMessage(role="user", content="hi"),
        ModelMessage(role="assistant", content="", tool_calls=(call,)),
        ModelMessage(role="tool", content="result", tool_call_id="c1"),
    ]
    out = to_openai_messages(messages)
    assert out[0] == {"role": "system", "content": "sys"}
    assert "tool_calls" not in out[1] and "tool_call_id" not in out[1]
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
        }
    ]
    assert isinstance(out[2]["tool_calls"][0]["function"]["arguments"], str)
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.txt"}
    assert out[3]["tool_call_id"] == "c1"
    assert out[3]["content"] == "result"


def test_to_openai_tools_thaws_input_schema() -> None:
    tool = ToolDefinition(
        name="read_file",
        description="Read a file.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    (entry,) = to_openai_tools([tool])
    assert entry["type"] == "function"
    function = entry["function"]
    assert function["name"] == "read_file"
    assert function["description"] == "Read a file."
    assert function["parameters"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }
    assert type(function["parameters"]) is dict  # thaw 后为可变 plain dict


def test_to_openai_helpers_are_pure_empty_inputs() -> None:
    assert to_openai_messages([]) == []
    assert to_openai_tools([]) == []

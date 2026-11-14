"""Ollama 原生适配器:``/api/chat`` NDJSON 流式 + 原生 tool calling(设计 §5.9;D3/D9)。

出站 body 的唯一构造点是本模块的 ``ollama_payload``——它是 provider 的 wire 形状,
不是 OpenAI 兼容形状(``options.num_predict``、assistant ``tool_calls`` 的 arguments
是对象而非 JSON 字符串)。``build_request`` 用 ``wire_json`` 一次定型
``serialized_json``;``stream()`` 直接把这份文本作为请求体发出(逐字节同一口径),
``complete()`` 仍是 ``assemble_model_events(await collect_events(stream(req)))``,
与 Task 1/2 共享唯一装配路径。

B7 已消亡:``spec.supports_tools is False`` 且请求带工具时抛
``ToolsNotSupportedError``——**绝不**降级成"把工具写进 prompt 再解析文本"。

``base_url`` 必须是 loopback/私有主机(移植旧 location 约束的有用部分);Ollama
没有凭据概念,因此不发 ``Authorization``,客户端 ``trust_env=False`` 避免环境代理
把本地流量带出机器。NDJSON 一帧都没读到必须响亮失败(R-T4-2),绝不当成成功空回复。
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

import httpx

from praxis.adapters.models.base import WIRE_OWNED_FIELDS, to_openai_tools, wire_json
from praxis.protocol.json import thaw_json
from praxis.protocol.models import (
    ModelEvent,
    ModelMessage,
    ModelReasoningDelta,
    ModelResponse,
    ModelSettings,
    ModelSpec,
    ModelStreamFinished,
    ModelTextDelta,
    ModelToolCallDelta,
    Usage,
    WireRequest,
    assemble_model_events,
    collect_events,
)
from praxis.protocol.tools import ToolDefinition

_WIRE_REVISION = "praxis-ollama-chat-v1"
_CHAT_PATH = "/api/chat"
# 带内 error 最多带出这么多字符;错误帧的其余字段绝不回显。
_MAX_ERROR_TEXT = 200

# Ollama body 里由本模块独占的字段:provider_options 不得覆盖(含 num_predict,
# 它等价于 OpenAI 的 max_tokens;options 整体也不允许被顶掉)。
_OLLAMA_WIRE_OWNED_FIELDS: frozenset[str] = WIRE_OWNED_FIELDS | frozenset(
    {"num_predict", "options"}
)
# 允许进入 options 的采样键;其余未知键明确拒绝,不生长第二套合并规则。
# temperature 仍归 wire-owned(由 ModelSettings 提供),num_predict/options 亦然。
_OLLAMA_OPTION_KEYS: frozenset[str] = frozenset(
    {
        "top_p",
        "seed",
        "num_ctx",
        "format",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "repeat_penalty",
        "repeat_last_n",
        "stop",
        "num_gpu",
        "num_thread",
    }
)


class ToolsNotSupportedError(ValueError):
    """模型注册为不支持工具,却被要求携带 tools;绝不降级成 prompt 解析(B7)。"""


def _is_local_host(hostname: str) -> bool:
    """loopback / 私有网段 / link-local 才算本地 Ollama 端点。"""
    if hostname.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def to_ollama_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """ModelMessage 序列 → ``/api/chat`` 的 messages。

    与 OpenAI 形状的唯一差别:assistant ``tool_calls[].function.arguments``
    是对象而不是 JSON 字符串(Ollama 的 Go 结构体按 map 反序列化,字符串会被拒);
    tool 消息只带 role/content(原生协议没有 tool_call_id 配对字段)。
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            entry["tool_calls"] = [
                {"function": {"name": call.name, "arguments": thaw_json(call.arguments)}}
                for call in message.tool_calls
            ]
        out.append(entry)
    return out


def ollama_payload(req: WireRequest) -> dict[str, Any]:
    """WireRequest → ``/api/chat`` body(本 provider 的唯一构造点)。

    ``provider_options`` 只接受 ``_OLLAMA_OPTION_KEYS`` 里的采样键(``top_p``/
    ``seed``/``num_ctx``/``format``/``mirostat*``/``repeat_*``/``stop``/``num_gpu``/
    ``num_thread``)并落入 ``options``;命中 wire-owned 字段或未知键一律
    ``ValueError``——宁可边界失败,也不生长第二套与真实请求不一致的合并规则。
    """
    options: dict[str, Any] = {"num_predict": req.max_output_tokens}
    if req.temperature is not None:
        options["temperature"] = req.temperature
    for key, value in req.provider_options.items():
        if key in _OLLAMA_WIRE_OWNED_FIELDS:
            raise ValueError(f"provider option cannot override wire-owned field: {key}")
        if key not in _OLLAMA_OPTION_KEYS:
            raise ValueError(f"unsupported ollama provider option: {key}")
        options[key] = thaw_json(value)

    payload: dict[str, Any] = {
        "model": req.model,
        "messages": to_ollama_messages(req.messages),
        "stream": True,
        "options": options,
    }
    if req.tools:
        payload["tools"] = to_openai_tools(req.tools)
    return payload


def _count(value: object) -> int | None:
    """只接受真正的整数(``True``/``False`` 是 bool,不是 token 计数)。"""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _usage_from_done_frame(frame: Mapping[str, Any]) -> Usage:
    """``prompt_eval_count`` / ``eval_count`` 有多少用多少(缺的记 0)。

    两个计数都在才敢称整份 usage 是 provider 记账,只有一个时如实标 ``estimated``。
    """
    prompt_eval = _count(frame.get("prompt_eval_count"))
    eval_count = _count(frame.get("eval_count"))
    return Usage(
        0 if prompt_eval is None else prompt_eval,
        0 if eval_count is None else eval_count,
        source=(
            "provider"
            if prompt_eval is not None and eval_count is not None
            else "estimated"
        ),
    )


def _error_frame_text(error: object) -> str:
    """带内 error 的有界文本;绝不回显整帧。"""
    text = error if isinstance(error, str) else json.dumps(error, ensure_ascii=False)
    return " ".join(text.split())[:_MAX_ERROR_TEXT]


class OllamaAdapter:
    """原生 ``/api/chat`` 补全;满足 protocol.models.ModelAdapter 协议。"""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if spec.provider != "ollama":
            raise ValueError(
                f"ollama adapter requires provider='ollama', got {spec.provider!r}"
            )
        if not spec.base_url:
            raise ValueError("ollama provider requires base_url in ModelSpec")
        parsed = urlsplit(spec.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(
                f"ollama base_url must be an absolute http(s) URL, got {spec.base_url!r}"
            )
        if not _is_local_host(parsed.hostname):
            raise ValueError(
                "ollama base_url must target a loopback or private host, "
                f"got {parsed.hostname!r}"
            )
        self._spec = spec
        self._base_url = spec.base_url.rstrip("/")
        self._client = (
            http_client
            if http_client is not None
            else httpx.AsyncClient(trust_env=False, timeout=spec.timeout_seconds)
        )

    @property
    def spec(self) -> ModelSpec:
        """静态模型配置;探针据此判断工具步是否适用(不读取适配器私有状态)。"""
        return self._spec

    def build_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        settings: ModelSettings,
    ) -> WireRequest:
        """唯一的合并点;工具能力不足在构造期响亮拒绝(B7)。"""
        if tools and not self._spec.supports_tools:
            raise ToolsNotSupportedError(
                f"model {self._spec.model_id} is registered without tool support"
            )
        request = WireRequest(
            model=self._spec.wire_model_id,
            messages=tuple(messages),
            tools=tuple(tools),
            max_output_tokens=(
                settings.max_output_tokens
                if settings.max_output_tokens is not None
                else self._spec.max_output_tokens
            ),
            provider_options=dict(self._spec.provider_options),
            temperature=settings.temperature,
            wire_revision=_WIRE_REVISION,
            tokenizer_model=self._spec.tokenizer_model,
        )
        return replace(request, serialized_json=wire_json(ollama_payload(request)))

    async def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]:
        """逐行消费 NDJSON;任何帧都不属于"空成功"(R-T4-2)。"""
        if not req.serialized_json:
            raise ValueError("ollama stream requires a request built by build_request")
        usage = Usage(0, 0, source="estimated")
        finish_reason = "stop"
        saw_frame = False
        tool_index = 0
        async with self._client.stream(
            "POST",
            f"{self._base_url}{_CHAT_PATH}",
            content=req.serialized_json,
            headers={"content-type": "application/json"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                saw_frame = True
                frame = self._parse_frame(line, model=req.model)
                error = frame.get("error")
                if error is not None:
                    # 带内错误帧(HTTP 200 但 body 是 {"error": ...})绝不是成功空应答。
                    raise ValueError(
                        f"provider returned an error frame for model {req.model!r}: "
                        f"{_error_frame_text(error)}"
                    )
                message = frame.get("message")
                if not isinstance(message, dict) and frame.get("done") is not True:
                    raise ValueError(
                        f"provider returned a frame without a message for model {req.model!r}"
                    )
                if isinstance(message, dict):
                    reasoning = message.get("thinking")
                    if not isinstance(reasoning, str) or not reasoning:
                        reasoning = message.get("reasoning")
                    if isinstance(reasoning, str) and reasoning:
                        yield ModelReasoningDelta(reasoning)
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        yield ModelTextDelta(content)
                    tool_calls = message.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for call in tool_calls:
                            function = call.get("function") if isinstance(call, dict) else None
                            if not isinstance(function, dict):
                                continue
                            name = function.get("name")
                            arguments = function.get("arguments")
                            # Ollama 不按帧碎片化一条 tool call:每个对象都是完整调用,
                            # 因此每个对象分配新的跨流 index。若未来出现 name/arguments
                            # 分帧,需改为按 provider 调用 id 合并(当前不做,YAGNI)。
                            yield ModelToolCallDelta(
                                index=tool_index,
                                call_id=f"ollama_call_{tool_index}",
                                name_fragment=name if isinstance(name, str) else "",
                                arguments_fragment=(
                                    json.dumps(arguments, ensure_ascii=False)
                                    if arguments is not None
                                    else ""
                                ),
                            )
                            tool_index += 1
                if frame.get("done") is True:
                    usage = _usage_from_done_frame(frame)
                    done_reason = frame.get("done_reason")
                    finish_reason = (
                        done_reason
                        if isinstance(done_reason, str) and done_reason
                        else "stop"
                    )
        if not saw_frame:
            raise ValueError(
                f"provider returned no stream events for model {req.model!r}; "
                "it may not support streaming"
            )
        yield ModelStreamFinished(usage=usage, finish_reason=finish_reason)

    @staticmethod
    def _parse_frame(line: str, *, model: str) -> dict[str, Any]:
        """单行 NDJSON → dict;畸形帧抛有界 ValueError(不回显帧内容)。"""
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(
                f"provider returned malformed stream frame for model {model!r}"
            ) from None
        if not isinstance(frame, dict):
            raise ValueError(
                f"provider returned malformed stream frame for model {model!r}"
            )
        return frame

    async def complete(self, req: WireRequest) -> ModelResponse:
        return assemble_model_events(await collect_events(self.stream(req)))


__all__ = [
    "OllamaAdapter",
    "ToolsNotSupportedError",
    "ollama_payload",
    "to_ollama_messages",
]

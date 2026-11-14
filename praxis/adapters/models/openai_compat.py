"""OpenAI 兼容 provider 适配器(设计 §5.9:一个构造器 + 一条流式装配路径)。

provider_options 与 max_output_tokens 只在 ``build_request`` 合并一次,
``complete`` 只消费不可变的 WireRequest——探针/直调等旁路路径无从泄漏
provider_options(旧 B6 根因在此结构下消亡)。``base_url`` 单一来源是
``spec.base_url``,调用方不再能传第二个地址。

``build_request`` 同时定型 ``serialized_json``(canonical JSON),供后续
token 估算/记账只测这一份出站文本(D2)。本阶段 ``stream()`` 仍以非流式
补全产出等价事件帧;真正的增量流式由后续 adapters 任务替换,事件形状不变。
SDK 错误原样抛出,交由 turn 循环的错误边界处置。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

import httpx
from openai import AsyncOpenAI

from praxis.adapters.models.base import (
    collect_events,
    to_openai_messages,
    to_openai_tools,
)
from praxis.protocol.models import (
    ModelEvent,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelSpec,
    ModelStreamFinished,
    ModelTextDelta,
    ModelToolCallDelta,
    Usage,
    WireRequest,
    assemble_model_events,
)
from praxis.protocol.tools import ToolDefinition

_WIRE_REVISION = "praxis-openai-wire-v1"


def _completion_payload(req: WireRequest) -> dict[str, Any]:
    """WireRequest → chat.completions.create 的 SDK kwargs(唯一合并点)。"""
    payload: dict[str, Any] = {
        "model": req.model,
        "messages": to_openai_messages(req.messages),
    }
    if req.tools:
        payload["tools"] = to_openai_tools(req.tools)
    payload.update(req.provider_options)
    # 标准字段最后落位:与 provider_options 冲突时 settings 显式值胜出
    # (max_tokens 恒胜;temperature 仅在 settings 给了值时覆盖,
    # 否则 provider_options 可继续提供默认值)。
    payload["max_tokens"] = req.max_output_tokens
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    return payload


def _canonical_json(payload: dict[str, Any]) -> str:
    """稳定键序/无空白的出站文本:同一请求必得同一字符串(D2 的测量对象)。"""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


class OpenAICompatAdapter:
    """chat.completions 补全;满足 protocol.models.ModelAdapter 协议。"""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._spec = spec
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=spec.base_url, http_client=http_client
        )

    def build_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        settings: ModelSettings,
    ) -> WireRequest:
        """唯一的合并点:spec 为底,settings 覆盖采样可调项;出站文本一并定型。"""
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
        )
        return replace(request, serialized_json=_canonical_json(_completion_payload(request)))

    async def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]:
        """把一次补全转成事件帧:text → tool-call 分片 → finish。"""
        completion = await self._client.chat.completions.create(**_completion_payload(req))
        if not completion.choices:
            raise ValueError(f"provider returned no choices for model {req.model!r}")
        choice = completion.choices[0]
        message = choice.message
        content = message.content or ""
        if content:
            yield ModelTextDelta(content)
        for index, call in enumerate(message.tool_calls or ()):
            yield ModelToolCallDelta(
                index=index,
                call_id=call.id,
                name_fragment=call.function.name or "",
                arguments_fragment=call.function.arguments or "",
            )
        usage = completion.usage
        yield ModelStreamFinished(
            usage=(
                Usage(usage.prompt_tokens or 0, usage.completion_tokens or 0)
                if usage is not None
                else Usage(0, 0)
            ),
            finish_reason=choice.finish_reason or "",
        )

    async def complete(self, req: WireRequest) -> ModelResponse:
        return assemble_model_events(await collect_events(self.stream(req)))

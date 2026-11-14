"""OpenAI 兼容 provider 适配器(设计 §5.7/§5.9:一条流式装配路径)。

出站 body 的唯一构造是 ``base.openai_payload``;``stream()`` 真实消费 SSE 帧
并逐帧产出 ModelEvent(不缓冲、不冻结首帧)。``complete()`` 只是
``assemble_model_events(await collect_events(stream(req)))``,因此流式与非流式
不存在两套记账/聚合路径(B9/B24/B25 根因在此结构下消亡)。

``build_request`` 定型的 ``serialized_json`` 是**实际发出去的那一份** body:
core 只走 ``stream()``,``complete()`` 也经由它派生,所以固定用
``openai_payload(req, stream=True)`` 定型(R-T2-1)。``base_url`` 单一来源是
``spec.base_url``;SDK 自带重试关闭(max_retries=0),重试归上层。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from praxis.adapters.models.base import openai_payload, wire_json
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

_WIRE_REVISION = "praxis-openai-wire-v1"


def _reasoning_text(delta: Any) -> str:
    """provider 怪癖:思考文本可能叫 ``reasoning_content`` 或 ``reasoning``。"""
    value = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
    return value if isinstance(value, str) else ""


def _usage_from_provider(usage: Any) -> Usage:
    """provider usage → 协议 Usage;cache/reasoning 是 input/output 的子集。"""
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return Usage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cached_input_tokens=int(getattr(prompt_details, "cached_tokens", 0) or 0),
        reasoning_tokens=int(getattr(completion_details, "reasoning_tokens", 0) or 0),
        source="provider",
    )


def _whole_completion_events(completion: ChatCompletion, *, model: str) -> list[ModelEvent]:
    """provider 忽略 stream 时的诚实整体转换:一条 text delta + finish。"""
    if not completion.choices:
        raise ValueError(f"provider returned no choices for model {model!r}")
    choice = completion.choices[0]
    message = choice.message
    events: list[ModelEvent] = []
    reasoning = _reasoning_text(message)
    if reasoning:
        events.append(ModelReasoningDelta(reasoning))
    if message.content:
        events.append(ModelTextDelta(message.content))
    for index, call in enumerate(message.tool_calls or ()):
        function = getattr(call, "function", None)
        events.append(
            ModelToolCallDelta(
                index=index,
                call_id=call.id,
                name_fragment=(function.name or "") if function is not None else "",
                arguments_fragment=(function.arguments or "") if function is not None else "",
            )
        )
    events.append(
        ModelStreamFinished(
            usage=(
                _usage_from_provider(completion.usage)
                if completion.usage is not None
                else Usage(0, 0, source="estimated")
            ),
            finish_reason=choice.finish_reason or "",
        )
    )
    return events


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
            api_key=api_key,
            base_url=spec.base_url,
            http_client=http_client,
            max_retries=0,
            timeout=spec.timeout_seconds,
        )

    def build_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        settings: ModelSettings,
    ) -> WireRequest:
        """唯一的合并点;出站文本按真正走 stream() 的那一份定型(R-T2-1)。"""
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
        return replace(request, serialized_json=wire_json(openai_payload(request, stream=True)))

    async def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]:
        """真实增量流:逐 chunk 产出 text/reasoning/tool 分片,末尾 finish。"""
        result = await self._client.chat.completions.create(
            **openai_payload(req, stream=True)
        )
        if isinstance(result, ChatCompletion):
            # provider 忽略 stream=True,返回完整对象:按"诚实整体"转换。
            for event in _whole_completion_events(result, model=req.model):
                yield event
            return

        upstream_id: str | None = None
        usage = Usage(0, 0, source="estimated")
        finish_reason = ""
        async for chunk in result:
            chunk_id = getattr(chunk, "id", None)
            if chunk_id is not None:
                if upstream_id is None:
                    upstream_id = chunk_id
                elif chunk_id != upstream_id:
                    raise ValueError("provider changed response identity during stream")
            if chunk.usage is not None:
                usage = _usage_from_provider(chunk.usage)
            for choice in chunk.choices:
                delta = choice.delta
                reasoning = _reasoning_text(delta)
                if reasoning:
                    yield ModelReasoningDelta(reasoning)
                if delta.content:
                    yield ModelTextDelta(delta.content)
                for call in delta.tool_calls or ():
                    function = getattr(call, "function", None)
                    yield ModelToolCallDelta(
                        index=call.index,
                        call_id=call.id,
                        name_fragment=(
                            (function.name or "") if function is not None else ""
                        ),
                        arguments_fragment=(
                            (function.arguments or "") if function is not None else ""
                        ),
                    )
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        yield ModelStreamFinished(usage=usage, finish_reason=finish_reason)

    async def complete(self, req: WireRequest) -> ModelResponse:
        return assemble_model_events(await collect_events(self.stream(req)))

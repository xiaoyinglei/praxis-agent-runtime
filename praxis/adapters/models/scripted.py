"""脚本驱动适配器:按预置剧本依次返回 ModelResponse。

测试与验收的确定性引擎,不触网;``delay_seconds`` 提供 kill -9 窗口
控制(恢复路径测试用)。剧本耗尽即抛 ScriptExhausted——绝不静默循环。

``stream()`` 把剧本切成真实的事件序列,``complete()`` 与生产适配器一样
走 ``assemble_model_events(await collect_events(...))``:验收测试因此驱动的
是**同一条**装配路径,而不是脚本专属的旁路。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence

from praxis.adapters.models.base import collect_events
from praxis.protocol.json import thaw_json
from praxis.protocol.models import (
    ModelEvent,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelStreamFinished,
    ModelTextDelta,
    ModelToolCallDelta,
    WireRequest,
    assemble_model_events,
)
from praxis.protocol.tools import ToolDefinition

_SCRIPTED_MODEL_ID = "scripted"
_SCRIPTED_MAX_OUTPUT_TOKENS = 4096


class ScriptExhausted(RuntimeError):  # noqa: N818 — 名称由 Task 9 brief 契约固定
    """剧本已用完而 turn 循环仍在请求补全。"""


class ScriptedAdapter:
    """满足 ModelAdapter 协议的固定剧本实现。"""

    def __init__(self, responses: Sequence[ModelResponse], *, delay_seconds: float = 0.0) -> None:
        self._responses = tuple(responses)
        self._delay_seconds = delay_seconds
        self._next_index = 0
        self.last_request: WireRequest | None = None

    def build_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        settings: ModelSettings,
    ) -> WireRequest:
        """构建并记录最近一次请求,供测试断言循环传给适配器的内容。"""
        request = WireRequest(
            model=_SCRIPTED_MODEL_ID,
            messages=tuple(messages),
            tools=tuple(tools),
            max_output_tokens=(
                settings.max_output_tokens
                if settings.max_output_tokens is not None
                else _SCRIPTED_MAX_OUTPUT_TOKENS
            ),
            provider_options={},
            temperature=settings.temperature,
        )
        self.last_request = request
        return request

    async def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]:
        """消费下一条剧本,按内容/工具调用/finish 切成事件帧。"""
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        if self._next_index >= len(self._responses):
            raise ScriptExhausted(
                f"script exhausted after {len(self._responses)} response(s); "
                f"request model={req.model!r}"
            )
        response = self._responses[self._next_index]
        self._next_index += 1
        if response.content:
            yield ModelTextDelta(response.content)
        for index, call in enumerate(response.tool_calls):
            yield ModelToolCallDelta(
                index=index,
                call_id=call.call_id,
                name_fragment=call.name,
                arguments_fragment=json.dumps(thaw_json(call.arguments), ensure_ascii=False),
            )
        yield ModelStreamFinished(response.usage, response.finish_reason)

    async def complete(self, req: WireRequest) -> ModelResponse:
        return assemble_model_events(await collect_events(self.stream(req)))

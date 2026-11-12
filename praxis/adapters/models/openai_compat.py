"""OpenAI 兼容 provider 适配器(设计 §5.9 最小面:非流式)。

provider_options 与 max_output_tokens 只在 ``build_request`` 合并一次,
``complete`` 只消费不可变的 WireRequest——探针/直调等旁路路径无从泄漏
provider_options(旧 B6 根因在此结构下消亡)。流式与重试属阶段 3。
SDK 错误原样抛出,交由 turn 循环的错误边界处置。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
from openai import AsyncOpenAI

from praxis.adapters.models.base import to_openai_messages, to_openai_tools
from praxis.protocol.models import (
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelSpec,
    ModelToolCall,
    Usage,
    WireRequest,
)
from praxis.protocol.tools import ToolDefinition


class OpenAICompatAdapter:
    """chat.completions 非流式补全;满足 protocol.models.ModelAdapter 协议。"""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        api_key: str,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._spec = spec
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def build_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        settings: ModelSettings,
    ) -> WireRequest:
        """唯一的合并点:spec 为底,settings 覆盖采样可调项。"""
        return WireRequest(
            model=self._spec.model_id,
            messages=tuple(messages),
            tools=tuple(tools),
            max_output_tokens=(
                settings.max_output_tokens
                if settings.max_output_tokens is not None
                else self._spec.max_output_tokens
            ),
            provider_options=dict(self._spec.provider_options),
            temperature=settings.temperature,
        )

    async def complete(self, req: WireRequest) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": to_openai_messages(req.messages),
        }
        if req.tools:
            kwargs["tools"] = to_openai_tools(req.tools)
        kwargs.update(req.provider_options)
        # 标准字段最后落位:与 provider_options 冲突时 settings 显式值胜出
        # (max_tokens 恒胜;temperature 仅在 settings 给了值时覆盖,
        # 否则 provider_options 可继续提供默认值)。
        kwargs["max_tokens"] = req.max_output_tokens
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        completion = await self._client.chat.completions.create(**kwargs)
        if not completion.choices:
            raise ValueError(f"provider returned no choices for model {req.model!r}")
        choice = completion.choices[0]
        message = choice.message
        tool_calls: list[ModelToolCall] = []
        for index, call in enumerate(message.tool_calls or ()):
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"tool_calls[{index}] ({call.function.name!r}) arguments is not valid JSON: "
                    f"{call.function.arguments!r}"
                ) from error
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"tool_calls[{index}] ({call.function.name!r}) arguments must be a JSON "
                    f"object, got {type(arguments).__name__}"
                )
            tool_calls.append(
                ModelToolCall(call_id=call.id, name=call.function.name, arguments=arguments)
            )
        usage = completion.usage
        return ModelResponse(
            content=message.content or "",
            tool_calls=tuple(tool_calls),
            usage=(
                Usage(usage.prompt_tokens or 0, usage.completion_tokens or 0)
                if usage is not None
                else Usage(0, 0)
            ),
            finish_reason=choice.finish_reason or "",
        )

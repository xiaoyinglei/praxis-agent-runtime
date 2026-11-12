"""协议层模型契约:ModelSpec/ModelMessage/WireRequest 与 ModelAdapter 边界(设计 §5.14)。

core 只面对 ModelAdapter 协议;provider 差异(openai-compatible/scripted)
封在 adapters 层。所有值为冻结数据类:构建一次、传递多处、无人能改。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from praxis.protocol.json import JsonValue
from praxis.protocol.tools import ToolDefinition


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """一个模型的选择与静态配置;provider_options 原样透传给 adapter。"""

    model_id: str
    provider: Literal["openai-compatible", "scripted"] = "openai-compatible"
    max_output_tokens: int = 4096
    provider_options: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ModelToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    tool_calls: tuple[ModelToolCall, ...]
    usage: Usage
    finish_reason: str


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """每轮采样的可调项;None 表示交给 provider 默认。"""

    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class WireRequest:
    """build_request 的产物:一次 provider 调用的完整不可变请求。"""

    model: str
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolDefinition, ...]
    max_output_tokens: int
    provider_options: Mapping[str, JsonValue]
    temperature: float | None


@runtime_checkable
class ModelAdapter(Protocol):
    """模型适配边界(T9/T13 消费):先构建请求,再异步补全。"""

    def build_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        settings: ModelSettings,
    ) -> WireRequest: ...

    async def complete(self, req: WireRequest) -> ModelResponse: ...

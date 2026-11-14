"""协议层模型契约:ModelSpec/ModelMessage/WireRequest 与 ModelAdapter 边界(设计 §5.9)。

core 只面对 ModelAdapter 协议;provider 差异(openai-compatible/ollama/scripted)
封在 adapters 层。所有值为冻结数据类:构建一次、传递多处、无人能改。

流式是唯一事实来源:``ModelTextDelta``/``ModelReasoningDelta``/
``ModelToolCallDelta``/``ModelStreamFinished`` 由 adapter 逐帧产出,
``assemble_model_events`` 是**唯一**装配实现——``complete()`` 只是
``assemble_model_events(await collect_events(stream(req)))``。流式与非流式
因此不存在两套记账/聚合路径(B24/B25 根因)。
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from praxis.protocol.json import JsonValue
from praxis.protocol.tools import ToolDefinition

ModelProvider = Literal["openai-compatible", "ollama", "scripted"]

_API_KEY_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 坏 tool arguments 的有界提示:只留计数,不把 provider 原文抄进会话内容。
_DROPPED_ARGUMENTS_HINT = "[invalid tool call arguments dropped: {count}]"


@dataclass(frozen=True, slots=True)
class CostMicros:
    """每百万 token 的单价(micros);全零 = 未配置计价。"""

    input_per_1m: int = 0
    output_per_1m: int = 0
    cache_read_per_1m: int = 0
    cache_write_per_1m: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("input_per_1m", self.input_per_1m),
            ("output_per_1m", self.output_per_1m),
            ("cache_read_per_1m", self.cache_read_per_1m),
            ("cache_write_per_1m", self.cache_write_per_1m),
        ):
            if value < 0:
                raise ValueError(f"CostMicros.{name} must be >= 0, got {value}")

    @property
    def is_zero(self) -> bool:
        """未配置任何计价;预算逻辑据此跳过成本结算。"""
        return (
            self.input_per_1m == 0
            and self.output_per_1m == 0
            and self.cache_read_per_1m == 0
            and self.cache_write_per_1m == 0
        )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """一个模型的单一选择与静态配置;provider_options 原样透传给 adapter。

    ``model_id`` 是本地别名(配置/日志用),``remote_model_id`` 才是发给
    provider 的 wire 名——两者分离后,别名可稳定而不随上游改名漂移。
    """

    model_id: str
    provider: ModelProvider = "openai-compatible"
    remote_model_id: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    context_window_tokens: int = 8192
    max_output_tokens: int = 4096
    supports_tools: bool = True
    tokenizer_model: str | None = None
    timeout_seconds: float = 120.0
    auto_compact_token_limit: int | None = None
    cost: CostMicros = CostMicros()
    provider_options: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 输出预算必须为正且严格小于上下文窗口,否则请求在 provider 侧必然被拒。
        if self.max_output_tokens <= 0:
            raise ValueError(
                f"max_output_tokens must be > 0, got {self.max_output_tokens}"
            )
        if self.context_window_tokens <= self.max_output_tokens:
            raise ValueError(
                "context_window_tokens must be > max_output_tokens "
                f"({self.context_window_tokens} <= {self.max_output_tokens})"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if (
            self.api_key_env is not None
            and _API_KEY_ENV_PATTERN.fullmatch(self.api_key_env) is None
        ):
            raise ValueError(
                f"api_key_env must be an environment variable name, got {self.api_key_env!r}"
            )

    @property
    def wire_model_id(self) -> str:
        """实际发给 provider 的模型名;未设别名时即本地 model_id。"""
        return self.remote_model_id or self.model_id


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
    """一次补全的 token 记账;``source="estimated"`` 表示 provider 未给 usage。"""

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    source: Literal["provider", "estimated"] = "provider"

    @property
    def total_tokens(self) -> int:
        """input/output 已各自包含 cache/reasoning 子集,总量不重复相加。"""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """一次补全的装配结果;``reasoning`` 是思考文本,绝不并入 ``content``。

    两者分离是硬约束:``content`` 会作为 assistant 历史回灌 provider,
    思考文本一旦混入就会被当成正文重复发送。
    """

    content: str
    tool_calls: tuple[ModelToolCall, ...]
    usage: Usage
    finish_reason: str
    reasoning: str = ""


# ---------------------------------------------------------------------------
# 流式事件:adapter 的唯一产出形状,assemble_model_events 的唯一输入
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelTextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ModelReasoningDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ModelToolCallDelta:
    """一个 tool-call 的分片;``call_id`` 只在首个非 None 帧定值(B9)。"""

    index: int
    call_id: str | None
    name_fragment: str = ""
    arguments_fragment: str = ""


@dataclass(frozen=True, slots=True)
class ModelStreamFinished:
    usage: Usage
    finish_reason: str


type ModelEvent = (
    ModelTextDelta | ModelReasoningDelta | ModelToolCallDelta | ModelStreamFinished
)


@dataclass
class _PartialToolCall:
    """装配期的可变累积器;不对外暴露,只是 index → 分片缓冲。"""

    call_id: str | None = None
    name_fragments: list[str] = field(default_factory=list)
    argument_fragments: list[str] = field(default_factory=list)


def _parse_arguments(raw: str) -> Mapping[str, JsonValue] | None:
    """空串视作 ``{}``(provider 省略 arguments);非法/非 object 返回 None。"""
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def assemble_model_events(events: Sequence[ModelEvent]) -> ModelResponse:
    """把流式事件装配成 ModelResponse——流式/非流式共用的唯一实现。

    语义(设计 §5.9;B9):
    - text 分片累加进 ``content``;reasoning 分片**单独**累加进 ``reasoning``,
      两者绝不混拼(``content`` 会作为 assistant 历史回灌 provider);
    - tool-call 按 ``index`` 累积:``call_id`` 首个非 None 定值,``name``/
      ``arguments`` 分片一直拼到 finish,不冻结首帧;
    - 空 arguments 视作 ``{}``;非 JSON 或非 object 的调用**丢弃**并在 content
      尾部留下有界计数提示,绝不抛穿 turn(畸形 provider 响应是常态);
    - 未收到 ``ModelStreamFinished`` → ``Usage(0, 0, source="estimated")``
      且 ``finish_reason=""``,如实标注估算来源。
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    partials: dict[int, _PartialToolCall] = {}
    usage = Usage(0, 0, source="estimated")
    finish_reason = ""

    for event in events:
        if isinstance(event, ModelTextDelta):
            content_parts.append(event.text)
        elif isinstance(event, ModelReasoningDelta):
            reasoning_parts.append(event.text)
        elif isinstance(event, ModelToolCallDelta):
            partial = partials.setdefault(event.index, _PartialToolCall())
            if partial.call_id is None and event.call_id is not None:
                partial.call_id = event.call_id
            partial.name_fragments.append(event.name_fragment)
            partial.argument_fragments.append(event.arguments_fragment)
        elif isinstance(event, ModelStreamFinished):
            usage = event.usage
            finish_reason = event.finish_reason

    calls: list[ModelToolCall] = []
    dropped = 0
    for index in sorted(partials):
        partial = partials[index]
        if partial.call_id is None:
            # 无 id 的调用无法与 tool 结果配对,不装配(也不计入 arguments 提示)。
            continue
        arguments = _parse_arguments("".join(partial.argument_fragments))
        if arguments is None:
            dropped += 1
            continue
        calls.append(
            ModelToolCall(
                call_id=partial.call_id,
                name="".join(partial.name_fragments),
                arguments=arguments,
            )
        )

    content = "".join(content_parts)
    if dropped:
        hint = _DROPPED_ARGUMENTS_HINT.format(count=dropped)
        content = f"{content} {hint}" if content else hint

    return ModelResponse(
        content=content,
        tool_calls=tuple(calls),
        usage=usage,
        finish_reason=finish_reason,
        reasoning="".join(reasoning_parts),
    )


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """每轮采样的可调项;None 表示交给 provider 默认。"""

    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class WireRequest:
    """build_request 的产物:一次 provider 调用的完整不可变请求。

    ``serialized_json`` 是 build_request 内一次定型的真正出站 JSON 文本
    (记账/token 估算只测这一份,杜绝第二口径);``wire_revision`` 标记
    wire 形状版本。旧构造不传两者时保持空串。
    """

    model: str
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolDefinition, ...]
    max_output_tokens: int
    provider_options: Mapping[str, JsonValue]
    temperature: float | None
    serialized_json: str = ""
    wire_revision: str = ""


@runtime_checkable
class ModelAdapter(Protocol):
    """模型适配边界:先构建请求,再流式补全;``complete`` 是流式装配的薄门面。"""

    def build_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        settings: ModelSettings,
    ) -> WireRequest: ...

    async def complete(self, req: WireRequest) -> ModelResponse: ...

    def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]: ...

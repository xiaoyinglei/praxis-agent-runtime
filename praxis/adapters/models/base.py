"""模型适配层的唯一出站序列化点:协议消息/工具 → OpenAI 兼容 wire body。

纯函数,无 provider 状态;所有 adapter 的出站 body 都经由 ``openai_payload``,
wire 形状只有一处定义(设计 §5.7/§5.9)。``provider_options`` 在这里一次性
分流:命中 wire-owned 字段直接拒绝(B6 根因:不允许第二口径覆盖真实请求),
SDK 认识的顶层参数直落,其余合并进 ``extra_body``。

``wire_json`` 是 build_request 定型的记账文本来源:同一 payload 必得同一字符串;
非 JSON 兼容值在这里失败(有界 ValueError),不许静默字符串化后混进测量文本。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from praxis.protocol.json import thaw_json
from praxis.protocol.models import ModelMessage, WireRequest, collect_events
from praxis.protocol.tools import ToolDefinition

# serializer 独占的字段:provider_options 不得覆盖,否则"实际发出去的"与
# "记账/测量文本"会出现第二口径(B6/B24)。
WIRE_OWNED_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "max_tokens",
        "temperature",
        "stream",
        "stream_options",
    }
)

# SDK 原生认识的顶层扩展参数;其余未知键统一进 extra_body(合并、键序稳定)。
# temperature 已在 WIRE_OWNED_FIELDS,不重复列出(否则这一项永不可达)。
_TOP_LEVEL_PROVIDER_OPTION_KEYS: frozenset[str] = frozenset(
    {"top_p", "seed", "parallel_tool_calls", "reasoning_effort"}
)


def to_openai_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """ModelMessage 序列 → chat.completions ``messages`` 列表。

    - assistant 的 tool_calls 按 OpenAI 形状序列化,arguments 为 JSON 字符串;
    - tool 角色消息携带 tool_call_id,回指其应答的调用。
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(thaw_json(call.arguments), ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            entry["tool_call_id"] = message.tool_call_id
        out.append(entry)
    return out


def to_openai_tools(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    """ToolDefinition 序列 → chat.completions ``tools`` 列表(function 形状)。"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": thaw_json(tool.input_schema),
            },
        }
        for tool in tools
    ]


def openai_payload(req: WireRequest, *, stream: bool) -> dict[str, Any]:
    """WireRequest → chat.completions.create 的完整 body(唯一构造点)。

    ``provider_options`` 逐项落位:
    - 命中 ``WIRE_OWNED_FIELDS`` → ``ValueError``(不得覆盖真实请求字段);
    - ``extra_body`` 键是受支持形状:其内容**合并**进顶层 ``extra_body``,
      不会产生 ``extra_body={"extra_body": ...}`` 双层嵌套;
    - SDK 认识的顶层键直落;
    - 其余键合并进 ``extra_body``,保持 provider_options 的插入顺序。
    ``stream=True`` 时加 ``stream_options={"include_usage": True}``——usage 是
    finish 事件的事实来源。

    注意:返回的是 **SDK kwargs 形状**;``extra_body`` 会被 SDK 展平进真实
    HTTP body,所以 ``serialized_json`` 与线上 body 仅在这一处不逐字节相同
    (结构可无损还原)。本阶段 ``build_request`` 固定以 ``stream=True`` 定型
    记账文本:core 只走 ``stream()``,``complete()`` 也由它派生(R-T2-1)。
    """
    payload: dict[str, Any] = {
        "model": req.model,
        "messages": to_openai_messages(req.messages),
    }
    if req.tools:
        payload["tools"] = to_openai_tools(req.tools)
    payload["max_tokens"] = req.max_output_tokens
    if req.temperature is not None:
        payload["temperature"] = req.temperature

    extra_body: dict[str, Any] = {}
    for key, value in req.provider_options.items():
        if key in WIRE_OWNED_FIELDS:
            raise ValueError(f"provider option cannot override wire-owned field: {key}")
        thawed = thaw_json(value)
        if key == "extra_body":
            if not isinstance(thawed, dict):
                raise ValueError("provider option 'extra_body' must be an object")
            # R-T4-1:SDK 把 extra_body 合并在 JSON body 之上且后者胜出,嵌套键
            # 同样不得覆盖 wire-owned 字段——否则真实请求与记账文本出现第二口径。
            for nested_key in thawed:
                if nested_key in WIRE_OWNED_FIELDS:
                    raise ValueError(
                        "provider option cannot override wire-owned field via extra_body: "
                        f"{nested_key}"
                    )
            extra_body.update(thawed)
        elif key in _TOP_LEVEL_PROVIDER_OPTION_KEYS:
            payload[key] = thawed
        else:
            extra_body[key] = thawed
    if extra_body:
        payload["extra_body"] = extra_body

    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def wire_json(payload: dict[str, Any]) -> str:
    """稳定键序/无空白的出站文本:同一请求必得同一字符串(记账测量对象)。

    非 JSON 兼容值必须在 build 期失败——静默 ``default=str`` 会把无法发送的
    值变成"看起来能记账"的文本(B24 口径分叉的温床)。错误消息只带类型,
    不回显整个值。
    """
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"request payload is not JSON-serializable: {exc}") from None


__all__ = [
    "WIRE_OWNED_FIELDS",
    "collect_events",
    "openai_payload",
    "to_openai_messages",
    "to_openai_tools",
    "wire_json",
]

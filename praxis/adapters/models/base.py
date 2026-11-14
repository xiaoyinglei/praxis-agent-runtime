"""模型适配层的唯一序列化点:协议消息/工具 → OpenAI 兼容 wire 结构。

纯函数,无 provider 状态;所有 adapter 的出站消息/工具格式都从这里出去,
保证 wire 形状只有一处定义(设计 §5.9)。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from praxis.protocol.json import thaw_json
from praxis.protocol.models import ModelEvent, ModelMessage
from praxis.protocol.tools import ToolDefinition


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


async def collect_events(stream: AsyncIterator[ModelEvent]) -> list[ModelEvent]:
    """把一个 adapter 的流收成事件列表,供 ``assemble_model_events`` 装配。

    ``complete()`` 与探针/CLI 共用这一条路径——没有并行的第二聚合实现。
    """
    return [event async for event in stream]

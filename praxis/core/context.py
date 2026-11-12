"""core 上下文:boundary 折叠与 Item→ModelMessage 投影(设计 §5.1/§5.7 · findings B22)。

压缩不复制正文:写端只 append summary + compaction_boundary(boundary 仅记录
covered_from/covered_to 的 seq 区间),原 Item 永远留在 store;折叠决策全在
读端(本模块)——旧架构把整个活动上下文复制进每条压缩日志的 B22 在此结构性消亡。

fail-open 读:algorithm_version 未知、boundary payload 畸形(非 int 区间等)
→ 该 boundary 不折叠任何东西;to_messages 对未知 kind 与畸形 payload 一律跳过。
两个函数都是纯函数:不修改输入、幂等、永不抛。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, TypeGuard

from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelMessage, ModelToolCall

_SUPPORTED_BOUNDARY_VERSION: Final = "v1"
_SUMMARY_PREFIX: Final = "[conversation summary]\n"


def build_context(items: Sequence[Item]) -> list[Item]:
    """按 compaction_boundary 折叠:丢弃覆盖区间内的 item,保留 boundary/summary 本身。

    - 已知版本 "v1" 且区间为 int → 折叠 [covered_from, covered_to](多 boundary 取并集);
    - 未知版本或畸形 payload → 该 boundary 不折叠(fail-open,原样保留全部 item);
    - 纯函数:返回新 list,元素与输入同一对象;幂等。
    """
    ranges: list[tuple[int, int]] = []
    for item in items:
        if item.kind != ItemKind.COMPACTION_BOUNDARY:
            continue
        covered = _boundary_range(item.payload)
        if covered is not None:
            ranges.append(covered)
    if not ranges:
        return list(items)
    folded: list[Item] = []
    for item in items:
        if item.kind == ItemKind.COMPACTION_BOUNDARY or item.kind == ItemKind.SUMMARY:
            folded.append(item)  # boundary 与 summary 自身永远保留
            continue
        if any(low <= item.seq <= high for low, high in ranges):
            continue
        folded.append(item)
    return folded


def to_messages(items: Sequence[Item]) -> list[ModelMessage]:
    """把(已折叠的)Item 序列投影为模型可见消息;未知 kind 与畸形 payload 跳过,永不抛。

    - user_message/steering → user(payload["text"])
    - model_response → assistant(payload["content"] + tool_calls 重建);content 缺失/非 str
      但 tool_calls 可重建时投 content=""(R7 读端 fail-open,保住配对 tool_result)
    - tool_result → tool(文本口径与 protocol.tools.ToolResult.text 一致,tool_call_id=payload["call_id"])
    - summary → user,前缀 "[conversation summary]\\n"
    - 其余 kind(turn_status/approval_*/tool_call/未知)→ 跳过
    """
    messages: list[ModelMessage] = []
    for item in items:
        message = _to_message(item)
        if message is not None:
            messages.append(message)
    return messages


# ---------------------------------------------------------------------------
# 折叠内部实现
# ---------------------------------------------------------------------------


def _is_int(value: object) -> TypeGuard[int]:
    """bool 是 int 子类,但 JSON 里 True/1 语义不同:畸形处理,bool 不算 int。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _boundary_range(payload: Mapping[str, JsonValue]) -> tuple[int, int] | None:
    """解析 boundary payload;非已知版本或畸形区间返回 None(即:不折叠)。"""
    if not isinstance(payload, Mapping):  # 防御:违反契约的 payload 也 fail-open
        return None
    if payload.get("algorithm_version") != _SUPPORTED_BOUNDARY_VERSION:
        return None
    low = payload.get("covered_from")
    high = payload.get("covered_to")
    if _is_int(low) and _is_int(high):
        return (low, high)
    return None


# ---------------------------------------------------------------------------
# to_messages 内部实现
# ---------------------------------------------------------------------------


def _to_message(item: Item) -> ModelMessage | None:
    kind = item.kind
    payload = item.payload
    if not isinstance(payload, Mapping):  # 防御:违反契约的 payload 跳过(fail-open)
        return None
    if kind == ItemKind.USER_MESSAGE or kind == ItemKind.STEERING:
        text = _text_field(payload, "text")
        return None if text is None else ModelMessage(role="user", content=text)
    if kind == ItemKind.MODEL_RESPONSE:
        return _model_response_message(payload)
    if kind == ItemKind.TOOL_RESULT:
        return _tool_result_message(payload)
    if kind == ItemKind.SUMMARY:
        text = _text_field(payload, "text")
        return None if text is None else ModelMessage(role="user", content=_SUMMARY_PREFIX + text)
    return None  # turn_status / approval_* / tool_call / 未知 kind → 跳过


def _text_field(payload: Mapping[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _model_response_message(payload: Mapping[str, JsonValue]) -> ModelMessage | None:
    content = _text_field(payload, "content")
    raw_calls = payload.get("tool_calls")
    calls: list[ModelToolCall] | None = None  # None=key 缺失;list=重建成功(可为空)
    if raw_calls is not None:
        calls = _reconstruct_tool_calls(raw_calls)
        if calls is None:  # tool_calls 畸形/条目非法 → 整个 item 跳过(不部分抢救)
            return None
    if content is not None:
        if calls is None:
            return ModelMessage(role="assistant", content=content)
        return ModelMessage(role="assistant", content=content, tool_calls=tuple(calls))
    # R7 读端 fail-open:content 缺失/非 str 的 tool-call-only 轮次(许多模型 API 的
    # 形状)不静默丢弃——只要有可重建的 tool_calls 就投 content="" 的 assistant 消息,
    # 否则配对的 tool_result 会在投影里成为孤儿。两者皆无才跳过。
    if calls:
        return ModelMessage(role="assistant", content="", tool_calls=tuple(calls))
    return None


def _reconstruct_tool_calls(raw_calls: object) -> list[ModelToolCall] | None:
    """严格重建:非序列或任一条目非法返回 None(= 不可重建)。"""
    if not isinstance(raw_calls, (list, tuple)):
        return None
    calls: list[ModelToolCall] = []
    for entry in raw_calls:
        if not isinstance(entry, Mapping):
            return None
        call_id = entry.get("call_id")
        name = entry.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            return None
        arguments = entry.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            return None
        calls.append(ModelToolCall(call_id=call_id, name=name, arguments=arguments))
    return calls


def _tool_result_message(payload: Mapping[str, JsonValue]) -> ModelMessage | None:
    call_id = payload.get("call_id")
    if not isinstance(call_id, str):
        return None
    raw_blocks = payload.get("content")
    blocks: Sequence[JsonValue] = ()
    if raw_blocks is not None:
        if not isinstance(raw_blocks, (list, tuple)):
            return None
        blocks = raw_blocks
    # 文本口径逐字镜像 protocol.tools.ToolResult.text:type=="text" 块取 data["text"],
    # 非字符串跳过,"\n".join;Task 12 的 tool_result item 按同一形状写入 payload。
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            return None
        if block.get("type") != "text":
            continue
        data = block.get("data")
        if not isinstance(data, Mapping):
            continue
        value = data.get("text")
        if isinstance(value, str):
            parts.append(value)
    return ModelMessage(role="tool", content="\n".join(parts), tool_call_id=call_id)

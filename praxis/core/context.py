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
_UNPAIRED_TOOL_RESULT_NOTE: Final = "[tool result unavailable: turn ended before execution]"


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

    配对修复(R9):只改投影,不改 store 事实。两遍处理——
    第一遍收集"有 tool_result item 的 (turn_id, call_id) 集合"(存在性按
    kind+turn_id+call_id 判定,与 status/文本无关,denied/not_executed 也算
    存在);第二遍顺序投影,紧跟在携带 tool_calls 的 assistant 消息之后,按其
    tool_calls 顺序为该 assistant 所属 turn 中没有结果的 call_id 合成 tool 消息
    (覆盖 cancelled/failed/abandoned-paused 轮次;INTERRUPTED 轮次由 recovery
    写入真实 not_executed/outcome_unknown 结果,属于结果集合,原样投影)。真实
    结果 item 遇到时正常投影,不会重复。
    这样任何 provider 的 tool_use/tool_result 配对校验(OpenAI/Anthropic 会
    拒绝孤儿 tool_calls)都不会因中断轮次而失败。

    结果覆盖(R10,同一第一遍的图完成):同一 turn 内一个 call_id 出现多条
    tool_result 时只投影**最后**一条(最新 seq 胜;seq 相同取列表靠后者)。
    reconcile 会在 outcome_unknown 之后追加对账结果——被覆盖的旧结果仍留在
    item 日志做审计,但不再进入模型上下文,避免同一调用出现两个互相矛盾的
    工具消息。

    turn 作用域(整分支评审修复):存在性与覆盖判定都按 (turn_id, call_id)
    分键——call_id 只在单个 turn 内是调用身份,跨 turn 不唯一(local provider
    每轮从 call_0 重新起号、CLI 在 default 线程多轮累积)。全局按 call_id 分键
    会把后一 turn 同 id 的真实结果判为"更新"而丢弃前一 turn 的结果、同时 R9
    也不再为缺结果的 turn 合成 → 前一 turn 的 assistant 成为孤儿 tool_call,
    真实 provider 拒收请求。turn_id 为 None 的 item 退化为 (None, call_id),
    单线程行为不变。
    """
    last_result_keys = _last_result_keys(items)
    messages: list[ModelMessage] = []
    for position, item in enumerate(items):
        if item.kind == ItemKind.TOOL_RESULT:
            call_id = item.payload.get("call_id")
            if isinstance(call_id, str) and last_result_keys.get((item.turn_id, call_id)) != (
                item.seq,
                position,
            ):
                continue  # R10:同 turn 内非最新结果,留在日志不进投影
        message = _to_message(item)
        if message is None:
            continue
        messages.append(message)
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                if (item.turn_id, call.call_id) not in last_result_keys:
                    messages.append(
                        ModelMessage(
                            role="tool",
                            content=_UNPAIRED_TOOL_RESULT_NOTE,
                            tool_call_id=call.call_id,
                        )
                    )
    return messages


def _last_result_keys(items: Sequence[Item]) -> dict[tuple[str | None, str], tuple[int, int]]:
    """两遍处理第一遍:(turn_id, call_id) → 该 turn 该 call 最新 tool_result 的
    (seq, 列表位置)。

    键集合即"已有结果的 (turn_id, call_id)"(R9 配对判据);值给出该 call 在本
    turn 内哪条结果进投影(R10 覆盖判据)。call_id 跨 turn 不唯一,键必须带
    turn_id(见 to_messages docstring);存在性与覆盖判定只看 kind+turn_id+
    call_id,与 status 无关。
    """
    latest: dict[tuple[str | None, str], tuple[int, int]] = {}
    for position, item in enumerate(items):
        if item.kind != ItemKind.TOOL_RESULT:
            continue
        call_id = item.payload.get("call_id")
        if not isinstance(call_id, str):
            continue
        key = (item.seq, position)
        result_id = (item.turn_id, call_id)
        current = latest.get(result_id)
        if current is None or key >= current:
            latest[result_id] = key
    return latest


# ---------------------------------------------------------------------------
# 折叠内部实现
# ---------------------------------------------------------------------------


def _is_int(value: object) -> TypeGuard[int]:
    """bool 是 int 子类,但 JSON 里 True/1 语义不同:畸形处理,bool 不算 int。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _boundary_range(payload: Mapping[str, JsonValue]) -> tuple[int, int] | None:
    """解析 boundary payload;非已知版本或畸形区间返回 None(即:不折叠)。

    payload 为 Mapping 是 Item 契约(见 protocol/items.py),不再做运行时的
    isinstance 防御(Item 是 frozen dataclass,store 读出的 payload 必为 Mapping)。
    """
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
    # payload 为 Mapping 由 Item 契约保证,不做运行时 isinstance 防御
    kind = item.kind
    payload = item.payload
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

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


# 声明窗口配对键/结果引用(B31/R-T7-1 共用,见 pair_tool_results)。
type DeclarationKey = tuple[str | None, int, str]
type ResultRef = tuple[int, int]  # (seq, 列表位置):进投影的最新结果


def pair_tool_results(
    items: Sequence[Item],
) -> tuple[dict[DeclarationKey, ResultRef], set[DeclarationKey]]:
    """把 tool_result 配给"声明窗口"——``to_messages`` 与 ``classify_pending`` 的唯一判据。

    键 = ``(turn_id, 声明它的 model_response.seq, call_id)``:同一 turn 内复用
    call_id(本地小模型每轮从 call_0 重新起号)不再互相覆盖。按 seq 升序扫一遍:

    - ``model_response`` 的 ``tool_calls`` **整条可重建**(与
      ``_reconstruct_tool_calls``/投影同一严格度)时,为其中每个 call_id 打开/
      更新该 ``(turn_id, call_id)`` 的当前声明窗口;部分畸形的响应不开窗——
      其配对结果因此成为孤儿、在自身位置 fail-open 投影,绝不被静默吞掉
      (同一条响应声明两次同 id 折叠进同一窗口,``to_messages`` 侧为第二次
      声明补合成);
    - ``tool_result`` 配给它之前**最近**的同 ``(turn_id, call_id)`` 声明;
      同窗口多条结果取最新 ``(seq, 列表位置)`` 胜(R10),旧结果留库审计;
    - 之前没有任何声明的结果进 ``orphans``(键用结果自身的
      ``(turn_id, result.seq, call_id)``,供读端原样投影);同 ``(turn_id,
      call_id)`` 的多条孤儿同样最新 seq 胜——否则会投影出多条同
      tool_call_id 的 tool 消息,provider 直接拒收。

    纯读、有界(一次排序 + 一次线性扫)、永不抛:未知 kind/畸形 payload 一律
    跳过,与 status/文本无关(denied/not_executed 也算"有结果")。返回值:
    ``paired`` = 窗口键 → 进投影结果的 ``(seq, 列表位置)``;
    ``orphans`` = 配不上声明的**最新**结果自身键集合。
    """
    ordered = sorted(enumerate(items), key=lambda entry: (entry[1].seq, entry[0]))
    open_declarations: dict[tuple[str | None, str], DeclarationKey] = {}
    paired: dict[DeclarationKey, ResultRef] = {}
    orphan_latest: dict[tuple[str | None, str], ResultRef] = {}
    for position, item in ordered:
        if item.kind == ItemKind.MODEL_RESPONSE:
            raw_calls = item.payload.get("tool_calls")
            if not isinstance(raw_calls, (list, tuple)):
                continue
            calls = _reconstruct_tool_calls(raw_calls)
            if calls is None:
                continue  # 部分畸形:不开窗(结果降级为孤儿,保持可投影)
            for call in calls:
                open_declarations[(item.turn_id, call.call_id)] = (
                    item.turn_id,
                    item.seq,
                    call.call_id,
                )
        elif item.kind == ItemKind.TOOL_RESULT:
            call_id = item.payload.get("call_id")
            if not isinstance(call_id, str):
                continue
            declaration = open_declarations.get((item.turn_id, call_id))
            candidate = (item.seq, position)
            if declaration is None:
                holder = (item.turn_id, call_id)
                current_orphan = orphan_latest.get(holder)
                if current_orphan is None or candidate >= current_orphan:
                    orphan_latest[holder] = candidate
                continue
            current = paired.get(declaration)
            if current is None or candidate >= current:
                paired[declaration] = candidate
    orphans = {
        (turn_id, seq, call_id)
        for (turn_id, call_id), (seq, _position) in orphan_latest.items()
    }
    return paired, orphans


def to_messages(items: Sequence[Item]) -> list[ModelMessage]:
    """把(已折叠的)Item 序列投影为模型可见消息;未知 kind 与畸形 payload 跳过,永不抛。

    - user_message/steering → user(payload["text"])
    - model_response → assistant(payload["content"] + tool_calls 重建);content 缺失/非 str
      但 tool_calls 可重建时投 content=""(R7 读端 fail-open,保住配对 tool_result)。
      payload["reasoning"] 永不读取:思考文本一旦进助手历史就会被回灌 provider。
    - tool_result → tool(文本口径与 protocol.tools.ToolResult.text 一致,tool_call_id=payload["call_id"])
    - summary → user,前缀 "[conversation summary]\\n"
    - 其余 kind(turn_status/approval_*/tool_call/未知)→ 跳过

    配对(B31/R9/R10,判据唯一来源是 ``pair_tool_results``):投影按 seq 顺序
    进行(列表顺序不是事实,seq 才是),紧跟每个携带 tool_calls 的 assistant
    消息之后,按声明顺序逐条输出该声明的配对结果;没有可用结果(未闭合声明,
    或结果 payload 不可重建)时补 ``_UNPAIRED_TOOL_RESULT_NOTE`` 合成消息
    (R9,覆盖 cancelled/failed/abandoned-paused 轮次;INTERRUPTED 轮次由
    recovery 写入真实 not_executed/outcome_unknown 结果,属已配结果)。
    同一声明窗口内多条结果只投影最新 seq 的一条(R10;reconcile 在
    outcome_unknown 之后追加对账结果时旧结果仅留库审计),同一响应重复声明
    同一 call_id 时第一次消费真实结果、后续声明补合成——每个声明恰有一条
    tool 消息,provider 的 tool_use/tool_result 配对校验不会因复用 call_id
    或中断轮次而失败。

    配不上任何声明的**孤儿结果**在自身位置原样投影(fail-open 读:CLI 直调
    遗留等旧数据不静默丢失);配对成功的结果只在声明 assistant 后出现一次。
    turn_id 为 None 的 item 退化为 (None, seq, call_id),单线程行为不变。
    """
    ordered = sorted(enumerate(items), key=lambda entry: (entry[1].seq, entry[0]))
    paired, orphans = pair_tool_results(items)
    latest_position = {key: position for key, (_seq, position) in paired.items()}
    messages: list[ModelMessage] = []
    used_keys: set[DeclarationKey] = set()
    for _position, item in ordered:
        if item.kind == ItemKind.TOOL_RESULT:
            call_id = item.payload.get("call_id")
            if not isinstance(call_id, str):
                continue
            if (item.turn_id, item.seq, call_id) in orphans:
                message = _to_message(item)  # 孤儿:原样投影(fail-open,永不抛)
                if message is not None:
                    messages.append(message)
            # 已配对/被更新结果覆盖的结果由其声明窗口统一投影,自身位置不再重复
            continue
        message = _to_message(item)
        if message is None:
            continue
        messages.append(message)
        if message.role != "assistant" or not message.tool_calls:
            continue
        for call in message.tool_calls:
            key: DeclarationKey = (item.turn_id, item.seq, call.call_id)
            result_message: ModelMessage | None = None
            if key not in used_keys:
                used_keys.add(key)
                result_position = latest_position.get(key)
                if result_position is not None:
                    result_message = _to_message(items[result_position])
            if result_message is None:  # 未闭合/不可重建 → R9 合成
                result_message = ModelMessage(
                    role="tool",
                    content=_UNPAIRED_TOOL_RESULT_NOTE,
                    tool_call_id=call.call_id,
                )
            messages.append(result_message)
    return messages


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

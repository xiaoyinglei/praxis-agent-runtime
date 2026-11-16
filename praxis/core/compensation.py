"""Praxis 唯一清算协议(阶段4 T4;B14 对称性/B29 全组清算;裁决 R-PF-2 两层结构)。

终态(cancelled/complete 守卫)与恢复(resume 分类)共用这一份未决调用清算:
旧架构的取消路径零补偿 item、恢复路径只审最后窗口、同窗口第 1 个缺结果即
早退(旧 ``agent_runtime/harness/ownership.py:181-190`` 病灶)在此结构性消亡。

两层(R-PF-2:T11 只 import 判据层,不得另写窗口匹配):

1. ``unclosed_windows(items, tools) -> list[UnresolvedCall]`` —— 纯判据:
   枚举**所有**未闭合声明窗口的调用(含跨窗口孤儿声明,不限最后一条
   model_response——与 ``recovery.classify_pending`` 的"最后窗口"语义职责分清)。
   闭合判据唯一来源是 ``core.context.pair_tool_results``:键 =
   ``(turn_id, 声明它的 model_response.seq, call_id)``,同 turn 跨轮复用 call_id
   不互相覆盖;幂等判据 = pair 已配(补过结果的窗口第二次清算零产出)。
2. ``close_unresolved(items, tools, *, text_for) -> (entries, unknown_ids)`` ——
   在判据层上生成 tool_result entries:started(已有 tool_call item,副作用可能
   已发生)→ outcome_unknown;未 started → not_executed。清算永不重放——
   重放是 resume 决策矩阵(session)的事,不是终态清算的事。

``text_result_entry`` 是会话层/管道合成 tool_result 的唯一形状助手(与管道
落库字段约定一致,勿漂移);interrupted 文案逐字保留旧 session 常量(提取
零行为变化),cancelled 文案独立(取消不是进程死亡)。

读路径纪律(B1):畸形 payload 一律跳过、永不抛。窗口开闭严格度与
``pair_tool_results`` 逐字对齐——部分畸形的 model_response 不开窗(其结果
保持孤儿身份,由读侧 ``_UNPAIRED_TOOL_RESULT_NOTE`` 兜底投影;全组清算后
该合成文本降为兜底而非常态)。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from praxis.core.context import DeclarationKey, pair_tool_results
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelToolCall
from praxis.protocol.tools import ResultStatus, Tool
from praxis.store.log import ItemEntry

#: 清算文案(按终态语境):interrupted 两条逐字 = 旧 session 私有常量(提取零变化)。
NOT_EXECUTED_TEXTS: Final[Mapping[str, str]] = {
    "cancelled": "[not executed: turn cancelled before dispatch]",
    "interrupted": "[not executed: turn interrupted before dispatch]",
}
OUTCOME_UNKNOWN_TEXTS: Final[Mapping[str, str]] = {
    "cancelled": "[outcome unknown: turn cancelled during execution]",
    "interrupted": "[outcome unknown: process died during execution]",
}


@dataclass(frozen=True, slots=True)
class UnresolvedCall:
    """一条未闭合声明窗口里的调用及其清算信息(纯判据产物,无副作用)。

    - started:本窗口声明之后已有同 call_id 的 tool_call item(按最近声明归属,
      更早窗口同 id 的 dispatch 不算)——副作用可能已经发生;
    - idempotent:tools 表给出的重放安全性(未知工具一律 False);
    - declaration_seq:声明它的 model_response 的 seq(pipeline cancel 检查点
      比较 ``cancel_request.seq > declaration_seq`` 用)。
    """

    call: ModelToolCall
    started: bool
    idempotent: bool
    declaration_seq: int


def _window_calls(raw_calls: object) -> list[ModelToolCall] | None:
    """严格重建声明(与 context._reconstruct_tool_calls 同严格度:非序列或任一
    条目非法 → None = 整条响应不开窗)。判据必须与 pair_tool_results 的开窗
    行为逐字一致,否则清算会补到配对根本不承认的窗口上。"""
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


def unclosed_windows(items: Sequence[Item], tools: Mapping[str, Tool]) -> list[UnresolvedCall]:
    """枚举所有未闭合声明窗口的调用(B29:全组、跨窗口,无早退)。

    按 seq 升序单遍扫:model_response 开窗(同响应重复声明同 id 折叠进同一
    窗口键,只产一条);tool_call 归属给它之前最近的同 ``(turn_id, call_id)``
    声明(与 pair_tool_results 的结果归属同纪律)。闭合判据 = pair_tool_results
    的 paired 键集合——denied/not_executed/outcome_unknown 也算"有结果"。
    输出按 (声明 seq, 声明顺序) 升序;纯函数、永不抛。
    """
    paired, _orphans = pair_tool_results(items)
    ordered = sorted(items, key=lambda item: item.seq)
    open_declarations: dict[tuple[str | None, str], DeclarationKey] = {}
    started: set[DeclarationKey] = set()
    windows: list[tuple[DeclarationKey, ModelToolCall]] = []
    seen: set[DeclarationKey] = set()
    for item in ordered:
        if item.kind == ItemKind.MODEL_RESPONSE:
            calls = _window_calls(item.payload.get("tool_calls"))
            if calls is None:
                continue  # 部分畸形:不开窗(与 pair_tool_results 逐字对齐)
            for call in calls:
                key: DeclarationKey = (item.turn_id, item.seq, call.call_id)
                open_declarations[(item.turn_id, call.call_id)] = key
                if key not in seen:
                    seen.add(key)
                    windows.append((key, call))
        elif item.kind == ItemKind.TOOL_CALL:
            call_id = item.payload.get("call_id")
            if not isinstance(call_id, str):
                continue
            declaration = open_declarations.get((item.turn_id, call_id))
            if declaration is not None:
                started.add(declaration)
    unresolved: list[UnresolvedCall] = []
    for key, call in windows:
        if key in paired:
            continue  # 本窗口已有结果(幂等判据:pair 已配)
        tool = tools.get(call.name)
        unresolved.append(
            UnresolvedCall(
                call=call,
                started=key in started,
                idempotent=tool.idempotent if tool is not None else False,
                declaration_seq=key[1],
            )
        )
    return unresolved


def text_result_entry(
    call_id: str,
    tool_name: str,
    status: ResultStatus,
    text: str,
    metadata: Mapping[str, JsonValue],
) -> ItemEntry:
    """合成 tool_result item 的唯一形状助手:与管道落库字段一致(勿漂移)。

    原 session._text_result_entry 提取至此(形状逐字不变):清算/resume 对账
    共用,不存在第二份 payload 约定。
    """
    payload: dict[str, JsonValue] = {
        "call_id": call_id,
        "tool_name": tool_name,
        "status": status.value,
        "content": ({"type": "text", "data": {"text": text}},),
        "is_error": status is ResultStatus.ERROR,
        "error_code": "user_reconciled_failed" if status is ResultStatus.ERROR else None,
        "error_message": text if status is ResultStatus.ERROR else None,
        "truncated": False,
        "metadata": dict(metadata),
    }
    return ItemEntry(kind=ItemKind.TOOL_RESULT, payload=payload)


def settlement_outcome(
    started: bool, *, text_for: Literal["cancelled", "interrupted"]
) -> tuple[ResultStatus, str]:
    """唯一清算矩阵(T16 ㉔):started → outcome_unknown;未 started → not_executed。

    close_unresolved(终态清算)、resume 决策矩阵(session._classify_and_prepare)
    与 cancel 检查点(pipeline._cancel_refusal)共用这一份 status/文案映射——
    ``ResultStatus.NOT_EXECUTED`` 字面构造只在本模块(CI 规则 27/计划㉔)。
    消费面只取矩阵产物,不得另写第二份映射或裸构造 not_executed。
    """
    if started:
        return ResultStatus.OUTCOME_UNKNOWN, OUTCOME_UNKNOWN_TEXTS[text_for]
    return ResultStatus.NOT_EXECUTED, NOT_EXECUTED_TEXTS[text_for]


def close_unresolved(
    items: Sequence[Item],
    tools: Mapping[str, Tool],
    *,
    text_for: Literal["cancelled", "interrupted"],
) -> tuple[list[ItemEntry], list[str]]:
    """在判据层上生成全组清算 entries + outcome_unknown 清单(B29 无早退)。

    started → outcome_unknown(无论幂等与否:清算不重放,已 dispatch 的调用
    结局只能如实标未知);未 started → not_executed。返回的 entries 由调用方
    落库(本函数纯函数);unknown ids 供 TurnResult/PAUSED 展示面。
    """
    entries: list[ItemEntry] = []
    unknown_ids: list[str] = []
    for unresolved in unclosed_windows(items, tools):
        call = unresolved.call
        status, text = settlement_outcome(unresolved.started, text_for=text_for)
        entries.append(text_result_entry(call.call_id, call.name, status, text, {}))
        if unresolved.started:
            unknown_ids.append(call.call_id)
    return entries, unknown_ids
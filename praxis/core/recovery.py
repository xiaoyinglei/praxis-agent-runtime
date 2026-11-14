"""Praxis recovery:sweep 扫描与 classify_pending 待决工具调用分类(设计 §5.8)。

本文件是 praxis/ 中 ``transition(..., TurnStatus.INTERRUPTED, ...)`` 唯一合法
调用点(CI 硬规则 5 强制):interrupted 只能由 recovery 写入——人为中断走
pause/cancel,进程死亡走 sweep。

sweep(执行锁粒度 = thread,§5.1):
    遍历本库所有 status="running" 的 turn,先**试取**该 thread 的执行锁——
    - ThreadBusy:另一进程正在执行,跳过,绝不误伤活 turn;
    - 取得到:flock 随进程死亡由内核自动释放,取到即原主已死 →
      transition(INTERRUPTED, reason="executor_process_died"),
      turn_status item 由 write_status 自动携带 last_seq(打断瞬间日志里
      已存在的最后一条 seq),recovery 续跑据此决定从哪读。
    任何打开本库的进程都可触发(如 composition 启动时);自身全程只借锁判定
    生死,标记完立即释放锁。

classify_pending:resume 前的纯函数分类器(Task 9/14 消费),无副作用——
把最后一条 model_response 里尚未拿到 tool_result 的调用,按"是否已写下
tool_call item"(可能已有副作用)与"工具是否幂等"(能否安全重放)标注。
结果配对判据与投影共用 ``core.context.pair_tool_results``:调用身份是
"最后一条 model_response 的声明窗口",不是裸 call_id——同 turn 跨轮复用
call_id 时,第一轮的结果绝不能把第二轮尚未执行的同 id 调用判为已解决
(R-T7-1,否则 resume 静默丢掉一次调用)。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from praxis.core.context import pair_tool_results
from praxis.core.locks import ThreadBusy, thread_execution_lock
from praxis.core.statemachine import IllegalTransition, TurnStatus, transition
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelToolCall
from praxis.store.turns import list_turns

if TYPE_CHECKING:
    from praxis.core.events import EventBus
    from praxis.protocol.tools import Tool
    from praxis.store.log import Store


def sweep(store: Store, bus: EventBus, workspace: Path) -> list[str]:
    """清扫:running 且执行锁空闲(原主已死)的 turn → interrupted。

    返回被标记的 turn_id 列表(按 list_turns 的 created_at 顺序)。
    单条 turn 的任何失败绝不中断整次清扫:sweep 是 recovery 入口,
    漏扫其余 dead turn 的代价远大于跳过一条。
    """
    marked: list[str] = []
    for turn in list_turns(store, status=TurnStatus.RUNNING.value):
        try:
            with thread_execution_lock(workspace, turn.thread_id):
                transition(
                    store,
                    bus,
                    turn.turn_id,
                    TurnStatus.INTERRUPTED,
                    reason="executor_process_died",
                )
        except ThreadBusy:
            continue
        except IllegalTransition:
            # R6 竞态:快照与取锁之间,该 turn 已被活执行者合法改写(如 completed)
            # ——不再是 dead,跳过并继续清扫其余。执行锁保证这不是"执行中":
            # 活执行者改写状态全程持锁,能取到锁就说明它已经交还了该 thread。
            continue
        marked.append(turn.turn_id)
    return marked


@dataclass(frozen=True, slots=True)
class PendingCall:
    """一条已发出但未拿到 tool_result 的工具调用。

    - started:日志里已有对应 tool_call item —— 副作用可能已经发生;
    - idempotent:tools 表给出的重放安全性(未知工具一律按 False)。
    resume 决策矩阵(§5.9):未 started 的幂等调用可直接重放;
    已 started 或非幂等的必须人工确认后才能重试。
    """

    call: ModelToolCall
    started: bool
    idempotent: bool


def classify_pending(items: Sequence[Item], tools: Mapping[str, Tool]) -> list[PendingCall]:
    """从 items(一个 turn 的完整日志)提取最后一条 model_response 的待决调用。

    规则:
    - 只看**最后**一条 model_response(seq 最大)——更早轮次的调用已被模型
      自己接管;
    - 结果判据走 ``pair_tool_results`` 的声明窗口:只有配给**本条**
      model_response 的 tool_result 才算已解决。同 turn 跨轮复用 call_id 时,
      更早轮次的结果属于更早的声明窗口,不得吞掉本轮的调用(R-T7-1);
    - started = 存在**本条声明之后**(seq 更大)的同 call_id tool_call item
      ——更早轮次同 id 的 tool_call 不是本窗口的 dispatch;
    - idempotent = tools[name].idempotent,未知工具按 False;
    - 读路径零校验(B1):结构不符的条目静默跳过,绝不抛异常。
    """
    last_response: Item | None = None
    for item in items:
        if item.kind == ItemKind.MODEL_RESPONSE and (
            last_response is None or item.seq >= last_response.seq
        ):
            last_response = item

    if last_response is None:
        return []
    raw_calls = last_response.payload.get("tool_calls")
    if not isinstance(raw_calls, (list, tuple)):
        return []

    paired, _orphans = pair_tool_results(items)
    started_call_ids: set[str] = set()
    for item in items:
        if item.kind != ItemKind.TOOL_CALL or item.seq <= last_response.seq:
            continue
        call_id = item.payload.get("call_id")
        if isinstance(call_id, str) and item.turn_id == last_response.turn_id:
            started_call_ids.add(call_id)

    pending: list[PendingCall] = []
    for entry in raw_calls:
        if not isinstance(entry, Mapping):
            continue
        call_id = entry.get("call_id")
        name = entry.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            continue
        if (last_response.turn_id, last_response.seq, call_id) in paired:
            continue  # 本窗口已有结果(更早窗口的结果不算)
        raw_arguments = entry.get("arguments")
        arguments: Mapping[str, JsonValue] = raw_arguments if isinstance(raw_arguments, Mapping) else {}
        tool = tools.get(name)
        pending.append(
            PendingCall(
                call=ModelToolCall(call_id=call_id, name=name, arguments=arguments),
                started=call_id in started_call_ids,
                idempotent=tool.idempotent if tool is not None else False,
            )
        )
    return pending

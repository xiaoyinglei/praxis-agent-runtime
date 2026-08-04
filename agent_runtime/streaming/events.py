"""
StreamEvent 定义 — 流式输出的基础类型。

设计原则：
- 不可变（frozen=True），可安全在协程间传递
- 高频字段提到顶层，不全塞 metadata
- EventType 用 Enum，方便 UI 层 pattern match
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from agent_runtime.tools.tool import JsonValue


class EventType(StrEnum):
    """流式事件类型。"""

    # ── LLM 流式输出 ──────────────────────────────────────
    TEXT_DELTA = "text_delta"  # 文本增量
    THINKING_DELTA = "thinking_delta"  # 思考增量（extended thinking，可选）

    # ── 工具生命周期 ──────────────────────────────────────
    TOOL_USE_START = "tool_use_start"  # 工具开始执行
    TOOL_USE_PROGRESS = "tool_use_progress"  # 工具执行进度
    TOOL_USE_RESULT = "tool_use_result"  # 工具执行完成
    TOOL_USE_ERROR = "tool_use_error"  # 工具执行失败

    # ── 上下文管理 ────────────────────────────────────────
    COMPACT_LAYER = "compact_layer"  # 单层压缩完成

    # ── 计划状态 ──────────────────────────────────────────
    PLAN_UPDATED = "plan_updated"  # update_plan 已写入 canonical PlanState

    # ── 会话控制 ──────────────────────────────────────────
    TURN_START = "turn_start"  # 一轮开始
    TURN_END = "turn_end"  # 一轮结束
    LOOP_END = "loop_end"  # 循环结束
    HUMAN_INPUT_REQUIRED = "human_input_required"  # 已持久化的人机交互请求
    RECOVERY = "recovery"  # 恢复尝试
    ABORT = "abort"  # 用户取消

    # ── Token 预算 ────────────────────────────────────────
    BUDGET_UPDATE = "budget_update"  # 预算消耗更新


@dataclass(frozen=True)
class StreamEvent:
    """流式事件 — 不可变，可安全在协程间传递。"""

    type: EventType
    turn_id: str = ""
    iteration: int = 0
    sequence: int = 0
    timestamp_ms: int = 0
    data: dict[str, JsonValue] = field(default_factory=dict)
    span_id: str | None = None  # 关联同一个 tool 调用的 start/progress/result
    parent_id: str | None = None  # 子 agent / 嵌套事件

    def __post_init__(self) -> None:
        if self.timestamp_ms == 0:
            object.__setattr__(self, "timestamp_ms", _now_ms())


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


_sequence_counter = 0


def next_sequence() -> int:
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter


# ── 工厂函数 ──────────────────────────────────────────────


def text_delta(
    text: str,
    *,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.TEXT_DELTA,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        data={"text": text},
    )


def tool_use_start(
    tool_name: str,
    tool_id: str,
    *,
    input_preview: str = "",
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    span = f"tool:{tool_id}"
    return StreamEvent(
        type=EventType.TOOL_USE_START,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        span_id=span,
        data={
            "tool_name": tool_name,
            "tool_id": tool_id,
            "input_preview": input_preview,
        },
    )


def tool_use_progress(
    tool_id: str,
    progress: str,
    *,
    percent: float | None = None,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    span = f"tool:{tool_id}"
    d: dict[str, JsonValue] = {"tool_id": tool_id, "progress": progress}
    if percent is not None:
        d["percent"] = percent
    return StreamEvent(
        type=EventType.TOOL_USE_PROGRESS,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        span_id=span,
        data=d,
    )


def tool_use_result(
    tool_name: str,
    tool_id: str,
    result: JsonValue,
    *,
    elapsed_ms: float | None = 0,
    details: Mapping[str, JsonValue] | None = None,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    span = f"tool:{tool_id}"
    data: dict[str, JsonValue] = {
        "tool_name": tool_name,
        "tool_id": tool_id,
        "result": result,
    }
    if elapsed_ms is not None:
        data["elapsed_ms"] = elapsed_ms
    if details:
        data["details"] = dict(details)
    return StreamEvent(
        type=EventType.TOOL_USE_RESULT,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        span_id=span,
        data=data,
    )


def tool_use_error(
    tool_id: str,
    error: str,
    *,
    recoverable: bool | None = True,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    span = f"tool:{tool_id}"
    data: dict[str, JsonValue] = {
        "tool_id": tool_id,
        "error": error,
    }
    if recoverable is not None:
        data["recoverable"] = recoverable
    return StreamEvent(
        type=EventType.TOOL_USE_ERROR,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        span_id=span,
        data=data,
    )


def compact_layer(
    *,
    channels: Sequence[str],
    warnings: Sequence[str],
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.COMPACT_LAYER,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        data={
            "channels": tuple(channels),
            "warnings": tuple(warnings),
        },
    )


def turn_start(*, turn_id: str = "", iteration: int = 0) -> StreamEvent:
    return StreamEvent(
        type=EventType.TURN_START,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
    )


def turn_end(
    *,
    turn_id: str = "",
    iteration: int = 0,
    stop_reason: str = "",
) -> StreamEvent:
    return StreamEvent(
        type=EventType.TURN_END,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        data={"stop_reason": stop_reason},
    )


def loop_end(
    *,
    reason: str,
    turn_id: str = "",
    total_turns: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.LOOP_END,
        turn_id=turn_id,
        sequence=next_sequence(),
        data={"reason": reason, "total_turns": total_turns},
    )


def recovery_event(
    strategy: str,
    detail: str = "",
    *,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.RECOVERY,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        data={"strategy": strategy, "detail": detail},
    )

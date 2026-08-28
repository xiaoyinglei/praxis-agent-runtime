"""
StreamEvent 定义 — 流式输出的基础类型。

设计原则：
- 不可变（frozen=True），可安全在协程间传递
- 高频字段提到顶层，不全塞 metadata
- EventType 用 Enum，方便 UI 层 pattern match
"""

from __future__ import annotations

import time
from dataclasses import InitVar, dataclass, field
from enum import StrEnum
from typing import Literal

from rag.agent.tools.tool import JsonValue


class EventType(StrEnum):
    """流式事件类型。"""

    # ── Canonical protocol v2 ─────────────────────────────
    TURN_STARTED = "turn_started"
    TURN_PAUSED = "turn_paused"
    TURN_RESUMED = "turn_resumed"
    TURN_CANCELLATION_REQUESTED = "turn_cancellation_requested"
    TURN_COMPLETED = "turn_completed"
    TURN_ABORTED = "turn_aborted"
    ITEM_STARTED = "item_started"
    ITEM_DELTA = "item_delta"
    ITEM_COMPLETED = "item_completed"

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


class TurnItemKind(StrEnum):
    AGENT_MESSAGE = "agent_message"
    REASONING = "reasoning"
    PLAN = "plan"
    TOOL = "tool"
    COMMAND = "command"
    RECONCILIATION = "reconciliation"
    LEGACY_MESSAGE = "legacy_message"


class ItemDeltaKind(StrEnum):
    TEXT = "text"
    REASONING = "reasoning"
    PLAN = "plan"
    TOOL_PROGRESS = "tool_progress"
    COMMAND_STDOUT = "command_stdout"
    COMMAND_STDERR = "command_stderr"


class ItemStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OUTCOME_UNKNOWN = "outcome_unknown"


_TURN_EVENTS = frozenset(
    {
        EventType.TURN_STARTED,
        EventType.TURN_PAUSED,
        EventType.TURN_RESUMED,
        EventType.TURN_CANCELLATION_REQUESTED,
        EventType.TURN_COMPLETED,
        EventType.TURN_ABORTED,
    }
)
_ITEM_EVENTS = frozenset(
    {EventType.ITEM_STARTED, EventType.ITEM_DELTA, EventType.ITEM_COMPLETED}
)


@dataclass(frozen=True)
class StreamEvent:
    """Canonical v2 event envelope plus import-compatible legacy event values."""

    protocol_version: Literal[2] = field(default=2, init=False)
    type: EventType
    turn_id: str = ""
    item_id: str | None = None
    item_kind: TurnItemKind | None = None
    delta_kind: ItemDeltaKind | None = None
    status: ItemStatus | None = None
    iteration: int = 0
    sequence: int = 0
    timestamp_ms: int = 0
    data: dict[str, JsonValue] = field(default_factory=dict)
    error: str | None = None
    parent_item_id: str | None = None
    span_id: InitVar[str | None] = None
    parent_id: InitVar[str | None] = None

    def __post_init__(
        self,
        span_id: str | None,
        parent_id: str | None,
    ) -> None:
        del span_id
        if self.parent_item_id is None and parent_id is not None:
            object.__setattr__(self, "parent_item_id", parent_id)
        if self.timestamp_ms == 0:
            object.__setattr__(self, "timestamp_ms", _now_ms())
        if self.type in _TURN_EVENTS:
            self._validate_turn_event()
        elif self.type in _ITEM_EVENTS:
            self._validate_item_event()

    def _validate_turn_event(self) -> None:
        if not self.turn_id:
            raise ValueError("turn_id is required for canonical Turn events")
        if any(
            value is not None
            for value in (
                self.item_id,
                self.item_kind,
                self.delta_kind,
                self.status,
                self.error,
                self.parent_item_id,
            )
        ):
            raise ValueError("item fields are forbidden on canonical Turn events")

    def _validate_item_event(self) -> None:
        if not self.turn_id:
            raise ValueError("turn_id is required for canonical Item events")
        if not self.item_id:
            raise ValueError("item_id is required for canonical Item events")
        if self.item_kind is None:
            raise ValueError("item_kind is required for canonical Item events")
        if self.type is EventType.ITEM_STARTED:
            if self.delta_kind is not None or self.status is not None or self.error is not None:
                raise ValueError("delta_kind, status, and error are forbidden on item_started")
            return
        if self.type is EventType.ITEM_DELTA:
            if self.delta_kind is None:
                raise ValueError("delta_kind is required for item_delta")
            if self.status is not None or self.error is not None:
                raise ValueError("status and error are forbidden on item_delta")
            delta = self.data.get("delta")
            if not isinstance(delta, str):
                raise ValueError("data.delta must be a string for item_delta")
            return
        if self.delta_kind is not None:
            raise ValueError("delta_kind is forbidden on item_completed")
        if self.status is None:
            raise ValueError("status is required for item_completed")
        if self.status in {ItemStatus.FAILED, ItemStatus.OUTCOME_UNKNOWN}:
            if not self.error:
                raise ValueError("error is required for failed or outcome_unknown item_completed")
        elif self.status is ItemStatus.SUCCESS and self.error is not None:
            raise ValueError("error is forbidden for successful item_completed")


def _now_ms() -> int:
    return int(time.time() * 1000)


_sequence_counter = 0


def next_sequence() -> int:
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter


# ── 工厂函数 ──────────────────────────────────────────────


def turn_started(turn_id: str) -> StreamEvent:
    return _turn_lifecycle(EventType.TURN_STARTED, turn_id, status="running")


def turn_paused(turn_id: str, *, reason: str) -> StreamEvent:
    return _turn_lifecycle(
        EventType.TURN_PAUSED,
        turn_id,
        status="paused",
        reason=reason,
    )


def turn_resumed(turn_id: str) -> StreamEvent:
    return _turn_lifecycle(EventType.TURN_RESUMED, turn_id, status="running")


def turn_cancellation_requested(turn_id: str) -> StreamEvent:
    return _turn_lifecycle(
        EventType.TURN_CANCELLATION_REQUESTED,
        turn_id,
        status="cancelling",
    )


def turn_completed(
    turn_id: str,
    *,
    status: Literal["completed", "failed"] = "completed",
    reason: str | None = None,
) -> StreamEvent:
    return _turn_lifecycle(
        EventType.TURN_COMPLETED,
        turn_id,
        status=status,
        reason=reason,
    )


def turn_aborted(turn_id: str, *, reason: str) -> StreamEvent:
    return _turn_lifecycle(
        EventType.TURN_ABORTED,
        turn_id,
        status="aborted",
        reason=reason,
    )


def _turn_lifecycle(
    event_type: EventType,
    turn_id: str,
    *,
    status: str,
    reason: str | None = None,
) -> StreamEvent:
    data: dict[str, JsonValue] = {"status": status}
    if reason is not None:
        data["reason"] = reason
    return StreamEvent(
        type=event_type,
        turn_id=turn_id,
        sequence=next_sequence(),
        data=data,
    )


def item_started(
    *,
    turn_id: str,
    item_id: str,
    item_kind: TurnItemKind,
    iteration: int = 0,
    data: dict[str, JsonValue] | None = None,
    parent_item_id: str | None = None,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.ITEM_STARTED,
        turn_id=turn_id,
        item_id=item_id,
        item_kind=item_kind,
        iteration=iteration,
        sequence=next_sequence(),
        data={} if data is None else dict(data),
        parent_item_id=parent_item_id,
    )


def item_delta(
    *,
    turn_id: str,
    item_id: str,
    item_kind: TurnItemKind,
    delta_kind: ItemDeltaKind,
    delta: str,
    iteration: int = 0,
    parent_item_id: str | None = None,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.ITEM_DELTA,
        turn_id=turn_id,
        item_id=item_id,
        item_kind=item_kind,
        delta_kind=delta_kind,
        iteration=iteration,
        sequence=next_sequence(),
        data={"delta": delta},
        parent_item_id=parent_item_id,
    )


def item_completed(
    *,
    turn_id: str,
    item_id: str,
    item_kind: TurnItemKind,
    status: ItemStatus,
    data: dict[str, JsonValue],
    iteration: int = 0,
    error: str | None = None,
    parent_item_id: str | None = None,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.ITEM_COMPLETED,
        turn_id=turn_id,
        item_id=item_id,
        item_kind=item_kind,
        status=status,
        iteration=iteration,
        sequence=next_sequence(),
        data=dict(data),
        error=error,
        parent_item_id=parent_item_id,
    )


def text_delta(
    text: str,
    *,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.TEXT_DELTA,
        turn_id=turn_id,
        item_kind=TurnItemKind.AGENT_MESSAGE,
        delta_kind=ItemDeltaKind.TEXT,
        iteration=iteration,
        sequence=next_sequence(),
        data={"text": text},
    )


def thinking_delta(
    text: str,
    *,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.THINKING_DELTA,
        turn_id=turn_id,
        item_kind=TurnItemKind.REASONING,
        delta_kind=ItemDeltaKind.REASONING,
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
    return StreamEvent(
        type=EventType.TOOL_USE_START,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        item_id=tool_id,
        item_kind=TurnItemKind.TOOL,
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
    d: dict[str, JsonValue] = {"tool_id": tool_id, "progress": progress}
    if percent is not None:
        d["percent"] = percent
    return StreamEvent(
        type=EventType.TOOL_USE_PROGRESS,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        item_id=tool_id,
        item_kind=TurnItemKind.TOOL,
        data=d,
    )


def tool_use_result(
    tool_name: str,
    tool_id: str,
    result: JsonValue,
    *,
    elapsed_ms: float = 0,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.TOOL_USE_RESULT,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        item_id=tool_id,
        item_kind=TurnItemKind.TOOL,
        data={
            "tool_name": tool_name,
            "tool_id": tool_id,
            "result": result,
            "elapsed_ms": elapsed_ms,
        },
    )


def tool_use_error(
    tool_id: str,
    error: str,
    *,
    recoverable: bool = True,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.TOOL_USE_ERROR,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        item_id=tool_id,
        item_kind=TurnItemKind.TOOL,
        data={
            "tool_id": tool_id,
            "error": error,
            "recoverable": recoverable,
        },
    )


def compact_layer(
    layer_name: str,
    before_tokens: int,
    after_tokens: int,
    *,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.COMPACT_LAYER,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        data={
            "layer": layer_name,
            "before": before_tokens,
            "after": after_tokens,
            "reduction": before_tokens - after_tokens,
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


def budget_update(
    used: int,
    remaining: int,
    *,
    turn_id: str = "",
    iteration: int = 0,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.BUDGET_UPDATE,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        data={"used": used, "remaining": remaining},
    )

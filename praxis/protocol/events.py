"""协议层运行事件:turn 生命周期中对外广播的最小不可变事实(T6/T7 消费)。

事件是 Item 流之外的进程内通知通道,不进 append-only log;
字段刻意扁平(纯 str),让 UI/CLI 无需理解 payload 结构即可渲染。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TurnStatusChanged:
    turn_id: str
    thread_id: str
    from_status: str
    to_status: str
    reason: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    turn_id: str
    call_id: str
    tool_name: str


@dataclass(frozen=True, slots=True)
class ToolFinished:
    turn_id: str
    call_id: str
    status: str


@dataclass(frozen=True, slots=True)
class ApprovalRequested:
    turn_id: str
    call_id: str
    tool_name: str
    args_digest: str


@dataclass(frozen=True, slots=True)
class ModelResponseReceived:
    turn_id: str
    content_preview: str


@dataclass(frozen=True, slots=True)
class ModelOutputDelta:
    """最终成功那一次尝试的增量(重试期间绝不 emit:失败首帧会留下重影)。"""

    turn_id: str
    kind: Literal["text", "reasoning", "tool_arguments"]
    text: str


@dataclass(frozen=True, slots=True)
class ModelRetried:
    """一次自动重试事实的提示事件;持久事实仍以 model_attempt item 为准。"""

    turn_id: str
    attempt: int
    kind: str
    retry_at: float


type Event = (
    TurnStatusChanged
    | ToolStarted
    | ToolFinished
    | ApprovalRequested
    | ModelResponseReceived
    | ModelOutputDelta
    | ModelRetried
)

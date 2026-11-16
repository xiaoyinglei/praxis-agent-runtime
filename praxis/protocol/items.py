"""协议层 Item:唯一事实源的最小记录单元(设计 §4.2)。

kind 字段类型是 str 而非 ItemKind——ItemKind 只是写端命名约定,读端不校验白名单。
旧架构里新增 kind 需要全链路改枚举的 B1 问题,在 praxis 中结构性消亡:
未知 kind 的 Item 必须能被原样读出、存储与展示。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from praxis.protocol.json import JsonValue


class ItemKind(StrEnum):
    USER_MESSAGE = "user_message"
    STEERING = "steering"
    MODEL_RESPONSE = "model_response"
    MODEL_ATTEMPT = "model_attempt"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_DECISION = "approval_decision"
    HOOK_INVOCATION = "hook_invocation"
    TURN_STATUS = "turn_status"
    SUMMARY = "summary"
    COMPACTION_BOUNDARY = "compaction_boundary"
    # 阶段4:预算/取消/子分配(设计 §5.5/§5.13)。追加在末尾,既有顺序不变。
    CANCEL_REQUEST = "cancel_request"
    BUDGET_EXTENSION = "budget_extension"
    CHILD_ALLOCATION = "child_allocation"


@dataclass(frozen=True, slots=True)
class Item:
    """一条 append-only 记录;payload 为 JSON 兼容字典,kind 为开放集合。"""

    item_id: str
    thread_id: str
    turn_id: str | None
    seq: int
    kind: str
    payload: Mapping[str, JsonValue]
    created_at: float

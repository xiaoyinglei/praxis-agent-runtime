from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HumanInputRequestIdMismatchError(RuntimeError):
    """A resume response does not match the currently persisted request."""


class ToolCallSummary(BaseModel):
    """面向用户的工具摘要，不暴露完整 ToolCallPlan。"""

    tool_call_id: str
    approval_id: str | None = None
    tool_name: str
    args_preview: str  # 如 "query='公积金政策', top_k=8"
    risk_level: str = "low"  # "low" | "medium" | "high"
    reason: str = ""  # 为什么需要审批


class HumanInputRequest(BaseModel):
    """Agent 暂停时向用户发送的输入请求。

    由工具执行服务（经 ApprovalPolicy）或模型 turn provider 生成，
    持久化在 canonical loop checkpoint 中。
    """

    request_id: str  # 唯一 ID，如 "hir_1234abcd"
    kind: Literal[
        "tool_approval",
        "tool_reconciliation",
        "choice",
        "clarification",
    ]
    question: str  # 面向用户的自然语言问题
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)
    context: dict[str, object] = Field(default_factory=dict)
    options: list[str] = Field(default_factory=list)


class HumanInputResponse(BaseModel):
    """用户对 HumanInputRequest 的响应。

    由 AgentLoop resume 边界接收、校验并写回 loop state。
    """

    request_id: str
    decision: Literal[
        "allow_once",
        "deny",
        "continue",
        "abort",
        "mark_completed",
        "mark_failed",
    ]
    approved_tool_call_ids: list[str] = Field(default_factory=list)
    denied_tool_call_ids: list[str] = Field(default_factory=list)
    user_message: str | None = None

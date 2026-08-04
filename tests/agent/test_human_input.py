from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_runtime.core.human_input import (
    HumanInputRequest,
    HumanInputResponse,
    ToolCallSummary,
)


class TestToolCallSummary:
    def test_minimal_summary(self) -> None:
        s = ToolCallSummary(
            tool_call_id="tc_001",
            tool_name="vector_search",
            args_preview="query='公积金', top_k=8",
        )
        assert s.tool_call_id == "tc_001"
        assert s.risk_level == "low"
        assert s.reason == ""

    def test_high_risk_summary(self) -> None:
        s = ToolCallSummary(
            tool_call_id="tc_002",
            tool_name="kg_upsert",
            args_preview="node=公积金, data=...",
            risk_level="high",
            reason="知识图谱写入需要审批",
        )
        assert s.risk_level == "high"
        assert "知识图谱" in s.reason


class TestHumanInputRequest:
    def test_tool_approval_request(self) -> None:
        req = HumanInputRequest(
            request_id="hir_001",
            kind="tool_approval",
            question="确认执行以下工具？",
            tool_calls=[
                ToolCallSummary(
                    tool_call_id="tc_001",
                    tool_name="vector_search",
                    args_preview="query='公积金', top_k=8",
                    risk_level="low",
                    reason="向量检索",
                ),
                ToolCallSummary(
                    tool_call_id="tc_002",
                    tool_name="kg_upsert",
                    args_preview="node=公积金, data=...",
                    risk_level="high",
                    reason="知识图谱写入需要审批",
                ),
            ],
            context={"task": "查询公积金政策"},
            options=["allow_once", "deny", "abort"],
        )
        assert req.request_id == "hir_001"
        assert req.kind == "tool_approval"
        assert len(req.tool_calls) == 2
        assert req.options == ["allow_once", "deny", "abort"]

    def test_choice_request(self) -> None:
        req = HumanInputRequest(
            request_id="hir_002",
            kind="choice",
            question="请选择数据源",
            options=["source_a", "source_b"],
        )
        assert req.kind == "choice"
        assert len(req.tool_calls) == 0

    def test_tool_reconciliation_request(self) -> None:
        req = HumanInputRequest(
            request_id="hir_reconcile",
            kind="tool_reconciliation",
            question="工具状态不明确，请选择恢复方式。",
            context={
                "tool_call_id": "tc_001",
                "operation_id": "op_001",
            },
            options=[
                "mark_completed",
                "mark_failed",
            ],
        )

        assert req.kind == "tool_reconciliation"
        assert req.context["operation_id"] == "op_001"


class TestHumanInputResponse:
    def test_allow_once_response(self) -> None:
        resp = HumanInputResponse(
            request_id="hir_001",
            decision="allow_once",
            approved_tool_call_ids=["tc_001", "tc_002"],
        )
        assert resp.decision == "allow_once"
        assert resp.approved_tool_call_ids == ["tc_001", "tc_002"]

    def test_deny_response(self) -> None:
        resp = HumanInputResponse(
            request_id="hir_001",
            decision="deny",
            denied_tool_call_ids=["tc_002"],
            user_message="知识图谱写入暂不执行",
        )
        assert resp.decision == "deny"
        assert resp.denied_tool_call_ids == ["tc_002"]
        assert resp.user_message == "知识图谱写入暂不执行"

    def test_abort_response(self) -> None:
        resp = HumanInputResponse(
            request_id="hir_001",
            decision="abort",
        )
        assert resp.decision == "abort"
        assert resp.approved_tool_call_ids == []
        assert resp.denied_tool_call_ids == []

    def test_invalid_decision_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HumanInputResponse(
                request_id="hir_001",
                decision="invalid_choice",
            )

    @pytest.mark.parametrize(
        "decision",
        ["mark_completed", "mark_failed"],
    )
    def test_tool_reconciliation_decisions(self, decision: str) -> None:
        response = HumanInputResponse(
            request_id="hir_reconcile",
            decision=decision,
        )

        assert response.decision == decision

    def test_unknown_outcome_replay_decision_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HumanInputResponse(
                request_id="hir_reconcile",
                decision="retry_new_operation",
            )

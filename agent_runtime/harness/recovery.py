"""Read durable recovery facts; execution remains owned by Session/TurnExecutor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_runtime.harness.ownership import OWNERSHIP_REVISION, ConversationBusyError, assert_recovery_leases
from agent_runtime.harness.state import RuntimeState


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    turn_id: str
    status: str
    route: str
    actions: tuple[str, ...]
    reason_code: str
    evidence_ids: tuple[str, ...]
    request_id: str | None = None
    requires_input: bool = False
    question: str | None = None
    choices: tuple[str, ...] = ()

    def accepts(self, action: str) -> bool:
        if action == "approve":
            action = "allow_once"
        elif action == "continue" and self.route == "retry_model":
            action = "retry"
        elif action == "retry" and self.route != "retry_model":
            action = "continue"
        if self.route == "approval_resolved":
            return action in {"allow_once", "deny"}  # Acknowledge a reply; never execute a terminal Turn.
        return action in self.actions

    def as_dict(self) -> dict[str, Any]:
        return {"turn_id": self.turn_id, "status": self.status, "actions": list(self.actions),
                "reason_code": self.reason_code, "evidence_ids": list(self.evidence_ids),
                "request_id": self.request_id, "requires_input": self.requires_input, "question": self.question,
                "choices": list(self.choices)}


def pending_context_overflow(store: RuntimeState, turn_id: str) -> bool:
    operations = [op for op in store.list_model_operations(turn_id)
                  if op.request_ref.get("purpose") != "context_summary"]
    return bool(operations and any(
        record.record_type == "model_attempt_rejected"
        and record.payload.get("operation_id") == operations[-1].operation_id
        and record.payload.get("error_type") == "context_overflow"
        for record in store.list_turn_records(turn_id)
    ))


def pending_input_supersession(store: RuntimeState, turn_id: str) -> bool:
    operations = store.list_model_operations(turn_id)
    if not operations or operations[-1].status != "failed":
        return False
    attempt = operations[-1].active_attempt_id
    records = store.list_turn_records(turn_id)
    return any(record.record_type == "model_attempt_rejected" and record.payload.get("attempt_id") == attempt
               and record.payload.get("error_type") == "input_superseded" for record in records) and not any(
        record.record_type == "model_attempt_dispatched" and record.payload.get("attempt_id") == attempt
        for record in records
    )


def recovery_plan(store: RuntimeState, turn_id: str) -> RecoveryPlan:
    turn = store.read_turn(turn_id)
    models = store.list_model_operations(turn_id)
    tools = store.list_tool_operations(turn_id)
    pending = tuple(item for item in store.list_interactions(turn_id) if item.status == "pending")
    uncertain = tuple(op for op in tools if op.status in {"running", "unknown"})
    owned = turn.binding_manifest.get("execution_ownership") == OWNERSHIP_REVISION
    abort = ("abort",) if (turn.status in {"paused", "interrupted"}
                          or turn.status == "running" and owned) and not uncertain else ()

    def plan(route: str, actions: tuple[str, ...], reason: str, *evidence: str,
             request_id: str | None = None, requires_input: bool = False, question: str | None = None,
             choices: tuple[str, ...] = ()) -> RecoveryPlan:
        return RecoveryPlan(turn_id, turn.status, route, actions + abort, reason, (turn_id, *evidence),
                            request_id, requires_input, question, choices)

    if turn.status not in {"running", "paused", "interrupted"}:
        resolved = [item for item in store.list_interactions(turn_id)
                    if item.kind == "tool_approval" and item.status == "resolved"]
        if turn.status == "completed" and resolved:
            return plan("approval_resolved", (), "approval_already_resolved", resolved[-1].request_id,
                        request_id=resolved[-1].request_id)
        return plan("terminal", (), "turn_terminal")
    for operation in models:
        try:
            store.request_input_revision(turn_id, operation.request_ref)
        except ValueError:
            # Keep safe cancellation available, but never execute unsupported
            # persisted requests or guess which user inputs they consumed.
            return plan("unsupported_request", (), "input_revision_unsupported", operation.operation_id)
    if any(item.kind == "tool_reconciliation" for item in pending) or any(op.status == "unknown" for op in uncertain):
        request_id = next((item.request_id for item in pending if item.kind == "tool_reconciliation"), None)
        return plan("reconcile", (), "tool_outcome_unknown", *(op.operation_id for op in uncertain),
                    *((request_id,) if request_id is not None else ()), request_id=request_id)
    dispatched = tuple(op for op in models if op.status == "dispatched")
    if dispatched or uncertain:
        try:
            assert_recovery_leases(owned=owned,
                leases=(*(store.list_model_attempts(op.operation_id)[-1].lease_expires_at for op in dispatched),
                        *(op.lease_expires_at for op in uncertain)))
        except ConversationBusyError:
            return RecoveryPlan(turn_id, turn.status, "unavailable", (), "execution_lease_active", (turn_id,))
        return plan("recover_owner", ("continue",), "execution_owner_recovery",
                    *(op.operation_id for op in dispatched), *(op.operation_id for op in uncertain))
    if pending and store.turn_cancellation_pending(turn_id):
        return plan("unavailable", (), "cancellation_pending", *(item.request_id for item in pending))
    if len(pending) == 1:
        request = pending[0]
        question = str(request.request.get("question") or request.request.get("reason") or request.kind)
        if request.kind == "tool_approval":
            actions: tuple[str, ...] = ("allow_once", "deny", "recheck")
            if request.request.get("session_grant"):
                actions = ("allow_once", "allow_session", "deny", "recheck")
            if models:
                if store.user_input_revision(turn_id) != store.request_input_revision(turn_id, models[-1].request_ref):
                    actions += ("continue",)
            return plan("approval", actions, "tool_approval_required", request.request_id,
                        request_id=request.request_id, question=question)
        if request.kind in {"clarification", "choice"}:
            return plan("input", ("continue",), request.kind + "_required", request.request_id,
                        request_id=request.request_id, requires_input=True, question=question,
                        choices=tuple(request.request["options"]) if request.kind == "choice" else ())
    if pending:
        return plan("unavailable", (), "pending_interaction", *(item.request_id for item in pending))
    if turn.status == "paused" and store.turn_cancellation_pending(turn_id):
        return plan("resume_cancelled", ("continue",), "cancellation_settled")
    unknown = tuple(op for op in models if op.status == "unknown")
    if len(unknown) == 1:
        return plan("retry_model", ("retry",), "model_outcome_unknown", unknown[0].operation_id,
                    request_id=unknown[0].operation_id)
    for request in store.list_interactions(turn_id):
        if (request.kind == "tool_approval" and request.status == "resolved"
                and request.response.get("decision") == "approve" and request.operation_id is not None):
            tool_operation = store.read_tool_operation(request.operation_id)
            if tool_operation.status == "ready" and tool_operation.approval_request_id == request.request_id:
                return plan("approved_ready", ("continue", "allow_once"), "approved_tool_not_started",
                            request.request_id, tool_operation.operation_id, request_id=request.request_id)
    if turn.status == "running" and models:
        operation = models[-1]
        if operation.status == "completed" and operation.response_item_id is not None:
            response = store.read_item(operation.response_item_id)
            if response.kind == "model_response":
                return plan("committed_response", ("continue",), "model_response_committed",
                            operation.operation_id, response.item_id)
        if operation.status == "prepared":
            return plan("prepared_model", ("continue",), "model_not_dispatched", operation.operation_id)
    if pending_context_overflow(store, turn_id):
        return plan("context_overflow", ("continue",), "context_overflow_rejected")
    if (turn.status == "running" and models and models[-1].status == "completed"
            and models[-1].request_ref.get("purpose") == "context_summary" and models[-1].response_item_id is not None):
        return plan("committed_response", ("continue",), "summary_response_committed",
                    models[-1].operation_id, models[-1].response_item_id)
    if turn.status == "running" and pending_input_supersession(store, turn_id):
        return plan("input_superseded", ("continue",), "request_superseded", models[-1].operation_id)
    if (turn.status == "running" and not models and not tools
            and turn.binding_manifest.get("execution_ownership") == OWNERSHIP_REVISION):
        return plan("first_dispatch", ("continue",), "before_first_dispatch")
    return plan("unavailable", (), "no_resume_point")

"""Read durable recovery facts; execution remains owned by Session/TurnRunner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agent_runtime.harness.ownership import OWNERSHIP_REVISION, ConversationBusyError, assert_recovery_leases
from agent_runtime.harness.runtime_store import RuntimeStore


class ContinuationKind(StrEnum):
    APPROVAL_RESOLVED = "approval_resolved"
    CHILD_WAIT = "child_wait"
    TERMINAL = "terminal"
    UNSUPPORTED_REQUEST = "unsupported_request"
    RECONCILE = "reconcile"
    UNAVAILABLE = "unavailable"
    RECOVER_OWNER = "recover_owner"
    APPROVAL = "approval"
    INPUT = "input"
    RESUME_CANCELLED = "resume_cancelled"
    RETRY_MODEL = "retry_model"
    WAIT_RETRY = "wait_retry"
    APPROVED_READY = "approved_ready"
    COMMITTED_RESPONSE = "committed_response"
    PREPARED_MODEL = "prepared_model"
    CONTEXT_OVERFLOW = "context_overflow"
    RESUME_COMPACTION = "resume_compaction"
    INPUT_SUPERSEDED = "input_superseded"
    FIRST_DISPATCH = "first_dispatch"
    NEXT_STEP = "next_step"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    turn_id: str
    status: str
    kind: ContinuationKind
    actions: tuple[str, ...]
    reason_code: str
    evidence_ids: tuple[str, ...]
    request_id: str | None = None
    requires_input: bool = False
    question: str | None = None
    choices: tuple[str, ...] = ()
    operation_id: str | None = None
    response_item_id: str | None = None
    expected_generation: int | None = None
    expected_sequence: int = 0

    @property
    def automatically_resumable(self) -> bool:
        """Continuation authority that needs no new user decision.

        Owner recovery only fences lost dispatches. The decision must be read
        again afterwards, before an unknown result can become a retry.
        """
        return not self.requires_input and self.kind in {
            ContinuationKind.TERMINAL,
            ContinuationKind.APPROVAL_RESOLVED,
            ContinuationKind.RECOVER_OWNER,
            ContinuationKind.CANCEL,
            ContinuationKind.CHILD_WAIT,
            ContinuationKind.WAIT_RETRY,
            ContinuationKind.APPROVED_READY,
            ContinuationKind.COMMITTED_RESPONSE,
            ContinuationKind.PREPARED_MODEL,
            ContinuationKind.CONTEXT_OVERFLOW,
            ContinuationKind.RESUME_COMPACTION,
            ContinuationKind.INPUT_SUPERSEDED,
            ContinuationKind.FIRST_DISPATCH,
            ContinuationKind.NEXT_STEP,
        }

    def accepts(self, action: str) -> bool:
        if action == "approve":
            action = "allow_once"
        elif action == "continue" and self.kind == "retry_model":
            action = "retry"
        elif action == "retry" and self.kind != "retry_model":
            action = "continue"
        if self.kind == "approval_resolved":
            return action in {"allow_once", "deny"}  # Acknowledge a reply; never execute a terminal Turn.
        return action in self.actions

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "status": self.status,
            "actions": list(self.actions),
            "reason_code": self.reason_code,
            "evidence_ids": list(self.evidence_ids),
            "request_id": self.request_id,
            "requires_input": self.requires_input,
            "question": self.question,
            "choices": list(self.choices),
        }


def pending_context_overflow(store: RuntimeStore, turn_id: str) -> bool:
    operations = [
        op for op in store.list_model_operations(turn_id) if op.request_ref.get("purpose") != "context_summary"
    ]
    return bool(
        operations
        and any(
            record.record_type == "model_attempt_rejected"
            and record.payload.get("operation_id") == operations[-1].operation_id
            and record.payload.get("error_type") == "context_overflow"
            for record in store.list_turn_records(turn_id)
        )
    )


def pending_input_supersession(store: RuntimeStore, turn_id: str) -> bool:
    operations = store.list_model_operations(turn_id)
    if not operations or operations[-1].status != "failed":
        return False
    attempt = operations[-1].active_attempt_id
    records = store.list_turn_records(turn_id)
    if any(
        record.record_type == "model_retry_cancelled"
        and record.payload.get("attempt_id") == attempt
        and record.payload.get("reason_code")
        in {"input_superseded", "context_source_superseded", "model_retry_cancelled"}
        for record in records
    ):
        return True
    if any(
        record.record_type == "model_attempt_rejected"
        and record.payload.get("attempt_id") == attempt
        and record.payload.get("error_type") == "context_source_superseded"
        for record in records
    ):
        return True
    return any(
        record.record_type == "model_attempt_rejected"
        and record.payload.get("attempt_id") == attempt
        and record.payload.get("error_type") == "input_superseded"
        for record in records
    ) and not any(
        record.record_type == "model_attempt_dispatched" and record.payload.get("attempt_id") == attempt
        for record in records
    )


def recovery_plan(store: RuntimeStore, turn_id: str) -> RecoveryDecision:
    """Inspection and execution share one consistent, detached decision."""
    return store.read_snapshot(lambda: _recovery_decision(store, turn_id))


def active_capacity_root(store: RuntimeStore, turn_id: str) -> str | None:
    """A rejection chain survives input supersession until a successful consumed Step."""

    def read() -> str | None:
        from agent_runtime.harness.execution_facts import handled_response_ids

        records = store.list_turn_records(turn_id)
        handled = handled_response_ids(records)
        rejected = {
            r.payload["operation_id"]
            for r in records
            if r.record_type == "model_attempt_rejected" and r.payload.get("error_type") == "context_overflow"
        }
        for operation in reversed(store.list_model_operations(turn_id)):
            if operation.request_ref.get("purpose") != "agent_step":
                continue
            if operation.status == "completed" and operation.response_item_id in handled:
                item = store.read_item(operation.response_item_id)
                if item.payload.get("response_status", "completed") == "completed":
                    return None
            root = operation.request_ref.get("capacity_recovery_root_id")
            if isinstance(root, str):
                return root
            if operation.operation_id in rejected:
                return operation.operation_id
        return None

    return store.read_snapshot(read)


def _recovery_decision(store: RuntimeStore, turn_id: str) -> RecoveryDecision:
    from agent_runtime.harness.execution_facts import handled_response_ids

    records = store.list_turn_records(turn_id)
    handled = handled_response_ids(records)
    turn = store.read_turn(turn_id)
    models = store.list_model_operations(turn_id)
    tools = store.list_tool_operations(turn_id)
    pending = tuple(item for item in store.list_interactions(turn_id) if item.status == "pending")
    uncertain = tuple(op for op in tools if op.status in {"running", "unknown"})
    owned = turn.binding_manifest.get("execution_ownership") == OWNERSHIP_REVISION
    abort = (
        ("abort",)
        if (turn.status in {"paused", "interrupted"} or turn.status == "running" and owned) and not uncertain
        else ()
    )

    def plan(
        route: str,
        actions: tuple[str, ...],
        reason: str,
        *evidence: str,
        request_id: str | None = None,
        requires_input: bool = False,
        question: str | None = None,
        choices: tuple[str, ...] = (),
    ) -> RecoveryDecision:
        operation = next((op for op in models if op.operation_id in evidence), None)
        return RecoveryDecision(
            turn_id,
            turn.status,
            ContinuationKind(route),
            actions + abort,
            reason,
            (turn_id, *evidence),
            request_id,
            requires_input,
            question,
            choices,
            operation_id=None if operation is None else operation.operation_id,
            response_item_id=None if operation is None else operation.response_item_id,
            expected_generation=None if operation is None else operation.generation,
            expected_sequence=turn.applied_thread_sequence,
        )

    if turn.status not in {"running", "paused", "interrupted"}:
        resolved = [
            item
            for item in store.list_interactions(turn_id)
            if item.kind == "tool_approval" and item.status == "resolved"
        ]
        if turn.status == "completed" and resolved:
            return plan(
                "approval_resolved",
                (),
                "approval_already_resolved",
                resolved[-1].request_id,
                request_id=resolved[-1].request_id,
            )
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
        return plan(
            "reconcile",
            (),
            "tool_outcome_unknown",
            *(op.operation_id for op in uncertain),
            *((request_id,) if request_id is not None else ()),
            request_id=request_id,
        )
    dispatched = tuple(op for op in models if op.status == "dispatched")
    if dispatched or uncertain:
        try:
            assert_recovery_leases(
                owned=owned,
                leases=(
                    *(store.list_model_attempts(op.operation_id)[-1].lease_expires_at for op in dispatched),
                    *(op.lease_expires_at for op in uncertain),
                ),
            )
        except ConversationBusyError:
            return RecoveryDecision(
                turn_id, turn.status, ContinuationKind.UNAVAILABLE, (), "execution_lease_active", (turn_id,)
            )
        return plan(
            "recover_owner",
            ("continue",),
            "execution_owner_recovery",
            *(op.operation_id for op in dispatched),
            *(op.operation_id for op in uncertain),
        )
    revoked = store.turn_admission_revoked_by(turn_id)
    if turn.status == "running" and revoked is not None:
        return plan("cancel", (), "cancellation_pending", revoked)
    if revoked is not None and revoked != turn_id:
        # A child cannot resume authority revoked by its persisted ancestor.
        # Reconciliation above remains available for unknown tool effects.
        if any(op.status == "unknown" for op in models):
            return plan("unavailable", (), "ancestor_cancelled_outcome_unknown", revoked)
        return plan("cancel", (), "ancestor_cancelled", revoked)
    if pending and revoked is not None:
        return plan("unavailable", (), "cancellation_pending", *(item.request_id for item in pending))
    pause = next((r for r in reversed(records) if r.record_type == "turn_paused"), None)
    if (
        turn.status == "paused"
        and revoked is None
        and pause is not None
        and pause.payload.get("reason_code") == "attached_children"
    ):
        children = store.tx.rows("child_budget_allocations", where={"parent_turn_id": turn_id, "status": "active"})
        if children and store.user_input_revision(turn_id) == pause.payload.get("input_revision"):
            return plan("unavailable", (), "attached_children_pending", *(str(c["child_turn_id"]) for c in children))
        return plan("child_wait", ("continue",), "attached_children_ready")
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
            return plan(
                "approval",
                actions,
                "tool_approval_required",
                request.request_id,
                request_id=request.request_id,
                question=question,
            )
        if request.kind in {"clarification", "choice"}:
            return plan(
                "input",
                ("continue",),
                request.kind + "_required",
                request.request_id,
                request_id=request.request_id,
                requires_input=True,
                question=question,
                choices=tuple(request.request["options"]) if request.kind == "choice" else (),
            )
    if pending:
        return plan("unavailable", (), "pending_interaction", *(item.request_id for item in pending))
    if turn.status == "paused" and store.turn_cancellation_pending(turn_id):
        return plan("resume_cancelled", ("continue",), "cancellation_settled")
    waiting = tuple(op for op in models if op.status == "retry_wait")
    if len(waiting) == 1 and turn.status == "running":
        return plan(
            "wait_retry",
            ("continue",),
            "model_retry_scheduled",
            waiting[0].operation_id,
            request_id=waiting[0].operation_id,
        )
    unknown = tuple(op for op in models if op.status == "unknown")
    if len(unknown) == 1:
        return plan(
            "retry_model",
            ("retry",),
            "model_outcome_unknown",
            unknown[0].operation_id,
            request_id=unknown[0].operation_id,
        )
    for request in store.list_interactions(turn_id):
        if (
            request.kind == "tool_approval"
            and request.status == "resolved"
            and request.response.get("decision") == "approve"
            and request.operation_id is not None
        ):
            tool_operation = store.read_tool_operation(request.operation_id)
            if tool_operation.status == "ready" and tool_operation.approval_request_id == request.request_id:
                return plan(
                    "approved_ready",
                    ("continue", "allow_once"),
                    "approved_tool_not_started",
                    request.request_id,
                    tool_operation.operation_id,
                    request_id=request.request_id,
                )
    if turn.status == "running" and models:
        operation = models[-1]
        if operation.status == "completed" and operation.response_item_id is not None:
            response = store.read_item(operation.response_item_id)
            if response.kind == "model_response" and response.item_id not in handled:
                return plan(
                    "committed_response",
                    ("continue",),
                    "model_response_committed",
                    operation.operation_id,
                    response.item_id,
                )
        if operation.status == "prepared":
            return plan("prepared_model", ("continue",), "model_not_dispatched", operation.operation_id)
    if pending_context_overflow(store, turn_id):
        return plan("context_overflow", ("continue",), "context_overflow_rejected")
    if (
        turn.status == "running"
        and models
        and models[-1].request_ref.get("purpose") == "context_summary"
        and (
            (models[-1].status == "completed" and models[-1].response_item_id is not None)
            or (
                models[-1].status == "failed"
                and any(
                    key in (models[-1].request_ref.get("compaction_plan") or {})
                    for key in ("checkpoint_revision", "summary_history_start")
                )
            )
        )
    ):
        return plan(
            "resume_compaction",
            ("continue",),
            "summary_response_committed",
            models[-1].operation_id,
            models[-1].response_item_id,
        )
    if turn.status == "running":
        from agent_runtime.harness import compaction_jobs

        job = compaction_jobs.find_job(store, turn_id=turn_id)
        if job is not None:
            return plan("resume_compaction", ("continue",), "compaction_admitted", job.job_id)
    if turn.status == "running" and pending_input_supersession(store, turn_id):
        return plan("input_superseded", ("continue",), "request_superseded", models[-1].operation_id)
    if (
        turn.status == "running"
        and not models
        and not tools
        and (
            turn.binding_manifest.get("execution_ownership") == OWNERSHIP_REVISION
            or any(record.record_type == "execution_protocol_adopted" for record in store.list_turn_records(turn_id))
        )
    ):
        return plan("first_dispatch", ("continue",), "before_first_dispatch")
    if turn.status == "running" and models and models[-1].response_item_id in handled:
        return plan("next_step", ("continue",), "response_consumed", models[-1].operation_id)
    return plan("unavailable", (), "no_resume_point")

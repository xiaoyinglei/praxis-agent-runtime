"""Explicit completion ops over owner-serialized durable facts."""

from __future__ import annotations

from uuid import uuid4

from agent_runtime.harness import facts, turn_ops
from agent_runtime.harness.records import (
    ItemSnapshot,
)
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation


@owner_operation
def record_final_proposal(
    store: RuntimeStore,
    *,
    turn_id: str,
    answer: str,
    source_item_id: str | None = None,
) -> ItemSnapshot:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be non-empty")
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        thread_id = turn_ops._running_turn_thread_id(store, turn_id)
        payload = {"text": answer}
        if source_item_id is not None:
            source = store.read_item(source_item_id)
            if source.turn_id != turn_id or source.kind != "model_response" or source.status != "completed":
                raise RuntimeError("final proposal requires a completed model response from this Turn")
            payload["source_item_id"] = source_item_id
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="model",
            payload={"item_id": item_id, "kind": "final_proposal"},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="model",
            payload={"item_id": item_id, "payload": payload},
        )
    return store._latest_item(turn_id)


@owner_operation
def record_completion_decision(
    store: RuntimeStore,
    *,
    turn_id: str,
    proposal_item_id: str,
    action: str,
    reason: str,
    validation_status: str = "not_evaluated",
    validation_scope: str | None = None,
    reason_code: str | None = None,
) -> ItemSnapshot:
    if action not in {"accept", "continue", "pause", "fail"}:
        raise ValueError(f"unsupported completion action: {action}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("completion reason must be non-empty")
    if validation_status not in {"not_evaluated", "passed", "failed"}:
        raise ValueError(f"unsupported completion validation status: {validation_status}")
    if validation_scope is not None and (not isinstance(validation_scope, str) or not validation_scope.strip()):
        raise ValueError("completion validation scope must be non-empty or None")
    if reason_code is not None and (not isinstance(reason_code, str) or not reason_code.strip()):
        raise ValueError("completion reason code must be non-empty or None")
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        thread_id = turn_ops._running_turn_thread_id(store, turn_id)
        proposal = store.tx.first("items", where={"item_id": proposal_item_id, "turn_id": turn_id})
        if proposal is None or proposal["kind"] != "final_proposal" or proposal["status"] != "completed":
            raise RuntimeError("completion decision requires a completed final proposal")
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="verifier",
            payload={"item_id": item_id, "kind": "completion_decision"},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="verifier",
            payload={
                "item_id": item_id,
                "payload": {
                    "proposal_item_id": proposal_item_id,
                    "action": action,
                    "reason": reason,
                    "validation_status": validation_status,
                    "validation_scope": validation_scope,
                    **({"reason_code": reason_code} if reason_code is not None else {}),
                },
            },
        )
    return store._latest_item(turn_id)


@owner_operation
def record_completion_feedback(
    store: RuntimeStore,
    *,
    turn_id: str,
    reason: str,
    source_item_id: str | None = None,
    reason_code: str | None = None,
) -> ItemSnapshot:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("completion feedback reason must be non-empty")
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        thread_id = turn_ops._running_turn_thread_id(store, turn_id)
        payload = {"text": reason}
        if reason_code is not None:
            payload["reason_code"] = reason_code
        if source_item_id is not None:
            source = store.read_item(source_item_id)
            if (
                source.turn_id != turn_id
                or source.status != "completed"
                or source.kind not in {"model_response", "completion_decision"}
            ):
                raise RuntimeError("feedback requires a completed source from this Turn")
            payload["source_item_id"] = source_item_id
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="verifier",
            payload={"item_id": item_id, "kind": "completion_feedback"},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="verifier",
            payload={"item_id": item_id, "payload": payload},
        )
    return store._latest_item(turn_id)

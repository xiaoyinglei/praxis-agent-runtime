"""Explicit turn ops over owner-serialized durable facts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime.harness import artifact_ops, facts
from agent_runtime.harness.operation_values import (
    TurnCancellationPendingError,
    TurnChildrenPendingError,
    TurnInputChangedError,
    _json_object,
)
from agent_runtime.harness.protocol import (
    item_text,
    restore_model_response,
)
from agent_runtime.harness.records import (
    InteractionSnapshot,
    ItemSnapshot,
    ThreadSnapshot,
    TurnSnapshot,
)
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation
from agent_runtime.streaming.events import (
    derive_plan_public_item_id,
)


@owner_operation
def create_thread(
    store: RuntimeStore,
    *,
    workspace: Path,
    thread_id: str | None = None,
) -> ThreadSnapshot:
    store.tx.assert_writable()
    resolved_workspace = Path(workspace).resolve()
    if not resolved_workspace.is_dir():
        raise ValueError("workspace must be an existing directory")
    if thread_id is not None and (not isinstance(thread_id, str) or not thread_id.strip()):
        raise ValueError("thread_id must be non-empty when provided")
    thread_id = thread_id or f"thread_{uuid4().hex}"
    from agent_runtime.harness.workspace_stores import register_workspace_store

    register_workspace_store(resolved_workspace, store.tx.path)
    with store.tx.transaction():
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=None,
            record_type="thread_created",
            producer="runtime",
            payload={"workspace": str(resolved_workspace)},
        )
        from agent_runtime.harness.memory_identity import bind_memory_thread

        bind_memory_thread(store, thread_id, resolved_workspace)
    return store.read_thread(thread_id)


@owner_operation
def update_session_settings(
    store: RuntimeStore,
    thread_id: str,
    settings: Mapping[str, Any],
    *,
    expected_request_revision: int | None = None,
) -> None:
    frozen = _json_object(settings, field="settings")
    with store.tx.transaction():
        thread = store.read_thread(thread_id)
        if (
            expected_request_revision is not None
            and thread.settings.get("request_settings_revision", 0) != expected_request_revision
        ):
            from agent_runtime.model_config_io import ConfigVersionConflict

            raise ConfigVersionConflict("Request settings changed in another Session; reopen before changing settings")
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=None,
            record_type="session_settings_updated",
            producer="runtime",
            payload={"settings": frozen},
        )


@owner_operation
def fork_thread(store: RuntimeStore, *, from_turn_id: str) -> ThreadSnapshot:
    thread_id = f"thread_{uuid4().hex}"
    with store.tx.transaction():
        source = store.tx.turn_with_workspace(from_turn_id)
        if source is None:
            raise KeyError(f"unknown fork Turn: {from_turn_id}")
        if source["status"] not in {"completed", "failed", "cancelled"}:
            raise RuntimeError("only a terminal Turn can be a fork point")
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=None,
            record_type="thread_forked",
            producer="runtime",
            payload={
                "workspace": source["workspace"],
                "parent_thread_id": source["thread_id"],
                "fork_turn_id": from_turn_id,
            },
        )
        from agent_runtime.harness.memory_identity import bind_memory_thread

        bind_memory_thread(store, thread_id, Path(source["workspace"]))
    return store.read_thread(thread_id)


@owner_operation
def start_turn(
    store: RuntimeStore,
    *,
    thread_id: str,
    user_message: str,
    binding_manifest: Mapping[str, Any],
    input_files: tuple[Mapping[str, Any], ...] = (),
    turn_id: str | None = None,
    turn_producer: str = "runtime",
) -> TurnSnapshot:
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must be non-empty")
    frozen_manifest = _json_object(binding_manifest, field="binding_manifest")
    frozen_input_files = tuple(_json_object(value, field="input file") for value in input_files)
    if turn_id is not None and (not isinstance(turn_id, str) or not turn_id):
        raise ValueError("turn_id must be non-empty when provided")
    if not isinstance(turn_producer, str) or not turn_producer:
        raise ValueError("turn_producer must be non-empty")
    turn_id = turn_id or f"turn_{uuid4().hex}"
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        thread = store.tx.first("threads", where={"thread_id": thread_id})
        if thread is None:
            raise KeyError(f"unknown thread: {thread_id}")
        if thread["active_turn_id"] is not None:
            raise RuntimeError(f"thread already has an active turn: {thread_id}")
        turn_index = (store.tx.maximum("turns", "turn_index", where={"thread_id": thread_id}) + 1,)[0]
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="turn_started",
            producer=turn_producer,
            payload={
                "turn_id": turn_id,
                "predecessor_turn_id": thread["head_turn_id"],
                "turn_index": turn_index,
                "binding_manifest": frozen_manifest,
            },
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="user",
            payload={"item_id": item_id, "kind": "user_message"},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="user",
            payload={"item_id": item_id, "payload": {"text": user_message}},
        )
        for input_file in frozen_input_files:
            input_item_id = f"item_{uuid4().hex}"
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="user",
                payload={"item_id": input_item_id, "kind": "input_file"},
            )
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="user",
                payload={"item_id": input_item_id, "payload": input_file},
            )
    return store.read_turn(turn_id)


@owner_operation
def read_input_receipt(
    store: RuntimeStore,
    *,
    thread_id: str,
    turn_id: str,
    input_id: str,
    text: str,
    input_file_paths: tuple[str, ...],
) -> ItemSnapshot | None:
    """Match a frozen input intent before touching referenced files again."""
    if store.tx.first("items", where={"item_id": input_id}) is None:
        return None
    saved = store.read_item(input_id)
    request = {"text": text, "files": list(input_file_paths)}
    if (
        saved.thread_id != thread_id or saved.turn_id != turn_id or saved.kind != "user_message"
        or saved.producer != "user" or saved.payload.get("input_request") != request
    ):
        raise ValueError("input receipt conflicts with an earlier command")
    return saved


@owner_operation
def receive_turn_input(
    store: RuntimeStore,
    *,
    thread_id: str,
    turn_id: str,
    text: str,
    input_files: tuple[Mapping[str, Any], ...] = (),
    input_id: str | None = None,
    input_file_paths: tuple[str, ...] | None = None,
) -> ItemSnapshot:
    """Accept guidance once; a host receipt may be replayed after the Turn ends."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input must be non-empty")
    if input_id is not None and (not isinstance(input_id, str) or not input_id.strip()):
        raise ValueError("input receipt must be non-empty")
    item_id = input_id or f"item_{uuid4().hex}"
    request = None if input_file_paths is None else {"text": text, "files": list(input_file_paths)}
    digest = hashlib.sha256(json.dumps(
        {"text": text, "files": [dict(value) for value in input_files]},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    with store.tx.transaction():
        prior = store.tx.first("items", where={"item_id": item_id}) if input_id is not None else None
        if prior is not None:
            saved = store.read_item(item_id)
            if (
                saved.thread_id != thread_id or saved.turn_id != turn_id or saved.kind != "user_message"
                or saved.producer != "user" or saved.payload.get("input_receipt_digest") != digest
                or request is not None and saved.payload.get("input_request") != request
            ):
                raise ValueError("input receipt conflicts with an earlier command")
            return saved
        thread = store.read_thread(thread_id)
        turn = store.read_turn(turn_id) if thread.active_turn_id == turn_id else None
        if turn is None or turn.status not in {"running", "paused", "interrupted"}:
            raise RuntimeError("expected Turn is no longer active; input was not accepted")
        _assert_no_pending_cancellation(store, turn_id)
        revision = store.user_input_revision(turn_id) + 1
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="user",
            payload={"item_id": item_id, "kind": "user_message"},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="user",
            payload={"item_id": item_id, "payload": {
                "text": text, "input_revision": revision,
                **({"input_receipt_digest": digest} if input_id is not None else {}),
                **({"input_request": request} if input_id is not None and request is not None else {}),
            }},
        )
        for value in input_files:
            file_id = f"item_{uuid4().hex}"
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="user",
                payload={"item_id": file_id, "kind": "input_file"},
            )
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="user",
                payload={"item_id": file_id, "payload": _json_object(value, field="input file")},
            )
    return store.read_item(item_id)


@owner_operation
def complete_turn(
    store: RuntimeStore,
    *,
    turn_id: str,
    answer: str,
    producer: str = "runtime",
    expected_input_revision: int | None = None,
    source_item_id: str | None = None,
) -> TurnSnapshot:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be non-empty")
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        _assert_no_pending_cancellation(store, turn_id)
        if expected_input_revision is not None and store.user_input_revision(turn_id) != expected_input_revision:
            raise TurnInputChangedError("new user input superseded this final answer")
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "running":
            raise RuntimeError(f"turn is not running: {turn_id}")
        pending_children = tuple(
            str(child["child_turn_id"])
            for child in store.tx.rows(
                "child_budget_allocations", where={"parent_turn_id": turn_id, "status": "active"}, order_by="rowid"
            )
        )
        if pending_children:
            # The allocator checks the same parent's running state under the
            # same SQLite write lock. Completion and late allocation therefore
            # serialize: either the allocation blocks completion, or a terminal
            # parent rejects allocation. A preceding asynchronous drain alone
            # cannot establish this invariant.
            raise TurnChildrenPendingError(pending_children)
        from agent_runtime.harness.child_workspace_ops import unresolved

        pending_integration = unresolved(store, turn_id)
        if pending_integration:
            raise TurnChildrenPendingError(pending_integration)
        thread_id = turn["thread_id"]
        active = store.tx.first("threads", where={"thread_id": thread_id})
        if active is None or active["active_turn_id"] != turn_id:
            raise RuntimeError(f"turn does not own the thread active slot: {turn_id}")
        if source_item_id is not None:
            source = store.read_item(source_item_id)
            if (
                source.turn_id != turn_id
                or source.kind != "model_response"
                or source.status != "completed"
                or item_text(source) != answer
                or restore_model_response(source.payload).status != "completed"
            ):
                raise ValueError("accepted answer requires the matching completed response from this Turn")
            item_id = source_item_id
        else:
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer=producer,
                payload={"item_id": item_id, "kind": "agent_message"},
            )
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer=producer,
                payload={"item_id": item_id, "payload": {"text": answer}},
            )
        _append_terminal_plan_state(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            status="complete",
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="turn_completed",
            producer=producer,
            payload={"turn_id": turn_id, "answer_item_id": item_id},
        )
    return store.read_turn(turn_id)


@owner_operation
def fail_turn(
    store: RuntimeStore,
    *,
    turn_id: str,
    reason_code: str,
    message: str,
) -> TurnSnapshot:
    if not isinstance(reason_code, str) or not reason_code.strip():
        raise ValueError("Turn failure reason_code must be non-empty")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Turn failure message must be non-empty")
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "running":
            raise RuntimeError("only a running Turn can fail")
        active = store.tx.first("threads", where={"thread_id": turn["thread_id"]})
        if active is None or active["active_turn_id"] != turn_id:
            raise RuntimeError("failed Turn does not own the Thread active slot")
        _append_terminal_plan_state(
            store,
            thread_id=turn["thread_id"],
            turn_id=turn_id,
            status="blocked",
            message=message.strip(),
        )
        from agent_runtime.harness.compaction_jobs import finish_active_jobs

        finish_active_jobs(store, turn_id=turn_id, status="failed")
        facts.append_fact(
            store,
            thread_id=turn["thread_id"],
            turn_id=turn_id,
            record_type="turn_failed",
            producer="verifier",
            payload={
                "turn_id": turn_id,
                "reason_code": reason_code.strip(),
                "message": message.strip(),
            },
        )
    return store.read_turn(turn_id)


def _append_terminal_plan_state(
    store: RuntimeStore,
    *,
    thread_id: str,
    turn_id: str,
    status: str,
    message: str | None = None,
) -> None:
    latest = store.tx.first(
        "items",
        where={"turn_id": turn_id, "kind": "plan_state", "status": "completed"},
        order_by="sequence",
        newest=True,
        limit=1,
    )
    if latest is None:
        return
    payload = latest["payload"]
    raw_plan = payload.get("plan")
    if not isinstance(raw_plan, Mapping):
        raise RuntimeError("canonical plan_state payload is malformed")
    revision = int(raw_plan["revision"]) + 1
    steps = [dict(step) for step in raw_plan.get("steps", ()) if isinstance(step, Mapping)]
    if status == "complete" and any(step.get("status") not in {"completed", "skipped"} for step in steps):
        status = "needs_replan"
    plan = {
        **dict(raw_plan),
        "revision": revision,
        "status": status,
        "active_step_id": None,
        "steps": steps,
    }
    event_type = "completed" if status == "complete" else status
    event = {
        "event_type": event_type,
        "message": message
        or (
            "Plan completed as reported; task validation is separate."
            if status == "complete"
            else (
                "Turn accepted with unfinished plan steps; review remaining work."
                if status == "needs_replan"
                else "Plan blocked by the failed Turn; reported steps retained."
            )
        ),
        "tool_call_ids": [],
    }
    item_id = f"item_{uuid4().hex}"
    public_item_id = derive_plan_public_item_id(
        turn_id=turn_id,
        revision=revision,
    )
    facts.append_fact(
        store,
        thread_id=thread_id,
        turn_id=turn_id,
        record_type="item_started",
        producer="runtime",
        payload={
            "item_id": item_id,
            "kind": "plan_state",
            "public_item_id": public_item_id,
            "revision": revision,
        },
    )
    facts.append_fact(
        store,
        thread_id=thread_id,
        turn_id=turn_id,
        record_type="item_completed",
        producer="runtime",
        payload={
            "item_id": item_id,
            "public_item_id": public_item_id,
            "revision": revision,
            "payload": {"plan": plan, "event": event},
        },
    )


@owner_operation
def pause_turn(
    store: RuntimeStore,
    *,
    turn_id: str,
    reason: str,
    producer: str = "runtime",
    reason_code: str | None = None,
    input_revision: int | None = None,
) -> TurnSnapshot:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Turn pause reason must be non-empty")
    if not isinstance(producer, str) or not producer.strip():
        raise ValueError("Turn pause producer must be non-empty")
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "running":
            raise RuntimeError("only a running Turn can pause")
        active = store.tx.first("threads", where={"thread_id": turn["thread_id"]})
        if active is None or active["active_turn_id"] != turn_id:
            raise RuntimeError("paused Turn does not own the Thread active slot")
        facts.append_fact(
            store,
            thread_id=turn["thread_id"],
            turn_id=turn_id,
            record_type="turn_paused",
            producer=producer,
            payload={
                "turn_id": turn_id, "reason": reason,
                **({"reason_code": reason_code} if reason_code is not None else {}),
                **({"input_revision": input_revision} if input_revision is not None else {}),
            },
        )
    return store.read_turn(turn_id)


@owner_operation
def interrupt_orphaned_turn(
    store: RuntimeStore,
    *,
    turn_id: str,
    reason: str,
    maintenance_confirmed: bool,
) -> TurnSnapshot:
    """Recover only the safe gap before any model/tool operation exists."""

    if not maintenance_confirmed:
        raise RuntimeError(
            "orphan recovery requires explicit maintenance confirmation that the prior worker is stopped"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Turn interruption reason must be non-empty")
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "running":
            raise RuntimeError("orphan recovery requires a running Turn")
        active = store.tx.first("threads", where={"thread_id": turn["thread_id"]})
        if active is None or active["active_turn_id"] != turn_id:
            raise RuntimeError("orphaned Turn does not own the Thread active slot")
        operation_count = 0
        for table in (
            "model_operations",
            "tool_operations",
            "interactions",
            "approvals",
        ):
            row = {"count": store.tx.count(table, where={"turn_id": turn_id})}
            operation_count += int(row["count"])
        if operation_count:
            raise RuntimeError(
                "pre-dispatch orphan recovery refuses Turns with durable operations; "
                "use operation-specific reconciliation"
            )
        facts.append_fact(
            store,
            thread_id=str(turn["thread_id"]),
            turn_id=turn_id,
            record_type="turn_interrupted",
            producer="recovery",
            payload={
                "turn_id": turn_id,
                "reason": reason,
                "maintenance_confirmed": True,
                "recovery_scope": "pre_dispatch_orphan",
            },
        )
    return store.read_turn(turn_id)


@owner_operation
def cancel_turn(store: RuntimeStore, *, turn_id: str) -> TurnSnapshot:
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] not in {"paused", "interrupted"}:
            raise RuntimeError("only a safely paused or interrupted Turn can be cancelled")
        thread_id = str(turn["thread_id"])
        unsafe_operation = store.tx.first(
            "tool_operations",
            where={
                "turn_id": turn_id,
                "status": (
                    "running",
                    "unknown",
                ),
            },
            order_by="rowid",
            limit=1,
        )
        if unsafe_operation is not None:
            raise RuntimeError(
                f"cancel cannot discard a running or unknown tool operation: {unsafe_operation['operation_id']}"
            )
        pending = store.tx.rows("interactions", where={"turn_id": turn_id, "status": "pending"}, order_by="rowid")
        for interaction in pending:
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="user",
                payload={
                    "request_id": interaction["request_id"],
                    "version": int(interaction["version"]) + 1,
                    "response": {"action": "abort"},
                    "approval_status": "denied",
                },
            )
            if interaction["operation_id"] is not None:
                operation = store.tx.first("tool_operations", where={"operation_id": interaction["operation_id"]})
                if operation is None or operation["status"] != "awaiting_approval":
                    raise RuntimeError("cancel cannot discard a started or unknown tool operation")
                facts.append_fact(
                    store,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="tool_operation_approval_resolved",
                    producer="runtime",
                    payload={
                        "operation_id": interaction["operation_id"],
                        "request_id": interaction["request_id"],
                        "status": "cancelled",
                    },
                )
        from agent_runtime.harness.compaction_jobs import finish_active_jobs

        finish_active_jobs(store, turn_id=turn_id, status="cancelled")
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="turn_cancelled",
            producer="user",
            payload={"turn_id": turn_id},
        )
    return store.read_turn(turn_id)


@owner_operation
def request_turn_cancellation(
    store: RuntimeStore,
    *,
    turn_id: str,
    producer: str = "runtime",
    reason: str = "controlling stream closed",
) -> TurnSnapshot:
    """Append the durable request before live work receives cancellation."""

    if not isinstance(producer, str) or not producer.strip():
        raise ValueError("cancellation producer must be non-empty")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("cancellation reason must be non-empty")
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] not in {"running", "paused", "interrupted"}:
            return store.read_turn(turn_id)
        thread_id = str(turn["thread_id"])
        active = store.tx.first("threads", where={"thread_id": thread_id})
        if active is None or active["active_turn_id"] != turn_id:
            raise RuntimeError("cancellation request cannot target a foreign active slot")
        if not store.turn_cancellation_pending(turn_id):
            facts.append_fact(
                store,
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_cancellation_requested",
                producer=producer.strip(),
                payload={
                    "turn_id": turn_id,
                    "reason": reason.strip()[:2_000],
                },
            )
    return store.read_turn(turn_id)


def _assert_no_pending_cancellation(store: RuntimeStore, turn_id: str) -> None:
    """Check canonical lineage inside the caller's admission transaction.

    Binding fields and live child drivers are not ownership evidence. A paused
    ancestor still permits admitted children to run; cancellation or a terminal
    ancestor closes admission without discarding already dispatched outcomes.
    """
    revoked = store.turn_admission_revoked_by(turn_id)
    if revoked is not None:
        raise TurnCancellationPendingError(
            f"Turn cancellation was requested or authority ended at {revoked}; new work was not accepted"
        )


@owner_operation
def resume_child_wait(store: RuntimeStore, *, turn_id: str) -> TurnSnapshot:
    """Continue a child-wait pause only after settlement or new user guidance."""
    with store.tx.transaction():
        turn = store.read_turn(turn_id)
        _assert_no_pending_cancellation(store, turn_id)
        pause = next(
            (r for r in reversed(store.list_turn_records(turn_id)) if r.record_type == "turn_paused"), None
        )
        if turn.status != "paused" or pause is None or pause.payload.get("reason_code") != "attached_children":
            raise RuntimeError("Turn is not waiting for attached children")
        active = store.tx.rows("child_budget_allocations", where={"parent_turn_id": turn_id, "status": "active"})
        if active and store.user_input_revision(turn_id) == pause.payload.get("input_revision"):
            raise RuntimeError("Attached children still require attention")
        if store.read_thread(turn.thread_id).active_turn_id != turn_id:
            raise RuntimeError("child-wait Turn does not own the active slot")
        facts.append_fact(
            store, thread_id=turn.thread_id, turn_id=turn_id, record_type="turn_resumed", producer="runtime",
            payload={"turn_id": turn_id, "reason": "attached children settled or new guidance received"},
        )
    return store.read_turn(turn_id)


@owner_operation
def resume_cancelled_turn(store: RuntimeStore, *, turn_id: str) -> TurnSnapshot:
    """Explicitly continue a quiescent cancellation without replaying results."""
    with store.tx.transaction():
        turn = store.read_turn(turn_id)
        if turn.status != "paused" or not store.turn_cancellation_pending(turn_id):
            raise RuntimeError("Turn has no paused cancellation to resume")
        if store.read_thread(turn.thread_id).active_turn_id != turn_id:
            raise RuntimeError("cancelled Turn does not own the active slot")
        if (
            any(item.status == "pending" for item in store.list_interactions(turn_id))
            or any(op.status in {"running", "unknown"} for op in store.list_tool_operations(turn_id))
            or any(op.status in {"dispatched", "unknown"} for op in store.list_model_operations(turn_id))
        ):
            raise RuntimeError("pending or uncertain execution must be resolved before resume")
        facts.append_fact(
            store,
            thread_id=turn.thread_id,
            turn_id=turn_id,
            record_type="turn_resumed",
            producer="user",
            payload={"turn_id": turn_id, "reason": "explicit cancellation resume"},
        )
    return store.read_turn(turn_id)


@owner_operation
def request_clarification(
    store: RuntimeStore,
    *,
    turn_id: str,
    question: str,
) -> InteractionSnapshot:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("clarification question must be non-empty")
    request_id = f"interaction_{uuid4().hex}"
    with store.tx.transaction():
        thread_id = _running_turn_thread_id(store, turn_id)
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="interaction_requested",
            producer="verifier",
            payload={
                "request_id": request_id,
                "kind": "clarification",
                "operation_id": None,
                "request": {"question": question},
            },
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="turn_paused",
            producer="verifier",
            payload={"turn_id": turn_id, "request_id": request_id},
        )
    return store.read_interaction(request_id)


@owner_operation
def resolve_clarification(
    store: RuntimeStore,
    *,
    turn_id: str,
    request_id: str,
    response: str,
) -> InteractionSnapshot:
    if not isinstance(response, str) or not response.strip():
        raise ValueError("clarification response must be non-empty")
    artifact_ops._assert_artifact_integrity(store, turn_id)
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "paused":
            raise RuntimeError("clarification response requires a paused Turn")
        _assert_no_pending_cancellation(store, turn_id)
        interaction = store.tx.first("interactions", where={"request_id": request_id, "turn_id": turn_id})
        if interaction is None:
            raise KeyError(f"unknown clarification request: {request_id}")
        if interaction["kind"] != "clarification" or interaction["status"] != "pending":
            raise RuntimeError("interaction is not a pending clarification")
        thread_id = str(turn["thread_id"])
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="interaction_resolved",
            producer="user",
            payload={
                "request_id": request_id,
                "version": int(interaction["version"]) + 1,
                "response": {"text": response},
            },
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="user",
            payload={"item_id": item_id, "kind": "user_message"},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="user",
            payload={"item_id": item_id, "payload": {"text": response}},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="turn_resumed",
            producer="runtime",
            payload={"turn_id": turn_id, "request_id": request_id},
        )
    return store.read_interaction(request_id)


@owner_operation
def request_choice(
    store: RuntimeStore,
    *,
    turn_id: str,
    question: str,
    options: tuple[str, ...],
) -> InteractionSnapshot:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("choice question must be non-empty")
    if (
        len(options) < 2
        or any(not isinstance(option, str) or not option.strip() for option in options)
        or len(set(options)) != len(options)
    ):
        raise ValueError("choice options must contain at least two unique non-empty values")
    request_id = f"interaction_{uuid4().hex}"
    with store.tx.transaction():
        thread_id = _running_turn_thread_id(store, turn_id)
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="interaction_requested",
            producer="runtime",
            payload={
                "request_id": request_id,
                "kind": "choice",
                "operation_id": None,
                "request": {"question": question, "options": list(options)},
            },
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="turn_paused",
            producer="runtime",
            payload={"turn_id": turn_id, "request_id": request_id},
        )
    return store.read_interaction(request_id)


@owner_operation
def resolve_choice(
    store: RuntimeStore,
    *,
    turn_id: str,
    request_id: str,
    selection: str,
) -> InteractionSnapshot:
    if not isinstance(selection, str) or not selection.strip():
        raise ValueError("choice selection must be non-empty")
    artifact_ops._assert_artifact_integrity(store, turn_id)
    item_id = f"item_{uuid4().hex}"
    with store.tx.transaction():
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "paused":
            raise RuntimeError("choice response requires a paused Turn")
        _assert_no_pending_cancellation(store, turn_id)
        interaction = store.tx.first("interactions", where={"request_id": request_id, "turn_id": turn_id})
        if interaction is None:
            raise KeyError(f"unknown choice request: {request_id}")
        if interaction["kind"] != "choice" or interaction["status"] != "pending":
            raise RuntimeError("interaction is not a pending choice")
        request = interaction["request"]
        options = request.get("options")
        if not isinstance(options, list) or selection not in options:
            raise ValueError("choice selection is not one of the frozen options")
        thread_id = str(turn["thread_id"])
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="interaction_resolved",
            producer="user",
            payload={
                "request_id": request_id,
                "version": int(interaction["version"]) + 1,
                "response": {"selection": selection},
            },
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="user",
            payload={"item_id": item_id, "kind": "user_message"},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="user",
            payload={"item_id": item_id, "payload": {"text": selection}},
        )
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="turn_resumed",
            producer="runtime",
            payload={"turn_id": turn_id, "request_id": request_id},
        )
    return store.read_interaction(request_id)


def _running_turn_thread_id(store: RuntimeStore, turn_id: str) -> str:
    turn = store.tx.first("turns", where={"turn_id": turn_id})
    if turn is None:
        raise KeyError(f"unknown turn: {turn_id}")
    if turn["status"] != "running":
        raise RuntimeError(f"turn is not running: {turn_id}")
    return str(turn["thread_id"])

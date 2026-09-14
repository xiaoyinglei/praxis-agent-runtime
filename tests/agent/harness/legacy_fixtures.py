"""Historical rollout fixtures retained after retiring the importer."""

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from agent_runtime.harness.rollout import ItemSnapshot, RolloutStore, _json_object


def record_migrated_context_item(
    store: RolloutStore,
    *,
    turn_id: str,
    kind: str,
    payload: Mapping[str, Any],
) -> ItemSnapshot:
    """Seed historical records for replay tests; not a runtime import API."""

    allowed_kinds = {
        "user_message",
        "agent_message",
        "model_response",
        "tool_result",
        "context_message",
        "input_file",
    }
    if kind not in allowed_kinds:
        raise ValueError(f"unsupported migrated Item kind: {kind}")
    frozen_payload = _json_object(payload, field="migrated Item payload")
    if kind == "tool_result":
        store._validate_artifact_references(frozen_payload)
    item_id = f"item_{uuid4().hex}"
    public_projection = (
        {
            "public_item_id": item_id,
            "public_item_kind": "legacy_message",
        }
        if kind == "model_response"
        else {}
    )
    with store._transaction():
        turn = store._connection.execute(
            "SELECT thread_id, status FROM turns WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "running":
            raise RuntimeError("migrated Items require a running Turn")
        store._append_and_reduce(
            thread_id=str(turn["thread_id"]),
            turn_id=turn_id,
            record_type="item_started",
            producer="migration",
            payload={"item_id": item_id, "kind": kind, **public_projection},
        )
        store._append_and_reduce(
            thread_id=str(turn["thread_id"]),
            turn_id=turn_id,
            record_type="item_completed",
            producer="migration",
            payload={
                "item_id": item_id,
                "payload": frozen_payload,
                **public_projection,
            },
        )
    return store.list_items(turn_id)[-1]

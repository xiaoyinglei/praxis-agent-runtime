"""Explicit facts over owner-serialized durable facts."""

from __future__ import annotations

from typing import Any

from agent_runtime.harness.runtime_store import RuntimeStore


def append_fact(store: RuntimeStore, **fact: Any) -> None:
    # Business identity enrichment belongs with commands, before storage.
    if fact["record_type"] == "item_completed":
        start = store.read_event_reference("item_started", str(fact["payload"]["item_id"]))
        if start is None:
            raise RuntimeError("completed internal Item has no durable start")
        fact["payload"] = {**start.payload, **fact["payload"]}
    elif fact["record_type"] in {"interaction_resolved", "approval_invalidated"}:
        interaction = store.tx.first("interactions", where={"request_id": fact["payload"]["request_id"]})
        if interaction is None:
            raise RuntimeError("interaction resolution has no committed request")
        fact["payload"] = {
            "kind": interaction["kind"],
            "operation_id": interaction["operation_id"],
            **fact["payload"],
        }
    store.tx.append_record(**fact)

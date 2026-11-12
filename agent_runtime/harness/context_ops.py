"""Explicit context ops over owner-serialized durable facts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from agent_runtime.harness import facts
from agent_runtime.harness.operation_values import _canonical_json, _context_durable_state, _json_object
from agent_runtime.harness.protocol import (
    ContextSourceChangedError,
    restore_model_response,
)
from agent_runtime.harness.records import (
    ItemSnapshot,
)
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation


@owner_operation
def record_context_compaction(
    store: RuntimeStore,
    *,
    turn_id: str,
    covered_item_ids: tuple[str, ...],
    summary: str,
    preserved_facts: Mapping[str, Any],
    artifact_refs: tuple[Mapping[str, Any], ...] = (),
    context_version: int,
    preserved_item_ids: tuple[str, ...] = (),
    expected_source_revision: str | None = None,
    durable_state: Mapping[str, Any] | None = None,
    algorithm_revision: str | None = None,
    tool_result_overrides: Mapping[str, str] | None = None,
    capacity_diagnostics: Mapping[str, Any] | None = None,
    replacement_history: tuple[Mapping[str, Any], ...] | None = None,
) -> ItemSnapshot:
    """Append an auditable replacement for one committed context prefix."""

    if not covered_item_ids or any(not isinstance(item_id, str) or not item_id for item_id in covered_item_ids):
        raise ValueError("compaction must cover non-empty Item IDs")
    if len(set(covered_item_ids)) != len(covered_item_ids):
        raise ValueError("compaction covered Item IDs must be unique")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("compaction summary must be non-empty")
    if isinstance(context_version, bool) or not isinstance(context_version, int) or context_version < 1:
        raise ValueError("context_version must be a positive integer")

    with store.tx.transaction():
        if expected_source_revision is not None and store.context_source_revision(turn_id) != expected_source_revision:
            raise ContextSourceChangedError("Context source changed before compaction commit.")
        turn = store.tx.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "running":
            raise RuntimeError("context compaction requires a running Turn")
        visible_items = tuple(
            item
            for item in (
                store.list_model_context_items(turn_id)
                if tool_result_overrides is not None
                else store.list_context_items(turn_id)
            )
            if item.status == "completed"
        )
        visible_ids = tuple(item.item_id for item in visible_items)
        if visible_ids[: len(covered_item_ids)] != covered_item_ids:
            raise ValueError("compaction must cover a contiguous context prefix in exact order")
        if len(set(preserved_item_ids)) != len(preserved_item_ids) or any(
            item_id not in covered_item_ids for item_id in preserved_item_ids
        ):
            raise ValueError("preserved Items must belong to the covered prefix")
        if algorithm_revision in {"semantic-compaction-v5", "semantic-compaction-v6"}:
            pending: set[tuple[str, str]] = set()
            replay_results: set[str] = set()
            for item in visible_items:
                if item.item_id not in preserved_item_ids:
                    continue
                calls = restore_model_response(item.payload).tool_calls if item.kind == "model_response" else ()
                if calls:
                    if pending:
                        raise ValueError("preserved tool exchanges must be complete")
                    pending.update((item.turn_id, call.id) for call in calls)
                elif item.kind == "tool_result":
                    key = (item.turn_id, item.payload.get("tool_call_id"))
                    if key not in pending:
                        raise ValueError("preserved tool result must have its original call")
                    pending.remove(key)
                    replay_results.add(item.item_id)
                elif item.kind != "user_message" or pending:
                    raise ValueError("only user Items and complete tool exchanges can be preserved")
            if pending or (
                algorithm_revision == "semantic-compaction-v5" and replay_results != set(tool_result_overrides or {})
            ):
                raise ValueError("preserved tool exchanges require all paired archived results")
            for source_turn in {item.turn_id for item in visible_items if item.item_id in replay_results}:
                if any(
                    op.result_item_id in replay_results
                    and (op.requires_reconciliation or op.status not in {"succeeded", "failed", "denied", "cancelled"})
                    for op in store.list_tool_operations(source_turn)
                ):
                    raise ValueError("unsettled tool results cannot be archived in semantic replay")
        elif algorithm_revision != "semantic-compaction-v7" and any(
            item.kind != "user_message" for item in visible_items if item.item_id in preserved_item_ids
        ):
            raise ValueError("only user Items can be preserved inside a covered prefix")

        if algorithm_revision == "semantic-compaction-v7":
            from agent_runtime.core.messages import restore_model_message
            from agent_runtime.harness.context import _validate_tool_groups

            if not replacement_history or sum(entry.get("item_id") is None for entry in replacement_history) != 1:
                raise ValueError("replacement history must contain exactly one summary")
            original_ids = {entry["item_id"] for entry in replacement_history if entry["item_id"] is not None}
            if not original_ids.issubset(visible_ids):
                raise ValueError("replacement history references a nonvisible Item")
            if original_ids.intersection(covered_item_ids) != set(preserved_item_ids):
                raise ValueError("replacement history must replay exactly the preserved Items inside its prefix")
            restored = tuple(restore_model_message(entry["message"]) for entry in replacement_history)
            summary_message = next(
                message
                for entry, message in zip(replacement_history, restored, strict=True)
                if entry["item_id"] is None
            )
            if summary_message.role != "context" or summary_message.context_category != "continuation_summary":
                raise ValueError("replacement history must identify its continuation summary")
            _validate_tool_groups(restored)
        elif replacement_history is not None:
            raise ValueError("replacement history requires a checkpoint record")

        if tool_result_overrides is not None:
            covered_by_id = {item.item_id: item for item in visible_items[: len(covered_item_ids)]}
            if not tool_result_overrides or any(
                target not in covered_by_id
                or covered_by_id[target].kind != "tool_result"
                or not isinstance(content, str)
                or not content
                for target, content in tool_result_overrides.items()
            ):
                raise ValueError("tool result overrides require completed covered tool Items and nonempty text")

        frozen_facts = _json_object(
            preserved_facts,
            field="compaction preserved_facts",
        )
        required_fact_categories = {
            "architecture_and_safety_constraints",
            "file_changes",
            "verification_results",
            "unresolved_work",
            "uncertain_side_effects",
        }
        missing_categories = sorted(required_fact_categories - frozen_facts.keys())
        if missing_categories:
            raise ValueError(f"compaction preserved_facts lacks critical fact categories: {missing_categories}")
        frozen_artifacts = tuple(_json_object(value, field="compaction artifact ref") for value in artifact_refs)
        covered_items = visible_items[: len(covered_item_ids)]
        covered_refs = [
            {
                "item_id": item.item_id,
                "thread_id": item.thread_id,
                "turn_id": item.turn_id,
                "item_sequence": item.sequence,
                "applied_thread_sequence": item.applied_thread_sequence,
                "payload_sha256": hashlib.sha256(_canonical_json(dict(item.payload)).encode()).hexdigest(),
            }
            for item in covered_items
        ]
        grouped_sequences: dict[str, list[int]] = {}
        for item in covered_items:
            grouped_sequences.setdefault(item.thread_id, []).append(item.applied_thread_sequence)
        covered_ranges = [
            {
                "thread_id": thread_id,
                "start_thread_sequence": min(sequences),
                "end_thread_sequence": max(sequences),
            }
            for thread_id, sequences in grouped_sequences.items()
        ]
        frozen_state = (
            _context_durable_state(store, turn_id)
            if durable_state is None
            else _json_object(durable_state, field="compaction durable_state")
        )
        item_id = f"item_{uuid4().hex}"
        payload = {
            "context_version": context_version,
            "covered_item_ids": list(covered_item_ids),
            "covered_item_refs": covered_refs,
            "covered_sequence_ranges": covered_ranges,
            "summary": summary,
            "preserved_facts": frozen_facts,
            "artifact_refs": list(frozen_artifacts),
            "durable_state": frozen_state,
        }
        if tool_result_overrides is not None:
            payload["tool_result_overrides"] = dict(tool_result_overrides)
        if replacement_history is not None:
            payload["replacement_history"] = list(replacement_history)
        if capacity_diagnostics is not None:
            payload["capacity_diagnostics"] = _json_object(capacity_diagnostics, field="capacity diagnostics")
        if expected_source_revision is not None:
            payload.update(
                source_revision=expected_source_revision,
                preserved_item_ids=list(preserved_item_ids),
                algorithm_revision=algorithm_revision,
            )
        facts.append_fact(
            store,
            thread_id=str(turn["thread_id"]),
            turn_id=turn_id,
            record_type="item_started",
            producer="runtime",
            payload={"item_id": item_id, "kind": "context_compaction"},
        )
        facts.append_fact(
            store,
            thread_id=str(turn["thread_id"]),
            turn_id=turn_id,
            record_type="item_completed",
            producer="runtime",
            payload={"item_id": item_id, "payload": payload},
        )
    return store._latest_item(turn_id)

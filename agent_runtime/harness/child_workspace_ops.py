"""Canonical child environment observations and parent completion admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime.approval import MODE_ORDER
from agent_runtime.harness import facts, turn_ops
from agent_runtime.harness.runtime_store import RuntimeStore
from agent_runtime.permission_policy import ResolvedPermissions, relocate_workspace_permissions
from agent_runtime.tools.permissions import ToolExecutionContext, restore_tool_policy
from agent_runtime.workspace_snapshot import capture_snapshot, working_entries

KIND = "child_workspace"


def record(store: RuntimeStore, parent: str, payload: Mapping[str, Any]) -> None:
    with store.tx.transaction():
        thread = turn_ops._running_turn_thread_id(store, parent)
        identity = f"item_{uuid4().hex}"
        facts.append_fact(
            store,
            thread_id=thread,
            turn_id=parent,
            record_type="item_started",
            producer="runtime",
            payload={"item_id": identity, "kind": KIND},
        )
        facts.append_fact(
            store,
            thread_id=thread,
            turn_id=parent,
            record_type="item_completed",
            producer="runtime",
            payload={"item_id": identity, "payload": dict(payload)},
        )


def state(store: RuntimeStore, parent: str, child: str) -> dict[str, Any] | None:
    for item in reversed(store.list_items(parent)):
        if item.kind == KIND and item.status == "completed" and item.payload.get("child_turn_id") == child:
            return dict(item.payload)
    return None


def unresolved(store: RuntimeStore, parent: str, *, settled_only: bool = False) -> tuple[str, ...]:
    latest = {}
    for item in store.list_items(parent):
        if item.kind == KIND and item.status == "completed":
            latest[str(item.payload["child_turn_id"])] = item.payload
    result = []
    current_tree = None
    for child, environment in latest.items():
        if settled_only:
            try:
                if store.read_child_budget_allocation(child)["status"] == "active":
                    continue
            except KeyError:
                pass
        if environment["status"] in {"discarded", "read_only"}:
            continue
        if environment.get("mode") == "read-only":
            try:
                if (
                    store.read_turn(child).status in {"completed", "failed", "cancelled"}
                    and store.read_child_budget_allocation(child)["status"] == "settled"
                ):
                    continue
            except KeyError:
                pass
        if environment["status"] == "integrated" and environment.get("verification") == "recorded":
            if current_tree is None:
                workspace = Path(store.read_thread(store.read_turn(parent).thread_id).workspace)
                current_tree = capture_snapshot(
                    workspace,
                    object_root=Path(environment["repository"]),
                    excluded=tuple(Path(p) for p in environment.get("excluded_paths", ())),
                )["tree"]
            current_entries = working_entries(
                workspace,
                object_root=Path(environment["repository"]),
                paths=tuple(environment.get("changed_paths", ())),
                include_directories=True,
            )
            if current_tree == environment.get("verified_tree") and current_entries == environment.get(
                "verified_entries"
            ):
                continue
        result.append(child)
    return tuple(result)


def ancestor_context(store: RuntimeStore, turn_id: str, context: ToolExecutionContext) -> ToolExecutionContext:
    """Intersect current persistent ancestor controls at the execution connection."""
    turn = store.read_turn(turn_id)
    destination = Path(store.read_thread(turn.thread_id).workspace)
    seen = {turn_id}
    while True:
        try:
            edge = store.read_child_budget_allocation(turn.turn_id)
        except KeyError:
            break
        parent_id = str(edge["parent_turn_id"])
        if parent_id in seen:
            raise RuntimeError("cyclic canonical child lineage")
        seen.add(parent_id)
        turn = store.read_turn(parent_id)
        thread = store.read_thread(turn.thread_id)
        policy = dict(
            thread.settings.get("tool_execution_policy", turn.binding_manifest.get("tool_execution_policy", {}))
        )
        if "max_parallel_calls" in policy:
            policy["max_parallel_calls"] = min(policy["max_parallel_calls"], context.max_parallel_calls)
        if policy.get("permissions") is not None:
            saved = ResolvedPermissions.model_validate(policy["permissions"])
            if saved.workspace != str(destination):
                saved = relocate_workspace_permissions(saved, Path(thread.workspace), destination)
            policy["permissions"] = saved.model_dump(mode="json")
        context = restore_tool_policy(context, policy)
        mode = policy.get("approval_mode", "ask")
        context = replace(context, approval_mode=min((context.approval_mode, mode), key=MODE_ORDER.__getitem__))
    return context

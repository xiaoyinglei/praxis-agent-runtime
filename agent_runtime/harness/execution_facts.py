"""Durable handling receipts; execution position is derived, never assigned."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_runtime.harness import tool_ops
from agent_runtime.harness.facts import append_fact
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation
from agent_runtime.tools.tool import ToolCall, _thaw_json

EXECUTION_PROTOCOL = "durable-turn-v1"


@owner_operation
def adopt_execution_protocol(store: RuntimeStore, turn_id: str) -> None:
    with store.tx.transaction():
        if any(r.record_type == "execution_protocol_adopted" for r in store.list_turn_records(turn_id)):
            return
        turn = store.read_turn(turn_id)
        append_fact(
            store,
            thread_id=turn.thread_id,
            turn_id=turn_id,
            producer="runtime",
            record_type="execution_protocol_adopted",
            payload={"revision": EXECUTION_PROTOCOL},
        )


@owner_operation
def response_handled(
    store: RuntimeStore, *, turn_id: str, operation_id: str, response_item_id: str, disposition: str
) -> None:
    with store.tx.transaction():
        prior = [
            r
            for r in store.list_turn_records(turn_id)
            if r.record_type == "response_handling_decided" and r.payload["response_item_id"] == response_item_id
        ]
        if prior:
            return
        operation = next(op for op in store.list_model_operations(turn_id) if op.operation_id == operation_id)
        if operation.status != "completed" or operation.response_item_id != response_item_id:
            raise RuntimeError("response handling requires the canonical completed operation")
        append_fact(
            store,
            thread_id=operation.thread_id,
            turn_id=turn_id,
            producer="runtime",
            record_type="response_handling_decided",
            payload={
                "operation_id": operation_id,
                "response_item_id": response_item_id,
                "step_index": operation.request_ref.get("step_snapshot", {}).get("step", 1),
                "disposition": disposition,
            },
        )


@owner_operation
def admit_tool_batch(store: RuntimeStore, *, turn_id: str, response_item_id: str, calls: Sequence[ToolCall]) -> None:
    """Persist all call intents and order before the first tool can start."""
    with store.tx.transaction():
        prior = next(
            (
                r
                for r in store.list_turn_records(turn_id)
                if r.record_type == "tool_batch_admitted" and r.payload["response_item_id"] == response_item_id
            ),
            None,
        )
        identity = tuple(call.tool_call_id for call in calls)
        if prior is not None:
            original = tuple(prior.payload["tool_call_ids"])
            if any(call_id not in original for call_id in identity):
                raise RuntimeError("recovered tool batch conflicts with its committed response")
            return
        response = store.read_item(response_item_id)
        if response.turn_id != turn_id or response.kind != "model_response" or response.status != "completed":
            raise RuntimeError("tool batch requires a committed model response")
        for call in calls:
            arguments = _thaw_json(call.arguments)
            if not isinstance(arguments, Mapping):
                raise TypeError("tool arguments must be an object")
            tool_ops.record_tool_call(
                store,
                turn_id=turn_id,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                arguments=dict(arguments),
                argument_error=call.argument_error,
                origin={
                    "request_id": call.origin.request_id,
                    "toolset_revision": call.origin.toolset_revision,
                    "exposed_tool_names": list(call.origin.exposed_tool_names),
                },
            )
        append_fact(
            store,
            thread_id=response.thread_id,
            turn_id=turn_id,
            producer="runtime",
            record_type="tool_batch_admitted",
            payload={"response_item_id": response_item_id, "tool_call_ids": list(identity)},
        )


def handled_response_ids(records: Sequence[Any]) -> frozenset[str]:
    return frozenset(
        record.payload["response_item_id"] for record in records if record.record_type == "response_handling_decided"
    )


def canonical_tool_results(store: RuntimeStore, turn_id: str) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Read the accepted outcome; keep superseded unknowns only in the audit ledger."""
    operations = {op.tool_call_id: op for op in store.list_tool_operations(turn_id)}
    results: list[tuple[str, Mapping[str, Any]]] = []
    for item in store.list_items(turn_id):
        if item.status != "completed":
            continue
        payload = item.payload
        if item.kind == "tool_reconciliation" and item.producer == "reconciler":
            reconciled = item.payload.get("result")
            if not isinstance(reconciled, Mapping):
                continue
            payload = reconciled
            call_id = payload.get("tool_call_id")
            if not isinstance(call_id, str):
                continue
            operation = operations.get(call_id)
            if operation is None or operation.result_item_id != item.item_id:
                continue
        elif item.kind == "tool_result":
            call_id = payload.get("tool_call_id")
            if not isinstance(call_id, str):
                continue
            operation = operations.get(call_id)
            if (
                operation is not None
                and operation.result_item_id is not None
                and operation.result_item_id != item.item_id
            ):
                continue
        else:
            continue
        results.append((item.item_id, payload))
    return tuple(results)


def repeated_tool_failure(store: RuntimeStore, turn_id: str) -> bool:
    """Use committed results so a process restart cannot reset the failure streak."""
    items = store.list_items(turn_id)
    calls = {
        item.payload.get("tool_call_id"): item.payload
        for item in items
        if item.kind == "tool_call" and item.status == "completed"
    }
    canonical = canonical_tool_results(store, turn_id)
    results = [result for _, result in canonical]
    if len(results) < 3:
        return False
    operations = {op.result_item_id: op for op in store.list_tool_operations(turn_id)}
    result_items = {result.get("tool_call_id"): item_id for item_id, result in canonical}
    target = None
    repeats = 0
    for result in reversed(results):
        call = calls.get(result.get("tool_call_id"))
        if call is None:
            break
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if metadata.get("workspace_tree_changed") is True:
            break
        if result.get("is_error") is not True:
            operation = operations.get(result_items.get(result.get("tool_call_id")))
            # Inspections and proven no-op writes do not reset a failure streak.
            # Unknown/external effects remain a conservative progress boundary.
            if operation is None:
                break
            effects = set(operation.effects)
            readonly = effects <= {"read_workspace"}
            noop = metadata.get("workspace_tree_changed") is False and effects <= {
                "read_workspace",
                "write_workspace",
                "execute_process",
                "destructive",
            }
            if not (readonly or noop):
                break
            if target and (call.get("tool_name"), call.get("arguments")) == (target[0], target[3]):
                break  # The same call succeeded: actual recovery.
            if target and isinstance(target[3], tuple) and call.get("tool_name") == target[0]:
                resources = tuple(
                    sorted(
                        resource["identity"]
                        for resource in operation.resources
                        if resource.get("kind") == "filesystem" and resource.get("access") == "read"
                    )
                )
                if set(target[3]) <= set(resources):
                    break
            continue
        # Argument validation permits a corrected call, not identical retries.
        deterministic = result.get("retryable") is False or result.get("error_code") == "invalid_arguments"
        if not deterministic:
            break
        signature = (
            result.get("tool_name"),
            result.get("error_code"),
            None,
            call.get("arguments"),
        )
        operation = operations.get(result_items.get(result.get("tool_call_id")))
        constraint = metadata.get("permission_constraint")
        if result.get("error_code") == "filesystem_denied" and isinstance(constraint, Mapping):
            # The same policy/resource/action remains the same obstacle
            # when a model changes tool names or rewrites its arguments.
            denied_resources = tuple(sorted({(r["path"], r["access"]) for r in constraint.get("resources", ())}))
            if denied_resources:
                signature = (
                    "filesystem_permission",
                    "filesystem_denied",
                    None,
                    (constraint.get("permissions_revision"), denied_resources),
                )
        if (
            result.get("error_code") == "file_not_found"
            and operation is not None
            and set(operation.effects) <= {"read_workspace"}
        ):
            resources = tuple(
                sorted(
                    resource["identity"]
                    for resource in operation.resources
                    if resource.get("kind") == "filesystem" and resource.get("access") == "read"
                )
            )
            if resources:
                # The committed outcome already attests a missing resource.
                # Replaying it must not perform unrecorded filesystem probes.
                signature = (result.get("tool_name"), "file_not_found", None, resources)
        if target is None:
            target = signature
        if signature == target:
            repeats += 1
            if repeats == 3:
                return True
    return False

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

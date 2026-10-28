"""Runtime-owned identities and durable lifecycle for generated model outputs.

Output completion records generated content, not authorization to execute it.
Only response acceptance may expose tool calls to the execution orchestrator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from agent_runtime.harness.records import ItemSnapshot
from agent_runtime.harness.state import RuntimeState, _state_operation
from agent_runtime.modeling.events import ModelOutput, output_from_payload, output_to_payload
from agent_runtime.streaming.events import derive_model_public_item_id


def _active_attempt(state: RuntimeState, operation_id: str, attempt_id: str, generation: int) -> Mapping[str, Any]:
    operation = state.persistence.first("model_operations", where={"operation_id": operation_id})
    if operation is None:
        raise KeyError(f"unknown model operation: {operation_id}")
    if (
        operation["status"] != "dispatched"
        or operation["active_attempt_id"] != attempt_id
        or operation["generation"] != generation
    ):
        raise RuntimeError("model output requires the active dispatched attempt")
    return operation


def _starts(state: RuntimeState, attempt_id: str) -> tuple[Mapping[str, Any], ...]:
    attempt = state.persistence.first("model_attempts", where={"attempt_id": attempt_id})
    if attempt is None:
        raise KeyError(f"unknown model attempt: {attempt_id}")
    operation = state.persistence.first("model_operations", where={"operation_id": attempt["operation_id"]})
    if operation is None:
        raise RuntimeError("model output attempt has no operation")
    return state.persistence.rows(
        "rollout_records",
        where={"record_type": "item_started", "turn_id": operation["turn_id"]},
        payload_fields={"attempt_id": attempt_id, "kind": "model_output"},
        order_by="record_id",
    )


def _start(state: RuntimeState, attempt_id: str, output_id: str) -> Mapping[str, Any] | None:
    rows = [row["payload"] for row in _starts(state, attempt_id) if row["payload"]["output_id"] == output_id]
    if len(rows) > 1:
        raise RuntimeError("model output has duplicate durable starts")
    return rows[0] if rows else None


def _channel(output: ModelOutput) -> str:
    return {
        "text": "agent_message",
        "refusal": "agent_message",
        "tool_call": "agent_message",
        "reasoning": "reasoning",
        "reasoning_summary": "reasoning",
        "plan": "plan",
    }[output.kind]


@_state_operation
def record_model_response_started(
    state: RuntimeState,
    *,
    operation_id: str,
    attempt_id: str,
    generation: int,
    provider_response_id: str | None,
) -> None:
    with state.persistence.transaction():
        operation = _active_attempt(state, operation_id, attempt_id, generation)
        state._append_and_reduce(
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="model_response_started",
            producer="runtime",
            payload={
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "generation": generation,
                "provider_response_id": provider_response_id,
            },
        )


@_state_operation
def start_model_output(
    state: RuntimeState,
    *,
    operation_id: str,
    attempt_id: str,
    generation: int,
    output: ModelOutput,
) -> ItemSnapshot:
    with state.persistence.transaction():
        operation = _active_attempt(state, operation_id, attempt_id, generation)
        if operation["request_ref"].get("purpose") == "context_summary":
            raise RuntimeError("context summary must not publish model output Items")
        if _start(state, attempt_id, output.output_id) is not None:
            raise RuntimeError("model output already started")
        item_id = f"item_{uuid4().hex}"
        state._append_and_reduce(
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="item_started",
            producer="model",
            payload={
                "item_id": item_id,
                "kind": "model_output",
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "generation": generation,
                "output_id": output.output_id,
                "output_kind": output.kind,
                "public_item_id": derive_model_public_item_id(
                    turn_id=operation["turn_id"],
                    model_attempt_id=attempt_id,
                    channel=_channel(output),
                    output_id=output.output_id,
                ),
                "output": output_to_payload(output),
            },
        )
    return state.read_item(item_id)


@_state_operation
def require_open_model_output(
    state: RuntimeState,
    *,
    operation_id: str,
    attempt_id: str,
    generation: int,
    output_id: str,
    item_id: str,
) -> None:
    _active_attempt(state, operation_id, attempt_id, generation)
    record = state.read_event_reference("item_started", item_id)
    if (
        record is None
        or record.payload.get("attempt_id") != attempt_id
        or record.payload.get("output_id") != output_id
        or record.payload.get("operation_id") != operation_id
        or record.payload.get("generation") != generation
        or record.payload.get("kind") != "model_output"
        or state.read_item(item_id).status != "started"
    ):
        raise RuntimeError("model delta requires an open output")


@_state_operation
def complete_model_output(
    state: RuntimeState,
    *,
    operation_id: str,
    attempt_id: str,
    generation: int,
    output: ModelOutput,
) -> ItemSnapshot:
    with state.persistence.transaction():
        operation = _active_attempt(state, operation_id, attempt_id, generation)
        start = _start(state, attempt_id, output.output_id)
        if start is None:
            raise RuntimeError("model output not started")
        if state.read_item(str(start["item_id"])).status != "started":
            raise RuntimeError("model output already completed")
        original = output_from_payload(start["output"])
        if (original.kind, original.tool_call_id, original.tool_name) != (
            output.kind,
            output.tool_call_id,
            output.tool_name,
        ):
            raise RuntimeError("model output identity changed")
        state._append_and_reduce(
            thread_id=operation["thread_id"],
            turn_id=operation["turn_id"],
            record_type="item_completed",
            producer="model",
            payload={
                "item_id": start["item_id"],
                "status": "success",
                "payload": {"output": output_to_payload(output)},
            },
        )
    return state.read_item(str(start["item_id"]))


@_state_operation
def read_model_outputs(state: RuntimeState, *, attempt_id: str) -> tuple[ModelOutput, ...]:
    outputs = []
    for row in _starts(state, attempt_id):
        start = row["payload"]
        item = state.read_item(start["item_id"])
        outputs.append(output_from_payload(item.payload.get("output", start["output"])))
    return tuple(outputs)


@_state_operation
def close_model_outputs(
    state: RuntimeState,
    *,
    attempt_id: str,
    status: str,
    error: str,
    partial_outputs: tuple[ModelOutput, ...] = (),
) -> None:
    """Close only unfinished objects; completed generated content stays complete."""
    if status not in {"failed", "cancelled", "outcome_unknown"} or not error:
        raise ValueError("invalid interrupted model output outcome")
    partial_by_id = {output.output_id: output for output in partial_outputs}
    with state.persistence.transaction():
        for row in _starts(state, attempt_id):
            start = row["payload"]
            if state.read_item(start["item_id"]).status != "started":
                continue
            output = partial_by_id.get(start["output_id"], output_from_payload(start["output"]))
            state._append_and_reduce(
                thread_id=row["thread_id"],
                turn_id=row["turn_id"],
                record_type="item_completed",
                producer="runtime",
                payload={
                    "item_id": start["item_id"],
                    "status": status,
                    "error": error,
                    "payload": {"output": output_to_payload(output)},
                },
            )


@_state_operation
def finish_model_outputs(
    state: RuntimeState,
    *,
    operation_id: str,
    attempt_id: str,
    generation: int,
    outputs: tuple[ModelOutput, ...],
) -> None:
    """Reconcile a successful return with completed objects and nonstream results."""
    with state.persistence.transaction():
        _active_attempt(state, operation_id, attempt_id, generation)
        starts = _starts(state, attempt_id)
        observed_ids = [row["payload"]["output_id"] for row in starts]
        expected_ids = [output.output_id for output in outputs]
        if observed_ids != expected_ids[: len(observed_ids)]:
            raise RuntimeError("model response output order conflicts with its stream")
        for output in outputs:
            start = _start(state, attempt_id, output.output_id)
            if start is None:
                start_model_output(
                    state, operation_id=operation_id, attempt_id=attempt_id, generation=generation, output=output
                )
            elif state.read_item(start["item_id"]).status == "completed":
                committed = state.read_item(start["item_id"]).payload["output"]
                if committed != output_to_payload(output):
                    raise RuntimeError("completed model output conflicts with response snapshot")
                continue
            complete_model_output(
                state, operation_id=operation_id, attempt_id=attempt_id, generation=generation, output=output
            )

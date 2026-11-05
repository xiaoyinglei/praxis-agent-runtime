"""Read command evidence from canonical calls, outcomes and continuations."""

from __future__ import annotations

from collections.abc import Mapping

from agent_runtime.harness.records import ItemSnapshot, ToolOperationSnapshot
from agent_runtime.harness.runtime_store import RuntimeStore

type Evidence = tuple[ItemSnapshot, ToolOperationSnapshot, Mapping[str, object]]


def trusted_tool_results(store: RuntimeStore, turn_id: str) -> tuple[Evidence, ...]:
    items = {item.item_id: item for item in store.list_items(turn_id)}
    calls = {
        item.payload.get("tool_call_id"): item
        for item in items.values()
        if item.kind == "tool_call" and item.status == "completed" and item.producer == "model"
    }
    trusted = []
    for operation in store.list_tool_operations(turn_id):
        if (
            operation.status not in {"succeeded", "failed", "denied", "cancelled", "unknown"}
            or operation.result_item_id is None
        ):
            continue
        item = items.get(operation.result_item_id)
        call = calls.get(operation.tool_call_id)
        if (
            item is None
            or call is None
            or item.kind != "tool_result"
            or item.status != "completed"
            or item.producer != "tool"
            or call.payload.get("tool_name") != operation.tool_name
            or item.payload.get("tool_name") != operation.tool_name
            or item.payload.get("tool_call_id") != operation.tool_call_id
            or item.sequence <= call.sequence
            or not isinstance(call.payload.get("arguments"), Mapping)
        ):
            continue
        trusted.append((item, operation, call.payload["arguments"]))
    # A live owner can close while a Turn is paused. Its confirmed outcome
    # remains available after process handles and the Session are gone.
    launches = {
        operation.operation_id: (item, operation, arguments)
        for item, operation, arguments in trusted
        if operation.tool_name == "run_command"
    }
    for item in items.values():
        if item.kind != "tool_continuation" or item.producer != "runtime" or item.status != "completed":
            continue
        operation_id = item.payload.get("operation_id")
        source = launches.get(operation_id) if isinstance(operation_id, str) else None
        if source is None:
            continue
        launch, operation, arguments = source
        initial = launch.payload.get("structured_content")
        output = item.payload.get("structured_content")
        if (
            item.payload.get("source_result_item_id") == launch.item_id
            and isinstance(initial, Mapping)
            and isinstance(output, Mapping)
            and item.payload.get("key") == initial.get("session_id") == output.get("session_id")
            and output.get("status") in {"completed", "cancelled"}
        ):
            trusted.append((item, operation, arguments))
    return tuple(trusted)


def trusted_command_results(store: RuntimeStore, turn_id: str) -> tuple[Evidence, ...]:
    launches: dict[str, tuple[ItemSnapshot, Mapping[str, object]]] = {}
    commands = []
    for item, operation, arguments in sorted(trusted_tool_results(store, turn_id), key=lambda entry: entry[0].sequence):
        output = item.payload.get("structured_content")
        if not isinstance(output, Mapping):
            # A failed command can have no structured output and still
            # supersedes earlier successful evidence for the same command.
            if operation.tool_name == "run_command":
                commands.append((item, operation, arguments))
            continue
        session_id = output.get("session_id")
        if operation.tool_name == "run_command":
            commands.append((item, operation, arguments))
            if isinstance(session_id, str) and operation.status == "succeeded":
                launches[session_id] = (item, arguments)
        elif (
            operation.tool_name == "manage_command"
            and isinstance(session_id, str)
            and arguments.get("session_id") == session_id
            and session_id in launches
            and output.get("status") in {"completed", "cancelled"}
        ):
            launch, original_arguments = launches[session_id]
            if item.sequence > launch.sequence:
                commands.append((item, operation, original_arguments))
    return tuple(commands)


def command_succeeded(
    *, operation_status: object, is_error: object, output: Mapping[str, object],
) -> bool:
    exit_code = output.get("exit_code")
    return (
        operation_status == "succeeded"
        and is_error is False
        and type(exit_code) is int
        and exit_code == 0
        and output.get("status") in (None, "completed")
        and not output.get("timed_out")
        and not output.get("sandbox_error")
    )

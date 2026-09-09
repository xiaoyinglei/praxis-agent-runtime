"""Provider-neutral context projection from committed rollout Items."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent_runtime.harness.protocol import (
    ContextBudgetExceededError,
    HarnessMessage,
    HarnessToolCall,
)
from agent_runtime.harness.rollout import ItemSnapshot, RolloutStore


class RolloutContextManager:
    """Build model-visible messages without owning provider serialization."""

    def __init__(
        self,
        store: RolloutStore,
        *,
        max_item_bytes: int = 500_000,
        max_total_bytes: int = 4_000_000,
        max_messages: int = 2_000,
    ) -> None:
        for name, value in (
            ("max_item_bytes", max_item_bytes),
            ("max_total_bytes", max_total_bytes),
            ("max_messages", max_messages),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._store = store
        self._max_item_bytes = max_item_bytes
        self._max_total_bytes = max_total_bytes
        self._max_messages = max_messages

    def compact(
        self,
        *,
        turn_id: str,
        covered_item_ids: tuple[str, ...],
        summary: str,
        preserved_facts: Mapping[str, Any],
        artifact_refs: tuple[Mapping[str, Any], ...] = (),
        context_version: int,
    ) -> ItemSnapshot:
        """Persist a compaction Item; original Items remain canonical history."""

        return self._store.record_context_compaction(
            turn_id=turn_id,
            covered_item_ids=covered_item_ids,
            summary=summary,
            preserved_facts=preserved_facts,
            artifact_refs=artifact_refs,
            context_version=context_version,
        )

    def build(self, turn_id: str) -> tuple[HarnessMessage, ...]:
        messages: list[HarnessMessage] = []
        total_bytes = 0

        def append(message: HarnessMessage) -> None:
            nonlocal total_bytes
            item_bytes = _message_size_bytes(message)
            if item_bytes > self._max_item_bytes:
                raise ContextBudgetExceededError(
                    "Model context single Item exceeds the configured byte limit "
                    f"({item_bytes} > {self._max_item_bytes})."
                )
            if len(messages) >= self._max_messages:
                raise ContextBudgetExceededError(
                    "Model context exceeds the configured message-count limit "
                    f"({len(messages) + 1} > {self._max_messages})."
                )
            if total_bytes + item_bytes > self._max_total_bytes:
                raise ContextBudgetExceededError(
                    "Model context exceeds the configured total byte limit "
                    f"({total_bytes + item_bytes} > {self._max_total_bytes})."
                )
            messages.append(message)
            total_bytes += item_bytes

        for _item_id, message in self._projected_messages(turn_id):
            append(message)
        return tuple(messages)

    def compact_for_budget(
        self,
        *,
        turn_id: str,
        retained_tail_messages: int,
    ) -> ItemSnapshot:
        if (
            isinstance(retained_tail_messages, bool)
            or not isinstance(retained_tail_messages, int)
            or retained_tail_messages < 0
        ):
            raise ValueError("retained_tail_messages must be a non-negative integer")
        context_items = tuple(
            item
            for item in self._store.list_context_items(turn_id)
            if item.status == "completed"
        )
        projected = self._projected_messages(turn_id)
        latest_user_index = next(
            (
                index
                for index in range(len(projected) - 1, -1, -1)
                if projected[index][1].role == "user"
            ),
            None,
        )
        if latest_user_index is None:
            raise ContextBudgetExceededError(
                "Durable compaction cannot remove the final model-visible user message."
            )
        retained = max(retained_tail_messages, len(projected) - latest_user_index)
        first_retained_index = max(0, len(projected) - retained)
        if first_retained_index == 0:
            raise ContextBudgetExceededError(
                "No older model-visible context is available for durable compaction."
            )
        first_retained_item_id = projected[first_retained_index][0]
        raw_index = next(
            (
                index
                for index, item in enumerate(context_items)
                if item.item_id == first_retained_item_id
            ),
            None,
        )
        if raw_index is None or raw_index == 0:
            raise RuntimeError("durable compaction projection lost its Item boundary")
        covered_items = context_items[:raw_index]
        versions: list[int] = []
        for item in context_items:
            raw_version = item.payload.get("context_version")
            if (
                item.kind == "context_compaction"
                and isinstance(raw_version, int)
                and not isinstance(raw_version, bool)
            ):
                versions.append(raw_version)
        return self.compact(
            turn_id=turn_id,
            covered_item_ids=tuple(item.item_id for item in covered_items),
            summary=(
                f"Runtime compacted {len(covered_items)} canonical rollout items. "
                "Use preserved facts and re-read referenced workspace resources when needed."
            ),
            preserved_facts=_runtime_preserved_facts(
                self._store,
                turn_id=turn_id,
                covered_items=covered_items,
            ),
            context_version=max(versions, default=0) + 1,
        )

    def _projected_messages(
        self,
        turn_id: str,
    ) -> tuple[tuple[str, HarnessMessage], ...]:
        context_items = self._store.list_context_items(turn_id)
        replacements, suppressed_item_ids = _compaction_projection(context_items)
        projected: list[tuple[str, HarnessMessage]] = []
        for item in context_items:
            if item.status != "completed":
                continue
            replacement = replacements.get(item.item_id)
            if replacement is not None:
                projected.append((item.item_id, replacement))
            if item.item_id in suppressed_item_ids:
                continue
            message = _item_message(item)
            if message is not None:
                projected.append((item.item_id, message))
        return tuple(projected)


def _item_message(item: ItemSnapshot) -> HarnessMessage | None:
    text = item.payload.get("text")
    if item.kind == "user_message" and isinstance(text, str):
        return HarnessMessage(role="user", content=text)
    if item.kind == "agent_message" and isinstance(text, str):
        return HarnessMessage(role="assistant", content=text)
    if item.kind == "model_response" and isinstance(text, str):
        calls = item.payload.get("tool_calls")
        if isinstance(calls, (list, tuple)) and calls:
            return HarnessMessage(
                role="assistant",
                content=text,
                tool_calls=tuple(
                    _tool_call(call) for call in calls if isinstance(call, Mapping)
                ),
            )
    if item.kind == "tool_result":
        model_content = item.payload.get("model_content")
        tool_call_id = item.payload.get("tool_call_id")
        if isinstance(model_content, str) and isinstance(tool_call_id, str):
            return HarnessMessage(
                role="tool",
                content=model_content,
                tool_call_id=tool_call_id,
            )
    if item.kind in {"completion_feedback", "context_message"} and isinstance(
        text, str
    ):
        return HarnessMessage(role="context", content=text)
    if item.kind == "input_file":
        workspace_path = item.payload.get("workspace_path")
        sha256 = item.payload.get("sha256")
        if isinstance(workspace_path, str) and isinstance(sha256, str):
            return HarnessMessage(
                role="context",
                content=(
                    "Attached input file available in the workspace: "
                    f"{workspace_path} (sha256={sha256})."
                ),
            )
    return None


def _runtime_preserved_facts(
    store: RolloutStore,
    *,
    turn_id: str,
    covered_items: tuple[ItemSnapshot, ...],
) -> dict[str, Any]:
    covered_ids = {item.item_id for item in covered_items}
    turn_ids = {item.turn_id for item in covered_items}
    operations = tuple(
        operation
        for source_turn_id in turn_ids
        for operation in store.list_tool_operations(source_turn_id)
    )
    operation_by_result = {
        operation.result_item_id: operation
        for operation in operations
        if operation.result_item_id is not None
    }
    constraints = [
        {
            "item_id": item.item_id,
            "kind": item.kind,
            "text": item.payload["text"],
        }
        for item in covered_items
        if item.kind in {"user_message", "context_message", "completion_feedback"}
        and isinstance(item.payload.get("text"), str)
    ]
    file_changes = [
        {
            "operation_id": operation.operation_id,
            "tool_name": operation.tool_name,
            "resources": [dict(resource) for resource in operation.resources],
        }
        for item in covered_items
        if (operation := operation_by_result.get(item.item_id)) is not None
        and "write_workspace" in operation.effects
        and operation.status == "succeeded"
    ]
    verification_results = [
        dict(item.payload)
        for item in covered_items
        if item.kind == "verification" and item.producer == "runtime"
    ]
    plan_items = [
        item
        for item in store.list_context_items(turn_id)
        if item.kind == "plan_state"
        and item.status == "completed"
        and isinstance(item.payload.get("plan"), Mapping)
    ]
    unresolved_work = (
        []
        if not plan_items
        else [dict(plan_items[-1].payload["plan"])]
    )
    uncertain_side_effects = [
        {
            "operation_id": operation.operation_id,
            "tool_name": operation.tool_name,
            "status": operation.status,
            "resources": [dict(resource) for resource in operation.resources],
        }
        for operation in operations
        if operation.status == "unknown" or operation.requires_reconciliation
    ]
    inspectable_sources = [
        {
            "operation_id": operation.operation_id,
            "tool_name": operation.tool_name,
            "resources": [dict(resource) for resource in operation.resources],
        }
        for operation in operations
        if operation.result_item_id in covered_ids
        and operation.status == "succeeded"
        and "read_workspace" in operation.effects
    ]
    return {
        "architecture_and_safety_constraints": constraints,
        "file_changes": file_changes,
        "verification_results": verification_results,
        "unresolved_work": unresolved_work,
        "uncertain_side_effects": uncertain_side_effects,
        "inspectable_sources": inspectable_sources,
    }


def _tool_call(payload: Mapping[str, object]) -> HarnessToolCall:
    call_id = payload.get("id")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise RuntimeError("committed model tool call is malformed")
    return HarnessToolCall(id=call_id, name=name, arguments=arguments)


def _compaction_projection(
    items: tuple[ItemSnapshot, ...],
) -> tuple[dict[str, HarnessMessage], set[str]]:
    replacements: dict[str, HarnessMessage] = {}
    suppressed: set[str] = set()
    visible_ids = {item.item_id for item in items}
    for item in items:
        if item.status != "completed" or item.kind != "context_compaction":
            continue
        covered = item.payload.get("covered_item_ids")
        if (
            not isinstance(covered, (list, tuple))
            or not covered
            or not all(isinstance(item_id, str) for item_id in covered)
            or any(item_id not in visible_ids for item_id in covered)
        ):
            raise RuntimeError("committed context compaction coverage is malformed")
        replacements[covered[0]] = _compaction_message(item.payload)
        suppressed.update(covered)
        suppressed.add(item.item_id)
    return replacements, suppressed


def _compaction_message(payload: Mapping[str, Any]) -> HarnessMessage:
    projected = {
        "artifact_refs": payload.get("artifact_refs", []),
        "context_version": payload.get("context_version"),
        "durable_state": payload.get("durable_state", {}),
        "preserved_facts": payload.get("preserved_facts", {}),
        "summary": payload.get("summary"),
    }
    try:
        content = json.dumps(
            projected,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("committed context compaction payload is malformed") from exc
    return HarnessMessage(role="context", content=f"Context compaction:\n{content}")


def _message_size_bytes(message: HarnessMessage) -> int:
    payload = {
        "content": message.content,
        "role": message.role,
        "tool_call_id": message.tool_call_id,
        "tool_calls": [
            {
                "arguments": call.arguments,
                "id": call.id,
                "name": call.name,
            }
            for call in message.tool_calls
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return len(encoded.encode("utf-8"))

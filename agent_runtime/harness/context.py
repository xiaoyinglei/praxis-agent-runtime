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
        context_items = self._store.list_context_items(turn_id)
        replacements, suppressed_item_ids = _compaction_projection(context_items)
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

        for item in context_items:
            if item.status != "completed":
                continue
            replacement = replacements.get(item.item_id)
            if replacement is not None:
                append(replacement)
            if item.item_id in suppressed_item_ids:
                continue
            text = item.payload.get("text")
            if item.kind == "user_message" and isinstance(text, str):
                append(HarnessMessage(role="user", content=text))
            elif item.kind == "agent_message" and isinstance(text, str):
                append(HarnessMessage(role="assistant", content=text))
            elif item.kind == "model_response" and isinstance(text, str):
                calls = item.payload.get("tool_calls")
                if isinstance(calls, (list, tuple)) and calls:
                    append(
                        HarnessMessage(
                            role="assistant",
                            content=text,
                            tool_calls=tuple(_tool_call(call) for call in calls if isinstance(call, Mapping)),
                        )
                    )
            elif item.kind == "tool_result":
                model_content = item.payload.get("model_content")
                tool_call_id = item.payload.get("tool_call_id")
                if isinstance(model_content, str) and isinstance(tool_call_id, str):
                    append(
                        HarnessMessage(
                            role="tool",
                            content=model_content,
                            tool_call_id=tool_call_id,
                        )
                    )
            elif item.kind == "completion_feedback" and isinstance(text, str):
                append(HarnessMessage(role="context", content=text))
            elif item.kind == "context_message" and isinstance(text, str):
                append(HarnessMessage(role="context", content=text))
            elif item.kind == "input_file":
                workspace_path = item.payload.get("workspace_path")
                sha256 = item.payload.get("sha256")
                if isinstance(workspace_path, str) and isinstance(sha256, str):
                    append(
                        HarnessMessage(
                            role="context",
                            content=(
                                f"Attached input file available in the workspace: {workspace_path} (sha256={sha256})."
                            ),
                        )
                    )
        return tuple(messages)


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

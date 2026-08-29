"""Streaming event sink contracts and explicit legacy projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from agent_runtime.streaming.events import (
    EventType,
    ItemDeltaKind,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
    loop_end,
    text_delta,
    thinking_delta,
    tool_use_error,
    tool_use_progress,
    tool_use_start,
    turn_start,
)
from agent_runtime.tools.tool import JsonValue


class StreamEventSink(Protocol):
    """Streaming event sink protocol."""

    async def emit(self, event: StreamEvent) -> None: ...


class LegacyStreamProjectionSink:
    """Explicitly project canonical v2 events to the legacy wire vocabulary.

    The wrapped sink receives projections only; canonical events are never
    dual-emitted through this adapter.
    """

    def __init__(self, target: StreamEventSink) -> None:
        self._target = target
        self._tool_names: dict[str, str] = {}
        self._tool_ids: dict[str, str] = {}

    async def emit(self, event: StreamEvent) -> None:
        projected = self._project(event)
        if projected is not None:
            await self._target.emit(projected)

    def _project(self, event: StreamEvent) -> StreamEvent | None:
        if event.type in {EventType.TURN_STARTED, EventType.TURN_RESUMED}:
            return turn_start(turn_id=event.turn_id, iteration=event.iteration)
        if event.type in {
            EventType.TURN_PAUSED,
            EventType.TURN_COMPLETED,
            EventType.TURN_ABORTED,
        }:
            reason = event.data.get("reason") or event.data.get("status")
            return loop_end(
                turn_id=event.turn_id,
                reason=reason if isinstance(reason, str) else event.type.value,
                total_turns=event.iteration,
            )
        if event.type is EventType.ITEM_STARTED:
            return self._project_item_started(event)
        if event.type is EventType.ITEM_DELTA:
            return self._project_item_delta(event)
        if event.type is EventType.ITEM_COMPLETED:
            return self._project_item_completed(event)
        return None

    def _project_item_started(self, event: StreamEvent) -> StreamEvent | None:
        if event.item_kind not in {TurnItemKind.TOOL, TurnItemKind.COMMAND}:
            return None
        tool_name = event.data.get("tool_name")
        name = tool_name if isinstance(tool_name, str) else event.item_kind.value
        if event.item_id is None:
            return None
        self._tool_names[event.item_id] = name
        tool_call_id = event.data.get("tool_call_id")
        legacy_tool_id = (
            tool_call_id if isinstance(tool_call_id, str) else event.item_id
        )
        self._tool_ids[event.item_id] = legacy_tool_id
        preview = event.data.get("input_preview")
        return tool_use_start(
            name,
            legacy_tool_id,
            input_preview=preview if isinstance(preview, str) else "",
            turn_id=event.turn_id,
            iteration=event.iteration,
        )

    def _project_item_delta(self, event: StreamEvent) -> StreamEvent | None:
        delta = event.data.get("delta")
        if not isinstance(delta, str):
            return None
        if event.delta_kind is ItemDeltaKind.TEXT:
            return text_delta(delta, turn_id=event.turn_id, iteration=event.iteration)
        if event.delta_kind is ItemDeltaKind.REASONING:
            return thinking_delta(
                delta,
                turn_id=event.turn_id,
                iteration=event.iteration,
            )
        if event.delta_kind in {
            ItemDeltaKind.TOOL_PROGRESS,
            ItemDeltaKind.COMMAND_STDOUT,
            ItemDeltaKind.COMMAND_STDERR,
        } and event.item_id is not None:
            return tool_use_progress(
                self._tool_ids.get(event.item_id, event.item_id),
                delta,
                turn_id=event.turn_id,
                iteration=event.iteration,
            )
        return None

    def _project_item_completed(self, event: StreamEvent) -> StreamEvent | None:
        if event.item_kind is TurnItemKind.PLAN:
            plan = event.data.get("plan")
            if isinstance(plan, Mapping):
                data: dict[str, JsonValue] = {
                    "plan": cast(JsonValue, dict(plan)),
                }
                plan_event = event.data.get("event")
                if isinstance(plan_event, Mapping):
                    data["event"] = cast(JsonValue, dict(plan_event))
                return StreamEvent(
                    type=EventType.PLAN_UPDATED,
                    turn_id=event.turn_id,
                    iteration=event.iteration,
                    data=data,
                )
            return None
        if event.item_kind not in {TurnItemKind.TOOL, TurnItemKind.COMMAND}:
            return None
        if event.item_id is None:
            return None
        tool_name = self._tool_names.get(event.item_id, event.item_kind.value)
        legacy_tool_id = self._tool_ids.get(event.item_id, event.item_id)
        if event.status is ItemStatus.SUCCESS:
            result_text, details = _legacy_tool_result(event.data.get("result"))
            return StreamEvent(
                type=EventType.TOOL_USE_RESULT,
                turn_id=event.turn_id,
                iteration=event.iteration,
                item_id=legacy_tool_id,
                item_kind=TurnItemKind.TOOL,
                data={
                    "tool_name": tool_name,
                    "tool_id": legacy_tool_id,
                    "result": result_text,
                    "elapsed_ms": 0,
                    "details": details,
                },
            )
        return tool_use_error(
            legacy_tool_id,
            event.error
            or (event.status.value if event.status is not None else "tool failed"),
            recoverable=event.status is not ItemStatus.OUTCOME_UNKNOWN,
            turn_id=event.turn_id,
            iteration=event.iteration,
        )


def _legacy_tool_result(result: JsonValue | None) -> tuple[str, JsonValue]:
    if not isinstance(result, Mapping):
        return str(result)[:500], {}
    structured = result.get("structured_content")
    if structured is not None:
        text = str(structured)
    else:
        blocks = result.get("content")
        rendered: list[str] = []
        if isinstance(blocks, Sequence) and not isinstance(blocks, (str, bytes)):
            for block in blocks:
                if not isinstance(block, Mapping) or block.get("type") != "text":
                    continue
                data = block.get("data")
                if not isinstance(data, Mapping):
                    continue
                value = data.get("text")
                if isinstance(value, str):
                    rendered.append(value)
        text = "\n".join(rendered)
    metadata = result.get("metadata")
    details: dict[str, JsonValue] = {}
    if isinstance(metadata, Mapping):
        file_path = metadata.get("file_path")
        diff = metadata.get("diff")
        diff_truncated = metadata.get("diff_truncated")
        if (
            isinstance(file_path, str)
            and isinstance(diff, str)
            and type(diff_truncated) is bool
        ):
            details = {
                "file_path": file_path,
                "diff": diff,
                "diff_truncated": diff_truncated,
            }
    return text[:500], details

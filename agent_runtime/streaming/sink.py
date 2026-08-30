"""Streaming event sink contracts and explicit legacy projection."""

from __future__ import annotations

import asyncio
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


class EventChannelClosed(RuntimeError):  # noqa: N818 - protocol name
    """A controlling event consumer closed its channel."""


class ObserverLagged(RuntimeError):  # noqa: N818 - protocol name
    """A passive observer fell behind its bounded event channel."""

    def __init__(self, *, last_cursor: str | None) -> None:
        super().__init__("passive observer lagged; reconnect from durable history")
        self.last_cursor = last_cursor


class _EventChannel:
    def __init__(self, *, capacity: int, last_cursor: str | None = None) -> None:
        self.queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=capacity)
        self.closed = asyncio.Event()
        self.error: Exception = EventChannelClosed("event channel is closed")
        self.last_cursor = last_cursor

    async def put(self, event: StreamEvent) -> None:
        if self.closed.is_set():
            raise self.error
        put = asyncio.create_task(self.queue.put(event))
        closing = asyncio.create_task(self.closed.wait())
        await asyncio.wait((put, closing), return_when=asyncio.FIRST_COMPLETED)
        if self.closed.is_set():
            put.cancel()
            await asyncio.gather(put, return_exceptions=True)
            self._discard_queued()
            raise self.error
        closing.cancel()
        await asyncio.gather(closing, return_exceptions=True)
        await put

    async def receive(self) -> StreamEvent:
        if self.closed.is_set():
            raise self.error
        receive = asyncio.create_task(self.queue.get())
        closing = asyncio.create_task(self.closed.wait())
        await asyncio.wait((receive, closing), return_when=asyncio.FIRST_COMPLETED)
        if self.closed.is_set():
            receive.cancel()
            await asyncio.gather(receive, return_exceptions=True)
            self._discard_queued()
            raise self.error
        closing.cancel()
        await asyncio.gather(closing, return_exceptions=True)
        return await receive

    def put_passive(self, event: StreamEvent, *, cursor: str | None) -> None:
        if self.closed.is_set():
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.close(error=ObserverLagged(last_cursor=self.last_cursor))
            return
        if cursor is not None:
            self.last_cursor = cursor

    def close(self, *, error: Exception | None = None) -> None:
        if self.closed.is_set():
            return
        if error is not None:
            self.error = error
        self.closed.set()
        self._discard_queued()

    def _discard_queued(self) -> None:
        while not self.queue.empty():
            self.queue.get_nowait()


class TurnEventStream:
    """One controlling consumer of a Turn's canonical event stream."""

    def __init__(self, channel: _EventChannel) -> None:
        self._channel = channel

    async def receive(self) -> StreamEvent:
        return await self._channel.receive()

    def close(self) -> None:
        self._channel.close()


class TurnEventDispatcher:
    """Fan out one Turn's events with controlling-consumer backpressure."""

    def __init__(self, *, capacity: int = 64) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._controlling: list[_EventChannel] = []
        self._controlling_sinks: list[StreamEventSink] = []
        self._passive: list[_EventChannel] = []

    def subscribe_controlling(self) -> TurnEventStream:
        channel = _EventChannel(capacity=self._capacity)
        self._controlling.append(channel)
        return TurnEventStream(channel)

    def subscribe_passive(self, *, last_cursor: str | None = None) -> TurnEventStream:
        channel = _EventChannel(capacity=self._capacity, last_cursor=last_cursor)
        self._passive.append(channel)
        return TurnEventStream(channel)

    def subscribe_controlling_sink(self, sink: StreamEventSink) -> None:
        self._controlling_sinks.append(sink)

    async def emit(self, event: StreamEvent, *, cursor: str | None = None) -> None:
        for channel in tuple(self._controlling):
            await channel.put(event)
        for channel in tuple(self._passive):
            channel.put_passive(event, cursor=cursor)
        for sink in tuple(self._controlling_sinks):
            try:
                await sink.emit(event)
            except Exception as exc:
                raise EventChannelClosed(
                    f"controlling event sink failed: {exc}"
                ) from exc


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

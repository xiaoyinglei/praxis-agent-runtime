"""
StreamEventSink — 流式事件的出口。

QueueStreamEventSink 把事件放进 asyncio.Queue，供外部 async generator 消费。

使用方式：
    sink = QueueStreamEventSink()
    task = asyncio.create_task(agent_loop.run(..., stream_sink=sink))
    async for event in sink.stream():
        render(event)
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

from agent_runtime.core.messages import ModelMessage, ToolCall, canonical_json_text
from agent_runtime.streaming.events import (
    EventType,
    ItemDeltaKind,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
    item_completed,
    loop_end,
    text_delta,
    thinking_delta,
    tool_use_error,
    tool_use_progress,
    tool_use_start,
    turn_start,
)
from agent_runtime.tools.tool import JsonValue

if TYPE_CHECKING:
    from agent_runtime.turns import TurnStore


class StreamEventSink(Protocol):
    """流式事件 sink 协议。"""

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
            return turn_start(
                turn_id=event.turn_id,
                iteration=event.iteration,
            )
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
            return text_delta(
                delta,
                turn_id=event.turn_id,
                iteration=event.iteration,
            )
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
            result = event.data.get("result")
            result_text, details = _legacy_tool_result(result)
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
            (
                event.error
                or (event.status.value if event.status is not None else "tool failed")
            ),
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


class DurableStreamEventSink:
    """Persist authoritative Item completion before delivering it live."""

    def __init__(
        self,
        *,
        turn_store: TurnStore,
        live_sink: StreamEventSink,
    ) -> None:
        self._turn_store = turn_store
        self._live_sink: StreamEventSink | None = live_sink
        self._started_at_ms: dict[tuple[str, str], int] = {}
        self._started_items: dict[tuple[str, str], StreamEvent] = {}
        self._item_deltas: dict[tuple[str, str], list[str]] = {}

    async def emit(self, event: StreamEvent) -> None:
        key = (event.turn_id, event.item_id or "")
        if event.type is EventType.ITEM_STARTED and event.item_id is not None:
            self._started_at_ms[key] = event.timestamp_ms
            self._started_items[key] = event
            self._item_deltas[key] = []
        elif event.type is EventType.ITEM_DELTA:
            self._item_deltas.setdefault(key, []).append(
                cast(str, event.data["delta"])
            )
        elif event.type is EventType.ITEM_COMPLETED:
            message = _completed_item_message(event)
            self._turn_store.commit_completed_item(
                event,
                message=message,
                started_at_ms=self._started_at_ms.pop(key, None),
            )
            self._started_items.pop(key, None)
            self._item_deltas.pop(key, None)
        if self._live_sink is not None:
            await self._live_sink.emit(event)

    async def close_open_items(self) -> None:
        """Durably cancel every started Item before its Turn is aborted."""

        for key, started in tuple(self._started_items.items()):
            if started.item_id is None or started.item_kind is None:
                continue
            content = "".join(self._item_deltas.get(key, ()))
            data: dict[str, JsonValue] = (
                {"content": content}
                if started.item_kind
                in {
                    TurnItemKind.AGENT_MESSAGE,
                    TurnItemKind.REASONING,
                    TurnItemKind.PLAN,
                }
                else {}
            )
            completed = item_completed(
                turn_id=started.turn_id,
                item_id=started.item_id,
                item_kind=started.item_kind,
                status=ItemStatus.CANCELLED,
                data=data,
                iteration=started.iteration,
                parent_item_id=started.parent_item_id,
            )
            self._turn_store.commit_completed_item(
                completed,
                started_at_ms=self._started_at_ms.get(key),
            )

    def disconnect_live(self) -> None:
        """Stop delivery without changing durable completion semantics."""

        self._live_sink = None

class QueueStreamEventSink:
    """基于 asyncio.Queue 的流式事件 sink。

    外部通过 stream() 消费事件。
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(
            maxsize=maxsize
        )
        self._closed = asyncio.Event()

    async def emit(self, event: StreamEvent) -> None:
        """往 queue 里放一个事件。"""
        if self._closed.is_set():
            raise RuntimeError("stream sink is closed")
        put_task = asyncio.create_task(self._queue.put(event))
        close_task = asyncio.create_task(self._closed.wait())
        done, _pending = await asyncio.wait(
            {put_task, close_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if put_task in done:
            close_task.cancel()
            await asyncio.gather(close_task, return_exceptions=True)
            return
        put_task.cancel()
        await asyncio.gather(put_task, return_exceptions=True)
        raise RuntimeError("stream sink closed during emit")

    async def stream(self) -> AsyncGenerator[StreamEvent, None]:
        """消费事件流，直到收到 LOOP_END 或 ABORT。"""
        while not self._closed.is_set() or not self._queue.empty():
            get_task = asyncio.create_task(self._queue.get())
            close_task = asyncio.create_task(self._closed.wait())
            done, _pending = await asyncio.wait(
                {get_task, close_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if get_task in done:
                close_task.cancel()
                await asyncio.gather(close_task, return_exceptions=True)
                event = get_task.result()
                if event is not None:
                    yield event
                continue
            get_task.cancel()
            await asyncio.gather(get_task, return_exceptions=True)

    async def close(self) -> None:
        """通知消费者结束。"""
        self._closed.set()

    @property
    def queue(self) -> asyncio.Queue[StreamEvent | None]:
        return self._queue


def _completed_item_message(event: StreamEvent) -> ModelMessage | None:
    if event.item_kind in {
        TurnItemKind.TOOL,
        TurnItemKind.COMMAND,
        TurnItemKind.PLAN,
    } and isinstance(event.data.get("result"), Mapping):
        return _completed_tool_message(event)
    if (
        event.item_kind is not TurnItemKind.AGENT_MESSAGE
        or event.status is not ItemStatus.SUCCESS
    ):
        return None
    content = event.data.get("content")
    raw_calls = event.data.get("tool_calls")
    if not isinstance(content, str):
        raise ValueError("completed agent_message requires string content")
    if not isinstance(raw_calls, (list, tuple)):
        raise ValueError("completed agent_message requires tool_calls sequence")
    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            raise ValueError("completed agent_message tool_calls must be objects")
        call_id = raw_call.get("id")
        name = raw_call.get("name")
        arguments = raw_call.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("completed agent_message tool call requires id")
        if not isinstance(name, str) or not name:
            raise ValueError("completed agent_message tool call requires name")
        if not isinstance(arguments, Mapping):
            raise ValueError("completed agent_message tool call requires arguments")
        calls.append(
            ToolCall(
                id=call_id,
                name=name,
                input=cast(dict[str, object], dict(arguments)),
            )
        )
    return ModelMessage(
        role="assistant",
        content=content,
        tool_calls=tuple(calls),
    )


def _completed_tool_message(event: StreamEvent) -> ModelMessage | None:
    result = event.data.get("result")
    if not isinstance(result, Mapping):
        return None
    tool_call_id = result.get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("completed tool item result requires tool_call_id")
    content = result.get("content")
    if not isinstance(content, (list, tuple)):
        raise ValueError("completed tool item result requires content sequence")
    payload = {
        "content": content,
        "structured_content": result.get("structured_content"),
        "is_error": result.get("is_error"),
        "error_code": result.get("error_code"),
        "error_message": result.get("error_message"),
        "retryable": result.get("retryable"),
        "truncated": result.get("truncated"),
    }
    return ModelMessage(
        role="tool",
        content=canonical_json_text(cast(JsonValue, payload)),
        tool_call_id=tool_call_id,
    )

"""
StreamEventSink — 流式事件的出口。

两种实现：
1. QueueStreamEventSink — 把事件放进 asyncio.Queue，外部 async generator 消费
2. NoopStreamEventSink — 什么都不做（非流式场景）

使用方式：
    sink = QueueStreamEventSink()
    task = asyncio.create_task(agent_loop.run(..., stream_sink=sink))
    async for event in sink.stream():
        render(event)
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, Protocol, cast

from rag.agent.core.messages import ModelMessage, ToolCall
from rag.agent.streaming.events import (
    EventType,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
)

if TYPE_CHECKING:
    from rag.agent.turns import TurnStore


class StreamEventSink(Protocol):
    """流式事件 sink 协议。"""

    async def emit(self, event: StreamEvent) -> None: ...


class NoopStreamEventSink:
    """空实现，非流式场景使用。"""

    async def emit(self, event: StreamEvent) -> None:
        pass


class DurableStreamEventSink:
    """Persist authoritative Item completion before delivering it live."""

    def __init__(
        self,
        *,
        turn_store: TurnStore,
        live_sink: StreamEventSink,
    ) -> None:
        self._turn_store = turn_store
        self._live_sink = live_sink
        self._started_at_ms: dict[tuple[str, str], int] = {}

    async def emit(self, event: StreamEvent) -> None:
        key = (event.turn_id, event.item_id or "")
        if event.type is EventType.ITEM_STARTED and event.item_id is not None:
            self._started_at_ms[key] = event.timestamp_ms
        elif event.type is EventType.ITEM_COMPLETED:
            message = _completed_item_message(event)
            self._turn_store.commit_completed_item(
                event,
                message=message,
                started_at_ms=self._started_at_ms.pop(key, None),
            )
        await self._live_sink.emit(event)


class QueueStreamEventSink:
    """基于 asyncio.Queue 的流式事件 sink。

    外部通过 stream() 消费事件。
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(
            maxsize=maxsize
        )

    async def emit(self, event: StreamEvent) -> None:
        """往 queue 里放一个事件。"""
        await self._queue.put(event)

    async def stream(self) -> AsyncGenerator[StreamEvent, None]:
        """消费事件流，直到收到 LOOP_END 或 ABORT。"""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event
            if event.type in {EventType.LOOP_END, EventType.ABORT}:
                break

    async def close(self) -> None:
        """通知消费者结束。"""
        await self._queue.put(None)

    @property
    def queue(self) -> asyncio.Queue[StreamEvent | None]:
        return self._queue


def _completed_item_message(event: StreamEvent) -> ModelMessage | None:
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

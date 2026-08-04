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
from collections.abc import AsyncGenerator
from typing import Protocol

from agent_runtime.streaming.events import EventType, StreamEvent


class StreamEventSink(Protocol):
    """流式事件 sink 协议。"""

    async def emit(self, event: StreamEvent) -> None: ...


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

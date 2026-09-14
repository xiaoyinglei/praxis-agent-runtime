"""Canonical streaming event channels and sink contracts."""

from __future__ import annotations

import asyncio
import math
from typing import Protocol

from agent_runtime.streaming.events import StreamEvent


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
        self.available = asyncio.Event()
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
        self.available.set()

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
        event = await receive
        if self.queue.empty():
            self.available.clear()
        return event

    def receive_nowait(self) -> StreamEvent:
        if self.closed.is_set():
            raise self.error
        event = self.queue.get_nowait()
        if self.queue.empty():
            self.available.clear()
        return event

    def put_passive(self, event: StreamEvent, *, cursor: str | None) -> None:
        if self.closed.is_set():
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.close(error=ObserverLagged(last_cursor=self.last_cursor))
            return
        self.available.set()
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
        self.available.clear()


class TurnEventStream:
    """One controlling consumer of a Turn's canonical event stream."""

    def __init__(self, channel: _EventChannel) -> None:
        self._channel = channel

    async def receive(self) -> StreamEvent:
        return await self._channel.receive()

    def receive_nowait(self) -> StreamEvent:
        return self._channel.receive_nowait()

    async def wait_available(self) -> None:
        await self._channel.available.wait()

    def close(self) -> None:
        self._channel.close()

    @property
    def empty(self) -> bool:
        return self._channel.queue.empty()


class TurnEventDispatcher:
    """Fan out one Turn's events with controlling-consumer backpressure."""

    def __init__(
        self,
        *,
        capacity: int = 64,
        sink_cancel_grace_seconds: float = 1.0,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if (
            isinstance(sink_cancel_grace_seconds, bool)
            or not isinstance(sink_cancel_grace_seconds, (int, float))
            or not math.isfinite(sink_cancel_grace_seconds)
            or sink_cancel_grace_seconds <= 0
        ):
            raise ValueError("sink cancellation grace must be finite and positive")
        self._capacity = capacity
        self._sink_cancel_grace_seconds = float(sink_cancel_grace_seconds)
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

    def unsubscribe_controlling_sink(self, sink: StreamEventSink) -> None:
        self._controlling_sinks.remove(sink)

    def close(self) -> None:
        for channel in (*self._controlling, *self._passive):
            channel.close()
        self._controlling.clear()
        self._passive.clear()
        self._controlling_sinks.clear()

    async def emit(self, event: StreamEvent, *, cursor: str | None = None) -> None:
        for channel in tuple(self._controlling):
            await channel.put(event)
        for channel in tuple(self._passive):
            channel.put_passive(event, cursor=cursor)
        for sink in tuple(self._controlling_sinks):
            delivery = asyncio.create_task(sink.emit(event))
            done, _pending = await asyncio.wait(
                {delivery},
                timeout=self._sink_cancel_grace_seconds,
            )
            if not done:
                delivery.cancel()
                await asyncio.sleep(0)
                delivery.add_done_callback(_consume_task_result)
                raise EventChannelClosed(
                    "controlling event sink exceeded cancellation grace"
                )
            try:
                delivery.result()
            except Exception as exc:
                raise EventChannelClosed(
                    f"controlling event sink failed: {exc}"
                ) from exc


def _consume_task_result(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    task.exception()

"""进程内非阻塞事件总线 — UI hint channel (设计 §5.6)。

核心语义：emit 是同步、非阻塞的，永远不抛异常、永不阻塞。
订阅者通过 asyncio.Queue[Event] 接收提示事件；core loop 从不 await 总线。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from praxis.protocol.events import Event


@dataclass
class Subscription:
    """一个事件总线的订阅者。"""
    queue: asyncio.Queue[Event]
    dropped: int = field(default=0, init=False)
    alive: bool = field(default=True, init=False)


class EventBus:
    """非阻塞事件总线。

    - subscribe() → Subscription(maxsize=1000 queue)
    - unsubscribe(sub) removes sub from the active list
    - emit(event) iterates live subscribers, put_nowait per subscriber;
      QueueFull → sub.dropped+=1; any other exception → mark dead & remove
      **Never raises, never blocks.**
    - dropped_total sums dropped across all tracked subscriptions (live + dead).
    """

    _maxsize: int = 1000

    def __init__(self) -> None:
        self._subscriptions: list[Subscription] = []
        self._dead_dropped_accumulator: int = 0

    # -- public API ----------------------------------------------------------

    def subscribe(self) -> Subscription:
        sub = Subscription(queue=asyncio.Queue(maxsize=self._maxsize))
        self._subscriptions.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        try:
            self._subscriptions.remove(sub)
        except ValueError:
            pass

    def emit(self, event: Event) -> None:
        to_remove: list[Subscription] = []
        for sub in list(self._subscriptions):
            if not sub.alive:
                to_remove.append(sub)
                continue
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                sub.dropped += 1
            except BaseException:
                sub.alive = False
                to_remove.append(sub)
        for sub in to_remove:
            self._subscriptions.remove(sub)
            self._dead_dropped_accumulator += sub.dropped

    @property
    def dropped_total(self) -> int:
        total = sum(s.dropped for s in self._subscriptions if s.alive)
        total += self._dead_dropped_accumulator
        return total

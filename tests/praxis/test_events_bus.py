"""非阻塞事件总线测试：Subscription + EventBus。"""

from __future__ import annotations

from praxis.core.events import EventBus
from praxis.protocol.events import ToolStarted


def test_emit_never_raises_on_full_queue():
    bus = EventBus()
    sub = bus.subscribe()
    for _ in range(2000):
        bus.emit(ToolStarted("t", "c", "read"))  # capacity 1000
    assert sub.dropped == 1000 and bus.dropped_total == 1000


def test_emit_never_raises_on_broken_subscriber(monkeypatch):
    bus = EventBus()
    bad = bus.subscribe()
    good = bus.subscribe()
    monkeypatch.setattr(
        bad.queue, "put_nowait", lambda e: (_ for _ in ()).throw(RuntimeError("dead"))
    )
    bus.emit(ToolStarted("t", "c", "read"))       # should not raise
    bus.emit(ToolStarted("t", "c2", "read"))
    assert good.queue.qsize() == 2 and not bad.alive  # dead sub removed, good unaffected


def test_emit_no_subscribers_is_noop():
    EventBus().emit(ToolStarted("t", "c", "read"))


def test_unsubscribe_removes_subscriber():
    """After unsubscribe, emit does not reach subscriber."""
    bus = EventBus()
    sub = bus.subscribe()
    bus.unsubscribe(sub)
    bus.emit(ToolStarted("t", "c", "read"))
    assert sub.queue.empty()
    assert sub.dropped == 0

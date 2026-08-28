from __future__ import annotations

from dataclasses import dataclass

from rag.agent.streaming.events import StreamEvent


@dataclass(frozen=True, slots=True)
class DurableTurnEvent:
    """One replayable Turn fact ordered independently from live delivery."""

    durable_ordinal: int
    event: StreamEvent


__all__ = ["DurableTurnEvent"]

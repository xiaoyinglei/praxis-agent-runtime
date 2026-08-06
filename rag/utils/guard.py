from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

_logger = logging.getLogger("rag.guard")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# ═══════════════════════════════════════════════════
# Rate Limiter
# ═══════════════════════════════════════════════════


@dataclass(slots=True)
class _UserWindow:
    timestamps: deque[float] = field(default_factory=deque)


class RateLimiter:
    """Sliding-window rate limiter with per-user and global limits.

    Only *allowed* requests have their timestamps recorded, preventing OOM
    from rejected-request timestamp accumulation (Trap 4).
    """

    def __init__(
        self,
        *,
        window_seconds: float | None = None,
        max_per_user: int | None = None,
        max_global: int | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        self._window_seconds = (
            window_seconds if window_seconds is not None else _env_float("RAG_RATE_WINDOW_SECONDS", 60.0)
        )
        self._max_per_user = max_per_user if max_per_user is not None else _env_int("RAG_RATE_MAX_PER_USER", 60)
        self._max_global = max_global if max_global is not None else _env_int("RAG_RATE_MAX_GLOBAL", 300)
        max_conc = max_concurrent if max_concurrent is not None else _env_int("RAG_RATE_MAX_CONCURRENT", 10)
        self._semaphore = threading.BoundedSemaphore(max_conc)
        self._lock = threading.Lock()
        self._users: dict[str, _UserWindow] = {}
        self._global_timestamps: deque[float] = deque()

    def try_acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()

    def allow(self, *, user_id: str = "anonymous") -> bool:
        now = time.monotonic()
        cutoff = now - self._window_seconds

        with self._lock:
            self._purge_expired(self._global_timestamps, cutoff)
            if len(self._global_timestamps) >= self._max_global:
                return False

            user = self._users.get(user_id)
            if user is None:
                user = _UserWindow()
                self._users[user_id] = user
            else:
                self._purge_expired(user.timestamps, cutoff)

            if len(user.timestamps) >= self._max_per_user:
                return False

            user.timestamps.append(now)
            self._global_timestamps.append(now)
            return True

    @staticmethod
    def _purge_expired(timestamps: deque[float], cutoff: float) -> None:
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "global_requests": len(self._global_timestamps),
                "max_global": self._max_global,
                "max_per_user": self._max_per_user,
                "max_concurrent": self._semaphore._initial_value,  # type: ignore[attr-defined]
                "user_count": len(self._users),
            }


RateLimitExceeded = type("RateLimitExceeded", (Exception,), {})


# ═══════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════


class _State(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@dataclass(slots=True)
class CircuitConfig:
    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    half_open_max_probes: int = 1
    success_threshold: int = 2


class CircuitBreaker:
    """Thread-safe circuit breaker with HALF_OPEN probe gating (Trap 3)."""

    def __init__(self, name: str, config: CircuitConfig | None = None) -> None:
        self.name = name
        self._config = config or CircuitConfig()
        self._state = _State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = 0.0
        self._half_open_probes = 0
        self._lock = threading.Lock()

    # ── public API ──────────────────────────────────

    def allow(self) -> bool:
        """Return True if the call should proceed, False to fast-fail."""
        with self._lock:
            return self._allow_locked()

    def on_success(self) -> None:
        with self._lock:
            self._on_success_locked()

    def on_failure(self) -> None:
        with self._lock:
            self._on_failure_locked()

    @contextmanager
    def guard(self, *, default: Any = None) -> Generator[None, None, None]:
        """Context manager: raises _CircuitOpenError or returns default on open."""
        if not self.allow():
            raise _CircuitOpenError(f"circuit {self.name} is OPEN")
        try:
            yield
        except Exception:
            self.on_failure()
            raise
        else:
            self.on_success()

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_probes": self._half_open_probes,
            }

    # ── internal state machine ──────────────────────

    def _allow_locked(self) -> bool:
        if self._state is _State.CLOSED:
            return True

        if self._state is _State.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.cooldown_seconds:
                self._state = _State.HALF_OPEN
                self._half_open_probes = 0
                self._success_count = 0
                _logger.info("circuit %s: OPEN → HALF_OPEN (cooldown elapsed)", self.name)
            else:
                return False

        if self._state is _State.HALF_OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.cooldown_seconds:
                self._half_open_probes = 0
                self._opened_at = time.monotonic()
            if self._half_open_probes >= self._config.half_open_max_probes:
                return False
            self._half_open_probes += 1
            return True

        return False

    def _on_success_locked(self) -> None:
        if self._state is _State.CLOSED:
            self._failure_count = 0
            return

        if self._state is _State.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._config.success_threshold:
                self._state = _State.CLOSED
                self._failure_count = 0
                self._success_count = 0
                self._half_open_probes = 0
                _logger.info("circuit %s: HALF_OPEN → CLOSED (recovered)", self.name)

    def _on_failure_locked(self) -> None:
        if self._state is _State.HALF_OPEN:
            self._state = _State.OPEN
            self._opened_at = time.monotonic()
            self._failure_count = self._config.failure_threshold
            self._half_open_probes = 0
            _logger.warning("circuit %s: HALF_OPEN → OPEN (probe failed)", self.name)
            return

        if self._state is _State.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._config.failure_threshold:
                self._state = _State.OPEN
                self._opened_at = time.monotonic()
                _logger.warning(
                    "circuit %s: CLOSED → OPEN (%d consecutive failures)",
                    self.name,
                    self._failure_count,
                )


class _CircuitOpenError(Exception):
    pass


# ═══════════════════════════════════════════════════
# Circuit Breaker decorator
# ═══════════════════════════════════════════════════


def guarded(
    breaker: CircuitBreaker,
    *,
    default: Any = None,
    reraise: bool = False,
) -> Any:
    """Decorator that wraps a function with a circuit breaker.

    On circuit-open: returns *default* (or raises if reraise=True).
    On circuit-closed/half-open: calls the function normally; failures
    are reported to the breaker.
    """

    def decorator(fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not breaker.allow():
                if reraise:
                    raise _CircuitOpenError(f"circuit {breaker.name} is OPEN")
                return default
            try:
                result = fn(*args, **kwargs)
            except Exception:
                breaker.on_failure()
                raise
            else:
                breaker.on_success()
                return result

        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        return wrapper

    return decorator


__all__ = [
    "CircuitBreaker",
    "CircuitConfig",
    "RateLimiter",
    "RateLimitExceeded",
    "guarded",
]

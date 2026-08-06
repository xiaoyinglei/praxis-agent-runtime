from __future__ import annotations

import threading
import time

from rag.utils.guard import CircuitBreaker, CircuitConfig, RateLimiter, guarded

# ═══════════════════════════════════════════════════
# RateLimiter tests
# ═══════════════════════════════════════════════════


def test_rate_limiter_allows_within_limit() -> None:
    rl = RateLimiter(max_per_user=5, max_global=100, max_concurrent=20)
    for _ in range(5):
        assert rl.allow(user_id="u1") is True


def test_rate_limiter_blocks_user_above_limit() -> None:
    rl = RateLimiter(max_per_user=3, max_global=100, max_concurrent=20)
    for _ in range(3):
        assert rl.allow(user_id="u1") is True
    assert rl.allow(user_id="u1") is False


def test_rate_limiter_blocks_global_limit() -> None:
    rl = RateLimiter(max_per_user=100, max_global=3, max_concurrent=20)
    for i in range(3):
        assert rl.allow(user_id=f"u{i}") is True
    assert rl.allow(user_id="u3") is False


def test_rate_limiter_independent_users() -> None:
    rl = RateLimiter(max_per_user=2, max_global=100, max_concurrent=20)
    assert rl.allow(user_id="u1") is True
    assert rl.allow(user_id="u1") is True
    assert rl.allow(user_id="u1") is False
    assert rl.allow(user_id="u2") is True
    assert rl.allow(user_id="u2") is True


def test_rate_limiter_concurrent_semaphore() -> None:
    rl = RateLimiter(max_per_user=100, max_global=100, max_concurrent=2)
    assert rl.try_acquire() is True
    assert rl.try_acquire() is True
    assert rl.try_acquire() is False
    rl.release()
    assert rl.try_acquire() is True


def test_rate_limiter_thread_safety() -> None:
    rl = RateLimiter(max_per_user=50, max_global=1000, max_concurrent=100)
    errors: list[Exception] = []

    def worker(uid: str) -> None:
        try:
            for _ in range(10):
                rl.allow(user_id=uid)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(f"u{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# ═══════════════════════════════════════════════════
# CircuitBreaker tests
# ═══════════════════════════════════════════════════


def test_circuit_breaker_starts_closed() -> None:
    cb = CircuitBreaker("test")
    assert cb.allow() is True


def test_circuit_breaker_opens_after_failures() -> None:
    cb = CircuitBreaker("test", CircuitConfig(failure_threshold=3, cooldown_seconds=60.0))
    for _ in range(3):
        assert cb.allow() is True
        cb.on_failure()
    assert cb.allow() is False


def test_circuit_breaker_resets_on_success_in_closed() -> None:
    cb = CircuitBreaker("test", CircuitConfig(failure_threshold=3, cooldown_seconds=60.0))
    cb.on_failure()
    cb.on_failure()
    cb.on_success()
    cb.on_success()
    cb.on_failure()
    cb.on_failure()
    cb.on_failure()
    assert cb.allow() is False


def test_circuit_breaker_half_open_probe_limit() -> None:
    cb = CircuitBreaker(
        "test",
        CircuitConfig(failure_threshold=1, cooldown_seconds=0.01, half_open_max_probes=1),
    )
    cb.on_failure()
    assert cb.allow() is False

    time.sleep(0.02)
    assert cb.allow() is True
    assert cb.allow() is False


def test_circuit_breaker_half_open_recovery() -> None:
    cb = CircuitBreaker(
        "test",
        CircuitConfig(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_probes=1, success_threshold=2
        ),
    )
    cb.on_failure()

    time.sleep(0.02)
    assert cb.allow() is True
    cb.on_success()
    assert cb.allow() is False

    time.sleep(0.02)
    assert cb.allow() is True
    cb.on_success()
    assert cb.allow() is True


def test_circuit_breaker_half_open_back_to_open_on_failure() -> None:
    cb = CircuitBreaker(
        "test",
        CircuitConfig(failure_threshold=1, cooldown_seconds=0.01, half_open_max_probes=1, success_threshold=2),
    )
    cb.on_failure()

    time.sleep(0.02)
    assert cb.allow() is True
    cb.on_failure()
    assert cb.allow() is False


def test_guarded_decorator_returns_default_on_open() -> None:
    cb = CircuitBreaker("test", CircuitConfig(failure_threshold=1, cooldown_seconds=60.0))

    @guarded(cb, default="FALLBACK")
    def might_fail() -> str:
        raise RuntimeError("boom")

    try:
        might_fail()
    except RuntimeError:
        pass

    assert might_fail() == "FALLBACK"


def test_guarded_decorator_passes_through_on_closed() -> None:
    cb = CircuitBreaker("test")

    @guarded(cb)
    def succeed() -> str:
        return "OK"

    assert succeed() == "OK"


def test_guarded_decorator_reraises_on_open() -> None:
    cb = CircuitBreaker("test", CircuitConfig(failure_threshold=1, cooldown_seconds=60.0))

    @guarded(cb, reraise=True)
    def might_fail() -> str:
        raise RuntimeError("boom")

    try:
        might_fail()
    except RuntimeError:
        pass
    assert cb.allow() is False


def test_circuit_breaker_thread_safety() -> None:
    cb = CircuitBreaker("test", CircuitConfig(failure_threshold=2, cooldown_seconds=0.02, half_open_max_probes=1))
    cb.on_failure()
    cb.on_failure()
    assert cb.allow() is False

    time.sleep(0.03)

    probes: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        result = cb.allow()
        with lock:
            probes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(probes) == 1


__all__ = [
    "test_rate_limiter_allows_within_limit",
    "test_rate_limiter_blocks_user_above_limit",
    "test_rate_limiter_blocks_global_limit",
    "test_rate_limiter_independent_users",
    "test_rate_limiter_concurrent_semaphore",
    "test_rate_limiter_thread_safety",
    "test_circuit_breaker_starts_closed",
    "test_circuit_breaker_opens_after_failures",
    "test_circuit_breaker_resets_on_success_in_closed",
    "test_circuit_breaker_half_open_probe_limit",
    "test_circuit_breaker_half_open_recovery",
    "test_circuit_breaker_half_open_back_to_open_on_failure",
    "test_guarded_decorator_returns_default_on_open",
    "test_guarded_decorator_passes_through_on_closed",
    "test_guarded_decorator_reraises_on_open",
    "test_circuit_breaker_thread_safety",
]

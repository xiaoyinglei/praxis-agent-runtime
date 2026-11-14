"""唯一模型重试实现与 model_attempt 事实(D7/§5.9;R-T3-1 分层)。

两层职责在此分别钉住:

- ``protocol/retry.py`` = 纯数据与纯策略(无 SDK import):``FailureKind``/
  ``ModelFailure``/``RetryPolicy``/``clock_floor``/``ModelUnavailableError``;
- ``adapters/models/retry.py`` = 唯一机制:``classify_failure`` 移植旧分类规则
  (只按 SDK 类型/HTTP status/code,绝不含 provider body),``stream_with_retry``
  是唯一退避实现,失败尝试追加 ``AttemptRecord``、成功不追加(成功记录由
  core 在 turn 归属下补写,见 ``core/turn.py``)。

测试里不真 sleep:``sleep`` 是注入参数,``_Clock.sleep`` 只推进假时钟。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path

import httpx
import openai
import pytest

from praxis.adapters.models.retry import (
    classify_failure,
    make_retry_stream,
    stream_with_retry,
)
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.composition import build_session
from praxis.core.turn import (
    TurnDeps,
    TurnState,
    _attempt_records_from_items,
    run_turn,
)
from praxis.core.statemachine import TurnStatus
from praxis.protocol.events import ModelOutputDelta, ModelRetried, ModelResponseReceived
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import (
    ModelEvent,
    ModelResponse,
    ModelStreamFinished,
    ModelTextDelta,
    Usage,
    collect_events,
)
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.retry import (
    AttemptRecord,
    ModelFailure,
    ModelUnavailableError,
    NoRetryPlan,
    RetryPlan,
    RetryPolicy,
    clock_floor,
)
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn

# ---------------------------------------------------------------------------
# 脚手架:假时钟 / provider 异常 / 流工厂
# ---------------------------------------------------------------------------


_Factory = Callable[[], AsyncIterator[ModelEvent]]


class _Clock:
    """可注入时钟;sleep 推进它而不是真等待(测试禁真 sleep)。"""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        assert delay >= 0
        self.now += delay


def _status_error(
    status: int,
    *,
    headers: dict[str, str] | None = None,
    body: object | None = None,
) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://unit.test/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return openai.APIStatusError("provider body must never leak", response=response, body=body)


async def _ok_stream(text: str = "ok") -> AsyncIterator[ModelEvent]:
    yield ModelTextDelta(text)
    yield ModelStreamFinished(Usage(1, 1), "stop")


def _failing_factory(
    exc: BaseException,
    *,
    fail_calls: int = 1,
    prefix: Sequence[ModelEvent] = (),
) -> tuple[_Factory, list[int]]:
    """前 ``fail_calls`` 次调用抛 ``exc``,之后返回一条成功流。"""
    calls: list[int] = []

    async def failing() -> AsyncIterator[ModelEvent]:
        for event in prefix:
            yield event
        raise exc

    def factory() -> AsyncIterator[ModelEvent]:
        calls.append(1)
        if len(calls) <= fail_calls:
            return failing()
        return _ok_stream()

    return factory, calls


# ---------------------------------------------------------------------------
# classify_failure:只按类型/status/code,message 固定,diagnostic 有界
# ---------------------------------------------------------------------------


def test_rate_limited_retry_after_header_wins() -> None:
    failure = classify_failure(
        _status_error(429, headers={"retry-after": "7"}), now=1000.0
    )
    assert failure.kind == "rate_limited" and failure.retryable
    assert failure.retry_at == pytest.approx(1007.0, abs=0.5)


def test_retry_after_ms_precedes_retry_after() -> None:
    failure = classify_failure(
        _status_error(429, headers={"retry-after-ms": "2500", "retry-after": "7"}),
        now=1000.0,
    )
    assert failure.retry_at == pytest.approx(1002.5, abs=0.01)


def test_output_started_is_never_retryable() -> None:
    failure = classify_failure(_status_error(503), output_started=True, now=0.0)
    assert failure.kind == "server_error" and failure.phase == "stream"
    assert failure.retryable is False
    decision = RetryPolicy().plan(
        retry_count=0,
        first_dispatched_at=0,
        failure=ModelFailure("server_error", "x", True, None, True, "stream"),
        now=0,
        random_fraction=0.5,
    )
    assert isinstance(decision, NoRetryPlan)


def test_full_jitter_and_limits() -> None:
    ok = ModelFailure("server_error", "x", True, None, False, "request")
    plan = RetryPolicy().plan(
        retry_count=1, first_dispatched_at=0, failure=ok, now=0, random_fraction=0.5
    )
    assert isinstance(plan, RetryPlan)
    assert plan.delay_ms == int(min(20_000, 500 * 2) * 0.5)
    assert plan.retry_at == pytest.approx(0.5)
    exhausted = RetryPolicy().plan(
        retry_count=2, first_dispatched_at=0, failure=ok, now=0, random_fraction=0.5
    )
    assert isinstance(exhausted, NoRetryPlan)
    assert exhausted.reason_code == "model_retry_exhausted"
    deadline = RetryPolicy().plan(
        retry_count=0, first_dispatched_at=0, failure=ok, now=121.0, random_fraction=0.5
    )
    assert isinstance(deadline, NoRetryPlan)
    assert deadline.reason_code == "model_retry_deadline_exceeded"


def test_retryable_set_is_only_rate_limited_server_error_transport_not_sent() -> None:
    policy = RetryPolicy()
    for kind in ("authentication", "permission", "invalid_request", "context_overflow",
                 "quota_exhausted", "transport_unknown", "protocol_error"):
        failure = ModelFailure(kind, "x", True, None, False, "request")
        decision = policy.plan(
            retry_count=0, first_dispatched_at=0, failure=failure, now=0, random_fraction=0.5
        )
        assert isinstance(decision, NoRetryPlan) and decision.reason_code == kind


def test_status_code_classification_and_x_should_retry() -> None:
    assert classify_failure(_status_error(401)).kind == "authentication"
    assert classify_failure(_status_error(403)).kind == "permission"
    assert classify_failure(_status_error(404)).kind == "invalid_request"
    assert classify_failure(
        _status_error(400, body={"code": "context_length_exceeded"})
    ).kind == "context_overflow"
    assert classify_failure(
        _status_error(429, body={"code": "insufficient_quota"})
    ).kind == "quota_exhausted"
    assert classify_failure(_status_error(501)).retryable is False
    assert classify_failure(_status_error(502)).retryable is True
    vetoed = classify_failure(_status_error(429, headers={"x-should-retry": "false"}))
    assert vetoed.kind == "rate_limited" and vetoed.retryable is False


def test_connect_error_in_cause_chain_is_transport_not_sent() -> None:
    request = httpx.Request("POST", "https://unit.test/v1/chat/completions")
    try:
        try:
            raise httpx.ConnectError("connection refused")
        except httpx.ConnectError as cause:
            raise openai.APIConnectionError(request=request) from cause
    except openai.APIConnectionError as exc:
        failure = classify_failure(exc)
    assert failure.kind == "transport_not_sent" and failure.retryable


def test_unknown_transport_failure_is_not_retryable() -> None:
    failure = classify_failure(RuntimeError("mystery"))
    assert failure.kind == "protocol_error" and failure.retryable is False
    assert failure.message == "Provider response violated the model protocol."
    assert failure.diagnostic == "mystery"


def test_provider_without_streaming_is_non_retryable_but_diagnostic_survives() -> None:
    """Task 2 的两条 ValueError 必须不重试,且原文不被固定文案彻底吞掉。"""
    for text in (
        "provider returned no stream events; it may not support streaming",
        "provider returned no choices for model 'm'",
    ):
        failure = classify_failure(ValueError(text))
        assert failure.kind == "protocol_error" and failure.retryable is False
        assert text not in failure.message  # 固定文案,不回显原文
        error = ModelUnavailableError(failure)
        assert text in str(error)  # 诊断字段让上层仍能看到根因


def test_provider_body_is_never_echoed_in_diagnostic() -> None:
    failure = classify_failure(_status_error(500, body={"error": {"message": "SECRET-BODY"}}))
    assert "SECRET-BODY" not in failure.message
    assert "SECRET-BODY" not in (failure.diagnostic or "")


# ---------------------------------------------------------------------------
# clock_floor:所有观测都是地板(跨重启复算)
# ---------------------------------------------------------------------------


def test_clock_backwards_stops_retrying() -> None:
    floor = clock_floor([AttemptRecord(0, "ok", None, 1000.0, False, "ok")], now=999.0)
    assert floor == 1000.0  # 调用方据此拒绝再次自动重试
    assert clock_floor([], now=5.0) == 5.0


def test_clock_floor_blocks_retry_after_provider_failure() -> None:
    """floor 来自历史 item;时钟早于 floor 时即使 503 也不重试。"""
    clock = _Clock(999.0)
    attempts = [AttemptRecord(0, "ok", None, 1000.0, False, "ok")]
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    factory, calls = _failing_factory(_status_error(503), fail_calls=10)

    async def drive() -> None:
        await collect_events(
            stream_with_retry(
                factory,
                policy=RetryPolicy(),
                clock=clock,
                sleep=sleep,
                attempts=attempts,
                floor=1000.0,
            )
        )

    with pytest.raises(ModelUnavailableError):
        asyncio.run(drive())
    assert calls == [1]
    assert sleeps == []
    assert len(attempts) == 2 and attempts[-1].outcome == "failed"


# ---------------------------------------------------------------------------
# stream_with_retry:唯一退避实现
# ---------------------------------------------------------------------------


def test_stream_with_retry_resumes_after_one_failure() -> None:
    clock = _Clock(0.0)
    attempts: list[AttemptRecord] = []
    factory, calls = _failing_factory(_status_error(429), fail_calls=1)

    async def drive() -> list[ModelEvent]:
        return await collect_events(
            stream_with_retry(
                factory,
                policy=RetryPolicy(),
                clock=clock,
                sleep=clock.sleep,
                attempts=attempts,
                floor=0.0,
            )
        )

    events = asyncio.run(drive())

    assert [type(event) for event in events] == [ModelTextDelta, ModelStreamFinished]
    assert calls == [1, 1]
    assert len(attempts) == 1 and attempts[0].kind == "rate_limited"
    assert attempts[0].outcome == "failed" and attempts[0].retry_at is not None
    assert attempts[0].retry_at_source == "backoff"


def test_retry_after_header_becomes_retry_at_source() -> None:
    clock = _Clock(1000.0)
    attempts: list[AttemptRecord] = []
    factory, _calls = _failing_factory(
        _status_error(429, headers={"retry-after": "7"}), fail_calls=1
    )

    async def drive() -> None:
        await collect_events(
            stream_with_retry(
                factory,
                policy=RetryPolicy(),
                clock=clock,
                sleep=clock.sleep,
                attempts=attempts,
                floor=0.0,
            )
        )

    asyncio.run(drive())
    assert attempts[0].retry_at_source == "retry_after"
    assert attempts[0].retry_at == pytest.approx(1007.0)


def test_stream_with_retry_exhausts_and_records_every_attempt() -> None:
    clock = _Clock(0.0)
    attempts: list[AttemptRecord] = []
    factory, calls = _failing_factory(_status_error(503), fail_calls=10)

    async def drive() -> None:
        await collect_events(
            stream_with_retry(
                factory,
                policy=RetryPolicy(),
                clock=clock,
                sleep=clock.sleep,
                attempts=attempts,
                floor=0.0,
            )
        )

    with pytest.raises(ModelUnavailableError) as excinfo:
        asyncio.run(drive())

    error = excinfo.value
    assert error.reason_code == "model_retry_exhausted"
    assert error.failure.kind == "server_error"
    assert error.failure.message == "Provider server failed."
    assert calls == [1, 1, 1]  # max_auto_retries + 1
    assert len(attempts) == 3
    assert all(record.outcome == "failed" for record in attempts)
    assert [record.retry_at is not None for record in attempts] == [True, True, False]


def test_output_started_in_stream_is_never_retried() -> None:
    clock = _Clock(0.0)
    attempts: list[AttemptRecord] = []
    factory, calls = _failing_factory(
        _status_error(500), fail_calls=1, prefix=(ModelTextDelta("partial"),)
    )

    async def drive() -> None:
        await collect_events(
            stream_with_retry(
                factory,
                policy=RetryPolicy(),
                clock=clock,
                sleep=clock.sleep,
                attempts=attempts,
                floor=0.0,
            )
        )

    with pytest.raises(ModelUnavailableError):
        asyncio.run(drive())
    assert calls == [1]
    assert len(attempts) == 1
    assert attempts[0].output_started is True and attempts[0].retry_at is None


def test_deadline_denial_raises_with_reason_code() -> None:
    clock = _Clock(0.0)
    attempts: list[AttemptRecord] = []
    calls: list[int] = []

    async def failing_after_deadline() -> AsyncIterator[ModelEvent]:
        clock.now = 121.0  # 首次派发在 0,失败观察已越过 120s deadline
        raise _status_error(503)
        yield  # pragma: no cover — 让函数成为 async generator

    def factory() -> AsyncIterator[ModelEvent]:
        calls.append(1)
        return failing_after_deadline()

    async def drive() -> None:
        await collect_events(
            stream_with_retry(
                factory,
                policy=RetryPolicy(),
                clock=clock,
                sleep=clock.sleep,
                attempts=attempts,
                floor=0.0,
            )
        )

    with pytest.raises(ModelUnavailableError) as excinfo:
        asyncio.run(drive())
    assert excinfo.value.reason_code == "model_retry_deadline_exceeded"
    assert calls == [1]
    assert attempts[0].retry_at is None


def test_make_retry_stream_binds_three_argument_contract() -> None:
    clock = _Clock(0.0)
    attempts: list[AttemptRecord] = []
    factory, calls = _failing_factory(_status_error(503), fail_calls=1)
    retry_stream = make_retry_stream(policy=RetryPolicy(), clock=clock, sleep=clock.sleep)

    events = asyncio.run(retry_stream(factory, attempts, 0.0))

    assert isinstance(events, list) and isinstance(events[0], ModelTextDelta)
    assert calls == [1, 1] and len(attempts) == 1


def test_retry_policy_defaults_are_the_brief_numbers() -> None:
    policy = RetryPolicy()
    assert (
        policy.max_auto_retries,
        policy.max_elapsed_ms,
        policy.base_delay_ms,
        policy.max_delay_ms,
    ) == (2, 120_000, 500, 20_000)


# ---------------------------------------------------------------------------
# core/turn.py:尝试事实的读写(跨重启 floor 的来源)
# ---------------------------------------------------------------------------

_THREAD = "retry-thread"
_TURN = "retry-turn"


def _turn(store: Store) -> None:
    store.ensure_thread(_THREAD)
    create_turn(store, turn_id=_TURN, thread_id=_THREAD, binding={})
    store.append_items(
        _THREAD, _TURN, [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "hi"})]
    )


def _deps(
    workspace: Path,
    store: Store,
    bus: object,
    retry_stream: object,
    *,
    adapter: object | None = None,
) -> TurnDeps:
    return TurnDeps(
        store=store,
        bus=bus,  # type: ignore[arg-type]
        adapter=adapter if adapter is not None else ScriptedAdapter([]),  # type: ignore[arg-type]
        tools={},
        permissions=PermissionContext(
            workspace=str(workspace), policy=PermissionPolicy(), mode="on-request"
        ),
        workspace=workspace,
        system_prompt="retry test",
        retry_stream=retry_stream,  # type: ignore[arg-type]
    )


def _drain(sub: object) -> list[object]:
    queue = sub.queue  # type: ignore[attr-defined]
    events: list[object] = []
    while True:
        try:
            events.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            return events


def test_run_turn_writes_model_attempt_items_for_failure_and_success(
    store: Store, bus: object, workspace: Path
) -> None:
    """失败尝试与成功尝试各一条 item;retry 与 delta 事件按序 emit。"""
    _turn(store)

    async def retry_stream(
        stream_factory: object, attempts: list[AttemptRecord], floor: float
    ) -> list[ModelEvent]:
        attempts.append(
            AttemptRecord(
                attempt=len(attempts),
                kind="rate_limited",
                retry_at=1.0,
                dispatched_at=0.0,
                output_started=False,
                outcome="failed",
                retry_at_source="backoff",
            )
        )
        return [ModelTextDelta("hi"), ModelStreamFinished(Usage(1, 1), "stop")]

    deps = _deps(workspace, store, bus, retry_stream)
    sub = bus.subscribe()  # type: ignore[attr-defined]

    result = asyncio.run(run_turn(deps, TurnState(_TURN, _THREAD)))

    assert result.status is TurnStatus.COMPLETED and result.output == "hi"
    attempts = [item for item in store.read_turn_items(_TURN) if item.kind == ItemKind.MODEL_ATTEMPT]
    assert [(a.payload["attempt"], a.payload["kind"], a.payload["outcome"]) for a in attempts] == [
        (0, "rate_limited", "failed"),
        (1, "ok", "ok"),
    ]
    assert set(attempts[0].payload) == {
        "attempt",
        "kind",
        "retry_at",
        "dispatched_at",
        "output_started",
        "outcome",
        "retry_at_source",
    }
    events = _drain(sub)
    assert any(isinstance(event, ModelRetried) for event in events)
    deltas = [event for event in events if isinstance(event, ModelOutputDelta)]
    assert [(d.kind, d.text) for d in deltas] == [("text", "hi")]
    assert any(isinstance(event, ModelResponseReceived) for event in events)


def test_run_turn_persists_final_failed_attempt_and_reason(
    store: Store, bus: object, workspace: Path
) -> None:
    """最终失败仍写 model_attempt;固定文案 + 诊断摘要一起进入 turn reason。"""
    _turn(store)

    async def retry_stream(
        stream_factory: object, attempts: list[AttemptRecord], floor: float
    ) -> list[ModelEvent]:
        attempts.append(
            AttemptRecord(
                attempt=len(attempts),
                kind="server_error",
                retry_at=None,
                dispatched_at=0.0,
                output_started=False,
                outcome="failed",
            )
        )
        raise ModelUnavailableError(
            ModelFailure(
                "server_error",
                "",
                retryable=False,
                retry_at=None,
                output_started=False,
                phase="request",
                diagnostic="provider 503",
            ),
            reason_code="model_retry_exhausted",
        )

    deps = _deps(workspace, store, bus, retry_stream)

    result = asyncio.run(run_turn(deps, TurnState(_TURN, _THREAD)))

    assert result.status is TurnStatus.FAILED
    assert result.reason is not None and result.reason.startswith("ModelUnavailableError:")
    assert "Provider server failed." in result.reason and "provider 503" in result.reason
    attempts = [item for item in store.read_turn_items(_TURN) if item.kind == ItemKind.MODEL_ATTEMPT]
    assert len(attempts) == 1 and attempts[0].payload["outcome"] == "failed"


def test_attempt_records_from_items_is_fail_open_on_malformed_payload() -> None:
    """畸形 model_attempt payload 一律跳过,绝不炸读路径。"""
    base = {
        "attempt": 0,
        "kind": "server_error",
        "retry_at": None,
        "dispatched_at": 10.0,
        "output_started": False,
        "outcome": "failed",
        "retry_at_source": None,
    }
    items = [
        Item("i1", "t", "turn", 1, ItemKind.MODEL_ATTEMPT, base, 0.0),
        Item("i2", "t", "turn", 2, ItemKind.MODEL_ATTEMPT, {"attempt": "zero"}, 0.0),
        Item("i3", "t", "turn", 3, ItemKind.MODEL_ATTEMPT, {**base, "outcome": "maybe"}, 0.0),
        Item("i4", "t", "turn", 4, ItemKind.MODEL_ATTEMPT, {**base, "dispatched_at": True}, 0.0),
        Item("i5", "t", "turn", 5, ItemKind.MODEL_RESPONSE, {"content": "x"}, 0.0),
        Item(
            "i6", "t", "turn", 6, ItemKind.MODEL_ATTEMPT,
            {**base, "attempt": 1, "kind": "ok", "outcome": "ok", "retry_at": 12.5},
            0.0,
        ),
    ]

    records = _attempt_records_from_items(items)

    assert [(r.attempt, r.kind, r.outcome, r.retry_at) for r in records] == [
        (0, "server_error", "failed", None),
        (1, "ok", "ok", 12.5),
    ]


# ---------------------------------------------------------------------------
# 是否写 attempt item 由"机制是否注入"决定,而不是分支特判(R-T3-3 附加钉住)
# ---------------------------------------------------------------------------


def _scripted_response(content: str = "ok") -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=(),
        usage=Usage(1, 1),
        finish_reason="stop",
    )


def test_composition_injected_retry_records_exactly_one_ok_attempt(workspace: Path) -> None:
    """生产组合根注入真实机制:一次无故障调用恰好一条 outcome=ok 的 item。"""
    session = build_session(workspace, ScriptedAdapter([_scripted_response("done")]))
    try:
        result = asyncio.run(session.submit("hi"))
        assert result.status is TurnStatus.COMPLETED and result.output == "done"
        attempts = [item for item in session.items() if item.kind == ItemKind.MODEL_ATTEMPT]
        assert len(attempts) == 1
        assert attempts[0].payload["attempt"] == 0
        assert attempts[0].payload["outcome"] == "ok"
        assert attempts[0].payload["kind"] == "ok"
    finally:
        session.close()


def test_retry_stream_none_writes_no_attempt_items(
    store: Store, bus: object, workspace: Path
) -> None:
    """scripted/单测路径(无机制)不产生尝试账,且照常完成。"""
    _turn(store)
    deps = _deps(
        workspace, store, bus, None, adapter=ScriptedAdapter([_scripted_response("fine")])
    )

    result = asyncio.run(run_turn(deps, TurnState(_TURN, _THREAD)))

    assert result.status is TurnStatus.COMPLETED and result.output == "fine"
    assert [i for i in store.read_turn_items(_TURN) if i.kind == ItemKind.MODEL_ATTEMPT] == []
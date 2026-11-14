"""协议层重试纯数据与纯策略(设计 §5.9/D7;R-T3-1 分层)。

本模块零第三方依赖——**不得 import openai/httpx**:SDK 异常分类与重试机制
都在 ``adapters/models/retry.py``。core 只能认识这些结构签名,经 composition
注入真实 ``RetryingStream``(与 ``SandboxCompiler`` 同一先例)。

为什么这样切:

- ``ModelFailure.message`` 恒为固定文案,provider body 永不进入持久化事实;
  原文摘要放 ``diagnostic``(有界、白名单字符),只供上层诊断,不参与重试判定。
- ``AttemptRecord`` 是 ``model_attempt`` item 的内存形状;真正的 append-only
  写点在 core(turn 归属),本模块不碰 store。
- ``RetryPolicy.plan`` 是纯授权:给定失败、已用重试数与随机分数,返回值只有
  两种——“要重试”的 ``RetryPlan`` 或“不重试”的 ``NoRetryPlan``(R-T3-3)。
- ``clock_floor`` 把历史尝试的派发时刻与当前观察时刻都当作地板:任何一条
  观测都足以证明时钟曾经到达过那里,回拨即拒绝自动重试(跨进程重启成立)。
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from praxis.protocol.models import ModelEvent

FailureKind = Literal[
    "authentication",
    "permission",
    "invalid_request",
    "context_overflow",
    "rate_limited",
    "quota_exhausted",
    "server_error",
    "transport_not_sent",
    "transport_unknown",
    "protocol_error",
]
FailurePhase = Literal["request", "stream"]
AttemptOutcome = Literal["ok", "failed"]
RetryAtSource = Literal["retry_after", "backoff"]

# 固定文案表:provider 原始消息绝不进入 message(只进 diagnostic)。
_MESSAGES: dict[FailureKind, str] = {
    "authentication": "Provider authentication failed.",
    "permission": "Provider permission denied.",
    "invalid_request": "Provider rejected the model request.",
    "context_overflow": "Provider context window exceeded.",
    "rate_limited": "Provider rate limit reached.",
    "quota_exhausted": "Provider quota exhausted.",
    "server_error": "Provider server failed.",
    "transport_not_sent": "Provider connection failed before the request was sent.",
    "transport_unknown": "Provider transport failed; request outcome is unknown.",
    "protocol_error": "Provider response violated the model protocol.",
}

_RETRYABLE_KINDS: frozenset[str] = frozenset(
    {"rate_limited", "server_error", "transport_not_sent"}
)


def _is_finite_number(value: object) -> bool:
    if type(value) is int:
        return math.isfinite(float(value))
    if type(value) is float:
        return math.isfinite(value)
    return False


@dataclass(frozen=True, slots=True)
class ModelFailure:
    """一次 provider 尝试的安全失败事实。

    ``message`` 若为空则由 ``kind`` 填固定文案;调用方(仅 classification)
    不应传入 provider 文本。``diagnostic`` 承载有界原文摘要,绝不含 body。
    ``output_started=True`` 会强制 ``retryable=False`` 且 ``phase="stream"``
    ——已经产出过增量的请求重放会在 UI 上留下回滚不掉的重影。
    """

    kind: FailureKind
    message: str
    retryable: bool
    retry_at: float | None
    output_started: bool
    phase: FailurePhase
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _MESSAGES:
            raise ValueError(f"unsupported model failure kind: {self.kind!r}")
        if self.phase not in ("request", "stream"):
            raise ValueError(f"unsupported model failure phase: {self.phase!r}")
        if type(self.retryable) is not bool or type(self.output_started) is not bool:
            raise TypeError("model failure flags must be booleans")
        if self.retry_at is not None and not _is_finite_number(self.retry_at):
            raise ValueError("retry_at must be a finite epoch time or None")
        if not self.message:
            object.__setattr__(self, "message", _MESSAGES[self.kind])
        if self.output_started:
            object.__setattr__(self, "retryable", False)
            object.__setattr__(self, "phase", "stream")


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """一次模型尝试的事实(与 ``model_attempt`` item payload 一一对应)。

    ``attempt`` 是本 turn 内单调尝试序号(跨重启由 item 续号);``retry_at``
    只在该次失败被允许重试时给出;``retry_at_source`` 标记它来自 provider
    头(``retry_after``)还是本地退避(``backoff``)。成功记录的 retry_at 恒
    为 None,由 core 在 turn 归属下补写。
    """

    attempt: int
    kind: str
    retry_at: float | None
    dispatched_at: float
    output_started: bool
    outcome: AttemptOutcome
    retry_at_source: RetryAtSource | None = None

    def __post_init__(self) -> None:
        if type(self.attempt) is not int or self.attempt < 0:
            raise ValueError(f"attempt must be a non-negative int, got {self.attempt!r}")
        if not self.kind:
            raise ValueError("attempt kind must be non-empty")
        if self.outcome not in ("ok", "failed"):
            raise ValueError(f"unsupported attempt outcome: {self.outcome!r}")
        if not _is_finite_number(self.dispatched_at):
            raise ValueError("dispatched_at must be a finite epoch time")
        if self.retry_at is not None and not _is_finite_number(self.retry_at):
            raise ValueError("retry_at must be a finite epoch time or None")
        if type(self.output_started) is not bool:
            raise TypeError("output_started must be a boolean")
        if self.retry_at_source not in (None, "retry_after", "backoff"):
            raise ValueError(f"unsupported retry_at_source: {self.retry_at_source!r}")


@dataclass(frozen=True, slots=True)
class RetryPlan:
    """一次允许重试的调度计划(唯一表示"要重试")。"""

    retry_at: float
    deadline_at: float
    delay_ms: int

    def __post_init__(self) -> None:
        if not _is_finite_number(self.retry_at) or not _is_finite_number(self.deadline_at):
            raise ValueError("retry plan times must be finite")
        if type(self.delay_ms) is not int or self.delay_ms < 0:
            raise ValueError("delay_ms must be a non-negative int")


@dataclass(frozen=True, slots=True)
class NoRetryPlan:
    """不重试的唯一表示与机器可读原因(R-T3-3:不再有 None/allowed 两套口径)。"""

    reason_code: str

    def __post_init__(self) -> None:
        if not self.reason_code:
            raise ValueError("NoRetryPlan.reason_code must be non-empty")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """冻结的自动重试授权参数(D7 逐字数值;不建 lease/账本族)。"""

    max_auto_retries: int = 2
    max_elapsed_ms: int = 120_000
    base_delay_ms: int = 500
    max_delay_ms: int = 20_000

    def __post_init__(self) -> None:
        for name, value in (
            ("max_auto_retries", self.max_auto_retries),
            ("max_elapsed_ms", self.max_elapsed_ms),
            ("base_delay_ms", self.base_delay_ms),
            ("max_delay_ms", self.max_delay_ms),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_elapsed_ms < 1 or self.max_delay_ms < self.base_delay_ms:
            raise ValueError("invalid model retry delay bounds")

    def plan(
        self,
        *,
        retry_count: int,
        first_dispatched_at: float,
        failure: ModelFailure,
        now: float,
        random_fraction: float,
    ) -> RetryPlan | NoRetryPlan:
        """返回 full-jitter 重试计划;``NoRetryPlan`` 是"不重试"的唯一表示。

        方法名必须是 ``plan`` 而非 ``decide``(R-T3-2):CI 把权限决策入口
        钉死为全仓唯一、只在 ``core/permission.py``,策略对象不开路径豁免。
        """
        if not all(math.isfinite(value) for value in (now, first_dispatched_at, random_fraction)):
            raise ValueError("retry times and jitter must be finite")
        if not 0 <= random_fraction <= 1:
            raise ValueError("random_fraction must lie between zero and one")
        if type(retry_count) is not int or retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        if now < first_dispatched_at:
            return NoRetryPlan("model_retry_clock_invalid")
        if retry_count >= self.max_auto_retries:
            return NoRetryPlan("model_retry_exhausted")
        if (
            failure.output_started
            or not failure.retryable
            or failure.kind not in _RETRYABLE_KINDS
        ):
            return NoRetryPlan(failure.kind)
        deadline_at = first_dispatched_at + self.max_elapsed_ms / 1000
        ceiling = min(self.max_delay_ms, self.base_delay_ms * 2 ** min(retry_count, 63))
        delay_ms = int(ceiling * random_fraction)
        retry_at = max(now + delay_ms / 1000, failure.retry_at if failure.retry_at is not None else now)
        if now > deadline_at or retry_at > deadline_at:
            return NoRetryPlan("model_retry_deadline_exceeded")
        return RetryPlan(retry_at=retry_at, deadline_at=deadline_at, delay_ms=delay_ms)


def clock_floor(attempt_records: Sequence[AttemptRecord], now: float) -> float:
    """所有已观测时刻都是地板:历史派发时刻 + 本次观察时刻取最大。

    调用方用 ``now < floor`` 判定时钟回拨并拒绝自动重试;跨重启时历史
    时刻来自 ``model_attempt`` item,所以判定不依赖进程内状态。
    """
    if not math.isfinite(now):
        raise ValueError("clock observation must be finite")
    return max([now, *(record.dispatched_at for record in attempt_records)])


class ModelUnavailableError(RuntimeError):  # noqa: N818 — 名称由计划契约固定
    """模型调用在重试耗尽/不可重试后仍失败;``failure`` 是安全事实。

    ``str(error)`` 附带固定文案与有界诊断摘要(例如 Task 2 的
    "provider returned no stream events"),让 turn 的 reason 仍能指向根因,
    同时绝不回显 provider body。
    """

    def __init__(self, failure: ModelFailure, *, reason_code: str | None = None) -> None:
        self.failure = failure
        self.reason_code = reason_code
        detail = failure.message
        if failure.diagnostic:
            detail = f"{detail} ({failure.diagnostic})"
        super().__init__(detail)


# core 与 adapters 共用的唯一结构签名(定义在最低层,避免任一侧 import 另一侧)。
type StreamFactory = Callable[[], AsyncIterator[ModelEvent]]
type RetryingStream = Callable[[StreamFactory, list[AttemptRecord], float], Awaitable[list[ModelEvent]]]

__all__ = [
    "AttemptOutcome",
    "AttemptRecord",
    "FailureKind",
    "FailurePhase",
    "ModelFailure",
    "ModelUnavailableError",
    "NoRetryPlan",
    "RetryAtSource",
    "RetryPlan",
    "RetryPolicy",
    "RetryingStream",
    "StreamFactory",
    "clock_floor",
]
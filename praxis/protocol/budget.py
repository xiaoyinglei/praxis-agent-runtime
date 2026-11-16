"""协议层预算数据:Limits、份额公式与取消/子分配 payload 键(设计 §5.5/§5.13/§5.1)。

``Limits`` 是预算树(阶段4 T2)与子代理(T5)共用的唯一限额形状:``None`` 表示
该维不 enforce(无限),不是"未知";零是"有界且已耗尽"。

``limits_merge_grant`` 是设计 §5.5 份额公式的**唯一表达**:spawn 子代理时按维
独立取 ``min(requested or remaining, floor(remaining × fraction))``。旧 B2 病灶
(``agent_runtime/harness/budget_ops.py:88`` 在 requested 为 None 时直接
``granted = remaining``)让单个子代理吸干父预算;这里份额上限是算法而不是调用方
特判,子代理结构性地拿不到超过父剩余的 ``fraction``。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Final

from praxis.protocol.json import JsonValue


@dataclass(frozen=True, slots=True)
class Limits:
    """一棵预算节点的分维限额;每维 ``None`` = 不 enforce(无限)。

    五维覆盖模型调用与子代理共用的全部资源面:tokens/cost 来自 §5.5,
    steps 是轮数上限,web/browser 请求数是阶段3 已有配额的数据形态。
    """

    tokens: int | None = None
    cost_micros: int | None = None
    steps: int | None = None
    web_requests: int | None = None
    browser_requests: int | None = None

    # 全 None 的共享实例(冻结,可安全跨模块传递)。
    UNLIMITED: ClassVar[Limits]


Limits.UNLIMITED = Limits()
UNLIMITED: Final[Limits] = Limits.UNLIMITED

_LIMIT_DIMENSIONS: Final[tuple[str, ...]] = (
    "tokens",
    "cost_micros",
    "steps",
    "web_requests",
    "browser_requests",
)

# 子分配 spawn 的默认份额上限(设计 §5.5:配置数据,不是特判)。
MAX_CHILD_FRACTION: Final[float] = 0.5

# 子代理并发/深度/总量默认上限;对照旧 agent_runtime/harness/subagents.py:25
# 的 DEFAULT_LIMITS,数值不变,形状改为不可变映射。
SUBAGENT_LIMITS_DEFAULT: Final[Mapping[str, int]] = MappingProxyType(
    {"max_concurrent": 4, "max_depth": 2, "max_total": 16}
)

# child_allocation item 的 payload 键(写端命名约定;读端仍 fail-open)。
# ``grant``/``requested`` 的形状是 limits_payload 的产物。
ALLOCATION_PAYLOAD_KEYS: Final[tuple[str, ...]] = (
    "allocation_id",
    "parent_turn_id",
    "child_thread_id",
    "child_root_turn_id",
    "workspace_mode",
    "workspace",
    "grant",
    "requested",
    "source",
)
ALLOCATION_SOURCES: Final[frozenset[str]] = frozenset({"model", "memory"})

# budget_extension item 的 payload 键:delta 是 Limits 形状的增量。
EXTENSION_KEYS: Final[tuple[str, ...]] = ("limits_delta",)

# cancel_request item 的 payload 键:source=user 为本机取消,parent 为父代理级联。
CANCEL_KEYS: Final[tuple[str, ...]] = ("source", "parent_turn_id")
CANCEL_SOURCES: Final[frozenset[str]] = frozenset({"user", "parent"})


def _merge_dimension(
    remaining: int | None,
    requested: int | None,
    fraction: float,
) -> int | None:
    """单维份额:``min(requested or remaining, floor(remaining × fraction))``。"""
    if remaining is None:
        # 父无限:该维没有份额上限,透传请求(None 仍表示无限)。
        return requested
    share = math.floor(remaining * fraction)
    if share <= 0:
        # 向下取整后无份额可授:grant=0(不是 None)。父该维有界,调用方
        # admission 据此拒;本函数只做算术,不抛错。
        return 0
    return min(remaining if requested is None else requested, share)


def limits_merge_grant(
    requested: Limits | None,
    parent_remaining: Limits,
    fraction: float = MAX_CHILD_FRACTION,
) -> Limits:
    """按设计 §5.5 计算子分配 grant——份额公式的唯一表达。

    每维独立:
    - 父该维无限(None)→ grant 该维 = requested(可为 None);
    - 父该维有界 → ``min(requested or remaining, floor(remaining × fraction))``,
      requested 超份额时**截断而不报错**(拒绝是调用方 admission 的事);
    - 份额向下取整后 ≤ 0 → grant 该维 = 0(有界且耗尽,不是 None)。
    """
    want = UNLIMITED if requested is None else requested
    return Limits(
        tokens=_merge_dimension(parent_remaining.tokens, want.tokens, fraction),
        cost_micros=_merge_dimension(parent_remaining.cost_micros, want.cost_micros, fraction),
        steps=_merge_dimension(parent_remaining.steps, want.steps, fraction),
        web_requests=_merge_dimension(parent_remaining.web_requests, want.web_requests, fraction),
        browser_requests=_merge_dimension(parent_remaining.browser_requests, want.browser_requests, fraction),
    )


def limits_payload(limits: Limits) -> dict[str, JsonValue]:
    """Limits → JSON 兼容 dict;只落已设置的维度(缺键读回即 None)。"""
    payload: dict[str, JsonValue] = {}
    for name in _LIMIT_DIMENSIONS:
        value = getattr(limits, name)
        if value is not None:
            payload[name] = value
    return payload


def _limit_dimension(payload: Mapping[str, JsonValue], name: str) -> int | None:
    raw = payload.get(name)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"Limits.{name} must be an integer or null, got {type(raw).__name__}")
    return raw


def limits_from_payload(payload: Mapping[str, JsonValue]) -> Limits:
    """payload dict → Limits;缺键为 None(对应 limits_payload 的省略写端)。"""
    return Limits(
        tokens=_limit_dimension(payload, "tokens"),
        cost_micros=_limit_dimension(payload, "cost_micros"),
        steps=_limit_dimension(payload, "steps"),
        web_requests=_limit_dimension(payload, "web_requests"),
        browser_requests=_limit_dimension(payload, "browser_requests"),
    )


# ---------------------------------------------------------------------------
# payload 拼装助手(T16 ㉓:child_allocation/cancel_request 唯一拼装点)
#
# 键常量在本模块(T1 落点),拼装助手与键同模块内聚——写端命名约定与
# 序列化形状一处持有;core.agents/core.session 只调用,不再裸拼 dict。
# CI 规则 26(计划㉓)钉住特征键集 dict 字面量不得在别处出现。
# ---------------------------------------------------------------------------


def allocation_payload(
    *,
    allocation_id: str,
    parent_turn_id: str,
    child_thread_id: str,
    child_root_turn_id: str,
    workspace_mode: str,
    workspace: str,
    grant: Limits,
    requested: Limits | None,
    source: str,
) -> dict[str, JsonValue]:
    """child_allocation item payload 的唯一拼装点(键序 = ALLOCATION_PAYLOAD_KEYS)。

    ``grant``/``requested`` 经 limits_payload 序列化(requested=None → {},
    与读端 _parse_allocation 的 fail-open 口径配对);source 边界拒绝未知值
    (写端命名约定,读端仍不校验)。
    """
    if source not in ALLOCATION_SOURCES:
        raise ValueError(f"unknown allocation source: {source!r} (expected model/memory)")
    return {
        "allocation_id": allocation_id,
        "parent_turn_id": parent_turn_id,
        "child_thread_id": child_thread_id,
        "child_root_turn_id": child_root_turn_id,
        "workspace_mode": workspace_mode,
        "workspace": workspace,
        "grant": limits_payload(grant),
        "requested": limits_payload(requested) if requested is not None else {},
        "source": source,
    }


def cancel_request_payload(*, source: str, parent_turn_id: str | None) -> dict[str, JsonValue]:
    """cancel_request item payload 的唯一拼装点(键序 = CANCEL_KEYS)。

    source=user 为本机取消(parent_turn_id=None),parent 为父代理级联。
    """
    if source not in CANCEL_SOURCES:
        raise ValueError(f"unknown cancel source: {source!r} (expected user/parent)")
    return {"source": source, "parent_turn_id": parent_turn_id}

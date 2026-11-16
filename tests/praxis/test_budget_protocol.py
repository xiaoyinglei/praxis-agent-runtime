"""protocol.budget:预算 Limits/份额公式/取消/子分配数据定义(设计 §5.5/§5.13/§5.1)。

本文件是阶段4 协议层的唯一验收面:
- 新 kind 注册与 payload 键常量(写端命名约定);
- 读端 fail-open:未知 kind 的 Item 仍可原样构造读出(B1);
- ``limits_merge_grant`` 份额公式穷举表——B2(子代理吸干预算)的回归钉;
- Limits 的 payload 序列化对 round-trip(R-PF-3,供 T3 消费)。
"""

from __future__ import annotations

import pytest

from praxis.protocol.budget import (
    ALLOCATION_PAYLOAD_KEYS,
    ALLOCATION_SOURCES,
    CANCEL_KEYS,
    CANCEL_SOURCES,
    EXTENSION_KEYS,
    MAX_CHILD_FRACTION,
    SUBAGENT_LIMITS_DEFAULT,
    UNLIMITED,
    Limits,
    limits_from_payload,
    limits_merge_grant,
    limits_payload,
)
from praxis.protocol.items import Item, ItemKind


# ---------------------------------------------------------------------------
# kind 注册与读端 fail-open
# ---------------------------------------------------------------------------


def test_three_new_kinds_are_registered_at_the_tail_of_the_enum() -> None:
    # 新 kind 追加在末尾,既有 12 个序不变(写端命名约定可增量扩展)。
    assert [kind.value for kind in ItemKind][-3:] == [
        "cancel_request",
        "budget_extension",
        "child_allocation",
    ]


@pytest.mark.parametrize(
    "kind",
    (ItemKind.CANCEL_REQUEST, ItemKind.BUDGET_EXTENSION, ItemKind.CHILD_ALLOCATION),
)
def test_item_accepts_new_budget_kinds_as_str(kind: ItemKind) -> None:
    item = Item(
        item_id="i1",
        thread_id="t",
        turn_id="turn-1",
        seq=1,
        kind=kind,
        payload={"source": "user"},
        created_at=0.0,
    )
    assert item.kind == kind.value


def test_unknown_kind_is_still_readable_fail_open() -> None:
    # B1 读端纪律:ItemKind 只是写端约定,未来 kind 必须能原样读出、不抛。
    item = Item(
        item_id="i1",
        thread_id="t",
        turn_id=None,
        seq=1,
        kind="kind_from_phase_9",
        payload={"x": 1},
        created_at=0.0,
    )
    assert item.kind == "kind_from_phase_9"


# ---------------------------------------------------------------------------
# payload 键常量组与数据默认值
# ---------------------------------------------------------------------------


def test_allocation_payload_keys() -> None:
    assert set(ALLOCATION_PAYLOAD_KEYS) == {
        "allocation_id",
        "parent_turn_id",
        "child_thread_id",
        "child_root_turn_id",
        "workspace_mode",
        "workspace",
        "grant",
        "requested",
        "source",
    }


def test_extension_and_cancel_payload_keys() -> None:
    assert set(EXTENSION_KEYS) == {"limits_delta"}
    assert set(CANCEL_KEYS) == {"source", "parent_turn_id"}


def test_source_value_sets() -> None:
    assert ALLOCATION_SOURCES == {"model", "memory"}
    assert CANCEL_SOURCES == {"user", "parent"}


def test_budget_constants_are_the_agreed_values() -> None:
    assert MAX_CHILD_FRACTION == 0.5
    assert dict(SUBAGENT_LIMITS_DEFAULT) == {
        "max_concurrent": 4,
        "max_depth": 2,
        "max_total": 16,
    }


def test_unlimited_is_all_none() -> None:
    assert UNLIMITED == Limits()
    assert (UNLIMITED.tokens, UNLIMITED.cost_micros, UNLIMITED.steps) == (None, None, None)
    assert (UNLIMITED.web_requests, UNLIMITED.browser_requests) == (None, None)


# ---------------------------------------------------------------------------
# limits_merge_grant:份额公式唯一表达(设计 §5.5)
#
# B2 病灶对照:旧代码 agent_runtime/harness/budget_ops.py:88 ``granted = remaining``
# ——requested 为 None 时子代理拿满父剩余,单个子代理吸干预算。
# 新公式:grant 该维 = min(requested or remaining, floor(remaining × fraction))。
# ---------------------------------------------------------------------------


def test_b2_regression_child_never_gets_full_parent_remaining() -> None:
    # B2 回归钉:父剩 1000 tokens、子未请求 → 默认份额 0.5 → 500,绝不等于 remaining。
    granted = limits_merge_grant(None, Limits(tokens=1000))
    assert granted.tokens == 500
    assert granted.tokens != 1000


@pytest.mark.parametrize(
    ("requested", "parent_remaining", "fraction", "expected"),
    [
        # 父无限维:透传 requested(可为 None)。
        (None, Limits(), MAX_CHILD_FRACTION, Limits()),
        (Limits(tokens=123, steps=4), Limits(), MAX_CHILD_FRACTION, Limits(tokens=123, steps=4)),
        # 请求未超份额:原样授予。
        (Limits(tokens=100), Limits(tokens=1000), MAX_CHILD_FRACTION, Limits(tokens=100)),
        # 请求超出份额:截断到份额,不抛错(拒绝是调用方 admission 的事)。
        (Limits(tokens=900), Limits(tokens=1000), MAX_CHILD_FRACTION, Limits(tokens=500)),
        (
            Limits(cost_micros=10_000, steps=99),
            Limits(cost_micros=400, steps=60),
            MAX_CHILD_FRACTION,
            Limits(cost_micros=200, steps=30),
        ),
        # requested 显式 0 不等于 None:授予 0。
        (Limits(tokens=0), Limits(tokens=1000), MAX_CHILD_FRACTION, Limits(tokens=0)),
        # share 向下取整:7 × 0.5 = 3.5 → 3。
        (None, Limits(tokens=7), MAX_CHILD_FRACTION, Limits(tokens=3)),
        # 向下取整后 ≤ 0 → 该维 grant = 0(不是 None)。
        (None, Limits(tokens=1), MAX_CHILD_FRACTION, Limits(tokens=0)),
        (None, Limits(tokens=3), 0.0, Limits(tokens=0)),
        # 父 remaining 显式 0(有界且耗尽)→ 0。
        (None, Limits(tokens=0), MAX_CHILD_FRACTION, Limits(tokens=0)),
        # 三维独立:None/有界/零 混合。
        (
            Limits(steps=50, web_requests=10),
            Limits(tokens=1000, cost_micros=400, steps=60, web_requests=0),
            MAX_CHILD_FRACTION,
            Limits(tokens=500, cost_micros=200, steps=30, web_requests=0),
        ),
        # 自定义 fraction:10 × 0.25 = 2.5 → 2。
        (None, Limits(browser_requests=10), 0.25, Limits(browser_requests=2)),
        # 五维全有界:每维独立取 min。
        (
            Limits(tokens=50, cost_micros=500, steps=5, web_requests=2, browser_requests=1),
            Limits(tokens=1000, cost_micros=1000, steps=100, web_requests=100, browser_requests=100),
            MAX_CHILD_FRACTION,
            Limits(tokens=50, cost_micros=500, steps=5, web_requests=2, browser_requests=1),
        ),
    ],
)
def test_limits_merge_grant_table(
    requested: Limits | None,
    parent_remaining: Limits,
    fraction: float,
    expected: Limits,
) -> None:
    assert limits_merge_grant(requested, parent_remaining, fraction) == expected


def test_limits_merge_grant_defaults_to_max_child_fraction() -> None:
    assert limits_merge_grant(None, Limits(tokens=1000)) == limits_merge_grant(
        None, Limits(tokens=1000), MAX_CHILD_FRACTION
    )


# ---------------------------------------------------------------------------
# Limits payload 序列化对(R-PF-3,T3 消费)
# ---------------------------------------------------------------------------


def test_limits_payload_round_trip() -> None:
    limits = Limits(tokens=100, cost_micros=7, steps=3, web_requests=1, browser_requests=2)
    payload = limits_payload(limits)
    assert payload == {
        "tokens": 100,
        "cost_micros": 7,
        "steps": 3,
        "web_requests": 1,
        "browser_requests": 2,
    }
    assert limits_from_payload(payload) == limits


def test_limits_payload_omits_unset_dimensions() -> None:
    assert limits_payload(Limits(tokens=5)) == {"tokens": 5}
    assert limits_payload(UNLIMITED) == {}


def test_limits_from_payload_missing_keys_are_none() -> None:
    assert limits_from_payload({}) == UNLIMITED
    assert limits_from_payload({"steps": 2}) == Limits(steps=2)


@pytest.mark.parametrize("bad", [True, "5", 1.5, (1, 2)])
def test_limits_from_payload_rejects_non_integer(bad: object) -> None:
    with pytest.raises(TypeError):
        limits_from_payload({"tokens": bad})  # type: ignore[dict-item]

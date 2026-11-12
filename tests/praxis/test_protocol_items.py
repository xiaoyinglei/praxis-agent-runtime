"""protocol.items:Item 开放集合 kind(B1)与冻结性测试(设计 §4.2)。"""

from __future__ import annotations

import dataclasses

import pytest

from praxis.protocol.items import Item, ItemKind


def _make_item(**overrides: object) -> Item:
    kwargs: dict[str, object] = dict(
        item_id="i1",
        thread_id="t",
        turn_id=None,
        seq=1,
        kind="user_message",
        payload={},
        created_at=0.0,
    )
    kwargs.update(overrides)
    return Item(**kwargs)


def test_item_kind_is_open_set() -> None:
    item = Item(
        item_id="i1",
        thread_id="t",
        turn_id=None,
        seq=1,
        kind="alien_kind_from_future",
        payload={"x": 1},
        created_at=0.0,
    )
    assert item.kind == "alien_kind_from_future"  # 构造不校验白名单(B1)


def test_item_is_frozen() -> None:
    item = _make_item()
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.seq = 2


def test_item_kind_values_are_the_ten_agreed() -> None:
    assert {kind.value for kind in ItemKind} == {
        "user_message",
        "steering",
        "model_response",
        "tool_call",
        "tool_result",
        "approval_request",
        "approval_decision",
        "turn_status",
        "summary",
        "compaction_boundary",
    }


def test_item_accepts_item_kind_member_as_str() -> None:
    # ItemKind 是 StrEnum:kind 字段类型是 str,枚举成员可直接传入且按值比较。
    item = _make_item(kind=ItemKind.TOOL_RESULT)
    assert item.kind == "tool_result"


def test_item_supports_turn_id_and_payload_shape() -> None:
    item = _make_item(turn_id="turn-7", seq=12, payload={"text": "hello"})
    assert item.turn_id == "turn-7"
    assert item.seq == 12
    assert item.payload["text"] == "hello"

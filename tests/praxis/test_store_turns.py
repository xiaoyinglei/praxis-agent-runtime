"""store.turns:turn CRUD 与状态写入唯一入口(write_status 的事务原子性)。"""

from __future__ import annotations

import sqlite3

import pytest

from praxis.protocol.items import ItemKind
from praxis.protocol.json import thaw_json
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import Turn, create_turn, get_turn, list_turns, write_status


def test_create_and_get_roundtrip(store: Store) -> None:
    binding = {"model": "m1", "params": {"temp": 0.2}, "tags": ["a", "b"]}
    turn = create_turn(store, turn_id="turn-1", thread_id="t", binding=binding)
    assert isinstance(turn, Turn)
    assert turn.status == "running"  # 初始 INSERT,不走 UPDATE 路径
    assert turn.created_at == turn.updated_at
    assert thaw_json(turn.binding) == binding
    assert get_turn(store, "turn-1") == turn
    assert get_turn(store, "ghost") is None


def test_create_turn_duplicate_id_rejected(store: Store) -> None:
    create_turn(store, turn_id="x", thread_id="t", binding={})
    with pytest.raises(sqlite3.IntegrityError):
        create_turn(store, turn_id="x", thread_id="t", binding={})


def test_list_turns_filters(store: Store) -> None:
    create_turn(store, turn_id="a", thread_id="t1", binding={})
    create_turn(store, turn_id="b", thread_id="t2", binding={})
    write_status(store, "a", "completed", reason="done")
    assert [t.turn_id for t in list_turns(store)] == ["a", "b"]  # created_at,turn_id 稳定序
    assert [t.turn_id for t in list_turns(store, thread_id="t2")] == ["b"]
    assert [t.turn_id for t in list_turns(store, status="completed")] == ["a"]
    assert [t.turn_id for t in list_turns(store, status="running")] == ["b"]
    assert list_turns(store, thread_id="t1", status="running") == []


def test_write_status_updates_turn_and_appends_turn_status_item(store: Store) -> None:
    create_turn(store, turn_id="turn-1", thread_id="t", binding={"model": "test-model"})
    store.append_items("t", "turn-1", [ItemEntry(ItemKind.USER_MESSAGE, {"text": "hi"})])

    updated = write_status(store, "turn-1", "completed", reason="模型正常返回")

    assert updated.status == "completed"
    assert updated.updated_at >= updated.created_at
    assert get_turn(store, "turn-1") == updated

    items = store.read_turn_items("turn-1")
    assert [i.kind for i in items] == ["user_message", "turn_status"]
    status_item = items[1]
    assert status_item.seq == 2  # 走同一 seq 分配
    assert status_item.payload == {"from": "running", "to": "completed", "reason": "模型正常返回"}


def test_write_status_interrupted_records_last_seq(store: Store) -> None:
    create_turn(store, turn_id="turn-1", thread_id="t", binding={})
    store.append_items(
        "t",
        "turn-1",
        [
            ItemEntry("user_message", {"n": 1}),
            ItemEntry("model_response", {"n": 2}),
            ItemEntry("tool_result", {"n": 3}),
        ],
    )
    store.append_items("other-thread", None, [ItemEntry("user_message", {"n": 99})])  # 不得污染

    write_status(store, "turn-1", "interrupted", reason="用户打断")

    status_item = store.read_turn_items("turn-1")[-1]
    assert status_item.kind == "turn_status"
    assert status_item.payload["last_seq"] == 3  # 打断时 thread 已有的最后一条 seq
    assert status_item.seq == 4  # turn_status 本身排在 last_seq 之后


def test_write_status_unknown_turn_raises_and_writes_nothing(store: Store) -> None:
    with pytest.raises(ValueError, match="unknown turn"):
        write_status(store, "ghost", "completed", reason="r")
    assert store.read_items("t") == []


def test_write_status_multi_step_history(store: Store) -> None:
    create_turn(store, turn_id="turn-1", thread_id="t", binding={})
    write_status(store, "turn-1", "waiting_approval", reason="需要审批")
    final = write_status(store, "turn-1", "completed", reason="审批通过")
    assert final.status == "completed"
    kinds = [i for i in store.read_turn_items("turn-1") if i.kind == "turn_status"]
    assert [i.payload["from"] for i in kinds] == ["running", "waiting_approval"]
    assert [i.payload["to"] for i in kinds] == ["waiting_approval", "completed"]

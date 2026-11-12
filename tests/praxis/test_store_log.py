"""store.log:append-only 事件日志(B1/B12 回归)、原子性与单进程多线程串行化。

R3 裁定:单进程内 MAX+1 分配不存在 seq 冲突,原子性以两种方式验证——
(a) transact 内 INSERT 后抛 RuntimeError,行不落库;(b) append 批量中途
json.dumps 抛错,整批回滚不留残行。另附 brief 的 IntegrityError 回滚场景。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue, thaw_json
from praxis.store.log import ItemEntry, Store

# ---------------------------------------------------------------------------
# brief Step 1 用例
# ---------------------------------------------------------------------------


def test_append_returns_known_ids(store: Store) -> None:
    r = store.append_items(
        "t",
        None,
        [
            ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "a"}),
            ItemEntry(kind=ItemKind.STEERING, payload={"text": "b"}),
        ],
    )
    items = store.read_items("t")
    assert [i.item_id for i in items] == [a.item_id for a in r]  # B12:返回即所写
    assert [i.seq for i in items] == [1, 2]
    assert [i.kind for i in items] == ["user_message", "steering"]
    assert items[0].payload == {"text": "a"}


def test_read_unknown_kind_fail_open(store: Store) -> None:
    store.transact(
        lambda c: c.execute(
            "INSERT INTO items VALUES (?,?,?,?,?,?,?)",
            ("x", "t", None, 1, "alien_kind", '{"weird": true}', 0.0),
        )
    )
    items = store.read_items("t")  # B1:不抛
    assert items[0].kind == "alien_kind" and items[0].payload == {"weird": True}


def test_seq_unique_per_thread(store: Store) -> None:
    store.append_items("t1", None, [ItemEntry("user_message", {"text": "1"})])
    store.append_items("t2", None, [ItemEntry("user_message", {"text": "2"})])
    assert store.read_items("t1")[0].seq == store.read_items("t2")[0].seq == 1


def test_append_batch_rolls_back_on_integrity_error(store: Store) -> None:
    appended = store.append_items("t", None, [ItemEntry("user_message", {"text": "preset"})])
    with pytest.raises(sqlite3.IntegrityError):
        store.transact(
            lambda c: c.execute(
                "INSERT INTO items VALUES (?,?,?,?,?,?,?)",
                ("dup", "t", None, 1, "k", "{}", 0.0),
            )
        )
    items = store.read_items("t")
    assert [i.item_id for i in items] == [appended[0].item_id]  # dup 未留下


# ---------------------------------------------------------------------------
# 原子性(R3 裁定的两种形态)
# ---------------------------------------------------------------------------


def test_transact_rolls_back_on_exception(store: Store) -> None:
    def fn(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO items VALUES (?,?,?,?,?,?,?)",
            ("bad", "t", None, 1, "k", "{}", 0.0),
        )
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        store.transact(fn)
    assert store.read_items("t") == []


def test_append_batch_rolls_back_on_encode_failure(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    real_dumps = json.dumps
    calls = {"n": 0}

    def flaky_dumps(*args: object, **kwargs: object) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("encode failure")
        return real_dumps(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(json, "dumps", flaky_dumps)
    with pytest.raises(RuntimeError, match="encode failure"):
        store.append_items(
            "t",
            None,
            [ItemEntry("user_message", {"text": "a"}), ItemEntry("steering", {"text": "b"})],
        )
    assert calls["n"] == 2
    assert store.read_items("t") == []  # 第一条也已回滚,无残行
    monkeypatch.undo()
    after = store.append_items("t", None, [ItemEntry("user_message", {"text": "c"})])
    assert after[0].seq == 1  # 回滚后 seq 从 0 重新分配


# ---------------------------------------------------------------------------
# 读路径与线程助手
# ---------------------------------------------------------------------------


def test_read_items_after_seq_and_limit(store: Store) -> None:
    store.append_items(
        "t",
        None,
        [ItemEntry("user_message", {"n": i}) for i in range(1, 6)],
    )
    assert [i.seq for i in store.read_items("t", after_seq=2, limit=2)] == [3, 4]
    assert [i.seq for i in store.read_items("t", after_seq=4)] == [5]
    assert store.read_items("other") == []


def test_read_turn_items(store: Store) -> None:
    store.append_items("t", "turn-1", [ItemEntry("user_message", {"n": 1})])
    store.append_items("t", "turn-2", [ItemEntry("user_message", {"n": 2})])
    store.append_items("t", None, [ItemEntry("steering", {"n": 3})])
    items = store.read_turn_items("turn-1")
    assert [i.seq for i in items] == [1]
    assert [i.payload["n"] for i in items] == [1]


def test_thread_helpers(store: Store) -> None:
    a = store.create_thread()
    b = store.create_thread()
    assert a != b and len(a) == 32 and all(c in "0123456789abcdef" for c in a)
    store.ensure_thread("pinned")
    store.ensure_thread("pinned")  # 幂等,不报错
    assert {a, b, "pinned"} <= set(store.list_threads())


def test_schema_version_recorded(store: Store) -> None:
    row = store.transact(
        lambda c: c.execute("SELECT value FROM store_metadata WHERE key = 'schema_version'").fetchone()
    )
    assert row is not None and row["value"] == "1"


def test_close_is_idempotent_and_blocks_later_use(tmp_path: Path) -> None:
    s = Store(tmp_path / "closed.db")
    s.append_items("t", None, [ItemEntry("user_message", {"x": 1})])
    s.close()
    s.close()  # 幂等
    with pytest.raises(RuntimeError, match="closed"):
        s.append_items("t", None, [ItemEntry("user_message", {"x": 2})])


def test_inprocess_concurrent_append_serializes(store: Store) -> None:
    """写者线程模型:4 个调用线程并发 append,由单连接写者串行化,seq 连续无冲突。"""

    def worker() -> None:
        for _ in range(50):
            store.append_items("t", None, [ItemEntry("user_message", {"x": 1})])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    items = store.read_items("t")
    assert [i.seq for i in items] == list(range(1, 201))
    assert len({i.item_id for i in items}) == 200


# ---------------------------------------------------------------------------
# hypothesis:任意 JSON payload 无损往返(freeze/thaw)
# ---------------------------------------------------------------------------

_json_values: st.SearchStrategy[object] = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(), children, max_size=4),
    max_leaves=12,
)
_payloads = st.dictionaries(st.text(), _json_values, max_size=6)


@settings(
    deadline=None,
    max_examples=50,
    # 复用同一 store fixture 是故意的:每个 example 独立 append 一条再读末条,
    # 不依赖 fixture 重置(seq 只是单调递增的簿记)。
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(payload=_payloads)
def test_payload_json_roundtrip_lossless(store: Store, payload: dict[str, JsonValue]) -> None:
    appended = store.append_items("hyp", None, [ItemEntry(kind="user_message", payload=payload)])
    got = store.read_items("hyp", after_seq=0)[-1]
    assert got.item_id == appended[0].item_id  # B12:返回即所写
    assert thaw_json(got.payload) == payload  # freeze/thaw 无损

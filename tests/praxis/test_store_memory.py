"""store.memory:store_metadata 唯一缝——sha256 包络、CAS 与有界 pruning(B23)。

两向纪律(Review Focus #4):items 读路径对未知 kind fail-open;完整性校验
(sha256 包络)失配 fail-closed——MemoryEnvelopeCorrupt 而不是返回可疑数据。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from praxis.protocol.json import canonical_json
from praxis.store.log import Store
from praxis.store.memory import (
    MEMORY_ENTRY_HISTORY_KEEP,
    MEMORY_JOB_HISTORY_KEEP,
    MEMORY_VERSION_HISTORY_KEEP,
    MemoryEnvelopeCorrupt,
    cas_put_metadata,
    delete_metadata,
    get_metadata,
    prune_job_history,
    prune_versions,
    put_state_enveloped,
    read_state_verified,
)

_STATE: dict[str, object] = {
    "entries": [{"key": "k1", "title": "T", "text": "hello"}],
    "use_memories": True,
}


def _raw(store: Store, key: str) -> str | None:
    return get_metadata(store, key)


def _put(store: Store, key: str, value: str) -> None:
    store.transact(lambda c: cas_put_metadata(c, key, None, value))


def _write_raw(store: Store, key: str, value: str) -> None:
    def _write(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE store_metadata SET value = ? WHERE key = ?", (value, key))

    store.transact(_write)


# ---------------------------------------------------------------------------
# canonical_json 唯一来源(protocol/json.py)
# ---------------------------------------------------------------------------


def test_canonical_json_is_key_order_independent_and_compact() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == canonical_json({"a": (2, 3), "b": 1})
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


# ---------------------------------------------------------------------------
# 基础 KV:get / delete / CAS
# ---------------------------------------------------------------------------


def test_get_metadata_absent_is_none(store: Store) -> None:
    assert get_metadata(store, "missing") is None


def test_cas_insert_when_absent(store: Store) -> None:
    assert store.transact(lambda c: cas_put_metadata(c, "k", None, '"v"')) is True
    assert get_metadata(store, "k") == '"v"'


def test_cas_insert_conflict_has_zero_side_effect(store: Store) -> None:
    _put(store, "k", '"existing"')
    assert store.transact(lambda c: cas_put_metadata(c, "k", None, '"other"')) is False
    assert get_metadata(store, "k") == '"existing"'


def test_cas_update_when_expected_matches(store: Store) -> None:
    _put(store, "k", '"v1"')
    assert store.transact(lambda c: cas_put_metadata(c, "k", '"v1"', '"v2"')) is True
    assert get_metadata(store, "k") == '"v2"'


def test_cas_mismatch_returns_false_and_keeps_value(store: Store) -> None:
    _put(store, "k", '"v1"')
    assert store.transact(lambda c: cas_put_metadata(c, "k", '"wrong"', '"v2"')) is False
    assert get_metadata(store, "k") == '"v1"'


def test_delete_metadata(store: Store) -> None:
    _put(store, "k", '"v"')
    store.transact(lambda c: delete_metadata(c, "k"))
    assert get_metadata(store, "k") is None
    # 幂等:再删不抛
    store.transact(lambda c: delete_metadata(c, "k"))


# ---------------------------------------------------------------------------
# 包络往返 + 版本快照(旧 memory_ops.py:48-53 形状移植)
# ---------------------------------------------------------------------------


def test_envelope_roundtrip_and_snapshot(store: Store) -> None:
    def _put_state(conn: sqlite3.Connection) -> int:
        return put_state_enveloped(conn, "memory:x", _STATE)

    version = store.transact(_put_state)
    assert version == 1

    latest = _raw(store, "memory:x")
    snapshot = _raw(store, "memory:x:version:1")
    assert latest is not None and snapshot is not None
    assert latest == snapshot  # 快照与最新指针同值(replace=False 不可变)

    envelope = json.loads(latest)
    assert envelope["version"] == 1
    assert envelope["state"]["version"] == 1
    assert envelope["sha256"] == hashlib.sha256(
        canonical_json(envelope["state"]).encode("utf-8")
    ).hexdigest()

    state = store.transact(lambda c: read_state_verified(c, "memory:x"))
    assert state == {**_STATE, "version": 1}


def test_envelope_version_increments_and_old_snapshot_immutable(store: Store) -> None:
    first = store.transact(lambda c: put_state_enveloped(c, "memory:x", _STATE))
    snapshot_v1 = _raw(store, f"memory:x:version:{first}")
    second = store.transact(lambda c: put_state_enveloped(c, "memory:x", {**_STATE, "flag": True}))
    assert second == first + 1
    assert _raw(store, f"memory:x:version:{first}") == snapshot_v1
    state = store.transact(lambda c: read_state_verified(c, "memory:x"))
    assert state is not None and state["version"] == second and state["flag"] is True


def test_read_state_absent_is_none(store: Store) -> None:
    assert store.transact(lambda c: read_state_verified(c, "nope")) is None


def test_envelope_tampered_state_fails_closed(store: Store) -> None:
    store.transact(lambda c: put_state_enveloped(c, "memory:x", _STATE))
    raw = _raw(store, "memory:x")
    assert raw is not None
    envelope = json.loads(raw)
    envelope["state"]["entries"][0]["text"] = "tampered"  # 一位改动,sha 不变
    _write_raw(store, "memory:x", json.dumps(envelope))

    with pytest.raises(MemoryEnvelopeCorrupt):
        store.transact(lambda c: read_state_verified(c, "memory:x"))


def test_envelope_tampered_sha_fails_closed(store: Store) -> None:
    store.transact(lambda c: put_state_enveloped(c, "memory:x", _STATE))
    raw = _raw(store, "memory:x")
    assert raw is not None
    envelope = json.loads(raw)
    envelope["sha256"] = "0" * 64
    _write_raw(store, "memory:x", json.dumps(envelope))

    with pytest.raises(MemoryEnvelopeCorrupt):
        store.transact(lambda c: read_state_verified(c, "memory:x"))


def test_envelope_tampered_version_fails_closed(store: Store) -> None:
    store.transact(lambda c: put_state_enveloped(c, "memory:x", _STATE))
    raw = _raw(store, "memory:x")
    assert raw is not None
    envelope = json.loads(raw)
    envelope["version"] = 99  # 包络顶层版本与 state 内版本不一致
    _write_raw(store, "memory:x", json.dumps(envelope))

    with pytest.raises(MemoryEnvelopeCorrupt):
        store.transact(lambda c: read_state_verified(c, "memory:x"))


def test_envelope_non_finite_state_fails_closed(store: Store) -> None:
    # json.loads 默认接受 NaN;完整性校验必须把它归为损坏包络而不是泄漏出去。
    _put(store, "memory:x", '{"state":{"version":1,"x":NaN},"sha256":"0","version":1}')
    with pytest.raises(MemoryEnvelopeCorrupt):
        store.transact(lambda c: read_state_verified(c, "memory:x"))


def test_envelope_malformed_json_fails_closed(store: Store) -> None:
    _put(store, "memory:x", "{not-json")
    with pytest.raises(MemoryEnvelopeCorrupt):
        store.transact(lambda c: read_state_verified(c, "memory:x"))
    # 畸形包络不能被"续版"掩盖:写入同样 fail-closed
    with pytest.raises(MemoryEnvelopeCorrupt):
        store.transact(lambda c: put_state_enveloped(c, "memory:x", _STATE))


# ---------------------------------------------------------------------------
# pruning(B23):版本快照有界、latest/当前 state/在跑 job 永不删、幂等
# ---------------------------------------------------------------------------


def test_pruning_constants_are_q7_values() -> None:
    assert MEMORY_VERSION_HISTORY_KEEP == 20
    assert MEMORY_ENTRY_HISTORY_KEEP == 100
    assert MEMORY_JOB_HISTORY_KEEP == 10


def test_prune_versions_keeps_bounded_window_and_latest(store: Store) -> None:
    for _ in range(25):
        store.transact(lambda c: put_state_enveloped(c, "memory:x", _STATE))

    removed = store.transact(lambda c: prune_versions(c, "memory:x", keep=20))
    assert removed == 5

    # 只删 n ≤ max−keep;21..25 与 latest 指针都在
    remaining = [n for n in range(1, 26) if _raw(store, f"memory:x:version:{n}") is not None]
    assert remaining == list(range(6, 26))
    assert _raw(store, "memory:x") is not None
    assert store.transact(lambda c: read_state_verified(c, "memory:x")) is not None

    # 幂等:第二次零删除
    removed_again = store.transact(lambda c: prune_versions(c, "memory:x", keep=20))
    assert removed_again == 0
    assert list(range(6, 26)) == [
        n for n in range(1, 26) if _raw(store, f"memory:x:version:{n}") is not None
    ]


def test_prune_versions_below_keep_is_noop(store: Store) -> None:
    for _ in range(3):
        store.transact(lambda c: put_state_enveloped(c, "memory:x", _STATE))
    assert store.transact(lambda c: prune_versions(c, "memory:x", keep=3)) == 0
    assert store.transact(lambda c: read_state_verified(c, "memory:x")) is not None


def test_prune_versions_respects_custom_keep(store: Store) -> None:
    for _ in range(5):
        store.transact(lambda c: put_state_enveloped(c, "memory:x", _STATE))
    assert store.transact(lambda c: prune_versions(c, "memory:x", keep=2)) == 3
    assert _raw(store, "memory:x:version:4") is not None
    assert _raw(store, "memory:x:version:5") is not None


def test_prune_versions_rejects_keep_below_one(store: Store) -> None:
    with pytest.raises(ValueError):
        store.transact(lambda c: prune_versions(c, "memory:x", keep=0))


def test_prune_job_history_bounded_idempotent_and_live_jobs_survive(store: Store) -> None:
    _put(store, "memory:jobs", "{}")  # latest 指针 / 在跑作业聚合行
    for n in range(1, 16):
        _put(store, f"memory:jobs:job:{n}", "{}")
    _put(store, "memory:jobs:job:running", "{}")  # 非数字后缀=在跑,永不删

    assert store.transact(lambda c: prune_job_history(c, "memory:jobs", keep=10)) == 5
    assert [n for n in range(1, 16) if _raw(store, f"memory:jobs:job:{n}") is not None] == list(
        range(6, 16)
    )
    assert _raw(store, "memory:jobs") is not None
    assert _raw(store, "memory:jobs:job:running") is not None

    assert store.transact(lambda c: prune_job_history(c, "memory:jobs", keep=10)) == 0


# ---------------------------------------------------------------------------
# 并发:两 Store 实例同库交替 CAS,恰一方赢
# ---------------------------------------------------------------------------


def test_two_store_instances_cas_exactly_one_wins(tmp_path: Path) -> None:
    db = tmp_path / "shared.db"
    s1 = Store(db)
    s2 = Store(db)
    try:
        s1.transact(lambda c: cas_put_metadata(c, "k", None, '"seed"'))
        barrier = threading.Barrier(2)
        results: list[bool] = []
        lock = threading.Lock()

        def _worker(store: Store, value: str) -> None:
            barrier.wait(timeout=10)
            won = store.transact(lambda c: cas_put_metadata(c, "k", '"seed"', value))
            with lock:
                results.append(won)

        threads = [
            threading.Thread(target=_worker, args=(s1, '"a"'), daemon=True),
            threading.Thread(target=_worker, args=(s2, '"b"'), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert sorted(results) == [False, True]
        assert get_metadata(s1, "k") in ('"a"', '"b"')
    finally:
        s1.close()
        s2.close()
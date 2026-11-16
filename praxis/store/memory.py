"""store_metadata 唯一读写缝:sha256 包络、CAS 与有界 pruning(设计 §5.14;B23)。

items 表"只 INSERT"的硬规则不适用于 store_metadata:memory 状态/版本快照/
作业历史的 UPDATE/DELETE 是设计 §5.14 的字面授权,且**全仓只允许出现在本文件**
(CI 硬规则㉑)。本模块因此是 store_metadata 的唯一读写缝,其他模块不拼 SQL。

不变量:
- latest 指针(``key``)与当前 state 永不 prune;只删历史快照
  ``key:version:<n>``(n ≤ max−keep)。快照写入用裸 INSERT(replace=False):
  已存在的版本号不可改写,冲突即整事务回滚。
- 完整性校验 fail-closed:包络 JSON 畸形、sha256 失配或顶层版本与 state
  内版本不一致 → ``MemoryEnvelopeCorrupt``。这是 Review Focus #4 的另一向:
  items 读路径对未知 kind fail-open,memory 包络校验失配绝不返回可疑数据。
- 所有 ``conn`` 函数都在**调用方已开启的事务内**执行(BEGIN IMMEDIATE 由
  ``Store.transact`` 提供,与 ``append_items_tx`` 同纪律);CAS 的
  "比较+写入"因此是原子的。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from typing import cast

from praxis.protocol.json import JsonValue, canonical_json
from praxis.store.log import Store

# Q7 pruning 常量(设计 §5.14;B23):版本快照留最近 20,每 entry 修订历史
# 留 100(纯 state 内列表截断,由 T11 消费),终态作业行留最近 10。
MEMORY_VERSION_HISTORY_KEEP = 20
MEMORY_ENTRY_HISTORY_KEEP = 100
MEMORY_JOB_HISTORY_KEEP = 10

VERSION_SNAPSHOT_SEGMENT = "version"
JOB_HISTORY_SEGMENT = "job"


class MemoryEnvelopeCorrupt(RuntimeError):  # noqa: N818 — 任务契约固定异常名(T10/T11 消费)
    """store_metadata 中的 memory 包络结构畸形或完整性校验失败(fail-closed)。"""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(f"memory envelope corrupt for {key!r}: {reason}")
        self.key = key
        self.reason = reason


def _select_raw(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM store_metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else cast(str, row[0])


def get_metadata(store: Store, key: str) -> str | None:
    """读一个 store_metadata 原始值;键不存在返回 None。

    走 ``Store.transact``(既有读方法先例,如 ``get_turn``)——store 层只暴露
    这一条事务入口,避免另开读连接/私有字段访问。
    """
    return store.transact(lambda conn: _select_raw(conn, key))


def cas_put_metadata(
    conn: sqlite3.Connection, key: str, expected_json: str | None, value_json: str
) -> bool:
    """比较并写入(调用方事务内);返回是否赢得 CAS。

    ``expected_json=None`` 表示要求键**不存在**(INSERT,冲突即失败);
    否则要求当前原始值逐字等于 ``expected_json``(UPDATE … WHERE value=?)。
    未命中返回 False 且零副作用——调用方无需回滚即可继续其他判断。
    """
    if expected_json is None:
        cursor = conn.execute(
            "INSERT INTO store_metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO NOTHING",
            (key, value_json),
        )
        return cursor.rowcount == 1
    cursor = conn.execute(
        "UPDATE store_metadata SET value = ? WHERE key = ? AND value = ?",
        (value_json, key, expected_json),
    )
    return cursor.rowcount == 1


def delete_metadata(conn: sqlite3.Connection, key: str) -> None:
    """删除一个键(调用方事务内);键不存在是无操作(幂等)。"""
    conn.execute("DELETE FROM store_metadata WHERE key = ?", (key,))


def _load_envelope(key: str, raw: str) -> Mapping[str, JsonValue]:
    """解析包络并校验结构;任何畸形都抛 MemoryEnvelopeCorrupt。"""
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MemoryEnvelopeCorrupt(key, f"invalid JSON: {exc.msg}") from None
    if not isinstance(loaded, dict):
        raise MemoryEnvelopeCorrupt(key, "envelope must be a JSON object")
    for field in ("state", "sha256", "version"):
        if field not in loaded:
            raise MemoryEnvelopeCorrupt(key, f"missing envelope field {field!r}")
    if not isinstance(loaded["state"], dict):
        raise MemoryEnvelopeCorrupt(key, "envelope state must be a JSON object")
    if not isinstance(loaded["sha256"], str):
        raise MemoryEnvelopeCorrupt(key, "envelope sha256 must be a string")
    if type(loaded["version"]) is not int:
        raise MemoryEnvelopeCorrupt(key, "envelope version must be an integer")
    return cast("Mapping[str, JsonValue]", loaded)


def read_state_verified(conn: sqlite3.Connection, key: str) -> Mapping[str, JsonValue] | None:
    """读 latest 包络并校验 sha256;缺失返回 None,失配 fail-closed。

    校验覆盖三处:sha256(state) 与包络记录一致、包络顶层 version 与
    state["version"] 一致、结构可解析。任一不满足 → ``MemoryEnvelopeCorrupt``。
    """
    raw = _select_raw(conn, key)
    if raw is None:
        return None
    envelope = _load_envelope(key, raw)
    state = cast("Mapping[str, JsonValue]", envelope["state"])
    recorded = cast(str, envelope["sha256"])
    version = cast(int, envelope["version"])
    try:
        actual = hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        # json.loads 默认接受 NaN/Infinity;这里与"非 JSON 兼容值"一并 fail-closed。
        raise MemoryEnvelopeCorrupt(key, f"state is not canonical JSON: {exc}") from None
    if actual != recorded:
        raise MemoryEnvelopeCorrupt(key, "sha256 mismatch")
    if state.get("version") != version:
        raise MemoryEnvelopeCorrupt(key, "version mismatch between envelope and state")
    return state


def put_state_enveloped(conn: sqlite3.Connection, key: str, state: Mapping[str, JsonValue]) -> int:
    """写入下一版本的 sha256 包络 + 不可变快照,返回新版本号。

    形状移植自旧 ``memory_ops.py:48-53``:``committed = state + version``;
    latest 指针 ``key`` 与快照 ``key:version:<n>`` 存同一份
    ``canonical_json({"state": committed, "sha256": sha, "version": n})``。
    版本号由当前包络的 version+1 得出(初版 1);当前包络损坏时 fail-closed,
    绝不在无法验证的旧状态上"续版"。快照键已存在 → IntegrityError 整事务回滚。
    """
    current = read_state_verified(conn, key)
    version = 1 if current is None else cast(int, current["version"]) + 1

    committed: dict[str, JsonValue] = dict(state)
    committed["version"] = version
    digest = hashlib.sha256(canonical_json(committed).encode("utf-8")).hexdigest()
    envelope = canonical_json({"state": committed, "sha256": digest, "version": version})

    # replace=False:裸 INSERT,版本快照不可改写(冲突→整事务回滚)。
    conn.execute(
        "INSERT INTO store_metadata(key, value) VALUES (?, ?)",
        (f"{key}:{VERSION_SNAPSHOT_SEGMENT}:{version}", envelope),
    )
    conn.execute(
        "INSERT INTO store_metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, envelope),
    )
    return version


def _numbered_keys(conn: sqlite3.Connection, prefix: str) -> list[tuple[int, str]]:
    """取 ``<prefix><n>`` 形态的键(n 为 ASCII 数字),返回 (n, key) 列表。"""
    rows = conn.execute(
        "SELECT key FROM store_metadata WHERE substr(key, 1, ?) = ?",
        (len(prefix), prefix),
    ).fetchall()
    numbered: list[tuple[int, str]] = []
    for row in rows:
        key = cast(str, row[0])
        suffix = key[len(prefix) :]
        if suffix.isascii() and suffix.isdigit():
            numbered.append((int(suffix), key))
    return numbered


def _prune_numbered(conn: sqlite3.Connection, prefix: str, keep: int) -> int:
    """删 ``<prefix><n>`` 中 n ≤ max−keep 的行;返回删除数(幂等)。"""
    if keep < 1:
        raise ValueError("keep must be >= 1")
    numbered = _numbered_keys(conn, prefix)
    if not numbered:
        return 0
    cutoff = max(number for number, _ in numbered) - keep
    doomed = [key for number, key in numbered if number <= cutoff]
    for key in doomed:
        delete_metadata(conn, key)
    return len(doomed)


def prune_versions(conn: sqlite3.Connection, key: str, keep: int = MEMORY_VERSION_HISTORY_KEEP) -> int:
    """只删 ``key:version:<n>`` 中 n ≤ max−keep 的历史快照(B23)。

    latest 指针 ``key``(当前 state)不匹配该前缀,永不删除;最大版本快照
    也因 n=max > max−keep(keep≥1)保留。重复调用零删除(幂等)。
    """
    return _prune_numbered(conn, f"{key}:{VERSION_SNAPSHOT_SEGMENT}:", keep)


def prune_job_history(conn: sqlite3.Connection, key: str, keep: int = MEMORY_JOB_HISTORY_KEEP) -> int:
    """只删终态作业历史行 ``key:job:<n>`` 中 n ≤ max−keep 的旧行(B23)。

    ``key`` 本身(latest/在跑作业聚合行)与非数字后缀的作业键(如
    ``key:job:running``)不匹配数字后缀过滤,永不删除。
    """
    return _prune_numbered(conn, f"{key}:{JOB_HISTORY_SEGMENT}:", keep)


__all__ = [
    "JOB_HISTORY_SEGMENT",
    "MEMORY_ENTRY_HISTORY_KEEP",
    "MEMORY_JOB_HISTORY_KEEP",
    "MEMORY_VERSION_HISTORY_KEEP",
    "VERSION_SNAPSHOT_SEGMENT",
    "MemoryEnvelopeCorrupt",
    "cas_put_metadata",
    "delete_metadata",
    "get_metadata",
    "prune_job_history",
    "prune_versions",
    "put_state_enveloped",
    "read_state_verified",
]
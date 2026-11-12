"""Praxis turns 表:创建、查询与 turn 状态写入唯一入口(设计 §5.1)。

本文件是全代码库唯一允许出现 UPDATE 作用于 turns 表 SQL 字符串的地方
(CI 硬规则扫描强制)。状态变更只能经 write_status,它在一个事务内同时
改写 turns.status 并追加一条 turn_status item,保证表状态与日志永不漂移。
create_turn 的初始 status="running" 走 INSERT——插入不是改写。
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue, freeze_json, thaw_json
from praxis.store.log import ItemEntry, Store, append_items_tx

_TURN_COLUMNS = "turn_id, thread_id, status, created_at, updated_at, binding"


@dataclass(frozen=True, slots=True)
class Turn:
    """一次 turn 的当前状态快照;binding 为 JSON 兼容映射(能力绑定,设计 §4.3)。"""

    turn_id: str
    thread_id: str
    status: str
    created_at: float
    updated_at: float
    binding: Mapping[str, JsonValue]


def _row_to_turn(row: sqlite3.Row) -> Turn:
    binding = freeze_json(json.loads(row["binding"]), path="$")
    return Turn(
        turn_id=row["turn_id"],
        thread_id=row["thread_id"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        binding=cast("Mapping[str, JsonValue]", binding),
    )


def _fetch_turn(conn: sqlite3.Connection, turn_id: str) -> Turn | None:
    row = conn.execute(f"SELECT {_TURN_COLUMNS} FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
    return _row_to_turn(row) if row is not None else None


def create_turn(
    store: Store,
    *,
    turn_id: str,
    thread_id: str,
    binding: Mapping[str, JsonValue],
) -> Turn:
    """INSERT 一条 status="running" 的 turn(初始状态不经由改写路径产生)。"""

    def _insert(conn: sqlite3.Connection) -> Turn:
        now = time.time()
        conn.execute(
            "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?)",
            (turn_id, thread_id, "running", now, now, json.dumps(thaw_json(binding), sort_keys=True)),
        )
        turn = _fetch_turn(conn, turn_id)
        assert turn is not None
        return turn

    return store.transact(_insert)


def get_turn(store: Store, turn_id: str) -> Turn | None:
    return store.transact(lambda conn: _fetch_turn(conn, turn_id))


def list_turns(
    store: Store,
    *,
    thread_id: str | None = None,
    status: str | None = None,
) -> list[Turn]:
    def _read(conn: sqlite3.Connection) -> list[Turn]:
        sql = f"SELECT {_TURN_COLUMNS} FROM turns"
        conditions: list[str] = []
        params: list[object] = []
        if thread_id is not None:
            conditions.append("thread_id = ?")
            params.append(thread_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at, turn_id"
        return [_row_to_turn(row) for row in conn.execute(sql, params)]

    return store.transact(_read)


def write_status(store: Store, turn_id: str, to: str, *, reason: str) -> Turn:
    """turn 状态变更唯一入口:同一事务内改写 turns.status 并追加 turn_status item。

    item payload 为 {from, to, reason};to == "interrupted" 时附加 last_seq =
    写入前该 thread 的当前 MAX(seq),即被打断瞬间日志里已存在的最后一条 seq,
    recovery(Task 9)据此知道从哪里续读(turn_status 自身排在它之后)。
    """

    def _tx(conn: sqlite3.Connection) -> Turn:
        row = conn.execute("SELECT thread_id, status FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown turn_id: {turn_id!r}")
        thread_id: str = row["thread_id"]
        from_status: str = row["status"]
        payload: dict[str, JsonValue] = {"from": from_status, "to": to, "reason": reason}
        if to == "interrupted":
            last_seq_row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM items WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            payload["last_seq"] = int(last_seq_row[0])
        conn.execute("UPDATE turns SET status = ?, updated_at = ? WHERE turn_id = ?", (to, time.time(), turn_id))
        append_items_tx(conn, thread_id, turn_id, [ItemEntry(kind=ItemKind.TURN_STATUS, payload=payload)])
        turn = _fetch_turn(conn, turn_id)
        assert turn is not None
        return turn

    return store.transact(_tx)

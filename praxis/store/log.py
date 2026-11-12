"""Praxis append-only 事件日志(设计 §5.1;findings B1/B12 写侧)。

线程模型:每个 Store 实例独占一个写者线程,该线程持有唯一 sqlite3 连接;
所有公开方法(读包括在内)把工作提交到写者线程并阻塞等待结果(queue + Future)。
进程内串行来自写者线程;跨进程串行来自 BEGIN IMMEDIATE + busy_timeout=5000。
后续任务的 async 调用方用 asyncio.to_thread 包装这些同步方法即可。

不变量:
- items 表只 INSERT,永不改写(B12 结构性消亡):append_items 直接返回刚写入的
  (item_id, seq),调用方无需再查"最新一条"。
- 读路径对 kind 零校验(B1):未知 kind 原样读出。
- seq 在同一 IMMEDIATE 事务内 SELECT COALESCE(MAX(seq),0) 分配,
  UNIQUE(thread_id, seq) 是最后防线。
- WAL + synchronous=NORMAL;schema_version 记入 store_metadata。
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from praxis.protocol.items import Item
from praxis.protocol.json import JsonValue, freeze_json, thaw_json

T = TypeVar("T")

SCHEMA_VERSION = "1"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS threads(thread_id TEXT PRIMARY KEY, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS turns(turn_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL,
  status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, binding TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items(item_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL,
  turn_id TEXT, seq INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
  created_at REAL NOT NULL, UNIQUE(thread_id, seq));
CREATE INDEX IF NOT EXISTS items_turn ON items(turn_id, seq);
CREATE TABLE IF NOT EXISTS store_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_ITEM_COLUMNS = "item_id, thread_id, turn_id, seq, kind, payload, created_at"

_Work = tuple[Callable[[sqlite3.Connection], Any], Future[Any]]


@dataclass(frozen=True, slots=True)
class ItemEntry:
    """写入端最小单元:kind 为开放集合,payload 必须是 JSON 兼容映射。"""

    kind: str
    payload: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class AppendedItem:
    """刚写入行的身份——返回即所写,不存在"事后查最新"的歧义。"""

    item_id: str
    seq: int


def _row_to_item(row: sqlite3.Row) -> Item:
    payload = freeze_json(json.loads(row["payload"]), path="$")
    return Item(
        item_id=row["item_id"],
        thread_id=row["thread_id"],
        turn_id=row["turn_id"],
        seq=row["seq"],
        kind=row["kind"],
        payload=cast("Mapping[str, JsonValue]", payload),
        created_at=row["created_at"],
    )


def append_items_tx(
    conn: sqlite3.Connection,
    thread_id: str,
    turn_id: str | None,
    entries: Sequence[ItemEntry],
) -> list[AppendedItem]:
    """在**调用方已开启的事务内**追加一批 item 并返回其身份。

    seq 分配:先 SELECT COALESCE(MAX(seq),0) 取基准,再逐条 INSERT base+i。
    并发安全完全依赖外层 BEGIN IMMEDIATE 把写事务串到同一临界区;
    UNIQUE(thread_id, seq) 兜底任何逻辑外冲突(触发即整批回滚)。
    公开此函数是让 store.turns 的 write_status 能在同一事务内追加
    turn_status item——跨模块只走公开接口(CI 硬规则)。
    """
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) FROM items WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    base = int(row[0])
    now = time.time()
    out: list[AppendedItem] = []
    for i, entry in enumerate(entries, start=1):
        item_id = uuid.uuid4().hex
        payload_text = json.dumps(thaw_json(entry.payload), sort_keys=True)
        conn.execute(
            "INSERT INTO items VALUES (?,?,?,?,?,?,?)",
            (item_id, thread_id, turn_id, base + i, entry.kind, payload_text, now),
        )
        out.append(AppendedItem(item_id=item_id, seq=base + i))
    return out


class Store:
    """append-only 事件日志的持久化门面;单写者线程持有唯一连接。"""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._queue: queue.Queue[_Work | None] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        ready: Future[None] = Future()
        self._writer = threading.Thread(
            target=self._writer_loop, args=(ready,), name="praxis-store-writer", daemon=True
        )
        self._writer.start()
        ready.result()  # 打开连接/初始化 schema 的失败在此同步抛出

    # -- 写者线程 --------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None)  # autocommit,事务由 transact 手管
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA_SQL)
        conn.execute(
            "INSERT OR IGNORE INTO store_metadata(key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        return conn

    def _writer_loop(self, ready: Future[None]) -> None:
        try:
            conn = self._open()
        except BaseException as exc:  # noqa: BLE001 — 必须把失败经 Future 传回构造方
            ready.set_exception(exc)
            return
        ready.set_result(None)
        try:
            while True:
                work = self._queue.get()
                if work is None:  # close() 哨兵
                    break
                fn, future = work
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    future.set_result(fn(conn))
                except BaseException as exc:  # noqa: BLE001 — 每个 Future 必须有归宿
                    future.set_exception(exc)
        finally:
            conn.close()

    def _submit(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        future: Future[T] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError(f"store closed: {self._db_path}")
            self._queue.put((fn, future))
        return future.result()

    # -- 事务入口与写入 ----------------------------------------------------------

    def transact(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """唯一事务入口:BEGIN IMMEDIATE → fn(conn) → COMMIT;异常 ROLLBACK 重抛。"""

        def _run(conn: sqlite3.Connection) -> T:
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(conn)
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
            return result

        return self._submit(_run)

    def append_items(
        self,
        thread_id: str,
        turn_id: str | None,
        entries: Sequence[ItemEntry],
    ) -> list[AppendedItem]:
        """一批原子追加,返回刚写入的 (item_id, seq)(B12:返回即所写)。"""
        return self.transact(lambda conn: append_items_tx(conn, thread_id, turn_id, entries))

    # -- 读取(对 kind 零校验,B1)-------------------------------------------------

    def read_items(self, thread_id: str, *, after_seq: int = 0, limit: int | None = None) -> list[Item]:
        def _read(conn: sqlite3.Connection) -> list[Item]:
            sql = f"SELECT {_ITEM_COLUMNS} FROM items WHERE thread_id = ? AND seq > ? ORDER BY seq"
            params: list[Any] = [thread_id, after_seq]
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            return [_row_to_item(row) for row in conn.execute(sql, params)]

        return self._submit(_read)

    def read_turn_items(self, turn_id: str) -> list[Item]:
        def _read(conn: sqlite3.Connection) -> list[Item]:
            rows = conn.execute(
                f"SELECT {_ITEM_COLUMNS} FROM items WHERE turn_id = ? ORDER BY seq", (turn_id,)
            )
            return [_row_to_item(row) for row in rows]

        return self._submit(_read)

    # -- threads ----------------------------------------------------------------

    def create_thread(self) -> str:
        thread_id = uuid.uuid4().hex

        def _insert(conn: sqlite3.Connection) -> str:
            conn.execute("INSERT INTO threads(thread_id, created_at) VALUES (?, ?)", (thread_id, time.time()))
            return thread_id

        return self.transact(_insert)

    def ensure_thread(self, thread_id: str) -> None:
        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR IGNORE INTO threads(thread_id, created_at) VALUES (?, ?)", (thread_id, time.time())
            )

        self.transact(_insert)

    def list_threads(self) -> list[str]:
        def _read(conn: sqlite3.Connection) -> list[str]:
            rows = conn.execute("SELECT thread_id FROM threads ORDER BY created_at, thread_id")
            return [str(row["thread_id"]) for row in rows]

        return self._submit(_read)

    # -- 生命周期 ----------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        self._writer.join()

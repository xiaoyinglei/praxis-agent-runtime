"""Praxis verify 离线校验。

只读扫描,问题累积,永不修复。设计参考 §5.1。

已知状态的合法取值(核心统计机见 Task 7 构建):core/statemachine.TurnStatus;
现阶段以硬编码集合为准,新增 status 值时仅报错不抛业务异常。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from sqlite3 import Connection, Row
from typing import TYPE_CHECKING, Any

_KNOWN_TURN_STATUSES: frozenset[str] = frozenset(
    {"running", "paused", "interrupted", "completed", "failed", "cancelled"}
)


@dataclass(frozen=True)
class VerifyReport:
    """一次校验的只读报告。"""

    ok: bool
    problems: tuple[str, ...]
    checked_items: int


def _check_thread_rows(rows: list[Row]) -> list[str]:
    """Check seq contiguity + raw-payload JSON/type + orphan tool_call for one thread."""
    problems: list[str] = []
    expected_seq = 1
    tid = rows[0]["thread_id"] if rows else "?"

    # -- seq contiguous ---------------------------------------------------------
    for row in rows:
        if row["seq"] != expected_seq:
            problems.append(
                f"seq gap in thread {tid}: expected {expected_seq}, got {row['seq']}"
            )
        expected_seq = row["seq"] + 1

    # -- Single-pass: JSON parse/type-check + collect call_ids ------------------
    call_ids: dict[str, str] = {}   # call_id -> turn_id
    result_ids: dict[str, str] = {}  # call_id -> turn_id
    for row in rows:
        raw = str(row["payload"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            problems.append(
                f"corrupt payload at {tid}/seq={row['seq']} "
                f"(item_id={row['item_id']}): JSON parse failure"
            )
            continue

        # Finding 1: valid JSON but non-object type — not usable as tool metadata
        if not isinstance(parsed, dict):
            problems.append(
                f"non-object payload at {tid}/seq={row['seq']} "
                f"(item_id={row['item_id']}): expected dict, got {type(parsed).__name__}"
            )
            continue

        if row["kind"] == "tool_call":
            cid = parsed.get("call_id")
            if isinstance(cid, str):
                call_ids[cid] = row["turn_id"]
        elif row["kind"] == "tool_result":
            cid = parsed.get("call_id")
            if isinstance(cid, str):
                result_ids[cid] = row["turn_id"]

    # -- orphan tool_call detection ---------------------------------------------
    for cid, t_id in call_ids.items():
        if cid not in result_ids or result_ids[cid] != t_id:
            problems.append(
                f"orphan tool_call {cid!r} in thread {tid}, "
                f"turn {t_id}: no matching tool_result"
            )

    return problems


if TYPE_CHECKING:
    from praxis.store.log import Store as _StoreType
else:
    _StoreType = Any


def verify_store(store: _StoreType) -> VerifyReport:
    """Read-only verification scan of the entire store.

    Checks performed (never raises business exceptions, never repairs):

    1. Per-thread: ``seq`` values are contiguous integers starting from 1.
    2. Each item's stored payload parses as valid JSON.
    3. Every turn's ``status`` is in the known set.
    4. ``tool_call`` items without a matching ``tool_result`` in the same turn
       are listed as informational problems.

    All problems are accumulated into :class:`VerifyReport.problems`.
    """

    def _scan(conn: Connection) -> VerifyReport:
        problems: list[str] = []

        # -- Read all items grouped by thread -----------------------------------
        item_rows = conn.execute(
            "SELECT item_id, thread_id, turn_id, seq, kind, payload, created_at "
            "FROM items ORDER BY thread_id, seq"
        ).fetchall()

        thread_map: dict[str, list[Row]] = defaultdict(list)
        for r in item_rows:
            thread_map[r["thread_id"]].append(r)

        # -- Per-thread checks --------------------------------------------------
        for _tid, rows in thread_map.items():
            problems.extend(_check_thread_rows(rows))

        # -- Turn status check --------------------------------------------------
        turn_rows = conn.execute(
            "SELECT turn_id, thread_id, status FROM turns ORDER BY created_at, turn_id"
        ).fetchall()
        for tr in turn_rows:
            if tr["status"] not in _KNOWN_TURN_STATUSES:
                problems.append(
                    f"unknown turn status {tr['status']!r} for turn {tr['turn_id']!r} "
                    f"(thread {tr['thread_id']})"
                )

        result: VerifyReport = VerifyReport(
            ok=len(problems) == 0,
            problems=tuple(problems),
            checked_items=len(item_rows),
        )
        return result

    report = store.transact(_scan)
    return report

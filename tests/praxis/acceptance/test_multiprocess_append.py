"""Task 4 阶段验收:真实多进程并发 append 同一 db 文件(设计 §5.1;B12 回归)。

4 个 spawn 进程各建一个独立 Store(同一 db 路径),每进程 250 批 × 1 条 append,
全部写同一 thread。验收标准:
- 1000 条不丢不重,seq 恰好覆盖 1..1000(UNIQUE(thread_id, seq) + BEGIN IMMEDIATE);
- 每个进程拿到的 AppendedItem.item_id 精确对应自己写下的 payload——
  "返回即所写"在跨进程并发下成立,旧架构 B12(事后查最新)结构性消亡。

worker 函数必须是模块级(picklable,spawn 子进程按限定名导入本模块)。
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from praxis.store.log import AppendedItem, ItemEntry, Store

WORKERS = 4
ITEMS_PER_WORKER = 250
THREAD_ID = "t"


def _append_worker(job: tuple[str, int]) -> tuple[int, list[AppendedItem]]:
    db_path, worker_id = job
    store = Store(Path(db_path))
    try:
        appended: list[AppendedItem] = []
        for index in range(ITEMS_PER_WORKER):
            batch = store.append_items(
                THREAD_ID,
                None,
                [ItemEntry(kind="user_message", payload={"worker": worker_id, "index": index})],
            )
            assert len(batch) == 1
            appended.append(batch[0])
        return worker_id, appended
    finally:
        store.close()


def test_multiprocess_append_no_loss_no_dup(tmp_path: Path) -> None:
    db = tmp_path / "multiprocess.db"

    bootstrap = Store(db)  # 先建 schema,避免 4 进程争抢 DDL
    bootstrap.close()

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=WORKERS) as pool:
        results = pool.map(_append_worker, [(str(db), wid) for wid in range(WORKERS)])

    total = WORKERS * ITEMS_PER_WORKER
    store = Store(db)
    try:
        items = store.read_items(THREAD_ID)
        assert len(items) == total  # 不丢
        assert sorted(i.seq for i in items) == list(range(1, total + 1))  # 不重、连续
        assert len({i.item_id for i in items}) == total  # id 唯一

        rows = {i.item_id: i.payload for i in items}
        by_worker = dict(results)
        assert sorted(by_worker) == list(range(WORKERS))
        for worker_id, returned in by_worker.items():
            assert len(returned) == ITEMS_PER_WORKER
            for app in returned:
                assert rows[app.item_id]["worker"] == worker_id  # B12:返回即所写
    finally:
        store.close()

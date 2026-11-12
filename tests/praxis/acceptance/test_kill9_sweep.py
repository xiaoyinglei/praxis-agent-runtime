"""Task 8 阶段验收:真实 SIGKILL → sweep 标记 interrupted(设计 §5.8;硬规则 8)。

不是模拟:执行锁由**另一个 OS 进程**(_child_hold_lock.py,sys.executable 启动)持有,
被 SIGKILL 杀死——没有任何用户态清理,锁释放完全依赖内核。父进程作为"任何打开
本库的进程"的代表执行 sweep:

1. kill 前:sweep 必须跳过活 turn(取不到锁 = 执行者活着,绝不误伤);
2. SIGKILL 后:新开 Store sweep 必须标记 interrupted,turn_status item 携带
   last_seq(= 子进程留下的最后一条日志 seq),证明续跑锚点真实存在。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from praxis.core.events import EventBus
from praxis.core.recovery import sweep
from praxis.protocol.items import ItemKind
from praxis.store.log import Store
from praxis.store.turns import get_turn

CHILD_SCRIPT = Path(__file__).resolve().parent / "_child_hold_lock.py"
# 与 _child_hold_lock.py 中的固定常量一致(子进程无法向父进程回传,双方共享此约定)。
THREAD_ID = "kill9-sweep-thread"
TURN_ID = "kill9-sweep-turn"


def _wait_for_ready(tmp_path: Path, proc: subprocess.Popen[bytes]) -> None:
    ready = tmp_path / "ready"
    deadline = time.monotonic() + 10.0
    while not ready.exists():
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            pytest.fail(f"child exited early rc={proc.returncode}: {stderr[:2000]}")
        if time.monotonic() > deadline:
            pytest.fail("child did not write ready file within 10s")
        time.sleep(0.02)


def test_kill9_then_sweep_marks_interrupted(tmp_path: Path) -> None:
    proc = subprocess.Popen(
        [sys.executable, str(CHILD_SCRIPT), str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_ready(tmp_path, proc)

        # ── 活锁阶段:执行者进程还活着并持有 flock → sweep 必须空手而归 ──
        live_store = Store(tmp_path / ".praxis" / "praxis.sqlite")
        try:
            assert sweep(live_store, EventBus(), tmp_path) == []
            live = get_turn(live_store, TURN_ID)
            assert live is not None and live.status == "running"
        finally:
            live_store.close()

        # ── 真实 kill -9:无清理,锁由内核随进程死亡自动释放 ──
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()

        # ── 另一个进程打开新 Store:sweep 取得到锁 = 原主已死 → interrupted ──
        store = Store(tmp_path / ".praxis" / "praxis.sqlite")
        try:
            assert sweep(store, EventBus(), tmp_path) == [TURN_ID]
            dead = get_turn(store, TURN_ID)
            assert dead is not None and dead.status == "interrupted"

            item = store.read_turn_items(TURN_ID)[-1]
            assert item.kind == ItemKind.TURN_STATUS.value
            assert item.payload["to"] == "interrupted"
            assert item.payload["reason"] == "executor_process_died"
            assert item.payload["last_seq"] == 1  # 子进程留下的 user_message
        finally:
            store.close()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

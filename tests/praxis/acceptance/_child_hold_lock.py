"""kill -9 sweep 验收子进程:真实进程持 thread 执行锁,ready 后长睡,等着被 SIGKILL。

由父测试用 ``sys.executable`` 以独立进程启动(python _child_hold_lock.py <workspace>),
只依赖 praxis 包本身。流程:建库建 running turn → 取执行锁 → 写 ready 文件 → sleep。
ready 在取锁**之后**写,保证父进程 kill 时锁确实在持有中;SIGKILL 不给任何
清理机会,锁的释放完全依赖内核(flock 挂文件描述)——这正是被验收的性质。

TURN_ID/THREAD_ID 为固定常量,父测试以同值断言(两处必须一致)。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 脚本位于 tests/praxis/acceptance/,parents[3] = 仓库根;置于模块顶部保证 praxis 可导入。
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from praxis.core.locks import thread_execution_lock  # noqa: E402
from praxis.store.log import ItemEntry, Store  # noqa: E402
from praxis.store.turns import create_turn  # noqa: E402

THREAD_ID = "kill9-sweep-thread"
TURN_ID = "kill9-sweep-turn"  # 与 test_kill9_sweep.py 的常量保持一致


def main() -> None:
    workspace = Path(sys.argv[1])
    (workspace / ".praxis").mkdir(parents=True, exist_ok=True)
    store = Store(workspace / ".praxis" / "praxis.sqlite")
    store.ensure_thread(THREAD_ID)
    create_turn(store, turn_id=TURN_ID, thread_id=THREAD_ID, binding={})
    store.append_items(THREAD_ID, TURN_ID, [ItemEntry("user_message", {"text": "hold the lock"})])
    with thread_execution_lock(workspace, THREAD_ID):
        (workspace / "ready").write_text("locked\n", encoding="utf-8")
        time.sleep(300)


if __name__ == "__main__":
    main()

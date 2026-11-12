"""Praxis thread 粒度执行锁(设计 §5.1/§5.8)。

锁粒度 = thread:执行 turn 的进程持有该 thread 的 lockfile
``workspace/.praxis/locks/<thread_id>.lock``,经 flock 持有至 with 退出/进程退出。
steering/审批/confirm 都只是向日志 append item,不取执行锁(§5.8)——
取锁的只有"正在执行该 thread 的那一个进程"。

关键性质(kill -9 验收的根基):flock 由内核挂靠在文件打开描述上,
进程退出——无论正常、异常还是被 SIGKILL——都会自动释放,不存在残留锁。
recovery.sweep 的"试取锁判生死"完全建立在这点上:取不到 = 有活的执行者;
取到 = 原主已死。因此这里**绝不阻塞等待**(LOCK_NB):会等的锁无法区分两者。
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ThreadBusy(RuntimeError):  # noqa: N818 — 设计 §5.8/§5.1 钉死的名字
    """该 thread 的执行锁正被另一进程持有(thread 正在被执行)。"""


def _check_thread_id_filename(thread_id: str) -> None:
    """thread_id 直接拼成 lockfile 名:路径语义的 id 必须在碰文件系统前整体拒绝。

    生产 thread_id 是 uuid hex,但 SDK(Task 14)接受外部传入——"../x" 之类
    会把 lockfile 写到 .praxis/locks/ 之外,破坏"锁目录 = 本库全部执行锁"的
    不变量。规则:非空、不含 '/' 或 '\\'、不含 '..' 路径成分。
    """
    if not thread_id:
        raise ValueError("thread_id must be non-empty")
    if "/" in thread_id or "\\" in thread_id:
        raise ValueError(f"thread_id must not contain path separators: {thread_id!r}")
    if ".." in Path(thread_id).parts:
        raise ValueError(f"thread_id must not contain '..' components: {thread_id!r}")


@contextmanager
def thread_execution_lock(workspace: Path, thread_id: str) -> Iterator[None]:
    """持有 thread 的独占执行锁;已被占用 → 立即 ``ThreadBusy``,不排队等待。

    - 进入:先校验 thread_id 可安全作文件名(违规 → ValueError,不碰文件系统),
      再创建 ``workspace/.praxis/locks/`` 并打开/创建 lockfile,
      ``flock(fd, LOCK_EX | LOCK_NB)``;``BlockingIOError``(EWOULDBLOCK)→ ThreadBusy。
    - 退出(含 with 体内异常):LOCK_UN + close,锁必然归还。
    - 进程死亡:内核关闭 fd 即自动释放——这就是 sweep 能安全"捡尸"的原因。
    """
    _check_thread_id_filename(thread_id)
    lock_dir = workspace / ".praxis" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_dir / f"{thread_id}.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ThreadBusy(f"thread {thread_id!r} is executing in another process") from exc
    except BaseException:
        # 取锁前的任何失败:fd 必须关闭,否则句柄泄漏(锁本身未持有,无需 UN)。
        os.close(fd)
        raise
    try:
        yield
    finally:
        # 正常/异常两条退出路径共用:先解锁再关句柄,锁不粘留。
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

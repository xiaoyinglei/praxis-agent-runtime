"""子进程组纪律:内置工具共用的公开原语(R-T11-1)。

``bash``(command.py)与 ``search_text``(search.py)都需要同一套纪律:
``start_new_session=True`` 让子进程成为独立进程组组长 → 取消/超时时对**整组**
SIGTERM → 有界宽限 → SIGKILL 整组 → 收尾读取。信号发送、主进程退出判据与
管道排空逻辑只在这里实现一次;消费者只组合这些原语,不得另写 os.killpg。

原语边界(刻意保持最小):
- ``signal_group``:整组发信号,只吞 ProcessLookupError/PermissionError。
  这是全仓库唯一一份 ``os.killpg`` 实现;core 的孤儿清扫不 import tools,
  而是由 composition 把本函数作为 ``ProcessSignaler`` 注入(R-T15-7);
- ``await_main_exit``:等主进程回收事件(returncode),不等管道 EOF——
  CPython 的 ``Process.wait()`` 把"进程退出"与"所有管道 EOF"混为一谈,
  后台孙进程持写端时会拖到孙进程自然退出(B20);
- ``drain_stream``:持续读到 EOF,缓冲至多 ``limit`` 字节,超限字节仍读但丢弃
  (停止读取会让管道灌满、阻塞子进程)。

超时宽限、取消收敛、EOF 收敛策略属于各工具的执行纪律,由消费者用上述
原语组合;本模块不替它们决定"等多久"或"何时升级 KILL"。
"""

from __future__ import annotations

import asyncio
import os
import signal


def signal_group(pgid: int, sig: signal.Signals) -> bool:
    """向整个进程组发信号。

    ``start_new_session=True`` ⇒ 子进程组 id == 主进程 pid;整组收发信号。
    返回"组内可能仍有活体":False 仅当 ESRCH(组已空,持管道者必在组外)。
    EPERM 视为仍有活体(信号未送达不是"组为空"的证据)。
    """
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


async def await_main_exit(proc: asyncio.subprocess.Process, timeout: float | None) -> int:
    """等主进程"退出"这一事件(returncode 由子进程回收线程置位),不等管道 EOF。"""

    async def poll() -> int:
        while proc.returncode is None:
            await asyncio.sleep(0.02)
        return proc.returncode

    return await asyncio.wait_for(poll(), timeout)


async def drain_stream(
    stream: asyncio.StreamReader,
    buffer: list[bytes],
    discarded: list[int],
    *,
    limit: int,
) -> None:
    """读到 EOF;已收字节留在 buffer,任务被取消也不丢。

    buffer 至多 ``limit`` 字节;超限字节仍继续读但只计入 ``discarded``
    ——停止读取会让管道灌满、阻塞子进程,那是另一种绑架。
    """
    buffered = 0
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return
        room = limit - buffered
        if room >= len(chunk):
            buffer.append(chunk)
        elif room > 0:
            buffer.append(chunk[:room])
            discarded[0] += len(chunk) - room
        else:
            discarded[0] += len(chunk)
        buffered += len(chunk)

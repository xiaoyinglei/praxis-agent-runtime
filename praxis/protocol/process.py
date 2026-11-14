"""进程组信号原语(唯一实现;R-T15-2 让 core 也能消费)。

``signal_group`` 是"向整个进程组发信号"的唯一实现:
- ``praxis/tools/builtins/process.py`` 公开它给 bash/search/run_python 等执行器;
- ``praxis/core/recovery.py`` 的孤儿清扫直接从这里导入——core 不得 import
  tools(import-linter ``praxis_core_purity``),但可以用协议层原语。把实现放在
  最底层、由 tools 侧 re-export,保证全仓库只有一份 ``os.killpg``。

不放 ``await_main_exit``/``drain_stream``:它们只服务工具执行纪律,留在
``tools/builtins/process.py``。
"""

from __future__ import annotations

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
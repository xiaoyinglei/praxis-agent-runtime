"""沙箱后端可用性探测:唯一探测入口(设计 §5.3;D7 bwrap 延后)。

darwin 且 /usr/bin/sandbox-exec 可执行 → ("seatbelt", True);
其他平台一律 (None, False)——不探测 bwrap,Linux fail-closed(D2/D7)。
"""

from __future__ import annotations

import os
import sys

from praxis.protocol.permission import SandboxAvailability

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# 后端词汇的唯一来源(S1 反模式规则:profile/后端字面量只允许出现在
# adapters/sandbox/)。fail-closed 的消费方(如 MCP stdio 生命周期)拼自己的
# 拒绝文案时从这里取后端名字,不得把 "bwrap"/"sandbox-exec" 抄进自己的字符串。
BACKEND_CONFIGURATION_HINT = "configure bwrap/seatbelt or run on a supported platform"


def probe_backend() -> SandboxAvailability:
    """唯一可用性探测:seatbelt only(bwrap 延后,其他平台不可用)。"""
    if sys.platform == "darwin" and os.path.isfile(_SANDBOX_EXEC) and os.access(_SANDBOX_EXEC, os.X_OK):
        return SandboxAvailability(backend="seatbelt", available=True)
    return SandboxAvailability(backend=None, available=False)
"""Praxis 沙箱适配器:唯一 profile/argv 构造包(S1 结构消亡;设计 §5.3/§4.2)。

外部只经 compile_sandbox/finalize_sandbox/probe_backend 取得沙箱执行形态;
profile 字符串与 argv 构造只出现在本包内(Task 10 CI 反模式钉死包外构造)。
bwrap 延后(D7):其他平台 probe 不可用,compile fail-closed 拒绝。
"""

from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox, validate_aliases
from praxis.adapters.sandbox.seatbelt import compile_seatbelt, render_profile

__all__ = [
    "compile_sandbox",
    "compile_seatbelt",
    "finalize_sandbox",
    "probe_backend",
    "render_profile",
    "validate_aliases",
]
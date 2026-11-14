"""协议层沙箱 ACI:执行上下文 + 编译产物形状(设计 §5.4;阶段 2 Task 1;D3/D5)。

零 praxis 依赖——纯数据与一个有界异常。"policy→SandboxSpec" 的编译逻辑在
core.sandbox(Task 6/7 落地);本模块只定义跨层传递的不可变形状。

约定:
- backend ∈ {"seatbelt", "bwrap"}(macOS sandbox-exec / Linux bubblewrap);
- argv_prefix 是子进程命令前缀(拼在真实命令 argv 之前);
- scratch_dir 是沙箱内可写临时区(绝对路径);
- view_dir 仅 bwrap 使用(commit 视图挂载点);
- profile_text 仅 seatbelt 使用(编译出的 profile 原文,供审计)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from praxis.protocol.permission import PermissionContext

_MAX_REFUSAL_REASON_CHARS = 512


class SandboxRefusal(RuntimeError):  # noqa: N818 — 异常名由阶段 2 计划 ACI 固定(后续任务编译器/管道同名引用)
    """沙箱编译拒绝:命令别名、不可表示的边界、后端不可用等一律抛此异常。

    管道(Task 7 接线)捕获后转 denied 结果;reason 构造时截断至 512 字符——
    它会进入模型可见文本,不得无界。
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason[:_MAX_REFUSAL_REASON_CHARS]
        super().__init__(self.reason)


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    """一份可直接拼进子进程 argv 的沙箱执行形态(编译产物,不可变)。"""

    backend: str
    argv_prefix: tuple[str, ...]
    scratch_dir: str
    view_dir: str | None = None
    profile_text: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("sandbox backend must be a non-empty string")
        prefix = tuple(self.argv_prefix)
        if not prefix or not all(isinstance(part, str) and part for part in prefix):
            raise ValueError("sandbox argv_prefix must be a non-empty tuple of non-empty strings")
        if not isinstance(self.scratch_dir, str) or not self.scratch_dir.startswith("/"):
            raise ValueError("sandbox scratch_dir must be an absolute path")
        object.__setattr__(self, "argv_prefix", prefix)


@dataclass(frozen=True, slots=True)
class ToolExecContext:
    """工具执行环境上下文:随每次 tool.run 调用传入(双参 ToolRunner)。

    阶段 2 Task 1:仅 sandbox 字段(无沙箱即 None,阶段 1 行为不变);
    后续字段(审批通道、取消令牌等)按需在后续任务扩展,只增不改。

    ``permissions``(R-T11-2):该次调用的权限快照。目录型读取工具(search_text/
    list_files)在触达**每一个具体文件**时用它做执行侧 deny 过滤:``decide`` 仍
    是唯一裁决函数,这里只是把已由 ``decide`` 授权的目录级 target 细化为逐条目
    检查(规则代数复用 ``protocol.policy.path_access``,不是第二套决策入口)。
    None = 无快照(单测直跑路径),不做逐条目过滤。
    """

    sandbox: SandboxSpec | None = None
    permissions: PermissionContext | None = None

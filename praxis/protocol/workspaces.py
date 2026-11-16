"""协议层子工作区 provisioner 类型(设计 §5.13;阶段4 T5 定义,T7 实装)。

``ChildWorkspaceProvisioner`` 是 write 模式子代理的工作区供给面:prepare 建
独立快照目录、diff/apply/discard 服务 T8 的 integrate 交付管道。仿
``protocol/sandbox.py`` 的 EnvBuilder 注入先例——core 只认识这个结构签名,
git worktree 实现(``adapters/workspaces/git.py``)由 composition 绑定。

T5 阶段 write 模式 provisioner=None 时 spawn 明确拒绝(read_only 不需要
provisioner:共享父工作区,权限收窄在构造期完成)。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

# 子工作区模式的唯一类型别名(spawn/build_child_session 共用)。
type WorkspaceMode = Literal["read_only", "write"]

# inspect 预览的字节上限(旧 child_workspaces.py:184 diff_limit ≤200_000 parity)。
DIFF_PREVIEW_LIMIT_BYTES = 200_000


class ChildWorkspaceRequiresGit(RuntimeError):  # noqa: N818 — T7 brief 钉死的名字
    """非 git 工作区无法供给 write 子工作区快照(Q5:明确拒绝,不静默降级)。"""


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """一次 apply 的逐文件结局:preimage 复检不一致的文件进 conflicts 保留不写。"""

    applied: list[str]
    conflicts: list[str]


@dataclass(frozen=True, slots=True)
class WorkspaceInspection:
    """T8 inspect 的读形:变更路径集 + 双方 sha256 表 + 有界 diff 预览。

    - ``preimages``:HEAD(快照基线)侧逐路径 sha256;基线不存在 = None。
      apply 的 preimage 复检以此为期望值(父仓快照后漂移 → conflict)。
    - ``postimages``:子工作区当前侧逐路径 sha256(删除 = None;非普通文件 = "")。
    - ``diff_preview``:tracked diff 的 ≤200_000 字节预览(UTF-8 边界安全);
      ``diff_truncated``/``diff_bytes`` 如实披露截断与全文尺寸。
    """

    paths: tuple[str, ...]
    preimages: Mapping[str, str | None]
    postimages: Mapping[str, str | None]
    diff_preview: str
    diff_truncated: bool
    diff_bytes: int


class ChildWorkspaceProvisioner(Protocol):
    """write 子工作区的供给/交付面(T7 GitWorktreeProvisioner 结构匹配)。

    - ``prepare``:在库外宿主目录建 detach 快照,返回子工作区路径;
      非 git 工作区抛 ``ChildWorkspaceRequiresGit``。
    - ``diff``:子工作区相对 HEAD 的变更 bundle(含 untracked;格式常量唯一
      在 T7 实现侧);``paths`` 给定 = 只含该子集(精确文件或目录前缀)。
    - ``inspect``:T8 交付管道的读面(路径集/双方 sha 表/有界预览);
      bundle 格式对 core/tools 保持不透明。
    - ``apply``:逐文件 preimage sha256 复检后整组应用;不一致 → conflicts,
      失败整组拒(旧事务性 patch_failed 语义 parity)。
    - ``discard``:移除子工作区快照。
    """

    def prepare(self, parent_root: Path, child_thread_id: str, mode: str) -> Path: ...

    def diff(self, ws: Path, *, paths: Sequence[str] | None = None) -> bytes: ...

    def inspect(self, ws: Path, *, paths: Sequence[str] | None = None) -> WorkspaceInspection: ...

    def apply(
        self, parent_root: Path, patch: bytes, *, preimages: Mapping[str, str | None]
    ) -> ApplyResult: ...

    def discard(self, ws: Path) -> None: ...

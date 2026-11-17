"""协议层子代理数据形状与工具面结构签名(设计 §5.13;阶段4 T6)。

仿 ``protocol/sandbox.py`` 的 EnvBuilder 注入先例:跨层传递的不可变形状与
结构签名住在 protocol,行为实现(``core.agents.SubagentManager``)住在 core,
模型工具面(``tools.builtins.agent_tools``)只认识本模块——tools 不 import
core(import-linter praxis_tools_adapters_boundary),composition 绑定。

T5 的形状(ChildHandle/ChildStatus/ChildSummary/两个拒绝异常/五件名字常量)
自 core.agents 迁居于此:工具面协议签名需要引用它们,而 protocol 不得反向
import core。core.agents 原样 re-export,既有调用方零改动。
``ApprovalSummary`` 同迁 protocol.permission(ChildStatus 的待批摘要形状)。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

from praxis.protocol.budget import Limits
from praxis.protocol.json import JsonValue
from praxis.protocol.permission import ApprovalSummary
from praxis.protocol.workspaces import ApplyResult, WorkspaceInspection, WorkspaceMode

# 子代理状态标签(wait/children_summary 共用):completed/failed 都表达
# "不再消耗",但 failed 保留独立标签供上层区分结局(与 ChildState 的 done
# 折叠不同——展示面不合并失败)。
type ChildStateLabel = Literal["running", "done", "paused", "interrupted", "cancelled", "failed"]

# T6 模型工具面五件套的名字常量:子工具表 = 父表减这五件(depth 结构门——
# 孙代即使谎报 depth 也拿不到 spawn 工具)。
AGENT_ORCHESTRATION_NAMES: Final[frozenset[str]] = frozenset(
    {"spawn_subagent", "wait_subagent", "send_to_subagent", "cancel_subagent"}
)
AGENT_TOOL_NAMES: Final[frozenset[str]] = AGENT_ORCHESTRATION_NAMES | frozenset(
    {"integrate_child_workspace"}
)


class ChildGrantError(RuntimeError):  # noqa: N818 — 计划钉死的名字
    """份额公式算出的 grant 在有界维 ≤ 0:父该维已耗尽,拒绝 spawn(B2)。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ChildLimitError(RuntimeError):  # noqa: N818 — 与 ChildGrantError 同族
    """并发/总数/深度/父终态 guard 拒绝(T6 映射 child_limit_reached 错误码)。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ChildActiveError(RuntimeError):  # noqa: N818 — 与 ChildGrantError 同族
    """子根 turn 非终态时拒绝 integrate(T8 映射 child_still_active 错误码)。

    读不到子 turn 同样保守拒绝(与 _child_is_active 的 fail-open 同向:
    结局不可知时绝不消费子工作区)。
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ChildHandle:
    """spawn 的返回身份:注册表 item 与子根 turn 的互指键 + 工作区事实。"""

    child_thread_id: str
    child_root_turn_id: str
    allocation_id: str
    workspace: Path
    workspace_mode: WorkspaceMode
    grant: Limits


@dataclass(frozen=True, slots=True)
class ChildStatus:
    """wait 的如实展示面:子的状态/结论/待批/未知调用/已用步数与成本。

    paused/interrupted 可见但不代批(Q3 直面裁决:人类对子 session 各自
    resume/confirm,父不替子审批)。
    """

    state: ChildStateLabel
    conclusion: str
    pending_approvals: tuple[ApprovalSummary, ...]
    unknown_calls: tuple[str, ...]
    steps_used: int
    cost_used: int
    workspace: str


@dataclass(frozen=True, slots=True)
class ChildSummary:
    """TurnResult.children_summary 的一条注记:父 turn 结束时子的状态快照。"""

    child_thread_id: str
    child_root_turn_id: str
    state: ChildStateLabel


class SubagentSurface(Protocol):
    """模型工具面消费的子代理编排结构签名(core.agents.SubagentManager 结构匹配)。

    composition 经 provider 闭包延迟绑定(manager 在 Session 之后构造,
    与 web authorizer 的 session_holder 同先例);tools 只经此协议触达行为。
    拒绝语义:份额耗尽抛 ``ChildGrantError``,guard 拒绝抛 ``ChildLimitError``,
    未知 child 抛 ``ValueError``——工具面映射为诚实错误码。
    """

    async def spawn(
        self,
        task: str,
        *,
        context_summary: str = "",
        requested: Limits | None = None,
        workspace_mode: WorkspaceMode = "read_only",
        depth: int = 0,
    ) -> ChildHandle: ...

    async def wait(self, child_thread_id: str, *, timeout: float = 10.0) -> ChildStatus: ...

    async def send(self, child_thread_id: str, message: str) -> None: ...

    def request_cancel(self, child_thread_id: str) -> str: ...

    # T8 integrate 交付面:子须终态(否则 ChildActiveError);read_only 子/
    # provisioner 未接线抛 ValueError(工具面如实上抛,管道归一 tool_error)。
    def integrate_inspect(
        self, child_thread_id: str, *, paths: Sequence[str] | None = None
    ) -> WorkspaceInspection: ...

    def integrate_apply(
        self, child_thread_id: str, *, paths: Sequence[str] | None = None
    ) -> ApplyResult: ...

    def integrate_discard(self, child_thread_id: str) -> None: ...


# 延迟绑定 provider:composition 的 manager_holder 闭包(工具表先于 manager 装配)。
type SurfaceProvider = Callable[[], SubagentSurface]


# ---------------------------------------------------------------------------
# 子装配身份(R-FINAL-C1:跨 resume 存活的装配事实,钉进子根 turn binding)
# ---------------------------------------------------------------------------

# binding 里子装配身份的唯一键名(spawn 写端 / resume 读端共用,不另写第二份)。
CHILD_ASSEMBLY_KEY: Final = "child_assembly"


@dataclass(frozen=True, slots=True)
class ChildAssembly:
    """子 Session 的装配身份读形(R-FINAL-C1)。

    spawn 时把这些事实钉进子根 turn binding,使跨进程/CLI 的裸 resume 能经
    build_child_session 等价装配重建子身份——而不只是同实例 resume 才存活:
    - ``workspace_mode``:read_only 收窄(ceiling/persist_always)的判据;
    - ``workspace``:子工作区根绝对路径(read_only=父工作区,write=独立 worktree);
    - ``parent_db_path``:父库文件绝对路径(同库纪律的路由出口——write 子从
      自身工作区 resume 时 Store 仍指向父库,绝不新建空库);
    - ``parent_workspace``:父工作区根绝对路径(策略声明来源/发现面)。
    """

    workspace_mode: WorkspaceMode
    workspace: str
    parent_db_path: str
    parent_workspace: str


def child_assembly_payload(
    *,
    workspace_mode: WorkspaceMode,
    workspace: str,
    parent_db_path: str,
    parent_workspace: str,
) -> dict[str, JsonValue]:
    """子装配身份的唯一拼装助手(spawn 写端;键常量同模块,规则 26 同纪律)。"""
    return {
        "workspace_mode": workspace_mode,
        "workspace": workspace,
        "parent_db_path": parent_db_path,
        "parent_workspace": parent_workspace,
    }


def child_assembly_from_binding(binding: Mapping[str, JsonValue]) -> ChildAssembly | None:
    """从 turn binding 读子装配身份;缺键/畸形一律 None(fail-open 读端)。

    None = 这不是子根 turn(普通 root turn 或历史 turn)——resume 走既有
    root 装配,零扰动。四字段必须全是非空 str 且 workspace_mode 合法,否则
    视为畸形 None(绝不拿半份身份重建出一个错装配的子)。
    """
    raw = binding.get(CHILD_ASSEMBLY_KEY)
    if not isinstance(raw, Mapping):
        return None
    mode = raw.get("workspace_mode")
    workspace = raw.get("workspace")
    parent_db_path = raw.get("parent_db_path")
    parent_workspace = raw.get("parent_workspace")
    if (
        not isinstance(mode, str)
        or mode not in ("read_only", "write")
        or not isinstance(workspace, str)
        or not workspace
        or not isinstance(parent_db_path, str)
        or not parent_db_path
        or not isinstance(parent_workspace, str)
        or not parent_workspace
    ):
        return None
    return ChildAssembly(
        workspace_mode=cast("WorkspaceMode", mode),
        workspace=workspace,
        parent_db_path=parent_db_path,
        parent_workspace=parent_workspace,
    )

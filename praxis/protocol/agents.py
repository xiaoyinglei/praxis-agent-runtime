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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

from praxis.protocol.budget import Limits
from praxis.protocol.permission import ApprovalSummary
from praxis.protocol.workspaces import WorkspaceMode

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


# 延迟绑定 provider:composition 的 manager_holder 闭包(工具表先于 manager 装配)。
type SurfaceProvider = Callable[[], SubagentSurface]

"""协议层权限判定结果:Decision 与权限上下文(设计 §5.3、§5.9)。

Decision 是 core 判定管线(T12)对外的最小结论形状:verdict + reason + 建议规则;
PermissionUpdate 描述"如果同意,可以把哪条规则落到哪里";
HookDecision 为 §5.14 的 hook 决策预留;PermissionContext 是纯数据快照,
解析逻辑在阶段 2 core。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from praxis.protocol.policy import (
    NetworkRule,
    PathRule,
    PermissionPolicy,
    PolicyRule,
    PolicyVerdict,
    ToolRule,
)

UpdateDestination = Literal["once", "session", "always"]
PermissionMode = Literal["on-request", "never", "bypass"]


@dataclass(frozen=True, slots=True)
class PermissionUpdate:
    """一条可落盘的权限建议:规则本体 + 生效范围。"""

    rule: PathRule | ToolRule | NetworkRule
    destination: UpdateDestination


@dataclass(frozen=True, slots=True)
class PermissionNotice:
    """权限更新生效后的如实披露(D9;阶段 2 留档②)。

    ``kind`` 是唯一机器可读分支面,``text`` 是人类可读的单行文案。当前唯一
    产者:session/always 的 NetworkRule 授权使命令网络 fail-closed(旧代码
    静默,是运维陷阱)。UI 只渲染、不据此改行为。
    """

    kind: str
    text: str


@dataclass(frozen=True)
class ApprovalSummary:
    """一条待批调用的人类可读摘要(源自 approval_request item,非第二事实源)。

    阶段4 T6 自 core.turn 迁居 protocol:ChildStatus(protocol.agents)的
    待批摘要形状需要它,而 protocol 不得 import core。core.turn 原样
    re-export,既有调用方(sdk/Session/测试)零改动。
    """

    call_id: str
    tool_name: str
    args_digest: str
    effects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    """一次权限判定的不可变结论。构造一律走 allow/ask/deny 三个 classmethod。"""

    verdict: PolicyVerdict
    reason: str
    suggestions: tuple[PermissionUpdate, ...] = ()

    @classmethod
    def allow(cls, reason: str) -> Decision:
        return cls(verdict="allow", reason=reason)

    @classmethod
    def ask(cls, reason: str) -> Decision:
        return cls(verdict="ask", reason=reason)

    @classmethod
    def deny(cls, reason: str) -> Decision:
        return cls(verdict="deny", reason=reason)

    @property
    def is_allow(self) -> bool:
        return self.verdict == "allow"

    @property
    def is_ask(self) -> bool:
        return self.verdict == "ask"

    @property
    def is_deny(self) -> bool:
        return self.verdict == "deny"


@dataclass(frozen=True, slots=True)
class HookDecision:
    """hook 返回的权限裁决(§5.14)。

    ``feedback`` 是 R-PF-1 的尾段(构造期已按 2048 字节截断,见
    ``protocol.hooks.HookOutcome``):deny/ask 的 reason 由 core 拼成
    ``hook:<source>: <feedback>``;缺省空串,既有 3A 构造零改动。
    """

    source: str
    verdict: PolicyVerdict
    feedback: str = ""


@dataclass(frozen=True, slots=True)
class SandboxAvailability:
    """沙箱能力探测快照(protocol 层纯数据;探测实现在 core.sandbox,Task 6)。

    backend=None + available=False 表示未启用/不可用——阶段 1 的默认形态。
    """

    backend: str | None
    available: bool


@dataclass(frozen=True, slots=True)
class PermissionContext:
    """权限判定的输入快照:工作区 + 策略 + 审批模式 + hook 侧裁决 + 规则层 + 沙箱。

    阶段 2 Task 1 扩展:mode 类型放宽为 PermissionMode(旧 Approval 值是子集,
    composition 传 policy.approval 仍合法);规则层字段 ceilings/session_rules/
    once_rules/always_rules 与 sandbox 由 core.permission(Task 2+)解析填充;
    全部新字段带默认,阶段 1 构造零改动。
    """

    workspace: str
    policy: PermissionPolicy
    mode: PermissionMode
    hook_decisions: tuple[HookDecision, ...] = ()
    ceilings: tuple[PermissionPolicy, ...] = ()
    session_rules: tuple[PolicyRule, ...] = ()
    once_rules: tuple[PolicyRule, ...] = ()
    always_rules: tuple[PolicyRule, ...] = ()
    sandbox: SandboxAvailability = SandboxAvailability(backend=None, available=False)

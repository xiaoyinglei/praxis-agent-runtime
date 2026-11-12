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
    Approval,
    NetworkRule,
    PathRule,
    PermissionPolicy,
    PolicyVerdict,
    ToolRule,
)

UpdateDestination = Literal["once", "session", "always"]


@dataclass(frozen=True, slots=True)
class PermissionUpdate:
    """一条可落盘的权限建议:规则本体 + 生效范围。"""

    rule: PathRule | ToolRule | NetworkRule
    destination: UpdateDestination


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
    """hook 返回的权限裁决(§5.14 预留位)。"""

    source: str
    verdict: PolicyVerdict


@dataclass(frozen=True, slots=True)
class PermissionContext:
    """权限判定的输入快照:工作区 + 策略 + 审批模式 + hook 侧裁决。"""

    workspace: str
    policy: PermissionPolicy
    mode: Approval
    hook_decisions: tuple[HookDecision, ...] = ()

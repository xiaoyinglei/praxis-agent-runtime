"""Praxis 唯一权限决策入口(设计 §5.4;findings S 类结构预防)。

decide() 的签名是冻结的接线契约:管道(T12 execute)只经模块属性
``permission.decide`` 调用它——阶段 2 把函数体换成步骤表(policy 解析、
hook、地板规则),所有调用方零改动。测试经 monkeypatch 本模块的 decide
验证"唯一入口"不是口号:换实现不换接线。

Phase 1 决策恒 allow:没有策略解析前,任何 deny/ask 都是臆造的严格性。
"""

from __future__ import annotations

from praxis.protocol.permission import Decision, PermissionContext
from praxis.protocol.tools import ResolvedUse, Tool


def decide(tool: Tool, use: ResolvedUse, ctx: PermissionContext) -> Decision:
    """一次工具调用的权限裁决。

    输入:工具 + 本次调用实际触达的副作用/目标(ResolvedUse)+ 权限上下文快照。
    输出:allow/ask/deny 三态 Decision(唯一结论形状)。
    """
    return Decision.allow("phase1-default")

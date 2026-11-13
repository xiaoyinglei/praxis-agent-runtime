"""protocol.permission 权限上下文 ACI 回归(设计 §5.3;阶段 2 Task 1 扩展)。

阶段 1 构造 PermissionContext(workspace, policy, mode) 必须零改动:
新增字段全部带默认值,mode 类型放宽为 PermissionMode 后旧 Approval 值仍是子集。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from praxis.protocol.permission import PermissionContext, SandboxAvailability
from praxis.protocol.policy import PathRule, PermissionPolicy, ToolRule


def test_context_defaults_backward_compatible() -> None:
    ctx = PermissionContext(workspace="/w", policy=PermissionPolicy(), mode="on-request")
    assert ctx.ceilings == () and ctx.session_rules == () and ctx.once_rules == ()
    assert ctx.sandbox == SandboxAvailability(backend=None, available=False)


def test_context_frozen() -> None:
    ctx = PermissionContext(workspace="/w", policy=PermissionPolicy(), mode="bypass")
    with pytest.raises(FrozenInstanceError):
        ctx.mode = "never"


def test_context_accepts_rule_layers_and_sandbox() -> None:
    ctx = PermissionContext(
        workspace="/w",
        policy=PermissionPolicy(),
        mode="never",
        ceilings=(PermissionPolicy(approval="never"),),
        session_rules=(ToolRule(pattern="bash", decision="ask"),),
        once_rules=(),
        always_rules=(PathRule(path="/w/out", access="write"),),
        sandbox=SandboxAvailability(backend="seatbelt", available=True),
    )
    assert len(ctx.ceilings) == 1
    assert ctx.session_rules[0].pattern == "bash"
    assert ctx.always_rules[0].path == "/w/out"
    assert ctx.sandbox.backend == "seatbelt" and ctx.sandbox.available
    assert ctx.once_rules == ()

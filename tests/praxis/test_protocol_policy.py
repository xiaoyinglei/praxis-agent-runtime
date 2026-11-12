"""protocol.policy / protocol.permission:冻结策略模型与校验器移植测试(设计 §5.3、§5.9)。"""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from praxis.protocol.permission import (
    Decision,
    HookDecision,
    PermissionContext,
    PermissionUpdate,
)
from praxis.protocol.policy import (
    NetworkPolicy,
    NetworkRule,
    PathRule,
    PermissionPolicy,
    ToolRule,
)


def test_policy_frozen_and_extra_forbid():
    p = PermissionPolicy()
    with pytest.raises(ValidationError):
        PermissionPolicy(profile="workspace", evil=1)
    with pytest.raises(ValidationError):
        p.profile = "read-only"


def test_policy_revision_deterministic():
    assert PermissionPolicy().revision == PermissionPolicy().revision
    assert PermissionPolicy().revision != PermissionPolicy(approval="never").revision


def test_readonly_cannot_widen():
    with pytest.raises(ValidationError):
        PermissionPolicy(profile="read-only", filesystem=(PathRule(path="x", access="write"),))


def test_domain_normalization():
    assert NetworkPolicy(allowed_domains=("EXAMPLE.com.",)).allowed_domains == ("example.com",)


def test_decision_constructors():
    assert Decision.ask("default").is_ask and Decision.ask("default").reason == "default"


# --- 校验器移植保真度 -------------------------------------------------------


def test_path_rule_rejects_empty_and_nul() -> None:
    with pytest.raises(ValidationError):
        PathRule(path="", access="read")
    with pytest.raises(ValidationError):
        PathRule(path="a\0b", access="write")
    assert PathRule(path="src/main.py", access="read").access == "read"


def test_tool_rule_rejects_empty_pattern() -> None:
    with pytest.raises(ValidationError):
        ToolRule(pattern="", decision="allow")
    assert ToolRule(pattern="read_*", decision="ask").decision == "ask"


def test_network_rule_domain_is_normalized() -> None:
    assert NetworkRule(domain="API.Example.COM.", decision="deny").domain == "api.example.com"
    assert NetworkRule(domain="*.Example.com", decision="ask").domain == "*.example.com"


@pytest.mark.parametrize("bad", ["http://example.com", "example.com:443", "example.com/path", "", "."])
def test_network_policy_rejects_malformed_domains(bad: str) -> None:
    with pytest.raises(ValidationError):
        NetworkPolicy(allowed_domains=(bad,))


def test_list_inputs_coerced_to_tuples() -> None:
    policy = PermissionPolicy(filesystem=[PathRule(path="x", access="read")])
    assert isinstance(policy.filesystem, tuple)
    assert policy.filesystem[0].path == "x"


# --- permission.py ----------------------------------------------------------


def test_decision_all_three_verdicts() -> None:
    allow = Decision.allow("public read")
    deny = Decision.deny("outside workspace")
    assert allow.is_allow and not allow.is_ask and not allow.is_deny
    assert deny.is_deny and deny.reason == "outside workspace"
    assert Decision.ask("needs approval").suggestions == ()


def test_decision_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Decision.allow("x").reason = "y"


def test_permission_update_is_frozen() -> None:
    update = PermissionUpdate(rule=PathRule(path="x", access="read"), destination="session")
    with pytest.raises(dataclasses.FrozenInstanceError):
        update.destination = "always"


def test_hook_decision_and_permission_context() -> None:
    hook = HookDecision(source="pre-tool-use", verdict="deny")
    assert hook.source == "pre-tool-use" and hook.verdict == "deny"
    ctx = PermissionContext(
        workspace="/tmp/ws",
        policy=PermissionPolicy(),
        mode="on-request",
        hook_decisions=(hook,),
    )
    assert ctx.mode == "on-request"
    assert ctx.hook_decisions[0].verdict == "deny"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.workspace = "/other"


def test_permission_context_defaults_to_no_hook_decisions() -> None:
    ctx = PermissionContext(
        workspace="/tmp/ws", policy=PermissionPolicy(), mode="never"
    )
    assert ctx.hook_decisions == ()

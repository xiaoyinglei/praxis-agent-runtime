"""Task 8:策略文件入口 + 权限更新代数(设计 §5.3/§5.12;C5/D6/D10/D9;裁决 R6)。

三层覆盖:
1. apply_update R6 代数(decide 级):session/always 同步扩展 ctx.policy
   (grant 经既有第 1/6 步生效,decide/步骤表零改动)且保留记录层追加;
   once 只进 once_rules(调用域,policy 不动);
2. Session.update_permission(行为级,经真实 submit):always 原子落盘
   .praxis/permissions.json 并跨 close/reopen 生效;session/once 永不落盘;
   once 在下一 _execute 批后清除;
3. composition/SDK 入口:policy 来源序(显式 > 文件 > 默认)、坏文件边界
   拒绝(ValueError,不静默)、mode 正交(C5:bypass 永不进 policy/永不落盘)。

D9 钉死:session/always 的 NetworkRule 授权对 URL 级工具放行,但使
command_network_decision 对 EXECUTE_PROCESS(bash)fail-closed deny。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from praxis.core.permission import apply_update, decide
from praxis.core.statemachine import TurnStatus
from praxis.interfaces import sdk as sdk_pkg
from praxis.interfaces.sdk import ScriptedAdapter, open_session
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import (
    PermissionContext,
    PermissionUpdate,
    SandboxAvailability,
)
from praxis.protocol.policy import (
    NetworkPolicy,
    NetworkRule,
    PathRule,
    PermissionPolicy,
    ToolRule,
)
from praxis.protocol.tools import (
    ResolvedUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
)
from tests.praxis.conftest import make_minimal_tool

_URL = "https://ok.com/x"


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


@pytest.fixture
def ws(workspace: Path) -> Path:
    """conftest `workspace` 的本文件别名(与 test_permission_pipeline 同名习惯)。"""
    return workspace


def base_ctx(ws: Path) -> PermissionContext:
    return PermissionContext(
        workspace=str(ws),
        policy=PermissionPolicy(),
        mode="on-request",
        sandbox=SandboxAvailability(backend=None, available=False),
    )


def make_named_tool(name: str, effects: frozenset[ToolEffect], targets: tuple[ToolTarget, ...]) -> Tool:
    return make_minimal_tool(
        definition=ToolDefinition(name=name, description=f"{name} task8 tool", input_schema={"type": "object"}),
        static_effects=effects,
        resolve_use=lambda arguments: ResolvedUse(effects=effects, targets=targets),
    )


def probe_tool(name: str = "probe") -> Tool:
    """NETWORK 无 url target:默认策略下第 6 步覆盖不成立 → fallback ask(Task 4 钉死)。"""
    return make_named_tool(name, frozenset({ToolEffect.NETWORK}), ())


def url_tool(name: str = "probe", url: str = _URL) -> Tool:
    return make_named_tool(name, frozenset({ToolEffect.NETWORK}), (ToolTarget(kind="url", value=url),))


def exec_tool(name: str = "bash") -> Tool:
    return make_named_tool(name, frozenset({ToolEffect.EXECUTE_PROCESS}), ())


def write_use(target: str) -> ResolvedUse:
    return ResolvedUse(
        effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
        targets=(ToolTarget(kind="path", value=target),),
    )


def empty_use() -> ResolvedUse:
    return ResolvedUse(effects=frozenset({ToolEffect.NETWORK}), targets=())


def script_call(call_id: str, name: str, arguments: dict[str, object] | None = None) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ModelToolCall(call_id=call_id, name=name, arguments=arguments or {}),),
        usage=Usage(0, 0),
        finish_reason="tool_calls",
    )


def script_stop(text: str) -> ModelResponse:
    return ModelResponse(content=text, tool_calls=(), usage=Usage(0, 0), finish_reason="stop")


def tool_result_for(session, call_id: str) -> Item:
    matches = [
        item
        for item in session.items()
        if item.kind == ItemKind.TOOL_RESULT and item.payload.get("call_id") == call_id
    ]
    assert matches, f"no tool_result for {call_id!r} in {[i.kind for i in session.items()]}"
    return matches[-1]


def permissions_file(ws: Path) -> Path:
    return ws / ".praxis" / "permissions.json"


# ---------------------------------------------------------------------------
# R6 apply_update 代数(decide 级)
# ---------------------------------------------------------------------------


def test_r6_once_destination_touches_only_once_rules(ws: Path) -> None:
    from dataclasses import replace  # 局部 import 仅限测试(源码头文件已顶层导入)

    ctx = base_ctx(ws)
    rule = ToolRule(pattern="probe", decision="allow")
    out = apply_update(ctx, PermissionUpdate(rule=rule, destination="once"))
    # R6:once 保持纯调用域——policy 恒等,不扩展、revision 不变
    assert out.policy is ctx.policy
    assert out.once_rules == (rule,) and out.session_rules == () and out.always_rules == ()
    # 生效通道仍是既有 _rule_layers(_tool_grant 含 session/once 层)
    assert decide(probe_tool(), empty_use(), out).is_allow
    # 调用域:once 层清空后授权即失效
    assert decide(probe_tool(), empty_use(), replace(out, once_rules=())).is_ask


def test_r6_session_path_grant_extends_policy_and_widens(ws: Path) -> None:
    ctx = base_ctx(ws)
    outside = ws.parent / "outside_data"
    outside.mkdir()
    use = write_use(str(outside / "f.txt"))
    tool = make_named_tool("write", frozenset({ToolEffect.WRITE_WORKSPACE}), ())
    assert decide(tool, use, ctx).is_deny  # 预授权缺口:workspace 外路径第 1 步拒绝
    rule = PathRule(path=str(outside), access="write")
    out = apply_update(ctx, PermissionUpdate(rule=rule, destination="session"))
    # R6:session = policy 扩展(生效通道)+ session_rules 追加(记录)
    assert out.policy.filesystem == (rule,)
    assert out.policy.revision != ctx.policy.revision
    assert ctx.policy.filesystem == ()  # 原 ctx 不变(纯函数)
    assert out.session_rules == (rule,)
    decision = decide(tool, use, out)
    assert decision.is_allow and decision.reason == "policy-grant"


def test_r6_always_tool_grant_extends_policy_and_records_both(ws: Path) -> None:
    ctx = base_ctx(ws)
    rule = ToolRule(pattern="probe", decision="allow")
    out = apply_update(ctx, PermissionUpdate(rule=rule, destination="always"))
    assert out.policy.tools == (rule,)  # R6:policy 扩展
    assert out.policy.revision != ctx.policy.revision
    assert out.always_rules == (rule,) and out.session_rules == (rule,)  # 记录层保留
    assert out.once_rules == ()
    assert decide(probe_tool(), empty_use(), out).is_allow


def test_r6_network_session_grant_allows_url_tool_but_bash_stays_denied(ws: Path) -> None:
    # D9:NetworkRule 授权使 network.rules 非空 ⇒ command_network_decision 一律 deny。
    ctx = base_ctx(ws)
    web = url_tool("web_fetch")
    use = ResolvedUse(effects=frozenset({ToolEffect.NETWORK}), targets=(ToolTarget(kind="url", value=_URL),))
    assert decide(web, use, ctx).is_ask  # 默认 network decision=ask
    rule = NetworkRule(domain="ok.com", decision="allow")
    out = apply_update(ctx, PermissionUpdate(rule=rule, destination="session"))
    assert out.policy.network.rules == (rule,)
    assert out.policy.revision != ctx.policy.revision
    assert out.session_rules == (rule,)
    assert decide(web, use, out).is_allow  # URL 级工具放行(第 6 步 policy-grant)
    exec_use = ResolvedUse(effects=frozenset({ToolEffect.EXECUTE_PROCESS}), targets=())
    bash_decision = decide(exec_tool(), exec_use, out)
    assert bash_decision.is_deny and bash_decision.reason == "rule: command network denied"  # bash fail-closed


# ---------------------------------------------------------------------------
# Session.update_permission 行为层(经真实 submit)
# ---------------------------------------------------------------------------


def test_policy_file_loaded_denies_bash(ws: Path) -> None:
    praxis_dir = ws / ".praxis"
    praxis_dir.mkdir()
    policy = PermissionPolicy(tools=(ToolRule(pattern="bash", decision="deny"),))
    (praxis_dir / "permissions.json").write_text(policy.model_dump_json(), encoding="utf-8")
    session = open_session(
        ws, ScriptedAdapter([script_call("c1", "bash", {"command": "echo hi"}), script_stop("done")])
    )
    try:
        result = asyncio.run(session.submit("run it"))
        assert result.status is TurnStatus.COMPLETED
        item = tool_result_for(session, "c1")
        assert item.payload["status"] == "denied"
        assert "rule: tool 'bash' denied" in str(item.payload.get("error_message"))
    finally:
        session.close()


def test_always_update_persists_and_roundtrips(ws: Path) -> None:
    adapter = [script_call("c1", "bash", {"command": "echo hi"}), script_stop("done")]
    session = open_session(ws, ScriptedAdapter(adapter))
    try:
        session.update_permission(
            PermissionUpdate(rule=ToolRule(pattern="bash", decision="deny"), destination="always")
        )
        result = asyncio.run(session.submit("run it"))
        assert result.status is TurnStatus.COMPLETED
        assert tool_result_for(session, "c1").payload["status"] == "denied"
    finally:
        session.close()
    path = permissions_file(ws)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["approval"] == "on-request"
    assert {"pattern": "bash", "decision": "deny"} in data["tools"]
    # 重新打开:文件即策略声明,bash 依旧 denied(无需再授权)
    session2 = open_session(ws, ScriptedAdapter(adapter))
    try:
        result2 = asyncio.run(session2.submit("run it again"))
        assert result2.status is TurnStatus.COMPLETED
        item2 = tool_result_for(session2, "c1")
        assert item2.payload["status"] == "denied"
        assert "rule: tool 'bash' denied" in str(item2.payload.get("error_message"))
    finally:
        session2.close()


def test_session_update_not_persisted(ws: Path) -> None:
    session = open_session(
        ws, ScriptedAdapter([script_call("c1", "bash", {"command": "echo hi"}), script_stop("done")])
    )
    try:
        session.update_permission(
            PermissionUpdate(rule=ToolRule(pattern="bash", decision="deny"), destination="session")
        )
        result = asyncio.run(session.submit("run it"))
        assert result.status is TurnStatus.COMPLETED
        assert tool_result_for(session, "c1").payload["status"] == "denied"  # 本会话生效
    finally:
        session.close()
    assert not permissions_file(ws).exists()  # session 授权永不落盘


def test_once_grant_effective_cleared_after_batch(ws: Path) -> None:
    adapter = [
        script_call("c1", "probe"),
        script_stop("turn1 done"),
        script_call("c2", "probe"),
        script_stop("turn2 never reached"),
    ]
    session = open_session(ws, ScriptedAdapter(adapter), tools=(probe_tool(),))
    try:
        session.update_permission(
            PermissionUpdate(rule=ToolRule(pattern="probe", decision="allow"), destination="once")
        )
        r1 = asyncio.run(session.submit("go 1"))
        assert r1.status is TurnStatus.COMPLETED  # once 授权:本批放行
        assert tool_result_for(session, "c1").payload["status"] == "ok"
        r2 = asyncio.run(session.submit("go 2"))  # _execute finally 已清除 once
        assert r2.status is TurnStatus.PAUSED
        assert [approval.call_id for approval in r2.pending_approvals] == ["c2"]
        assert not permissions_file(ws).exists()  # once 永不落盘
    finally:
        session.close()


def test_mode_bypass_orthogonal_rules_survive(ws: Path) -> None:
    # C5:mode=bypass 不改 policy.approval、不碰预授权规则;bypass 免疫规则 deny(第 1 步先)。
    policy = PermissionPolicy(tools=(ToolRule(pattern="probe", decision="deny"),))
    session = open_session(
        ws, ScriptedAdapter([script_call("c1", "probe"), script_stop("x")]), tools=(probe_tool(),),
        policy=policy, mode="bypass",
    )
    try:
        result = asyncio.run(session.submit("go"))
        assert result.status is TurnStatus.COMPLETED
        assert tool_result_for(session, "c1").payload["status"] == "denied"
        session.update_permission(
            PermissionUpdate(rule=NetworkRule(domain="ok.com", decision="allow"), destination="always")
        )
    finally:
        session.close()
    data = json.loads(permissions_file(ws).read_text(encoding="utf-8"))
    assert data["approval"] == "on-request"  # bypass 永不进 policy/永不落盘
    assert {"pattern": "probe", "decision": "deny"} in data["tools"]  # 预授权规则保持
    assert {"domain": "ok.com", "decision": "allow"} in data["network"]["rules"]
    # 切回默认(on-request)重开:deny 规则依旧生效
    session2 = open_session(
        ws, ScriptedAdapter([script_call("c2", "probe"), script_stop("y")]), tools=(probe_tool(),)
    )
    try:
        result2 = asyncio.run(session2.submit("go again"))
        assert result2.status is TurnStatus.COMPLETED
        assert tool_result_for(session2, "c2").payload["status"] == "denied"
    finally:
        session2.close()


def test_explicit_policy_param_overrides_file(ws: Path) -> None:
    praxis_dir = ws / ".praxis"
    praxis_dir.mkdir()
    (praxis_dir / "permissions.json").write_text(
        PermissionPolicy(network=NetworkPolicy(decision="deny")).model_dump_json(), encoding="utf-8"
    )

    def adapter() -> ScriptedAdapter:
        return ScriptedAdapter([script_call("c1", "probe", {"url": _URL}), script_stop("done")])

    session_a = open_session(ws, adapter(), tools=(url_tool(),))
    try:
        result_a = asyncio.run(session_a.submit("fetch"))
        assert result_a.status is TurnStatus.COMPLETED
        item = tool_result_for(session_a, "c1")
        assert item.payload["status"] == "denied"  # 文件策略:network deny 生效
        assert "network denied" in str(item.payload.get("error_message"))
    finally:
        session_a.close()
    session_b = open_session(ws, adapter(), tools=(url_tool(),), policy=PermissionPolicy())
    try:
        result_b = asyncio.run(session_b.submit("fetch"))
        assert result_b.status is TurnStatus.PAUSED  # 显式参数 > 文件:默认 ask
    finally:
        session_b.close()


@pytest.mark.parametrize("bad_text", ["{ not json", '{"profile": "turbo"}', '{"nope": 1}', "[]"])
def test_bad_policy_file_rejected_at_boundary(ws: Path, bad_text: str) -> None:
    praxis_dir = ws / ".praxis"
    praxis_dir.mkdir()
    (praxis_dir / "permissions.json").write_text(bad_text, encoding="utf-8")
    with pytest.raises(ValueError):
        open_session(ws, ScriptedAdapter([]))


def test_invalid_mode_rejected(ws: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        open_session(ws, ScriptedAdapter([]), mode="turbo")


# ---------------------------------------------------------------------------
# SDK 出口(R6-7)
# ---------------------------------------------------------------------------


def test_sdk_reexports_update_surface() -> None:
    for name in ("PermissionUpdate", "PathRule", "ToolRule", "NetworkRule", "PermissionMode", "apply_update"):
        assert name in sdk_pkg.__all__, f"sdk.__all__ missing {name}"
        assert hasattr(sdk_pkg, name), f"sdk missing attribute {name}"

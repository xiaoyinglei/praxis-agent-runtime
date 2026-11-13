"""Task 4:decide 八步表 + apply_update(设计 §5.3;findings S2/S8;D1/D2/D5/D6/D10)。

每步独立穷举(命中/不命中矩阵)+ decide 合并器代数属性(hypothesis):
- deny-never-downgrades:deny 来源在场时,加 allow 规则/hook allow 绝不降级;
- ceiling-never-widens:加 ceiling 严格度单调不降(deny>ask>allow);
- decide-totality:任意输入(含 NUL/断链/loop symlink)不抛(B10);
- floor-bypass-immune:地板 target × 任意 mode ⇒ deny(S2)。

步骤序冻结:deny_rules → ask_rules → tool_callback → safety_floor →
mode_bypass → allow_rules → sandbox_auto → dontask_guard → fallback ask。
两个由步骤序直接推出的边界行为在此钉死(见 report):
- never 模式只转换 fallback ask;第 2 步的 ask 原样返回(步骤表无 mode 条件);
- bypass(第 5 步)先于沙箱裁决(第 7 步),但地板/规则/回调 deny 免疫。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from praxis.core.permission import (
    PIPELINE,
    allow_rules,
    apply_update,
    ask_rules,
    decide,
    deny_rules,
    dontask_guard,
    mode_bypass,
    safety_floor,
    sandbox_auto,
    tool_callback,
)
from praxis.protocol.floor import host_private_paths
from praxis.protocol.permission import (
    Decision,
    HookDecision,
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
    ToolDefinition,
    ToolEffect,
    ToolTarget,
)
from tests.praxis.conftest import make_minimal_tool

NEVER_REASON = "approval disabled (never mode)"
_ORDER = {"allow": 0, "ask": 1, "deny": 2}
# 默认 NetworkPolicy.decision="ask" ⇒ bash(EXECUTE_PROCESS)在第 2 步命令网络 ask(D9/S8)。
# 隔离第 5/7/8 步的 bash 测试须显式 network allow。
NET_ALLOW = PermissionPolicy(network=NetworkPolicy(decision="allow"))


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def make_ctx(
    workspace: Path,
    *,
    policy: PermissionPolicy | None = None,
    mode: str = "on-request",
    ceilings: tuple[PermissionPolicy, ...] = (),
    hook_decisions: tuple[HookDecision, ...] = (),
    session_rules: tuple = (),
    once_rules: tuple = (),
    always_rules: tuple = (),
    sandbox: SandboxAvailability | None = None,
) -> PermissionContext:
    return PermissionContext(
        workspace=str(workspace),
        policy=policy if policy is not None else PermissionPolicy(),
        mode=mode,  # type: ignore[arg-type]
        ceilings=ceilings,
        hook_decisions=hook_decisions,
        session_rules=session_rules,
        once_rules=once_rules,
        always_rules=always_rules,
        sandbox=sandbox if sandbox is not None else SandboxAvailability(backend=None, available=False),
    )


def make_tool(name: str, effects: frozenset[ToolEffect], targets: tuple[ToolTarget, ...], **overrides):
    return make_minimal_tool(
        definition=ToolDefinition(name=name, description=f"{name} test tool", input_schema={"type": "object"}),
        static_effects=effects,
        resolve_use=lambda arguments: ResolvedUse(effects=effects, targets=targets),
        **overrides,
    )


def path_use(workspace: Path, rel: str, effect: ToolEffect = ToolEffect.WRITE_WORKSPACE) -> ResolvedUse:
    return ResolvedUse(effects=frozenset({effect}), targets=(ToolTarget(kind="path", value=str(workspace / rel)),))


def url_use(url: str, effect: ToolEffect = ToolEffect.NETWORK) -> ResolvedUse:
    return ResolvedUse(effects=frozenset({effect}), targets=(ToolTarget(kind="url", value=url),))


@pytest.fixture
def ws(workspace: Path) -> Path:
    (workspace / "sub").mkdir()
    (workspace / "docs").mkdir()
    (workspace / "dangling").symlink_to(workspace / "missing")
    return workspace


@pytest.fixture
def bash_tool(ws: Path):
    return make_tool("bash", frozenset({ToolEffect.EXECUTE_PROCESS}), (ToolTarget(kind="path", value=str(ws)),))


@pytest.fixture
def bash_use(ws: Path) -> ResolvedUse:
    return ResolvedUse(
        effects=frozenset({ToolEffect.EXECUTE_PROCESS}), targets=(ToolTarget(kind="path", value=str(ws)),)
    )


@pytest.fixture
def write_tool(ws: Path):
    return make_tool("write", frozenset({ToolEffect.WRITE_WORKSPACE}), ())


@pytest.fixture
def web_tool():
    return make_tool("web_fetch", frozenset({ToolEffect.NETWORK}), ())


# ---------------------------------------------------------------------------
# 第 1 步 deny_rules:命中/不命中矩阵
# ---------------------------------------------------------------------------


def test_step1_tool_deny_any_source(ws, bash_tool, bash_use):
    ctx = make_ctx(ws, policy=PermissionPolicy(tools=(ToolRule(pattern="bash", decision="deny"),)))
    assert deny_rules(bash_tool, bash_use, ctx).is_deny
    ctx2 = make_ctx(ws, ceilings=(PermissionPolicy(tools=(ToolRule(pattern="*", decision="deny"),)),))
    assert deny_rules(bash_tool, bash_use, ctx2).is_deny  # ceiling 来源
    ctx3 = make_ctx(ws, session_rules=(ToolRule(pattern="bash", decision="deny"),))
    assert deny_rules(bash_tool, bash_use, ctx3).is_deny  # session 来源
    ctx4 = make_ctx(ws, once_rules=(ToolRule(pattern="ba*", decision="deny"),))
    assert deny_rules(bash_tool, bash_use, ctx4).is_deny  # once 来源


def test_step1_tool_most_restrictive_wins(ws, bash_tool, bash_use):
    ctx = make_ctx(
        ws,
        policy=PermissionPolicy(tools=(ToolRule(pattern="bash", decision="allow"),)),
        ceilings=(PermissionPolicy(tools=(ToolRule(pattern="bash", decision="ask"),)),),
        session_rules=(ToolRule(pattern="bash", decision="deny"),),
    )
    assert deny_rules(bash_tool, bash_use, ctx).is_deny


def test_step1_no_deny_returns_none(ws, bash_tool, bash_use):
    ctx = make_ctx(ws, policy=PermissionPolicy(tools=(ToolRule(pattern="bash", decision="allow"),)))
    assert deny_rules(bash_tool, bash_use, ctx) is None
    assert deny_rules(bash_tool, bash_use, make_ctx(ws)) is None  # 无规则 = 无意见


def test_step1_path_outside_workspace_denied(ws, write_tool):
    use = path_use(ws, "../outside.txt")
    d = deny_rules(write_tool, use, make_ctx(ws))
    assert d.is_deny and "outside permitted paths" in d.reason


def test_step1_path_write_flag_from_effects(ws, write_tool):
    policy = PermissionPolicy(filesystem=(PathRule(path="docs", access="read"),))
    read_use = path_use(ws, "docs/a.md", ToolEffect.READ_WORKSPACE)
    write_use = path_use(ws, "docs/a.md", ToolEffect.WRITE_WORKSPACE)
    assert deny_rules(write_tool, read_use, make_ctx(ws, policy=policy)) is None
    assert deny_rules(write_tool, write_use, make_ctx(ws, policy=policy)).is_deny


def test_step1_ceiling_path_intersection(ws, write_tool):
    ceiling = PermissionPolicy(profile="read-only")
    write_use = path_use(ws, "f.txt")
    read_use = path_use(ws, "f.txt", ToolEffect.READ_WORKSPACE)
    assert deny_rules(write_tool, write_use, make_ctx(ws, ceilings=(ceiling,))).is_deny
    assert deny_rules(write_tool, read_use, make_ctx(ws, ceilings=(ceiling,))) is None


def test_step1_session_path_rule_tightens_only(ws, write_tool):
    deny_ctx = make_ctx(ws, session_rules=(PathRule(path="sub", access="deny"),))
    assert deny_rules(write_tool, path_use(ws, "sub/x"), deny_ctx).is_deny
    # session allow 规则不放宽:workspace 外路径仍拒(交集语义,ceiling/地板同理)
    widen_ctx = make_ctx(ws, session_rules=(PathRule(path="../outside", access="write"),))
    assert deny_rules(write_tool, path_use(ws, "../outside.txt"), widen_ctx).is_deny


def test_step1_network_deny(ws, web_tool):
    policy = PermissionPolicy(network=NetworkPolicy(decision="allow", allowed_domains=("ok.com",)))
    assert deny_rules(web_tool, url_use("https://evil.com/x"), make_ctx(ws, policy=policy)).is_deny
    assert deny_rules(web_tool, url_use("https://ok.com/x"), make_ctx(ws, policy=policy)) is None


def test_step1_network_deny_from_ceiling(ws, web_tool):
    ceiling = PermissionPolicy(network=NetworkPolicy(decision="deny"))
    assert deny_rules(web_tool, url_use("https://ok.com"), make_ctx(ws, ceilings=(ceiling,))).is_deny


def test_step1_command_network_deny_d9(ws, bash_tool, bash_use):
    policy = PermissionPolicy(network=NetworkPolicy(decision="allow", allowed_domains=("ok.com",)))
    d = deny_rules(bash_tool, bash_use, make_ctx(ws, policy=policy))
    assert d.is_deny  # D9:子进程无代理不可逐域强制 → fail closed
    assert deny_rules(bash_tool, bash_use, make_ctx(ws)) is None


def test_step1_hook_deny(ws, bash_tool, bash_use):
    ctx = make_ctx(ws, hook_decisions=(HookDecision(source="h", verdict="deny"),))
    d = deny_rules(bash_tool, bash_use, ctx)
    assert d.is_deny and d.reason == "hook:h"


def test_step1_hook_allow_no_effect(ws, bash_tool, bash_use):
    ctx = make_ctx(
        ws,
        policy=PermissionPolicy(tools=(ToolRule(pattern="bash", decision="deny"),)),
        hook_decisions=(HookDecision(source="h", verdict="allow"),),
    )
    assert deny_rules(bash_tool, bash_use, ctx).is_deny


def test_step1_other_target_kinds_ignored(ws, write_tool):
    use = ResolvedUse(effects=frozenset({ToolEffect.WRITE_WORKSPACE}), targets=(ToolTarget(kind="blob", value="../x"),))
    assert deny_rules(write_tool, use, make_ctx(ws)) is None


# ---------------------------------------------------------------------------
# 第 2 步 ask_rules(S8:network ask 先于 bypass/allow/沙箱)
# ---------------------------------------------------------------------------


def test_step2_tool_ask_any_source(ws, bash_tool, bash_use):
    for ctx in (
        make_ctx(ws, policy=PermissionPolicy(tools=(ToolRule(pattern="bash", decision="ask"),))),
        make_ctx(ws, ceilings=(PermissionPolicy(tools=(ToolRule(pattern="bash", decision="ask"),)),)),
        make_ctx(ws, session_rules=(ToolRule(pattern="bash", decision="ask"),)),
        make_ctx(ws, once_rules=(ToolRule(pattern="bash", decision="ask"),)),
    ):
        assert ask_rules(bash_tool, bash_use, ctx).is_ask


def test_step2_tool_most_restrictive_and_deny_passthrough(ws, bash_tool, bash_use, write_tool):
    ctx = make_ctx(
        ws,
        policy=PermissionPolicy(tools=(ToolRule(pattern="bash", decision="allow"),)),
        session_rules=(ToolRule(pattern="bash", decision="ask"),),
    )
    assert ask_rules(bash_tool, bash_use, ctx).is_ask
    # deny 归第 1 步:用无 EXECUTE_PROCESS 的 write,排除命令网络 ask 干扰
    deny_ctx = make_ctx(ws, policy=PermissionPolicy(tools=(ToolRule(pattern="write", decision="deny"),)))
    assert ask_rules(write_tool, path_use(ws, "f.txt"), deny_ctx) is None


def test_step2_network_ask(ws, web_tool):
    ctx = make_ctx(ws, policy=PermissionPolicy(network=NetworkPolicy(decision="ask")))
    assert ask_rules(web_tool, url_use("https://ok.com/x"), ctx).is_ask
    allow_ctx = make_ctx(ws, policy=PermissionPolicy(network=NetworkPolicy(decision="allow")))
    assert ask_rules(web_tool, url_use("https://ok.com/x"), allow_ctx) is None


def test_step2_network_default_is_ask(ws, web_tool):
    # NetworkPolicy 默认 decision="ask":无声明即 ask(S8 的正常可达面)
    assert ask_rules(web_tool, url_use("https://ok.com/x"), make_ctx(ws)).is_ask


def test_step2_command_network_ask(ws, bash_tool, bash_use):
    policy = PermissionPolicy(network=NetworkPolicy(decision="ask"))
    assert ask_rules(bash_tool, bash_use, make_ctx(ws, policy=policy)).is_ask


def test_step2_hook_ask(ws, bash_tool, bash_use):
    ctx = make_ctx(ws, policy=NET_ALLOW, hook_decisions=(HookDecision(source="h", verdict="ask"),))
    d = ask_rules(bash_tool, bash_use, ctx)
    assert d.is_ask and d.reason == "hook:h"


def test_step2_no_ask_returns_none(ws, bash_tool, bash_use):
    ctx = make_ctx(
        ws,
        policy=PermissionPolicy(
            tools=(ToolRule(pattern="bash", decision="allow"),), network=NetworkPolicy(decision="allow")
        ),
    )
    assert ask_rules(bash_tool, bash_use, ctx) is None


def test_network_ask_survives_allow_rule(ws, web_tool):  # S8(brief 逐字)
    ctx = make_ctx(
        ws,
        policy=PermissionPolicy(
            tools=(ToolRule(pattern="web*", decision="allow"),), network=NetworkPolicy(decision="ask")
        ),
    )
    assert decide(web_tool, url_use("https://ok.com/x"), ctx).is_ask  # 第 2 步先于第 6 步


def test_network_ask_survives_bypass_and_sandbox(ws, web_tool):
    ctx = make_ctx(
        ws,
        policy=PermissionPolicy(network=NetworkPolicy(decision="ask")),
        mode="bypass",
        sandbox=SandboxAvailability(backend="fake", available=True),
    )
    assert decide(web_tool, url_use("https://ok.com/x"), ctx).is_ask  # 任何入口不可绕过


# ---------------------------------------------------------------------------
# 第 3 步 tool_callback(D5:回调只能收紧)
# ---------------------------------------------------------------------------


def test_step3_callback_deny_ask_fire_immediately(ws, write_tool):
    use = path_use(ws, "f.txt")
    for verdict, factory in (("deny", Decision.deny), ("ask", Decision.ask)):
        tool = make_minimal_tool(
            definition=ToolDefinition(name="cb", description="cb", input_schema={"type": "object"}),
            check_permission=lambda u, c, _f=factory: _f("tool"),
        )
        d = tool_callback(tool, use, make_ctx(ws))
        assert d is not None and d.verdict == verdict and d.reason == "tool"


def test_step3_callback_allow_and_none_continue(ws, write_tool):
    use = path_use(ws, "f.txt")
    allow_tool = make_minimal_tool(check_permission=lambda u, c: Decision.allow("tool"))
    none_tool = make_minimal_tool(check_permission=lambda u, c: None)
    assert tool_callback(allow_tool, use, make_ctx(ws)) is None
    assert tool_callback(none_tool, use, make_ctx(ws)) is None
    assert tool_callback(write_tool, use, make_ctx(ws)) is None  # 无回调


def test_callback_allow_does_not_widen(ws):  # brief 逐字
    net_no_target = ResolvedUse(effects=frozenset({ToolEffect.NETWORK}), targets=())
    tool = make_minimal_tool(check_permission=lambda use, ctx: Decision.allow("tool"))
    assert decide(tool, net_no_target, make_ctx(ws, mode="on-request")).is_ask
    assert decide(tool, net_no_target, make_ctx(ws, mode="never")).is_deny


def test_callback_deny_immune_to_bypass(ws, write_tool):
    tool = make_minimal_tool(check_permission=lambda u, c: Decision.deny("tool"))
    assert decide(tool, path_use(ws, "f.txt"), make_ctx(ws, mode="bypass")).is_deny


# ---------------------------------------------------------------------------
# 第 4 步 safety_floor(S2:bypass 免疫;双候选)
# ---------------------------------------------------------------------------


def test_step4_floor_bypass_immune(ws, write_tool):  # brief 逐字
    ctx = make_ctx(ws, mode="bypass")
    use = ResolvedUse(
        effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
        targets=(ToolTarget(kind="path", value=str(ws / ".praxis" / "worktree.json")),),
    )
    assert safety_floor(write_tool, use, ctx).is_deny  # S2
    assert decide(write_tool, use, ctx).is_deny  # bypass 也拒


def test_step4_floor_access_matrix(ws, write_tool):
    # .git 只读:write 拒,read 放;.praxis/.agent_memory 读写都拒
    git_write = path_use(ws, ".git/config")
    git_read = path_use(ws, ".git/config", ToolEffect.READ_WORKSPACE)
    assert safety_floor(write_tool, git_write, make_ctx(ws)).is_deny
    assert safety_floor(write_tool, git_read, make_ctx(ws)) is None
    for rel in (".praxis/x", ".agent_memory/m"):
        for effect in (ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE):
            assert safety_floor(write_tool, path_use(ws, rel, effect), make_ctx(ws)).is_deny


def test_step4_floor_destructive_write_flag(ws):
    tool = make_tool("deliver", frozenset({ToolEffect.DESTRUCTIVE}), ())
    use = ResolvedUse(
        effects=frozenset({ToolEffect.DESTRUCTIVE}),
        targets=(ToolTarget(kind="path", value=str(ws / ".git" / "config")),),
    )
    assert safety_floor(tool, use, make_ctx(ws)).is_deny  # DESTRUCTIVE ⇒ write=True


def test_step4_floor_private_paths(ws, write_tool):
    private = host_private_paths()[0]
    use = ResolvedUse(
        effects=frozenset({ToolEffect.READ_WORKSPACE}),
        targets=(ToolTarget(kind="path", value=str(Path(private) / "credentials.json")),),
    )
    d = safety_floor(write_tool, use, make_ctx(ws))
    assert d.is_deny and d.reason.startswith("floor: ")


def test_step4_floor_symlink_real_candidate(ws, write_tool):
    (ws / "sneaky").symlink_to(ws / ".praxis")
    use = path_use(ws, "sneaky/worktree.json")
    assert safety_floor(write_tool, use, make_ctx(ws)).is_deny  # real 候选命中
    assert decide(write_tool, use, make_ctx(ws, mode="bypass")).is_deny


def test_step4_floor_non_path_targets_ignored(ws, write_tool):
    use = ResolvedUse(
        effects=frozenset({ToolEffect.WRITE_WORKSPACE}), targets=(ToolTarget(kind="blob", value=".praxis"),)
    )
    assert safety_floor(write_tool, use, make_ctx(ws)) is None


def test_step4_floor_broken_symlink_no_raise(ws, write_tool):
    assert safety_floor(write_tool, path_use(ws, "dangling"), make_ctx(ws)) is None


# ---------------------------------------------------------------------------
# 第 5 步 mode_bypass
# ---------------------------------------------------------------------------


def test_step5_bypass_allow_other_modes_none(ws, bash_tool, bash_use):
    d = mode_bypass(bash_tool, bash_use, make_ctx(ws, mode="bypass"))
    assert d.is_allow and d.reason == "mode:bypass"
    assert mode_bypass(bash_tool, bash_use, make_ctx(ws, mode="on-request")) is None
    assert mode_bypass(bash_tool, bash_use, make_ctx(ws, mode="never")) is None


# ---------------------------------------------------------------------------
# 第 6 步 allow_rules(D1 覆盖;S5 预防;ceiling 不授权)
# ---------------------------------------------------------------------------


def test_step6_destructive_skips_allow(ws):  # brief 逐字
    ctx = make_ctx(ws, policy=PermissionPolicy(tools=(ToolRule(pattern="*", decision="allow"),)))
    tool = make_tool("deliver", frozenset({ToolEffect.DESTRUCTIVE}), ())
    use = ResolvedUse(effects=frozenset({ToolEffect.DESTRUCTIVE}), targets=())
    assert allow_rules(tool, use, ctx) is None  # S5 预防


def test_step6_execute_process_skips_to_sandbox(ws, bash_tool, bash_use):
    ctx = make_ctx(ws, policy=PermissionPolicy(tools=(ToolRule(pattern="*", decision="allow"),)))
    assert allow_rules(bash_tool, bash_use, ctx) is None  # D1:沙箱裁决


def test_step6_tool_allow_any_grant_source(ws, write_tool):
    use = path_use(ws, "f.txt")
    for ctx in (
        make_ctx(ws, policy=PermissionPolicy(tools=(ToolRule(pattern="write", decision="allow"),))),
        make_ctx(ws, session_rules=(ToolRule(pattern="write", decision="allow"),)),
        make_ctx(ws, once_rules=(ToolRule(pattern="write", decision="allow"),)),
    ):
        d = allow_rules(write_tool, use, ctx)
        assert d.is_allow and d.reason == "rule"


def test_step6_ceiling_allow_does_not_grant(ws, web_tool):
    # ceiling 只否决不授权:ceiling allow + 无覆盖(NETWORK 无 url)⇒ None,不得 allow
    ctx = make_ctx(ws, ceilings=(PermissionPolicy(tools=(ToolRule(pattern="*", decision="allow"),)),))
    use = ResolvedUse(effects=frozenset({ToolEffect.NETWORK}), targets=())
    assert allow_rules(web_tool, use, ctx) is None


def test_step6_coverage_workspace_read_write(ws, write_tool):
    d = allow_rules(write_tool, path_use(ws, "f.txt"), make_ctx(ws))
    assert d.is_allow and d.reason == "policy-grant"
    read_use = path_use(ws, "f.txt", ToolEffect.READ_WORKSPACE)
    assert allow_rules(write_tool, read_use, make_ctx(ws)).is_allow
    assert allow_rules(write_tool, path_use(ws, "../out.txt"), make_ctx(ws)) is None


def test_step6_coverage_ceiling_intersection(ws, write_tool):
    ceiling = PermissionPolicy(profile="read-only")
    assert allow_rules(write_tool, path_use(ws, "f.txt"), make_ctx(ws, ceilings=(ceiling,))) is None


def test_step6_network_coverage(ws, web_tool):
    policy = PermissionPolicy(network=NetworkPolicy(decision="allow"))
    ctx = make_ctx(ws, policy=policy)
    assert allow_rules(web_tool, url_use("https://ok.com/x"), ctx).is_allow
    no_target = ResolvedUse(effects=frozenset({ToolEffect.NETWORK}), targets=())
    assert allow_rules(web_tool, no_target, ctx) is None  # 无 url ⇒ 覆盖不成立 ⇒ fallback ask
    mixed = ResolvedUse(
        effects=frozenset({ToolEffect.NETWORK}),
        targets=(ToolTarget(kind="url", value="https://ok.com"), ToolTarget(kind="url", value="https://evil.com")),
    )
    deny_policy = PermissionPolicy(network=NetworkPolicy(decision="allow", denied_domains=("evil.com",)))
    assert allow_rules(web_tool, mixed, make_ctx(ws, policy=deny_policy)) is None


def test_step6_empty_effects_covered(ws):
    # 阶段 1 玩具工具(空 effects/无 target)⇒ 覆盖成立(迁移保绿的关键)
    tool = make_tool("toy", frozenset(), ())
    use = ResolvedUse(effects=frozenset(), targets=())
    assert allow_rules(tool, use, make_ctx(ws)).is_allow


def test_step6_session_path_allow_does_not_widen_coverage(ws, write_tool):
    ctx = make_ctx(ws, session_rules=(PathRule(path="../outside", access="write"),))
    assert allow_rules(write_tool, path_use(ws, "../out.txt"), ctx) is None


# ---------------------------------------------------------------------------
# 第 7 步 sandbox_auto(D2 fail-closed)
# ---------------------------------------------------------------------------


def test_step7_sandbox_fail_closed(ws, bash_tool, bash_use):  # brief 逐字
    ctx = make_ctx(ws, sandbox=SandboxAvailability(backend=None, available=False))
    assert sandbox_auto(bash_tool, bash_use, ctx).is_deny  # D2
    ctx2 = make_ctx(ws, sandbox=SandboxAvailability(backend="seatbelt", available=True))
    assert sandbox_auto(bash_tool, bash_use, ctx2).is_allow


def test_step7_non_execute_none(ws, write_tool):
    use = path_use(ws, "f.txt")
    assert sandbox_auto(write_tool, use, make_ctx(ws)) is None


def test_step7_decide_bash_sandbox_matrix(ws, bash_tool, bash_use):
    assert decide(bash_tool, bash_use, make_ctx(ws, policy=NET_ALLOW)).is_deny
    d = decide(
        bash_tool, bash_use,
        make_ctx(ws, policy=NET_ALLOW, sandbox=SandboxAvailability(backend="fake", available=True)),
    )
    assert d.is_allow and d.reason == "sandbox"


def test_decide_bash_default_network_asks(ws, bash_tool, bash_use):
    # D9/S8 正常可达面:默认 network ask ⇒ bash 第 2 步 ask(先于沙箱/bypass)
    assert decide(bash_tool, bash_use, make_ctx(ws)).is_ask
    assert decide(bash_tool, bash_use, make_ctx(ws, mode="bypass")).is_ask


# ---------------------------------------------------------------------------
# 第 8 步 dontask_guard + fallback
# ---------------------------------------------------------------------------


def test_step8_never_converts_ask(ws, web_tool):  # brief 逐字
    net_no_target = ResolvedUse(effects=frozenset({ToolEffect.NETWORK}), targets=())
    assert decide(web_tool, net_no_target, make_ctx(ws, mode="on-request")).is_ask
    assert decide(web_tool, net_no_target, make_ctx(ws, mode="never")).is_deny
    assert dontask_guard(web_tool, net_no_target, make_ctx(ws, mode="never")).is_deny
    assert dontask_guard(web_tool, net_no_target, make_ctx(ws, mode="on-request")) is None


def test_fallback_ask_reason_and_suggestions(ws, web_tool):
    use = ResolvedUse(effects=frozenset({ToolEffect.NETWORK}), targets=())
    d = decide(web_tool, use, make_ctx(ws))
    assert d.is_ask and d.reason == "default"
    assert {u.destination for u in d.suggestions} == {"session", "always"}


def test_never_mode_step2_ask_returns_ask(ws, bash_tool, bash_use):
    # 步骤表钉死:dontask_guard 只转换 fallback ask;第 2 步 ask 无 mode 条件。
    # never 下该 ask 会让回合停在审批(不可批准=不执行),fail-safe 但非 deny。
    ctx = make_ctx(ws, policy=PermissionPolicy(tools=(ToolRule(pattern="bash", decision="ask"),)), mode="never")
    assert decide(bash_tool, bash_use, ctx).is_ask


def test_bypass_precedes_sandbox(ws, bash_tool, bash_use):
    # 步骤表钉死:第 5 步 bypass 先于第 7 步沙箱;地板/规则 deny 仍免疫(见 floor 测试)。
    assert decide(bash_tool, bash_use, make_ctx(ws, policy=NET_ALLOW, mode="bypass")).is_allow


# ---------------------------------------------------------------------------
# decide 合并器:顺序、suggestions、全函数
# ---------------------------------------------------------------------------


def test_pipeline_order_frozen():
    assert PIPELINE == (
        deny_rules,
        ask_rules,
        tool_callback,
        safety_floor,
        mode_bypass,
        allow_rules,
        sandbox_auto,
        dontask_guard,
    )


def test_hook_allow_never_exempts(ws, bash_tool, bash_use):  # brief 逐字(D5)
    ctx = make_ctx(
        ws,
        policy=PermissionPolicy(tools=(ToolRule(pattern="bash", decision="deny"),)),
        hook_decisions=(HookDecision(source="h", verdict="allow"),),
    )
    assert decide(bash_tool, bash_use, ctx).is_deny


def test_ask_carries_suggestions_deny_does_not(ws, bash_tool, bash_use):  # brief 逐字
    ask_policy = PermissionPolicy(tools=(ToolRule(pattern="bash", decision="ask"),))
    d = decide(bash_tool, bash_use, make_ctx(ws, policy=ask_policy))
    assert d.is_ask and any(u.rule == ToolRule(pattern="bash", decision="allow") for u in d.suggestions)
    assert {u.destination for u in d.suggestions} == {"session", "always"}
    assert decide(
        bash_tool, bash_use, make_ctx(ws, policy=PermissionPolicy(tools=(ToolRule(pattern="bash", decision="deny"),)))
    ).suggestions == ()


def test_deny_suggestions_empty_all_sources(ws, bash_tool, bash_use, write_tool):
    floor_use = path_use(ws, ".praxis/x")
    deny_policy = PermissionPolicy(tools=(ToolRule(pattern="bash", decision="deny"),))
    cases = [
        (bash_tool, bash_use, make_ctx(ws, policy=deny_policy)),
        (bash_tool, bash_use, make_ctx(ws, policy=NET_ALLOW)),  # sandbox_unavailable
        (write_tool, floor_use, make_ctx(ws, mode="bypass")),  # floor
        (
            bash_tool, bash_use,
            make_ctx(ws, policy=NET_ALLOW, hook_decisions=(HookDecision(source="h", verdict="deny"),)),
        ),
        (bash_tool, bash_use, make_ctx(ws, policy=NET_ALLOW, mode="never")),  # sandbox_unavailable(never 同拒)
    ]
    for tool, use, ctx in cases:
        d = decide(tool, use, ctx)
        assert d.is_deny and d.suggestions == ()


def test_decide_total_on_hostile_inputs(ws, write_tool):
    # B10:NUL workspace/断链/loop symlink/NUL target/抛异常回调,一律三态返回
    loop = ws / "loop"
    loop.symlink_to(loop)
    hostile_uses = [
        path_use(ws, "dangling"),
        ResolvedUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(ToolTarget(kind="path", value=str(loop / "x")),),
        ),
        ResolvedUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(ToolTarget(kind="path", value="a\0b"),),
        ),
    ]
    for use in hostile_uses:
        for mode in ("on-request", "never", "bypass"):
            assert decide(write_tool, use, make_ctx(ws, mode=mode)).verdict in _ORDER
    for bad_ws in ("\0", "/x\0y", ""):
        ctx = make_ctx(ws, mode="on-request")
        object.__setattr__(ctx, "workspace", bad_ws)  # frozen dataclass 构造敌意输入
        assert decide(write_tool, path_use(ws, "f.txt"), ctx).verdict in _ORDER

    def boom(use, ctx):
        raise RuntimeError("callback exploded")

    boom_tool = make_minimal_tool(check_permission=boom)
    assert decide(boom_tool, path_use(ws, "f.txt"), make_ctx(ws)).is_deny  # fail closed


def test_apply_update_roundtrip_grants_session_allow(ws, web_tool):
    use = ResolvedUse(effects=frozenset({ToolEffect.NETWORK}), targets=())
    ctx = make_ctx(ws)
    assert decide(web_tool, use, ctx).is_ask
    update = PermissionUpdate(rule=ToolRule(pattern="web_fetch", decision="allow"), destination="session")
    updated = apply_update(ctx, update)
    assert decide(web_tool, use, updated).is_allow  # 第 6 步 ToolRule allow 先于覆盖检查


# ---------------------------------------------------------------------------
# apply_update:三 destination 纯函数
# ---------------------------------------------------------------------------


def test_apply_update_once(ws):
    ctx = make_ctx(ws)
    rule = ToolRule(pattern="bash", decision="allow")
    out = apply_update(ctx, PermissionUpdate(rule=rule, destination="once"))
    assert out.once_rules == (rule,)
    assert out.session_rules == () and out.always_rules == ()
    assert ctx.once_rules == () and out is not ctx


def test_apply_update_session(ws):
    ctx = make_ctx(ws)
    rule = PathRule(path="sub", access="write")
    out = apply_update(ctx, PermissionUpdate(rule=rule, destination="session"))
    assert out.session_rules == (rule,)
    assert out.once_rules == () and out.always_rules == ()
    assert ctx.session_rules == ()


def test_apply_update_always_hits_both(ws):
    ctx = make_ctx(ws)
    rule = NetworkRule(domain="ok.com", decision="allow")
    out = apply_update(ctx, PermissionUpdate(rule=rule, destination="always"))
    assert out.always_rules == (rule,) and out.session_rules == (rule,)
    assert out.once_rules == ()
    assert ctx.always_rules == () and ctx.session_rules == ()


def test_apply_update_appends_and_preserves_other_fields(ws):
    existing = ToolRule(pattern="read", decision="allow")
    ctx = make_ctx(ws, session_rules=(existing,), mode="bypass")
    rule = ToolRule(pattern="write", decision="allow")
    out = apply_update(ctx, PermissionUpdate(rule=rule, destination="session"))
    assert out.session_rules == (existing, rule)
    assert out.mode == "bypass" and out.workspace == ctx.workspace and out.policy is ctx.policy


def test_apply_update_unknown_destination_raises(ws):
    ctx = make_ctx(ws)
    with pytest.raises(ValueError, match="destination"):
        apply_update(ctx, PermissionUpdate(rule=ToolRule(pattern="x", decision="allow"), destination="bogus"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# hypothesis:合并器代数属性(机器强制面)
# ---------------------------------------------------------------------------

_TOOL_NAMES = ["bash", "read", "write", "web_fetch", "minimal"]
_PATTERNS = ["*", "bash", "read", "write", "web*", "minimal", "ba*"]
_VERDICTS = ["allow", "ask", "deny"]
_DOMAINS = ["ok.com", "*.corp.io", "evil.com"]


@st.composite
def policy_strategy(draw):
    profile = draw(st.sampled_from(["workspace", "read-only", "trusted-local"]))
    accesses = ["read", "deny"] if profile == "read-only" else ["read", "write", "deny"]
    filesystem = draw(
        st.lists(
            st.builds(
                PathRule,
                path=st.sampled_from(["sub", "sub/ok", "docs", "../outside", "."]),
                access=st.sampled_from(accesses),
            ),
            max_size=3,
        )
    )
    tools = draw(
        st.lists(
            st.builds(ToolRule, pattern=st.sampled_from(_PATTERNS), decision=st.sampled_from(_VERDICTS)), max_size=3
        )
    )
    network = draw(
        st.builds(
            NetworkPolicy,
            decision=st.sampled_from(_VERDICTS),
            allowed_domains=st.lists(st.sampled_from(_DOMAINS), max_size=2).map(tuple),
            denied_domains=st.lists(st.sampled_from(_DOMAINS), max_size=2).map(tuple),
            rules=st.lists(
                st.builds(NetworkRule, domain=st.sampled_from(_DOMAINS), decision=st.sampled_from(_VERDICTS)),
                max_size=2,
            ).map(tuple),
        )
    )
    return PermissionPolicy(profile=profile, filesystem=tuple(filesystem), tools=tuple(tools), network=network)


def target_strategy(ws: Path):
    paths = [
        str(ws / "f.txt"), str(ws / "sub" / "x"), str(ws / "docs" / "a"), str(ws / ".praxis" / "x"),
        str(ws / ".git" / "config"), str(ws / "dangling"), str(ws / ".." / "out.txt"), "relative.txt",
    ]
    urls = ["https://ok.com/x", "https://evil.com", "http://sub.corp.io", "not a url", "ftp://ok.com"]
    return st.one_of(
        st.builds(ToolTarget, kind=st.just("path"), value=st.sampled_from(paths)),
        st.builds(ToolTarget, kind=st.just("url"), value=st.sampled_from(urls)),
        st.builds(ToolTarget, kind=st.just("blob"), value=st.sampled_from(["anything"])),
    )


@st.composite
def use_strategy(draw, ws: Path):
    effects = draw(st.sets(st.sampled_from(list(ToolEffect)), max_size=3))
    targets = draw(st.lists(target_strategy(ws), max_size=3))
    return ResolvedUse(effects=frozenset(effects), targets=tuple(targets))


def _property_tool(name: str):
    return make_minimal_tool(
        definition=ToolDefinition(name=name, description="property tool", input_schema={"type": "object"})
    )


_HYP = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


@_HYP
@given(data=st.data())
def test_deny_never_downgrades(data, ws):
    # deny 来源在场 ⇒ 加 allow 规则(session ToolRule/PathRule/NetworkRule)+ hook allow 不降级。
    # never 模式的 fallback deny 是 mode 产物(第 8 步),第 6 步规则 allow 先于它——排除该 reason。
    policy = data.draw(policy_strategy())
    use = data.draw(use_strategy(ws))
    mode = data.draw(st.sampled_from(["on-request", "never", "bypass"]))
    tool = _property_tool(data.draw(st.sampled_from(_TOOL_NAMES)))
    ctx = make_ctx(ws, policy=policy, mode=mode)
    base = decide(tool, use, ctx)
    if not base.is_deny or base.reason == NEVER_REASON:
        return
    widened = replace(
        ctx,
        session_rules=ctx.session_rules + (
            ToolRule(pattern="*", decision="allow"),
            PathRule(path="/", access="write"),
            NetworkRule(domain="ok.com", decision="allow"),
        ),
        hook_decisions=ctx.hook_decisions + (HookDecision(source="h", verdict="allow"),),
    )
    assert decide(tool, use, widened).is_deny


@_HYP
@given(data=st.data())
def test_bypass_preserves_source_denials(data, ws):
    # bypass 决策表(机器强制面):第 1 步 deny ⇒ deny;否则第 2 步 ask ⇒ ask
    # (ask 先于回调/地板/bypass);否则回调/地板 deny ⇒ deny(S2/D5 免疫)。
    policy = data.draw(policy_strategy())
    use = data.draw(use_strategy(ws))
    tool = _property_tool(data.draw(st.sampled_from(_TOOL_NAMES)))
    ctx = make_ctx(ws, policy=policy, mode="on-request")
    bypassed = decide(tool, use, replace(ctx, mode="bypass"))
    deny1 = deny_rules(tool, use, ctx)
    if deny1 is not None and deny1.is_deny:
        assert bypassed.is_deny
        return
    if ask_rules(tool, use, ctx) is not None:
        assert bypassed.is_ask
        return
    callback = tool_callback(tool, use, ctx)
    if callback is not None and callback.is_deny:
        assert bypassed.is_deny
        return
    floor = safety_floor(tool, use, ctx)
    if floor is not None and floor.is_deny:
        assert bypassed.is_deny  # 地板 bypass 免疫(S2)


@_HYP
@given(data=st.data())
def test_ceiling_never_widens(data, ws):
    # ceiling 永不授权:capped=allow ⇒ base=allow(allow 集合只缩不涨)。
    # 注:完整 deny>ask>allow 序在冻结步骤表下不成立——第 2 步 ask 先于第 3/4/7 步,
    # ceiling ask 可把后段 deny 软化为 ask(仍不自动执行);故另钉第 1 步 deny 单调。
    policy = data.draw(policy_strategy())
    ceiling = data.draw(policy_strategy())
    use = data.draw(use_strategy(ws))
    tool = _property_tool(data.draw(st.sampled_from(_TOOL_NAMES)))
    ctx = make_ctx(ws, policy=policy, mode="on-request")
    base = decide(tool, use, ctx)
    capped = decide(tool, use, replace(ctx, ceilings=(ceiling,)))
    if capped.is_allow:
        assert base.is_allow
    base_deny = deny_rules(tool, use, ctx)
    if base_deny is not None and base_deny.is_deny:
        capped_deny = deny_rules(tool, use, replace(ctx, ceilings=(ceiling,)))
        assert capped_deny is not None and capped_deny.is_deny  # 第 1 步 deny 来源单调


@_HYP
@given(data=st.data())
def test_decide_total(data, ws):
    # 任意输入(含 NUL workspace/断链 target/抛异常回调)不抛,返回三态之一(B10)。
    policy = data.draw(policy_strategy())
    use = data.draw(use_strategy(ws))
    mode = data.draw(st.sampled_from(["on-request", "never", "bypass"]))
    name = data.draw(st.sampled_from(_TOOL_NAMES))
    bad_workspace = data.draw(st.sampled_from([str(ws), "\0", "/x\0y", "", "~/nowhere"]))
    exploding = data.draw(st.booleans())
    tool = _property_tool(name)
    if exploding:
        tool = make_minimal_tool(
            definition=ToolDefinition(name=name, description="exploding", input_schema={"type": "object"}),
            check_permission=lambda u, c: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    ctx = make_ctx(ws, policy=policy, mode=mode)
    object.__setattr__(ctx, "workspace", bad_workspace)
    d = decide(tool, use, ctx)
    assert isinstance(d, Decision) and d.verdict in _ORDER


@_HYP
@given(data=st.data())
def test_floor_bypass_immune_property(data, ws):
    # 地板 target × 任意 mode ⇒ floor deny(S2)。私有目录显式授权 write,
    # 保证第 1 步放行、地板是唯一拒绝者(防御纵深:授权也压不过地板)。
    mode = data.draw(st.sampled_from(["on-request", "never", "bypass"]))
    private = host_private_paths()[0]
    effect = data.draw(st.sampled_from([ToolEffect.WRITE_WORKSPACE, ToolEffect.DESTRUCTIVE]))
    if data.draw(st.booleans()):
        rel = data.draw(st.sampled_from([".praxis/x", ".git/config", ".agent_memory/m"]))
        target, policy = str(ws / rel), PermissionPolicy()
    else:
        target = str(Path(private) / "secret")
        policy = PermissionPolicy(filesystem=(PathRule(path=private, access="write"),))
    tool = _property_tool(data.draw(st.sampled_from(_TOOL_NAMES)))
    use = ResolvedUse(effects=frozenset({effect}), targets=(ToolTarget(kind="path", value=target),))
    d = decide(tool, use, make_ctx(ws, policy=policy, mode=mode))
    assert d.is_deny and d.reason.startswith("floor: ") and d.suggestions == ()

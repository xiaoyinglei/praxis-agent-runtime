"""Task 11:web URL 面进唯一 decide(S8 消亡机制;阶段 2 留档①②)。

brief Step 1 ①–⑤ 逐条:
① URL target 的 ask 在 bypass/allow 规则/沙箱可用三态下全部先返回 ask
  (reason ``rule: network requires approval: <url>``),三入口 × 三裁决矩阵;
② URL ask 的 suggestions = 域名级 NetworkRule(session/always,**最前**=推荐窄授权)
  + 既有工具级 ToolRule(session/always);主机归一化与 network_decision 同源;
③ update_permission 返回 D9 PermissionNotice 且 live policy 已生效
  (仅 NetworkRule + session/always;once/PathRule/ToolRule 返回 None);
④ authorizer 闭包:allow 放行、ask 拒绝且带 approval required 文案、deny 拒绝,
  且读 **live** policy(session 中途授权后下一跳立刻放行);
⑤ 重定向换域:初始 allow、落地 ask ⇒ 中止;落地 allow ⇒ 成功。

S8 对照表钉死:「旧实现 ask==放行;praxis 实现 ask==拒绝本次请求并指向
NetworkRule 授权路径」。逐跳/子资源是**执行期检查点**,不是审批暂停面。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.web.client import PublicHttpClient, PublicWebError
from praxis.adapters.web.limits import WebLimits
from praxis.composition import make_web_authorizer
from praxis.core.permission import decide, network_verdict
from praxis.core.statemachine import TurnStatus
from praxis.core.turn import TurnResult
from praxis.interfaces.cli.main import _report_turn
from praxis.interfaces.sdk import ScriptedAdapter, open_session
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import (
    HookDecision,
    PermissionContext,
    PermissionNotice,
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
from tests.praxis.test_web_urls_dns import (
    PUBLIC_IP,
    FakeBackend,
    FakeResolver,
    FakeStream,
    response_chunks,
)

TOOL_NAME = "web_probe"
URL = "https://a.b.c/d"
HOST = "a.b.c"
D9_TEXT = (
    "granting this network rule makes bash command network fail-closed (D9); "
    "approvals for bash will be required as whole-tool asks"
)
NETWORK_EFFECTS = frozenset({ToolEffect.NETWORK})
EXEC_EFFECTS = frozenset({ToolEffect.EXECUTE_PROCESS})


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


@pytest.fixture
def ws(workspace: Path) -> Path:
    """conftest `workspace` 的本文件别名(与 test_permission_pipeline 同名习惯)。"""
    return workspace


def make_ctx(
    ws: Path,
    *,
    policy: PermissionPolicy | None = None,
    mode: str = "on-request",
    hook_decisions: tuple[HookDecision, ...] = (),
    session_rules: tuple = (),
    sandbox: SandboxAvailability | None = None,
) -> PermissionContext:
    return PermissionContext(
        workspace=str(ws),
        policy=policy if policy is not None else PermissionPolicy(),
        mode=mode,  # type: ignore[arg-type]
        hook_decisions=hook_decisions,
        session_rules=session_rules,
        sandbox=(
            sandbox
            if sandbox is not None
            else SandboxAvailability(backend=None, available=False)
        ),
    )


def make_tool(
    name: str,
    effects: frozenset[ToolEffect],
    targets: tuple[ToolTarget, ...],
) -> Tool:
    return make_minimal_tool(
        definition=ToolDefinition(
            name=name, description=f"{name} web-decide tool", input_schema={"type": "object"}
        ),
        static_effects=effects,
        resolve_use=lambda arguments: ResolvedUse(effects=effects, targets=targets),
    )


def url_tool(name: str = TOOL_NAME, url: str = URL) -> Tool:
    return make_tool(name, NETWORK_EFFECTS, (ToolTarget(kind="url", value=url),))


def url_use(url: str = URL) -> ResolvedUse:
    return ResolvedUse(
        effects=NETWORK_EFFECTS, targets=(ToolTarget(kind="url", value=url),)
    )


def exec_tool(name: str = "bash") -> Tool:
    return make_tool(name, EXEC_EFFECTS, ())


def exec_use() -> ResolvedUse:
    return ResolvedUse(effects=EXEC_EFFECTS, targets=())


def resp(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=Usage(0, 0),
        finish_reason="tool_calls" if calls else "stop",
    )


def tool_suggestions(name: str = TOOL_NAME) -> tuple[PermissionUpdate, ...]:
    rule = ToolRule(pattern=name, decision="allow")
    return (
        PermissionUpdate(rule=rule, destination="session"),
        PermissionUpdate(rule=rule, destination="always"),
    )


# ---------------------------------------------------------------------------
# ① URL ask 的 reason + S8 三入口 × 三裁决矩阵
# ---------------------------------------------------------------------------


def test_url_ask_reason(ws: Path) -> None:
    decision = decide(url_tool(), url_use(), make_ctx(ws))
    assert decision.is_ask
    assert decision.reason == f"rule: network requires approval: {URL}"


def _policy_for_verdict(verdict: str) -> PermissionPolicy:
    if verdict == "allow":
        network = NetworkPolicy(rules=(NetworkRule(domain=HOST, decision="allow"),))
    elif verdict == "deny":
        network = NetworkPolicy(rules=(NetworkRule(domain=HOST, decision="deny"),))
    else:
        network = NetworkPolicy()
    # ToolRule allow 在场:用来证明第 2 步先于第 6 步——allow 规则不能越过网络 ask。
    return PermissionPolicy(
        network=network, tools=(ToolRule(pattern=TOOL_NAME, decision="allow"),)
    )


@pytest.mark.parametrize("mode", ["on-request", "bypass", "never"])
@pytest.mark.parametrize("verdict", ["allow", "ask", "deny"])
def test_url_decision_matrix_across_modes(ws: Path, mode: str, verdict: str) -> None:
    """三入口 × {on-request,bypass,never} × {allow,ask,deny}。

    ask 由第 2 步先返回:bypass(第 5 步)/allow 规则(第 6 步)/never(末端转换)
    都不能把它放行;never 只把 ask 变 deny,绝不降级成 allow。
    """
    decision = decide(
        url_tool(), url_use(), make_ctx(ws, policy=_policy_for_verdict(verdict), mode=mode)
    )
    expected = verdict
    if mode == "never" and verdict == "ask":
        expected = "deny"
    assert decision.verdict == expected, (mode, verdict, decision)
    if verdict == "ask" and mode == "on-request":
        assert decision.reason == f"rule: network requires approval: {URL}"
    if verdict == "deny":
        assert decision.reason == f"rule: network denied: {URL}"


def test_network_ask_precedes_sandbox_available(ws: Path) -> None:
    """沙箱可用也不能越过 URL ask(第 2 步先于第 7 步;web 只声明 NETWORK)。"""
    ctx = make_ctx(
        ws,
        policy=_policy_for_verdict("ask"),
        sandbox=SandboxAvailability(backend="seatbelt", available=True),
    )
    decision = decide(url_tool(), url_use(), ctx)
    assert decision.is_ask
    assert decision.reason == f"rule: network requires approval: {URL}"


# ---------------------------------------------------------------------------
# ② 域名级 suggestions(最前=推荐窄授权)+ 其余 ask 形状零改动
# ---------------------------------------------------------------------------


def test_url_ask_suggestions_domain_first_then_tool(ws: Path) -> None:
    decision = decide(url_tool(), url_use(), make_ctx(ws))
    network_rule = NetworkRule(domain=HOST, decision="allow")
    assert decision.suggestions == (
        PermissionUpdate(rule=network_rule, destination="session"),
        PermissionUpdate(rule=network_rule, destination="always"),
        *tool_suggestions(),
    )


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("https://A.B.C:443/d", "a.b.c"),
        ("https://a.b.c./d", "a.b.c"),
        ("HTTP://A.B.C./x?y=1", "a.b.c"),
    ],
)
def test_url_ask_suggestion_host_normalization(ws: Path, url: str, host: str) -> None:
    decision = decide(url_tool(url=url), url_use(url), make_ctx(ws))
    assert decision.is_ask
    rule = decision.suggestions[0].rule
    assert isinstance(rule, NetworkRule)
    assert rule.domain == host
    assert rule.decision == "allow"


def test_url_ask_skips_domain_suggestion_for_unrepresentable_host(ws: Path) -> None:
    """IPv6 字面量无法表达成 NetworkRule(domain 校验只收主机名)——保留工具级建议。"""
    url = "https://[2606:4700::1111]/"
    decision = decide(url_tool(url=url), url_use(url), make_ctx(ws))
    assert decision.is_ask
    assert decision.suggestions == tool_suggestions()


def test_tool_rule_ask_suggestions_unchanged(ws: Path) -> None:
    policy = PermissionPolicy(tools=(ToolRule(pattern="bash", decision="ask"),))
    decision = decide(
        exec_tool(), exec_use(), make_ctx(ws, policy=policy)
    )
    assert decision.is_ask
    assert decision.reason == "rule: tool 'bash' requires approval"
    assert decision.suggestions == tool_suggestions("bash")


def test_hook_ask_suggestions_unchanged(ws: Path) -> None:
    tool = make_tool("read_probe", frozenset({ToolEffect.READ_WORKSPACE}), ())
    ctx = make_ctx(
        ws, hook_decisions=(HookDecision(source="guard.sh", verdict="ask"),)
    )
    decision = decide(tool, ResolvedUse(effects=frozenset({ToolEffect.READ_WORKSPACE}), targets=()), ctx)
    assert decision.is_ask
    assert decision.reason == "hook:guard.sh"
    assert decision.suggestions == tool_suggestions("read_probe")


def test_command_network_ask_keeps_tool_level_suggestions(ws: Path) -> None:
    """D9 命令网络 ask 不是 URL ask——建议保持工具级,不冒充域授权。"""
    decision = decide(exec_tool(), exec_use(), make_ctx(ws))
    assert decision.is_ask
    assert decision.reason == "rule: command network requires approval"
    assert decision.suggestions == tool_suggestions("bash")


# ---------------------------------------------------------------------------
# network_verdict(公开单一 fold;decide 与 authorizer 共用)
# ---------------------------------------------------------------------------


def test_network_verdict_folds_all_sources_most_restrictive(ws: Path) -> None:
    ctx = make_ctx(
        ws,
        policy=PermissionPolicy(
            network=NetworkPolicy(rules=(NetworkRule(domain=HOST, decision="allow"),))
        ),
        session_rules=(NetworkRule(domain=HOST, decision="deny"),),
    )
    assert network_verdict(ctx, URL) == "deny"


# ---------------------------------------------------------------------------
# ③ update_permission → D9 PermissionNotice(仅 NetworkRule + session/always)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("destination", ["session", "always"])
def test_update_permission_network_grant_returns_notice_and_takes_effect(
    ws: Path, destination: str
) -> None:
    session = open_session(ws, ScriptedAdapter([]))
    try:
        notice = session.update_permission(
            PermissionUpdate(
                rule=NetworkRule(domain=HOST, decision="allow"),
                destination=destination,  # type: ignore[arg-type]
            )
        )
        assert notice == PermissionNotice(kind="command_network_blocked", text=D9_TEXT)
        assert network_verdict(session.permissions, URL) == "allow"
    finally:
        session.close()


def test_update_permission_once_network_returns_none(ws: Path) -> None:
    session = open_session(ws, ScriptedAdapter([]))
    try:
        notice = session.update_permission(
            PermissionUpdate(
                rule=NetworkRule(domain=HOST, decision="allow"), destination="once"
            )
        )
        assert notice is None
        assert network_verdict(session.permissions, URL) == "ask"  # once 不折进 policy
    finally:
        session.close()


@pytest.mark.parametrize(
    "rule", [PathRule(path="sub", access="read"), ToolRule(pattern=TOOL_NAME, decision="allow")]
)
def test_update_permission_non_network_returns_none(ws: Path, rule: object) -> None:
    session = open_session(ws, ScriptedAdapter([]))
    try:
        assert session.update_permission(PermissionUpdate(rule=rule, destination="session")) is None  # type: ignore[arg-type]
    finally:
        session.close()


def test_network_grant_fail_closes_bash_command_network(ws: Path) -> None:
    """D9 联动:授权 URL 域后 bash 的命令网络从 ask 变 deny(fail-closed)。"""
    session = open_session(ws, ScriptedAdapter([]))
    try:
        tool, use = exec_tool(), exec_use()
        before = decide(tool, use, session.permissions)
        assert before.is_ask
        assert before.reason == "rule: command network requires approval"

        session.update_permission(
            PermissionUpdate(
                rule=NetworkRule(domain=HOST, decision="allow"), destination="session"
            )
        )
        after = decide(tool, use, session.permissions)
        assert after.is_deny
        assert after.reason == "rule: command network denied"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# ④ authorizer 闭包:allow / ask / deny + live policy
# ---------------------------------------------------------------------------


def test_web_authorizer_reads_live_policy_allow_ask_deny(ws: Path) -> None:
    session = open_session(ws, ScriptedAdapter([]))
    try:
        authorize = make_web_authorizer(lambda: session.permissions)
        with pytest.raises(PublicWebError) as ask_error:
            authorize(URL)
        assert ask_error.value.code == "denied"
        assert str(ask_error.value) == f"approval required for {HOST}"

        # live:同一条闭包,中途授予域规则后下一跳立刻放行(不缓存快照)。
        session.update_permission(
            PermissionUpdate(
                rule=NetworkRule(domain=HOST, decision="allow"), destination="session"
            )
        )
        assert authorize(URL) is None
    finally:
        session.close()

    deny_ctx = make_ctx(ws, policy=PermissionPolicy(network=NetworkPolicy(decision="deny")))
    authorize_deny = make_web_authorizer(lambda: deny_ctx)
    with pytest.raises(PublicWebError) as deny_error:
        authorize_deny(URL)
    assert deny_error.value.code == "denied"
    assert str(deny_error.value) == ""


def test_web_authorizer_reads_mutated_provider(ws: Path) -> None:
    holder: dict[str, PermissionContext] = {"ctx": make_ctx(ws)}
    authorize = make_web_authorizer(lambda: holder["ctx"])
    with pytest.raises(PublicWebError):
        authorize(URL)
    holder["ctx"] = make_ctx(
        ws,
        policy=PermissionPolicy(
            network=NetworkPolicy(rules=(NetworkRule(domain=HOST, decision="allow"),))
        ),
    )
    assert authorize(URL) is None


# ---------------------------------------------------------------------------
# ⑤ 重定向换域:逐跳检查点不是审批暂停面
# ---------------------------------------------------------------------------


def _redirect_client(
    *,
    policy: PermissionPolicy,
    calls: list[str],
) -> tuple[PublicHttpClient, FakeBackend, FakeResolver]:
    resolver = FakeResolver([PUBLIC_IP])
    backend = FakeBackend(
        [
            FakeStream(
                response_chunks(
                    "302 Found",
                    headers=[("Location", "https://second.test/next")],
                )
            ),
            FakeStream(response_chunks("200 OK", body=b"ok")),
        ]
    )
    inner = make_web_authorizer(lambda: make_ctx(Path("."), policy=policy))

    def authorize(url: str) -> None:
        calls.append(url)
        inner(url)

    client = PublicHttpClient(
        authorize=authorize,
        limits=WebLimits(),
        resolver=resolver,
        backend=backend,
        environ={},
    )
    return client, backend, resolver


def _allow_policy(*hosts: str) -> PermissionPolicy:
    return PermissionPolicy(
        network=NetworkPolicy(
            rules=tuple(NetworkRule(domain=host, decision="allow") for host in hosts)
        )
    )


def test_redirect_to_ask_host_aborts_at_execution_checkpoint() -> None:
    calls: list[str] = []
    client, backend, resolver = _redirect_client(
        policy=_allow_policy("first.test"), calls=calls
    )
    try:
        with pytest.raises(PublicWebError) as error:
            asyncio.run(client.fetch("https://first.test/start"))
        assert error.value.code == "denied"
        assert str(error.value) == "approval required for second.test"
    finally:
        asyncio.run(client.aclose())
    # 第一跳放行并连接;第二跳在 authorize 处中止——validate/connect 从未触及。
    assert calls == ["https://first.test/start", "https://second.test/next"]
    assert len(backend.connects) == 1
    assert resolver.calls == [("first.test", 443)]


def test_redirect_to_allowed_host_succeeds() -> None:
    calls: list[str] = []
    client, backend, _resolver = _redirect_client(
        policy=_allow_policy("first.test", "second.test"),
        calls=calls,
    )
    try:
        response = asyncio.run(client.fetch("https://first.test/start"))
    finally:
        asyncio.run(client.aclose())
    assert response.status == 200
    assert response.body == b"ok"
    assert response.final_url == "https://second.test/next"
    assert calls == ["https://first.test/start", "https://second.test/next"]
    assert len(backend.connects) == 2


# ---------------------------------------------------------------------------
# 审批流:建议在 resume 时显式落地,notice 进入 TurnResult;CLI 单行渲染
# ---------------------------------------------------------------------------


def test_resume_applies_suggested_domain_rule_and_surfaces_notice(ws: Path) -> None:
    tool = url_tool()
    session = open_session(
        ws,
        ScriptedAdapter([resp("", (("c1", TOOL_NAME, {}),)), resp("done")]),
        tools=(tool,),
    )
    try:
        paused = asyncio.run(session.submit("fetch it"))
        assert paused.status is TurnStatus.PAUSED
        assert [approval.call_id for approval in paused.pending_approvals] == ["c1"]

        # 真实 URL ask 的建议(域名级 session,第一条=推荐窄授权)
        suggestion = decide(tool, url_use(), session.permissions).suggestions[0]
        assert suggestion.destination == "session"

        resumed = asyncio.run(
            session.resume(
                paused.turn_id,
                approval={"c1": "allow"},
                permission_updates=(suggestion,),
            )
        )
        assert resumed.status is TurnStatus.COMPLETED
        assert resumed.notices == (
            PermissionNotice(kind="command_network_blocked", text=D9_TEXT),
        )
        assert network_verdict(session.permissions, URL) == "allow"
    finally:
        session.close()


def test_cli_report_turn_prints_notice_lines_and_keeps_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    notice = PermissionNotice(kind="command_network_blocked", text=D9_TEXT)
    result = TurnResult(
        turn_id="t1",
        status=TurnStatus.COMPLETED,
        output="done",
        reason="model_finished",
        notices=(notice,),
    )
    assert _report_turn(result) == 0
    output = capsys.readouterr().out
    assert f"notice: {D9_TEXT}" in output
    assert output.count("notice:") == 1
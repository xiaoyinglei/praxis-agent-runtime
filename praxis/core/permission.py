"""Praxis 唯一权限决策入口(设计 §5.3/§5.4;findings S2/S8 结构预防)。

decide() 的签名是冻结的接线契约:管道(T12 execute)只经模块属性
``permission.decide`` 调用它——阶段 2 把函数体换成步骤表,所有调用方零改动。
测试经 monkeypatch 本模块的 decide 验证"唯一入口"不是口号:换实现不换接线。

八步表(固定顺序,每步独立公开函数,可单测):
1. deny_rules    规则/hook deny 来源(tool 跨源取最严;path/network 逐策略交集;D9)
2. ask_rules     规则/hook ask 来源(S8:network ask 先于 bypass/allow/沙箱)
3. tool_callback 工具回调 deny/ask 立即生效;allow/None 继续(D5:只能收紧)
4. safety_floor  地板(protocol/floor.py 单一数据源),bypass 免疫(S2)
5. mode_bypass   bypass 模式 allow
6. allow_rules   ToolRule allow 或完整授权覆盖(D1);DESTRUCTIVE/EXECUTE_PROCESS 跳过
7. sandbox_auto  EXECUTE_PROCESS 沙箱裁决,不可用 → deny(D2 fail-closed)
8. dontask_guard never 模式把 fallback ask 转 deny(合并器末端再把任何早退 ask 统一转 deny)

规则匹配一律消费 protocol/policy.py 代数,地板一律消费 protocol/floor.py——
core 没有第二套匹配器/地板列表。ceiling 组合 = 逐一 resolve、决策时最严者胜
(deny>ask>allow),不建 intersection 对象;ceiling 只否决、永不授权
(ceiling-never-widens 的机器强制面)。session/once 规则层同样只收紧:
层基线("/" write / network allow)= 无意见,显式规则比基线更具体即生效。

decide 是全函数(B10):任何输入(NUL workspace、断链/loop symlink、抛异常的
回调)都不抛——意外异常 fail-closed 为 deny;八步皆无意见则 fallback ask。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path

from praxis.protocol.floor import floor_reason, host_private_paths
from praxis.protocol.permission import Decision, PermissionContext, PermissionUpdate
from praxis.protocol.policy import (
    NetworkPolicy,
    NetworkRule,
    PathRule,
    PermissionPolicy,
    PolicyVerdict,
    ResolvedPolicy,
    ToolRule,
    command_network_decision,
    network_decision,
    path_access,
    path_reason,
    resolve_policy,
    tool_decision,
)
from praxis.protocol.tools import ResolvedUse, Tool, ToolEffect

type Step = Callable[[Tool, ResolvedUse, PermissionContext], Decision | None]

# 裁决合成序(非规则匹配):deny>ask>allow,None=无意见。规则匹配本身只在 protocol/policy.py。
_VERDICT_ORDER: dict[PolicyVerdict | None, int] = {None: -1, "allow": 0, "ask": 1, "deny": 2}
_COVERABLE_EFFECTS = frozenset({ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE, ToolEffect.NETWORK})


# ---------------------------------------------------------------------------
# 策略解析与裁决合成(公开代数之上的组合层,无第二套匹配)
# ---------------------------------------------------------------------------


def _most_restrictive(verdicts: Iterable[PolicyVerdict | None]) -> PolicyVerdict | None:
    opinions = [verdict for verdict in verdicts if verdict is not None]
    return max(opinions, key=_VERDICT_ORDER.__getitem__) if opinions else None


def _policy_sources(ctx: PermissionContext) -> tuple[ResolvedPolicy, ...]:
    """[policy] + ceilings,逐一 resolve(ceiling 组合 = 各自解析,决策时取最严)。"""
    workspace = Path(ctx.workspace)
    return tuple(resolve_policy(policy, workspace) for policy in (ctx.policy, *ctx.ceilings))


def _rule_layers(ctx: PermissionContext) -> tuple[ResolvedPolicy, ...]:
    """session/once 规则层:只收紧、不放宽(交集语义的一个来源)。

    基线即"无意见":filesystem 以 "/" write 打底、network 以 allow 打底——
    显式规则比基线更具体(或同路径更严)即生效;无命中 = 放行该层。
    ToolRule 无基线问题:无命中 tool_decision 返回 None。
    """
    layers: list[ResolvedPolicy] = []
    for rules in (ctx.session_rules, ctx.once_rules):
        if not rules:
            continue
        path_rules = tuple(rule for rule in rules if isinstance(rule, PathRule))
        tool_rules = tuple(rule for rule in rules if isinstance(rule, ToolRule))
        network_rules = tuple(rule for rule in rules if isinstance(rule, NetworkRule))
        layers.append(
            resolve_policy(
                PermissionPolicy(
                    filesystem=(PathRule(path="/", access="write"), *path_rules),
                    tools=tool_rules,
                    network=NetworkPolicy(decision="allow", rules=network_rules),
                ),
                Path(ctx.workspace),
            )
        )
    return tuple(layers)


def _all_sources(ctx: PermissionContext) -> tuple[ResolvedPolicy, ...]:
    return (*_policy_sources(ctx), *_rule_layers(ctx))


def _path_targets(use: ResolvedUse) -> tuple[str, ...]:
    return tuple(target.value for target in use.targets if target.kind == "path")


def _url_targets(use: ResolvedUse) -> tuple[str, ...]:
    return tuple(target.value for target in use.targets if target.kind == "url")


def _tool_verdict(tool: Tool, ctx: PermissionContext) -> PolicyVerdict | None:
    """tool 裁决跨 [policy]+ceilings+session/once 取最严(第 1/2 步用)。"""
    name = tool.definition.name
    return _most_restrictive(tool_decision(source, name) for source in _all_sources(ctx))


def _tool_grant(tool: Tool, ctx: PermissionContext) -> PolicyVerdict | None:
    """第 6 步授权来源 = policy + session/once;ceiling 只否决、永不授权。"""
    name = tool.definition.name
    sources = (_policy_sources(ctx)[0], *_rule_layers(ctx))
    return _most_restrictive(tool_decision(source, name) for source in sources)


def _path_denial(use: ResolvedUse, ctx: PermissionContext, *, write: bool) -> str:
    """任一 path target 在任一来源不过 → 返回 path_reason;""=全部通过。"""
    for value in _path_targets(use):
        for source in _all_sources(ctx):
            reason = path_reason(source, value, write=write)
            if reason:
                return reason
    return ""


def _network_verdict(ctx: PermissionContext, url: str) -> PolicyVerdict:
    return _most_restrictive(network_decision(source, url) for source in _all_sources(ctx)) or "deny"


def _command_network_verdict(ctx: PermissionContext) -> PolicyVerdict:
    return _most_restrictive(command_network_decision(source) for source in _all_sources(ctx)) or "deny"


def _hook_source(ctx: PermissionContext, verdict: PolicyVerdict) -> str | None:
    for hook in ctx.hook_decisions:
        if hook.verdict == verdict:
            return hook.source
    return None


def _tool_ask(tool: Tool, reason: str) -> Decision:
    """ask 附 ToolRule allow 建议(session/always 两条);deny 不建议放宽。"""
    rule = ToolRule(pattern=tool.definition.name, decision="allow")
    return replace(
        Decision.ask(reason),
        suggestions=(
            PermissionUpdate(rule=rule, destination="session"),
            PermissionUpdate(rule=rule, destination="always"),
        ),
    )


# ---------------------------------------------------------------------------
# 八步(每步独立公开函数;None = 无意见,交下一步)
# ---------------------------------------------------------------------------


def deny_rules(tool: Tool, use: ResolvedUse, ctx: PermissionContext) -> Decision | None:
    """第 1 步:全部 deny 来源——tool 最严、path/network 交集、D9 命令网络、hook。"""
    if _tool_verdict(tool, ctx) == "deny":
        return Decision.deny(f"rule: tool '{tool.definition.name}' denied")
    reason = _path_denial(use, ctx, write=ToolEffect.WRITE_WORKSPACE in use.effects)
    if reason:
        return Decision.deny(f"rule: {reason}")
    for url in _url_targets(use):
        if _network_verdict(ctx, url) == "deny":
            return Decision.deny(f"rule: network denied: {url}")
    if ToolEffect.EXECUTE_PROCESS in use.effects and _command_network_verdict(ctx) == "deny":
        return Decision.deny("rule: command network denied")
    source = _hook_source(ctx, "deny")
    if source is not None:
        return Decision.deny(f"hook:{source}")
    return None


def ask_rules(tool: Tool, use: ResolvedUse, ctx: PermissionContext) -> Decision | None:
    """第 2 步:全部 ask 来源。S8:network ask 先于 bypass/allow/沙箱,任何入口不可绕过。"""
    if _tool_verdict(tool, ctx) == "ask":
        return _tool_ask(tool, f"rule: tool '{tool.definition.name}' requires approval")
    for url in _url_targets(use):
        if _network_verdict(ctx, url) == "ask":
            return _tool_ask(tool, f"rule: network requires approval: {url}")
    if ToolEffect.EXECUTE_PROCESS in use.effects and _command_network_verdict(ctx) == "ask":
        return _tool_ask(tool, "rule: command network requires approval")
    source = _hook_source(ctx, "ask")
    if source is not None:
        return _tool_ask(tool, f"hook:{source}")
    return None


def tool_callback(tool: Tool, use: ResolvedUse, ctx: PermissionContext) -> Decision | None:
    """第 3 步:回调 deny/ask 立即返回;allow/None → None(D5:回调只能收紧)。"""
    if tool.check_permission is None:
        return None
    decision = tool.check_permission(use, ctx)
    if decision is None or decision.is_allow:
        return None
    return decision


def safety_floor(tool: Tool, use: ResolvedUse, ctx: PermissionContext) -> Decision | None:
    """第 4 步:地板(protocol/floor.py 单一数据源),bypass 免疫(S2,先于第 5 步)。

    双候选(floor.py 调用约定):词法路径(floor_reason 内部 expanduser+abspath)
    与真实路径(non-strict resolve())任一命中即 deny——symlink 不转移地板豁免。
    write 旗标 = WRITE_WORKSPACE 或 DESTRUCTIVE ∈ effects。
    """
    write = ToolEffect.WRITE_WORKSPACE in use.effects or ToolEffect.DESTRUCTIVE in use.effects
    private_paths = host_private_paths()
    for value in _path_targets(use):
        candidates = [Path(value)]
        try:
            candidates.append(Path(value).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            pass  # 词法候选仍裁决;不可解析路径在第 1 步已被 path_reason 拒绝
        for candidate in candidates:
            reason = floor_reason(candidate, write=write, private_paths=private_paths)
            if reason:
                return Decision.deny(f"floor: {reason}")
    return None


def mode_bypass(tool: Tool, use: ResolvedUse, ctx: PermissionContext) -> Decision | None:
    """第 5 步:bypass 模式 allow(规则/回调/地板 deny 已在第 1-4 步先返回)。"""
    if ctx.mode == "bypass":
        return Decision.allow("mode:bypass")
    return None


def allow_rules(tool: Tool, use: ResolvedUse, ctx: PermissionContext) -> Decision | None:
    """第 6 步:ToolRule allow(授权来源)或完整授权覆盖(D1)。"""
    if ToolEffect.DESTRUCTIVE in use.effects:
        return None  # S5 预防:破坏性副作用永不自动 allow
    if ToolEffect.EXECUTE_PROCESS in use.effects:
        return None  # D1:交第 7 步沙箱裁决
    if _tool_grant(tool, ctx) == "allow":
        return Decision.allow("rule")
    if not use.effects <= _COVERABLE_EFFECTS:
        return None
    write = ToolEffect.WRITE_WORKSPACE in use.effects
    sources = _all_sources(ctx)
    for value in _path_targets(use):
        if any(not path_access(source, value, write=write) for source in sources):
            return None
    if ToolEffect.NETWORK in use.effects:
        urls = _url_targets(use)
        if not urls:
            return None  # NETWORK 无 url target ⇒ 覆盖不成立 ⇒ fallback ask(正常可达路径)
        if any(_network_verdict(ctx, url) != "allow" for url in urls):
            return None
    return Decision.allow("policy-grant")


def sandbox_auto(tool: Tool, use: ResolvedUse, ctx: PermissionContext) -> Decision | None:
    """第 7 步:EXECUTE_PROCESS 沙箱裁决;不可用 → deny(D2 fail-closed,无逃生门)。"""
    if ToolEffect.EXECUTE_PROCESS not in use.effects:
        return None
    if ctx.sandbox.available:
        return Decision.allow("sandbox")
    return Decision.deny("sandbox_unavailable")


def dontask_guard(tool: Tool, use: ResolvedUse, ctx: PermissionContext) -> Decision | None:
    """第 8 步:never 模式把 fallback ask 转 deny(审批禁用)。"""
    if ctx.mode == "never":
        return Decision.deny("approval disabled (never mode)")
    return None


PIPELINE: tuple[Step, ...] = (
    deny_rules,
    ask_rules,
    tool_callback,
    safety_floor,
    mode_bypass,
    allow_rules,
    sandbox_auto,
    dontask_guard,
)


# ---------------------------------------------------------------------------
# 合并器与 update 代数
# ---------------------------------------------------------------------------


def decide(tool: Tool, use: ResolvedUse, ctx: PermissionContext) -> Decision:
    """一次工具调用的权限裁决(唯一入口;签名冻结)。

    全函数(B10):任何输入不抛——意外异常 fail-closed 为 deny。
    八步皆无意见 → fallback ask(附建议规则)。
    never 模式在合并器末端把**任何** ask 统一转 deny(design §5.3 第 8 步
    "放最后防绕过":第 2/3 步的早退 ask 同样转换,不只 fallback)。
    """
    try:
        decision: Decision | None = None
        for step in PIPELINE:
            decision = step(tool, use, ctx)
            if decision is not None:
                break
        if decision is None:
            decision = _tool_ask(tool, "default")
    except Exception:  # 全函数围栏:敌意输入/回调异常一律 fail-closed,绝不炸穿管道
        return Decision.deny("rule: permission evaluation failed (fail closed)")
    if ctx.mode == "never" and decision.is_ask:
        return Decision.deny("approval disabled (never mode)")
    return decision


def apply_update(ctx: PermissionContext, update: PermissionUpdate) -> PermissionContext:
    """纯函数 update 代数(D10):once/session/always 追加对应规则层。

    always 同时进 always_rules(调用方落盘)与 session_rules(立即生效);
    返回新冻结上下文,原 ctx 不变;无磁盘 IO(落盘是 Task 8 调用方的事)。
    """
    if update.destination == "once":
        return replace(ctx, once_rules=(*ctx.once_rules, update.rule))
    if update.destination == "session":
        return replace(ctx, session_rules=(*ctx.session_rules, update.rule))
    if update.destination == "always":
        return replace(
            ctx,
            always_rules=(*ctx.always_rules, update.rule),
            session_rules=(*ctx.session_rules, update.rule),
        )
    raise ValueError(f"unknown update destination: {update.destination!r}")

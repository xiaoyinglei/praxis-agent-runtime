"""protocol.policy 策略解析与规则匹配代数(设计 §5.3;移植旧 permission_policy.py:134-414)。

覆盖 brief 的 TDD 用例(逐字)+ 边界补充 + hypothesis(全函数 B10、规则序无关性)。
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from praxis.protocol.policy import (
    NetworkPolicy,
    NetworkRule,
    PathRule,
    PermissionPolicy,
    ResolvedPolicy,
    ToolRule,
    command_network_decision,
    network_decision,
    path_access,
    path_reason,
    resolve_policy,
    tool_decision,
)

# --- brief Step 1 用例(逐字)-------------------------------------------------


def test_resolve_workspace_profile_grants_write(tmp_path):
    r = resolve_policy(PermissionPolicy(), tmp_path)
    assert path_access(r, tmp_path / "a.txt", write=True)
    assert not path_access(r, tmp_path.parent / "out.txt")


def test_readonly_profile_denies_writes(tmp_path):
    r = resolve_policy(PermissionPolicy(profile="read-only"), tmp_path)
    assert path_access(r, tmp_path / "a.txt") and not path_access(r, tmp_path / "a.txt", write=True)


def test_most_specific_rule_wins(tmp_path):
    p = PermissionPolicy(filesystem=(PathRule(path="sub", access="deny"), PathRule(path="sub/ok", access="write")))
    r = resolve_policy(p, tmp_path)
    assert not path_access(r, tmp_path / "sub/x", write=True)
    assert path_access(r, tmp_path / "sub/ok/y", write=True)


def test_symlink_escape_denied(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside)
    r = resolve_policy(PermissionPolicy(), tmp_path)
    assert not path_access(r, tmp_path / "link/f.txt")          # real 候选越界


def test_broken_symlink_no_raise(tmp_path):                     # B10
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    r = resolve_policy(PermissionPolicy(), tmp_path)
    assert path_reason(r, tmp_path / "dangling") == ""          # 目标在 workspace 内,non-strict 不抛
    assert isinstance(path_reason(r, tmp_path / "dangling", write=True), str)


def test_tool_decision_most_restrictive():
    r = resolve_policy(PermissionPolicy(tools=(ToolRule(pattern="ba*", decision="allow"),
                                               ToolRule(pattern="bash", decision="deny"))), Path("/w"))
    assert tool_decision(r, "bash") == "deny" and tool_decision(r, "read") is None


def test_network_decision_matrix():
    # 穷举:base decision × rules × allowed/denied × *.domain 匹配
    p = PermissionPolicy(network=NetworkPolicy(decision="ask", allowed_domains=("example.com",)))
    r = resolve_policy(p, Path("/w"))
    assert network_decision(r, "https://example.com/x") == "ask"
    assert network_decision(r, "https://evil.com/x") == "deny"      # 白名单外
    assert network_decision(r, "ftp://example.com") == "deny"       # scheme 外
    assert network_decision(r, "https://sub.example.com") == "deny" # 裸域不匹配子域


def test_command_network_filters_fail_closed():
    r = resolve_policy(
        PermissionPolicy(network=NetworkPolicy(decision="allow", allowed_domains=("x.com",))), Path("/w"))
    assert command_network_decision(r) == "deny"                    # D9:无代理不可逐域强制
    r2 = resolve_policy(PermissionPolicy(network=NetworkPolicy(decision="allow")), Path("/w"))
    assert command_network_decision(r2) == "allow"


# --- 边界补充 ----------------------------------------------------------------


def test_resolve_absolute_paths_and_workspace(tmp_path):
    r = resolve_policy(PermissionPolicy(filesystem=(PathRule(path="src", access="read"),)), tmp_path)
    assert isinstance(r, ResolvedPolicy)
    assert r.workspace == str(tmp_path.resolve())
    assert all(Path(rule.path).is_absolute() for rule in r.filesystem)
    assert r.filesystem[0].path == str(tmp_path.resolve())          # base 授权 = workspace 本体
    assert r.filesystem[1].path == str(tmp_path.resolve() / "src")


def test_trusted_local_profile_allows_reads_outside(tmp_path):
    r = resolve_policy(PermissionPolicy(profile="trusted-local"), tmp_path)
    assert any(rule.path == "/" and rule.access == "read" for rule in r.filesystem)
    assert path_access(r, "/etc/hosts")
    assert not path_access(r, "/etc/hosts", write=True)             # read 规则不授权写


def test_private_paths_denied(tmp_path):
    secret = tmp_path.parent / "secret"
    secret.mkdir()
    (secret / "token").write_text("x")
    r = resolve_policy(PermissionPolicy(profile="trusted-local"), tmp_path, private_paths=(str(secret),))
    assert r.private_paths == (str(secret),)
    assert not path_access(r, secret / "token")
    assert "private" in path_reason(r, secret / "token")
    assert not path_access(r, secret)                               # 目录本体也拒绝


def test_outside_and_symlink_reason_strings(tmp_path):
    outside = tmp_path.parent / "out2"
    outside.mkdir()
    (tmp_path / "link2").symlink_to(outside)
    r = resolve_policy(PermissionPolicy(), tmp_path)
    assert path_reason(r, tmp_path.parent / "nope.txt") == "path is outside permitted paths"
    assert path_reason(r, tmp_path / "link2" / "f").startswith("symlink target is outside permitted paths")


def test_symlink_loop_fails_closed(tmp_path):
    # 依赖受支持解释器范围(>=3.12,<3.14,见 .python-version):3.14 起 Path.resolve
    # 移除 loop 抛错语义(non-strict 直接返回 loop 路径),届时此契约需随迁移重审——超出本任务范围。
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    r = resolve_policy(PermissionPolicy(), tmp_path)
    assert path_reason(r, tmp_path / "loop") == "path cannot be safely resolved"  # ELOOP → OSError,不抛


@pytest.mark.parametrize("path", ["a\0b", "/w/a\0b"])
def test_nul_path_denied_not_raised(path: str, tmp_path):
    r = resolve_policy(PermissionPolicy(), tmp_path)
    assert path_reason(r, path) == "path cannot be safely resolved"


def test_write_denied_by_read_rule_reason(tmp_path):
    r = resolve_policy(PermissionPolicy(filesystem=(PathRule(path="docs", access="read"),)), tmp_path)
    assert path_access(r, tmp_path / "docs" / "a.md")
    reason = path_reason(r, tmp_path / "docs" / "a.md", write=True)
    assert reason == f"path denied by read rule: {tmp_path.resolve() / 'docs'}"


def test_deny_rule_beats_sibling_allow_by_specificity(tmp_path):
    # 同深度 deny 与 write 并存时,最具体(parts)优先;等深度等 access 不产生歧义
    p = PermissionPolicy(filesystem=(
        PathRule(path="a", access="read"),
        PathRule(path="a/b", access="deny"),
        PathRule(path="a/b/c", access="write"),
    ))
    r = resolve_policy(p, tmp_path)
    assert path_access(r, tmp_path / "a" / "x") and not path_access(r, tmp_path / "a" / "x", write=True)
    assert not path_access(r, tmp_path / "a" / "b" / "y")
    assert path_access(r, tmp_path / "a" / "b" / "c" / "z", write=True)


def test_tool_decision_case_sensitive_and_ask_over_allow():
    r = resolve_policy(PermissionPolicy(tools=(
        ToolRule(pattern="Bash", decision="allow"),
        ToolRule(pattern="ba*", decision="ask"),
    )), Path("/w"))
    assert tool_decision(r, "bash") == "ask"                        # 仅 ba* 命中:Bash 规则大小写敏感不命中
    assert tool_decision(r, "Bash") == "allow"                      # 仅 Bash 规则命中
    assert tool_decision(r, "read") is None                         # 无命中的默认交给上层
    both = resolve_policy(PermissionPolicy(tools=(
        ToolRule(pattern="Ba*", decision="allow"),
        ToolRule(pattern="Bash", decision="ask"),
    )), Path("/w"))
    assert tool_decision(both, "Bash") == "ask"                     # 全匹配取最严:ask > allow


@pytest.mark.parametrize("url", ["", "not a url", "https://", "http://[bad", "example.com/path", "//example.com"])
def test_network_decision_malformed_denied(url: str):
    r = resolve_policy(PermissionPolicy(), Path("/w"))
    assert network_decision(r, url) == "deny"


def test_network_rule_widens_non_base_deny_only():
    # base deny 不被 rules 覆盖;base ask 可被 rule 收紧/放宽到 allow
    r_deny = resolve_policy(PermissionPolicy(network=NetworkPolicy(
        decision="deny", rules=(NetworkRule(domain="ok.com", decision="allow"),))), Path("/w"))
    assert network_decision(r_deny, "https://ok.com") == "deny"
    assert network_decision(r_deny, "https://other.com") == "deny"
    r_ask = resolve_policy(PermissionPolicy(network=NetworkPolicy(
        decision="ask", rules=(NetworkRule(domain="ok.com", decision="allow"),))), Path("/w"))
    assert network_decision(r_ask, "https://ok.com") == "allow"
    assert network_decision(r_ask, "https://other.com") == "ask"


def test_network_denied_domains_and_wildcard_rules():
    r = resolve_policy(PermissionPolicy(network=NetworkPolicy(
        decision="allow", denied_domains=("bad.com", "*.nope.io"),
        rules=(NetworkRule(domain="*.corp.io", decision="ask"),))),
        Path("/w"))
    assert network_decision(r, "https://bad.com/x") == "deny"       # denied 压过 base allow
    assert network_decision(r, "https://sub.bad.com") == "allow"    # 裸域 denied 仅精确匹配
    assert network_decision(r, "https://a.corp.io") == "ask"
    assert network_decision(r, "https://corp.io") == "allow"        # *.corp.io 不匹配裸域
    assert network_decision(r, "https://deep.a.corp.io") == "ask"
    assert network_decision(r, "https://nope.io") == "allow"        # *.nope.io 不匹配裸域
    assert network_decision(r, "https://x.nope.io") == "deny"       # 通配 denied 压过 rules


def test_command_network_denied_or_rules_also_fail_closed():
    r1 = resolve_policy(PermissionPolicy(network=NetworkPolicy(decision="ask", denied_domains=("x.com",))), Path("/w"))
    assert command_network_decision(r1) == "deny"
    r2 = resolve_policy(PermissionPolicy(network=NetworkPolicy(
        decision="allow", rules=(NetworkRule(domain="y.com", decision="allow"),))), Path("/w"))
    assert command_network_decision(r2) == "deny"
    r3 = resolve_policy(PermissionPolicy(network=NetworkPolicy(decision="deny")), Path("/w"))
    assert command_network_decision(r3) == "deny"                   # 无过滤时回落 network.decision


def test_path_access_is_not_path_reason(tmp_path):
    r = resolve_policy(PermissionPolicy(), tmp_path)
    for candidate in (tmp_path / "x", tmp_path.parent / "y", tmp_path):
        assert path_access(r, candidate) == (not path_reason(r, candidate))
        assert path_access(r, candidate, write=True) == (not path_reason(r, candidate, write=True))


# --- hypothesis:全函数(B10)+ 规则序无关性 -----------------------------------

_SEGMENT = st.one_of(
    st.sampled_from(["", ".", "..", "sub", "ok", "x.txt", "link", "dangling", "loop", "missing", "private"]),
    st.text(st.characters(exclude_characters="\0/"), max_size=8),
)
_PATH_STR = st.lists(_SEGMENT, max_size=6).map("/".join)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(path=_PATH_STR)
def test_path_reason_total_never_raises(path: str, tmp_path: Path) -> None:
    # B10:任意字符串(绝对/相对/含 ..,NUL 除外)path_reason 必返回 str,不抛
    link = tmp_path / "link"
    if not link.is_symlink():
        link.symlink_to(tmp_path / "nowhere")
    r = resolve_policy(PermissionPolicy(filesystem=(PathRule(path="sub", access="deny"),)), tmp_path)
    for write in (False, True):
        reason = path_reason(r, path, write=write)
        assert isinstance(reason, str)
        assert bool(reason) == (not path_access(r, path, write=write))


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(shuffled=st.permutations([
    PathRule(path="sub", access="deny"),
    PathRule(path="sub/ok", access="write"),
    PathRule(path="private", access="read"),
    PathRule(path="deep/a/b", access="deny"),
]))
def test_filesystem_rule_order_irrelevant(shuffled: list[PathRule], tmp_path: Path) -> None:
    r = resolve_policy(PermissionPolicy(filesystem=tuple(shuffled)), tmp_path)
    candidates = [
        tmp_path / "sub" / "x", tmp_path / "sub" / "ok" / "y", tmp_path / "private" / "z",
        tmp_path / "deep" / "a" / "b" / "w", tmp_path / "free",
    ]
    baseline = {str(c): (path_reason(r, c), path_reason(r, c, write=True)) for c in candidates}
    rng = random.Random(0)
    for _ in range(5):
        perm = list(shuffled)
        rng.shuffle(perm)
        r2 = resolve_policy(PermissionPolicy(filesystem=tuple(perm)), tmp_path)
        assert {str(c): (path_reason(r2, c), path_reason(r2, c, write=True)) for c in candidates} == baseline

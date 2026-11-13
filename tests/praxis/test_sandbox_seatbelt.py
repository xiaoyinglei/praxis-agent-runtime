"""Task 5:seatbelt 渲染结构(全平台)+ 真实强制测试(darwin only)。

渲染结构:deny default、scratch 可写、policy 不相交子树(require-not 子区域)、
地板段由 FLOOR_ENTRIES 循环生成(无第二份清单)、private_paths 双形态 deny、
网络段 iff allow_network、escape、realpath 语义、broad-root+write 拒绝。

真实强制(subprocess 直跑 /usr/bin/sandbox-exec,非 mock):
- S2:.git 写阻断 / .praxis 读阻断 / workspace 写 + .git 读放行;
- 网络默认阻断(profile 无 network-outbound + best-effort curl,离线优雅降级);
- S9/D8 实证:沙箱内 ln .git/config 硬链接攻击;
- S9/D8 findings 补充(xfail strict):先存硬链接别名写穿透——seatbelt 按路径
  裁决无法表达 inode 规则,实证记录不假装解决(结构性修复随 bwrap 视图延后)。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox
from praxis.adapters.sandbox.seatbelt import render_profile
from praxis.protocol.floor import FLOOR_ENTRIES
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PathRule, PermissionPolicy, resolve_policy
from praxis.protocol.sandbox import SandboxRefusal, SandboxSpec
from praxis.protocol.tools import ResolvedUse, ToolEffect, ToolTarget

_AVAILABLE = probe_backend()
_DARWIN = sys.platform == "darwin"


def make_ctx(workspace: Path, *, policy: PermissionPolicy | None = None) -> PermissionContext:
    return PermissionContext(
        workspace=str(workspace),
        policy=policy if policy is not None else PermissionPolicy(),
        mode="on-request",
    )


def bash_use_of(workspace: Path) -> ResolvedUse:
    return ResolvedUse(
        effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
        targets=(ToolTarget(kind="path", value=str(workspace)),),
    )


def run_sandboxed_bash(spec: SandboxSpec, workspace: Path, command: str) -> subprocess.CompletedProcess[str]:
    """真实沙箱执行助手:argv_prefix + bash,subprocess timeout=30(全局 60s 之内)。"""
    return subprocess.run(
        [*spec.argv_prefix, "/bin/bash", "-lc", command],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# 渲染结构(全平台可跑)
# ---------------------------------------------------------------------------


def test_profile_denies_default_and_allows_scratch(tmp_path: Path):
    ws = tmp_path / "ws"
    scratch = tmp_path / "scratch"
    ws.mkdir()
    scratch.mkdir()
    resolved = resolve_policy(PermissionPolicy(), ws)
    text = render_profile(resolved, workspace=ws, scratch=scratch, allow_network=False, allow_write=True, private_paths=())
    assert "(version 1)" in text and "(deny default)" in text
    assert f'(subpath "{scratch}")' in text
    assert "(allow process-exec)" in text and "(allow process-fork)" in text


def test_policy_regions_disjoint(tmp_path: Path):
    # 父 deny + 子 write 授权 → require-not 子区域,父不影子化子
    ws = tmp_path / "ws"
    (ws / "locked" / "child").mkdir(parents=True)
    policy = PermissionPolicy(
        filesystem=(PathRule(path="locked", access="deny"), PathRule(path="locked/child", access="write"))
    )
    resolved = resolve_policy(policy, ws)
    text = render_profile(resolved, workspace=ws, scratch=tmp_path / "sc", allow_network=False, allow_write=True, private_paths=())
    locked = str(ws / "locked")
    child = str(ws / "locked" / "child")
    assert f'(deny file-read* (require-all (subpath "{locked}")' in text
    assert f'(require-not (subpath "{child}"))' in text
    assert f'(allow file-write* (require-all (subpath "{child}")' in text


def test_scratch_excluded_from_policy_regions(tmp_path: Path):
    ws = tmp_path / "ws"
    (ws / "locked").mkdir(parents=True)
    scratch = tmp_path / "sc"
    scratch.mkdir()
    resolved = resolve_policy(PermissionPolicy(filesystem=(PathRule(path="locked", access="deny"),)), ws)
    text = render_profile(resolved, workspace=ws, scratch=scratch, allow_network=False, allow_write=True, private_paths=())
    assert f'(require-not (subpath "{scratch}"))' in text


def test_floor_section_matches_floor_entries(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    resolved = resolve_policy(PermissionPolicy(), ws)
    text = render_profile(resolved, workspace=ws, scratch=tmp_path / "sc", allow_network=False, allow_write=True, private_paths=())
    # 地板 regex deny 行数 == FLOOR_ENTRIES 条数(无第二份地板清单)
    regex_denies = [line for line in text.splitlines() if "(regex" in line and line.startswith("(deny")]
    assert len(regex_denies) == len(FLOOR_ENTRIES)
    assert ".praxis" in text and ".agent_memory" in text and ".git" in text
    assert "[gG][iI][tT]" in text
    # deny 条目 read+write 双拒;read 条目仅 write 拒
    for entry in FLOOR_ENTRIES:
        if entry.access == "deny":
            assert any(line.startswith("(deny file-read* file-write* (regex") for line in regex_denies)
        else:
            assert any(line.startswith("(deny file-write* (regex") for line in regex_denies)
    # 旧 runtime 的第二份地板清单不得出现
    for legacy in (".venv", "node_modules", ".codex", ".aws", ".agents"):
        assert legacy not in text


def test_floor_identical_across_profiles(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    sections = []
    for profile in ("workspace", "read-only", "trusted-local"):
        resolved = resolve_policy(PermissionPolicy(profile=profile), ws)  # type: ignore[arg-type]
        text = render_profile(
            resolved, workspace=ws, scratch=tmp_path / "sc", allow_network=False, allow_write=False, private_paths=()
        )
        sections.append([line for line in text.splitlines() if "(regex" in line])
    assert sections[0] == sections[1] == sections[2]  # 地板与 profile 无关


def test_private_paths_rendered_both_forms(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    real = tmp_path / "priv-real"
    real.mkdir()
    alias = tmp_path / "priv-link"
    alias.symlink_to(real)
    resolved = resolve_policy(PermissionPolicy(), ws)
    text = render_profile(
        resolved, workspace=ws, scratch=tmp_path / "sc", allow_network=False, allow_write=True, private_paths=(str(alias),)
    )
    # 词法形态与内核解析形态都要 deny(/var ↔ /private/var 同理)
    assert f'(deny file-read* file-write* (subpath "{alias}"))' in text
    assert f'(deny file-read* file-write* (subpath "{real}"))' in text


def test_network_section_iff_allow(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    resolved = resolve_policy(PermissionPolicy(), ws)
    denied = render_profile(resolved, workspace=ws, scratch=tmp_path / "sc", allow_network=False, allow_write=True, private_paths=())
    assert "network-outbound" not in denied
    allowed = render_profile(resolved, workspace=ws, scratch=tmp_path / "sc", allow_network=True, allow_write=True, private_paths=())
    assert '(allow network-outbound (remote ip "*:*"))' in allowed
    assert "/private/var/run/mDNSResponder" in allowed and "/var/run/mDNSResponder" in allowed
    assert '(global-name "com.apple.networkd")' in allowed


def test_broad_root_read_with_write_refused(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    resolved = resolve_policy(PermissionPolicy(profile="trusted-local"), ws)
    with pytest.raises(SandboxRefusal):
        render_profile(resolved, workspace=ws, scratch=tmp_path / "sc", allow_network=False, allow_write=True, private_paths=())
    readonly = render_profile(resolved, workspace=ws, scratch=tmp_path / "sc", allow_network=False, allow_write=False, private_paths=())
    assert '(subpath "/")' in readonly


def test_symlink_boundary_refused_in_render(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "link").symlink_to(ws / "real")
    resolved = resolve_policy(PermissionPolicy(filesystem=(PathRule(path="link", access="read"),)), ws)
    with pytest.raises(SandboxRefusal):
        render_profile(resolved, workspace=ws, scratch=tmp_path / "sc", allow_network=False, allow_write=True, private_paths=())


def test_escape_quotes_and_backslashes(tmp_path: Path):
    ws = tmp_path / "ws"
    weird = ws / 'we"ird\\dir'
    weird.mkdir(parents=True)
    resolved = resolve_policy(PermissionPolicy(filesystem=(PathRule(path='we"ird\\dir', access="read"),)), ws)
    text = render_profile(resolved, workspace=ws, scratch=tmp_path / "sc", allow_network=False, allow_write=True, private_paths=())
    assert 'we\\"ird\\\\dir' in text


def test_workspace_rendered_as_realpath(tmp_path: Path):
    real = tmp_path / "real-ws"
    real.mkdir()
    link = tmp_path / "link-ws"
    link.symlink_to(real)
    resolved = resolve_policy(PermissionPolicy(), link)
    text = render_profile(resolved, workspace=link, scratch=tmp_path / "sc", allow_network=False, allow_write=True, private_paths=())
    assert f'(subpath "{real}")' in text
    assert f'(subpath "{link}")' not in text


# ---------------------------------------------------------------------------
# 真实 seatbelt 强制测试(darwin only;subprocess 直跑 sandbox-exec)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DARWIN or not _AVAILABLE.available, reason="seatbelt is macOS only")
def test_seatbelt_blocks_floor_write(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("orig")
    spec = compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    try:
        result = run_sandboxed_bash(spec, tmp_path, "echo hacked > .git/config")
        assert result.returncode != 0
        assert (tmp_path / ".git" / "config").read_text() == "orig"
    finally:
        finalize_sandbox(spec, tmp_path)


@pytest.mark.skipif(not _DARWIN or not _AVAILABLE.available, reason="seatbelt is macOS only")
def test_seatbelt_blocks_praxis_read(tmp_path: Path):
    (tmp_path / ".praxis").mkdir()
    (tmp_path / ".praxis" / "worktree.json").write_text("{}")
    spec = compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    try:
        result = run_sandboxed_bash(spec, tmp_path, "cat .praxis/worktree.json")
        assert result.returncode != 0  # 无访问
    finally:
        finalize_sandbox(spec, tmp_path)


@pytest.mark.skipif(not _DARWIN or not _AVAILABLE.available, reason="seatbelt is macOS only")
def test_seatbelt_allows_workspace_write_and_git_read(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("orig")
    spec = compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    try:
        result = run_sandboxed_bash(spec, tmp_path, "echo ok > f.txt && cat .git/config")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "f.txt").read_text() == "ok\n"
        assert "orig" in result.stdout
    finally:
        finalize_sandbox(spec, tmp_path)


@pytest.mark.skipif(not _DARWIN or not _AVAILABLE.available, reason="seatbelt is macOS only")
def test_seatbelt_scratch_writable(tmp_path: Path):
    spec = compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    try:
        probe = Path(spec.scratch_dir) / "probe.txt"
        result = run_sandboxed_bash(spec, tmp_path, f"echo ok > {probe}")
        assert result.returncode == 0, result.stderr
        assert probe.read_text() == "ok\n"
    finally:
        finalize_sandbox(spec, tmp_path)


@pytest.mark.skipif(not _DARWIN or not _AVAILABLE.available, reason="seatbelt is macOS only")
def test_seatbelt_write_via_symlink_form_workspace(tmp_path: Path):
    # 回归:workspace 以词法 symlink 形态传入(macOS /var/folders 生产形态),写仍须放行
    real = tmp_path / "real-ws"
    real.mkdir()
    alias = tmp_path / "alias-ws"
    alias.symlink_to(real)
    spec = compile_sandbox(make_ctx(alias), bash_use_of(alias), alias)
    try:
        result = run_sandboxed_bash(spec, alias, "echo ok > f.txt")
        assert result.returncode == 0, result.stderr
        assert (real / "f.txt").read_text() == "ok\n"
    finally:
        finalize_sandbox(spec, alias)


@pytest.mark.skipif(not _DARWIN or not _AVAILABLE.available, reason="seatbelt is macOS only")
def test_seatbelt_network_blocked_by_default(tmp_path: Path):
    spec = compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    try:
        # 硬断言:默认(network ask)编译产物无 network-outbound 段
        assert "network-outbound" not in (spec.profile_text or "")
        # best-effort 真实检查:沙箱内 curl 必须失败;离线时也失败 → 优雅降级
        result = run_sandboxed_bash(spec, tmp_path, "/usr/bin/curl -m 5 -sS -o /dev/null https://example.com")
        assert result.returncode != 0
    finally:
        finalize_sandbox(spec, tmp_path)


@pytest.mark.skipif(not _DARWIN or not _AVAILABLE.available, reason="seatbelt is macOS only")
def test_seatbelt_runtime_hardlink_probe(tmp_path: Path):
    # S9/D8 实证:沙箱内创建硬链接别名再写——断言安全期望(.git/config 内容不变)。
    # 若穿透:本测试失败 = 发现新缺陷,走 findings 补充节流程,不得弱化断言。
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("orig")
    spec = compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    try:
        run_sandboxed_bash(spec, tmp_path, "ln .git/config alias && echo hacked > alias")
        assert (tmp_path / ".git" / "config").read_text() == "orig"
    finally:
        finalize_sandbox(spec, tmp_path)


@pytest.mark.skipif(not _DARWIN or not _AVAILABLE.available, reason="seatbelt is macOS only")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "S9/D8 findings 补充(实证 2026-11-13 darwin):先存硬链接别名写穿透——"
        "seatbelt 按路径裁决无法表达 inode 规则,echo > alias 改写 .git/config。"
        "结构性修复 = bwrap 复制视图断硬链接(Task 6 延后);若本测试 XPASS 说明平台已修复,应重评。"
    ),
)
def test_seatbelt_preexisting_hardlink_alias_write(tmp_path: Path):
    # 沙箱外预先创建硬链接(模拟攻击者预置),沙箱内经别名写:安全期望仍是 config 不变。
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("orig")
    os.link(tmp_path / ".git" / "config", tmp_path / "prealias")
    spec = compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    try:
        run_sandboxed_bash(spec, tmp_path, "echo hacked > prealias")
        assert (tmp_path / ".git" / "config").read_text() == "orig"
    finally:
        finalize_sandbox(spec, tmp_path)
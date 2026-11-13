"""Task 5:沙箱编译单一来源(设计 §5.3/§4.2;findings S1/S2/S9;D4/D8/D9)。

compile_sandbox 是唯一 profile 构造入口:
- D9 网络单一来源:decide(bash) 的网络裁决 ⟺ compile 的 network-outbound 段;
- S9 preflight:硬链接授权/symlink 边界/symlink 逃逸扫描/扫描失败一律 SandboxRefusal;
- S2 地板穷举:任何 profile 的编译产物恒含 FLOOR_ENTRIES 地板段;
- fail-closed:无后端平台拒绝;trusted-local + write 拒绝(controller ruling 6);
- ceiling 只收紧:网络/写授权被 ceiling 否决;
- scratch 生命周期:mkdtemp 创建、finalize 幂等清理、编译失败不泄漏。

brief 偏差(已记录于 report):test_floor_in_profile_every_policy 的 trusted-local
分支改走 render_profile(allow_write=False)——compile 对 trusted-local 恒派生
allow_write=True,与 "/" read 组合按 ruling 6 必拒,无法产出 spec。
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox, validate_aliases
from praxis.adapters.sandbox.seatbelt import render_profile
from praxis.core.permission import decide
from praxis.protocol.floor import host_private_paths
from praxis.protocol.permission import PermissionContext, SandboxAvailability
from praxis.protocol.policy import (
    NetworkPolicy,
    PathRule,
    PermissionPolicy,
    command_network_decision,
    resolve_policy,
)
from praxis.protocol.sandbox import SandboxRefusal, SandboxSpec
from praxis.protocol.tools import ResolvedUse, ToolDefinition, ToolEffect, ToolTarget
from tests.praxis.conftest import make_minimal_tool

_AVAILABLE = probe_backend()


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def make_ctx(
    workspace: Path,
    *,
    policy: PermissionPolicy | None = None,
    ceilings: tuple[PermissionPolicy, ...] = (),
    sandbox: SandboxAvailability | None = None,
) -> PermissionContext:
    return PermissionContext(
        workspace=str(workspace),
        policy=policy if policy is not None else PermissionPolicy(),
        mode="on-request",
        ceilings=ceilings,
        sandbox=sandbox if sandbox is not None else SandboxAvailability(backend=None, available=False),
    )


def bash_use_of(workspace: Path) -> ResolvedUse:
    return ResolvedUse(
        effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
        targets=(ToolTarget(kind="path", value=str(workspace)),),
    )


def path_use_of(workspace: Path, rel: str, effect: ToolEffect) -> ResolvedUse:
    return ResolvedUse(
        effects=frozenset({effect}),
        targets=(ToolTarget(kind="path", value=str(workspace / rel)),),
    )


def bash_tool_of(workspace: Path):
    return make_minimal_tool(
        definition=ToolDefinition(name="bash", description="bash test tool", input_schema={"type": "object"}),
        static_effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
        resolve_use=lambda arguments: bash_use_of(workspace),
    )


# ---------------------------------------------------------------------------
# probe:唯一可用性探测(ruling 1,bwrap 延后)
# ---------------------------------------------------------------------------


def test_probe_backend_shape():
    availability = probe_backend()
    if sys.platform == "darwin" and os.path.isfile("/usr/bin/sandbox-exec") and os.access("/usr/bin/sandbox-exec", os.X_OK):
        assert availability.backend == "seatbelt" and availability.available
    else:
        # 其他平台一律不可用——不探测 bwrap(D7 Linux fail-closed)
        assert availability.backend is None and not availability.available


# ---------------------------------------------------------------------------
# D9 网络单一来源:decide(bash) ⟺ compile 的 network-outbound 段
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend (bwrap deferred)")
def test_compile_is_single_source_of_network_verdict(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    tool = bash_tool_of(ws)
    use = bash_use_of(ws)
    for net in (
        NetworkPolicy(decision="ask"),
        NetworkPolicy(decision="allow"),
        NetworkPolicy(decision="deny"),
        NetworkPolicy(decision="allow", allowed_domains=("x.com",)),
    ):
        policy = PermissionPolicy(network=net)
        ctx = make_ctx(ws, policy=policy, sandbox=probe_backend())
        decision = decide(tool, use, ctx)
        # 直接调 compile:deny/ask 时管道不会调,但直接调仍须 fail-closed 一致
        spec = compile_sandbox(ctx, use, ws)
        has_network = "network-outbound" in (spec.profile_text or "")
        expected = command_network_decision(resolve_policy(policy, ws)) == "allow"
        assert has_network == expected, f"network={net}"
        # bash 的 decide 放行 ⟺ 命令网络 allow(沙箱可用时第 7 步放行)
        assert decision.is_allow == expected, f"network={net}: {decision}"
        finalize_sandbox(spec, ws)


# ---------------------------------------------------------------------------
# S9 preflight:硬链接授权 / symlink 边界 / symlink 逃逸 / 扫描失败
# ---------------------------------------------------------------------------


def test_hardlink_grant_refused(tmp_path: Path):
    target = tmp_path / "real.txt"
    target.write_text("x")
    os.link(target, tmp_path / "alias.txt")
    policy = PermissionPolicy(filesystem=(PathRule(path="alias.txt", access="read"),))
    with pytest.raises(SandboxRefusal):
        compile_sandbox(make_ctx(tmp_path, policy=policy), path_use_of(tmp_path, "alias.txt", ToolEffect.READ_WORKSPACE), tmp_path)


def test_symlink_boundary_refused(tmp_path: Path):
    (tmp_path / "link").symlink_to(tmp_path / "real")
    policy = PermissionPolicy(filesystem=(PathRule(path="link", access="write"),))
    with pytest.raises(SandboxRefusal):
        compile_sandbox(make_ctx(tmp_path, policy=policy), path_use_of(tmp_path, "link/f", ToolEffect.WRITE_WORKSPACE), tmp_path)


def test_symlink_escape_scan_refused(tmp_path: Path):
    # 授权根内 deny 子树中的 symlink 指向授权文件:alias 无权限但 target 有权限 → 拒绝
    (tmp_path / "locked").mkdir()
    (tmp_path / "open.txt").write_text("x")
    (tmp_path / "locked" / "alias").symlink_to(tmp_path / "open.txt")
    policy = PermissionPolicy(filesystem=(PathRule(path="locked", access="deny"),))
    with pytest.raises(SandboxRefusal):
        compile_sandbox(make_ctx(tmp_path, policy=policy), bash_use_of(tmp_path), tmp_path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permission bits")
def test_scan_failure_refused(tmp_path: Path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o000)
    try:
        with pytest.raises(SandboxRefusal):
            compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    finally:
        blocked.chmod(0o755)


def test_validate_aliases_clean_policy_passes(tmp_path: Path):
    resolved = resolve_policy(PermissionPolicy(), tmp_path, private_paths=host_private_paths())
    validate_aliases(resolved, tmp_path, allow_write=True)  # 不抛即通过


# ---------------------------------------------------------------------------
# S2 地板穷举:任何 profile 的产物恒含地板段(FLOOR_ENTRIES 单一来源)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend (bwrap deferred)")
def test_floor_in_profile_every_policy(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    texts: list[str] = []
    for profile in ("workspace", "read-only"):
        spec = compile_sandbox(make_ctx(ws, policy=PermissionPolicy(profile=profile)), bash_use_of(ws), ws)  # type: ignore[arg-type]
        texts.append(spec.profile_text or "")
        finalize_sandbox(spec, ws)
    # trusted-local:compile 恒派生 allow_write=True → 与 "/" read 组合按 ruling 6 拒绝;
    # 地板断言改走 render_profile(allow_write=False)(只读命令形态)。
    resolved = resolve_policy(PermissionPolicy(profile="trusted-local"), ws)
    texts.append(
        render_profile(resolved, workspace=ws, scratch=tmp_path / "scratch", allow_network=False, allow_write=False, private_paths=())
    )
    for text in texts:
        assert ".praxis" in text and ".agent_memory" in text  # 地板段恒在
        assert re.search(r"\[gG\]\[iI\]\[tT\]", text)  # .git write-deny regex


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend (bwrap deferred)")
def test_trusted_local_write_combination_refused(tmp_path: Path):
    # ruling 6: broad host-root read + allow_write → SandboxRefusal(移植旧 ValueError 语义)
    with pytest.raises(SandboxRefusal):
        compile_sandbox(make_ctx(tmp_path, policy=PermissionPolicy(profile="trusted-local")), bash_use_of(tmp_path), tmp_path)


def test_private_paths_denied_in_profile(tmp_path: Path):
    spec = compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    text = spec.profile_text or ""
    for private in host_private_paths():
        lexical = os.path.abspath(os.path.expanduser(private))
        for form in dict.fromkeys((lexical, os.path.realpath(lexical))):
            assert f'(deny file-read* file-write* (subpath "{form}"))' in text
    finalize_sandbox(spec, tmp_path)


# ---------------------------------------------------------------------------
# fail-closed 分派与 ceiling 收紧
# ---------------------------------------------------------------------------


def test_no_backend_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "praxis.adapters.sandbox.compile.probe_backend",
        lambda: SandboxAvailability(backend=None, available=False),
    )
    with pytest.raises(SandboxRefusal, match="no sandbox backend on this platform"):
        compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend (bwrap deferred)")
def test_ceiling_denies_network(tmp_path: Path):
    ctx = make_ctx(
        tmp_path,
        policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
        ceilings=(PermissionPolicy(network=NetworkPolicy(decision="deny")),),
    )
    spec = compile_sandbox(ctx, bash_use_of(tmp_path), tmp_path)
    assert "network-outbound" not in (spec.profile_text or "")
    finalize_sandbox(spec, tmp_path)


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend (bwrap deferred)")
def test_ceiling_denies_write(tmp_path: Path):
    ctx = make_ctx(tmp_path, ceilings=(PermissionPolicy(profile="read-only"),))
    spec = compile_sandbox(ctx, bash_use_of(tmp_path), tmp_path)
    text = spec.profile_text or ""
    workspace = os.path.realpath(tmp_path)
    assert f'(allow file-write* (require-all (subpath "{workspace}")' not in text
    assert f'(deny file-write* (require-all (subpath "{workspace}")' in text
    finalize_sandbox(spec, tmp_path)


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend (bwrap deferred)")
def test_symlink_form_workspace_still_grants_write(tmp_path: Path):
    # 回归:workspace 以词法 symlink 形态传入(macOS /var/folders ↔ /private/var)
    # 时,allow_write 不得误判 False——规范化与 resolve_policy 同源。
    real = tmp_path / "real-ws"
    real.mkdir()
    alias = tmp_path / "alias-ws"
    alias.symlink_to(real)
    spec = compile_sandbox(make_ctx(alias), bash_use_of(alias), alias)
    text = spec.profile_text or ""
    assert f'(allow file-write* (require-all (subpath "{real}")' in text
    assert f'(deny file-write* (require-all (subpath "{real}")' not in text
    finalize_sandbox(spec, alias)


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend (bwrap deferred)")
def test_workspace_grant_region_always_rendered(tmp_path: Path):
    # ruling 11:无显式 filesystem 规则时,workspace 基础授权区域仍渲染
    spec = compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    assert f'(subpath "{os.path.realpath(tmp_path)}")' in (spec.profile_text or "")
    finalize_sandbox(spec, tmp_path)


# ---------------------------------------------------------------------------
# scratch 生命周期:创建 / 幂等清理 / 异常不抛 / 编译失败不泄漏
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend (bwrap deferred)")
def test_scratch_created_and_finalize_idempotent(tmp_path: Path):
    spec = compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    scratch = Path(spec.scratch_dir)
    assert scratch.is_dir() and scratch.name.startswith("praxis-scratch-")
    assert scratch.parent == Path(tempfile.gettempdir())
    finalize_sandbox(spec, tmp_path)
    assert not scratch.exists()
    finalize_sandbox(spec, tmp_path)  # 幂等:二次清理不抛
    finalize_sandbox(spec, tmp_path)


def test_finalize_never_raises(tmp_path: Path):
    victim = tmp_path / "not-a-dir"
    victim.write_text("x")
    spec = SandboxSpec(backend="seatbelt", argv_prefix=("/usr/bin/sandbox-exec", "-p", "(version 1)"), scratch_dir=str(victim))
    finalize_sandbox(spec, tmp_path)  # rmtree 失败只记日志,不抛
    assert victim.exists()
    deferred = SandboxSpec(backend="bwrap", argv_prefix=("bwrap",), scratch_dir=str(tmp_path / "missing"))
    finalize_sandbox(deferred, tmp_path)  # bwrap 延后:非 seatbelt 一律 no-op


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend (bwrap deferred)")
def test_compile_failure_does_not_leak_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured: list[Path] = []

    def exploding_compile(resolved, **kwargs):
        captured.append(kwargs["scratch"])
        raise SandboxRefusal("render exploded")

    monkeypatch.setattr("praxis.adapters.sandbox.compile.compile_seatbelt", exploding_compile)
    with pytest.raises(SandboxRefusal, match="render exploded"):
        compile_sandbox(make_ctx(tmp_path), bash_use_of(tmp_path), tmp_path)
    assert captured and not captured[0].exists()
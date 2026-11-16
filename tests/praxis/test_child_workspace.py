"""阶段4 T7:子工作区 provisioner——detach 快照 / diff bundle / preimage apply。

覆盖面(逐条对应 brief 与控制器裁决):
- prepare:真 git repo 的 ``git worktree add --detach HEAD`` 快照,父脏文件与
  untracked 都不入快照;落点 = ``host_config_dir()/worktrees/<sha256>/<child>``
  (库外、宿主目录);
- diff/apply round-trip:tracked 修改 + untracked 新文件(含二进制)可重建;
- preimage 逐文件 sha256 复检:不一致 → conflicts 且**整组不写**(双方字节都在);
- ``git apply --check`` 失败 → ``patch_failed`` 整组拒(untracked 也不写);
- 安全面:diff bundle 与 apply 路径集都排除 ``.praxis``/``.git``(嵌套也排除),
  子 ``.praxis/permissions.json`` 改动不出现在 parent;
- discard 幂等且 ``git worktree list`` 干净;
- 非 git 工作区 → ``ChildWorkspaceRequiresGit``(Q5);
- argv 探针:固定 argv 数组、无 shell 拼接;env 只含 child_env 白名单 + 最小
  GIT_* 注入,宿主键零泄漏;
- composition 绑定 ``GitWorktreeProvisioner(env_builder=child_env)``。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from praxis.adapters.workspaces.git import GitWorktreeProvisioner
from praxis.composition import build_session
from praxis.interfaces.sdk import ScriptedAdapter
from praxis.protocol.floor import host_config_dir
from praxis.protocol.workspaces import ChildWorkspaceRequiresGit
from praxis.tools.builtins.child_env import child_env


def _git(cwd: Path, *args: str, input: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        input=input,
        capture_output=True,
        check=True,
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """真 git repo(非 mock):一个 commit + 文本/二进制 tracked 文件。"""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "praxis-test@example.com")
    _git(root, "config", "user.name", "Praxis Test")
    (root / "a.txt").write_text("base\n", encoding="utf-8")
    (root / "bin.dat").write_bytes(b"\x00base-bytes")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    return root


@pytest.fixture
def provisioner() -> GitWorktreeProvisioner:
    return GitWorktreeProvisioner(child_env)


def _expected_workspace(repo: Path, child_id: str) -> Path:
    digest = hashlib.sha256(os.fsencode(str(repo.resolve()))).hexdigest()[:16]
    return host_config_dir() / "worktrees" / digest / child_id


# ---------------------------------------------------------------------------
# prepare:detach 快照 + 落点
# ---------------------------------------------------------------------------


def test_prepare_detached_snapshot_hides_parent_dirty_bytes(
    repo: Path, provisioner: GitWorktreeProvisioner
) -> None:
    # 父工作区脏:修改 tracked + 新增 untracked——都不得进入快照。
    (repo / "a.txt").write_text("parent-dirty\n", encoding="utf-8")
    (repo / "dirty-only.txt").write_text("dirty\n", encoding="utf-8")

    ws = provisioner.prepare(repo, "child-1", "write")

    assert ws == _expected_workspace(repo, "child-1")
    assert ws.is_dir()
    assert ws.is_relative_to(host_config_dir())
    assert not ws.is_relative_to(repo)
    # 快照 = HEAD 字节,不含父脏修改/untracked。
    assert (ws / "a.txt").read_text(encoding="utf-8") == "base\n"
    assert not (ws / "dirty-only.txt").exists()
    # detach:无分支 symbolic ref。
    detached = subprocess.run(
        ["git", "-C", str(ws), "symbolic-ref", "-q", "HEAD"], capture_output=True
    )
    assert detached.returncode != 0
    assert str(ws) in _git(repo, "worktree", "list").stdout.decode()


# ---------------------------------------------------------------------------
# diff/apply round-trip
# ---------------------------------------------------------------------------


def test_diff_apply_roundtrip_includes_untracked_and_binary(
    repo: Path, provisioner: GitWorktreeProvisioner
) -> None:
    ws = provisioner.prepare(repo, "child-1", "write")
    (ws / "a.txt").write_text("child\n", encoding="utf-8")
    (ws / "bin.dat").write_bytes(b"\x00child-bytes")
    (ws / "sub").mkdir()
    (ws / "sub" / "new.txt").write_text("new\n", encoding="utf-8")

    bundle = provisioner.diff(ws)
    assert isinstance(bundle, bytes)
    # 未 apply 前 parent 不变。
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base\n"
    assert not (repo / "sub" / "new.txt").exists()

    result = provisioner.apply(
        repo,
        bundle,
        preimages={
            "a.txt": _sha(b"base\n"),
            "bin.dat": _sha(b"\x00base-bytes"),
            "sub/new.txt": None,
        },
    )
    assert result.conflicts == []
    assert result.applied == ["a.txt", "bin.dat", "sub/new.txt"]
    assert (repo / "a.txt").read_text(encoding="utf-8") == "child\n"
    assert (repo / "bin.dat").read_bytes() == b"\x00child-bytes"
    assert (repo / "sub" / "new.txt").read_text(encoding="utf-8") == "new\n"


# ---------------------------------------------------------------------------
# preimage 复检:conflicts → 整组不写(双方字节都在)
# ---------------------------------------------------------------------------


def test_apply_preimage_mismatch_keeps_both_sides_and_writes_nothing(
    repo: Path, provisioner: GitWorktreeProvisioner
) -> None:
    ws = provisioner.prepare(repo, "child-1", "write")
    (ws / "a.txt").write_text("child\n", encoding="utf-8")
    (ws / "new.txt").write_text("new\n", encoding="utf-8")
    bundle = provisioner.diff(ws)

    # parent 在快照后被并发修改。
    (repo / "a.txt").write_text("parent-moved\n", encoding="utf-8")
    result = provisioner.apply(
        repo,
        bundle,
        preimages={"a.txt": _sha(b"base\n"), "new.txt": None},
    )

    assert result.applied == []
    assert result.conflicts == ["a.txt"]
    # 双方字节都在:parent 保留自己的新字节,child 快照保留 child 字节。
    assert (repo / "a.txt").read_text(encoding="utf-8") == "parent-moved\n"
    assert (ws / "a.txt").read_text(encoding="utf-8") == "child\n"
    # 不半写:非冲突的 untracked 也不落 parent。
    assert not (repo / "new.txt").exists()


# ---------------------------------------------------------------------------
# git apply 失败:patch_failed 整组拒
# ---------------------------------------------------------------------------


def test_apply_patch_failure_rejects_whole_group(
    repo: Path, provisioner: GitWorktreeProvisioner
) -> None:
    ws = provisioner.prepare(repo, "child-1", "write")
    (ws / "a.txt").write_text("child\n", encoding="utf-8")
    (ws / "new.txt").write_text("new\n", encoding="utf-8")
    bundle = provisioner.diff(ws)

    # 等长破坏 hunk 头:尺寸/untracked 完好,只有 patch 解析失败。
    corrupted = bundle.replace(b"@@ -1 +1 @@", b"@@ X1 +1 @@")
    assert corrupted != bundle
    with pytest.raises(RuntimeError, match="patch_failed"):
        provisioner.apply(
            repo,
            corrupted,
            preimages={"a.txt": _sha(b"base\n"), "new.txt": None},
        )

    assert (repo / "a.txt").read_text(encoding="utf-8") == "base\n"
    assert not (repo / "new.txt").exists()


# ---------------------------------------------------------------------------
# 安全面:diff/apply 路径集排除 .praxis(与 .git)
# ---------------------------------------------------------------------------


def test_praxis_and_git_are_excluded_from_bundle_and_apply(
    repo: Path, provisioner: GitWorktreeProvisioner
) -> None:
    ws = provisioner.prepare(repo, "child-1", "write")
    (ws / "a.txt").write_text("child\n", encoding="utf-8")
    # 顶层与嵌套 .praxis(floor:任意路径成分命中)。
    (ws / ".praxis").mkdir()
    (ws / ".praxis" / "permissions.json").write_text('{"always": []}\n', encoding="utf-8")
    nested = ws / "nested" / "proj" / ".praxis"
    nested.mkdir(parents=True)
    (nested / "permissions.json").write_text("{}\n", encoding="utf-8")

    bundle = provisioner.diff(ws)
    assert b".praxis" not in bundle

    result = provisioner.apply(repo, bundle, preimages={"a.txt": _sha(b"base\n")})
    assert result.applied == ["a.txt"]
    assert result.conflicts == []
    assert not (repo / ".praxis" / "permissions.json").exists()
    assert not (repo / "nested" / "proj" / ".praxis" / "permissions.json").exists()


# ---------------------------------------------------------------------------
# discard:幂等 + worktree list 干净
# ---------------------------------------------------------------------------


def test_discard_removes_worktree_and_is_idempotent(
    repo: Path, provisioner: GitWorktreeProvisioner
) -> None:
    ws = provisioner.prepare(repo, "child-1", "write")
    assert ws.is_dir()

    provisioner.discard(ws)
    assert not ws.exists()
    assert str(ws) not in _git(repo, "worktree", "list").stdout.decode()
    # 第二次 discard 是幂等 no-op。
    provisioner.discard(ws)


# ---------------------------------------------------------------------------
# Q5:非 git 工作区明确拒绝
# ---------------------------------------------------------------------------


def test_prepare_non_git_workspace_raises(
    tmp_path: Path, provisioner: GitWorktreeProvisioner
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ChildWorkspaceRequiresGit, match="git"):
        provisioner.prepare(plain, "child-1", "write")


def test_prepare_rejects_repository_subdirectory(
    repo: Path, provisioner: GitWorktreeProvisioner
) -> None:
    """快照以 HEAD 检出整仓;子目录工作区会让快照根 ≠ Session 工作区,拒绝。"""
    subdir = repo / "pkg"
    subdir.mkdir()
    with pytest.raises(RuntimeError, match="repository root"):
        provisioner.prepare(subdir, "child-1", "write")


# ---------------------------------------------------------------------------
# argv/env 探针:固定数组、无 shell、宿主 env 零泄漏
# ---------------------------------------------------------------------------


def test_git_subprocess_uses_argv_array_and_isolated_env(
    repo: Path, provisioner: GitWorktreeProvisioner, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run = subprocess.run
    records: list[tuple[object, dict[str, object]]] = []

    def spy(argv: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        records.append((argv, kwargs))
        return real_run(argv, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setenv("PRAXIS_HOST_SECRET_PROBE", "must-not-leak")
    monkeypatch.setattr("praxis.adapters.workspaces.git.subprocess.run", spy)

    provisioner.prepare(repo, "child-1", "write")

    assert records
    allowed = set(child_env(Path("/tmp"), repo)) | {
        "GIT_TERMINAL_PROMPT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
    }
    for argv, kwargs in records:
        # 固定 argv 数组:list[str],首项 git,任何元素都不是 shell 命令行。
        assert isinstance(argv, list)
        assert argv and argv[0] == "git"
        assert all(isinstance(part, str) for part in argv)
        assert all(" " not in part for part in argv)
        assert kwargs.get("shell") is not True
        env = kwargs["env"]
        assert isinstance(env, dict)
        assert set(env) <= allowed
        assert "PRAXIS_HOST_SECRET_PROBE" not in env
        assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["HOME"] != str(Path.home())


# ---------------------------------------------------------------------------
# composition:provisioner 由组合根绑定(child_env 唯一实现注入)
# ---------------------------------------------------------------------------


def test_composition_binds_git_provisioner_with_child_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    created: list[object] = []

    class SpyProvisioner(GitWorktreeProvisioner):
        def __init__(self, env_builder: object) -> None:
            created.append(env_builder)
            super().__init__(env_builder)  # type: ignore[arg-type]

    monkeypatch.setattr("praxis.composition.GitWorktreeProvisioner", SpyProvisioner)

    async def scenario() -> None:
        session = build_session(
            workspace,
            ScriptedAdapter([]),
            thread_id="host",
            enable_subagents=True,
        )
        await session.aclose()

    asyncio.run(scenario())

    assert created == [child_env]
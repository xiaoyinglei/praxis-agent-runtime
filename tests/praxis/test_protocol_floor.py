"""praxis.protocol.floor — 地板单一数据源测试(Task 3 + review fixes)。

验证:
- FloorEntry dataclass + FLOOR_ENTRIES tuple 存在且正确。
- FloorEntry.access 校验（仅 "read"/"deny"）。
- host_private_paths() 返回两个宿主私有目录字符串（monkeypatch PRAXIS_CONFIG_DIR）。
- floor_reason 矩阵穷举、任意成分匹配、private containment、broken symlink non-raise。
- 输入规范化：~ 相对路径、.. 穿越均被 lexically resolved 后正确拒绝。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from praxis.protocol.floor import (
    FLOOR_ENTRIES,
    FloorEntry,
    floor_reason,
    host_private_paths,
)

# ── FLOOR_ENTRIES 常量检查 ────────────────────────────────────────────

def test_floor_entries_count() -> None:
    assert len(FLOOR_ENTRIES) == 3


def test_floor_entries_values() -> None:
    expected = (
        FloorEntry(".git", "read"),
        FloorEntry(".praxis", "deny"),
        FloorEntry(".agent_memory", "deny"),
    )
    assert FLOOR_ENTRIES == expected


# ── Finding 2: FloorEntry.access 校验 ─────────────────────────────────

def test_floor_entry_valid_read() -> None:
    e = FloorEntry("test", "read")
    assert e.access == "read"


def test_floor_entry_valid_deny() -> None:
    e = FloorEntry("test", "deny")
    assert e.access == "deny"


def test_floor_entry_invalid_access_raises() -> None:
    with pytest.raises(ValueError):
        FloorEntry("test", "write")


def test_floor_entry_invalid_access_none_raises() -> None:
    with pytest.raises(ValueError):
        FloorEntry("test", "")


# ── 矩阵穷举测试 ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [".git", ".praxis", ".agent_memory", ".GIT", ".Git"])
@pytest.mark.parametrize("write", [False, True])
def test_floor_matrix(tmp_path: Path, name: str, write: bool) -> None:
    target = tmp_path / name / "config"
    reason = floor_reason(target, write=write, private_paths=())
    if name.casefold() == ".git":
        assert (reason != "") == write  # 只读：write 拒 read 放
    else:
        assert reason != ""            # deny：读写都拒


def test_floor_any_component(tmp_path: Path) -> None:
    assert floor_reason(
        tmp_path / "sub" / ".praxis" / "x",
        write=False,
        private_paths=(),
    ) != ""


def test_host_private_denied(tmp_path: Path) -> None:
    priv = str(tmp_path / "private")
    assert floor_reason(
        Path(priv) / "f",
        write=False,
        private_paths=(priv,),
    ) != ""


# ── Finding 3: Broken symlink WITH non-existent private root ───────────
# Exercises _is_under OSError path: samefile on missing candidate/root.

def test_floor_no_raise_on_broken_symlink(tmp_path: Path) -> None:
    link = tmp_path / "l"
    link.symlink_to(tmp_path / "missing")
    floor_reason(link, write=True, private_paths=())  # 不抛


def test_floor_broken_symlink_with_missing_private_root(tmp_path: Path) -> None:
    """Broken symlink + non-existent private root → _is_under OSError path covered."""
    fake_private = str(tmp_path / "ghost_private")
    link = tmp_path / "l"
    link.symlink_to(tmp_path / "also_missing")
    # Should not raise even though private root doesn't exist at all
    result = floor_reason(link, write=True, private_paths=(fake_private,))
    assert isinstance(result, str)  # 不抛


# ── Normal paths ───────────────────────────────────────────────────────

def test_normal_path_no_hit(tmp_path: Path) -> None:
    assert floor_reason(
        tmp_path / "src" / "a.py",
        write=True,
        private_paths=(),
    ) == ""


def test_empty_private_paths_no_hit(tmp_path: Path) -> None:
    assert floor_reason(
        tmp_path / "src",
        write=True,
        private_paths=(),
    ) == ""


def test_write_read_entry_deny(tmp_path: Path) -> None:
    """access='read' 时 write=True → 拒绝。"""
    target = tmp_path / ".git" / "HEAD"
    assert floor_reason(target, write=True, private_paths=()) != ""


def test_read_read_entry_allow(tmp_path: Path) -> None:
    """access='read' 时 write=False 且不命中 deny → 放行（无 hit）。"""
    target = tmp_path / ".git" / "HEAD"
    assert floor_reason(target, write=False, private_paths=()) == ""


# ── Finding 5: ~ 相对路径与 .. 穿越 — 实际拒绝测试 ─────────────────────
# 以下测试使用真实的 ~ 路径和含 .. 的相对路径，验证 lexical resolve 后落入私有根被拒绝。

def test_tilde_relative_normalized_into_private(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real tilde path Path('~/...') expands → contained in private root → denied."""
    monkeypatch.delenv("PRAXIS_CONFIG_DIR", raising=False)
    paths = host_private_paths()
    # ~ 路径经 expanduser+abspath 归一到 ~/.config/praxis，应触发 deny
    tilde_candidate = Path("~/.config/praxis/secret.json")
    assert floor_reason(tilde_candidate, write=False, private_paths=paths) != ""
    assert floor_reason(tilde_candidate, write=True, private_paths=paths) != ""


def test_dotdot_traversal_into_private_monkeypatch_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relative path with '..' that resolves INSIDE a private root → denied."""
    monkeypatch.chdir(str(tmp_path))
    priv_jobs = f"{tempfile.gettempdir()}/praxis-command-jobs-{os.getuid()}"
    rel = os.path.relpath(priv_jobs, str(tmp_path))  # e.g. '../../../usr/lib/foo'
    candidate = Path(rel) / "x"                       # relative path containing ..
    paths = host_private_paths()
    # 经 expanduser + os.path.abspath 解析后落入私有根，应拒绝（写）
    assert floor_reason(candidate, write=True, private_paths=paths) != ""


# ── Finding 5: 绝对路径在合成私有根内被正确拒绝 ────────────────────────

def test_abs_private_canonical_path_denied(tmp_path: Path) -> None:
    """Absolute path inside private root is correctly denied."""
    priv_dir = tmp_path / "secure"
    priv_dir.mkdir()
    inner_file = priv_dir / "data.bin"
    abs_priv = str(priv_dir)
    assert floor_reason(inner_file, write=True, private_paths=(abs_priv,)) != ""
    assert floor_reason(inner_file, write=False, private_paths=(abs_priv,)) != ""


# ── Finding 4: host_private_paths 测试 — monkeypatch PRAXIS_CONFIG_DIR ─

def test_host_private_paths_returns_two_strings() -> None:
    paths = host_private_paths()
    assert isinstance(paths, tuple)
    assert len(paths) == 2
    assert all(isinstance(p, str) for p in paths)


def test_host_private_config_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env → ~/.config/praxis is entry 0; jobs-dir matches tempdir+uid."""
    monkeypatch.delenv("PRAXIS_CONFIG_DIR", raising=False)
    paths = host_private_paths()
    expected_config = str(Path.home().resolve() / ".config" / "praxis")
    assert paths[0] == expected_config
    expected_jobs = f"{tempfile.gettempdir()}/praxis-command-jobs-{os.getuid()}"
    assert paths[1] == expected_jobs


def test_host_private_config_custom_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """PRAXIS_CONFIG_DIR set → exact expanded value is entry 0; jobs-dir unchanged."""
    custom = "/tmp/my-custom-praxis-config"
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", custom)
    paths = host_private_paths()
    assert paths[0] == os.path.expanduser(os.path.expandvars(custom))
    expected_jobs = f"{tempfile.gettempdir()}/praxis-command-jobs-{os.getuid()}"
    assert paths[1] == expected_jobs


def test_host_private_jobs_dir_uid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jobs-dir contains the current UID, deterministic across calls."""
    monkeypatch.delenv("PRAXIS_CONFIG_DIR", raising=False)
    uid = os.getuid()
    paths_a = host_private_paths()
    paths_b = host_private_paths()
    assert paths_a[1].endswith(f"-{uid}")
    assert paths_b[1].endswith(f"-{uid}")
    assert paths_a[1] == paths_b[1]

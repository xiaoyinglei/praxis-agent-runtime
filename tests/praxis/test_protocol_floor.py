"""praxis.protocol.floor — 地板单一数据源测试(Task 3)。

验证:
- FloorEntry dataclass + FLOOR_ENTRIES tuple 存在且正确。
- host_private_paths() 返回两个宿主私有目录字符串。
- floor_reason 矩阵穷举、任意成分匹配、private containment、broken symlink non-raise。
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


def test_floor_no_raise_on_broken_symlink(tmp_path: Path) -> None:
    link = tmp_path / "l"
    link.symlink_to(tmp_path / "missing")
    floor_reason(link, write=True, private_paths=())  # 不抛


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


# ── host_private_paths ────────────────────────────────────────────────

def test_host_private_paths_returns_two_strings() -> None:
    paths = host_private_paths()
    assert isinstance(paths, tuple)
    assert len(paths) == 2
    assert all(isinstance(p, str) for p in paths)


def test_host_private_contains_config_dir() -> None:
    config = host_private_paths()
    home_config = str(Path.home().resolve() / ".config" / "praxis")
    assert home_config in config[0] or config[0].endswith("/praxis")

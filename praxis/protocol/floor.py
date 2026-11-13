"""协议层地板单一数据源(设计 §5.3 第 4 步;findings S2;D4)。

这是全项目唯一的安全地板定义。核心判定管线(Task 4)和沙箱 profile
编译器(Task 5)均消费此模块——不得有第二套地板列表。
"""

from __future__ import annotations

import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from praxis.protocol.policy import Access


@dataclass(frozen=True, slots=True)
class FloorEntry:
    """一个地板条目的不可变描述。

    Parameters:
        name: 目录名（casefold + NFC 匹配任意路径成分）。
        access: "read" = 只读（write 拒，read 放）；"deny" = 无访问。
    """

    name: str
    access: Access


# ── 地板常量：唯一来源 ────────────────────────────────────────────────

FLOOR_ENTRIES: tuple[FloorEntry, ...] = (
    FloorEntry(".git", "read"),
    FloorEntry(".praxis", "deny"),
    FloorEntry(".agent_memory", "deny"),
)


def _normalize_component(name: str) -> str:
    """NFC 规范化 + casefold，用于匹配比较。"""
    return unicodedata.normalize("NFC", name).casefold()


def _is_under(candidate: Path, root_str: str) -> bool:
    """判断 candidate 是否位于 root_str 下，使用 NFC/casefold 归一化。

    non-strict: 不抛 OSError / broken symlink / missing target。
    逻辑与 policy._under_path 同风格，但本地实现、不跨模块导入私有符号。
    """
    try:
        if candidate.is_relative_to(root_str):
            return True
        root_normed = _normalize_component(root_str)
        for ancestor in (candidate, *candidate.parents):
            try:
                ancestor_normed = _normalize_component(str(ancestor))
                if ancestor_normed == root_normed:
                    try:
                        if ancestor.samefile(root_str):
                            return True
                    except OSError:
                        pass
            except (OSError, ValueError):
                continue
    except (OSError, ValueError):
        pass
    return False


def host_private_paths() -> tuple[str, ...]:
    """返回宿主私有目录元组。

    - PRAXIS_CONFIG_DIR env 展开（不存在则为 ~/.config/praxis）。
    - <tempdir>/praxis-command-jobs-<uid>。
    """
    config_env = os.environ.get("PRAXIS_CONFIG_DIR")
    if config_env:
        config_dir = os.path.expanduser(os.path.expandvars(config_env))
    else:
        config_dir = str(Path.home().resolve() / ".config" / "praxis")

    temp_base = tempfile.gettempdir()
    uid = os.getuid()
    private_jobs_dir = f"{temp_base}/praxis-command-jobs-{uid}"

    return (config_dir, private_jobs_dir)


def floor_reason(
    path: Path,
    *,
    write: bool = False,
    private_paths: tuple[str, ...],
) -> str:
    """判定给定 path 是否触达安全地板。

    Returns:
        "" = 无地板命中（通过），否则为拒绝理由字符串。

    Logic:
        1. 检查 path 的任何路径成分是否命中 FLOOR_ENTRIES（casefold+NFC）。
           - deny → 读写都拒。
           - read → write 时拒，read 时放行。
        2. 检查 path 是否在 private_paths 下（NFC/casefold+samefile tolerant）。
           - 在私有目录下 → 读写都拒。
        3. 全程 non-strict：不对文件系统做任何存在性假设，不抛异常。
    """
    # ── 步骤 1: FLOOR_ENTRIES 成分匹配 ───────────────────────────────
    path_str = str(path)
    floor_hit: str | None = None

    for entry in FLOOR_ENTRIES:
        normed_entry_name = _normalize_component(entry.name)
        parts = Path(path_str).parts
        for part in parts:
            if _normalize_component(part) == normed_entry_name:
                if entry.access == "deny":
                    floor_hit = (
                        f"floor denied by protected directory: {entry.name}"
                    )
                elif entry.access == "read" and write:
                    floor_hit = (
                        f"floor denied by read-only directory: {entry.name}"
                    )
                break  # 命中第一个就够，继续其他 entry 找更严的

    if floor_hit is not None:
        return floor_hit

    # ── 步骤 2: private_paths Containment ────────────────────────────
    for priv in private_paths:
        if _is_under(path, priv):
            return "path is under a private directory"

    return ""

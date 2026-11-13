"""协议层地板单一数据源(设计 §5.3 第 4 步;findings S2;D4)。

这是全项目唯一的安全地板定义。核心判定管线(Task 4)和沙箱 profile
编译器(Task 5)均消费此模块——不得有第二套地板列表。

调用约定：
- floor_reason 在内部对 path 做 expanduser() + os.path.abspath 词法规范化
  （不解析 symlink，保持 B10 非严格 totality）。
- caller 检查 symlink 目标时应同时传入词法路径与真实路径两个候选（沿用
  protocol/policy.py:path_reason 的双候选语义）；Task 4 将在 decide 层完成
  这一调用模式。
"""

from __future__ import annotations

import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class FloorEntry:
    """一个地板条目的不可变描述。

    Parameters:
        name: 目录名（casefold + NFC 匹配任意路径成分）。
        access: "read" = 只读（write 拒，read 放）；"deny" = 无访问。

    Validation: access is restricted to "read"/"deny"; any other value
    raises ValueError (D4 floor semantics only use read/deny).
    """

    name: str
    access: Literal["read", "deny"]

    def __post_init__(self) -> None:
        if self.access not in ("read", "deny"):
            raise ValueError(
                f"FloorEntry.access must be 'read' or 'deny', got {self.access!r}"
            )


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
    对 candidate 做 expanduser() + os.path.abspath 词法规范化（不解析 symlink）。
    """
    try:
        lex = candidate.expanduser()
        canon = os.path.abspath(str(lex))
    except (OSError, ValueError):
        return False
    try:
        if Path(canon).is_relative_to(root_str):
            return True
        root_normed = _normalize_component(root_str)
        for ancestor in (Path(canon), *Path(canon).parents):
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
    write: bool,
    private_paths: tuple[str, ...],
) -> str:
    """判定给定 path 是否触达安全地板。

    Returns:
        "" = 无地板命中（通过），否则为拒绝理由字符串。

    Logic:
        1. 对输入做词法规范化：expanduser() + os.path.abspath（不解析 symlink，
           保持 B10 non-strict totality）。
        2. 检查任何路径成分是否命中 FLOOR_ENTRIES（casefold+NFC）。
           - deny → 读写都拒。
           - read → write 时拒，read 时放行。
        3. 检查是否在 private_paths 下（NFC/casefold+samefile tolerant）。
           - 在私有目录下 → 读写都拒。
        4. 全程 non-strict：不对文件系统做任何存在性假设，不抛异常。
    """
    # ── 步骤 0: 输入词法规范化 ───────────────────────────────────────
    try:
        lex = Path(path).expanduser()
        path_str = os.path.abspath(str(lex))
    except (OSError, ValueError):
        return "path cannot be safely normalized"

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

    # ── 步骤 1: private_paths Containment ────────────────────────────
    for priv in private_paths:
        if _is_under(path, priv):
            return "path is under a private directory"

    return ""

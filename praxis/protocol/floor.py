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
from functools import lru_cache
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


@lru_cache(maxsize=8192)
def _cached_path_forms(value: str) -> tuple[Path, ...]:
    """一个路径的两种容器形态:词法(expanduser+abspath)与内核 realpath。

    S-3:``host_private_paths()`` 刻意不 resolve env 分支(返回值逐字不变),
    但 macOS 的 ``/var/folders`` ↔ ``/private/var/folders`` 意味着「父目录给的
    词法形态」与「目标 realpath 形态」可能不同。containment 判定必须两种形态
    都比——单一词法比较会让进程内工具绕过私有目录地板。

    N-5:逐条目过滤会为每个文件重复判定同一份私有根;``resolve()`` 是 syscall
    密集操作。这里按路径字符串做**有界**缓存(唯一实现仍在本函数),把常量
    root 的 realpath 计算摊薄到每条目一次候选 resolve。

    non-strict:broken symlink / 缺失目标不抛(``Path.resolve`` 默认 strict=False)。
    """
    try:
        lexical = Path(os.path.abspath(str(Path(value).expanduser())))
    except (OSError, ValueError):
        return ()
    forms = [lexical]
    try:
        real = lexical.resolve()
    except (OSError, RuntimeError, ValueError):
        real = None
    if real is not None:
        forms.append(real)
    return tuple(dict.fromkeys(forms))


def _path_forms(value: Path | str) -> tuple[Path, ...]:
    """``_cached_path_forms`` 的 Path/str 入口(缓存键统一为 str)。"""
    try:
        key = os.fspath(value)
    except TypeError:
        return ()
    return _cached_path_forms(key)


def _is_under(candidate: Path, root_str: str) -> bool:
    """判断 candidate 是否位于 root_str 下，使用 NFC/casefold 归一化。

    non-strict: 不抛 OSError / broken symlink / missing target。
    逻辑与 policy._under_path 同风格，但本地实现、不跨模块导入私有符号。
    S-3:候选与私有根都按词法/realpath 双形态比较(``_path_forms``),因此
    ``/var/folders/...`` 与 ``/private/var/folders/...`` 不再互相漏判。
    """
    candidates = _path_forms(candidate)
    roots = _path_forms(root_str)
    if not candidates or not roots:
        return False
    for canon in candidates:
        try:
            if any(canon.is_relative_to(root) for root in roots):
                return True
            for ancestor in (canon, *canon.parents):
                ancestor_normed = _normalize_component(str(ancestor))
                for root in roots:
                    if ancestor_normed != _normalize_component(str(root)):
                        continue
                    try:
                        if ancestor.samefile(root):
                            return True
                    except OSError:
                        pass
        except (OSError, ValueError):
            continue
    return False


def host_config_dir() -> Path:
    """宿主全局配置目录(唯一来源;adapters 只消费,分层不破)。

    ``PRAXIS_CONFIG_DIR`` 给出时先 expandvars + expanduser,再经 ``Path()``
    归一化——尾斜杠、重复的 ``//`` 与 ``.`` 成分会被折叠,因此返回值不保证与
    环境变量字符串逐字相同(也不 resolve symlink);未给出时
    ``~/.config/praxis``。其下固定为 ``config.sqlite`` 与 ``credentials/``。
    """
    config_env = os.environ.get("PRAXIS_CONFIG_DIR")
    if config_env:
        return Path(os.path.expanduser(os.path.expandvars(config_env)))
    return Path.home().resolve() / ".config" / "praxis"


def command_jobs_dir() -> Path:
    """后台命令会话 marker 目录(Task 15 复用;已在地板私有集合里)。

    返回值语义与阶段 2 的 host_private_paths()[1] 逐字一致——刻意不做
    resolve():tempdir 在 macOS 上可能经 /var → /private/var 改变字面形态,
    host_private_paths() 的返回值集合必须保持不变。
    """
    return Path(tempfile.gettempdir()) / f"praxis-command-jobs-{os.getuid()}"


def host_private_paths() -> tuple[str, ...]:
    """返回宿主私有目录元组(两个公开定义函数的组合)。

    - PRAXIS_CONFIG_DIR env 展开(不存在则为 ~/.config/praxis)。
    - <tempdir>/praxis-command-jobs-<uid>。
    """
    return (str(host_config_dir()), str(command_jobs_dir()))


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

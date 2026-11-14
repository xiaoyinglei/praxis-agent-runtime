"""workspace 路径围栏(移植旧 filesystem.py:849-856 的 _checked_path,去掉 WorkspaceRuntime 依赖)。

解析相对/绝对路径后 ``resolve()``,结果必须严格位于 workspace 内(含符号链接解析),
越界一律 ``ToolValidationError``。阶段 2 的地板规则在此之前拦截。

另提供 ``atomic_write_text``:同目录 tmp + ``os.replace`` 的原子落盘语义。
filesystem 的 write 与 patch 的逐文件写入共用这一份实现(计划裁决 R-P1,
禁止复制第二份)。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from praxis.protocol.tools import ToolValidationError


def checked_path(
    workspace: Path,
    value: str,
    *,
    write: bool = False,
    allow_root: bool = False,
) -> Path:
    """Resolve ``value`` against ``workspace`` and confine it strictly inside.

    ``write=True`` 表示允许目标尚不存在(父目录由调用方创建);围栏本身
    不做存在性检查——read 工具把缺失文件映射为 ``not_found`` 结果而非异常。

    ``allow_root=True`` 只为目录清单(list_files)开口:工作区根本身是合法
    的**被列出对象**(它仍严格在围栏内,只是不再是"必须位于其下"的严格子路径)。
    read/write 一律保持默认,不允许把工作区根当作文件目标。
    """
    base = workspace.resolve()
    candidate = Path(value)
    target = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if not target.is_relative_to(base) or (target == base and not allow_root):
        raise ToolValidationError(
            path="$.path",
            message=f"path is outside the workspace: {value!r}",
        )
    return target


def atomic_write_text(path: Path, content: str) -> None:
    """同目录 tmp + os.replace(移植旧 _atomic_write_text/atomic_bytes 的原子语义)。"""
    encoded = content.encode("utf-8")
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(directory))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

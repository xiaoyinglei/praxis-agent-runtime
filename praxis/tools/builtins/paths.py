"""workspace 路径围栏(移植旧 filesystem.py:849-856 的 _checked_path,去掉 WorkspaceRuntime 依赖)。

解析相对/绝对路径后 ``resolve()``,结果必须严格位于 workspace 内(含符号链接解析),
越界一律 ``ToolValidationError``。阶段 2 的地板规则在此之前拦截。
"""

from __future__ import annotations

from pathlib import Path

from praxis.protocol.tools import ToolValidationError


def checked_path(workspace: Path, value: str, *, write: bool = False) -> Path:
    """Resolve ``value`` against ``workspace`` and confine it strictly inside.

    ``write=True`` 表示允许目标尚不存在(父目录由调用方创建);围栏本身
    不做存在性检查——read 工具把缺失文件映射为 ``not_found`` 结果而非异常。
    """
    base = workspace.resolve()
    candidate = Path(value)
    target = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if target == base or not target.is_relative_to(base):
        raise ToolValidationError(
            path="$.path",
            message=f"path is outside the workspace: {value!r}",
        )
    return target

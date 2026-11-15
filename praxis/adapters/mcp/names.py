"""MCP 名字规范化与工具名拼装的**全仓唯一实现点**(3B 笔记 §2.1;T20 规则⑯)。

旧实现有 6 处各自拼装:``tools/integrations/mcp.py:83`` 的 canonical 名,
``runtime/mcp.py:219/441`` 与 ``runtime/mcp_catalog.py:256`` 的 server
通配,``harness/session.py:141/1367`` 的全域通配。它们逐字收敛到本模块:
任何其他模块都不得再写字面量拼装(T20 静态规则钉死),分类/展示一律消费
``tool.source`` 而不是前缀字符串。

规范化步骤逐字移植旧 ``normalize_mcp_name``(``tools/integrations/mcp.py:
70-75``):strip → lower → ``-``/空格 → ``_`` → 非法字符 → ``_`` → **折叠
任意连续下划线**为单个 ``_`` → 去首尾 ``_``。注意旧实现折叠的是 ``_+``
(不只是 ``__``),因此 ``a--b`` 与 ``a__b`` 都得到 ``a_b``。
"""

from __future__ import annotations

import re

_INVALID_NAME = re.compile(r"[^a-z0-9_]+")


def normalize_mcp_name(value: str) -> str:
    """把 server/tool 名规范化为 ``[a-z0-9_]`` 串(可能为空)。"""
    if not isinstance(value, str):
        raise TypeError("MCP name must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _INVALID_NAME.sub("_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def canonical_tool_name(server: str, tool: str) -> str:
    """``<server>`` 与 ``<tool>`` 的规范化工具名(旧 canonical 名的唯一等价物)。"""
    server_name = normalize_mcp_name(server)
    tool_name = normalize_mcp_name(tool)
    if not server_name or not tool_name:
        raise ValueError("MCP server and tool names must normalize to non-empty values")
    return f"mcp__{server_name}__{tool_name}"


def server_wildcard(server: str) -> str:
    """某个 server 的工具通配名(策略预检/目录过滤的唯一拼装点)。"""
    server_name = normalize_mcp_name(server)
    if not server_name:
        raise ValueError("MCP server name must normalize to a non-empty value")
    return f"mcp__{server_name}__*"


__all__ = ["canonical_tool_name", "normalize_mcp_name", "server_wildcard"]
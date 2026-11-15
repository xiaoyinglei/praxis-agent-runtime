"""MCP ``tools/list`` 完整目录:分页、资源上限与全或无发布(T16 交接;T17)。

逐条移植旧 ``agent_runtime/runtime/mcp.py:522-549`` 的 ``_list_all_tools``
(审查点名的正确设计):

- **cursor 续取**:首页无参,后续页用 ``PaginatedRequestParams(cursor=...)``;
  ``nextCursor`` 为 None 才是终点。
- **全或无**:任何一页越界/畸形都直接抛 ``ValueError``,绝不把第一页的
  半成品目录交给调用方(部分目录会让多页 server 的工具静默消失)。
- **资源上限**(逐字):``_MAX_CATALOG_TOOLS=4096``、
  ``_MAX_CATALOG_PAGES=256``、``_MAX_CATALOG_BYTES=4 MiB``。字节口径是
  ``tool.model_dump_json()`` 的 UTF-8 长度(旧实现同源)。
- **重复工具名拒绝**:跨页同名即错(同名工具会让 canonical 名碰撞)。
- **重复/空 cursor 拒绝**:``nextCursor`` 指向已请求过的 cursor 或空串,
  说明 server 在打转;立即拒绝而不是无限循环。

本模块只做目录读取,不建连接、不做策略判定(T19 的 manager 负责逐 server
捕获异常并转 ``mcp_server_unavailable``)。
"""

from __future__ import annotations

from mcp.client.session import ClientSession
from mcp.types import ListToolsResult, PaginatedRequestParams
from mcp.types import Tool as McpTool

_MAX_CATALOG_TOOLS = 4096
_MAX_CATALOG_PAGES = 256
_MAX_CATALOG_BYTES = 4 * 1024 * 1024


async def list_all_tools(session: ClientSession) -> tuple[McpTool, ...]:
    """取回**完整且受上限约束**的工具目录;失败绝不返回部分页。

    Raises:
        ValueError: 重复工具名、重复/空 cursor、工具数/字节数/页数越界。
    """
    tools: list[McpTool] = []
    names: set[str] = set()
    cursors: set[str] = set()
    cursor: str | None = None
    size = 0
    for _ in range(_MAX_CATALOG_PAGES):
        page: ListToolsResult = (
            await session.list_tools()
            if cursor is None
            else await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
        )
        for tool in page.tools:
            if tool.name in names:
                raise ValueError("MCP tools/list returned a duplicate tool name")
            names.add(tool.name)
            size += len(tool.model_dump_json().encode("utf-8"))
            if len(names) > _MAX_CATALOG_TOOLS or size > _MAX_CATALOG_BYTES:
                raise ValueError("MCP tools/list exceeded the catalog resource limit")
            tools.append(tool)
        cursor = page.nextCursor
        if cursor is None:
            return tuple(tools)
        if not cursor or cursor in cursors:
            raise ValueError("MCP tools/list returned a repeated or empty cursor")
        cursors.add(cursor)
    raise ValueError("MCP tools/list exceeded the catalog page limit")


__all__ = [
    "_MAX_CATALOG_BYTES",
    "_MAX_CATALOG_PAGES",
    "_MAX_CATALOG_TOOLS",
    "list_all_tools",
]
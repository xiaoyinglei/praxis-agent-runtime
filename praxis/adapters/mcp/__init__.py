"""Praxis MCP 适配器(T15-T18)。

**有意不做包级 re-export**(T18 修复轮 M4):各子模块按职责直接 import——
``praxis.adapters.mcp.config`` / ``.names`` / ``.lifecycle`` / ``.catalog`` /
``.descriptors`` / ``.http_transport``。包级聚合会与
``adapters.config.credentials`` 对 ``.names`` 的依赖形成工程化循环
(``credentials`` → ``mcp.__init__`` → ``http_transport`` → ``credentials``);
空 ``__init__`` 让任何装载顺序(先 mcp / 先 credentials / 先 composition)
都安全。``owned_connection`` 是包内 owner-task 原语,不是包级公开面。

后续任务在此包内扩展:manager/CLI 接线(T19)。
"""

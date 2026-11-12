"""Praxis 唯一组合根(设计 §4.1;CI 规则 7:Session 构造只允许出现在本文件)。

build_session 把分层组件装配成一个可用会话,顺序即依赖序:
建 ``.praxis/`` 目录 → Store(单写者线程) → EventBus → **sweep**(打开即收编
崩溃遗留的 running turn → interrupted,§5.8;活执行者持锁,Sweep 跳过,绝不误伤)
→ 默认三内置工具(read/write/bash,可由 tools 参数整体替换)→ PermissionContext
→ 线程 ensure/create → Session。

thread_id 是外部输入且直接参与 lockfile 命名:在这里(边界)先验拒绝空串、路径
分隔符、NUL 与 ``..`` 成分——locks.py 会再拦一次,但边界拒绝让错误在碰文件系统
与建库之前发生(fail early at the boundary)。

任何装配失败都要关掉已建的 Store,绝不留写者线程。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from praxis.core.events import EventBus
from praxis.core.recovery import sweep
from praxis.core.session import Session
from praxis.protocol.models import ModelAdapter
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.tools import Tool
from praxis.store.log import Store
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.filesystem import create_read_tool, create_write_tool


def _validate_thread_id(thread_id: str) -> None:
    if not thread_id:
        raise ValueError("thread_id must be non-empty")
    if "/" in thread_id or "\\" in thread_id:
        raise ValueError(f"thread_id must not contain path separators: {thread_id!r}")
    if "\x00" in thread_id:
        raise ValueError("thread_id must not contain NUL bytes")
    if ".." in Path(thread_id).parts:
        raise ValueError(f"thread_id must not contain '..' components: {thread_id!r}")


def build_session(
    workspace: Path,
    adapter: ModelAdapter,
    *,
    policy: PermissionPolicy | None = None,
    thread_id: str | None = None,
    tools: Sequence[Tool] | None = None,
    max_steps: int = 50,
) -> Session:
    """装配一个 Session:唯一组合根。CLI/TUI/host 一律经 sdk.open_session 走到这里。"""
    if thread_id is not None:
        _validate_thread_id(thread_id)
    root = Path(workspace)
    praxis_dir = root / ".praxis"
    praxis_dir.mkdir(parents=True, exist_ok=True)
    store = Store(praxis_dir / "praxis.sqlite")
    try:
        bus = EventBus()
        sweep(store, bus, root)
        tool_seq = (
            tools
            if tools is not None
            else (create_read_tool(root), create_write_tool(root), create_bash_tool(root))
        )
        tool_map: dict[str, Tool] = {tool.definition.name: tool for tool in tool_seq}
        effective_policy = policy if policy is not None else PermissionPolicy()
        permissions = PermissionContext(
            workspace=str(root), policy=effective_policy, mode=effective_policy.approval
        )
        resolved_thread = thread_id if thread_id is not None else store.create_thread()
        store.ensure_thread(resolved_thread)
        return Session(store, bus, root, adapter, tool_map, permissions, resolved_thread, max_steps)
    except BaseException:
        store.close()
        raise

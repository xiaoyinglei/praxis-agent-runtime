"""T2:ToolSource 枚举与 Tool.source(设计 §5.10:替代 revision 字符串前缀分类)。

分类唯一来源是 ``tool.source`` 枚举;任何代码不得用 "integration-mcp"/"mcp__"
之类字符串前缀解析来源(T20 上 CI 规则)。本文件钉住三件事:
1. 枚举值逐字(BUILTIN="builtin"、MCP="mcp")与 Tool.source 的末位默认;
2. 管道写出的 tool_call payload 带 ``"source": str``;
3. 旧 item(payload 无 source)读侧 fail-open:消费方不得因缺键崩溃,按 BUILTIN 归类。
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest

from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.interfaces.cli.render import render_item
from praxis.protocol.items import ItemKind
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.tools import Tool, ToolCall, ToolSource
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn
from tests.praxis.conftest import make_minimal_tool


def _ctx(store: Store, workspace: Path, tools: dict[str, Tool]) -> ExecCtx:
    """最小 ExecCtx(与 test_pipeline.make_ctx 同口径):turn 以 running 起步。"""
    turn = create_turn(store, turn_id="turn-1", thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools=tools,
        permissions=PermissionContext(
            workspace=str(workspace), policy=PermissionPolicy(), mode="on-request"
        ),
        turn_id=turn.turn_id,
        thread_id="thread-1",
        policy_revision=PermissionPolicy().revision,
    )


# ---------------------------------------------------------------------------
# 枚举与 Tool 字段面
# ---------------------------------------------------------------------------


def test_tool_source_member_values_are_verbatim() -> None:
    assert ToolSource.BUILTIN.value == "builtin"
    assert ToolSource.MCP.value == "mcp"
    assert isinstance(ToolSource.MCP, str)  # StrEnum:payload 可直接存字符串


def test_tool_source_defaults_to_builtin() -> None:
    assert make_minimal_tool().source is ToolSource.BUILTIN


def test_tool_source_accepts_mcp() -> None:
    tool = make_minimal_tool(source=ToolSource.MCP)
    assert tool.source is ToolSource.MCP


def test_source_is_last_field_with_default() -> None:
    """frozen slots dataclass 末位带默认:现有全部位置构造零改动。"""
    field = dataclasses.fields(Tool)[-1]
    assert field.name == "source"
    assert field.default is ToolSource.BUILTIN


def test_positional_construction_of_all_previous_fields_still_works() -> None:
    base = make_minimal_tool()
    positional = [
        getattr(base, field.name) for field in dataclasses.fields(Tool) if field.name != "source"
    ]
    clone = Tool(*positional)
    assert clone.source is ToolSource.BUILTIN


@pytest.mark.parametrize("bad", ["mcp", "builtin", 1, None, object()])
def test_non_enum_source_raises_type_error(bad: object) -> None:
    with pytest.raises(TypeError, match="source must be a ToolSource"):
        make_minimal_tool(source=bad)


# ---------------------------------------------------------------------------
# 管道 payload:tool_call item 带 source 字符串
# ---------------------------------------------------------------------------


def test_pipeline_payload_carries_builtin_source(store: Store, workspace: Path) -> None:
    ctx = _ctx(store, workspace, {"minimal": make_minimal_tool()})
    asyncio.run(execute(ToolCall("c1", "minimal", {}), ctx))
    call_item = ctx.store.read_turn_items(ctx.turn_id)[0]
    assert call_item.kind == ItemKind.TOOL_CALL
    assert call_item.payload["source"] == "builtin"


def test_pipeline_payload_carries_mcp_source(store: Store, workspace: Path) -> None:
    ctx = _ctx(store, workspace, {"minimal": make_minimal_tool(source=ToolSource.MCP)})
    asyncio.run(execute(ToolCall("c1", "minimal", {}), ctx))
    call_item = ctx.store.read_turn_items(ctx.turn_id)[0]
    assert call_item.payload["source"] == "mcp"


# ---------------------------------------------------------------------------
# 旧 item 兼容:payload 无 source 的读侧 fail-open
# ---------------------------------------------------------------------------


def _legacy_tool_call_payload() -> dict[str, object]:
    """3B 之前的 tool_call payload(键集合:call_id/tool_name/arguments/effects/
    targets/args_digest/idempotent),有意不含 "source"。"""
    return {
        "call_id": "legacy-1",
        "tool_name": "integration-mcp__search",
        "arguments": {"q": "x"},
        "effects": ["network"],
        "targets": [],
        "args_digest": "0" * 64,
        "idempotent": True,
    }


def test_legacy_tool_call_without_source_is_fail_open(store: Store, workspace: Path) -> None:
    """旧库 item 读侧不得崩溃:缺 source 键按 BUILTIN 归类(消费口径统一为 .get)。

    注意 tool_name 故意带 "integration-mcp" 前缀:分类只看 source 缺省值,
    绝不从名字前缀反推来源(§5.10 反目标)。
    """
    turn = create_turn(store, turn_id="turn-legacy", thread_id="thread-1", binding={})
    store.append_items(
        "thread-1",
        turn.turn_id,
        [ItemEntry(kind=ItemKind.TOOL_CALL, payload=_legacy_tool_call_payload())],
    )
    item = store.read_turn_items(turn.turn_id)[0]
    assert "source" not in item.payload  # 旧库原样读出,不被读路径改写
    # 消费方口径:payload.get(key, default) —— 缺键回退 BUILTIN,不抛。
    assert ToolSource(item.payload.get("source", ToolSource.BUILTIN.value)) is ToolSource.BUILTIN
    # 审计展示最后一道读端永不抛(B1 fail-open)。
    assert "legacy-1" in render_item(item)
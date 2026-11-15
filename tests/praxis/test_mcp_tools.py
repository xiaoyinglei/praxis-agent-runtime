"""Task 17:MCP descriptor→Tool 暴露、目录上限、输出归一与调用期守卫唯一表达(Q3)。

**覆盖面**(brief Step 1 + controller 裁决):
- effects/targets 矩阵(stdio/http × hints × destructive),执行策略常量;
- allowlist 缺失 ⇒ 构造期 ValueError;allowlist 过滤与 HTTP ``read_only`` 过滤;
- 目录分页(cursor 续取、全或无)、重复工具名/重复 cursor/空 cursor 拒绝、
  4096 工具 / 256 页 / 4 MiB 资源上限(常量逐字 + 边界行为);
- 输出归一:content text/image/embedded-resource(文本与 blob)、
  ``structuredContent``、``isError``、legacy fallback 形状、真实 SDK 模型;
- 调用期守卫唯一两处:config sha 漂移 ⇒ ``mcp_config_changed``(经真实
  ``pipeline.execute``);catalog 成员缺失 ⇒ ``mcp_tool_unknown``;
  意外 SDK 异常(ExceptionGroup 摊平)⇒ ``mcp_call_failed`` 有界;
- 超限输出经管道按 ``max_model_output_bytes=1_000_000`` 截断;
- 三入口一致性形状样本:同一 MCP Tool 经 ``permission.decide`` 与 builtin
  完全同判;经 ``pipeline.execute`` 三次(模型/CLI/resume 共同入口)结果一致;
  全仓 grep 无 ``integration-mcp`` 前缀分类字符串。

**R-T17-1 收敛**(controller 裁决):brief 的"live schema 指纹漂移 ⇒
``mcp_tool_changed``"不实现——praxis 没有 live schema 通道,跨 server 的
schema 变化由重连(config sha + T19 重启语义)承接,旧批准失效由
decide + bound ``policy_revision``/``args_digest`` 承担。descriptor 仍计算
``schema_fingerprint`` 供重连/审计使用,但 run 不读它。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent, ToolAnnotations
from mcp.types import Tool as McpTool

from praxis.adapters.mcp import catalog
from praxis.adapters.mcp.catalog import (
    _MAX_CATALOG_BYTES,
    _MAX_CATALOG_PAGES,
    _MAX_CATALOG_TOOLS,
    list_all_tools,
)
from praxis.adapters.mcp.config import HttpServerConfig, StdioServerConfig
from praxis.adapters.mcp.descriptors import (
    McpToolDescriptor,
    _build_tool,
    mcp_tools_for_connection,
)
from praxis.adapters.mcp.lifecycle import McpConnection
from praxis.adapters.mcp.output import normalize_mcp_output
from praxis.core import permission
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.protocol.items import ItemKind
from praxis.protocol.json import thaw_json
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy, ToolRule
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResultStatus,
    Tool,
    ToolCall,
    ToolEffect,
    ToolSource,
    ToolTarget,
    ToolValidationError,
)
from praxis.store.log import Store
from praxis.store.turns import create_turn

# ---------------------------------------------------------------------------
# 脚手架:配置 / 假 session / 假连接 / 最小 ExecCtx
# ---------------------------------------------------------------------------


def _stdio_config(**overrides: Any) -> StdioServerConfig:
    values: dict[str, Any] = dict(
        name="srv",
        command="python",
        args=(),
        env={},
        cwd=None,
        tools_allowlist=frozenset(),
        allow_all_tools=True,
        startup_timeout_seconds=20.0,
    )
    values.update(overrides)
    return StdioServerConfig(**values)


def _http_config(**overrides: Any) -> HttpServerConfig:
    values: dict[str, Any] = dict(
        name="remote",
        url="https://mcp.example.com/rpc",
        headers={},
        bearer_token_env_var=None,
        keychain=False,
        read_only=False,
        tools_allowlist=frozenset(),
        allow_all_tools=True,
        startup_timeout_seconds=20.0,
    )
    values.update(overrides)
    return HttpServerConfig(**values)


def _mcp_tool(
    name: str,
    *,
    description: str = "does things",
    schema: Mapping[str, Any] | None = None,
    read_only: bool | None = None,
    destructive: bool | None = None,
    idempotent: bool | None = None,
) -> McpTool:
    annotations = None
    if any(value is not None for value in (read_only, destructive, idempotent)):
        annotations = ToolAnnotations(
            readOnlyHint=bool(read_only),
            destructiveHint=bool(destructive),
            idempotentHint=bool(idempotent),
        )
    return McpTool(
        name=name,
        description=description,
        inputSchema=dict(schema or {"type": "object", "properties": {}}),
        annotations=annotations,
    )


class _FakeSession:
    """只实现 T17 消费的两个协程:call_tool 与 list_tools 由子类补。"""

    def __init__(self, result: object | None = None, *, error: BaseException | None = None) -> None:
        self.result = result if result is not None else CallToolResult(content=[TextContent(type="text", text="ok")])
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, **_: Any) -> object:
        self.calls.append((name, dict(arguments or {})))
        if self.error is not None:
            raise self.error
        return self.result


def _connection(
    *,
    server_name: str = "srv",
    tools: tuple[McpTool, ...] = (),
    session: object | None = None,
) -> McpConnection:
    return McpConnection(
        server_name=server_name,
        session=session if session is not None else _FakeSession(),  # type: ignore[arg-type]
        tools=tools,
    )


def _ctx(
    store: Store,
    workspace: Path,
    tool: Tool,
    *,
    policy: PermissionPolicy | None = None,
    mode: str = "on-request",
    turn_id: str = "turn-1",
) -> ExecCtx:
    effective = policy if policy is not None else PermissionPolicy()
    turn = create_turn(store, turn_id=turn_id, thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools={tool.definition.name: tool},
        permissions=PermissionContext(workspace=str(workspace), policy=effective, mode=mode),  # type: ignore[arg-type]
        turn_id=turn.turn_id,
        thread_id="thread-1",
        policy_revision=effective.revision,
    )


def _calls(items: list[Any], kind: str) -> list[Mapping[str, Any]]:
    return [item.payload for item in items if item.kind == kind]


# ---------------------------------------------------------------------------
# effects / targets 矩阵与执行策略
# ---------------------------------------------------------------------------


def test_stdio_tool_has_no_process_or_network_effect_and_catalog_target() -> None:
    connection = _connection(tools=(_mcp_tool("search"),))
    (tool,) = mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")

    assert tool.source is ToolSource.MCP
    assert tool.definition.name == "mcp__srv__search"
    assert tool.static_effects == frozenset()
    use = tool.resolve_use(tool.validate_input({}))
    assert use.effects == frozenset()
    assert use.targets == (ToolTarget(kind="mcp_tool", value="mcp__srv__search"),)


def test_stdio_tool_never_declares_execute_process() -> None:
    # server 进程已在 T16 沙箱内;每次调用只是 client RPC,不得再声明进程副作用。
    connection = _connection(tools=(_mcp_tool("run", destructive=True),))
    (tool,) = mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")
    assert ToolEffect.EXECUTE_PROCESS not in tool.static_effects
    assert tool.static_effects == frozenset({ToolEffect.DESTRUCTIVE})


def test_destructive_hint_blocks_auto_allow_and_concurrency() -> None:
    connection = _connection(tools=(_mcp_tool("drop", destructive=True, idempotent=True),))
    (tool,) = mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")
    assert tool.concurrency_safe is False
    # brief/controller: idempotent = idempotent_hint(默认 False),不被 read_only 放大。
    assert tool.idempotent is True


def test_idempotent_hint_defaults_false_and_sets_true() -> None:
    plain = _connection(tools=(_mcp_tool("plain"),))
    (plain_tool,) = mcp_tools_for_connection(plain, _stdio_config(), config_sha_provider=lambda: "sha")
    assert plain_tool.idempotent is False
    assert plain_tool.concurrency_safe is True

    hinted = _connection(tools=(_mcp_tool("safe", idempotent=True, read_only=True),))
    (hinted_tool,) = mcp_tools_for_connection(hinted, _stdio_config(), config_sha_provider=lambda: "sha")
    assert hinted_tool.idempotent is True
    assert hinted_tool.concurrency_safe is True


def test_http_tool_declares_network_effect_and_url_target() -> None:
    config = _http_config()
    connection = _connection(server_name="remote", tools=(_mcp_tool("query"),))
    (tool,) = mcp_tools_for_connection(connection, config, config_sha_provider=lambda: "sha-a")

    assert tool.static_effects == frozenset({ToolEffect.NETWORK})
    use = tool.resolve_use(tool.validate_input({}))
    assert use.effects == frozenset({ToolEffect.NETWORK})
    assert use.targets == (
        ToolTarget(kind="mcp_tool", value="mcp__remote__query"),
        ToolTarget(kind="url", value=config.url),
    )


def test_http_destructive_tool_adds_destructive_effect() -> None:
    connection = _connection(server_name="remote", tools=(_mcp_tool("purge", destructive=True),))
    (tool,) = mcp_tools_for_connection(
        connection, _http_config(), config_sha_provider=lambda: "sha-a"
    )
    assert tool.static_effects == frozenset({ToolEffect.NETWORK, ToolEffect.DESTRUCTIVE})
    assert tool.concurrency_safe is False


def test_execution_policy_matches_legacy_values() -> None:
    connection = _connection(tools=(_mcp_tool("echo"),))
    (tool,) = mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")
    assert tool.timeout_seconds == 30.0
    assert tool.max_model_output_bytes == 1_000_000
    assert tool.cancellation_mode is CancellationMode.REMOTE_BEST_EFFORT
    assert tool.interrupt_behavior is InterruptBehavior.CANCEL
    # Q3:MCP 永不自带权限回调;策略统一交 decide。
    assert tool.check_permission is None


def test_description_falls_back_like_legacy() -> None:
    connection = _connection(tools=(_mcp_tool("nameless", description=""),))
    (tool,) = mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")
    assert tool.definition.description == "Configured MCP tool."


# ---------------------------------------------------------------------------
# allowlist / read_only 过滤(构造期语义)
# ---------------------------------------------------------------------------


def test_missing_allowlisted_tool_fails_at_conversion() -> None:
    config = _stdio_config(tools_allowlist=frozenset({"present", "missing"}), allow_all_tools=False)
    connection = _connection(tools=(_mcp_tool("present"),))
    with pytest.raises(ValueError, match="missing allowlisted tools: missing"):
        mcp_tools_for_connection(connection, config, config_sha_provider=lambda: "sha-a")


def test_allowlist_filters_and_allow_all_tools_includes_everything() -> None:
    tools = (_mcp_tool("alpha"), _mcp_tool("beta"))
    restricted = _stdio_config(tools_allowlist=frozenset({"alpha"}), allow_all_tools=False)
    (alpha,) = mcp_tools_for_connection(
        _connection(tools=tools), restricted, config_sha_provider=lambda: "sha-a"
    )
    assert alpha.definition.name == "mcp__srv__alpha"

    permissive = _stdio_config(tools_allowlist=frozenset({"alpha"}), allow_all_tools=True)
    names = {
        tool.definition.name
        for tool in mcp_tools_for_connection(
            _connection(tools=tools), permissive, config_sha_provider=lambda: "sha-a"
        )
    }
    assert names == {"mcp__srv__alpha", "mcp__srv__beta"}


def test_http_read_only_filter_keeps_only_reader_tools() -> None:
    tools = (
        _mcp_tool("reader", read_only=True),
        _mcp_tool("reader_destructive", read_only=True, destructive=True),
        _mcp_tool("writer"),
    )
    (tool,) = mcp_tools_for_connection(
        _connection(server_name="remote", tools=tools),
        _http_config(read_only=True),
        config_sha_provider=lambda: "sha-a",
    )
    assert tool.definition.name == "mcp__remote__reader"


def test_duplicate_canonical_names_are_rejected() -> None:
    connection = _connection(tools=(_mcp_tool("a-b"), _mcp_tool("a_b")))
    with pytest.raises(ValueError, match="duplicate canonical MCP tool name: mcp__srv__a_b"):
        mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")


def test_descriptor_schema_fingerprint_is_canonical_and_sensitive() -> None:
    """descriptor 的指纹是 canonical JSON 的 sha256(R-T17-1:run 不读它)。

    它保留给 T19 重连/审计;跨 server 的 schema 变化由重连承接,不在工具层
    制造第二决策路径。键序无关、schema 内容变化即变,证明指纹真实可用。
    """
    base = McpToolDescriptor(
        server_name="srv",
        tool_name="t",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        },
    )
    reordered = McpToolDescriptor(
        server_name="srv",
        tool_name="t",
        description="d",
        input_schema={
            "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
            "type": "object",
        },
    )
    changed = McpToolDescriptor(
        server_name="srv",
        tool_name="t",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
        },
    )
    assert len(base.schema_fingerprint) == 64
    assert base.schema_fingerprint == reordered.schema_fingerprint
    assert base.schema_fingerprint != changed.schema_fingerprint
    assert base.canonical_name == "mcp__srv__t"


# ---------------------------------------------------------------------------
# 目录分页与资源上限(旧 _list_all_tools 逐条移植)
# ---------------------------------------------------------------------------


class _FakeListSession:
    def __init__(self, pages: list[ListToolsResult]) -> None:
        self.pages = pages
        self.requested: list[str | None] = []

    async def list_tools(
        self, cursor: str | None = None, *, params: PaginatedRequestParams | None = None
    ) -> ListToolsResult:
        requested = None if params is None else params.cursor
        self.requested.append(requested)
        return self.pages[len(self.requested) - 1]


def test_catalog_constants_are_verbatim() -> None:
    assert _MAX_CATALOG_TOOLS == 4096
    assert _MAX_CATALOG_PAGES == 256
    assert _MAX_CATALOG_BYTES == 4_194_304


def test_multi_page_catalog_is_followed_to_the_end() -> None:
    session = _FakeListSession(
        [
            ListToolsResult(tools=[_mcp_tool("a"), _mcp_tool("b")], nextCursor="c1"),
            ListToolsResult(tools=[_mcp_tool("c")]),
        ]
    )
    tools = asyncio.run(list_all_tools(session))  # type: ignore[arg-type]
    assert tuple(tool.name for tool in tools) == ("a", "b", "c")
    assert session.requested == [None, "c1"]


def test_duplicate_tool_name_across_pages_is_rejected() -> None:
    session = _FakeListSession(
        [
            ListToolsResult(tools=[_mcp_tool("same")], nextCursor="c1"),
            ListToolsResult(tools=[_mcp_tool("same")]),
        ]
    )
    with pytest.raises(ValueError, match="duplicate tool name"):
        asyncio.run(list_all_tools(session))  # type: ignore[arg-type]


@pytest.mark.parametrize("cursor", ["c1", ""])
def test_repeated_or_empty_cursor_is_rejected(cursor: str) -> None:
    session = _FakeListSession(
        [
            ListToolsResult(tools=[_mcp_tool("a")], nextCursor="c1"),
            ListToolsResult(tools=[_mcp_tool("b")], nextCursor=cursor),
        ]
    )
    with pytest.raises(ValueError, match="repeated or empty cursor"):
        asyncio.run(list_all_tools(session))  # type: ignore[arg-type]


def test_page_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "_MAX_CATALOG_PAGES", 2)
    session = _FakeListSession(
        [
            ListToolsResult(tools=[_mcp_tool("a")], nextCursor="c1"),
            ListToolsResult(tools=[_mcp_tool("b")], nextCursor="c2"),
            ListToolsResult(tools=[_mcp_tool("c")]),
        ]
    )
    with pytest.raises(ValueError, match="page limit"):
        asyncio.run(list_all_tools(session))  # type: ignore[arg-type]


def test_tool_count_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "_MAX_CATALOG_TOOLS", 2)
    session = _FakeListSession([ListToolsResult(tools=[_mcp_tool("a"), _mcp_tool("b"), _mcp_tool("c")])])
    with pytest.raises(ValueError, match="catalog resource limit"):
        asyncio.run(list_all_tools(session))  # type: ignore[arg-type]


def test_byte_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "_MAX_CATALOG_BYTES", 64)
    session = _FakeListSession([ListToolsResult(tools=[_mcp_tool("big", description="x" * 256)])])
    with pytest.raises(ValueError, match="catalog resource limit"):
        asyncio.run(list_all_tools(session))  # type: ignore[arg-type]


def test_catalog_never_publishes_a_partial_page(monkeypatch: pytest.MonkeyPatch) -> None:
    # 第二页越界 ⇒ 整体失败,调用方拿不到第一页的半成品(全或无)。
    monkeypatch.setattr(catalog, "_MAX_CATALOG_TOOLS", 2)
    session = _FakeListSession(
        [
            ListToolsResult(tools=[_mcp_tool("a")], nextCursor="c1"),
            ListToolsResult(tools=[_mcp_tool("b"), _mcp_tool("c")]),
        ]
    )
    with pytest.raises(ValueError):
        asyncio.run(list_all_tools(session))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 输出归一(legacy :172-241 形状)
# ---------------------------------------------------------------------------


def _normalize(raw: object) -> Any:
    return normalize_mcp_output(raw, server_name="Srv", tool_name="do-it")


def test_normalize_text_block() -> None:
    result = _normalize({"content": [{"type": "text", "text": "hello"}]})
    assert result.tool_name == "mcp__srv__do_it"
    assert result.status is ResultStatus.OK
    assert result.is_error is False
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].data["text"] == "hello"
    assert result.metadata == {"mcp_server": "Srv", "mcp_tool": "do-it"}


def test_normalize_image_block_keeps_legacy_mime_type_key() -> None:
    result = _normalize(
        {"content": [{"type": "image", "data": "aGk=", "mimeType": "image/png"}]}
    )
    assert result.content[0].type == "image"
    assert result.content[0].data == {"data": "aGk=", "mime_type": "image/png"}


def test_normalize_embedded_resource_with_text_becomes_text_block() -> None:
    result = _normalize(
        {
            "content": [
                {
                    "type": "resource",
                    "resource": {"uri": "file:///w/note.txt", "text": "body"},
                }
            ]
        }
    )
    assert result.content[0].type == "text"
    assert result.content[0].data == {
        "text": "body",
        "resource_uri": "file:///w/note.txt",
        "truncated": False,
    }


def test_normalize_embedded_blob_resource_becomes_resource_block() -> None:
    result = _normalize(
        {
            "content": [
                {
                    "type": "resource",
                    "resource": {"uri": "file:///w/blob.bin", "blob": "AA=="},
                }
            ]
        }
    )
    assert result.content[0].type == "resource"
    assert result.content[0].data == {"uri": "file:///w/blob.bin"}


def test_normalize_structured_content_is_preserved_in_metadata() -> None:
    # praxis protocol 删除了 structured_content 字段;结构化值进 metadata,
    # 与 mcp_server/mcp_tool 同仓(旧 NormalizedToolOutput.structured_content 的落点)。
    result = _normalize(
        {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {"count": 2, "items": ["a", "b"]},
        }
    )
    assert thaw_json(result.metadata["structured_content"]) == {"count": 2, "items": ["a", "b"]}


def test_normalize_is_error_maps_to_mcp_tool_error() -> None:
    result = _normalize({"content": [{"type": "text", "text": "boom\n"}], "isError": True})
    assert result.status is ResultStatus.ERROR
    assert result.is_error is True
    assert result.error_code == "mcp_tool_error"
    assert result.error_message == "boom\n"
    assert result.metadata == {"mcp_server": "Srv", "mcp_tool": "do-it"}


def test_normalize_is_error_without_text_uses_fallback_message() -> None:
    result = _normalize({"content": [], "isError": True})
    assert result.error_code == "mcp_tool_error"
    assert result.error_message == "MCP tool reported an error"


def test_normalize_legacy_fallback_text_images_resources() -> None:
    result = _normalize({"text": "legacy", "images": ["aW1n"], "resources": ["file:///r"]})
    assert [block.type for block in result.content] == ["text", "image", "resource"]
    assert result.content[0].data["text"] == "legacy"
    assert result.content[1].data["data"] == "aW1n"
    assert result.content[2].data["uri"] == "file:///r"


def test_normalize_real_sdk_call_tool_result() -> None:
    result = _normalize(
        CallToolResult(
            content=[TextContent(type="text", text="sdk")],
            structuredContent={"n": 1},
        )
    )
    assert result.content[0].data["text"] == "sdk"
    assert result.metadata["structured_content"] == {"n": 1}


# ---------------------------------------------------------------------------
# 调用期守卫:Q3 唯一两处(配置漂移 + catalog 成员)
# ---------------------------------------------------------------------------


def test_config_sha_drift_is_refused_before_the_call(store: Store, workspace: Path) -> None:
    session = _FakeSession()
    holder = {"sha": "sha-before"}
    connection = _connection(tools=(_mcp_tool("echo"),), session=session)
    (tool,) = mcp_tools_for_connection(
        connection, _stdio_config(), config_sha_provider=lambda: holder["sha"]
    )
    holder["sha"] = "sha-after"

    result = asyncio.run(execute(ToolCall("c1", tool.definition.name, {}), _ctx(store, workspace, tool)))

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "mcp_config_changed"
    assert "config changed; reconnect" in (result.error_message or "")
    assert session.calls == []


def test_unknown_tool_against_catalog_snapshot_is_refused() -> None:
    session = _FakeSession()
    descriptor = McpToolDescriptor(
        server_name="srv",
        tool_name="ghost",
        description="ghost tool",
        input_schema={"type": "object"},
    )
    tool = _build_tool(
        descriptor,
        connection=_connection(tools=(), session=session),
        catalog_names=frozenset(),
        config_sha_provider=lambda: "sha-a",
        config_sha_at_connect="sha-a",
    )
    result = asyncio.run(tool.run({}, ToolExecContext()))  # type: ignore[arg-type]

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "mcp_tool_unknown"
    # 同一事实在 use 解析期也 fail-closed(管道会折成 invalid_input,不执行)。
    with pytest.raises(ToolValidationError, match="not in the connected catalog"):
        tool.resolve_use({})
    assert session.calls == []


def test_sdk_exception_group_failure_is_bounded_and_call_scoped(store: Store, workspace: Path) -> None:
    error = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [RuntimeError("transport exploded"), ValueError("x" * 2_000)],
    )
    connection = _connection(tools=(_mcp_tool("echo"),), session=_FakeSession(error=error))
    (tool,) = mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")

    result = asyncio.run(execute(ToolCall("c1", tool.definition.name, {}), _ctx(store, workspace, tool)))

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "mcp_call_failed"
    message = result.error_message or ""
    assert "transport exploded" in message
    assert "unhandled errors in a TaskGroup" not in message
    assert len(message) <= 512


def test_is_error_result_flows_through_pipeline(store: Store, workspace: Path) -> None:
    session = _FakeSession(
        CallToolResult(content=[TextContent(type="text", text="server said no")], isError=True)
    )
    connection = _connection(tools=(_mcp_tool("echo"),), session=session)
    (tool,) = mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")

    result = asyncio.run(execute(ToolCall("c1", tool.definition.name, {}), _ctx(store, workspace, tool)))

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "mcp_tool_error"
    assert result.metadata["mcp_server"] == "srv"
    assert result.metadata["mcp_tool"] == "echo"


def test_oversized_output_is_truncated_at_the_declared_budget(store: Store, workspace: Path) -> None:
    session = _FakeSession(CallToolResult(content=[TextContent(type="text", text="y" * 1_000_001)]))
    connection = _connection(tools=(_mcp_tool("echo"),), session=session)
    (tool,) = mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")
    assert tool.max_model_output_bytes == 1_000_000

    result = asyncio.run(execute(ToolCall("c1", tool.definition.name, {}), _ctx(store, workspace, tool)))

    assert result.truncated is True
    assert len(result.text.encode("utf-8")) <= 1_000_000


# ---------------------------------------------------------------------------
# 三入口一致性形状样本(唯一 decide / 唯一管道)
# ---------------------------------------------------------------------------


def _mcp_tool_for_consistency() -> Tool:
    connection = _connection(tools=(_mcp_tool("echo"),))
    (tool,) = mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")
    return tool


@pytest.mark.parametrize(
    "policy",
    [
        PermissionPolicy(),
        PermissionPolicy(tools=(ToolRule(pattern="mcp__srv__echo", decision="allow"),)),
        PermissionPolicy(tools=(ToolRule(pattern="mcp__srv__echo", decision="ask"),)),
        PermissionPolicy(tools=(ToolRule(pattern="mcp__srv__echo", decision="deny"),)),
    ],
)
def test_decide_is_identical_for_mcp_and_builtin_sources(policy: PermissionPolicy) -> None:
    mcp_tool = _mcp_tool_for_consistency()
    builtin_tool = dataclasses.replace(mcp_tool, source=ToolSource.BUILTIN)
    args = mcp_tool.validate_input({})
    use = mcp_tool.resolve_use(args)
    ctx = PermissionContext(workspace="/tmp/ws", policy=policy, mode="on-request")

    mcp_decision = permission.decide(mcp_tool, use, ctx)
    builtin_decision = permission.decide(builtin_tool, use, ctx)

    assert (mcp_decision.verdict, mcp_decision.reason, mcp_decision.suggestions) == (
        builtin_decision.verdict,
        builtin_decision.reason,
        builtin_decision.suggestions,
    )


def test_same_mcp_tool_is_identical_across_three_entries(store: Store, workspace: Path) -> None:
    """模型/CLI/resume 三入口共用 execute;同一 MCP Tool 三次结局逐字一致。"""
    session = _FakeSession(CallToolResult(content=[TextContent(type="text", text="pong")]))
    connection = _connection(tools=(_mcp_tool("echo"),), session=session)
    (tool,) = mcp_tools_for_connection(connection, _stdio_config(), config_sha_provider=lambda: "sha-a")

    outcomes: list[tuple[Any, ...]] = []
    for index, entry in enumerate(("model", "cli", "resume")):
        turn_id = f"turn-{entry}"
        ctx = _ctx(store, workspace, tool, turn_id=turn_id)
        result = asyncio.run(execute(ToolCall(f"c{index}", tool.definition.name, {}), ctx))
        items = ctx.store.read_turn_items(turn_id)
        call_payload = _calls(items, ItemKind.TOOL_CALL)[0]
        assert call_payload["source"] == "mcp"
        outcomes.append((result.status, result.text, result.error_code, call_payload["source"]))

    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert len(session.calls) == 3


def test_praxis_contains_no_integration_mcp_prefix_classification() -> None:
    """§5.10 反目标:分类只读 tool.source,任何代码不得用名字前缀。"""
    root = Path(__file__).resolve().parents[2] / "praxis"
    offenders = [
        str(path)
        for path in root.rglob("*.py")
        if "integration-mcp" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_auxiliary_json_payload_is_bounded() -> None:
    # 形状样本:结构化值必须是 JSON 兼容(与 protocol/freeze_json 同口径)。
    result = _normalize({"content": [], "structuredContent": {"ok": True}})
    payload = thaw_json(result.metadata)
    assert isinstance(payload, dict)
    assert payload["structured_content"] == {"ok": True}
    assert json.loads(json.dumps(payload))["structured_content"] == {"ok": True}
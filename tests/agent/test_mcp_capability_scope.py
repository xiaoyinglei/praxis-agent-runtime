from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from mcp.types import ListToolsResult, Tool, ToolAnnotations
from typer.testing import CliRunner

from agent_runtime import mcp_admin
from agent_runtime.cli import agent_app
from agent_runtime.permission_policy import NetworkPolicy, NetworkRule, PermissionPolicy
from agent_runtime.runtime import mcp as runtime
from agent_runtime.tools.permissions import ToolExecutionContext, UseToolDecision, can_use_tool
from agent_runtime.tools.tool import ToolApprovalProfile


@pytest.fixture
def anyio_backend():
    return "asyncio"


def entry(config):
    return yaml.safe_load(config.read_text())["servers"][0]


@pytest.mark.parametrize("read_only", [True, False])
def test_github_preset_discovers_server_tools_without_an_implicit_allowlist(tmp_path, monkeypatch, read_only):
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "settings"))
    args = ["mcp", "add", "github", "--bearer-token-env-var", "TOKEN"]
    if not read_only:
        args.append("--allow-actions")
    result = CliRunner().invoke(agent_app, args)
    assert result.exit_code == 0, result.output
    configured = entry(mcp_admin.user_mcp_config())
    assert configured["read_only"] is read_only
    assert configured["tools_allowlist"] == []
    assert configured["allow_all_tools"] is True
    assert configured["url"] == (mcp_admin.GITHUB_READONLY_URL if read_only else mcp_admin.GITHUB_URL)


def test_cli_and_sdk_change_existing_scope_independently_without_removing_login(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "settings"))
    config = mcp_admin.register_remote_mcp(
        "github",
        bearer_token_env_var="TOKEN",
        tools=("issue_read",),
        workspace=tmp_path,
    )
    before = config.read_bytes()
    conflict = CliRunner().invoke(agent_app, ["mcp", "configure", "github", "--all-tools", "--tool", "issue_read"])
    assert conflict.exit_code != 0 and config.read_bytes() == before
    result = CliRunner().invoke(
        agent_app,
        [
            "mcp",
            "configure",
            "github",
            "--allow-actions",
            "--workspace",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert entry(config)["url"] == mcp_admin.GITHUB_URL
    assert entry(config)["tools_allowlist"] == ["issue_read"]
    assert entry(config)["bearer_token_env_var"] == "TOKEN"
    assert "X-MCP-Readonly" not in entry(config)["headers"]
    mcp_admin.configure_mcp_connection("github", tools=(), workspace=tmp_path)
    assert entry(config)["allow_all_tools"] is True
    assert entry(config)["read_only"] is False
    mcp_admin.configure_mcp_connection("github", read_only=True, workspace=tmp_path)
    assert entry(config)["allow_all_tools"] is True
    assert entry(config)["headers"]["X-MCP-Readonly"] == "true"
    listing = CliRunner().invoke(agent_app, ["mcp", "list"])
    assert json.loads(listing.output)[0]["allow_all_tools"] is True


def test_generic_connection_scope_does_not_rewrite_endpoint_or_access_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "settings"))
    config = mcp_admin.register_remote_mcp(
        "docs",
        url="https://docs.example/mcp",
        bearer_token_env_var="TOKEN",
        workspace=tmp_path,
    )
    monkeypatch.setattr(mcp_admin, "_read_keychain", lambda _: pytest.fail("must not read credentials"))
    mcp_admin.configure_mcp_connection("docs", read_only=False, tools=("create_document",), workspace=tmp_path)
    assert entry(config)["url"] == "https://docs.example/mcp"
    assert entry(config)["tools_allowlist"] == ["create_document"]
    assert entry(config)["allow_all_tools"] is False


def test_failed_legacy_endpoint_migration_restores_credential_and_keeps_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "settings"))
    config = mcp_admin.register_remote_mcp("github", bearer_token_env_var="TOKEN", workspace=tmp_path)
    configured = entry(config)
    configured.pop("bearer_token_env_var")
    configured["keychain"] = True
    config.write_text(yaml.safe_dump({"servers": [configured]}))
    original = config.read_bytes()
    credentials = {"github": json.dumps({"url": configured["url"], "token": "fake-test-token"})}
    saved = credentials["github"]
    monkeypatch.setattr(mcp_admin, "_read_keychain", lambda name: credentials[name])
    monkeypatch.setattr(mcp_admin, "_write_keychain", lambda name, value: credentials.update({name: value}))

    def fail_config(*_):
        raise OSError("disk error")

    monkeypatch.setattr(mcp_admin, "_atomic_write", fail_config)
    with pytest.raises(OSError, match="disk error"):
        mcp_admin.configure_mcp_connection("github", read_only=False, workspace=tmp_path)
    assert config.read_bytes() == original
    assert credentials["github"] == saved


def test_scope_update_cannot_override_current_network_deny(tmp_path, monkeypatch):
    root = tmp_path / "settings"
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(root))
    config = mcp_admin.register_remote_mcp("github", bearer_token_env_var="TOKEN", workspace=tmp_path)
    before = config.read_bytes()
    (root / "permissions.json").write_text(json.dumps({"network": {"denied_domains": ["api.githubcopilot.com"]}}))
    with pytest.raises(ValueError, match="permission policy"):
        mcp_admin.configure_mcp_connection("github", read_only=False, workspace=tmp_path)
    assert config.read_bytes() == before


def test_scope_update_preserves_other_server_catalogs_and_rolls_back_on_write_failure(tmp_path, monkeypatch):
    from agent_runtime.runtime.mcp_catalog import _read_catalog, write_catalog
    from agent_runtime.tools.integrations.mcp import MCPToolDescriptor

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "settings"))
    config = mcp_admin.register_remote_mcp(
        "github",
        bearer_token_env_var="TOKEN",
        tools=("issue_read",),
        workspace=tmp_path,
    )
    mcp_admin.register_remote_mcp(
        "docs",
        url="https://docs.example/mcp",
        bearer_token_env_var="TOKEN",
        workspace=tmp_path,
    )
    write_catalog(
        config,
        [
            MCPToolDescriptor(
                server_name=name,
                tool_name="read",
                description="Read fixture data",
                input_schema={"type": "object"},
                read_only_hint=True,
                endpoint_url=url,
            )
            for name, url in (("github", mcp_admin.GITHUB_READONLY_URL), ("docs", "https://docs.example/mcp"))
        ],
        server_names=frozenset({"github", "docs"}),
    )
    before_config = config.read_bytes()
    before_catalog = _read_catalog(config)
    original_write = mcp_admin._atomic_write

    def fail_config(path, content):
        if path == config:
            raise OSError("config write failed")
        original_write(path, content)

    monkeypatch.setattr(mcp_admin, "_atomic_write", fail_config)
    with pytest.raises(OSError, match="config write failed"):
        mcp_admin.configure_mcp_connection("github", tools=(), workspace=tmp_path)
    assert config.read_bytes() == before_config
    assert _read_catalog(config) == before_catalog
    monkeypatch.setattr(mcp_admin, "_atomic_write", original_write)
    mcp_admin.configure_mcp_connection("github", tools=(), workspace=tmp_path)
    assert _read_catalog(config) == {"docs": before_catalog["docs"]}


def tool(name, *, read_only=False, destructive=False):
    return Tool(
        name=name,
        inputSchema={"type": "object"},
        annotations=ToolAnnotations(
            readOnlyHint=read_only,
            destructiveHint=destructive,
        ),
    )


@pytest.mark.anyio
async def test_catalog_follows_pages_and_does_not_publish_duplicates_or_cursor_loops():
    client = SimpleNamespace(
        list_tools=AsyncMock(
            side_effect=[
                ListToolsResult(tools=[tool("read", read_only=True)], nextCursor="next"),
                ListToolsResult(tools=[tool("write")]),
            ]
        )
    )
    listed = await runtime._list_all_tools(client)
    assert [item.name for item in listed.tools] == ["read", "write"]
    assert client.list_tools.call_args_list[1].kwargs["params"].cursor == "next"
    for second in [ListToolsResult(tools=[], nextCursor="next"), ListToolsResult(tools=[tool("read")])]:
        client.list_tools = AsyncMock(
            side_effect=[
                ListToolsResult(tools=[tool("read", read_only=True)], nextCursor="next"),
                second,
            ]
        )
        with pytest.raises(ValueError, match="repeated|duplicate"):
            await runtime._list_all_tools(client)


@pytest.mark.anyio
async def test_catalog_limits_fail_instead_of_installing_a_partial_page(monkeypatch):
    monkeypatch.setattr(runtime, "_MAX_CATALOG_TOOLS", 1)
    client = SimpleNamespace(list_tools=AsyncMock(return_value=ListToolsResult(tools=[tool("a"), tool("b")])))
    with pytest.raises(ValueError, match="resource limit"):
        await runtime._list_all_tools(client)


@pytest.mark.anyio
async def test_catalog_page_and_byte_limits_are_explicit_failures(monkeypatch):
    client = SimpleNamespace(list_tools=AsyncMock(return_value=ListToolsResult(tools=[], nextCursor="more")))
    monkeypatch.setattr(runtime, "_MAX_CATALOG_PAGES", 1)
    with pytest.raises(ValueError, match="page limit"):
        await runtime._list_all_tools(client)
    monkeypatch.setattr(runtime, "_MAX_CATALOG_BYTES", 1)
    client.list_tools = AsyncMock(return_value=ListToolsResult(tools=[tool("read")]))
    with pytest.raises(ValueError, match="resource limit"):
        await runtime._list_all_tools(client)


@pytest.mark.anyio
@pytest.mark.parametrize("read_only", [True, False])
async def test_discovery_filters_capabilities_and_actions_still_require_approval(tmp_path, monkeypatch, read_only):
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "settings"))
    config = mcp_admin.register_remote_mcp(
        "docs",
        url="https://docs.example/mcp",
        bearer_token_env_var="TOKEN",
        read_only=read_only,
        workspace=tmp_path,
    )

    @asynccontextmanager
    async def connected(*_):
        client = SimpleNamespace(
            list_tools=AsyncMock(
                side_effect=[
                    ListToolsResult(tools=[tool("read", read_only=True)], nextCursor="more"),
                    ListToolsResult(tools=[tool("create"), tool("contradictory", read_only=True, destructive=True)]),
                ]
            )
        )
        listed = await runtime._list_all_tools(client)
        yield client, SimpleNamespace(serverInfo=SimpleNamespace(version="fixture")), listed, None

    monkeypatch.setattr(runtime, "_owned_mcp_connection", connected)
    permissions = PermissionPolicy(
        network=NetworkPolicy(rules=(NetworkRule(domain="docs.example", decision="allow"),))
    ).resolve(tmp_path)
    async with runtime.open_product_mcp_tools(config, permissions=permissions) as tools:
        assert [item.definition.name for item in tools] == (
            ["mcp__docs__read"] if read_only else ["mcp__docs__read", "mcp__docs__create", "mcp__docs__contradictory"]
        )
        if not read_only:
            for action in tools[1:]:
                assert action.approval_profile is ToolApprovalProfile.MCP_ACTION
                context = ToolExecutionContext(approval_mode="full", permissions=permissions)
                assert can_use_tool(action, {}, action.resolve_use({}), context).decision is UseToolDecision.ASK

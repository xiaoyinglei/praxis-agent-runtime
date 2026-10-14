from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import pytest
import yaml
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from agent_runtime.permission_policy import NetworkPolicy, NetworkRule, PermissionPolicy
from agent_runtime.runtime import mcp as runtime


@pytest.fixture
def anyio_backend():
    return "asyncio"


def config_file(tmp_path, **overrides):
    config = tmp_path / "remote.yaml"
    config.write_text(yaml.safe_dump({"servers": [{
        "name": "remote", "transport": "http", "url": "https://mcp.example.com/mcp",
        "enabled": True, "headers": {"Authorization": "Bearer ${TEST_MCP_TOKEN}"},
        "tools_allowlist": ["echo"], "read_only": True, **overrides,
    }]}))
    return config


def policy(tmp_path):
    return PermissionPolicy(network=NetworkPolicy(rules=(
        NetworkRule(domain="mcp.example.com", decision="allow"),
    ))).resolve(tmp_path)


@asynccontextmanager
async def echo_server(monkeypatch):
    server = FastMCP("remote-test", stateless_http=True, json_response=True,
                     transport_security=TransportSecuritySettings(allowed_hosts=["mcp.example.com"]))

    @server.tool(annotations={"readOnlyHint": True})
    def echo(message: str) -> str:
        return "echo: " + message

    app = server.streamable_http_app()
    requests = []

    class Transport(httpx.ASGITransport):
        async def handle_async_request(self, request):
            requests.append((request.method, str(request.url), request.headers.get("authorization")))
            return await super().handle_async_request(request)

    monkeypatch.setattr(runtime, "public_http_transport", lambda: Transport(app=app), raising=False)
    async with server.session_manager.run():
        yield requests


@pytest.mark.anyio
async def test_http_mcp_initializes_discovers_calls_and_closes(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    config = config_file(tmp_path)
    async with echo_server(monkeypatch) as requests:
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path)) as tools:
            assert len(tools) == 1
            assert tools[0].definition.name == "mcp__remote__echo"
            result = await tools[0].run({"message": "hello"})
            assert "echo: hello" in result.content[0].text
            resolved = tools[0].resolve_use({"message": "hello"})
            assert any(target.kind == "public_web" and target.value == "https://mcp.example.com/mcp"
                       for target in resolved.targets)
        assert len(requests) >= 4
        assert all(authorization == "Bearer test-secret" for _, _, authorization in requests)
        assert all(url == "https://mcp.example.com/mcp" for _, url, _ in requests)


@pytest.mark.anyio
@pytest.mark.parametrize("decision", ["ask", "deny"])
async def test_http_startup_requires_allow_before_any_io(tmp_path, monkeypatch, decision):
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    config = config_file(tmp_path)
    async with echo_server(monkeypatch) as requests:
        diagnostics = []
        current = PermissionPolicy(network=NetworkPolicy(decision=decision)).resolve(tmp_path)
        async with runtime.open_product_mcp_tools(config, permissions=current, diagnostics=diagnostics) as tools:
            assert tools == ()
        assert requests == []
        assert diagnostics


@pytest.mark.anyio
async def test_http_live_revoke_blocks_existing_tool_before_io(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    config = config_file(tmp_path)
    current = [policy(tmp_path)]
    async with echo_server(monkeypatch) as requests:
        async with runtime.open_product_mcp_tools(
            config, permissions=current[0], permission_provider=lambda: current[0],
        ) as tools:
            assert tools
            await tools[0].run({"message": "first"})
            count = len(requests)
            current[0] = PermissionPolicy(network=NetworkPolicy(decision="deny")).resolve(tmp_path)
            with pytest.raises(PermissionError):
                await tools[0].run({"message": "revoked"})
            assert len(requests) == count


@pytest.mark.parametrize("url", ["http://mcp.example.com/mcp", "https://127.0.0.1/mcp",
                                    "https://user:password@mcp.example.com/mcp", "https://mcp.example.com/mcp#frag"])
def test_http_config_rejects_insecure_or_credentialed_urls(tmp_path, monkeypatch, url):
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    with pytest.raises(ValueError):
        runtime._load_enabled_servers(config_file(tmp_path, url=url))


@pytest.mark.anyio
@pytest.mark.parametrize("status", [302, 401])
async def test_http_redirect_and_auth_failure_do_not_expose_credentials_or_follow(
    tmp_path, monkeypatch, caplog, status,
):
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret-value")
    config = config_file(tmp_path)
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(status, headers={"Location": "https://evil.example/mcp"},
                              json={"message": "unavailable"})

    monkeypatch.setattr(runtime, "public_http_transport", lambda: httpx.MockTransport(respond))
    diagnostics = []
    async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), diagnostics=diagnostics) as tools:
        assert tools == ()
    assert requests == ["https://mcp.example.com/mcp"]
    assert diagnostics
    assert "test-secret-value" not in caplog.text
    assert "test-secret-value" not in str(diagnostics)


@pytest.mark.anyio
async def test_exact_tool_deny_and_config_drift_block_before_request(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    config = config_file(tmp_path)
    async with echo_server(monkeypatch) as requests:
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path)) as tools:
            assert tools
            count = len(requests)
            config.write_text(config.read_text() + "\n# user changed configuration\n")
            with pytest.raises(PermissionError, match="config changed"):
                await tools[0].run({"message": "after config change"})
            assert len(requests) == count


def test_remote_actions_require_approval_even_with_named_allow(tmp_path):
    from agent_runtime.permission_policy import ToolRule
    from agent_runtime.tools.integrations.mcp import MCPToolDescriptor, create_mcp_tools
    from agent_runtime.tools.permissions import ToolExecutionContext, can_use_tool

    tool = create_mcp_tools((MCPToolDescriptor(
        server_name="remote", tool_name="write", description="Remote action",
        input_schema={"type": "object"}, endpoint_url="https://mcp.example.com/mcp", require_action_approval=True,
    ),), lambda *_: {})[0]
    allowed = PermissionPolicy(network=NetworkPolicy(decision="allow"),
                               tools=(ToolRule(pattern="mcp__remote__*", decision="allow"),)).resolve(tmp_path)
    result = can_use_tool(tool, {}, tool.resolve_use({}), ToolExecutionContext(permissions=allowed))
    assert result.decision.value == "ask"


@pytest.mark.anyio
async def test_public_agent_reuses_remote_connection_across_submissions(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from agent_runtime.mcp_admin import register_remote_mcp
    from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel

    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    register_remote_mcp("remote", url="https://mcp.example.com/mcp", bearer_token_env_var="TEST_MCP_TOKEN",
                        tools=("echo",), workspace=tmp_path)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db")
    monkeypatch.setattr(agent, "_model_factory", PublicHarnessModel)
    async with echo_server(monkeypatch) as requests:
        async with agent.session(require_workspace_change=False) as session:
            tool = session.tools["mcp__remote__echo"]
            before = await tool.run({"message": "before"})
            assert "before" in before.content[0].text
            first = await session.submit("first analysis")
            second = await session.submit("second analysis")
            after = await tool.run({"message": "after"})
            assert "after" in after.content[0].text
            assert first.status == second.status == "done"
            assert session.tools["mcp__remote__echo"] is tool
        assert len(requests) >= 5


@pytest.mark.anyio
async def test_host_header_cannot_change_authenticated_request_authority(tmp_path, monkeypatch):
    config = config_file(tmp_path, headers={"Host": "evil.example"})
    current = runtime._load_enabled_servers(config)[0]
    requests = []
    monkeypatch.setattr(runtime, "public_http_transport", lambda: httpx.MockTransport(
        lambda request: (requests.append(request) or httpx.Response(200)),
    ))
    transport = runtime._MCPHTTPTransport(current, config, policy(tmp_path), None)
    try:
        request = httpx.Request("POST", "https://mcp.example.com/mcp", headers={"Host": "evil.example"})
        with pytest.raises(PermissionError, match="authority"):
            await transport.handle_async_request(request)
        assert requests == []
    finally:
        await transport.aclose()


@pytest.mark.anyio
async def test_session_runtime_network_revoke_reaches_existing_http_connection(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from agent_runtime.mcp_admin import register_remote_mcp
    from agent_runtime.tools.tool import ToolEffect
    from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel

    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    register_remote_mcp("remote", url="https://mcp.example.com/mcp", bearer_token_env_var="TEST_MCP_TOKEN",
                        tools=("echo",), workspace=tmp_path)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db")
    monkeypatch.setattr(agent, "_model_factory", PublicHarnessModel)
    async with echo_server(monkeypatch) as requests:
        async with agent.session(require_workspace_change=False) as session:
            tool = session.tools["mcp__remote__echo"]
            count = len(requests)
            session.update_tool_policy(deny_effects=frozenset({ToolEffect.NETWORK}))
            with pytest.raises(PermissionError):
                await tool.run({"message": "must not be sent"})
            assert len(requests) == count


def test_network_effect_revoke_preserves_external_filesystem_grants(tmp_path):
    from agent_runtime.harness.session import Session
    from agent_runtime.permission_policy import PathRule
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolEffect

    session = Session()
    session.workspace_path = tmp_path / "project"
    external = tmp_path / "reference"
    granted = PermissionPolicy(filesystem=(PathRule(path=str(external), access="write"),),
                               network=NetworkPolicy(decision="allow")).resolve(session.workspace_path)
    session.tool_execution_context = ToolExecutionContext(permissions=granted,
        deny_effects=frozenset({ToolEffect.NETWORK}))
    effective = session._resource_permissions()
    assert effective.network_decision("https://example.com") == "deny"
    assert effective.path_access(external / "notes.txt", write=True)
    assert not effective.path_access(tmp_path / "unauthorized.txt")

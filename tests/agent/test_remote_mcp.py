from __future__ import annotations

import asyncio
import json
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
async def echo_server(monkeypatch, *, calls=None, methods=None):
    server = FastMCP("remote-test", stateless_http=True, json_response=True,
                     transport_security=TransportSecuritySettings(allowed_hosts=["mcp.example.com"]))

    @server.tool(annotations={"readOnlyHint": True})
    def echo(message: str) -> str:
        if calls is not None:
            calls.append(message)
        return "echo: " + message

    app = server.streamable_http_app()
    requests = []

    class Transport(httpx.ASGITransport):
        async def handle_async_request(self, request):
            requests.append((request.method, str(request.url), request.headers.get("authorization")))
            if methods is not None and request.method == "POST":
                methods.append(json.loads(request.content).get("method"))
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
async def test_lazy_remote_mcp_does_no_startup_io_and_reuses_connection_on_use(tmp_path, monkeypatch):
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "personal"))
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    config = config_file(tmp_path)
    async with echo_server(monkeypatch) as requests:
        # An explicit connection check records the schema, never a credential.
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), cache_catalog=True):
            pass
        requests.clear()
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), lazy=True) as tools:
            assert requests == []
            assert [tool.definition.name for tool in tools] == ["mcp__remote__echo"]
            result = await tools[0].run({"message": "first"})
            assert "echo: first" in result.content[0].text
            count = len(requests)
            await tools[0].run({"message": "second"})
            assert len(requests) == count + 1
        for path in (tmp_path / "personal" / "mcp_catalog").glob("*.json"):
            assert "test-secret" not in path.read_text()


@pytest.mark.anyio
async def test_uncached_remote_service_prompts_only_when_used(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "personal"))
    config = config_file(tmp_path)
    monkeypatch.delenv("TEST_MCP_TOKEN", raising=False)
    async with echo_server(monkeypatch) as requests:
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), lazy=True) as tools:
            assert requests == []
            assert len(tools) == 1
            result = tools[0].normalize_output(await tools[0].run({}))
            assert result.error_code == "mcp_connection_required"
            assert result.metadata["mcp_server"] == "remote"
            assert "/connections" in result.error_message
        assert requests == []
        assert not caplog.records


@pytest.mark.anyio
async def test_public_agent_ordinary_chat_never_loads_remote_credentials_or_opens_browser(tmp_path, monkeypatch):
    from agent_runtime import Agent, mcp_oauth
    from agent_runtime.mcp_admin import register_remote_mcp
    from agent_runtime.tools.executor import ToolExecutor
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
    from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel

    register_remote_mcp("remote", url="https://mcp.example.com/mcp", oauth=True, workspace=tmp_path,
                        tools=("echo",))

    def unexpected(*args, **kwargs):
        pytest.fail("Ordinary chat must not load MCP credentials, connect, or open a browser")

    monkeypatch.setattr(runtime, "public_http_transport", unexpected)
    monkeypatch.setattr(mcp_oauth, "create_oauth_provider", unexpected)
    monkeypatch.setattr("webbrowser.open", unexpected)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db")
    monkeypatch.setattr(agent, "_model_factory", PublicHarnessModel)
    async with agent.session(require_workspace_change=False) as session:
        result = await session.submit("Explain this local project")
        assert result.status == "done"
        tool = session.tools["mcp__remote__connect"]
        execution = await ToolExecutor({tool.definition.name: tool}).execute(
            ToolCall(tool_call_id="connection", tool_name=tool.definition.name, arguments={},
                     origin=ToolCallOrigin(request_id="request", toolset_revision="catalog-v1",
                                           exposed_tool_names=(tool.definition.name,))),
            context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path, permissions=session.permissions()),
        )
        assert execution.result.error_code == "mcp_connection_required"


@pytest.mark.anyio
@pytest.mark.parametrize("drift", ["credential", "schema", "safety"])
async def test_lazy_reconnect_revalidates_cached_approval_identity_before_dispatch(tmp_path, monkeypatch, drift):
    from agent_runtime.runtime.mcp_catalog import _catalog_path
    monkeypatch.setenv("TEST_MCP_TOKEN", "original-credential")
    config = config_file(tmp_path)
    calls = []
    async with echo_server(monkeypatch, calls=calls):
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), cache_catalog=True):
            pass
        if drift == "credential":
            monkeypatch.setenv("TEST_MCP_TOKEN", "updated-credential")
        else:
            path = _catalog_path(config)
            cached = json.loads(path.read_text())
            descriptor = cached["servers"]["remote"][0]
            if drift == "schema":
                descriptor["input_schema"]["properties"]["message"]["description"] = "changed schema"
            else:
                descriptor["idempotent_hint"] = not descriptor["idempotent_hint"]
            path.write_text(json.dumps(cached))
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), lazy=True) as tools:
            result = tools[0].normalize_output(await tools[0].run({"message": "must not dispatch"}))
            assert result.error_code == "mcp_connection_required"
            assert calls == []


@pytest.mark.anyio
async def test_concurrent_lazy_calls_initialize_one_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    config = config_file(tmp_path)
    methods = []
    async with echo_server(monkeypatch, methods=methods):
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), cache_catalog=True):
            pass
        methods.clear()
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), lazy=True) as tools:
            await asyncio.gather(tools[0].run({"message": "one"}), tools[0].run({"message": "two"}))
        assert methods.count("initialize") == 1
        assert methods.count("tools/call") == 2


@pytest.mark.anyio
async def test_lazy_failure_prompts_once_without_repeated_network_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    config = config_file(tmp_path)
    async with echo_server(monkeypatch):
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), cache_catalog=True):
            pass
    requests = []
    monkeypatch.setattr(runtime, "public_http_transport", lambda: httpx.MockTransport(
        lambda request: (requests.append(request) or httpx.Response(401)),
    ))
    async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), lazy=True) as tools:
        assert requests == []
        for _ in range(2):
            result = tools[0].normalize_output(await tools[0].run({"message": "retry"}))
            assert result.error_code == "mcp_connection_required"
        assert len(requests) == 1


@pytest.mark.anyio
async def test_cancelled_lazy_connect_closes_transport_and_owner_task(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    config = config_file(tmp_path)
    async with echo_server(monkeypatch):
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), cache_catalog=True):
            pass
    started = asyncio.Event()
    closed = []

    async def blocked(request):
        started.set()
        await asyncio.Event().wait()

    class Transport(httpx.MockTransport):
        async def aclose(self):
            closed.append(True)
            await super().aclose()

    monkeypatch.setattr(runtime, "public_http_transport", lambda: Transport(blocked))
    async with runtime.open_product_mcp_tools(
        config, permissions=policy(tmp_path), lazy=True, close_timeout_seconds=0.05,
    ) as tools:
        call = asyncio.create_task(tools[0].run({"message": "cancel"}))
        await asyncio.wait_for(started.wait(), 3)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 3)
    assert closed == [True]
    assert not any(task.get_name() == "mcp-http:remote" and not task.done() for task in asyncio.all_tasks())


@pytest.mark.anyio
async def test_explicit_host_check_populates_workspace_config_catalog_without_granting_network(tmp_path, monkeypatch):
    from agent_runtime.mcp_admin import register_remote_mcp
    from agent_runtime.mcp_cli import check_mcp_connection
    monkeypatch.setenv("TEST_MCP_TOKEN", "test-secret")
    config = config_file(tmp_path)
    async with echo_server(monkeypatch) as requests:
        denied = await check_mcp_connection("remote", tmp_path, config_path=config)
        assert not denied["connected"]
        assert requests == []
        # Separate host registration grants the domain; --config itself does not.
        register_remote_mcp("remote", url="https://mcp.example.com/mcp", workspace=tmp_path,
                            bearer_token_env_var="TEST_MCP_TOKEN", tools=("echo",))
        assert (await check_mcp_connection("remote", tmp_path, config_path=config))["connected"]
        requests.clear()
        async with runtime.open_product_mcp_tools(config, permissions=policy(tmp_path), lazy=True) as tools:
            assert requests == []
            assert tools[0].definition.name == "mcp__remote__echo"


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
        from agent_runtime.mcp_cli import check_mcp_connection
        assert (await check_mcp_connection("remote", tmp_path))["connected"]
        requests.clear()
        async with agent.session(require_workspace_change=False) as session:
            assert requests == []
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
        from agent_runtime.mcp_cli import check_mcp_connection
        assert (await check_mcp_connection("remote", tmp_path))["connected"]
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

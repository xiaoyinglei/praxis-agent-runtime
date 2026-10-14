import httpx
import pytest

from agent_runtime.tools.web_http import PublicWebClient, PublicWebError


@pytest.mark.anyio
async def test_redirect_destination_checked_before_network_io():
    seen = []

    def transport(request):
        seen.append(request.url.host)
        return httpx.Response(302, headers={"location": "https://denied.example/secret"})

    def authorize(url):
        if httpx.URL(url).host != "allowed.example":
            raise PublicWebError("network_policy_denied", "destination denied")

    client = PublicWebClient(transport=httpx.MockTransport(transport), authorize_url=authorize)
    try:
        with pytest.raises(PublicWebError, match="destination denied"):
            await client.get("https://allowed.example/")
    finally:
        await client.aclose()
    assert seen == ["allowed.example"]


@pytest.mark.parametrize(
    "network,approval,expected",
    [
        ("allow", "on-request", "allow"),
        ("allow", "never", "allow"),
        ("ask", "on-request", "ask"),
        ("ask", "never", "deny"),
        ("deny", "on-request", "deny"),
        ("deny", "never", "deny"),
    ],
)
def test_network_and_approval_modes_compose(tmp_path, network, approval, expected):
    from agent_runtime.permission_policy import NetworkPolicy, PermissionPolicy
    from agent_runtime.tools.permissions import ToolExecutionContext, can_use_tool
    from agent_runtime.tools.tool import ResolvedToolUse, ToolEffect, ToolTarget
    from tests.agent.test_single_tool_executor import _tool

    resolved = ResolvedToolUse(frozenset({ToolEffect.NETWORK}), (ToolTarget("public_web", "https://example.com"),))
    tool = _tool(static_effects=resolved.effects, resolve_use=lambda _args: resolved)
    context = ToolExecutionContext(
        permissions=PermissionPolicy(
            approval=approval,
            network=NetworkPolicy(decision=network),
        ).resolve(tmp_path)
    )
    assert can_use_tool(tool, {}, resolved, context).decision.value == expected


@pytest.mark.anyio
async def test_preapproved_process_network_allow_does_not_prompt(tmp_path):
    from agent_runtime.permission_policy import NetworkPolicy, PermissionPolicy
    from agent_runtime.tools.executor import ToolExecutor
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ResolvedToolUse, ToolEffect, ToolTarget
    from tests.agent.test_single_tool_executor import _call, _tool

    resolved = ResolvedToolUse(
        frozenset({ToolEffect.NETWORK, ToolEffect.EXECUTE_PROCESS}), (ToolTarget("cwd_path", str(tmp_path)),)
    )
    tool = _tool(static_effects=resolved.effects, resolve_use=lambda _args: resolved)
    context = ToolExecutionContext(
        workspace_root=tmp_path,
        cwd=tmp_path,
        allow_execute_tools=True,
        permissions=PermissionPolicy(approval="never", network=NetworkPolicy(decision="allow")).resolve(tmp_path),
    )
    execution = await ToolExecutor({"demo": tool}).execute(_call(), context=context)
    assert not execution.result.is_error

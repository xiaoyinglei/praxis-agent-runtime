from pathlib import Path

from agent_runtime.permission_policy import NetworkPolicy, PermissionPolicy


def test_named_destination_allow_preserves_ask_for_other_hosts(tmp_path: Path):
    policy = PermissionPolicy(network=NetworkPolicy.model_validate({
        "rules": [{"domain": "api.githubcopilot.com", "decision": "allow"}],
    })).resolve(tmp_path)
    assert policy.network_decision("https://api.githubcopilot.com/mcp/readonly") == "allow"
    assert policy.network_decision("https://example.com") == "ask"
    assert policy.network_decision("https://api.githubcopilot.com.evil.example") == "ask"
    assert policy.command_network_allowed() == "deny"


def test_destination_rules_do_not_override_denials_or_managed_ceiling(tmp_path: Path):
    policy = PermissionPolicy(network=NetworkPolicy.model_validate({
        "rules": [{"domain": "*.example.com", "decision": "allow"},
                  {"domain": "secret.example.com", "decision": "deny"}],
    })).resolve(tmp_path)
    assert policy.network_decision("https://public.example.com") == "allow"
    assert policy.network_decision("https://secret.example.com") == "deny"
    ceiling = PermissionPolicy(network=NetworkPolicy(decision="deny")).resolve(tmp_path)
    assert policy.intersection(ceiling).network_decision("https://public.example.com") == "deny"
    filtered = PermissionPolicy(network=NetworkPolicy.model_validate({
        "allowed_domains": ["allowed.example.com"],
        "rules": [{"domain": "other.example.com", "decision": "allow"}],
    })).resolve(tmp_path)
    assert filtered.network_decision("https://other.example.com") == "deny"


def test_destination_allow_cannot_override_runtime_network_effect_deny(tmp_path):
    from agent_runtime.tools.integrations.mcp import MCPToolDescriptor, create_mcp_tools
    from agent_runtime.tools.permissions import ToolExecutionContext, can_use_tool
    from agent_runtime.tools.tool import ToolEffect

    tool = create_mcp_tools((MCPToolDescriptor(
        server_name="remote", tool_name="read", description="Read remotely", input_schema={"type": "object"},
        endpoint_url="https://mcp.example.com/mcp", read_only_hint=True,
    ),), lambda *_: {})[0]
    policy = PermissionPolicy(network=NetworkPolicy(decision="allow")).resolve(tmp_path)
    decision = can_use_tool(tool, {}, tool.resolve_use({}), ToolExecutionContext(
        permissions=policy, deny_effects=frozenset({ToolEffect.NETWORK}),
    ))
    assert decision.decision.value == "deny"


def test_readonly_ceiling_rejects_remote_actions_before_approval(tmp_path):
    from agent_runtime.tools.integrations.mcp import MCPToolDescriptor, create_mcp_tools
    from agent_runtime.tools.permissions import ToolExecutionContext, can_use_tool

    tool = create_mcp_tools((MCPToolDescriptor(
        server_name="remote", tool_name="write", description="Remote action", input_schema={"type": "object"},
        endpoint_url="https://mcp.example.com/mcp", require_action_approval=True,
    ),), lambda *_: {})[0]
    policy = PermissionPolicy(profile="read-only", network=NetworkPolicy(decision="allow")).resolve(tmp_path)
    result = can_use_tool(tool, {}, tool.resolve_use({}), ToolExecutionContext(permissions=policy))
    assert result.decision.value == "deny"

"""Praxis MCP 适配器:名字唯一拼装 + 配置加载/信任门(T15)+ stdio 生命周期(T16)
+ descriptor→Tool 暴露/目录上限/输出归一(T17)+ HTTP 传输/bearer(T18)。

后续任务在此包内扩展:manager/CLI 接线(T19)。
"""

from praxis.adapters.mcp.catalog import list_all_tools
from praxis.adapters.mcp.config import (
    HttpServerConfig,
    McpConfigTrustDecision,
    McpServerConfig,
    StdioServerConfig,
    config_sha256,
    decide_mcp_config_trust,
    load_mcp_configs,
    resolve_mcp_config_path,
)
from praxis.adapters.mcp.descriptors import McpToolDescriptor, mcp_tools_for_connection
from praxis.adapters.mcp.http_transport import (
    GuardedMcpTransport,
    McpTransportError,
    credential_revision,
    owned_http_connection,
)
from praxis.adapters.mcp.lifecycle import McpConnection, owned_connection, owned_stdio_connection
from praxis.adapters.mcp.names import (
    canonical_tool_name,
    global_wildcard,
    normalize_mcp_name,
    server_wildcard,
)
from praxis.adapters.mcp.output import normalize_mcp_output

__all__ = [
    "GuardedMcpTransport",
    "HttpServerConfig",
    "McpConfigTrustDecision",
    "McpConnection",
    "McpServerConfig",
    "McpToolDescriptor",
    "McpTransportError",
    "StdioServerConfig",
    "canonical_tool_name",
    "config_sha256",
    "credential_revision",
    "decide_mcp_config_trust",
    "global_wildcard",
    "list_all_tools",
    "load_mcp_configs",
    "mcp_tools_for_connection",
    "normalize_mcp_name",
    "normalize_mcp_output",
    "owned_connection",
    "owned_http_connection",
    "owned_stdio_connection",
    "resolve_mcp_config_path",
    "server_wildcard",
]

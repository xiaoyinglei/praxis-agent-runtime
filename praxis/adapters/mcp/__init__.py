"""Praxis MCP 适配器:名字唯一拼装 + 配置加载/信任门(T15)+ stdio 生命周期(T16)。

后续任务在此包内扩展:descriptor→Tool(T17)、HTTP 传输(T18)、manager/CLI
接线(T19)。
"""

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
from praxis.adapters.mcp.lifecycle import McpConnection, owned_stdio_connection
from praxis.adapters.mcp.names import (
    canonical_tool_name,
    global_wildcard,
    normalize_mcp_name,
    server_wildcard,
)

__all__ = [
    "HttpServerConfig",
    "McpConfigTrustDecision",
    "McpConnection",
    "McpServerConfig",
    "StdioServerConfig",
    "canonical_tool_name",
    "config_sha256",
    "decide_mcp_config_trust",
    "global_wildcard",
    "load_mcp_configs",
    "normalize_mcp_name",
    "owned_stdio_connection",
    "resolve_mcp_config_path",
    "server_wildcard",
]

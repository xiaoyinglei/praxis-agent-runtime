"""Praxis MCP 适配器:名字唯一拼装 + ``mcp_servers.yaml`` 加载/信任门/漂移(T15)。

后续任务在此包内扩展:stdio 生命周期(T16)、descriptor→Tool(T17)、HTTP
传输(T18)、manager/CLI 接线(T19)。
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
from praxis.adapters.mcp.names import (
    canonical_tool_name,
    normalize_mcp_name,
    server_wildcard,
)

__all__ = [
    "HttpServerConfig",
    "McpConfigTrustDecision",
    "McpServerConfig",
    "StdioServerConfig",
    "canonical_tool_name",
    "config_sha256",
    "decide_mcp_config_trust",
    "load_mcp_configs",
    "normalize_mcp_name",
    "resolve_mcp_config_path",
    "server_wildcard",
]
"""CLI connection setup; configuration and credentials belong to the human host."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from agent_runtime import mcp_admin
from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic

mcp_app = typer.Typer(no_args_is_help=True)


async def check_mcp_connection(name: str, workspace: Path) -> dict[str, object]:
    """Check one personal connection without invoking a model or MCP tool."""
    from agent_runtime.permission_policy import resolve_permission_sources
    from agent_runtime.runtime.mcp import decide_mcp_config_trust, open_trusted_product_mcp_tools

    config = mcp_admin.user_mcp_config()
    policy = resolve_permission_sources(workspace)
    trust = decide_mcp_config_trust(config, workspace_root=workspace, trust_workspace=False)
    diagnostics: list[RuntimeDiagnostic] = []
    async with open_trusted_product_mcp_tools(
        config, trust=trust, permissions=policy, diagnostics=diagnostics, server_names=frozenset({name}),
    ) as tools:
        return {"name": name, "connected": bool(tools), "tools": [tool.definition.name for tool in tools],
                "diagnostics": [item.model_dump(mode="json") for item in diagnostics]}


@mcp_app.command("add")
def add(
    name: Annotated[str, typer.Argument(help="服务名称；github 使用官方只读预设")],
    url: Annotated[str | None, typer.Option("--url", help="远程 HTTPS MCP 地址")] = None,
    bearer_token_env_var: Annotated[str | None, typer.Option("--bearer-token-env-var")] = None,
    tools: Annotated[list[str] | None, typer.Option("--tool", help="可重复指定工具白名单")] = None,
    read_only: Annotated[bool, typer.Option("--read-only/--allow-actions")] = True,
    oauth: Annotated[bool, typer.Option("--oauth", help="使用标准浏览器 OAuth 登录")] = False,
    oauth_client_id: Annotated[str | None, typer.Option("--oauth-client-id")] = None,
    oauth_callback_port: Annotated[int, typer.Option("--oauth-callback-port", min=0, max=65535)] = 0,
    oauth_scopes: Annotated[list[str] | None, typer.Option("--oauth-scope")] = None,
    oauth_domains: Annotated[
        list[str] | None, typer.Option("--oauth-domain", help="授权 OAuth 使用的额外精确域名"),
    ] = None,
) -> None:
    """注册远程服务及其域名授权；github 默认使用现有 gh 登录。"""
    try:
        config = mcp_admin.register_remote_mcp(name, url=url, bearer_token_env_var=bearer_token_env_var,
                                               tools=tuple(tools or ()), read_only=read_only, oauth=oauth,
                                               oauth_client_id=oauth_client_id, oauth_scopes=tuple(oauth_scopes or ()),
                                               oauth_allowed_domains=tuple(oauth_domains or ()),
                                               oauth_callback_port=oauth_callback_port)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None
    print(json.dumps({"registered": name, "config": str(config), "read_only": read_only}, ensure_ascii=False))
    print(f"连接检查：agent mcp check {name}")


@mcp_app.command("list")
def list_connections() -> None:
    """显示个人连接；不读取凭据、不连接远程服务。"""
    keys = ("name", "transport", "url", "enabled", "read_only", "bearer_token_env_var", "keychain",
            "oauth", "oauth_client_id", "oauth_scopes", "oauth_allowed_domains", "oauth_callback_port")
    print(json.dumps([{key: entry[key] for key in keys if key in entry}
                      for entry in mcp_admin.mcp_connections()], ensure_ascii=False, indent=2))


@mcp_app.command("login")
def login(
    name: Annotated[str, typer.Argument()],
    from_gh: Annotated[bool, typer.Option("--from-gh")] = False,
    callback_port: Annotated[int | None, typer.Option("--callback-port", min=0, max=65535)] = None,
) -> None:
    """打开 OAuth 浏览器登录，或隐藏输入并更新本机令牌。"""
    try:
        entries = [entry for entry in mcp_admin.mcp_connections() if entry["name"] == name]
        if entries and entries[0].get("oauth"):
            if from_gh:
                raise ValueError("OAuth connections cannot use --from-gh")
            from agent_runtime.mcp_oauth import login_mcp_oauth
            asyncio.run(login_mcp_oauth(name, callback_port=callback_port))
            print(f"已完成 {name} OAuth 登录；后续会话自动刷新凭据。")
            return
        with mcp_admin.configuration_lock():
            entries = [entry for entry in mcp_admin.mcp_connections() if entry["name"] == name]
            if not entries or entries[0].get("transport") != "http" or not entries[0].get("keychain"):
                raise ValueError("服务须已注册并使用本机令牌凭据")
            if from_gh and name != "github":
                raise ValueError("--from-gh is available for the GitHub connection")
            if from_gh:
                mcp_admin.authorize_github_cli_endpoint(entries[0]["url"])
            token = mcp_admin.github_cli_token() if from_gh else typer.prompt("Token", hide_input=True)
            mcp_admin.save_mcp_credential(name, entries[0]["url"], token)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None
    print(f"已更新 {name} 本机凭据；请重新打开 Agent 会话。")


@mcp_app.command("check")
def check(
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
) -> None:
    """验证认证和工具发现；不调用模型或修改远程数据。"""
    try:
        result = asyncio.run(check_mcp_connection(name, workspace))
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["connected"]:
        raise typer.Exit(1)


@mcp_app.command("remove")
def remove(name: Annotated[str, typer.Argument()]) -> None:
    """移除个人连接及其本机凭据；保留可能共享的网络域名授权。"""
    try:
        mcp_admin.remove_mcp_connection(name)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None
    print(f"已移除 {name} 连接及本机凭据。")

"""Configure and inspect the same web backend used by ordinary Agent tasks."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from contextlib import aclosing
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import typer

from agent_runtime.model_credentials import resolve_model_credential
from agent_runtime.models import ModelSpec
from agent_runtime.user_config import user_config_directory
from agent_runtime.web_config import (
    WebConfiguration,
    install_search_key,
    load_search_key,
    resolve_web_configuration,
    save_web_configuration,
)

web_app = typer.Typer(add_completion=False, no_args_is_help=True)


def search_setup_key_file(*, workspace: Path, provider: str, key_file: Path | None) -> Path | None:
    candidate = key_file if key_file is not None else user_config_directory() / f"{provider}.key"
    if key_file is not None or candidate.exists():
        load_search_key(candidate, workspace=workspace)
        return candidate
    return None


def reusable_model_search_key(provider: str, model: ModelSpec | None) -> str | None:
    if (provider == "deepseek" and model is not None
            and (model.base_url or "").rstrip("/") in {
                "https://api.deepseek.com", "https://api.deepseek.com/v1",
            }):
        return resolve_model_credential(model.api_key_env, model.base_url)
    return None


def configure_search(
    *, workspace: Path, provider: str = "deepseek", key_file: Path | None = None,
    model: ModelSpec | None = None,
) -> WebConfiguration:
    """Human-only prompt shared by first-run setup, chat settings and the CLI."""
    key_file = search_setup_key_file(workspace=workspace, provider=provider, key_file=key_file)
    if key_file is None:
        key = None
        available = reusable_model_search_key(provider, model)
        if available is not None and typer.confirm("复用当前 DeepSeek 模型账号的 Key？", default=True):
            key = available
        if key is None:
            key = typer.prompt(f"{provider} API Key", hide_input=True)
        key_file = install_search_key(provider, key, workspace=workspace)
    return save_web_configuration(provider, key_file, workspace=workspace)


def onboard_search(*, workspace: Path, model: ModelSpec) -> WebConfiguration:
    """Offer discovery once; preserve configured services and explicit skips."""
    current = resolve_web_configuration()
    if current.source != "default":
        label = current.provider or "未启用（聊天中可选择设置搜索）"
        typer.echo(f"联网搜索：{label}")
        return current
    if not typer.confirm("启用联网搜索（DeepSeek，使用独立工具 API，可能产生费用）？", default=True):
        configuration = save_web_configuration(None, None, workspace=workspace)
        typer.echo("已跳过联网搜索；聊天界面提供设置入口，普通网页读取仍可使用。")
        return configuration
    configuration = configure_search(workspace=workspace, model=model)
    typer.echo("联网搜索已配置：DeepSeek；凭据不会进入聊天记录。")
    return configuration


@web_app.command("configure")
def configure(
    provider: Annotated[Literal["deepseek", "brave", "tavily"], typer.Option("--provider")] = "deepseek",
    key_file: Annotated[Path | None, typer.Option("--key-file", help="工作区外的受保护密钥文件")] = None,
) -> None:
    """保存一次，后续 CLI 与 Python SDK 自动使用；不会写入聊天历史。"""
    try:
        configuration = configure_search(workspace=Path.cwd(), provider=provider, key_file=key_file)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None
    typer.echo(f"已保存联网配置：{configuration.provider}；获取方式：{configuration.acquisition}")
    typer.echo("新启动的 agent chat/run 和 Agent() 会自动加载；agent web check 可检查配置。")


@web_app.command("status")
def status(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """显示有效配置来源及获取方式，不输出密钥内容。"""
    try:
        configuration = resolve_web_configuration()
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None
    data = configuration.public_status()
    if json_output:
        typer.echo(json.dumps(data, ensure_ascii=False))
    else:
        typer.echo(f"搜索：{configuration.provider or '未配置（agent web configure）'}")
        acquisition = 'DeepSeek 托管 Fetch' if configuration.acquisition == 'provider' else '本机直接读取'
        typer.echo(f"默认网页获取：{acquisition}")
        typer.echo(f"配置来源：{configuration.source}")
        if configuration.key_file:
            typer.echo(f"凭据文件：{configuration.key_file}")


async def _check_live(configuration: WebConfiguration, key: str) -> None:
    from agent_runtime.tools.builtins.web import create_web_tools
    from agent_runtime.tools.web_http import PublicWebClient, web_proxy_configuration

    sources: dict[str, bytes] = {}

    def save(content: bytes) -> str:
        identity = f"artifact_{uuid4().hex}"
        sources[identity] = content
        return identity

    proxy, no_proxy = web_proxy_configuration(None, dict(os.environ))
    async with aclosing(PublicWebClient(proxy_url=proxy, no_proxy=no_proxy)) as client:
        tools = {tool.definition.name: tool for tool in create_web_tools(
            client, save_source=save, load_source=lambda identity: sources[identity],
            search_key=key, search_provider=configuration.provider,
        )}
        for name, arguments in (
            ("web_search", {"query": "Python documentation"}),
            ("web_fetch", {"url": "https://docs.python.org/3/"}),
        ):
            result = tools[name].run(arguments)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                raise ValueError(f"{name}: unexpected diagnostic result")
            if result.get("error_code"):
                raise ValueError(f"{name}: {result['error_code']}: {result.get('error_message', '')}")
            typer.echo(f"{name}：已连接")


@web_app.command("check")
def check(
    live: Annotated[bool, typer.Option("--live", help="实际调用一次搜索和一次读取，可能产生 API 费用")] = False,
) -> None:
    """默认只检查本地配置；--live 才访问服务，不调用聊天模型。"""
    try:
        configuration = resolve_web_configuration()
        if configuration.provider is None:
            raise ValueError("搜索未配置；运行 agent web configure --provider deepseek")
        key = load_search_key(configuration.key_file, workspace=Path.cwd())
        assert key is not None
        if live:
            asyncio.run(_check_live(configuration, key))
        else:
            typer.echo(f"本地配置有效：{configuration.provider}；未发送网络请求（--live 可检查连接）。")
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None

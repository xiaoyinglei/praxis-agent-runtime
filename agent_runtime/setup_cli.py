"""First-run terminal setup using the existing model and MCP control planes."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import click
import typer

from agent_runtime import mcp_admin
from agent_runtime.core.llm_config import ModelProvider, normalize_model_endpoint, validate_api_key_env_name
from agent_runtime.core.runtime_diagnostics import redact_sensitive_text
from agent_runtime.mcp_cli import check_mcp_connection
from agent_runtime.model_admin import ModelAdminService, ModelDefinitionArguments, openai_model_arguments
from agent_runtime.model_config_io import validate_user_config_path
from agent_runtime.model_credentials import credential_directory, resolve_model_credential, save_model_credential
from agent_runtime.model_probe import ProbeLevel
from agent_runtime.terminal_render import safe_terminal_text
from agent_runtime.web_cli import onboard_search
from agent_runtime.workspace import DEFAULT_MODEL_SESSION_PATH


@contextmanager
def _probe_key(name: str | None, key: str | None) -> Iterator[None]:
    """Expose entered credentials only during the synchronous CLI probe phase."""
    previous = None if name is None else os.environ.get(name)
    if name is not None and key is not None:
        os.environ[name] = key
    try:
        yield
    finally:
        if name is not None and key is not None:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _configure_model(service: ModelAdminService, model_id: str | None, *, simple: bool = False) -> str:
    typer.echo("配置 OpenAI 兼容模型" if simple else "1/3 选择模型")
    if simple and service.whole_catalog_override_active:
        source = safe_terminal_text(service.source_path("override"))
        typer.echo(f"正在使用整库覆盖（{source}）。个人模型注册表未加载。")
    entries = service.list_models()
    adding = False
    if model_id is None and len(entries) == 1 and not simple:
        model_id = entries[0].model_id
        typer.echo("默认模型：" + safe_terminal_text(model_id))
    elif model_id is None:
        for entry in entries:
            typer.echo("  " + safe_terminal_text(f"{entry.model_id} · {entry.spec.provider}"))
        model_id = typer.prompt("模型 ID（输入 new 添加模型）", default=service.current().id)
        if model_id == "new":
            adding = True
            model_id = typer.prompt("服务商提供的实际模型 ID")
    known = not adding and model_id in {entry.model_id for entry in entries}
    arguments: ModelDefinitionArguments | None = None
    if known:
        definition = service.show(model_id).definition
        endpoint = normalize_model_endpoint(
            provider=definition.provider, base_url=definition.base_url, location=definition.location
        ).base_url
        name = definition.api_key_env
    else:
        if service.whole_catalog_override_active:
            raise ValueError("当前使用整库覆盖，请取消 RAG_AGENT_MODELS_PATH / RAG_AGENT_MODELS 后再添加个人模型")
        typer.echo("添加兼容 OpenAI 接口的模型；本地模型服务需先启动。")
        endpoint = typer.prompt("Base URL（例如 https://provider.example/v1）").strip().rstrip("/")
        normalized = normalize_model_endpoint(
            provider=ModelProvider.OPENAI_COMPATIBLE, base_url=endpoint, location=None
        )
        endpoint = normalized.base_url
        remote_model_id = None
        connection = urlsplit(endpoint).netloc
        if simple:
            if any(entry.model_id == model_id for entry in entries):
                connection = typer.prompt("连接名称", default=connection).strip()
                if not connection or any(char.isspace() for char in connection):
                    raise ValueError("连接名称不能为空或包含空格")
                remote_model_id, model_id = model_id, f"{model_id}@{connection}"
                if any(entry.model_id == model_id for entry in entries):
                    raise ValueError(f"连接已存在，请选择 {model_id} 或使用其他连接名称")
            arguments = openai_model_arguments(
                base_url=endpoint, connection_name=connection, remote_model_id=remote_model_id,
            )
            name = arguments.api_key_env
        else:
            window = typer.prompt("上下文窗口 tokens（按服务说明填写）", type=click.IntRange(min=1))
            name = typer.prompt("API Key 环境变量名（无认证服务输入 none）", default="PRAXIS_MODEL_API_KEY")
            name = None if name == "none" else name
            validate_api_key_env_name(name)
            arguments = ModelDefinitionArguments(
                provider=ModelProvider.OPENAI_COMPATIBLE,
                base_url=endpoint,
                api_key_env=name,
                context_window_tokens=window,
                location=normalized.location,
            )
        arguments.for_add()  # Validate before requesting a secret.

    typer.echo("验证并保存模型连接（会发送测试消息）" if simple else "2/3 验证并保存模型连接（会产生少量模型请求）")
    key = None
    if name is not None:
        existing = resolve_model_credential(name, endpoint)
        if existing is None or not typer.confirm("已有凭据，是否复用？", default=True):
            validate_user_config_path(credential_directory(), workspace=service.workspace, worktree=service.worktree)
            if simple and arguments is not None and arguments.location == "local":
                key = typer.prompt("API Key（本地无认证可留空）", hide_input=True, default="", show_default=False)
                if not key:
                    arguments = replace(arguments, api_key_env=None)
                    name, key = None, None
            else:
                key = typer.prompt("API Key", hide_input=True)
            if key is not None and (not key.strip() or any(ord(char) < 32 for char in key)):
                raise ValueError("API Key 不能为空或包含控制字符")
        else:
            typer.echo("复用已有凭据。")
    try:
        level = ProbeLevel.CHAT if simple else ProbeLevel.FULL
        if simple and arguments is not None:
            asyncio.run(service.add(
                model_id, arguments=arguments, from_path=None, skip_probe=False,
                probe_level=level, make_default=True, api_key_to_save=key,
            ))
            if key is not None:
                typer.echo("凭据已保存。")
                key = None
        else:
            with _probe_key(name, key):
                if known:
                    asyncio.run(service.probe(model_id, level=level))
                else:
                    assert arguments is not None
                    asyncio.run(service.add(model_id, arguments=arguments, from_path=None, skip_probe=False))
        if name is not None and key is not None:
            save_model_credential(name, endpoint, key)
            location = "macOS Keychain" if sys.platform == "darwin" else "工作区外的私有凭据文件"
            typer.echo("凭据已保存到" + location + "。")
            if os.environ.get(name):
                typer.echo(f"当前终端的环境变量仍优先；使用新保存的 Key 前请执行：unset {name}")
    except Exception as error:
        # Entered keys are no longer in os.environ after the probe, so redact
        # their exact value as well as the existing environment-based redactor.
        message = redact_sensitive_text(str(error))
        if key:
            message = message.replace(key, "[redacted]")
        raise ValueError(message) from None
    refreshed = ModelAdminService.from_env(workspace=service.workspace, session_path=service.session_path)
    if known or not simple:
        refreshed.set_default(model_id)
    refreshed.switch(model_id)
    typer.echo("模型验证通过，已选择：" + safe_terminal_text(model_id))
    return model_id


def configure_model(model_id: str | None = None) -> None:
    try:
        service = ModelAdminService.from_env(session_path=DEFAULT_MODEL_SESSION_PATH)
        _configure_model(service, model_id, simple=True)
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo("模型配置未完成：" + safe_terminal_text(redact_sensitive_text(error)), err=True)
        raise typer.Exit(1) from None


def setup(
    model: Annotated[str | None, typer.Option("--model", help="直接选择模型 ID；未知 ID 进入添加流程")] = None,
    session_path: Annotated[
        Path, typer.Option("--session-path", help="当前项目的模型选择文件")
    ] = DEFAULT_MODEL_SESSION_PATH,
    no_mcp: Annotated[bool, typer.Option("--no-mcp", help="跳过可选的 GitHub MCP 配置")] = False,
    no_web: Annotated[bool, typer.Option("--no-web", help="跳过可选的联网搜索配置，不更改已有设置")] = False,
) -> None:
    """首次配置：模型连接、联网搜索，再按需连接 GitHub。"""
    try:
        service = ModelAdminService.from_env(session_path=session_path)
        _configure_model(service, model)
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo("模型配置未完成：" + safe_terminal_text(redact_sensitive_text(error)), err=True)
        raise typer.Exit(1) from None
    typer.echo("3/3 可选服务")
    if not no_web:
        try:
            refreshed = ModelAdminService.from_env(workspace=service.workspace, session_path=service.session_path)
            onboard_search(workspace=service.workspace, model=refreshed.current())
        except (OSError, ValueError, RuntimeError) as error:
            typer.echo("模型已配置；联网搜索未完成：" + safe_terminal_text(redact_sensitive_text(error)), err=True)
            raise typer.Exit(1) from None
    if not no_mcp and typer.confirm("连接只读 GitHub MCP？", default=False):
        try:
            connections = mcp_admin.mcp_connections()
            if not any(entry["name"] == "github" for entry in connections):
                typer.echo("使用本机 GitHub CLI 登录；未登录时先运行 gh auth login --hostname github.com。")
                mcp_admin.register_remote_mcp("github", workspace=service.workspace)
            result = asyncio.run(check_mcp_connection("github", service.workspace))
            if not result["connected"]:
                raise ValueError("连接检查未发现可用工具；请运行 agent mcp check github 查看详情")
            typer.echo("GitHub 连接检查通过。")
        except (OSError, ValueError, RuntimeError) as error:
            typer.echo("模型已配置；GitHub 连接未完成：" + safe_terminal_text(redact_sensitive_text(error)), err=True)
            raise typer.Exit(1) from None
    typer.echo("配置完成。运行 agent 启动，聊天中用 /model 切换。")

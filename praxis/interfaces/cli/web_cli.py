"""``praxis web``:搜索 provider 凭据的 CLI 面(Task 13;留档③)。

本模块只经 ``praxis.interfaces.sdk`` 触达适配器(与 models_cli 同一纪律):
- ``login <provider> [--key-env NAME]``:hidden prompt → ``save_web_search_credential``
  (**值不进 argv/日志**,写 Keychain 或 0600 文件);**绝不写 web.json** ——
  该文件是 operator-authored 配置,含 provider 与 api_key_env 变量名,不含密钥。
- ``status``:配置来源、provider、凭据"存在/缺失"(经 credential_source 风格
  探针,**绝不打印值**)。

web.json 缺失 ⇒ login 给出可执行错误(创建文件 + 运行命令),status 正常报告
"not configured"。provider 名集合与环境变量名正则都从 ``search.py`` 经 sdk
re-export 消费(单一事实,R-T13-M8)。凭据层共享文案里的 "model credential"
在本 CLI 边界改写为 web search 措辞(R-T13-M5),不改 credentials.py。

退出码沿用主 CLI:0 完成 / 1 运行时失败 / 3 用法错误(``typer.BadParameter``
由主 app 的 ``_UsageExit3Group`` 统一重映射);main.py 已挂载 ``praxis web``。
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from typing import NoReturn, cast

import typer

from praxis.interfaces.sdk import (
    API_KEY_ENV_NAME,
    SEARCH_ENDPOINTS,
    SEARCH_PROVIDERS,
    SearchProvider,
    SearchProviderConfig,
    load_web_search_config,
    save_web_search_credential,
    web_configuration_path,
    web_search_credential_source,
)

EXIT_COMPLETED = 0
EXIT_FAILED = 1

_PROVIDER_ENV = "PRAXIS_WEB_SEARCH_PROVIDER"
_PROVIDER_TEXT = ", ".join(SEARCH_PROVIDERS[:-1]) + " or " + SEARCH_PROVIDERS[-1]

app = typer.Typer(
    help="Manage web search provider credentials (web.json stays operator-authored).",
    no_args_is_help=True,
)


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(EXIT_FAILED)


def _credential_message(error: Exception) -> str:
    """把凭据层共享的 "model credential" 固定文案改写成 web 搜索措辞。"""
    return (
        str(error)
        .replace("Model credential", "Web search credential")
        .replace("model credential", "web search credential")
    )


def _validate_provider(provider: str) -> None:
    if provider not in SEARCH_PROVIDERS:
        raise typer.BadParameter(f"provider must be {_PROVIDER_TEXT} (got {provider!r})")


def _config_or_fail(environ: Mapping[str, str] | None = None) -> SearchProviderConfig | None:
    try:
        return load_web_search_config(environ=environ)
    except ValueError as error:
        _fail(str(error))


def _provider_source(env_provider: str | None, path: object) -> str:
    return f"{_PROVIDER_ENV}={env_provider}" if env_provider is not None else str(path)


@app.command("login")
def login(
    provider: str = typer.Argument(..., help=f"{_PROVIDER_TEXT}."),
    key_env: str | None = typer.Option(
        None, "--key-env", help="Environment variable name that holds the key."
    ),
) -> None:
    """为 provider 保存搜索凭据(hidden prompt;不修改 web.json)。"""
    _validate_provider(provider)
    if key_env is not None and API_KEY_ENV_NAME.fullmatch(key_env) is None:
        raise typer.BadParameter("--key-env must be an environment variable name")
    # 文件视图(忽略 env 覆盖)用于取 api_key_env;生效视图用于归因与实际 provider。
    file_config = _config_or_fail(environ={})
    effective = _config_or_fail()
    path = web_configuration_path()
    if file_config is None:
        _fail(
            f"web search configuration not found: {path}; create it with "
            '{"version": 1, "provider": "deepseek|brave|tavily", "api_key_env": "NAME"} '
            "then run: praxis web login <provider>"
        )
    env_provider = os.environ.get(_PROVIDER_ENV) or None
    source = _provider_source(env_provider, path)
    effective_provider = effective.provider if effective is not None else file_config.provider
    if key_env is None:
        if provider not in {file_config.provider, effective_provider}:
            _fail(
                f"web search provider is {effective_provider!r} (from {source}); "
                f"run: praxis web login {effective_provider}, or pass --key-env NAME"
            )
        key_env = file_config.api_key_env
    value = typer.prompt(f"Search API key for {key_env}", hide_input=True)
    if not value.strip():
        _fail("empty credential; nothing saved")
    try:
        save_web_search_credential(
            key_env, SEARCH_ENDPOINTS[cast("SearchProvider", provider)], value
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        _fail(_credential_message(error))
    note = ""
    if env_provider is not None and provider != effective_provider:
        note = f" (note: runtime provider is {effective_provider} from {_PROVIDER_ENV})"
    typer.echo(f"ok: web search credential saved for {provider} ({key_env}){note}")


@app.command("status")
def status() -> None:
    """报告 web.json 来源、provider 与凭据存在性(绝不打印值)。"""
    path = web_configuration_path()
    env_provider = os.environ.get(_PROVIDER_ENV) or None
    suffix = f" (provider overridden by {_PROVIDER_ENV})" if env_provider else ""
    typer.echo(f"config: {path}{suffix}")
    config = _config_or_fail()
    if config is None:
        typer.echo(
            "web search: not configured; create "
            f'{path} with {{"version": 1, "provider": "deepseek|brave|tavily", '
            '"api_key_env": "NAME"} and run: praxis web login <provider>'
        )
        raise typer.Exit(EXIT_COMPLETED)
    typer.echo(f"provider: {config.provider} (from {_provider_source(env_provider, path)})")
    typer.echo(f"api_key_env: {config.api_key_env}")
    try:
        source = web_search_credential_source(
            config.api_key_env, SEARCH_ENDPOINTS[config.provider]
        )
    except ValueError as error:
        _fail(_credential_message(error))
    if source == "missing":
        typer.echo(
            f"credential: missing; run: praxis web login {config.provider} "
            f"--key-env {config.api_key_env}"
        )
    else:
        typer.echo(f"credential: present ({source})")
    raise typer.Exit(EXIT_COMPLETED)


__all__ = ["app"]
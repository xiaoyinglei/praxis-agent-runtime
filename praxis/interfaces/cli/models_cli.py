"""``praxis models``:全局模型注册表/连接与凭据的唯一 CLI 面(设计 §5.15)。

本模块只经 ``praxis.interfaces.sdk`` 触达内核/适配器(契约 praxis_cli_via_sdk):
- 模型与连接全部经 ``ModelConfigStore``;
- 凭据值只经 ``resolve_model_credential``/``save_model_credential``,绝不打印,
  绝不进入 argv(prompt 是隐藏输入);
- 适配器构造只经 sdk 的 ``model_adapter_for``/``validate_model_spec``(R-T6-2)。

退出码与主 CLI 一致:0 完成 / 1 运行时失败(含 CAS 冲突,干净报错)/
3 用法错误(typer.BadParameter 由主 app 的 _UsageExit3Group 统一重映射)。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, NoReturn, cast

import click
import typer

from praxis.interfaces.sdk import (
    ConfigVersionConflict,
    ConnectionEntry,
    CostMicros,
    ModelConfigStore,
    ModelEntry,
    ModelSpec,
    model_adapter_for,
    probe_model,
    resolve_model_credential,
    save_model_credential,
    validate_model_spec,
)

EXIT_COMPLETED = 0
EXIT_FAILED = 1

_PROVIDERS = ("openai-compatible", "ollama")
_PROVIDER_CHOICE = click.Choice(list(_PROVIDERS))
_OPTION_ARG = typer.Option(
    [], "--option", help="Provider option KEY=JSON (repeatable; validated before writing)."
)

app = typer.Typer(
    help="Register models/connections and manage their credentials (global registry).",
    no_args_is_help=True,
)
connection_app = typer.Typer(help="Manage provider connections.", no_args_is_help=True)
app.add_typer(connection_app, name="connection")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(EXIT_FAILED)


@contextmanager
def _store() -> Iterator[ModelConfigStore]:
    """打开全局库;KeyError/CAS/ValueError 一律干净失败(退出码 1,无 traceback)。"""
    store: ModelConfigStore | None = None
    try:
        store = ModelConfigStore()
        yield store
    except (ConfigVersionConflict, KeyError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        _fail(str(message))
    finally:
        if store is not None:
            store.close()


def dollars_per_1m_to_micros(value: str | int | float | Decimal) -> int:
    """美元/1M token → micros/1M token;本模块唯一换算实现。"""
    return int(Decimal(str(value)) * 1_000_000)


def _price(value: str | None, option: str) -> int:
    if value is None:
        return 0
    try:
        return dollars_per_1m_to_micros(value)
    except (ArithmeticError, ValueError):
        raise typer.BadParameter(f"{option} expects a number of US dollars per 1M tokens") from None


def _parse_options(raw: list[str]) -> dict[str, Any]:
    """``--option KEY=JSON``(可重复);非法 JSON 回落为字符串值。"""
    options: dict[str, Any] = {}
    for text in raw:
        key, separator, value_text = text.partition("=")
        if not separator or not key:
            raise typer.BadParameter(f"--option expects KEY=JSON, got {text!r}")
        if key in options:
            raise typer.BadParameter(f"duplicate --option for {key!r}")
        try:
            options[key] = json.loads(value_text)
        except json.JSONDecodeError:
            options[key] = value_text
    return options


def _dollars(micros: int) -> str:
    return format(Decimal(micros) / 1_000_000, "f")


# ---------------------------------------------------------------------------
# models list / show
# ---------------------------------------------------------------------------


@app.command("list")
def list_models() -> None:
    """列出已注册模型;``*`` 标记默认模型。"""
    with _store() as store:
        entries = store.list_models()
        default = store.default_model()
        if not entries:
            typer.echo("no models registered; run: praxis models add <model-id> --provider …")
            return
        for entry in entries:
            marker = "*" if entry.spec.model_id == default else " "
            typer.echo(
                f"{marker} {entry.spec.model_id}\t{entry.spec.provider}"
                f"\t{entry.spec.base_url}\t{entry.spec.wire_model_id}"
            )


@app.command("show")
def show(model_id: str) -> None:
    """打印一个模型的完整配置(凭据只显示环境变量名,绝不显示值)。"""
    with _store() as store:
        entry = store.get(model_id)
        if entry is None:
            raise KeyError(f"model {model_id!r} is not registered; run: praxis models list")
        spec = entry.spec
        typer.echo(f"model: {spec.model_id}")
        typer.echo(f"  provider: {spec.provider}")
        typer.echo(f"  connection: {entry.connection.name}")
        typer.echo(f"  base_url: {spec.base_url}")
        typer.echo(f"  api_key_env: {spec.api_key_env or '-'}")
        typer.echo(f"  remote_model_id: {spec.wire_model_id}")
        typer.echo(f"  context_window_tokens: {spec.context_window_tokens}")
        typer.echo(f"  max_output_tokens: {spec.max_output_tokens}")
        typer.echo(f"  supports_tools: {str(spec.supports_tools).lower()}")
        typer.echo(f"  tokenizer: {spec.tokenizer_model or '-'}")
        typer.echo(
            "  cost (USD per 1M tokens): "
            f"input={_dollars(spec.cost.input_per_1m)} "
            f"output={_dollars(spec.cost.output_per_1m)} "
            f"cache_read={_dollars(spec.cost.cache_read_per_1m)} "
            f"cache_write={_dollars(spec.cost.cache_write_per_1m)}"
        )


# ---------------------------------------------------------------------------
# models add / remove / set-default
# ---------------------------------------------------------------------------


@app.command("add")
def add(
    model_id: str,
    provider: str = typer.Option(..., "--provider", click_type=_PROVIDER_CHOICE),
    base_url: str = typer.Option(..., "--base-url", help="Absolute provider endpoint."),
    api_key_env: str | None = typer.Option(
        None, "--api-key-env", help="Environment variable name (never the value)."
    ),
    connection: str | None = typer.Option(
        None, "--connection", help="Connection name (default: provider)."
    ),
    remote_model_id: str | None = typer.Option(None, "--remote-model-id"),
    context_window: int = typer.Option(8192, "--context-window"),
    max_output: int = typer.Option(4096, "--max-output"),
    price_in_per_1m: str | None = typer.Option(None, "--price-in-per-1m"),
    price_out_per_1m: str | None = typer.Option(None, "--price-out-per-1m"),
    supports_tools: bool = typer.Option(True, "--supports-tools/--no-tools"),
    tokenizer: str | None = typer.Option(None, "--tokenizer"),
    option: list[str] = _OPTION_ARG,
) -> None:
    """注册/更新一个模型;写库前先用目标 provider 的真实 adapter 干跑(R-T6-1)。"""
    try:
        cost = CostMicros(
            input_per_1m=_price(price_in_per_1m, "--price-in-per-1m"),
            output_per_1m=_price(price_out_per_1m, "--price-out-per-1m"),
        )
        options = _parse_options(option)
        spec = ModelSpec(
            model_id=model_id,
            provider=cast("Any", provider),
            remote_model_id=remote_model_id,
            base_url=base_url,
            api_key_env=api_key_env,
            context_window_tokens=context_window,
            max_output_tokens=max_output,
            supports_tools=supports_tools,
            tokenizer_model=tokenizer,
            cost=cost,
            provider_options=options,
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from None

    credential: str | None = None
    if spec.api_key_env is not None:
        try:
            credential = resolve_model_credential(spec.api_key_env, base_url)
        except ValueError as exc:
            _fail(str(exc))
    if provider == "openai-compatible" and credential is None:
        hint = (
            f"run: praxis models login {model_id}"
            if spec.api_key_env is not None
            else "pass --api-key-env and export that variable"
        )
        raise typer.BadParameter(
            f"openai-compatible model {model_id!r} has no resolvable credential; {hint}"
        )

    try:
        validate_model_spec(spec, credential)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None

    entry = ConnectionEntry(
        name=connection or provider,
        provider=cast("Any", provider),
        base_url=base_url,
        api_key_env=api_key_env,
    )
    with _store() as store:
        store.upsert_connection(entry)
        store.upsert_model(ModelEntry(spec=spec, connection=entry, revision=0))
    typer.echo(f"ok: registered {model_id}")


@app.command("remove")
def remove(model_id: str) -> None:
    """删除模型(若是默认模型则同时清空默认)。"""
    with _store() as store:
        store.remove_model(model_id, expected_revision=store.revision)
    typer.echo(f"ok: removed {model_id}")


@app.command("set-default")
def set_default(model_id: str) -> None:
    """把模型设为默认(省略 --model 时使用)。"""
    with _store() as store:
        store.set_default_model(model_id, expected_revision=store.revision)
    typer.echo(f"ok: default model {model_id}")


# ---------------------------------------------------------------------------
# models connection add / list
# ---------------------------------------------------------------------------


@connection_app.command("add")
def connection_add(
    name: str,
    provider: str = typer.Option(..., "--provider", click_type=_PROVIDER_CHOICE),
    base_url: str = typer.Option(..., "--base-url"),
    api_key_env: str | None = typer.Option(None, "--api-key-env"),
    location: str | None = typer.Option(None, "--location"),
) -> None:
    """注册/更新一个 provider 连接(模型通过同名连接共享 base_url/api_key_env)。"""
    entry = ConnectionEntry(
        name=name,
        provider=cast("Any", provider),
        base_url=base_url,
        api_key_env=api_key_env,
        location=location,
    )
    with _store() as store:
        store.upsert_connection(entry)
    typer.echo(f"ok: connection {name}")


@connection_app.command("list")
def connection_list() -> None:
    """列出已注册连接(只显示 api_key_env 名字,不显示凭据值)。"""
    with _store() as store:
        entries = store.list_connections()
    if not entries:
        typer.echo("no connections")
        return
    for entry in entries:
        typer.echo(
            f"{entry.name}\t{entry.provider}\t{entry.base_url}"
            f"\t{entry.api_key_env or '-'}\t{entry.location or '-'}"
        )


# ---------------------------------------------------------------------------
# models login / check
# ---------------------------------------------------------------------------


@app.command("login")
def login(model_id: str) -> None:
    """为已注册模型保存凭据(hidden prompt;值不进 argv/日志)。"""
    with _store() as store:
        spec = store.resolve_spec(model_id)
    if spec.api_key_env is None:
        _fail(
            f"model {model_id!r} has no api_key_env; "
            "re-register it with praxis models add --api-key-env"
        )
    value = typer.prompt(f"API key for {spec.api_key_env}", hide_input=True)
    if not value.strip():
        _fail("empty credential; nothing saved")
    try:
        save_model_credential(spec.api_key_env, spec.base_url or "", value)
    except ValueError as exc:
        _fail(str(exc))
    typer.echo(f"ok: credential saved for {model_id}")


@app.command("check")
def check(model_id: str) -> None:
    """轻量探针:文本步(+ 支持工具时的工具步);ok → 0,失败 → 1。"""
    with _store() as store:
        spec = store.resolve_spec(model_id)
    try:
        api_key = (
            resolve_model_credential(spec.api_key_env, spec.base_url or "")
            if spec.api_key_env is not None
            else None
        )
        adapter = model_adapter_for(spec, api_key)
    except ValueError as exc:
        _fail(str(exc))
    outcome = asyncio.run(probe_model(adapter))
    if outcome.detail:
        typer.echo(outcome.detail, err=True)
    if outcome.usage is not None:
        typer.echo(
            f"usage: input={outcome.usage.input_tokens} "
            f"output={outcome.usage.output_tokens} ({outcome.usage.source})"
        )
    steps = ",".join(outcome.steps) or "-"
    if outcome.ok:
        typer.echo(f"ok: {model_id} (steps: {steps})")
        raise typer.Exit(EXIT_COMPLETED)
    typer.echo(f"failed: {model_id} (steps: {steps})", err=True)
    raise typer.Exit(EXIT_FAILED)


__all__ = ["app", "connection_app", "dollars_per_1m_to_micros"]
"""Task 13:``web_search`` + ``web.json`` provider 配置 + 搜索凭据纪律(留档③)。

brief Step 1 清单逐条:
- 三 provider 请求形状在 wire 上逐条验证(method/URL/鉴权头/params/body);
- 缺配置 ⇒ ``web_search_unconfigured`` 且文案指向 ``praxis web login``;
- provider 401/403 不回显 body,key 不出现在 text/metadata/str(exc);
- URL target = provider endpoint 使真实 decide 的 ask/deny 生效(一条管道用例);
- ``max_results`` 越界 invalid_input(1..20);
- web.json live reload(改写文件 ⇒ 下一次调用立刻看到);
- ``PRAXIS_WEB_SEARCH_PROVIDER`` 覆盖;``PRAXIS_WEB_SEARCH_KEY_FILE`` 完全惰性;
- 凭据 env 优先 + POSIX 0600 往返 + Keychain service 前缀 ``praxis.web.``;
- 取消:await 点可取消 ⇒ COOPERATIVE/CANCEL(声明与实测一致)。

全部经假 resolver + 假 httpcore backend 注入,绝不触网;CLI 用例强制 POSIX
凭据分支,不碰开发者真钥匙串。
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from praxis.adapters.config import credentials
from praxis.adapters.config.credentials import (
    delete_web_search_credential,
    resolve_web_search_credential,
    save_web_search_credential,
    web_search_credential_source,
)
from praxis.adapters.web.limits import web_configuration_path
from praxis.adapters.web.search import (
    DEFAULT_MAX_RESULTS,
    MAX_MODEL_OUTPUT_BYTES,
    MAX_RESULTS_LIMIT,
    SEARCH_ENDPOINTS,
    TOOL_TIMEOUT_SECONDS,
    WEB_SEARCH_TOOL_NAME,
    SearchProviderConfig,
    create_web_search_tools,
    load_web_search_config,
    run_web_search,
)
from praxis.composition import make_web_authorizer
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.interfaces.cli import app as cli_app
from praxis.interfaces.cli import web_cli
from praxis.protocol.items import ItemKind
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy, ToolRule
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResultStatus,
    Tool,
    ToolCall,
    ToolEffect,
    ToolSource,
    ToolValidationError,
)
from praxis.store.log import Store
from praxis.store.turns import create_turn
from tests.praxis.test_web_urls_dns import (
    PUBLIC_IP,
    FakeBackend,
    FakeResolver,
    FakeStream,
    make_client,
    response_chunks,
)

KEY = "sk-WEBSECRET"
QUERY = "praxis web search"
DEEPSEEK = SearchProviderConfig(provider="deepseek", api_key_env="DEEPSEEK_SEARCH_KEY")
BRAVE = SearchProviderConfig(provider="brave", api_key_env="BRAVE_SEARCH_KEY")
TAVILY = SearchProviderConfig(provider="tavily", api_key_env="TAVILY_SEARCH_KEY")

_PAYLOADS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "results": [
            {"title": "Deep Title", "url": "https://example.com/deep", "snippet": "deep snippet"},
            {"title": "Deep Two", "url": "https://example.com/two", "snippet": "second"},
        ]
    },
    "brave": {
        "web": {
            "results": [
                {"title": "Brave Title", "url": "https://example.com/brave", "description": "brave snippet"}
            ]
        }
    },
    "tavily": {
        "results": [
            {"title": "Tavily Title", "url": "https://example.com/tavily", "content": "tavily snippet"}
        ]
    },
}


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_web_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """不继承真实 provider env;凭据统一走 POSIX 分支(不碰真钥匙串)。"""
    monkeypatch.delenv("PRAXIS_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("PRAXIS_WEB_SEARCH_KEY_FILE", raising=False)
    monkeypatch.setattr(credentials, "_PLATFORM", "linux")


def json_stream(payload: object, *, status: str = "200 OK") -> FakeStream:
    return FakeStream(
        response_chunks(
            status,
            headers=[("Content-Type", "application/json")],
            body=json.dumps(payload).encode(),
        )
    )


def make_tool(
    streams: Sequence[FakeStream],
    *,
    config: SearchProviderConfig | None = DEEPSEEK,
    key: str | None = KEY,
    authorize: object = None,
    resolver: FakeResolver | None = None,
    config_provider: object = None,
    credential_provider: object = None,
) -> tuple[Tool, FakeBackend, FakeResolver]:
    backend = FakeBackend(streams)
    resolver = resolver if resolver is not None else FakeResolver([PUBLIC_IP])
    client, _ = make_client(resolver=resolver, backend=backend, authorize=authorize)
    (tool,) = create_web_search_tools(
        client,
        config_provider=config_provider if config_provider is not None else (lambda: config),  # type: ignore[arg-type]
        credential_provider=credential_provider if credential_provider is not None else (lambda _cfg: key),  # type: ignore[arg-type]
    )
    return tool, backend, resolver


def run_tool(tool: Tool, arguments: dict[str, Any] | None = None) -> Any:
    return asyncio.run(
        tool.run({"query": QUERY, **(arguments or {})}, ToolExecContext())
    )


def request_parts(stream: FakeStream) -> tuple[str, str, dict[str, str], bytes]:
    """拆出首个请求的 (method, target, headers, body)。"""
    raw = b"".join(stream.writes)
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    method, target, _version = lines[0].split(" ")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return method, target, headers, body


def query_params(target: str) -> dict[str, str]:
    parsed = urllib.parse.parse_qs(urllib.parse.urlsplit(target).query)
    return {key: values[0] for key, values in parsed.items()}


def write_config(path: Path, **overrides: object) -> None:
    data: dict[str, object] = {
        "version": 1,
        "provider": "deepseek",
        "api_key_env": "DEEPSEEK_SEARCH_KEY",
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# web.json 配置加载
# ---------------------------------------------------------------------------


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_web_search_config(tmp_path / "absent.json") is None


def test_valid_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "web.json"
    write_config(path, provider="tavily", api_key_env="TAVILY_SEARCH_KEY")
    assert load_web_search_config(path) == TAVILY


def test_limits_only_shared_file_is_not_search_config(tmp_path: Path) -> None:
    path = tmp_path / "web.json"
    path.write_text(json.dumps({"version": 1, "limits": {"timeout_seconds": 5}}), encoding="utf-8")
    assert load_web_search_config(path) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "not json",
        [],
        {"version": 2, "provider": "deepseek", "api_key_env": "KEY"},
        {"version": True, "provider": "deepseek", "api_key_env": "KEY"},
        {"version": 1, "provider": "google", "api_key_env": "KEY"},
        {"version": 1, "provider": "deepseek", "api_key_env": "1KEY"},
        {"version": 1, "provider": "deepseek", "api_key_env": "KEY-X"},
        {"version": 1, "provider": "deepseek", "api_key_env": "KEY", "extra": 1},
        {"version": 1, "provider": "deepseek"},
    ],
)
def test_malformed_config_names_file(tmp_path: Path, mutation: object) -> None:
    path = tmp_path / "web.json"
    text = mutation if isinstance(mutation, str) else json.dumps(mutation)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_web_search_config(path)
    assert str(path) in str(excinfo.value)


def test_env_provider_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "web.json"
    write_config(path, provider="deepseek")
    monkeypatch.setenv("PRAXIS_WEB_SEARCH_PROVIDER", "brave")
    assert load_web_search_config(path) == SearchProviderConfig(
        provider="brave", api_key_env="DEEPSEEK_SEARCH_KEY"
    )
    monkeypatch.setenv("PRAXIS_WEB_SEARCH_PROVIDER", "google")
    with pytest.raises(ValueError, match="deepseek, brave or tavily"):
        load_web_search_config(path)


def test_key_file_env_is_inert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "web.json"
    write_config(path)
    monkeypatch.setenv("PRAXIS_WEB_SEARCH_KEY_FILE", str(tmp_path / "does-not-exist.key"))
    # 不读、不校验、不报错:配置与行为完全同 KEY_FILE 未设置。
    assert load_web_search_config(path) == DEEPSEEK


def test_config_is_live_not_snapshotted(tmp_path: Path) -> None:
    path = tmp_path / "web.json"
    write_config(path, provider="brave", api_key_env="BRAVE_SEARCH_KEY")
    tool, _backend, resolver = make_tool(
        [json_stream(_PAYLOADS["brave"]), json_stream(_PAYLOADS["deepseek"])],
        config_provider=lambda: load_web_search_config(path),
    )
    first = tool.resolve_use({"query": QUERY})
    assert [target.value for target in first.targets] == [SEARCH_ENDPOINTS["brave"]]
    # run 现读:重写文件前的调用命中 brave(GET)。
    assert run_tool(tool).metadata["provider"] == "brave"

    write_config(path, provider="deepseek", api_key_env="DEEPSEEK_SEARCH_KEY")
    second = tool.resolve_use({"query": QUERY})
    assert [target.value for target in second.targets] == [SEARCH_ENDPOINTS["deepseek"]]
    # 同一次调用后的下一次 run 立刻看到新 provider(POST)。
    assert run_tool(tool).metadata["provider"] == "deepseek"
    assert request_parts(_backend.issued[0])[0] == "GET"
    assert request_parts(_backend.issued[1])[0] == "POST"
    assert len(resolver.calls) == 2


def test_credential_binding_is_atomic_with_live_config(tmp_path: Path) -> None:
    """R-T13-1:credential_provider 收到同一个 live cfg ⇒ 中途改写 web.json 也无串用。"""
    path = tmp_path / "web.json"
    write_config(path, provider="brave", api_key_env="BRAVE_SEARCH_KEY")
    seen: list[tuple[str, str]] = []

    def credential_provider(config: SearchProviderConfig) -> str:
        seen.append((config.provider, config.api_key_env))
        # 在凭据解析中途把 web.json 翻成另一个 provider:本次调用必须仍用旧快照。
        write_config(path, provider="deepseek", api_key_env="DEEPSEEK_SEARCH_KEY")
        return f"key-{config.provider}"

    tool, backend, _ = make_tool(
        [json_stream(_PAYLOADS["brave"]), json_stream(_PAYLOADS["deepseek"])],
        config_provider=lambda: load_web_search_config(path),
        credential_provider=credential_provider,
    )
    first = run_tool(tool)
    assert first.metadata["provider"] == "brave"
    assert request_parts(backend.issued[0])[2]["x-subscription-token"] == "key-brave"

    second = run_tool(tool)
    assert second.metadata["provider"] == "deepseek"
    assert request_parts(backend.issued[1])[2]["authorization"] == "Bearer key-deepseek"
    assert seen == [
        ("brave", "BRAVE_SEARCH_KEY"),
        ("deepseek", "DEEPSEEK_SEARCH_KEY"),
    ]


# ---------------------------------------------------------------------------
# ACI:schema / 效果 / target / 取消声明
# ---------------------------------------------------------------------------


def test_tool_aci_shape() -> None:
    tool, _, _ = make_tool([])
    assert tool.definition.name == WEB_SEARCH_TOOL_NAME
    schema = tool.definition.input_schema
    assert set(schema["properties"]) == {"query", "max_results"}  # type: ignore[arg-type]
    assert schema["additionalProperties"] is False
    assert tuple(schema["required"]) == ("query",)
    assert schema["properties"]["max_results"]["default"] == DEFAULT_MAX_RESULTS == 8  # type: ignore[index]
    assert schema["properties"]["query"]["maxLength"] == 512  # type: ignore[index]

    assert tool.static_effects == frozenset({ToolEffect.NETWORK})
    assert tool.idempotent is True
    assert tool.concurrency_safe is True
    assert tool.timeout_seconds == TOOL_TIMEOUT_SECONDS == 45.0
    assert tool.max_model_output_bytes == MAX_MODEL_OUTPUT_BYTES == 65_536
    # await 点(asyncio HTTP)全程可取消:CANCEL 是如实声明。
    assert tool.cancellation_mode is CancellationMode.COOPERATIVE
    assert tool.interrupt_behavior is InterruptBehavior.CANCEL
    assert tool.source is ToolSource.BUILTIN
    assert "untrusted" in tool.definition.description
    assert "web_search_unconfigured" in tool.definition.description


@pytest.mark.parametrize("provider", ["deepseek", "brave", "tavily"])
def test_resolve_use_target_is_provider_endpoint(provider: str) -> None:
    config = SearchProviderConfig(provider=provider, api_key_env="NAME")  # type: ignore[arg-type]
    tool, _, _ = make_tool([], config=config)
    use = tool.resolve_use({"query": QUERY})
    assert use.effects == frozenset({ToolEffect.NETWORK})
    assert [(target.kind, target.value) for target in use.targets] == [
        ("url", SEARCH_ENDPOINTS[provider])  # type: ignore[index]
    ]


def test_unconfigured_resolve_use_has_no_target() -> None:
    tool, _, _ = make_tool([], config=None)
    assert tool.resolve_use({"query": QUERY}).targets == ()


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": "x" * 513},
        {"max_results": 0},
        {"max_results": MAX_RESULTS_LIMIT + 1},
        {"max_results": 1.5},
        {"extra": 1},
    ],
)
def test_schema_rejects_out_of_range(arguments: dict[str, Any]) -> None:
    tool, _, _ = make_tool([])
    with pytest.raises(ToolValidationError):
        tool.validate_input({"query": QUERY, **arguments})


@pytest.mark.parametrize(
    "arguments",
    [{}, {"max_results": 1}, {"max_results": MAX_RESULTS_LIMIT}, {"query": "x" * 512}],
)
def test_schema_accepts_in_range(arguments: dict[str, Any]) -> None:
    tool, _, _ = make_tool([])
    validated = tool.validate_input({"query": QUERY, **arguments})
    assert validated["query"] == arguments.get("query", QUERY)


def test_integral_float_max_results_is_coerced() -> None:
    tool, _, _ = make_tool([json_stream(_PAYLOADS["deepseek"])])
    result = run_tool(tool, {"max_results": 2.0})
    assert result.is_error is False
    assert result.metadata["result_count"] == 2


def test_cancellation_propagates_from_await_point() -> None:
    started = asyncio.Event()

    class SlowClient:
        async def fetch(self, url: str, **kwargs: object) -> object:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    (tool,) = create_web_search_tools(
        SlowClient(),  # type: ignore[arg-type]
        config_provider=lambda: DEEPSEEK,
        credential_provider=lambda _cfg: KEY,
    )

    async def scenario() -> None:
        task = asyncio.create_task(tool.run({"query": QUERY}, ToolExecContext()))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


class BlockingReadStream(FakeStream):
    """响应头之后的 read 永久挂起,用于真实 client 取消/关闭验证。"""

    def __init__(self, started: asyncio.Event) -> None:
        super().__init__()
        self.started = started

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.reads += 1
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def test_real_client_cancellation_closes_response_stream() -> None:
    """R-T13-M7:CANCEL 声明由真实 client.fetch 取消 + socket 关闭钉住。"""
    started = asyncio.Event()
    stream = BlockingReadStream(started)
    backend = FakeBackend([stream])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    (tool,) = create_web_search_tools(
        client,
        config_provider=lambda: DEEPSEEK,
        credential_provider=lambda _cfg: KEY,
    )

    async def scenario() -> None:
        task = asyncio.create_task(tool.run({"query": QUERY}, ToolExecContext()))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert stream.closed is True


# ---------------------------------------------------------------------------
# 三 provider 请求形状(wire capture)+ 归一
# ---------------------------------------------------------------------------


def test_deepseek_request_shape_and_normalization() -> None:
    resolver = FakeResolver([PUBLIC_IP])
    tool, backend, _ = make_tool([json_stream(_PAYLOADS["deepseek"])], config=DEEPSEEK, resolver=resolver)
    result = run_tool(tool, {"max_results": 3})
    assert result.is_error is False
    assert result.text.splitlines() == [
        "1. Deep Title — deep snippet (https://example.com/deep)",
        "2. Deep Two — second (https://example.com/two)",
    ]
    assert result.metadata == {"provider": "deepseek", "result_count": 2, "truncated": False}
    assert resolver.calls == [("api.deepseek.com", 443)]

    stream = backend.issued[0]
    method, target, headers, body = request_parts(stream)
    assert method == "POST"
    assert target == "/v1/tools/search"
    assert headers["authorization"] == f"Bearer {KEY}"
    assert headers["accept"] == "application/json"
    assert json.loads(body) == {"query": QUERY, "count": 3, "need_content": False}


def test_tavily_request_shape_and_normalization() -> None:
    resolver = FakeResolver([PUBLIC_IP])
    tool, backend, _ = make_tool([json_stream(_PAYLOADS["tavily"])], config=TAVILY, resolver=resolver)
    result = run_tool(tool, {"max_results": 4})
    assert result.is_error is False
    assert result.text == "1. Tavily Title — tavily snippet (https://example.com/tavily)"
    assert result.metadata["provider"] == "tavily"
    assert resolver.calls == [("api.tavily.com", 443)]

    method, target, headers, body = request_parts(backend.issued[0])
    assert method == "POST"
    assert target == "/search"
    assert headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(body) == {
        "query": QUERY,
        "max_results": 4,
        "search_depth": "basic",
        "auto_parameters": False,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }


def test_brave_request_shape_and_normalization() -> None:
    resolver = FakeResolver([PUBLIC_IP])
    tool, backend, _ = make_tool([json_stream(_PAYLOADS["brave"])], config=BRAVE, resolver=resolver)
    result = run_tool(tool, {"max_results": 5})
    assert result.is_error is False
    assert result.text == "1. Brave Title — brave snippet (https://example.com/brave)"
    assert result.metadata["provider"] == "brave"
    assert resolver.calls == [("api.search.brave.com", 443)]

    method, target, headers, body = request_parts(backend.issued[0])
    assert method == "GET"
    assert urllib.parse.urlsplit(target).path == "/res/v1/web/search"
    assert query_params(target) == {"q": QUERY, "count": "5"}
    assert headers["x-subscription-token"] == KEY
    assert headers["accept"] == "application/json"
    assert body == b""


@pytest.mark.parametrize(
    "config", [DEEPSEEK, BRAVE, TAVILY], ids=["deepseek", "brave", "tavily"]
)
def test_search_rejects_redirects_without_second_hop(config: SearchProviderConfig) -> None:
    stream = FakeStream(
        response_chunks(
            "302 Found",
            headers=[("Location", "https://evil.example/next")],
            body=b"",
        )
    )
    tool, backend, resolver = make_tool([stream], config=config)
    result = run_tool(tool)
    assert result.error_code == "web_search_failed:invalid_redirect"
    assert len(backend.connects) == 1
    assert len(resolver.calls) == 1


def test_max_results_caps_and_reports_truncated() -> None:
    tool, _, _ = make_tool([json_stream(_PAYLOADS["deepseek"])])
    capped = run_tool(tool, {"max_results": 1})
    assert capped.metadata["result_count"] == 1
    assert capped.metadata["truncated"] is True
    assert len(capped.text.splitlines()) == 1


def test_result_text_fields_stay_one_line() -> None:
    payload = {
        "results": [
            {
                "title": "Line one\n2. Forged line",
                "url": "https://example.com/lines",
                "snippet": "snip\nmore",
            }
        ]
    }
    tool, _, _ = make_tool([json_stream(payload)])
    result = run_tool(tool)
    assert result.text.splitlines() == [
        "1. Line one 2. Forged line — snip more (https://example.com/lines)"
    ]


def test_private_result_urls_are_skipped_not_fatal() -> None:
    payload = {
        "results": [
            {"title": "Private", "url": "https://127.0.0.1/x", "snippet": "never"},
            {"title": "Public", "url": "https://example.com/ok", "snippet": "yes"},
        ]
    }
    tool, _, _ = make_tool([json_stream(payload)])
    result = run_tool(tool)
    assert result.is_error is False
    assert [line for line in result.text.splitlines()] == [
        "1. Public — yes (https://example.com/ok)"
    ]


# ---------------------------------------------------------------------------
# 错误面:unconfigured / 401-403 / 凭证回显 / malformed
# ---------------------------------------------------------------------------


def test_unconfigured_is_actionable() -> None:
    tool, backend, _ = make_tool([], config=None)
    result = run_tool(tool)
    assert result.status is ResultStatus.ERROR
    assert result.error_code == "web_search_unconfigured"
    assert "praxis web login" in (result.error_message or "")
    assert backend.connects == []


def test_missing_credential_is_unconfigured_login_hint() -> None:
    tool, backend, _ = make_tool([], key=None)
    result = run_tool(tool)
    assert result.error_code == "web_search_unconfigured"
    assert "praxis web login deepseek --key-env DEEPSEEK_SEARCH_KEY" in (
        result.error_message or ""
    )
    assert backend.connects == []


def test_malformed_config_in_run_is_unconfigured_not_invalid_input(tmp_path: Path) -> None:
    path = tmp_path / "web.json"
    path.write_text("{broken", encoding="utf-8")
    tool, _, _ = make_tool([], config_provider=lambda: load_web_search_config(path))
    assert tool.resolve_use({"query": QUERY}).targets == ()
    result = run_tool(tool)
    assert result.error_code == "web_search_unconfigured"
    assert str(path) in (result.error_message or "")


@pytest.mark.parametrize("status", ["401 Unauthorized", "403 Forbidden"])
def test_rejected_credential_is_bounded_without_body_echo(status: str) -> None:
    body = f"provider says: {KEY} is invalid; secret-body-marker"
    stream = FakeStream(
        response_chunks(status, headers=[("Content-Type", "text/plain")], body=body.encode())
    )
    tool, _, _ = make_tool([stream])
    result = run_tool(tool)
    assert result.status is ResultStatus.ERROR
    assert result.error_code == "web_search_failed:unauthorized"
    outputs = [result.text, result.error_message or "", str(dict(result.metadata))]
    for output in outputs:
        assert KEY not in output
        assert "secret-body-marker" not in output
    assert status[:3] in (result.error_message or "")


def test_provider_rate_limit_class() -> None:
    stream = FakeStream(response_chunks("429 Too Many Requests", body=b"slow down"))
    tool, _, _ = make_tool([stream])
    result = run_tool(tool)
    assert result.error_code == "web_search_failed:rate_limited"
    assert "slow down" not in (result.error_message or "")


def test_other_non_2xx_is_http_error() -> None:
    stream = FakeStream(response_chunks("503 Service Unavailable", body=b"backend detail"))
    tool, _, _ = make_tool([stream])
    result = run_tool(tool)
    assert result.error_code == "web_search_failed:http_error"
    assert "503" in (result.error_message or "")
    assert "backend detail" not in (result.error_message or "")


def test_credential_echo_marks_response_invalid() -> None:
    payload = {
        "results": [
            {"title": "Echo", "url": "https://example.com/e", "snippet": f"token {KEY}"}
        ]
    }
    tool, _, _ = make_tool([json_stream(payload)])
    result = run_tool(tool)
    assert result.error_code == "web_search_failed:invalid_response"
    assert KEY not in result.text
    assert KEY not in str(dict(result.metadata))


def test_malformed_provider_response_is_invalid_response() -> None:
    tool, _, _ = make_tool([json_stream({"web": {}})])  # brave shape on deepseek
    assert run_tool(tool).error_code == "web_search_failed:invalid_response"

    body_stream = FakeStream(response_chunks("200 OK", body=b"not json"))
    tool, _, _ = make_tool([body_stream])
    assert run_tool(tool).error_code == "web_search_failed:invalid_response"


def test_success_output_never_contains_key() -> None:
    tool, _, _ = make_tool([json_stream(_PAYLOADS["deepseek"])])
    result = run_tool(tool)
    assert KEY not in result.text
    assert KEY not in str(dict(result.metadata))


# ---------------------------------------------------------------------------
# 真 decide 管道:URL target 让 ask/deny 生效(一条用例)
# ---------------------------------------------------------------------------


def _exec_ctx(
    store: Store,
    workspace: Path,
    tool: Tool,
    permissions: PermissionContext,
    *,
    turn_id: str,
) -> ExecCtx:
    turn = create_turn(store, turn_id=turn_id, thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools={WEB_SEARCH_TOOL_NAME: tool},
        permissions=permissions,
        turn_id=turn.turn_id,
        thread_id="thread-1",
        policy_revision=permissions.policy.revision,
    )


def test_pipeline_ask_deny_and_allow_use_url_target(
    store: Store, workspace: Path
) -> None:
    holder: dict[str, PermissionContext] = {}

    def permissions() -> PermissionContext:
        return holder["ctx"]

    tool, backend, _ = make_tool(
        [json_stream(_PAYLOADS["deepseek"])], authorize=make_web_authorizer(permissions)
    )

    ask_policy = PermissionPolicy(
        tools=(ToolRule(pattern=WEB_SEARCH_TOOL_NAME, decision="allow"),),
        network=NetworkPolicy(decision="ask"),
    )
    holder["ctx"] = PermissionContext(workspace=str(workspace), policy=ask_policy, mode="on-request")
    paused = asyncio.run(
        execute(
            ToolCall("c-ask", WEB_SEARCH_TOOL_NAME, {"query": QUERY}),
            _exec_ctx(store, workspace, tool, holder["ctx"], turn_id="turn-ask"),
        )
    )
    assert paused.status is ResultStatus.PAUSED
    assert backend.connects == []
    kinds = [item.kind for item in store.read_turn_items("turn-ask")]
    assert ItemKind.APPROVAL_REQUEST in kinds

    deny_policy = PermissionPolicy(
        tools=(ToolRule(pattern=WEB_SEARCH_TOOL_NAME, decision="allow"),),
        network=NetworkPolicy(decision="allow", denied_domains=("api.deepseek.com",)),
    )
    holder["ctx"] = PermissionContext(workspace=str(workspace), policy=deny_policy, mode="on-request")
    denied = asyncio.run(
        execute(
            ToolCall("c-deny", WEB_SEARCH_TOOL_NAME, {"query": QUERY}),
            _exec_ctx(store, workspace, tool, holder["ctx"], turn_id="turn-deny"),
        )
    )
    assert denied.status is ResultStatus.DENIED
    assert denied.error_code == "permission_denied"
    assert backend.connects == []

    allow_policy = PermissionPolicy(
        tools=(ToolRule(pattern=WEB_SEARCH_TOOL_NAME, decision="allow"),),
        network=NetworkPolicy(decision="allow"),
    )
    holder["ctx"] = PermissionContext(workspace=str(workspace), policy=allow_policy, mode="on-request")
    allowed = asyncio.run(
        execute(
            ToolCall("c-allow", WEB_SEARCH_TOOL_NAME, {"query": QUERY}),
            _exec_ctx(store, workspace, tool, holder["ctx"], turn_id="turn-allow"),
        )
    )
    assert allowed.status is ResultStatus.OK
    assert allowed.is_error is False
    assert "1. Deep Title" in allowed.text
    assert len(backend.connects) == 1


# ---------------------------------------------------------------------------
# 搜索凭据纪律:env 优先 / 0600 往返 / Keychain service 前缀
# ---------------------------------------------------------------------------


def test_web_credential_posix_roundtrip_and_endpoint_binding() -> None:
    endpoint = SEARCH_ENDPOINTS["deepseek"]
    save_web_search_credential("DEEPSEEK_SEARCH_KEY", endpoint, f"  {KEY}  ")
    assert resolve_web_search_credential("DEEPSEEK_SEARCH_KEY", endpoint) == KEY
    assert web_search_credential_source("DEEPSEEK_SEARCH_KEY", endpoint) == "store"
    # 同一 env 名换 endpoint(另一个 provider)不串用。
    assert resolve_web_search_credential("DEEPSEEK_SEARCH_KEY", SEARCH_ENDPOINTS["tavily"]) is None

    delete_web_search_credential("DEEPSEEK_SEARCH_KEY", endpoint)
    assert resolve_web_search_credential("DEEPSEEK_SEARCH_KEY", endpoint) is None
    delete_web_search_credential("DEEPSEEK_SEARCH_KEY", endpoint)  # no-op


def test_web_credential_env_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = SEARCH_ENDPOINTS["brave"]
    monkeypatch.setenv("BRAVE_SEARCH_KEY", "from-env")
    assert resolve_web_search_credential("BRAVE_SEARCH_KEY", endpoint) == "from-env"
    assert web_search_credential_source("BRAVE_SEARCH_KEY", endpoint) == "env"


def test_web_keychain_uses_praxis_web_service_and_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeRun:
        def __init__(self) -> None:
            self.responses = [(0, "", ""), (0, f"{KEY}\n", "")]

        def __call__(self, argv: list[str], **kwargs: object) -> Any:
            calls.append((list(argv), dict(kwargs)))
            returncode, stdout, stderr = self.responses.pop(0)
            import subprocess

            return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    monkeypatch.setattr(credentials, "_PLATFORM", "darwin")
    monkeypatch.setattr(credentials.subprocess, "run", FakeRun())
    save_web_search_credential("TAVILY_SEARCH_KEY", SEARCH_ENDPOINTS["tavily"], KEY)

    argv, kwargs = calls[0]
    assert argv == ["/usr/bin/security", "-i"]
    command = kwargs["input"]
    assert isinstance(command, str)
    assert "praxis.web." in command
    assert "praxis.model." not in command
    assert KEY not in argv

    read_argv, _read_kwargs = calls[1]
    assert read_argv[:2] == ["/usr/bin/security", "find-generic-password"]
    service = read_argv[read_argv.index("-s") + 1]
    assert service.startswith("praxis.web.")
    assert KEY not in read_argv


# ---------------------------------------------------------------------------
# CLI:praxis web login / status(不写 web.json,不打印 key)
# ---------------------------------------------------------------------------


def _runner() -> CliRunner:
    return CliRunner()


def write_host_config(**overrides: object) -> Path:
    path = web_configuration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_config(path, **overrides)
    return path


def test_cli_login_saves_credential_without_touching_web_json(tmp_path: Path) -> None:
    path = write_host_config()
    before = path.read_bytes()

    result = _runner().invoke(
        web_cli.app, ["login", "deepseek"], input=f"{KEY}\n"
    )
    assert result.exit_code == 0, result.output
    assert "credential saved" in result.output
    assert KEY not in result.output
    assert path.read_bytes() == before
    assert (
        resolve_web_search_credential("DEEPSEEK_SEARCH_KEY", SEARCH_ENDPOINTS["deepseek"])
        == KEY
    )


def test_cli_status_reports_presence_without_value(tmp_path: Path) -> None:
    write_host_config()
    save_web_search_credential("DEEPSEEK_SEARCH_KEY", SEARCH_ENDPOINTS["deepseek"], KEY)
    result = _runner().invoke(web_cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "provider: deepseek" in result.output
    assert "api_key_env: DEEPSEEK_SEARCH_KEY" in result.output
    assert "credential: present (store)" in result.output
    assert KEY not in result.output


def test_cli_status_unconfigured(tmp_path: Path) -> None:
    result = _runner().invoke(web_cli.app, ["status"])
    assert result.exit_code == 0
    assert "not configured" in result.output
    assert "praxis web login" in result.output


def test_cli_login_missing_web_json_is_actionable(tmp_path: Path) -> None:
    result = _runner().invoke(web_cli.app, ["login", "deepseek"], input=f"{KEY}\n")
    assert result.exit_code == 1
    assert "configuration not found" in result.output
    assert "api_key_env" in result.output


def test_cli_login_validates_provider_and_key_env(tmp_path: Path) -> None:
    write_host_config()
    bad_provider = _runner().invoke(web_cli.app, ["login", "google"], input=f"{KEY}\n")
    assert bad_provider.exit_code != 0
    assert "deepseek, brave or tavily" in bad_provider.output

    write_host_config(provider="brave")
    mismatch = _runner().invoke(web_cli.app, ["login", "deepseek"], input=f"{KEY}\n")
    assert mismatch.exit_code == 1
    assert "provider is 'brave'" in mismatch.output
    assert "web login brave" in mismatch.output

    bad_env = _runner().invoke(
        web_cli.app, ["login", "deepseek", "--key-env", "1BAD"], input=f"{KEY}\n"
    )
    assert bad_env.exit_code != 0
    assert "environment variable name" in bad_env.output

# ---------------------------------------------------------------------------
# main.py 挂载与 env 覆盖归因(R-T13-2 / M5 / M6)
# ---------------------------------------------------------------------------


def test_main_app_mounts_praxis_web() -> None:
    help_result = _runner().invoke(cli_app, ["web", "login", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "--key-env" in help_result.output

    status_result = _runner().invoke(cli_app, ["web", "status"])
    assert status_result.exit_code == 0, status_result.output
    assert "not configured" in status_result.output


def test_cli_login_env_override_names_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_host_config(provider="deepseek", api_key_env="DEEPSEEK_SEARCH_KEY")
    monkeypatch.setenv("PRAXIS_WEB_SEARCH_PROVIDER", "brave")

    # 文件 provider(deepseek)仍可显式登录;提示运行时 provider 来自 env。
    allowed = _runner().invoke(web_cli.app, ["login", "deepseek"], input=f"{KEY}\n")
    assert allowed.exit_code == 0, allowed.output
    assert "runtime provider is brave from PRAXIS_WEB_SEARCH_PROVIDER" in allowed.output

    # 与文件/生效 provider 都不同 ⇒ 失败文案归因 env,而不是 "web.json selects"。
    blocked = _runner().invoke(web_cli.app, ["login", "tavily"], input=f"{KEY}\n")
    assert blocked.exit_code == 1
    assert "PRAXIS_WEB_SEARCH_PROVIDER=brave" in blocked.output
    assert "web.json selects" not in blocked.output


def test_cli_status_rewrites_model_credential_wording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_host_config()
    monkeypatch.setattr(credentials, "_PLATFORM", "win32")
    result = _runner().invoke(web_cli.app, ["status"])
    assert result.exit_code == 1
    assert "web search credentials require macOS or POSIX" in result.output
    assert "model credential" not in result.output.lower()

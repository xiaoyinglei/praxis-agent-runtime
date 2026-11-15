"""Task 12:web_fetch Tool(行窗口 + 管道外部化;S8 URL 面工具化)。

brief Step 1 清单逐条:
- schema 阈值逐个:max_bytes 4096..16000、max_lines 1..500、timeout 5..120;
- 有意差异显式断言:`{render, source_id, compare_to_source_id, view, section_id,
  start_link, start_section, find, max_tokens, refresh}` **不在 schema**(R-PF-2');
- 行窗口跨多字节文本(UTF-8 字节预算)、默认 12000/100、长行 512 字符切块;
- 200_000 源抽取上限;65_536 输出上限经管道 artifact 指针(用缩小上限的 Tool 副本
  复现,因为 schema 的 max_bytes ≤16000 本身不可能触到 65_536);
- gzip 页面、非 HTML 透传、二进制拒收;
- 错误映射:invalid_input(URL 预检逐字形状)、denied、web_fetch_failed:<code>
  (invalid_redirect / too_many_redirects / invalid_url / timeout / http_error);
- 真管道 execute:happy path + denied path + invalid_input path + artifact 外部化
  (工具表注入,同 test_pipeline 姿势)。

全部经假 resolver + 假 httpcore backend(T10 fixture)注入,绝不触网。
"""

from __future__ import annotations

import asyncio
import gzip
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.web.client import PublicWebError, WebFetchResponse
from praxis.adapters.web.extract import MAX_SOURCE_CHARACTERS, MAX_VIRTUAL_LINE_CHARACTERS
from praxis.adapters.web.limits import WebLimits
from praxis.adapters.web.tools import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    MAX_MODEL_OUTPUT_BYTES,
    TOOL_TIMEOUT_SECONDS,
    WEB_FETCH_TOOL_NAME,
    create_web_fetch_tools,
)
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
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
from praxis.store.artifacts import ArtifactStore
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

URL = "https://example.com/page"
_ABSENT_KEYS = {
    "render",
    "source_id",
    "compare_to_source_id",
    "view",
    "section_id",
    "start_link",
    "start_section",
    "find",
    "max_tokens",
    "refresh",
}


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def make_tool(
    streams: list[FakeStream],
    *,
    client_limits: WebLimits | None = None,
    tool_limits: WebLimits | None = None,
    authorize: object = None,
) -> tuple[Tool, FakeBackend]:
    backend = FakeBackend(streams)
    client, _ = make_client(
        resolver=FakeResolver([PUBLIC_IP]),
        backend=backend,
        limits=client_limits,
        authorize=authorize,
    )
    (tool,) = create_web_fetch_tools(client, limits=tool_limits if tool_limits is not None else WebLimits())
    return tool, backend


def run_tool(tool: Tool, arguments: dict[str, Any]) -> Any:
    return asyncio.run(tool.run({"url": URL, **arguments}, ToolExecContext()))


def page(
    body: bytes,
    *,
    content_type: str = "text/html; charset=utf-8",
    status: str = "200 OK",
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> FakeStream:
    return FakeStream(
        response_chunks(
            status,
            headers=[("Content-Type", content_type), *extra_headers],
            body=body,
        )
    )


def fetch(
    body: bytes,
    *,
    content_type: str = "text/html; charset=utf-8",
    arguments: dict[str, Any] | None = None,
    client_limits: WebLimits | None = None,
    tool_limits: WebLimits | None = None,
    authorize: object = None,
) -> tuple[Any, FakeBackend]:
    tool, backend = make_tool(
        [page(body, content_type=content_type)],
        client_limits=client_limits,
        tool_limits=tool_limits,
        authorize=authorize,
    )
    return run_tool(tool, arguments or {}), backend


# ---------------------------------------------------------------------------
# Tool ACI:R-PF-2' 键集、效果、执行策略、target 形状
# ---------------------------------------------------------------------------


def test_tool_aci_shape_and_absent_keys() -> None:
    tool, _ = make_tool([])
    assert tool.definition.name == WEB_FETCH_TOOL_NAME
    schema = tool.definition.input_schema
    properties = set(schema["properties"])  # type: ignore[arg-type]
    assert properties == {"url", "max_bytes", "timeout_seconds", "start_line", "max_lines"}
    # R-PF-2' 有意差异:schema 完全不含这些旧键(T14 才加 render)。
    assert _ABSENT_KEYS.isdisjoint(properties)
    assert schema["additionalProperties"] is False
    assert tuple(schema["required"]) == ("url",)

    assert tool.static_effects == frozenset({ToolEffect.NETWORK})
    assert tool.idempotent is True
    assert tool.concurrency_safe is True
    assert tool.timeout_seconds == TOOL_TIMEOUT_SECONDS == 75.0
    assert tool.max_model_output_bytes == MAX_MODEL_OUTPUT_BYTES == 65_536
    assert tool.cancellation_mode is CancellationMode.COOPERATIVE
    # 抽取是进程内线程,线程不可杀:如实 FINISH_CURRENT(3A inspect_data_file 同款)。
    assert tool.interrupt_behavior is InterruptBehavior.FINISH_CURRENT
    assert tool.source is ToolSource.BUILTIN
    # M2:≤16000 的摘录预算使管道外部化在本工具上不可达,不得向模型承诺 artifact
    # 指针;M4:错误枚举覆盖抽取/行窗口码。
    description = tool.definition.description
    assert "artifact" not in description
    assert "web_line_out_of_range" in description
    assert "web_content_empty" in description
    assert "source_truncated" in description


def test_resolve_use_target_is_raw_url() -> None:
    tool, _ = make_tool([])
    raw = "HTTPS://Example.COM/a?b=1#frag"
    use = tool.resolve_use({"url": raw})
    assert use.effects == frozenset({ToolEffect.NETWORK})
    assert [(target.kind, target.value) for target in use.targets] == [("url", raw)]


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_bytes": 4_095},
        {"max_bytes": 16_001},
        {"max_lines": 0},
        {"max_lines": 501},
        {"timeout_seconds": 4.9},
        {"timeout_seconds": 120.1},
        {"start_line": 0},
        {"view": "content"},
        {"url": ""},
        {"url": "https://" + "a" * 4_090},
    ],
)
def test_schema_rejects_out_of_range(arguments: dict[str, Any]) -> None:
    tool, _ = make_tool([])
    with pytest.raises(ToolValidationError):
        tool.validate_input({"url": URL, **arguments})


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"max_bytes": 4_096},
        {"max_bytes": 16_000},
        {"max_lines": 1},
        {"max_lines": 500},
        {"timeout_seconds": 5},
        {"timeout_seconds": 120},
        {"start_line": 1},
    ],
)
def test_schema_accepts_in_range(arguments: dict[str, Any]) -> None:
    tool, _ = make_tool([])
    assert tool.validate_input({"url": URL, **arguments})["url"] == URL


def test_integral_float_integer_arguments_are_coerced() -> None:
    # JSON Schema 2020-12 接受 2.0 作为 integer(3.5/true/"3" 在 schema 层拒绝)。
    # 不做 int 化会让 lines[2.0] 抛裸 TypeError ⇒ 管道折成不透明 tool_error。
    lines = [f"line {index}" for index in range(1, 151)]
    body = "\n".join(lines).encode()
    int_result, _ = fetch(
        body,
        content_type="text/plain; charset=utf-8",
        arguments={"start_line": 2, "max_lines": 100, "max_bytes": 4_096},
    )
    float_result, _ = fetch(
        body,
        content_type="text/plain; charset=utf-8",
        arguments={"start_line": 2.0, "max_lines": 100.0, "max_bytes": 4_096.0},
    )
    assert float_result.is_error is False
    assert float_result.text == int_result.text
    assert float_result.metadata == int_result.metadata
    assert int_result.text.splitlines()[0] == "2: line 2"


def test_invalid_url_precheck_is_legacy_shaped_invalid_input() -> None:
    tool, _ = make_tool([])
    with pytest.raises(ToolValidationError) as excinfo:
        tool.validate_input({"url": "https://127.0.0.1/"})
    assert excinfo.value.path == "$.url"
    assert excinfo.value.message == (
        "a public HTTP(S) URL without credentials on port 80 or 443 is required"
    )


# ---------------------------------------------------------------------------
# 抽取:HTML 噪声/块边界、非 HTML 透传、二进制拒收
# ---------------------------------------------------------------------------


def test_html_extraction_strips_noise_and_keeps_block_breaks() -> None:
    body = (
        b"<html><head><title>TITLENOISE</title><style>.x{}</style></head>"
        b"<body><nav>NAVNOISE</nav><aside>ASIDENOISE</aside>"
        b"<script>SCRIPTNOISE</script>"
        b"<p>Hello <b>world</b>.</p><p>\xe7\xac\xac\xe4\xba\x8c\xe6\xae\xb5</p>"
        b"</body></html>"
    )
    result, _ = fetch(body)
    assert result.is_error is False
    assert result.text.splitlines() == ["1: Hello world.", "2: ", "3: 第二段"]
    for noise in ("TITLENOISE", "NAVNOISE", "ASIDENOISE", "SCRIPTNOISE"):
        assert noise not in result.text
    assert result.metadata["line_count"] == 3
    assert result.metadata["start_line"] == 1
    assert result.metadata["end_line"] == 3
    assert result.metadata["truncated"] is False
    assert result.metadata["source_truncated"] is False


def test_metadata_keys_are_exactly_the_brief_set() -> None:
    result, _ = fetch(b"<p>hello</p>")
    assert set(result.metadata) == {
        "content_type",
        "final_url",
        "total_bytes",
        "truncated",
        "source_truncated",
        "line_count",
        "start_line",
        "end_line",
    }
    assert result.metadata["content_type"] == "text/html; charset=utf-8"
    assert result.metadata["final_url"] == URL
    assert result.metadata["total_bytes"] == len(b"<p>hello</p>")


def test_non_html_text_passthrough() -> None:
    body = "第一行\n第二行 日本語\n".encode()
    result, _ = fetch(body, content_type="text/plain; charset=utf-8")
    assert result.is_error is False
    assert result.text.splitlines() == ["1: 第一行", "2: 第二行 日本語"]
    assert result.metadata["total_bytes"] == len(body)


def test_json_passthrough() -> None:
    body = b'{"k": "v"}'
    result, _ = fetch(body, content_type="application/json")
    assert result.is_error is False
    assert "1: " in result.text
    assert '{"k": "v"}' in result.text


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"%PDF-1.7 binary", "application/pdf"),
        (b"\x00\x01binary", "text/plain"),
        (b"<html></html>", "image/png"),
    ],
)
def test_unsupported_or_binary_content_rejected(body: bytes, content_type: str) -> None:
    result, _ = fetch(body, content_type=content_type)
    assert result.is_error is True
    assert result.status is ResultStatus.ERROR
    assert result.error_code == "web_fetch_failed:web_content_unsupported"


def test_empty_html_maps_web_content_empty() -> None:
    result, _ = fetch(b"<html><script>only noise</script></html>")
    assert result.error_code == "web_fetch_failed:web_content_empty"


def test_gzip_page_decodes_and_reports_decoded_bytes() -> None:
    html = b"<p>gzip page</p>"
    compressed = gzip.compress(html)
    tool, _ = make_tool(
        [
            FakeStream(
                response_chunks(
                    headers=[
                        ("Content-Type", "text/html; charset=utf-8"),
                        ("Content-Encoding", "gzip"),
                    ],
                    body=compressed,
                )
            )
        ]
    )
    result = run_tool(tool, {})
    assert result.is_error is False
    assert "1: gzip page" in result.text
    assert result.metadata["total_bytes"] == len(html)


def test_source_extraction_cap_200_000_characters() -> None:
    body = ("x" * 99 + "\n").encode() * 3_000
    assert len(body) > MAX_SOURCE_CHARACTERS
    result, _ = fetch(body, content_type="text/plain; charset=utf-8")
    assert result.is_error is False
    assert result.metadata["source_truncated"] is True
    assert result.metadata["line_count"] <= MAX_SOURCE_CHARACTERS // 99 + 2
    assert result.metadata["total_bytes"] == len(body)


# ---------------------------------------------------------------------------
# 行窗口:1-based 虚拟行、跨多字节文本、默认值、游标
# ---------------------------------------------------------------------------


def test_line_window_across_multibyte_text() -> None:
    lines = [f"第{index}行 日本語テキスト" for index in range(1, 31)]
    body = "\n".join(lines).encode()
    result, _ = fetch(
        body,
        content_type="text/plain; charset=utf-8",
        arguments={"start_line": 3, "max_lines": 5},
    )
    assert result.is_error is False
    assert result.text.splitlines() == [
        f"{number}: {lines[number - 1]}" for number in range(3, 8)
    ]
    assert result.metadata["line_count"] == 30
    assert result.metadata["start_line"] == 3
    assert result.metadata["end_line"] == 7
    assert result.metadata["truncated"] is True


def test_max_bytes_is_utf8_byte_budget_and_cursor_continues() -> None:
    lines = ["日本語" * 33 for _ in range(100)]  # 每个虚拟行约 300 UTF-8 字节
    body = "\n".join(lines).encode()
    result, _ = fetch(
        body,
        content_type="text/plain; charset=utf-8",
        arguments={"max_bytes": 4_096, "max_lines": 500},
    )
    assert result.is_error is False
    assert len(result.text.encode("utf-8")) <= 4_096
    assert result.metadata["end_line"] < result.metadata["line_count"]
    assert result.metadata["truncated"] is True
    next_start = result.metadata["end_line"] + 1
    continued, _ = fetch(
        body,
        content_type="text/plain; charset=utf-8",
        arguments={"start_line": next_start, "max_lines": 1},
    )
    assert continued.text == f"{next_start}: {lines[next_start - 1]}"


def test_default_max_bytes_and_max_lines() -> None:
    # 默认 12000 字节:200 字节 × 100 行 ⇒ 约 60 行。
    body = ("x" * 199 + "\n").encode() * 100
    default_result, _ = fetch(body, content_type="text/plain; charset=utf-8")
    assert len(default_result.text.encode("utf-8")) <= DEFAULT_MAX_BYTES
    assert DEFAULT_MAX_BYTES == 12_000
    assert default_result.metadata["truncated"] is True
    wide_result, _ = fetch(
        body, content_type="text/plain; charset=utf-8", arguments={"max_bytes": 16_000}
    )
    assert wide_result.metadata["end_line"] > default_result.metadata["end_line"]
    # 默认 100 行:300 条短行 ⇒ 恰 100 行。
    short_body = "\n".join(f"line {index}" for index in range(300)).encode()
    lines_result, _ = fetch(short_body, content_type="text/plain; charset=utf-8")
    assert lines_result.metadata["end_line"] == DEFAULT_MAX_LINES == 100
    assert lines_result.metadata["truncated"] is True


def test_long_physical_line_wraps_at_512_characters() -> None:
    body = ("a" * (MAX_VIRTUAL_LINE_CHARACTERS * 2 + 100)).encode()
    result, _ = fetch(
        body,
        content_type="text/plain; charset=utf-8",
        arguments={"start_line": 2, "max_lines": 1},
    )
    assert result.metadata["line_count"] == 3
    rendered = result.text.splitlines()[0]
    assert rendered.startswith("2: ")
    assert len(rendered) == len("2: ") + MAX_VIRTUAL_LINE_CHARACTERS
    assert result.metadata["end_line"] == 2


def test_start_line_past_end_is_bounded_error() -> None:
    result, _ = fetch(
        b"a\nb\nc", content_type="text/plain; charset=utf-8", arguments={"start_line": 10}
    )
    assert result.error_code == "web_fetch_failed:web_line_out_of_range"
    assert result.is_error is True


# ---------------------------------------------------------------------------
# 错误映射:denied / invalid_redirect / too_many_redirects / invalid_url / timeout / http_error
# ---------------------------------------------------------------------------


def test_authorize_denied_maps_denied_without_connecting() -> None:
    def deny(url: str) -> None:
        raise PublicWebError("denied", f"approval required for {url}")

    result, backend = fetch(b"<p>never</p>", authorize=deny)
    assert result.status is ResultStatus.ERROR
    assert result.error_code == "denied"
    assert "approval required" in result.error_message
    assert backend.connects == []


def test_ask_authorizer_from_t11_denies_fetch_without_connecting() -> None:
    # T11 fixture 语义:初始 URL 的 ask 由 decide 暂停审批;若请求已在客户端内部
    # 到达逐跳检查点(例如重定向/子资源),ask 是**拒绝本次请求**而非静默放行。
    from praxis.composition import make_web_authorizer

    ctx = PermissionContext(workspace="/tmp", policy=PermissionPolicy(), mode="on-request")
    tool, backend = make_tool(
        [FakeStream(response_chunks(body=b"never"))],
        authorize=make_web_authorizer(lambda: ctx),
    )
    result = run_tool(tool, {})
    assert result.error_code == "denied"
    assert "approval required for example.com" in result.error_message
    assert backend.connects == []


def test_malformed_redirect_location_maps_invalid_redirect() -> None:
    tool, backend = make_tool(
        [
            FakeStream(response_chunks("302 Found", headers=[("Location", "https://[::1")])),
            FakeStream(response_chunks(body=b"never")),
        ]
    )
    result = run_tool(tool, {})
    assert result.error_code == "web_fetch_failed:invalid_redirect"
    assert len(backend.connects) == 1


def test_too_many_redirects_maps() -> None:
    streams = [
        FakeStream(response_chunks("302 Found", headers=[("Location", f"https://h{i}.example/")]))
        for i in range(6)
    ]
    tool, _ = make_tool(streams)
    result = run_tool(tool, {})
    assert result.error_code == "web_fetch_failed:too_many_redirects"


def test_redirect_to_private_url_maps_invalid_url() -> None:
    tool, backend = make_tool(
        [
            FakeStream(response_chunks("302 Found", headers=[("Location", "https://127.0.0.1/")])),
            FakeStream(response_chunks(body=b"never")),
        ]
    )
    result = run_tool(tool, {})
    assert result.error_code == "web_fetch_failed:invalid_url"
    assert len(backend.connects) == 1


def test_timeout_maps_web_fetch_failed_timeout() -> None:
    # 确定性 timeout:流在读取时抛 TimeoutError(与真实 deadline 同映射路径),
    # 不依赖墙钟 sleep。
    stream = FakeStream(response_chunks(body=b"never"), read_error=TimeoutError("deadline"))
    backend = FakeBackend([stream])
    client, _ = make_client(
        resolver=FakeResolver([PUBLIC_IP]),
        backend=backend,
        limits=WebLimits(timeout_seconds=20),
    )
    (tool,) = create_web_fetch_tools(client, limits=WebLimits())
    result = run_tool(tool, {})
    assert result.error_code == "web_fetch_failed:timeout"


def test_non_2xx_maps_legacy_http_error() -> None:
    tool, _ = make_tool([page(b"missing", status="404 Not Found")])
    result = run_tool(tool, {})
    assert result.error_code == "web_fetch_failed:http_error"
    assert "404" in result.error_message
    assert "missing" not in result.text


# ---------------------------------------------------------------------------
# 参数转发与宿主上限钳制(用 record client 钉住逐请求 timeout)
# ---------------------------------------------------------------------------


class RecordingClient:
    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: object = None,
        json_body: object = None,
        max_bytes: int | None = None,
        timeout_seconds: float | None = None,
    ) -> WebFetchResponse:
        self.timeouts.append(timeout_seconds)
        return WebFetchResponse(
            status=200, headers={}, body=b"hello", final_url=url, content_type="text/plain"
        )


def test_timeout_arg_forwarded_and_clamped_by_limits_and_tool_budget() -> None:
    # 宿主上限高于工具级 75s:120s 请求必须夹到 75s(不是 120,也不是宿主上限)。
    client = RecordingClient()
    (tool,) = create_web_fetch_tools(client, limits=WebLimits(timeout_seconds=120))  # type: ignore[arg-type]
    assert run_tool(tool, {}).is_error is False
    assert run_tool(tool, {"timeout_seconds": 120}).is_error is False
    assert run_tool(tool, {"timeout_seconds": 5}).is_error is False
    assert client.timeouts == [45.0, 75.0, 5.0]

    # 宿主上限低于 75s:宿主硬上限胜出(模型参数不能抬高 limits)。
    tighter = RecordingClient()
    (clamped,) = create_web_fetch_tools(tighter, limits=WebLimits(timeout_seconds=30))  # type: ignore[arg-type]
    run_tool(clamped, {"timeout_seconds": 120})
    assert tighter.timeouts == [30.0]


def test_excerpt_budget_clamped_by_host_limits() -> None:
    body = ("y" * 199 + "\n").encode() * 100
    result, _ = fetch(
        body,
        content_type="text/plain; charset=utf-8",
        arguments={"max_bytes": 16_000},
        tool_limits=WebLimits(max_bytes=4_096),
    )
    assert len(result.text.encode("utf-8")) <= 4_096
    assert result.metadata["truncated"] is True


# ---------------------------------------------------------------------------
# 真管道 execute:工具表注入(happy / denied / invalid_input / artifact 外部化)
# ---------------------------------------------------------------------------


def make_exec_ctx(
    store: Store,
    workspace: Path,
    tool: Tool,
    *,
    artifacts: ArtifactStore | None = None,
) -> ExecCtx:
    policy = PermissionPolicy(
        tools=(ToolRule(pattern=WEB_FETCH_TOOL_NAME, decision="allow"),),
        network=NetworkPolicy(decision="allow"),
    )
    turn = create_turn(store, turn_id="turn-1", thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools={WEB_FETCH_TOOL_NAME: tool},
        permissions=PermissionContext(
            workspace=str(workspace), policy=policy, mode="on-request"
        ),
        turn_id=turn.turn_id,
        thread_id="thread-1",
        policy_revision=policy.revision,
        artifacts=artifacts,
    )


def test_pipeline_happy_path_executes_tool(store: Store, workspace: Path) -> None:
    tool, _ = make_tool([page(b"<p>pipeline hello</p>")])
    ctx = make_exec_ctx(store, workspace, tool)
    result = asyncio.run(execute(ToolCall("c1", WEB_FETCH_TOOL_NAME, {"url": URL}), ctx))
    assert result.status is ResultStatus.OK
    assert result.is_error is False
    assert "pipeline hello" in result.text
    assert result.metadata["final_url"] == URL
    kinds = [item.kind for item in store.read_turn_items(ctx.turn_id)]
    assert kinds == [ItemKind.TOOL_CALL, ItemKind.TOOL_RESULT]


def test_pipeline_denied_path_executes_authorizer_denial(store: Store, workspace: Path) -> None:
    def deny(url: str) -> None:
        raise PublicWebError("denied", "approval required for example.com")

    tool, backend = make_tool(
        [FakeStream(response_chunks(body=b"never"))], authorize=deny
    )
    ctx = make_exec_ctx(store, workspace, tool)
    result = asyncio.run(execute(ToolCall("c1", WEB_FETCH_TOOL_NAME, {"url": URL}), ctx))
    assert result.status is ResultStatus.ERROR
    assert result.error_code == "denied"
    assert backend.connects == []
    kinds = [item.kind for item in store.read_turn_items(ctx.turn_id)]
    assert kinds == [ItemKind.TOOL_CALL, ItemKind.TOOL_RESULT]


def test_pipeline_invalid_url_is_invalid_input_before_decide(store: Store, workspace: Path) -> None:
    tool, backend = make_tool([FakeStream(response_chunks(body=b"never"))])
    ctx = make_exec_ctx(store, workspace, tool)
    result = asyncio.run(
        execute(ToolCall("c1", WEB_FETCH_TOOL_NAME, {"url": "https://127.0.0.1/"}), ctx)
    )
    assert result.is_error is True
    assert result.status is ResultStatus.ERROR
    assert result.error_code == "invalid_input"
    assert (
        "a public HTTP(S) URL without credentials on port 80 or 443 is required"
        in result.text
    )
    assert backend.connects == []
    # 校验失败不形成 tool_call item(未 dispatch)。
    assert [item.kind for item in store.read_turn_items(ctx.turn_id)] == [ItemKind.TOOL_RESULT]


def test_pipeline_externalizes_oversized_output_to_artifact(
    store: Store, workspace: Path, tmp_path: Path
) -> None:
    body = "".join(f"<p>{'line ' * 20}{index}</p>" for index in range(120)).encode()
    tool, _ = make_tool([page(body)])
    # 65_536 的真实上限在 schema(max_bytes≤16000)下不可达;用同一 Tool 的缩小副本
    # 证明管道外部化路径对它同样生效(没有任何工具侧第二套截断)。
    small = replace(tool, max_model_output_bytes=1_024)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    ctx = make_exec_ctx(store, workspace, small, artifacts=artifacts)
    result = asyncio.run(execute(ToolCall("c1", WEB_FETCH_TOOL_NAME, {"url": URL}), ctx))
    assert result.status is ResultStatus.OK
    assert result.truncated is True
    assert "artifact://" in result.text
    digest = result.metadata["artifact_digest"]
    assert isinstance(digest, str)
    full = artifacts.get(digest)
    assert full is not None
    assert full.decode("utf-8").startswith("1: line line")
    assert result.metadata["artifact_bytes"] == len(full)
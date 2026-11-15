"""Task 10:DNS 固定 HTTP 客户端——URL 预检 / 限额 / 连接期钉死 / 逐跳授权(无网确定性)。

 brief Step 1 清单逐条:
 ① redirect 到 127.0.0.1 ⇒ 第二跳 authorize 被调**且** validate 拒;
 ② DNS 返回 [93.184.216.34, 10.0.0.1] ⇒ nonpublic_address 且从不连接私有地址;
 ③ github.com.evil.com 对 *.github.com 不匹配(protocol/policy._domain_matches 语义核对);
 ④ TLS 首候选失败未发字节 ⇒ 回退第二候选;已发字节 ⇒ 不回退直接 tls_failed;
 ⑤ gzip bomb:压缩体小、解码体 >max_bytes ⇒ max_bytes_exceeded(流中途);
 ⑥ 跨跳 Authorization/Cookie 剥离、Location 4097 拒绝、6 跳 ⇒ too_many_redirects;
 ⑦ IPv4-mapped ::ffff:127.0.0.1 拒;⑧ 端口 8443 拒;
 ⑨ trusted_proxy 旁路本地 DNS 固定,validate_proxy_url 拒带凭证代理。

全部经假 resolver + 假 httpcore backend 注入,绝不触网。
"""

from __future__ import annotations

import asyncio
import gzip
import json
import socket
import ssl
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

import httpcore
import pytest

from praxis.adapters.web.client import (
    CLIENT_ERROR_CODES,
    PublicHttpClient,
    PublicWebError,
    WebFetchResponse,
    _ConnectionCandidates,
    validate_proxy_url,
    web_proxy_configuration,
)
from praxis.adapters.web.limits import WebLimits, load_web_limits
from praxis.adapters.web.urls import _is_public_address, validate_public_url
from praxis.protocol.policy import _domain_matches

# ---------------------------------------------------------------------------
# 确定性测试双工:假流 / 假 backend / 假 resolver
# ---------------------------------------------------------------------------


class FakeStream(httpcore.AsyncNetworkStream):
    """脚本化 HTTP/1.1 响应流;记录 write/read/TLS/close 供机制断言。"""

    def __init__(
        self,
        chunks: Sequence[bytes] = (),
        *,
        tls_error: Exception | None = None,
        read_error: Exception | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.tls_error = tls_error
        self.read_error = read_error
        self.writes: list[bytes] = []
        self.reads = 0
        self.tls_started = 0
        self.sni: str | None = None
        self.closed = False

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.reads += 1
        if self.read_error is not None:
            raise self.read_error
        if not self.chunks:
            return b""
        return self.chunks.pop(0)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.writes.append(bytes(buffer))

    async def aclose(self) -> None:
        self.closed = True

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.tls_started += 1
        self.sni = server_hostname
        if self.tls_error is not None:
            raise self.tls_error
        return self

    def get_extra_info(self, info: str) -> object:
        return None


class FakeBackend(httpcore.AsyncNetworkBackend):
    """按脚本逐次弹出流或异常;记录每次 connect_tcp 的 (host, port, timeout)。"""

    def __init__(
        self,
        script: Sequence[FakeStream | Exception] = (),
        *,
        delays: Sequence[float] = (),
        clock: dict[str, float] | None = None,
        burn: float = 0.0,
    ) -> None:
        self.script: list[FakeStream | Exception] = list(script)
        self.delays = list(delays)
        self.clock = clock
        self.burn = burn
        self.connects: list[tuple[str, int, float | None]] = []
        self.issued: list[FakeStream] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> httpcore.AsyncNetworkStream:
        self.connects.append((host, port, timeout))
        if self.clock is not None:
            self.clock["now"] += self.burn
        delay = self.delays.pop(0) if self.delays else 0.0
        if delay:
            await asyncio.sleep(delay)
        if not self.script:
            raise AssertionError("FakeBackend script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        self.issued.append(item)
        return item

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: object = None
    ) -> httpcore.AsyncNetworkStream:
        raise AssertionError("unix socket must not be used")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeResolver:
    """getaddrinfo 注入点:固定答案列表或异常;记录调用。"""

    def __init__(self, answers: Sequence[str] | Exception) -> None:
        self.answers = answers
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append((host, port))
        if isinstance(self.answers, Exception):
            raise self.answers
        return list(self.answers)


def response_chunks(
    status: str = "200 OK",
    *,
    headers: Sequence[tuple[str, str]] = (),
    body: bytes = b"",
    chunk_size: int = 0,
) -> list[bytes]:
    """构造脚本化 HTTP/1.1 响应字节块(Content-Length 定界)。"""
    head = (
        f"HTTP/1.1 {status}\r\n"
        + "".join(f"{name}: {value}\r\n" for name, value in headers)
        + f"Content-Length: {len(body)}\r\n\r\n"
    ).encode()
    if not body:
        return [head]
    if chunk_size > 0:
        return [head] + [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]
    return [head, body]


def make_client(
    *,
    resolver: FakeResolver | None = None,
    backend: FakeBackend | None = None,
    limits: WebLimits | None = None,
    environ: dict[str, str] | None = None,
    proxy_url: str | None = None,
    no_proxy: str | None = None,
    authorize: object = None,
) -> tuple[PublicHttpClient, list[str]]:
    calls: list[str] = []

    def recording_authorize(url: str) -> None:
        calls.append(url)

    client = PublicHttpClient(
        authorize=authorize if authorize is not None else recording_authorize,  # type: ignore[arg-type]
        limits=limits if limits is not None else WebLimits(),
        resolver=resolver,
        backend=backend,
        environ=environ if environ is not None else {},
        proxy_url=proxy_url,
        no_proxy=no_proxy,
    )
    return client, calls


PUBLIC_IP = "93.184.216.34"
PUBLIC_IP2 = "93.184.216.35"


# ---------------------------------------------------------------------------
# urls.py:URL 预检(不做 DNS;⑦⑧ 与 S6 主机名纪律)
# ---------------------------------------------------------------------------


def test_validate_public_url_accepts_and_normalizes() -> None:
    assert validate_public_url("https://example.com") == ("example.com", 443, "/")
    assert validate_public_url("http://example.com") == ("example.com", 80, "/")
    assert validate_public_url("http://example.com:80/a/b?x=1#frag") == ("example.com", 80, "/a/b?x=1")
    assert validate_public_url("https://example.com:443/") == ("example.com", 443, "/")
    assert validate_public_url("http://example.com:443/") == ("example.com", 443, "/")
    assert validate_public_url("HTTPS://EXAMPLE.COM./Path") == ("example.com", 443, "/Path")
    assert validate_public_url(f"https://{PUBLIC_IP}/x") == (PUBLIC_IP, 443, "/x")
    assert validate_public_url("https://[2606:4700::1111]/") == ("2606:4700::1111", 443, "/")


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://example.com/",
        "file:///etc/passwd",
        "https://user@example.com/",
        "https://user:pass@example.com/",
        "https://@example.com/",
        "https://example.com:8443/",  # ⑧ 端口 8443 拒
        "https://example.com:99999/",
        "https://example.com:abc/",
        "https://[fe80::1%25eth0]/",
        "https://localhost/",
        "https://LOCALHOST/",
        "https://foo.localhost/",
        "https://127.0.0.1/",
        "https://10.0.0.1/",
        "https://169.254.169.254/",
        "https://[::1]/",
        "https://[::ffff:127.0.0.1]/",  # ⑦ IPv4-mapped 私有拒
        "https://224.0.0.1/",
        "https://240.0.0.1/",
        "https://example.com/\n",
        "https://exa mple.com/",
        "https://example.com/\x7f",
        "https:///path",
    ],
)
def test_validate_public_url_rejects(url: str) -> None:
    with pytest.raises(PublicWebError) as excinfo:
        validate_public_url(url)
    assert excinfo.value.code == "invalid_url"


def test_ipv4_mapped_public_unwrapped() -> None:
    # IPv4-mapped 先还原再判:映射到公网 IP 放行,映射到私有拒(⑦ 的另半边)。
    assert _is_public_address("::ffff:93.184.216.34") is True
    assert _is_public_address("::ffff:127.0.0.1") is False
    assert _is_public_address("::ffff:10.0.0.1") is False
    assert validate_public_url("https://[::ffff:93.184.216.34]/") == ("::ffff:93.184.216.34", 443, "/")


def test_is_public_address_non_literal() -> None:
    assert _is_public_address("example.com") is False
    assert _is_public_address("") is False


def test_hostname_discipline_exact_not_substring() -> None:
    # ③ S6 同族纪律:域名规则是精确/标签边界匹配,任何子串匹配禁止。
    assert _domain_matches("api.github.com", "*.github.com") is True
    assert _domain_matches("github.com.evil.com", "*.github.com") is False
    assert _domain_matches("github.com", "*.github.com") is False
    assert _domain_matches("github.com", "github.com") is True
    # URL 预检的主机名裁决同样是精确/域名后缀,不是子串:
    assert validate_public_url("https://github.com.evil.com/")[0] == "github.com.evil.com"
    assert validate_public_url("https://localhost.evil.com/")[0] == "localhost.evil.com"
    assert validate_public_url("https://evillocalhost/")[0] == "evillocalhost"
    with pytest.raises(PublicWebError):
        validate_public_url("https://x.localhost/")


def test_validate_public_url_does_no_dns() -> None:
    # 预检对不存在的主机名也放行(裁决在连接期);这里没有任何 resolver 可调用。
    host, port, target = validate_public_url("https://nonexistent.invalid./a?b=1")
    assert (host, port, target) == ("nonexistent.invalid", 443, "/a?b=1")


# ---------------------------------------------------------------------------
# limits.py:范围校验 / 配置加载 / Q7 有意差异
# ---------------------------------------------------------------------------


def test_web_limits_defaults_and_fields() -> None:
    limits = WebLimits()
    assert (limits.max_bytes, limits.timeout_seconds, limits.max_redirects) == (2_000_000, 20.0, 5)
    # Q7 有意差异显式断言:只有逐请求硬上限,没有 turn 级账本字段。
    assert tuple(f.name for f in fields(WebLimits)) == ("max_bytes", "timeout_seconds", "max_redirects")
    assert not hasattr(limits, "turn_requests")
    assert not hasattr(limits, "browser_requests")


def test_web_limits_range_validation() -> None:
    assert WebLimits(max_bytes=4096).max_bytes == 4096
    assert WebLimits(max_bytes=32_000_000).max_bytes == 32_000_000
    assert WebLimits(timeout_seconds=300).timeout_seconds == 300
    assert WebLimits(max_redirects=0).max_redirects == 0
    assert WebLimits(max_redirects=10).max_redirects == 10
    for kwargs in (
        {"max_bytes": 4095},
        {"max_bytes": 32_000_001},
        {"max_bytes": 0},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1},
        {"timeout_seconds": 301},
        {"max_redirects": -1},
        {"max_redirects": 11},
    ):
        with pytest.raises(ValueError):
            WebLimits(**kwargs)  # type: ignore[arg-type]


def test_load_web_limits_defaults_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRAXIS_WEB_LIMITS", raising=False)
    assert load_web_limits() == WebLimits()


def test_load_web_limits_from_host_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRAXIS_WEB_LIMITS", raising=False)
    config = tmp_path / "web.json"
    config.write_text(json.dumps({"limits": {"max_bytes": 8192, "max_redirects": 2}}), encoding="utf-8")
    limits = load_web_limits(config)
    assert limits == WebLimits(max_bytes=8192, max_redirects=2)
    # "limits" 块可选:缺省即默认(其余键留给 T13 搜索配置)。
    config.write_text(json.dumps({"provider": "brave"}), encoding="utf-8")
    assert load_web_limits(config) == WebLimits()


def test_load_web_limits_default_file_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRAXIS_WEB_LIMITS", raising=False)
    from praxis.protocol.floor import host_config_dir

    config = host_config_dir() / "web.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"limits": {"timeout_seconds": 7.5}}), encoding="utf-8")
    assert load_web_limits() == WebLimits(timeout_seconds=7.5)


def test_load_web_limits_malformed_names_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRAXIS_WEB_LIMITS", raising=False)
    config = tmp_path / "web.json"
    config.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match=str(config)):
        load_web_limits(config)
    config.write_text(json.dumps({"limits": {"max_bytes": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match=str(config)):
        load_web_limits(config)
    config.write_text(json.dumps({"limits": {"turn_requests": 32}}), encoding="utf-8")
    with pytest.raises(ValueError, match=str(config)):
        load_web_limits(config)


def test_load_web_limits_env_is_file_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 旧语义保留:PRAXIS_WEB_LIMITS 的值是 JSON 文件路径,不是内联 JSON。
    env_file = tmp_path / "limits.json"
    env_file.write_text(json.dumps({"max_bytes": 16384}), encoding="utf-8")
    monkeypatch.setenv("PRAXIS_WEB_LIMITS", str(env_file))
    assert load_web_limits() == WebLimits(max_bytes=16384)
    # env 覆盖显式 path 与宿主文件。
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"limits": {"max_bytes": 4096}}), encoding="utf-8")
    assert load_web_limits(other) == WebLimits(max_bytes=16384)


def test_load_web_limits_env_file_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "limits.json"
    env_file.write_bytes(b"x" * 4097)
    monkeypatch.setenv("PRAXIS_WEB_LIMITS", str(env_file))
    with pytest.raises(ValueError, match=str(env_file)):
        load_web_limits()
    env_file.write_text(json.dumps({"max_bytes": 8192}), encoding="utf-8")
    assert load_web_limits() == WebLimits(max_bytes=8192)
    monkeypatch.setenv("PRAXIS_WEB_LIMITS", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="missing.json"):
        load_web_limits()


def test_no_turn_budget_ledger_q7() -> None:
    # Q7 有意差异:turn/会话计数归阶段 4 预算树;旧账本机制在 3B 不存在。
    # 钉"代码里不存在"(docstring 允许点名旧机制做对照说明)。
    import ast
    import inspect

    import praxis.adapters.web.limits as limits_module

    tree = ast.parse(inspect.getsource(limits_module))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    code_strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]
    assert "web_request_budget_exceeded" not in code_strings
    assert "reserve_resource_unit" not in code_strings
    assert not hasattr(limits_module, "reserve_resource_unit")
    assert not any(
        isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "sqlite3" for alias in node.names)
        for node in ast.walk(tree)
    )


# ---------------------------------------------------------------------------
# 连接期 DNS 钉死(②;数字直连;去重 ≤8;全部答案判公网)
# ---------------------------------------------------------------------------


def test_dns_mixed_answers_rejected_before_any_connect() -> None:
    # ② DNS 返回 [公网, 私有] ⇒ nonpublic_address,且从不连接任何地址。
    resolver = FakeResolver([PUBLIC_IP, "10.0.0.1"])
    backend = FakeBackend([FakeStream(response_chunks(body=b"ok"))])
    client, _ = make_client(resolver=resolver, backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "nonpublic_address"
    assert backend.connects == []
    assert resolver.calls == [("example.com", 443)]


@pytest.mark.parametrize(
    "answers",
    [
        ["10.0.0.1"],
        ["127.0.0.1", PUBLIC_IP],
        ["::ffff:127.0.0.1"],
        ["169.254.169.254"],
        [],
    ],
)
def test_dns_nonpublic_answers_rejected(answers: list[str]) -> None:
    backend = FakeBackend([FakeStream(response_chunks(body=b"ok"))])
    client, _ = make_client(resolver=FakeResolver(answers), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "nonpublic_address"
    assert backend.connects == []


def test_numeric_url_skips_resolver_and_connects_checked_ip() -> None:
    # 只连已校验数字地址:字面 IP 不经 resolver,backend 收到的就是该数字地址。
    resolver = FakeResolver(["10.0.0.1"])
    stream = FakeStream(response_chunks(body=b"ok"))
    backend = FakeBackend([stream])
    client, _ = make_client(resolver=resolver, backend=backend)
    result = asyncio.run(client.fetch(f"https://{PUBLIC_IP}/path"))
    assert result.body == b"ok"
    assert resolver.calls == []
    assert [(host, port) for host, port, _ in backend.connects] == [(PUBLIC_IP, 443)]
    assert stream.sni == PUBLIC_IP


def test_hostname_connects_only_numeric_answers_with_origin_sni() -> None:
    stream = FakeStream(response_chunks(body=b"ok"))
    backend = FakeBackend([stream])
    resolver = FakeResolver([PUBLIC_IP])
    client, _ = make_client(resolver=resolver, backend=backend)
    asyncio.run(client.fetch("https://example.com/"))
    assert [(host, port) for host, port, _ in backend.connects] == [(PUBLIC_IP, 443)]
    # 原始主机名留给 TLS SNI/Host,不被数字地址替换。
    assert stream.sni == "example.com"
    assert b"host: example.com" in b"".join(stream.writes).lower()


def test_candidates_dedup_order_and_cap() -> None:
    # 去重、保序、上限 8:12 个答案(含重复)只尝试前 8 个唯一地址。
    answers = [f"93.184.216.{i}" for i in range(1, 11)] + [PUBLIC_IP, PUBLIC_IP2]
    backend = FakeBackend([httpcore.ConnectError("refused")] * 12)
    client, _ = make_client(resolver=FakeResolver(answers), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "connect_failed"
    expected = list(dict.fromkeys(answers))[:8]
    assert [host for host, _, _ in backend.connects] == expected


def test_resolver_failure_maps_connect_failed() -> None:
    backend = FakeBackend([FakeStream(response_chunks(body=b"ok"))])
    client, _ = make_client(resolver=FakeResolver(socket.gaierror("name resolution failed")), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "connect_failed"
    assert backend.connects == []


def test_private_literal_url_rejected_before_network() -> None:
    backend = FakeBackend([FakeStream(response_chunks(body=b"ok"))])
    client, calls = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://127.0.0.1/"))
    assert excinfo.value.code == "invalid_url"
    assert backend.connects == []
    assert calls == ["https://127.0.0.1/"]


# ---------------------------------------------------------------------------
# 共享 deadline 与每候选预算 min(3.0, remaining/剩余数)
# ---------------------------------------------------------------------------


def test_candidate_budget_formula_with_fake_clock() -> None:
    # 确定性时钟:预算公式与共享 deadline 耗尽,不依赖真实时间。
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        clock = {"now": 100.0}
        original_time = loop.time
        loop.time = lambda: clock["now"]  # type: ignore[method-assign]
        try:
            backend = FakeBackend(
                [httpcore.ConnectError("refused")] * 3, clock=clock, burn=5.0
            )
            candidates = _ConnectionCandidates(
                backend, [PUBLIC_IP, PUBLIC_IP2, "93.184.216.36"], 443, 10.0, None, None
            )
            with pytest.raises(PublicWebError) as excinfo:
                await candidates.connect()
            assert excinfo.value.code == "timeout"
            # 预算:min(3, 10/3)=3 → 烧 5 → min(3, 5/2)=2.5 → 烧 5 → 耗尽。
            assert [timeout for _, _, timeout in backend.connects] == [3.0, 2.5]
        finally:
            loop.time = original_time  # type: ignore[method-assign]

    asyncio.run(scenario())


def test_candidate_budget_cap_and_dedup_unit() -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        clock = {"now": 0.0}
        original_time = loop.time
        loop.time = lambda: clock["now"]  # type: ignore[method-assign]
        try:
            stream = FakeStream(response_chunks(body=b"ok"))
            backend = FakeBackend([stream], clock=clock)
            candidates = _ConnectionCandidates(
                backend, [PUBLIC_IP, PUBLIC_IP, PUBLIC_IP2], 443, 100.0, None, None
            )
            assert candidates.addresses == [PUBLIC_IP, PUBLIC_IP2]
            await candidates.connect()
            # 大预算下每候选上限 3.0。
            assert backend.connects[0][2] == 3.0
        finally:
            loop.time = original_time  # type: ignore[method-assign]

    asyncio.run(scenario())


def test_first_candidate_burns_budget_second_gets_remainder() -> None:
    # 真实小时钟:首候选烧掉大部分预算(被每候选上限截断),第二候选共享剩余。
    # 注意:delay 在脚本弹出之前,被取消的首尝试不消费脚本项。
    stream = FakeStream(response_chunks(body=b"ok"))
    backend = FakeBackend(
        [stream, FakeStream(response_chunks(body=b"never"))], delays=[0.45, 0.0]
    )
    client, _ = make_client(
        resolver=FakeResolver([PUBLIC_IP, PUBLIC_IP2]),
        backend=backend,
        limits=WebLimits(timeout_seconds=0.6),
    )
    result = asyncio.run(client.fetch("https://example.com/"))
    assert result.body == b"ok"
    assert len(backend.connects) == 2
    first_budget = backend.connects[0][2]
    second_budget = backend.connects[1][2]
    assert first_budget is not None and second_budget is not None
    assert first_budget <= 0.31  # min(3.0, 0.6/2)
    assert second_budget <= 0.31  # 剩余 ~0.3 全给最后一个候选
    assert second_budget < 0.6


# ---------------------------------------------------------------------------
# TLS 回退规则(④):仅未发 HTTP 字节前回退
# ---------------------------------------------------------------------------


def test_tls_fallback_before_bytes_sent() -> None:
    # ④ 前半:首候选 TLS 失败、未发字节 ⇒ 回退第二候选成功。
    first = FakeStream(response_chunks(body=b"never"), tls_error=httpcore.ConnectError("tls reset"))
    second = FakeStream(response_chunks(body=b"ok"))
    backend = FakeBackend([first, second])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP, PUBLIC_IP2]), backend=backend)
    result = asyncio.run(client.fetch("https://example.com/"))
    assert result.body == b"ok"
    assert [host for host, _, _ in backend.connects] == [PUBLIC_IP, PUBLIC_IP2]
    assert first.tls_started == 1
    assert first.writes == []  # 未发任何 HTTP 字节
    assert first.closed
    assert second.sni == "example.com"


def test_tls_no_fallback_after_bytes_sent() -> None:
    # ④ 后半:TLS 已成功、请求字节已发,其后 TLS 层失败 ⇒ 不回退,直接 tls_failed。
    stream = FakeStream(response_chunks(body=b"ok"), read_error=ssl.SSLError("mid-stream tls failure"))
    backend = FakeBackend([stream, FakeStream(response_chunks(body=b"never"))])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP, PUBLIC_IP2]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "tls_failed"
    assert len(backend.connects) == 1  # 绝不回退第二候选
    assert stream.writes != []  # 请求字节确已发出


def test_tls_fallback_exhausted_maps_tls_failed() -> None:
    backend = FakeBackend(
        [
            FakeStream(response_chunks(body=b"x"), tls_error=httpcore.ConnectError("reset")),
            FakeStream(response_chunks(body=b"x"), tls_error=httpcore.ConnectError("reset")),
        ]
    )
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP, PUBLIC_IP2]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "tls_failed"
    assert len(backend.connects) == 2


def test_tls_nonrecoverable_error_no_fallback() -> None:
    stream = FakeStream(response_chunks(body=b"x"), tls_error=RuntimeError("certificate policy"))
    backend = FakeBackend([stream, FakeStream(response_chunks(body=b"never"))])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP, PUBLIC_IP2]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "tls_failed"
    assert len(backend.connects) == 1


# ---------------------------------------------------------------------------
# 重定向自处理(①⑥):逐跳 authorize、凭证剥离、Location 上限、跳数上限
# ---------------------------------------------------------------------------


def test_redirect_to_private_authorizes_then_rejects() -> None:
    # ① redirect 到 127.0.0.1:第二跳 authorize 被调,且 validate 拒(invalid_url)。
    hop1 = FakeStream(response_chunks("302 Found", headers=[("Location", "https://127.0.0.1/")]))
    backend = FakeBackend([hop1, FakeStream(response_chunks(body=b"never"))])
    client, calls = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://a.example/"))
    assert excinfo.value.code == "invalid_url"
    assert calls == ["https://a.example/", "https://127.0.0.1/"]
    assert len(backend.connects) == 1  # 第二跳从未连接


def test_redirect_chain_authorize_sequence_and_header_stripping() -> None:
    # ⑥ 跨跳剥凭证:只留 accept/accept-language/accept-encoding/user-agent。
    hop1 = FakeStream(response_chunks("302 Found", headers=[("Location", "https://b.example/landing")]))
    hop2 = FakeStream(response_chunks(body=b"final"))
    backend = FakeBackend([hop1, hop2])
    client, calls = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    result = asyncio.run(
        client.fetch(
            "https://a.example/",
            headers={
                "Authorization": "Bearer secret-token",
                "Cookie": "session=secret",
                "X-Custom": "leak",
                "Accept": "text/html",
                "Accept-Language": "zh-CN",
                "User-Agent": "praxis-test",
            },
        )
    )
    assert result.body == b"final"
    assert result.final_url == "https://b.example/landing"
    assert calls == ["https://a.example/", "https://b.example/landing"]
    second_request = b"".join(hop2.writes).lower()
    assert b"authorization" not in second_request
    assert b"cookie" not in second_request
    assert b"x-custom" not in second_request
    assert b"accept: text/html" in second_request
    # httpx.Headers 在线路上把值小写化(legacy 同一行为)。
    assert b"accept-language: zh-cn" in second_request
    assert b"user-agent: praxis-test" in second_request
    assert b"accept-encoding: gzip, identity" in second_request
    # 首跳保留原始凭证头。
    first_request = b"".join(hop1.writes).lower()
    assert b"authorization: bearer secret-token" in first_request


def test_redirect_relative_location_joined() -> None:
    hop1 = FakeStream(response_chunks("302 Found", headers=[("Location", "/next?q=1")]))
    hop2 = FakeStream(response_chunks(body=b"ok"))
    backend = FakeBackend([hop1, hop2])
    client, calls = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    result = asyncio.run(client.fetch("https://a.example/first"))
    assert result.final_url == "https://a.example/next?q=1"
    assert calls == ["https://a.example/first", "https://a.example/next?q=1"]


def test_redirect_location_4097_rejected() -> None:
    # ⑥ Location 4097 字符拒绝(4096 上限)。
    long_location = "https://b.example/" + "a" * (4097 - len("https://b.example/"))
    assert len(long_location) == 4097
    hop1 = FakeStream(response_chunks("302 Found", headers=[("Location", long_location)]))
    backend = FakeBackend([hop1])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://a.example/"))
    assert excinfo.value.code == "invalid_redirect"
    assert len(backend.connects) == 1


def test_redirect_missing_location_invalid() -> None:
    hop1 = FakeStream(response_chunks("302 Found"))
    backend = FakeBackend([hop1])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://a.example/"))
    assert excinfo.value.code == "invalid_redirect"


def test_six_hops_too_many_redirects() -> None:
    # ⑥ 6 跳 302(max_redirects=5)⇒ too_many_redirects;authorize 每跳各一次。
    script = [
        FakeStream(response_chunks("302 Found", headers=[("Location", f"https://h{i}.example/")]))
        for i in range(6)
    ]
    backend = FakeBackend(script)
    client, calls = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://h0.example/"))
    assert excinfo.value.code == "too_many_redirects"
    assert len(calls) == 6
    assert len(backend.connects) == 6


def test_five_hops_within_limit_succeeds() -> None:
    script = [
        FakeStream(response_chunks("302 Found", headers=[("Location", f"https://h{i + 1}.example/")]))
        for i in range(5)
    ] + [FakeStream(response_chunks(body=b"done"))]
    backend = FakeBackend(script)
    client, calls = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    result = asyncio.run(client.fetch("https://h0.example/"))
    assert result.body == b"done"
    assert len(calls) == 6


def test_post_json_never_follows_redirects() -> None:
    hop1 = FakeStream(response_chunks("302 Found", headers=[("Location", "https://b.example/")]))
    backend = FakeBackend([hop1])
    client, calls = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://a.example/api", method="POST", json_body={"k": "v"}))
    assert excinfo.value.code == "invalid_redirect"
    assert calls == ["https://a.example/api"]
    request = b"".join(hop1.writes)
    assert request.startswith(b"POST /api HTTP/1.1")
    assert b"content-type: application/json" in request.lower()
    assert request.endswith(b'{"k": "v"}')


def test_post_json_body_over_limit_rejected_before_network() -> None:
    backend = FakeBackend([FakeStream(response_chunks(body=b"never"))])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(
            client.fetch("https://a.example/", method="POST", json_body={"k": "x" * 5000}, max_bytes=4096)
        )
    assert excinfo.value.code == "max_bytes_exceeded"
    assert backend.connects == []


def test_non_2xx_returned_with_status_not_raised() -> None:
    # 有意差异:旧 http_error 码不在 3B 码集;非 2xx 经 status 字段原样返回。
    stream = FakeStream(response_chunks("404 Not Found", body=b"missing"))
    backend = FakeBackend([stream])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    result = asyncio.run(client.fetch("https://example.com/"))
    assert isinstance(result, WebFetchResponse)
    assert result.status == 404
    assert result.body == b"missing"
    assert result.content_type == ""


def test_response_headers_whitelist_only() -> None:
    stream = FakeStream(
        response_chunks(
            headers=[
                ("Content-Type", "text/html; charset=utf-8"),
                ("Set-Cookie", "secret=1"),
                ("Content-Encoding", "identity"),
                ("Referrer-Policy", "no-referrer"),
                ("Access-Control-Allow-Origin", "*"),
            ],
            body=b"ok",
        )
    )
    backend = FakeBackend([stream])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    result = asyncio.run(client.fetch("https://example.com/"))
    assert result.content_type == "text/html; charset=utf-8"
    assert result.headers == {"referrer-policy": "no-referrer", "access-control-allow-origin": "*"}


# ---------------------------------------------------------------------------
# 字节上限与 gzip(⑤):增量解压、解码后同限、流中途触发
# ---------------------------------------------------------------------------


def test_gzip_bomb_decoded_cap_mid_stream() -> None:
    # ⑤ 压缩体小、解码体远超 max_bytes ⇒ 流中途 max_bytes_exceeded。
    bomb = gzip.compress(b"A" * 200_000)
    assert len(bomb) < 4096  # 压缩体本身小于限额
    chunks = response_chunks(headers=[("Content-Encoding", "gzip")], body=bomb, chunk_size=64)
    stream = FakeStream(chunks)
    backend = FakeBackend([stream])
    client, _ = make_client(
        resolver=FakeResolver([PUBLIC_IP]), backend=backend, limits=WebLimits(max_bytes=4096)
    )
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "max_bytes_exceeded"
    assert stream.chunks  # 未读完全部压缩块:中途触发,不是事后检查
    assert stream.reads < len(chunks)


def test_gzip_within_limit_decodes() -> None:
    payload = gzip.compress(b"hello public web")
    stream = FakeStream(response_chunks(headers=[("Content-Encoding", "gzip")], body=payload))
    backend = FakeBackend([stream])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    result = asyncio.run(client.fetch("https://example.com/"))
    assert result.body == b"hello public web"


def test_identity_wire_cap() -> None:
    stream = FakeStream(response_chunks(body=b"x" * 5000, chunk_size=1000))
    backend = FakeBackend([stream])
    client, _ = make_client(
        resolver=FakeResolver([PUBLIC_IP]), backend=backend, limits=WebLimits(max_bytes=4096)
    )
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "max_bytes_exceeded"


def test_max_bytes_override_per_request() -> None:
    stream = FakeStream(response_chunks(body=b"x" * 5000))
    backend = FakeBackend([stream])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/", max_bytes=4096))
    assert excinfo.value.code == "max_bytes_exceeded"


@pytest.mark.parametrize("encoding", ["br", "deflate", "zstd", "compress"])
def test_unsupported_encoding_rejected(encoding: str) -> None:
    stream = FakeStream(response_chunks(headers=[("Content-Encoding", encoding)], body=b"xx"))
    backend = FakeBackend([stream])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "encoding_unsupported"


def test_malformed_gzip_rejected() -> None:
    stream = FakeStream(response_chunks(headers=[("Content-Encoding", "gzip")], body=b"not gzip at all"))
    backend = FakeBackend([stream])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "encoding_unsupported"


def test_truncated_gzip_rejected() -> None:
    payload = gzip.compress(b"hello" * 100)
    stream = FakeStream(response_chunks(headers=[("Content-Encoding", "gzip")], body=payload[:10]))
    backend = FakeBackend([stream])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "encoding_unsupported"


# ---------------------------------------------------------------------------
# authorize 契约:任意异常原样传播、先于网络、逐跳
# ---------------------------------------------------------------------------


def test_authorize_exception_propagates_unchanged_before_network() -> None:
    boom = OSError("authorizer says no")  # 故意选会被网络映射误捕的类型
    backend = FakeBackend([FakeStream(response_chunks(body=b"never"))])

    def deny(url: str) -> None:
        raise boom

    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend, authorize=deny)
    with pytest.raises(OSError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value is boom  # 同一实例,未被包装/映射
    assert backend.connects == []


def test_authorize_denied_code_propagates() -> None:
    # T11 形状:回调 raise PublicWebError("denied") ⇒ 立即中止,不触网。
    backend = FakeBackend([FakeStream(response_chunks(body=b"never"))])

    def deny(url: str) -> None:
        raise PublicWebError("denied", "approval required at call time")

    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend, authorize=deny)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "denied"
    assert backend.connects == []


def test_authorize_runs_before_validate_each_hop() -> None:
    # 每跳先 authorize 再 validate:非法初始 URL 也先过 authorize。
    seen: list[str] = []

    def record(url: str) -> None:
        seen.append(url)

    backend = FakeBackend([FakeStream(response_chunks(body=b"never"))])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend, authorize=record)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com:8443/"))
    assert excinfo.value.code == "invalid_url"
    assert seen == ["https://example.com:8443/"]
    assert backend.connects == []


def test_authorize_redirect_hop_denial_stops_chain() -> None:
    hop1 = FakeStream(response_chunks("302 Found", headers=[("Location", "https://b.example/")]))
    backend = FakeBackend([hop1, FakeStream(response_chunks(body=b"never"))])
    seen: list[str] = []

    def deny_second(url: str) -> None:
        seen.append(url)
        if url == "https://b.example/":
            raise PublicWebError("denied")

    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP]), backend=backend, authorize=deny_second)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://a.example/"))
    assert excinfo.value.code == "denied"
    assert seen == ["https://a.example/", "https://b.example/"]
    assert len(backend.connects) == 1


# ---------------------------------------------------------------------------
# 超时 / 连接失败 / 错误码集
# ---------------------------------------------------------------------------


def test_connect_timeout_maps_timeout_code() -> None:
    backend = FakeBackend([FakeStream(response_chunks(body=b"never"))], delays=[5.0])
    client, _ = make_client(
        resolver=FakeResolver([PUBLIC_IP]), backend=backend, limits=WebLimits(timeout_seconds=0.2)
    )
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "timeout"


def test_all_candidates_refused_maps_connect_failed() -> None:
    backend = FakeBackend([httpcore.ConnectError("refused"), httpcore.ConnectError("refused")])
    client, _ = make_client(resolver=FakeResolver([PUBLIC_IP, PUBLIC_IP2]), backend=backend)
    with pytest.raises(PublicWebError) as excinfo:
        asyncio.run(client.fetch("https://example.com/"))
    assert excinfo.value.code == "connect_failed"
    assert len(backend.connects) == 2


def test_client_error_codes_match_brief() -> None:
    # 码集逐字钉死(brief §Interfaces);invalid_url 属预检、http_error 有意缺席。
    assert CLIENT_ERROR_CODES == frozenset(
        {
            "nonpublic_address",
            "invalid_redirect",
            "too_many_redirects",
            "max_bytes_exceeded",
            "encoding_unsupported",
            "connect_failed",
            "tls_failed",
            "timeout",
            "denied",
        }
    )
    assert "invalid_url" not in CLIENT_ERROR_CODES
    assert "http_error" not in CLIENT_ERROR_CODES
    assert "web_request_budget_exceeded" not in CLIENT_ERROR_CODES


def test_public_web_error_is_runtime_error_with_code() -> None:
    error = PublicWebError("denied")
    assert isinstance(error, RuntimeError)
    assert error.code == "denied"


# ---------------------------------------------------------------------------
# 代理(⑨):env 优先级 / no_proxy 归一化 / validate_proxy_url / 模式旁路
# ---------------------------------------------------------------------------


def test_proxy_env_precedence_and_direct_sentinel() -> None:
    env = {"PRAXIS_WEB_PROXY": "http://upper.example:8080", "https_proxy": "http://lower.example:8080"}
    proxy, _ = web_proxy_configuration(None, env)
    assert proxy == "http://upper.example:8080"
    proxy, _ = web_proxy_configuration(None, {"https_proxy": "http://lower.example:8080"})
    assert proxy == "http://lower.example:8080"
    proxy, _ = web_proxy_configuration(None, {"HTTPS_PROXY": "http://s.example:8080"})
    assert proxy == "http://s.example:8080"
    proxy, _ = web_proxy_configuration(None, {"http_proxy": "http://h.example:8080"})
    assert proxy == "http://h.example:8080"
    assert web_proxy_configuration(None, {"PRAXIS_WEB_PROXY": "direct"})[0] is None
    assert web_proxy_configuration(None, {"PRAXIS_WEB_PROXY": ""})[0] is None
    assert web_proxy_configuration("http://explicit.example:8080", {})[0] == "http://explicit.example:8080"


def test_no_proxy_normalization() -> None:
    _, no_proxy = web_proxy_configuration(None, {"no_proxy": "a.com, .b.com,,c.com"})
    assert no_proxy == "a.com,.b.com,c.com"
    _, no_proxy = web_proxy_configuration(None, {"NO_PROXY": "x.com"})
    assert no_proxy == "x.com"


@pytest.mark.parametrize(
    "value",
    [
        "http://user:pass@proxy.example:8080",
        "http://proxy.example:8080/path",
        "http://proxy.example:8080/?q=1",
        "http://proxy.example:8080/#frag",
        "socks5://proxy.example:1080",
        "not a url",
        "http://proxy.example:8080/\n",
    ],
)
def test_validate_proxy_url_rejects(value: str) -> None:
    with pytest.raises(ValueError):
        validate_proxy_url(value)


def test_validate_proxy_url_accepts_clean_endpoints() -> None:
    assert validate_proxy_url("http://proxy.example:8080") == "http://proxy.example:8080"
    assert validate_proxy_url("https://proxy.example/") == "https://proxy.example/"


def test_connection_mode_trusted_proxy_and_bypass() -> None:
    # ⑨ trusted_proxy 模式旁路本地 DNS 固定(结构:代理 transport 用普通 backend);
    # no_proxy 命中回落 direct。
    client, _ = make_client(environ={"PRAXIS_WEB_PROXY": "http://proxy.corp:8080"})
    assert client.connection_mode("https://example.com/") == "trusted_proxy"
    assert client.connection_mode("http://example.com/") == "trusted_proxy"
    bypass, _ = make_client(
        environ={"PRAXIS_WEB_PROXY": "http://proxy.corp:8080", "no_proxy": "example.com"}
    )
    assert bypass.connection_mode("https://example.com/") == "direct"
    assert bypass.connection_mode("https://other.com/") == "trusted_proxy"
    direct, _ = make_client(environ={})
    assert direct.connection_mode("https://example.com/") == "direct"


def test_constructor_rejects_credentialed_proxy() -> None:
    with pytest.raises(ValueError):
        make_client(proxy_url="http://user:pass@proxy.corp:8080")


def test_trusted_proxy_bypasses_pinned_backend_structurally() -> None:
    # ⑨ 结构钉:direct transport 用 DNS 固定 backend;proxy transport 用普通 backend
    # (上游代理负责最终 IP 裁决,本地钉死按设计旁路——legacy 语义)。
    from praxis.adapters.web.client import _PinnedNetworkBackend

    client, _ = make_client(environ={"PRAXIS_WEB_PROXY": "http://proxy.corp:8080"})
    assert client._proxy_transport is not None
    assert isinstance(client._transport._pool._network_backend, _PinnedNetworkBackend)
    assert isinstance(client._proxy_transport._pool._network_backend, httpcore.AnyIOBackend)
    direct, _ = make_client(environ={})
    assert direct._proxy_transport is None
    assert isinstance(direct._transport._pool._network_backend, _PinnedNetworkBackend)


def test_ipv6_literal_fetch_brackets_and_numeric_connect() -> None:
    stream = FakeStream(response_chunks(body=b"ok"))
    backend = FakeBackend([stream])
    resolver = FakeResolver(["10.0.0.1"])
    client, _ = make_client(resolver=resolver, backend=backend)
    result = asyncio.run(client.fetch("https://[2606:4700::1111]/x"))
    assert result.body == b"ok"
    assert result.final_url == "https://[2606:4700::1111]/x"
    assert resolver.calls == []
    assert [(host, port) for host, port, _ in backend.connects] == [("2606:4700::1111", 443)]


def test_keepalive_zero_no_dns_cache() -> None:
    # 无 DNS 缓存的结构钉:每请求新连接 ⇒ 同一 fetch 两跳各自解析一次。
    hop1 = FakeStream(response_chunks("302 Found", headers=[("Location", "https://b.example/")]))
    hop2 = FakeStream(response_chunks(body=b"ok"))
    backend = FakeBackend([hop1, hop2])
    resolver = FakeResolver([PUBLIC_IP])
    client, _ = make_client(resolver=resolver, backend=backend)
    asyncio.run(client.fetch("https://a.example/"))
    assert resolver.calls == [("a.example", 443), ("b.example", 443)]
    assert len(backend.connects) == 2
    assert hop1.closed and hop2.closed

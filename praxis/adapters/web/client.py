"""DNS 固定的公共 HTTP 客户端(agent_runtime/tools/web_http.py 好设计逐条移植)。

两段式:URL 预检不做 DNS(urls.py,可在批准前跑);**连接期**才逐 DNS 答案
校验,全部答案公网才放行,且只连已校验的**数字地址**(防第二次未校验 DNS /
rebinding)。无 DNS 缓存:``max_keepalive_connections=0`` ⇒ 每请求新连接、
每跳重解析。候选去重 ≤8、共享 deadline、每候选预算 ``min(3.0, remaining/剩余数)``;
TLS 失败**仅在未发出 HTTP 字节前**回退下一候选(start_tls 成功即丢弃候选引用,
结构上保证字节后无回退路径)。

fetch 路径错误码(本模块自产,``CLIENT_ERROR_CODES`` 逐字钉死):
nonpublic_address / invalid_redirect / too_many_redirects / max_bytes_exceeded /
encoding_unsupported / connect_failed / tls_failed / timeout;``denied`` 由注入的
authorize 回调产生(T11 绑 decide 闭包);``invalid_url`` 由 URL 预检产生
(urls.py,legacy parity)。**有意差异**:旧 http_error 不再抛——非 2xx 经
``WebFetchResponse.status`` 原样返回;旧 network_error 拆为 connect_failed /
tls_failed;旧 request_too_large/response_too_large 统一 max_bytes_exceeded;
旧 unsupported_encoding/invalid_encoding 统一 encoding_unsupported;
旧 redirect_not_allowed 归 invalid_redirect(POST 恒禁重定向);旧
proxy_connection_failed 归 connect_failed(不回显代理 reason phrase);
httpx.InvalidURL 兜底 ⇒ invalid_url(R-T10-1);无 turn 账本(Q7)。

authorize 契约:每跳先于 validate/connect 调 ``authorize(str(target))``,可抛
**任意**异常,客户端原样传播(同一实例)且绝不触碰该目标的网络。
代理:``PRAXIS_WEB_PROXY`` > https_proxy 族;trusted_proxy 模式由上游代理负责
最终 IP 裁决,本地 DNS 固定**按设计旁路**(legacy 语义);代理配置只来自启动
环境/显式参数,**workspace 文件绝不读**。
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
import socket
import ssl
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit
from urllib.request import proxy_bypass_environment  # type: ignore[attr-defined]  # typeshed 缺名,legacy 同款

import httpcore
import httpx

from praxis.adapters.web.limits import WebLimits
from praxis.adapters.web.urls import PublicWebError, is_public_address, validate_public_url

__all__ = [
    "CLIENT_ERROR_CODES",
    "PublicHttpClient",
    "PublicWebError",
    "Resolver",
    "WebFetchResponse",
    "authorize_callback",
    "public_http_transport",
    "validate_proxy_url",
    "web_proxy_configuration",
]

# brief 逐字码集:fetch 路径自产 + authorize 回调的 denied;invalid_url 属预检,不在此列。
CLIENT_ERROR_CODES = frozenset(
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

# 逐跳授权回调(T11 由 composition 绑到 decide 闭包);raise ⇒ 立即中止本次请求。
authorize_callback = Callable[[str], None]

# getaddrinfo 注入点:(host, port) → 全部地址字符串;默认 asyncio.to_thread(socket.getaddrinfo)。
Resolver = Callable[[str, int], Awaitable[Sequence[str]]]

_MAX_CANDIDATES = 8
_PER_CANDIDATE_CAP = 3.0
_MAX_LOCATION_CHARS = 4096
_DEFAULT_DEADLINE_SECONDS = 20.0
_CROSS_HOP_HEADERS = frozenset({"accept", "accept-language", "accept-encoding", "user-agent"})
# 响应头只保留真实有界的浏览器策略;绝不带 cookie / content-encoding 给模型。
_RESPONSE_HEADERS = (
    "access-control-allow-origin",
    "access-control-allow-credentials",
    "access-control-expose-headers",
    "referrer-policy",
)

# 与 httpcore 公开 backend 签名一致,不 import 私有模块。
SocketOption = tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]


def validate_proxy_url(value: str) -> str:
    """代理端点是可信应用配置,绝不是模型输入(legacy :43-55 parity)。"""
    try:
        url = httpx.URL(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError
        return str(url)
    except (ValueError, httpx.InvalidURL):
        raise ValueError(
            "Web proxy must be an HTTP(S) endpoint without credentials, path, query or fragment."
        ) from None


def web_proxy_configuration(explicit: str | None, environment: Mapping[str, str]) -> tuple[str | None, str]:
    """读调用方持有的启动环境一次;workspace 文件绝不读(legacy :58-69 parity)。"""
    value = explicit
    if value is None:
        value = next(
            (
                environment[key]
                for key in ("PRAXIS_WEB_PROXY", "https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY")
                if environment.get(key)
            ),
            None,
        )
    proxy = None if value is None or value in {"", "direct"} else validate_proxy_url(value)
    no_proxy = environment.get("no_proxy", environment.get("NO_PROXY", ""))
    return proxy, ",".join(no_proxy.replace(",", " ").split())


async def _default_resolver(host: str, port: int) -> list[str]:
    answers = await asyncio.to_thread(
        socket.getaddrinfo, host, port, socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP
    )
    return [str(answer[4][0]) for answer in answers]


class _ConnectionCandidates:
    """一份已校验 DNS 快照 + 一个 TCP/TLS 共享 deadline(legacy :145-185)。"""

    def __init__(
        self,
        backend: httpcore.AsyncNetworkBackend,
        addresses: Sequence[str],
        port: int,
        timeout: float | None,
        local_address: str | None,
        socket_options: Iterable[SocketOption] | None,
    ) -> None:
        self.backend = backend
        self.addresses = list(dict.fromkeys(addresses))[:_MAX_CANDIDATES]
        self.port = port
        self.deadline = asyncio.get_running_loop().time() + (
            timeout if timeout is not None else _DEFAULT_DEADLINE_SECONDS
        )
        self.local_address = local_address
        self.socket_options = socket_options
        self.index = 0
        self.attempt_deadline = self.deadline

    def remaining(self) -> float:
        return max(0.0, self.attempt_deadline - asyncio.get_running_loop().time())

    async def connect(self) -> httpcore.AsyncNetworkStream:
        last_error: Exception = httpcore.ConnectTimeout()
        while self.index < len(self.addresses):
            now = asyncio.get_running_loop().time()
            remaining = self.deadline - now
            if remaining <= 0:
                raise PublicWebError("timeout", "The public HTTP connection exceeded its deadline.")
            # 即使请求预算很小,也为其他候选保留时间。
            budget = min(_PER_CANDIDATE_CAP, remaining / (len(self.addresses) - self.index))
            self.attempt_deadline = now + budget
            address = self.addresses[self.index]
            self.index += 1
            try:
                async with asyncio.timeout(budget):
                    return await self.backend.connect_tcp(
                        address,
                        self.port,
                        timeout=budget,
                        local_address=self.local_address,
                        socket_options=self.socket_options,
                    )
            except (httpcore.NetworkError, httpcore.TimeoutException, OSError, TimeoutError) as error:
                last_error = error
        if isinstance(last_error, (httpcore.TimeoutException, TimeoutError)):
            raise PublicWebError("timeout", "The public HTTP connection exceeded its deadline.") from last_error
        raise PublicWebError(
            "connect_failed", "The public HTTP connection failed; no website response was received."
        ) from last_error


class _PublicStream(httpcore.AsyncNetworkStream):
    """只重试未发 HTTP 字节的连接;关闭每个失败/被取消的 socket(legacy :188-233)。

    start_tls 成功即返回**不带候选引用**的新包装——请求字节写出后的任何失败
    都没有回退路径(结构保证,而非运行时标记)。
    """

    def __init__(
        self, stream: httpcore.AsyncNetworkStream, candidates: _ConnectionCandidates | None = None
    ) -> None:
        self._stream = stream
        self._candidates = candidates

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return await self._stream.read(max_bytes, timeout)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        await self._stream.write(buffer, timeout)

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        while True:
            budget = self._candidates.remaining() if self._candidates is not None else timeout
            try:
                async with asyncio.timeout(budget):
                    secured = await self._stream.start_tls(ssl_context, server_hostname, budget)
                return _PublicStream(secured)
            except BaseException as error:
                try:
                    # 关闭失败 socket 不得掩盖原始失败或取消。
                    async with asyncio.timeout(0.1):
                        await self._stream.aclose()
                except Exception:
                    pass
                if not isinstance(error, Exception) or isinstance(error, PublicWebError):
                    raise  # 取消/键盘中断/已映射错误:原样传播
                recoverable = isinstance(
                    error, (httpcore.NetworkError, httpcore.TimeoutException, OSError, TimeoutError)
                )
                if (
                    self._candidates is None
                    or self._candidates.index >= len(self._candidates.addresses)
                    or not recoverable
                ):
                    if isinstance(error, (httpcore.TimeoutException, TimeoutError)):
                        raise PublicWebError(
                            "timeout", "The public HTTPS handshake exceeded its deadline."
                        ) from error
                    raise PublicWebError("tls_failed", "The public HTTPS TLS handshake failed.") from error
                self._stream = await self._candidates.connect()

    def get_extra_info(self, info: str) -> object:
        return self._stream.get_extra_info(info)


class _PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """连接期逐答案校验、只连数字地址(legacy :240-274;resolver 注入点)。"""

    def __init__(self, resolver: Resolver, backend: httpcore.AsyncNetworkBackend) -> None:
        self._resolver = resolver
        self._backend = backend

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SocketOption] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        # 数字直连避免第二次未校验 DNS(rebinding)。每个 A/AAAA 答案都必须公网。
        async with asyncio.timeout(timeout):
            started = asyncio.get_running_loop().time()
            try:
                ipaddress.ip_address(host)
            except ValueError:
                try:
                    addresses = list(await self._resolver(host, port))
                except OSError as error:
                    raise PublicWebError("connect_failed", "DNS resolution failed.") from error
            else:
                addresses = [host]
            if not addresses or any(not is_public_address(address) for address in addresses):
                raise PublicWebError(
                    "nonpublic_address",
                    "DNS returned a non-public address; direct public-web access was blocked before HTTP. "
                    "No website response was received, so this does not establish whether a repository "
                    "is private, missing or misspelled. Check DNS or configure a trusted HTTP proxy.",
                )
            remaining = (
                None
                if timeout is None
                else max(0.0, timeout - (asyncio.get_running_loop().time() - started))
            )
            candidates = _ConnectionCandidates(
                self._backend, addresses, port, remaining, local_address, socket_options
            )
            return _PublicStream(await candidates.connect(), candidates)


class _CoreStream(httpx.AsyncByteStream):
    def __init__(self, response: httpcore.Response) -> None:
        self._response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.aiter_stream():
            yield chunk

    async def aclose(self) -> None:
        await self._response.aclose()


class _PublicTransport(httpx.AsyncBaseTransport):
    """keepalive=0 ⇒ 无 DNS 缓存、每请求新连接(legacy :295-304 原样)。"""

    def __init__(self, network_backend: httpcore.AsyncNetworkBackend, proxy_url: str | None = None) -> None:
        self._pool: httpcore.AsyncConnectionPool | httpcore.AsyncHTTPProxy
        if proxy_url is None:
            self._pool = httpcore.AsyncConnectionPool(
                ssl_context=ssl.create_default_context(),
                network_backend=network_backend,
                retries=0,
                max_connections=10,
                max_keepalive_connections=0,
            )
        else:
            # 可信上游代理解析 origin 并负责目标 IP 裁决(本地 DNS 固定按设计旁路)。
            self._pool = httpcore.AsyncHTTPProxy(
                proxy_url=proxy_url,
                ssl_context=ssl.create_default_context(),
                proxy_ssl_context=(
                    ssl.create_default_context() if httpx.URL(proxy_url).scheme == "https" else None
                ),
                network_backend=network_backend,
                retries=0,
                max_connections=10,
                max_keepalive_connections=0,
            )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert isinstance(request.stream, httpx.AsyncByteStream)
        response = await self._pool.handle_async_request(
            httpcore.Request(
                method=request.method,
                url=httpcore.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=request.stream,
                extensions=request.extensions,
            )
        )
        return httpx.Response(
            response.status,
            headers=response.headers,
            stream=_CoreStream(response),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


def _direct_transport(
    *,
    resolver: Resolver | None = None,
    backend: httpcore.AsyncNetworkBackend | None = None,
) -> _PublicTransport:
    """DNS 固定直连传输的**唯一构造点**(PublicHttpClient 与 public_http_transport 共用)。"""
    inner = httpcore.AnyIOBackend() if backend is None else backend
    pinned = _PinnedNetworkBackend(_default_resolver if resolver is None else resolver, inner)
    return _PublicTransport(pinned)


def public_http_transport(
    *,
    resolver: Resolver | None = None,
    backend: httpcore.AsyncNetworkBackend | None = None,
) -> httpx.AsyncBaseTransport:
    """DNS 固定、仅公共地址的直连传输(旧 ``web_http.public_http_transport``)。

    供非 web 的宿主自有 HTTP 集成复用同一连接期校验(T18 的 MCP HTTP);
    ``resolver``/``backend`` 是确定性测试接缝,绝不暴露给模型。
    """
    return _direct_transport(resolver=resolver, backend=backend)


@dataclass(frozen=True, slots=True)
class WebFetchResponse:
    """有界抓取结果;headers 只含浏览器策略白名单,绝不含 cookie/content-encoding。"""

    status: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str
    content_type: str


class _AuthorizeAbortError(BaseException):
    """内部哨兵:把 authorizer 的任意异常原样带出网络错误映射层(同一实例)。"""

    def __init__(self, original: BaseException) -> None:
        self.original = original
        super().__init__()


def _canonical_target(scheme: str, host: str, port: int, path_query: str) -> str:
    hostname = f"[{host}]" if ":" in host else host
    default = 443 if scheme == "https" else 80
    authority = hostname if port == default else f"{hostname}:{port}"
    return f"{scheme}://{authority}{path_query}"


class PublicHttpClient:
    """有界公共读取 + 可信 JSON POST;逐跳 authorize;直连 DNS 固定。

    ``resolver``/``backend`` 是确定性测试接缝(仅可信应用代码可注入,绝不暴露
    给模型)。重定向自处理:每跳先 authorize 再 validate,Location join ≤4096,
    跨跳剥凭证(只留 accept*/user-agent);POST 恒禁重定向。gzip 增量解压且
    **解码后**同限计数;非 gzip/identity 拒绝。代理配置来自启动环境
    (``PRAXIS_WEB_PROXY`` > https_proxy 族)或显式参数;workspace 文件绝不读。
    """

    def __init__(
        self,
        *,
        authorize: authorize_callback,
        limits: WebLimits,
        resolver: Resolver | None = None,
        backend: httpcore.AsyncNetworkBackend | None = None,
        proxy_url: str | None = None,
        no_proxy: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._authorize = authorize
        self._limits = limits
        environment = os.environ if environ is None else environ
        proxy, normalized_no_proxy = web_proxy_configuration(proxy_url, environment)
        self._proxy_url = proxy
        self._no_proxy = (
            normalized_no_proxy if no_proxy is None else ",".join(no_proxy.replace(",", " ").split())
        )
        inner = httpcore.AnyIOBackend() if backend is None else backend
        self._transport = _direct_transport(resolver=resolver, backend=backend)
        # 代理路径用普通 backend(不经 pinned 包装):上游代理负责最终 IP 裁决
        # (legacy 语义,按设计旁路本地钉死)。backend 接缝对两条 transport 同时生效
        # (生产 backend=None ⇒ 两者皆 AnyIOBackend;测试注入假 backend 可演 CONNECT)。
        self._proxy_transport = _PublicTransport(inner, proxy) if proxy is not None else None

    def connection_mode(self, target: str) -> str:
        """'direct' | 'trusted_proxy'(legacy :388-392;no_proxy 命中回落 direct)。"""
        parsed = urlsplit(target)
        host = parsed.hostname or ""
        hostname = f"[{host}]" if ":" in host else host
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if self._proxy_url is not None and not proxy_bypass_environment(
            f"{hostname}:{port}", {"no": self._no_proxy}
        ):
            return "trusted_proxy"
        return "direct"

    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        max_bytes: int | None = None,
        timeout_seconds: float | None = None,
        follow_redirects: bool = True,
    ) -> WebFetchResponse:
        """有界抓取;max_bytes/timeout_seconds 是可信调用方的逐请求覆盖(默认取 limits)。

        ``follow_redirects=False``(R-T13-3,legacy ``allow_redirects=False`` parity):
        首个 3xx 直接 ``invalid_redirect``,不消费跳数、不发第二跳。搜索 provider
        端点按 legacy 语义禁止重定向;web_fetch 默认仍跟随并逐跳授权。

        **调用方不变量(composition/T20)**:client 与本客户端配套的工具工厂必须用
        **同一份** ``WebLimits`` 构造(逐请求上限只有一个事实来源)。逐请求覆盖只
        接受可信调用方——模型参数已由工具 schema 与工具侧 min(limits, 工具预算)
        钳制;client 层不再二次硬钳制,这是有意的分层(避免两处各自缩放同一上限)。

        json_body 须为 JSON 安全值(T12 控制请求体):NaN/Infinity 抛裸 ValueError
        (legacy web_http.py:415 allow_nan=False parity,调用方输入类,不属于
        PublicWebError 有界错误面)。
        """
        limit = self._limits.max_bytes if max_bytes is None else max_bytes
        if limit < 1:
            raise ValueError("Response byte limit must be positive.")
        timeout = self._limits.timeout_seconds if timeout_seconds is None else timeout_seconds
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Request timeout must be a positive finite number.")
        content: bytes | None = None
        request_headers = httpx.Headers(headers)
        if json_body is not None:
            content = json.dumps(json_body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(content) > limit:
                raise PublicWebError("max_bytes_exceeded", "The JSON request exceeds the byte limit.")
            request_headers["Content-Type"] = "application/json"
        request_headers["Accept-Encoding"] = "gzip, identity"
        try:
            async with asyncio.timeout(timeout):
                return await self._fetch_loop(
                    url, method, request_headers, content, limit, timeout, follow_redirects
                )
        except _AuthorizeAbortError as abort:
            raise abort.original from None
        except PublicWebError:
            raise
        except httpx.InvalidURL:
            # 兜底层 (b):预检 IDNA 三层之外残余的 httpx 解析失败(InvalidURL ⊄ HTTPError),
            # 绝不裸泄漏给调用方。
            raise PublicWebError(
                "invalid_url", "A public HTTP(S) URL on port 80 or 443 is required."
            ) from None
        except (TimeoutError, httpx.TimeoutException, httpcore.TimeoutException):
            raise PublicWebError("timeout", "The public HTTP request exceeded its deadline.") from None
        except httpcore.ProxyError:
            # ProxyError ⊄ NetworkError:CONNECT 非 2xx 单独映射;消息不回显代理 reason phrase。
            raise PublicWebError("connect_failed", "The configured proxy rejected the connection.") from None
        except ssl.SSLError:
            # TLS 已成功、字节已发之后的 TLS 层失败:无回退(结构保证),直接 tls_failed。
            raise PublicWebError("tls_failed", "The public HTTPS connection failed during TLS.") from None
        except (httpx.HTTPError, httpcore.NetworkError, httpcore.ProtocolError, OSError):
            raise PublicWebError("connect_failed", "The public HTTP request failed.") from None

    async def _fetch_loop(
        self,
        url: str,
        method: str,
        headers: httpx.Headers,
        content: bytes | None,
        limit: int,
        timeout: float,
        follow_redirects: bool,
    ) -> WebFetchResponse:
        target = url
        wire_bytes = 0
        for hop in range(self._limits.max_redirects + 1):
            # 每跳先 authorize(任意异常原样传播),再 validate,再连接。
            try:
                self._authorize(target)
            except BaseException as error:
                raise _AuthorizeAbortError(error) from None
            host, port, path_query = validate_public_url(target)
            canonical = _canonical_target(urlsplit(target).scheme, host, port, path_query)
            # 自处理重定向与原始解码,不用 httpx client(它会急切解析 Location)。
            request = httpx.Request(
                method,
                canonical,
                headers=headers,
                content=content,
                extensions={
                    "timeout": dict.fromkeys(
                        ("connect", "read", "write", "pool"), timeout
                    )
                },
            )
            mode = self.connection_mode(canonical)
            transport = self._proxy_transport if mode == "trusted_proxy" else self._transport
            assert transport is not None
            response = await transport.handle_async_request(request)
            try:
                if 300 <= response.status_code < 400:
                    if not follow_redirects:
                        raise PublicWebError(
                            "invalid_redirect", "Redirects are not allowed for this request."
                        )
                    if method != "GET":
                        raise PublicWebError(
                            "invalid_redirect", "Redirects are not allowed for this request."
                        )
                    if hop >= self._limits.max_redirects:
                        raise PublicWebError(
                            "too_many_redirects", "The public HTTP request exceeded its redirect limit."
                        )
                    location = response.headers.get("Location")
                    if not location:
                        raise PublicWebError("invalid_redirect", "The redirect has no valid destination.")
                    try:
                        joined = urljoin(canonical, location)
                    except ValueError:
                        # R-T10-2:攻击者可控 Location(如 "https://[::1")的 URL 解析会抛
                        # 裸 ValueError;有界错误面一律 invalid_redirect,消息静态、不回显 Location。
                        raise PublicWebError(
                            "invalid_redirect", "The redirect has no valid destination."
                        ) from None
                    if len(joined) > _MAX_LOCATION_CHARS:
                        raise PublicWebError("invalid_redirect", "The redirect has no valid destination.")
                    target = joined
                    headers = httpx.Headers(
                        {key: value for key, value in headers.items() if key.lower() in _CROSS_HOP_HEADERS}
                    )
                    continue
                response_headers = {
                    name: value
                    for name in _RESPONSE_HEADERS
                    if (value := response.headers.get(name)) is not None
                    and len(value) <= 1024
                    and all(ord(character) >= 32 and ord(character) != 127 for character in value)
                }
                body, wire_bytes = await self._read_body(response, wire_bytes, limit)
                return WebFetchResponse(
                    status=response.status_code,
                    headers=response_headers,
                    body=body,
                    final_url=canonical,
                    content_type=response.headers.get("Content-Type", ""),
                )
            finally:
                await response.aclose()
        raise AssertionError("Redirect loop must return or raise.")

    async def _read_body(self, response: httpx.Response, wire_bytes: int, limit: int) -> tuple[bytes, int]:
        encoding = response.headers.get("Content-Encoding", "identity").lower().strip()
        if encoding not in {"", "identity", "gzip"}:
            raise PublicWebError(
                "encoding_unsupported", "The response uses an unsupported content encoding."
            )
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None
        body = bytearray()
        try:
            async for chunk in response.aiter_raw():
                wire_bytes += len(chunk)
                if wire_bytes > limit:
                    raise PublicWebError("max_bytes_exceeded", "The response exceeds the byte limit.")
                if decoder is not None:
                    chunk = decoder.decompress(chunk, limit - len(body) + 1)
                body.extend(chunk)
                if len(body) > limit:
                    raise PublicWebError("max_bytes_exceeded", "The response exceeds the byte limit.")
            if decoder is not None and (not decoder.eof or decoder.unused_data):
                raise PublicWebError(
                    "encoding_unsupported", "The gzip response is incomplete or malformed."
                )
        except zlib.error:
            raise PublicWebError(
                "encoding_unsupported", "The gzip response is incomplete or malformed."
            ) from None
        return bytes(body), wire_bytes

    async def aclose(self) -> None:
        try:
            await self._transport.aclose()
        finally:
            if self._proxy_transport is not None:
                await self._proxy_transport.aclose()

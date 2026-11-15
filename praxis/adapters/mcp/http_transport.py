"""MCP HTTP 传输:精确 endpoint + Host 守卫、无 3xx、401/403 有界映射(3B §5.10;T18)。

**安装版 SDK 面(mcp 1.27.1)**:``mcp.client.streamable_http.streamable_http_client(
url, *, http_client: httpx.AsyncClient | None = None, terminate_on_close=True)``
接受**注入的 AsyncClient**。本模块因此自带客户端:``GuardedMcpTransport``
包住公共直连传输(``web.client.public_http_transport``:DNS 固定、仅公共地址),
``trust_env=False`` 关闭环境代理、``follow_redirects=False`` 关闭客户端重定向;
守卫类在**传输层**拒绝一切 3xx,不依赖客户端的跟随设置。SDK 的另一入口
``streamablehttp_client`` 在老签名里接受 headers/timeout/auth,但本版本已将其
标记 deprecated 并**忽略**——不用它。``timeout=httpx.Timeout(startup,
read=300)`` 逐字对齐旧 ``:509``(连接预算取配置的 startup 超时,SSE 读 300s)。

**Q3 收敛(controller 裁决 R-T18-1;与旧 :432-442 的有意差异)**:旧
``_MCPHTTPTransport.check_permissions`` 每次请求做 network_decision /
tool_decision / config 字节漂移三条检查。praxis 的 HTTP MCP 工具带
``ToolTarget("url", endpoint)``(T17),每次 ``execute`` 都由单一 ``decide``
用**当次** live PermissionContext 裁决:deny/ask 正常生效、ask 走审批暂停 +
``policy_revision`` 失效。因此本传输只保留**工具层无法表达**的传输完整性
守卫:

1. ``str(request.url)`` 与配置 endpoint **逐字相等**(path/query/大小写任一
   变化都拒——凭证不可能被带到同主机的另一个路径);
2. ``Host`` 头与 ``request.url.netloc`` 一致(防 Host-authority 混淆);
3. 一切 3xx 拒绝;
4. 401/403 ⇒ ``McpTransportError(code="mcp_http_unauthorized")``。

旧 OAuth 腿的 ``destination_check`` **不再移植**:策略漂移只有一个面
(``decide``),不在传输层造第二条网络裁决路径。连接期的
``network_verdict_at_connect`` 是**发现预授权**快照(Q4 fail-closed),不是
裁决路径;ask 视作未授权并给出可执行文案(MCP 发现发生在会话装配期,无
暂停面)。

**凭据链(Q6)**:``bearer_token_env_var`` → 非空环境变量优先,缺失再回落
``credentials.load_mcp_bearer``(与 ``resolve_model_credential`` 同款
env→Keychain 链);``keychain: true`` → 仅 ``load_mcp_bearer``。凭据值只进
``Authorization`` 头,绝不进错误消息/日志。``credential_revision`` 是
sha256(canonical JSON:headers 展开值 + bearer **是否存在**)——不读任何
OAuth 存储(T15 已拒 OAuth 配置键)。

**owner task**:与 stdio 共用 ``lifecycle.owned_connection`` 唯一骨架
(AnyIO cancel scope 任务局部性同一理由);本模块只提供 connect 闭包。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

# 模块式导入(而非 ``from ... import load_mcp_bearer``)是刻意的:凭据层反过来
# 依赖 ``adapters.mcp.names`` 的规范名,模块式绑定让"先 import credentials"
# 的装载顺序不构成循环属性读取(调用期属性才解析)。
from praxis.adapters.config import credentials
from praxis.adapters.mcp.catalog import list_all_tools
from praxis.adapters.mcp.config import HttpServerConfig
from praxis.adapters.mcp.lifecycle import McpConnection, owned_connection
from praxis.adapters.web.client import public_http_transport
from praxis.protocol.policy import PolicyVerdict

_READ_TIMEOUT_SECONDS = 300.0

_ENDPOINT_CODE = "mcp_http_endpoint_changed"
_HOST_CODE = "mcp_http_host_mismatch"
_REDIRECT_CODE = "mcp_http_redirect"
_UNAUTHORIZED_CODE = "mcp_http_unauthorized"

_ENDPOINT_MESSAGE = "MCP request cannot change its configured endpoint"
_HOST_MESSAGE = "MCP HTTP Host authority must match its configured endpoint"
_REDIRECT_MESSAGE = "MCP HTTP redirects are refused; credentials stay at the configured endpoint"
_UNAUTHORIZED_MESSAGE = "remote MCP authentication or authorization failed (HTTP {status})"


class McpTransportError(PermissionError):
    """MCP HTTP 传输的有界拒绝(旧 ``PermissionError`` 族)。

    ``code`` 是稳定机器码;``str(error)`` 只含固定文案与 HTTP 状态码,绝不
    回显 HTTP 头、凭据或请求 URL 参数。
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class GuardedMcpTransport(httpx.AsyncBaseTransport):
    """对**配置 endpoint** 的逐请求传输完整性守卫(httpx transport 适配器)。

    ``inner`` 是真实传输(生产为 ``public_http_transport()``;测试注入
    ``httpx.MockTransport`` 以无网验证矩阵)。ACL 语义见模块 docstring。
    """

    def __init__(self, endpoint: str, inner: httpx.AsyncBaseTransport) -> None:
        self._endpoint = endpoint
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) != self._endpoint:
            raise McpTransportError(_ENDPOINT_CODE, _ENDPOINT_MESSAGE)
        host = request.headers.get("host", "").casefold()
        if host != request.url.netloc.decode("ascii").casefold():
            raise McpTransportError(_HOST_CODE, _HOST_MESSAGE)
        response = await self._inner.handle_async_request(request)
        # 旧实现用 ``response.is_redirect``(依赖 Location);这里按状态码区间,
        # 300-399 一律拒绝,与"无论方法、有无 Location"的裁决一致。
        if 300 <= response.status_code < 400:
            await response.aclose()
            raise McpTransportError(_REDIRECT_CODE, _REDIRECT_MESSAGE)
        if response.status_code in {401, 403}:
            status = response.status_code
            await response.aclose()
            raise McpTransportError(_UNAUTHORIZED_CODE, _UNAUTHORIZED_MESSAGE.format(status=status))
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def _resolve_bearer(config: HttpServerConfig) -> str | None:
    """env 优先、Keychain 回落的 bearer 解析链;缺失时点名来源(绝不回显值)。"""
    if config.bearer_token_env_var is not None:
        value = os.environ.get(config.bearer_token_env_var)
        if isinstance(value, str) and value.strip():
            return value.strip()
        stored = credentials.load_mcp_bearer(config.name, config.url)
        if stored is not None:
            return stored
        raise ValueError(
            f"missing MCP credential environment variable: {config.bearer_token_env_var}"
        )
    if config.keychain:
        stored = credentials.load_mcp_bearer(config.name, config.url)
        if stored is None:
            raise ValueError(
                f"MCP credential missing for {config.name}; save a bearer token first"
            )
        return stored
    return None


def _headers_with_bearer(headers: Mapping[str, str], token: str | None) -> dict[str, str]:
    """配置 headers(已展开)+ bearer;无 bearer 时原样保留配置的 Authorization。"""
    result = dict(headers)
    if token is not None:
        # bearer 源显式覆盖配置里的任何 Authorization(大小写不敏感)。
        result = {key: value for key, value in result.items() if key.casefold() != "authorization"}
        result["Authorization"] = f"Bearer {token}"
    return result


def credential_revision(headers: Mapping[str, str], *, bearer_present: bool) -> str:
    """凭据形状的稳定指纹:headers 展开值 + bearer 是否存在(不含明文)。"""
    state = {"headers": dict(headers), "bearer": bearer_present}
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@asynccontextmanager
async def owned_http_connection(
    config: HttpServerConfig,
    *,
    workspace: Path,
    network_verdict_at_connect: Callable[[str], PolicyVerdict],
    close_timeout_seconds: float = 5.0,
) -> AsyncIterator[McpConnection]:
    """连接一个 HTTP MCP server,就绪后 yield,退出时有界关闭。

    Parameters:
        config: T15 已校验/已展开的 HTTP 配置(endpoint 已是 ``str(httpx.URL)``)。
        workspace: 与 stdio 兄弟函数的调用点对称而保留;HTTP 目前不需要任何
            workspace 作用域资源(endpoint 来自宿主配置,工具调用由 decide 裁决)。
        network_verdict_at_connect: 连接期策略快照(T19 传入闭包**当时**的 live
            裁决)。非 ``allow`` ⇒ **在派生 owner task 之前** fail-closed 拒绝;
            ask 与 deny 同等处理,并提示先在 policy 允许该域。
        close_timeout_seconds: 关闭宽限;超时即 cancel owner 并 gather。

    Raises:
        ValueError: 连接期网络未预授权,或配置的凭据源缺失/绑定不匹配。
        Exception: 传输/握手失败原样上抛(调用方逐 server 隔离,不重试)。
    """
    verdict = network_verdict_at_connect(config.url)
    if verdict != "allow":
        host = httpx.URL(config.url).host
        raise ValueError(
            f"mcp http server requires network allow for {host}; ask/deny at connect time "
            "is fail-closed (Q4 discovery preauthorization parity)"
        )
    bearer = _resolve_bearer(config)
    headers = _headers_with_bearer(config.headers, bearer)
    revision = credential_revision(config.headers, bearer_present=bearer is not None)

    async def connect(resources: AsyncExitStack) -> McpConnection:
        transport = GuardedMcpTransport(config.url, public_http_transport())
        client = await resources.enter_async_context(
            httpx.AsyncClient(
                transport=transport,
                headers=headers,
                timeout=httpx.Timeout(config.startup_timeout_seconds, read=_READ_TIMEOUT_SECONDS),
                trust_env=False,
                follow_redirects=False,
            )
        )
        streams = await resources.enter_async_context(
            streamable_http_client(config.url, http_client=client)
        )
        session = await resources.enter_async_context(ClientSession(streams[0], streams[1]))
        await session.initialize()
        catalog = await list_all_tools(session)
        return McpConnection(
            server_name=config.name,
            session=session,
            tools=catalog,
            credential_revision=revision,
        )

    async with owned_connection(
        config.name,
        connect,
        startup_timeout_seconds=config.startup_timeout_seconds,
        close_timeout_seconds=close_timeout_seconds,
    ) as connection:
        yield connection


__all__ = [
    "GuardedMcpTransport",
    "McpTransportError",
    "credential_revision",
    "owned_http_connection",
]
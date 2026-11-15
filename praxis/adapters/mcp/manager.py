"""MCP server manager:eager 连接、策略预检、逐 server 隔离与 bundle 收尾(T19;Q4/Q9)。

**装配语义(controller 钉死)**:

1. 配置只经 T15 唯一入口加载(信任门 + ``${VAR}`` 展开 + 字节 sha 基线);
   隐式默认文件缺失 ⇒ 空 bundle(无 MCP 的既有路径零行为变化)。
2. **策略预检在连任何 server 之前**(声明快照,不是第二决策入口):
   ``tool_decision(resolve_policy(policy, workspace), server_wildcard(name))``
   为 ``deny`` ⇒ 跳过 + ``mcp_server_skipped`` 诊断;显式 allowlist 全部
   ``deny`` 同样跳过;HTTP 另要求 ``network_decision(...) == "allow"``
   (ask/deny 一律 fail-closed 跳过——Q4 发现预授权)。
3. 逐 server 独立连接:stdio 在 T19 编译的沙箱 profile 内派生(T16 owner
   task),HTTP 经 T18 守卫传输。**单 server 失败只关闭自己的栈**,诊断码
   ``mcp_server_unavailable``,其余 server 照常可用。
4. ``McpBundle.aclose`` **幂等**:关闭全部 owner 栈(进程组回收在 SDK 内闭合)
   并 ``finalize_sandbox`` 逐个清理编译产物。Session 的 integrations 收尾与
   CLI 的 ``finally`` 双触达无害。
5. connect 期异常组经 ``_errors`` 唯一摊平(外层 kind + 第一片叶子摘要,
   ≤512 字符);配置 env/header 展开值做兜底 redact,秘密绝不进诊断。

**注入纪律(R-T15-7)**:adapters 不得 import tools,``env_builder`` 由
``composition`` 绑定 ``tools.builtins.child_env`` 的唯一实现;``sandbox_compiler``
同族注入(生产为 ``adapters.sandbox.compile.compile_sandbox``)。本模块不读
环境凭据、不建第二策略入口(``decide`` 仍在管道里裁决每次调用)。
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from praxis.adapters.mcp._errors import bounded_message, diagnostic_message
from praxis.adapters.mcp.config import (
    HttpServerConfig,
    McpServerConfig,
    StdioServerConfig,
    config_sha256,
    load_mcp_configs,
    resolve_mcp_config_path,
)
from praxis.adapters.mcp.descriptors import mcp_tools_for_connection
from praxis.adapters.mcp.http_transport import owned_http_connection
from praxis.adapters.mcp.lifecycle import McpConnection, owned_stdio_connection
from praxis.adapters.mcp.names import canonical_tool_name, normalize_mcp_name, server_wildcard
from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import finalize_sandbox
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import (
    PermissionPolicy,
    PolicyVerdict,
    ResolvedPolicy,
    network_decision,
    resolve_policy,
    tool_decision,
)
from praxis.protocol.sandbox import EnvBuilder, SandboxSpec
from praxis.protocol.tools import ResolvedUse, Tool, ToolEffect, ToolTarget

logger = logging.getLogger(__name__)

type SandboxCompiler = Callable[[PermissionContext, ResolvedUse, Path], SandboxSpec]

_SKIPPED = "mcp_server_skipped"
_UNAVAILABLE = "mcp_server_unavailable"
_CLOSE_TIMEOUT_SECONDS = 5.0
# 环境/header 展开值进入诊断前的兜底 redact 门槛:短值(如 "C.UTF-8")
# 出现在异常文本里属于正常噪声,重写它只会让诊断更难读。
_REDACT_MIN_LENGTH = 8


@dataclass(frozen=True, slots=True)
class McpServerStatus:
    """一个 server 的 connect 结果(成功/策略跳过/不可用三类之一)。"""

    server_name: str
    transport: Literal["stdio", "http"]
    connected: bool
    tool_count: int = 0
    diagnostic_code: str | None = None
    diagnostic_message: str | None = None


class McpBundle:
    """一次 eager connect 的产物:工具投影 + 状态 + 幂等收尾面。

    ``tools`` 是各连接目录的**构造期快照**(T17 factory);``statuses`` 与
    ``tools`` 顺序无关但覆盖每个参与本次 connect 的 server。``aclose`` 可在
    任意 task 调用(owner 栈关闭由 ``owned_connection`` 的 shield 语义兜住)。

    收尾采用**单飞行任务**(repair #2):并发 ``aclose`` 共享同一个受 shield
    保护的 teardown;``_closed`` 只在 teardown **成功完成后**置位——调用方在
    teardown 中被取消不会留下未关闭的栈,后续调用可重试(任务自身被取消时
    ``_closing`` 复位)。
    """

    def __init__(
        self,
        tools: tuple[Tool, ...],
        statuses: tuple[McpServerStatus, ...],
        *,
        _stacks: tuple[AsyncExitStack, ...] = (),
        _specs: tuple[SandboxSpec, ...] = (),
        _workspace: Path | None = None,
    ) -> None:
        self.tools = tools
        self.statuses = statuses
        self._stacks = _stacks
        self._specs = _specs
        self._workspace = _workspace
        self._closed = False
        self._closing: asyncio.Task[None] | None = None

    async def aclose(self) -> None:
        """关闭全部连接与沙箱产物;**重复/并发调用安全**。

        stdout/进程组回收由 owner task 的 ``owned_connection`` 完成;这里只
        按逆序关栈,并在栈关闭后逐个 ``finalize_sandbox``。单个栈清理失败只
        记类型名日志(不展开原始异常文本——它可能含配置展开值),不让一个坏
        server 阻止其余收尾。
        """
        if self._closed:
            return
        if self._closing is None:
            self._closing = asyncio.ensure_future(self._teardown())
        # shield:调用方被取消时 teardown 继续;并发调用者 await 同一个任务。
        await asyncio.shield(self._closing)

    async def _teardown(self) -> None:
        try:
            for stack in reversed(self._stacks):
                try:
                    await stack.aclose()
                except Exception as error:  # noqa: BLE001 — 一个 server 不得阻塞其余
                    # 不记 exc_info:异常文本可能携带配置展开值(repair #2)。
                    logger.warning("MCP connection teardown failed: %s", type(error).__name__)
            workspace = self._workspace if self._workspace is not None else Path.cwd()
            for spec in self._specs:
                try:
                    finalize_sandbox(spec, workspace)
                except Exception as error:  # noqa: BLE001 — finalize 自身不抛
                    logger.warning("MCP sandbox cleanup failed: %s", type(error).__name__)
            self._closed = True
        finally:
            # 任务被取消时 _closed 仍未置位;清空让后续 aclose 重试。
            self._closing = None


async def connect_mcp_servers(
    workspace: Path,
    *,
    policy: PermissionPolicy | None = None,
    sandbox_compiler: SandboxCompiler,
    env_builder: EnvBuilder,
    network_verdict_at_connect: Callable[[str], PolicyVerdict] | None = None,
    close_timeout_seconds: float = _CLOSE_TIMEOUT_SECONDS,
) -> McpBundle:
    """按配置 eager 连接全部 enabled MCP server,返回工具投影与状态。

    Parameters:
        workspace: 真实工作区;配置/``cwd``/沙箱都以它为准(绝不用进程 cwd)。
        policy: **声明策略快照**(composition ``resolve_declared_policy`` 的
            唯一加载产物)。``None`` 只表示"调用方未声明"(空策略),声明文件
            不会被本模块读取——适配器层不建第二个策略加载入口。
        sandbox_compiler: stdio 沙箱编译(composition 注入生产实现)。
        env_builder: 子进程最小环境(composition 注入 ``child_env``)。
        network_verdict_at_connect: HTTP 连接期网络凭据的裁决源;缺省按
            声明快照的 ``network_decision``。ask/deny 都在派生 owner task 前
            跳过一个 server(fail-closed),不是第二条调用期裁决路径。
        close_timeout_seconds: owner 栈关闭宽限(T16 语义)。

    Raises:
        ValueError: 配置路径/信任门/YAML/条目校验失败(装配错误,不是单 server
            失败——没有可信配置就没有可隔离的 server)。
    """
    root = workspace.expanduser().resolve()
    snapshot = policy if policy is not None else PermissionPolicy()
    config_path = resolve_mcp_config_path(root)
    if config_path is None:
        return McpBundle((), ())
    configs = load_mcp_configs(root, config_path)
    resolved = resolve_policy(snapshot, root)
    permissions = PermissionContext(
        workspace=str(root),
        policy=snapshot,
        mode=snapshot.approval,
        sandbox=probe_backend(),
    )
    verdict_for = network_verdict_at_connect or (lambda url: network_decision(resolved, url))

    def sha_provider() -> str:
        """调用期重读配置字节(T17 ``run`` 的漂移基线来源)。"""
        return config_sha256(config_path)

    tools: list[Tool] = []
    statuses: list[McpServerStatus] = []
    stacks: list[AsyncExitStack] = []
    specs: list[SandboxSpec] = []

    for config in configs:
        transport: Literal["stdio", "http"] = (
            "http" if isinstance(config, HttpServerConfig) else "stdio"
        )
        skip_reason = _skip_reason(config, resolved)
        if skip_reason is not None:
            statuses.append(
                McpServerStatus(
                    server_name=config.name,
                    transport=transport,
                    connected=False,
                    diagnostic_code=_SKIPPED,
                    diagnostic_message=skip_reason,
                )
            )
            continue
        stack: AsyncExitStack | None = None
        spec: SandboxSpec | None = None
        try:
            connection, stack, spec = await _open_one(
                config,
                root=root,
                permissions=permissions,
                sandbox_compiler=sandbox_compiler,
                env_builder=env_builder,
                network_verdict_at_connect=verdict_for,
                close_timeout_seconds=close_timeout_seconds,
            )
            server_tools = mcp_tools_for_connection(
                connection, config, config_sha_provider=sha_provider
            )
        except Exception as error:  # noqa: BLE001 — 逐 server 隔离
            if stack is not None:
                await _close_local(stack, spec, root)
            secrets = _config_secrets(config)

            def redact(text: str, _secrets: tuple[str, ...] = secrets) -> str:
                return _redact(text, _secrets)

            message = diagnostic_message(
                error,
                fallback="MCP server unavailable",
                redact=redact,
            )
            statuses.append(
                McpServerStatus(
                    server_name=config.name,
                    transport=transport,
                    connected=False,
                    diagnostic_code=_UNAVAILABLE,
                    diagnostic_message=message,
                )
            )
            logger.warning("MCP server %s unavailable: %s", config.name, message)
            continue
        stacks.append(stack)
        if spec is not None:
            specs.append(spec)
        tools.extend(server_tools)
        statuses.append(
            McpServerStatus(
                server_name=config.name,
                transport=transport,
                connected=True,
                tool_count=len(server_tools),
            )
        )

    return McpBundle(
        tuple(tools),
        tuple(statuses),
        _stacks=tuple(stacks),
        _specs=tuple(specs),
        _workspace=root,
    )


def _skip_reason(config: McpServerConfig, resolved: ResolvedPolicy) -> str | None:
    """连接前策略预检(Q4):返回可执行跳过文案,None = 可连接。

    ``resolved`` 是 ``resolve_policy`` 的产物;本函数只消费 ``protocol.policy``
    的同一代数,不新建决策入口。所有插值都经 ``bounded_message``(repair #4):
    server 名/通配/URL 都不得让诊断无界,整体 ≤512(与 unavailable 路径同形)。
    """
    bounded_name = bounded_message(config.name, fallback="?")
    wildcard_denied = tool_decision(resolved, server_wildcard(config.name))
    if wildcard_denied == "deny":
        wildcard = bounded_message(server_wildcard(config.name), fallback="mcp__*")
        return bounded_message(
            f"MCP server {bounded_name!r} is denied by the permission policy "
            f"({wildcard}); remove the deny rule to enable it",
            fallback="MCP server skipped: denied by the permission policy",
        )
    if not config.allow_all_tools:
        decisions = [
            tool_decision(resolved, canonical_tool_name(config.name, tool))
            for tool in config.tools_allowlist
        ]
        if decisions and all(decision == "deny" for decision in decisions):
            normalized = bounded_message(normalize_mcp_name(config.name), fallback="?")
            return bounded_message(
                f"every allowlisted tool of MCP server {bounded_name!r} is denied by the "
                "permission policy; grant at least one mcp__"
                f"{normalized}__<tool> rule to enable it",
                fallback="MCP server skipped: all allowlisted tools are denied",
            )
    if isinstance(config, HttpServerConfig):
        verdict = network_decision(resolved, config.url)
        if verdict != "allow":
            return bounded_message(
                f"MCP server {bounded_name!r} requires network allow for its endpoint at "
                f"connect time (got {verdict}); add a NetworkRule with decision allow for "
                "the endpoint host, then reconnect (Q4 discovery preauthorization)",
                fallback="MCP server skipped: endpoint network is not preauthorized",
            )
    return None


async def _open_one(
    config: McpServerConfig,
    *,
    root: Path,
    permissions: PermissionContext,
    sandbox_compiler: SandboxCompiler,
    env_builder: EnvBuilder,
    network_verdict_at_connect: Callable[[str], PolicyVerdict],
    close_timeout_seconds: float,
) -> tuple[McpConnection, AsyncExitStack, SandboxSpec | None]:
    """连接一个 server;成功时栈/spec 的所有权交给调用方(失败由调用方清理)。"""
    stack = AsyncExitStack()
    spec: SandboxSpec | None = None
    try:
        if isinstance(config, StdioServerConfig):
            spec = sandbox_compiler(permissions, _sandbox_use(config, root), root)
            effective = replace(config, cwd=config.cwd or str(root))
            connection = await stack.enter_async_context(
                owned_stdio_connection(
                    effective,
                    workspace=root,
                    sandbox_spec=spec,
                    env_builder=env_builder,
                    close_timeout_seconds=close_timeout_seconds,
                )
            )
        else:
            connection = await stack.enter_async_context(
                owned_http_connection(
                    config,
                    workspace=root,
                    network_verdict_at_connect=network_verdict_at_connect,
                    close_timeout_seconds=close_timeout_seconds,
                )
            )
    except BaseException:
        # 失败只关自己的栈(含已编译的 spec);调用方收到的异常不带泄漏资源。
        await _close_local(stack, spec, root)
        raise
    return connection, stack, spec


def _sandbox_use(config: StdioServerConfig, root: Path) -> ResolvedUse:
    """R-PF-5 合成 use:派生 server 进程要 EXECUTE_PROCESS + READ_WORKSPACE。

    targets 点名真实命令与 cwd(缺省取 workspace,T16 交接 ③):沙箱 profile
    的路径授权因此与"实际会读什么/从哪启动"同源,而不是无目标的空编译。
    """
    return ResolvedUse(
        effects=frozenset({ToolEffect.EXECUTE_PROCESS, ToolEffect.READ_WORKSPACE}),
        targets=(
            ToolTarget(kind="path", value=config.command),
            ToolTarget(kind="path", value=config.cwd or str(root)),
        ),
    )


async def _close_local(
    stack: AsyncExitStack, spec: SandboxSpec | None, workspace: Path
) -> None:
    """单个 server 的局部收尾(失败隔离:绝不抛给其余 server)。"""
    try:
        await stack.aclose()
    except Exception as error:  # noqa: BLE001 — 失败栈的清理同样容忍
        # 不记 exc_info:原始异常文本可能携带配置展开值(repair #2)。
        logger.warning("MCP failed-server teardown failed: %s", type(error).__name__)
    if spec is not None:
        try:
            finalize_sandbox(spec, workspace)
        except Exception as error:  # noqa: BLE001 — finalize 自身不抛
            logger.warning("MCP failed-server sandbox cleanup failed: %s", type(error).__name__)


def _config_secrets(config: McpServerConfig) -> tuple[str, ...]:
    """配置里**已展开**的值(env/header/bearer env 值),仅用于诊断 redact。"""
    if isinstance(config, StdioServerConfig):
        values = list(config.env.values())
    else:
        values = list(config.headers.values())
        if config.bearer_token_env_var is not None:
            bearer = os.environ.get(config.bearer_token_env_var)
            if bearer:
                values.append(bearer)
    return tuple(value for value in values if isinstance(value, str) and value)


def _redact(message: str, secrets: Iterable[str]) -> str:
    """从诊断文本里抹掉已知配置秘密值(长于门槛的整词替换)。

    修复轮 #2:同时替换**原始形态**与**空白折叠形态**——含换行/多空格的秘密
    在 collapse 之后只剩后者,单形态匹配会漏。调用点在 ``_errors`` 里先对
    原始 leaf 文本、再对 collapse 后的文本各执行一次,跨越 512 截断边界的
    秘密因此不会留下前缀。
    """
    redacted = message
    for secret in sorted(secrets, key=len, reverse=True):
        if len(secret) < _REDACT_MIN_LENGTH:
            continue
        for form in dict.fromkeys((secret, " ".join(secret.split()))):
            if form and form in redacted:
                redacted = redacted.replace(form, "[redacted]")
    return redacted


__all__ = ["McpBundle", "McpServerStatus", "connect_mcp_servers"]

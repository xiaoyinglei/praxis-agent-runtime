"""MCP descriptor → praxis ``Tool`` 暴露(legacy ``:116-169`` 映射;T17)。

**每工具一张 descriptor**(旧 ``MCPToolDescriptor`` 去掉 ``approval_profile`` /
``execution_revision`` / ``connection_revision``):名字、描述、输入 schema、
三个 hint、HTTP endpoint,以及 **schema 指纹**。

**effects / targets 映射**(controller 钉死的值):

- ``static_effects``:HTTP server 的工具 ``{NETWORK}``,stdio server 的工具
  **空集**——server 进程已在 T16 沙箱内,调用只是 client RPC;两条路径都再
  叠加 ``DESTRUCTIVE``(``destructiveHint`` 为真时)。
- ``targets``:恒有 ``ToolTarget("mcp_tool", canonical_name)``;HTTP 工具再加
  ``ToolTarget("url", config.url)``,让 ``decide`` 用**当次** live policy 对
  endpoint 做 network 裁决(Q3 的完整闭环)。
- ``idempotent = idempotentHint``(默认 False);``concurrency_safe =
  not destructiveHint``;``timeout_seconds=30.0``;``max_model_output_bytes=
  1_000_000``;``cancellation_mode=REMOTE_BEST_EFFORT``;
  ``interrupt_behavior=CANCEL``(legacy 姿态);``source=ToolSource.MCP``。

**调用期守卫 = 唯一两处**(Q3/R-T17-1):

1. (a) 配置字节 sha256 漂移:factory 构造时经 ``config_sha_provider`` 捕获
   基线,``run`` 每次重读;不一致 ⇒ ``mcp_config_changed``
   ("config changed; reconnect")。这是**重连问题**,不是权限问题。
2. (b) catalog 成员缺失:``run`` 对照 factory 时的目录快照,未知
   canonical 名 ⇒ ``mcp_tool_unknown``。

brief 原有的 (b) "live schema 指纹漂移 ⇒ ``mcp_tool_changed``" 在
**R-T17-1 收敛下不实现**:praxis 没有 live schema 通道(不重列就看不见
server 的新 schema),跨 server 的 schema 变化只能由重连承接(config sha +
T19 重启语义);"schema 变化后旧批准仍有效"的风险由 decide + 审批绑定的
``args_digest``/``policy_revision`` 在**管道层**兜住。descriptor 仍计算
``schema_fingerprint``(canonical JSON 的 sha256),供 T19 重连/审计消费,
但 ``run`` 不读它——这里不新增第二条决策路径。

**allowlist / read_only**:缺失的 allowlist 工具在**构造期**抛 ``ValueError``
(启动错误,legacy ``runtime/mcp.py:242-246``);HTTP ``read_only=True`` 的
server 只暴露 ``readOnlyHint and not destructiveHint`` 的工具
(legacy ``:257-262``)。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from mcp.types import Tool as McpTool

from praxis.adapters.mcp.config import HttpServerConfig, McpServerConfig
from praxis.adapters.mcp.lifecycle import McpConnection
from praxis.adapters.mcp.names import canonical_tool_name
from praxis.adapters.mcp.output import normalize_mcp_output
from praxis.protocol.json import JsonValue, freeze_json, thaw_json
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    ToolSource,
    ToolTarget,
    ToolValidationError,
    json_schema_input,
)

_TOOL_TIMEOUT_SECONDS = 30.0
_MAX_MODEL_OUTPUT_BYTES = 1_000_000
_CONFIG_CHANGED_MESSAGE = "config changed; reconnect"
_MAX_ERROR_MESSAGE_CHARS = 512
_MAX_LEAF_MESSAGES = 8
_DEFAULT_DESCRIPTION = "Configured MCP tool."
_PENDING_TOOL_CALL_ID = "pending"


@dataclass(frozen=True, slots=True)
class McpToolDescriptor:
    """一个已发现 MCP 工具的自包含描述(不携带策略/修订字段)。"""

    server_name: str
    tool_name: str
    description: str
    input_schema: Mapping[str, JsonValue]
    read_only_hint: bool = False
    destructive_hint: bool = False
    idempotent_hint: bool = False
    endpoint_url: str | None = None
    canonical_name: str = field(init=False)
    schema_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("server_name", "tool_name", "description"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"MCP tool {name} must be a string")
        for name in ("read_only_hint", "destructive_hint", "idempotent_hint"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"MCP tool {name} must be a bool")
        schema = freeze_json(self.input_schema, path="input_schema")
        if not isinstance(schema, Mapping):
            raise TypeError("MCP tool input_schema must be a mapping")
        object.__setattr__(self, "input_schema", schema)
        object.__setattr__(
            self,
            "canonical_name",
            canonical_tool_name(self.server_name, self.tool_name),
        )
        object.__setattr__(self, "schema_fingerprint", _schema_fingerprint(schema))


def mcp_tools_for_connection(
    connection: McpConnection,
    config: McpServerConfig,
    *,
    config_sha_provider: Callable[[], str],
) -> tuple[Tool, ...]:
    """把一条已就绪连接的工具目录投影为 praxis ``Tool`` 元组。

    Parameters:
        connection: T16/T18 产出的已就绪连接;``tools`` 是**完整**目录
            (T17 的 ``catalog.list_all_tools`` 已分页取全)。
        config: 该 server 的已校验配置(allowlist/read_only/transport 判定)。
        config_sha_provider: 当前配置字节 sha256 的提供者(每次 ``run`` 重读)。
            factory 调用即捕获基线,之后的漂移由 ``run`` 拒绝。

    Raises:
        ValueError: allowlist 点名的工具不在目录中,或 canonical 名重复
            (启动错误;legacy 同文案/同边界)。
    """
    config_sha_at_connect = config_sha_provider()
    catalog = tuple(connection.tools)
    missing = config.tools_allowlist - {item.name for item in catalog}
    if missing:
        raise ValueError(
            f"MCP server {config.name!r} is missing allowlisted tools: "
            + ", ".join(sorted(missing))
        )

    endpoint_url = config.url if isinstance(config, HttpServerConfig) else None
    descriptors: list[McpToolDescriptor] = []
    names: set[str] = set()
    for item in catalog:
        if not config.allow_all_tools and item.name not in config.tools_allowlist:
            continue
        annotations = item.annotations
        read_only = bool(getattr(annotations, "readOnlyHint", False)) and not bool(
            getattr(annotations, "destructiveHint", False)
        )
        if isinstance(config, HttpServerConfig) and config.read_only and not read_only:
            continue
        descriptor = _descriptor_for(item, config, endpoint_url=endpoint_url)
        if descriptor.canonical_name in names:
            raise ValueError(f"duplicate canonical MCP tool name: {descriptor.canonical_name}")
        names.add(descriptor.canonical_name)
        descriptors.append(descriptor)

    catalog_names = frozenset(descriptor.canonical_name for descriptor in descriptors)
    return tuple(
        _build_tool(
            descriptor,
            connection=connection,
            catalog_names=catalog_names,
            config_sha_provider=config_sha_provider,
            config_sha_at_connect=config_sha_at_connect,
        )
        for descriptor in descriptors
    )


def _descriptor_for(
    item: McpTool,
    config: McpServerConfig,
    *,
    endpoint_url: str | None,
) -> McpToolDescriptor:
    annotations = item.annotations
    description = item.description or _DEFAULT_DESCRIPTION
    return McpToolDescriptor(
        server_name=config.name,
        tool_name=item.name,
        description=description.strip()
        or f"Call {item.name} on the configured MCP server {config.name}.",
        input_schema=cast("Mapping[str, JsonValue]", item.inputSchema),
        read_only_hint=bool(getattr(annotations, "readOnlyHint", False)),
        destructive_hint=bool(getattr(annotations, "destructiveHint", False)),
        idempotent_hint=bool(getattr(annotations, "idempotentHint", False)),
        endpoint_url=endpoint_url,
    )


def _build_tool(
    descriptor: McpToolDescriptor,
    *,
    connection: McpConnection,
    catalog_names: frozenset[str],
    config_sha_provider: Callable[[], str],
    config_sha_at_connect: str,
) -> Tool:
    effects: set[ToolEffect] = set()
    if descriptor.endpoint_url is not None:
        effects.add(ToolEffect.NETWORK)
    if descriptor.destructive_hint:
        effects.add(ToolEffect.DESTRUCTIVE)
    static_effects = frozenset(effects)

    targets: tuple[ToolTarget, ...] = (
        ToolTarget(kind="mcp_tool", value=descriptor.canonical_name),
    )
    if descriptor.endpoint_url is not None:
        targets = (
            *targets,
            ToolTarget(kind="url", value=descriptor.endpoint_url),
        )

    def resolve_use(_arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        # 目录成员校验:factory 只投影目录内工具,这里仍 fail-closed——工具对象
        # 若脱离原目录(重连/替换)则不再产生可执行 use。
        if descriptor.canonical_name not in catalog_names:
            raise ToolValidationError(
                path="$",
                message=f"MCP tool is not in the connected catalog: {descriptor.canonical_name}",
            )
        return ResolvedUse(effects=static_effects, targets=targets)

    async def run(
        arguments: Mapping[str, JsonValue],
        _exec_ctx: ToolExecContext,
    ) -> ToolResult:
        # Q3 唯一两处调用期守卫(见模块 docstring):配置漂移在前,目录成员在后。
        drift = _config_drift_result(
            descriptor,
            config_sha_provider=config_sha_provider,
            config_sha_at_connect=config_sha_at_connect,
        )
        if drift is not None:
            return drift
        if descriptor.canonical_name not in catalog_names:
            return _guard_result(
                descriptor,
                "mcp_tool_unknown",
                f"MCP tool is not in the connected catalog: {descriptor.canonical_name}",
            )
        try:
            raw = await connection.session.call_tool(
                descriptor.tool_name,
                arguments=dict(arguments),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — SDK/传输异常一律归一为调用失败
            return _guard_result(descriptor, "mcp_call_failed", _call_failure_message(exc))
        try:
            return normalize_mcp_output(
                raw,
                server_name=descriptor.server_name,
                tool_name=descriptor.tool_name,
            )
        except Exception as exc:  # noqa: BLE001 — 畸形输出不得炸穿管道
            return _guard_result(
                descriptor,
                "mcp_call_failed",
                f"invalid MCP tool output ({type(exc).__name__})",
            )

    return Tool(
        definition=ToolDefinition(
            name=descriptor.canonical_name,
            description=descriptor.description,
            input_schema=descriptor.input_schema,
        ),
        validate_input=json_schema_input(descriptor.input_schema),
        run=run,
        static_effects=static_effects,
        resolve_use=resolve_use,
        idempotent=descriptor.idempotent_hint,
        concurrency_safe=not descriptor.destructive_hint,
        cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=_TOOL_TIMEOUT_SECONDS,
        max_model_output_bytes=_MAX_MODEL_OUTPUT_BYTES,
        # Q3:MCP never ships permission callbacks; the single decide owns policy.
        check_permission=None,
        source=ToolSource.MCP,
    )


def _config_drift_result(
    descriptor: McpToolDescriptor,
    *,
    config_sha_provider: Callable[[], str],
    config_sha_at_connect: str,
) -> ToolResult | None:
    try:
        current = config_sha_provider()
    except Exception:  # noqa: BLE001 — 读失败同样 fail-closed 到重连
        return _guard_result(descriptor, "mcp_config_changed", _CONFIG_CHANGED_MESSAGE)
    if current != config_sha_at_connect:
        return _guard_result(descriptor, "mcp_config_changed", _CONFIG_CHANGED_MESSAGE)
    return None


def _guard_result(descriptor: McpToolDescriptor, error_code: str, message: str) -> ToolResult:
    bounded = _bounded(message)
    return ToolResult(
        tool_call_id=_PENDING_TOOL_CALL_ID,
        tool_name=descriptor.canonical_name,
        content=(
            ToolContentBlock(type="text", data={"text": f"{error_code}: {bounded}"}),
        ),
        status=ResultStatus.ERROR,
        is_error=True,
        error_code=error_code,
        error_message=bounded,
        metadata={
            "mcp_server": descriptor.server_name,
            "mcp_tool": descriptor.tool_name,
        },
    )


def _call_failure_message(error: BaseException) -> str:
    """摊平 ExceptionGroup 的叶子消息并截断——绝不带出 TaskGroup 包装文案。

    T16 的实测形状是 ``ExceptionGroup(ExceptionGroup(TimeoutError))``;T19 在
    connect 阶段负责把同一形状转诊断,工具层这里只保证调用失败可读且有界。
    """
    leaves = _leaves(error)[:_MAX_LEAF_MESSAGES]
    detail = "; ".join(f"{type(leaf).__name__}: {leaf}" for leaf in leaves)
    return _bounded(detail or "MCP tool call failed")


def _leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for sub in error.exceptions for leaf in _leaves(sub)]
    return [error]


def _bounded(message: str) -> str:
    collapsed = " ".join(message.split())
    return collapsed[:_MAX_ERROR_MESSAGE_CHARS] or "MCP tool call failed"


def _schema_fingerprint(schema: Mapping[str, JsonValue]) -> str:
    """canonical JSON(键序无关)的 sha256;descriptor 计算,run 不重查。"""
    payload = json.dumps(thaw_json(schema), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["McpToolDescriptor", "mcp_tools_for_connection"]
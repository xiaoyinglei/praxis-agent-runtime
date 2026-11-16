"""子代理模型工具面五件套(Q4 裁决;设计 §5.13/§7.4;阶段4 T6)。

五件:``spawn_subagent`` / ``wait_subagent`` / ``send_to_subagent`` /
``cancel_subagent`` / ``integrate_child_workspace``。旧 ``task`` 同步壳
(spawn+wait 复制)按"不留双实现"不带走——同步语义用 spawn+wait 两步表达。

effects 裁决(不新增效果枚举值,YAGNI):
- 四件编排声明 ``{}``——纯编排,子的副作用各自过子 session 的管道审批;
  ``enable_subagents=True`` 时 composition 自动附加这四件的 **session 域
  ToolRule allow**(预算份额与子限额才是闸门;对照旧表钉为有意差异)。
  session 域 allow 只活在内存,绝不进 always 落盘。
- ``integrate_child_workspace`` 声明 ``DESTRUCTIVE + WRITE_WORKSPACE``
  (S5:破坏性副作用永不自动 allow,显式审批不可豁免);T8 实装
  inspect/apply/discard,本任务交付 schema + ``integrate_not_wired`` 诚实
  错误码(R-PF-2 先例)。

注入纪律(仿 EnvBuilder 先例):tools 不 import core——行为经
``protocol.agents.SubagentSurface`` 结构签名触达,composition 以 provider
闭包延迟绑定 ``SubagentManager``(manager 在 Session 之后构造,与 web
authorizer 的 session_holder 同先例)。

args 上限(旧 integrations/subagent.py:52-59 parity):task/context_summary
≤20_000;wait timeout ≤60(manager 钳位同值)。错误码面:``grant_exhausted``
(ChildGrantError)/``child_limit_reached``(ChildLimitError)/
``child_not_found``(未知 child 的 ValueError)/``integrate_not_wired``;
其余异常(如终态子再 cancel、write 模式 provisioner 未接线)如实上抛,
由管道唯一归一为 ``tool_error``(消息自解释,不发明第二套码)。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from praxis.protocol.agents import (
    ChildGrantError,
    ChildLimitError,
    SurfaceProvider,
)
from praxis.protocol.budget import Limits, limits_payload
from praxis.protocol.json import JsonValue
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
    ToolTarget,
    json_schema_input,
)
from praxis.protocol.workspaces import WorkspaceMode
from praxis.tools.builtins.filesystem import PENDING_TOOL_CALL_ID

# 旧 subagent.py:52-59 parity:task/context_summary 的有界字符上限。
MAX_TASK_CHARS = 20_000
MAX_CONTEXT_SUMMARY_CHARS = 20_000
MAX_STEERING_CHARS = 20_000
# wait 的最长等待(manager 钳位同值);工具超时须覆盖它。
MAX_WAIT_TIMEOUT_SECONDS = 60.0

_SPAWN_NAME = "spawn_subagent"
_WAIT_NAME = "wait_subagent"
_SEND_NAME = "send_to_subagent"
_CANCEL_NAME = "cancel_subagent"
_INTEGRATE_NAME = "integrate_child_workspace"

# child_thread_id 是 uuid4().hex(32 字符);上限留余量但拒绝无界输入。
_MAX_CHILD_ID_CHARS = 64

_SPAWN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_TASK_CHARS,
            "description": "Bounded task delegated to an isolated child agent.",
        },
        "context_summary": {
            "type": "string",
            "maxLength": MAX_CONTEXT_SUMMARY_CHARS,
            "description": "Only the parent context needed to complete this task.",
        },
        "max_tokens": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Requested child model-token ceiling; granted only from this turn's "
                "remaining budget (at most half per dimension), never minted."
            ),
        },
        "max_cost_micros": {
            "type": "integer",
            "minimum": 1,
            "description": "Requested child monetary ceiling in integer micro-dollars.",
        },
        "max_steps": {
            "type": "integer",
            "minimum": 1,
            "description": "Requested child step ceiling (may only narrow the share).",
        },
        "workspace_mode": {
            "type": "string",
            "enum": ["read_only", "write"],
            "description": (
                "read_only (default) shares the parent workspace with writes vetoed; "
                "write isolates the child in a git worktree snapshot (requires a git workspace)."
            ),
        },
    },
    "required": ["task"],
    "additionalProperties": False,
}

_WAIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "child_thread_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": _MAX_CHILD_ID_CHARS,
            "description": "child_thread_id returned by spawn_subagent.",
        },
        "timeout": {
            "type": "number",
            "minimum": 0,
            "maximum": MAX_WAIT_TIMEOUT_SECONDS,
            "description": "Seconds to wait (0..60, default 10); expiry does not cancel the child.",
        },
    },
    "required": ["child_thread_id"],
    "additionalProperties": False,
}

_SEND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "child_thread_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": _MAX_CHILD_ID_CHARS,
            "description": "child_thread_id returned by spawn_subagent.",
        },
        "message": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_STEERING_CHARS,
            "description": "Steering text the child reads at its next step boundary.",
        },
    },
    "required": ["child_thread_id", "message"],
    "additionalProperties": False,
}

_CANCEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "child_thread_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": _MAX_CHILD_ID_CHARS,
            "description": "child_thread_id returned by spawn_subagent.",
        },
    },
    "required": ["child_thread_id"],
    "additionalProperties": False,
}

_INTEGRATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "child_thread_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": _MAX_CHILD_ID_CHARS,
            "description": "child_thread_id of a write-mode subagent.",
        },
        "action": {
            "type": "string",
            "enum": ["inspect", "apply", "discard"],
            "description": "inspect the child diff, apply it to the parent, or discard the snapshot.",
        },
        "paths": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 4096},
            "maxItems": 1000,
            "description": "Optional relative-path subset to integrate (default: all changes).",
        },
    },
    "required": ["child_thread_id", "action"],
    "additionalProperties": False,
}

_validate_spawn = json_schema_input(_SPAWN_SCHEMA)
_validate_wait = json_schema_input(_WAIT_SCHEMA)
_validate_send = json_schema_input(_SEND_SCHEMA)
_validate_cancel = json_schema_input(_CANCEL_SCHEMA)
_validate_integrate = json_schema_input(_INTEGRATE_SCHEMA)


def _text_result(name: str, text: str, metadata: Mapping[str, JsonValue]) -> ToolResult:
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name=name,
        content=(ToolContentBlock(type="text", data={"text": text}),),
        status=ResultStatus.OK,
        metadata=metadata,
    )


def _error(name: str, code: str, message: str) -> ToolResult:
    """固定有界错误结果(与 skills 激活错误同形状);消息进模型可见文本。"""
    bounded = message[:512]
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name=name,
        content=(ToolContentBlock(type="text", data={"text": f"{code}: {bounded}"}),),
        status=ResultStatus.ERROR,
        is_error=True,
        error_code=code,
        error_message=bounded,
    )


def _child_id(arguments: Mapping[str, JsonValue]) -> str:
    raw = arguments.get("child_thread_id")
    return raw if isinstance(raw, str) else ""


def _requested_limits(arguments: Mapping[str, JsonValue]) -> Limits | None:
    """schema 已保证 integer≥1(bool 被 jsonschema integer 拒);缺键 = None。"""
    dims: dict[str, int] = {}
    for arg_name, dim in (("max_tokens", "tokens"), ("max_cost_micros", "cost_micros"), ("max_steps", "steps")):
        raw = arguments.get(arg_name)
        if isinstance(raw, int) and not isinstance(raw, bool):
            dims[dim] = raw
    return Limits(**dims) if dims else None


def _unknown_child(exc: ValueError) -> bool:
    """manager._ref_for 的未知 child 消息前缀(与终态拒绝的 ValueError 区分)。"""
    return str(exc).startswith("unknown child thread")


def create_agent_tools(surface_provider: SurfaceProvider) -> tuple[Tool, ...]:
    """装配五件套(provider 延迟绑定 SubagentManager;composition 唯一调用点)。

    返回序稳定:spawn/wait/send/cancel/integrate(默认表尾追加,口径 15+5)。
    """

    async def spawn_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        surface = surface_provider()
        raw_task = arguments.get("task")
        task = raw_task if isinstance(raw_task, str) else ""
        raw_summary = arguments.get("context_summary")
        summary = raw_summary if isinstance(raw_summary, str) else ""
        raw_mode = arguments.get("workspace_mode", "read_only")
        mode = cast("WorkspaceMode", raw_mode if isinstance(raw_mode, str) else "read_only")
        try:
            handle = await surface.spawn(
                task,
                context_summary=summary,
                requested=_requested_limits(arguments),
                workspace_mode=mode,
            )
        except ChildGrantError as exc:
            return _error(_SPAWN_NAME, "grant_exhausted", str(exc))
        except ChildLimitError as exc:
            return _error(_SPAWN_NAME, "child_limit_reached", str(exc))
        grant = limits_payload(handle.grant)
        return _text_result(
            _SPAWN_NAME,
            f"spawned subagent {handle.child_thread_id} "
            f"(workspace_mode={handle.workspace_mode}, workspace={handle.workspace})\n"
            f"grant: {json.dumps(grant, sort_keys=True)}\n"
            "Use wait_subagent to collect its conclusion; the child keeps running until then.",
            {
                "child_thread_id": handle.child_thread_id,
                "child_root_turn_id": handle.child_root_turn_id,
                "allocation_id": handle.allocation_id,
                "workspace": str(handle.workspace),
                "workspace_mode": handle.workspace_mode,
                "grant": grant,
            },
        )

    async def wait_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        surface = surface_provider()
        child_id = _child_id(arguments)
        raw_timeout = arguments.get("timeout", 10.0)
        timeout = float(raw_timeout) if isinstance(raw_timeout, (int, float)) else 10.0
        try:
            status = await surface.wait(child_id, timeout=timeout)
        except ValueError as exc:
            return _error(_WAIT_NAME, "child_not_found", str(exc))
        approvals: tuple[JsonValue, ...] = tuple(
            {
                "call_id": summary.call_id,
                "tool_name": summary.tool_name,
                "args_digest": summary.args_digest,
                "effects": tuple(summary.effects),
            }
            for summary in status.pending_approvals
        )
        text = f"subagent {child_id} state={status.state}\n{status.conclusion}"
        if approvals:
            text += (
                f"\npending approvals: {len(approvals)} "
                "(a human must resume the child session directly; this tool never approves)"
            )
        return _text_result(
            _WAIT_NAME,
            text,
            {
                "child_thread_id": child_id,
                "state": status.state,
                "conclusion": status.conclusion,
                "pending_approvals": approvals,
                "unknown_calls": tuple(status.unknown_calls),
                "steps_used": status.steps_used,
                "cost_used": status.cost_used,
                "workspace": status.workspace,
            },
        )

    async def send_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        surface = surface_provider()
        child_id = _child_id(arguments)
        raw_message = arguments.get("message")
        message = raw_message if isinstance(raw_message, str) else ""
        try:
            await surface.send(child_id, message)
        except ValueError as exc:
            return _error(_SEND_NAME, "child_not_found", str(exc))
        return _text_result(
            _SEND_NAME,
            f"steering delivered to subagent {child_id} (read at its next step boundary)",
            {"child_thread_id": child_id},
        )

    async def cancel_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        surface = surface_provider()
        child_id = _child_id(arguments)
        try:
            convergence = surface.request_cancel(child_id)
        except ValueError as exc:
            if _unknown_child(exc):
                return _error(_CANCEL_NAME, "child_not_found", str(exc))
            raise  # 终态拒绝等:如实上抛,管道唯一归一 tool_error
        return _text_result(
            _CANCEL_NAME,
            f"cancel {convergence} for subagent {child_id}",
            {"child_thread_id": child_id, "convergence": convergence},
        )

    async def integrate_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        # T8 实装 inspect/apply/discard(provisioner 交付管道);本任务诚实拒绝,
        # 零副作用(R-PF-2 先例)。schema 校验仍先行(管道 invalid_input 纪律)。
        return _error(
            _INTEGRATE_NAME,
            "integrate_not_wired",
            "integrate_child_workspace is not wired yet (T8 delivery pipeline); no action was performed",
        )

    def orchestration_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        return ResolvedUse(frozenset(), (ToolTarget(kind="subagent", value=_child_id(arguments)),))

    def integrate_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        return ResolvedUse(
            frozenset({ToolEffect.DESTRUCTIVE, ToolEffect.WRITE_WORKSPACE}),
            (ToolTarget(kind="subagent", value=_child_id(arguments)),),
        )

    spawn = Tool(
        definition=ToolDefinition(
            name=_SPAWN_NAME,
            description=(
                "Spawn an isolated subagent for a bounded task and return its child_thread_id "
                "immediately (does not wait). The child runs in its own thread with a budget "
                "share carved from this turn's remaining budget (at most half per dimension); "
                "it cannot spend beyond that share. Use wait_subagent to collect its conclusion. "
                "workspace_mode read_only (default) shares this workspace with writes vetoed; "
                "write requires a git workspace provisioner."
            ),
            input_schema=_SPAWN_SCHEMA,
        ),
        validate_input=_validate_spawn,
        run=spawn_run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(frozenset(), ()),
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=60.0,
        max_model_output_bytes=64_000,
    )
    wait = Tool(
        definition=ToolDefinition(
            name=_WAIT_NAME,
            description=(
                "Wait up to timeout seconds (0..60, default 10) for a subagent to finish and "
                "return its state, conclusion, and pending approvals. Expiry does NOT cancel "
                "the child; it keeps running. paused/interrupted children need a human to "
                "resume their own session; this tool never approves on their behalf."
            ),
            input_schema=_WAIT_SCHEMA,
        ),
        validate_input=_validate_wait,
        run=wait_run,
        static_effects=frozenset(),
        resolve_use=orchestration_use,
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=65.0,
        max_model_output_bytes=64_000,
    )
    send = Tool(
        definition=ToolDefinition(
            name=_SEND_NAME,
            description=(
                "Append a steering message to a subagent's thread; the child reads it at its "
                "next step boundary. Does not interrupt the child's current step."
            ),
            input_schema=_SEND_SCHEMA,
        ),
        validate_input=_validate_send,
        run=send_run,
        static_effects=frozenset(),
        resolve_use=orchestration_use,
        idempotent=False,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=10.0,
        max_model_output_bytes=4_000,
    )
    cancel = Tool(
        definition=ToolDefinition(
            name=_CANCEL_NAME,
            description=(
                "Request cancellation of a subagent. Returns the convergence path: signalled "
                "(in-process driver injected) or queued (item appended; the child executor "
                "converges at its step boundary). Terminal children cannot be cancelled."
            ),
            input_schema=_CANCEL_SCHEMA,
        ),
        validate_input=_validate_cancel,
        run=cancel_run,
        static_effects=frozenset(),
        resolve_use=orchestration_use,
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=10.0,
        max_model_output_bytes=4_000,
    )
    integrate = Tool(
        definition=ToolDefinition(
            name=_INTEGRATE_NAME,
            description=(
                "Inspect, apply, or discard a write-mode subagent's workspace changes. "
                "Destructive: explicit approval is always required (never auto-allowed). "
                "Delivery pipeline lands in T8; until then this tool honestly reports "
                "integrate_not_wired and performs no action."
            ),
            input_schema=_INTEGRATE_SCHEMA,
        ),
        validate_input=_validate_integrate,
        run=integrate_run,
        static_effects=frozenset({ToolEffect.DESTRUCTIVE, ToolEffect.WRITE_WORKSPACE}),
        resolve_use=integrate_use,
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=60.0,
        max_model_output_bytes=256_000,
    )
    return (spawn, wait, send, cancel, integrate)

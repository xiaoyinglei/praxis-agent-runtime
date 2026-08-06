from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from agent_runtime.core.human_input import HumanInputRequest, ToolCallSummary
from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic
from agent_runtime.knowledge import AgentCitation, AgentEvidence, agent_citation_from_value, agent_evidence_from_value
from agent_runtime.modeling.contracts import LLMUsage
from agent_runtime.planning import AgentPlan, PlanEvent
from agent_runtime.tools.tool import JsonValue, ToolResult

if TYPE_CHECKING:
    from agent_runtime.service import AgentRunResult

type AgentResultStatus = Literal["done", "paused", "failed"]
type AgentPauseKind = Literal[
    "tool_approval",
    "tool_reconciliation",
    "choice",
    "clarification",
]
type AgentDiagnosticSeverity = Literal["warning", "error"]


def _empty_json_mapping() -> Mapping[str, JsonValue]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, JsonValue] | None = None
    structured_output: JsonValue | None = None
    is_error: bool = False
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    truncated: bool = False
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        if self.arguments is not None:
            object.__setattr__(
                self,
                "arguments",
                _freeze_json_mapping(self.arguments),
            )
        if self.structured_output is not None:
            object.__setattr__(
                self,
                "structured_output",
                _freeze_json_value(self.structured_output),
            )


@dataclass(frozen=True, slots=True)
class AgentDiagnostic:
    code: str
    component: str
    message: str
    severity: AgentDiagnosticSeverity = "warning"
    degraded: bool = True
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class AgentToolSummary:
    tool_call_id: str
    tool_name: str
    args_preview: str
    approval_id: str | None = None
    risk_level: str = "low"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AgentPause:
    request_id: str
    kind: AgentPauseKind
    question: str
    tool_calls: tuple[AgentToolSummary, ...] = ()
    options: tuple[str, ...] = ()
    context: Mapping[str, JsonValue] = field(default_factory=_empty_json_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _freeze_json_mapping(self.context))


@dataclass(frozen=True, slots=True)
class AgentUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    model_calls: int = 0
    latency_ms: float = 0.0
    logical_input_tokens: int | None = None
    uncached_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    usage_source: str | None = None
    startup_ms: float = 0.0
    build_service_ms: float = 0.0
    model_ready_ms: float = 0.0
    prepare_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    finalize_latency_ms: float = 0.0
    prompt_bytes: int = 0
    tool_schema_bytes: int = 0
    native_calls: int = 0
    native_errors: int = 0
    native_latency_ms_total: float = 0.0
    deferred_calls: int = 0
    mcp_calls: int = 0
    mcp_errors: int = 0
    mcp_latency_ms_total: float = 0.0


@dataclass(frozen=True, slots=True)
class AgentResult:
    answer: str | None
    status: AgentResultStatus
    files: tuple[str, ...]
    tool_calls: tuple[AgentToolCall, ...]
    evidence: tuple[AgentEvidence, ...]
    citations: tuple[AgentCitation, ...]
    usage: AgentUsage
    diagnostics: tuple[AgentDiagnostic, ...]
    turn_id: str
    stop_reason: str | None
    pause: AgentPause | None
    workspace_path: str | None
    groundedness: bool
    insufficient_evidence: bool
    plan: AgentPlan | None
    plan_events: tuple[PlanEvent, ...]
    needs_user_input: str | None = None

    @classmethod
    def _from_internal(
        cls,
        result: AgentRunResult,
        *,
        files: tuple[str, ...] = (),
    ) -> AgentResult:
        status = _project_status(result.status)
        arguments_by_id = _tool_call_arguments(result)
        projected_files = files or tuple(result.input_files)
        return cls(
            answer=result.final_answer,
            status=status,
            files=projected_files,
            tool_calls=tuple(_project_tool_call(tool, arguments_by_id=arguments_by_id) for tool in result.tool_results),
            evidence=tuple(_project_evidence(item) for item in result.evidence),
            citations=tuple(_project_citation(item) for item in result.citations),
            usage=_project_usage(result),
            diagnostics=tuple(_project_diagnostic(item) for item in result.runtime_diagnostics),
            turn_id=result.turn_id,
            stop_reason=result.stop_reason,
            pause=_project_pause(result.human_input_request),
            workspace_path=result.workspace_path,
            groundedness=result.groundedness_flag,
            insufficient_evidence=result.insufficient_evidence_flag,
            plan=(None if result.plan is None else result.plan.model_copy(deep=True)),
            plan_events=tuple(event.model_copy(deep=True) for event in result.plan_events),
            needs_user_input=result.needs_user_input,
        )


def _project_status(value: str) -> AgentResultStatus:
    if value not in {"done", "paused", "failed"}:
        raise ValueError(f"unsupported internal Agent result status: {value!r}")
    return cast(AgentResultStatus, value)


def _tool_call_arguments(
    result: AgentRunResult,
) -> Mapping[str, Mapping[str, JsonValue]]:
    arguments = result.tool_call_arguments
    return MappingProxyType({} if arguments is None else arguments)


def _project_tool_call(
    result: ToolResult,
    *,
    arguments_by_id: Mapping[str, Mapping[str, JsonValue]],
) -> AgentToolCall:
    arguments = arguments_by_id.get(result.tool_call_id)
    return AgentToolCall(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        arguments=arguments,
        structured_output=result.structured_content,
        is_error=result.is_error,
        error_code=result.error_code,
        error_message=result.error_message,
        retryable=result.retryable,
        truncated=result.truncated,
        latency_ms=None,
    )


def _project_evidence(item: object) -> AgentEvidence:
    return agent_evidence_from_value(item)


def _project_citation(item: object) -> AgentCitation:
    return agent_citation_from_value(item)


def _project_diagnostic(item: RuntimeDiagnostic) -> AgentDiagnostic:
    return AgentDiagnostic(
        code=item.code,
        component=item.component,
        message=item.message,
        severity=item.severity,
        degraded=item.degraded,
        error_type=item.error_type,
    )


def _project_pause(value: HumanInputRequest | None) -> AgentPause | None:
    if value is None:
        return None
    return AgentPause(
        request_id=value.request_id,
        kind=value.kind,
        question=value.question,
        tool_calls=tuple(_project_tool_summary(item) for item in value.tool_calls),
        options=tuple(value.options),
        context=_freeze_json_mapping(value.context),
    )


def _project_tool_summary(item: ToolCallSummary) -> AgentToolSummary:
    return AgentToolSummary(
        tool_call_id=item.tool_call_id,
        approval_id=item.approval_id,
        tool_name=item.tool_name,
        args_preview=item.args_preview,
        risk_level=item.risk_level,
        reason=item.reason,
    )


def _project_usage(result: AgentRunResult) -> AgentUsage:
    usages = tuple(record.usage for record in result.model_call_records)
    latency = result.latency_profile
    metrics = result.tool_call_metrics
    return AgentUsage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
        tool_calls=len(result.tool_results),
        model_calls=len(result.model_call_records),
        latency_ms=0.0 if latency is None else latency.total_ms,
        logical_input_tokens=_sum_optional(tuple(usage.logical_input_tokens for usage in usages)),
        uncached_input_tokens=_sum_optional(tuple(usage.uncached_input_tokens for usage in usages)),
        cache_read_input_tokens=_sum_optional(tuple(usage.cache_read_input_tokens for usage in usages)),
        cache_write_input_tokens=_sum_optional(tuple(usage.cache_write_input_tokens for usage in usages)),
        usage_source=_usage_source(usages),
        startup_ms=0.0 if latency is None else latency.startup_ms,
        build_service_ms=(0.0 if latency is None else latency.build_service_ms),
        model_ready_ms=0.0 if latency is None else latency.model_ready_ms,
        prepare_latency_ms=(0.0 if latency is None else latency.prepare_latency_ms),
        model_latency_ms=0.0 if latency is None else latency.model_latency_ms,
        tool_latency_ms=0.0 if latency is None else latency.tool_latency_ms,
        finalize_latency_ms=(0.0 if latency is None else latency.finalize_latency_ms),
        prompt_bytes=0 if latency is None else latency.prompt_bytes,
        tool_schema_bytes=(0 if latency is None else latency.tool_schema_bytes),
        native_calls=0 if metrics is None else metrics.native_calls,
        native_errors=0 if metrics is None else metrics.native_errors,
        native_latency_ms_total=(0.0 if metrics is None else metrics.native_latency_ms_total),
        deferred_calls=0 if metrics is None else metrics.deferred_calls,
        mcp_calls=0 if metrics is None else metrics.mcp_calls,
        mcp_errors=0 if metrics is None else metrics.mcp_errors,
        mcp_latency_ms_total=(0.0 if metrics is None else metrics.mcp_latency_ms_total),
    )


def _sum_optional(values: tuple[int | None, ...]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _usage_source(usages: tuple[LLMUsage, ...]) -> str | None:
    values = tuple(usage.usage_source for usage in usages if usage.usage_source is not None)
    if not values:
        return None
    return values[0] if len(set(values)) == 1 else "mixed"


def _freeze_json_mapping(
    value: Mapping[str, object],
) -> Mapping[str, JsonValue]:
    return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})


def _freeze_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("public result JSON values must be finite")
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if any(not isinstance(key, str) for key in mapping):
            raise TypeError("public result JSON object keys must be strings")
        return MappingProxyType({cast(str, key): _freeze_json_value(item) for key, item in mapping.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sequence = cast(Sequence[object], value)
        return tuple(_freeze_json_value(item) for item in sequence)
    raise TypeError(f"public result dynamic value is not JSON-compatible: {type(value).__name__}")


__all__ = [
    "AgentCitation",
    "AgentDiagnosticSeverity",
    "AgentDiagnostic",
    "AgentEvidence",
    "AgentPause",
    "AgentPauseKind",
    "AgentPlan",
    "AgentResult",
    "AgentResultStatus",
    "AgentToolCall",
    "AgentToolSummary",
    "AgentUsage",
    "JsonValue",
    "PlanEvent",
]

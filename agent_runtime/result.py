from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from agent_runtime.knowledge import AgentCitation, AgentEvidence, agent_evidence_from_value
from agent_runtime.planning import AgentPlan, PlanEvent
from agent_runtime.tools.tool import JsonValue

if TYPE_CHECKING:
    from agent_runtime.harness.protocol import TurnResult
    from agent_runtime.harness.rollout import ItemSnapshot, RolloutStore

type AgentResultStatus = Literal["done", "paused", "failed"]
type AgentPauseKind = Literal[
    "tool_approval",
    "tool_reconciliation",
    "choice",
    "clarification",
    "model_retry",
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
    thread_id: str = ""

    @classmethod
    def _from_harness(
        cls,
        result: TurnResult,
        *,
        store: RolloutStore,
        files: tuple[str, ...] = (),
    ) -> AgentResult:
        turn = store.read_turn(result.turn_id)
        thread = store.read_thread(result.thread_id)
        attempts = tuple(
            attempt
            for operation in store.list_model_operations(result.turn_id)
            for attempt in store.list_model_attempts(operation.operation_id)
            if attempt.status == "completed"
        )
        usage_values = tuple(attempt.usage for attempt in attempts)
        sources = tuple(
            value
            for usage in usage_values
            if isinstance((value := usage.get("usage_source")), str)
        )
        items = store.list_items(result.turn_id)
        arguments_by_id = {
            str(item.payload["tool_call_id"]): item.payload.get("arguments")
            for item in items
            if item.kind == "tool_call"
            and isinstance(item.payload.get("tool_call_id"), str)
            and isinstance(item.payload.get("arguments"), Mapping)
        }
        projected_calls: list[AgentToolCall] = []
        evidence: list[AgentEvidence] = []
        citations: list[AgentCitation] = []
        groundedness = False
        insufficient_evidence = False
        for item in items:
            if item.kind != "tool_result":
                continue
            payload = item.payload
            tool_call_id = payload.get("tool_call_id")
            tool_name = payload.get("tool_name")
            if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                continue
            raw_structured = payload.get("structured_content")
            structured = (
                None
                if raw_structured is None
                else _freeze_json_value(raw_structured)
            )
            projected_calls.append(
                AgentToolCall(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=cast(
                        Mapping[str, JsonValue] | None,
                        arguments_by_id.get(tool_call_id),
                    ),
                    structured_output=structured,
                    is_error=payload.get("is_error") is True,
                    error_code=(
                        value
                        if isinstance((value := payload.get("error_code")), str)
                        else None
                    ),
                    error_message=(
                        value
                        if isinstance((value := payload.get("error_message")), str)
                        else None
                    ),
                    retryable=payload.get("retryable") is True,
                    truncated=payload.get("truncated") is True,
                )
            )
            if tool_name != "search_knowledge" or not isinstance(
                raw_structured, Mapping
            ):
                continue
            groundedness = raw_structured.get("groundedness_flag") is True
            insufficient_evidence = (
                raw_structured.get("insufficient_evidence") is True
            )
            raw_evidence = raw_structured.get("results")
            if isinstance(raw_evidence, Sequence) and not isinstance(
                raw_evidence, (str, bytes)
            ):
                for value in raw_evidence:
                    if isinstance(value, Mapping):
                        evidence.append(agent_evidence_from_value(value))
            evidence_by_anchor = {
                item.citation_anchor: item for item in evidence
            }
            raw_citations = raw_structured.get("citations")
            if isinstance(raw_citations, Sequence) and not isinstance(
                raw_citations, (str, bytes)
            ):
                for value in raw_citations:
                    if not isinstance(value, str):
                        continue
                    source = evidence_by_anchor.get(value)
                    citations.append(
                        AgentCitation(
                            citation_id=value,
                            evidence_id=("" if source is None else source.evidence_id),
                            record_type=(
                                "knowledge"
                                if source is None or source.source_type is None
                                else source.source_type
                            ),
                            file_name=(None if source is None else source.file_name),
                            citation_anchor=value,
                            doc_id=(None if source is None else source.doc_id),
                            source_type=(None if source is None else source.source_type),
                        )
                    )
        pause: AgentPause | None = None
        pending = tuple(
            interaction
            for interaction in store.list_interactions(result.turn_id)
            if interaction.status == "pending"
        )
        if len(pending) == 1 and pending[0].kind in {
            "tool_approval",
            "tool_reconciliation",
            "choice",
            "clarification",
        }:
            interaction = pending[0]
            reason = interaction.request.get("reason")
            requested_question = interaction.request.get("question")
            question = (
                reason
                if isinstance(reason, str) and reason
                else requested_question
                if isinstance(requested_question, str) and requested_question
                else f"Input required for {interaction.kind}"
            )
            summaries: tuple[AgentToolSummary, ...] = ()
            if interaction.operation_id is not None:
                operation = store.read_tool_operation(interaction.operation_id)
                arguments = arguments_by_id.get(operation.tool_call_id)
                effects = interaction.request.get("effects")
                summaries = (
                    AgentToolSummary(
                        tool_call_id=operation.tool_call_id,
                        tool_name=operation.tool_name,
                        args_preview=(
                            "" if arguments is None else str(dict(arguments))[:1000]
                        ),
                        approval_id=interaction.request_id,
                        risk_level=("high" if effects else "low"),
                        reason=question,
                    ),
                )
            pause = AgentPause(
                request_id=interaction.request_id,
                kind=cast(AgentPauseKind, interaction.kind),
                question=question,
                tool_calls=summaries,
                options=_interaction_options(interaction.kind, interaction.request),
                context=cast(Mapping[str, JsonValue], interaction.request),
            )
        plan, plan_events = _project_harness_plan(items, turn_status=turn.status)
        if pause is None and turn.status in {"paused", "interrupted"}:
            unknown = [op for op in store.list_model_operations(result.turn_id) if op.status == "unknown"]
            if len(unknown) == 1:
                pause = AgentPause(
                    request_id=unknown[0].operation_id,
                    kind="model_retry",
                    question="模型响应中断。可重试本次模型请求，或结束这一轮。",
                    options=("retry", "abort"),
                    context={"operation_id": unknown[0].operation_id},
                )
        unknown_model_diagnostics = tuple(
            record
            for record in store.list_records(thread.thread_id)
            if record.turn_id == turn.turn_id
            and record.record_type == "model_attempt_unknown"
            and isinstance(record.payload.get("error_type"), str)
            and isinstance(record.payload.get("error_message"), str)
        )
        incomplete_model_responses = tuple(
            item
            for item in items
            if item.kind == "model_response"
            and item.payload.get("response_status") == "incomplete"
            and isinstance(item.payload.get("incomplete_reason"), str)
        )
        rejected_model_diagnostics = tuple(
            record
            for record in store.list_records(thread.thread_id)
            if record.turn_id == turn.turn_id
            and record.record_type == "model_attempt_rejected"
            and isinstance(record.payload.get("error_type"), str)
            and isinstance(record.payload.get("error_message"), str)
        )
        projected_diagnostics: list[AgentDiagnostic] = []
        if unknown_model_diagnostics:
            projected_diagnostics.append(
                AgentDiagnostic(
                    code="model_dispatch_outcome_unknown",
                    component="model",
                    message=str(
                        unknown_model_diagnostics[-1].payload["error_message"]
                    ),
                    severity="warning",
                    degraded=True,
                    error_type=str(
                        unknown_model_diagnostics[-1].payload["error_type"]
                    ),
                ),
            )
        if rejected_model_diagnostics:
            rejected = rejected_model_diagnostics[-1]
            projected_diagnostics.append(
                AgentDiagnostic(
                    code="model_dispatch_rejected",
                    component="model",
                    message=str(rejected.payload["error_message"]),
                    severity="error",
                    degraded=True,
                    error_type=str(rejected.payload["error_type"]),
                )
            )
        if incomplete_model_responses:
            reason = str(incomplete_model_responses[-1].payload["incomplete_reason"])
            projected_diagnostics.append(
                AgentDiagnostic(
                    code="model_response_incomplete",
                    component="model",
                    message=f"Model response was incomplete: {reason}.",
                    severity="error",
                    degraded=True,
                    error_type="IncompleteModelResponse",
                )
            )
        if (
            turn.status == "failed"
            and turn.terminal_reason_code is not None
            and turn.terminal_message is not None
            and not any(
                diagnostic.code == turn.terminal_reason_code
                for diagnostic in projected_diagnostics
            )
        ):
            projected_diagnostics.insert(
                0,
                AgentDiagnostic(
                    code=turn.terminal_reason_code,
                    component="runtime",
                    message=turn.terminal_message,
                    severity="error",
                    degraded=True,
                ),
            )
        diagnostics = tuple(projected_diagnostics)
        return cls(
            answer=result.answer,
            status={
                "completed": "done",
                "paused": "paused",
            }.get(result.status, "failed"),  # type: ignore[arg-type]
            files=files,
            tool_calls=tuple(projected_calls),
            evidence=tuple(evidence),
            citations=tuple(citations),
            usage=AgentUsage(
                input_tokens=sum(_usage_integer(usage, "input_tokens") for usage in usage_values),
                output_tokens=sum(_usage_integer(usage, "output_tokens") for usage in usage_values),
                total_tokens=sum(_usage_integer(usage, "total_tokens") for usage in usage_values),
                tool_calls=len(projected_calls),
                model_calls=len(attempts),
                usage_source=(
                    None
                    if not sources
                    else sources[0] if len(set(sources)) == 1 else "mixed"
                ),
            ),
            diagnostics=diagnostics,
            turn_id=turn.turn_id,
            stop_reason=turn.terminal_reason_code or turn.status,
            pause=pause,
            workspace_path=thread.workspace,
            groundedness=groundedness,
            insufficient_evidence=insufficient_evidence,
            plan=plan,
            plan_events=plan_events,
            needs_user_input=None if pause is None else pause.question,
            thread_id=thread.thread_id,
        )


def _project_harness_plan(
    items: Sequence[ItemSnapshot],
    *,
    turn_status: str,
) -> tuple[AgentPlan | None, tuple[PlanEvent, ...]]:
    del turn_status
    plan_items = [
        item
        for item in items
        if item.kind == "plan_state"
        and item.status == "completed"
        and isinstance(item.payload.get("plan"), Mapping)
    ]
    if not plan_items:
        return None, ()

    events: list[PlanEvent] = []
    for item in plan_items:
        raw_plan = cast(Mapping[str, object], item.payload["plan"])
        raw_event = item.payload.get("event")
        event = raw_event if isinstance(raw_event, Mapping) else {}
        revision = raw_plan.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise RuntimeError("canonical plan_state revision is malformed")
        raw_event_type = event.get("event_type")
        event_type: Literal[
            "initialized",
            "llm_update",
            "decision_progress",
            "observation_progress",
            "needs_replan",
            "completed",
            "blocked",
        ]
        if raw_event_type in {
            "initialized",
            "llm_update",
            "decision_progress",
            "observation_progress",
            "needs_replan",
            "completed",
            "blocked",
        }:
            event_type = cast(
                Literal[
                    "initialized",
                    "llm_update",
                    "decision_progress",
                    "observation_progress",
                    "needs_replan",
                    "completed",
                    "blocked",
                ],
                raw_event_type,
            )
        else:
            plan_status = raw_plan.get("status")
            if plan_status == "complete":
                event_type = "completed"
            elif plan_status == "blocked":
                event_type = "blocked"
            else:
                event_type = "llm_update"
        message = event.get("message")
        related_step_id = event.get("related_step_id")
        tool_call_ids = event.get("tool_call_ids")
        events.append(
            PlanEvent(
                event_id=f"plan_event_{item.item_id}",
                event_type=event_type,
                plan_revision=revision,
                message=(
                    message
                    if isinstance(message, str) and message
                    else "Applied canonical plan transition."
                ),
                related_step_id=(
                    related_step_id if isinstance(related_step_id, str) else None
                ),
                tool_call_ids=(
                    [str(value) for value in tool_call_ids]
                    if isinstance(tool_call_ids, Sequence)
                    and not isinstance(tool_call_ids, (str, bytes))
                    else []
                ),
            )
        )

    canonical = cast(Mapping[str, object], plan_items[-1].payload["plan"])
    raw_steps = canonical.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise RuntimeError("canonical plan_state steps are malformed")
    plan = AgentPlan.model_validate({**dict(canonical), "steps": list(raw_steps)})
    return plan, tuple(events)


def _usage_integer(usage: Mapping[str, object], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _interaction_options(
    kind: str,
    request: Mapping[str, object],
) -> tuple[str, ...]:
    if kind == "tool_approval":
        return ("approve", "deny")
    if kind != "choice":
        return ()
    options = request.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        return ()
    return tuple(option for option in options if isinstance(option, str))


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

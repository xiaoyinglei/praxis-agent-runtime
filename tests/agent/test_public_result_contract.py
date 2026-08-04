from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields

import pytest

import agent_runtime.result as result_module
from agent_runtime.core.human_input import HumanInputRequest, ToolCallSummary
from agent_runtime.core.model_request import ModelCallRecord
from agent_runtime.core.runtime_diagnostics import (
    AgentLatencyProfile,
    RuntimeDiagnostic,
    ToolCallMetrics,
)
from agent_runtime.planning import AgentPlan, PlanEvent, PlanStep
from agent_runtime.result import (
    AgentCitation,
    AgentDiagnostic,
    AgentEvidence,
    AgentPause,
    AgentResult,
    AgentToolCall,
    AgentToolSummary,
    AgentUsage,
)
from agent_runtime.service import AgentRunResult
from agent_runtime.tools.tool import ToolResult
from rag.schema.llm import LLMUsage
from rag.schema.query import AnswerCitation, EvidenceItem, GroundingTarget


def test_public_result_dtos_are_frozen_and_have_the_stable_surface() -> None:
    assert tuple(field.name for field in fields(AgentResult)) == (
        "answer",
        "status",
        "files",
        "tool_calls",
        "evidence",
        "citations",
        "usage",
        "diagnostics",
        "turn_id",
        "stop_reason",
        "pause",
        "workspace_path",
        "groundedness",
        "insufficient_evidence",
        "plan",
        "plan_events",
    )
    assert not hasattr(AgentResult, "run_id")
    assert "Any" not in inspect.getsource(result_module)

    frozen_values = (
        AgentToolCall(tool_call_id="call-1", tool_name="read_file"),
        AgentEvidence(
            evidence_id="evidence-1",
            doc_id=1,
            citation_anchor="report#summary",
            text="Revenue increased.",
            score=0.9,
        ),
        AgentCitation(
            citation_id="citation-1",
            evidence_id="evidence-1",
            record_type="section",
        ),
        AgentDiagnostic(
            code="degraded_provider",
            component="provider",
            message="Fallback used.",
        ),
        AgentToolSummary(
            tool_call_id="call-1",
            tool_name="read_file",
            args_preview="path='README.md'",
        ),
        AgentPause(
            request_id="request-1",
            kind="tool_approval",
            question="Allow read_file?",
        ),
        AgentUsage(),
    )
    for value in frozen_values:
        with pytest.raises(FrozenInstanceError):
            value.__setattr__(next(iter(value.__dataclass_fields__)), "changed")


def test_internal_projection_builds_stable_result_dtos_without_internal_objects() -> None:
    first_tool = ToolResult(
        tool_call_id="call-read",
        tool_name="read_file",
        structured_content={"path": "README.md", "lines": ("one", "two")},
    )
    second_tool = ToolResult(
        tool_call_id="call-search",
        tool_name="knowledge_search",
        is_error=True,
        error_code="not_found",
        error_message="No matching source.",
        retryable=True,
        truncated=True,
    )
    evidence = EvidenceItem(
        evidence_id="evidence-1",
        doc_id=7,
        benchmark_doc_id="benchmark-7",
        source_id=3,
        citation_anchor="report#summary",
        text="Revenue increased.",
        score=0.91,
        evidence_kind="retrieved",
        record_type="section",
        file_name="report.pdf",
        section_path=["Summary", "Revenue"],
        page_start=2,
        page_end=3,
        source_type="pdf",
        retrieval_channels=["dense", "lexical"],
        retrieval_family="hybrid",
        grounding_target=GroundingTarget(
            kind="section",
            doc_id=7,
            source_id=3,
            section_id=9,
            page_start=2,
            page_end=3,
            section_path=["Summary", "Revenue"],
            raw_locator={"section": "revenue"},
        ),
    )
    citation = AnswerCitation(
        citation_id="citation-1",
        file_name="report.pdf",
        section_path=["Summary", "Revenue"],
        page_start=2,
        page_end=3,
        evidence_id="evidence-1",
        record_type="section",
        citation_anchor="report#summary",
        doc_id=7,
        benchmark_doc_id="benchmark-7",
        source_id=3,
        source_type="pdf",
    )
    diagnostic = RuntimeDiagnostic(
        code="provider_fallback",
        component="model",
        message="Used fallback provider.",
        severity="warning",
        degraded=True,
        error_type="ProviderUnavailable",
    )
    human_request = HumanInputRequest(
        request_id="request-approval",
        kind="tool_approval",
        question="Allow read_file?",
        tool_calls=[
            ToolCallSummary(
                tool_call_id="call-read",
                approval_id="approval-read",
                tool_name="read_file",
                args_preview="path='README.md'",
                risk_level="low",
                reason="Read the requested file.",
            )
        ],
        options=["allow_once", "deny"],
        context={"policy": {"effects": ["read"]}, "attempt": 1},
    )
    plan = AgentPlan(
        objective="Read and summarize the report.",
        steps=[PlanStep(step_id="step-read", title="Read the report.")],
    )
    plan_event = PlanEvent(
        event_id="event-read",
        event_type="initialized",
        plan_revision=0,
        message="Plan initialized.",
    )
    raw = AgentRunResult(
        turn_id="turn-1",
        status="paused",
        final_answer="Partial answer.",
        stop_reason="approval_required",
        tool_results=[first_tool, second_tool],
        tool_call_arguments={
            "call-read": {"path": "README.md", "line_end": 20},
        },
        model_call_records=[
            _model_call_record(
                request_id="model-request-1",
                input_tokens=20,
                output_tokens=5,
                logical_input_tokens=20,
                uncached_input_tokens=14,
                cache_read_input_tokens=4,
                cache_write_input_tokens=2,
            ),
            _model_call_record(
                request_id="model-request-2",
                input_tokens=10,
                output_tokens=3,
                logical_input_tokens=10,
                uncached_input_tokens=8,
                cache_read_input_tokens=2,
                cache_write_input_tokens=0,
            ),
        ],
        evidence=[evidence],
        citations=[citation],
        groundedness_flag=True,
        insufficient_evidence_flag=False,
        human_input_request=human_request,
        workspace_path="/workspace",
        runtime_diagnostics=[diagnostic],
        tool_call_metrics=ToolCallMetrics(
            native_calls=1,
            native_errors=0,
            native_latency_ms_total=4.5,
            deferred_calls=1,
            mcp_calls=2,
            mcp_errors=1,
            mcp_latency_ms_total=8.5,
        ),
        latency_profile=AgentLatencyProfile(
            startup_ms=1.0,
            build_service_ms=2.0,
            model_ready_ms=3.0,
            prepare_latency_ms=4.0,
            model_latency_ms=5.0,
            tool_latency_ms=6.0,
            finalize_latency_ms=7.0,
            total_ms=28.0,
            prompt_bytes=2048,
            tool_schema_bytes=4096,
        ),
        plan=plan,
        plan_events=[plan_event],
    )

    public = AgentResult._from_internal(raw, files=("report.pdf",))

    assert public.answer == "Partial answer."
    assert public.status == "paused"
    assert public.files == ("report.pdf",)
    assert public.turn_id == "turn-1"
    assert not hasattr(public, "session_id")
    assert public.stop_reason == "approval_required"
    assert public.workspace_path == "/workspace"
    assert public.groundedness is True
    assert public.insufficient_evidence is False
    assert not hasattr(public, "raw")
    assert not hasattr(public, "thread_id")
    assert not hasattr(public, "run_id")

    assert [tool.tool_call_id for tool in public.tool_calls] == [
        "call-read",
        "call-search",
    ]
    assert public.tool_calls[0].tool_name == "read_file"
    assert public.tool_calls[0].arguments == {
        "path": "README.md",
        "line_end": 20,
    }
    assert public.tool_calls[0].structured_output == {
        "path": "README.md",
        "lines": ("one", "two"),
    }
    assert public.tool_calls[0].latency_ms is None
    assert public.tool_calls[1].arguments is None
    assert public.tool_calls[1].is_error is True
    assert public.tool_calls[1].error_code == "not_found"
    assert public.tool_calls[1].error_message == "No matching source."
    assert public.tool_calls[1].retryable is True
    assert public.tool_calls[1].truncated is True

    assert public.evidence[0].section_path == ("Summary", "Revenue")
    assert public.evidence[0].retrieval_channels == ("dense", "lexical")
    assert public.evidence[0].grounding_target == {
        "kind": "section",
        "doc_id": 7,
        "source_id": 3,
        "section_id": 9,
        "asset_id": None,
        "page_start": 2,
        "page_end": 3,
        "section_path": ("Summary", "Revenue"),
        "raw_locator": {"section": "revenue"},
    }
    assert public.citations[0].citation_id == "citation-1"
    assert public.citations[0].section_path == ("Summary", "Revenue")
    assert public.diagnostics[0].code == "provider_fallback"

    assert public.pause is not None
    assert public.pause.request_id == "request-approval"
    assert public.pause.tool_calls == (
        AgentToolSummary(
            tool_call_id="call-read",
            approval_id="approval-read",
            tool_name="read_file",
            args_preview="path='README.md'",
            risk_level="low",
            reason="Read the requested file.",
        ),
    )
    assert public.pause.options == ("allow_once", "deny")
    assert public.pause.context == {
        "policy": {"effects": ("read",)},
        "attempt": 1,
    }

    assert public.usage == AgentUsage(
        input_tokens=30,
        output_tokens=8,
        total_tokens=38,
        tool_calls=2,
        model_calls=2,
        latency_ms=28.0,
        logical_input_tokens=30,
        uncached_input_tokens=22,
        cache_read_input_tokens=6,
        cache_write_input_tokens=2,
        usage_source="provider",
        startup_ms=1.0,
        build_service_ms=2.0,
        model_ready_ms=3.0,
        prepare_latency_ms=4.0,
        model_latency_ms=5.0,
        tool_latency_ms=6.0,
        finalize_latency_ms=7.0,
        prompt_bytes=2048,
        tool_schema_bytes=4096,
        native_calls=1,
        native_errors=0,
        native_latency_ms_total=4.5,
        deferred_calls=1,
        mcp_calls=2,
        mcp_errors=1,
        mcp_latency_ms_total=8.5,
    )
    assert public.plan == plan
    assert public.plan is not plan
    assert public.plan_events == (plan_event,)
    assert public.plan_events[0] is not plan_event

    with pytest.raises(FrozenInstanceError):
        public.__setattr__("answer", "changed")
    assert public.tool_calls[0].arguments is not None
    with pytest.raises(TypeError):
        public.tool_calls[0].arguments["path"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        public.pause.context["attempt"] = 2  # type: ignore[index]


def test_public_result_restores_input_files_from_resumed_turn() -> None:
    raw = AgentRunResult(
        turn_id="resumed-turn",
        status="done",
        final_answer="done",
        input_files=["/tmp/report.pdf"],
    )

    public = AgentResult._from_internal(raw)

    assert public.files == ("/tmp/report.pdf",)


def test_internal_projection_preserves_the_single_turn_identity() -> None:
    result = AgentRunResult(
        turn_id="turn-root",
        status="done",
        tool_results=[ToolResult(tool_call_id="legacy-call", tool_name="read_file")],
    )
    continuation = result.model_copy(update={"turn_id": "turn-continuation"})
    legacy_without_arguments = result.model_copy(
        update={"tool_call_arguments": None}
    )

    assert AgentResult._from_internal(result).turn_id == "turn-root"
    assert AgentResult._from_internal(result).tool_calls[0].arguments is None
    assert (
        AgentResult._from_internal(legacy_without_arguments)
        .tool_calls[0]
        .arguments
        is None
    )
    assert AgentResult._from_internal(continuation).turn_id == "turn-continuation"
    assert not hasattr(AgentResult._from_internal(result), "session_id")


def _model_call_record(
    *,
    request_id: str,
    input_tokens: int,
    output_tokens: int,
    logical_input_tokens: int,
    uncached_input_tokens: int,
    cache_read_input_tokens: int,
    cache_write_input_tokens: int,
) -> ModelCallRecord:
    return ModelCallRecord(
        request_id=request_id,
        prompt_revision=f"prompt-{request_id}",
        toolset_revision=f"tools-{request_id}",
        provider_wire_hash=f"wire-{request_id}",
        usage=LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            source="provider",
            logical_input_tokens=logical_input_tokens,
            uncached_input_tokens=uncached_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            usage_source="provider",
        ),
    )

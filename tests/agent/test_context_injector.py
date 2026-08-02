from __future__ import annotations

import inspect

from langchain_core.messages import HumanMessage

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.messages import ModelMessage, tool_result_message
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import LoopState, PendingToolCall, create_loop_state
from rag.agent.memory.injector import ContextBuilder
from rag.agent.memory.models import ExtractedFact, MemoryRef, WorkingSummary
from rag.agent.tools.tool import ToolContentBlock, ToolResult


class _CharacterTokenAccounting:
    def count(self, text: str) -> int:
        return len(text)

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str:
        clipped = text[: max(token_budget, 0)]
        if add_ellipsis and len(clipped) < len(text) and token_budget >= 4:
            return clipped[: token_budget - 4].rstrip() + " ..."
        return clipped


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="System prompt",
        allowed_tools=["search_text"],
    )


def _state() -> LoopState:
    state = create_loop_state(
        current_message="Explain policy",
        run_config=AgentRunConfig(
            turn_id="ctx",
            llm_budget_total=1000,
        ),
        messages=[HumanMessage(content="recent tail", id="h-tail")],
    )
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc1",
            tool_name="search_text",
            is_error=True,
            error_code="tool_not_implemented",
            error_message="not wired",
        )
    ]
    state["memory_state"].working_summary = WorkingSummary(
        summary="Prior working summary",
        covered_message_ids=["h1"],
        updated_at="2026-05-08T00:00:00Z",
        token_count=3,
    )
    state["memory_state"].extracted_facts = [
        ExtractedFact(
            fact_id="f1",
            text="Memory fact",
            evidence_ids=["ev1"],
        )
    ]
    return state


def test_context_sections_follow_authority_order() -> None:
    context = ContextBuilder(max_context_tokens=1000).assemble_loop(
        definition=_definition(),
        state=_state(),
    )

    names = [section.name for section in context.sections]
    assert names[:3] == ["system", "task", "tool_results"]
    assert names.index("tool_results") < names.index("message_tail")
    rendered = context.as_text()
    assert "ev1" in rendered
    assert "tool_call_id=tc1" in rendered
    assert '"error_code":"tool_not_implemented"' in rendered


def test_model_visible_conversation_uses_only_canonical_transcript() -> None:
    state = _state()
    state["conversation_history"] = [
        ModelMessage(role="user", content="canonical prior user"),
        ModelMessage(role="assistant", content="canonical prior answer"),
    ]
    state["turn_transcript"] = [
        ModelMessage(role="user", content="canonical current request"),
    ]
    state["messages"] = [
        HumanMessage(content="FORGED LEGACY MESSAGE", id="legacy-forged"),
    ]
    state["memory_state"].working_summary = WorkingSummary(
        summary="FORGED LEGACY SUMMARY",
        covered_message_ids=["legacy-forged"],
        updated_at="2026-05-08T00:00:00Z",
        token_count=3,
    )

    context = ContextBuilder(max_context_tokens=2_000).assemble_loop(
        definition=_definition(),
        state=state,
    )

    rendered = context.as_text()
    assert "canonical prior user" in rendered
    assert "canonical prior answer" in rendered
    assert "canonical current request" in rendered
    assert "FORGED LEGACY MESSAGE" not in rendered
    assert "FORGED LEGACY SUMMARY" not in rendered


def test_tool_result_uses_fixed_canonical_content_without_formatter() -> None:
    state = _state()
    path = "input_files/销售情况对标  区域双周会.xlsx"
    result = ToolResult(
        tool_call_id="tc-read",
        tool_name="read_file",
        content=(ToolContentBlock(type="text", data={"text": path}),),
        structured_content={"path": path, "truncated": False},
    )
    state["tool_results"] = [result]

    context = ContextBuilder(max_context_tokens=2000).assemble_loop(
        definition=_definition(),
        state=state,
    )

    canonical_content = tool_result_message(result).content
    assert canonical_content in context.section("tool_results").content
    assert path in context.section("tool_results").content
    assert "formatter_resolver" not in inspect.signature(ContextBuilder).parameters


def test_context_injects_memory_refs_without_raw_paths() -> None:
    state = _state()
    state["memory_state"].memory_refs = [
        MemoryRef(
            ref_id="mem_big",
            path=".agent_memory/records/mem_big.json",
            summary="command produced total=42",
            source_tool_call_id="tc-command",
            source_tool_name="run_command",
            size_bytes=9999,
        )
    ]

    context = ContextBuilder(max_context_tokens=1000).assemble_loop(
        definition=_definition(),
        state=state,
    )

    memory = context.section("memory").content
    assert "mem_big" in memory
    assert "total=42" in memory
    assert ".agent_memory/records/mem_big.json" not in memory


def test_pending_decisions_are_kept_ahead_of_optional_context() -> None:
    state = _state()
    state["messages"] = [HumanMessage(content="tail " * 200, id="tail")]
    state["pending_tool_calls"] = [
        PendingToolCall(
            plan=ToolCallPlan.create("search_text", {"query": "policy"}),
            status="pending",
        )
    ]

    context = ContextBuilder(max_context_tokens=350).assemble_loop(
        definition=_definition(),
        state=state,
    )

    names = [section.name for section in context.sections]
    assert "open_decisions" in names
    if "message_tail" not in names:
        assert "message_tail" in context.context_budget.dropped_sections


def test_context_builder_uses_injected_token_accounting() -> None:
    accounting = _CharacterTokenAccounting()
    state = _state()
    state["tool_results"] = []
    state["messages"] = []

    context = ContextBuilder(
        max_context_tokens=500,
        token_accounting=accounting,
    ).assemble_loop(
        definition=_definition(),
        state=state,
    )

    assert context.context_budget.used_context_tokens == accounting.count(context.as_text())


def test_required_overflow_is_explicit_and_never_hashes_content() -> None:
    state = _state()
    state["tool_results"] = []
    state["messages"] = []
    definition = AgentRuntimePolicy.test_factory(
        system_prompt="SYSTEM_REAL_CONTENT",
        allowed_tools=[],
    )

    context = ContextBuilder(
        max_context_tokens=20,
        token_accounting=_CharacterTokenAccounting(),
        max_section_chars=10_000,
    ).assemble_loop(
        definition=definition,
        state=state,
    )

    assert context.context_budget.overflow is True
    assert "system" in context.context_budget.required_truncated
    assert all("sha256=" not in section.content for section in context.sections)

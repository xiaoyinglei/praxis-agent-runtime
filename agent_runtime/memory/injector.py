from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.messages import (
    ModelMessage,
    canonical_json_text,
    model_message_payload,
    tool_result_message,
)
from agent_runtime.memory.models import (
    ContextBudgetSnapshot,
    ContextSection,
    ContextSectionName,
    InjectedContext,
    MemoryRef,
)
from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract
from agent_runtime.tools.tool import ToolResult

if TYPE_CHECKING:
    from agent_runtime.loop.state import LoopState

    type ContextState = LoopState


@dataclass
class _ContextSelection:
    sections: list[ContextSection]
    dropped_sections: list[ContextSectionName] = field(default_factory=list)
    summarized_sections: list[ContextSectionName] = field(default_factory=list)
    required_truncated: list[ContextSectionName] = field(default_factory=list)
    dropped_section_reasons: dict[str, str] = field(default_factory=dict)
    overflow: bool = False
    degraded: bool = False
    warnings: list[str] = field(default_factory=list)


class ContextTokenAccounting(Protocol):
    def count(self, text: str) -> int: ...

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str: ...


class ContextBuilder:
    """Assemble bounded LLM context in the authority order defined by the spec."""

    _MAX_LOCATORS_PER_OBSERVATION = 20
    _MAX_COLUMNS_PER_LOCATOR = 40
    _MAX_HEAD_ROWS_PER_LOCATOR = 8
    _MAX_ROW_CELLS = 12
    _SECTION_PRIORITY: dict[ContextSectionName, int] = {
        "instructions": 0,
        "system": 0,
        "policy_hints": 0,
        "task": 1,
        "open_decisions": 2,
        "plan": 3,
        "call_context": 4,
        "tool_results": 4,
        "evidence": 5,
        "memory": 6,
        "working_memory": 7,
        "message_tail": 8,
    }

    def __init__(
        self,
        *,
        max_context_tokens: int,
        max_section_chars: int = 4000,
        token_accounting: ContextTokenAccounting | None = None,
    ) -> None:
        if max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative")
        if max_section_chars <= 0:
            raise ValueError("max_section_chars must be positive")
        self._max_context_tokens = max_context_tokens
        self._max_section_chars = max_section_chars
        self._token_accounting = token_accounting or TokenAccountingService(
            TokenizerContract(
                embedding_model_name="agent-context",
                tokenizer_model_name="agent-context",
                chunking_tokenizer_model_name="agent-context",
                tokenizer_backend="simple",
                max_context_tokens=max(max_context_tokens, 1),
                prompt_reserved_tokens=0,
                local_files_only=True,
            )
        )

    def assemble_loop(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: LoopState,
        policy_hints: Sequence[str] = (),
        included_sections: frozenset[ContextSectionName] | None = None,
        required_sections: frozenset[ContextSectionName] | None = None,
    ) -> InjectedContext:
        """Assemble bounded context from the canonical loop state."""

        candidates: list[ContextSection] = []

        def add(
            name: ContextSectionName,
            content: str,
            *,
            required: bool = False,
        ) -> None:
            if included_sections is not None and name not in included_sections:
                return
            self._add_section(
                candidates,
                name,
                content,
                required=(required if required_sections is None else name in required_sections),
            )

        add("system", definition.system_instructions, required=True)
        add("policy_hints", self._format_policy_hints(policy_hints))
        add(
            "task",
            self._format_task(state.get("current_message", "")),
            required=True,
        )
        add(
            "open_decisions",
            self._format_loop_open_decisions(state),
            required=True,
        )
        add(
            "plan",
            self._format_plan(self._plan_from_state(state)),
            required=True,
        )
        add(
            "tool_results",
            self._format_tool_context(state),
            required=True,
        )
        ms = state.get("memory_state")
        add(
            "memory",
            self._format_memory_refs(
                ms.memory_refs if ms is not None else [],
                ms.memory_warnings if ms is not None else [],
            ),
        )
        add(
            "working_memory",
            self._format_working_memory(
                ms.extracted_facts if ms is not None else [],
            ),
        )
        add(
            "message_tail",
            self._format_message_tail(
                (
                    *state.get("conversation_history", []),
                    *state.get("turn_transcript", []),
                )
            ),
        )
        selection = self._select_sections(candidates)
        return InjectedContext(
            sections=selection.sections,
            context_budget=self._budget_snapshot(
                selection,
                state=state,
            ),
        )

    def _add_section(
        self,
        sections: list[ContextSection],
        name: ContextSectionName,
        content: str,
        *,
        required: bool = False,
    ) -> None:
        normalized = content.strip()
        if not normalized:
            return
        bounded = normalized if required else self._truncate(normalized)
        sections.append(
            ContextSection(
                name=name,
                content=bounded,
                token_count=self._section_token_count(name, bounded),
                required=required,
            )
        )

    def _select_sections(
        self,
        candidates: list[ContextSection],
    ) -> _ContextSelection:
        if self._max_context_tokens == 0:
            return _ContextSelection(sections=candidates)

        selected_by_name: dict[ContextSectionName, ContextSection] = {}
        dropped: list[ContextSectionName] = []
        summarized: list[ContextSectionName] = []
        required_truncated: list[ContextSectionName] = []
        dropped_reasons: dict[str, str] = {}
        warnings: list[str] = []
        overflow = False
        degraded = False
        indexed = list(enumerate(candidates))
        ordered = sorted(
            indexed,
            key=lambda item: (
                0 if item[1].required else 1,
                self._SECTION_PRIORITY[item[1].name],
                item[0],
            ),
        )
        for _, section in ordered:
            candidate_selection = {
                **selected_by_name,
                section.name: section,
            }
            if self._selected_token_count(candidates, candidate_selection) <= (self._max_context_tokens):
                selected_by_name[section.name] = section
                continue

            if section.required:
                overflow = True
                degraded = True
                warnings.append("context_overflow")
                required_truncated.append(section.name)
                dropped.append(section.name)
                dropped_reasons[section.name] = "required_section_overflow"
                continue

            clipped = self._clip_optional_section(
                candidates=candidates,
                selected_by_name=selected_by_name,
                section=section,
            )
            if clipped is not None:
                selected_by_name[section.name] = clipped
                summarized.append(section.name)
                degraded = True
                continue
            dropped.append(section.name)
            dropped_reasons[section.name] = "budget_priority"
            degraded = True
        return _ContextSelection(
            sections=[selected_by_name[section.name] for section in candidates if section.name in selected_by_name],
            dropped_sections=dropped,
            summarized_sections=list(dict.fromkeys(summarized)),
            required_truncated=list(dict.fromkeys(required_truncated)),
            dropped_section_reasons=dropped_reasons,
            overflow=overflow,
            degraded=degraded,
            warnings=list(dict.fromkeys(warnings)),
        )

    def _budget_snapshot(
        self,
        selection: _ContextSelection,
        *,
        state: ContextState,
    ) -> ContextBudgetSnapshot:
        sections = selection.sections
        by_name = {section.name: section.token_count for section in sections}
        summarized = [section.name for section in sections if section.content.endswith("[truncated]")]
        summarized = list(dict.fromkeys([*summarized, *selection.summarized_sections]))
        return ContextBudgetSnapshot(
            max_context_tokens=self._max_context_tokens,
            used_context_tokens=self._token_accounting.count(self._render_sections(sections)),
            system_tokens=by_name.get("system", 0) + by_name.get("policy_hints", 0),
            planning_tokens=by_name.get("plan", 0),
            evidence_tokens=by_name.get("evidence", 0),
            memory_tokens=by_name.get("memory", 0),
            working_memory_tokens=by_name.get("working_memory", 0),
            message_tail_tokens=by_name.get("message_tail", 0),
            tool_result_tokens=by_name.get("tool_results", 0) + by_name.get("open_decisions", 0),
            dropped_sections=selection.dropped_sections,
            summarized_sections=summarized,
            overflow=selection.overflow,
            degraded=selection.degraded,
            required_truncated=selection.required_truncated,
            section_token_counts={str(key): value for key, value in by_name.items()},
            dropped_section_reasons=selection.dropped_section_reasons,
            memory_ref_count=len(state["memory_state"].memory_refs if "memory_state" in state else []),
            externalized_record_count=self._externalized_tool_output_count(state.get("tool_results", [])),
            warnings=list(
                dict.fromkeys(
                    [
                        *(state["memory_state"].memory_warnings if "memory_state" in state else []),
                        *selection.warnings,
                    ]
                )
            ),
        )

    def _clip_optional_section(
        self,
        *,
        candidates: list[ContextSection],
        selected_by_name: dict[ContextSectionName, ContextSection],
        section: ContextSection,
    ) -> ContextSection | None:
        if not section.content:
            return None
        low = 1
        high = self._token_accounting.count(section.content)
        best: ContextSection | None = None
        while low <= high:
            midpoint = (low + high) // 2
            content = self._token_accounting.clip(
                section.content,
                midpoint,
                add_ellipsis=True,
            ).strip()
            if not content:
                high = midpoint - 1
                continue
            clipped = ContextSection(
                name=section.name,
                content=content,
                token_count=self._section_token_count(section.name, content),
                required=False,
            )
            candidate_selection = {
                **selected_by_name,
                section.name: clipped,
            }
            if self._selected_token_count(candidates, candidate_selection) <= (self._max_context_tokens):
                best = clipped
                low = midpoint + 1
            else:
                high = midpoint - 1
        return best

    def _selected_token_count(
        self,
        candidates: Sequence[ContextSection],
        selected_by_name: dict[ContextSectionName, ContextSection],
    ) -> int:
        selected = [selected_by_name[section.name] for section in candidates if section.name in selected_by_name]
        return self._token_accounting.count(self._render_sections(selected))

    def _section_token_count(
        self,
        name: ContextSectionName,
        content: str,
    ) -> int:
        return self._token_accounting.count(f"[{name}]\n{content}")

    @staticmethod
    def _render_sections(sections: Sequence[ContextSection]) -> str:
        return "\n\n".join(f"[{section.name}]\n{section.content}" for section in sections)

    @staticmethod
    def _format_policy_hints(policy_hints: Sequence[str]) -> str:
        hints = [hint.strip() for hint in policy_hints if hint.strip()]
        if not hints:
            return ""
        lines = ["Instruction and policy hints:"]
        lines.extend(f"- {hint}" for hint in hints)
        return "\n".join(lines)

    @staticmethod
    def _format_task(task: str) -> str:
        task_text = task.strip()
        if not task_text:
            return ""
        return f"Current task:\n{task_text}"

    def _format_plan(self, plan: Any) -> str:
        if plan is None:
            return ""
        steps = getattr(plan, "steps", []) or []
        lines = ["Current autonomous plan:"]
        objective = getattr(plan, "objective", None)
        if isinstance(objective, str) and objective.strip():
            lines.append(f"objective: {self._one_line(objective)}")
        status = getattr(plan, "status", None)
        revision = getattr(plan, "revision", None)
        active_step_id = getattr(plan, "active_step_id", None)
        plan_bits: list[str] = []
        if status:
            plan_bits.append(f"status={status}")
        if revision is not None:
            plan_bits.append(f"revision={revision}")
        if active_step_id:
            plan_bits.append(f"active_step_id={self._format_identifier(active_step_id)}")
        if plan_bits:
            lines.append(" ".join(plan_bits))
        summary = getattr(plan, "summary", None)
        if isinstance(summary, str) and summary.strip():
            lines.append(f"summary: {self._one_line(summary)}")
        if steps:
            lines.append("steps:")
            for step in steps[:12]:
                title = self._one_line(str(getattr(step, "title", "")))
                step_line = (
                    f"- step_id={self._format_identifier(getattr(step, 'step_id', '<unknown>'))} "
                    f"status={getattr(step, 'status', '<unknown>')} "
                    f"title={title}"
                )
                lines.append(step_line)
                expected_tools = getattr(step, "expected_tool_names", None)
                if isinstance(expected_tools, list) and expected_tools:
                    lines.append("  expected_tool_names: " + self._format_list([str(item) for item in expected_tools]))
                tool_call_ids = getattr(step, "tool_call_ids", None)
                if isinstance(tool_call_ids, list) and tool_call_ids:
                    lines.append("  tool_call_ids: " + self._format_list([str(item) for item in tool_call_ids]))
                notes = getattr(step, "notes", None)
                if isinstance(notes, str) and notes.strip():
                    lines.append(f"  notes: {self._one_line(notes)}")
        return "\n".join(lines)

    @staticmethod
    def _plan_from_state(state: LoopState) -> Any:
        plan_state = state.get("plan_state")
        if plan_state is not None:
            plan = getattr(plan_state, "agent_plan", None)
            if plan is not None:
                return plan
        return state.get("agent_plan")

    def _format_working_memory(self, facts: Sequence[Any]) -> str:
        if not facts:
            return ""
        lines = [
            "Working memory is current-run context, not an authority above retrieved evidence.",
        ]
        lines.append("extracted_facts:")
        for fact in facts:
            evidence_ids = ", ".join(fact.evidence_ids)
            source_ids = ", ".join(fact.source_message_ids)
            stale = " stale=true" if fact.stale else ""
            suffix = f"{stale} confidence={fact.confidence:.3g}"
            lines.append(f"- fact_id={fact.fact_id}{suffix}")
            lines.append(f"  text: {self._one_line(fact.text)}")
            if evidence_ids:
                lines.append(f"  evidence_ids: {evidence_ids}")
            if source_ids:
                lines.append(f"  source_message_ids: {source_ids}")
        return "\n".join(lines)

    def _format_memory_refs(
        self,
        memory_refs: Sequence[Any],
        memory_warnings: Sequence[str],
    ) -> str:
        refs = [ref for ref in memory_refs if isinstance(ref, MemoryRef)]
        warnings = [warning for warning in memory_warnings if warning]
        if not refs and not warnings:
            return ""
        lines = [
            "Run-local externalized memory refs. Use summaries for reasoning; "
            "raw payloads require internal resolution.",
        ]
        if refs:
            for ref in refs[:20]:
                lines.append(
                    "- "
                    + self._metadata_line(
                        ref_id=ref.ref_id,
                        status=ref.status,
                        source_tool=ref.source_tool_name,
                        tool_call_id=ref.source_tool_call_id,
                        size_bytes=ref.size_bytes,
                    )
                )
                lines.append(f"  summary: {self._one_line(ref.summary)}")
            remaining = len(refs) - 20
            if remaining > 0:
                lines.append(f"- ... {remaining} more memory refs")
        if warnings:
            lines.append("memory_warnings: " + ", ".join(dict.fromkeys(warnings)))
        return "\n".join(lines)

    def _format_message_tail(
        self,
        messages: Sequence[ModelMessage],
    ) -> str:
        if not messages:
            return ""
        lines = ["Canonical conversation transcript:"]
        lines.extend(self._format_message(message) for message in messages)
        return "\n".join(lines)

    def _format_list(self, values: Sequence[Any], *, limit: int | None = None) -> str:
        effective_limit = limit if limit is not None else len(values)
        shown = [self._one_line(str(value)) for value in values[:effective_limit]]
        remaining = len(values) - effective_limit
        suffix = f", ...(+{remaining})" if remaining > 0 else ""
        return "[" + ", ".join(shown) + suffix + "]"

    @staticmethod
    def _format_identifier(value: object) -> str:
        return ContextBuilder._preserve_spaces_one_line(str(value))

    @staticmethod
    def _preserve_spaces_one_line(text: str) -> str:
        return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()

    def _format_tool_results(self, tool_results: Sequence[ToolResult]) -> str:
        if not tool_results:
            return ""
        lines = ["Tool results:"]
        for result in tool_results:
            message = tool_result_message(result)
            lines.append(f"- tool_call_id={result.tool_call_id} tool_name={result.tool_name} content={message.content}")
        return "\n".join(lines)

    def _format_tool_context(self, state: ContextState) -> str:
        return self._format_tool_results(state.get("tool_results", []))

    def _format_loop_open_decisions(self, state: LoopState) -> str:
        lines: list[str] = []
        request = state.get("approval_request")
        if request is not None:
            lines.append(
                "approval_request: "
                f"kind={request.kind} request_id={request.request_id} "
                f"question={self._one_line(request.question)}"
            )
            for approval_call in request.tool_calls:
                lines.append(
                    f"- tool_call_id={approval_call.tool_call_id} "
                    f"tool_name={approval_call.tool_name} "
                    f"risk_level={approval_call.risk_level}"
                )
        response = state.get("approval_response")
        if response is not None:
            lines.append(f"approval_response: request_id={response.request_id} decision={response.decision}")
            if response.user_message:
                lines.append(f"user_message: {self._one_line(response.user_message)}")
        pending_tool_calls = state.get("pending_tool_calls", [])
        if pending_tool_calls:
            lines.append("pending_tool_calls:")
            for pending_call in pending_tool_calls:
                lines.append(
                    f"- tool_call_id={pending_call.tool_call_id} "
                    f"tool_name={pending_call.tool_name} "
                    f"arguments={self._one_line(str(pending_call.plan.arguments))}"
                )
        fs = state.get("finish_state")
        feedback = fs.feedback if fs is not None else []
        if feedback:
            lines.append("finish_feedback:")
            for item in feedback:
                lines.append(
                    f"- code={item.code} occurrences={item.occurrences} message={self._one_line(item.message)}"
                )
        fs = state.get("finish_state")
        warnings = fs.warnings if fs is not None else []
        if warnings:
            lines.append("finish_warnings:")
            for item in warnings:
                lines.append(
                    f"- code={item.code} occurrences={item.occurrences} message={self._one_line(item.message)}"
                )
        return "\n".join(lines)

    @staticmethod
    def _metadata_line(**values: object) -> str:
        return " ".join(f"{key}={value}" for key, value in values.items() if value not in (None, "", []))

    @staticmethod
    def _format_message(message: ModelMessage) -> str:
        return "- " + canonical_json_text(model_message_payload(message))

    @staticmethod
    def _externalized_tool_output_count(tool_results: Sequence[ToolResult]) -> int:
        del tool_results
        return 0

    @staticmethod
    def _one_line(text: str) -> str:
        return " ".join(text.split())

    def _truncate(self, content: str) -> str:
        if len(content) <= self._max_section_chars:
            return content
        truncated = content[: self._max_section_chars].rstrip()
        return f"{truncated}\n[truncated]"


__all__ = ["ContextBuilder"]

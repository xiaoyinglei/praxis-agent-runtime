from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.llm_prompts import (
    build_loop_turn_prompt,
)
from agent_runtime.loop.state import LoopState
from agent_runtime.memory.injector import ContextBuilder, ContextTokenAccounting
from agent_runtime.memory.models import (
    ContextBudgetSnapshot,
    ContextSection,
    ContextSectionName,
    InjectedContext,
)
from agent_runtime.modeling.contracts import LLMCallStage, LLMStageBudget
from agent_runtime.modeling.gateway import structured_accounted_prompt

_OPTIONAL_STATE_SECTIONS: frozenset[ContextSectionName] = frozenset(
    {
        "evidence",
        "tool_results",
        "memory",
        "working_memory",
        "message_tail",
    }
)
_DECISION_STATE_SECTIONS: frozenset[ContextSectionName] = frozenset[ContextSectionName](
    {"open_decisions", "plan"}
).union(_OPTIONAL_STATE_SECTIONS)


@dataclass(frozen=True, slots=True)
class AssembledAgentLLMContext:
    stage: LLMCallStage
    prompt: str
    context: InjectedContext


class AgentLLMContextOverflowError(RuntimeError):
    def __init__(
        self,
        *,
        stage: LLMCallStage,
        context_budget: ContextBudgetSnapshot,
    ) -> None:
        super().__init__(
            f"Required Agent LLM context does not fit stage {stage.value}: "
            f"{', '.join(context_budget.required_truncated) or 'unknown section'}"
        )
        self.stage = stage
        self.context_budget = context_budget


class AgentLLMContextAssembler:
    def __init__(
        self,
        *,
        token_accounting: ContextTokenAccounting,
        stage_budgets: Mapping[LLMCallStage, LLMStageBudget],
    ) -> None:
        self._token_accounting = token_accounting
        self._stage_budgets = dict(stage_budgets)

    @property
    def token_accounting(self) -> ContextTokenAccounting:
        return self._token_accounting

    def assemble_loop_turn(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: LoopState,
        budget_remaining: int | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> AssembledAgentLLMContext:
        return self._assemble(
            stage=LLMCallStage.TOOL_DECISION,
            state=state,
            prefix_sections=[
                self._required_section("system", definition.system_instructions),
                self._required_section(
                    "instructions",
                    build_loop_turn_prompt(
                        state,
                        budget_remaining=budget_remaining,
                        allowed_tools=definition.configured_tool_names,
                    ),
                ),
            ],
            included_state_sections=_DECISION_STATE_SECTIONS,
            required_state_sections=frozenset({"open_decisions", "plan"}),
            output_schema=output_schema,
        )

    def assemble_generate(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: LoopState,
        prompt: str,
        context_sections: Sequence[str],
        stage: LLMCallStage = LLMCallStage.LLM_GENERATE,
    ) -> AssembledAgentLLMContext:
        prefix = [
            self._required_section("system", definition.system_instructions),
            self._required_section("task", prompt),
        ]
        if call_context := self._call_context(context_sections):
            prefix.append(self._required_section("call_context", call_context))
        return self._assemble(
            stage=stage,
            state=state,
            prefix_sections=prefix,
            included_state_sections=_OPTIONAL_STATE_SECTIONS,
            required_state_sections=frozenset(),
        )

    def assemble_summarize(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: LoopState,
        task: str,
        context_sections: Sequence[str],
    ) -> AssembledAgentLLMContext:
        instruction = (
            "Summarize only the supplied and trusted context. Do not invent facts. "
            "If the context reports generated artifacts or computations, preserve "
            "their result and path."
        )
        prefix = [
            self._required_section(
                "system",
                f"{definition.system_instructions}\n\n{instruction}".strip(),
            ),
            self._required_section("task", task),
        ]
        if call_context := self._call_context(context_sections):
            prefix.append(self._required_section("call_context", call_context))
        return self._assemble(
            stage=LLMCallStage.LLM_SUMMARIZE,
            state=state,
            prefix_sections=prefix,
            included_state_sections=_OPTIONAL_STATE_SECTIONS,
            required_state_sections=frozenset(),
        )

    def assemble_compare(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: LoopState,
        question: str,
        left_context_sections: Sequence[str],
        right_context_sections: Sequence[str],
    ) -> AssembledAgentLLMContext:
        context_parts: list[str] = []
        if left := self._call_context(left_context_sections):
            context_parts.append(f"Left context:\n{left}")
        if right := self._call_context(right_context_sections):
            context_parts.append(f"Right context:\n{right}")
        prefix = [
            self._required_section("system", definition.system_instructions),
            self._required_section("task", question),
        ]
        if context_parts:
            prefix.append(
                self._required_section(
                    "call_context",
                    "\n\n".join(context_parts),
                )
            )
        return self._assemble(
            stage=LLMCallStage.LLM_COMPARE,
            state=state,
            prefix_sections=prefix,
            included_state_sections=_OPTIONAL_STATE_SECTIONS,
            required_state_sections=frozenset(),
        )

    def assemble_final_output(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: LoopState,
        candidate_text: str,
        validation_feedback: str | None,
        output_schema: type[BaseModel],
    ) -> AssembledAgentLLMContext:
        instructions = (
            "Produce the final response as one object matching the required "
            "output schema. Use only the supplied task and trusted context. "
            "Do not add unsupported claims."
        )
        prefix = [
            self._required_section("system", definition.system_instructions),
            self._required_section(
                "task",
                f"{instructions}\n\nCurrent user message:\n{state['current_message'].strip()}",
            ),
        ]
        call_context = self._call_context(
            [
                (f"Candidate synthesis:\n{candidate_text.strip()}" if candidate_text.strip() else ""),
                (
                    f"Previous validation errors to repair:\n{validation_feedback.strip()}"
                    if validation_feedback and validation_feedback.strip()
                    else ""
                ),
            ]
        )
        if call_context:
            prefix.append(self._required_section("call_context", call_context))
        return self._assemble(
            stage=LLMCallStage.FINAL_SYNTHESIS,
            state=state,
            prefix_sections=prefix,
            included_state_sections=_OPTIONAL_STATE_SECTIONS,
            required_state_sections=frozenset(),
            output_schema=output_schema,
        )

    def _assemble(
        self,
        *,
        stage: LLMCallStage,
        state: LoopState,
        prefix_sections: Sequence[ContextSection],
        included_state_sections: frozenset[ContextSectionName],
        required_state_sections: frozenset[ContextSectionName],
        output_schema: type[BaseModel] | None = None,
    ) -> AssembledAgentLLMContext:
        budget = self._stage_budget(stage)
        prefix_text = self._render(prefix_sections)
        accounted_prefix = self._accounted_prompt(prefix_text, output_schema)
        if self._token_accounting.count(accounted_prefix) > budget.max_input_tokens:
            snapshot = self._prefix_overflow_snapshot(
                prefix_sections,
                max_input_tokens=budget.max_input_tokens,
                output_schema=output_schema,
            )
            raise AgentLLMContextOverflowError(
                stage=stage,
                context_budget=snapshot,
            )

        state_budget = max(
            budget.max_input_tokens
            - self._token_accounting.count(accounted_prefix)
            - self._token_accounting.count("\n\n"),
            0,
        )
        state_context = self._assemble_state_context(
            state=state,
            max_context_tokens=state_budget,
            included_sections=included_state_sections,
            required_sections=required_state_sections,
        )
        prompt = self._combine(prefix_text, state_context.as_text())
        accounted_prompt = self._accounted_prompt(prompt, output_schema)
        attempts = 0
        while (
            not state_context.context_budget.overflow
            and self._token_accounting.count(accounted_prompt) > budget.max_input_tokens
            and state_budget > 0
            and attempts < 4
        ):
            overflow = self._token_accounting.count(accounted_prompt) - budget.max_input_tokens
            state_budget = max(state_budget - overflow - 1, 0)
            state_context = self._assemble_state_context(
                state=state,
                max_context_tokens=state_budget,
                included_sections=included_state_sections,
                required_sections=required_state_sections,
            )
            prompt = self._combine(prefix_text, state_context.as_text())
            accounted_prompt = self._accounted_prompt(prompt, output_schema)
            attempts += 1

        combined = self._combined_context(
            prefix_sections=prefix_sections,
            state_context=state_context,
            prompt=prompt,
            max_input_tokens=budget.max_input_tokens,
            output_schema=output_schema,
        )
        if combined.context_budget.overflow or self._token_accounting.count(accounted_prompt) > budget.max_input_tokens:
            raise AgentLLMContextOverflowError(
                stage=stage,
                context_budget=combined.context_budget,
            )
        return AssembledAgentLLMContext(
            stage=stage,
            prompt=prompt,
            context=combined,
        )

    def _assemble_state_context(
        self,
        *,
        state: LoopState,
        max_context_tokens: int,
        included_sections: frozenset[ContextSectionName],
        required_sections: frozenset[ContextSectionName],
    ) -> InjectedContext:
        if max_context_tokens <= 0:
            if required_sections:
                builder = ContextBuilder(
                    max_context_tokens=1,
                    token_accounting=self._token_accounting,
                )
                required_probe = builder.assemble_loop(
                    definition=self._empty_definition(),
                    state=state,
                    included_sections=required_sections,
                    required_sections=required_sections,
                )
                if required_probe.sections or required_probe.context_budget.overflow:
                    required_names = list(
                        dict.fromkeys(
                            [
                                *required_probe.context_budget.required_truncated,
                                *(section.name for section in required_probe.sections),
                            ]
                        )
                    )
                    snapshot = required_probe.context_budget.model_copy(
                        update={
                            "max_context_tokens": 0,
                            "used_context_tokens": 0,
                            "overflow": True,
                            "degraded": True,
                            "required_truncated": required_names,
                            "dropped_sections": required_names,
                            "dropped_section_reasons": {
                                str(name): "required_section_overflow" for name in required_names
                            },
                            "warnings": list(
                                dict.fromkeys(
                                    [
                                        *required_probe.context_budget.warnings,
                                        "context_overflow",
                                    ]
                                )
                            ),
                        }
                    )
                    return InjectedContext(sections=[], context_budget=snapshot)
            return InjectedContext(
                sections=[],
                context_budget=ContextBudgetSnapshot(max_context_tokens=0),
            )

        builder = ContextBuilder(
            max_context_tokens=max_context_tokens,
            token_accounting=self._token_accounting,
        )
        return builder.assemble_loop(
            definition=self._empty_definition(),
            state=state,
            included_sections=included_sections,
            required_sections=required_sections,
        )

    @staticmethod
    def _empty_definition() -> AgentRuntimePolicy:
        empty_definition = AgentRuntimePolicy(
            system_instructions="",
            core_tool_names=(),
            deferred_tool_names=(),
            max_iterations=10,
        )
        return empty_definition

    def _combined_context(
        self,
        *,
        prefix_sections: Sequence[ContextSection],
        state_context: InjectedContext,
        prompt: str,
        max_input_tokens: int,
        output_schema: type[BaseModel] | None,
    ) -> InjectedContext:
        sections = [*prefix_sections, *state_context.sections]
        accounted_prompt = self._accounted_prompt(prompt, output_schema)
        section_token_counts = {
            **state_context.context_budget.section_token_counts,
            **{str(section.name): section.token_count for section in prefix_sections},
        }
        if output_schema is not None:
            section_token_counts["output_schema"] = self._schema_token_count(output_schema)
        snapshot = state_context.context_budget.model_copy(
            update={
                "max_context_tokens": max_input_tokens,
                "used_context_tokens": self._token_accounting.count(accounted_prompt),
                "system_tokens": (
                    state_context.context_budget.system_tokens
                    + sum(
                        section.token_count for section in prefix_sections if section.name in {"instructions", "system"}
                    )
                ),
                "section_token_counts": section_token_counts,
            }
        )
        return InjectedContext(sections=sections, context_budget=snapshot)

    def _prefix_overflow_snapshot(
        self,
        sections: Sequence[ContextSection],
        *,
        max_input_tokens: int,
        output_schema: type[BaseModel] | None,
    ) -> ContextBudgetSnapshot:
        selected: list[ContextSection] = []
        truncated: list[ContextSectionName] = []
        if (
            output_schema is not None
            and self._token_accounting.count(self._accounted_prompt("", output_schema)) > max_input_tokens
        ):
            truncated.append("output_schema")
        for section in sections:
            candidate = self._accounted_prompt(
                self._render([*selected, section]),
                output_schema,
            )
            if self._token_accounting.count(candidate) <= max_input_tokens:
                selected.append(section)
            else:
                truncated.append(section.name)
        return ContextBudgetSnapshot(
            max_context_tokens=max_input_tokens,
            used_context_tokens=self._token_accounting.count(
                self._accounted_prompt(
                    self._render(selected),
                    output_schema,
                )
            ),
            system_tokens=sum(
                section.token_count for section in selected if section.name in {"instructions", "system"}
            ),
            dropped_sections=truncated,
            required_truncated=truncated,
            dropped_section_reasons={str(name): "required_section_overflow" for name in truncated},
            overflow=True,
            degraded=True,
            warnings=["context_overflow"],
            section_token_counts={
                **{str(section.name): section.token_count for section in selected},
                **({"output_schema": self._schema_token_count(output_schema)} if output_schema is not None else {}),
            },
        )

    def _required_section(
        self,
        name: ContextSectionName,
        content: str,
    ) -> ContextSection:
        normalized = content.strip()
        return ContextSection(
            name=name,
            content=normalized,
            token_count=self._token_accounting.count(f"[{name}]\n{normalized}"),
            required=True,
        )

    def _stage_budget(self, stage: LLMCallStage) -> LLMStageBudget:
        try:
            return self._stage_budgets[stage]
        except KeyError as exc:
            raise ValueError(f"No Agent LLM context budget configured for {stage.value}") from exc

    @staticmethod
    def _call_context(sections: Sequence[str]) -> str:
        return "\n\n".join(section.strip() for section in sections if section.strip())

    @staticmethod
    def _render(sections: Sequence[ContextSection]) -> str:
        return "\n\n".join(f"[{section.name}]\n{section.content}" for section in sections)

    @staticmethod
    def _combine(prefix: str, context: str) -> str:
        if prefix and context:
            return f"{prefix}\n\n{context}"
        return prefix or context

    @staticmethod
    def _accounted_prompt(
        prompt: str,
        output_schema: type[BaseModel] | None,
    ) -> str:
        if output_schema is None:
            return prompt
        return structured_accounted_prompt(prompt, output_schema)

    def _schema_token_count(self, schema: type[BaseModel]) -> int:
        accounted = structured_accounted_prompt("", schema)
        return self._token_accounting.count(accounted)


__all__ = [
    "AgentLLMContextAssembler",
    "AgentLLMContextOverflowError",
    "AssembledAgentLLMContext",
]

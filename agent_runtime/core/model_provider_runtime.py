"""Resolve the one model-turn provider used by the canonical AgentLoop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.goal_contract import GoalSpec
from agent_runtime.core.llm_providers import create_loop_model_turn_provider
from agent_runtime.core.llm_registry import ModelResolver
from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic
from agent_runtime.loop.state import LoopState, ModelTurnDraft, ModelTurnProvider
from agent_runtime.skills.runtime import SkillRuntime
from agent_runtime.tools.tool import Tool


class ResultDrivenModelTurnProvider:
    """Fail-closed fallback used only when strict model setup is disabled."""

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["finish_state"].feedback:
            return ModelTurnDraft(
                action="pause",
                pause_reason=(
                    "No model provider is available to address stop-hook "
                    "feedback."
                ),
            )
        if state["tool_results"]:
            latest = state["tool_results"][-1]
            if latest.is_error:
                return ModelTurnDraft(
                    action="finish",
                    final_answer=latest.error_message or "Tool execution failed.",
                )
            payload = latest.structured_content
            if isinstance(payload, Mapping):
                for key in ("text", "result", "output_text", "conclusion"):
                    value = payload.get(key)
                    if isinstance(value, str) and value:
                        return ModelTurnDraft(
                            action="finish",
                            final_answer=value,
                        )
            return ModelTurnDraft(
                action="pause",
                pause_reason="Tool execution produced no extractable answer.",
            )
        return ModelTurnDraft(
            action="pause",
            pause_reason="No model turn provider is configured.",
        )


@dataclass
class ModelProviderResolver:
    model_turn_provider: ModelTurnProvider | None
    model_registry: ModelResolver | None
    policy: AgentRuntimePolicy
    registry_snapshot: Mapping[str, Tool]
    goal_spec: GoalSpec | None = None
    strict_model_provider: bool = True
    stream_sink: object | None = None
    skill_runtime: SkillRuntime | None = None

    def resolve(self, state: LoopState | None) -> ModelTurnProvider:
        if self.model_turn_provider is not None:
            return self.model_turn_provider
        if self.model_registry is not None:
            try:
                resident: tuple[str, ...] = ()
                disabled: tuple[str, ...] = ()
                if state is not None:
                    resident = (
                        *state.get("resident_tool_names", ()),
                        *state.get("explicit_tool_names", ()),
                    )
                    disabled = tuple(state.get("disabled_tool_names", ()))
                return create_loop_model_turn_provider(
                    self.model_registry,
                    self.policy.model_selection,
                    registry_snapshot=self.registry_snapshot,
                    resident_tool_names=resident,
                    disabled_tool_names=disabled,
                    goal_spec=self.goal_spec,
                    stream_sink=self.stream_sink,
                    skill_runtime=self.skill_runtime,
                )
            except Exception as exc:
                if self.strict_model_provider:
                    raise
                if state is not None:
                    state["runtime_diagnostics"].append(
                        RuntimeDiagnostic.from_exception(
                            code="default_providers_initialization_failed",
                            component="model_providers",
                            error=exc,
                        )
                    )
        return ResultDrivenModelTurnProvider()


__all__ = ["ModelProviderResolver", "ResultDrivenModelTurnProvider"]

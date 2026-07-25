from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, cast

from pydantic import BaseModel, Field

from rag.agent.core.definition import AgentRuntimePolicy, ModelSelectionPolicy
from rag.agent.core.goal_contract import GoalSpec
from rag.agent.core.llm_registry import ModelResolver
from rag.agent.core.messages import (
    ModelMessage,
    StopReason,
    ToolUseResult,
    canonical_json_text,
    context_event_message,
    model_message_payload,
)
from rag.agent.core.messages import ToolCall as ModelToolCall
from rag.agent.core.model_request import (
    ContextBlock,
    ModelRequest,
    ModelSettings,
    StableModelContext,
    bind_model_call_record,
    build_model_request,
    build_stable_context,
    split_turn_context,
    tool_definition_payload,
)
from rag.agent.core.observations import (
    grounded_workspace_paths,
    runtime_workspace_change,
)
from rag.agent.core.runtime_diagnostics import AgentLatencyProfile
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import LoopState, ModelTurnDraft, ModelTurnEnvelope
from rag.agent.skills.runtime import SkillRuntime
from rag.agent.streaming.events import text_delta
from rag.agent.tools.selection import select_tools
from rag.agent.tools.tool import JsonValue, Tool, ToolCallOrigin
from rag.providers.llm_gateway import (
    LLMGateway,
    model_request_input_text,
)
from rag.schema.llm import LLMCallStage

_MAX_WORKING_STATE_GROUNDED_PATHS = 200


class LoopModelDecision(BaseModel):
    """Small compatibility input accepted by ``parse_loop_model_turn``."""

    action: Literal["execute", "finish", "pause"]
    tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    final_answer: str | None = None
    pause_reason: str | None = None
    needs_user_input: str | None = None
    stop_reason: str | None = None
    thought: str | None = None


def parse_loop_model_turn(
    value: ModelTurnDraft | LoopModelDecision | Mapping[str, object],
) -> ModelTurnDraft:
    """Normalize a typed decision without giving labels routing authority."""

    if isinstance(value, ModelTurnDraft):
        return value
    decision = (
        value
        if isinstance(value, LoopModelDecision)
        else LoopModelDecision.model_validate(value)
    )
    calls = tuple(decision.tool_calls)
    if calls:
        return ModelTurnDraft(action="execute", tool_calls=calls)
    if decision.action == "finish":
        return ModelTurnDraft(action="finish", final_answer=decision.final_answer)
    if decision.action == "pause":
        return ModelTurnDraft(
            action="pause",
            pause_reason=(
                decision.pause_reason
                or decision.needs_user_input
                or decision.stop_reason
                or decision.thought
            ),
        )
    return ModelTurnDraft(action="execute")


class LLMLoopModelTurnProvider:
    """Build one canonical request and delegate only wire work to the gateway."""

    manages_llm_context = True

    def __init__(
        self,
        gateway: LLMGateway,
        *,
        model: str,
        provider: str,
        supports_native_tools: bool,
        registry_snapshot: Mapping[str, Tool],
        resident_tool_names: Sequence[str],
        disabled_tool_names: Sequence[str] = (),
        kwargs: Mapping[str, object] | None = None,
        context_window_tokens: int = 32_768,
        stream_sink: object | None = None,
        skill_runtime: SkillRuntime | None = None,
        goal_spec: GoalSpec | None = None,
    ) -> None:
        if not hasattr(gateway, "agenerate_model_request"):
            raise TypeError("gateway must execute canonical ModelRequest values")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be non-empty")
        if not isinstance(provider, str) or not provider:
            raise ValueError("provider must be non-empty")
        if type(supports_native_tools) is not bool:
            raise TypeError("supports_native_tools must be a bool")
        if context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        self._gateway = gateway
        self._model = model
        self._provider = provider
        self._supports_native_tools = supports_native_tools
        self._registry_snapshot = registry_snapshot
        self._resident_tool_names = tuple(resident_tool_names)
        self._disabled_tool_names = tuple(disabled_tool_names)
        self._kwargs = dict(kwargs or {})
        self._context_window_tokens = context_window_tokens
        self._stream_sink = stream_sink
        self._skill_runtime = skill_runtime
        self._goal_spec = (
            None if goal_spec is None else goal_spec.model_copy(deep=True)
        )

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnEnvelope:
        del budget_remaining
        state_resident_names = (
            *state.get("resident_tool_names", ()),
            *state.get("explicit_tool_names", ()),
        )
        resident_names = tuple(
            state_resident_names or self._resident_tool_names
        )
        disabled_names = tuple(
            state.get("disabled_tool_names") or self._disabled_tool_names
        )
        selected_tools = select_tools(
            self._registry_snapshot,
            resident_names=resident_names,
            active_names=tuple(state.get("active_tool_names", ())),
            disabled_names=disabled_names,
        )
        skill_context = (
            ""
            if self._skill_runtime is None
            else self._skill_runtime.render_prompt_context(state)
        )
        instructions = [
            definition.system_instructions or "You are a helpful agent."
        ]
        if skill_context:
            instructions.append(skill_context)
        file_manifest = state.get("file_manifest")
        frozen_run_context: list[ContextBlock] = []
        if self._goal_spec is not None:
            frozen_run_context.append(
                ContextBlock(
                    name="goal_contract",
                    content={
                        "authority": "runtime",
                        "fingerprint": self._goal_spec.fingerprint,
                        "spec": self._goal_spec.model_dump(mode="json"),
                    },
                )
            )
        if file_manifest is not None and file_manifest.files:
            frozen_run_context.append(
                ContextBlock(
                    name="input_files",
                    content={
                        "instruction": (
                            "Use these exact workspace-relative paths when calling "
                            "file tools."
                        ),
                        "files": tuple(
                            {
                                "path": entry.path,
                                "kind": entry.file_kind,
                                "size_bytes": entry.size_bytes,
                            }
                            for entry in file_manifest.files
                        ),
                    },
                )
            )
        settings = self._model_settings(definition.model_selection)
        initial_message, context_transcript = split_turn_context(
            conversation_history=state["conversation_history"],
            turn_transcript=state["turn_transcript"],
        )
        context = build_stable_context(
            instructions=tuple(instructions),
            frozen_run_context=tuple(frozen_run_context),
            initial_user_task=initial_message,
            initial_memory=tuple(state.get("persistent_memories", ())),
            transcript=context_transcript,
        )
        working_state = _working_state_message(
            state,
            goal_spec=self._goal_spec,
        )
        if working_state is not None:
            context = context.append_message(working_state)
        context_limit = (
            state["run_config"].max_context_tokens
            or self._context_window_tokens
        )
        model_input_limit = max(
            256,
            context_limit - settings.max_output_tokens - 1_024,
        )
        stage_input_limit = self._gateway.effective_stage_budget(
            LLMCallStage.TOOL_DECISION,
            kwargs={"max_tokens": settings.max_output_tokens},
        ).max_input_tokens
        request_id = (
            f"{state['run_config'].turn_id}:turn:{state['iteration']}"
        )

        def request_for(
            candidate_context: StableModelContext,
        ) -> ModelRequest:
            return build_model_request(
                request_id=request_id,
                context=candidate_context,
                selected_tools=selected_tools,
                settings=settings,
            )

        def request_input_tokens(
            candidate_context: StableModelContext,
        ) -> int:
            candidate_request = request_for(candidate_context)
            accounted_input = model_request_input_text(
                candidate_request,
                provider=self._provider,
                supports_native_tools=self._supports_native_tools,
            )
            return self._gateway.token_accounting.count(
                accounted_input
            )

        context = _project_model_context(
            context,
            max_input_tokens=min(model_input_limit, stage_input_limit),
            max_summary_chars=(
                state["run_config"].memory_policy.max_working_summary_chars
            ),
            protected_tail_start=(
                len(state["conversation_history"]) - 1
                if state["conversation_history"]
                else None
            ),
            input_token_count=request_input_tokens,
        )
        request = request_for(context)
        _record_request_sizes(state, request)
        response = await self._gateway.agenerate_model_request(
            stage=LLMCallStage.TOOL_DECISION,
            request=request,
            provider=self._provider,
            supports_native_tools=self._supports_native_tools,
            stream=self._stream_sink is not None,
            text_delta_sink=self._emit_text_delta,
        )
        turn = response.turn
        if not isinstance(turn, ToolUseResult):
            raise TypeError("gateway must return a provider-neutral ToolUseResult")
        turn = _scope_model_tool_call_ids(
            turn,
            request_id=request.request_id,
        )
        record = bind_model_call_record(
            request=request,
            provider_wire_hash=response.provider_wire_hash,
            usage=response.usage,
        )
        assistant_message = ModelMessage(
            role="assistant",
            content=turn.text,
            reasoning_content=turn.reasoning_content,
            tool_calls=tuple(turn.tool_calls),
        )
        return ModelTurnEnvelope(
            draft=_draft_from_turn(turn, request=request),
            request=request,
            model_call_record=record,
            assistant_message=assistant_message,
            context_revision=context.context_revision,
            provider_serializer_revision=response.serializer_revision,
        )

    def _model_settings(
        self,
        selection: ModelSelectionPolicy,
    ) -> ModelSettings:
        max_output_tokens = self._kwargs.get(
            "max_tokens",
            selection.tool_decision_max_tokens or 2048,
        )
        temperature = self._kwargs.get(
            "temperature",
            selection.tool_decision_temperature,
        )
        top_p = self._kwargs.get("top_p", 1.0)
        seed = self._kwargs.get("seed")
        provider_options = self._kwargs.get("provider_options", {})
        if not isinstance(provider_options, Mapping):
            raise TypeError("provider_options must be a mapping")
        return ModelSettings(
            model=self._model,
            max_output_tokens=_int_setting(max_output_tokens),
            temperature=_float_setting(temperature),
            top_p=(
                None
                if top_p is None
                else _float_setting(top_p)
            ),
            parallel_tool_calls=bool(
                self._kwargs.get("parallel_tool_calls", True)
            ),
            seed=(
                None
                if seed is None
                else _int_setting(seed)
            ),
            provider_options=cast(
                Mapping[str, JsonValue],
                provider_options,
            ),
        )

    async def _emit_text_delta(self, value: str) -> None:
        sink = self._stream_sink
        if sink is None:
            return
        emit = getattr(sink, "emit", None)
        if not callable(emit):
            return
        await emit(text_delta(value))


def _project_model_context(
    context: StableModelContext,
    *,
    max_input_tokens: int,
    max_summary_chars: int,
    protected_tail_start: int | None,
    input_token_count: Callable[[StableModelContext], int],
) -> StableModelContext:
    """Apply a final measured model-window projection."""

    current_tokens = input_token_count(context)
    if (
        not context.transcript
        or current_tokens <= max_input_tokens
    ):
        return context

    maximum_tail_start = (
        len(context.transcript)
        if protected_tail_start is None
        else min(
            max(protected_tail_start, 0),
            len(context.transcript),
        )
    )
    if maximum_tail_start <= 0:
        return context

    best = context
    best_tokens = current_tokens
    summary_limits = _projection_summary_limits(
        max_summary_chars
    )
    for tail_start in range(1, maximum_tail_start + 1):
        for summary_limit in summary_limits:
            candidate = context.project_compaction(
                tail_start=tail_start,
                max_summary_chars=summary_limit,
            )
            if candidate == context:
                continue
            candidate_tokens = input_token_count(candidate)
            if candidate_tokens < best_tokens:
                best = candidate
                best_tokens = candidate_tokens
            if candidate_tokens <= max_input_tokens:
                return candidate
    return best


def _projection_summary_limits(
    max_summary_chars: int,
) -> tuple[int, ...]:
    current = min(max_summary_chars, 12_000)
    limits: list[int] = []
    while current > 80:
        limits.append(current)
        current = max(80, current // 2)
    limits.append(current)
    if limits[-1] != 1:
        limits.append(1)
    return tuple(dict.fromkeys(limits))


def _working_state_message(
    state: LoopState,
    *,
    goal_spec: GoalSpec | None,
) -> ModelMessage | None:
    memory_state = state["memory_state"]
    plan = state["plan_state"].agent_plan
    workspace_change_tool_call_ids = tuple(
        result.tool_call_id
        for result in state["tool_results"]
        if runtime_workspace_change(result) is not None
    )
    latest_change_index = max(
        (
            index
            for index, result in enumerate(state["tool_results"])
            if runtime_workspace_change(result) is not None
        ),
        default=-1,
    )
    verification_tool_call_ids = tuple(
        result.tool_call_id
        for result in state["tool_results"][latest_change_index + 1 :]
        if (
            latest_change_index >= 0
            and result.tool_name == "run_command"
            and not result.is_error
        )
    )
    runtime_requirements = tuple(
        {
            "constraint_id": constraint.constraint_id,
            "constraint_type": constraint.constraint_type,
            "expected_value": cast(JsonValue, constraint.expected_value),
            "observation": (
                "observed"
                if (
                    constraint.constraint_type == "workspace_change"
                    and workspace_change_tool_call_ids
                )
                or (
                    constraint.constraint_type == "verification_after_change"
                    and verification_tool_call_ids
                )
                else "pending"
            ),
            "requirement": _runtime_requirement_description(
                constraint.constraint_type
            ),
        }
        for constraint in (() if goal_spec is None else goal_spec.constraints)
        if (
            constraint.required
            and constraint.expected_value is True
            and constraint.constraint_type
            in {"workspace_change", "verification_after_change"}
        )
    )
    if (
        not memory_state.recent_observations
        and not memory_state.known_locators
        and not memory_state.verified_workspace_paths
        and plan is None
        and not runtime_requirements
        and goal_spec is None
    ):
        return None

    plan_payload: JsonValue = None
    if plan is not None:
        plan_payload = {
            "authority": "advisory",
            "objective": plan.objective,
            "status": plan.status,
            "active_step_id": plan.active_step_id,
            "steps": tuple(
                {
                    "step_id": step.step_id,
                    "title": step.title,
                    "status": step.status,
                }
                for step in plan.steps
            ),
        }
    file_manifest = state.get("file_manifest")
    manifest_paths = (
        ()
        if file_manifest is None
        else tuple(entry.path for entry in file_manifest.files)
    )
    verified_grounded_paths = grounded_workspace_paths(
        locators=memory_state.known_locators,
        input_paths=(
            *manifest_paths,
            *memory_state.verified_workspace_paths,
        ),
        tool_results=state["tool_results"],
        tool_calls=state["canonical_tool_calls"],
    )
    grounded_paths = _project_grounded_workspace_paths(
        verified_grounded_paths,
    )
    return context_event_message(
        "working_state",
        {
            "active_goal": (
                None
                if goal_spec is None
                else {
                    "authority": "runtime",
                    "fingerprint": goal_spec.fingerprint,
                    "original_query": goal_spec.original_query,
                }
            ),
            "plan_claims": plan_payload,
            "runtime_requirements": runtime_requirements,
            "runtime_evidence": {
                "authority": "runtime",
                "grounded_paths": grounded_paths,
                "grounded_path_count": len(verified_grounded_paths),
                "grounded_paths_truncated": (
                    len(grounded_paths) < len(verified_grounded_paths)
                ),
                "recent_observations": tuple(
                    {
                        "tool_call_id": observation.tool_call_id,
                        "tool_name": observation.tool_name,
                        "status": observation.status,
                        "error": observation.error,
                        "warnings": tuple(observation.warnings),
                    }
                    for observation in memory_state.recent_observations
                ),
                "known_locators": tuple(
                    cast(Mapping[str, JsonValue], locator)
                    for locator in memory_state.known_locators
                ),
                "workspace_change_tool_call_ids": tuple(
                    workspace_change_tool_call_ids
                ),
                "verification_tool_call_ids": verification_tool_call_ids,
            },
        },
    )


def _runtime_requirement_description(constraint_type: str) -> str:
    if constraint_type == "workspace_change":
        return (
            "A runtime-observed write must change workspace contents; "
            "prose and pre-change verification do not satisfy this."
        )
    return (
        "A recognized verification command must succeed after the latest "
        "workspace change; pre-change commands do not satisfy this."
    )


def _project_grounded_workspace_paths(
    paths: Sequence[str],
) -> tuple[str, ...]:
    if len(paths) <= _MAX_WORKING_STATE_GROUNDED_PATHS:
        return tuple(paths)
    selected: dict[str, None] = {}
    for path in reversed(paths):
        selected.setdefault(path, None)
        if len(selected) >= _MAX_WORKING_STATE_GROUNDED_PATHS:
            break
    return tuple(selected)


def _record_request_sizes(state: LoopState, request: ModelRequest) -> None:
    profile = state.get("latency_profile")
    if not isinstance(profile, AgentLatencyProfile):
        profile = AgentLatencyProfile()
    prompt_bytes = len(
        canonical_json_text(
            tuple(model_message_payload(message) for message in request.messages)
        ).encode("utf-8")
    )
    tool_schema_bytes = len(
        canonical_json_text(
            tuple(tool_definition_payload(tool) for tool in request.tools)
        ).encode("utf-8")
    )
    state["latency_profile"] = profile.model_copy(
        update={
            "prompt_bytes": profile.prompt_bytes + prompt_bytes,
            "tool_schema_bytes": (
                profile.tool_schema_bytes + tool_schema_bytes
            ),
        }
    )


def _int_setting(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError("model integer setting must be numeric")
    return int(value)


def _float_setting(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError("model float setting must be numeric")
    return float(value)


def _draft_from_turn(
    turn: ToolUseResult,
    *,
    request: ModelRequest,
) -> ModelTurnDraft:
    if turn.tool_calls:
        origin = ToolCallOrigin(
            request_id=request.request_id,
            toolset_revision=request.toolset_revision,
            exposed_tool_names=request.exposed_tool_names,
        )
        return ModelTurnDraft(
            action="execute",
            tool_calls=tuple(
                ToolCallPlan(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=dict(call.input),
                    origin=origin,
                )
                for call in turn.tool_calls
            ),
        )
    if turn.stop_reason is StopReason.MAX_TOKENS:
        return ModelTurnDraft(
            action="pause",
            pause_reason="Model output reached its configured token limit.",
        )
    return ModelTurnDraft(
        action="finish",
        final_answer=turn.text or "The model returned an empty final response.",
    )


def _scope_model_tool_call_ids(
    turn: ToolUseResult,
    *,
    request_id: str,
) -> ToolUseResult:
    """Make provider-local tool IDs deterministic and unique per request."""

    if not turn.tool_calls:
        return turn
    scoped_calls = [
        ModelToolCall(
            id=(
                "tc_"
                + hashlib.sha256(
                    f"{request_id}\0{index}\0{call.id}".encode()
                ).hexdigest()[:20]
            ),
            name=call.name,
            input=dict(call.input),
        )
        for index, call in enumerate(turn.tool_calls)
    ]
    return turn.model_copy(update={"tool_calls": scoped_calls})


def create_loop_model_turn_provider(
    registry: ModelResolver,
    selection: ModelSelectionPolicy,
    *,
    registry_snapshot: Mapping[str, Tool],
    resident_tool_names: Sequence[str],
    disabled_tool_names: Sequence[str] = (),
    stream_sink: object | None = None,
    skill_runtime: SkillRuntime | None = None,
    goal_spec: GoalSpec | None = None,
) -> LLMLoopModelTurnProvider:
    resolved = registry.resolve_for_node(
        node_model=selection.tool_decision_model,
        node_name="tool_decision",
    )
    gateway = resolved.gateway
    if gateway is None:
        raise RuntimeError("resolved model does not provide an LLM gateway")
    provider = resolved.provider
    model = resolved.model
    supports_native_tools = resolved.supports_native_tools
    return LLMLoopModelTurnProvider(
        gateway,
        model=model,
        provider=provider,
        supports_native_tools=supports_native_tools,
        registry_snapshot=registry_snapshot,
        resident_tool_names=resident_tool_names,
        disabled_tool_names=disabled_tool_names,
        kwargs=resolved.kwargs,
        context_window_tokens=resolved.context_window_tokens,
        stream_sink=stream_sink,
        skill_runtime=skill_runtime,
        goal_spec=goal_spec,
    )


__all__ = [
    "LLMLoopModelTurnProvider",
    "LoopModelDecision",
    "create_loop_model_turn_provider",
    "parse_loop_model_turn",
]

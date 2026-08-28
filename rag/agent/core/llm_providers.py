from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from rag.agent.core.definition import AgentRuntimePolicy, ModelSelectionPolicy
from rag.agent.core.llm_registry import ModelResolver
from rag.agent.core.messages import (
    ModelMessage,
    StopReason,
    ToolUseResult,
    canonical_json_text,
    model_message_payload,
)
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
from rag.agent.core.runtime_diagnostics import AgentLatencyProfile
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import ModelTurnEnvelope
from rag.agent.loop.state import LoopState, ModelTurnDraft
from rag.agent.skills.runtime import SkillRuntime
from rag.agent.streaming.events import (
    ItemDeltaKind,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
    item_completed,
    item_delta,
    item_started,
)
from rag.agent.tools.selection import select_tools
from rag.agent.tools.tool import JsonValue, Tool, ToolCallOrigin
from rag.providers.llm_gateway import (
    LLMGateway,
    ProviderDelta,
    ProviderDeltaChannel,
)
from rag.schema.llm import LLMCallStage


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
        frozen_run_context = (
            ()
            if file_manifest is None or not file_manifest.files
            else (
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
                ),
            )
        )
        settings = self._model_settings(definition.model_selection)
        initial_message, context_transcript = split_turn_context(
            conversation_history=state["conversation_history"],
            turn_transcript=state["turn_transcript"],
        )
        context = build_stable_context(
            instructions=tuple(instructions),
            frozen_run_context=frozen_run_context,
            initial_user_task=initial_message,
            initial_memory=tuple(state.get("persistent_memories", ())),
            transcript=context_transcript,
        )
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
        context = _project_model_context(
            context,
            max_input_tokens=min(model_input_limit, stage_input_limit),
        )
        request = build_model_request(
            request_id=(
                f"{state['run_config'].turn_id}:turn:{state['iteration']}"
            ),
            context=context,
            selected_tools=selected_tools,
            settings=settings,
        )
        _record_request_sizes(state, request)
        items = _ModelItemEmitter(
            sink=self._stream_sink,
            turn_id=state["run_config"].turn_id,
            iteration=state["iteration"],
        )
        try:
            response = await self._gateway.agenerate_model_request(
                stage=LLMCallStage.TOOL_DECISION,
                request=request,
                provider=self._provider,
                supports_native_tools=self._supports_native_tools,
                stream=self._stream_sink is not None,
                delta_sink=(
                    items.emit_delta
                    if self._stream_sink is not None
                    else None
                ),
            )
            turn = response.turn
            if not isinstance(turn, ToolUseResult):
                raise TypeError(
                    "gateway must return a provider-neutral ToolUseResult"
                )
            await items.complete(turn)
        except BaseException as exc:
            await items.close_partial(exc)
            raise
        record = bind_model_call_record(
            request=request,
            provider_wire_hash=response.provider_wire_hash,
            usage=response.usage,
        )
        assistant_message = ModelMessage(
            role="assistant",
            content=turn.text,
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
        )



class _ModelItemEmitter:
    """Translate provider channels into canonical Item lifecycles."""

    _KINDS = {
        ProviderDeltaChannel.TEXT: (
            TurnItemKind.AGENT_MESSAGE,
            ItemDeltaKind.TEXT,
            "agent",
        ),
        ProviderDeltaChannel.REASONING: (
            TurnItemKind.REASONING,
            ItemDeltaKind.REASONING,
            "reasoning",
        ),
        ProviderDeltaChannel.PLAN: (
            TurnItemKind.PLAN,
            ItemDeltaKind.PLAN,
            "provider_plan",
        ),
    }

    def __init__(
        self,
        *,
        sink: object | None,
        turn_id: str,
        iteration: int,
    ) -> None:
        self._sink = sink
        self._turn_id = turn_id
        self._iteration = iteration
        self._started: set[ProviderDeltaChannel] = set()
        self._completed: set[ProviderDeltaChannel] = set()
        self._buffers: dict[ProviderDeltaChannel, list[str]] = {
            channel: [] for channel in ProviderDeltaChannel
        }

    async def emit_delta(self, delta: ProviderDelta) -> None:
        if not isinstance(delta, ProviderDelta):
            raise TypeError("provider delta sink requires ProviderDelta values")
        if not delta.content:
            return
        item_kind, delta_kind, prefix = self._KINDS[delta.channel]
        item_id = f"{prefix}:{self._turn_id}:{self._iteration}"
        if delta.channel not in self._started:
            await self._emit(
                item_started(
                    turn_id=self._turn_id,
                    item_id=item_id,
                    item_kind=item_kind,
                    iteration=self._iteration,
                )
            )
            self._started.add(delta.channel)
        self._buffers[delta.channel].append(delta.content)
        await self._emit(
            item_delta(
                turn_id=self._turn_id,
                item_id=item_id,
                item_kind=item_kind,
                delta_kind=delta_kind,
                delta=delta.content,
                iteration=self._iteration,
            )
        )

    async def complete(self, turn: ToolUseResult) -> None:
        if self._sink is None:
            return
        for channel in (
            ProviderDeltaChannel.REASONING,
            ProviderDeltaChannel.PLAN,
        ):
            if channel in self._started:
                await self._complete_channel(
                    channel,
                    status=ItemStatus.SUCCESS,
                    data={"content": "".join(self._buffers[channel])},
                )
        if ProviderDeltaChannel.TEXT not in self._started:
            item_kind, _, prefix = self._KINDS[ProviderDeltaChannel.TEXT]
            await self._emit(
                item_started(
                    turn_id=self._turn_id,
                    item_id=f"{prefix}:{self._turn_id}:{self._iteration}",
                    item_kind=item_kind,
                    iteration=self._iteration,
                )
            )
            self._started.add(ProviderDeltaChannel.TEXT)
        message = ModelMessage(
            role="assistant",
            content=turn.text,
            tool_calls=tuple(turn.tool_calls),
        )
        payload = model_message_payload(message)
        await self._complete_channel(
            ProviderDeltaChannel.TEXT,
            status=ItemStatus.SUCCESS,
            data={
                "content": payload["content"],
                "tool_calls": payload["tool_calls"],
            },
        )

    async def close_partial(self, exc: BaseException) -> None:
        status = (
            ItemStatus.CANCELLED
            if isinstance(exc, BaseExceptionGroup)
            and any(
                isinstance(child, asyncio.CancelledError)
                for child in exc.exceptions
            )
            else ItemStatus.CANCELLED
            if isinstance(exc, asyncio.CancelledError)
            else ItemStatus.FAILED
        )
        for channel in (
            ProviderDeltaChannel.REASONING,
            ProviderDeltaChannel.PLAN,
            ProviderDeltaChannel.TEXT,
        ):
            if channel not in self._started:
                continue
            if channel in self._completed:
                continue
            await self._complete_channel(
                channel,
                status=status,
                data={"content": "".join(self._buffers[channel])},
                error=None if status is ItemStatus.CANCELLED else str(exc),
            )

    async def _complete_channel(
        self,
        channel: ProviderDeltaChannel,
        *,
        status: ItemStatus,
        data: dict[str, JsonValue],
        error: str | None = None,
    ) -> None:
        item_kind, _, prefix = self._KINDS[channel]
        self._completed.add(channel)
        await self._emit(
            item_completed(
                turn_id=self._turn_id,
                item_id=f"{prefix}:{self._turn_id}:{self._iteration}",
                item_kind=item_kind,
                status=status,
                data=data,
                iteration=self._iteration,
                error=error,
            )
        )

    async def _emit(self, event: StreamEvent) -> None:
        emit = getattr(self._sink, "emit", None)
        if callable(emit):
            await emit(event)


def _project_model_context(
    context: StableModelContext,
    *,
    max_input_tokens: int,
) -> StableModelContext:
    """Bound model-visible history without mutating canonical Session history."""

    if not context.transcript:
        return context
    maximum_bytes = max_input_tokens * 4
    visible = (*context.stable_messages, *context.transcript)
    if _messages_size(visible) <= maximum_bytes:
        return context

    tail_budget = max(512, maximum_bytes // 2)
    tail_start = len(context.transcript)
    used = 0
    for index in range(len(context.transcript) - 1, -1, -1):
        size = _message_size(context.transcript[index])
        if tail_start < len(context.transcript) and used + size > tail_budget:
            break
        tail_start = index
        used += size
    tail_start = _extend_tail_for_tool_pair(context.transcript, tail_start)
    tail = context.transcript[tail_start:]
    covered = context.transcript[:tail_start]
    summary_limit = min(12_000, max(256, maximum_bytes // 4))
    summary = _deterministic_transcript_summary(
        covered,
        max_chars=summary_limit,
    )
    return context.compact(summary=summary, retained_tail=tail)


def _extend_tail_for_tool_pair(
    transcript: tuple[ModelMessage, ...],
    start: int,
) -> int:
    if start <= 0 or start >= len(transcript):
        return start
    first = transcript[start]
    if first.role != "tool" or first.tool_call_id is None:
        return start
    for index in range(start - 1, -1, -1):
        message = transcript[index]
        if any(call.id == first.tool_call_id for call in message.tool_calls):
            return index
    return start


def _deterministic_transcript_summary(
    messages: tuple[ModelMessage, ...],
    *,
    max_chars: int,
) -> str:
    if not messages:
        return "Earlier conversation omitted to fit the model context window."
    lines = [
        f"{message.role}: {canonical_json_text(model_message_payload(message))}"
        for message in messages
    ]
    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    return summary[:max_chars].rstrip() + " [truncated]"


def _messages_size(messages: Sequence[ModelMessage]) -> int:
    return sum(_message_size(message) for message in messages)


def _message_size(message: ModelMessage) -> int:
    return len(
        canonical_json_text(model_message_payload(message)).encode("utf-8")
    )


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


def create_loop_model_turn_provider(
    registry: ModelResolver,
    selection: ModelSelectionPolicy,
    *,
    registry_snapshot: Mapping[str, Tool],
    resident_tool_names: Sequence[str],
    disabled_tool_names: Sequence[str] = (),
    stream_sink: object | None = None,
    skill_runtime: SkillRuntime | None = None,
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
    )


__all__ = [
    "LLMLoopModelTurnProvider",
    "LoopModelDecision",
    "create_loop_model_turn_provider",
    "parse_loop_model_turn",
]

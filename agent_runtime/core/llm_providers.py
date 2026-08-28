from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, cast

from pydantic import BaseModel, Field

from agent_runtime.core.definition import AgentRuntimePolicy, ModelSelectionPolicy
from agent_runtime.core.goal_contract import GoalSpec
from agent_runtime.core.llm_registry import ModelResolver
from agent_runtime.core.messages import (
    ModelMessage,
    StopReason,
    ToolUseResult,
    canonical_json_text,
    context_event_message,
    model_message_payload,
)
from agent_runtime.core.messages import ToolCall as ModelToolCall
from agent_runtime.core.model_request import (
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
from agent_runtime.core.observations import (
    grounded_workspace_paths,
    runtime_file_inspection_paths,
    runtime_workspace_change,
    runtime_workspace_file_changes,
)
from agent_runtime.core.runtime_diagnostics import AgentLatencyProfile
from agent_runtime.core.turn_contracts import ToolCallPlan
from agent_runtime.loop.state import LoopState, ModelTurnDraft, ModelTurnEnvelope
from agent_runtime.loop.stop_hooks import runtime_verification_after_latest_change
from agent_runtime.modeling.contracts import LLMCallStage
from agent_runtime.modeling.gateway import (
    LLMGateway,
    ProviderDelta,
    ProviderDeltaChannel,
    model_request_input_text,
)
from agent_runtime.skills.runtime import SkillRuntime
from agent_runtime.streaming.events import (
    ItemDeltaKind,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
    item_completed,
    item_delta,
    item_started,
)
from agent_runtime.tools.selection import select_tools
from agent_runtime.tools.tool import JsonValue, Tool, ToolCallOrigin

_MAX_WORKING_STATE_GROUNDED_PATHS = 32
_MAX_WORKING_STATE_LOCATORS = 32
_MAX_POST_CHANGE_FILE_EVIDENCE = 8
_REPEATED_TOOL_FAILURE_CODE = "repeated_tool_failure"
_MODEL_TOOL_CALL_REJECTED_EVENT = "model_tool_call_rejected"
_TOOL_CALL_CORRECTION_EVENT = "tool_call_correction"
_PATH_ONLY_LOCATOR_FIELDS = frozenset(
    {
        "source_tool",
        "path",
        "name",
        "mime_type",
        "size_bytes",
        "is_dir",
        "file_kind",
        "is_binary",
        "readable_as_text",
        "truncated",
    }
)


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
    decision = value if isinstance(value, LoopModelDecision) else LoopModelDecision.model_validate(value)
    calls = tuple(decision.tool_calls)
    if calls:
        return ModelTurnDraft(action="execute", tool_calls=calls)
    if decision.action == "finish":
        return ModelTurnDraft(action="finish", final_answer=decision.final_answer)
    if decision.action == "pause":
        return ModelTurnDraft(
            action="pause",
            pause_reason=(
                decision.pause_reason or decision.needs_user_input or decision.stop_reason or decision.thought
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
        self._goal_spec = None if goal_spec is None else goal_spec.model_copy(deep=True)

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
        resident_names = tuple(state_resident_names or self._resident_tool_names)
        disabled_names = tuple(state.get("disabled_tool_names") or self._disabled_tool_names)
        selected_tools = select_tools(
            self._registry_snapshot,
            resident_names=resident_names,
            active_names=tuple(state.get("active_tool_names", ())),
            disabled_names=disabled_names,
        )
        skill_context = "" if self._skill_runtime is None else self._skill_runtime.render_prompt_context(state)
        instructions = [definition.system_instructions or "You are a helpful agent."]
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
                        "instruction": ("Use these exact workspace-relative paths when calling file tools."),
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
        context_transcript, tool_call_correction = _project_tool_call_correction_context(
            context_transcript,
            selected_tools=selected_tools,
        )
        context = build_stable_context(
            instructions=tuple(instructions),
            frozen_run_context=tuple(frozen_run_context),
            initial_user_task=initial_message,
            transcript=context_transcript,
        )
        working_state = _working_state_message(
            state,
            goal_spec=self._goal_spec,
        )
        if working_state is not None:
            context = context.append_message(working_state)
        if tool_call_correction is not None:
            context = context.append_message(tool_call_correction)
        context_limit = state["run_config"].max_context_tokens or self._context_window_tokens
        model_input_limit = max(
            256,
            context_limit - settings.max_output_tokens - 1_024,
        )
        stage_input_limit = self._gateway.effective_stage_budget(
            LLMCallStage.TOOL_DECISION,
            kwargs={"max_tokens": settings.max_output_tokens},
        ).max_input_tokens
        request_id = f"{state['run_config'].turn_id}:turn:{state['iteration']}"

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
            return self._gateway.token_accounting.count(accounted_input)

        context = _project_model_context(
            context,
            max_input_tokens=min(model_input_limit, stage_input_limit),
            max_summary_chars=(state["run_config"].memory_policy.max_working_summary_chars),
            protected_tail_start=(len(state["conversation_history"]) - 1 if state["conversation_history"] else None),
            input_token_count=request_input_tokens,
        )
        request = request_for(context)
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
            turn = _scope_model_tool_call_ids(
                turn,
                request_id=request.request_id,
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
        assistant_message = _assistant_message_from_turn(turn)
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
            top_p=(None if top_p is None else _float_setting(top_p)),
            parallel_tool_calls=bool(self._kwargs.get("parallel_tool_calls", True)),
            seed=(None if seed is None else _int_setting(seed)),
            provider_options=cast(
                Mapping[str, JsonValue],
                provider_options,
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


def _project_tool_call_correction_context(
    transcript: Sequence[ModelMessage],
    *,
    selected_tools: Sequence[Tool],
) -> tuple[tuple[ModelMessage, ...], ModelMessage | None]:
    """Replace trailing provider rejections with one schema-derived retry hint.

    The checkpoint transcript remains append-only.  This projection is only for
    the next model request, so raw rejected generations stay available for
    diagnostics without teaching the model to copy them.
    """

    rejection_payloads = tuple(_tool_call_rejection_payload(message) for message in transcript)
    trailing: list[Mapping[str, object]] = []
    for payload in reversed(rejection_payloads):
        if payload is None:
            break
        trailing.append(payload)
    retained = tuple(
        message
        for message, payload in zip(
            transcript,
            rejection_payloads,
            strict=True,
        )
        if payload is None
    )
    if not trailing:
        return retained, None
    trailing.reverse()
    return retained, context_event_message(
        _TOOL_CALL_CORRECTION_EVENT,
        _tool_call_correction_payload(
            trailing,
            selected_tools=selected_tools,
        ),
    )


def _tool_call_rejection_payload(
    message: ModelMessage,
) -> Mapping[str, object] | None:
    if message.role != "context":
        return None
    try:
        event = json.loads(message.content)
    except (TypeError, ValueError):
        return None
    if not isinstance(event, Mapping) or event.get("event_type") != _MODEL_TOOL_CALL_REJECTED_EVENT:
        return None
    payload = event.get("payload")
    return payload if isinstance(payload, Mapping) else None


def _tool_call_correction_payload(
    rejections: Sequence[Mapping[str, object]],
    *,
    selected_tools: Sequence[Tool],
) -> Mapping[str, JsonValue]:
    latest = rejections[-1]
    failed_generation = latest.get("failed_generation")
    raw = failed_generation if isinstance(failed_generation, str) else ""
    validation_error_value = latest.get("validation_error")
    validation_error = (
        validation_error_value[:1_000]
        if isinstance(validation_error_value, str)
        else "Provider rejected the generated tool call."
    )
    if raw and raw in validation_error:
        validation_error = validation_error.replace(
            raw,
            "[rejected generation omitted]",
        )

    tools_by_name = {tool.definition.name: tool for tool in selected_tools}
    tool_name, attempted_arguments, failure_kind = _inspect_rejected_tool_generation(
        raw,
        selected_tool_names=frozenset(tools_by_name),
    )
    selected = tools_by_name.get(tool_name or "")
    allowed_names: tuple[str, ...] = ()
    required_names: tuple[str, ...] = ()
    additional_properties_allowed: bool | None = None
    if selected is not None:
        schema = selected.definition.input_schema
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            allowed_names = tuple(sorted(name for name in properties if isinstance(name, str)))
        required = schema.get("required")
        if isinstance(required, Sequence) and not isinstance(
            required,
            (str, bytes),
        ):
            required_names = tuple(sorted(name for name in required if isinstance(name, str)))
        additional_properties_allowed = schema.get("additionalProperties", True) is not False
    rejected_names = (
        ()
        if selected is None or attempted_arguments is None
        else tuple(sorted(name for name in attempted_arguments if isinstance(name, str) and name not in allowed_names))
    )
    return {
        "recovery": "retry_native_tool_call",
        "instruction": (
            "Retry one native tool call with a strict JSON object. Use only "
            "argument names allowed by the attached tool schema; do not copy "
            "the rejected generation."
        ),
        "attempt_count": len(rejections),
        "failure_kind": failure_kind,
        "tool_name": tool_name,
        "allowed_argument_names": allowed_names,
        "required_argument_names": required_names,
        "rejected_argument_names": rejected_names,
        "additional_properties_allowed": additional_properties_allowed,
        "validation_error": validation_error,
        "failed_generation_chars": len(raw),
        "failed_generation_sha256": (hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else None),
    }


def _inspect_rejected_tool_generation(
    value: str,
    *,
    selected_tool_names: frozenset[str],
) -> tuple[str | None, Mapping[str, object] | None, str]:
    text = value.strip()
    tool_name: str | None = None
    arguments: Mapping[str, object] | None = None
    parsed_json = False
    candidate = text

    if text.startswith("<function=") and text.endswith("</function>"):
        header_end = text.find(">")
        if header_end > len("<function="):
            tool_name = text[len("<function=") : header_end]
            candidate = text[header_end + 1 : -len("</function>")]
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError):
        parsed = None
    else:
        parsed_json = True

    if isinstance(parsed, Mapping):
        function = parsed.get("function")
        envelope = function if isinstance(function, Mapping) else parsed
        candidate_name = envelope.get("name")
        if not isinstance(candidate_name, str):
            candidate_name = envelope.get("tool_name")
        if isinstance(candidate_name, str):
            tool_name = candidate_name
        raw_arguments = envelope.get("arguments")
        if raw_arguments is None:
            raw_arguments = envelope.get("attempted_arguments")
        if isinstance(raw_arguments, Mapping):
            arguments = raw_arguments
        elif isinstance(raw_arguments, str):
            try:
                decoded_arguments = json.loads(raw_arguments)
            except (TypeError, ValueError):
                parsed_json = False
            else:
                if isinstance(decoded_arguments, Mapping):
                    arguments = decoded_arguments
                else:
                    parsed_json = False
        elif text.startswith("<function="):
            arguments = parsed

    if tool_name not in selected_tool_names:
        match = re.search(
            r'"(?:name|tool_name)"\s*:\s*"([A-Za-z0-9_.:-]+)"',
            text,
        )
        tool_name = match.group(1) if match is not None and match.group(1) in selected_tool_names else None
    return (
        tool_name,
        arguments,
        "schema_validation" if parsed_json else "invalid_json",
    )


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
    if not context.transcript or current_tokens <= max_input_tokens:
        return context

    maximum_tail_start = (
        len(context.transcript)
        if protected_tail_start is None
        else min(
            max(protected_tail_start, 0),
            len(context.transcript),
        )
    )
    latest_tool_pair_start = _latest_tool_pair_start(context.transcript)
    if latest_tool_pair_start is not None:
        maximum_tail_start = min(
            maximum_tail_start,
            latest_tool_pair_start,
        )

    best = context
    best_tokens = current_tokens
    summary_limits = _projection_summary_limits(max_summary_chars)
    for tail_start in range(0, maximum_tail_start + 1):
        for summary_limit in summary_limits:
            candidate = context.project_compaction(
                tail_start=tail_start,
                max_summary_chars=summary_limit,
                project_tool_results=True,
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


def _latest_tool_pair_start(
    transcript: Sequence[ModelMessage],
) -> int | None:
    for tool_index in range(len(transcript) - 1, -1, -1):
        tool_message = transcript[tool_index]
        if tool_message.role != "tool" or tool_message.tool_call_id is None:
            continue
        for assistant_index in range(tool_index - 1, -1, -1):
            assistant_message = transcript[assistant_index]
            if any(call.id == tool_message.tool_call_id for call in assistant_message.tool_calls):
                original_call_id = _referenced_original_tool_call_id(tool_message)
                if original_call_id is None:
                    return assistant_index
                original_index = _assistant_tool_call_index(
                    transcript[:assistant_index],
                    original_call_id,
                )
                return assistant_index if original_index is None else original_index
        return tool_index
    return None


def _referenced_original_tool_call_id(
    message: ModelMessage,
) -> str | None:
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(payload, Mapping)
        or payload.get("is_error") is not True
        or payload.get("error_code") != _REPEATED_TOOL_FAILURE_CODE
    ):
        return None
    structured = payload.get("structured_content")
    if not isinstance(structured, Mapping) or structured.get("repeated_failure") is not True:
        return None
    original = structured.get("original_tool_call_id")
    return original if isinstance(original, str) and original else None


def _assistant_tool_call_index(
    transcript: Sequence[ModelMessage],
    tool_call_id: str,
) -> int | None:
    for index in range(len(transcript) - 1, -1, -1):
        if any(call.id == tool_call_id for call in transcript[index].tool_calls):
            return index
    return None


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
    protected_tool_call_ids = _protected_transcript_tool_call_ids(state["turn_transcript"])
    workspace_change_tool_call_ids = tuple(
        result.tool_call_id for result in state["tool_results"] if runtime_workspace_change(result) is not None
    )
    latest_change_index = max(
        (index for index, result in enumerate(state["tool_results"]) if runtime_workspace_change(result) is not None),
        default=-1,
    )
    verification_evidence = runtime_verification_after_latest_change(state)
    verification_tool_call_ids = verification_evidence.successful_tool_call_ids
    post_change_file_inspection = _post_change_file_inspection(
        state,
        latest_change_index=latest_change_index,
    )
    verification_constraint_pending = any(
        constraint.required
        and constraint.expected_value is True
        and constraint.constraint_type == "verification_after_change"
        and not verification_evidence.satisfied
        for constraint in (() if goal_spec is None else goal_spec.constraints)
    )
    runtime_requirements = tuple(
        {
            "constraint_id": constraint.constraint_id,
            "constraint_type": constraint.constraint_type,
            "expected_value": cast(JsonValue, constraint.expected_value),
            "observation": (
                "observed"
                if (constraint.constraint_type == "workspace_change" and workspace_change_tool_call_ids)
                or (
                    constraint.constraint_type == "verification_after_change"
                    and verification_evidence.satisfied
                )
                else "pending"
            ),
            "requirement": _runtime_requirement_description(constraint.constraint_type),
        }
        for constraint in (() if goal_spec is None else goal_spec.constraints)
        if (
            constraint.required
            and constraint.expected_value is True
            and constraint.constraint_type in {"workspace_change", "verification_after_change"}
        )
    )
    if (
        not memory_state.recent_observations
        and not memory_state.known_locators
        and not memory_state.verified_workspace_paths
        and plan is None
        and not runtime_requirements
        and post_change_file_inspection is None
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
    manifest_paths = () if file_manifest is None else tuple(entry.path for entry in file_manifest.files)
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
    known_locators = _project_working_locators(
        memory_state.known_locators,
    )
    runtime_evidence: dict[str, JsonValue] = {
        "authority": "runtime",
        "grounded_paths": grounded_paths,
        "grounded_path_count": len(verified_grounded_paths),
        "grounded_paths_truncated": (len(grounded_paths) < len(verified_grounded_paths)),
        "recent_observations": tuple(
            {
                "tool_call_id": observation.tool_call_id,
                "tool_name": observation.tool_name,
                "status": observation.status,
                "error": observation.error,
                "warnings": tuple(observation.warnings),
            }
            for observation in memory_state.recent_observations
            if (observation.tool_call_id not in protected_tool_call_ids)
        ),
        "known_locators": tuple(cast(Mapping[str, JsonValue], locator) for locator in known_locators),
        "known_locator_count": len(memory_state.known_locators),
        "known_locators_compacted": (len(known_locators) < len(memory_state.known_locators)),
        "workspace_change_tool_call_ids": tuple(workspace_change_tool_call_ids),
        "verification_tool_call_ids": verification_tool_call_ids,
    }
    if post_change_file_inspection is not None:
        runtime_evidence["post_change_file_inspection"] = post_change_file_inspection
        if (
            post_change_file_inspection["observation"] == "observed"
            and not verification_constraint_pending
        ):
            runtime_evidence["completion_guidance"] = {
                "authority": "runtime",
                "condition": "literal_file_task_and_existing_result_satisfies_target",
                "action": "finish",
                "prohibited_reconfirmation": (
                    "repeat_inspection",
                    "repeat_mutation",
                    "run_command_only_to_reconfirm_file_content",
                ),
            }

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
            "runtime_evidence": runtime_evidence,
        },
    )


def _post_change_file_inspection(
    state: LoopState,
    *,
    latest_change_index: int,
) -> Mapping[str, JsonValue] | None:
    if latest_change_index < 0:
        return None
    change_result = state["tool_results"][latest_change_index]
    if change_result.is_error:
        return None
    changed_paths = tuple(
        path
        for path, _before_sha256, _after_sha256 in runtime_workspace_file_changes(change_result)
    )
    if not changed_paths:
        return None

    changed_path_set = frozenset(changed_paths)
    inspection_tool_call_ids: list[str] = []
    inspected_paths: dict[str, None] = {}
    calls = state["canonical_tool_calls"]
    for result in state["tool_results"][latest_change_index + 1 :]:
        paths = runtime_file_inspection_paths(
            result,
            call=calls.get(result.tool_call_id),
        )
        related_paths = tuple(path for path in paths if path in changed_path_set)
        if not related_paths:
            continue
        inspection_tool_call_ids.append(result.tool_call_id)
        for path in related_paths:
            inspected_paths.setdefault(path, None)

    observation = "observed" if inspection_tool_call_ids else "pending"
    data_artifact = any(
        path.casefold().endswith(
            (".xlsx", ".xlsm", ".pdf", ".csv", ".tsv", ".json")
        )
        for path in changed_paths
    )
    if observation == "observed":
        literal_file_guidance: Mapping[str, JsonValue] = {
            "action": "finish_if_existing_result_satisfies_task",
            "maximum_additional_file_inspections": 0,
            "next_response_tool_calls": 0,
        }
    else:
        literal_file_guidance = {
            "action": "choose_one_targeted_inspection_then_finish",
            "choose_exactly_one_of": (
                ("inspect_data_file",)
                if data_artifact
                else ("read_file", "search_text")
            ),
            "maximum_additional_file_inspections": 1,
            "never_batch_alternatives": True,
            "do_not_pair_positive_and_negative_searches": True,
        }

    return {
        "authority": "runtime",
        "latest_change_tool_call_id": change_result.tool_call_id,
        "changed_paths": changed_paths,
        "observation": observation,
        "inspection_tool_call_ids": tuple(
            inspection_tool_call_ids[-_MAX_POST_CHANGE_FILE_EVIDENCE:]
        ),
        "inspected_paths": tuple(inspected_paths)[-_MAX_POST_CHANGE_FILE_EVIDENCE:],
        "scope": (
            "data_file_structure_and_content"
            if data_artifact
            else "file_content"
        ),
        "semantic_target_satisfied": "not_evaluated",
        "guidance": {
            "literal_file_task": literal_file_guidance,
            "behavioral_code_task": (
                "file_content_evidence_does_not_replace_pending_runtime_verification"
            ),
        },
    }


def _runtime_requirement_description(constraint_type: str) -> str:
    if constraint_type == "workspace_change":
        return (
            "A runtime-observed write must change workspace contents; "
            "prose and pre-change verification do not satisfy this."
        )
    return (
        "A recognized behavior check or exact generated-artifact inspection "
        "must succeed after the latest workspace change; stale or failed checks "
        "do not satisfy this."
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


def _project_working_locators(
    locators: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    precise = tuple(locator for locator in locators if not set(locator).issubset(_PATH_ONLY_LOCATOR_FIELDS))
    if len(precise) <= _MAX_WORKING_STATE_LOCATORS:
        return precise
    return precise[-_MAX_WORKING_STATE_LOCATORS:]


def _protected_transcript_tool_call_ids(
    transcript: Sequence[ModelMessage],
) -> frozenset[str]:
    protected_start = _latest_tool_pair_start(transcript)
    if protected_start is None:
        return frozenset()
    return frozenset(
        message.tool_call_id
        for message in transcript[protected_start:]
        if (message.role == "tool" and message.tool_call_id is not None)
    )


def _record_request_sizes(state: LoopState, request: ModelRequest) -> None:
    profile = state.get("latency_profile")
    if not isinstance(profile, AgentLatencyProfile):
        profile = AgentLatencyProfile()
    prompt_bytes = len(
        canonical_json_text(tuple(model_message_payload(message) for message in request.messages)).encode("utf-8")
    )
    tool_schema_bytes = len(
        canonical_json_text(tuple(tool_definition_payload(tool) for tool in request.tools)).encode("utf-8")
    )
    state["latency_profile"] = profile.model_copy(
        update={
            "prompt_bytes": profile.prompt_bytes + prompt_bytes,
            "tool_schema_bytes": (profile.tool_schema_bytes + tool_schema_bytes),
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


def _assistant_message_from_turn(turn: ToolUseResult) -> ModelMessage | None:
    """Return only provider-replayable assistant history.

    Some reasoning models can consume their entire output budget before
    producing visible content or a tool call.  Persisting that response as an
    assistant message leaves ``content=null`` and no ``tool_calls`` on resume,
    which OpenAI-compatible providers reject.  The model-call record and pause
    reason still preserve what happened; the unusable wire message must not
    enter canonical conversation history.
    """

    if (
        turn.stop_reason is StopReason.MAX_TOKENS
        and not turn.text
        and not turn.tool_calls
    ):
        return None
    return ModelMessage(
        role="assistant",
        content=turn.text,
        reasoning_content=turn.reasoning_content,
        tool_calls=tuple(turn.tool_calls),
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
            id=("tc_" + hashlib.sha256(f"{request_id}\0{index}\0{call.id}".encode()).hexdigest()[:20]),
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

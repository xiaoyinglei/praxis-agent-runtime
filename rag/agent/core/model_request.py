from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real

from rag.agent.core.messages import (
    ModelMessage,
    canonical_json_text,
    context_event_message,
    model_message_payload,
    snapshot_model_message,
    tool_result_message,
)
from rag.agent.core.turn_contracts import ToolManifest, ToolManifestEntry
from rag.agent.tools.tool import (
    JsonValue,
    Tool,
    ToolDefinition,
    ToolResult,
    _require_non_empty_string,
    json_schema_output,
)
from rag.schema.llm import LLMUsage

CANONICAL_REQUEST_REVISION = "canonical-model-request-v1"
STABLE_CONTEXT_REVISION = "stable-model-context-v1"
COMPACTION_REVISION = "context-compaction-v1"


class ToolChoiceMode(StrEnum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    NAMED = "named"


@dataclass(frozen=True, slots=True)
class ToolChoice:
    mode: ToolChoiceMode
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ToolChoiceMode):
            raise TypeError("tool choice mode must be a ToolChoiceMode")
        if self.mode is ToolChoiceMode.NAMED:
            _require_non_empty_string(self.name, field_name="named tool choice")
        elif self.name is not None:
            raise ValueError("tool choice name is only valid for named mode")

    @classmethod
    def auto(cls) -> ToolChoice:
        return cls(ToolChoiceMode.AUTO)

    @classmethod
    def none(cls) -> ToolChoice:
        return cls(ToolChoiceMode.NONE)

    @classmethod
    def required(cls) -> ToolChoice:
        return cls(ToolChoiceMode.REQUIRED)

    @classmethod
    def named(cls, name: str) -> ToolChoice:
        return cls(ToolChoiceMode.NAMED, name=name)


@dataclass(frozen=True, slots=True)
class ModelSettings:
    model: str
    max_output_tokens: int = 2048
    temperature: float = 0.0
    top_p: float | None = 1.0
    parallel_tool_calls: bool = True
    seed: int | None = None
    provider_options: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string(self.model, field_name="model")
        if (
            not isinstance(self.max_output_tokens, int)
            or isinstance(self.max_output_tokens, bool)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        _validate_real_range(
            self.temperature,
            field_name="temperature",
            minimum=0.0,
            maximum=2.0,
            minimum_inclusive=True,
        )
        if self.top_p is not None:
            _validate_real_range(
                self.top_p,
                field_name="top_p",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            )
        if type(self.parallel_tool_calls) is not bool:
            raise TypeError("parallel_tool_calls must be a bool")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise TypeError("seed must be an integer or None")
        frozen_options = json_schema_output(None, self.provider_options)
        if not isinstance(frozen_options, Mapping):
            raise TypeError("provider_options must be an object")
        object.__setattr__(self, "provider_options", frozen_options)


@dataclass(frozen=True, slots=True)
class ContextBlock:
    name: str
    content: JsonValue

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, field_name="context block name")
        object.__setattr__(
            self,
            "content",
            json_schema_output(None, self.content),
        )


@dataclass(frozen=True, slots=True)
class StableModelContext:
    instructions: tuple[str, ...]
    frozen_run_context: tuple[ContextBlock, ...]
    initial_user_task: str
    initial_memory: tuple[str, ...]
    transcript: tuple[ModelMessage, ...]
    context_revision: str
    parent_context_revision: str | None = None
    revision_reason: str = "initial"

    def __post_init__(self) -> None:
        instructions = _ordered_strings(
            self.instructions,
            field_name="instructions",
            require_non_empty_sequence=True,
        )
        if isinstance(self.frozen_run_context, (str, bytes)) or not isinstance(
            self.frozen_run_context,
            Sequence,
        ):
            raise TypeError("frozen_run_context must be a sequence of ContextBlock values")
        blocks: list[ContextBlock] = []
        for block in self.frozen_run_context:
            if not isinstance(block, ContextBlock):
                raise TypeError("frozen_run_context must contain ContextBlock values")
            blocks.append(ContextBlock(name=block.name, content=block.content))
        _require_non_empty_string(self.initial_user_task, field_name="initial_user_task")
        memory = _ordered_strings(self.initial_memory, field_name="initial_memory")
        transcript = _snapshot_messages(self.transcript, field_name="transcript")
        _require_non_empty_string(self.context_revision, field_name="context_revision")
        if self.parent_context_revision is not None:
            _require_non_empty_string(
                self.parent_context_revision,
                field_name="parent_context_revision",
            )
        _require_non_empty_string(self.revision_reason, field_name="revision_reason")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "frozen_run_context", tuple(blocks))
        object.__setattr__(self, "initial_memory", memory)
        object.__setattr__(self, "transcript", transcript)

    @property
    def stable_messages(self) -> tuple[ModelMessage, ...]:
        messages: list[ModelMessage] = [
            ModelMessage(
                role="system",
                content="\n\n".join(self.instructions),
            )
        ]
        messages.extend(
            context_event_message(
                "frozen_run_context",
                {
                    "name": block.name,
                    "content": block.content,
                },
            )
            for block in self.frozen_run_context
        )
        if self.initial_memory:
            messages.append(
                context_event_message(
                    "initial_memory",
                    {"items": self.initial_memory},
                )
            )
        messages.append(ModelMessage(role="user", content=self.initial_user_task))
        return tuple(messages)

    def append_message(self, message: ModelMessage) -> StableModelContext:
        return self._with_transcript((*self.transcript, snapshot_model_message(message)))

    def append_tool_result(self, result: ToolResult) -> StableModelContext:
        return self.append_message(tool_result_message(result))

    def append_context_event(
        self,
        event_type: str,
        payload: Mapping[str, JsonValue],
    ) -> StableModelContext:
        return self.append_message(context_event_message(event_type, payload))

    def append_skill_activation(
        self,
        activation_event: Mapping[str, JsonValue],
    ) -> StableModelContext:
        return self.append_context_event(
            "skill_activation",
            activation_event,
        )

    def compact(
        self,
        *,
        summary: str,
        retained_tail: Sequence[ModelMessage] = (),
    ) -> StableModelContext:
        _require_non_empty_string(summary, field_name="compaction summary")
        if len(summary) > 100_000:
            raise ValueError("compaction summary exceeds 100000 characters")
        tail = _snapshot_messages(retained_tail, field_name="retained_tail")
        if tail and (len(tail) > len(self.transcript) or self.transcript[-len(tail) :] != tail):
            raise ValueError("retained_tail must be a suffix of the transcript")
        event = context_event_message(
            "context_compaction",
            {
                "summary": summary,
                "parent_context_revision": self.context_revision,
            },
        )
        revision = _revision(
            "context",
            {
                "serializer_revision": COMPACTION_REVISION,
                "parent_context_revision": self.context_revision,
                "summary": summary,
                "retained_tail": tuple(model_message_payload(message) for message in tail),
            },
        )
        return StableModelContext(
            instructions=self.instructions,
            frozen_run_context=self.frozen_run_context,
            initial_user_task=self.initial_user_task,
            initial_memory=self.initial_memory,
            transcript=(event, *tail),
            context_revision=revision,
            parent_context_revision=self.context_revision,
            revision_reason="compaction",
        )

    def project_compaction(
        self,
        *,
        tail_start: int,
        max_summary_chars: int,
    ) -> StableModelContext:
        projected = project_transcript_compaction(
            self.transcript,
            parent_context_revision=self.context_revision,
            tail_start=tail_start,
            max_summary_chars=max_summary_chars,
        )
        if projected == self.transcript:
            return self
        revision = _revision(
            "context",
            {
                "serializer_revision": COMPACTION_REVISION,
                "parent_context_revision": self.context_revision,
                "transcript": tuple(
                    model_message_payload(message)
                    for message in projected
                ),
            },
        )
        return StableModelContext(
            instructions=self.instructions,
            frozen_run_context=self.frozen_run_context,
            initial_user_task=self.initial_user_task,
            initial_memory=self.initial_memory,
            transcript=projected,
            context_revision=revision,
            parent_context_revision=self.context_revision,
            revision_reason="compaction",
        )

    def _with_transcript(
        self,
        transcript: tuple[ModelMessage, ...],
    ) -> StableModelContext:
        return StableModelContext(
            instructions=self.instructions,
            frozen_run_context=self.frozen_run_context,
            initial_user_task=self.initial_user_task,
            initial_memory=self.initial_memory,
            transcript=transcript,
            context_revision=self.context_revision,
            parent_context_revision=self.parent_context_revision,
            revision_reason=self.revision_reason,
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    request_id: str
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolDefinition, ...]
    exposed_tool_names: tuple[str, ...]
    tool_choice: ToolChoice
    settings: ModelSettings
    prompt_revision: str
    toolset_revision: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.request_id, field_name="request_id")
        messages = _snapshot_messages(self.messages, field_name="messages")
        tools = tuple(self.tools)
        if any(not isinstance(tool, ToolDefinition) for tool in tools):
            raise TypeError("tools must contain ToolDefinition values")
        exposed_names = _ordered_names(
            self.exposed_tool_names,
            field_name="exposed_tool_names",
        )
        definition_names = tuple(tool.name for tool in tools)
        if definition_names != exposed_names:
            raise ValueError("exposed_tool_names must exactly match ToolDefinition order")
        if not isinstance(self.tool_choice, ToolChoice):
            raise TypeError("tool_choice must be a ToolChoice")
        if not isinstance(self.settings, ModelSettings):
            raise TypeError("settings must be ModelSettings")
        if self.tool_choice.mode is ToolChoiceMode.NAMED and self.tool_choice.name not in exposed_names:
            raise ValueError("named tool choice must be exposed")
        if self.tool_choice.mode is ToolChoiceMode.REQUIRED and not exposed_names:
            raise ValueError("required tool choice needs at least one tool")
        _require_non_empty_string(
            self.prompt_revision,
            field_name="prompt_revision",
        )
        _require_non_empty_string(
            self.toolset_revision,
            field_name="toolset_revision",
        )
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "exposed_tool_names", exposed_names)


@dataclass(frozen=True, slots=True)
class ModelCallRecord:
    request_id: str
    prompt_revision: str
    toolset_revision: str
    provider_wire_hash: str
    usage: LLMUsage

    def __post_init__(self) -> None:
        _require_non_empty_string(self.request_id, field_name="request_id")
        _require_non_empty_string(
            self.prompt_revision,
            field_name="prompt_revision",
        )
        _require_non_empty_string(
            self.toolset_revision,
            field_name="toolset_revision",
        )
        _require_non_empty_string(
            self.provider_wire_hash,
            field_name="provider_wire_hash",
        )
        if not isinstance(self.usage, LLMUsage):
            raise TypeError("usage must be an LLMUsage")
        if self.usage.usage_source is None:
            raise ValueError("ModelCallRecord requires normalized usage")
        object.__setattr__(self, "usage", self.usage.model_copy(deep=True))


def split_turn_context(
    *,
    conversation_history: Sequence[ModelMessage],
    turn_transcript: Sequence[ModelMessage],
) -> tuple[str, tuple[ModelMessage, ...]]:
    """Project explicit Conversation/Turn state into the model's stable prefix and transcript."""

    history = _snapshot_messages(
        conversation_history,
        field_name="conversation_history",
    )
    current = _snapshot_messages(
        turn_transcript,
        field_name="turn_transcript",
    )
    combined = (*history, *current)
    if not combined:
        raise ValueError("Turn context requires at least one user message")
    first = combined[0]
    if first.role != "user" or not first.content.strip():
        raise ValueError("Turn context must begin with a non-empty user message")
    return first.content, tuple(combined[1:])


def build_stable_context(
    *,
    instructions: Sequence[str],
    frozen_run_context: Sequence[ContextBlock] = (),
    initial_user_task: str,
    initial_memory: Sequence[str] = (),
    transcript: Sequence[ModelMessage] = (),
) -> StableModelContext:
    instruction_values = _ordered_strings(
        instructions,
        field_name="instructions",
        require_non_empty_sequence=True,
    )
    _require_non_empty_string(initial_user_task, field_name="initial_user_task")
    memory_values = _ordered_strings(
        initial_memory,
        field_name="initial_memory",
    )
    if isinstance(frozen_run_context, (str, bytes)) or not isinstance(
        frozen_run_context,
        Sequence,
    ):
        raise TypeError("frozen_run_context must be a sequence of ContextBlock values")
    blocks: list[ContextBlock] = []
    for block in frozen_run_context:
        if not isinstance(block, ContextBlock):
            raise TypeError("frozen_run_context must contain ContextBlock values")
        blocks.append(ContextBlock(name=block.name, content=block.content))
    block_tuple = tuple(blocks)
    transcript_tuple = _snapshot_messages(transcript, field_name="transcript")
    revision_payload: Mapping[str, JsonValue] = {
        "serializer_revision": STABLE_CONTEXT_REVISION,
        "instructions": instruction_values,
        "frozen_run_context": tuple({"name": block.name, "content": block.content} for block in block_tuple),
        "initial_user_task": initial_user_task,
        "initial_memory": memory_values,
    }
    return StableModelContext(
        instructions=instruction_values,
        frozen_run_context=block_tuple,
        initial_user_task=initial_user_task,
        initial_memory=memory_values,
        transcript=transcript_tuple,
        context_revision=_revision("context", revision_payload),
    )


def build_model_request(
    *,
    request_id: str,
    context: StableModelContext,
    selected_tools: Sequence[Tool],
    settings: ModelSettings,
    tool_choice: ToolChoice | None = None,
    dynamic_tail: Sequence[ModelMessage] = (),
) -> ModelRequest:
    if not isinstance(context, StableModelContext):
        raise TypeError("context must be a StableModelContext")
    if not isinstance(settings, ModelSettings):
        raise TypeError("settings must be ModelSettings")
    if isinstance(selected_tools, (str, bytes)) or not isinstance(
        selected_tools,
        Sequence,
    ):
        raise TypeError("selected_tools must be a sequence of Tool values")
    tools = tuple(selected_tools)
    if any(not isinstance(tool, Tool) for tool in tools):
        raise TypeError("selected_tools must contain Tool values")
    names = tuple(tool.definition.name for tool in tools)
    if len(set(names)) != len(names):
        raise ValueError("selected_tools must have unique names")
    choice = tool_choice or (ToolChoice.auto() if tools else ToolChoice.none())
    if not isinstance(choice, ToolChoice):
        raise TypeError("tool_choice must be a ToolChoice")
    tail = _snapshot_messages(dynamic_tail, field_name="dynamic_tail")
    toolset_revision = toolset_revision_for_tools(tools)
    prompt_revision = _revision(
        "prompt",
        {
            "serializer_revision": CANONICAL_REQUEST_REVISION,
            "stable_context": _stable_context_payload(context),
            "context_revision": context.context_revision,
            "toolset_revision": toolset_revision,
            "tool_choice": tool_choice_payload(choice),
            "settings": model_settings_payload(settings),
        },
    )
    return ModelRequest(
        request_id=request_id,
        messages=(*context.stable_messages, *context.transcript, *tail),
        tools=tuple(tool.definition for tool in tools),
        exposed_tool_names=names,
        tool_choice=choice,
        settings=settings,
        prompt_revision=prompt_revision,
        toolset_revision=toolset_revision,
    )


def bind_model_call_record(
    *,
    request: ModelRequest,
    provider_wire_hash: str,
    usage: LLMUsage,
) -> ModelCallRecord:
    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a ModelRequest")
    return ModelCallRecord(
        request_id=request.request_id,
        prompt_revision=request.prompt_revision,
        toolset_revision=request.toolset_revision,
        provider_wire_hash=provider_wire_hash,
        usage=usage,
    )


def toolset_revision_for_tools(tools: Sequence[Tool]) -> str:
    ordered_tools = _ordered_tools(tools)
    return _revision(
        "tools",
        tuple(_tool_contract_payload(tool) for tool in ordered_tools),
    )


def build_tool_manifest(
    *,
    tools: Sequence[Tool],
    resident_tool_names: Sequence[str] = (),
    explicit_tool_names: Sequence[str] = (),
    active_tool_names: Sequence[str] = (),
    provider_serializer_revision: str,
) -> ToolManifest:
    ordered_tools = _ordered_tools(tools)
    resident = _ordered_names(
        resident_tool_names,
        field_name="resident_tool_names",
    )
    explicit = _ordered_names(
        explicit_tool_names,
        field_name="explicit_tool_names",
    )
    active = _ordered_names(
        active_tool_names,
        field_name="active_tool_names",
    )
    ordered_names = (*resident, *explicit, *active)
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("tool manifest orders must contain unique names")
    if tuple(tool.definition.name for tool in ordered_tools) != ordered_names:
        raise ValueError("tools must match resident, explicit, and active order")
    _require_non_empty_string(
        provider_serializer_revision,
        field_name="provider_serializer_revision",
    )
    return ToolManifest(
        entries=tuple(_tool_manifest_entry(tool) for tool in ordered_tools),
        resident_tool_names=resident,
        explicit_tool_names=explicit,
        active_tool_names=active,
        toolset_revision=toolset_revision_for_tools(ordered_tools),
        provider_serializer_revision=provider_serializer_revision,
    )


def model_call_record_payload(
    record: ModelCallRecord,
) -> dict[str, object]:
    if not isinstance(record, ModelCallRecord):
        raise TypeError("record must be a ModelCallRecord")
    usage: dict[str, object] = {
        "logical_input_tokens": record.usage.logical_input_tokens,
        "uncached_input_tokens": record.usage.uncached_input_tokens,
        "cache_read_input_tokens": record.usage.cache_read_input_tokens,
        "cache_write_input_tokens": record.usage.cache_write_input_tokens,
        "output_tokens": record.usage.output_tokens,
        "usage_source": record.usage.usage_source,
        "raw_provider_usage": record.usage.raw_provider_usage,
    }
    return {
        "request_id": record.request_id,
        "prompt_revision": record.prompt_revision,
        "toolset_revision": record.toolset_revision,
        "provider_wire_hash": record.provider_wire_hash,
        "usage": usage,
    }


def canonical_model_request_json(request: ModelRequest) -> str:
    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a ModelRequest")
    return canonical_json_text(
        {
            "request_id": request.request_id,
            "messages": tuple(model_message_payload(message) for message in request.messages),
            "tools": tuple(tool_definition_payload(tool) for tool in request.tools),
            "exposed_tool_names": request.exposed_tool_names,
            "tool_choice": tool_choice_payload(request.tool_choice),
            "settings": model_settings_payload(request.settings),
            "prompt_revision": request.prompt_revision,
            "toolset_revision": request.toolset_revision,
        }
    )


def stable_context_json(context: StableModelContext) -> str:
    if not isinstance(context, StableModelContext):
        raise TypeError("context must be a StableModelContext")
    return canonical_json_text(
        {
            **dict(_stable_context_payload(context)),
            "context_revision": context.context_revision,
            "parent_context_revision": context.parent_context_revision,
            "revision_reason": context.revision_reason,
        }
    )


def tool_definition_payload(
    definition: ToolDefinition,
) -> Mapping[str, JsonValue]:
    if not isinstance(definition, ToolDefinition):
        raise TypeError("definition must be a ToolDefinition")
    return {
        "name": definition.name,
        "description": definition.description,
        "input_schema": definition.input_schema,
    }


def tool_choice_payload(choice: ToolChoice) -> Mapping[str, JsonValue]:
    return {
        "mode": choice.mode.value,
        "name": choice.name,
    }


def model_settings_payload(settings: ModelSettings) -> Mapping[str, JsonValue]:
    return {
        "model": settings.model,
        "max_output_tokens": settings.max_output_tokens,
        "temperature": float(settings.temperature),
        "top_p": None if settings.top_p is None else float(settings.top_p),
        "parallel_tool_calls": settings.parallel_tool_calls,
        "seed": settings.seed,
        "provider_options": settings.provider_options,
    }


def freeze_json_mapping(
    value: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    frozen = json_schema_output(None, value)
    if not isinstance(frozen, Mapping):
        raise TypeError("value must be a JSON object")
    return frozen


def canonical_hash(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_text(value).encode("utf-8")).hexdigest()


def canonical_transcript_revision(
    transcript: Sequence[ModelMessage],
) -> str:
    """Return the deterministic revision owned by a canonical transcript."""

    messages = _snapshot_messages(
        transcript,
        field_name="transcript",
    )
    return _revision(
        "transcript",
        tuple(model_message_payload(message) for message in messages),
    )


def project_transcript_compaction(
    transcript: Sequence[ModelMessage],
    *,
    parent_context_revision: str,
    tail_start: int,
    max_summary_chars: int,
) -> tuple[ModelMessage, ...]:
    """Project a smaller canonical transcript with a verifiable compaction event."""

    messages = _snapshot_messages(
        transcript,
        field_name="transcript",
    )
    _require_non_empty_string(
        parent_context_revision,
        field_name="parent_context_revision",
    )
    if (
        not isinstance(tail_start, int)
        or isinstance(tail_start, bool)
        or tail_start < 0
        or tail_start > len(messages)
    ):
        raise ValueError("tail_start must index the transcript")
    if (
        not isinstance(max_summary_chars, int)
        or isinstance(max_summary_chars, bool)
        or max_summary_chars <= 0
    ):
        raise ValueError("max_summary_chars must be a positive integer")

    actual_tail_start = _extend_tail_for_tool_pair(messages, tail_start)
    covered = messages[:actual_tail_start]
    tail = messages[actual_tail_start:]
    if not covered:
        return messages
    summary_limit = min(max_summary_chars, 12_000)
    summary = _deterministic_transcript_summary(
        covered,
        max_chars=summary_limit,
    )
    projection: Mapping[str, JsonValue] = {
        "covered_count": len(covered),
        "retained_tail_count": len(tail),
        "summary_max_chars": summary_limit,
        "source_digest": canonical_hash(
            tuple(model_message_payload(message) for message in messages)
        ),
        "retained_tail_digest": canonical_hash(
            tuple(model_message_payload(message) for message in tail)
        ),
    }
    event = context_event_message(
        "context_compaction",
        {
            "summary": summary,
            "parent_context_revision": parent_context_revision,
            "projection": projection,
        },
    )
    candidate = (event, *tail)
    if _model_messages_size(candidate) >= _model_messages_size(messages):
        return messages
    return candidate


def is_verified_transcript_compaction_rewrite(
    existing_turn: Sequence[ModelMessage],
    candidate_turn: Sequence[ModelMessage],
    *,
    message_compaction_min_count: int,
    max_message_tail_count: int,
    reactive_compact_tail_count: int,
    max_summary_chars: int,
) -> bool:
    """Verify a rewrite against projections allowed by trusted runtime policy."""

    existing = _snapshot_messages(
        existing_turn,
        field_name="existing_turn",
    )
    candidate = _snapshot_messages(
        candidate_turn,
        field_name="candidate_turn",
    )
    if (
        len(existing) < 2
        or len(candidate) < 2
        or existing[0].role != "user"
        or candidate[0] != existing[0]
        or candidate[1].role != "context"
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                message_compaction_min_count,
                max_message_tail_count,
                reactive_compact_tail_count,
                max_summary_chars,
            )
        )
        or message_compaction_min_count <= 0
        or max_message_tail_count < 0
        or reactive_compact_tail_count <= 0
        or max_summary_chars <= 0
    ):
        return False

    existing_body = existing[1:]
    parent_revision = canonical_transcript_revision(existing)
    allowed: list[tuple[ModelMessage, ...]] = []
    if len(existing) >= message_compaction_min_count:
        proactive = project_transcript_compaction(
            existing_body,
            parent_context_revision=parent_revision,
            tail_start=max(
                0,
                len(existing_body) - max_message_tail_count,
            ),
            max_summary_chars=max_summary_chars,
        )
        if proactive != existing_body:
            allowed.append(proactive)

    reactive_tail_start = max(
        0,
        len(existing_body) - reactive_compact_tail_count,
    )
    if reactive_tail_start == 0:
        reactive_tail_start = 1
    reactive = project_transcript_compaction(
        existing_body,
        parent_context_revision=parent_revision,
        tail_start=reactive_tail_start,
        max_summary_chars=max_summary_chars,
    )
    if reactive == existing_body:
        reactive = project_transcript_compaction(
            existing_body,
            parent_context_revision=parent_revision,
            tail_start=len(existing_body),
            max_summary_chars=max_summary_chars,
        )
    if reactive != existing_body:
        allowed.append(reactive)

    for projection in allowed:
        expected_prefix = (existing[0], *projection)
        if (
            len(candidate) >= len(expected_prefix)
            and candidate[: len(expected_prefix)] == expected_prefix
        ):
            return True
    return False


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
    lines = [
        f"{message.role}: {canonical_json_text(model_message_payload(message))}"
        for message in messages
    ]
    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    return summary[:max_chars].rstrip() + " [truncated]"


def _model_messages_size(messages: Sequence[ModelMessage]) -> int:
    return sum(
        len(
            canonical_json_text(
                model_message_payload(message)
            ).encode("utf-8")
        )
        for message in messages
    )


def _tool_contract_payload(tool: Tool) -> Mapping[str, JsonValue]:
    return {
        "definition": tool_definition_payload(tool.definition),
        "output_schema": tool.output_schema,
        "static_effects": tuple(sorted(effect.value for effect in tool.static_effects)),
        "execution_revision": tool.execution_revision,
        "idempotent": tool.idempotent,
        "concurrency_safe": tool.concurrency_safe,
        "cancellation_mode": tool.cancellation_mode.value,
        "interrupt_behavior": tool.interrupt_behavior.value,
        "timeout_seconds": float(tool.timeout_seconds),
        "max_model_output_bytes": tool.max_model_output_bytes,
        "approval_profile": (
            tool.approval_profile.value
            if tool.approval_profile is not None
            else None
        ),
    }


def _tool_manifest_entry(tool: Tool) -> ToolManifestEntry:
    return ToolManifestEntry(
        name=tool.definition.name,
        description_hash=canonical_hash(tool.definition.description),
        input_schema_hash=canonical_hash(tool.definition.input_schema),
        static_effects_hash=canonical_hash(
            tuple(sorted(effect.value for effect in tool.static_effects))
        ),
        execution_contract_hash=canonical_hash(_tool_contract_payload(tool)),
    )


def _stable_context_payload(
    context: StableModelContext,
) -> Mapping[str, JsonValue]:
    return {
        "serializer_revision": STABLE_CONTEXT_REVISION,
        "instructions": context.instructions,
        "frozen_run_context": tuple(
            {"name": block.name, "content": block.content} for block in context.frozen_run_context
        ),
        "initial_user_task": context.initial_user_task,
        "initial_memory": context.initial_memory,
    }


def _revision(prefix: str, value: JsonValue) -> str:
    return f"{prefix}_{canonical_hash(value)}"


def _snapshot_messages(
    messages: Sequence[ModelMessage],
    *,
    field_name: str,
) -> tuple[ModelMessage, ...]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError(f"{field_name} must be a sequence of ModelMessage values")
    return tuple(snapshot_model_message(message) for message in messages)


def _ordered_tools(tools: Sequence[Tool]) -> tuple[Tool, ...]:
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise TypeError("tools must be a sequence of Tool values")
    ordered = tuple(tools)
    if any(not isinstance(tool, Tool) for tool in ordered):
        raise TypeError("tools must contain Tool values")
    names = tuple(tool.definition.name for tool in ordered)
    if len(set(names)) != len(names):
        raise ValueError("tools must have unique names")
    return ordered


def _ordered_names(
    names: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    values = _ordered_strings(names, field_name=field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique names")
    return values


def _ordered_strings(
    values: Sequence[str],
    *,
    field_name: str,
    require_non_empty_sequence: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    result = tuple(values)
    if require_non_empty_sequence and not result:
        raise ValueError(f"{field_name} must not be empty")
    for value in result:
        _require_non_empty_string(value, field_name=field_name)
    return result
def _validate_real_range(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    below_minimum = numeric < minimum if minimum_inclusive else numeric <= minimum
    if below_minimum or numeric > maximum:
        minimum_operator = ">=" if minimum_inclusive else ">"
        raise ValueError(f"{field_name} must be {minimum_operator} {minimum} and <= {maximum}")


__all__ = [
    "CANONICAL_REQUEST_REVISION",
    "COMPACTION_REVISION",
    "STABLE_CONTEXT_REVISION",
    "ContextBlock",
    "ModelCallRecord",
    "ModelRequest",
    "ModelSettings",
    "StableModelContext",
    "ToolChoice",
    "ToolChoiceMode",
    "bind_model_call_record",
    "build_model_request",
    "build_stable_context",
    "build_tool_manifest",
    "canonical_hash",
    "canonical_model_request_json",
    "canonical_transcript_revision",
    "freeze_json_mapping",
    "model_settings_payload",
    "model_call_record_payload",
    "split_turn_context",
    "stable_context_json",
    "tool_choice_payload",
    "tool_definition_payload",
    "toolset_revision_for_tools",
]

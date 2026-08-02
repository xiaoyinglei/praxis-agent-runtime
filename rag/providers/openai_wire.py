from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from rag.agent.core.messages import (
    ModelMessage,
    StopReason,
    ToolCall,
    ToolUseResult,
    canonical_json_text,
    snapshot_model_message,
)
from rag.agent.core.model_request import (
    ModelRequest,
    ToolChoice,
    ToolChoiceMode,
    canonical_hash,
    freeze_json_mapping,
)
from rag.agent.tools.tool import JsonValue, json_schema_output
from rag.schema.llm import LLMUsage, normalize_llm_usage

OPENAI_WIRE_REVISION = "openai-compatible-chat-v2"
_RESERVED_PAYLOAD_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "seed",
    }
)
_CACHE_CONTROLLED_FIELDS = frozenset(
    {
        "prompt_cache_key",
        "prompt_cache_retention",
    }
)
_SERIALIZER_OWNED_FIELDS = _RESERVED_PAYLOAD_FIELDS | _CACHE_CONTROLLED_FIELDS


@dataclass(frozen=True, slots=True)
class OpenAIWireRequest:
    payload: Mapping[str, JsonValue]
    serialized_json: str
    provider_wire_hash: str
    serializer_revision: str = OPENAI_WIRE_REVISION


def serialize_openai_request(
    request: ModelRequest,
    *,
    cache_key: str | None = None,
    cache_parameters: Mapping[str, JsonValue] | None = None,
    supported_cache_parameters: Collection[str] = (),
) -> OpenAIWireRequest:
    """Serialize one already-selected canonical request to chat wire data."""

    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a ModelRequest")
    supported = _supported_parameter_names(supported_cache_parameters)
    payload: dict[str, JsonValue] = {
        "model": request.settings.model,
        "messages": _message_payloads(request.messages),
        "max_completion_tokens": request.settings.max_output_tokens,
        "temperature": float(request.settings.temperature),
        "parallel_tool_calls": request.settings.parallel_tool_calls,
    }
    if request.settings.top_p is not None:
        payload["top_p"] = float(request.settings.top_p)
    if request.settings.seed is not None:
        payload["seed"] = request.settings.seed
    if request.tools:
        payload["tools"] = tuple(
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                },
            }
            for definition in request.tools
        )
        payload["tool_choice"] = _tool_choice_payload(request.tool_choice)
    elif request.tool_choice.mode is not ToolChoiceMode.NONE:
        payload["tool_choice"] = _tool_choice_payload(request.tool_choice)

    for key, value in request.settings.provider_options.items():
        if key in _SERIALIZER_OWNED_FIELDS:
            raise ValueError(f"provider option cannot override serializer-owned field: {key}")
        payload[key] = value

    if "prompt_cache_key" in supported:
        effective_cache_key = cache_key or request.prompt_revision
        if not isinstance(effective_cache_key, str) or not effective_cache_key:
            raise ValueError("cache_key must be a non-empty string")
        payload["prompt_cache_key"] = effective_cache_key

    if cache_parameters is not None:
        if not isinstance(cache_parameters, Mapping):
            raise TypeError("cache_parameters must be a mapping")
        for key, value in cache_parameters.items():
            if not isinstance(key, str) or not key:
                raise TypeError("cache parameter names must be non-empty strings")
            if key == "prompt_cache_key" or key not in supported:
                continue
            if key in _RESERVED_PAYLOAD_FIELDS or key in request.settings.provider_options:
                raise ValueError(f"cache parameter cannot override request field: {key}")
            payload[key] = value

    frozen_payload = freeze_json_mapping(payload)
    serialized = canonical_json_text(frozen_payload)
    wire_hash = "wire_" + canonical_hash(
        {
            "serializer_revision": OPENAI_WIRE_REVISION,
            "payload": frozen_payload,
        }
    )
    return OpenAIWireRequest(
        payload=frozen_payload,
        serialized_json=serialized,
        provider_wire_hash=wire_hash,
    )


def parse_openai_response(response: object) -> ToolUseResult:
    """Parse chat response data into the shared provider-neutral turn value."""

    choices = _field(response, "choices", default=None)
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        raise ValueError("response choices must be a non-empty sequence")
    if not choices:
        raise ValueError("response choices must not be empty")
    choice = choices[0]
    message = _field(choice, "message", default=None)
    if message is None:
        raise ValueError("response choice is missing message")
    raw_stop = _field(choice, "finish_reason", default="unknown")
    raw_stop_text = str(raw_stop or "unknown")
    if raw_stop_text in {"tool_calls", "tool_use"}:
        stop_reason = StopReason.TOOL_USE
    elif raw_stop_text == "length":
        stop_reason = StopReason.MAX_TOKENS
    else:
        stop_reason = StopReason.END_TURN

    raw_calls = _field(message, "tool_calls", default=())
    if raw_calls is None:
        raw_calls = ()
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        raise ValueError("message tool_calls must be a sequence")
    calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls, start=1):
        function = _field(raw_call, "function", default=None)
        if function is None:
            raise ValueError("tool call is missing function")
        name = _field(function, "name", default="")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call function name must be non-empty")
        raw_arguments = _field(function, "arguments", default={})
        arguments = _parse_arguments(raw_arguments)
        call_id = _field(raw_call, "id", default=None)
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"
        calls.append(ToolCall(id=call_id, name=name, input=arguments))

    content = _field(message, "content", default="")
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)
    reasoning_content = _field(message, "reasoning_content", default=None)
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        reasoning_content = str(reasoning_content)
    return ToolUseResult(
        tool_calls=calls,
        text=content,
        reasoning_content=reasoning_content,
        stop_reason=stop_reason,
        raw_stop_reason=raw_stop_text,
    )


def parse_openai_usage(response: object) -> LLMUsage | None:
    """Normalize reported OpenAI-compatible usage without guessing cache data."""

    raw_usage = _field(response, "usage", default=None)
    if raw_usage is None:
        return None
    usage_payload = _plain_usage_mapping(raw_usage)
    prompt_details = _field(raw_usage, "prompt_tokens_details", default=None)
    if prompt_details is None:
        prompt_details = _field(raw_usage, "input_tokens_details", default=None)
    input_tokens = _first_token_count(
        raw_usage,
        ("prompt_tokens", "input_tokens"),
    )
    output_tokens = _first_token_count(
        raw_usage,
        ("completion_tokens", "output_tokens"),
    )
    cache_read = _first_token_count(
        prompt_details,
        ("cached_tokens", "cache_read_tokens"),
    )
    if cache_read is None:
        cache_read = _first_token_count(
            raw_usage,
            ("cache_read_input_tokens",),
        )
    cache_write = _first_token_count(
        prompt_details,
        ("cache_write_tokens", "cache_creation_tokens"),
    )
    if cache_write is None:
        cache_write = _first_token_count(
            raw_usage,
            ("cache_write_input_tokens", "cache_creation_input_tokens"),
        )
    return normalize_llm_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        input_tokens_include_cache=True,
        usage_source="provider",
        raw_provider_usage=usage_payload,
    )


def _message_payloads(
    messages: Sequence[ModelMessage],
) -> tuple[Mapping[str, JsonValue], ...]:
    payloads: list[Mapping[str, JsonValue]] = []
    leading_system_content: list[str] = []
    conversation_started = False
    for raw_message in messages:
        message = snapshot_model_message(raw_message)
        if not conversation_started and message.role in {"system", "context"}:
            leading_system_content.append(message.content)
            continue
        if not conversation_started:
            if leading_system_content:
                payloads.append(
                    {
                        "role": "system",
                        "content": "\n\n".join(leading_system_content),
                    }
                )
            conversation_started = True
        if message.role == "system":
            raise ValueError("non-leading system message is not supported")
        if message.role == "context":
            payloads.append({"role": "user", "content": message.content})
            continue
        payloads.append(_message_payload(message))
    if not conversation_started and leading_system_content:
        payloads.append(
            {
                "role": "system",
                "content": "\n\n".join(leading_system_content),
            }
        )
    return tuple(payloads)


def _message_payload(message: ModelMessage) -> Mapping[str, JsonValue]:
    message = snapshot_model_message(message)
    if message.role in {"system", "user"}:
        return {"role": message.role, "content": message.content}
    if message.role == "assistant":
        payload: dict[str, JsonValue] = {
            "role": "assistant",
            "content": message.content or None,
        }
        if message.reasoning_content is not None:
            payload["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            payload["tool_calls"] = tuple(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": canonical_json_text(json_schema_output(None, call.input)),
                    },
                }
                for call in message.tool_calls
            )
        return payload
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }
    raise ValueError(f"unsupported model message role: {message.role}")


def _tool_choice_payload(choice: ToolChoice) -> JsonValue:
    if choice.mode is ToolChoiceMode.NAMED:
        assert choice.name is not None
        return {
            "type": "function",
            "function": {"name": choice.name},
        }
    return choice.mode.value


def _parse_arguments(raw: object) -> dict[str, object]:
    if isinstance(raw, Mapping):
        parsed: object = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"_raw": raw[:20_000]}
    else:
        parsed = {"_raw": str(raw)[:20_000]}
    if not isinstance(parsed, Mapping):
        parsed = {"_raw": parsed}
    frozen = json_schema_output(None, cast(JsonValue, parsed))
    if not isinstance(frozen, Mapping):
        raise TypeError("parsed tool arguments must be an object")
    return {key: _thaw_json(value) for key, value in frozen.items()}


def _first_token_count(
    raw: object,
    names: Sequence[str],
) -> int | None:
    if raw is None:
        return None
    missing = object()
    for name in names:
        value = _field(raw, name, default=missing)
        if value is missing or value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"usage field {name} must be an integer or None")
        if value < 0:
            raise ValueError(f"usage field {name} must be non-negative")
        return value
    return None


def _plain_usage_mapping(raw: object) -> Mapping[str, object]:
    plain = _plain_usage_value(raw, depth=0)
    if not isinstance(plain, Mapping):
        raise TypeError("provider usage must serialize as an object")
    return plain


def _plain_usage_value(raw: object, *, depth: int) -> object:
    if depth > 20:
        raise ValueError("provider usage exceeds maximum nesting depth")
    if raw is None or isinstance(raw, (str, int, float, bool)):
        return raw
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        return _plain_usage_value(model_dump(mode="json"), depth=depth + 1)
    if isinstance(raw, Mapping):
        result: dict[str, object] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise TypeError("provider usage keys must be strings")
            result[key] = _plain_usage_value(value, depth=depth + 1)
        return result
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [_plain_usage_value(value, depth=depth + 1) for value in raw]
    try:
        attributes = vars(raw)
    except TypeError:
        attributes = None
    if isinstance(attributes, Mapping):
        return _plain_usage_value(
            {key: value for key, value in attributes.items() if isinstance(key, str) and not key.startswith("_")},
            depth=depth + 1,
        )
    raise TypeError("provider usage must contain only JSON-compatible values")


def _thaw_json(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _supported_parameter_names(values: Collection[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise TypeError("supported_cache_parameters must be a collection of names")
    names: set[str] = set()
    for name in values:
        if not isinstance(name, str) or not name:
            raise TypeError("supported cache parameter names must be non-empty strings")
        names.add(name)
    return frozenset(names)


def _field(raw: object, name: str, *, default: object) -> object:
    if isinstance(raw, Mapping):
        return raw.get(name, default)
    return getattr(raw, name, default)


__all__ = [
    "OPENAI_WIRE_REVISION",
    "OpenAIWireRequest",
    "parse_openai_response",
    "parse_openai_usage",
    "serialize_openai_request",
]

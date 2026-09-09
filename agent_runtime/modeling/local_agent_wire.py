from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.core.messages import (
    StopReason,
    ToolCall,
    ToolUseResult,
    canonical_json_text,
    model_message_payload,
)
from agent_runtime.core.model_request import (
    ModelRequest,
    canonical_hash,
    freeze_json_mapping,
    model_settings_payload,
    tool_choice_payload,
    tool_definition_payload,
)
from agent_runtime.modeling.contracts import LLMUsage, normalize_llm_usage
from agent_runtime.tools.tool import JsonValue

LOCAL_AGENT_WIRE_REVISION = "local-agent-flat-json-v1"
_SUPPORTED_PROVIDERS = frozenset({"mlx", "ollama"})
_RESPONSE_EXAMPLE = '{"text":"","tool_calls":[{"id":"call_1","name":"read_file","arguments":{"path":"README.md"}}]}'
_RESERVED_GENERATION_FIELDS = frozenset(
    {
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "parallel_tool_calls",
    }
)


class LocalAgentWireMode(StrEnum):
    NATIVE = "native"
    FLAT_JSON = "flat_json"


@dataclass(frozen=True, slots=True)
class LocalAgentWireRequest:
    provider: str
    prompt: str
    generation_options: Mapping[str, JsonValue]
    serialized_json: str
    provider_wire_hash: str
    serializer_revision: str = LOCAL_AGENT_WIRE_REVISION


class _LocalToolCallEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=500)
    name: str = Field(min_length=1, max_length=500)
    arguments: dict[str, Any] = Field(default_factory=dict)


class _LocalResponseEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(default="", max_length=100_000)
    tool_calls: list[_LocalToolCallEnvelope] = Field(
        default_factory=list,
        max_length=32,
    )


def resolve_local_agent_wire(
    provider: str,
    *,
    supports_native_tools: bool,
) -> LocalAgentWireMode:
    """Resolve only declared provider capability, never task text."""

    if not isinstance(provider, str) or provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported local model provider: {provider}")
    if type(supports_native_tools) is not bool:
        raise TypeError("supports_native_tools must be a bool")
    return LocalAgentWireMode.NATIVE if supports_native_tools else LocalAgentWireMode.FLAT_JSON


def render_local_agent_request(
    request: ModelRequest,
    *,
    provider: str,
) -> LocalAgentWireRequest:
    """Render the canonical request as one deterministic flat prompt."""

    resolve_local_agent_wire(provider, supports_native_tools=False)
    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a ModelRequest")
    tools_json = canonical_json_text(tuple(tool_definition_payload(tool) for tool in request.tools))
    messages_json = canonical_json_text(tuple(model_message_payload(message) for message in request.messages))
    tool_choice_json = canonical_json_text(tool_choice_payload(request.tool_choice))
    response_schema_json = canonical_json_text(
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ("id", "name", "arguments"),
                        "additionalProperties": False,
                    },
                },
            },
            "required": ("text", "tool_calls"),
            "additionalProperties": False,
        }
    )
    prompt = "\n\n".join(
        (
            f"[Canonical Agent Request: {LOCAL_AGENT_WIRE_REVISION}]",
            f"[Selected Tools]\n{tools_json}",
            f"[Tool Choice]\n{tool_choice_json}",
            f"[Canonical Messages]\n{messages_json}",
            (
                "[Response Contract]\n"
                "Return exactly one JSON object. Do not use markdown fences or "
                "add prose outside the object. The object must satisfy this schema:\n"
                f"{response_schema_json}\n"
                "Example:\n"
                f"{_RESPONSE_EXAMPLE}"
            ),
        )
    )

    settings = dict(model_settings_payload(request.settings))

    if request.settings.max_output_tokens is None:
        settings.pop("max_output_tokens", None)
    generation_options: dict[str, JsonValue] = {
    "model": request.settings.model,
    "temperature": float(request.settings.temperature),
    "parallel_tool_calls": request.settings.parallel_tool_calls,}

    if request.settings.max_output_tokens is not None:
        generation_options["max_tokens"] = (
            request.settings.max_output_tokens
        )
    if request.settings.top_p is not None:
        generation_options["top_p"] = float(request.settings.top_p)
    if request.settings.seed is not None:
        generation_options["seed"] = request.settings.seed
    for key, value in request.settings.provider_options.items():
        if key in _RESERVED_GENERATION_FIELDS:
            raise ValueError(f"provider option cannot override generation field: {key}")
        generation_options[key] = value
    frozen_options = freeze_json_mapping(generation_options)
    wire_payload: Mapping[str, JsonValue] = {
        "provider": provider,
        "serializer_revision": LOCAL_AGENT_WIRE_REVISION,
        "prompt": prompt,
        "generation_options": frozen_options,
        "canonical_settings": settings,
    }
    serialized = canonical_json_text(wire_payload)
    wire_hash = "wire_" + canonical_hash(wire_payload)
    return LocalAgentWireRequest(
        provider=provider,
        prompt=prompt,
        generation_options=frozen_options,
        serialized_json=serialized,
        provider_wire_hash=wire_hash,
    )


def parse_local_agent_response(raw: str | Mapping[str, object]) -> ToolUseResult:
    """Validate one strict response envelope into the shared turn value."""

    if isinstance(raw, str):
        envelope = _LocalResponseEnvelope.model_validate_json(raw)
    elif isinstance(raw, Mapping):
        envelope = _LocalResponseEnvelope.model_validate(raw)
    else:
        raise TypeError("local agent response must be JSON text or an object")
    calls = [
        ToolCall(
            id=call.id,
            name=call.name,
            input=dict(call.arguments),
        )
        for call in envelope.tool_calls
    ]
    has_calls = bool(calls)
    return ToolUseResult(
        tool_calls=calls,
        text=envelope.text,
        stop_reason=(StopReason.TOOL_USE if has_calls else StopReason.END_TURN),
        raw_stop_reason=("tool_calls" if has_calls else "end_turn"),
    )


def estimate_local_agent_usage(
    *,
    input_tokens: int,
    output_tokens: int,
) -> LLMUsage:
    """Record tokenizer totals without claiming unobserved cache behavior."""

    return normalize_llm_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_include_cache=True,
        usage_source="tokenizer_estimate",
    )


__all__ = [
    "LOCAL_AGENT_WIRE_REVISION",
    "LocalAgentWireMode",
    "LocalAgentWireRequest",
    "estimate_local_agent_usage",
    "parse_local_agent_response",
    "render_local_agent_request",
    "resolve_local_agent_wire",
]

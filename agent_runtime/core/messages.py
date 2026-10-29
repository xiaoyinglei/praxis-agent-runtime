"""Provider-neutral typed messages for Harness model adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_runtime.modeling.events import (
    ModelOutput,
    ModelOutputKind,
    output_from_payload,
    output_to_payload,
    validate_model_outputs,
)
from agent_runtime.tools.tool import JsonValue, _thaw_json, json_schema_output

if TYPE_CHECKING:
    from agent_runtime.tools.tool import ToolResult

# ── Stop reason ──


class StopReason(StrEnum):
    """Normalized provider stop reason used by the Harness model boundary."""

    TOOL_USE = "tool_use"
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    INCOMPLETE = "incomplete"


# ── Tool call ──


@dataclass(frozen=True)
class ModelToolCall:
    """A single tool invocation requested by the model.

    ``id`` is model-provided or fallback-generated (``tc_<hex>``).
    ``arguments`` is an immutable parsed JSON object.
    """

    id: str
    name: str
    arguments: Mapping[str, Any]
    argument_error: str | None = None

    @classmethod
    def create(cls, name: str, arguments: Mapping[str, Any]) -> ModelToolCall:
        return cls(id=f"tc_{uuid4().hex[:12]}", name=name, arguments=arguments)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or not isinstance(self.name, str) or not self.name:
            raise ValueError("model tool-call id and name must be non-empty strings")
        frozen = json_schema_output(None, self.arguments)
        if not isinstance(frozen, Mapping):
            raise TypeError("model tool-call input must be an object")
        object.__setattr__(self, "arguments", frozen)

    def __deepcopy__(self, memo: dict[int, object]) -> ModelToolCall:
        return self


def model_tool_call_payload(call: ModelToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments": _thaw_json(call.arguments),
        "argument_error": call.argument_error,
    }


def decode_tool_arguments(raw: object) -> tuple[dict[str, Any], str | None]:
    """Decode an object without disguising syntax errors as schema errors."""
    if isinstance(raw, Mapping):
        parsed = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError as error:
            return {"_raw": raw}, f"Invalid tool arguments JSON: {error}"
        if not isinstance(parsed, Mapping):
            return {"_raw": raw}, "Invalid tool arguments JSON: expected an object"
    else:
        return {"_raw": str(raw)}, "Invalid tool arguments JSON: expected an object or JSON string"
    frozen = json_schema_output(None, dict(parsed))
    return cast(dict[str, Any], _thaw_json(frozen)), None


# ── Model message ──


def outputs_from_parts(
    *,
    text: str = "",
    tool_calls: tuple[ModelToolCall, ...] = (),
    reasoning_content: str | None = None,
    plan_content: str | None = None,
) -> tuple[ModelOutput, ...]:
    """Build ordered blocks at a flat protocol or synthetic-model boundary."""
    outputs = []
    parts: tuple[tuple[ModelOutputKind, str | None], ...] = (
        ("reasoning", reasoning_content),
        ("plan", plan_content),
        ("text", text),
    )
    for kind, content in parts:
        if content is not None and not isinstance(content, str):
            raise TypeError(f"model {kind} content must be a string or None")
        if content:
            outputs.append(ModelOutput(output_id=kind, kind=kind, content=content))
    for call in tool_calls:
        outputs.append(
            ModelOutput(
                output_id=f"tool:{call.id}",
                kind="tool_call",
                tool_call_id=call.id,
                tool_name=call.name,
                arguments=call.arguments,
                argument_error=call.argument_error,
                content=canonical_json_text(call.arguments),
            )
        )
    return validate_model_outputs(tuple(outputs))


def outputs_tool_calls(outputs: tuple[ModelOutput, ...]) -> tuple[ModelToolCall, ...]:
    return tuple(
        ModelToolCall(
            id=cast(str, output.tool_call_id),
            name=cast(str, output.tool_name),
            arguments=cast(dict[str, Any], output.arguments),
            argument_error=output.argument_error,
        )
        for output in outputs
        if output.kind == "tool_call" and output.arguments is not None
    )


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """One immutable history message, shared by Runtime and model adapters.

    Assistant bodies are ordered output blocks. Other roles carry text. Flat
    views are computed, never stored alongside the authoritative body.
    """

    role: Literal["system", "user", "assistant", "tool", "context"]
    body: str | tuple[ModelOutput, ...]
    tool_call_id: str | None = None
    context_category: Literal["continuation_summary"] | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool", "context"}:
            raise ValueError(f"unsupported model message role: {self.role}")
        if self.role == "assistant":
            body = outputs_from_parts(text=self.body) if isinstance(self.body, str) else self.body
            object.__setattr__(self, "body", validate_model_outputs(body))
        elif not isinstance(self.body, str):
            raise ValueError("only assistant messages may contain model output blocks")
        if self.role == "tool":
            if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        if self.context_category is not None and (
            self.role != "context" or self.context_category != "continuation_summary"
        ):
            raise ValueError("continuation_summary provenance requires a context message")

    @classmethod
    def from_parts(
        cls,
        *,
        role: Literal["system", "user", "assistant", "tool", "context"],
        content: str = "",
        tool_calls: tuple[ModelToolCall, ...] = (),
        reasoning_content: str | None = None,
        tool_call_id: str | None = None,
        context_category: Literal["continuation_summary"] | None = None,
    ) -> ModelMessage:
        if role != "assistant" and (tool_calls or reasoning_content is not None):
            raise ValueError("only assistant messages may contain tool calls or reasoning_content")
        body = (
            outputs_from_parts(text=content, tool_calls=tool_calls, reasoning_content=reasoning_content)
            if role == "assistant"
            else content
        )
        return cls(role=role, body=body, tool_call_id=tool_call_id, context_category=context_category)

    @property
    def outputs(self) -> tuple[ModelOutput, ...]:
        return self.body if isinstance(self.body, tuple) else ()

    @property
    def content(self) -> str:
        if isinstance(self.body, str):
            return self.body
        return "".join(output.content for output in self.body if output.kind in {"text", "refusal"})

    @property
    def reasoning_content(self) -> str | None:
        return "".join(output.content for output in self.outputs if output.kind == "reasoning") or None

    @property
    def tool_calls(self) -> tuple[ModelToolCall, ...]:
        return outputs_tool_calls(self.outputs)


class ToolUseResult(BaseModel):
    """Flat wire parser result; converted to output blocks at the gateway."""

    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    text: str = ""
    reasoning_content: str | None = None
    stop_reason: StopReason
    raw_stop_reason: str


def restore_model_message(payload: Mapping[str, Any]) -> ModelMessage:
    """Read the canonical message body."""
    raw_body = payload["body"]
    body = raw_body if isinstance(raw_body, str) else tuple(output_from_payload(p) for p in raw_body)
    return ModelMessage(
        role=payload["role"],
        body=body,
        tool_call_id=payload.get("tool_call_id"),
        context_category=payload.get("context_category"),
    )


def context_event_message(
    event_type: str,
    payload: Mapping[str, JsonValue],
) -> ModelMessage:
    """Encode one append-only provider-neutral context event."""

    if not isinstance(event_type, str) or not event_type:
        raise ValueError("event_type must be a non-empty string")
    frozen_payload = json_schema_output(None, payload)
    if not isinstance(frozen_payload, Mapping):
        raise TypeError("context event payload must be an object")
    return ModelMessage(
        role="context",
        body=canonical_json_text(
            {
                "event_type": event_type,
                "payload": frozen_payload,
            }
        ),
    )


def tool_result_message(result: ToolResult) -> ModelMessage:
    """Fix model-visible ToolResult content at transcript insertion time."""

    from agent_runtime.tools.tool import ToolResult

    if not isinstance(result, ToolResult):
        raise TypeError("result must be a ToolResult")
    payload = tool_result_payload(result)
    return ModelMessage(
        role="tool",
        body=canonical_json_text(payload),
        tool_call_id=result.tool_call_id,
    )


def tool_result_payload(result: ToolResult) -> Mapping[str, JsonValue]:
    """Build the same model envelope for budget admission and transcript insertion."""
    structured = result.structured_content
    if (
        result.tool_name == "web_fetch"
        and not result.is_error
        and isinstance(structured, Mapping)
        and isinstance(structured.get("source_id"), str)
    ):
        runtime_fields = {
            "network_bytes",
            "cache_hit",
            "connection_mode",
            "failure_stage",
            "extraction_method",
            "error_code",
            "error_message",
            "raw_body_base64",
        }
        structured = {
            key: value
            for key, value in structured.items()
            if key not in runtime_fields and (value is not None or key in {"next_line", "next_link", "next_section"})
        }
        return {"structured_content": structured, "is_error": False, "truncated": result.truncated}
    return {
        "content": tuple(
            {
                "type": block.type,
                "data": block.data,
            }
            for block in result.content
        ),
        "structured_content": structured,
        "is_error": result.is_error,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "retryable": result.retryable,
        "truncated": result.truncated,
    }


def model_message_payload(message: ModelMessage) -> Mapping[str, JsonValue]:
    """Serialize the one authoritative body, including private replay metadata."""
    if not isinstance(message, ModelMessage):
        raise TypeError("message must be a ModelMessage")
    return {
        "role": message.role,
        "body": message.body
        if isinstance(message.body, str)
        else cast(JsonValue, tuple(output_to_payload(output) for output in message.body)),
        "tool_call_id": message.tool_call_id,
        "context_category": message.context_category,
    }


def canonical_json_text(value: JsonValue) -> str:
    """Serialize JSON with normalized mapping keys and ordered arrays."""

    frozen = json_schema_output(None, value)
    return json.dumps(
        _thaw_json(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "ModelMessage",
    "StopReason",
    "ModelToolCall",
    "ToolUseResult",
    "canonical_json_text",
    "context_event_message",
    "model_message_payload",
    "tool_result_message",
]

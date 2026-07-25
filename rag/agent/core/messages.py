"""Provider-neutral typed messages for the agent loop.

These types are the internal protocol between AgentLoop, ModelTurnProvider,
and checkpoint storage.  Provider adapters (OpenAI, Anthropic, fallback)
translate to/from these types at the boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from rag.agent.tools.tool import JsonValue, _thaw_json, json_schema_output

if TYPE_CHECKING:
    from rag.agent.tools.tool import ToolResult

# ── Stop reason ──


class StopReason(StrEnum):
    """Normalized stop reason.  AgentLoop control flow depends *only* on this."""

    TOOL_USE = "tool_use"
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"


# ── Tool call ──


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model.

    ``id`` is model-provided or fallback-generated (``tc_<hex>``).
    ``input`` is already parsed into a dict — never a JSON string.
    """

    id: str
    name: str
    input: dict[str, Any]

    @classmethod
    def create(cls, name: str, input: dict[str, Any]) -> ToolCall:
        return cls(id=f"tc_{uuid4().hex[:12]}", name=name, input=input)

    def __deepcopy__(self, memo: dict[int, object]) -> ToolCall:
        frozen_input = json_schema_output(None, self.input)
        if not isinstance(frozen_input, Mapping):
            raise TypeError("model tool-call input must be an object")
        copied = ToolCall(
            id=self.id,
            name=self.name,
            input=cast(dict[str, Any], frozen_input),
        )
        memo[id(self)] = copied
        return copied


# ── Model message ──


@dataclass(frozen=True)
class ModelMessage:
    """A single message in the agent conversation.

    Roles:
    - ``system``: generated each turn by AgentMessageAssembler, never stored
    - ``user``: task or user input
    - ``assistant``: model response, optionally with tool_calls
    - ``tool``: tool result, requires ``tool_call_id``
    - ``context``: an appended canonical context event
    """

    role: Literal["system", "user", "assistant", "tool", "context"]
    content: str
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None  # required when role="tool"


# ── Tool use result ──


class ToolUseResult(BaseModel):
    """Model output after a turn.  The loop kernel uses ``stop_reason``
    (the normalized enum) for control flow; ``raw_stop_reason`` is
    debug-only and must NOT enter checkpoints."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    text: str = ""
    reasoning_content: str | None = None
    stop_reason: StopReason
    raw_stop_reason: str  # debug / logging only


def snapshot_model_message(message: ModelMessage) -> ModelMessage:
    """Detach one canonical message from caller-owned argument mappings."""

    if not isinstance(message, ModelMessage):
        raise TypeError("message must be a ModelMessage")
    if message.role not in {"system", "user", "assistant", "tool", "context"}:
        raise ValueError(f"unsupported model message role: {message.role}")
    if not isinstance(message.content, str):
        raise TypeError("model message content must be a string")
    if message.reasoning_content is not None:
        if not isinstance(message.reasoning_content, str):
            raise TypeError("model message reasoning_content must be a string or None")
        if message.role != "assistant":
            raise ValueError(
                "reasoning_content is only valid for assistant messages"
            )
    if message.tool_calls and message.role != "assistant":
        raise ValueError("only assistant messages may contain tool calls")
    if message.role == "tool":
        if not isinstance(message.tool_call_id, str) or not message.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
    elif message.tool_call_id is not None:
        raise ValueError("tool_call_id is only valid for tool messages")
    calls: list[ToolCall] = []
    for call in message.tool_calls:
        if not isinstance(call, ToolCall):
            raise TypeError("message tool_calls must contain ToolCall values")
        if not isinstance(call.id, str) or not call.id:
            raise ValueError("model tool-call id must be a non-empty string")
        if not isinstance(call.name, str) or not call.name:
            raise ValueError("model tool-call name must be a non-empty string")
        frozen_input = json_schema_output(None, call.input)
        if not isinstance(frozen_input, Mapping):
            raise TypeError("model tool-call input must be an object")
        calls.append(
            ToolCall(
                id=call.id,
                name=call.name,
                input=cast(dict[str, Any], frozen_input),
            )
        )
    return ModelMessage(
        role=message.role,
        content=message.content,
        reasoning_content=message.reasoning_content,
        tool_calls=tuple(calls),
        tool_call_id=message.tool_call_id,
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
        content=canonical_json_text(
            {
                "event_type": event_type,
                "payload": frozen_payload,
            }
        ),
    )


def tool_result_message(result: ToolResult) -> ModelMessage:
    """Fix model-visible ToolResult content at transcript insertion time."""

    from rag.agent.tools.tool import ToolResult

    if not isinstance(result, ToolResult):
        raise TypeError("result must be a ToolResult")
    payload: Mapping[str, JsonValue] = {
        "content": tuple(
            {
                "type": block.type,
                "data": block.data,
            }
            for block in result.content
        ),
        "structured_content": result.structured_content,
        "is_error": result.is_error,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "retryable": result.retryable,
        "truncated": result.truncated,
    }
    return ModelMessage(
        role="tool",
        content=canonical_json_text(payload),
        tool_call_id=result.tool_call_id,
    )


def model_message_payload(message: ModelMessage) -> Mapping[str, JsonValue]:
    """Return the canonical JSON value for one snapshotted message."""

    snapshotted = snapshot_model_message(message)
    payload: dict[str, JsonValue] = {
        "role": snapshotted.role,
        "content": snapshotted.content,
        "tool_calls": tuple(
            {
                "id": call.id,
                "name": call.name,
                "arguments": json_schema_output(None, call.input),
            }
            for call in snapshotted.tool_calls
        ),
        "tool_call_id": snapshotted.tool_call_id,
    }
    if snapshotted.reasoning_content is not None:
        payload["reasoning_content"] = snapshotted.reasoning_content
    return payload


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
    "ToolCall",
    "ToolUseResult",
    "canonical_json_text",
    "context_event_message",
    "model_message_payload",
    "snapshot_model_message",
    "tool_result_message",
]

"""Provider-neutral output facts for one model request.

These types know no Runtime turn, execution, approval, or public Item identity.
The final output snapshot is authoritative; deltas are transient previews.
Dispatch exceptions classify failed/cancelled/unknown transport outcomes, rather
than emitting a second, competing terminal outcome through this callback.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, cast

type ModelOutputKind = Literal["text", "reasoning", "reasoning_summary", "plan", "tool_call", "refusal"]
type ModelDeltaKind = Literal["text", "reasoning", "reasoning_summary", "plan", "tool_arguments", "signature"]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(part) for key, part in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(part) for part in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(part) for key, part in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(part) for part in value]
    return value


def _snapshot(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("model JSON object must be a mapping")
    return cast(Mapping[str, Any], _freeze(json.loads(json.dumps(_plain(value), allow_nan=False))))


@dataclass(frozen=True, slots=True)
class ModelOutput:
    output_id: str
    kind: ModelOutputKind
    content: str = ""
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: Mapping[str, Any] | None = None
    argument_error: str | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def delta_kind(self) -> ModelDeltaKind:
        if self.kind == "tool_call":
            return "tool_arguments"
        if self.kind == "refusal":
            return "text"
        return self.kind

    def __post_init__(self) -> None:
        if not isinstance(self.output_id, str) or not self.output_id:
            raise ValueError("model output_id must be a non-empty string")
        if self.kind not in {"text", "reasoning", "reasoning_summary", "plan", "tool_call", "refusal"}:
            raise ValueError(f"unsupported model output kind: {self.kind}")
        if not isinstance(self.content, str):
            raise TypeError("model output content must be a string")
        if self.argument_error is not None and (
            not isinstance(self.argument_error, str) or not self.argument_error.strip()
        ):
            raise ValueError("model output argument_error must be a non-empty string or None")
        if self.kind == "tool_call":
            if (
                not isinstance(self.tool_call_id, str)
                or not self.tool_call_id
                or not isinstance(self.tool_name, str)
                or not self.tool_name
            ):
                raise ValueError("tool output requires tool_call_id and tool_name")
        elif any(
            value is not None for value in (self.tool_call_id, self.tool_name, self.arguments, self.argument_error)
        ):
            raise ValueError("tool fields are only valid on tool_call outputs")
        if self.arguments is not None:
            object.__setattr__(self, "arguments", _snapshot(self.arguments))
            if not self.content and self.argument_error is None:
                object.__setattr__(
                    self, "content", json.dumps(_plain(self.arguments), ensure_ascii=False, sort_keys=True)
                )
            else:
                try:
                    parsed = json.loads(self.content)
                except json.JSONDecodeError as error:
                    if self.argument_error is None:
                        raise ValueError(
                            "complete tool arguments require valid object JSON or an argument_error"
                        ) from error
                else:
                    if not isinstance(parsed, Mapping):
                        if self.argument_error is None:
                            raise ValueError("complete tool arguments require JSON object input or an argument_error")
                    elif json.dumps(_plain(_snapshot(parsed)), sort_keys=True) != json.dumps(
                        _plain(self.arguments), sort_keys=True
                    ):
                        raise ValueError("tool arguments conflict with raw output content")
        object.__setattr__(self, "provider_data", _snapshot(self.provider_data))


def output_to_payload(output: ModelOutput) -> dict[str, Any]:
    return {
        "output_id": output.output_id,
        "kind": output.kind,
        "content": output.content,
        "tool_call_id": output.tool_call_id,
        "tool_name": output.tool_name,
        "arguments": _plain(output.arguments),
        "argument_error": output.argument_error,
        "provider_data": _plain(output.provider_data),
    }


def output_from_payload(payload: Mapping[str, Any]) -> ModelOutput:
    return ModelOutput(**dict(payload))


@dataclass(frozen=True, slots=True)
class ModelResponseStarted:
    provider_response_id: str | None = None

    def __post_init__(self) -> None:
        _validate_response_identity(self.provider_response_id)


def _validate_response_identity(value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("model response identity must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class ModelOutputStarted:
    output: ModelOutput


@dataclass(frozen=True, slots=True)
class ModelOutputDelta:
    output_id: str
    delta_kind: ModelDeltaKind
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.output_id, str) or not self.output_id:
            raise ValueError("model delta output_id must be a non-empty string")
        if self.delta_kind not in {"text", "reasoning", "reasoning_summary", "plan", "tool_arguments", "signature"}:
            raise ValueError(f"unsupported model delta kind: {self.delta_kind}")
        if not isinstance(self.content, str):
            raise TypeError("model delta content must be a string")


@dataclass(frozen=True, slots=True)
class ModelOutputCompleted:
    output: ModelOutput


@dataclass(frozen=True, slots=True)
class ModelResponseCompleted:
    usage: Mapping[str, Any]
    stop_reason: str
    provider_response_id: str | None = None
    status: Literal["completed", "incomplete"] = "completed"
    incomplete_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_response_identity(self.provider_response_id)
        if not isinstance(self.stop_reason, str) or not self.stop_reason.strip():
            raise ValueError("model stop_reason must be a non-empty string")
        if self.status not in {"completed", "incomplete"}:
            raise ValueError("invalid model response status")
        if self.status == "completed" and self.incomplete_reason is not None:
            raise ValueError("completed model responses cannot have an incomplete reason")
        if self.status == "incomplete" and (
            not isinstance(self.incomplete_reason, str) or not self.incomplete_reason.strip()
        ):
            raise ValueError("incomplete model responses require a non-empty reason")
        object.__setattr__(self, "usage", _snapshot(self.usage))


type ModelEvent = (
    ModelResponseStarted | ModelOutputStarted | ModelOutputDelta | ModelOutputCompleted | ModelResponseCompleted
)
type ModelEventSink = Callable[[ModelEvent], None | Awaitable[None]]


async def emit_model_event(sink: ModelEventSink | None, event: ModelEvent) -> None:
    if sink is not None:
        emitted = sink(event)
        if inspect.isawaitable(emitted):
            await emitted


class ModelStreamState:
    """Validate a single ordered response stream and retain output snapshots."""

    def __init__(self) -> None:
        self.started = False
        self.completed = False
        self.provider_response_id: str | None = None
        self.terminal: ModelResponseCompleted | None = None
        self._outputs: dict[str, ModelOutput] = {}
        self._closed: set[str] = set()

    @property
    def outputs(self) -> tuple[ModelOutput, ...]:
        return tuple(self._outputs.values())

    def validate_response(
        self,
        *,
        outputs: tuple[ModelOutput, ...],
        provider_response_id: str | None,
        status: str,
        incomplete_reason: str | None,
        usage: Mapping[str, Any],
        stop_reason: str | None = None,
    ) -> None:
        """Reject contradictory callback and returned terminal snapshots."""
        if self.terminal is None:
            raise ValueError("model stream has no terminal response")
        if self.outputs != outputs:
            raise ValueError("model stream outputs disagree with returned response")
        if self.provider_response_id != provider_response_id:
            raise ValueError("model stream response identity disagrees with returned response")
        if self.terminal.status != status or self.terminal.incomplete_reason != incomplete_reason:
            raise ValueError("model stream terminal outcome disagrees with returned response")
        if stop_reason is not None and self.terminal.stop_reason != stop_reason:
            raise ValueError("model stream stop reason disagrees with returned response")
        returned_usage = _snapshot(usage)
        # Adapter pricing and Runtime timings may add fields. Every observed
        # provider fact must still survive unchanged, including nested raw usage.
        for key, value in self.terminal.usage.items():
            if key not in returned_usage or returned_usage[key] != value:
                raise ValueError(f"model stream usage field {key!r} disagrees with returned response")

    def accept(self, event: ModelEvent) -> None:
        if self.completed:
            raise ValueError("model response already terminated")
        if isinstance(event, ModelResponseStarted):
            if self.started:
                raise ValueError("model response already started")
            self.started = True
            self.provider_response_id = event.provider_response_id
            return
        if not self.started:
            raise ValueError("model response not started")
        if isinstance(event, ModelOutputStarted):
            if event.output.output_id in self._outputs:
                raise ValueError("model output already started")
            self._outputs[event.output.output_id] = event.output
            return
        if isinstance(event, ModelResponseCompleted):
            if set(self._outputs) != self._closed:
                raise ValueError("model response has open outputs")
            if (
                self.provider_response_id
                and event.provider_response_id
                and self.provider_response_id != event.provider_response_id
            ):
                raise ValueError("model response identity changed")
            self.provider_response_id = event.provider_response_id or self.provider_response_id
            self.terminal = event
            self.completed = True
            return
        output_id = event.output.output_id if isinstance(event, ModelOutputCompleted) else event.output_id
        if output_id not in self._outputs:
            raise ValueError("model output not started")
        if output_id in self._closed:
            raise ValueError("model output already completed")
        output = self._outputs[output_id]
        if isinstance(event, ModelOutputCompleted):
            if (
                output.kind != event.output.kind
                or output.tool_call_id != event.output.tool_call_id
                or output.tool_name != event.output.tool_name
            ):
                raise ValueError("model output identity or kind changed")
            self._outputs[output_id] = event.output
            self._closed.add(output_id)
            return
        if event.delta_kind == "signature":
            if output.kind not in {"reasoning", "reasoning_summary"}:
                raise ValueError("signature delta requires reasoning output")
            provider_data = _plain(output.provider_data)
            provider_data["signature"] = provider_data.get("signature", "") + event.content
            self._outputs[output_id] = replace(output, provider_data=provider_data)
            return
        expected = (
            "tool_arguments" if output.kind == "tool_call" else ("text" if output.kind == "refusal" else output.kind)
        )
        if event.delta_kind != expected:
            raise ValueError("model delta kind does not match output")
        self._outputs[output_id] = replace(output, content=output.content + event.content)

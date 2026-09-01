from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterator
from typing import Any, TypeVar, cast

from agent_runtime.modeling.contracts import LLMProviderResult, LLMUsage
from agent_runtime.modeling.openai_wire import parse_openai_usage

T = TypeVar("T")
_JSON_CODE_FENCE_RE = re.compile(r"^\s*```\s*(?:[A-Za-z0-9_-]+)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)


class OpenAICompatibleChatGenerator:
    """Lazy OpenAI-compatible chat client used by the agent and RAG assembly."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        supports_tools: bool | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.chat_model_name = model
        self._base_url = base_url
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=timeout_seconds,
        )
        self._supports_tools = (
            (os.environ.get("RAG_NATIVE_TOOL_CALLING", "").lower() not in {"0", "false", "no", "off"})
            if supports_tools is None
            else supports_tools
        )
        self._stream_lock = threading.Lock()
        self._active_stream: object | None = None

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    def generate_text(self, *, prompt: str, system_prompt: str | None = None, **kwargs: object) -> str:
        return self.generate_text_with_usage(prompt=prompt, system_prompt=system_prompt, **kwargs).value

    def generate_text_with_usage(
        self, *, prompt: str, system_prompt: str | None = None, **kwargs: object
    ) -> LLMProviderResult[str]:
        system_instructions = kwargs.pop("system_instructions", None)
        if system_prompt is None and isinstance(system_instructions, str):
            system_prompt = system_instructions
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(model=self.chat_model_name, messages=messages, **kwargs)  # type: ignore[call-overload]
        content = response.choices[0].message.content
        return LLMProviderResult(
            value=str(content) if content is not None else "", usage=_openai_response_usage(response)
        )

    def generate_structured(
        self, *, prompt: str, schema: type[T], system_prompt: str | None = None, **kwargs: object
    ) -> T:
        return self.generate_structured_with_usage(
            prompt=prompt, schema=schema, system_prompt=system_prompt, **kwargs
        ).value

    def generate_structured_with_usage(
        self, *, prompt: str, schema: type[T], system_prompt: str | None = None, **kwargs: object
    ) -> LLMProviderResult[T]:
        schema_json = json.dumps(cast(Any, schema).model_json_schema(), ensure_ascii=False)
        structured_prompt = (
            f"Return ONLY valid JSON matching this schema.\n\nJSON schema:\n{schema_json}\n\nUser task:\n{prompt}"
        )
        generated = self.generate_text_with_usage(
            prompt=structured_prompt,
            system_prompt=system_prompt,
            **kwargs,
        )
        raw = _extract_json_object(_strip_json_code_fence(generated.value)).strip()
        try:
            value = schema.model_validate(json.loads(raw))  # type: ignore[attr-defined]
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI-compatible structured fallback returned invalid JSON") from exc
        return LLMProviderResult(value=value, usage=generated.usage)

    def generate_with_tools(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: object
    ) -> LLMProviderResult[Any]:
        response = self._client.chat.completions.create(
            model=self.chat_model_name, messages=messages, tools=tools or None, **kwargs
        )  # type: ignore[call-overload]
        return LLMProviderResult(value=response, usage=_openai_response_usage(response))

    def list_models(self) -> tuple[str, ...]:
        """Return provider-advertised model ids through the configured client."""

        response = self._client.models.list()
        return tuple(str(model.id) for model in response.data)

    def stream_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: object,
    ) -> Iterator[dict[str, object]]:
        """Translate the OpenAI-compatible SSE stream into gateway chunks."""

        stream = self._client.chat.completions.create(
            model=self.chat_model_name,
            messages=messages,
            tools=tools or None,
            stream=True,
            **kwargs,
        )  # type: ignore[call-overload]
        if _is_complete_chat_response(stream):
            yield from _complete_response_chunks(stream)
            return
        stream_lock = getattr(self, "_stream_lock", None)
        if stream_lock is None:
            stream_lock = threading.Lock()
            self._stream_lock = stream_lock
        with stream_lock:
            self._active_stream = stream
        tool_blocks: dict[int, tuple[str, str]] = {}
        authoritative_stop = False
        try:
            for response_chunk in stream:
                choices = response_chunk.choices
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.delta
                content = getattr(delta, "content", None)
                if content:
                    yield {"type": "text_delta", "content": str(content)}
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield {"type": "thinking_delta", "content": str(reasoning)}
                plan = getattr(delta, "plan", None)
                if plan:
                    yield {"type": "plan_delta", "content": str(plan)}
                for tool_call in getattr(delta, "tool_calls", None) or ():
                    index = int(tool_call.index)
                    function = tool_call.function
                    if index not in tool_blocks:
                        tool_id = str(tool_call.id or f"tool-{index}")
                        tool_name = str(function.name or "")
                        tool_blocks[index] = (tool_id, tool_name)
                        yield {
                            "type": "tool_use_start",
                            "tool_name": tool_name,
                            "tool_id": tool_id,
                        }
                    if function.arguments:
                        tool_id, _tool_name = tool_blocks[index]
                        yield {
                            "type": "tool_input_delta",
                            "content": str(function.arguments),
                            "tool_id": tool_id,
                        }
                finish_reason = choice.finish_reason
                if finish_reason is not None:
                    for _index in sorted(tool_blocks):
                        tool_id, _tool_name = tool_blocks[_index]
                        yield {"type": "content_block_stop", "tool_id": tool_id}
                    yield {
                        "type": "message_stop",
                        "stop_reason": _stream_stop_reason(str(finish_reason)),
                        "usage": _openai_response_usage(response_chunk),
                    }
                    authoritative_stop = True
            if not authoritative_stop:
                raise RuntimeError("OpenAI-compatible stream ended without a finish reason")
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
            with stream_lock:
                if self._active_stream is stream:
                    self._active_stream = None

    def cancel_stream(self) -> None:
        """Best-effort cancellation hook used by the gateway's async bridge."""

        stream_lock = getattr(self, "_stream_lock", None)
        if stream_lock is None:
            return
        with stream_lock:
            stream = getattr(self, "_active_stream", None)
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    def __repr__(self) -> str:
        return f"OpenAICompatibleChatGenerator(model={self.chat_model_name!r}, base_url={self._base_url!r})"


def _strip_json_code_fence(text: str) -> str:
    match = _JSON_CODE_FENCE_RE.match(text)
    return match.group("body") if match else text


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    return stripped[start : end + 1] if start != -1 and start < end else stripped


def _openai_response_usage(response: object) -> LLMUsage | None:
    normalized = parse_openai_usage(response)
    if normalized is None:
        return None
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return normalized.model_copy(update={"reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0)})


def _stream_stop_reason(value: str) -> str:
    if value == "tool_calls":
        return "tool_use"
    if value == "length":
        return "max_tokens"
    return "end_turn"


def _is_complete_chat_response(value: object) -> bool:
    choices = getattr(value, "choices", None)
    return isinstance(choices, (list, tuple)) and bool(choices) and hasattr(choices[0], "message")


def _complete_response_chunks(response: object) -> Iterator[dict[str, object]]:
    """Normalize providers that ignore stream=True and return one completion."""

    choice = response.choices[0]  # type: ignore[attr-defined]
    message = choice.message
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        yield {"type": "thinking_delta", "content": str(reasoning)}
    content = getattr(message, "content", None)
    if content:
        yield {"type": "text_delta", "content": str(content)}
    tool_calls = getattr(message, "tool_calls", None) or ()
    for index, tool_call in enumerate(tool_calls):
        tool_id = str(tool_call.id or f"tool-{index}")
        function = tool_call.function
        yield {
            "type": "tool_use_start",
            "tool_name": str(function.name or ""),
            "tool_id": tool_id,
        }
        if function.arguments:
            yield {
                "type": "tool_input_delta",
                "content": str(function.arguments),
                "tool_id": tool_id,
            }
        yield {"type": "content_block_stop", "tool_id": tool_id}
    yield {
        "type": "message_stop",
        "stop_reason": _stream_stop_reason(str(choice.finish_reason or "stop")),
        "usage": _openai_response_usage(response),
    }


__all__ = ["OpenAICompatibleChatGenerator"]

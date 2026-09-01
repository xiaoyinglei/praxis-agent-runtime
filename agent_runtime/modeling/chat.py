from __future__ import annotations

import json
import os
import re
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


__all__ = ["OpenAICompatibleChatGenerator"]

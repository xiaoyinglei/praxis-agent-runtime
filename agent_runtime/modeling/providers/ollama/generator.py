from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class OllamaGenerator:
    """
    Ollama 文本生成专员。
    只负责文本生成与结构化生成，不负责 embedding。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen2.5:7b",
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
        )

    def generate_text(self, *, prompt: str, **kwargs: Any) -> str:
        model = kwargs.pop("model", self._default_model)
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", None)
        response_format = kwargs.pop("format", None)

        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        request_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
        }
        if response_format is not None:
            request_payload["format"] = response_format

        try:
            response = self._client.post(
                f"{self._base_url}/api/chat",
                json=request_payload,
            )
            response.raise_for_status()
            response_payload = response.json()
            return str(response_payload["message"]["content"]).strip()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama text generation failed: {exc}") from exc
        except KeyError as exc:
            raise RuntimeError("Ollama response missing message.content") from exc

    def generate_structured(self, *, prompt: str, schema: type[T], **kwargs: Any) -> T:
        schema_json = schema.model_json_schema()

        structured_prompt = f"""
Return ONLY valid JSON matching this schema.
JSON schema:
{schema_json}

User task:
{prompt}
""".strip()

        # 触发 Ollama 的底层的强制 JSON 模式，配合你的 Prompt，可以说是双保险，成功率 99.99%
        raw = self.generate_text(prompt=structured_prompt, format="json", **kwargs)

        try:
            return schema.model_validate_json(raw)
        except Exception as exc:
            raise RuntimeError(f"Ollama structured generation failed to parse into {schema.__name__}: {exc}") from exc

    def close(self) -> None:
        self._client.close()

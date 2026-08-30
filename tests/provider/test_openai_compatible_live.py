from __future__ import annotations

import json
import os

import pytest

from agent_runtime.core.model_request import (
    ModelSettings,
    build_model_request,
    build_stable_context,
)
from agent_runtime.modeling.contracts import LLMCallStage
from agent_runtime.modeling.gateway import LLMGateway
from rag.assembly.support import _OpenAICompatibleChatGenerator

_BASE_URL = os.environ.get("RAG_TEST_OPENAI_BASE_URL")
_MODEL = os.environ.get("RAG_TEST_OPENAI_MODEL")
_API_KEY = os.environ.get("RAG_TEST_OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not (_BASE_URL and _MODEL),
    reason=("set RAG_TEST_OPENAI_BASE_URL and RAG_TEST_OPENAI_MODEL to run live OpenAI-compatible tests"),
)


class _WhitespaceTokenAccounting:
    def count(self, text: str) -> int:
        return len(text.split())

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str:
        words = text.split()[:token_budget]
        clipped = " ".join(words)
        if add_ellipsis and len(words) < len(text.split()):
            return clipped + "..."
        return clipped


def _live_gateway() -> LLMGateway:
    assert _BASE_URL is not None
    assert _MODEL is not None
    return LLMGateway(
        generator=_OpenAICompatibleChatGenerator(
            model=_MODEL,
            base_url=_BASE_URL,
            api_key=_API_KEY,
        ),
        token_accounting=_WhitespaceTokenAccounting(),
        model_context_tokens=32_768,
    )


@pytest.mark.anyio
async def test_live_canonical_text_completion() -> None:
    assert _MODEL is not None
    request = build_model_request(
        request_id="live-openai-text",
        context=build_stable_context(
            instructions=("Answer briefly.",),
            initial_user_task="Reply with: live adapter ok",
        ),
        selected_tools=(),
        settings=ModelSettings(
            model=_MODEL,
            max_output_tokens=32,
            temperature=0.0,
        ),
    )

    result = await _live_gateway().agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=request,
        provider="openai-compatible",
        supports_native_tools=True,
    )

    assert result.turn.text.strip()
    assert result.turn.tool_calls == []
    assert result.usage.usage_source == "provider"


@pytest.mark.anyio
async def test_live_simulated_streaming_tool_completion() -> None:
    chunks = [
        chunk
        async for chunk in _live_gateway().astream_with_tools(
            stage=LLMCallStage.TOOL_DECISION,
            messages=[
                {
                    "role": "user",
                    "content": "Call live_probe with value exactly ok.",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "live_probe",
                        "description": "Return the supplied probe value.",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            kwargs={
                "temperature": 0.0,
                "max_tokens": 64,
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "live_probe"},
                },
            },
        )
    ]

    starts = [chunk for chunk in chunks if chunk.type == "tool_use_start"]
    inputs = [chunk for chunk in chunks if chunk.type == "tool_input_delta"]
    assert len(starts) == 1
    assert starts[0].tool_name == "live_probe"
    assert len(inputs) == 1
    assert json.loads(inputs[0].content) == {"value": "ok"}
    assert chunks[-1].type == "message_stop"
    assert chunks[-1].stop_reason == "tool_use"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.usage_source == "provider"

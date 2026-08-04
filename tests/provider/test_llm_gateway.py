from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from agent_runtime.core.context import LLMBudgetLedger
from agent_runtime.core.model_request import (
    ModelSettings,
    build_model_request,
    build_stable_context,
)
from agent_runtime.modeling.contracts import (
    LLMCallStage,
    LLMProviderResult,
    LLMStageBudget,
    LLMUsage,
)
from agent_runtime.modeling.gateway import (
    LLMContextOverflowError,
    LLMGateway,
    LLMToolCallValidationError,
    StreamChunk,
)
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    json_schema_input,
)
from rag.assembly.support import _OpenAICompatibleChatGenerator


class _WordTokenAccounting:
    def count(self, text: str) -> int:
        return len(text.split())


class _UsageAwareGenerator:
    def __init__(
        self,
        *,
        output: str = "final answer",
        usage: LLMUsage | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.output = output
        self.usage = usage
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate_text_with_usage(
        self,
        *,
        prompt: str,
        **kwargs: Any,
    ) -> LLMProviderResult[str]:
        self.calls.append((prompt, kwargs))
        if self.error is not None:
            raise self.error
        return LLMProviderResult(value=self.output, usage=self.usage)


class _Decision(BaseModel):
    action: str


class _StructuredUsageAwareGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_structured_with_usage(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> LLMProviderResult[BaseModel]:
        del prompt
        self.calls.append(kwargs)
        return LLMProviderResult(
            value=schema.model_validate({"action": "execute"}),
            usage=LLMUsage(
                input_tokens=6,
                output_tokens=2,
                cached_input_tokens=3,
                reasoning_tokens=1,
                source="provider",
            ),
        )


class _StreamingGenerator:
    def stream_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[StreamChunk]:
        del messages, tools, kwargs
        return [
            StreamChunk(type="text_delta", content="one two"),
            StreamChunk(type="message_stop", stop_reason="end_turn"),
        ]


class _CanonicalNativeGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMProviderResult[dict[str, object]]:
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        return LLMProviderResult(
            value={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "native answer"},
                    }
                ]
            },
            usage=LLMUsage(
                input_tokens=13,
                output_tokens=2,
                cached_input_tokens=5,
                source="provider",
                logical_input_tokens=13,
                uncached_input_tokens=8,
                cache_read_input_tokens=5,
                cache_write_input_tokens=0,
                usage_source="provider",
                raw_provider_usage={"prompt_tokens": 13, "cached_tokens": 5},
            ),
        )


class _ReasoningNativeGenerator:
    def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMProviderResult[dict[str, object]]:
        del messages, tools, kwargs
        return LLMProviderResult(
            value={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "Inspect the source before editing.",
                            "tool_calls": [
                                {
                                    "id": "call_reasoning",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
            usage=LLMUsage(
                input_tokens=20,
                output_tokens=8,
                source="provider",
            ),
        )


class _RejectedToolCallGenerator:
    def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMProviderResult[dict[str, object]]:
        del messages, tools, kwargs
        error = RuntimeError("provider rejected generated tool call")
        error.body = {  # type: ignore[attr-defined]
            "message": "Tool call validation failed: max_bytes exceeds maximum",
            "code": "tool_use_failed",
            "failed_generation": ('<function=read_file>{"path":"README.md","max_bytes":2000000}</function>'),
        }
        raise error


class _StructuredRejectedToolCallGenerator:
    def generate_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMProviderResult[dict[str, object]]:
        del messages, tools, kwargs
        error = RuntimeError("provider rejected generated tool call")
        error.body = {  # type: ignore[attr-defined]
            "error": {
                "message": "Tool call validation failed: invalid arguments",
                "code": "tool_use_failed",
                "failed_generation": {
                    "reason": "schema_validation",
                    "tool_call_id": "call_read_file",
                    "attempted_arguments": {
                        "path": "README.md",
                        "line_start": 20,
                    },
                },
            }
        }
        raise error


class _CanonicalLocalGenerator(_UsageAwareGenerator):
    def __init__(self) -> None:
        super().__init__(
            output='{"text":"local answer","tool_calls":[]}',
            usage=LLMUsage(
                input_tokens=17,
                output_tokens=3,
                source="provider",
                logical_input_tokens=17,
                uncached_input_tokens=17,
                usage_source="provider",
                raw_provider_usage={"input_tokens": 17},
            ),
        )


class _CanonicalStreamingGenerator:
    def stream_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[StreamChunk]:
        del messages, tools, kwargs
        return [
            StreamChunk(type="text_delta", content="stream answer"),
            StreamChunk(
                type="message_stop",
                stop_reason="end_turn",
                usage=LLMUsage(
                    input_tokens=21,
                    output_tokens=2,
                    cached_input_tokens=9,
                    source="provider",
                    logical_input_tokens=21,
                    uncached_input_tokens=12,
                    cache_read_input_tokens=9,
                    cache_write_input_tokens=0,
                    usage_source="provider",
                    raw_provider_usage={"prompt_tokens": 21, "cached_tokens": 9},
                ),
            ),
        ]


def _canonical_tool(name: str) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Use {name}.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=lambda _arguments: {},
        normalize_output=lambda _raw: NormalizedToolOutput(),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(effects=frozenset(), targets=()),
        execution_revision=f"{name}-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


def _canonical_request():
    return build_model_request(
        request_id="request-parity",
        context=build_stable_context(
            instructions=("Use the selected tools when needed.",),
            initial_user_task="Inspect README.md.",
        ),
        selected_tools=(
            _canonical_tool("list_files"),
            _canonical_tool("read_file"),
        ),
        settings=ModelSettings(
            model="test-model",
            max_output_tokens=128,
            temperature=0.0,
            parallel_tool_calls=True,
        ),
    )


def _gateway(
    generator: object,
    *,
    max_input_tokens: int = 8,
    max_output_tokens: int = 4,
    model_context_tokens: int = 32,
) -> LLMGateway:
    return LLMGateway(
        generator=generator,
        token_accounting=_WordTokenAccounting(),  # type: ignore[arg-type]
        model_context_tokens=model_context_tokens,
        stage_budgets={
            LLMCallStage.TOOL_DECISION: LLMStageBudget(
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                safety_margin_tokens=2,
            )
        },
    )


def _openai_generator_for_response(
    response: ChatCompletion,
) -> _OpenAICompatibleChatGenerator:
    generator = _OpenAICompatibleChatGenerator.__new__(_OpenAICompatibleChatGenerator)
    generator.chat_model_name = "test-model"
    generator._base_url = "https://example.test/v1"
    generator._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))
    return generator


@pytest.mark.anyio
async def test_gateway_reserves_worst_case_and_commits_provider_usage() -> None:
    generator = _UsageAwareGenerator(usage=LLMUsage(input_tokens=3, output_tokens=2, source="provider"))
    ledger = LLMBudgetLedger(total=20)

    result = await _gateway(generator).agenerate_text(
        stage=LLMCallStage.TOOL_DECISION,
        prompt="one two three",
        ledger=ledger,
        lease_id="decision:1",
    )

    assert result.value == "final answer"
    assert result.usage.source == "provider"
    assert result.usage.total_tokens == 5
    assert await ledger.committed() == 5
    assert await ledger.reserved() == 0
    assert generator.calls[0][1]["max_tokens"] == 4


@pytest.mark.anyio
async def test_gateway_estimates_usage_when_provider_does_not_report_it() -> None:
    generator = _UsageAwareGenerator(usage=None, output="four five")
    ledger = LLMBudgetLedger(total=20)

    result = await _gateway(generator).agenerate_text(
        stage=LLMCallStage.TOOL_DECISION,
        prompt="one two three",
        ledger=ledger,
        lease_id="decision:2",
    )

    assert result.usage == LLMUsage(
        input_tokens=3,
        output_tokens=2,
        source="tokenizer_estimate",
    )
    assert await ledger.committed() == 5


@pytest.mark.anyio
async def test_gateway_refunds_reservation_when_provider_fails() -> None:
    generator = _UsageAwareGenerator(error=RuntimeError("provider unavailable"))
    ledger = LLMBudgetLedger(total=20)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await _gateway(generator).agenerate_text(
            stage=LLMCallStage.TOOL_DECISION,
            prompt="one two three",
            ledger=ledger,
            lease_id="decision:3",
        )

    assert await ledger.committed() == 0
    assert await ledger.reserved() == 0
    assert await ledger.remaining() == 20


@pytest.mark.anyio
async def test_gateway_refunds_reservation_when_provider_call_is_cancelled() -> None:
    generator = _UsageAwareGenerator(error=asyncio.CancelledError())
    ledger = LLMBudgetLedger(total=20)

    with pytest.raises(asyncio.CancelledError):
        await _gateway(generator).agenerate_text(
            stage=LLMCallStage.TOOL_DECISION,
            prompt="one two three",
            ledger=ledger,
            lease_id="decision:cancelled",
        )

    assert await ledger.committed() == 0
    assert await ledger.reserved() == 0
    assert await ledger.remaining() == 20


@pytest.mark.anyio
async def test_gateway_rejects_prompt_above_stage_input_budget() -> None:
    generator = _UsageAwareGenerator()
    ledger = LLMBudgetLedger(total=100)

    with pytest.raises(LLMContextOverflowError) as exc_info:
        await _gateway(generator, max_input_tokens=3).agenerate_text(
            stage=LLMCallStage.TOOL_DECISION,
            prompt="one two three four",
            ledger=ledger,
            lease_id="decision:4",
        )

    assert exc_info.value.input_tokens == 4
    assert exc_info.value.max_input_tokens == 3
    assert generator.calls == []
    assert await ledger.remaining() == 100


@pytest.mark.anyio
async def test_gateway_respects_model_window_after_output_and_safety_reserve() -> None:
    generator = _UsageAwareGenerator()

    with pytest.raises(LLMContextOverflowError) as exc_info:
        await _gateway(
            generator,
            max_input_tokens=20,
            max_output_tokens=4,
            model_context_tokens=10,
        ).agenerate_text(
            stage=LLMCallStage.TOOL_DECISION,
            prompt="one two three four five",
            ledger=LLMBudgetLedger(total=100),
            lease_id="decision:5",
        )

    assert exc_info.value.max_input_tokens == 4
    assert generator.calls == []


def test_gateway_exposes_effective_stage_budget_for_context_assembly() -> None:
    gateway = _gateway(
        _UsageAwareGenerator(),
        max_input_tokens=20,
        max_output_tokens=4,
        model_context_tokens=10,
    )

    budget = gateway.effective_stage_budget(LLMCallStage.TOOL_DECISION)

    assert budget.max_input_tokens == 4
    assert budget.max_output_tokens == 4


@pytest.mark.anyio
async def test_gateway_accounts_for_structured_generation_as_one_call() -> None:
    generator = _StructuredUsageAwareGenerator()
    ledger = LLMBudgetLedger(total=30)

    result = await _gateway(
        generator,
        max_input_tokens=100,
        model_context_tokens=200,
    ).agenerate_structured(
        stage=LLMCallStage.TOOL_DECISION,
        prompt="choose action",
        schema=_Decision,
        ledger=ledger,
        lease_id="decision:structured",
    )

    assert result.value == _Decision(action="execute")
    assert result.usage.cached_input_tokens == 3
    assert result.usage.reasoning_tokens == 1
    assert result.usage.total_tokens == 8
    assert await ledger.committed() == 8
    assert generator.calls[0]["max_tokens"] == 4


@pytest.mark.anyio
async def test_streaming_gateway_commits_tokenizer_estimate_usage() -> None:
    ledger = LLMBudgetLedger(total=20)

    chunks = [
        chunk
        async for chunk in _gateway(_StreamingGenerator()).astream_with_tools(
            stage=LLMCallStage.TOOL_DECISION,
            messages=[{"role": "user", "content": "hello prompt"}],
            tools=[],
            ledger=ledger,
            lease_id="decision:stream",
        )
    ]

    assert [chunk.type for chunk in chunks] == ["text_delta", "message_stop"]
    assert chunks[0].content == "one two"
    assert await ledger.committed() == 5
    assert await ledger.reserved() == 0


@pytest.mark.anyio
async def test_streaming_gateway_refunds_when_consumer_closes_early() -> None:
    ledger = LLMBudgetLedger(total=20)
    stream = _gateway(_StreamingGenerator()).astream_with_tools(
        stage=LLMCallStage.TOOL_DECISION,
        messages=[{"role": "user", "content": "hello prompt"}],
        tools=[],
        ledger=ledger,
        lease_id="decision:stream-close",
    )

    chunk = await stream.__anext__()
    assert chunk.type == "text_delta"

    await stream.aclose()

    assert await ledger.committed() == 0
    assert await ledger.reserved() == 0
    assert await ledger.remaining() == 20


@pytest.mark.anyio
async def test_streaming_fallback_parses_sdk_text_completion(
    chat_completion_factory: Callable[..., ChatCompletion],
) -> None:
    response = chat_completion_factory(content="provider answer", tool_calls=None)

    chunks = [
        chunk
        async for chunk in _gateway(_openai_generator_for_response(response)).astream_with_tools(
            stage=LLMCallStage.TOOL_DECISION,
            messages=[{"role": "user", "content": "hello prompt"}],
            tools=[],
        )
    ]

    assert [chunk.type for chunk in chunks] == ["text_delta", "message_stop"]
    assert chunks[0].content == "provider answer"
    assert chunks[-1].stop_reason == "end_turn"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.usage_source == "provider"


@pytest.mark.anyio
async def test_streaming_fallback_parses_sdk_tool_completion(
    chat_completion_factory: Callable[..., ChatCompletion],
) -> None:
    response = chat_completion_factory(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_fixture",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                },
            }
        ],
    )

    chunks = [
        chunk
        async for chunk in _gateway(_openai_generator_for_response(response)).astream_with_tools(
            stage=LLMCallStage.TOOL_DECISION,
            messages=[{"role": "user", "content": "read the file"}],
            tools=[],
        )
    ]

    assert [chunk.type for chunk in chunks] == [
        "tool_use_start",
        "tool_input_delta",
        "content_block_stop",
        "message_stop",
    ]
    assert chunks[0].tool_id == "call_fixture"
    assert chunks[0].tool_name == "read_file"
    assert chunks[1].content == '{"path":"README.md"}'
    assert chunks[-1].stop_reason == "tool_use"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.usage_source == "provider"


def test_openai_compatible_generator_normalizes_sdk_response_usage(
    chat_completion_factory: Callable[..., ChatCompletion],
) -> None:
    response = chat_completion_factory(
        content="provider answer",
        prompt_tokens=11,
        completion_tokens=7,
        cached_tokens=5,
        reasoning_tokens=3,
    )
    generator = _openai_generator_for_response(response)

    result = generator.generate_text_with_usage(prompt="question", max_tokens=20)

    assert result.value == "provider answer"
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.reasoning_tokens == 3
    assert result.usage.logical_input_tokens == 11
    assert result.usage.uncached_input_tokens == 6
    assert result.usage.cache_read_input_tokens == 5
    assert result.usage.usage_source == "provider"
    assert result.usage.raw_provider_usage is not None


@pytest.mark.anyio
async def test_gateway_adapts_one_canonical_request_to_openai_wire() -> None:
    generator = _CanonicalNativeGenerator()
    request = _canonical_request()

    result = await _gateway(
        generator,
        max_input_tokens=2_000,
        model_context_tokens=4_000,
    ).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=request,
        provider="openai-compatible",
        supports_native_tools=True,
    )

    assert result.turn.text == "native answer"
    assert result.usage.cache_read_input_tokens == 5
    assert result.usage.usage_source == "provider"
    assert result.provider_wire_hash.startswith("wire_")
    assert result.serializer_revision == "openai-compatible-chat-v2"
    assert result.wire_kind == "openai"
    assert [item["function"]["name"] for item in generator.calls[0]["tools"]] == [
        "list_files",
        "read_file",
    ]
    assert "model" not in generator.calls[0]


@pytest.mark.anyio
async def test_canonical_provider_options_use_openai_extra_body_transport() -> None:
    generator = _CanonicalNativeGenerator()
    original = _canonical_request()
    request = replace(
        original,
        settings=replace(
            original.settings,
            provider_options={"thinking": {"type": "enabled"}},
        ),
    )

    await _gateway(
        generator,
        max_input_tokens=2_000,
        model_context_tokens=4_000,
    ).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=request,
        provider="openai-compatible",
        supports_native_tools=True,
    )

    assert "thinking" not in generator.calls[0]
    assert generator.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.mark.anyio
async def test_canonical_request_output_limit_reaches_native_provider() -> None:
    generator = _CanonicalNativeGenerator()
    original = _canonical_request()
    request = replace(
        original,
        settings=replace(original.settings, max_output_tokens=2),
    )

    await _gateway(
        generator,
        max_input_tokens=2_000,
        max_output_tokens=4,
        model_context_tokens=4_000,
    ).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=request,
        provider="openai-compatible",
        supports_native_tools=True,
    )

    assert generator.calls[0]["max_tokens"] == 2


@pytest.mark.anyio
async def test_gateway_normalizes_provider_tool_call_validation_failure() -> None:
    with pytest.raises(LLMToolCallValidationError) as exc_info:
        await _gateway(
            _RejectedToolCallGenerator(),
            max_input_tokens=2_000,
            model_context_tokens=4_000,
        ).agenerate_model_request(
            stage=LLMCallStage.TOOL_DECISION,
            request=_canonical_request(),
            provider="openai-compatible",
            supports_native_tools=True,
        )

    assert "max_bytes exceeds maximum" in exc_info.value.validation_error
    assert '"max_bytes":2000000' in exc_info.value.failed_generation


@pytest.mark.anyio
async def test_gateway_preserves_structured_failed_generation() -> None:
    with pytest.raises(LLMToolCallValidationError) as exc_info:
        await _gateway(
            _StructuredRejectedToolCallGenerator(),
            max_input_tokens=2_000,
            model_context_tokens=4_000,
        ).agenerate_model_request(
            stage=LLMCallStage.TOOL_DECISION,
            request=_canonical_request(),
            provider="openai-compatible",
            supports_native_tools=True,
        )

    failed_generation = json.loads(exc_info.value.failed_generation)
    assert failed_generation == {
        "attempted_arguments": {
            "line_start": 20,
            "path": "README.md",
        },
        "reason": "schema_validation",
        "tool_call_id": "call_read_file",
    }


@pytest.mark.anyio
async def test_streaming_gateway_preserves_rejected_tool_call_feedback() -> None:
    with pytest.raises(LLMToolCallValidationError) as exc_info:
        await _gateway(
            _RejectedToolCallGenerator(),
            max_input_tokens=2_000,
            model_context_tokens=4_000,
        ).agenerate_model_request(
            stage=LLMCallStage.TOOL_DECISION,
            request=_canonical_request(),
            provider="openai-compatible",
            supports_native_tools=True,
            stream=True,
        )

    assert "max_bytes exceeds maximum" in exc_info.value.validation_error
    assert '"max_bytes":2000000' in exc_info.value.failed_generation


@pytest.mark.anyio
@pytest.mark.parametrize("provider", ["mlx", "ollama"])
async def test_gateway_adapts_one_canonical_request_to_local_envelope(provider: str) -> None:
    generator = _CanonicalLocalGenerator()
    request = _canonical_request()

    result = await _gateway(
        generator,
        max_input_tokens=2_000,
        model_context_tokens=4_000,
    ).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=request,
        provider=provider,
        supports_native_tools=False,
    )

    assert result.turn.text == "local answer"
    assert result.usage.usage_source == "provider"
    assert result.provider_wire_hash.startswith("wire_")
    assert result.serializer_revision == "local-agent-flat-json-v1"
    assert result.wire_kind == provider
    prompt, kwargs = generator.calls[0]
    assert "[Selected Tools]" in prompt
    assert "list_files" in prompt
    assert "read_file" in prompt
    assert kwargs["model"] == "test-model"


@pytest.mark.anyio
async def test_streaming_canonical_request_uses_final_provider_usage() -> None:
    request = _canonical_request()
    deltas: list[str] = []

    result = await _gateway(
        _CanonicalStreamingGenerator(),
        max_input_tokens=2_000,
        model_context_tokens=4_000,
    ).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=request,
        provider="openai-compatible",
        supports_native_tools=True,
        stream=True,
        text_delta_sink=deltas.append,
    )

    assert result.turn.text == "stream answer"
    assert result.usage.cache_read_input_tokens == 9
    assert result.usage.usage_source == "provider"
    assert deltas == ["stream answer"]


@pytest.mark.anyio
async def test_streaming_fallback_preserves_reasoning_for_next_tool_turn() -> None:
    result = await _gateway(
        _ReasoningNativeGenerator(),
        max_input_tokens=2_000,
        max_output_tokens=128,
        model_context_tokens=4_000,
    ).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=_canonical_request(),
        provider="openai-compatible",
        supports_native_tools=True,
        stream=True,
    )

    assert result.turn.reasoning_content == "Inspect the source before editing."
    assert result.turn.tool_calls[0].name == "read_file"

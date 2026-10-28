from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from agent_runtime.core.messages import StopReason
from agent_runtime.core.model_request import (
    ModelSettings,
    build_model_request,
    build_stable_context,
)
from agent_runtime.modeling.budget import LLMBudgetLedger
from agent_runtime.modeling.contracts import (
    LLMCallStage,
    LLMProviderResult,
    LLMStageBudget,
    LLMUsage,
)
from agent_runtime.modeling.events import ModelEvent, ModelOutputDelta
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


class _BlockingStreamingGenerator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def stream_with_tools(self, **_kwargs: object) -> list[StreamChunk]:
        self.started.set()
        self.release.wait()
        return []


class _FastStreamingGenerator:
    def __init__(self) -> None:
        self.produced = 0

    def stream_with_tools(self, **_kwargs: object):
        for _index in range(1_000):
            self.produced += 1
            yield StreamChunk(type="text_delta", content="x")
        yield StreamChunk(type="message_stop", stop_reason="end_turn")


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
            StreamChunk(type="thinking_delta", content="inspect evidence"),
            StreamChunk(type="plan_delta", content="read then answer"),
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
async def test_sync_provider_bridge_is_bounded_by_consumer_backpressure() -> None:
    generator = _FastStreamingGenerator()
    stream = _gateway(generator).astream_with_tools(
        stage=LLMCallStage.TOOL_DECISION,
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    assert (await stream.__anext__()).content == "x"
    await asyncio.sleep(0.05)

    assert generator.produced < 100
    await stream.aclose()


@pytest.mark.anyio
async def test_cancelling_permanently_blocked_sync_provider_returns_promptly() -> None:
    generator = _BlockingStreamingGenerator()
    stream = _gateway(generator).astream_with_tools(
        stage=LLMCallStage.TOOL_DECISION,
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )
    task = asyncio.create_task(stream.__anext__())
    try:
        await asyncio.wait_for(
            asyncio.to_thread(generator.started.wait),
            timeout=0.5,
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
    finally:
        generator.release.set()
        await stream.aclose()


@pytest.mark.anyio
async def test_streaming_fallback_parses_sdk_text_completion(
    chat_completion_factory: Callable[..., ChatCompletion],
) -> None:
    response = chat_completion_factory(
        content="provider answer that is deliberately longer than twenty bytes",
        tool_calls=None,
    )

    chunks = [
        chunk
        async for chunk in _gateway(_openai_generator_for_response(response)).astream_with_tools(
            stage=LLMCallStage.TOOL_DECISION,
            messages=[{"role": "user", "content": "hello prompt"}],
            tools=[],
        )
    ]

    assert [chunk.type for chunk in chunks] == ["text_delta", "message_stop"]
    assert chunks[0].content == (
        "provider answer that is deliberately longer than twenty bytes"
    )
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
    deltas: list[ModelEvent] = []

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
        event_sink=deltas.append,
    )

    assert result.turn.text == "stream answer"
    assert result.usage.cache_read_input_tokens == 9
    assert result.usage.usage_source == "provider"
    assert [(event.delta_kind, event.content) for event in deltas if isinstance(event, ModelOutputDelta)] == [
        ("reasoning", "inspect evidence"), ("plan", "read then answer"), ("text", "stream answer"),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("arguments", ['{"path":', '{"path":"file"}'])
async def test_streaming_length_stop_has_priority_over_tool_fragments(arguments):
    class Generator:
        def stream_with_tools(self, **kwargs):
            yield {"type": "tool_use_start", "tool_id": "frag", "tool_name": "read_file"}
            yield {"type": "tool_input_delta", "tool_id": "frag", "content": arguments}
            yield {"type": "content_block_stop", "tool_id": "frag"}
            yield {"type": "message_stop", "stop_reason": "max_tokens", "usage": LLMUsage(
                input_tokens=10, output_tokens=20, total_tokens=30, source="provider",
            )}

    result = await _gateway(Generator(), max_input_tokens=2000, model_context_tokens=4000).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION, request=_canonical_request(), provider="openai-compatible",
        supports_native_tools=True, stream=True,
    )
    assert result.turn.stop_reason is StopReason.MAX_TOKENS
    assert result.turn.tool_calls == []
    assert result.usage.total_tokens == 30


@pytest.mark.anyio
@pytest.mark.parametrize("reason", ["content_filter", "unknown", None])
async def test_streaming_unsuccessful_terminal_never_exposes_tool_calls(reason):
    class Generator:
        def stream_with_tools(self, **kwargs):
            yield {"type": "tool_use_start", "tool_id": "frag", "tool_name": "read_file"}
            yield {"type": "tool_input_delta", "tool_id": "frag", "content": '{"path":"README.md"}'}
            yield {"type": "content_block_stop", "tool_id": "frag"}
            yield {"type": "message_stop", "stop_reason": reason, "usage": LLMUsage(
                input_tokens=10, output_tokens=20, total_tokens=30, source="provider",
            )}
    result = await _gateway(Generator(), max_input_tokens=2000, model_context_tokens=4000).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION, request=_canonical_request(), provider="openai-compatible",
        supports_native_tools=True, stream=True,
    )
    assert result.turn.stop_reason is StopReason.INCOMPLETE
    assert result.turn.tool_calls == []
    assert result.turn.raw_stop_reason == (reason or "unknown")


@pytest.mark.anyio
async def test_stream_without_terminal_does_not_promote_tools_to_success():
    class Generator:
        def stream_with_tools(self, **kwargs):
            yield {"type": "tool_use_start", "tool_id": "frag", "tool_name": "read_file"}
            yield {"type": "tool_input_delta", "tool_id": "frag", "content": '{"path":"README.md"}'}
            yield {"type": "content_block_stop", "tool_id": "frag"}
    with pytest.raises(ConnectionError, match="without a terminal stop"):
        await _gateway(Generator(), max_input_tokens=2000, model_context_tokens=4000).agenerate_model_request(
            stage=LLMCallStage.TOOL_DECISION, request=_canonical_request(), provider="openai-compatible",
            supports_native_tools=True, stream=True,
        )


@pytest.mark.anyio
@pytest.mark.parametrize("reason, expected", [
    ("length", StopReason.MAX_TOKENS), ("content_filter", StopReason.INCOMPLETE),
    ("unrecognized", StopReason.INCOMPLETE), ("tool_calls", StopReason.TOOL_USE),
])
async def test_openai_sse_terminal_reaches_gateway_without_success_coercion(reason, expected):
    from agent_runtime.modeling.chat import OpenAICompatibleChatGenerator

    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=reason,
        delta=SimpleNamespace(content="prefix", tool_calls=[SimpleNamespace(
            index=0, id="call", function=SimpleNamespace(name="read_file", arguments="{}"),
        )]),
    )], usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
    generator = OpenAICompatibleChatGenerator.__new__(OpenAICompatibleChatGenerator)
    generator.chat_model_name = "test-model"
    generator._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: iter([response]),
    )))
    result = await _gateway(generator, max_input_tokens=2000, model_context_tokens=4000).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION, request=_canonical_request(), provider="openai-compatible",
        supports_native_tools=True, stream=True,
    )
    assert result.turn.stop_reason is expected
    assert len(result.turn.tool_calls) == (1 if expected is StopReason.TOOL_USE else 0)
    assert result.usage.total_tokens == 30


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


@pytest.mark.anyio
async def test_model_events_preserve_native_same_kind_blocks_and_signature() -> None:
    class Generator:
        def stream_with_tools(self, **kwargs):
            yield {"type": "message_start", "provider_response_id": "response-native"}
            yield {"type": "content_block_start", "output_kind": "text", "block_index": 0}
            yield {"type": "text_delta", "block_index": 0, "content": "first"}
            yield {"type": "content_block_stop", "block_index": 0}
            yield {"type": "content_block_start", "output_kind": "text", "block_index": 1}
            yield {"type": "text_delta", "block_index": 1, "content": "second"}
            yield {"type": "content_block_stop", "block_index": 1}
            yield {"type": "content_block_start", "output_kind": "reasoning", "block_index": 2}
            yield {"type": "thinking_delta", "block_index": 2, "content": "summary"}
            yield {"type": "signature_delta", "block_index": 2, "content": "opaque"}
            yield {"type": "content_block_stop", "block_index": 2}
            yield {"type": "message_stop", "stop_reason": "end_turn"}

    events = []
    result = await _gateway(Generator(), max_input_tokens=2000, model_context_tokens=4000).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION, request=_canonical_request(), provider="openai-compatible",
        supports_native_tools=True, stream=True, event_sink=events.append,
    )
    assert result.provider_response_id == "response-native"
    assert [(output.output_id, output.kind, output.content) for output in result.outputs] == [
        ("block:0", "text", "first"), ("block:1", "text", "second"), ("block:2", "reasoning", "summary"),
    ]
    assert result.outputs[2].provider_data["signature"] == "opaque"
    assert result.turn.text == "firstsecond"


@pytest.mark.anyio
async def test_model_snapshot_retains_refusal_and_provider_identity_without_sink() -> None:
    response = SimpleNamespace(id="response-refusal", choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=None, refusal="I cannot do that.", tool_calls=[]),
    )], usage={"prompt_tokens": 10, "completion_tokens": 2})
    gateway = _gateway(_openai_generator_for_response(response), max_input_tokens=2000, model_context_tokens=4000)
    result = await gateway.agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION, request=_canonical_request(), provider="openai-compatible",
        supports_native_tools=True,
    )
    assert result.provider_response_id == "response-refusal"
    assert result.outputs[0].kind == "refusal"
    assert result.outputs[0].content == "I cannot do that."
    assert result.turn.text == "I cannot do that."


@pytest.mark.anyio
@pytest.mark.parametrize("stop_reason", ["tool_use", "max_tokens"])
async def test_native_tool_output_ids_route_interleaved_call_and_block_references(stop_reason) -> None:
    class Generator:
        def stream_with_tools(self, **kwargs):
            yield {
                "type": "tool_use_start", "output_id": "native-one", "tool_id": "call-one",
                "tool_name": "read_file", "block_index": 3,
            }
            yield {
                "type": "tool_use_start", "output_id": "native-two", "tool_id": "call-two",
                "tool_name": "read_file", "block_index": 7,
            }
            yield {"type": "tool_input_delta", "tool_id": "call-one", "content": '{"path":'}
            yield {"type": "tool_input_delta", "block_index": 7, "content": '{"path":"two.py"}'}
            yield {"type": "tool_input_delta", "block_index": 3, "content": '"one.py"}'}
            yield {"type": "content_block_stop", "tool_id": "call-two"}
            yield {"type": "content_block_stop", "tool_id": "call-one"}
            yield {"type": "message_stop", "stop_reason": stop_reason}

    events = []
    result = await _gateway(Generator(), max_input_tokens=2000, model_context_tokens=4000).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION, request=_canonical_request(), provider="openai-compatible",
        supports_native_tools=True, stream=True, event_sink=events.append,
    )
    assert [(output.output_id, output.tool_call_id, output.arguments["path"]) for output in result.outputs] == [
        ("native-one", "call-one", "one.py"), ("native-two", "call-two", "two.py"),
    ]
    assert [(event.output_id, event.content) for event in events if isinstance(event, ModelOutputDelta)] == [
        ("native-one", '{"path":'), ("native-two", '{"path":"two.py"}'), ("native-one", '"one.py"}'),
    ]
    expected_calls = ["call-one", "call-two"] if stop_reason == "tool_use" else []
    assert [call.id for call in result.turn.tool_calls] == expected_calls


@pytest.mark.anyio
async def test_native_tool_identity_references_cannot_point_at_different_outputs() -> None:
    class Generator:
        def stream_with_tools(self, **kwargs):
            yield {"type": "tool_use_start", "output_id": "native-one", "tool_id": "call-one", "tool_name": "read_file"}
            yield {"type": "tool_use_start", "output_id": "native-two", "tool_id": "call-two", "tool_name": "read_file"}
            yield {"type": "tool_input_delta", "output_id": "native-one", "tool_id": "call-two", "content": "{}"}
            yield {"type": "message_stop", "stop_reason": "tool_use"}

    with pytest.raises(ValueError, match="identity references disagree"):
        await _gateway(Generator(), max_input_tokens=2000, model_context_tokens=4000).agenerate_model_request(
            stage=LLMCallStage.TOOL_DECISION, request=_canonical_request(), provider="openai-compatible",
            supports_native_tools=True, stream=True,
        )


def test_openai_sse_reports_response_start_once_before_content() -> None:
    from agent_runtime.modeling.chat import OpenAICompatibleChatGenerator

    def chunk(*, content=None, finish_reason=None, choices=True):
        return SimpleNamespace(
            id="response-before-content",
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)]
            if choices else [],
            usage=None,
        )

    responses = [chunk(choices=False), chunk(), chunk(content="answer"), chunk(finish_reason="stop")]
    generator = OpenAICompatibleChatGenerator.__new__(OpenAICompatibleChatGenerator)
    generator.chat_model_name = "test-model"
    generator._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: iter(responses),
    )))

    chunks = list(generator.stream_with_tools(messages=[], tools=[]))

    assert chunks[0] == {"type": "response_start", "provider_response_id": "response-before-content"}
    assert [part["type"] for part in chunks] == ["response_start", "text_delta", "message_stop"]
    assert chunks[1]["content"] == "answer"


@pytest.mark.anyio
@pytest.mark.parametrize("metadata_only", [False, True])
async def test_openai_early_sse_identity_survives_gateway_runtime_failure_and_reopen(
    tmp_path, metadata_only, monkeypatch,
) -> None:
    from agent_runtime.harness import RolloutEventReader, RuntimeState, Session
    from agent_runtime.modeling.chat import OpenAICompatibleChatGenerator
    from tests.agent.harness.test_model_adapter import BoundGatewayHarnessModel, _resolved_model

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "config"))

    def responses():
        yield SimpleNamespace(
            id="response-before-content",
            choices=[] if metadata_only else [SimpleNamespace(
                delta=SimpleNamespace(role="assistant", content=None), finish_reason=None,
            )],
            usage=None,
        )
        raise ConnectionError("transport lost before the first output")

    generator = OpenAICompatibleChatGenerator.__new__(OpenAICompatibleChatGenerator)
    generator.chat_model_name = "test-model"
    generator._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: responses(),
    )))
    gateway = LLMGateway(
        generator=generator,
        token_accounting=_WordTokenAccounting(),
        model_context_tokens=8192,
        stage_budgets={LLMCallStage.AGENT_STEP: LLMStageBudget(
            max_input_tokens=4096, max_output_tokens=128, safety_margin_tokens=32,
        )},
    )
    model = BoundGatewayHarnessModel(
        model_id="test-model",
        resolved=_resolved_model(gateway=gateway, max_output_tokens=128),
        instructions=("Answer the request.",),
    )
    database = tmp_path / "run.sqlite3"
    async with await Session.open(database=database, workspace=tmp_path, model=model) as session:
        result = await session.submit("inspect")
        assert result.status == "paused"
        thread_id = session.thread_id

    async with await RuntimeState.open(database) as state:
        [operation] = await state.run_async(state.list_model_operations)
        [attempt] = await state.run_async(state.list_model_attempts, operation.operation_id)
        assert attempt.status == "unknown"
        assert attempt.provider_response_id == "response-before-content"
        replay = await state.run_async(RolloutEventReader(state).read, thread_id)
        responding = [entry.event for entry in replay if entry.event.data.get("status") == "responding"]
        assert len(responding) == 1
        assert responding[0].data["provider_response_id"] == "response-before-content"
        assert not any(entry.event_type in {"item_started", "item_completed"}
                       and entry.event.item_kind is not None
                       and entry.event.item_kind.value == "agent_message" for entry in replay)

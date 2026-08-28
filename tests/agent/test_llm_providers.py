from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

import pytest

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.llm_providers import LLMLoopModelTurnProvider
from rag.agent.core.messages import StopReason, ToolCall, ToolUseResult
from rag.agent.loop.runtime import ModelTurnEnvelope
from rag.agent.loop.state import create_loop_state
from rag.agent.streaming.events import (
    EventType,
    ItemDeltaKind,
    ItemStatus,
    TurnItemKind,
)
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    json_schema_input,
)
from rag.providers.llm_gateway import ProviderDelta, ProviderDeltaChannel
from rag.schema.llm import LLMCallStage, LLMStageBudget, LLMUsage


def _tool(name: str) -> Tool:
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


class _CanonicalGateway:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def effective_stage_budget(
        self,
        stage: LLMCallStage,
        *,
        kwargs: Mapping[str, object] | None = None,
    ) -> LLMStageBudget:
        del stage, kwargs
        return LLMStageBudget(
            max_input_tokens=32_000,
            max_output_tokens=4_096,
        )

    async def agenerate_model_request(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            turn=ToolUseResult(
                text="canonical answer",
                tool_calls=[],
                stop_reason=StopReason.END_TURN,
                raw_stop_reason="stop",
            ),
            usage=LLMUsage(
                input_tokens=12,
                output_tokens=2,
                cached_input_tokens=4,
                source="provider",
                logical_input_tokens=12,
                uncached_input_tokens=8,
                cache_read_input_tokens=4,
                cache_write_input_tokens=0,
                usage_source="provider",
                raw_provider_usage={"prompt_tokens": 12},
            ),
            provider_wire_hash="wire_provider_parity",
            serializer_revision="provider-wire-v1",
            wire_kind=str(kwargs["provider"]),
        )


class _StreamingCanonicalGateway(_CanonicalGateway):
    def __init__(
        self,
        *,
        turn: ToolUseResult,
        deltas: tuple[ProviderDelta, ...] = (),
        error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self._turn = turn
        self._deltas = deltas
        self._error = error

    async def agenerate_model_request(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        sink = kwargs["delta_sink"]
        assert callable(sink)
        for delta in self._deltas:
            await sink(delta)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            turn=self._turn,
            usage=LLMUsage(
                input_tokens=2,
                output_tokens=2,
                source="provider",
                logical_input_tokens=2,
                uncached_input_tokens=2,
                cache_read_input_tokens=0,
                cache_write_input_tokens=0,
                usage_source="provider",
            ),
            provider_wire_hash="wire_stream",
            serializer_revision="provider-wire-v1",
            wire_kind="openai",
        )


class _CollectingSink:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def emit(self, event: object) -> None:
        self.events.append(event)


def _state():
    return create_loop_state(
        current_message="Inspect README.md.",
        run_config=AgentRunConfig(
            turn_id="provider-parity",
        ),
    )


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Use canonical tools.",
        allowed_tools=["list_files", "read_file"],
    )


class TestLLMLoopModelTurnProviderBasic:
    """Basic sanity checks for LLMLoopModelTurnProvider."""

    def test_module_exports_expected_symbols(self) -> None:
        """Verify the module exports the expected public API after cleanup."""
        import rag.agent.core.llm_providers as m

        assert hasattr(m, "LLMLoopModelTurnProvider")
        assert hasattr(m, "LoopModelDecision")
        assert hasattr(m, "parse_loop_model_turn")
        assert hasattr(m, "create_loop_model_turn_provider")

    def test_retrieval_hint_code_is_removed(self) -> None:
        """Verify LLMRetrievalHintProvider no longer exists in the module."""
        import rag.agent.core.llm_providers as m

        assert not hasattr(m, "LLMRetrievalHintProvider")
        assert not hasattr(m, "create_default_providers")
        assert not hasattr(m, "RetrievalHintDecision")
        assert not hasattr(m, "_extract_quoted_terms")
        assert not hasattr(m, "_validate_retrieval_signals")


@pytest.mark.anyio
async def test_all_supported_providers_receive_the_same_canonical_request() -> None:
    snapshot = MappingProxyType(
        {
            "list_files": _tool("list_files"),
            "read_file": _tool("read_file"),
        }
    )
    captured = []
    envelopes = []

    for provider_name, supports_native_tools in (
        ("openai-compatible", True),
        ("mlx", False),
        ("ollama", False),
    ):
        gateway = _CanonicalGateway()
        provider = LLMLoopModelTurnProvider(
            gateway,
            model="test-model",
            provider=provider_name,
            supports_native_tools=supports_native_tools,
            registry_snapshot=snapshot,
            resident_tool_names=("list_files", "read_file"),
        )
        envelope = await provider.next_turn(
            _state(),
            definition=_definition(),
            budget_remaining=10_000,
        )
        assert isinstance(envelope, ModelTurnEnvelope)
        captured.append(gateway.calls[0]["request"])
        envelopes.append(envelope)

    assert [request.exposed_tool_names for request in captured] == [
        ("list_files", "read_file"),
        ("list_files", "read_file"),
        ("list_files", "read_file"),
    ]
    assert len({request.request_id for request in captured}) == 1
    assert len({request.prompt_revision for request in captured}) == 1
    assert len({request.toolset_revision for request in captured}) == 1
    assert all(envelope.model_call_record.request_id == captured[0].request_id for envelope in envelopes)
    assert all(envelope.draft.final_answer == "canonical answer" for envelope in envelopes)


@pytest.mark.anyio
async def test_model_channels_emit_complete_canonical_item_lifecycles() -> None:
    gateway = _StreamingCanonicalGateway(
        turn=ToolUseResult(
            text="hello",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            raw_stop_reason="stop",
        ),
        deltas=(
            ProviderDelta(ProviderDeltaChannel.REASONING, "check"),
            ProviderDelta(ProviderDeltaChannel.PLAN, "read"),
            ProviderDelta(ProviderDeltaChannel.TEXT, "hel"),
            ProviderDelta(ProviderDeltaChannel.TEXT, "lo"),
        ),
    )
    sink = _CollectingSink()
    provider = LLMLoopModelTurnProvider(
        gateway,
        model="test-model",
        provider="openai-compatible",
        supports_native_tools=True,
        registry_snapshot=MappingProxyType({}),
        resident_tool_names=(),
        stream_sink=sink,
    )

    await provider.next_turn(
        _state(), definition=_definition(), budget_remaining=10_000
    )

    events = sink.events
    assert [event.type for event in events] == [
        EventType.ITEM_STARTED,
        EventType.ITEM_DELTA,
        EventType.ITEM_STARTED,
        EventType.ITEM_DELTA,
        EventType.ITEM_STARTED,
        EventType.ITEM_DELTA,
        EventType.ITEM_DELTA,
        EventType.ITEM_COMPLETED,
        EventType.ITEM_COMPLETED,
        EventType.ITEM_COMPLETED,
    ]
    assert [
        event.item_kind
        for event in events
        if event.type is EventType.ITEM_STARTED
    ] == [
        TurnItemKind.REASONING,
        TurnItemKind.PLAN,
        TurnItemKind.AGENT_MESSAGE,
    ]
    assert [
        event.delta_kind
        for event in events
        if event.type is EventType.ITEM_DELTA
    ] == [
        ItemDeltaKind.REASONING,
        ItemDeltaKind.PLAN,
        ItemDeltaKind.TEXT,
        ItemDeltaKind.TEXT,
    ]
    completed = [
        event for event in events if event.type is EventType.ITEM_COMPLETED
    ]
    assert [(event.item_kind, event.status) for event in completed] == [
        (TurnItemKind.REASONING, ItemStatus.SUCCESS),
        (TurnItemKind.PLAN, ItemStatus.SUCCESS),
        (TurnItemKind.AGENT_MESSAGE, ItemStatus.SUCCESS),
    ]
    assert completed[0].data == {"content": "check"}
    assert completed[1].data == {"content": "read"}
    assert completed[2].data == {"content": "hello", "tool_calls": ()}


@pytest.mark.anyio
async def test_tool_only_response_still_completes_agent_message_item() -> None:
    gateway = _StreamingCanonicalGateway(
        turn=ToolUseResult(
            text="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="read_file",
                    input={"path": "README.md"},
                )
            ],
            stop_reason=StopReason.TOOL_USE,
            raw_stop_reason="tool_calls",
        )
    )
    sink = _CollectingSink()
    provider = LLMLoopModelTurnProvider(
        gateway,
        model="test-model",
        provider="openai-compatible",
        supports_native_tools=True,
        registry_snapshot=MappingProxyType({}),
        resident_tool_names=(),
        stream_sink=sink,
    )

    await provider.next_turn(
        _state(), definition=_definition(), budget_remaining=10_000
    )

    assert [event.type for event in sink.events] == [
        EventType.ITEM_STARTED,
        EventType.ITEM_COMPLETED,
    ]
    assert sink.events[-1].data == {
        "content": "",
        "tool_calls": (
            {
                "id": "call-1",
                "name": "read_file",
                "arguments": {"path": "README.md"},
            },
        ),
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (RuntimeError("provider failed"), ItemStatus.FAILED),
        (asyncio.CancelledError(), ItemStatus.CANCELLED),
    ],
)
async def test_partial_model_items_close_on_error_or_cancellation(
    error: BaseException,
    expected_status: ItemStatus,
) -> None:
    gateway = _StreamingCanonicalGateway(
        turn=ToolUseResult(
            text="",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            raw_stop_reason="stop",
        ),
        deltas=(ProviderDelta(ProviderDeltaChannel.TEXT, "partial"),),
        error=error,
    )
    sink = _CollectingSink()
    provider = LLMLoopModelTurnProvider(
        gateway,
        model="test-model",
        provider="openai-compatible",
        supports_native_tools=True,
        registry_snapshot=MappingProxyType({}),
        resident_tool_names=(),
        stream_sink=sink,
    )

    with pytest.raises(type(error)):
        await provider.next_turn(
            _state(), definition=_definition(), budget_remaining=10_000
        )

    assert [event.type for event in sink.events] == [
        EventType.ITEM_STARTED,
        EventType.ITEM_DELTA,
        EventType.ITEM_COMPLETED,
    ]
    assert sink.events[-1].status is expected_status

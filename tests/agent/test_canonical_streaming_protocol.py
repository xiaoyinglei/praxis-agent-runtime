from __future__ import annotations

import asyncio
from dataclasses import fields
from types import MappingProxyType, SimpleNamespace

import pytest

from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.llm_providers import LLMLoopModelTurnProvider
from agent_runtime.core.messages import StopReason, ToolUseResult
from agent_runtime.modeling.contracts import LLMStageBudget, LLMUsage
from agent_runtime.modeling.gateway import ProviderDelta, ProviderDeltaChannel
from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract
from agent_runtime.service import AgentRunRequest, AgentService
from agent_runtime.streaming import events
from agent_runtime.streaming.sink import (
    DurableStreamEventSink,
    LegacyStreamProjectionSink,
    QueueStreamEventSink,
)
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.turns import RuntimeBinding, TurnStore


def test_v2_protocol_types_and_wire_values_are_public() -> None:
    turn_item_kind = getattr(events, "TurnItemKind", None)
    delta_kind = getattr(events, "ItemDeltaKind", None)
    item_status = getattr(events, "ItemStatus", None)

    assert turn_item_kind is not None
    assert delta_kind is not None
    assert item_status is not None
    assert {member.value for member in turn_item_kind} == {
        "agent_message",
        "reasoning",
        "plan",
        "tool",
        "command",
        "reconciliation",
        "legacy_message",
    }
    assert {member.value for member in delta_kind} == {
        "text",
        "reasoning",
        "plan",
        "tool_progress",
        "command_stdout",
        "command_stderr",
    }
    assert {member.value for member in item_status} == {
        "success",
        "failed",
        "cancelled",
        "outcome_unknown",
    }
    assert events.EventType.TURN_STARTED.value == "turn_started"
    assert events.EventType.TURN_PAUSED.value == "turn_paused"
    assert events.EventType.TURN_RESUMED.value == "turn_resumed"
    assert (
        events.EventType.TURN_CANCELLATION_REQUESTED.value
        == "turn_cancellation_requested"
    )
    assert events.EventType.TURN_COMPLETED.value == "turn_completed"
    assert events.EventType.TURN_ABORTED.value == "turn_aborted"
    assert events.EventType.ITEM_STARTED.value == "item_started"
    assert events.EventType.ITEM_DELTA.value == "item_delta"
    assert events.EventType.ITEM_COMPLETED.value == "item_completed"


def test_v2_stream_event_has_explicit_item_contract() -> None:
    assert tuple(field.name for field in fields(events.StreamEvent)) == (
        "protocol_version",
        "type",
        "turn_id",
        "item_id",
        "item_kind",
        "delta_kind",
        "status",
        "iteration",
        "sequence",
        "timestamp_ms",
        "data",
        "error",
        "parent_item_id",
    )

    started = events.item_started(
        turn_id="turn-1",
        item_id="agent:turn-1:1",
        item_kind=events.TurnItemKind.AGENT_MESSAGE,
        iteration=1,
        data={"phase": "answer"},
    )
    delta = events.item_delta(
        turn_id="turn-1",
        item_id=started.item_id,
        item_kind=started.item_kind,
        delta_kind=events.ItemDeltaKind.TEXT,
        delta="hel",
        iteration=1,
    )
    completed = events.item_completed(
        turn_id="turn-1",
        item_id=started.item_id,
        item_kind=started.item_kind,
        status=events.ItemStatus.SUCCESS,
        iteration=1,
        data={"content": "hello", "tool_calls": []},
    )

    assert started.protocol_version == 2
    assert started.type is events.EventType.ITEM_STARTED
    assert delta.type is events.EventType.ITEM_DELTA
    assert delta.data == {"delta": "hel"}
    assert completed.type is events.EventType.ITEM_COMPLETED
    assert completed.status is events.ItemStatus.SUCCESS
    assert {started.turn_id, delta.turn_id, completed.turn_id} == {"turn-1"}
    assert {started.item_id, delta.item_id, completed.item_id} == {
        "agent:turn-1:1"
    }


def test_v2_stream_event_rejects_invalid_field_combinations() -> None:
    with pytest.raises(ValueError, match="item_id"):
        events.StreamEvent(
            type=events.EventType.ITEM_DELTA,
            turn_id="turn-1",
            item_kind=events.TurnItemKind.AGENT_MESSAGE,
            delta_kind=events.ItemDeltaKind.TEXT,
            data={"delta": "x"},
        )

    with pytest.raises(ValueError, match="error"):
        events.item_completed(
            turn_id="turn-1",
            item_id="tool:1",
            item_kind=events.TurnItemKind.TOOL,
            status=events.ItemStatus.FAILED,
            data={},
        )

    with pytest.raises(ValueError, match="forbidden"):
        events.StreamEvent(
            type=events.EventType.TURN_STARTED,
            turn_id="turn-1",
            item_id="not-allowed",
        )


def test_v2_turn_lifecycle_events_are_not_item_scoped() -> None:
    started = events.turn_started("turn-1")
    paused = events.turn_paused("turn-1", reason="approval")
    resumed = events.turn_resumed("turn-1")
    cancelling = events.turn_cancellation_requested("turn-1")
    aborted = events.turn_aborted("turn-1", reason="consumer_closed")

    assert [event.type for event in (started, paused, resumed, cancelling, aborted)] == [
        events.EventType.TURN_STARTED,
        events.EventType.TURN_PAUSED,
        events.EventType.TURN_RESUMED,
        events.EventType.TURN_CANCELLATION_REQUESTED,
        events.EventType.TURN_ABORTED,
    ]
    assert all(event.item_id is None for event in (started, paused, resumed, cancelling, aborted))


class _LiveSink:
    def __init__(self, *, fail: bool = False) -> None:
        self.events = []
        self.fail = fail

    async def emit(self, event: events.StreamEvent) -> None:
        self.events.append(event)
        if self.fail:
            raise RuntimeError("live consumer disconnected")


@pytest.mark.anyio
async def test_durable_sink_commits_completed_agent_item_before_live_delivery(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "turns.sqlite3")
    store.begin_turn("hello", RuntimeBinding(), turn_id="turn-1")
    live = _LiveSink()
    sink = DurableStreamEventSink(turn_store=store, live_sink=live)
    started = events.item_started(
        turn_id="turn-1",
        item_id="agent:turn-1:1",
        item_kind=events.TurnItemKind.AGENT_MESSAGE,
        iteration=1,
    )
    completed = events.item_completed(
        turn_id="turn-1",
        item_id="agent:turn-1:1",
        item_kind=events.TurnItemKind.AGENT_MESSAGE,
        status=events.ItemStatus.SUCCESS,
        iteration=1,
        data={"content": "world", "tool_calls": ()},
    )

    await sink.emit(started)
    await sink.emit(
        events.item_delta(
            turn_id="turn-1",
            item_id="agent:turn-1:1",
            item_kind=events.TurnItemKind.AGENT_MESSAGE,
            delta_kind=events.ItemDeltaKind.TEXT,
            delta="world",
            iteration=1,
        )
    )
    await sink.emit(completed)

    assert [record.event.type for record in store.replay_turn_events("turn-1")] == [
        events.EventType.TURN_STARTED,
        events.EventType.ITEM_COMPLETED,
    ]
    assert [(message.role, message.content) for message in store.turn_history("turn-1")] == [
        ("user", "hello"),
        ("assistant", "world"),
    ]
    assert live.events[-1] == completed
    store.close()


@pytest.mark.anyio
async def test_live_sink_failure_cannot_rollback_completed_item(tmp_path) -> None:
    store = TurnStore(tmp_path / "turns.sqlite3")
    store.begin_turn("hello", RuntimeBinding(), turn_id="turn-1")
    sink = DurableStreamEventSink(
        turn_store=store,
        live_sink=_LiveSink(fail=True),
    )
    completed = events.item_completed(
        turn_id="turn-1",
        item_id="agent:turn-1:1",
        item_kind=events.TurnItemKind.AGENT_MESSAGE,
        status=events.ItemStatus.SUCCESS,
        iteration=1,
        data={"content": "world", "tool_calls": ()},
    )

    with pytest.raises(RuntimeError, match="disconnected"):
        await sink.emit(completed)

    replayed = store.replay_turn_events("turn-1")[-1].event
    assert replayed.type is events.EventType.ITEM_COMPLETED
    assert replayed.item_id == completed.item_id
    assert replayed.data == {"content": "world", "tool_calls": []}
    assert store.turn_history("turn-1")[-1].content == "world"
    store.close()


@pytest.mark.anyio
async def test_synthetic_cancel_is_idempotent_with_producer_cancel_completion(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "turns.sqlite3")
    store.begin_turn("hello", RuntimeBinding(), turn_id="turn-cancel-item")
    sink = DurableStreamEventSink(turn_store=store, live_sink=_LiveSink())
    started = events.item_started(
        turn_id="turn-cancel-item",
        item_id="agent:turn-cancel-item:1",
        item_kind=events.TurnItemKind.AGENT_MESSAGE,
        iteration=1,
    )
    await sink.emit(started)
    await sink.emit(
        events.item_delta(
            turn_id=started.turn_id,
            item_id=started.item_id,
            item_kind=started.item_kind,
            delta_kind=events.ItemDeltaKind.TEXT,
            delta="partial",
            iteration=1,
        )
    )

    await sink.close_open_items()
    await sink.close_open_items()
    await sink.emit(
        events.item_completed(
            turn_id=started.turn_id,
            item_id=started.item_id,
            item_kind=started.item_kind,
            status=events.ItemStatus.CANCELLED,
            data={"content": "partial"},
            iteration=1,
        )
    )

    replayed = store.replay_turn_events("turn-cancel-item")
    assert [record.event.type for record in replayed] == [
        events.EventType.TURN_STARTED,
        events.EventType.ITEM_COMPLETED,
    ]
    store.close()


@pytest.mark.anyio
async def test_legacy_projection_emits_only_legacy_wire_events() -> None:
    target = _LiveSink()
    sink = LegacyStreamProjectionSink(target)
    canonical = (
        events.turn_started("turn-legacy"),
        events.item_started(
            turn_id="turn-legacy",
            item_id="agent:turn-legacy:1",
            item_kind=events.TurnItemKind.AGENT_MESSAGE,
            iteration=1,
        ),
        events.item_delta(
            turn_id="turn-legacy",
            item_id="agent:turn-legacy:1",
            item_kind=events.TurnItemKind.AGENT_MESSAGE,
            delta_kind=events.ItemDeltaKind.TEXT,
            delta="hello",
            iteration=1,
        ),
        events.item_completed(
            turn_id="turn-legacy",
            item_id="agent:turn-legacy:1",
            item_kind=events.TurnItemKind.AGENT_MESSAGE,
            status=events.ItemStatus.SUCCESS,
            data={"content": "hello", "tool_calls": []},
            iteration=1,
        ),
        events.turn_completed("turn-legacy"),
    )

    for event in canonical:
        await sink.emit(event)

    assert [event.type for event in target.events] == [
        events.EventType.TURN_START,
        events.EventType.TEXT_DELTA,
        events.EventType.LOOP_END,
    ]
    assert not any(
        event.type
        in {
            events.EventType.TURN_STARTED,
            events.EventType.ITEM_STARTED,
            events.EventType.ITEM_DELTA,
            events.EventType.ITEM_COMPLETED,
            events.EventType.TURN_COMPLETED,
        }
        for event in target.events
    )


@pytest.mark.anyio
async def test_legacy_projection_preserves_tool_id_result_details_and_plan_event() -> None:
    target = _LiveSink()
    sink = LegacyStreamProjectionSink(target)
    await sink.emit(
        events.item_started(
            turn_id="turn-legacy",
            item_id="tool:turn-legacy:call-1:1",
            item_kind=events.TurnItemKind.TOOL,
            data={
                "tool_name": "apply_patch",
                "tool_call_id": "call-1",
                "input_preview": "patch='***'",
            },
        )
    )
    await sink.emit(
        events.item_completed(
            turn_id="turn-legacy",
            item_id="tool:turn-legacy:call-1:1",
            item_kind=events.TurnItemKind.TOOL,
            status=events.ItemStatus.SUCCESS,
            data={
                "result": {
                    "tool_name": "apply_patch",
                    "content": ({"type": "text", "data": {"text": "patched"}},),
                    "structured_content": None,
                    "metadata": {
                        "file_path": "a.py",
                        "diff": "-old\n+new",
                        "diff_truncated": False,
                    },
                }
            },
        )
    )
    await sink.emit(
        events.item_completed(
            turn_id="turn-legacy",
            item_id="update_plan:turn-legacy:1",
            item_kind=events.TurnItemKind.PLAN,
            status=events.ItemStatus.SUCCESS,
            data={"plan": {"revision": 1}, "event": {"event_type": "llm_update"}},
        )
    )

    assert [event.type for event in target.events] == [
        events.EventType.TOOL_USE_START,
        events.EventType.TOOL_USE_RESULT,
        events.EventType.PLAN_UPDATED,
    ]
    assert target.events[0].data["tool_id"] == "call-1"
    assert target.events[1].data["result"] == "patched"
    assert target.events[1].data["details"] == {
        "file_path": "a.py",
        "diff": "-old\n+new",
        "diff_truncated": False,
    }
    assert target.events[2].data["event"] == {"event_type": "llm_update"}


class _EndToEndGateway:
    def __init__(self) -> None:
        self.token_accounting = TokenAccountingService(
            TokenizerContract(
                embedding_model_name="canonical-stream-test",
                tokenizer_model_name="canonical-stream-test",
                chunking_tokenizer_model_name="canonical-stream-test",
                tokenizer_backend="simple",
                max_context_tokens=8_000,
                prompt_reserved_tokens=256,
                local_files_only=True,
            )
        )

    def effective_stage_budget(self, stage, *, kwargs=None) -> LLMStageBudget:
        del stage, kwargs
        return LLMStageBudget(max_input_tokens=8_000, max_output_tokens=1_000)

    async def agenerate_model_request(self, **kwargs):
        await kwargs["delta_sink"](
            ProviderDelta(ProviderDeltaChannel.TEXT, "durable answer")
        )
        return SimpleNamespace(
            turn=ToolUseResult(
                text="durable answer",
                tool_calls=[],
                stop_reason=StopReason.END_TURN,
                raw_stop_reason="stop",
            ),
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
            provider_wire_hash="wire_e2e",
            serializer_revision="provider-wire-v1",
            wire_kind="openai",
        )


class _BlockingEndToEndGateway(_EndToEndGateway):
    def __init__(self) -> None:
        super().__init__()
        self.block = asyncio.Event()

    async def agenerate_model_request(self, **kwargs):
        await kwargs["delta_sink"](
            ProviderDelta(ProviderDeltaChannel.TEXT, "partial")
        )
        await self.block.wait()
        raise AssertionError("blocking gateway should be cancelled")


@pytest.mark.anyio
async def test_service_streams_turn_item_delta_completion_and_durable_history(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "turns.sqlite3")
    definition = AgentRuntimePolicy.test_factory(
        system_prompt="Answer directly.",
        allowed_tools=[],
    )
    provider = LLMLoopModelTurnProvider(
        _EndToEndGateway(),
        model="test-model",
        provider="openai-compatible",
        supports_native_tools=True,
        registry_snapshot=MappingProxyType({}),
        resident_tool_names=(),
    )
    service = AgentService(
        definition=definition,
        tool_registry=ToolRegistry(),
        model_turn_provider=provider,
        turn_store=store,
        runtime_binding=RuntimeBinding(workspace_path=str(tmp_path)),
    )

    streamed = [
        event
        async for event in service.run_streaming(
            AgentRunRequest(message="hello", turn_id="turn-e2e")
        )
    ]

    assert streamed[0].type is events.EventType.TURN_STARTED
    assert streamed[-1].type is events.EventType.TURN_COMPLETED
    assert [
        event.type
        for event in streamed
        if event.type
        in {
            events.EventType.ITEM_STARTED,
            events.EventType.ITEM_DELTA,
            events.EventType.ITEM_COMPLETED,
        }
    ] == [
        events.EventType.ITEM_STARTED,
        events.EventType.ITEM_DELTA,
        events.EventType.ITEM_COMPLETED,
    ]
    assert [record.event.type for record in store.replay_turn_events("turn-e2e")] == [
        events.EventType.TURN_STARTED,
        events.EventType.ITEM_COMPLETED,
        events.EventType.TURN_COMPLETED,
    ]
    assert [(message.role, message.content) for message in store.turn_history("turn-e2e")] == [
        ("user", "hello"),
        ("assistant", "durable answer"),
    ]
    await service.aclose()
    store.close()


@pytest.mark.anyio
async def test_nonstream_event_sink_receives_the_same_canonical_protocol(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "turns.sqlite3")
    live = _LiveSink()
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(
            system_prompt="Answer directly.",
            allowed_tools=[],
        ),
        tool_registry=ToolRegistry(),
        model_turn_provider=LLMLoopModelTurnProvider(
            _EndToEndGateway(),
            model="test-model",
            provider="openai-compatible",
            supports_native_tools=True,
            registry_snapshot=MappingProxyType({}),
            resident_tool_names=(),
        ),
        stream_sink=live,
        turn_store=store,
        runtime_binding=RuntimeBinding(workspace_path=str(tmp_path)),
    )

    result = await service.run(
        AgentRunRequest(message="hello", turn_id="turn-event-sink")
    )

    assert result.status == "done"
    assert [event.type for event in live.events] == [
        events.EventType.TURN_STARTED,
        events.EventType.ITEM_STARTED,
        events.EventType.ITEM_DELTA,
        events.EventType.ITEM_COMPLETED,
        events.EventType.TURN_COMPLETED,
    ]
    assert live.events[0] == store.replay_turn_events("turn-event-sink")[0].event
    assert live.events[-1] == store.replay_turn_events("turn-event-sink")[-1].event
    await service.aclose()
    store.close()


@pytest.mark.anyio
async def test_nonstream_run_without_live_sink_still_persists_agent_item(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "turns.sqlite3")
    service = AgentService(
        definition=AgentRuntimePolicy.test_factory(
            system_prompt="Answer directly.",
            allowed_tools=[],
        ),
        tool_registry=ToolRegistry(),
        model_turn_provider=LLMLoopModelTurnProvider(
            _EndToEndGateway(),
            model="test-model",
            provider="openai-compatible",
            supports_native_tools=True,
            registry_snapshot=MappingProxyType({}),
            resident_tool_names=(),
        ),
        turn_store=store,
        runtime_binding=RuntimeBinding(workspace_path=str(tmp_path)),
    )

    result = await service.run(
        AgentRunRequest(message="hello", turn_id="turn-no-live-sink")
    )

    assert result.status == "done"
    assert [
        record.event.type
        for record in store.replay_turn_events("turn-no-live-sink")
    ] == [
        events.EventType.TURN_STARTED,
        events.EventType.ITEM_COMPLETED,
        events.EventType.TURN_COMPLETED,
    ]
    await service.aclose()
    store.close()


@pytest.mark.anyio
async def test_full_public_queue_can_close_without_waiting_for_capacity() -> None:
    sink = QueueStreamEventSink(maxsize=1)
    await sink.emit(events.turn_started("turn-full"))
    blocked_emit = asyncio.create_task(
        sink.emit(events.turn_paused("turn-full", reason="backpressure"))
    )
    await asyncio.sleep(0)
    assert not blocked_emit.done()

    await asyncio.wait_for(sink.close(), timeout=0.1)
    with pytest.raises(RuntimeError, match="closed during emit"):
        await asyncio.wait_for(blocked_emit, timeout=0.1)

    replayed = [event async for event in sink.stream()]
    assert [event.type for event in replayed] == [events.EventType.TURN_STARTED]


@pytest.mark.anyio
async def test_stream_cancellation_closes_partial_item_before_aborted_turn(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "turns.sqlite3")
    definition = AgentRuntimePolicy.test_factory(
        system_prompt="Answer directly.",
        allowed_tools=[],
    )
    provider = LLMLoopModelTurnProvider(
        _BlockingEndToEndGateway(),
        model="test-model",
        provider="openai-compatible",
        supports_native_tools=True,
        registry_snapshot=MappingProxyType({}),
        resident_tool_names=(),
    )
    service = AgentService(
        definition=definition,
        tool_registry=ToolRegistry(),
        model_turn_provider=provider,
        turn_store=store,
        runtime_binding=RuntimeBinding(workspace_path=str(tmp_path)),
    )
    stream = service.run_streaming(
        AgentRunRequest(message="hello", turn_id="turn-cancel")
    )

    assert (await anext(stream)).type is events.EventType.TURN_STARTED
    assert (await anext(stream)).type is events.EventType.ITEM_STARTED
    assert (await anext(stream)).type is events.EventType.ITEM_DELTA
    await asyncio.wait_for(stream.aclose(), timeout=0.5)

    replayed = store.replay_turn_events("turn-cancel")
    assert [record.event.type for record in replayed] == [
        events.EventType.TURN_STARTED,
        events.EventType.ITEM_COMPLETED,
        events.EventType.TURN_CANCELLATION_REQUESTED,
        events.EventType.TURN_ABORTED,
    ]
    assert replayed[1].event.status is events.ItemStatus.CANCELLED
    await service.aclose()
    store.close()

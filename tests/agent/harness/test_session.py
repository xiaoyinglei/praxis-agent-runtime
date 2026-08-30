from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_runtime.core.llm_registry import ResolvedModel
from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    GatewayHarnessModel,
    HarnessMessage,
    HarnessModelDeltaSink,
    HarnessModelRequest,
    HarnessModelResponse,
    HarnessToolCall,
    ModelDispatchPreflightError,
    PreparedModelCall,
    RolloutContextManager,
    RolloutStore,
    RuntimeComposition,
    Session,
)
from agent_runtime.modeling.contracts import LLMCallStage, LLMStageBudget
from agent_runtime.modeling.gateway import LLMGateway, StreamChunk
from agent_runtime.streaming.events import EventType, ItemStatus, TurnItemKind
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    json_schema_input,
)


class InspectingModel:
    def __init__(self, store: RolloutStore) -> None:
        self._store = store
        self.prepared_requests: list[HarnessModelRequest] = []

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        self.prepared_requests.append(request)
        return PreparedModelCall(
            request_hash="request-hash",
            context_hash="context-hash",
            tool_hash="tool-hash",
            wire_hash="wire-hash",
            request_ref={"message_count": len(request.messages)},
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        [operation] = self._store.list_model_operations()
        [attempt] = self._store.list_model_attempts(operation.operation_id)
        assert operation.status == "dispatched"
        assert attempt.status == "dispatched"
        assert attempt.claim_owner == "turn-worker-a"
        assert operation.request_hash == prepared.request_hash
        return HarnessModelResponse(
            text="model answer",
            provider_response_id="response-1",
            usage={"input_tokens": 5, "output_tokens": 2},
        )


class InspectingCompletionGate:
    def __init__(self, store: RolloutStore) -> None:
        self._store = store
        self.evaluated = False

    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        self.evaluated = True
        assert any(item.kind == "model_response" for item in self._store.list_items(proposal.turn_id))
        assert any(item.kind == "final_proposal" for item in self._store.list_items(proposal.turn_id))
        return CompletionDecision(action="accept", reason="plain answer is complete")


class OverBudgetAnswerModel(InspectingModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        return HarnessModelResponse(
            text="must not be accepted",
            provider_response_id="response-over-budget",
            usage={"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
        )


class PreflightRejectedModel(InspectingModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        raise ModelDispatchPreflightError("provider call exceeds remaining budget")


class IncompleteResponseModel(InspectingModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        return HarnessModelResponse(
            text="partial answer must not be accepted",
            provider_response_id="response-incomplete",
            usage={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
            status="incomplete",
            incomplete_reason="max_output_tokens",
        )


class FailingDispatchModel(InspectingModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        raise ConnectionError("provider connection was lost after dispatch")


class RejectedDispatchModel(InspectingModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        raise RuntimeError("provider rejected a validly delivered request")


class RetryThenAnswerModel(InspectingModel):
    def __init__(self, store: RolloutStore) -> None:
        super().__init__(store)
        self.dispatch_count = 0

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        self.dispatch_count += 1
        if self.dispatch_count == 1:
            raise ConnectionError("first dispatch outcome is unknown")
        return HarnessModelResponse(
            text="recovered model answer",
            provider_response_id="response-recovered",
            usage={"input_tokens": 4, "output_tokens": 2},
        )


class ReviseAfterFeedbackModel(InspectingModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        request = self.prepared_requests[-1]
        if len(self.prepared_requests) == 1:
            return HarnessModelResponse(
                text="unverified draft",
                provider_response_id="response-draft",
                usage={"input_tokens": 3, "output_tokens": 2},
            )
        assert request.messages[-1] == HarnessMessage(
            role="context",
            content="verification evidence is missing",
        )
        return HarnessModelResponse(
            text="verified final answer",
            provider_response_id="response-final",
            usage={"input_tokens": 6, "output_tokens": 3},
        )


class ContinueThenAcceptGate:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        self.calls += 1
        if self.calls == 1:
            return CompletionDecision(
                action="continue",
                reason="verification evidence is missing",
            )
        return CompletionDecision(action="accept", reason="evidence is now complete")


class FixedDecisionGate:
    def __init__(self, action: str, reason: str) -> None:
        self.action = action
        self.reason = reason

    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        del proposal
        return CompletionDecision(  # type: ignore[arg-type]
            action=self.action,
            reason=self.reason,
        )


class PauseThenAcceptGate:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        del proposal
        self.calls += 1
        if self.calls == 1:
            return CompletionDecision(action="pause", reason="which target?")
        return CompletionDecision(action="accept", reason="target is explicit")


class AnswerAfterClarificationModel(InspectingModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        if len(self.prepared_requests) == 1:
            return HarnessModelResponse(
                text="ambiguous draft",
                provider_response_id="response-ambiguous",
                usage={"input_tokens": 3, "output_tokens": 2},
            )
        assert self.prepared_requests[-1].messages[-1] == HarnessMessage(
            role="user",
            content="target A",
        )
        return HarnessModelResponse(
            text="answer for target A",
            provider_response_id="response-target-a",
            usage={"input_tokens": 6, "output_tokens": 3},
        )


class CharacterAccounting:
    def count(self, text: str) -> int:
        return len(text)

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str:
        suffix = "..." if add_ellipsis and len(text) > token_budget else ""
        return text[: max(0, token_budget - len(suffix))] + suffix


class BlockingAfterFirstDeltaProvider:
    def __init__(self) -> None:
        self.blocked = threading.Event()
        self.release = threading.Event()

    def stream_with_tools(self, **_kwargs: object) -> Iterator[StreamChunk]:
        yield StreamChunk(type="text_delta", content="partial")
        self.blocked.set()
        self.release.wait()


class ZeroDeltaToolThenAnswerModel:
    def __init__(self) -> None:
        self.dispatch_count = 0

    def snapshot(self) -> dict[str, str]:
        return {"model_alias": "test-model"}

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        toolset_revision = toolset_revision_for_tools(request.tools)
        digest = str(request.step) * 64
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash=toolset_revision,
            wire_hash=digest,
            request_ref={
                "request_id": f"{request.turn_id}:step:{request.step}",
                "toolset_revision": toolset_revision,
                "exposed_tool_names": [tool.definition.name for tool in request.tools],
            },
        )

    async def dispatch(
        self,
        prepared: PreparedModelCall,
        *,
        delta_sink: HarnessModelDeltaSink | None = None,
    ) -> HarnessModelResponse:
        del prepared, delta_sink
        self.dispatch_count += 1
        if self.dispatch_count == 1:
            return HarnessModelResponse(
                text="",
                provider_response_id="response-tool-only",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="call-1",
                        name="read_file",
                        arguments={"path": "README.md"},
                    ),
                ),
            )
        return HarnessModelResponse(
            text="tool completed",
            provider_response_id="response-final",
            usage={"input_tokens": 4, "output_tokens": 2},
        )


class AcceptToolCompleted:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        assert proposal.answer == "tool completed"
        return CompletionDecision(action="accept", reason="tool loop completed")


def _read_file_tool() -> Tool:
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ("path",),
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name="read_file",
            description="Read a test file.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=lambda arguments: {"text": f"contents of {arguments['path']}"},
        normalize_output=lambda raw: NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data={"value": str(raw)}),),
        ),
        output_schema=None,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(),
        ),
        execution_revision="read-file-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4_096,
    )


@pytest.mark.anyio
async def test_harness_cancel_of_blocked_sync_provider_closes_item_without_join(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = BlockingAfterFirstDeltaProvider()
    gateway = LLMGateway(
        generator=provider,
        token_accounting=CharacterAccounting(),
        model_context_tokens=8_192,
        stage_budgets={
            LLMCallStage.AGENT_STEP: LLMStageBudget(
                max_input_tokens=4_096,
                max_output_tokens=256,
                safety_margin_tokens=0,
            )
        },
    )
    model = GatewayHarnessModel(
        model_alias="test-model",
        resolved=ResolvedModel(
            generator=provider,
            kwargs={"max_tokens": 256},
            gateway=gateway,
            provider="openai-compatible",
            model="provider-model",
        ),
        instructions=("Answer the user directly.",),
    )
    dispatcher = TurnEventDispatcher(capacity=8)
    stream = dispatcher.subscribe_controlling()
    try:
        with RolloutStore(tmp_path / "rollout.sqlite3") as store:
            thread = store.create_thread(workspace=workspace)
            runner = Session(
                thread_id=thread.thread_id,
                store=store,
                model=model,
                context_manager=RolloutContextManager(store),
                completion_gate=FixedDecisionGate("accept", "unused"),
                event_dispatcher=dispatcher,
            )
            running = asyncio.create_task(
                runner.run(
                    user_message="cancel a blocked sync provider",
                    binding_manifest={"model_alias": "test-model"},
                )
            )
            initial_events = [
                await asyncio.wait_for(stream.receive(), timeout=0.5)
                for _index in range(3)
            ]
            await asyncio.wait_for(
                asyncio.to_thread(provider.blocked.wait),
                timeout=0.5,
            )

            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, timeout=0.2)

            terminal_events = []
            while not stream.empty:
                terminal_events.append(stream.receive_nowait())

            [operation] = store.list_model_operations()
            [attempt] = store.list_model_attempts(operation.operation_id)
            [response_item] = [
                item
                for item in store.list_items(operation.turn_id)
                if item.kind == "model_response"
            ]

            assert [event.type for event in initial_events] == [
                EventType.TURN_STARTED,
                EventType.ITEM_STARTED,
                EventType.ITEM_DELTA,
            ]
            assert attempt.status == "unknown"
            assert response_item.status == "completed"
            assert response_item.payload["text"] == "partial"
            completed = [
                event
                for event in terminal_events
                if event.type is EventType.ITEM_COMPLETED
                and event.item_kind is TurnItemKind.AGENT_MESSAGE
            ]
            assert len(completed) == 1
            assert completed[0].status is ItemStatus.OUTCOME_UNKNOWN
            assert completed[0].data["content"] == "partial"
    finally:
        provider.release.set()


@pytest.mark.anyio
async def test_zero_text_tool_only_response_starts_then_completes_without_delta(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = ZeroDeltaToolThenAnswerModel()
    dispatcher = TurnEventDispatcher(capacity=64)
    stream = dispatcher.subscribe_controlling()
    tool = _read_file_tool()
    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=model,
        completion_gate=AcceptToolCompleted(),
        tools={tool.definition.name: tool},
    ) as runtime:
        result = await runtime.thread_manager.run(
            user_message="use one tool",
            event_dispatcher=dispatcher,
        )
        events = []
        while not stream.empty:
            events.append(stream.receive_nowait())

    tool_only_completion = next(
        event
        for event in events
        if event.type is EventType.ITEM_COMPLETED
        and event.item_kind is TurnItemKind.AGENT_MESSAGE
        and event.data["tool_calls"]
    )
    tool_only_lifecycle = [
        event for event in events if event.item_id == tool_only_completion.item_id
    ]

    assert result.answer == "tool completed"
    assert model.dispatch_count == 2
    assert [event.type for event in tool_only_lifecycle] == [
        EventType.ITEM_STARTED,
        EventType.ITEM_COMPLETED,
    ]
    assert tool_only_completion.data["content"] == ""
    tool_calls = tool_only_completion.data["tool_calls"]
    assert isinstance(tool_calls, (list, tuple))
    assert len(tool_calls) == 1


def test_session_persists_model_transaction_before_provider_io(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        model = InspectingModel(store)
        completion_gate = InspectingCompletionGate(store)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=completion_gate,
            worker_id="turn-worker-a",
        )

        result = asyncio.run(
            runner.run(
                user_message="answer plainly",
                binding_manifest={"model_alias": "test-model"},
            )
        )

        assert result.answer == "model answer"
        assert completion_gate.evaluated is True
        assert model.prepared_requests[0].messages == (HarnessMessage(role="user", content="answer plainly"),)
        assert result.thread_id == thread.thread_id
        assert store.read_turn(result.turn_id).status == "completed"
        [operation] = store.list_model_operations(result.turn_id)
        [attempt] = store.list_model_attempts(operation.operation_id)
        assert operation.status == "completed"
        assert attempt.status == "completed"
        assert attempt.provider_response_id == "response-1"
        assert attempt.usage == {"input_tokens": 5, "output_tokens": 2}
        assert [item.kind for item in store.list_items(result.turn_id)] == [
            "user_message",
            "model_request",
            "model_response",
            "final_proposal",
            "completion_decision",
            "agent_message",
        ]
        assert store.verify().valid is True


def test_single_final_response_over_frozen_token_budget_fails_the_turn(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        completion_gate = InspectingCompletionGate(store)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=OverBudgetAnswerModel(store),
            context_manager=RolloutContextManager(store),
            completion_gate=completion_gate,
        )

        result = asyncio.run(
            runner.run(
                user_message="stay within budget",
                binding_manifest={
                    "model_alias": "test-model",
                    "model_token_budget_total": 10,
                },
            )
        )

        assert result.status == "failed"
        assert result.answer is None
        assert store.read_turn(result.turn_id).status == "failed"
        assert completion_gate.evaluated is False
        assert not any(
            item.kind in {"final_proposal", "completion_decision", "agent_message"}
            for item in store.list_items(result.turn_id)
        )
        [attempt] = store.list_model_attempts(store.list_model_operations(result.turn_id)[0].operation_id)
        assert attempt.status == "completed"
        assert attempt.usage["total_tokens"] == 11
        assert store.verify().valid is True


def test_model_preflight_rejection_fails_without_an_unknown_outcome(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        model = PreflightRejectedModel(store)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=InspectingCompletionGate(store),
        )

        result = asyncio.run(
            runner.run(
                user_message="do not call above budget",
                binding_manifest={
                    "model_alias": "test-model",
                    "model_token_budget_total": 10,
                },
            )
        )

        assert result.status == "failed"
        assert model.prepared_requests[0].model_token_budget_remaining == 10
        [operation] = store.list_model_operations(result.turn_id)
        [attempt] = store.list_model_attempts(operation.operation_id)
        assert operation.status == "failed"
        assert attempt.status == "failed"
        assert not any(item.kind == "model_response" for item in store.list_items(result.turn_id))
        assert store.verify().valid is True


def test_incomplete_model_response_is_durable_failure_not_unknown(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        completion_gate = InspectingCompletionGate(store)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=IncompleteResponseModel(store),
            context_manager=RolloutContextManager(store),
            completion_gate=completion_gate,
        )

        result = asyncio.run(
            runner.run(
                user_message="return a complete response",
                binding_manifest={"model_alias": "test-model"},
            )
        )

        assert result.status == "failed"
        assert result.answer is None
        [operation] = store.list_model_operations(result.turn_id)
        [attempt] = store.list_model_attempts(operation.operation_id)
        assert operation.status == "completed"
        assert attempt.status == "completed"
        assert attempt.usage["total_tokens"] == 15
        [response_item] = [item for item in store.list_items(result.turn_id) if item.kind == "model_response"]
        assert response_item.payload["response_status"] == "incomplete"
        assert response_item.payload["incomplete_reason"] == "max_output_tokens"
        assert completion_gate.evaluated is False
        assert not any(record.record_type == "model_attempt_unknown" for record in store.list_records(thread.thread_id))
        assert store.verify().valid is True


def test_late_model_attempt_cannot_commit_a_second_response(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="one answer only",
            binding_manifest={"model_alias": "test-model"},
        )
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="request-hash",
            context_hash="context-hash",
            tool_hash="tool-hash",
            wire_hash="wire-hash",
            request_ref={"request_id": f"{turn.turn_id}:step:1", "message_count": 1},
        )
        first = store.dispatch_model_attempt(operation.operation_id)
        store.mark_model_attempt_unknown(
            operation_id=operation.operation_id,
            attempt_id=first.attempt_id,
            generation=first.generation,
        )
        second = store.prepare_model_retry(operation.operation_id)
        store.dispatch_model_attempt(operation.operation_id)

        assert store.complete_model_attempt(
            operation_id=operation.operation_id,
            attempt_id=second.attempt_id,
            generation=second.generation,
            text="new answer",
            provider_response_id="response-new",
            usage={"input_tokens": 5, "output_tokens": 2},
        )
        assert not store.complete_model_attempt(
            operation_id=operation.operation_id,
            attempt_id=first.attempt_id,
            generation=first.generation,
            text="late old answer",
            provider_response_id="response-old",
            usage={"input_tokens": 5, "output_tokens": 3},
        )

        [response] = [item for item in store.list_items(turn.turn_id) if item.kind == "model_response"]
        assert response.payload["text"] == "new answer"
        attempts = store.list_model_attempts(operation.operation_id)
        assert [(attempt.generation, attempt.status) for attempt in attempts] == [
            (1, "abandoned"),
            (2, "completed"),
        ]
        assert store.verify().valid is True


def test_provider_failure_leaves_durable_unknown_attempt_for_reconciliation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=FailingDispatchModel(store),
            context_manager=RolloutContextManager(store),
            completion_gate=InspectingCompletionGate(store),
        )

        result = asyncio.run(
            runner.run(
                user_message="provider may fail",
                binding_manifest={"model_alias": "test-model"},
            )
        )

        [operation] = store.list_model_operations()
        [attempt] = store.list_model_attempts(operation.operation_id)
        assert operation.status == "unknown"
        assert attempt.status == "unknown"
        assert result.turn_id == operation.turn_id
        assert result.status == "paused"
        assert store.read_turn(operation.turn_id).status == "paused"
        unknown = [
            record for record in store.list_records(thread.thread_id) if record.record_type == "model_attempt_unknown"
        ]
        assert unknown[-1].payload["error_type"] == "ConnectionError"
        assert unknown[-1].payload["error_message"] == ("provider connection was lost after dispatch")
        assert store.verify().valid is True


def test_known_model_rejection_fails_without_unknown_or_retry_state(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=RejectedDispatchModel(store),
            context_manager=RolloutContextManager(store),
            completion_gate=InspectingCompletionGate(store),
        )

        result = asyncio.run(
            runner.run(
                user_message="do not retry a deterministic rejection",
                binding_manifest={"model_alias": "test-model"},
            )
        )

        [operation] = store.list_model_operations(result.turn_id)
        [attempt] = store.list_model_attempts(operation.operation_id)
        assert result.status == "failed"
        assert operation.status == "failed"
        assert attempt.status == "failed"
        rejected = [
            record for record in store.list_records(thread.thread_id) if record.record_type == "model_attempt_rejected"
        ]
        assert rejected[-1].payload["error_type"] == "RuntimeError"
        assert rejected[-1].payload["error_message"] == ("provider rejected a validly delivered request")
        assert not any(record.record_type == "model_attempt_unknown" for record in store.list_records(thread.thread_id))
        assert store.verify().valid is True


def test_model_retry_uses_a_new_attempt_on_the_same_logical_operation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        model = RetryThenAnswerModel(store)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=InspectingCompletionGate(store),
        )
        paused = asyncio.run(
            runner.run(
                user_message="recover provider",
                binding_manifest={"model_alias": "test-model"},
            )
        )

        resumed = asyncio.run(runner.retry_unknown_model(turn_id=paused.turn_id))

        assert resumed.status == "completed"
        assert resumed.answer == "recovered model answer"
        [operation] = store.list_model_operations(paused.turn_id)
        assert operation.status == "completed"
        assert [
            (attempt.generation, attempt.status) for attempt in store.list_model_attempts(operation.operation_id)
        ] == [
            (1, "abandoned"),
            (2, "completed"),
        ]
        assert model.dispatch_count == 2
        assert store.read_turn(paused.turn_id).status == "completed"
        assert store.verify().valid is True


def test_completion_gate_continue_feeds_the_gap_back_into_the_same_turn(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        model = ReviseAfterFeedbackModel(store)
        gate = ContinueThenAcceptGate()
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=gate,
        )

        result = asyncio.run(
            runner.run(
                user_message="finish with evidence",
                binding_manifest={"model_alias": "test-model"},
            )
        )

        assert result.status == "completed"
        assert result.answer == "verified final answer"
        assert gate.calls == 2
        assert len(store.list_model_operations(result.turn_id)) == 2
        assert [
            item.payload["action"] for item in store.list_items(result.turn_id) if item.kind == "completion_decision"
        ] == ["continue", "accept"]
        [feedback] = [item for item in store.list_items(result.turn_id) if item.kind == "completion_feedback"]
        assert feedback.payload == {"text": "verification evidence is missing"}
        assert store.verify().valid is True


def test_completion_gate_pause_creates_a_durable_clarification(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=InspectingModel(store),
            context_manager=RolloutContextManager(store),
            completion_gate=FixedDecisionGate(
                "pause",
                "choose the intended target",
            ),
            worker_id="turn-worker-a",
        )

        result = asyncio.run(
            runner.run(
                user_message="ambiguous task",
                binding_manifest={"model_alias": "test-model"},
            )
        )

        assert result.status == "paused"
        assert result.interaction_id is not None
        [interaction] = store.list_interactions(result.turn_id)
        assert interaction.kind == "clarification"
        assert interaction.status == "pending"
        assert interaction.request == {"question": "choose the intended target"}
        assert store.read_turn(result.turn_id).status == "paused"
        assert store.read_thread(thread.thread_id).active_turn_id == result.turn_id


def test_completion_gate_fail_releases_the_thread_without_an_agent_answer(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=InspectingModel(store),
            context_manager=RolloutContextManager(store),
            completion_gate=FixedDecisionGate("fail", "safety evidence rejected"),
            worker_id="turn-worker-a",
        )

        result = asyncio.run(
            runner.run(
                user_message="unsafe completion",
                binding_manifest={"model_alias": "test-model"},
            )
        )

        assert result.status == "failed"
        assert result.answer is None
        assert store.read_turn(result.turn_id).status == "failed"
        assert store.read_thread(thread.thread_id).active_turn_id is None
        assert not any(item.kind == "agent_message" for item in store.list_items(result.turn_id))
        assert store.verify().valid is True


def test_clarification_response_resumes_the_same_turn_without_granting_permission(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        model = AnswerAfterClarificationModel(store)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=PauseThenAcceptGate(),
            worker_id="turn-worker-a",
        )
        paused = asyncio.run(
            runner.run(
                user_message="answer the ambiguous task",
                binding_manifest={"model_alias": "test-model"},
            )
        )
        assert paused.interaction_id is not None

        resumed = asyncio.run(
            runner.respond_interaction(
                turn_id=paused.turn_id,
                request_id=paused.interaction_id,
                response="target A",
            )
        )

        assert resumed.turn_id == paused.turn_id
        assert resumed.status == "completed"
        assert resumed.answer == "answer for target A"
        [interaction] = store.list_interactions(paused.turn_id)
        assert interaction.status == "resolved"
        assert interaction.response == {"text": "target A"}
        assert store.list_approvals(paused.turn_id) == ()
        assert [item.payload["text"] for item in store.list_items(paused.turn_id) if item.kind == "user_message"] == [
            "answer the ambiguous task",
            "target A",
        ]
        assert store.verify().valid is True


def test_repeated_identical_clarification_response_returns_the_completed_turn(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        model = AnswerAfterClarificationModel(store)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=PauseThenAcceptGate(),
            worker_id="turn-worker-a",
        )
        paused = asyncio.run(
            runner.run(
                user_message="answer the ambiguous task",
                binding_manifest={"model_alias": "test-model"},
            )
        )
        assert paused.interaction_id is not None
        completed = asyncio.run(
            runner.respond_interaction(
                turn_id=paused.turn_id,
                request_id=paused.interaction_id,
                response="target A",
            )
        )

        repeated = asyncio.run(
            runner.respond_interaction(
                turn_id=paused.turn_id,
                request_id=paused.interaction_id,
                response="target A",
            )
        )

        assert repeated == completed
        assert len(model.prepared_requests) == 2
        assert len(store.list_model_operations(paused.turn_id)) == 2
        assert store.verify().valid is True


def test_conflicting_or_wrong_clarification_response_fails_before_model_io(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        model = AnswerAfterClarificationModel(store)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=PauseThenAcceptGate(),
            worker_id="turn-worker-a",
        )
        paused = asyncio.run(
            runner.run(
                user_message="answer the ambiguous task",
                binding_manifest={"model_alias": "test-model"},
            )
        )
        assert paused.interaction_id is not None

        for request_id, response, expected in (
            ("interaction_missing", "target A", KeyError),
            (paused.interaction_id, "target A", None),
            (paused.interaction_id, "target B", RuntimeError),
        ):
            if expected is None:
                asyncio.run(
                    runner.respond_interaction(
                        turn_id=paused.turn_id,
                        request_id=request_id,
                        response=response,
                    )
                )
            else:
                try:
                    asyncio.run(
                        runner.respond_interaction(
                            turn_id=paused.turn_id,
                            request_id=request_id,
                            response=response,
                        )
                    )
                except expected:
                    pass
                else:
                    raise AssertionError(f"expected {expected.__name__}")

        assert len(model.prepared_requests) == 2
        assert len(store.list_model_operations(paused.turn_id)) == 2
        assert store.verify().valid is True


def test_choice_uses_the_durable_interaction_lifecycle_without_granting_permission(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="pick a deployment target",
            binding_manifest={"model_alias": "test-model"},
        )
        choice = store.request_choice(
            turn_id=turn.turn_id,
            question="Which target?",
            options=("staging", "production"),
        )
        model = InspectingModel(store)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=InspectingCompletionGate(store),
            worker_id="turn-worker-a",
        )

        result = asyncio.run(
            runner.respond_interaction(
                turn_id=turn.turn_id,
                request_id=choice.request_id,
                response="staging",
            )
        )

        assert result.status == "completed"
        assert store.read_interaction(choice.request_id).response == {"selection": "staging"}
        assert store.list_approvals(turn.turn_id) == ()
        assert model.prepared_requests[0].messages[-1] == HarnessMessage(
            role="user",
            content="staging",
        )
        assert store.verify().valid is True


def test_choice_response_is_idempotent_and_rejects_invalid_or_conflicting_input(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="pick a deployment target",
            binding_manifest={"model_alias": "test-model"},
        )
        choice = store.request_choice(
            turn_id=turn.turn_id,
            question="Which target?",
            options=("staging", "production"),
        )
        model = InspectingModel(store)
        runner = Session(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=InspectingCompletionGate(store),
            worker_id="turn-worker-a",
        )

        try:
            asyncio.run(
                runner.respond_interaction(
                    turn_id=turn.turn_id,
                    request_id=choice.request_id,
                    response="development",
                )
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid choice must fail before model I/O")
        assert model.prepared_requests == []

        completed = asyncio.run(
            runner.respond_interaction(
                turn_id=turn.turn_id,
                request_id=choice.request_id,
                response="staging",
            )
        )
        repeated = asyncio.run(
            runner.respond_interaction(
                turn_id=turn.turn_id,
                request_id=choice.request_id,
                response="staging",
            )
        )
        assert repeated == completed

        try:
            asyncio.run(
                runner.respond_interaction(
                    turn_id=turn.turn_id,
                    request_id=choice.request_id,
                    response="production",
                )
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("conflicting terminal choice must fail loud")
        assert len(model.prepared_requests) == 1
        assert store.verify().valid is True

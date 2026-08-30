"""Live Harness execution spine: Session -> TurnContext -> StepContext."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from agent_runtime.harness.events import RolloutEventReader
from agent_runtime.harness.protocol import (
    CompletionGate,
    CompletionProposal,
    ContextBudgetExceededError,
    ContextManager,
    HarnessMessage,
    HarnessModel,
    HarnessModelDelta,
    HarnessModelRequest,
    HarnessModelResponse,
    HarnessToolCall,
    ModelDispatchCancelledError,
    ModelDispatchOutcomeUnknownError,
    ModelDispatchPreflightError,
    PreparedModelCall,
    ToolRouter,
    TurnResult,
)
from agent_runtime.harness.rollout import ItemSnapshot, ModelOperationSnapshot, RolloutStore
from agent_runtime.harness.tool_orchestrator import (
    ToolApprovalInvalidatedError,
    ToolApprovalRequiredError,
    ToolOrchestrator,
)
from agent_runtime.streaming.events import (
    ItemDeltaKind,
    TurnItemKind,
    derive_model_public_item_id,
    item_delta,
)
from agent_runtime.streaming.sink import EventChannelClosed, TurnEventDispatcher
from agent_runtime.tools.tool import Tool, ToolCall, ToolCallOrigin


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Immutable identity and durable settings for one Turn in this Session."""

    thread_id: str
    turn_id: str
    binding_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding_manifest",
            MappingProxyType(dict(self.binding_manifest)),
        )


@dataclass(frozen=True, slots=True)
class StepContext:
    """One immutable model-request view captured inside a Turn."""

    turn: TurnContext
    step: int
    messages: tuple[HarnessMessage, ...]
    tools: tuple[Tool, ...]
    model_token_budget_remaining: int | None

    def model_request(self) -> HarnessModelRequest:
        return HarnessModelRequest(
            thread_id=self.turn.thread_id,
            turn_id=self.turn.turn_id,
            messages=self.messages,
            binding_manifest=self.turn.binding_manifest,
            tools=self.tools,
            step=self.step,
            model_token_budget_remaining=self.model_token_budget_remaining,
        )


class Session:
    """Live owner of one thread and its Turn/Step execution."""

    def __init__(
        self,
        *,
        thread_id: str,
        store: RolloutStore,
        model: HarnessModel,
        context_manager: ContextManager,
        completion_gate: CompletionGate,
        tool_router: ToolRouter | None = None,
        tool_orchestrator: ToolOrchestrator | None = None,
        max_steps: int = 16,
        worker_id: str | None = None,
        model_lease_seconds: float = 300.0,
        event_dispatcher: TurnEventDispatcher | None = None,
    ) -> None:
        store.read_thread(thread_id)
        self.thread_id = thread_id
        self._store = store
        self._model = model
        self._context_manager = context_manager
        self._completion_gate = completion_gate
        self._tool_router = tool_router
        self._tool_orchestrator = tool_orchestrator
        self._max_steps = max_steps
        self._worker_id = worker_id or f"worker_{uuid4().hex}"
        self._model_lease_seconds = model_lease_seconds
        self._event_dispatcher = event_dispatcher

    def attach_event_dispatcher(
        self,
        event_dispatcher: TurnEventDispatcher,
    ) -> None:
        self._event_dispatcher = event_dispatcher
        if self._tool_orchestrator is not None:
            self._tool_orchestrator.attach_event_dispatcher(event_dispatcher)

    async def run(
        self,
        *,
        user_message: str,
        binding_manifest: Mapping[str, Any],
        input_files: tuple[Mapping[str, Any], ...] = (),
    ) -> TurnResult:
        turn = await self._commit(
            lambda: self._store.start_turn(
                thread_id=self.thread_id,
                user_message=user_message,
                binding_manifest=binding_manifest,
                input_files=input_files,
            )
        )
        return await self.run_turn(
            self.restore_turn_context(turn.turn_id),
            start_step=1,
        )

    async def _commit[T](self, operation: Callable[[], T]) -> T:
        mutation = self._store.capture_mutation(operation)
        if self._event_dispatcher is not None:
            for replayed in RolloutEventReader(self._store).project_committed_batch(mutation.records):
                await self._event_dispatcher.emit(
                    replayed.event,
                    cursor=replayed.cursor,
                )
        return mutation.value

    def restore_turn_context(self, turn_id: str) -> TurnContext:
        turn = self._store.read_turn(turn_id)
        if turn.thread_id != self.thread_id:
            raise RuntimeError("Turn belongs to a different Session")
        return TurnContext(
            thread_id=self.thread_id,
            turn_id=turn.turn_id,
            binding_manifest=turn.binding_manifest,
        )

    def capture_step_context(
        self,
        turn_context: TurnContext,
        *,
        step: int,
    ) -> StepContext:
        if turn_context.thread_id != self.thread_id:
            raise RuntimeError("Turn belongs to a different Session")
        messages = self._context_manager.build(turn_context.turn_id)
        tools = (
            ()
            if self._tool_router is None
            else self._tool_router.select(
                turn_id=turn_context.turn_id,
                messages=messages,
            )
        )
        return StepContext(
            turn=turn_context,
            step=step,
            messages=messages,
            tools=tools,
            model_token_budget_remaining=_remaining_model_tokens(
                turn_context.binding_manifest,
                consumed=self._consumed_model_tokens(turn_context.turn_id),
            ),
        )

    async def resume(self, *, turn_id: str, decision: str) -> TurnResult:
        if self._tool_orchestrator is None:
            raise RuntimeError("Turn has no ToolOrchestrator for approval resume")
        turn = self._store.read_turn(turn_id)
        turn_context = self.restore_turn_context(turn_id)
        approval_interactions = [
            interaction for interaction in self._store.list_interactions(turn_id) if interaction.kind == "tool_approval"
        ]
        if approval_interactions and approval_interactions[-1].status == "resolved":
            prior_decision = approval_interactions[-1].response.get("decision")
            if prior_decision != decision:
                raise RuntimeError(f"decision conflicts with resolved approval: {prior_decision} != {decision}")
            if turn.status == "completed":
                answers = [
                    item.payload.get("text")
                    for item in self._store.list_items(turn_id)
                    if item.kind == "agent_message" and isinstance(item.payload.get("text"), str)
                ]
                if len(answers) != 1:
                    raise RuntimeError("completed Turn has no unique canonical answer")
                return TurnResult(
                    thread_id=turn.thread_id,
                    turn_id=turn_id,
                    answer=answers[0],
                    status="completed",
                )
            if turn.status == "running" and decision == "approve":
                try:
                    await self._tool_orchestrator.recover_resolved_approval(turn_id=turn_id)
                except ToolApprovalInvalidatedError as invalidated:
                    return TurnResult(
                        thread_id=turn.thread_id,
                        turn_id=turn_id,
                        answer=None,
                        status="paused",
                        interaction_id=invalidated.interaction_id,
                    )
                return await self.run_turn(
                    turn_context,
                    start_step=len(self._store.list_model_operations(turn_id)) + 1,
                )
        if turn.status != "paused":
            raise RuntimeError(f"turn is not paused: {turn_id}")
        try:
            await self._tool_orchestrator.resume_approval(
                turn_id=turn_id,
                decision=decision,
            )
        except ToolApprovalInvalidatedError as invalidated:
            return TurnResult(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                answer=None,
                status="paused",
                interaction_id=invalidated.interaction_id,
            )
        start_step = len(self._store.list_model_operations(turn_id)) + 1
        return await self.run_turn(
            turn_context,
            start_step=start_step,
        )

    async def retry_unknown_model(self, *, turn_id: str) -> TurnResult:
        turn = self._store.read_turn(turn_id)
        turn_context = self.restore_turn_context(turn_id)
        if turn.status not in {"paused", "interrupted"}:
            raise RuntimeError("model retry requires a paused or interrupted Turn")
        unknown = [
            operation for operation in self._store.list_model_operations(turn_id) if operation.status == "unknown"
        ]
        if len(unknown) != 1:
            raise RuntimeError("model retry requires one unknown logical operation")
        operation = unknown[0]
        step = len(self._store.list_model_operations(turn_id))
        request = self.capture_step_context(turn_context, step=step).model_request()
        prepared = self._model.prepare(request)
        if (
            prepared.request_hash != operation.request_hash
            or prepared.context_hash != operation.context_hash
            or prepared.tool_hash != operation.tool_hash
            or prepared.wire_hash != operation.wire_hash
        ):
            raise RuntimeError("unknown model request cannot be reproduced exactly")
        await self._commit(lambda: self._store.prepare_model_retry(operation.operation_id))
        dispatched = await self._dispatch_prepared(
            thread_id=turn.thread_id,
            turn_id=turn_id,
            operation=operation,
            prepared=prepared,
        )
        if isinstance(dispatched, TurnResult):
            return dispatched
        token_budget = turn.binding_manifest.get("model_token_budget_total")
        if token_budget is not None and self._consumed_model_tokens(turn_id) > token_budget:
            return await self._fail_turn(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                reason="Turn exceeded its frozen model token budget.",
            )
        handled = await self._handle_model_response(
            thread_id=turn.thread_id,
            turn_id=turn_id,
            response=dispatched,
            prepared=prepared,
        )
        if handled is not None:
            return handled
        return await self.run_turn(
            turn_context,
            start_step=step + 1,
        )

    async def recover_committed_model_response(self, *, turn_id: str) -> TurnResult:
        """Continue after a crash between response commit and response handling."""

        turn = self._store.read_turn(turn_id)
        turn_context = self.restore_turn_context(turn_id)
        if turn.status != "running":
            raise RuntimeError("committed response recovery requires a running Turn")
        operations = self._store.list_model_operations(turn_id)
        if not operations:
            raise RuntimeError("Turn has no committed model response to recover")
        operation = operations[-1]
        if operation.status != "completed" or operation.response_item_id is None:
            raise RuntimeError("latest model operation has no canonical completed response")
        response_item = self._store.read_item(operation.response_item_id)
        response = _response_from_committed_item(response_item)
        if not response.tool_calls:
            raise RuntimeError("latest committed response has no pending tool calls")
        tool_operations = {
            tool_operation.tool_call_id: tool_operation for tool_operation in self._store.list_tool_operations(turn_id)
        }
        pending_calls: list[HarnessToolCall] = []
        for call in response.tool_calls:
            tool_operation = tool_operations.get(call.id)
            if tool_operation is None:
                pending_calls.append(call)
                continue
            if tool_operation.result_item_id is None:
                if tool_operation.status in {"succeeded", "failed"}:
                    interaction = await self._commit(
                        partial(
                            self._store.mark_tool_result_missing,
                            operation_id=tool_operation.operation_id,
                        )
                    )
                    return TurnResult(
                        thread_id=turn.thread_id,
                        turn_id=turn_id,
                        answer=None,
                        status="paused",
                        interaction_id=interaction.request_id,
                    )
                raise RuntimeError("committed tool call has an uncertain operation; use tool reconciliation")
            result_item = self._store.read_item(tool_operation.result_item_id)
            if (
                result_item.kind != "tool_result"
                or result_item.status != "completed"
                or result_item.payload.get("tool_call_id") != call.id
            ):
                raise RuntimeError("committed tool result linkage is malformed")
        token_budget = turn.binding_manifest.get("model_token_budget_total")
        if token_budget is not None and self._consumed_model_tokens(turn_id) > token_budget:
            return await self._fail_turn(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                reason="Turn exceeded its frozen model token budget.",
            )
        prepared = PreparedModelCall(
            request_hash=operation.request_hash,
            context_hash=operation.context_hash,
            tool_hash=operation.tool_hash,
            wire_hash=operation.wire_hash,
            request_ref=operation.request_ref,
        )
        if pending_calls:
            handled = await self._handle_model_response(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                response=HarnessModelResponse(
                    text=response.text,
                    provider_response_id=response.provider_response_id,
                    usage=response.usage,
                    tool_calls=tuple(pending_calls),
                ),
                prepared=prepared,
            )
            if handled is not None:
                return handled
        return await self.run_turn(
            turn_context,
            start_step=len(operations) + 1,
        )

    async def respond_interaction(
        self,
        *,
        turn_id: str,
        request_id: str,
        response: str,
    ) -> TurnResult:
        turn = self._store.read_turn(turn_id)
        turn_context = self.restore_turn_context(turn_id)
        interaction = self._store.read_interaction(request_id)
        if interaction.turn_id != turn_id or interaction.kind not in {
            "clarification",
            "choice",
        }:
            raise RuntimeError("interaction is not a user-response request for this Turn")
        if interaction.status == "resolved":
            response_field = "text" if interaction.kind == "clarification" else "selection"
            prior_response = interaction.response.get(response_field)
            if prior_response != response:
                raise RuntimeError(
                    f"response conflicts with resolved {interaction.kind}: {prior_response!r} != {response!r}"
                )
            if turn.status == "completed":
                answers = [
                    item.payload.get("text")
                    for item in self._store.list_items(turn_id)
                    if item.kind == "agent_message" and isinstance(item.payload.get("text"), str)
                ]
                if len(answers) != 1:
                    raise RuntimeError("completed Turn has no unique canonical answer")
                return TurnResult(
                    thread_id=turn.thread_id,
                    turn_id=turn_id,
                    answer=answers[0],
                    status="completed",
                )
            raise RuntimeError(f"resolved {interaction.kind} is already being processed")
        if interaction.kind == "clarification":
            await self._commit(
                lambda: self._store.resolve_clarification(
                    turn_id=turn_id,
                    request_id=request_id,
                    response=response,
                )
            )
        else:
            await self._commit(
                lambda: self._store.resolve_choice(
                    turn_id=turn_id,
                    request_id=request_id,
                    selection=response,
                )
            )
        return await self.run_turn(
            turn_context,
            start_step=len(self._store.list_model_operations(turn_id)) + 1,
        )

    async def run_turn(
        self,
        turn_context: TurnContext,
        *,
        start_step: int,
    ) -> TurnResult:
        thread_id = turn_context.thread_id
        turn_id = turn_context.turn_id
        binding_manifest = turn_context.binding_manifest
        step_budget = binding_manifest.get("model_step_budget", self._max_steps)
        if isinstance(step_budget, bool) or not isinstance(step_budget, int) or step_budget < 1:
            raise RuntimeError("frozen model step budget is invalid")
        token_budget = binding_manifest.get("model_token_budget_total")
        if token_budget is not None and (
            isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 1
        ):
            raise RuntimeError("frozen model token budget is invalid")
        effective_step_budget = min(self._max_steps, step_budget)
        for step in range(start_step, effective_step_budget + 1):
            if token_budget is not None and self._consumed_model_tokens(turn_id) >= token_budget:
                return await self._fail_turn(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    reason="Turn exhausted its frozen model token budget.",
                )
            try:
                step_context = self.capture_step_context(turn_context, step=step)
                request = step_context.model_request()
            except ContextBudgetExceededError as exc:
                return await self._fail_turn(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    reason=str(exc),
                )
            prepared = self._model.prepare(request)
            durable_request_ref = {
                **prepared.request_ref,
                "request_id": prepared.request_ref.get("request_id") or f"{turn_id}:step:{step}",
            }
            operation = await self._commit(
                partial(
                    self._store.prepare_model_operation,
                    turn_id=turn_id,
                    request_hash=prepared.request_hash,
                    context_hash=prepared.context_hash,
                    tool_hash=prepared.tool_hash,
                    wire_hash=prepared.wire_hash,
                    request_ref=durable_request_ref,
                )
            )
            dispatched = await self._dispatch_prepared(
                thread_id=thread_id,
                turn_id=turn_id,
                operation=operation,
                prepared=prepared,
            )
            if isinstance(dispatched, TurnResult):
                return dispatched
            if token_budget is not None and self._consumed_model_tokens(turn_id) > token_budget:
                return await self._fail_turn(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    reason="Turn exceeded its frozen model token budget.",
                )
            handled = await self._handle_model_response(
                thread_id=thread_id,
                turn_id=turn_id,
                response=dispatched,
                prepared=prepared,
            )
            if handled is None:
                continue
            return handled
        return await self._fail_turn(
            thread_id=thread_id,
            turn_id=turn_id,
            reason="Turn exhausted its frozen model step budget.",
        )

    def _consumed_model_tokens(self, turn_id: str) -> int:
        consumed = 0
        for operation in self._store.list_model_operations(turn_id):
            for attempt in self._store.list_model_attempts(operation.operation_id):
                if attempt.status != "completed":
                    continue
                total = attempt.usage.get("total_tokens")
                if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
                    consumed += total
                    continue
                for key in ("input_tokens", "output_tokens"):
                    value = attempt.usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        consumed += value
        return consumed

    async def _fail_turn(
        self,
        *,
        thread_id: str,
        turn_id: str,
        reason: str,
    ) -> TurnResult:
        await self._commit(lambda: self._store.fail_turn(turn_id=turn_id, reason=reason))
        return TurnResult(
            thread_id=thread_id,
            turn_id=turn_id,
            answer=None,
            status="failed",
        )

    async def _dispatch_prepared(
        self,
        *,
        thread_id: str,
        turn_id: str,
        operation: ModelOperationSnapshot,
        prepared: PreparedModelCall,
    ) -> HarnessModelResponse | TurnResult:
        attempt = await self._commit(
            lambda: self._store.dispatch_model_attempt(
                operation.operation_id,
                worker_id=self._worker_id,
                lease_seconds=self._model_lease_seconds,
            )
        )
        streamed_content: dict[str, list[str]] = {
            "text": [],
            "reasoning": [],
            "plan": [],
        }

        async def publish_delta(delta: HarnessModelDelta) -> None:
            streamed_content[delta.channel].append(delta.content)
            channel = {
                "text": "agent_message",
                "reasoning": "reasoning",
                "plan": "plan",
            }[delta.channel]
            item_kind = {
                "text": TurnItemKind.AGENT_MESSAGE,
                "reasoning": TurnItemKind.REASONING,
                "plan": TurnItemKind.PLAN,
            }[delta.channel]
            delta_kind = {
                "text": ItemDeltaKind.TEXT,
                "reasoning": ItemDeltaKind.REASONING,
                "plan": ItemDeltaKind.PLAN,
            }[delta.channel]
            await self._commit(
                partial(
                    self._store.start_model_output_channel,
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    channel=channel,
                )
            )
            if self._event_dispatcher is not None:
                await self._event_dispatcher.emit(
                    item_delta(
                        turn_id=turn_id,
                        item_id=derive_model_public_item_id(
                            turn_id=turn_id,
                            model_attempt_id=attempt.attempt_id,
                            channel=channel,
                        ),
                        item_kind=item_kind,
                        delta_kind=delta_kind,
                        delta=delta.content,
                    )
                )

        try:
            dispatch = self._model.dispatch
            if "delta_sink" in inspect.signature(dispatch).parameters:
                response = await dispatch(prepared, delta_sink=publish_delta)
            else:
                response = await dispatch(prepared)
        except ModelDispatchPreflightError as exc:
            await self._commit(
                partial(
                    self._store.reject_model_attempt,
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    reason=str(exc),
                )
            )
            return await self._fail_turn(
                thread_id=thread_id,
                turn_id=turn_id,
                reason=str(exc),
            )
        except (ModelDispatchCancelledError, EventChannelClosed) as exc:
            reason = str(exc).strip() or "provider acknowledged model cancellation"
            await self._commit(
                partial(
                    self._store.request_turn_cancellation,
                    turn_id=turn_id,
                    reason=reason,
                )
            )
            await self._commit(
                partial(
                    self._store.cancel_model_attempt,
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    reason=reason,
                    channel_content={
                        "agent_message": "".join(streamed_content["text"]),
                        "reasoning": "".join(streamed_content["reasoning"]),
                        "plan": "".join(streamed_content["plan"]),
                    },
                )
            )
            return TurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
                answer=None,
                status="cancelled",
            )
        except (ModelDispatchOutcomeUnknownError, ConnectionError, TimeoutError) as exc:
            await self._commit(
                partial(
                    self._store.mark_model_attempt_unknown,
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    error_type=type(exc).__name__,
                    error_message=(str(exc).strip() or "model dispatch raised without a message"),
                    channel_content={
                        "agent_message": "".join(streamed_content["text"]),
                        "reasoning": "".join(streamed_content["reasoning"]),
                        "plan": "".join(streamed_content["plan"]),
                    },
                )
            )
            return TurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
                answer=None,
                status="paused",
            )
        except asyncio.CancelledError as exc:
            await asyncio.shield(
                self._commit(
                    partial(
                        self._store.request_turn_cancellation,
                        turn_id=turn_id,
                        reason="Turn task cancelled while provider outcome was unknown",
                    )
                )
            )
            await self._commit(
                partial(
                    self._store.mark_model_attempt_unknown,
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    error_type=type(exc).__name__,
                    error_message=("model dispatch was cancelled after provider I/O began"),
                    channel_content={
                        "agent_message": "".join(streamed_content["text"]),
                        "reasoning": "".join(streamed_content["reasoning"]),
                        "plan": "".join(streamed_content["plan"]),
                    },
                )
            )
            raise
        except Exception as exc:
            message = str(exc).strip() or "model dispatch failed with a known error"
            await self._commit(
                partial(
                    self._store.reject_model_attempt,
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    reason=message,
                    error_type=type(exc).__name__,
                    error_message=message,
                    channel_content={
                        "agent_message": "".join(streamed_content["text"]),
                        "reasoning": "".join(streamed_content["reasoning"]),
                        "plan": "".join(streamed_content["plan"]),
                    },
                )
            )
            return await self._fail_turn(
                thread_id=thread_id,
                turn_id=turn_id,
                reason=message,
            )
        accepted = await self._commit(
            lambda: self._store.complete_model_attempt(
                operation_id=operation.operation_id,
                attempt_id=attempt.attempt_id,
                generation=attempt.generation,
                text=response.text,
                provider_response_id=response.provider_response_id,
                usage=response.usage,
                tool_calls=tuple(
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": dict(call.arguments),
                    }
                    for call in response.tool_calls
                ),
                response_status=response.status,
                incomplete_reason=response.incomplete_reason,
                reasoning_content=(
                    response.reasoning_content
                    if response.reasoning_content is not None
                    else "".join(streamed_content["reasoning"])
                ),
                plan_content=(
                    response.plan_content
                    if response.plan_content is not None
                    else "".join(streamed_content["plan"])
                ),
            )
        )
        if not accepted:
            raise RuntimeError("current model attempt lost its commit generation")
        if response.status == "incomplete":
            return await self._fail_turn(
                thread_id=thread_id,
                turn_id=turn_id,
                reason=(f"Model returned an incomplete agent step: {response.incomplete_reason}."),
            )
        return response

    async def _handle_model_response(
        self,
        *,
        thread_id: str,
        turn_id: str,
        response: HarnessModelResponse,
        prepared: PreparedModelCall,
    ) -> TurnResult | None:
        if response.tool_calls:
            if self._tool_orchestrator is None:
                raise RuntimeError("model requested tools but no ToolOrchestrator exists")
            for call in response.tool_calls:
                try:
                    await self._tool_orchestrator.execute(
                        turn_id=turn_id,
                        call=_aci_tool_call(call, prepared.request_ref),
                    )
                except ToolApprovalRequiredError as pause:
                    return TurnResult(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        answer=None,
                        status="paused",
                        interaction_id=pause.interaction_id,
                    )
                turn = self._store.read_turn(turn_id)
                if turn.status == "paused":
                    pending = [
                        interaction
                        for interaction in self._store.list_interactions(turn_id)
                        if interaction.status == "pending"
                    ]
                    if len(pending) != 1:
                        raise RuntimeError("paused tool execution has no unique interaction")
                    return TurnResult(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        answer=None,
                        status="paused",
                        interaction_id=pending[0].request_id,
                    )
            return None
        return await self._finish_answer(
            thread_id=thread_id,
            turn_id=turn_id,
            answer=response.text,
        )

    async def _finish_answer(
        self,
        *,
        thread_id: str,
        turn_id: str,
        answer: str,
    ) -> TurnResult | None:
        proposal_item = await self._commit(
            lambda: self._store.record_final_proposal(
                turn_id=turn_id,
                answer=answer,
            )
        )
        proposal = CompletionProposal(
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=proposal_item.item_id,
            answer=answer,
        )
        decision = self._completion_gate.evaluate(proposal)
        await self._commit(
            lambda: self._store.record_completion_decision(
                turn_id=turn_id,
                proposal_item_id=proposal.item_id,
                action=decision.action,
                reason=decision.reason,
            )
        )
        if decision.action == "continue":
            await self._commit(
                lambda: self._store.record_completion_feedback(
                    turn_id=turn_id,
                    reason=decision.reason,
                )
            )
            return None
        if decision.action == "pause":
            interaction = await self._commit(
                lambda: self._store.request_clarification(
                    turn_id=turn_id,
                    question=decision.reason,
                )
            )
            return TurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
                answer=None,
                status="paused",
                interaction_id=interaction.request_id,
            )
        if decision.action == "fail":
            await self._commit(
                lambda: self._store.fail_turn(
                    turn_id=turn_id,
                    reason=decision.reason,
                )
            )
            return TurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
                answer=None,
                status="failed",
            )
        completed = await self._commit(
            lambda: self._store.complete_turn(
                turn_id=turn_id,
                answer=answer,
            )
        )
        return TurnResult(
            thread_id=thread_id,
            turn_id=completed.turn_id,
            answer=answer,
        )


def _aci_tool_call(
    call: HarnessToolCall,
    request_ref: Mapping[str, Any],
) -> ToolCall:
    request_id = request_ref.get("request_id")
    toolset_revision = request_ref.get("toolset_revision")
    exposed = request_ref.get("exposed_tool_names")
    if (
        not isinstance(request_id, str)
        or not isinstance(toolset_revision, str)
        or not isinstance(exposed, (list, tuple))
        or any(not isinstance(name, str) for name in exposed)
    ):
        raise RuntimeError("prepared model request omitted its tool origin manifest")
    return ToolCall(
        tool_call_id=call.id,
        tool_name=call.name,
        arguments=call.arguments,
        origin=ToolCallOrigin(
            request_id=request_id,
            toolset_revision=toolset_revision,
            exposed_tool_names=tuple(exposed),
        ),
    )


def _response_from_committed_item(item: ItemSnapshot) -> HarnessModelResponse:
    if item.kind != "model_response" or item.status != "completed":
        raise RuntimeError("canonical model response Item is malformed")
    text = item.payload.get("text")
    provider_response_id = item.payload.get("provider_response_id")
    usage = item.payload.get("usage")
    raw_calls = item.payload.get("tool_calls", ())
    response_status = item.payload.get("response_status", "completed")
    incomplete_reason = item.payload.get("incomplete_reason")
    if (
        not isinstance(text, str)
        or (provider_response_id is not None and not isinstance(provider_response_id, str))
        or not isinstance(usage, Mapping)
        or not isinstance(raw_calls, (list, tuple))
        or response_status not in {"completed", "incomplete"}
        or (incomplete_reason is not None and not isinstance(incomplete_reason, str))
    ):
        raise RuntimeError("canonical model response payload is malformed")
    calls: list[HarnessToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            raise RuntimeError("canonical model tool call is malformed")
        call_id = raw_call.get("id")
        name = raw_call.get("name")
        arguments = raw_call.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, Mapping):
            raise RuntimeError("canonical model tool call is malformed")
        calls.append(HarnessToolCall(id=call_id, name=name, arguments=arguments))
    return HarnessModelResponse(
        text=text,
        provider_response_id=provider_response_id,
        usage=usage,
        tool_calls=tuple(calls),
        status=response_status,
        incomplete_reason=incomplete_reason,
    )


def _remaining_model_tokens(
    binding_manifest: Mapping[str, Any],
    *,
    consumed: int,
) -> int | None:
    total = binding_manifest.get("model_token_budget_total")
    if isinstance(total, bool) or not isinstance(total, int):
        return None
    return max(total - consumed, 0)

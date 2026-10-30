"""Turn execution and ephemeral model/tool Step contexts."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from agent_runtime.budget import (
    BudgetLimitExceededError,
)
from agent_runtime.core.messages import model_message_payload
from agent_runtime.harness import model_ops, turn_ops
from agent_runtime.harness.events import commit_and_publish
from agent_runtime.harness.execution_context import StepContext
from agent_runtime.harness.model_outputs import (
    complete_model_output,
    record_model_response_started,
    require_open_model_output,
    start_model_output,
)
from agent_runtime.harness.operation_values import TurnInputChangedError
from agent_runtime.harness.prepared_requests import frozen_request_payload
from agent_runtime.harness.protocol import (
    ContextBudgetExceededError,
    ContextManager,
    HarnessModel,
    HarnessModelResponse,
    ModelContextOverflowError,
    ModelDispatchCancelledError,
    ModelDispatchOutcomeUnknownError,
    ModelDispatchPreflightError,
    ModelMessage,
    PreparedModelCall,
    ReplayingHarnessModel,
    TurnResult,
)
from agent_runtime.harness.records import ModelOperationSnapshot
from agent_runtime.harness.runtime_store import RuntimeStore
from agent_runtime.modeling.contracts import context_capacity_policy
from agent_runtime.modeling.events import (
    ModelEvent,
    ModelOutputCompleted,
    ModelOutputDelta,
    ModelOutputStarted,
    ModelResponseStarted,
    ModelStreamState,
)
from agent_runtime.streaming.events import (
    ItemDeltaKind,
    TurnItemKind,
    derive_model_public_item_id,
    item_delta,
)
from agent_runtime.streaming.sink import EventChannelClosed, TurnEventDispatcher


class ModelRunner:
    """Own one provider operation lifecycle, independently of Turn scheduling."""

    def __init__(
        self,
        *,
        store: RuntimeStore,
        model: HarnessModel,
        worker_id: str,
        model_lease_seconds: float = 300.0,
        event_dispatcher: TurnEventDispatcher | None = None,
        context_manager: ContextManager | None = None,
        prepare_binding: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._model = model
        self._worker_id = worker_id
        self._model_lease_seconds = model_lease_seconds
        self._event_dispatcher = event_dispatcher
        self._context_manager = context_manager
        self._prepare_binding = prepare_binding

    def attach_event_dispatcher(self, dispatcher: TurnEventDispatcher) -> None:
        self._event_dispatcher = dispatcher

    async def _commit[T](self, operation: Callable[[], T]) -> T:
        return await commit_and_publish(self._store, operation, self._event_dispatcher)

    async def _commit_cancellation(self, operation: Callable[[], None]) -> None:
        task = asyncio.create_task(self._commit(operation))
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
        task.result()

    async def _fail_turn(self, *, thread_id: str, turn_id: str, reason_code: str, message: str) -> TurnResult:
        await self._commit(
            lambda: turn_ops.fail_turn(self._store, turn_id=turn_id, reason_code=reason_code, message=message)
        )
        return TurnResult(thread_id=thread_id, turn_id=turn_id, answer=None, status="failed")

    async def prepare(self, step: StepContext) -> PreparedModelCall:
        policy = context_capacity_policy(step.binding_manifest)

        def fallback_summary(fork: StepContext) -> StepContext:
            binding = dict(fork.binding_manifest)
            fallback = binding.pop("summary_fork_fallback")
            binding.pop("summary_cache_fork")
            return replace(
                fork, binding_manifest=binding, messages=(ModelMessage(role="user", body=fallback),), tools=()
            )

        if policy is not None and step.purpose == "context_summary":
            # A large token window does not authorize an unbounded history
            # envelope. The planner splits complete work units before provider
            # serialization; an indivisible unit fails with originals intact.
            sizes = [len(message.content.encode("utf-8")) for message in step.messages]
            if max(sizes, default=0) > policy.max_item_bytes or sum(sizes) > policy.max_total_bytes:
                if step.binding_manifest.get("summary_cache_fork"):
                    return await self.prepare(fallback_summary(step))
                raise ContextBudgetExceededError(
                    "byte_guard: Summary fragment exceeds its frozen materialization limit.",
                    reason_code="byte_guard",
                )
        if self._prepare_binding is not None:
            await self._prepare_binding(step.binding_manifest)
        try:
            if step.frozen_request is not None:
                if not isinstance(self._model, ReplayingHarnessModel):
                    raise RuntimeError("model cannot restore its frozen provider request")
                prepared = self._model.replay(step.frozen_request, request=step.model_request())
            else:
                prepared = self._model.prepare(step.model_request())
        except ContextBudgetExceededError:
            if not step.binding_manifest.get("summary_cache_fork"):
                raise
            # A fork is an optimization, never a reason to weaken admission.
            # Keep the bounded legacy source in the frozen snapshot for fallback.
            step = fallback_summary(step)
            prepared = self._model.prepare(step.model_request())
        if (
            policy is not None
            and step.purpose == "context_summary"
            and prepared.request_ref.get("serialized_input_bytes", 0) > policy.max_total_bytes
        ):
            if step.binding_manifest.get("summary_cache_fork"):
                return await self.prepare(fallback_summary(step))
            raise ContextBudgetExceededError(
                "byte_guard: Serialized summary request exceeds its frozen materialization limit.",
                reason_code="byte_guard",
            )
        remaining = step.model_token_budget_remaining
        while (
            step.purpose == "agent_step"
            and step.frozen_request is None
            and step.budget_pressure
            and step.compaction_plan is None
            and remaining is not None
            and prepared.resource_request.total_tokens > remaining
        ):
            # Ordinary completion must fit the Turn's remaining capacity. Keep
            # joint summary/continuation plans intact and leave the atomic
            # dispatch reservation authoritative. Reprepare so wire, hashes,
            # pricing and the durable snapshot all use the same output cap.
            # Re-encoding may append changed input limits. Each pass must
            # strictly reduce the positive output cap, so this local refinement terminates.
            output_limit = remaining - prepared.resource_request.input_tokens
            if output_limit < 1:
                raise ContextBudgetExceededError(
                    f"Turn cumulative token budget cannot fit model input plus output: "
                    f"remaining={remaining}, input_tokens={prepared.resource_request.input_tokens}.",
                    reason_code="model_token_budget_exhausted",
                )
            previous_output = prepared.resource_request.output_tokens
            step = replace(step, output_token_limit=min(previous_output, output_limit))
            prepared = self._model.prepare(step.model_request())
            if prepared.resource_request.output_tokens >= previous_output:
                raise ContextBudgetExceededError(
                    f"Model preparation did not reduce the output cap: "
                    f"required={prepared.resource_request.total_tokens}, remaining={remaining}.",
                    reason_code="model_token_budget_exhausted",
                )
        if step.control.retry_count:
            prepared = replace(
                prepared, resource_request=replace(prepared.resource_request, retries=step.control.retry_count)
            )
        snapshot = {
            "input_revision": step.input_revision,
            "control": step.control.freeze(),
            "binding_manifest": dict(step.binding_manifest),
            "step": step.step,
            "purpose": step.purpose,
            "request_id": step.request_id,
            "input_token_limit": step.input_token_limit,
            "output_token_limit": step.output_token_limit,
            "continuation_summary_role": prepared.request_ref.get("continuation_summary_role", "context"),
            "compaction_plan": dict(step.compaction_plan) if step.compaction_plan is not None else None,
            "messages": [model_message_payload(message) for message in step.messages],
            "tools": [{"name": tool.definition.name, "revision": tool.execution_revision} for tool in step.tools],
            "model_token_budget_remaining": step.model_token_budget_remaining,
            "budget_pressure": step.budget_pressure,
        }
        budget_status = dict(step.control.budget_status)
        if step.frozen_request is None and context_capacity_policy(step.binding_manifest) is not None:
            from agent_runtime.harness.context import RolloutContextManager

            if (
                isinstance(self._context_manager, RolloutContextManager)
                and "context_projection" in prepared.request_ref
            ):
                limits = await self._store.run_async(self._context_manager.limits, step.turn.turn_id)
                projection = {
                    **prepared.request_ref["context_projection"],
                    "byte_guards": {"item": limits["max_item_bytes"], "total": limits["max_total_bytes"]},
                    "message_guard": limits["max_messages"],
                }
                prepared = replace(prepared, request_ref={**prepared.request_ref, "context_projection": projection})
        prepared = replace(
            prepared,
            request_ref={
                **prepared.request_ref,
                **(
                    {"context_updates": list(step.binding_manifest["context_updates"])}
                    if step.purpose == "agent_step" and step.binding_manifest.get("context_updates")
                    else {}
                ),
                **(
                    {"runtime_context_update": step.binding_manifest["runtime_context_update"]}
                    if step.purpose == "agent_step" and step.binding_manifest.get("runtime_context_update")
                    else {}
                ),
                **(
                    {"context_update_messages": step.binding_manifest["context_update_messages"]}
                    if step.purpose == "agent_step" and step.binding_manifest.get("context_update_messages")
                    else {}
                ),
                **(
                    {"runtime_context_message": step.binding_manifest["runtime_context_message"]}
                    if step.purpose == "agent_step" and step.binding_manifest.get("runtime_context_message")
                    else {}
                ),
                "purpose": step.purpose,
                **(
                    {"capacity_recovery_root_id": step.control.recovery_root_id}
                    if step.control.recovery_root_id is not None
                    else {}
                ),
                "input_revision": step.input_revision,
                **({"request_id": step.request_id} if step.request_id else {}),
                "step_snapshot": snapshot,
                **({"budget_status": budget_status} if budget_status is not None else {}),
                **({"compaction_plan": dict(step.compaction_plan)} if step.compaction_plan is not None else {}),
                **(
                    {"budget_finalization": True}
                    if step.binding_manifest.get("runtime_budget_phase") == "final_answer"
                    else {}
                ),
            },
        )

        payload = frozen_request_payload(prepared)
        if payload is None:
            return prepared
        return replace(prepared, request_ref={**prepared.request_ref, "frozen_request_payload": payload})

    async def dispatch(
        self,
        *,
        thread_id: str,
        turn_id: str,
        operation: ModelOperationSnapshot,
        prepared: PreparedModelCall,
        allow_protected_budget: bool = False,
    ) -> HarnessModelResponse | TurnResult:
        try:
            attempt = await self._commit(
                lambda: model_ops.dispatch_model_attempt(
                    self._store,
                    operation.operation_id,
                    worker_id=self._worker_id,
                    lease_seconds=self._model_lease_seconds,
                    resource_request=prepared.resource_request,
                    allow_protected_budget=allow_protected_budget,
                )
            )
        except asyncio.CancelledError:
            # The Store observed the claim's real outcome, but provider dispatch
            # has not been called. Release that known unspent reservation.
            def reject_undispatched() -> None:
                with self._store.persistence.transaction():
                    claimed = self._store.list_model_attempts(operation.operation_id)[-1]
                    if claimed.status == "dispatched" and claimed.claim_owner == self._worker_id:
                        model_ops.reject_model_attempt(
                            self._store,
                            operation_id=operation.operation_id,
                            attempt_id=claimed.attempt_id,
                            generation=claimed.generation,
                            reason="cancelled before provider dispatch",
                        )
                    turn_ops.request_turn_cancellation(
                        self._store, turn_id=turn_id, reason="Turn task cancelled before provider dispatch"
                    )

            await self._commit_cancellation(reject_undispatched)
            raise
        except TurnInputChangedError:
            await self._commit(lambda: model_ops.supersede_prepared_model(self._store, operation.operation_id))
            raise
        except BudgetLimitExceededError as exc:
            return await self._fail_turn(
                thread_id=thread_id,
                turn_id=turn_id,
                reason_code="model_budget_limit_exceeded",
                message=str(exc),
            )
        stream_state = ModelStreamState()
        output_item_ids: dict[str, str] = {}

        async def publish_model_event(event: ModelEvent) -> None:
            stream_state.accept(event)
            if isinstance(event, ModelResponseStarted):
                await self._commit(
                    partial(
                        record_model_response_started,
                        self._store,
                        operation_id=operation.operation_id,
                        attempt_id=attempt.attempt_id,
                        generation=attempt.generation,
                        provider_response_id=event.provider_response_id,
                    )
                )
            elif isinstance(event, ModelOutputStarted):
                item = await self._commit(
                    partial(
                        start_model_output,
                        self._store,
                        operation_id=operation.operation_id,
                        attempt_id=attempt.attempt_id,
                        generation=attempt.generation,
                        output=event.output,
                    )
                )
                output_item_ids[event.output.output_id] = item.item_id
            elif isinstance(event, ModelOutputCompleted):
                await self._commit(
                    partial(
                        complete_model_output,
                        self._store,
                        operation_id=operation.operation_id,
                        attempt_id=attempt.attempt_id,
                        generation=attempt.generation,
                        output=event.output,
                    )
                )
            elif isinstance(event, ModelOutputDelta):
                await self._store.run_async(
                    partial(
                        require_open_model_output,
                        self._store,
                        operation_id=operation.operation_id,
                        attempt_id=attempt.attempt_id,
                        generation=attempt.generation,
                        output_id=event.output_id,
                        item_id=output_item_ids[event.output_id],
                    )
                )
                if event.delta_kind == "signature" or self._event_dispatcher is None:
                    return
                output = next(output for output in stream_state.outputs if output.output_id == event.output_id)
                channel = {
                    "text": "agent_message",
                    "refusal": "agent_message",
                    "tool_call": "agent_message",
                    "reasoning": "reasoning",
                    "reasoning_summary": "reasoning",
                    "plan": "plan",
                }[output.kind]
                item_kind = {
                    "text": TurnItemKind.AGENT_MESSAGE,
                    "refusal": TurnItemKind.REFUSAL,
                    "tool_call": TurnItemKind.TOOL_CALL,
                    "reasoning": TurnItemKind.REASONING,
                    "reasoning_summary": TurnItemKind.REASONING,
                    "plan": TurnItemKind.PLAN,
                }[output.kind]
                delta_kind = {
                    "text": ItemDeltaKind.TEXT,
                    "reasoning": ItemDeltaKind.REASONING,
                    "reasoning_summary": ItemDeltaKind.REASONING,
                    "plan": ItemDeltaKind.PLAN,
                    "tool_arguments": ItemDeltaKind.TOOL_ARGUMENTS,
                }[event.delta_kind]
                await self._event_dispatcher.emit(
                    item_delta(
                        turn_id=turn_id,
                        item_id=derive_model_public_item_id(
                            turn_id=turn_id,
                            model_attempt_id=attempt.attempt_id,
                            channel=channel,
                            output_id=output.output_id,
                        ),
                        item_kind=item_kind,
                        delta_kind=delta_kind,
                        delta=event.content,
                    )
                )

        dispatch_started = time.monotonic()
        try:
            response = await self._model.dispatch(
                prepared,
                event_sink=(None if operation.request_ref.get("purpose") == "context_summary" else publish_model_event),
            )
            if stream_state.started:
                if response.stop_reason is None and stream_state.terminal is not None:
                    response = replace(response, stop_reason=stream_state.terminal.stop_reason)
                stream_state.validate_response(
                    usage=response.usage,
                    outputs=response.outputs,
                    provider_response_id=response.provider_response_id,
                    status=response.status,
                    incomplete_reason=response.incomplete_reason,
                    stop_reason=response.stop_reason,
                )
        except ModelContextOverflowError as exc:
            await self._commit(
                partial(
                    partial(model_ops.reject_model_attempt, self._store),
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    reason=str(exc),
                    error_type="context_overflow",
                )
            )
            if operation.request_ref.get("purpose") == "context_summary":
                return await self._fail_turn(
                    thread_id=thread_id, turn_id=turn_id, reason_code="summary_context_overflow", message=str(exc)
                )
            raise
        except ModelDispatchPreflightError as exc:
            await self._commit(
                partial(
                    partial(model_ops.reject_model_attempt, self._store),
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    reason=str(exc),
                )
            )
            return await self._fail_turn(
                thread_id=thread_id,
                turn_id=turn_id,
                reason_code="model_dispatch_preflight_rejected",
                message=str(exc),
            )
        except ModelDispatchCancelledError as exc:
            reason = str(exc).strip() or "provider acknowledged model cancellation"
            await self._commit(
                partial(
                    partial(turn_ops.request_turn_cancellation, self._store),
                    turn_id=turn_id,
                    reason=reason,
                )
            )
            await self._commit(
                partial(
                    partial(model_ops.cancel_model_attempt, self._store),
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    reason=reason,
                    outputs=stream_state.outputs,
                )
            )
            return TurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
                answer=None,
                status="cancelled",
            )
        except (ModelDispatchOutcomeUnknownError, ConnectionError, TimeoutError, EventChannelClosed) as exc:
            await self._commit(
                partial(
                    partial(model_ops.mark_model_attempt_unknown, self._store),
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    error_type=type(exc).__name__,
                    error_message=(str(exc).strip() or "model dispatch raised without a message"),
                    outputs=stream_state.outputs,
                )
            )
            return TurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
                answer=None,
                status="paused",
            )
        except asyncio.CancelledError as exc:
            error_type = type(exc).__name__

            def record_unknown() -> None:
                with self._store.persistence.transaction():
                    turn_ops.request_turn_cancellation(
                        self._store,
                        turn_id=turn_id,
                        reason="Turn task cancelled while provider outcome was unknown",
                    )
                    model_ops.mark_model_attempt_unknown(
                        self._store,
                        operation_id=operation.operation_id,
                        attempt_id=attempt.attempt_id,
                        generation=attempt.generation,
                        error_type=error_type,
                        outputs=stream_state.outputs,
                        error_message="model dispatch was cancelled after provider I/O began",
                    )

            await self._commit_cancellation(record_unknown)
            raise
        except Exception as exc:
            # The HarnessModel contract owns provider-outcome classification.
            # Explicit transport/unknown failures are handled above. A generic
            # exception is therefore a deterministic operation failure, but it
            # still does not prove zero provider billing. Fail the Attempt/Turn
            # while conservatively keeping the reservation as UNKNOWN exposure.
            message = str(exc).strip() or "model dispatch failed with a known error"
            await self._commit(
                partial(
                    partial(model_ops.reject_model_attempt, self._store),
                    operation_id=operation.operation_id,
                    attempt_id=attempt.attempt_id,
                    generation=attempt.generation,
                    reason=message,
                    error_type=type(exc).__name__,
                    error_message=message,
                    outputs=stream_state.outputs,
                    budget_outcome="unknown",
                )
            )
            return await self._fail_turn(
                thread_id=thread_id,
                turn_id=turn_id,
                reason_code="model_dispatch_rejected",
                message=message,
            )
        dispatch_ms = (time.monotonic() - dispatch_started) * 1000
        response = replace(
            response,
            usage={
                **response.usage,
                "latency_ms": dispatch_ms,
                "model_ms": int(dispatch_ms),
            },
        )
        accepted = await self._commit(
            lambda: model_ops.complete_model_attempt(
                self._store,
                operation_id=operation.operation_id,
                attempt_id=attempt.attempt_id,
                generation=attempt.generation,
                response=response,
            )
        )
        if not accepted:
            raise RuntimeError("current model attempt lost its commit generation")
        return response

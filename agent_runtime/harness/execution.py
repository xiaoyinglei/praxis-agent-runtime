"""Advance one durable Turn through execution, recovery and user controls."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from typing import Any
from uuid import uuid4

from agent_runtime.approval import CommandGrant
from agent_runtime.budget import (
    BudgetLimitExceededError,
    PricingUnavailableError,
)
from agent_runtime.harness import model_ops, model_retry_ops, turn_ops
from agent_runtime.harness.budget_context import agent_step_count
from agent_runtime.harness.context_engine import ContextEngine, SummaryStoppedError
from agent_runtime.harness.events import commit_and_publish
from agent_runtime.harness.execution_context import StepContext, TurnContext, step_limit_from_binding
from agent_runtime.harness.execution_facts import adopt_execution_protocol
from agent_runtime.harness.lifecycle import LifecycleHooks
from agent_runtime.harness.model_runner import ModelRunner
from agent_runtime.harness.operation_values import TurnCancellationPendingError, TurnInputChangedError
from agent_runtime.harness.ownership import ConversationRecoveryRequiredError, recover_owner_loss
from agent_runtime.harness.protocol import (
    BindingProvider,
    CompletionGate,
    ContextBudgetExceededError,
    ContextManager,
    ContextSourceChangedError,
    HarnessModel,
    HarnessModelResponse,
    ModelContextOverflowError,
    ModelMessage,
    ModelRetryScheduledError,
    ToolRouter,
    TurnResult,
    item_text,
)
from agent_runtime.harness.records import ModelOperationSnapshot, TurnSnapshot
from agent_runtime.harness.recovery import ContinuationKind, active_capacity_root, recovery_plan
from agent_runtime.harness.request_context import RequestContext
from agent_runtime.harness.responses import ResponseHandler
from agent_runtime.harness.runtime_store import RuntimeStore
from agent_runtime.harness.stop import TurnStop
from agent_runtime.harness.tool_orchestrator import (
    ToolApprovalInvalidatedError,
    ToolApprovalRequiredError,
    ToolOrchestrator,
)
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.tools.evidence import workspace_changed


class TurnRunner:
    """Execute a Turn using services borrowed from its owning Session."""

    def __init__(
        self,
        *,
        thread_id: str,
        store: RuntimeStore,
        model: HarnessModel,
        context_manager: ContextManager,
        completion_gate: CompletionGate,
        tool_router: ToolRouter | None = None,
        tool_orchestrator: ToolOrchestrator | None = None,
        max_steps: int | None = None,
        worker_id: str | None = None,
        model_lease_seconds: float = 300.0,
        event_dispatcher: TurnEventDispatcher | None = None,
        binding_provider: BindingProvider | None = None,
        prepare_binding: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
        before_completion: Callable[[str], Awaitable[None]] | None = None,
        lifecycle_hooks: LifecycleHooks | None = None,
        dispatch_guard: Callable[[RuntimeStore], None] | None = None,
    ) -> None:
        store.read_thread(thread_id)
        self.thread_id = thread_id
        self._store = store
        self._model = model
        self._context_manager = context_manager
        self._tool_orchestrator = tool_orchestrator
        self._max_steps = max_steps
        self._worker_id = worker_id or f"worker_{uuid4().hex}"
        self._event_dispatcher = event_dispatcher
        self._capacity_recovery_root: str | None = None
        self._dispatch_guard = dispatch_guard
        self._model_runner = ModelRunner(
            store=store,
            model=model,
            worker_id=self._worker_id,
            model_lease_seconds=model_lease_seconds,
            event_dispatcher=event_dispatcher,
            context_manager=context_manager,
            prepare_binding=prepare_binding,
            dispatch_guard=dispatch_guard,
        )
        self._requests = RequestContext(
            thread_id=thread_id,
            store=store,
            context_manager=context_manager,
            tool_router=tool_router,
            tool_orchestrator=tool_orchestrator,
            binding_provider=binding_provider,
        )
        self._stop = TurnStop(
            store=store, gate=completion_gate, commit=self._commit, before_completion=before_completion,
            lifecycle_hooks=lifecycle_hooks, dispatch_guard=dispatch_guard,
        )
        self._responses = ResponseHandler(
            store=store,
            tool_orchestrator=tool_orchestrator,
            stop=self._stop,
            commit=self._commit,
            fail_turn=self._fail_turn,
        )
        self._context_engine = ContextEngine(
            store=store,
            context_manager=context_manager,
            prepare_step=self._model_runner.prepare,
            capture_step_context=self.capture_step_context,
            restore_step_context=self.restore_step_context,
            dispatch_prepared=self._model_runner.dispatch,
            commit=self._commit,
            budget_status=self._budget_status,
        )

    def restore_turn_context(self, turn_id: str) -> TurnContext:
        return self._requests.restore_turn_context(turn_id)

    def capture_step_context(
        self,
        turn_context: TurnContext,
        *,
        step: int,
        budget_pressure: bool | None = None,
        messages: tuple[ModelMessage, ...] | None = None,
    ) -> StepContext:
        self._requests._capacity_recovery_root = self._capacity_recovery_root
        return self._requests.capture_step_context(
            turn_context, step=step, budget_pressure=budget_pressure, messages=messages
        )

    def restore_step_context(self, turn: TurnContext, operation: ModelOperationSnapshot) -> StepContext:
        return self._requests.restore_step_context(turn, operation)

    def _budget_status(self, turn: TurnContext) -> dict[str, Any]:
        return self._requests._budget_status(turn)

    def attach_event_dispatcher(
        self,
        event_dispatcher: TurnEventDispatcher,
    ) -> None:
        self._event_dispatcher = event_dispatcher
        self._model_runner.attach_event_dispatcher(event_dispatcher)
        if self._tool_orchestrator is not None:
            self._tool_orchestrator.attach_event_dispatcher(event_dispatcher)

    async def run(
        self,
        *,
        turn_id: str,
        user_message: str,
        binding_manifest: Mapping[str, Any],
        input_files: tuple[Mapping[str, Any], ...] = (),
        reference_user_paths: bool = False,
    ) -> TurnResult:
        binding_manifest = dict(binding_manifest)
        binding_manifest.setdefault("model_step_budget", self._max_steps)
        step_limit_from_binding(binding_manifest)
        def admit() -> TurnSnapshot:
            with self._store.tx.transaction():
                if self._dispatch_guard is not None:
                    self._dispatch_guard(self._store)
                return turn_ops.start_turn(
                    self._store, thread_id=self.thread_id, turn_id=turn_id,
                    user_message=user_message, binding_manifest=binding_manifest, input_files=input_files,
                    reference_user_paths=reference_user_paths,
                )

        turn = await self._commit(admit)
        return await self.drive(
            (await self._store.run_async(self.restore_turn_context, turn.turn_id)),
            start_step=1,
        )

    async def _commit[T](self, operation: Callable[[], T]) -> T:
        return await commit_and_publish(self._store, operation, self._event_dispatcher)

    async def _commit_cancellation(self, operation: Callable[[], None]) -> None:
        """Finish cancellation bookkeeping even when cancellation is repeated."""
        task = asyncio.create_task(self._commit(operation))
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
        task.result()

    async def _resolve_approval(
        self,
        *,
        turn_id: str,
        decision: str,
        on_session_grant: Callable[[CommandGrant], None] | None = None,
    ) -> TurnResult | None:
        await self._store.run_async(partial(model_ops.validate_turn_model_requests, self._store), turn_id)
        if self._tool_orchestrator is None:
            raise RuntimeError("Turn has no ToolOrchestrator for approval resume")
        turn = await self._store.run_async(self._store.read_turn, turn_id)
        approval_interactions = await self._store.run_async(
            lambda: [
                interaction
                for interaction in self._store.list_interactions(turn_id)
                if interaction.kind == "tool_approval"
            ]
        )
        if approval_interactions and approval_interactions[-1].status == "resolved":
            prior_decision = approval_interactions[-1].response.get("decision")
            if prior_decision != decision:
                raise RuntimeError(f"decision conflicts with resolved approval: {prior_decision} != {decision}")
            if turn.status == "completed":
                answer = await self._store.run_async(self._store.read_turn_answer, turn_id)
                return TurnResult(
                    thread_id=turn.thread_id,
                    turn_id=turn_id,
                    answer=item_text(answer),
                    status="completed",
                )
            if turn.status == "running" and decision == "approve":
                try:
                    await self._tool_orchestrator.recover_resolved_approval(turn_id=turn_id)
                except (ToolApprovalInvalidatedError, ToolApprovalRequiredError) as invalidated:
                    return TurnResult(
                        thread_id=turn.thread_id,
                        turn_id=turn_id,
                        answer=None,
                        status="paused",
                        interaction_id=invalidated.interaction_id,
                    )
                return None
        if turn.status != "paused":
            raise RuntimeError(f"turn is not paused: {turn_id}")
        try:
            await self._tool_orchestrator.resume_approval(
                turn_id=turn_id,
                decision=decision,
                on_session_grant=on_session_grant,
            )
        except (ToolApprovalInvalidatedError, ToolApprovalRequiredError) as invalidated:
            return TurnResult(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                answer=None,
                status="paused",
                interaction_id=invalidated.interaction_id,
            )
        return None

    async def _execute_saved_request(
        self, *, turn_id: str, operation_id: str, retry: bool = False, automatic_retry: bool = False
    ) -> HarnessModelResponse | TurnResult | None:
        turn = await self._store.run_async(self.restore_turn_context, turn_id)
        operation = next(
            op
            for op in await self._store.run_async(self._store.list_model_operations, turn_id)
            if op.operation_id == operation_id
        )
        expected = "retry_wait" if automatic_retry else "unknown" if retry else "prepared"
        if operation.status != expected:
            raise RuntimeError(f"saved model request changed: expected {expected}, got {operation.status}")
        if await self._commit(partial(self._context_engine.discard_stale_saved_summary, operation_id)):
            return None
        context = await self._store.run_async(self.restore_step_context, turn, operation)
        prepared = await self._model_runner.prepare(context)
        if (prepared.request_hash, prepared.context_hash, prepared.tool_hash, prepared.wire_hash) != (
            operation.request_hash,
            operation.context_hash,
            operation.tool_hash,
            operation.wire_hash,
        ):
            raise RuntimeError("saved model request cannot be reproduced exactly")
        if automatic_retry:
            await self._commit(
                partial(
                    model_retry_ops.prepare_scheduled_retry,
                    self._store,
                    operation_id=operation_id,
                    resource_request=prepared.resource_request,
                )
            )
        elif retry:

            def admit_retry() -> None:
                from agent_runtime.harness import compaction_jobs

                with self._store.tx.transaction():
                    if operation.request_ref.get("compaction_plan", {}).get("job_id") or operation.request_ref.get(
                        "compaction_job_id"
                    ):
                        compaction_jobs.admit_retry(
                            self._store, turn_id=turn_id, operation_id=operation_id, amount=prepared.resource_request
                        )
                    model_ops.prepare_model_retry(self._store, operation_id)

            await self._commit(admit_retry)
        try:
            return await self._model_runner.dispatch(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                operation=operation,
                prepared=prepared,
                allow_protected_budget=retry or context.budget_pressure,
            )
        except TurnInputChangedError:
            return None
        except ContextSourceChangedError:
            await self._commit(partial(self._context_engine.discard_stale_saved_summary, operation_id))
            return None

    async def _wait_model_retry(self, *, turn_id: str, operation_id: str) -> TurnResult | None:
        schedule = await self._store.run_async(model_retry_ops.read_schedule, self._store, operation_id)
        if schedule is None:
            raise RuntimeError("retry wait lost its canonical schedule")
        now = time.time()
        due = time.monotonic() + max(0.0, schedule["retry_at"] - now)
        try:
            while True:
                now = time.time()
                if now < schedule["observed_at"] or now > schedule["deadline_at"]:
                    await self._commit(
                        partial(
                            model_retry_ops.fail_unstarted_retry,
                            self._store,
                            operation_id=operation_id,
                            reason_code="model_retry_deadline_exceeded",
                            message="Automatic retry deadline expired or its clock moved backwards.",
                        )
                    )
                    return None
                if await self._store.run_async(self._store.turn_cancellation_pending, turn_id):
                    return None
                operations = await self._store.run_async(self._store.list_model_operations, turn_id)
                operation = next(op for op in operations if op.operation_id == operation_id)
                revision = await self._store.run_async(
                    self._store.request_input_revision, turn_id, operation.request_ref
                )
                if revision != await self._store.run_async(self._store.user_input_revision, turn_id):
                    discarded = await self._commit(
                        partial(self._context_engine.discard_stale_saved_summary, operation_id)
                    )
                    if not discarded:
                        await self._commit(
                            partial(
                                model_retry_ops.cancel_scheduled_retry,
                                self._store,
                                operation_id=operation_id,
                                reason_code="input_superseded",
                                reason="new user input invalidated the waiting request",
                            )
                        )
                    return None
                remaining = due - time.monotonic()
                if remaining <= 0:
                    break
                # Bound responsiveness to durable input/cancellation from another
                # owner without restarting the persisted server-advice deadline.
                await asyncio.sleep(min(remaining, 0.25))
            return_value = await self._execute_saved_request(
                turn_id=turn_id, operation_id=operation_id, automatic_retry=True
            )
            return return_value if isinstance(return_value, TurnResult) else None
        except TurnCancellationPendingError:
            return None
        except TurnInputChangedError:
            if await self._commit(partial(self._context_engine.discard_stale_saved_summary, operation_id)):
                return None
            await self._commit(
                partial(
                    model_retry_ops.cancel_scheduled_retry,
                    self._store,
                    operation_id=operation_id,
                    reason_code="input_superseded",
                    reason="input changed during retry admission",
                )
            )
            return None
        except ContextSourceChangedError:
            if await self._commit(partial(self._context_engine.discard_stale_saved_summary, operation_id)):
                return None
            await self._commit(
                partial(
                    model_retry_ops.cancel_scheduled_retry,
                    self._store,
                    operation_id=operation_id,
                    reason_code="context_source_superseded",
                    reason="summary source changed",
                )
            )
            return None
        except BudgetLimitExceededError as exc:
            await self._commit(
                partial(
                    model_retry_ops.fail_unstarted_retry,
                    self._store,
                    operation_id=operation_id,
                    reason_code="model_retry_budget_exhausted",
                    message=str(exc),
                )
            )
            return None
        except asyncio.CancelledError:

            def record_wait_cancellation() -> None:
                turn_ops.request_turn_cancellation(
                    self._store, turn_id=turn_id, reason="cancelled during model retry wait"
                )

            await self._commit_cancellation(record_wait_cancellation)
            raise

    async def respond_interaction(
        self,
        *,
        turn_id: str,
        request_id: str,
        response: str,
    ) -> TurnResult | None:
        # Validate retained request provenance before accepting a new reply.
        await self._store.run_async(self.restore_turn_context, turn_id)
        turn = await self._store.run_async(self._store.read_turn, turn_id)
        interaction = await self._store.run_async(self._store.read_interaction, request_id)
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
                answer = await self._store.run_async(self._store.read_turn_answer, turn_id)
                return TurnResult(
                    thread_id=turn.thread_id,
                    turn_id=turn_id,
                    answer=item_text(answer),
                    status="completed",
                )
            raise RuntimeError(f"resolved {interaction.kind} is already being processed")
        if interaction.kind == "clarification":
            await self._commit(
                lambda: turn_ops.resolve_clarification(
                    self._store,
                    turn_id=turn_id,
                    request_id=request_id,
                    response=response,
                )
            )
        else:
            await self._commit(
                lambda: turn_ops.resolve_choice(
                    self._store,
                    turn_id=turn_id,
                    request_id=request_id,
                    selection=response,
                )
            )
        return None

    def agent_step_count(self, turn_id: str) -> int:
        return agent_step_count(self._store, turn_id)

    async def drive(
        self,
        turn_context: TurnContext | None = None,
        *,
        turn_id: str | None = None,
        start_step: int | None = None,
        action: str | None = None,
        user_input: str | None = None,
        expected_request_id: str | None = None,
        automatic: bool = False,
        authorization_revision: int | None = None,
        on_session_grant: Callable[[CommandGrant], None] | None = None,
        command_sessions: Any = None,
        cancel_live_commands: Callable[[], Awaitable[None]] | None = None,
    ) -> TurnResult:
        """The only Turn advancement loop, for new execution and durable recovery."""
        if turn_context is None:
            if turn_id is None:
                raise TypeError("drive requires a Turn context or identity")
            if action == "abort":
                current_turn = await self._store.run_async(self._store.read_turn, turn_id)
                turn_context = TurnContext(self.thread_id, turn_id, current_turn.binding_manifest)
            else:
                turn_context = await self._store.run_async(self.restore_turn_context, turn_id)
        turn_id = turn_context.turn_id
        thread_id = turn_context.thread_id
        if thread_id != self.thread_id:
            raise RuntimeError("Turn belongs to a different Session")
        permit_retry = False
        if action is not None:
            # A queued reply authorizes the observed Turn revision. Replaying
            # its delivery may finish already committed work, but cannot reuse
            # that reply to approve a later interaction or retry another attempt.
            if authorization_revision is not None:
                current = await self._store.run_async(self._store.read_turn, turn_id)
                automatic = automatic or current.applied_thread_sequence != authorization_revision
            offered = await self._store.run_async(recovery_plan, self._store, turn_id)
            acknowledged = False
            if expected_request_id is not None and offered.request_id != expected_request_id:
                try:
                    receipt = await self._store.run_async(self._store.read_interaction, expected_request_id)
                except KeyError:
                    receipt = None
                if receipt is not None and receipt.turn_id == turn_id and receipt.status == "resolved":
                    if receipt.kind in {"clarification", "choice"}:
                        field = "text" if receipt.kind == "clarification" else "selection"
                        acknowledged = action == "continue" and receipt.response.get(field) == user_input
                    elif receipt.kind == "tool_approval":
                        reply = {"allow_once": "approve", "allow_session": "approve", "approve": "approve"}.get(
                            action, action
                        )
                        acknowledged = receipt.response.get("decision") == reply
                    if not acknowledged:
                        raise RuntimeError("response conflicts with the committed interaction receipt")
                if not acknowledged and offered.operation_id != expected_request_id:
                    raise ValueError("recovery request changed; inspect current state before responding")
            if offered.status in {"running", "paused", "interrupted"}:
                await self._commit(lambda: recover_owner_loss(self._store, turn_id, abort=action == "abort"))
            if action in {"continue", "retry"} and command_sessions is not None:
                assert self._tool_orchestrator is not None
                while await self._tool_orchestrator.reconcile_command_observation(command_sessions, turn_id):
                    await self._commit(lambda: recover_owner_loss(self._store, turn_id))
            offered = await self._store.run_async(recovery_plan, self._store, turn_id)
            if automatic and not offered.automatically_resumable:
                return TurnResult(
                    thread_id=thread_id, turn_id=turn_id, answer=None,
                    status=offered.status, interaction_id=offered.request_id,
                )
            if acknowledged:
                # The old response is already committed. Continue from current
                # facts without applying it to a newer outstanding interaction.
                return await self.drive(
                    turn_context,
                    start_step=start_step,
                    command_sessions=command_sessions,
                    cancel_live_commands=cancel_live_commands,
                )
            if action == "abort":
                if cancel_live_commands is not None:
                    await cancel_live_commands()
                if not offered.accepts("abort"):
                    raise ConversationRecoveryRequiredError(f"取消未完成：{offered.reason_code}；请先核对执行状态。")
                cancelled = await self._commit(lambda: turn_ops.cancel_turn(self._store, turn_id=turn_id))
                return TurnResult(thread_id=thread_id, turn_id=turn_id, answer=None, status=cancelled.status)
            if offered.kind == ContinuationKind.RECONCILE and action in {"continue", "retry"}:
                return TurnResult(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    answer=None,
                    status=offered.status,
                    interaction_id=offered.request_id,
                )
            if not offered.accepts(action):
                raise RuntimeError(
                    f"resume action {action!r} is unavailable: {offered.reason_code}; "
                    f"available actions: {', '.join(offered.actions) or 'none'}"
                )
            if offered.kind == ContinuationKind.CHILD_WAIT:
                await self._commit(lambda: turn_ops.resume_child_wait(self._store, turn_id=turn_id))
            elif offered.kind == ContinuationKind.RESUME_CANCELLED:
                await self._commit(lambda: turn_ops.resume_cancelled_turn(self._store, turn_id=turn_id))
            elif offered.kind in {
                ContinuationKind.APPROVAL,
                ContinuationKind.APPROVAL_RESOLVED,
                ContinuationKind.APPROVED_READY,
            }:
                approval_decision = {
                    "allow_once": "approve",
                    "approve": "approve",
                    "allow_session": "approve",
                    "deny": "deny",
                    "recheck": "recheck",
                    "continue": "approve",
                }.get(action, action)
                resolved = await self._resolve_approval(
                    turn_id=turn_id,
                    decision=approval_decision,
                    on_session_grant=on_session_grant if action == "allow_session" else None,
                )
                if resolved is not None:
                    return resolved
            elif offered.kind == ContinuationKind.INPUT:
                if user_input is None or offered.request_id is None:
                    raise ValueError("input resume requires action=continue and user_input")
                answered = await self.respond_interaction(
                    turn_id=turn_id, request_id=offered.request_id, response=user_input
                )
                if answered is not None:
                    return answered
            permit_retry = offered.kind == ContinuationKind.RETRY_MODEL
        await self._commit(lambda: adopt_execution_protocol(self._store, turn_id))
        first_step = start_step
        while True:
            decision = await self._store.run_async(recovery_plan, self._store, turn_id)
            if decision.kind == ContinuationKind.CANCEL:
                await self._commit(
                    lambda: turn_ops.request_turn_cancellation(
                        self._store, turn_id=turn_id, reason="Execution authority revoked"
                    )
                )
                for operation in await self._store.run_async(self._store.list_model_operations, turn_id):
                    schedule = await self._store.run_async(
                        model_retry_ops.read_schedule, self._store, operation.operation_id
                    )
                    if operation.status in {"retry_wait", "prepared"} and schedule is not None:
                        await self._commit(
                            partial(
                                model_retry_ops.cancel_scheduled_retry,
                                self._store,
                                operation_id=operation.operation_id,
                                reason="Turn cancellation requested",
                            )
                        )
                await self._commit(lambda: recover_owner_loss(self._store, turn_id, abort=True))
                current = await self._store.run_async(self._store.read_turn, turn_id)
                if current.status == "running":
                    # A reserved child may have no execution-ownership binding
                    # yet. Recovery already excluded live/unknown tool effects
                    # and active model leases before offering CANCEL.
                    await self._commit(
                        lambda: turn_ops.pause_turn(self._store, turn_id=turn_id, reason="Unstarted work cancelled")
                    )
                cancelled = await self._commit(lambda: turn_ops.cancel_turn(self._store, turn_id=turn_id))
                return TurnResult(thread_id=thread_id, turn_id=turn_id, answer=None, status=cancelled.status)
            if decision.kind in {ContinuationKind.TERMINAL, ContinuationKind.APPROVAL_RESOLVED}:
                current = await self._store.run_async(self._store.read_turn, turn_id)
                answer = None
                if current.status == "completed":
                    answer = item_text(await self._store.run_async(self._store.read_turn_answer, turn_id))
                return TurnResult(thread_id=thread_id, turn_id=turn_id, answer=answer, status=current.status)
            if decision.kind == ContinuationKind.COMMITTED_RESPONSE:
                assert decision.operation_id is not None
                try:
                    result = await self._responses._consume_response(
                        turn_id=turn_id, operation_id=decision.operation_id
                    )
                except TurnCancellationPendingError:
                    continue
                if result is not None:
                    return result
                first_step = None
                continue
            if decision.kind == ContinuationKind.WAIT_RETRY:
                assert decision.operation_id is not None
                try:
                    waiting_result = await self._wait_model_retry(turn_id=turn_id, operation_id=decision.operation_id)
                except ModelRetryScheduledError:
                    continue
                if waiting_result is not None:
                    return waiting_result
                first_step = None
                continue
            if decision.kind in {ContinuationKind.PREPARED_MODEL, ContinuationKind.RETRY_MODEL}:
                if decision.kind == ContinuationKind.RETRY_MODEL and not permit_retry:
                    return TurnResult(thread_id=thread_id, turn_id=turn_id, answer=None, status=decision.status)
                assert decision.operation_id is not None
                try:
                    dispatched = await self._execute_saved_request(
                        turn_id=turn_id,
                        operation_id=decision.operation_id,
                        retry=decision.kind == ContinuationKind.RETRY_MODEL,
                    )
                except (ModelContextOverflowError, ModelRetryScheduledError, TurnCancellationPendingError):
                    continue
                permit_retry = False
                if isinstance(dispatched, TurnResult):
                    return dispatched
                first_step = None
                continue
            if decision.kind in {
                ContinuationKind.RECONCILE,
                ContinuationKind.APPROVAL,
                ContinuationKind.INPUT,
                ContinuationKind.UNSUPPORTED_REQUEST,
                ContinuationKind.UNAVAILABLE,
            }:
                if not (
                    decision.reason_code == "no_resume_point"
                    and decision.status == "running"
                    and first_step is not None
                ):
                    return TurnResult(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        answer=None,
                        status=decision.status,
                        interaction_id=decision.request_id,
                    )
            count = await self._store.run_async(self.agent_step_count, turn_id)
            step = count + 1 if first_step is None else first_step
            if decision.kind in {ContinuationKind.CONTEXT_OVERFLOW, ContinuationKind.RESUME_COMPACTION}:
                models = await self._store.run_async(self._store.list_model_operations, turn_id)
                if models and (
                    decision.kind == ContinuationKind.CONTEXT_OVERFLOW
                    or models[-1].request_ref.get("purpose") == "context_summary"
                ):
                    step = models[-1].request_ref.get("step_snapshot", {}).get("step", max(1, count))
            remaining_tokens = await self._store.run_async(
                lambda: self._store.read_budget_state(turn_id).remaining("tokens")
            )
            if (
                remaining_tokens is not None and remaining_tokens < 1
                and decision.kind not in {ContinuationKind.RESUME_COMPACTION, ContinuationKind.CONTEXT_OVERFLOW}
            ):
                return await self._fail_turn(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    reason_code="model_token_budget_exhausted",
                    message="Turn exhausted its frozen model token budget.",
                )
            binding = turn_context.binding_manifest
            step_budget = step_limit_from_binding(binding)
            if step_budget is not None and step > step_budget:
                return await self._fail_turn(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    reason_code="model_step_budget_exhausted",
                    message=await self._store.run_async(self._step_budget_receipt, turn_id, step_budget),
                )
            reactive = decision.kind == ContinuationKind.CONTEXT_OVERFLOW
            rejected = None
            rejections = 0
            self._capacity_recovery_root = await self._store.run_async(active_capacity_root, self._store, turn_id)
            if reactive:
                operations = await self._store.run_async(self._store.list_model_operations, turn_id)
                rejected = next(op for op in reversed(operations) if op.request_ref.get("purpose") != "context_summary")
                root_id = rejected.request_ref.get("capacity_recovery_root_id", rejected.operation_id)
                by_id = {op.operation_id: op for op in operations}

                def count_rejections(
                    by_id: Mapping[str, ModelOperationSnapshot] = by_id, root_id: str = root_id
                ) -> int:
                    return sum(
                        r.record_type == "model_attempt_rejected"
                        and r.payload.get("error_type") == "context_overflow"
                        and r.payload.get("operation_id") in by_id
                        and by_id[r.payload["operation_id"]].request_ref.get(
                            "capacity_recovery_root_id", r.payload["operation_id"]
                        )
                        == root_id
                        for r in self._store.list_turn_records(turn_id)
                    )

                rejections = await self._store.run_async(count_rejections)
                if rejections > 2:
                    return await self._fail_turn(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        reason_code="model_context_window_exceeded",
                        message="Context overflow persisted after both recovery paths.",
                    )
                self._capacity_recovery_root = root_id
            observation = await self._store.run_async(self._context_engine.observation_revision, turn_id)
            try:
                if rejected is not None and rejections == 2:
                    context, prepared = await self._context_engine.retry_overflow_with_archived_results(
                        turn_context, step=step, rejected=rejected,
                    )
                else:
                    context, prepared = await self._context_engine.prepare_or_resume(
                        turn_context, step=step, pressure_reason="provider_overflow" if reactive else None,
                        rejected_request_bytes=(
                            rejected.request_ref.get("serialized_input_bytes") if rejected is not None else None
                        ),
                    )
            except ModelRetryScheduledError:
                continue
            except SummaryStoppedError as stopped:
                return stopped.result
            except TurnCancellationPendingError:
                continue
            except ContextSourceChangedError as exc:
                current_observation = await self._store.run_async(self._context_engine.observation_revision, turn_id)
                if current_observation == observation:
                    return await self._fail_turn(
                        thread_id=thread_id, turn_id=turn_id, reason_code="context_source_changed", message=str(exc)
                    )
                # New canonical input invalidated the candidate. Re-evaluate
                # without resetting the same failure chain or its budget.
                continue
            except (ContextBudgetExceededError, PricingUnavailableError) as exc:
                return await self._fail_turn(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    reason_code=getattr(
                        exc,
                        "reason_code",
                        "pricing_unavailable" if isinstance(exc, PricingUnavailableError) else "context_source_changed",
                    ),
                    message=str(exc),
                )
            if context.input_revision != await self._store.run_async(self._store.user_input_revision, turn_id):
                continue
            operation = await self._commit(
                partial(
                    model_ops.prepare_model_operation,
                    self._store,
                    turn_id=turn_id,
                    request_hash=prepared.request_hash,
                    context_hash=prepared.context_hash,
                    tool_hash=prepared.tool_hash,
                    wire_hash=prepared.wire_hash,
                    request_ref={
                        **prepared.request_ref,
                        "request_id": prepared.request_ref.get("request_id") or f"{turn_id}:step:{step}",
                    },
                )
            )
            try:
                dispatched = await self._model_runner.dispatch(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    operation=operation,
                    prepared=prepared,
                    allow_protected_budget=context.budget_pressure,
                )
            except (TurnInputChangedError, TurnCancellationPendingError):
                first_step = None
                continue
            except (ModelContextOverflowError, ModelRetryScheduledError):
                first_step = None
                continue
            if isinstance(dispatched, TurnResult):
                return dispatched
            first_step = None
            # The committed response is consumed by this same loop, including after restart.

    def _step_budget_receipt(self, turn_id: str, limit: int) -> str:
        results = [
            item
            for item in self._store.list_items(turn_id)
            if item.kind == "tool_result" and item.status == "completed"
        ]
        failed = sum(item.payload.get("is_error") is True for item in results)
        changed = any(workspace_changed(item.payload.get("metadata", {})) is True for item in results)
        return (
            f"本轮已达到 {limit} 步执行上限，尚未完成收尾。\n"
            f"已记录 {len(results)} 个工具结果，其中 {failed} 个报错。"
            + ("已记录实际工作区修改；现有修改已保留。" if changed else "未记录实际工作区修改。")
            + "\n本条为执行回执，任务是否满足需求请依据实际检查结果判断。可在当前会话继续处理剩余工作。"
        )

    async def _fail_turn(
        self,
        *,
        thread_id: str,
        turn_id: str,
        reason_code: str,
        message: str,
    ) -> TurnResult:
        await self._commit(
            lambda: turn_ops.fail_turn(
                self._store,
                turn_id=turn_id,
                reason_code=reason_code,
                message=message,
            )
        )
        return TurnResult(
            thread_id=thread_id,
            turn_id=turn_id,
            answer=None,
            status="failed",
        )

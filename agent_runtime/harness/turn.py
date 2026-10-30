"""Turn execution and ephemeral model/tool Step contexts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Any
from uuid import uuid4

from agent_runtime.approval import CommandGrant
from agent_runtime.budget import (
    PricingUnavailableError,
    budget_pressure_active,
)
from agent_runtime.core.messages import model_message_payload, restore_model_message
from agent_runtime.harness import completion_ops, model_ops, tool_ops, turn_ops
from agent_runtime.harness.context_engine import ContextEngine, SummaryStoppedError
from agent_runtime.harness.control import ContextSnapshot, StepControl, append_context_updates, append_runtime_state
from agent_runtime.harness.events import commit_and_publish
from agent_runtime.harness.execution_context import StepContext, TurnContext
from agent_runtime.harness.execution_facts import admit_tool_batch, adopt_execution_protocol, response_handled
from agent_runtime.harness.model_runner import ModelRunner
from agent_runtime.harness.operation_values import TurnCancellationPendingError, TurnInputChangedError
from agent_runtime.harness.ownership import ConversationRecoveryRequiredError, recover_owner_loss
from agent_runtime.harness.protocol import (
    DEFAULT_MAX_STEPS,
    BindingProvider,
    CompletionDecision,
    CompletionGate,
    CompletionProposal,
    ContextBudgetExceededError,
    ContextManager,
    ContextSourceChangedError,
    HarnessModel,
    HarnessModelResponse,
    ModelContextOverflowError,
    ModelMessage,
    ModelToolCall,
    PreparedModelCall,
    ToolRouter,
    TurnResult,
    item_text,
    restore_model_response,
)
from agent_runtime.harness.records import ItemSnapshot, ModelOperationSnapshot
from agent_runtime.harness.recovery import ContinuationKind, active_capacity_root, recovery_plan
from agent_runtime.harness.runtime_store import RuntimeStore
from agent_runtime.harness.tool_orchestrator import (
    ToolApprovalInvalidatedError,
    ToolApprovalRequiredError,
    ToolOrchestrator,
)
from agent_runtime.modeling.contracts import context_capacity_policy
from agent_runtime.runtime_clock import capture_runtime_clock
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin

_BUDGET_PRESSURE_MESSAGE = (
    "Runtime budget pressure is active. Stop broad exploration and converge on "
    "the best defensible completion from evidence already gathered. Prefer a "
    "final answer; use additional tools only when strictly required to avoid "
    "an incorrect or unsafe result."
)


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
        max_steps: int = DEFAULT_MAX_STEPS,
        worker_id: str | None = None,
        model_lease_seconds: float = 300.0,
        event_dispatcher: TurnEventDispatcher | None = None,
        binding_provider: BindingProvider | None = None,
        prepare_binding: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
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
        self._binding_provider = binding_provider
        self._prepare_binding = prepare_binding
        self._capacity_recovery_root: str | None = None
        self._model_runner = ModelRunner(
            store=store,
            model=model,
            worker_id=self._worker_id,
            model_lease_seconds=model_lease_seconds,
            event_dispatcher=event_dispatcher,
            context_manager=context_manager,
            prepare_binding=prepare_binding,
        )
        self._context_engine = ContextEngine(
            store=store,
            context_manager=context_manager,
            prepare_step=self._model_runner.prepare,
            capture_step_context=self.capture_step_context,
            restore_step_context=self.restore_step_context,
            dispatch_prepared=self._model_runner.dispatch,
            commit=self._commit,
            max_steps=max_steps,
            budget_status=self._budget_status,
        )

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
    ) -> TurnResult:
        turn = await self._commit(
            lambda: turn_ops.start_turn(
                self._store,
                thread_id=self.thread_id,
                turn_id=turn_id,
                user_message=user_message,
                binding_manifest=binding_manifest,
                input_files=input_files,
            )
        )
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

    def restore_turn_context(self, turn_id: str) -> TurnContext:
        turn = self._store.read_turn(turn_id)
        if turn.thread_id != self.thread_id:
            raise RuntimeError("Turn belongs to a different Session")
        if turn.status in {"running", "paused", "interrupted"}:
            model_ops.validate_turn_model_requests(self._store, turn_id)
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
        budget_pressure: bool | None = None,
        messages: tuple[ModelMessage, ...] | None = None,
    ) -> StepContext:
        if turn_context.thread_id != self.thread_id:
            raise RuntimeError("Turn belongs to a different Session")
        input_revision = self._store.user_input_revision(turn_context.turn_id)
        state = self._store.read_budget_state(turn_context.turn_id)
        if budget_pressure is None:
            pressure = budget_pressure_active(state)
        elif type(budget_pressure) is not bool:
            raise TypeError("budget_pressure must be a bool or None")
        else:
            pressure = budget_pressure
        if messages is None:
            messages = self._context_manager.build(turn_context.turn_id)
        from agent_runtime.project_instructions import instruction_context, restored_instructions

        project_instructions = restored_instructions(
            turn_context.binding_manifest,
            self._store.list_items(turn_context.turn_id),
        )
        execution_policy = turn_context.binding_manifest.get("execution_context_policy", {})
        append_context = execution_policy.get("context_updates") == "append-v1"
        if not append_context and project_instructions is not None and project_instructions["entries"]:
            messages = (ModelMessage(role="context", body=instruction_context(project_instructions)), *messages)
        snapshot = turn_context.binding_manifest.get("memory_snapshot")
        if not append_context and snapshot is not None:
            from agent_runtime.harness.memory import memory_navigation

            messages = (ModelMessage(role="context", body=memory_navigation(snapshot)), *messages)
        context_updates: tuple[ModelMessage, ...] = ()
        if append_context:
            messages, context_updates = self._append_context_updates(messages, project_instructions, snapshot)
        append_state = execution_policy.get("cache_continuity") == "append-v1"
        runtime_capacity = execution_policy.get("revision") == "working-context-v2"
        clock = capture_runtime_clock()
        if execution_policy.get("revision") in {"working-context-v1", "working-context-v2"} and not append_state:
            from agent_runtime.harness.context import RolloutContextManager

            if isinstance(self._context_manager, RolloutContextManager):
                plan = self._context_manager.current_plan(turn_context.turn_id)
                if plan is not None:
                    from agent_runtime.planning import model_visible_plan

                    messages = (
                        *messages,
                        ModelMessage(
                            role="context",
                            body=(
                                "Current advisory plan (canonical runtime state):\n"
                                + json.dumps(
                                    model_visible_plan(plan.payload["plan"]), ensure_ascii=False, sort_keys=True
                                )
                                + "\nKeep this plan current as phases finish or constraints change. "
                                "A completed plan is not proof of task correctness."
                            ),
                        ),
                    )
        if pressure and not append_state:
            messages = (
                *messages,
                ModelMessage(
                    role="context",
                    body=(
                        "Resource budget is running low. Prioritize remaining required actions and verification. "
                        "Report unfinished work honestly if the limit is reached."
                        if runtime_capacity
                        else _BUDGET_PRESSURE_MESSAGE
                    ),
                ),
            )
        completion = turn_context.binding_manifest.get("completion_policy", {})
        if not append_state and completion.get("revision") == "completion-v2" and completion.get("acceptance_command"):
            messages = (
                *messages,
                ModelMessage(
                    role="context",
                    body=(
                        "User-configured acceptance contract (immutable for this Turn):\n"
                        + json.dumps(dict(completion), ensure_ascii=False, sort_keys=True)
                        + "\nRun acceptance_command exactly via run_command in the workspace root. "
                        "The latest execution must pass after the latest workspace change. "
                        "Other commands or self-reported checks do not substitute for this contract. "
                        "Normal tool permissions and approval still apply."
                    ),
                ),
            )
        step_policy = turn_context.binding_manifest.get("step_budget_policy") == {"revision": "pressure-v1"}
        if (
            not append_state
            and not runtime_capacity
            and (context_capacity_policy(turn_context.binding_manifest) is not None or step_policy)
        ):
            messages = (
                *messages,
                ModelMessage(
                    role="context",
                    body="Runtime budget status:\n" + json.dumps(self._budget_status(turn_context), sort_keys=True),
                ),
            )
        if step_policy and not append_state:
            remaining_steps = min(self._max_steps, turn_context.binding_manifest["model_step_budget"]) - step + 1
            if remaining_steps <= 2:
                messages = (
                    *messages,
                    ModelMessage(
                        role="context",
                        body=(
                            f"Step budget is nearly exhausted: {remaining_steps} request(s) "
                            "remaining including this one. "
                            "Converge using existing evidence. Return a final answer when justified; "
                            "state any unfinished "
                            "work and blockers honestly. Do not claim unperformed verification or extend the budget."
                        ),
                    ),
                )
        update = None
        if append_state:
            messages, update = self._append_runtime_state(
                turn_context,
                messages,
                step=step,
                pressure=pressure,
                clock=clock,
            )
        tools = (
            ()
            if self._tool_router is None
            else self._tool_router.select(
                turn_id=turn_context.turn_id,
                messages=messages,
            )
        )
        binding = dict(
            turn_context.binding_manifest
            if self._binding_provider is None
            else self._binding_provider.snapshot(thread_id=self.thread_id, turn_id=turn_context.turn_id)
        )
        # Old Turns do not acquire recovery by reopening with new defaults.
        binding.pop("model_response_recovery", None)
        binding.pop("project_instructions", None)
        binding.pop("tool_failure_recovery", None)
        binding.pop("step_budget_policy", None)
        binding.pop("execution_context_policy", None)
        for name in (
            "completion_policy",
            "model_response_recovery",
            "tool_failure_recovery",
            "step_budget_policy",
            "execution_context_policy",
            "model_step_budget",
            "model_token_budget_total",
            "model_cost_budget_total_micros",
            "budget_root_turn_id",
            "budget_parent_turn_id",
            "web_limits",
            "context_capacity_policy",
            "auto_compact_token_limit",
            "memory_snapshot",
            "memory_generation_allowed",
        ):
            if name in turn_context.binding_manifest:
                binding[name] = turn_context.binding_manifest[name]
        if project_instructions is not None:
            binding["project_instructions"] = project_instructions
        # Freeze clock facts in the saved step, just like its messages and tools.
        # Repreparing/recovering this request must not read a different clock.
        binding["runtime_clock"] = clock
        binding.pop("context_updates", None)
        binding.pop("context_update_messages", None)
        if context_updates:
            binding["context_updates"] = tuple(m.content for m in context_updates)
            binding["context_update_messages"] = [model_message_payload(m) for m in context_updates]
        binding.pop("runtime_context_update", None)
        binding.pop("runtime_context_message", None)
        if update is not None:
            binding["runtime_context_update"] = update.content
            binding["runtime_context_message"] = model_message_payload(update)
        return StepContext(
            turn=turn_context,
            step=step,
            binding_manifest=binding,
            messages=messages,
            tools=tools,
            model_token_budget_remaining=state.remaining("tokens"),
            budget_pressure=pressure,
            purpose=turn_context.binding_manifest.get("maintenance_purpose", "agent_step"),
            input_token_limit=turn_context.binding_manifest.get("memory_input_tokens"),
            output_token_limit=turn_context.binding_manifest.get("memory_output_tokens"),
            input_revision=input_revision,
            request_id=(
                f"{turn_context.turn_id}:step:{step}:recovery:{self._capacity_recovery_root}"
                if self._capacity_recovery_root is not None
                else None
            ),
            control=StepControl(
                retry_count=1 if self._capacity_recovery_root is not None else 0,
                recovery_root_id=self._capacity_recovery_root,
                budget_status=self._budget_status(turn_context),
                context_snapshot=ContextSnapshot.from_messages(messages),
            ),
        )

    @staticmethod
    def _append_context_updates(
        messages: tuple[ModelMessage, ...], instructions: Mapping[str, Any] | None, memory: Mapping[str, Any] | None
    ) -> tuple[tuple[ModelMessage, ...], tuple[ModelMessage, ...]]:
        return append_context_updates(messages, instructions, memory)

    def _append_runtime_state(
        self,
        turn: TurnContext,
        messages: tuple[ModelMessage, ...],
        *,
        step: int,
        pressure: bool,
        clock: Mapping[str, Any],
    ) -> tuple[tuple[ModelMessage, ...], ModelMessage | None]:
        """Append changed facts; old updates are projected from durable request Items.

        Compaction removes old updates with the rest of covered history. Rebuild
        the baseline from visible messages so the first post-compaction update
        contains every current fact. No speculative preparation mutates storage.
        """
        from agent_runtime.harness.context import RolloutContextManager
        from agent_runtime.planning import model_visible_plan

        plan = (
            self._context_manager.current_plan(turn.turn_id)
            if isinstance(self._context_manager, RolloutContextManager)
            else None
        )
        state: dict[str, Any] = {
            "Current runtime clock facts": dict(clock),
            "Runtime budget status": self._budget_status(turn),
            "Current advisory plan": None if plan is None else model_visible_plan(plan.payload["plan"]),
            "budget_pressure": pressure,
            "requests_remaining_including_this": max(
                0,
                min(
                    self._max_steps,
                    turn.binding_manifest.get("model_step_budget", self._max_steps),
                )
                - step
                + 1,
            ),
        }
        if turn.binding_manifest.get("execution_context_policy", {}).get("revision") == "working-context-v2":
            remaining = state.pop("requests_remaining_including_this")
            state.pop("Runtime budget status")
            state.pop("budget_pressure")
            reminder = (
                "Resource budget is running low. Prioritize remaining required actions and verification. "
                "Report unfinished work honestly if the limit is reached."
                if pressure
                else None
            )
            if remaining <= 2:
                reminder = (
                    f"At most {remaining} model request(s) remain including this one. "
                    "Prioritize remaining required actions and verification; report unfinished work honestly."
                )
            state["resource_reminder"] = reminder
        completion = turn.binding_manifest.get("completion_policy", {})
        state["acceptance_contract"] = dict(completion) if completion.get("acceptance_command") else None
        return append_runtime_state(messages, state, acceptance=bool(completion.get("acceptance_command")))

    def _budget_status(self, turn: TurnContext) -> dict[str, Any]:
        state = self._store.read_budget_state(turn.turn_id)
        web = turn.binding_manifest.get("web_limits")
        web_remaining: int | str = "unset"
        if isinstance(web, Mapping) and isinstance(web.get("turn_requests"), int):
            used = self._store.read_resource_usage(turn.turn_id, "public_web_requests")["used"]
            web_remaining = max(0, web["turn_requests"] - used)
        return {
            "agent_steps_remaining": max(
                0, turn.binding_manifest.get("model_step_budget", self._max_steps) - self.agent_step_count(turn.turn_id)
            ),
            "model_requests_remaining": state.remaining("model_calls")
            if state.limits.model_calls is not None
            else "unset",
            "web_requests_remaining": web_remaining,
            "tokens_remaining": state.remaining("tokens") if state.limits.tokens is not None else "unset",
            "cost_remaining_micros": state.remaining("cost_micros")
            if state.limits.cost_micros is not None
            else "unset",
        }

    def restore_step_context(self, turn: TurnContext, operation: ModelOperationSnapshot) -> StepContext:
        snapshot = operation.request_ref.get("step_snapshot")
        if not isinstance(snapshot, Mapping):
            raise RuntimeError("model operation has no durable Step snapshot")
        # Rebuild the original request, never recapture current Session settings.
        messages = tuple(restore_model_message(item) for item in snapshot["messages"])
        tools = (
            ()
            if self._tool_orchestrator is None
            else self._tool_orchestrator.restore_tools({item["name"]: item["revision"] for item in snapshot["tools"]})
        )
        if snapshot["tools"] and not tools:
            raise RuntimeError("original request tools are unavailable")
        return StepContext(
            turn=turn,
            step=snapshot["step"],
            binding_manifest=snapshot["binding_manifest"],
            messages=messages,
            tools=tools,
            model_token_budget_remaining=snapshot["model_token_budget_remaining"],
            budget_pressure=snapshot["budget_pressure"],
            purpose=snapshot.get("purpose", "agent_step"),
            request_id=snapshot.get("request_id"),
            input_token_limit=snapshot.get("input_token_limit"),
            output_token_limit=snapshot.get("output_token_limit"),
            compaction_plan=snapshot.get("compaction_plan"),
            continuation_summary_role=snapshot.get("continuation_summary_role", "context"),
            input_revision=self._store.request_input_revision(turn.turn_id, operation.request_ref),
            control=StepControl.restore(snapshot.get("control")),
            frozen_request=(
                None
                if not operation.request_ref.get("frozen_request_artifact")
                else json.loads(self._store.read_artifact(operation.request_ref["frozen_request_artifact"]))
            ),
        )

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
        self, *, turn_id: str, operation_id: str, retry: bool = False
    ) -> HarnessModelResponse | TurnResult | None:
        turn = await self._store.run_async(self.restore_turn_context, turn_id)
        operation = next(
            op
            for op in await self._store.run_async(self._store.list_model_operations, turn_id)
            if op.operation_id == operation_id
        )
        expected = "unknown" if retry else "prepared"
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
        if retry:

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

    async def _consume_response(self, *, turn_id: str, operation_id: str) -> TurnResult | None:
        """Continue after a crash between response commit and response handling."""

        turn = await self._store.run_async(self._store.read_turn, turn_id)
        if turn.status != "running":
            raise RuntimeError("committed response recovery requires a running Turn")
        operations = await self._store.run_async(self._store.list_model_operations, turn_id)
        if not operations:
            raise RuntimeError("Turn has no committed model response to recover")
        operation = next(op for op in operations if op.operation_id == operation_id)
        if operation.status != "completed" or operation.response_item_id is None:
            raise RuntimeError("latest model operation has no canonical completed response")
        if operation.request_ref.get("purpose") == "context_summary":
            return None
        response_id = operation.response_item_id
        assert response_id is not None
        response_item = await self._store.run_async(self._store.read_item, operation.response_item_id)
        response = _response_from_committed_item(response_item)
        budget = await self._store.run_async(self._store.read_budget_state, turn_id)
        for resource, reason in (
            ("tokens", "model_token_budget_exhausted"),
            ("cost_micros", "model_cost_budget_exhausted"),
        ):
            limit = getattr(budget.limits, resource)
            actual = budget.used.total_tokens if resource == "tokens" else budget.used.cost_micros
            if limit is not None and actual > limit:
                return await self._fail_turn(
                    thread_id=turn.thread_id,
                    turn_id=turn_id,
                    reason_code=reason,
                    message=f"Turn actual model usage exceeded its frozen {resource} limit.",
                )
        if response.status == "incomplete" or not response.tool_calls:
            prepared = PreparedModelCall(
                request_hash=operation.request_hash,
                context_hash=operation.context_hash,
                tool_hash=operation.tool_hash,
                wire_hash=operation.wire_hash,
                request_ref=operation.request_ref,
            )
            handled = await self._handle_model_response(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                response=response,
                prepared=prepared,
            )
            await self._commit(
                lambda: response_handled(
                    self._store,
                    turn_id=turn_id,
                    operation_id=operation_id,
                    response_item_id=response_id,
                    disposition="answer_or_continue",
                )
            )
            return handled
        tool_operations = await self._store.run_async(
            lambda: {
                tool_operation.tool_call_id: tool_operation
                for tool_operation in self._store.list_tool_operations(turn_id)
            }
        )
        pending_calls: list[ModelToolCall] = []
        committed_results = await self._store.run_async(
            lambda: {
                item.payload.get("tool_call_id")
                for item in self._store.list_items(turn_id)
                if item.kind == "tool_result" and item.status == "completed"
            }
        )
        for call in response.tool_calls:
            if call.id in committed_results:
                continue
            tool_operation = tool_operations.get(call.id)
            if tool_operation is None:
                pending_calls.append(call)
                continue
            if tool_operation.result_item_id is None:
                if tool_operation.status in {"prepared", "ready"} and tool_operation.attempt_count == 0:
                    pending_calls.append(call)
                    continue
                if (
                    tool_operation.status == "failed"
                    and tool_operation.error_code == "input_superseded"
                    and tool_operation.attempt_count == 0
                ):
                    pending_calls.append(call)
                    continue
                if tool_operation.status == "denied" and (
                    await self._store.run_async(self._store.request_input_revision, turn_id, operation.request_ref)
                ) != (await self._store.run_async(self._store.user_input_revision, turn_id)):
                    pending_calls.append(call)
                    continue
                if tool_operation.status in {"succeeded", "failed"}:
                    interaction = await self._commit(
                        partial(
                            partial(tool_ops.mark_tool_result_missing, self._store),
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
            result_item = await self._store.run_async(self._store.read_item, tool_operation.result_item_id)
            result_payload = result_item.payload
            if result_item.kind == "tool_reconciliation" and result_item.producer == "reconciler":
                result_payload = result_item.payload.get("result", {})
            if (
                result_item.kind not in {"tool_result", "tool_reconciliation"}
                or result_item.status != "completed"
                or not isinstance(result_payload, Mapping)
                or result_payload.get("tool_call_id") != call.id
            ):
                raise RuntimeError("committed tool result linkage is malformed")
        token_budget = turn.binding_manifest.get("model_token_budget_total")
        if (
            token_budget is not None
            and (await self._store.run_async(self._consumed_model_tokens, turn_id)) > token_budget
        ):
            return await self._fail_turn(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                reason_code="model_token_budget_exhausted",
                message="Turn exceeded its frozen model token budget.",
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
                response=replace(
                    response,
                    outputs=tuple(
                        output
                        for output in response.outputs
                        if output.kind != "tool_call" or output.tool_call_id in {call.id for call in pending_calls}
                    ),
                ),
                prepared=prepared,
            )
            if handled is not None:
                return handled
        handled = await self._after_tool_batch(
            thread_id=turn.thread_id,
            turn_id=turn_id,
            source_item_id=operation.response_item_id,
        )
        if handled is not None:
            return handled
        await self._commit(
            lambda: response_handled(
                self._store,
                turn_id=turn_id,
                operation_id=operation_id,
                response_item_id=response_id,
                disposition="tool_batch_resolved",
            )
        )
        return None

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
        operations = [
            op
            for op in self._store.list_model_operations(turn_id)
            if op.request_ref.get("purpose") != "context_summary"
        ]
        return max(
            (int(op.request_ref.get("step_snapshot", {}).get("step", index)) for index, op in enumerate(operations, 1)),
            default=0,
        )

    async def drive(
        self,
        turn_context: TurnContext | None = None,
        *,
        turn_id: str | None = None,
        start_step: int | None = None,
        action: str | None = None,
        user_input: str | None = None,
        expected_request_id: str | None = None,
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
            if offered.kind == ContinuationKind.RESUME_CANCELLED:
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
                await self._commit(lambda: recover_owner_loss(self._store, turn_id, abort=True))
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
                    result = await self._consume_response(turn_id=turn_id, operation_id=decision.operation_id)
                except TurnCancellationPendingError:
                    continue
                if result is not None:
                    return result
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
                except (ModelContextOverflowError, TurnCancellationPendingError):
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
                if models:
                    step = models[-1].request_ref.get("step_snapshot", {}).get("step", max(1, count))
            remaining_tokens = await self._store.run_async(
                lambda: self._store.read_budget_state(turn_id).remaining("tokens")
            )
            if remaining_tokens is not None and remaining_tokens < 1:
                return await self._fail_turn(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    reason_code="model_token_budget_exhausted",
                    message="Turn exhausted its frozen model token budget.",
                )
            binding = turn_context.binding_manifest
            step_budget = binding.get("model_step_budget", self._max_steps)
            if type(step_budget) is not int or step_budget < 1:
                raise RuntimeError("frozen model step budget is invalid")
            if step > min(self._max_steps, step_budget):
                return await self._fail_turn(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    reason_code="model_step_budget_exhausted",
                    message=await self._store.run_async(self._step_budget_receipt, turn_id, step_budget),
                )
            reactive = decision.kind == ContinuationKind.CONTEXT_OVERFLOW
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
                if rejections > 1:
                    return await self._fail_turn(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        reason_code="model_context_window_exceeded",
                        message="Context overflow persisted after one recovery.",
                    )
                self._capacity_recovery_root = root_id
            observation = await self._store.run_async(self._context_engine.observation_revision, turn_id)
            try:
                context, prepared = await self._context_engine.prepare_or_resume(
                    turn_context, step=step, pressure_reason="provider_overflow" if reactive else None
                )
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
            except ModelContextOverflowError:
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
        changed = any(item.payload.get("metadata", {}).get("workspace_tree_changed") is True for item in results)
        return (
            f"本轮已达到 {limit} 步执行上限，尚未完成收尾。\n"
            f"已记录 {len(results)} 个工具结果，其中 {failed} 个报错。"
            + ("已记录实际工作区修改；现有修改已保留。" if changed else "未记录实际工作区修改。")
            + "\n本条为执行回执，任务验收以验证状态和实际检查为准。可在当前会话继续处理剩余工作。"
        )

    def _consumed_model_tokens(self, turn_id: str) -> int:
        # Transitional compatibility for existing callers. The authoritative
        # accounting view now comes from durable budget reservations.
        return self._store.read_budget_state(turn_id).used.total_tokens

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

    async def _handle_model_response(
        self,
        *,
        thread_id: str,
        turn_id: str,
        response: HarnessModelResponse,
        prepared: PreparedModelCall,
    ) -> TurnResult | None:
        source = next(
            op
            for op in await self._store.run_async(self._store.list_model_operations, turn_id)
            if op.request_ref.get("request_id") == prepared.request_ref.get("request_id")
        )
        source_item_id = source.response_item_id
        revision = await self._store.run_async(self._store.request_input_revision, turn_id, prepared.request_ref)
        if revision != (await self._store.run_async(self._store.user_input_revision, turn_id)) and (
            response.status == "incomplete" or not response.tool_calls
        ):
            return None
        if response.status == "incomplete":
            policy = await self._store.run_async(
                lambda: self._store.read_turn(turn_id).binding_manifest.get("model_response_recovery", {})
            )
            # Count canonical committed outcomes, not process-local retries.
            # One retry per Turn; summaries and unknown transport outcomes have
            # separate recovery contracts. Every retry spends an ordinary step.
            incomplete = await self._store.run_async(
                lambda: sum(
                    item.kind == "model_response" and item.payload.get("response_status") == "incomplete"
                    for item in self._store.list_items(turn_id)
                )
            )
            if (
                isinstance(policy, Mapping)
                and policy.get("revision") == "restart-step-v1"
                and type(policy.get("max_retries")) is int
                and policy["max_retries"] == 1
                and response.incomplete_reason == "max_output_tokens"
                and incomplete <= 1
            ):
                return None
            return await self._fail_turn(
                thread_id=thread_id,
                turn_id=turn_id,
                reason_code="model_response_incomplete",
                message=(
                    f"Model returned an incomplete agent step: {response.incomplete_reason}; "
                    "recovery unavailable or exhausted."
                ),
            )
        if response.tool_calls:
            if self._tool_orchestrator is None:
                raise RuntimeError("model requested tools but no ToolOrchestrator exists")
            calls = tuple(_aci_tool_call(call, prepared.request_ref) for call in response.tool_calls)
            if source_item_id is None:
                raise RuntimeError("tool batch lacks canonical response identity")
            await self._commit(
                lambda: admit_tool_batch(self._store, turn_id=turn_id, response_item_id=source_item_id, calls=calls)
            )
            try:
                await self._tool_orchestrator.execute_batch(
                    turn_id=turn_id,
                    calls=calls,
                )
            except asyncio.CancelledError:
                await asyncio.shield(
                    self._commit(
                        partial(
                            partial(turn_ops.request_turn_cancellation, self._store),
                            turn_id=turn_id,
                            reason="Turn task cancelled during tool execution",
                        )
                    )
                )
                if (await self._store.run_async(self._store.read_turn, turn_id)).status == "running":
                    await asyncio.shield(
                        self._commit(
                            partial(
                                partial(turn_ops.pause_turn, self._store),
                                turn_id=turn_id,
                                reason="tool execution was interrupted",
                            )
                        )
                    )
                raise
            except ToolApprovalRequiredError as pause:
                return TurnResult(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    answer=None,
                    status="paused",
                    interaction_id=pause.interaction_id,
                )
            return await self._after_tool_batch(
                thread_id=thread_id,
                turn_id=turn_id,
                source_item_id=source_item_id,
            )
        return await self._finish_answer(
            thread_id=thread_id,
            turn_id=turn_id,
            answer=response.text,
            input_revision=revision,
            source_item_id=source_item_id,
        )

    async def _after_tool_batch(
        self,
        *,
        thread_id: str,
        turn_id: str,
        source_item_id: str | None,
    ) -> TurnResult | None:
        """Apply the same durable post-tool transition during execution and recovery."""
        turn = await self._store.run_async(self._store.read_turn, turn_id)
        if turn.status == "paused":
            pending = await self._store.run_async(
                lambda: [i for i in self._store.list_interactions(turn_id) if i.status == "pending"]
            )
            if len(pending) != 1:
                raise RuntimeError("paused tool execution has no unique interaction")
            return TurnResult(
                thread_id=thread_id, turn_id=turn_id, answer=None, status="paused", interaction_id=pending[0].request_id
            )
        if source_item_id is None:
            raise RuntimeError("tool batch has no committed model response")
        operation = next(
            op
            for op in await self._store.run_async(self._store.list_model_operations, turn_id)
            if op.response_item_id == source_item_id
        )
        revision = await self._store.run_async(self._store.request_input_revision, turn_id, operation.request_ref)
        if revision != (await self._store.run_async(self._store.user_input_revision, turn_id)):
            return None
        items = await self._store.run_async(self._store.list_items, turn_id)
        if any(i.kind == "completion_feedback" and i.payload.get("source_item_id") == source_item_id for i in items):
            return None  # A committed recovery grant belongs to this batch, not a new attempt.
        call_ids = await self._store.run_async(
            lambda: {
                call.id for call in restore_model_response(self._store.read_item(source_item_id).payload).tool_calls
            }
        )
        failure = await self._store.run_async(
            lambda: next(
                (
                    result
                    for _, result in reversed(self._canonical_tool_results(turn_id))
                    if result.get("tool_call_id") in call_ids
                    and result.get("is_error") is True
                    and (result.get("error_code") != "input_superseded")
                ),
                None,
            )
        )
        if failure is None or not (await self._store.run_async(self._repeated_tool_failure, turn_id)):
            return None
        feedback = json.dumps(
            {
                "reason_code": "repeated_tool_failure_recovery",
                "message": "The same deterministic tool failure has repeated without progress. "
                "Inspect its resource or correct the request before retrying. Changing only paging or "
                "output limits cannot repair a missing resource. You have one recovery opportunity; "
                "use a different justified action or explain the blocker honestly.",
            },
            sort_keys=True,
        )
        recovery_used = any(
            i.kind == "completion_feedback" and _feedback_reason_code(i.payload) == "repeated_tool_failure_recovery"
            for i in items
        )
        recovery_policy = turn.binding_manifest.get("tool_failure_recovery")
        if (
            isinstance(recovery_policy, Mapping)
            and recovery_policy.get("revision") == "one-replan-v1"
            and not recovery_used
        ):
            await self._commit(
                lambda: completion_ops.record_completion_feedback(
                    self._store,
                    turn_id=turn_id,
                    reason=feedback,
                    source_item_id=source_item_id,
                    reason_code="repeated_tool_failure_recovery",
                )
            )
            return None
        return await self._fail_turn(
            thread_id=thread_id,
            turn_id=turn_id,
            reason_code="repeated_tool_failure",
            message=(
                f"未观察到有效进展，工具 {failure.get('tool_name')} 重复发生同一确定性故障"
                f"（{failure.get('error_code')}）：{failure.get('error_message')}。"
                "本轮恢复机会已用尽，已停止重复调用。"
            ),
        )

    def _canonical_tool_results(self, turn_id: str) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        """Read the accepted outcome; keep superseded unknowns only in the audit ledger."""
        operations = {op.tool_call_id: op for op in self._store.list_tool_operations(turn_id)}
        results: list[tuple[str, Mapping[str, Any]]] = []
        for item in self._store.list_items(turn_id):
            if item.status != "completed":
                continue
            payload = item.payload
            if item.kind == "tool_reconciliation" and item.producer == "reconciler":
                reconciled = item.payload.get("result")
                if not isinstance(reconciled, Mapping):
                    continue
                payload = reconciled
                call_id = payload.get("tool_call_id")
                if not isinstance(call_id, str):
                    continue
                operation = operations.get(call_id)
                if operation is None or operation.result_item_id != item.item_id:
                    continue
            elif item.kind == "tool_result":
                call_id = payload.get("tool_call_id")
                if not isinstance(call_id, str):
                    continue
                operation = operations.get(call_id)
                if (
                    operation is not None
                    and operation.result_item_id is not None
                    and operation.result_item_id != item.item_id
                ):
                    continue
            else:
                continue
            results.append((item.item_id, payload))
        return tuple(results)

    def _repeated_tool_failure(self, turn_id: str) -> bool:
        """Use committed results so a process restart cannot reset the failure streak."""
        items = self._store.list_items(turn_id)
        calls = {
            item.payload.get("tool_call_id"): item.payload
            for item in items
            if item.kind == "tool_call" and item.status == "completed"
        }
        canonical = self._canonical_tool_results(turn_id)
        results = [result for _, result in canonical]
        if len(results) < 3:
            return False
        operations = {op.result_item_id: op for op in self._store.list_tool_operations(turn_id)}
        result_items = {result.get("tool_call_id"): item_id for item_id, result in canonical}
        target = None
        repeats = 0
        for result in reversed(results):
            call = calls.get(result.get("tool_call_id"))
            if call is None:
                break
            metadata = result.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            if metadata.get("workspace_tree_changed") is True:
                break
            if result.get("is_error") is not True:
                operation = operations.get(result_items.get(result.get("tool_call_id")))
                # Inspections and proven no-op writes do not reset a failure streak.
                # Unknown/external effects remain a conservative progress boundary.
                if operation is None:
                    break
                effects = set(operation.effects)
                readonly = effects <= {"read_workspace"}
                noop = metadata.get("workspace_tree_changed") is False and effects <= {
                    "read_workspace",
                    "write_workspace",
                    "execute_process",
                    "destructive",
                }
                if not (readonly or noop):
                    break
                if target and (call.get("tool_name"), call.get("arguments")) == (target[0], target[3]):
                    break  # The same call succeeded: actual recovery.
                if target and isinstance(target[3], tuple) and call.get("tool_name") == target[0]:
                    resources = tuple(
                        sorted(
                            resource["identity"]
                            for resource in operation.resources
                            if resource.get("kind") == "filesystem" and resource.get("access") == "read"
                        )
                    )
                    if set(target[3]) <= set(resources):
                        break
                continue
            # Argument validation permits a corrected call, not identical retries.
            deterministic = result.get("retryable") is False or result.get("error_code") == "invalid_arguments"
            if not deterministic:
                break
            signature = (
                result.get("tool_name"),
                result.get("error_code"),
                None,
                call.get("arguments"),
            )
            operation = operations.get(result_items.get(result.get("tool_call_id")))
            constraint = metadata.get("permission_constraint")
            if result.get("error_code") == "filesystem_denied" and isinstance(constraint, Mapping):
                # The same policy/resource/action remains the same obstacle
                # when a model changes tool names or rewrites its arguments.
                denied_resources = tuple(sorted({(r["path"], r["access"]) for r in constraint.get("resources", ())}))
                if denied_resources:
                    signature = (
                        "filesystem_permission",
                        "filesystem_denied",
                        None,
                        (constraint.get("permissions_revision"), denied_resources),
                    )
            if (
                result.get("error_code") == "file_not_found"
                and operation is not None
                and set(operation.effects) <= {"read_workspace"}
            ):
                resources = tuple(
                    sorted(
                        resource["identity"]
                        for resource in operation.resources
                        if resource.get("kind") == "filesystem" and resource.get("access") == "read"
                    )
                )
                if resources:
                    # The committed outcome already attests a missing resource.
                    # Replaying it must not perform unrecorded filesystem probes.
                    signature = (result.get("tool_name"), "file_not_found", None, resources)
            if target is None:
                target = signature
            if signature == target:
                repeats += 1
                if repeats == 3:
                    return True
        return False

    async def _finish_answer(
        self,
        *,
        thread_id: str,
        turn_id: str,
        answer: str,
        input_revision: int,
        source_item_id: str | None = None,
    ) -> TurnResult | None:
        if (await self._store.run_async(self._store.user_input_revision, turn_id)) != input_revision:
            return None
        items = await self._store.run_async(self._store.list_items, turn_id)
        proposal_item = next(
            (
                i
                for i in items
                if i.kind == "final_proposal"
                and source_item_id is not None
                and i.payload.get("source_item_id") == source_item_id
            ),
            None,
        )
        if proposal_item is None:
            proposal_item = await self._commit(
                lambda: completion_ops.record_final_proposal(
                    self._store,
                    turn_id=turn_id,
                    answer=answer,
                    source_item_id=source_item_id,
                )
            )
        proposal = CompletionProposal(
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=proposal_item.item_id,
            answer=answer,
        )
        decision_item = next(
            (
                i
                for i in items
                if i.kind == "completion_decision" and i.payload.get("proposal_item_id") == proposal.item_id
            ),
            None,
        )
        if decision_item is None:
            decision = await self._store.run_async(self._completion_gate.evaluate, proposal)
            if decision.action == "continue" and (
                await self._store.run_async(self._completion_continuation_limit_reached, turn_id)
            ):
                decision = replace(
                    decision,
                    action="fail",
                    reason_code="completion_continuation_limit",
                    reason=(
                        "Completion checks exhausted their consecutive continuation allowance without a tool "
                        "exchange or new user input. The last check did not accept the result: " + decision.reason
                    ),
                )
            decision_item = await self._commit(
                lambda: completion_ops.record_completion_decision(
                    self._store,
                    turn_id=turn_id,
                    proposal_item_id=proposal.item_id,
                    action=decision.action,
                    reason=decision.reason,
                    validation_status=decision.validation_status,
                    validation_scope=decision.validation_scope,
                    reason_code=decision.reason_code,
                )
            )
        else:
            decision = CompletionDecision(
                **{key: value for key, value in decision_item.payload.items() if key != "proposal_item_id"}
            )
        if (await self._store.run_async(self._store.user_input_revision, turn_id)) != input_revision:
            return None
        if decision.action == "continue":
            if not any(
                i.kind == "completion_feedback" and i.payload.get("source_item_id") == decision_item.item_id
                for i in items
            ):
                await self._commit(
                    lambda: completion_ops.record_completion_feedback(
                        self._store,
                        turn_id=turn_id,
                        reason=decision.reason,
                        source_item_id=decision_item.item_id,
                    )
                )
            return None
        if decision.action == "pause":
            interaction = await self._commit(
                lambda: turn_ops.request_clarification(
                    self._store,
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
                lambda: turn_ops.fail_turn(
                    self._store,
                    turn_id=turn_id,
                    reason_code=decision.reason_code or "completion_rejected",
                    message=decision.reason,
                )
            )
            return TurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
                answer=None,
                status="failed",
            )
        try:
            completed = await self._commit(
                lambda: turn_ops.complete_turn(
                    self._store,
                    turn_id=turn_id,
                    answer=answer,
                    expected_input_revision=input_revision,
                    source_item_id=source_item_id,
                )
            )
        except TurnInputChangedError:
            return None
        return TurnResult(
            thread_id=thread_id,
            turn_id=completed.turn_id,
            answer=answer,
        )

    def _completion_continuation_limit_reached(self, turn_id: str) -> bool:
        """Bound Stop feedback itself; tool exchanges are not proof of task progress."""
        policy = self._store.read_turn(turn_id).binding_manifest.get("completion_continuation_policy", {})
        if not isinstance(policy, Mapping) or policy.get("revision") != "bounded-v1":
            return False  # Old Turns keep their frozen behavior.
        limit = policy.get("max_continuations")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise RuntimeError("frozen completion continuation limit is invalid")
        continuations = 0
        for item in reversed(self._store.list_items(turn_id)):
            if item.status != "completed":
                continue
            if item.kind == "user_message" or (
                item.kind == "model_response"
                and item.payload.get("response_status", "completed") == "completed"
                and restore_model_response(item.payload).tool_calls
            ):
                break
            if item.kind == "completion_decision" and item.payload.get("action") == "continue":
                continuations += 1
        return continuations >= limit


def _feedback_reason_code(payload: Mapping[str, Any]) -> str | None:
    code = payload.get("reason_code")
    if isinstance(code, str):
        return code
    return None


def _aci_tool_call(
    call: ModelToolCall,
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
        argument_error=call.argument_error,
        origin=ToolCallOrigin(
            request_id=request_id,
            toolset_revision=toolset_revision,
            exposed_tool_names=tuple(exposed),
        ),
    )


def _response_from_committed_item(item: ItemSnapshot) -> HarnessModelResponse:
    if item.kind != "model_response" or item.status != "completed":
        raise RuntimeError("canonical model response Item is malformed")
    return restore_model_response(item.payload)

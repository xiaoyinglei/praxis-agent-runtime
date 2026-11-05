"""Capture live request facts once; restore saved requests without recapturing them."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent_runtime.budget import budget_pressure_active
from agent_runtime.core.messages import model_message_payload, restore_model_message
from agent_runtime.harness import model_ops
from agent_runtime.harness.budget_context import budget_status, model_visible_budget
from agent_runtime.harness.control import ContextSnapshot, StepControl, append_context_updates, append_runtime_state
from agent_runtime.harness.execution_context import StepContext, TurnContext, step_limit_from_binding
from agent_runtime.harness.protocol import BindingProvider, ContextManager, ModelMessage, ToolRouter
from agent_runtime.harness.records import ModelOperationSnapshot
from agent_runtime.harness.runtime_store import RuntimeStore
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
from agent_runtime.runtime_clock import capture_runtime_clock
from agent_runtime.runtime_environment import capture_runtime_environment


class RequestContext:
    def __init__(
        self,
        *,
        thread_id: str,
        store: RuntimeStore,
        context_manager: ContextManager,
        tool_router: ToolRouter | None,
        tool_orchestrator: ToolOrchestrator | None,
        binding_provider: BindingProvider | None,
    ) -> None:
        self.thread_id = thread_id
        self._store = store
        self._context_manager = context_manager
        self._tool_router = tool_router
        self._tool_orchestrator = tool_orchestrator
        self._binding_provider = binding_provider
        self._capacity_recovery_root: str | None = None

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
        from agent_runtime.project_instructions import restored_instructions

        project_instructions = restored_instructions(
            turn_context.binding_manifest,
            self._store.list_items(turn_context.turn_id),
        )
        context_updates: tuple[ModelMessage, ...] = ()
        clock = capture_runtime_clock()
        update = None
        if turn_context.binding_manifest.get("maintenance_purpose", "agent_step") == "agent_step":
            messages, context_updates = append_context_updates(
                messages, project_instructions, turn_context.binding_manifest.get("memory_snapshot")
            )
            messages, update = self._append_runtime_state(turn_context, messages, step=step, clock=clock)
            from agent_runtime.harness.context import RolloutContextManager

            if isinstance(self._context_manager, RolloutContextManager):
                # Capacity includes the live facts appended to this request.
                # Candidate selection must measure this exact same projection.
                self._context_manager._validate_budget(messages, turn_id=turn_context.turn_id)
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
        # Absence in an old Turn also means no ceiling. Do not leak a live
        # Session default into saved request metadata or delegated tool calls.
        binding["model_step_budget"] = step_limit_from_binding(turn_context.binding_manifest)
        # All newly captured agent requests use the current append-only context
        # projection. Historical saved requests bypass capture and replay their
        # original frozen wire; there is no second live legacy prompt strategy.
        binding["execution_context_policy"] = {
            **binding.get("execution_context_policy", {}),
            "revision": "working-context-v2",
            "cache_continuity": "append-v1",
            "context_updates": "append-v1",
        }
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

    def _append_runtime_state(
        self,
        turn: TurnContext,
        messages: tuple[ModelMessage, ...],
        *,
        step: int,
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
            "Current execution environment": capture_runtime_environment(
                self._store.read_thread(turn.thread_id).workspace
            ),
            "Current runtime clock facts": dict(clock),
            "Current advisory plan": None if plan is None else model_visible_plan(plan.payload["plan"]),
        }
        previous = ContextSnapshot.from_messages(messages).runtime_state.get("Resource limits (runtime ledger)")
        budget = model_visible_budget(self._store, turn, step=step, previous=previous)
        if budget is not None or previous is not None:
            state["Resource limits (runtime ledger)"] = budget
        completion = turn.binding_manifest.get("completion_policy", {})
        state["acceptance_contract"] = dict(completion) if completion.get("acceptance_command") else None
        delivery_policy = turn.binding_manifest.get("delivery_policy", {})
        if delivery_policy.get("revision") == "delivery-v1" and delivery_policy.get("model_tools", True):
            plans = [i for i in self._store.list_items(turn.turn_id) if i.kind == "delivery_plan"]
            state["Delivery evidence contract"] = {
                "plan": dict(plans[-1].payload) if plans else None,
                "next_action": (
                    "Use record_delivery to declare scope, exact checks and omissions; record trusted tool evidence "
                    "and final review. Use record_delivery action=submit with the exact reviewed final answer "
                    "to propose that answer without regenerating it. A natural final can end unverified. "
                    "Versioned commands use explicit verification_dependencies and read-only inputs. "
                    "Verified delivery for explicit acceptance requires verification_mode=workspace "
                    "at the configured cwd; "
                    "otherwise prefer verification_mode=snapshot for a frozen input boundary."
                ),
            }
        return append_runtime_state(messages, state, acceptance=bool(completion.get("acceptance_command")))

    def _budget_status(self, turn: TurnContext) -> dict[str, Any]:
        return budget_status(self._store, turn)

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

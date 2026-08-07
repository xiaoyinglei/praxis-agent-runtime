from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Mapping, Sequence
from dataclasses import replace
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import uuid4

from pydantic import ValidationError

from agent_runtime.core.checkpointing import CheckpointStore
from agent_runtime.core.context import TurnRegistry
from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.finalization import (
    FinishCandidateBuilder,
    FinishCandidateBuildError,
)
from agent_runtime.core.human_input import HumanInputRequest, ToolCallSummary
from agent_runtime.core.llm_context import AgentLLMContextOverflowError
from agent_runtime.core.messages import (
    ModelMessage,
    canonical_json_text,
    context_event_message,
    tool_result_message,
)
from agent_runtime.core.model_request import build_tool_manifest
from agent_runtime.core.observations import (
    ObservationBatch,
    ObservationExtractor,
    grounded_workspace_paths,
    runtime_workspace_change,
)
from agent_runtime.core.output_models import ValidatedFinalOutput
from agent_runtime.core.runtime_diagnostics import (
    AgentLatencyProfile,
    RuntimeDiagnostic,
    redact_sensitive_text,
)
from agent_runtime.loop.state import (
    LoopPause,
    LoopState,
    LoopTerminal,
    LoopTransition,
    LoopTransitionReason,
    ModelTurn,
    ModelTurnEnvelope,
    ModelTurnProvider,
    PendingToolCall,
    append_loop_diagnostic,
    replace_latest_transition,
)
from agent_runtime.loop.stop_hooks import StopHookOutcome, StopHookRunner
from agent_runtime.memory.compactor import LoopCompactionResult
from agent_runtime.modeling.gateway import (
    LLMContextOverflowError,
    LLMToolCallValidationError,
)
from agent_runtime.planning import (
    MAX_PLAN_EVENTS,
    AgentPlan,
    PlanEvent,
    PlanStep,
    PlanTracker,
)
from agent_runtime.streaming.events import (
    EventType,
    StreamEvent,
    compact_layer,
    loop_end,
    next_sequence,
    recovery_event,
    tool_use_error,
    tool_use_result,
    tool_use_start,
    turn_end,
    turn_start,
)
from agent_runtime.streaming.sink import StreamEventSink
from agent_runtime.tools.builtins.planning import UpdatePlanInput
from agent_runtime.tools.executor import ToolExecutionRecord, ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.selection import FIND_TOOLS_NAME, reduce_tool_activation
from agent_runtime.tools.tool import JsonValue, Tool, ToolCall, ToolCallOrigin, ToolResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_runtime.skills.runtime import SkillRuntime

# Tool classification used by runtime metrics.
_NATIVE_TOOL_SET = frozenset(
    {
        "list_files",
        "search_text",
        "read_file",
        "inspect_data_file",
        "apply_patch",
        "run_command",
        "execute_python",
        "update_plan",
        "find_tools",
    }
)

_REPEATED_TOOL_FAILURE_CODE = "repeated_tool_failure"
_MAX_RETRYABLE_IDENTICAL_FAILURES = 2
_EXPLORATION_TOOL_NAMES = frozenset(
    {
        "list_files",
        "search_text",
        "read_file",
        "inspect_data_file",
        "run_command",
        "find_tools",
    }
)
_STABLE_INSPECTION_TOOL_NAMES = frozenset(
    {"list_files", "search_text", "read_file", "inspect_data_file", "find_tools"}
)


def _add_model_latency(state: LoopState, latency_ms: float) -> None:
    profile = state.get("latency_profile")
    if not isinstance(profile, AgentLatencyProfile):
        profile = AgentLatencyProfile()
    state["latency_profile"] = profile.model_copy(update={"model_latency_ms": profile.model_latency_ms + latency_ms})


def _append_turn_messages(
    state: LoopState,
    messages: Sequence[ModelMessage],
) -> None:
    if not messages:
        return
    state["turn_transcript"] = [
        *state.get("turn_transcript", []),
        *messages,
    ]


class LoopContextManager(Protocol):
    def prepare(
        self,
        state: LoopState,
    ) -> LoopCompactionResult | Awaitable[LoopCompactionResult]: ...


class LoopEventSink(Protocol):
    async def emit(self, transition: LoopTransition) -> None: ...


class NullLoopEventSink:
    async def emit(self, transition: LoopTransition) -> None:
        del transition


class AgentLoop:
    """Claude-like single-agent kernel implemented as an ordinary while loop."""

    def __init__(
        self,
        *,
        definition: AgentRuntimePolicy,
        model_provider: ModelTurnProvider,
        context_manager: LoopContextManager,
        tool_executor: ToolExecutor,
        registry_snapshot: Mapping[str, Tool],
        execution_context: ToolExecutionContext,
        checkpoint_store: CheckpointStore,
        stop_hook_runner: StopHookRunner,
        finish_candidate_builder: FinishCandidateBuilder,
        event_sink: LoopEventSink | None = None,
        stream_sink: StreamEventSink | None = None,
        observation_extractor: ObservationExtractor | None = None,
        plan_tracker: PlanTracker | None = None,
        max_model_retries: int = 1,
        skill_runtime: SkillRuntime | None = None,
        discoverable_tool_names: Sequence[str] | None = None,
    ) -> None:
        if max_model_retries < 0:
            raise ValueError("max_model_retries must be non-negative")
        self._definition = definition
        self._model_provider = model_provider
        self._context_manager = context_manager
        self._tool_executor = tool_executor
        self._registry_snapshot = registry_snapshot
        self._execution_context = execution_context
        self._checkpoint_store = checkpoint_store
        self._stop_hook_runner = stop_hook_runner
        self._finish_candidate_builder = finish_candidate_builder
        self._event_sink = event_sink or NullLoopEventSink()
        self._stream_sink: StreamEventSink | None = None
        self._set_stream_sink(stream_sink)
        self._observation_extractor = observation_extractor or ObservationExtractor()
        self._observed_tool_call_ids: set[str] = set()
        self._plan_tracker = plan_tracker or PlanTracker()
        self._max_model_retries = max_model_retries
        self._skill_runtime = skill_runtime
        self._discoverable_tool_names = None if discoverable_tool_names is None else tuple(discoverable_tool_names)

    def _set_stream_sink(self, sink: StreamEventSink | None) -> None:
        """Set the active stream sink, propagating to sub-components that support it."""
        self._stream_sink = sink
        for target in (self._model_provider,):
            if hasattr(target, "_stream_sink"):
                target._stream_sink = sink

    async def _emit_stream(self, event: Any) -> None:
        """Emit a stream event if sink is configured."""
        if self._stream_sink is not None:
            await self._stream_sink.emit(event)

    async def run_streaming(self, state: LoopState) -> AsyncGenerator[StreamEvent, None]:
        """Yield StreamEvents as the loop runs.  One queue sink, no monkey patches."""
        from agent_runtime.streaming.sink import QueueStreamEventSink

        sink = QueueStreamEventSink()
        original = self._stream_sink
        self._set_stream_sink(sink)

        run_task: asyncio.Task[LoopState] | None = None
        try:
            run_task = asyncio.create_task(self.run(state))

            def _on_run_done(t: asyncio.Task[LoopState]) -> None:
                del t
                asyncio.create_task(sink.close())

            run_task.add_done_callback(_on_run_done)

            async for event in sink.stream():
                yield event

            await run_task

        except BaseException:
            await sink.close()
            if run_task is not None and not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):
                    pass
            raise

        finally:
            self._set_stream_sink(original)

    async def run(self, state: LoopState) -> LoopState:
        if state["status"] != "running":
            return state
        handles = TurnRegistry.get_or_create(state["run_config"])
        if state["plan_state"].agent_plan is None:
            plan, events = self._plan_tracker.initialize_task(
                task=state["current_message"],
            )
            state["plan_state"].agent_plan = plan
            state["plan_state"].plan_events = list(events)
        await self._checkpoint_store.save_snapshot(state, reason="loop_start")

        retries = 0
        while state["status"] == "running":
            budget_remaining = await _remaining_llm_budget(handles)
            if budget_remaining is not None and budget_remaining <= 0:
                await self._fail(
                    state,
                    stop_reason="budget_exhausted",
                    error="LLM/tool budget exhausted.",
                    transition_reason="failed",
                    checkpoint_reason="budget_exhausted",
                )
                return state
            # Execute any pending tool calls first
            if state["pending_tool_calls"]:
                if await self._execute_pending_tools(state):
                    return state
                continue

            # Guard rails
            request_max_turns = state["run_config"].max_turns
            max_turns = self._definition.max_iterations
            stop_reason: LoopTransitionReason = "max_iterations"
            if request_max_turns is not None and request_max_turns <= max_turns:
                max_turns = request_max_turns
                stop_reason = "max_turns"
            if state["iteration"] >= max_turns:
                await self._fail(
                    state,
                    stop_reason=stop_reason,
                    error="Max turns reached.",
                    transition_reason=stop_reason,
                    checkpoint_reason=stop_reason,
                )
                return state

            if not await self._compact(state):
                return state

            await self._emit_stream(
                turn_start(
                    turn_id=state["run_config"].turn_id,
                    iteration=state["iteration"] + 1,
                )
            )

            # Model turn — call the LLM, handle errors, return (turn | None)
            state["iteration"] += 1
            turn, retries = await self._model_turn(
                state,
                retries,
                budget_remaining=(-1 if budget_remaining is None else budget_remaining),
            )
            if turn is None:
                if state["status"] != "running":
                    return state  # terminal
                continue  # retry

            # Dispatch
            state["last_model_turn"] = turn
            if turn.action == "execute":
                state["tool_call_ledger"].append_plans(turn.tool_calls, turn=state["iteration"])
                state["pending_tool_calls"] = [PendingToolCall(plan=call, status="pending") for call in turn.tool_calls]
                state["tool_call_ledger"].trim(
                    active_tool_call_ids={p.tool_call_id for p in state["pending_tool_calls"]},
                )
                await self._transition(
                    state, reason="next_turn", detail={"action": turn.action}, checkpoint_reason="model_turn"
                )
                await self._transition(
                    state,
                    reason="tool_execution",
                    detail={"phase": "scheduled", "tool_call_ids": [c.tool_call_id for c in turn.tool_calls]},
                    checkpoint_reason="tool_calls_scheduled",
                )
                await self._emit_stream(
                    turn_end(
                        turn_id=state["run_config"].turn_id,
                        iteration=state["iteration"],
                        stop_reason="tool_use",
                    )
                )
                continue

            await self._transition(
                state, reason="next_turn", detail={"action": turn.action}, checkpoint_reason="model_turn"
            )

            if turn.action == "pause":
                await self._emit_stream(
                    turn_end(
                        turn_id=state["run_config"].turn_id,
                        iteration=state["iteration"],
                        stop_reason="pause",
                    )
                )
                await self._pause(
                    state,
                    reason=cast(str, turn.pause_reason),
                    request=None,
                    checkpoint_reason="model_pause",
                    transition_reason="paused",
                )
                return state

            # finish
            await self._emit_stream(
                turn_end(
                    turn_id=state["run_config"].turn_id,
                    iteration=state["iteration"],
                    stop_reason="end_turn",
                )
            )
            if await self._evaluate_finish(state, turn):
                return state

        await self._emit_stream(
            loop_end(
                turn_id=state["run_config"].turn_id,
                reason=state["terminal"].stop_reason if state.get("terminal") else "loop_exited",
                total_turns=state["iteration"],
            )
        )
        return state

    async def _compact(self, state: LoopState) -> bool:
        """Run proactive compaction.  Returns False if compaction itself failed (state is terminal)."""
        try:
            result = await _await_value(self._context_manager.prepare(state))
        except Exception as exc:
            await self._fail(
                state,
                stop_reason="context_compaction_failed",
                error=str(exc) or type(exc).__name__,
                transition_reason="failed",
                checkpoint_reason="context_compaction_failed",
            )
            return False
        if not result.changed:
            return True
        transition = LoopTransition(
            reason="compaction",
            iteration=state["iteration"],
            detail={"channels": list(result.channels), "warnings": list(result.warnings)},
        )
        replace_latest_transition(state, transition)
        await self._event_sink.emit(transition)
        await self._emit_stream(
            compact_layer(
                channels=list(result.channels),
                warnings=list(result.warnings),
                turn_id=state["run_config"].turn_id,
                iteration=state["iteration"],
            )
        )
        return True

    async def _model_turn(
        self,
        state: LoopState,
        retries: int,
        *,
        budget_remaining: int,
    ) -> tuple[ModelTurn | None, int]:
        """Call the model, handle errors.  Returns (turn, retries) or (None, _) on terminal failure."""
        turn: ModelTurn | None = None
        try:
            model_started_at = time.perf_counter()
            try:
                provided = await self._model_provider.next_turn(
                    state,
                    definition=self._definition,
                    budget_remaining=budget_remaining,
                )
            finally:
                _add_model_latency(state, (time.perf_counter() - model_started_at) * 1000)
            envelope = provided if isinstance(provided, ModelTurnEnvelope) else ModelTurnEnvelope(draft=provided)
            self._apply_model_envelope(state, envelope)
            for t in envelope.transitions:
                tr = t.model_copy(update={"iteration": state["iteration"]})
                replace_latest_transition(state, tr)
                await self._event_sink.emit(tr)
            turn = await self._finish_candidate_builder.build(envelope.draft, state=state)
            return turn, 0  # success resets retry count

        except (AgentLLMContextOverflowError, LLMContextOverflowError) as exc:
            turn, _ = await self._handle_overflow(state, exc)
            return turn, retries

        except (FinishCandidateBuildError, ValidationError, ValueError) as exc:
            turn, retries = await self._handle_invalid_turn(state, exc, retries)
            return turn, retries

        except Exception as exc:
            turn, retries = await self._handle_provider_error(state, exc, retries)
            return turn, retries

    def _apply_model_envelope(
        self,
        state: LoopState,
        envelope: ModelTurnEnvelope,
    ) -> None:
        request = envelope.request
        if envelope.model_call_record is not None:
            state["model_call_records"] = [
                *state.get("model_call_records", []),
                envelope.model_call_record,
            ]
        if envelope.assistant_message is not None:
            _append_turn_messages(state, (envelope.assistant_message,))
        if envelope.context_revision is not None:
            state["context_revision"] = envelope.context_revision
        if request is None:
            return
        state["prompt_revision"] = request.prompt_revision
        serializer_revision = (
            envelope.provider_serializer_revision or state.get("provider_serializer_revision") or "provider-wire-v1"
        )
        state["provider_serializer_revision"] = serializer_revision
        selected = tuple(self._registry_snapshot[name] for name in request.exposed_tool_names)
        resident = tuple(name for name in state.get("resident_tool_names", ()) if name in request.exposed_tool_names)
        explicit = tuple(
            name
            for name in state.get("explicit_tool_names", ())
            if name in request.exposed_tool_names and name not in resident
        )
        active = tuple(
            name
            for name in state.get("active_tool_names", ())
            if name in request.exposed_tool_names and name not in resident and name not in explicit
        )
        state["tool_manifest"] = build_tool_manifest(
            tools=selected,
            resident_tool_names=resident,
            explicit_tool_names=explicit,
            active_tool_names=active,
            provider_serializer_revision=serializer_revision,
        )
        for plan in envelope.draft.tool_calls:
            origin = plan.origin or ToolCallOrigin(
                request_id=request.request_id,
                toolset_revision=request.toolset_revision,
                exposed_tool_names=request.exposed_tool_names,
            )
            state["canonical_tool_calls"][plan.tool_call_id] = ToolCall(
                tool_call_id=plan.tool_call_id,
                tool_name=plan.tool_name,
                arguments=_canonicalize_tool_arguments(
                    self._registry_snapshot,
                    tool_name=plan.tool_name,
                    arguments=cast(
                        Mapping[str, JsonValue],
                        plan.arguments,
                    ),
                ),
                origin=origin,
            )

    async def _handle_overflow(
        self,
        state: LoopState,
        exc: Exception,
    ) -> tuple[ModelTurn | None, int]:
        """Reactive compaction on context overflow.  Returns (turn, 0) to retry, (None, _) if failed."""
        if state["memory_state"].reactive_compact_used:
            append_loop_diagnostic(
                state,
                RuntimeDiagnostic.from_exception(
                    code="context_overflow", component="agent_loop", error=exc, severity="error"
                ),
            )
            await self._fail(
                state,
                stop_reason="context_overflow",
                error=str(exc) or type(exc).__name__,
                transition_reason="failed",
                checkpoint_reason="context_overflow",
            )
            return None, 0

        state["memory_state"].reactive_compact_used = True
        compact = getattr(self._context_manager, "reactive_compact", None)
        if compact is None:
            await self._fail(
                state,
                stop_reason="context_overflow",
                error="Context overflow, no reactive compaction available.",
                transition_reason="failed",
                checkpoint_reason="context_overflow",
            )
            return None, 0

        try:
            result = await _await_value(compact(state))
        except Exception as compact_exc:
            await self._fail(
                state,
                stop_reason="context_compaction_failed",
                error=str(compact_exc) or type(compact_exc).__name__,
                transition_reason="failed",
                checkpoint_reason="context_compaction_failed",
            )
            return None, 0

        append_loop_diagnostic(
            state,
            RuntimeDiagnostic.from_exception(
                code="context_overflow_recovered", component="agent_loop", error=exc, severity="warning"
            ),
        )

        if not result.changed:
            await self._fail(
                state,
                stop_reason="context_overflow",
                error="Reactive compaction did not free context space.",
                transition_reason="failed",
                checkpoint_reason="context_overflow",
            )
            return None, 0

        state["iteration"] = max(0, state["iteration"] - 1)
        tr = LoopTransition(
            reason="compaction",
            iteration=state["iteration"],
            detail={"mode": "reactive", "channels": list(result.channels), "warnings": list(result.warnings)},
        )
        replace_latest_transition(state, tr)
        await self._event_sink.emit(tr)
        await self._emit_stream(
            compact_layer(
                channels=list(result.channels),
                warnings=list(result.warnings),
                turn_id=state["run_config"].turn_id,
                iteration=state["iteration"],
            )
        )
        return None, 0  # caller will retry (turn is None, but state is still running)

    async def _handle_invalid_turn(
        self,
        state: LoopState,
        exc: Exception,
        retries: int,
    ) -> tuple[ModelTurn | None, int]:
        """Validation error or bad model output.  Retry or fail."""
        if isinstance(exc, FinishCandidateBuildError) and _has_tool_error(state):
            append_loop_diagnostic(
                state,
                RuntimeDiagnostic.from_exception(
                    code="tool_error", component="agent_loop", error=exc, severity="error"
                ),
            )
            await self._fail(
                state,
                stop_reason="tool_error",
                error=_latest_tool_error_message(state),
                transition_reason="failed",
                checkpoint_reason="tool_error",
            )
            return None, retries

        append_loop_diagnostic(
            state,
            RuntimeDiagnostic.from_exception(
                code="invalid_model_turn", component="agent_loop", error=exc, severity="error"
            ),
        )

        if retries >= self._max_model_retries:
            await self._fail(
                state,
                stop_reason="invalid_model_turn",
                error=str(exc) or type(exc).__name__,
                transition_reason="failed",
                checkpoint_reason="invalid_model_turn",
            )
            return None, retries

        retries += 1
        await self._emit_stream(
            recovery_event(
                strategy="model_retry",
                detail=f"attempt={retries}, error={str(exc)[:200]}",
                turn_id=state["run_config"].turn_id,
                iteration=state["iteration"],
            )
        )
        await self._transition(
            state,
            reason="retry",
            detail={"component": "model", "attempt": retries, "error": str(exc)},
            checkpoint_reason="model_retry",
        )
        return None, retries  # caller will retry

    async def _handle_provider_error(
        self,
        state: LoopState,
        exc: Exception,
        retries: int,
    ) -> tuple[ModelTurn | None, int]:
        """Model provider failure.  Retry or fail."""
        safe_error = redact_sensitive_text(str(exc) or type(exc).__name__)
        if isinstance(exc, LLMToolCallValidationError):
            append_loop_diagnostic(
                state,
                RuntimeDiagnostic.from_exception(
                    code="model_tool_call_rejected",
                    component="agent_loop",
                    error=exc,
                    severity="warning",
                ),
            )
            feedback: dict[str, JsonValue] = {
                "recovery": "correct_tool_arguments",
                "validation_error": redact_sensitive_text(exc.validation_error),
            }
            if exc.failed_generation:
                feedback["failed_generation"] = redact_sensitive_text(exc.failed_generation)
            _append_turn_messages(
                state,
                (
                    context_event_message(
                        "model_tool_call_rejected",
                        feedback,
                    ),
                ),
            )
            await self._emit_stream(
                recovery_event(
                    strategy="tool_call_correction",
                    detail=redact_sensitive_text(exc.validation_error)[:200],
                    turn_id=state["run_config"].turn_id,
                    iteration=state["iteration"],
                )
            )
            await self._transition(
                state,
                reason="retry",
                detail={
                    "component": "model_tool_call",
                    "attempt": 0,
                    "error": safe_error,
                },
                checkpoint_reason="model_tool_call_rejected",
            )
            return None, 0

        append_loop_diagnostic(
            state,
            RuntimeDiagnostic.from_exception(
                code="model_provider_failed", component="agent_loop", error=exc, severity="error"
            ),
        )

        if retries >= self._max_model_retries:
            await self._fail(
                state,
                stop_reason="model_provider_failed",
                error=safe_error,
                transition_reason="failed",
                checkpoint_reason="model_provider_failed",
            )
            return None, retries

        retries += 1
        await self._emit_stream(
            recovery_event(
                strategy="model_retry",
                detail=f"attempt={retries}, error={safe_error[:200]}",
                turn_id=state["run_config"].turn_id,
                iteration=state["iteration"],
            )
        )
        await self._transition(
            state,
            reason="retry",
            detail={
                "component": "model",
                "attempt": retries,
                "error": safe_error,
            },
            checkpoint_reason="model_retry",
        )
        return None, retries  # caller will retry

    async def _execute_pending_tools(self, state: LoopState) -> bool:
        turn_id = state["run_config"].turn_id
        turn = state["iteration"]
        pending = tuple(state["pending_tool_calls"])
        calls = tuple(self._canonical_call(state, pending_call) for pending_call in pending)
        circuit_checked_calls, circuit_results = _guard_repeated_tool_failures(
            state,
            calls,
        )
        executable_calls, repeated_inspection_results = _guard_repeated_successful_inspections(
            state,
            circuit_checked_calls,
        )
        for call in executable_calls:
            await self._emit_stream(
                tool_use_start(
                    tool_name=call.tool_name,
                    tool_id=call.tool_call_id,
                    input_preview=_tool_input_preview(call.arguments),
                    turn_id=turn_id,
                    iteration=turn,
                )
            )
        context = replace(
            self._execution_context,
            approved_tool_call_ids=frozenset(state["approved_tool_call_ids"]),
            denied_tool_call_ids=frozenset(state["denied_tool_call_ids"]),
            active_skill_ids=(
                frozenset() if self._skill_runtime is None else self._skill_runtime.validated_active_skill_ids(state)
            ),
        )
        executions = await self._tool_executor.execute_batch(
            executable_calls,
            context=context,
            records=state["tool_execution_records"],
            record_sink=self._checkpoint_store.write_execution_record,
        )
        for execution in executions:
            if execution.record is not None:
                state["tool_execution_records"][execution.record.tool_call_id] = execution.record

        approval_executions = tuple(
            execution for execution in executions if execution.result.error_code == "approval_required"
        )
        approval_execution = approval_executions[0] if approval_executions else None
        approval_ids = {execution.result.tool_call_id for execution in approval_executions}
        results_by_id = {
            result.tool_call_id: result
            for result in (
                *circuit_results,
                *repeated_inspection_results,
            )
        }
        results_by_id.update(
            {
                execution.result.tool_call_id: execution.result
                for execution in executions
                if execution.result.tool_call_id not in approval_ids
            }
        )
        new_results = [results_by_id[call.tool_call_id] for call in calls if call.tool_call_id in results_by_id]
        reconciliation_execution = next(
            (
                execution
                for execution in executions
                if execution.record is not None and execution.record.requires_reconciliation
            ),
            None,
        )
        pending_ids = set(approval_ids)
        if reconciliation_execution is not None:
            pending_ids.add(reconciliation_execution.result.tool_call_id)
        state["pending_tool_calls"] = [
            PendingToolCall(plan=item.plan, status="pending") for item in pending if item.tool_call_id in pending_ids
        ]

        observable_results = [result for result in new_results if result.tool_name != "update_plan"]
        batch = self._observation_extractor.extract(
            observable_results,
            seen_tool_call_ids=list(self._observed_tool_call_ids),
        )
        self._observed_tool_call_ids.update(observation.tool_call_id for observation in batch.structured_observations)
        self._merge_observations(
            state,
            batch,
            tool_results=observable_results,
        )

        new_results, plan_updates = self._apply_update_plan_results(
            state,
            calls=calls,
            results=new_results,
        )
        self._apply_activation_results(state, new_results)
        transcript_messages = _tool_result_transcript_messages(
            state,
            new_results,
        )
        state["tool_results"] = _merge_keyed(
            state["tool_results"],
            new_results,
        )
        _append_turn_messages(
            state,
            transcript_messages,
        )
        for tool_result in new_results:
            if tool_result.is_error:
                await self._emit_stream(
                    tool_use_error(
                        tool_id=tool_result.tool_call_id,
                        error=tool_result.error_message or "Unknown error",
                        recoverable=None,
                        turn_id=turn_id,
                        iteration=turn,
                    )
                )
                if tool_result.error_code == _REPEATED_TOOL_FAILURE_CODE:
                    failure_count = tool_result.metadata.get("failure_count", 0)
                    detail = f"tool={tool_result.tool_name}, matching_failures={failure_count}"
                    append_loop_diagnostic(
                        state,
                        RuntimeDiagnostic(
                            code=_REPEATED_TOOL_FAILURE_CODE,
                            component="agent_loop",
                            message=detail,
                            severity="warning",
                        ),
                    )
                    await self._emit_stream(
                        recovery_event(
                            strategy="tool_failure_circuit_breaker",
                            detail=detail,
                            turn_id=turn_id,
                            iteration=turn,
                        )
                    )
            else:
                await self._emit_stream(
                    tool_use_result(
                        tool_name=tool_result.tool_name,
                        tool_id=tool_result.tool_call_id,
                        result=_tool_result_text(tool_result)[:500],
                        elapsed_ms=None,
                        details=_tool_result_event_details(tool_result),
                        turn_id=turn_id,
                        iteration=turn,
                    )
                )

        for plan, event in plan_updates:
            await self._emit_stream(
                _stream_plan_updated(
                    plan=plan,
                    event=event,
                    turn_id=turn_id,
                    iteration=turn,
                )
            )

        if approval_execution is not None:
            approval_call = next(call for call in calls if call.tool_call_id == approval_execution.result.tool_call_id)
            request = _approval_request(
                approval_execution.result,
                approval_call,
            )
            state["approval_request"] = request
            await self._pause(
                state,
                reason=request.question,
                request=request,
                checkpoint_reason="tool_pause",
                transition_reason="approval_required",
            )
            return True
        if reconciliation_execution is not None:
            request = _reconciliation_request(
                reconciliation_execution.result,
                reconciliation_execution.record,
            )
            state["approval_request"] = request
            await self._pause(
                state,
                reason=request.question,
                request=request,
                checkpoint_reason="tool_reconciliation",
                transition_reason="approval_required",
            )
            return True

        await self._transition(
            state,
            reason="tool_execution",
            detail={
                "phase": "recorded",
                "result_count": len(new_results),
                "pending_count": len(state["pending_tool_calls"]),
                "circuit_breaker_count": len(circuit_results),
                "repeated_inspection_count": len(repeated_inspection_results),
            },
            checkpoint_reason="tool_results_recorded",
        )

        # Trim ledger: keep entries for pending + just-completed calls
        active_ids: set[str] = set()
        for p in state["pending_tool_calls"]:
            active_ids.add(p.tool_call_id)
        for tr in new_results:
            active_ids.add(tr.tool_call_id)
        state["tool_call_ledger"].trim(active_tool_call_ids=active_ids)

        # Record tool call metrics.
        self._record_metrics(state, new_results)

        circuit_remained_open = not executions and any(
            result.metadata.get("circuit_already_open") is True for result in circuit_results
        )
        if circuit_remained_open:
            await self._fail(
                state,
                stop_reason=_REPEATED_TOOL_FAILURE_CODE,
                error=("The model repeated an identical tool call after the failure circuit opened."),
                transition_reason="failed",
                checkpoint_reason=_REPEATED_TOOL_FAILURE_CODE,
            )
            return True

        return False

    def _apply_update_plan_results(
        self,
        state: LoopState,
        *,
        calls: Sequence[ToolCall],
        results: Sequence[ToolResult],
    ) -> tuple[list[ToolResult], list[tuple[AgentPlan, PlanEvent]]]:
        calls_by_id = {call.tool_call_id: call for call in calls}
        canonical_results: list[ToolResult] = []
        plan_updates: list[tuple[AgentPlan, PlanEvent]] = []
        for result in results:
            if result.tool_name != "update_plan" or result.is_error:
                canonical_results.append(result)
                continue
            call = calls_by_id[result.tool_call_id]
            submitted = UpdatePlanInput.model_validate(call.arguments)
            current = state["plan_state"].agent_plan
            if current is None:
                current, events = self._plan_tracker.initialize_task(
                    task=state["current_message"],
                )
                self._append_plan_events(state, events)
            steps = [
                PlanStep(
                    step_id=item.step_id or f"step_{index:03d}",
                    title=item.step,
                    status=item.status,
                )
                for index, item in enumerate(submitted.plan, start=1)
            ]
            updated, events = self._plan_tracker.replace_from_tool(
                current,
                steps=steps,
                summary=submitted.explanation,
            )
            state["plan_state"].agent_plan = updated
            self._append_plan_events(state, events)
            event = events[-1]
            plan_updates.append((updated, event))
            canonical_results.append(
                replace(
                    result,
                    structured_content={
                        "accepted": True,
                        "revision": updated.revision,
                        "message": (
                            "Plan persisted as advisory state; tool policy and "
                            "stop hooks retain execution and completion authority."
                        ),
                        "authority": "advisory",
                    },
                )
            )
        return canonical_results, plan_updates

    def _canonical_call(
        self,
        state: LoopState,
        pending: PendingToolCall,
    ) -> ToolCall:
        existing = state["canonical_tool_calls"].get(pending.tool_call_id)
        if existing is not None:
            return existing
        plan = pending.plan
        exposed = tuple(
            dict.fromkeys(
                (
                    *state.get("resident_tool_names", ()),
                    *state.get("explicit_tool_names", ()),
                    *state.get("active_tool_names", ()),
                    plan.tool_name,
                )
            )
        )
        manifest = state.get("tool_manifest")
        origin = plan.origin or ToolCallOrigin(
            request_id=f"{state['run_config'].turn_id}:initial",
            toolset_revision=(manifest.toolset_revision if manifest is not None else "tools_initial"),
            exposed_tool_names=exposed,
        )
        call = ToolCall(
            tool_call_id=plan.tool_call_id,
            tool_name=plan.tool_name,
            arguments=_canonicalize_tool_arguments(
                self._registry_snapshot,
                tool_name=plan.tool_name,
                arguments=cast(Mapping[str, JsonValue], plan.arguments),
            ),
            origin=origin,
        )
        state["canonical_tool_calls"][call.tool_call_id] = call
        return call

    def _apply_activation_results(
        self,
        state: LoopState,
        results: Sequence[ToolResult],
    ) -> None:
        for result in results:
            if (
                result.tool_name == "invoke_skill"
                and not result.is_error
                and self._skill_runtime is not None
                and isinstance(result.structured_content, Mapping)
            ):
                self._skill_runtime.apply_activation_event(
                    state,
                    result.structured_content,
                    iteration=state["iteration"],
                )
                continue
            if result.tool_name != FIND_TOOLS_NAME or result.is_error:
                continue
            proposals = result.metadata.get("proposed_activation_names", ())
            if not isinstance(proposals, Sequence) or isinstance(
                proposals,
                (str, bytes),
            ):
                continue
            reduction = reduce_tool_activation(
                self._registry_snapshot,
                proposed_names=tuple(str(name) for name in proposals),
                resident_names=(
                    *state.get("resident_tool_names", ()),
                    *state.get("explicit_tool_names", ()),
                ),
                active_names=state.get("active_tool_names", ()),
                disabled_names=state.get("disabled_tool_names", ()),
                max_active_tools=self._definition.max_active_deferred_tools,
                discoverable_names=self._discoverable_tool_names,
            )
            state["active_tool_names"] = list(reduction.active_names)

    def _record_metrics(
        self,
        state: LoopState,
        new_results: list[ToolResult],
    ) -> None:
        """Accumulate structured ToolCallMetrics across turns.

        ToolCallMetrics is stored in LoopState and exposed via AgentRunResult.
        The diagnostic string is kept as a compact summary for inline display.
        """
        from agent_runtime.core.runtime_diagnostics import ToolCallMetrics

        prev = state.get("tool_call_metrics")
        if not isinstance(prev, ToolCallMetrics):
            prev = ToolCallMetrics()

        native = sum(1 for tr in new_results if tr.tool_name in _NATIVE_TOOL_SET)
        deferred = sum(
            1
            for result in new_results
            if result.tool_name not in _NATIVE_TOOL_SET and not result.tool_name.startswith("mcp__")
        )
        mcp_calls = sum(1 for tr in new_results if tr.tool_name.startswith("mcp__"))
        native_err = sum(1 for tr in new_results if tr.tool_name in _NATIVE_TOOL_SET and tr.is_error)
        mcp_err = sum(1 for tr in new_results if tr.tool_name.startswith("mcp__") and tr.is_error)
        durations = {trace.tool_call_id: trace.duration_ms for trace in self._tool_executor.traces}
        native_lat = sum(durations.get(tr.tool_call_id, 0.0) for tr in new_results if tr.tool_name in _NATIVE_TOOL_SET)
        mcp_lat = sum(durations.get(tr.tool_call_id, 0.0) for tr in new_results if tr.tool_name.startswith("mcp__"))

        metrics = prev.model_copy(
            update={
                "native_calls": prev.native_calls + native,
                "native_errors": prev.native_errors + native_err,
                "native_latency_ms_total": prev.native_latency_ms_total + native_lat,
                "deferred_calls": prev.deferred_calls + deferred,
                "mcp_calls": prev.mcp_calls + mcp_calls,
                "mcp_errors": prev.mcp_errors + mcp_err,
                "mcp_latency_ms_total": prev.mcp_latency_ms_total + mcp_lat,
            },
        )
        state["tool_call_metrics"] = metrics  # type: ignore[typeddict-unknown-key]

        # Keep compact diagnostic summary for inline display
        msg = (
            f"native={metrics.native_calls}/{metrics.native_errors}err "
            f"deferred={metrics.deferred_calls} "
            f"mcp={metrics.mcp_calls}/{metrics.mcp_errors}err "
            f"lat={metrics.native_latency_ms_total:.0f}/{metrics.mcp_latency_ms_total:.0f}ms"
        )
        state["runtime_diagnostics"] = [
            *state["runtime_diagnostics"],
            RuntimeDiagnostic(
                code="tool_call_metrics",
                component="AgentLoop",
                message=msg[:500],
                severity="warning",
                degraded=False,
            ),
        ][-20:]

    async def _evaluate_finish(
        self,
        state: LoopState,
        turn: ModelTurn,
    ) -> bool:
        candidate = cast(str, turn.final_answer)
        outcome = await self._stop_hook_runner.evaluate(
            state=state,
            candidate=candidate,
        )
        if outcome.blocked:
            await self._transition(
                state,
                reason="stop_hook_blocked",
                detail={
                    "code": outcome.code,
                    "message": outcome.message or "",
                },
                checkpoint_reason="stop_hook_blocked",
            )
            return False
        if outcome.halted:
            await self._fail(
                state,
                stop_reason=outcome.code,
                error=outcome.message or outcome.code,
                transition_reason="failed",
                checkpoint_reason="stop_hook_halt",
                final_output=outcome.final_output,
            )
            return True
        await self._complete(
            state,
            candidate=candidate,
            outcome=outcome,
        )
        return True

    async def _complete(
        self,
        state: LoopState,
        *,
        candidate: str,
        outcome: StopHookOutcome,
    ) -> None:
        transcript = state["turn_transcript"]
        if (
            not transcript
            or transcript[-1].role != "assistant"
            or transcript[-1].content != candidate
            or transcript[-1].tool_calls
        ):
            _append_turn_messages(
                state,
                (ModelMessage(role="assistant", content=candidate),),
            )
        state["status"] = "completed"
        state["finish_state"].final_answer = candidate
        state["finish_state"].final_output = outcome.final_output
        state["pause"] = None
        state["terminal"] = LoopTerminal(
            status="completed",
            stop_reason=outcome.code,
            final_answer=candidate,
            final_output=outcome.final_output,
        )
        plan, events = self._plan_tracker.record_completion(state["plan_state"].agent_plan)
        state["plan_state"].agent_plan = plan
        state["plan_state"].plan_events = [*state["plan_state"].plan_events, *events][-MAX_PLAN_EVENTS:]
        await self._transition(
            state,
            reason="finished",
            detail={"stop_code": outcome.code},
            checkpoint_reason="terminal_completed",
        )
        await self._emit_stream(
            loop_end(
                turn_id=state["run_config"].turn_id,
                reason=outcome.code,
                total_turns=state["iteration"],
            )
        )

    async def _fail(
        self,
        state: LoopState,
        *,
        stop_reason: str,
        error: str,
        transition_reason: LoopTransitionReason,
        checkpoint_reason: str,
        final_output: ValidatedFinalOutput | None = None,
    ) -> None:
        error = redact_sensitive_text(error)
        state["status"] = "failed"
        state["pause"] = None
        state["terminal"] = LoopTerminal(
            status="failed",
            stop_reason=stop_reason,
            final_output=final_output,
            error=error,
        )
        plan, events = self._plan_tracker.record_completion(
            state["plan_state"].agent_plan,
            blocked=True,
        )
        state["plan_state"].agent_plan = plan
        state["plan_state"].plan_events = [*state["plan_state"].plan_events, *events][-MAX_PLAN_EVENTS:]
        await self._transition(
            state,
            reason=transition_reason,
            detail={
                "stop_reason": stop_reason,
                "error": error,
            },
            checkpoint_reason=checkpoint_reason,
        )
        # ── 流式事件：loop 结束 ──
        await self._emit_stream(
            loop_end(
                turn_id=state["run_config"].turn_id,
                reason=stop_reason,
                total_turns=state["iteration"],
            )
        )

    async def _pause(
        self,
        state: LoopState,
        *,
        reason: str,
        request: HumanInputRequest | None,
        checkpoint_reason: str,
        transition_reason: LoopTransitionReason,
    ) -> None:
        state["status"] = "paused"
        state["pause"] = LoopPause(
            reason=reason,
            request=request,
        )
        await self._transition(
            state,
            reason=transition_reason,
            detail={
                "reason": reason,
                "request_kind": (getattr(request, "kind", None) if request is not None else None),
            },
            checkpoint_reason=checkpoint_reason,
        )
        if request is not None:
            await self._emit_stream(
                _stream_human_input_required(
                    request=request,
                    turn_id=state["run_config"].turn_id,
                    iteration=state["iteration"],
                )
            )
        await self._emit_stream(
            loop_end(
                turn_id=state["run_config"].turn_id,
                reason=str(transition_reason),
                total_turns=state["iteration"],
            )
        )

    async def _transition(
        self,
        state: LoopState,
        *,
        reason: LoopTransitionReason,
        detail: dict[str, object],
        checkpoint_reason: str,
        checkpoint: bool = True,
    ) -> None:
        transition = LoopTransition(
            reason=reason,
            iteration=state["iteration"],
            detail=detail,
        )
        replace_latest_transition(state, transition)
        await self._event_sink.emit(transition)
        if checkpoint:
            await self._checkpoint_store.save_snapshot(
                state,
                reason=checkpoint_reason,
            )

    @staticmethod
    def _append_plan_events(
        state: LoopState,
        events: Sequence[PlanEvent],
    ) -> None:
        state["plan_state"].plan_events = [
            *state["plan_state"].plan_events,
            *events,
        ][-MAX_PLAN_EVENTS:]

    @staticmethod
    def _merge_observations(
        state: LoopState,
        batch: ObservationBatch,
        *,
        tool_results: Sequence[ToolResult] = (),
    ) -> None:
        memory_state = state["memory_state"]
        observation_limit = state["run_config"].memory_policy.reactive_compact_max_observations
        locator_limit = state["run_config"].memory_policy.reactive_compact_max_evidence
        observations_by_id = {observation.tool_call_id: observation for observation in memory_state.recent_observations}
        for observation in batch.structured_observations:
            observations_by_id[observation.tool_call_id] = observation

        locators_by_key = {
            json.dumps(locator, ensure_ascii=False, sort_keys=True): locator for locator in memory_state.known_locators
        }
        new_locators = [
            *batch.locators,
            *[locator for observation in batch.structured_observations for locator in observation.locators],
        ]
        for locator in new_locators:
            locators_by_key[json.dumps(locator, ensure_ascii=False, sort_keys=True)] = locator
        verified_paths = dict.fromkeys(memory_state.verified_workspace_paths)
        for path in grounded_workspace_paths(
            locators=new_locators,
            tool_results=(
                *state["tool_results"],
                *tool_results,
            ),
            tool_calls=state["canonical_tool_calls"],
        ):
            verified_paths.setdefault(path, None)

        state["memory_state"] = memory_state.model_copy(
            update={
                "recent_observations": list(observations_by_id.values())[-observation_limit:],
                "verified_workspace_paths": list(verified_paths),
                "known_locators": list(locators_by_key.values())[-locator_limit:],
            }
        )


async def _await_value[T](value: T | Awaitable[T]) -> T:
    if isawaitable(value):
        return await value
    return value


async def _remaining_llm_budget(handles: Any) -> int | None:
    ledger = getattr(handles, "llm_budget_ledger", None)
    if ledger is None:
        return None
    return cast(int, await ledger.remaining())


def _delivery_cycle_results(state: LoopState) -> list[ToolResult]:
    cycle: list[ToolResult] = []
    for result in reversed(state["tool_results"]):
        if runtime_workspace_change(result) is not None:
            break
        if result.tool_name != "update_plan" and result.tool_name not in _EXPLORATION_TOOL_NAMES:
            break
        cycle.append(result)
    cycle.reverse()
    return cycle


def _guard_repeated_tool_failures(
    state: LoopState,
    calls: Sequence[ToolCall],
) -> tuple[tuple[ToolCall, ...], tuple[ToolResult, ...]]:
    executable: list[ToolCall] = []
    blocked: list[ToolResult] = []
    for call in calls:
        if call.tool_call_id in state["tool_execution_records"]:
            # Recorded calls belong to ToolExecutor's replay/reconciliation
            # path. The circuit only correlates new model-generated attempts.
            executable.append(call)
            continue
        failures = _matching_tool_failures_since_recovery(state, call)
        if not failures:
            executable.append(call)
            continue
        failure_limit = _MAX_RETRYABLE_IDENTICAL_FAILURES if all(result.retryable for result in failures) else 1
        if len(failures) < failure_limit:
            executable.append(call)
            continue
        already_open = any(result.error_code == _REPEATED_TOOL_FAILURE_CODE for result in failures)
        blocked.append(
            ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                is_error=True,
                error_code=_REPEATED_TOOL_FAILURE_CODE,
                error_message=(
                    "Repeated identical tool call blocked after "
                    f"{len(failures)} matching failure(s) without a successful "
                    "recovery. Change the arguments, use a different tool, or "
                    "finish with the available evidence."
                ),
                retryable=False,
                structured_content={
                    "repeated_failure": True,
                    "original_tool_call_id": failures[0].tool_call_id,
                    "failure_count": len(failures),
                    "last_error_code": (failures[-1].error_code or "unknown"),
                },
                metadata={
                    "failure_count": len(failures),
                    "last_error_code": failures[-1].error_code or "unknown",
                    "circuit_already_open": already_open,
                },
            )
        )
    return tuple(executable), tuple(blocked)


def _guard_repeated_successful_inspections(
    state: LoopState,
    calls: Sequence[ToolCall],
) -> tuple[tuple[ToolCall, ...], tuple[ToolResult, ...]]:
    successful_cycle_results = tuple(
        result
        for result in _delivery_cycle_results(state)
        if not result.is_error and result.tool_name in _STABLE_INSPECTION_TOOL_NAMES
    )
    executable: list[ToolCall] = []
    blocked: list[ToolResult] = []
    for call in calls:
        if call.tool_call_id in state["tool_execution_records"] or call.tool_name not in _STABLE_INSPECTION_TOOL_NAMES:
            executable.append(call)
            continue
        previous = next(
            (
                result
                for result in reversed(successful_cycle_results)
                if (
                    (previous_call := state["canonical_tool_calls"].get(result.tool_call_id)) is not None
                    and _same_tool_invocation(previous_call, call)
                )
            ),
            None,
        )
        if previous is None:
            executable.append(call)
            continue
        blocked.append(
            ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                is_error=True,
                error_code="repeated_inspection",
                error_message=(
                    "This exact read-only inspection already succeeded in the "
                    "current unchanged workspace state. Reuse that result. If "
                    "it shows the requested state and no distinct requirement "
                    "remains, finish now. Do not repeat the mutation or request "
                    "run_command or execute_python solely to reconfirm it. Do not switch to another "
                    "inspection tool or pair positive and negative searches just "
                    "for stronger confirmation. Use another tool only for a "
                    "specific unmet requirement."
                ),
                retryable=False,
                structured_content={
                    "repeated_inspection": True,
                    "previous_tool_call_id": previous.tool_call_id,
                    "recommended_action": (
                        "finish_if_existing_result_satisfies_task"
                    ),
                    "do_not_escalate_for_reconfirmation": True,
                    "maximum_additional_file_inspections_for_same_claim": 0,
                    "do_not_substitute_another_inspection_tool": True,
                },
                metadata={
                    "previous_tool_call_id": previous.tool_call_id,
                },
            )
        )
    return tuple(executable), tuple(blocked)


def _matching_tool_failures_since_recovery(
    state: LoopState,
    call: ToolCall,
) -> tuple[ToolResult, ...]:
    failures: list[ToolResult] = []
    for result in reversed(state["tool_results"]):
        if (
            not result.is_error
            and result.tool_name != "update_plan"
            and result.tool_name not in _EXPLORATION_TOOL_NAMES
        ):
            break
        previous_call = state["canonical_tool_calls"].get(result.tool_call_id)
        if previous_call is None or not _same_failed_operation(previous_call, call, result):
            continue
        if not result.is_error:
            break
        failures.append(result)
    failures.reverse()
    return tuple(failures)


def _tool_result_transcript_messages(
    state: LoopState,
    results: Sequence[ToolResult],
) -> tuple[ModelMessage, ...]:
    seen: dict[str, tuple[str, int]] = {}
    prior_failures: list[ToolResult] = []
    for result in reversed(state["tool_results"]):
        if not result.is_error:
            break
        prior_failures.append(result)
    for result in reversed(prior_failures):
        fingerprint = _tool_failure_evidence_fingerprint(result)
        if fingerprint is None:
            continue
        original_id, count = seen.get(
            fingerprint,
            (result.tool_call_id, 0),
        )
        seen[fingerprint] = (original_id, count + 1)

    messages: list[ModelMessage] = []
    for result in results:
        if not result.is_error:
            seen.clear()
            messages.append(tool_result_message(result))
            continue
        fingerprint = _tool_failure_evidence_fingerprint(result)
        previous = None if fingerprint is None else seen.get(fingerprint)
        if fingerprint is None or previous is None:
            messages.append(tool_result_message(result))
            if fingerprint is not None:
                seen[fingerprint] = (result.tool_call_id, 1)
            continue
        original_id, count = previous
        repeat_count = count + 1
        visible_result = replace(
            result,
            content=(),
            structured_content={
                "repeated_failure": True,
                "evidence_fingerprint": fingerprint,
                "original_tool_call_id": original_id,
                "repeat_count": repeat_count,
            },
        )
        messages.append(tool_result_message(visible_result))
        seen[fingerprint] = (original_id, repeat_count)
    return tuple(messages)


def _tool_failure_evidence_fingerprint(
    result: ToolResult,
) -> str | None:
    if not result.is_error:
        return None
    structured_content = result.structured_content
    if result.tool_name == "run_command" and isinstance(
        structured_content,
        Mapping,
    ):
        structured_content = {key: value for key, value in structured_content.items() if key != "duration_ms"}
    visible_content = tuple(
        {
            "type": block.type,
            "data": block.data,
        }
        for block in result.content
    )
    payload = cast(
        JsonValue,
        {
            "tool_name": result.tool_name,
            "content": visible_content,
            "structured_content": structured_content,
            "is_error": result.is_error,
            "error_code": result.error_code,
            "error_message": result.error_message,
            "retryable": result.retryable,
            "truncated": result.truncated,
        },
    )
    return hashlib.sha256(canonical_json_text(payload).encode("utf-8")).hexdigest()


def _same_tool_invocation(left: ToolCall, right: ToolCall) -> bool:
    return left.tool_name == right.tool_name and left.arguments == right.arguments


def _same_failed_operation(
    left: ToolCall,
    right: ToolCall,
    result: ToolResult,
) -> bool:
    if left.tool_name != right.tool_name:
        return False
    if left.tool_name == "run_command" and result.error_code == "command_failed":
        left_arguments = dict(left.arguments)
        right_arguments = dict(right.arguments)
        left_arguments.pop("timeout_seconds", None)
        right_arguments.pop("timeout_seconds", None)
        return left_arguments == right_arguments
    return left.arguments == right.arguments


def _canonicalize_tool_arguments(
    registry_snapshot: Mapping[str, Tool],
    *,
    tool_name: str,
    arguments: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    """Materialize schema defaults before identity, policy, and execution."""

    tool = registry_snapshot.get(tool_name)
    if tool is None:
        return arguments
    try:
        return tool.validate_input(arguments)
    except Exception:
        # ToolExecutor remains the canonical fail-closed validation boundary.
        return arguments


def _merge_keyed[T](existing: list[T], additions: list[T]) -> list[T]:
    merged: dict[str, T] = {}
    for item in [*existing, *additions]:
        key = _item_key(item)
        merged.pop(key, None)
        merged[key] = item
    return list(merged.values())


def _item_key(item: object) -> str:
    key = getattr(item, "key", None)
    if isinstance(key, str) and key:
        return key
    for attribute in (
        "tool_call_id",
        "source_tool_call_id",
        "unit_id",
        "evidence_id",
        "citation_id",
    ):
        value = getattr(item, attribute, None)
        if value is not None:
            return f"{attribute}:{value}"
    if isinstance(item, dict):
        return repr(sorted(item.items()))
    return repr(item)


def _has_tool_error(state: LoopState) -> bool:
    return any(result.is_error for result in state["tool_results"])


def _latest_tool_error_message(state: LoopState) -> str:
    for result in reversed(state["tool_results"]):
        if not result.is_error:
            continue
        return result.error_message or f"Tool {result.tool_name} failed."
    return "Tool execution failed."


def _tool_result_text(result: ToolResult) -> str:
    if result.structured_content is not None:
        return str(result.structured_content)
    return "\n".join(str(block.data.get("text", "")) for block in result.content if block.type == "text")


def _tool_result_event_details(result: ToolResult) -> dict[str, Any]:
    if result.tool_name != "apply_patch":
        return {}
    file_path = result.metadata.get("file_path")
    diff = result.metadata.get("diff")
    diff_truncated = result.metadata.get("diff_truncated")
    if not isinstance(file_path, str) or not isinstance(diff, str) or type(diff_truncated) is not bool:
        return {}
    return {
        "file_path": file_path,
        "diff": diff,
        "diff_truncated": diff_truncated,
    }


_SENSITIVE_ARGUMENT_PARTS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


def _tool_input_preview(arguments: Mapping[str, object]) -> str:
    items = [
        f"{str(key)[:80]}={_preview_value(value, key=str(key), depth=0)}" for key, value in list(arguments.items())[:8]
    ]
    return ", ".join(items)[:500]


def _preview_value(value: object, *, key: str, depth: int) -> str:
    lowered = key.lower()
    if any(part in lowered for part in _SENSITIVE_ARGUMENT_PARTS):
        return "<redacted>"
    if depth >= 2:
        return "<nested>"
    if isinstance(value, Mapping):
        rendered = ", ".join(
            f"{str(child_key)[:40]}: {_preview_value(child, key=str(child_key), depth=depth + 1)}"
            for child_key, child in list(value.items())[:5]
        )
        return "{" + rendered + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rendered = ", ".join(_preview_value(item, key=key, depth=depth + 1) for item in list(value)[:5])
        return "[" + rendered + "]"
    text = str(value).replace("\n", " ")
    return repr(text[:120]) if isinstance(value, str) else text[:120]


def _approval_request(
    result: ToolResult,
    call: ToolCall,
) -> HumanInputRequest:
    reason = result.error_message or "Tool execution requires approval."
    approval_id = result.metadata.get("approval_id")
    if not isinstance(approval_id, str) or not approval_id:
        approval_id = result.tool_call_id
    approval_scope = result.metadata.get("approval_scope")
    if approval_scope not in {"tool", "network"}:
        approval_scope = "tool"
    cwd = result.metadata.get("cwd")
    execution_mode = result.metadata.get("execution_mode")
    network_requested = result.metadata.get("network_requested") is True
    workspace_path = result.metadata.get("workspace_path")
    workspace_write = result.metadata.get("workspace_write") is True
    args_preview = _tool_input_preview(call.arguments)
    question = f"Allow {result.tool_name} to run once? {reason}"
    if result.tool_name == "run_command":
        args_preview = _run_command_approval_preview(
            call.arguments,
            cwd=cwd if isinstance(cwd, str) else None,
            execution_mode=(execution_mode if isinstance(execution_mode, str) else "restricted_sandbox"),
            network_requested=network_requested,
            workspace_path=(workspace_path if isinstance(workspace_path, str) else None),
            workspace_write=workspace_write,
        )
        if approval_scope == "network":
            question = f"Allow network access for this run_command invocation? {reason}"
        else:
            question = f"Allow run_command to execute once in restricted_sandbox mode? {reason}"
            if workspace_write:
                question += (
                    " This approval grants destructive write access to the entire workspace except Git metadata."
                )
            if network_requested:
                question += " Network access is not included in this approval."
    return HumanInputRequest(
        request_id=f"hir_{uuid4().hex[:12]}",
        kind="tool_approval",
        question=question,
        tool_calls=[
            ToolCallSummary(
                tool_call_id=result.tool_call_id,
                approval_id=approval_id,
                tool_name=result.tool_name,
                args_preview=args_preview,
                risk_level="medium",
                reason=reason,
            )
        ],
        context={
            "tool_call_id": result.tool_call_id,
            "approval_id": approval_id,
            "approval_scope": approval_scope,
            "cwd": cwd,
            "network_requested": network_requested,
            "execution_mode": execution_mode,
            "workspace_path": workspace_path,
            "workspace_write": workspace_write,
        },
        options=["allow_once", "deny"],
    )


def _run_command_approval_preview(
    arguments: Mapping[str, object],
    *,
    cwd: str | None,
    execution_mode: str,
    network_requested: bool,
    workspace_path: str | None,
    workspace_write: bool,
) -> str:
    command = arguments.get("command")
    raw_command = command if isinstance(command, str) else str(command)
    command_text = json.dumps(raw_command, ensure_ascii=False)
    cwd_text = json.dumps(
        cwd or str(arguments.get("working_dir", ".")),
        ensure_ascii=False,
    )
    network_text = "requested (separate approval required)" if network_requested else "disabled"
    workspace_path_text = json.dumps(
        workspace_path or "<unknown>",
        ensure_ascii=False,
    )
    workspace_write_text = (
        f"requested for entire workspace {workspace_path_text} (Git metadata remains read-only)"
        if workspace_write
        else "disabled (workspace is read-only)"
    )
    return (
        f"command: {command_text}\n"
        f"cwd: {cwd_text}\n"
        f"workspace write: {workspace_write_text}\n"
        f"network: {network_text}\n"
        f"execution mode: {execution_mode}"
    )


def _reconciliation_request(
    result: ToolResult,
    record: ToolExecutionRecord | None,
) -> HumanInputRequest:
    return HumanInputRequest(
        request_id=f"hir_{uuid4().hex[:12]}",
        kind="tool_reconciliation",
        question=(
            f"Tool {result.tool_name} has an unknown external outcome; choose how to reconcile it before continuing."
        ),
        context={
            "tool_call_id": result.tool_call_id,
            "tool_name": result.tool_name,
            "operation_id": None if record is None else record.operation_id,
            "execution_status": (None if record is None else record.status.value),
        },
        options=["mark_completed", "mark_failed"],
    )


def _stream_human_input_required(
    *,
    request: HumanInputRequest,
    turn_id: str,
    iteration: int,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.HUMAN_INPUT_REQUIRED,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        data={
            "request_id": request.request_id,
            "kind": request.kind,
            "question": request.question,
            "tool_calls": tuple(
                {
                    "tool_call_id": item.tool_call_id,
                    "approval_id": item.approval_id,
                    "tool_name": item.tool_name,
                    "input_preview": item.args_preview,
                    "reason": item.reason,
                }
                for item in request.tool_calls
            ),
        },
    )


def _stream_plan_updated(
    *,
    plan: AgentPlan,
    event: PlanEvent,
    turn_id: str,
    iteration: int,
) -> StreamEvent:
    return StreamEvent(
        type=EventType.PLAN_UPDATED,
        turn_id=turn_id,
        iteration=iteration,
        sequence=next_sequence(),
        data={
            "plan": plan.model_dump(mode="json"),
            "event": event.model_dump(mode="json"),
        },
    )


__all__ = [
    "AgentLoop",
    "LoopContextManager",
    "LoopEventSink",
    "ModelTurnEnvelope",
    "ModelTurnProvider",
    "NullLoopEventSink",
]

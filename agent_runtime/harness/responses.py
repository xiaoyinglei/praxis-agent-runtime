"""Consume committed responses and hand tool/final outputs to their owners."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from agent_runtime.harness import completion_ops, tool_ops, turn_ops
from agent_runtime.harness.execution_facts import (
    admit_tool_batch,
    canonical_tool_results,
    repeated_tool_failure,
    response_handled,
)
from agent_runtime.harness.protocol import (
    HarnessModelResponse,
    ModelToolCall,
    PreparedModelCall,
    TurnResult,
    restore_model_response,
)
from agent_runtime.harness.records import ItemSnapshot
from agent_runtime.harness.runtime_store import RuntimeStore
from agent_runtime.harness.stop import TurnStop
from agent_runtime.harness.tool_orchestrator import (
    ToolApprovalRequiredError,
    ToolOrchestrator,
)
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin


class ResponseHandler:
    """Consume committed model responses; never dispatch a new model request."""

    def __init__(
        self,
        *,
        store: RuntimeStore,
        tool_orchestrator: ToolOrchestrator | None,
        stop: TurnStop,
        commit: Callable[[Callable[[], Any]], Awaitable[Any]],
        fail_turn: Callable[..., Awaitable[TurnResult]],
    ) -> None:
        self._store = store
        self._tool_orchestrator = tool_orchestrator
        self._stop = stop
        self._commit = commit
        self._fail_turn = fail_turn

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
            and (await self._store.run_async(lambda: self._store.read_budget_state(turn_id).used.total_tokens))
            > token_budget
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
        else:
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
        return await self._stop.finish(
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
        source = await self._store.run_async(self._store.read_item, source_item_id)
        calls = restore_model_response(source.payload).tool_calls
        if any(call.name == "record_delivery" and call.arguments.get("action") == "submit" for call in calls):
            handled = await self._stop.finish(
                thread_id=thread_id,
                turn_id=turn_id,
                answer=None,
                input_revision=revision,
                source_item_id=source_item_id,
            )
            if handled is not None:
                return handled
        call_ids = {call.id for call in calls}
        failure = await self._store.run_async(
            lambda: next(
                (
                    result
                    for _, result in reversed(canonical_tool_results(self._store, turn_id))
                    if result.get("tool_call_id") in call_ids
                    and result.get("is_error") is True
                    and (result.get("error_code") != "input_superseded")
                ),
                None,
            )
        )
        if failure is None or not (await self._store.run_async(repeated_tool_failure, self._store, turn_id)):
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

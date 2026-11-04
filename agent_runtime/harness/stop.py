"""Optional, explicit stop conditions and durable end-of-turn handling.

An accepted stop decision means only that no configured condition blocks stopping.
Raw tool evidence remains authoritative; no task-correctness score is produced.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, cast

from agent_runtime.harness import completion_ops, turn_ops
from agent_runtime.harness.operation_values import TurnChildrenPendingError, TurnInputChangedError
from agent_runtime.harness.protocol import (
    CompletionDecision,
    CompletionGate,
    CompletionProposal,
    TurnResult,
    restore_model_response,
)
from agent_runtime.harness.records import ItemSnapshot, ToolOperationSnapshot
from agent_runtime.harness.runtime_store import RuntimeStore


class StopChecks:
    def __init__(self, store: RuntimeStore) -> None:
        self._store = store

    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        policy = self._store.read_turn(proposal.turn_id).binding_manifest.get("completion_policy", {})
        if not isinstance(policy, Mapping):
            return CompletionDecision(
                "fail", "The frozen stop policy is invalid.", reason_code="invalid_completion_policy"
            )
        # Historical policies retain their explicit requirements, without reviving
        # inferred verifier classification. Saved decisions/wire stay immutable.
        return self._evaluate(proposal, policy)

    def _evaluate(self, proposal: CompletionProposal, policy: Mapping[str, object]) -> CompletionDecision:
        """Keep lifecycle, recorded mutations and user-selected acceptance distinct."""
        require_change = policy.get("require_workspace_change", False)
        raw_command = policy.get("acceptance_command")
        command = raw_command.strip() if isinstance(raw_command, str) else None

        def reject(code: str, message: str, **details: object) -> CompletionDecision:
            feedback = {"reason_code": code, "message": message, **details}
            if command:
                feedback["command"] = command
                feedback["working_dir"] = "."
            return CompletionDecision(
                action="continue",
                reason=json.dumps(feedback, ensure_ascii=False),
                reason_code=code,
            )

        if not isinstance(require_change, bool) or (raw_command is not None and not command):
            return reject("invalid_completion_policy", "The frozen completion policy is invalid.")
        if not require_change and command is None:
            return CompletionDecision(
                action="accept",
                reason="The model turn ended; task acceptance was not evaluated.",
            )

        records = self._trusted_results(proposal.turn_id)
        launches: dict[str, tuple[ItemSnapshot, Mapping[str, object]]] = {}
        command_results = []
        for item, operation, arguments in sorted(records, key=lambda entry: entry[0].sequence):
            output = item.payload.get("structured_content")
            if not isinstance(output, Mapping):
                continue
            session_id = output.get("session_id")
            if operation.tool_name == "run_command":
                command_results.append((item, operation, arguments))
                if isinstance(session_id, str) and operation.status == "succeeded":
                    launches[session_id] = (item, arguments)
            elif (
                operation.tool_name == "manage_command"
                and isinstance(session_id, str)
                and arguments.get("session_id") == session_id
                and session_id in launches
                and output.get("status") in {"completed", "cancelled"}
            ):
                launch, original_arguments = launches[session_id]
                if item.sequence > launch.sequence:
                    command_results.append((item, operation, original_arguments))

        latest_change, result_order = self._workspace_change_order(proposal.turn_id)
        if require_change and not latest_change:
            return reject(
                "workspace_change_missing",
                "The requested workspace mutation has not been recorded.",
                next_action="Make the requested workspace change before finishing.",
            )
        if command is None:
            return CompletionDecision(
                action="accept",
                reason="The requested workspace mutation was recorded; task acceptance was not evaluated.",
            )

        turn = self._store.read_turn(proposal.turn_id)
        workspace = Path(self._store.read_thread(turn.thread_id).workspace).resolve()
        candidates = []
        for item, operation, arguments in command_results:
            actual_command = arguments.get("command")
            if (
                not isinstance(actual_command, str)
                or actual_command.strip() != command
                or item.item_id not in result_order
            ):
                continue
            working_dir = arguments.get("working_dir")
            if working_dir is None or working_dir == "":
                working_dir = "."
            if not isinstance(working_dir, str):
                continue
            try:
                if (workspace / working_dir).resolve() != workspace:
                    continue
            except (OSError, ValueError):
                continue
            candidates.append((item, operation))
        if not candidates:
            return reject(
                "acceptance_command_missing",
                "The configured acceptance command has no trusted execution in this workspace.",
                next_action="Run command exactly with run_command from working_dir and inspect its result.",
            )
        item, operation = max(candidates, key=lambda pair: result_order[pair[0].item_id])
        result = item.payload.get("structured_content")
        exit_code = result.get("exit_code") if isinstance(result, Mapping) else None
        if isinstance(result, Mapping) and result.get("status") == "running":
            return reject(
                "acceptance_command_pending",
                "Observe the configured acceptance command with manage_command.",
                tool_call_id=operation.tool_call_id,
                session_id=result.get("session_id"),
            )
        if (
            operation.status != "succeeded"
            or item.payload.get("is_error") is not False
            or not isinstance(result, Mapping)
            or not isinstance(exit_code, int)
            or isinstance(exit_code, bool)
            or exit_code != 0
            or result.get("status") in {"running", "cancelled"}
            or result.get("timed_out")
            or result.get("sandbox_error")
        ):
            return reject(
                "acceptance_command_failed",
                "The latest configured acceptance command did not complete successfully.",
                tool_call_id=operation.tool_call_id,
                next_action="Inspect that execution's output, resolve the failure, and rerun command from working_dir.",
            )
        # A user-selected verifier may produce build outputs itself. Its own
        # attested mutation is part of that check; a subsequent operation's
        # mutation still requires a fresh execution of the frozen command.
        if result_order[item.item_id] < latest_change:
            return reject(
                "acceptance_command_stale",
                "The workspace changed after the configured acceptance command succeeded.",
                tool_call_id=operation.tool_call_id,
                next_action="Rerun command exactly with run_command from working_dir against the updated workspace.",
            )
        return CompletionDecision(
            action="accept",
            reason="The configured acceptance command completed successfully after the latest workspace mutation.",
        )

    def _workspace_change_order(self, turn_id: str) -> tuple[int, dict[str, int]]:
        """Order trusted shared-workspace mutations across durable descendants.

        Item sequences belong to individual Turns. Rollout record IDs supply
        the common commit order for parent checks and delegated mutations.
        Allocation records, rather than a claimed binding field, establish
        ancestry; unrelated work in the same workspace remains outside it.
        """
        turns = {turn.turn_id: turn for turn in self._store.list_turns()}
        workspaces = {thread.thread_id: Path(thread.workspace).resolve() for thread in self._store.list_threads()}
        workspace = workspaces[turns[turn_id].thread_id]
        children: dict[str, list[str]] = {}
        for child in turns.values():
            if workspaces[child.thread_id] != workspace:
                continue
            try:
                allocation = self._store.read_child_budget_allocation(child.turn_id)
            except KeyError:
                continue
            children.setdefault(str(allocation["parent_turn_id"]), []).append(child.turn_id)
        family: set[str] = set()
        pending = [turn_id]
        while pending:
            current = pending.pop()
            if current not in family:
                family.add(current)
                pending.extend(children.get(current, ()))
        result_order = {
            str(record.payload["item_id"]): record.record_id
            for thread_id in {turns[current].thread_id for current in family}
            for record in self._store.list_records(thread_id)
            if record.turn_id in family
            and record.record_type == "item_completed"
            and record.producer in {"tool", "runtime"}
            and isinstance(record.payload.get("item_id"), str)
        }
        latest_change = max(
            (
                result_order[item.item_id]
                for current in family
                for item, operation, _arguments in self._trusted_results(current)
                if item.item_id in result_order and _is_workspace_change(item, self._write_origin(item, operation))
            ),
            default=0,
        )
        latest_change = max(latest_change, max((
            result_order[item.item_id] for item in self._store.list_items(turn_id)
            if item.kind == "child_workspace" and item.payload.get("status") == "integrated"
            and item.payload.get("verification") == "required" and item.item_id in result_order
        ), default=0))
        return latest_change, result_order

    def _write_origin(self, item: ItemSnapshot, operation: ToolOperationSnapshot) -> ToolOperationSnapshot:
        metadata = item.payload.get("metadata")
        origin_id = metadata.get("runtime_workspace_origin_operation_id") if isinstance(metadata, Mapping) else None
        if isinstance(origin_id, str):
            origin = next(
                (
                    candidate
                    for candidate in self._store.list_tool_operations(operation.turn_id)
                    if candidate.operation_id == origin_id
                ),
                None,
            )
            if origin is not None and origin.result_item_id is not None:
                return origin
        return operation

    def _trusted_results(
        self,
        turn_id: str,
    ) -> tuple[tuple[ItemSnapshot, ToolOperationSnapshot, Mapping[str, object]], ...]:
        items = {item.item_id: item for item in self._store.list_items(turn_id)}
        calls = {
            item.payload.get("tool_call_id"): item
            for item in items.values()
            if item.kind == "tool_call" and item.status == "completed" and item.producer == "model"
        }
        trusted = []
        for operation in self._store.list_tool_operations(turn_id):
            if (
                operation.status not in {"succeeded", "failed", "denied", "cancelled", "unknown"}
                or operation.result_item_id is None
            ):
                continue
            item = items.get(operation.result_item_id)
            call = calls.get(operation.tool_call_id)
            if (
                item is None
                or call is None
                or item.kind != "tool_result"
                or item.status != "completed"
                or item.producer != "tool"
                or call.payload.get("tool_name") != operation.tool_name
                or item.payload.get("tool_name") != operation.tool_name
                or item.payload.get("tool_call_id") != operation.tool_call_id
                or item.sequence <= call.sequence
                or not isinstance(call.payload.get("arguments"), Mapping)
            ):
                continue
            trusted.append((item, operation, call.payload["arguments"]))
        # A live owner can close while a Turn is paused. Its confirmed outcome
        # remains available after process handles and the Session are gone.
        launches = {
            operation.operation_id: (item, operation, arguments)
            for item, operation, arguments in trusted
            if operation.tool_name == "run_command"
        }
        for item in items.values():
            if item.kind != "tool_continuation" or item.producer != "runtime" or item.status != "completed":
                continue
            operation_id = item.payload.get("operation_id")
            source = launches.get(operation_id) if isinstance(operation_id, str) else None
            if source is None:
                continue
            launch, operation, arguments = source
            initial = launch.payload.get("structured_content")
            output = item.payload.get("structured_content")
            if (
                item.payload.get("source_result_item_id") == launch.item_id
                and isinstance(initial, Mapping)
                and isinstance(output, Mapping)
                and item.payload.get("key") == initial.get("session_id") == output.get("session_id")
                and output.get("status") in {"completed", "cancelled"}
            ):
                trusted.append((item, operation, arguments))
        return tuple(trusted)


def _is_workspace_change(item: ItemSnapshot, operation: ToolOperationSnapshot) -> bool:
    metadata = item.payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    has_write_claim = any(
        resource.get("kind") == "filesystem" and resource.get("access") == "write" for resource in operation.resources
    )
    return (
        has_write_claim
        and metadata.get("runtime_workspace_write") is True
        and metadata.get("workspace_tree_changed") is True
    )


class TurnStop:
    """Commit a final response once, after optional conditions allow stopping."""

    def __init__(
        self,
        *,
        store: RuntimeStore,
        gate: CompletionGate,
        commit: Callable[[Callable[[], Any]], Awaitable[Any]],
        before_completion: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._gate = gate
        self._commit = commit
        self._before_completion = before_completion

    async def finish(
        self,
        *,
        thread_id: str,
        turn_id: str,
        answer: str,
        input_revision: int,
        source_item_id: str | None = None,
    ) -> TurnResult | None:
        def finish_atomically() -> TurnResult | None:
            # Evidence observation, decision, feedback/terminal transition share
            # one transaction. No accepted decision can be stranded before
            # completion while another owner appends a workspace mutation.
            with self._store.tx.transaction():
                return self._finish(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    answer=answer,
                    input_revision=input_revision,
                    source_item_id=source_item_id,
                )

        def pause_if_current(reason: str) -> TurnResult | None:
            with self._store.tx.transaction():
                if self._store.user_input_revision(turn_id) != input_revision:
                    return None
                turn_ops.pause_turn(
                    self._store, turn_id=turn_id, reason=reason,
                    reason_code="attached_children", input_revision=input_revision,
                )
                return TurnResult(thread_id=thread_id, turn_id=turn_id, answer=None, status="paused")

        while True:
            if self._before_completion is not None:
                if await self._store.run_async(self._store.user_input_revision, turn_id) != input_revision:
                    return None
                waiting = asyncio.ensure_future(self._before_completion(turn_id))
                try:
                    while not waiting.done():
                        await asyncio.wait({waiting}, timeout=0.05)
                        if await self._store.run_async(self._store.user_input_revision, turn_id) != input_revision:
                            return None
                    waiting.result()
                except TurnChildrenPendingError as exc:
                    return cast(TurnResult | None, await self._commit(partial(pause_if_current, str(exc))))
                finally:
                    if not waiting.done():
                        # Cancelling this wait leaves the actual child drivers
                        # alive. The parent must consume steering before deciding
                        # whether any child should be stopped.
                        waiting.cancel()
                    await asyncio.gather(waiting, return_exceptions=True)
            try:
                return cast(TurnResult | None, await self._commit(finish_atomically))
            except TurnChildrenPendingError as exc:
                if self._before_completion is None:
                    return cast(TurnResult | None, await self._commit(partial(pause_if_current, str(exc))))
                # A child was admitted after the drain but before the terminal
                # transaction. Drain it and retry this same answer; do not ask
                # the model to poll or regenerate an already received answer.

    def _finish(
        self,
        *,
        thread_id: str,
        turn_id: str,
        answer: str,
        input_revision: int,
        source_item_id: str | None = None,
    ) -> TurnResult | None:
        if self._store.user_input_revision(turn_id) != input_revision:
            return None
        items = self._store.list_items(turn_id)
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
            proposal_item = completion_ops.record_final_proposal(
                self._store,
                turn_id=turn_id,
                answer=answer,
                source_item_id=source_item_id,
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
                for i in reversed(items)
                if i.kind == "completion_decision" and i.payload.get("proposal_item_id") == proposal.item_id
            ),
            None,
        )
        if decision_item is None or decision_item.payload.get("action") == "accept":
            # Pre-upgrade records may contain an accept without a completed Turn.
            # Recheck current evidence: a child or a command continuation could
            # have changed the workspace while no parent owner was running.
            from agent_runtime.harness.child_workspace_ops import unresolved

            pending_artifacts = unresolved(self._store, turn_id, settled_only=True)
            decision = (
                CompletionDecision(
                    action="continue", reason_code="child_integration_required",
                    reason=json.dumps({"child_turn_ids": pending_artifacts,
                        "next_action": (
                            "Inspect and integrate each child artifact, run parent verification, or explicitly discard."
                        )}),
                ) if pending_artifacts else self._gate.evaluate(proposal)
            )
            if decision.action == "continue" and (self._completion_continuation_limit_reached(turn_id)):
                decision = replace(
                    decision,
                    action="fail",
                    reason_code="completion_continuation_limit",
                    reason=(
                        "Completion checks exhausted their consecutive continuation allowance without a tool "
                        "exchange or new user input. The last check did not accept the result: " + decision.reason
                    ),
                )
            decision_item = completion_ops.record_completion_decision(
                self._store,
                turn_id=turn_id,
                proposal_item_id=proposal.item_id,
                action=decision.action,
                reason=decision.reason,
                reason_code=decision.reason_code,
            )
        else:
            decision = CompletionDecision(
                **{
                    key: value
                    for key, value in decision_item.payload.items()
                    if key in {"action", "reason", "reason_code"}
                }
            )
        if self._store.user_input_revision(turn_id) != input_revision:
            return None
        if decision.action == "continue":
            if not any(
                i.kind == "completion_feedback" and i.payload.get("source_item_id") == decision_item.item_id
                for i in items
            ):
                completion_ops.record_completion_feedback(
                    self._store,
                    turn_id=turn_id,
                    reason=decision.reason,
                    source_item_id=decision_item.item_id,
                )
            return None
        if decision.action == "pause":
            interaction = turn_ops.request_clarification(
                self._store,
                turn_id=turn_id,
                question=decision.reason,
            )
            return TurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
                answer=None,
                status="paused",
                interaction_id=interaction.request_id,
            )
        if decision.action == "fail":
            turn_ops.fail_turn(
                self._store,
                turn_id=turn_id,
                reason_code=decision.reason_code or "completion_rejected",
                message=decision.reason,
            )
            return TurnResult(
                thread_id=thread_id,
                turn_id=turn_id,
                answer=None,
                status="failed",
            )
        try:
            policy = self._store.read_turn(turn_id).binding_manifest.get("delivery_policy", {})
            if policy.get("revision") == "delivery-v1":
                from agent_runtime.harness import delivery

                if not any(i.kind == "delivery_record" for i in self._store.list_items(turn_id)):
                    delivery.record(self._store, turn_id)
            completed = turn_ops.complete_turn(
                self._store,
                turn_id=turn_id,
                answer=answer,
                expected_input_revision=input_revision,
                source_item_id=source_item_id,
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

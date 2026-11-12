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

from agent_runtime.core.model_request import canonical_hash, freeze_json_mapping
from agent_runtime.harness import completion_ops, turn_ops
from agent_runtime.harness.command_evidence import (
    command_succeeded,
    trusted_command_results,
    trusted_tool_results,
)
from agent_runtime.harness.lifecycle import LifecycleHooks
from agent_runtime.harness.operation_values import (
    TurnCancellationPendingError,
    TurnChildrenPendingError,
    TurnInputChangedError,
)
from agent_runtime.harness.protocol import (
    CompletionDecision,
    CompletionGate,
    CompletionProposal,
    TurnResult,
    restore_model_response,
)
from agent_runtime.harness.records import ItemSnapshot, ToolOperationSnapshot
from agent_runtime.harness.runtime_store import RuntimeStore
from agent_runtime.tools.evidence import workspace_changed

_BOUNDED_STOP_FEEDBACK = frozenset({"stop_hook_blocked", "stop_evidence_changed", "delivery_evidence_stale"})

class StopChecks:
    def __init__(self, store: RuntimeStore) -> None:
        self._store = store

    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        return self.assess(proposal)[0]

    def assess(self, proposal: CompletionProposal) -> tuple[CompletionDecision, dict[str, Any]]:
        policy = self._store.read_turn(proposal.turn_id).binding_manifest.get("completion_policy", {})
        evidence: dict[str, Any] = {}
        if not isinstance(policy, Mapping):
            return (
                CompletionDecision(
                    "fail", "The frozen stop policy is invalid.", reason_code="invalid_completion_policy",
                ),
                evidence,
            )
        return self._evaluate(proposal, policy, evidence), evidence

    def _evaluate(
        self, proposal: CompletionProposal, policy: Mapping[str, object], evidence: dict[str, Any],
    ) -> CompletionDecision:
        """Keep lifecycle, recorded mutations and user-selected acceptance distinct."""
        require_change = policy.get("require_workspace_change", False)
        raw_command = policy.get("acceptance_command")
        command = raw_command.strip() if isinstance(raw_command, str) else None
        evidence.update(
            require_workspace_change=require_change,
            command=command,
            command_status="not_evaluated" if command else "not_configured",
        )

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

        latest_change, result_order = self._workspace_change_order(proposal.turn_id)
        evidence["workspace_change_item_id"] = next(
            (item_id for item_id, order in result_order.items() if order == latest_change), None,
        )
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
        for item, operation, arguments in trusted_command_results(self._store, proposal.turn_id):
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
            evidence["command_status"] = "missing"
            return reject(
                "acceptance_command_missing",
                "The configured acceptance command has no trusted execution in this workspace.",
                next_action="Run command exactly with run_command from working_dir and inspect its result.",
            )
        item, operation = max(candidates, key=lambda pair: result_order[pair[0].item_id])
        evidence.update(
            result_item_id=item.item_id, operation_id=operation.operation_id, tool_call_id=operation.tool_call_id,
        )
        result = item.payload.get("structured_content")
        if isinstance(result, Mapping) and result.get("status") == "running":
            evidence["command_status"] = "pending"
            return reject(
                "acceptance_command_pending",
                "Observe the configured acceptance command with manage_command.",
                tool_call_id=operation.tool_call_id,
                session_id=result.get("session_id"),
            )
        if not isinstance(result, Mapping) or not command_succeeded(
            operation_status=operation.status,
            is_error=item.payload.get("is_error"),
            output=result,
        ):
            evidence["command_status"] = "failed"
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
            evidence["command_status"] = "stale"
            return reject(
                "acceptance_command_stale",
                "The workspace changed after the configured acceptance command succeeded.",
                tool_call_id=operation.tool_call_id,
                next_action="Rerun command exactly with run_command from working_dir against the updated workspace.",
            )
        evidence["command_status"] = "passed"
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
                for item, operation, _arguments in trusted_tool_results(self._store, current)
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
        and workspace_changed(metadata) is True
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
        lifecycle_hooks: LifecycleHooks | None = None,
        dispatch_guard: Callable[[RuntimeStore], None] | None = None,
    ) -> None:
        self._store = store
        self._gate = gate
        self._commit = commit
        self._before_completion = before_completion
        self._lifecycle_hooks = lifecycle_hooks
        self._dispatch_guard = dispatch_guard

    async def finish(
        self,
        *,
        thread_id: str,
        turn_id: str,
        answer: str | None,
        input_revision: int,
        source_item_id: str | None = None,
    ) -> TurnResult | None:
        proposal: ItemSnapshot | None = None
        hook_block: str | None = None
        hook_evidence_revision: str | None = None

        def prepare_proposal() -> tuple[ItemSnapshot, Mapping[str, Any] | None, bool] | None:
            with self._store.tx.transaction():
                item = self._proposal(
                    turn_id, answer, input_revision, source_item_id,
                    proposal_id=None if proposal is None else proposal.item_id,
                )
                if item is None:
                    return None
                items = self._store.list_items(turn_id)
                prior = self._prior_decision(items, item.item_id)
                if prior is not None and prior.payload["action"] != "accept":
                    return item, None, False
                binding = self._store.read_turn(turn_id).binding_manifest
                if binding.get("lifecycle_hook_protocol") != "completion-evidence-v1":
                    return item, None, True
                _, evidence = self._review(item, items, include_hook_evidence=True)
                return item, evidence, True

        def finish_atomically() -> TurnResult | None:
            # Evidence observation, decision, feedback/terminal transition share
            # one transaction. No accepted decision can be stranded before
            # completion while another owner appends a workspace mutation.
            try:
                with self._store.tx.transaction():
                    if proposal is not None and self._dispatch_guard is not None:
                        self._dispatch_guard(self._store)
                    return self._finish(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        answer=answer,
                        input_revision=input_revision,
                        source_item_id=source_item_id,
                        hook_proposal_id=None if proposal is None else proposal.item_id,
                        hook_block=hook_block,
                        hook_evidence_revision=hook_evidence_revision,
                    )
            except TurnCancellationPendingError:
                turn_ops.request_turn_cancellation(
                    self._store, turn_id=turn_id, reason="Execution authority revoked before completion",
                )
                raise

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
            if self._lifecycle_hooks is not None and bool(await self._store.run_async(
                self._lifecycle_hooks.configured, turn_id, "Stop"
            )):
                prepared = await self._commit(prepare_proposal)
                if prepared is None:
                    return None
                proposal, evidence, invoke_hook = prepared
                hook_block = None
                hook_evidence_revision = None if evidence is None else canonical_hash(evidence)
                if invoke_hook:
                    hook_block = await self._lifecycle_hooks.run(
                        self._commit, "Stop", turn_id=turn_id, target_id=proposal.item_id,
                        source_item_id=proposal.item_id, answer=proposal.payload["text"], input_revision=input_revision,
                        completion=evidence, completion_revision=hook_evidence_revision,
                    )
            try:
                return cast(TurnResult | None, await self._commit(finish_atomically))
            except TurnChildrenPendingError as exc:
                if self._before_completion is None:
                    return cast(TurnResult | None, await self._commit(partial(pause_if_current, str(exc))))
                # A child was admitted after the drain but before the terminal
                # transaction. Drain it and retry this same answer; do not ask
                # the model to poll or regenerate an already received answer.

    def _proposal(
        self, turn_id: str, answer: str | None, input_revision: int, source_item_id: str | None,
        *, proposal_id: str | None = None,
    ) -> ItemSnapshot | None:
        turn_ops._assert_no_pending_cancellation(self._store, turn_id)
        if self._store.user_input_revision(turn_id) != input_revision:
            return None
        delivery_id = None
        if answer is None:
            from agent_runtime.harness import delivery

            submission = delivery.submitted_answer(self._store, turn_id, source_item_id) if source_item_id else None
            if submission is None:
                return None
            answer, delivery_id = submission
        for item in self._store.list_items(turn_id):
            if (
                item.kind == "final_proposal"
                and item.payload.get("source_item_id") == source_item_id
                and item.payload.get("delivery_id") == delivery_id
                and (source_item_id is not None or item.item_id == proposal_id)
            ):
                return item
        return completion_ops.record_final_proposal(
            self._store, turn_id=turn_id, answer=answer, source_item_id=source_item_id, delivery_id=delivery_id,
        )

    @staticmethod
    def _prior_decision(items: tuple[ItemSnapshot, ...], proposal_id: str) -> ItemSnapshot | None:
        return next(
            (
                item for item in reversed(items)
                if item.kind == "completion_decision" and item.payload.get("proposal_item_id") == proposal_id
            ),
            None,
        )

    def _review(
        self, item: ItemSnapshot, items: tuple[ItemSnapshot, ...], *, include_hook_evidence: bool,
    ) -> tuple[CompletionDecision, Mapping[str, Any]]:
        from agent_runtime.harness.child_workspace_ops import unresolved

        turn_id = item.turn_id
        proposal = CompletionProposal(item.thread_id, turn_id, item.item_id, item.payload["text"])
        pending_artifacts = unresolved(self._store, turn_id, settled_only=True)
        acceptance = None
        if pending_artifacts:
            decision = CompletionDecision(
                action="continue", reason_code="child_integration_required",
                reason=json.dumps({
                    "child_turn_ids": pending_artifacts,
                    "next_action": (
                        "Inspect and integrate each child artifact, run parent verification, or explicitly discard."
                    ),
                }),
            )
        elif include_hook_evidence and isinstance(self._gate, StopChecks):
            decision, acceptance = self._gate.assess(proposal)
        else:
            decision = self._gate.evaluate(proposal)
        delivery_id = item.payload.get("delivery_id")
        report = None
        if (delivery_id is not None and decision.action == "accept") or (
            include_hook_evidence and any(i.kind == "delivery_record" for i in items)
        ):
            from agent_runtime.harness import delivery

            report = delivery.inspect(self._store, turn_id, include_live_evidence=include_hook_evidence)
            if delivery_id is not None and decision.action == "accept" and report.get("freshness") == "stale":
                decision = CompletionDecision(
                    "continue",
                    "Delivery evidence changed before completion. Record current evidence and resubmit.",
                    reason_code="delivery_evidence_stale",
                )
        if not include_hook_evidence:
            return decision, {}
        stop_hook_continuations = sum(
            i.kind == "completion_decision" and i.payload.get("reason_code") == "stop_hook_blocked" for i in items
        )
        evidence = freeze_json_mapping({
            "input_revision": self._store.user_input_revision(turn_id),
            "decision": {"action": decision.action, "reason_code": decision.reason_code, "reason": decision.reason},
            "acceptance": acceptance,
            "pending_child_turn_ids": pending_artifacts,
            "stop_hook_active": stop_hook_continuations > 0,
            "stop_hook_continuations": stop_hook_continuations,
            "delivery": None if report is None else {
                key: report[key] for key in (
                    "delivery_id", "status", "freshness", "source_item_ids",
                    "live_workspace", "receipt_evidence", "verified_scope",
                ) if key in report
            },
        })
        return decision, evidence

    def _finish(
        self,
        *,
        thread_id: str,
        turn_id: str,
        answer: str | None,
        input_revision: int,
        source_item_id: str | None = None,
        hook_proposal_id: str | None = None,
        hook_block: str | None = None,
        hook_evidence_revision: str | None = None,
    ) -> TurnResult | None:
        proposal_item = self._proposal(
            turn_id, answer, input_revision, source_item_id, proposal_id=hook_proposal_id,
        )
        if proposal_item is None or (hook_proposal_id is not None and proposal_item.item_id != hook_proposal_id):
            return None
        answer = proposal_item.payload["text"]
        delivery_id = proposal_item.payload.get("delivery_id")
        items = self._store.list_items(turn_id)
        decision_item = self._prior_decision(items, proposal_item.item_id)
        if decision_item is None or decision_item.payload.get("action") == "accept":
            # Evaluate live facts again in the terminal transaction.
            decision, evidence = self._review(
                proposal_item, items, include_hook_evidence=hook_evidence_revision is not None,
            )
            applied_hook_revision = None
            if decision.action == "accept":
                if hook_evidence_revision is not None and canonical_hash(evidence) != hook_evidence_revision:
                    decision = CompletionDecision(
                        "continue",
                        "Completion evidence changed during Stop. Check the current facts and propose again.",
                        reason_code="stop_evidence_changed",
                    )
                else:
                    applied_hook_revision = hook_evidence_revision
                    if hook_block is not None:
                        decision = CompletionDecision("continue", hook_block, reason_code="stop_hook_blocked")
            bounded_across_batches = decision.reason_code in _BOUNDED_STOP_FEEDBACK
            if decision.action == "continue" and self._completion_continuation_limit_reached(
                turn_id, across_batches=bounded_across_batches
            ):
                decision = replace(
                    decision,
                    action="fail",
                    reason_code=(
                        "stop_hook_continuation_limit" if bounded_across_batches else "completion_continuation_limit"
                    ),
                    reason="Completion checks exhausted their continuation allowance. " + decision.reason,
                )
            decision_item = completion_ops.record_completion_decision(
                self._store,
                turn_id=turn_id,
                proposal_item_id=proposal_item.item_id,
                action=decision.action,
                reason=decision.reason,
                reason_code=decision.reason_code,
                hook_evidence_revision=applied_hook_revision,
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
            from agent_runtime.harness import delivery

            binding = self._store.read_turn(turn_id).binding_manifest
            if delivery.should_record_on_stop(binding, items):
                delivery.record(self._store, turn_id)
            completed = turn_ops.complete_turn(
                self._store,
                turn_id=turn_id,
                answer=answer,
                expected_input_revision=input_revision,
                source_item_id=source_item_id if delivery_id is None else None,
            )
        except TurnInputChangedError:
            return None
        return TurnResult(
            thread_id=thread_id,
            turn_id=completed.turn_id,
            answer=answer,
        )

    def _completion_continuation_limit_reached(self, turn_id: str, *, across_batches: bool) -> bool:
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
            if across_batches:
                if item.kind == "completion_decision" and item.payload.get("reason_code") in _BOUNDED_STOP_FEEDBACK:
                    continuations += 1
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

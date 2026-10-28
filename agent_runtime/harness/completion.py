"""Evidence-based completion policy for delivery Turns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from agent_runtime.harness.protocol import CompletionDecision, CompletionProposal
from agent_runtime.harness.records import ItemSnapshot, ToolOperationSnapshot
from agent_runtime.harness.state import RuntimeState
from agent_runtime.harness.verification import assess_tool_verification


class DeliveryCompletionGate:
    """Validate required workspace-change evidence, not answer correctness."""

    def __init__(self, store: RuntimeState) -> None:
        self._store = store

    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        turn = self._store.read_turn(proposal.turn_id)
        policy = turn.binding_manifest.get("completion_policy", {})
        if isinstance(policy, Mapping) and policy.get("revision") == "completion-v2":
            return self._evaluate_v2(proposal, policy)
        if not isinstance(policy, Mapping) or not policy.get("require_workspace_change", False):
            return CompletionDecision(
                action="accept",
                reason=(
                    "Workspace-change validation is not required for this Turn; task acceptance has not been evaluated."
                ),
            )

        # A failed multi-file operation may still have attested partial writes.
        trusted = self._trusted_results(proposal.turn_id, include_failed=True)
        changes = [(item, operation) for item, operation in trusted if _is_workspace_change(item, operation)]
        if not changes:
            return CompletionDecision(
                action="continue",
                reason="A verified workspace change is still required.",
                validation_status="failed",
                validation_scope="workspace_change",
            )

        latest_change_sequence = max(item.sequence for item, _operation in changes)
        latest_change_resources = {
            str(resource.get("identity"))
            for item, operation in changes
            if item.sequence == latest_change_sequence
            for resource in operation.resources
            if resource.get("kind") == "filesystem"
            and resource.get("access") == "write"
            and isinstance(resource.get("identity"), str)
        }
        verified_after_change = any(
            item.sequence > latest_change_sequence
            and _verification_matches_change(
                item,
                changed_resources=latest_change_resources,
            )
            for item in self._trusted_verifications(proposal.turn_id)
        )
        if not verified_after_change:
            return CompletionDecision(
                action="continue",
                reason=self._verification_feedback(
                    proposal.turn_id,
                    latest_change_sequence,
                    latest_change_resources,
                ),
                validation_status="failed",
                validation_scope="workspace_change",
            )
        return CompletionDecision(
            action="accept",
            reason="The latest workspace change has trusted post-change verification.",
            validation_status="passed",
            validation_scope="workspace_change",
        )

    def _evaluate_v2(self, proposal: CompletionProposal, policy: Mapping[str, object]) -> CompletionDecision:
        """Keep lifecycle, recorded mutations and user-selected acceptance distinct."""
        require_change = policy.get("require_workspace_change", False)
        raw_command = policy.get("acceptance_command")
        command = raw_command.strip() if isinstance(raw_command, str) else None

        def reject(code: str, message: str, scope: str, **details: object) -> CompletionDecision:
            feedback = {"reason_code": code, "message": message, **details}
            if command:
                feedback["command"] = command
                feedback["working_dir"] = "."
            return CompletionDecision(
                action="continue", reason=json.dumps(feedback, ensure_ascii=False),
                validation_status="failed", validation_scope=scope,
            )

        if not isinstance(require_change, bool) or (raw_command is not None and not command):
            return reject("invalid_completion_policy", "The frozen completion policy is invalid.", "completion_policy")
        records = self._trusted_v2_results(proposal.turn_id)
        launches: dict[str, tuple[ItemSnapshot, Mapping[str, object]]] = {}
        pending: set[str] = set()
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
                    if output.get("status") == "running":
                        pending.add(session_id)
                    else:
                        pending.discard(session_id)
            elif (operation.tool_name == "manage_command" and isinstance(session_id, str)
                  and arguments.get("session_id") == session_id and session_id in launches):
                launch, original_arguments = launches[session_id]
                if item.sequence > launch.sequence:
                    command_results.append((item, operation, original_arguments))
                    if output.get("status") in {"completed", "cancelled"}:
                        pending.discard(session_id)
        if pending:
            return reject(
                "command_session_pending", "Observe the terminal command result with manage_command before finishing.",
                "command_lifecycle", session_ids=sorted(pending),
            )
        if not require_change and command is None:
            return CompletionDecision(
                action="accept", reason="The model turn ended; task acceptance was not evaluated.",
            )

        latest_change, result_order = self._v2_workspace_change_order(proposal.turn_id)
        if require_change and not latest_change:
            return reject(
                "workspace_change_missing", "The requested workspace mutation has not been recorded.",
                "workspace_change_recorded", next_action="Make the requested workspace change before finishing.",
            )
        if command is None:
            return CompletionDecision(
                action="accept",
                reason="The requested workspace mutation was recorded; task acceptance was not evaluated.",
                validation_status="passed", validation_scope="workspace_change_recorded",
            )

        turn = self._store.read_turn(proposal.turn_id)
        workspace = Path(self._store.read_thread(turn.thread_id).workspace).resolve()
        candidates = []
        for item, operation, arguments in command_results:
            actual_command = arguments.get("command")
            if (
                not isinstance(actual_command, str)
                or actual_command.strip() != command or item.item_id not in result_order
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
                "acceptance_command",
                next_action="Run command exactly with run_command from working_dir and inspect its result.",
            )
        item, operation = max(candidates, key=lambda pair: result_order[pair[0].item_id])
        result = item.payload.get("structured_content")
        exit_code = result.get("exit_code") if isinstance(result, Mapping) else None
        if (
            operation.status != "succeeded" or item.payload.get("is_error") is not False
            or not isinstance(result, Mapping) or not isinstance(exit_code, int) or isinstance(exit_code, bool)
            or exit_code != 0 or result.get("timed_out") or result.get("sandbox_error")
        ):
            return reject(
                "acceptance_command_failed", "The latest configured acceptance command did not complete successfully.",
                "acceptance_command", tool_call_id=operation.tool_call_id,
                next_action="Inspect that execution's output, resolve the failure, and rerun command from working_dir.",
            )
        # A user-selected verifier may produce build outputs itself. Its own
        # attested mutation is part of that check; a subsequent operation's
        # mutation still requires a fresh execution of the frozen command.
        if result_order[item.item_id] < latest_change:
            return reject(
                "acceptance_command_stale", "The workspace changed after the configured acceptance command succeeded.",
                "acceptance_command", tool_call_id=operation.tool_call_id,
                next_action="Rerun command exactly with run_command from working_dir against the updated workspace.",
            )
        return CompletionDecision(
            action="accept",
            reason="The configured acceptance command completed successfully after the latest workspace mutation.",
            validation_status="passed", validation_scope="acceptance_command",
        )

    def _v2_workspace_change_order(self, turn_id: str) -> tuple[int, dict[str, int]]:
        """Order trusted shared-workspace mutations across durable descendants.

        Item sequences belong to individual Turns. Rollout record IDs supply
        the common commit order for parent checks and delegated mutations.
        Allocation records, rather than a claimed binding field, establish
        ancestry; unrelated work in the same workspace remains outside it.
        """
        turns = {turn.turn_id: turn for turn in self._store.list_turns()}
        workspaces = {
            thread.thread_id: Path(thread.workspace).resolve() for thread in self._store.list_threads()
        }
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
            if record.turn_id in family and record.record_type == "item_completed"
            and record.producer in {"tool", "runtime"}
            and isinstance(record.payload.get("item_id"), str)
        }
        latest_change = max(
            (result_order[item.item_id]
             for current in family for item, operation, _arguments in self._trusted_v2_results(current)
             if item.item_id in result_order and _is_workspace_change(item, self._write_origin(item, operation))),
            default=0,
        )
        return latest_change, result_order

    def _write_origin(self, item: ItemSnapshot, operation: ToolOperationSnapshot) -> ToolOperationSnapshot:
        metadata = item.payload.get("metadata")
        origin_id = metadata.get("runtime_workspace_origin_operation_id") if isinstance(metadata, Mapping) else None
        if isinstance(origin_id, str):
            origin = next((candidate for candidate in self._store.list_tool_operations(operation.turn_id)
                           if candidate.operation_id == origin_id), None)
            if origin is not None and origin.result_item_id is not None:
                return origin
        return operation

    def _trusted_v2_results(
        self, turn_id: str,
    ) -> tuple[tuple[ItemSnapshot, ToolOperationSnapshot, Mapping[str, object]], ...]:
        items = {item.item_id: item for item in self._store.list_items(turn_id)}
        calls = {
            item.payload.get("tool_call_id"): item for item in items.values()
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
                item is None or call is None or item.kind != "tool_result" or item.status != "completed"
                or item.producer != "tool" or call.payload.get("tool_name") != operation.tool_name
                or item.payload.get("tool_name") != operation.tool_name
                or item.payload.get("tool_call_id") != operation.tool_call_id or item.sequence <= call.sequence
                or not isinstance(call.payload.get("arguments"), Mapping)
            ):
                continue
            trusted.append((item, operation, call.payload["arguments"]))
        # A live owner can close while a Turn is paused. Its confirmed outcome
        # remains available after process handles and the Session are gone.
        launches = {operation.operation_id: (item, operation, arguments)
                    for item, operation, arguments in trusted if operation.tool_name == "run_command"}
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
            if (item.payload.get("source_result_item_id") == launch.item_id
                    and isinstance(initial, Mapping) and isinstance(output, Mapping)
                    and item.payload.get("key") == initial.get("session_id") == output.get("session_id")
                    and output.get("status") in {"completed", "cancelled"}):
                trusted.append((item, operation, arguments))
        return tuple(trusted)

    def _trusted_results(
        self,
        turn_id: str,
        *,
        include_failed: bool = False,
    ) -> tuple[tuple[ItemSnapshot, ToolOperationSnapshot], ...]:
        items = self._store.list_items(turn_id)
        items_by_id = {item.item_id: item for item in items}
        calls_by_id = {
            str(item.payload.get("tool_call_id")): item
            for item in items
            if item.kind == "tool_call"
            and item.status == "completed"
            and item.producer == "model"
            and isinstance(item.payload.get("tool_call_id"), str)
        }
        trusted: list[tuple[ItemSnapshot, ToolOperationSnapshot]] = []
        statuses = {"succeeded", "failed"} if include_failed else {"succeeded"}
        for operation in self._store.list_tool_operations(turn_id):
            if operation.status not in statuses or operation.result_item_id is None:
                continue
            result = items_by_id.get(operation.result_item_id)
            call = calls_by_id.get(operation.tool_call_id)
            if (
                result is None
                or result.kind != "tool_result"
                or result.status != "completed"
                or result.producer != "tool"
                or call is None
                or call.payload.get("tool_name") != operation.tool_name
            ):
                continue
            trusted.append((result, operation))
        return tuple(trusted)

    def _verification_feedback(
        self,
        turn_id: str,
        latest_change_sequence: int,
        changed_resources: set[str],
    ) -> str:
        calls = {
            item.payload.get("tool_call_id"): item
            for item in self._store.list_items(turn_id)
            if item.kind == "tool_call"
        }
        candidates = sorted(
            (
                (item, operation)
                for item, operation in self._trusted_results(turn_id, include_failed=True)
                if item.sequence > latest_change_sequence
                and operation.tool_name in {"run_command", "execute_python", "read_file", "inspect_data_file"}
            ),
            key=lambda pair: pair[0].sequence,
            reverse=True,
        )
        feedback: dict[str, object] = {
            "validation": "post_change_verification",
            "reason_code": "verification_missing",
            "message": "No successful verification of the latest workspace change has been recorded.",
            "next_action": "Run the project's documented checks directly after the latest workspace change, "
            "using a recognized test runner or static check, or inspect the changed artifacts with "
            "read_file or inspect_data_file.",
        }
        if candidates:
            item, operation = candidates[0]
            call = calls[operation.tool_call_id]
            arguments = call.payload.get("arguments", {})
            assessment = assess_tool_verification(
                tool_name=operation.tool_name,
                arguments=arguments,
                structured_content=item.payload.get("structured_content"),
                resources=operation.resources,
            )
            code, message = assessment.reason_code, assessment.message
            if operation.status == "failed" and assessment.evidence is not None:
                code, message = (
                    "tool_failed",
                    "The tool operation failed; its output cannot attest successful verification.",
                )
            elif assessment.evidence is not None and assessment.evidence.kind == "assertion":
                code, message = (
                    "assertion_targets_unattested",
                    "The managed interpreter observed successful assertions, but it did not attest which files "
                    "they verified. Assertion execution alone does not satisfy workspace-change verification.",
                )
                feedback["attested_write_targets"] = [target[:500] for target in sorted(changed_resources)[:8]]
                feedback["next_action"] = (
                    "Run the project's documented checks directly with a recognized test runner or static check "
                    "in run_command, or use read_file or inspect_data_file on attested_write_targets after the "
                    "latest write. File inspection attests inspection, not task correctness."
                )
            elif assessment.evidence is not None:
                code, message = (
                    "unrelated_inspection" if assessment.evidence.kind == "inspection" else "unrelated_verification",
                    "The observed verification targets could not be matched to the latest attested write targets. "
                    "Run the project's checks or obtain inspection evidence matching those targets.",
                )
                feedback["attested_write_targets"] = [target[:500] for target in sorted(changed_resources)[:8]]
                feedback["inspected_targets"] = [target[:500] for target in assessment.evidence.verified_resources[:8]]
            feedback.update(
                reason_code=code,
                message=message,
                tool_call_id=operation.tool_call_id,
                tool_name=operation.tool_name,
            )
            if assessment.suggested_command and len(assessment.suggested_command) <= 2000:
                feedback["suggested_command"] = assessment.suggested_command
                feedback["next_action"] = (
                    "Run suggested_command directly in the same working directory in a separate run_command call. "
                    "Then inspect the exit status and output."
                )
            command = arguments.get("command")
            if isinstance(command, str):
                feedback["command"] = command[:2000]
            structured = item.payload.get("structured_content")
            if isinstance(structured, Mapping):
                exit_code = structured.get("exit_code")
                if isinstance(exit_code, int) and not isinstance(exit_code, bool):
                    feedback["exit_code"] = exit_code
            if item.payload.get("error_code"):
                feedback["error_code"] = str(item.payload["error_code"])[:200]
        elif any(item.sequence <= latest_change_sequence for item in self._trusted_verifications(turn_id)):
            feedback.update(
                reason_code="verification_stale",
                message="The recorded verification happened before the latest workspace change. "
                "Rerun verification against the updated files.",
            )
        return json.dumps(feedback, ensure_ascii=False)

    def _trusted_verifications(self, turn_id: str) -> tuple[ItemSnapshot, ...]:
        operations = {operation.operation_id: operation for operation in self._store.list_tool_operations(turn_id)}
        items = {item.item_id: item for item in self._store.list_items(turn_id)}
        calls = {item.payload.get("tool_call_id"): item for item in items.values() if item.kind == "tool_call"}
        trusted: list[ItemSnapshot] = []
        for item in items.values():
            if item.kind != "verification" or item.status != "completed" or item.producer != "runtime":
                continue
            operation_id = item.payload.get("operation_id")
            operation = operations.get(operation_id) if isinstance(operation_id, str) else None
            if (
                operation is None
                or operation.status != "succeeded"
                or operation.result_item_id is None
                or item.payload.get("source_result_item_id") != operation.result_item_id
                or item.payload.get("arguments_digest") != operation.arguments_digest
            ):
                continue
            source = items.get(operation.result_item_id)
            call = calls.get(operation.tool_call_id)
            if (
                source is None
                or call is None
                or source.kind != "tool_result"
                or source.status != "completed"
                or source.producer != "tool"
                or call.kind != "tool_call"
                or call.status != "completed"
                or call.producer != "model"
                or call.payload.get("tool_name") != operation.tool_name
                or source.payload.get("tool_call_id") != operation.tool_call_id
                or source.payload.get("tool_name") != operation.tool_name
                or source.payload.get("is_error") is True
                or not isinstance(call.payload.get("arguments"), Mapping)
            ):
                continue
            evidence = assess_tool_verification(
                tool_name=operation.tool_name,
                arguments=call.payload.get("arguments", {}),
                structured_content=source.payload.get("structured_content"),
                resources=operation.resources,
            ).evidence
            # Old logs stay immutable; today's gate rechecks the bound source
            # instead of inheriting historical classification mistakes.
            if (
                evidence is None
                or item.payload.get("verification_kind") != evidence.kind
                or item.payload.get("verifier") != evidence.verifier
                or tuple(item.payload.get("verified_resources", ())) != evidence.verified_resources
            ):
                continue
            trusted.append(item)
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


def _verification_matches_change(
    item: ItemSnapshot,
    *,
    changed_resources: set[str],
) -> bool:
    kind = item.payload.get("verification_kind")
    if kind in {"test", "static_analysis"}:
        return True
    if kind != "inspection":
        return False
    resources = item.payload.get("verified_resources")
    if not isinstance(resources, (list, tuple)):
        return False
    return bool(changed_resources & {str(resource) for resource in resources})

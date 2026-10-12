"""Evidence-based completion policy for delivery Turns."""

from __future__ import annotations

import json
from collections.abc import Mapping

from agent_runtime.harness.protocol import CompletionDecision, CompletionProposal
from agent_runtime.harness.rollout import ItemSnapshot, RolloutStore, ToolOperationSnapshot
from agent_runtime.harness.verification import assess_tool_verification


class DeliveryCompletionGate:
    """Validate required workspace-change evidence, not answer correctness."""

    def __init__(self, store: RolloutStore) -> None:
        self._store = store

    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        turn = self._store.read_turn(proposal.turn_id)
        policy = turn.binding_manifest.get("completion_policy", {})
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
            "using a recognized test runner, static check, or explicit assertion.",
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

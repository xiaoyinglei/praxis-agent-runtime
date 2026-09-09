"""Evidence-based completion policy for delivery Turns."""

from __future__ import annotations

from collections.abc import Mapping

from agent_runtime.harness.protocol import CompletionDecision, CompletionProposal
from agent_runtime.harness.rollout import ItemSnapshot, RolloutStore, ToolOperationSnapshot


class DeliveryCompletionGate:
    """Accept delivery only after a trusted change and later verification."""

    def __init__(self, store: RolloutStore) -> None:
        self._store = store

    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        turn = self._store.read_turn(proposal.turn_id)
        policy = turn.binding_manifest.get("completion_policy", {})
        if not isinstance(policy, Mapping) or not policy.get(
            "require_workspace_change", False
        ):
            return CompletionDecision(
                action="accept",
                reason="No workspace-change evidence is required for this Turn.",
            )

        trusted = self._trusted_results(proposal.turn_id)
        changes = [
            (item, operation)
            for item, operation in trusted
            if _is_workspace_change(item, operation)
        ]
        if not changes:
            return CompletionDecision(
                action="continue",
                reason="A verified workspace change is still required.",
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
                reason=(
                    "Post-change verification is still required. Run a recognized test "
                    "runner (for example `uv run pytest -q ...`) successfully after the "
                    "latest workspace change; running a pytest file with `python` does "
                    "not execute its tests."
                ),
            )
        return CompletionDecision(
            action="accept",
            reason="The latest workspace change has trusted post-change verification.",
        )

    def _trusted_results(
        self, turn_id: str
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
        for operation in self._store.list_tool_operations(turn_id):
            if operation.status != "succeeded" or operation.result_item_id is None:
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

    def _trusted_verifications(self, turn_id: str) -> tuple[ItemSnapshot, ...]:
        operations = {
            operation.operation_id: operation
            for operation in self._store.list_tool_operations(turn_id)
        }
        trusted: list[ItemSnapshot] = []
        for item in self._store.list_items(turn_id):
            if (
                item.kind != "verification"
                or item.status != "completed"
                or item.producer != "runtime"
            ):
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
            trusted.append(item)
        return tuple(trusted)


def _is_workspace_change(
    item: ItemSnapshot, operation: ToolOperationSnapshot
) -> bool:
    metadata = item.payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    has_write_claim = any(
        resource.get("kind") == "filesystem" and resource.get("access") == "write"
        for resource in operation.resources
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
    if kind in {"test", "static_analysis", "assertion"}:
        return True
    if kind != "inspection":
        return False
    resources = item.payload.get("verified_resources")
    if not isinstance(resources, (list, tuple)):
        return False
    return bool(changed_resources & {str(resource) for resource in resources})

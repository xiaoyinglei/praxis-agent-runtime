"""Evidence-based completion policy for delivery Turns."""

from __future__ import annotations

import ast
import shlex
from collections.abc import Mapping
from typing import Any

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
            for item, operation, _arguments in trusted
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
            and _is_successful_verification(
                item,
                operation,
                arguments,
                changed_resources=latest_change_resources,
            )
            for item, operation, arguments in trusted
        )
        if not verified_after_change:
            return CompletionDecision(
                action="continue",
                reason=(
                    "Run a recognized test runner (for example `uv run pytest -q ...`) "
                    "successfully after the latest workspace change; running a pytest "
                    "file with `python` does not execute its tests."
                ),
            )
        return CompletionDecision(
            action="accept",
            reason="The latest workspace change has trusted post-change verification.",
        )

    def _trusted_results(
        self, turn_id: str
    ) -> tuple[
        tuple[ItemSnapshot, ToolOperationSnapshot, Mapping[str, Any]], ...
    ]:
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
        trusted: list[
            tuple[ItemSnapshot, ToolOperationSnapshot, Mapping[str, Any]]
        ] = []
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
            arguments = call.payload.get("arguments")
            if not isinstance(arguments, Mapping):
                continue
            trusted.append((result, operation, arguments))
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


def _is_successful_verification(
    item: ItemSnapshot,
    operation: ToolOperationSnapshot,
    arguments: Mapping[str, Any],
    *,
    changed_resources: set[str],
) -> bool:
    if operation.tool_name in {"read_file", "inspect_data_file"}:
        structured = item.payload.get("structured_content")
        if not isinstance(structured, Mapping):
            return False
        read_resources = {
            str(resource.get("identity"))
            for resource in operation.resources
            if resource.get("kind") == "filesystem"
            and resource.get("access") == "read"
            and isinstance(resource.get("identity"), str)
        }
        if not read_resources & changed_resources:
            return False
        return (
            structured.get("valid") is True
            if operation.tool_name == "inspect_data_file"
            else True
        )
    if operation.tool_name not in {"run_command", "execute_python"}:
        return False
    structured = item.payload.get("structured_content")
    if not isinstance(structured, Mapping):
        return False
    exit_code = structured.get("exit_code")
    if isinstance(exit_code, bool) or exit_code != 0:
        return False
    source = arguments.get("command")
    if operation.tool_name == "execute_python":
        source = arguments.get("code")
    return isinstance(source, str) and _looks_like_verification(source)


def _looks_like_verification(source: str) -> bool:
    try:
        words = shlex.split(source)
    except ValueError:
        return False
    if not words:
        return False
    executable = words[0].rsplit("/", maxsplit=1)[-1]
    if executable in {"python", "python3"} and "-c" in words:
        source_index = words.index("-c") + 1
        if source_index < len(words):
            try:
                tree = ast.parse(words[source_index])
            except SyntaxError:
                pass
            else:
                if any(isinstance(statement, ast.Assert) for statement in tree.body):
                    return True
    normalized = " ".join(words).lower()
    markers = (
        "pytest",
        "unittest",
        "ruff check",
        "mypy",
        "npm test",
        "npm run test",
        "pnpm test",
        "pnpm run test",
        "yarn test",
        "cargo test",
        "go test",
    )
    return any(marker in normalized for marker in markers)

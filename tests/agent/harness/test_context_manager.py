from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    ContextBudgetExceededError,
    HarnessMessage,
    HarnessModelRequest,
    HarnessModelResponse,
    PreparedModelCall,
    RolloutContextManager,
    RolloutStore,
    Session,
)


def test_context_manager_builds_followup_from_thread_history(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        first = store.start_turn(
            thread_id=thread.thread_id,
            user_message="first question",
            binding_manifest={"model_alias": "model-v1"},
        )
        store.complete_turn(turn_id=first.turn_id, answer="first answer")
        second = store.start_turn(
            thread_id=thread.thread_id,
            user_message="follow up",
            binding_manifest={"model_alias": "model-v2"},
        )

        messages = RolloutContextManager(store).build(second.turn_id)

        assert messages == (
            HarnessMessage(role="user", content="first question"),
            HarnessMessage(role="assistant", content="first answer"),
            HarnessMessage(role="user", content="follow up"),
        )
        assert store.read_turn(second.turn_id).predecessor_turn_id == first.turn_id


def test_context_manager_rejects_an_oversized_item_before_provider_serialization(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="x" * 200,
            binding_manifest={"model_alias": "model-v1"},
        )

        with pytest.raises(ContextBudgetExceededError, match="single Item"):
            RolloutContextManager(store, max_item_bytes=100).build(turn.turn_id)


def test_context_manager_enforces_total_bytes_and_message_count(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="a" * 80,
            binding_manifest={"model_alias": "model-v1"},
        )
        store.record_migrated_context_item(
            turn_id=turn.turn_id,
            kind="context_message",
            payload={"text": "b" * 80},
        )

        with pytest.raises(ContextBudgetExceededError, match="total byte"):
            RolloutContextManager(
                store,
                max_item_bytes=1_000,
                max_total_bytes=250,
            ).build(turn.turn_id)
        with pytest.raises(ContextBudgetExceededError, match="message-count"):
            RolloutContextManager(store, max_messages=1).build(turn.turn_id)


def test_context_manager_counts_tool_arguments_toward_the_item_limit(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="inspect",
            binding_manifest={"model_alias": "model-v1"},
        )
        store.record_migrated_context_item(
            turn_id=turn.turn_id,
            kind="model_response",
            payload={
                "text": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "read_file",
                        "arguments": {"path": "x" * 500},
                    }
                ],
            },
        )

        with pytest.raises(ContextBudgetExceededError, match="single Item"):
            RolloutContextManager(store, max_item_bytes=300).build(turn.turn_id)


@pytest.mark.anyio
async def test_context_budget_failure_fails_turn_without_calling_model(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class NeverCalledModel:
        def __init__(self) -> None:
            self.prepare_calls = 0
            self.dispatch_calls = 0

        def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
            del request
            self.prepare_calls += 1
            raise AssertionError("oversized context must not reach model preparation")

        async def dispatch(
            self,
            prepared: PreparedModelCall,
        ) -> HarnessModelResponse:
            del prepared
            self.dispatch_calls += 1
            raise AssertionError("oversized context must not reach provider I/O")

    class Accept:
        def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
            del proposal
            return CompletionDecision(action="accept", reason="accepted")

    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        model = NeverCalledModel()
        result = await Session(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store, max_item_bytes=100),
            completion_gate=Accept(),
        ).run(
            turn_id="turn-context-budget",
            user_message="x" * 200,
            binding_manifest={"model_step_budget": 1},
        )

        assert result.status == "failed"
        assert model.prepare_calls == 0
        assert model.dispatch_calls == 0
        assert store.read_turn(result.turn_id).status == "failed"


def test_compaction_replaces_a_committed_prefix_without_deleting_rollout_history(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    preserved_facts = {
        "architecture_and_safety_constraints": ["never write outside workspace"],
        "file_changes": ["src/app.py changed"],
        "verification_results": ["focused test passed"],
        "unresolved_work": ["run full suite"],
        "uncertain_side_effects": [],
    }
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        first = store.start_turn(
            thread_id=thread.thread_id,
            user_message="old question",
            binding_manifest={"model_alias": "model-v1"},
        )
        store.complete_turn(turn_id=first.turn_id, answer="old answer")
        second = store.start_turn(
            thread_id=thread.thread_id,
            user_message="current question",
            binding_manifest={"model_alias": "model-v1"},
        )
        manager = RolloutContextManager(store)
        original_items = store.list_context_items(second.turn_id)
        covered_ids = tuple(item.item_id for item in original_items[:2])

        compaction = manager.compact(
            turn_id=second.turn_id,
            covered_item_ids=covered_ids,
            summary="The old exchange established the workspace rule.",
            preserved_facts=preserved_facts,
            artifact_refs=({"path": "artifacts/context.json", "sha256": "a" * 64},),
            context_version=2,
        )

        projected = manager.build(second.turn_id)
        assert len(projected) == 2
        assert projected[0].role == "context"
        compacted_payload = json.loads(projected[0].content.removeprefix("Context compaction:\n"))
        assert compacted_payload["summary"].startswith("The old exchange")
        assert compacted_payload["preserved_facts"] == preserved_facts
        assert projected[1] == HarnessMessage(role="user", content="current question")
        assert "old question" not in tuple(message.content for message in projected)
        assert "old answer" not in tuple(message.content for message in projected)

        payload = compaction.payload
        assert payload["covered_item_ids"] == list(covered_ids)
        assert payload["covered_sequence_ranges"]
        assert len(payload["covered_item_refs"]) == 2
        assert payload["context_version"] == 2
        assert len(store.list_context_items(second.turn_id)) == 4
        assert store.verify().valid is True

        store.rebuild_projections()
        assert manager.build(second.turn_id) == projected


def test_compaction_requires_a_prefix_and_explicit_critical_fact_categories(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="current question",
            binding_manifest={"model_alias": "model-v1"},
            input_files=({"workspace_path": "input.txt", "sha256": "b" * 64},),
        )
        items = store.list_context_items(turn.turn_id)
        manager = RolloutContextManager(store)

        with pytest.raises(ValueError, match="contiguous context prefix"):
            manager.compact(
                turn_id=turn.turn_id,
                covered_item_ids=(items[1].item_id,),
                summary="invalid gap",
                preserved_facts={},
                context_version=1,
            )
        with pytest.raises(ValueError, match="critical fact categories"):
            manager.compact(
                turn_id=turn.turn_id,
                covered_item_ids=(items[0].item_id,),
                summary="missing safety accounting",
                preserved_facts={},
                context_version=1,
            )

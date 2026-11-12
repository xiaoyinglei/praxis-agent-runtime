"""Complete checkpoints retain prior work and admit the continuation before I/O."""

import asyncio

from agent_runtime.harness import RolloutContextManager, RuntimeStore, turn_ops
from agent_runtime.modeling.contracts import ContextCapacityPolicy
from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
from tests.agent.harness.test_context_capacity_policy import capacity_runner
from tests.agent.harness.test_tool_result_elision import exchange


def test_second_checkpoint_cannot_drop_first_checkpoint_to_fit(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "complete.db") as store:
        runner, turn, gateway = capacity_runner(store, tmp_path, window=8192)
        manager = runner._context_manager
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "old work " * 1000})
        digest, _ = manager.semantic_source(turn.turn_id, checkpoint=True)
        manager.commit_compaction(
            manager.semantic_candidate(
                turn.turn_id,
                source_hash=digest,
                summary="UNIQUE_OLD_DECISION: preserve the ABI; step one applied.",
                checkpoint=True,
            )
        )
        for i in range(8):
            seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": f"NEW_WORK_{i}: " + "x" * 3000})
        result = asyncio.run(runner.drive(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        sources = "\n".join(
            str(op.request_ref["step_snapshot"]["messages"])
            for op in store.list_model_operations(turn.turn_id)
            if op.request_ref["purpose"] == "context_summary"
        )
        assert "UNIQUE_OLD_DECISION" in sources
        assert all(f"NEW_WORK_{i}" in sources for i in range(8))
        assert store.verify().valid


def test_large_recent_tool_group_is_archived_before_summary_admission(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "large.db") as store:
        runner, turn, gateway = capacity_runner(
            store,
            tmp_path,
            policy=ContextCapacityPolicy(max_item_bytes=6000, max_messages=4),
        )
        original = exchange(store, turn.turn_id, "large", "ORIGINAL_EVIDENCE " * 10000)
        result = asyncio.run(runner.drive(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed", result
        summaries = [
            op for op in store.list_model_operations(turn.turn_id) if op.request_ref["purpose"] == "context_summary"
        ]
        assert summaries
        source = str(summaries[0].request_ref["step_snapshot"]["messages"])
        assert original.item_id in source and "read_context" in source
        assert store.read_item(original.item_id).payload == original.payload


def test_archival_projection_after_checkpoint_replays_the_same_window(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "elision-after-checkpoint.db") as store:
        runner, turn, _ = capacity_runner(store, tmp_path, messages=2000)
        manager = RolloutContextManager(store)
        exchange(store, turn.turn_id, "old", "old evidence " * 1000)
        digest, _ = manager.semantic_source(turn.turn_id, checkpoint=True)
        manager.commit_compaction(
            manager.semantic_candidate(
                turn.turn_id,
                source_hash=digest,
                summary="Old work done, continue.",
                checkpoint=True,
            )
        )
        exchange(store, turn.turn_id, "new", "new evidence " * 1000)
        exchange(store, turn.turn_id, "latest", "latest result")
        candidate = next(manager.cheap_candidates(turn.turn_id))
        manager.commit_compaction(candidate)
        assert manager.build(turn.turn_id) == candidate.messages
        assert store.verify().valid


def test_model_projection_starts_at_checkpoint_without_loading_archive(tmp_path, monkeypatch):
    database = tmp_path / "bounded.db"
    with RuntimeStore.open_sync(database) as store:
        runner, turn, _ = capacity_runner(store, tmp_path)
        manager = RolloutContextManager(store)
        original = exchange(store, turn.turn_id, "old", "large old output " * 10000)
        digest, _ = manager.semantic_source(turn.turn_id, checkpoint=True)
        manager.commit_compaction(
            manager.semantic_candidate(
                turn.turn_id,
                source_hash=digest,
                summary="Pending: verify the changed file.",
                checkpoint=True,
            )
        )
        expected = manager.build(turn.turn_id)
    with RuntimeStore.open_sync(database) as store:

        def no_archive(*args, **kwargs):
            raise AssertionError("normal model request loaded the full archive")

        monkeypatch.setattr(store, "list_conversation_items", no_archive)
        assert RolloutContextManager(store).build(turn.turn_id) == expected
        assert store.read_item(original.item_id).payload == original.payload
        from agent_runtime.harness import TurnRunner
        from agent_runtime.harness.tool_router import DurableToolRouter
        from tests.agent.harness.test_compaction_consistency import Accept, model_for

        rows = store.persistence.conversation_rows
        items = store.list_items

        def bounded_rows(*args, **kwargs):
            assert kwargs.get("after_sequence", 0) > 0 or kwargs.get("tool_name") == "find_tools"
            return rows(*args, **kwargs)

        def typed_items(*args, **kwargs):
            assert kwargs.get("kind") is not None
            return items(*args, **kwargs)

        monkeypatch.setattr(store.persistence, "conversation_rows", bounded_rows)
        monkeypatch.setattr(store, "list_items", typed_items)
        model, _ = model_for(100000)
        runner = TurnRunner(
            thread_id=turn.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            tool_router=DurableToolRouter(store=store, tools={}, resident_names=(), discoverable_names=()),
            completion_gate=Accept(),
        )
        context = runner.capture_step_context(runner.restore_turn_context(turn.turn_id), step=1)
        asyncio.run(runner._model_runner.prepare(context))


def test_checkpoint_reader_respects_fork_cutoff_and_context_reset(tmp_path):
    database = tmp_path / "ancestry.db"
    with RuntimeStore.open_sync(database) as store:
        _, parent, _ = capacity_runner(store, tmp_path)
        manager = RolloutContextManager(store)
        exchange(store, parent.turn_id, "old", "Original evidence " * 1000)
        digest, _ = manager.semantic_source(parent.turn_id, checkpoint=True)
        manager.commit_compaction(
            manager.semantic_candidate(
                parent.turn_id, source_hash=digest, summary="PARENT_CHECKPOINT", checkpoint=True
            )
        )
        turn_ops.complete_turn(store, turn_id=parent.turn_id, answer="Parent finished.")
        fork = turn_ops.fork_thread(store, from_turn_id=parent.turn_id)
        later = turn_ops.start_turn(
            store, thread_id=parent.thread_id, user_message="PARENT_FUTURE", binding_manifest={}
        )
        exchange(store, later.turn_id, "future", "Future evidence " * 1000)
        child = turn_ops.start_turn(store, thread_id=fork.thread_id, user_message="CHILD_WORK", binding_manifest={})
        digest, _ = manager.semantic_source(later.turn_id, checkpoint=True)
        manager.commit_compaction(
            manager.semantic_candidate(later.turn_id, source_hash=digest, summary="FUTURE_CHECKPOINT", checkpoint=True)
        )
    with RuntimeStore.open_sync(database) as store:
        manager = RolloutContextManager(store)
        messages = str(manager.build(child.turn_id))
        assert "PARENT_CHECKPOINT" in messages and "CHILD_WORK" in messages
        assert "PARENT_FUTURE" not in messages and "FUTURE_CHECKPOINT" not in messages
        turn_ops.complete_turn(store, turn_id=child.turn_id, answer="Child finished.")
        reset = turn_ops.start_turn(
            store,
            thread_id=fork.thread_id,
            user_message="RESET_WORK",
            binding_manifest={"context_reset": {"revision": "v1", "reason": "memory_revoked"}},
        )
        reset_messages = str(manager.build(reset.turn_id))
        assert "RESET_WORK" in reset_messages
        assert "PARENT_CHECKPOINT" not in reset_messages and "CHILD_WORK" not in reset_messages
        assert store.verify().valid

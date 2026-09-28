import json
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutContextManager, RolloutStore
from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
from tests.agent.harness.test_compaction_consistency import start


def exchange(store, turn_id, call_id, content):
    seed(store, turn_id=turn_id, kind="model_response", payload={
        "text": "Read evidence", "tool_calls": [{"id": call_id, "name": "read", "arguments": {}}],
    })
    return seed(store, turn_id=turn_id, kind="tool_result", payload={
        "tool_call_id": call_id, "model_content": content,
    })


def test_parallel_result_admission_keeps_all_calls_and_archives_originals(tmp_path):
    import asyncio

    from agent_runtime.harness import TurnExecutor
    from tests.agent.harness.test_compaction_consistency import Accept, model_for

    path = tmp_path / "batch-admission.db"
    with RolloutStore(path) as store:
        thread, turn = start(store, tmp_path)
        calls = [{"id": str(i), "name": "read", "arguments": {"path": f"{i}.md"}} for i in range(3)]
        seed(store, turn_id=turn.turn_id, kind="model_response", payload={"text": "Read", "tool_calls": calls})
        originals = [seed(store, turn_id=turn.turn_id, kind="tool_result", payload={
            "tool_call_id": str(i), "model_content": f"RULE_{i}: preserve this evidence. " + "sample " * 2000,
        }) for i in range(3)]
        model, gateway = model_for(9000)
        manager = RolloutContextManager(store, max_total_bytes=5000)
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=manager, completion_gate=Accept())
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        operations = store.list_model_operations(turn.turn_id)
        assert [op.request_ref["purpose"] for op in operations] == ["agent_step"]
        assert len(gateway.requests) == 1
        # Admission bounds the request. The final answer is appended afterwards;
        # it is canonical output, not part of that already-dispatched request.
        projected = RolloutContextManager(store).build(turn.turn_id)
        manager._validate_budget(projected[:-1])
        results = [m for m in projected if m.role == "tool"]
        assert [m.tool_call_id for m in results] == ["0", "1", "2"]
        for i, (item, message) in enumerate(zip(originals, results, strict=True)):
            assert f"RULE_{i}" in message.content
            assert "truncated" in message.content and item.item_id in message.content
            assert store.read_item(item.item_id).payload == item.payload
        assert len([m for m in projected if m.tool_calls][0].tool_calls) == 3
    with RolloutStore(path) as store:
        assert RolloutContextManager(store).build(turn.turn_id) == projected
        assert store.verify().valid


def test_elision_preserves_message_structure_and_restarts(tmp_path: Path):
    path = tmp_path / "r.db"
    with RolloutStore(path) as store:
        _, turn = start(store, tmp_path)
        exchange(store, turn.turn_id, "old", "old detail " * 1000)
        exchange(store, turn.turn_id, "middle-1", "intermediate result")
        exchange(store, turn.turn_id, "middle-2", "intermediate result")
        exchange(store, turn.turn_id, "recent", "Current file contents")
        manager = RolloutContextManager(store)
        original = manager.build(turn.turn_id)
        candidate = next(manager.cheap_candidates(turn.turn_id))
        assert [m.role for m in candidate.messages] == [m.role for m in original]
        assert candidate.messages[-2:] == original[-2:]
        assert candidate.messages[1].tool_calls == original[1].tool_calls
        assert candidate.messages[2].tool_call_id == "old"
        assert "read_context" in candidate.messages[2].content
        manager.commit_compaction(candidate)
        assert manager.build(turn.turn_id) == candidate.messages
        assert any("old detail " * 1000 in str(i.payload) for i in store.list_context_items(turn.turn_id))
    with RolloutStore(path) as store:
        assert RolloutContextManager(store).build(turn.turn_id) == candidate.messages
        assert store.verify().valid


def test_later_batch_admission_preserves_earlier_evidence_prefixes(tmp_path):
    import asyncio

    from agent_runtime.harness import TurnExecutor
    from tests.agent.harness.test_compaction_consistency import Accept, model_for

    with RolloutStore(tmp_path / "successive.db") as store:
        thread, turn = start(store, tmp_path)
        for i in range(3):
            exchange(store, turn.turn_id, str(i), f"RULE_{i}: " + "samples " * 2000)
        model, gateway = model_for(9000)
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=RolloutContextManager(store, max_total_bytes=5000),
                              completion_gate=Accept())
        context = runner.restore_turn_context(turn.turn_id)
        asyncio.run(runner._prepare_compacted_step(context, step=1))
        exchange(store, turn.turn_id, "next", "New file " * 200)
        _, prepared = asyncio.run(runner._prepare_compacted_step(context, step=2))
        visible = prepared.request_ref["step_snapshot"]["messages"]
        for i in range(3):
            message = next(m for m in visible if m["tool_call_id"] == str(i))
            assert f"RULE_{i}" in message["content"]
            assert message["content"].count("Tool result truncated") == 1
        assert not gateway.requests
        assert not store.list_model_operations(turn.turn_id)


def test_old_history_is_summarized_instead_of_starving_current_tool_result(tmp_path):
    import asyncio

    from agent_runtime.harness import TurnExecutor
    from tests.agent.harness.test_compaction_consistency import Accept, model_for

    with RolloutStore(tmp_path / "working-set.db") as store:
        thread, turn = start(store, tmp_path)
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "old finding " * 320})
        content = "Exact current file tail needed for patching.\n" * 25
        exchange(store, turn.turn_id, "current", content)
        model, _ = model_for(10000)
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=RolloutContextManager(store, max_total_bytes=5000),
                              completion_gate=Accept())
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        ops = store.list_model_operations(turn.turn_id)
        assert [op.request_ref["purpose"] for op in ops] == ["context_summary", "agent_step"]
        latest = ops[-1].request_ref["step_snapshot"]["messages"]
        assert next(m for m in latest if m["tool_call_id"] == "current")["content"] == content


def test_new_elision_and_semantic_compaction_do_not_resurrect_old_results(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        exchange(store, turn.turn_id, "old", "old detail " * 1000)
        exchange(store, turn.turn_id, "middle-1", "intermediate result")
        exchange(store, turn.turn_id, "middle-2", "intermediate result")
        exchange(store, turn.turn_id, "recent", "recent detail " * 1000)
        manager = RolloutContextManager(store)
        manager.commit_compaction(next(manager.cheap_candidates(turn.turn_id)))
        exchange(store, turn.turn_id, "new", "New evidence")
        candidates = list(manager.cheap_candidates(turn.turn_id))
        assert candidates, "Earlier elision must not hide other structured results from later cleaning"
        manager.commit_compaction(candidates[-1])
        assert [m.tool_call_id for m in manager.build(turn.turn_id) if m.role == "tool"] == [
            "old", "middle-1", "middle-2", "recent", "new",
        ]
        source_hash, _ = manager.semantic_source(turn.turn_id)
        manager.commit_compaction(manager.semantic_candidate(
            turn.turn_id, source_hash=source_hash, summary="Old evidence checked; work remains.",
        ))
        assert not any(m.role == "tool" for m in manager.build(turn.turn_id))
        assert store.verify().valid


def test_elision_after_summary_preserves_summary_and_survives_next_summary(tmp_path: Path):
    path = tmp_path / "r.db"
    with RolloutStore(path) as store:
        _, turn = start(store, tmp_path)
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "history " * 1000})
        manager = RolloutContextManager(store)
        source_hash, _ = manager.semantic_source(turn.turn_id)
        manager.commit_compaction(manager.semantic_candidate(
            turn.turn_id, source_hash=source_hash, summary="Keep decision A.",
        ))
        exchange(store, turn.turn_id, "old", "old detail " * 1000)
        exchange(store, turn.turn_id, "recent", "Current file")
        manager.commit_compaction(next(manager.cheap_candidates(turn.turn_id)))
        assert any("Keep decision A." in m.content for m in manager.build(turn.turn_id))
    with RolloutStore(path) as store:
        manager = RolloutContextManager(store)
        assert any("Keep decision A." in m.content for m in manager.build(turn.turn_id))
        source_hash, source = manager.semantic_source(turn.turn_id)
        assert "Keep decision A." in source
        manager.commit_compaction(manager.semantic_candidate(
            turn.turn_id, source_hash=source_hash, summary="Keep decision B, superseding A.",
        ))
        projected = manager.build(turn.turn_id)
        assert not any(m.role == "tool" for m in projected)
        assert sum(m.role == "context" for m in projected) == 1
        assert store.verify().valid


@pytest.mark.parametrize("trailing_kind", ["context_message", "user_message"])
def test_latest_exchange_is_protected_when_other_messages_follow(tmp_path: Path, trailing_kind):
    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        exchange(store, turn.turn_id, "recent", "recent detail " * 1000)
        seed(store, turn_id=turn.turn_id, kind=trailing_kind, payload={"text": "Continue with that evidence."})
        assert not list(RolloutContextManager(store).cheap_candidates(turn.turn_id))


def test_tool_admission_probes_do_not_dispatch_model_requests(tmp_path: Path):
    import asyncio

    from agent_runtime.harness import TurnExecutor
    from tests.agent.harness.test_compaction_consistency import Accept, model_for

    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        exchange(store, turn.turn_id, "old", "old detail " * 2000)
        exchange(store, turn.turn_id, "recent", "Current file")
        model, gateway = model_for()
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=RolloutContextManager(store), completion_gate=Accept())
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert len(gateway.requests) == 1
        assert [op.request_ref["purpose"] for op in store.list_model_operations(turn.turn_id)] == ["agent_step"]


def test_zero_retained_tail_is_summarized_without_synthetic_tool_replay(tmp_path):
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
    from tests.agent.harness.test_compaction_consistency import long_history, start

    database = tmp_path / "semantic-replay.db"
    with RolloutStore(database) as store:
        _, turn = start(store, tmp_path)
        long_history(store, turn)
        seed(store, turn_id=turn.turn_id, kind="model_response", payload={
            "text": "Read specs",
            "tool_calls": [{"id": "latest", "name": "read_file", "arguments": {"path": "spec.md"}}]})
        seed(store, turn_id=turn.turn_id, kind="tool_result", payload={
            "tool_call_id": "latest", "model_content": "SPEC_EVIDENCE " + "data " * 2000})
        manager = RolloutContextManager(store, max_total_bytes=5000)
        source_hash, source = manager.semantic_source(turn.turn_id)
        assert "SPEC_EVIDENCE" in source
        candidate = manager.semantic_candidate(turn.turn_id, source_hash=source_hash,
                                                summary="Read spec.md successfully. SPEC_EVIDENCE established.")
        assert not any(m.role == "tool" or m.tool_calls for m in candidate.messages)
        assert "tool_result_overrides" not in json.loads(candidate.payload_json)
        manager.commit_compaction(candidate)
        expected = candidate.messages
    with RolloutStore(database) as store:
        assert RolloutContextManager(store, max_total_bytes=5000).build(turn.turn_id) == expected
        assert store.verify().valid

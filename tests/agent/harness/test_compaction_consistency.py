from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_runtime.harness import (
    CompletionDecision,
    GatewayHarnessModel,
    HarnessModelRequest,
    RolloutContextManager,
    RolloutStore,
    TurnExecutor,
)
from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway, _resolved_model


class Accept:
    def evaluate(self, proposal):
        return CompletionDecision(action="accept", reason="complete")


def model_for(budget=9000):
    gateway = BudgetAwareCapturingGateway(max_input_tokens=budget)
    model = GatewayHarnessModel(
        model_id="test-model",
        resolved=_resolved_model(gateway=gateway, token_accounting=gateway.token_accounting),
        instructions=("Answer directly.",),
    )
    return model, gateway


def start(store, path):
    thread = store.create_thread(workspace=path)
    turn = store.start_turn(
        thread_id=thread.thread_id,
        user_message="Fix the package; preserve public API.",
        binding_manifest={"model_id": "test-model"},
    )
    return thread, turn


def long_history(store, turn):
    for i in range(8):
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": f"finding {i}: " + "x" * 3000})


def test_single_turn_compacts_and_dispatches(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        model, gateway = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
            worker_id="test-worker",
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert len([r for r in gateway.requests if ":context-summary:" not in r.request_id]) == 1
        assert any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))


def test_durable_boundary_preserves_tool_group(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        seed(
            store,
            turn_id=turn.turn_id,
            kind="model_response",
            payload={
                "text": "",
                "tool_calls": [
                    {"id": "a", "name": "read", "arguments": {}},
                    {"id": "b", "name": "read", "arguments": {}},
                ],
            },
        )
        for call in ("a", "b"):
            seed(
                store,
                turn_id=turn.turn_id,
                kind="tool_result",
                payload={"tool_call_id": call, "model_content": "result"},
            )
        store.complete_turn(turn_id=turn.turn_id, answer="done")
        follow = store.start_turn(thread_id=thread.thread_id, user_message="continue", binding_manifest={})
        manager = RolloutContextManager(store)
        manager.compact_for_budget(turn_id=follow.turn_id, retained_tail_messages=3)
        messages = manager.build(follow.turn_id)
        calls = {c.id for m in messages for c in m.tool_calls}
        assert all(m.tool_call_id in calls for m in messages if m.role == "tool")
        model, _ = model_for(100000)
        model.prepare(
            HarnessModelRequest(
                thread_id=thread.thread_id, turn_id=follow.turn_id, messages=messages, binding_manifest={}
            )
        )


def test_candidate_rebuild_and_stale_commit(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        candidate = list(
            manager.compaction_candidates(turn.turn_id, summary="Test summary of the completed findings.")
        )[-1]
        model, _ = model_for(100000)

        def prepare(messages):
            return model.prepare(
                HarnessModelRequest(
                    thread_id=thread.thread_id, turn_id=turn.turn_id, messages=messages, binding_manifest={}
                )
            )

        before = prepare(candidate.messages)
        manager.commit_compaction(candidate)
        assert manager.build(turn.turn_id) == candidate.messages
        assert prepare(manager.build(turn.turn_id)).wire_hash == before.wire_hash
        store.rebuild_projections()
        assert manager.build(turn.turn_id) == candidate.messages
        with pytest.raises(RuntimeError, match="changed"):
            manager.commit_compaction(candidate)


def test_unfit_candidate_does_not_commit(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        model, gateway = model_for(100)
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
            worker_id="test-worker",
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert not gateway.requests
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))


def test_repeated_compaction_reopen_and_fork(tmp_path: Path):
    database = tmp_path / "r.db"
    with RolloutStore(database) as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        seed(
            store, turn_id=turn.turn_id, kind="input_file", payload={"workspace_path": "notes.txt", "sha256": "a" * 64}
        )
        manager = RolloutContextManager(store)
        candidate = list(
            manager.compaction_candidates(turn.turn_id, summary="Test summary of the completed findings.")
        )[-1]
        manager.commit_compaction(candidate)
        long_history(store, turn)
        candidate = list(
            manager.compaction_candidates(turn.turn_id, summary="Test summary of the completed findings.")
        )[-1]
        manager.commit_compaction(candidate)
        expected = candidate.messages
        assert sum(m.content.startswith("Context compaction:") for m in expected) == 1
        assert sum(m.role == "user" for m in expected) == 1
        assert "notes.txt" in next(m.content for m in expected if m.role == "context")
        store.complete_turn(turn_id=turn.turn_id, answer="finished")
        fork = store.fork_thread(from_turn_id=turn.turn_id)
        follow = store.start_turn(thread_id=fork.thread_id, user_message="next", binding_manifest={})
        expected = manager.build(follow.turn_id)
        assert sum(m.content.startswith("Context compaction:") for m in expected) == 1
        assert "notes.txt" in next(m.content for m in expected if m.role == "context")
        assert store.verify().valid
    with RolloutStore(database) as store:
        assert RolloutContextManager(store).build(follow.turn_id) == expected


def test_other_connection_invalidates_candidate_on_non_message_state(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        candidate = list(
            manager.compaction_candidates(turn.turn_id, summary="Test summary of the completed findings.")
        )[-1]
        with RolloutStore(tmp_path / "r.db") as other:
            other.record_tool_execution_state(
                turn_id=turn.turn_id,
                operation_id="op-new",
                tool_call_id="call-new",
                tool_name="read",
                arguments_digest="digest",
                execution_revision="v1",
                idempotent=True,
                status="prepared",
                attempt_count=0,
                error_code=None,
                requires_reconciliation=False,
            )
        with pytest.raises(RuntimeError, match="changed"):
            manager.commit_compaction(candidate)
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))


def test_candidate_messages_do_not_share_mutable_tool_arguments(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        long_history(store, turn)
        seed(
            store,
            turn_id=turn.turn_id,
            kind="model_response",
            payload={
                "text": "",
                "tool_calls": [
                    {"id": "a", "name": "read", "arguments": {"paths": ["safe"]}},
                ],
            },
        )
        seed(store, turn_id=turn.turn_id, kind="tool_result", payload={"tool_call_id": "a", "model_content": "ok"})
        candidate = next(
            c
            for c in RolloutContextManager(store).compaction_candidates(
                turn.turn_id, summary="Test summary of the completed findings."
            )
            if any(m.tool_calls for m in c.messages)
        )
        call = next(c for m in candidate.messages for c in m.tool_calls)
        call.arguments["paths"].append("changed")
        assert next(c for m in candidate.messages for c in m.tool_calls).arguments["paths"] == ["safe"]


@pytest.mark.parametrize("limits", [{"max_messages": 5}, {"max_item_bytes": 1800}, {"max_total_bytes": 6000}])
def test_local_limits_also_use_precommit_candidates(tmp_path: Path, limits):
    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        model, gateway = model_for(9000)
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store, **limits),
            completion_gate=Accept(),
            worker_id="test-worker",
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert len([r for r in gateway.requests if ":context-summary:" not in r.request_id]) == 1


def test_previous_explicit_facts_survive_automatic_compaction(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        first = list(manager.compaction_candidates(turn.turn_id, summary="Test summary of the completed findings."))[-1]
        payload = json.loads(first.payload_json)
        payload["preserved_facts"]["architecture_and_safety_constraints"].append("Do not modify billing.")
        store.record_context_compaction(
            turn_id=turn.turn_id,
            covered_item_ids=tuple(payload["covered_item_ids"]),
            preserved_item_ids=tuple(payload["preserved_item_ids"]),
            expected_source_revision=first.source_revision,
            context_version=1,
            summary="Billing behavior was verified.",
            preserved_facts=payload["preserved_facts"],
            artifact_refs=({"path": "evidence.txt", "sha256": "b" * 64},),
        )
        long_history(store, turn)
        candidate = list(
            manager.compaction_candidates(turn.turn_id, summary="Test summary of the completed findings.")
        )[-1]
        assert "Do not modify billing." in next(m.content for m in candidate.messages if m.role == "context")
        assert "evidence.txt" in next(m.content for m in candidate.messages if m.role == "context")


def test_successful_tool_operations_are_not_pending_state(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        kwargs = dict(
            turn_id=turn.turn_id,
            operation_id="op",
            tool_call_id="a",
            tool_name="read",
            arguments_digest="digest",
            execution_revision="v1",
            idempotent=True,
            attempt_count=0,
            error_code=None,
            requires_reconciliation=False,
        )
        store.record_tool_execution_state(**kwargs, status="prepared")
        store.record_tool_execution_state(**kwargs, status="ready")
        claim = store.claim_tool_operation(operation_id="op", worker_id="test", lease_seconds=60)
        assert store.commit_tool_operation_outcome(
            operation_id="op",
            claim_generation=claim.claim_generation,
            fencing_token=claim.fencing_token,
            status="succeeded",
            attempt_count=1,
            error_code=None,
            requires_reconciliation=False,
        )
        assert store.context_durable_state(turn.turn_id)["tool_operations"] == []


def test_changed_source_is_replanned_before_dispatch(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)

        class ConcurrentManager(RolloutContextManager):
            changed = False

            def commit_compaction(self, candidate):
                if not self.changed:
                    self.changed = True
                    seed(store, turn_id=turn.turn_id, kind="context_message", payload={"text": "New constraint."})
                return super().commit_compaction(candidate)

        model, gateway = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=ConcurrentManager(store),
            completion_gate=Accept(),
            worker_id="test-worker",
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert len([r for r in gateway.requests if ":context-summary:" not in r.request_id]) == 1
        assert sum(i.kind == "context_compaction" for i in store.list_items(turn.turn_id)) == 1
        assert "New constraint." in str(gateway.requests[-1])


def test_uncertain_tool_group_is_not_covered(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        long_history(store, turn)
        call = seed(
            store,
            turn_id=turn.turn_id,
            kind="model_response",
            payload={
                "text": "",
                "tool_calls": [
                    {"id": "a", "name": "write", "arguments": {}},
                ],
            },
        )
        seed(
            store,
            turn_id=turn.turn_id,
            kind="tool_result",
            payload={"tool_call_id": "a", "model_content": "Outcome unknown; reconcile first."},
        )
        kwargs = dict(
            turn_id=turn.turn_id,
            operation_id="op",
            tool_call_id="a",
            tool_name="write",
            arguments_digest="digest",
            execution_revision="v1",
            idempotent=False,
            attempt_count=0,
            error_code=None,
            requires_reconciliation=False,
        )
        store.record_tool_execution_state(**kwargs, status="prepared")
        store.record_tool_execution_state(**kwargs, status="ready")
        claim = store.claim_tool_operation(operation_id="op", worker_id="test", lease_seconds=60)
        assert store.commit_tool_operation_outcome(
            operation_id="op",
            claim_generation=claim.claim_generation,
            fencing_token=claim.fencing_token,
            status="unknown",
            attempt_count=1,
            error_code="lost",
            requires_reconciliation=True,
        )
        candidates = list(
            RolloutContextManager(store).compaction_candidates(
                turn.turn_id, summary="Test summary of the completed findings."
            )
        )
        assert candidates
        for candidate in candidates:
            assert call.item_id not in json.loads(candidate.payload_json)["covered_item_ids"]
            assert any(m.tool_calls for m in candidate.messages)


def test_retained_successful_tool_result_is_not_clipped(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        long_history(store, turn)
        seed(
            store,
            turn_id=turn.turn_id,
            kind="model_response",
            payload={
                "text": "",
                "tool_calls": [
                    {"id": "a", "name": "read", "arguments": {}},
                ],
            },
        )
        content = "successful file content\n" * 1000
        seed(store, turn_id=turn.turn_id, kind="tool_result", payload={"tool_call_id": "a", "model_content": content})
        candidates = list(
            RolloutContextManager(store).compaction_candidates(
                turn.turn_id, summary="Test summary of the completed findings."
            )
        )
        results = [m.content for c in candidates for m in c.messages if m.role == "tool"]
        assert results
        assert all(value == content for value in results)


def test_resume_after_compaction_commit_does_not_repeat_it(tmp_path: Path):
    database = tmp_path / "r.db"
    model, gateway = model_for()
    with RolloutStore(database) as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        candidate = list(
            manager.compaction_candidates(turn.turn_id, summary="Test summary of the completed findings.")
        )[-1]
        prepared = model.prepare(
            HarnessModelRequest(
                thread_id=thread.thread_id,
                turn_id=turn.turn_id,
                messages=candidate.messages,
                binding_manifest={"model_id": "test-model"},
            )
        )
        manager.commit_compaction(candidate)
        # Simulate process loss after the durable transition, before model preparation is recorded.
    with RolloutStore(database) as store:
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
            worker_id="recovered-worker",
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert len([r for r in gateway.requests if ":context-summary:" not in r.request_id]) == 1
        assert sum(i.kind == "context_compaction" for i in store.list_items(turn.turn_id)) == 1
        assert store.list_model_operations(turn.turn_id)[0].wire_hash == prepared.wire_hash


def test_mid_turn_summary_follows_current_user_in_projection_and_provider_wire(tmp_path: Path):
    from agent_runtime.harness import HarnessModelRequest
    from agent_runtime.modeling.openai_wire import _message_payloads

    database = tmp_path / "order.db"
    with RolloutStore(database) as store:
        _, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        source_hash, source = manager.semantic_source(turn.turn_id)
        assert "Fix the package; preserve public API." in source
        candidate = manager.semantic_candidate(
            turn.turn_id,
            source_hash=source_hash,
            summary="Already read all three specifications; implementation pending.",
        )
        assert [m.role for m in candidate.messages] == ["user", "context"]
        manager.commit_compaction(candidate)
        expected = candidate.messages
    with RolloutStore(database) as store:
        messages = RolloutContextManager(store).build(turn.turn_id)
        assert messages == expected
        model, _ = model_for()
        prepared = model.prepare(
            HarnessModelRequest(thread_id="thread", turn_id=turn.turn_id, messages=messages, binding_manifest={})
        )
        wire = _message_payloads(prepared.dispatch_payload.request.messages)
        assert "Already read" not in wire[0]["content"]
        assert "Fix the package" in wire[-2]["content"]
        assert "Already read" in wire[-1]["content"]


def test_old_automatic_user_text_is_summarized_not_pinned_forever(tmp_path: Path):
    with RolloutStore(tmp_path / "facts.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        old = store.start_turn(
            thread_id=thread.thread_id, user_message="old request details " * 400, binding_manifest={}
        )
        store.complete_turn(turn_id=old.turn_id, answer="previous work complete")
        new = store.start_turn(thread_id=thread.thread_id, user_message="current task", binding_manifest={})
        manager = RolloutContextManager(store, max_total_bytes=5000)
        source_hash, source = manager.semantic_source(new.turn_id)
        assert "old request details " * 400 in source
        candidate = manager.semantic_candidate(new.turn_id, source_hash=source_hash, summary="Previous task complete.")
        assert all("old request details " * 400 not in m.content for m in candidate.messages)
        assert candidate.messages[-1].content == "current task"
        facts = json.loads(candidate.payload_json)["preserved_facts"]
        assert facts["architecture_and_safety_constraints"][0]["runtime_archive_reference"] is True
        manager.commit_compaction(candidate)
        assert store.verify().valid


def test_summary_input_includes_recent_exchange_even_when_replayed_verbatim(tmp_path: Path):
    with RolloutStore(tmp_path / "tail-source.db") as store:
        _, turn = start(store, tmp_path)
        long_history(store, turn)
        seed(
            store,
            turn_id=turn.turn_id,
            kind="model_response",
            payload={"text": "read current", "tool_calls": [{"id": "recent", "name": "read", "arguments": {}}]},
        )
        seed(
            store,
            turn_id=turn.turn_id,
            kind="tool_result",
            payload={
                "tool_call_id": "recent",
                "tool_name": "read",
                "model_content": "LATEST-TAIL-EVIDENCE",
                "is_error": False,
            },
        )
        manager = RolloutContextManager(store)
        retained = manager.semantic_retained_tail(turn.turn_id)
        with_tail = manager.semantic_source(turn.turn_id, retained_tail_messages=retained)
        assert "LATEST-TAIL-EVIDENCE" in with_tail[1]
        assert with_tail == manager.semantic_source(turn.turn_id, retained_tail_messages=0)

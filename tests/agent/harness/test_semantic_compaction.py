from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_runtime.harness import RolloutContextManager, RolloutStore, TurnExecutor
from tests.agent.harness.test_compaction_consistency import Accept, long_history, model_for, start


def test_recompaction_uses_committed_memory_and_new_history_not_raw_archive(tmp_path: Path):
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed

    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        original_hash, original = manager.semantic_source(turn.turn_id)
        manager.commit_compaction(
            manager.semantic_candidate(
                turn.turn_id, source_hash=original_hash, summary="Decision A, reason B; pending C."
            )
        )
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "New evidence supersedes A with D."})
        _, current = manager.semantic_source(turn.turn_id)
        assert "Decision A, reason B; pending C." in current
        assert "New evidence supersedes A with D." in current
        assert "x" * 3000 not in current
        assert len(current) < len(original) / 2
        # Compaction is a projection: the complete original remains recoverable.
        assert any("x" * 3000 in str(item.payload) for item in store.list_context_items(turn.turn_id))


def test_full_history_is_chunked_and_internal_summaries_are_durable(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        source_hash, source = manager.semantic_source(turn.turn_id)
        model, gateway = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        operations = store.list_model_operations(turn.turn_id)
        internal = [op for op in operations if op.request_ref.get("purpose") == "context_summary"]
        assert len(internal) > 1
        fragments = [
            op.request_ref["step_snapshot"]["messages"][0]["content"].split("HISTORY:\n", 1)[1] for op in internal
        ]
        original_history = json.loads(source)["history"]
        actual_history = [entry for text in fragments
                          for entry in json.loads(text)["history"] if "history_index" in entry]
        assert actual_history == original_history
        assert runner.agent_step_count(turn.turn_id) == 1
        assert all(store.read_item(op.response_item_id).kind == "context_summary_response" for op in internal)
        compactions = [i for i in store.list_items(turn.turn_id) if i.kind == "context_compaction"]
        assert len(compactions) == 1
        assert compactions[0].payload["algorithm_revision"] == "semantic-compaction-v4"
        assert store.read_budget_state(turn.turn_id).used.total_tokens > 0
        from agent_runtime.harness.events import RolloutEventReader

        events = RolloutEventReader(store).read(thread.thread_id)
        assert events is not None
        assert store.verify().valid


def test_latest_tool_evidence_survives_semantic_compaction_verbatim(tmp_path: Path):
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed

    with RolloutStore(tmp_path / "r.db") as store:
        _, turn = start(store, tmp_path)
        long_history(store, turn)
        seed(
            store,
            turn_id=turn.turn_id,
            kind="model_response",
            payload={"text": "", "tool_calls": [{"id": "read-a", "name": "read_context", "arguments": {}}]},
        )
        seed(
            store,
            turn_id=turn.turn_id,
            kind="tool_result",
            payload={"tool_call_id": "read-a", "model_content": "Verified original: TTL=17, implementation pending."},
        )
        manager = RolloutContextManager(store)
        retained = manager.semantic_retained_tail(turn.turn_id)
        assert retained == 2
        source_hash, source = manager.semantic_source(turn.turn_id, retained_tail_messages=retained)
        assert "Verified original:" in source
        manager.commit_compaction(
            manager.semantic_candidate(
                turn.turn_id, source_hash=source_hash, summary="Earlier work pending.", retained_tail_messages=retained
            )
        )
        messages = manager.build(turn.turn_id)
        assert messages[-2].tool_calls[0].id == "read-a"
        assert messages[-1].content == "Verified original: TTL=17, implementation pending."


def test_summary_with_exhausted_budget_stops_before_adapter(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from agent_runtime.harness.protocol import ContextBudgetExceededError

    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        model, gateway = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )
        context = runner.restore_turn_context(turn.turn_id)
        monkeypatch.setattr(store, "read_budget_state", lambda _: SimpleNamespace(remaining=lambda _: 0))
        with pytest.raises(ContextBudgetExceededError, match="No model token budget"):
            asyncio.run(runner._semantic_summary(context, step=1, source="history"))
        assert not gateway.requests
        assert not store.list_model_operations(turn.turn_id)


def test_committed_summary_is_reused_after_restart_before_compaction(tmp_path: Path):
    database = tmp_path / "r.db"
    with RolloutStore(database) as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        source_hash, source = manager.semantic_source(turn.turn_id)
        model, gateway = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        summary = asyncio.run(
            runner._semantic_summary(runner.restore_turn_context(turn.turn_id), step=1, source=source)
        )
        assert manager.semantic_source(turn.turn_id) == (source_hash, source)
        assert summary
    with RolloutStore(database) as store:
        manager = RolloutContextManager(store)
        model, gateway = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        result = asyncio.run(runner.recover_committed_model_response(turn_id=turn.turn_id))
        assert result.status == "completed"
        assert len(gateway.requests) == 1  # only the actual agent step, no summary redispatch
        assert store.verify().valid


def test_provider_overflow_retries_once_with_a_new_durable_request(tmp_path: Path):
    from agent_runtime.harness.protocol import ModelContextOverflowError

    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        model, gateway = model_for(100000)
        original_dispatch = model.dispatch
        agent_calls = []

        async def dispatch(prepared, *, delta_sink=None):
            if prepared.request_ref.get("purpose") != "context_summary":
                agent_calls.append(prepared.request_ref["request_id"])
                if len(agent_calls) == 1:
                    raise ModelContextOverflowError("provider context window rejected")
            return await original_dispatch(prepared, delta_sink=delta_sink)

        model.dispatch = dispatch
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert len(agent_calls) == 2 and len(set(agent_calls)) == 2
        assert runner.agent_step_count(turn.turn_id) == 1
        assert store.verify().valid


def test_second_provider_overflow_stops_without_retry_loop(tmp_path: Path):
    from agent_runtime.harness.protocol import ModelContextOverflowError

    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        model, gateway = model_for(100000)
        original_dispatch = model.dispatch
        calls = []

        async def dispatch(prepared, *, delta_sink=None):
            if prepared.request_ref.get("purpose") != "context_summary":
                calls.append(prepared.request_ref["request_id"])
                raise ModelContextOverflowError("still too large")
            return await original_dispatch(prepared, delta_sink=delta_sink)

        model.dispatch = dispatch
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert len(calls) == 2
        assert store.read_turn(turn.turn_id).terminal_reason_code == "context_budget_exhausted"
        assert store.verify().valid


def test_request_inside_effective_provider_budget_does_not_apply_another_percentage(tmp_path: Path):
    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        model, gateway = model_for(28000)
        from agent_runtime.harness import HarnessModelRequest

        request = model.prepare(
            HarnessModelRequest(
                thread_id=thread.thread_id,
                turn_id=turn.turn_id,
                messages=manager.build(turn.turn_id),
                binding_manifest={},
            )
        )
        pressure = request.request_ref["context_projection"]
        assert pressure["input_tokens"] < pressure["max_input_tokens"]
        assert pressure["input_tokens"] > pressure["max_input_tokens"] * 0.85
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert not any(
            op.request_ref.get("purpose") == "context_summary" for op in store.list_model_operations(turn.turn_id)
        )


def test_unknown_summary_retry_preserves_internal_purpose(tmp_path: Path):
    from agent_runtime.harness.protocol import ModelDispatchOutcomeUnknownError

    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        model, gateway = model_for()
        original_dispatch = model.dispatch
        calls = []

        async def dispatch(prepared, *, delta_sink=None):
            calls.append(prepared.request_ref["request_id"])
            if len(calls) == 1:
                assert prepared.request_ref["purpose"] == "context_summary"
                assert delta_sink is None
                raise ModelDispatchOutcomeUnknownError("connection lost after dispatch")
            return await original_dispatch(prepared, delta_sink=delta_sink)

        model.dispatch = dispatch
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )
        paused = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert paused.status == "paused"
        result = asyncio.run(runner.retry_unknown_model(turn_id=turn.turn_id))
        assert result.status == "completed"
        assert calls[0] == calls[1]
        assert runner.agent_step_count(turn.turn_id) == 1
        assert store.verify().valid


def test_crash_after_summary_prepare_recovers_same_operation(tmp_path: Path):
    import pytest

    database = tmp_path / "r.db"
    with RolloutStore(database) as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        model, _ = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )

        async def crash(**kwargs):
            raise RuntimeError("crash before dispatch")

        runner._dispatch_prepared = crash
        with pytest.raises(RuntimeError, match="crash before dispatch"):
            asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        [prepared] = store.list_model_operations(turn.turn_id)
        assert prepared.status == "prepared"
    with RolloutStore(database) as store:
        model, gateway = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )
        result = asyncio.run(runner.recover_prepared_model(turn_id=turn.turn_id))
        assert result.status == "completed"
        operations = store.list_model_operations(turn.turn_id)
        assert operations[0].operation_id == prepared.operation_id
        assert len(store.list_model_attempts(prepared.operation_id)) == 1
        assert store.verify().valid


def test_old_tool_outputs_are_archived_before_spending_a_summary_call(tmp_path: Path):
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed

    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        for index in range(5):
            seed(
                store,
                turn_id=turn.turn_id,
                kind="model_response",
                payload={
                    "text": f"finding {index}",
                    "tool_calls": [{"id": str(index), "name": "read", "arguments": {}}],
                },
            )
            seed(
                store,
                turn_id=turn.turn_id,
                kind="tool_result",
                payload={
                    "tool_call_id": str(index),
                    "model_content": "old detail " * 1000 if index < 2 else f"recent result {index}",
                },
            )
        model, gateway = model_for()
        runner = TurnExecutor(
            thread_id=thread.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed" and len(gateway.requests) == 1
        [compaction] = [i for i in store.list_items(turn.turn_id) if i.kind == "context_compaction"]
        assert compaction.payload["algorithm_revision"] == "tool-output-elision-v5"
        projected = RolloutContextManager(store).build(turn.turn_id)
        for index in range(5):
            assert any(m.role == "assistant" and m.content == f"finding {index}" for m in projected)
        for index in range(2, 5):
            assert any(m.role == "tool" and m.tool_call_id == str(index)
                       and m.content == f"recent result {index}" for m in projected)


def test_incomplete_summary_never_commits_a_context_transition(tmp_path: Path):
    from agent_runtime.harness import HarnessModelResponse

    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        original = manager.build(turn.turn_id)
        model, _ = model_for()

        async def dispatch(prepared, *, delta_sink=None):
            assert prepared.request_ref["purpose"] == "context_summary"
            assert delta_sink is None
            return HarnessModelResponse(
                text="cut off",
                provider_response_id="partial",
                usage={},
                status="incomplete",
                incomplete_reason="max_tokens",
            )

        model.dispatch = dispatch
        runner = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert manager.build(turn.turn_id) == original
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))
        assert store.verify().valid


def test_oversized_generated_summary_is_rejected_without_length_ladder(tmp_path: Path):
    from agent_runtime.harness import HarnessModelResponse

    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        model, _ = model_for(100000)
        original_dispatch = model.dispatch
        summaries = []

        async def dispatch(prepared, *, delta_sink=None):
            if prepared.request_ref["purpose"] == "context_summary":
                summaries.append(prepared.request_ref["step_snapshot"]["messages"][0]["content"])
                return HarnessModelResponse(
                    text="Long narrative. " * 500
                    if len(summaries) == 1
                    else "Decision kept; work and verification pending.",
                    provider_response_id="summary",
                    usage={},
                )
            return await original_dispatch(prepared, delta_sink=delta_sink)

        model.dispatch = dispatch
        manager = RolloutContextManager(store, max_total_bytes=5000)
        executor = TurnExecutor(
            thread_id=thread.thread_id, store=store, model=model, context_manager=manager, completion_gate=Accept()
        )
        result = asyncio.run(executor.run_turn(executor.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed" and len(summaries) == 1
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))
        assert store.verify().valid


def test_summary_disables_explicit_thinking_without_changing_agent_settings():
    from dataclasses import replace

    from agent_runtime.harness import GatewayHarnessModel, HarnessMessage, HarnessModelRequest
    from agent_runtime.model_definition import ProviderOptionsDefinition, ThinkingOptionsDefinition
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway, _resolved_model

    gateway = BudgetAwareCapturingGateway(max_input_tokens=9000)
    resolved = _resolved_model(gateway=gateway, token_accounting=gateway.token_accounting)
    resolved = replace(
        resolved,
        request_defaults=resolved.request_defaults.model_copy(
            update={"provider_options": ProviderOptionsDefinition(thinking=ThinkingOptionsDefinition(type="enabled"))}
        ),
    )
    model = GatewayHarnessModel(model_id="test", resolved=resolved, instructions=("Work.",))
    request = HarnessModelRequest(
        thread_id="t", turn_id="u", messages=(HarnessMessage(role="user", content="History"),), binding_manifest={}
    )
    summary = model.prepare(replace(request, purpose="context_summary", request_id="summary"))
    normal = model.prepare(request)
    assert summary.dispatch_payload.request.settings.provider_options["thinking"]["type"] == "disabled"
    assert normal.dispatch_payload.request.settings.provider_options["thinking"]["type"] == "enabled"

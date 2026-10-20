import pytest

from agent_runtime.modeling.contracts import DEFAULT_LLM_STAGE_BUDGETS, LLMCallStage, LLMStageBudget
from agent_runtime.modeling.gateway import LLMGateway
from tests.agent.harness.test_model_adapter import CharacterAccounting


def gateway(window, cap=None):
    budgets = {stage: budget.model_copy() for stage, budget in DEFAULT_LLM_STAGE_BUDGETS.items()}
    if cap is not None:
        budgets[LLMCallStage.AGENT_STEP] = LLMStageBudget(max_input_tokens=cap, max_output_tokens=32768)
    return LLMGateway(
        generator=object(), token_accounting=CharacterAccounting(), model_context_tokens=window, stage_budgets=budgets
    )


@pytest.mark.parametrize("window,output", [(8192, 1024), (131072, 8192), (1048576, 32768)])
def test_default_input_capacity_uses_model_window(window, output):
    effective = gateway(window).effective_stage_budget(LLMCallStage.AGENT_STEP, kwargs={"max_tokens": output})
    assert effective.max_input_tokens == window - output - 512
    assert DEFAULT_LLM_STAGE_BUDGETS[LLMCallStage.AGENT_STEP].max_input_tokens is None


def test_explicit_cap_does_not_limit_default_summary_capacity():
    model = gateway(1048576, cap=64000)
    assert model.effective_stage_budget(LLMCallStage.AGENT_STEP).max_input_tokens == 64000
    summary = model.effective_stage_budget(LLMCallStage.CONTEXT_COMPACTION)
    assert summary.max_input_tokens == 1048576 - summary.max_output_tokens - summary.safety_margin_tokens


def test_no_input_capacity_fails_without_changing_output():
    with pytest.raises(ValueError, match="No input budget"):
        gateway(1024).effective_stage_budget(LLMCallStage.AGENT_STEP, kwargs={"max_tokens": 1024})


def test_old_definition_roundtrip_preserves_canonical_bytes_and_integer_cap():
    import json

    from agent_runtime.core.llm_config import AgentModelsConfig, ModelSpec
    from agent_runtime.model_definition import (
        ModelExecutionDefinition,
        build_model_execution_definition,
        canonical_definition_json,
    )

    config = AgentModelsConfig(
        default_model="example",
        models={
            "example": ModelSpec(
                provider="openai_compatible", base_url="https://example.com/v1", context_window_tokens=1048576
            )
        },
    )
    definition = build_model_execution_definition(model_id="example", spec=config.models["example"], config=config)
    new = json.loads(canonical_definition_json(definition))
    assert new["context_capacity_policy"]["revision"] == "context-capacity-v2"
    old = {key: value for key, value in new.items() if key != "context_capacity_policy"}
    old["llm_stage_budgets"]["agent_step"]["max_input_tokens"] = 64000
    encoded = json.dumps(old, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    restored = ModelExecutionDefinition.model_validate_json(encoded)
    assert canonical_definition_json(restored) == encoded
    assert restored.llm_stage_budgets["agent_step"].max_input_tokens == 64000


def prepared_request(window, size, cap=None, output=32768, input_limit=None):
    from agent_runtime.harness import GatewayHarnessModel, HarnessMessage, HarnessModelRequest
    from tests.agent.harness.test_model_adapter import _resolved_model

    model_gateway = gateway(window, cap)
    resolved = _resolved_model(
        gateway=model_gateway,
        token_accounting=model_gateway.token_accounting,
        context_window_tokens=window,
        max_output_tokens=output,
    )
    model = GatewayHarnessModel(model_id="test-model", resolved=resolved, instructions=("Instructions.",))
    return model.prepare(
        HarnessModelRequest(
            thread_id="thread",
            turn_id="turn",
            binding_manifest={},
            messages=(HarnessMessage(role="user", content="x" * size),),
            input_token_limit=input_limit,
        )
    )


def test_one_million_window_admits_64k_and_reports_policy():
    projection = prepared_request(1048576, 64000).request_ref["context_projection"]
    assert projection["input_tokens"] > 64000  # includes full protocol/instructions
    assert projection["effective_input_limit"] == 1015296
    assert projection["auto_compact_limit"] == 943718
    assert projection["post_compact_target"] == 754974
    assert projection["explicit_input_cap"] is None


def test_soft_threshold_triggers_before_hard_limit():
    from agent_runtime.harness import ContextCompactionRequiredError

    with pytest.raises(ContextCompactionRequiredError) as failure:
        prepared_request(1048576, 950000)
    assert failure.value.max_input_tokens == 943718
    assert failure.value.reason_code == "token_threshold"


def test_explicit_cap_still_triggers_on_full_request():
    from agent_runtime.harness import ContextCompactionRequiredError

    with pytest.raises(ContextCompactionRequiredError):
        prepared_request(1048576, 64000, cap=64000)


@pytest.mark.parametrize("cursor", [297, None])
@pytest.mark.parametrize("body_budget,visible_end", [(0, None), (500, 146)])
def test_web_elision_preserves_fetch_cursor(cursor, body_budget, visible_end):
    import json

    from agent_runtime.harness.context import _web_result_excerpt

    content = json.dumps(
        {
            "structured_content": {
                "source_id": "artifact_page",
                "url": "https://example.com",
                "start_line": 97,
                "next_line": cursor,
                "content": "\n".join(f"L{n}: row" for n in range(97, 297)),
            }
        }
    )
    # Each line is 9 or 10 bytes; use exact prefix length for the 50-line case.
    if body_budget:
        body_budget = len("\n".join(f"L{n}: row" for n in range(97, 147)).encode())
    excerpt = json.loads(_web_result_excerpt(content, body_budget, "original"))
    source = excerpt["structured_content"]
    assert source["next_line"] == cursor
    assert source["visible_end_line"] == visible_end
    assert "fetch_next_line" not in source
    assert source["start_line"] == 97
    assert "original" in excerpt["archive_lookup"]


def test_semantic_source_restores_fact_hidden_by_old_elision(tmp_path):
    from agent_runtime.harness import RolloutContextManager, RolloutStore
    from tests.agent.harness.test_compaction_consistency import long_history
    from tests.agent.harness.test_tool_result_elision import exchange

    with RolloutStore(tmp_path / "source.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="task",
            binding_manifest={"context_capacity_policy": {"revision": "context-capacity-v2"}},
        )
        long_history(store, turn)
        original = exchange(store, turn.turn_id, "old", "prefix " * 1000 + "GOLD_FACT_TTL_17")
        exchange(store, turn.turn_id, "recent", "recent result")
        manager = RolloutContextManager(store)
        manager.commit_compaction(next(manager.cheap_candidates(turn.turn_id, max_result_bytes=20)))
        assert "GOLD_FACT_TTL_17" not in str(manager.build(turn.turn_id))
        source_hash, source = manager.semantic_source(turn.turn_id)
        assert "GOLD_FACT_TTL_17" in source
        assert store.read_item(original.item_id).payload["model_content"].endswith("GOLD_FACT_TTL_17")
        manager.commit_compaction(
            manager.semantic_candidate(turn.turn_id, source_hash=source_hash, summary="TTL=17 verified.")
        )
        exchange(store, turn.turn_id, "new", "NEW_EVIDENCE")
        _, second = manager.semantic_source(turn.turn_id)
        assert "TTL=17 verified." in second and "NEW_EVIDENCE" in second
        assert "GOLD_FACT_TTL_17" not in second  # summarized originals stay archived


def test_new_turn_body_guards_and_explicit_lower_guard(tmp_path):
    from agent_runtime.harness import RolloutContextManager, RolloutStore
    from agent_runtime.modeling.contracts import ContextCapacityPolicy
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed

    with RolloutStore(tmp_path / "guards.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="task",
            binding_manifest={"context_capacity_policy": ContextCapacityPolicy().model_dump()},
        )
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "x" * 600000})
        assert len(RolloutContextManager(store).build(turn.turn_id)) == 2
        with pytest.raises(Exception, match="byte_guard"):
            RolloutContextManager(store, max_item_bytes=500000).build(turn.turn_id)


def capacity_runner(store, path, *, tokens=200000, window=1048576, messages=3, summary=None, policy=None):
    from agent_runtime.harness import GatewayHarnessModel, RolloutContextManager, TurnExecutor
    from agent_runtime.modeling.contracts import ContextCapacityPolicy
    from tests.agent.harness.test_compaction_consistency import Accept
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway, _resolved_model

    thread = store.create_thread(workspace=path)
    turn = store.start_turn(
        thread_id=thread.thread_id,
        user_message="synthetic task",
        binding_manifest={
            "model_token_budget_total": tokens,
            "model_step_budget": 16,
            "context_capacity_policy": (policy or ContextCapacityPolicy(max_messages=messages)).model_dump(),
        },
    )
    model_gateway = BudgetAwareCapturingGateway(max_input_tokens=window - 1024)
    resolved = _resolved_model(
        gateway=model_gateway, token_accounting=model_gateway.token_accounting, context_window_tokens=window
    )
    model = GatewayHarnessModel(model_id="test-model", resolved=resolved, instructions=("Follow the task.",))
    if summary is not None:
        from dataclasses import replace

        original = model.dispatch

        async def dispatch(prepared, **kwargs):
            response = await original(prepared, **kwargs)
            return (
                replace(response, text=summary)
                if prepared.request_ref.get("purpose") == "context_summary"
                else response
            )

        model.dispatch = dispatch
    runner = TurnExecutor(
        thread_id=thread.thread_id,
        store=store,
        model=model,
        context_manager=RolloutContextManager(store),
        completion_gate=Accept(),
    )
    return runner, turn, model_gateway


def test_message_guard_with_1m_window_and_200k_budget_can_summarize(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutStore
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed

    with RolloutStore(tmp_path / "reserve.db") as store:
        runner, turn, model_gateway = capacity_runner(store, tmp_path)
        for index in range(8):
            seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": f"fact {index}: " + "x" * 8000})
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed", result
        operations = store.list_model_operations(turn.turn_id)
        assert [op.request_ref["purpose"] for op in operations] == ["context_summary", "agent_step"]
        plan = operations[0].request_ref["compaction_plan"]
        assert plan["continuation_input_limit"] < 40000
        assert plan["planned_token_upper_bound"] < 200000
        assert len(model_gateway.requests) == 2
        assert store.verify().valid


def test_auto_compaction_uses_summary_even_when_elision_would_fit(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutStore
    from tests.agent.harness.test_tool_result_elision import exchange

    with RolloutStore(tmp_path / "semantic-first.db") as store:
        runner, turn, _ = capacity_runner(store, tmp_path, messages=5)
        for index in range(3):
            exchange(store, turn.turn_id, str(index), "evidence " * 2000)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed", result
        compacted = [item for item in store.list_items(turn.turn_id) if item.kind == "context_compaction"]
        assert compacted and all(item.payload["algorithm_revision"].startswith("semantic-") for item in compacted)
        assert store.list_model_operations(turn.turn_id)[0].request_ref["purpose"] == "context_summary"


def test_oversized_serialized_summary_stops_before_continuation(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutStore
    from tests.agent.harness.test_compaction_consistency import long_history

    with RolloutStore(tmp_path / "oversized.db") as store:
        runner, turn, model_gateway = capacity_runner(store, tmp_path, summary="\\" * 20000)
        long_history(store, turn)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert len(model_gateway.requests) == 1
        assert not any(item.kind == "context_compaction" for item in store.list_items(turn.turn_id))


def test_acquisition_ranges_merge_only_same_source_version_and_keep_gaps():
    from agent_runtime.harness.context import _web_acquired_ranges

    refs = [
        {
            "source_id": "artifact_page",
            "content_hash": version,
            "view": "content",
            "line_basis": "saved",
            "start_line": start,
            "acquired_end_line": end,
        }
        for version, start, end in [
            ("v1", 20, 30),
            ("v1", 1, 10),
            ("v1", 5, 12),
            ("v1", 13, 19),
            ("v1", 40, 50),
            ("v2", 31, 39),
        ]
    ]
    groups = _web_acquired_ranges(refs)
    assert next(ref for ref in groups if ref["content_hash"] == "v1")["acquired_ranges"] == [[1, 30], [40, 50]]
    assert next(ref for ref in groups if ref["content_hash"] == "v2")["acquired_ranges"] == [[31, 39]]


def test_retained_web_exchange_keeps_canonical_acquisition_references(tmp_path):
    import json

    from agent_runtime.harness import RolloutContextManager, RolloutStore
    from tests.agent.harness.test_compaction_consistency import long_history
    from tests.agent.harness.test_tool_result_elision import exchange

    with RolloutStore(tmp_path / "retained-web.db") as store:
        _, turn, _ = capacity_runner(store, tmp_path, messages=2000)
        long_history(store, turn)
        item = exchange(
            store,
            turn.turn_id,
            "web",
            json.dumps(
                {
                    "structured_content": {
                        "source_id": "artifact_page",
                        "url": "https://example.com",
                        "content_hash": "v1",
                        "view": "content",
                        "line_basis": "saved",
                        "start_line": 97,
                        "next_line": 297,
                        "content": "\n".join(f"L{n}: evidence" for n in range(97, 297)),
                    }
                }
            ),
        )
        manager = RolloutContextManager(store)
        source_hash, _ = manager.semantic_source(turn.turn_id)
        candidate = manager.semantic_candidate(
            turn.turn_id, source_hash=source_hash, summary="Evidence summarized.", retained_tail_messages=2
        )
        payload = json.loads(candidate.payload_json)
        assert item.item_id in payload["preserved_item_ids"]
        ref = next(ref for ref in payload["artifact_refs"] if ref.get("item_id") == item.item_id)
        assert ref["next_line"] == 297 and ref["acquired_end_line"] == 296
        assert next(ref for ref in payload["artifact_refs"] if ref.get("kind") == "web_acquired_ranges")[
            "acquired_ranges"
        ] == [[97, 296]]


def test_small_window_summary_tree_measures_each_leaf_and_merge(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutStore
    from tests.agent.harness.test_compaction_consistency import long_history

    with RolloutStore(tmp_path / "tree.db") as store:
        runner, turn, model_gateway = capacity_runner(store, tmp_path, window=8192)
        long_history(store, turn)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed", result
        summaries = [
            op for op in store.list_model_operations(turn.turn_id) if op.request_ref["purpose"] == "context_summary"
        ]
        assert len(summaries) > 1
        for op in summaries:
            plan = op.request_ref["compaction_plan"]
            assert op.request_ref["context_projection"]["input_tokens"] <= plan["summary_input_limit"]
            assert plan["continuation_input_limit"] < 8192
        assert len(model_gateway.requests) == len(summaries) + 1
        assert store.verify().valid


def test_large_window_summary_leaves_respect_frozen_materialization_guard(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutStore
    from agent_runtime.modeling.contracts import ContextCapacityPolicy
    from tests.agent.harness.test_compaction_consistency import long_history

    with RolloutStore(tmp_path / "materialization.db") as store:
        runner, turn, model_gateway = capacity_runner(
            store, tmp_path, policy=ContextCapacityPolicy(max_item_bytes=5000, max_messages=3)
        )
        long_history(store, turn)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed", result
        summaries = [
            op for op in store.list_model_operations(turn.turn_id) if op.request_ref["purpose"] == "context_summary"
        ]
        assert len(summaries) > 1  # capacity alone could admit the whole history
        assert all(
            len(message["content"].encode("utf-8")) <= 5000
            for op in summaries
            for message in op.request_ref["step_snapshot"]["messages"]
        )
        assert len(model_gateway.requests) == len(summaries) + 1
        assert store.verify().valid


def test_indivisible_summary_work_unit_over_byte_guard_never_dispatches(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutStore
    from agent_runtime.modeling.contracts import ContextCapacityPolicy
    from tests.agent.harness.test_compaction_consistency import long_history
    from tests.agent.harness.test_tool_result_elision import exchange

    with RolloutStore(tmp_path / "indivisible.db") as store:
        runner, turn, model_gateway = capacity_runner(
            store, tmp_path, policy=ContextCapacityPolicy(max_item_bytes=5000)
        )
        long_history(store, turn)
        original = exchange(store, turn.turn_id, "large", "canonical evidence " * 1000)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert not model_gateway.requests and not store.list_model_operations(turn.turn_id)
        assert store.read_item(original.item_id).payload == original.payload
        assert store.verify().valid


def test_merge_exceeding_reserved_node_limit_is_not_dispatched(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutStore
    from tests.agent.harness.test_compaction_consistency import long_history

    with RolloutStore(tmp_path / "merge-too-large.db") as store:
        runner, turn, model_gateway = capacity_runner(store, tmp_path, window=8192, summary="x" * 3300)
        long_history(store, turn)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert len(model_gateway.requests) == 2  # two leaves; no unreserved merge or continuation
        assert not any(item.kind == "context_compaction" for item in store.list_items(turn.turn_id))
        assert store.verify().valid


def test_budget_planning_can_drop_redundant_latest_exchange_before_summary(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutStore
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
    from tests.agent.harness.test_tool_result_elision import exchange

    with RolloutStore(tmp_path / "tail-budget.db") as store:
        runner, turn, model_gateway = capacity_runner(store, tmp_path, tokens=150000, messages=4)
        for _index in range(8):
            seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "x" * 8000})
        exchange(store, turn.turn_id, "latest", "LAST_EXCHANGE_FACT " + "y" * 20000)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed", result
        summary = store.list_model_operations(turn.turn_id)[0]
        assert summary.request_ref["compaction_plan"]["retained_tail_messages"] == 0
        assert "LAST_EXCHANGE_FACT" in str(summary.request_ref["step_snapshot"]["messages"])
        assert len(model_gateway.requests) == 2


def test_capacity_and_compaction_are_visible_in_replayed_events(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutStore
    from agent_runtime.harness.events import RolloutEventReader
    from tests.agent.harness.test_compaction_consistency import long_history

    with RolloutStore(tmp_path / "events.db") as store:
        runner, turn, _ = capacity_runner(store, tmp_path)
        long_history(store, turn)
        asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        events = RolloutEventReader(store).read(runner.thread_id)
        started = next(event for event in events if event.event_type == "turn_started")
        assert started.data["context_capacity_policy"]["revision"] == "context-capacity-v2"
        compacted = next(event for event in events if event.event_type == "compact_layer")
        assert compacted.data["compacted"] is True
        assert compacted.data["trigger_reason"] == "message_guard"
        assert compacted.data["after_input_tokens"] < compacted.data["before_input_tokens"]
        assert compacted.data["summary_calls"] == 1
        budget = next(
            event for event in events if event.event_type == "budget_update" and event.data["purpose"] == "agent_step"
        )
        assert budget.data["context_projection"]["model_context_tokens"] == 1048576
        assert budget.data["budget_status"]["cost_remaining_micros"] == "unset"
        assert "agent_steps_remaining" in budget.data["budget_status"]
        message = store.list_model_operations(turn.turn_id)[-1].request_ref["step_snapshot"]["messages"][-1]
        assert "Runtime budget status:" in message["content"]


def test_terminal_displays_capacity_and_compaction_tokens_without_cost(capsys):
    import asyncio

    from agent_runtime.streaming.events import EventType, StreamEvent
    from agent_runtime.terminal_render import TerminalToolEventDisplay

    display = TerminalToolEventDisplay(interactive=False)
    display.set_verbose(True)
    asyncio.run(
        display.emit(
            StreamEvent(
                type=EventType.BUDGET_UPDATE,
                turn_id="turn",
                data={
                    "purpose": "agent_step",
                    "context_projection": {
                        "model_context_tokens": 1048576,
                        "effective_input_limit": 1015296,
                        "auto_compact_limit": 943718,
                        "explicit_input_cap": None,
                    },
                    "budget_status": {
                        "agent_steps_remaining": 15,
                        "web_requests_remaining": 30,
                        "tokens_remaining": "unset",
                        "cost_remaining_micros": "unknown",
                    },
                },
            )
        )
    )
    asyncio.run(
        display.emit(
            StreamEvent(
                type=EventType.COMPACT_LAYER,
                turn_id="turn",
                data={
                    "trigger_reason": "message_guard",
                    "before_input_tokens": 64000,
                    "after_input_tokens": 3000,
                    "summary_calls": 1,
                    "summary_tokens": 1000,
                    "summary_cost_micros": "unknown",
                },
            )
        )
    )
    rendered = capsys.readouterr().out
    assert "1015296" in rendered and "943718" in rendered
    assert "message_guard" in rendered and "64000" in rendered and "3000" in rendered
    assert "unset" in rendered
    assert "cost" not in rendered and "费用" not in rendered


@pytest.mark.anyio
async def test_context_details_are_available_on_request_without_chat_noise(capsys):
    from agent_runtime.streaming.events import EventType, StreamEvent
    from agent_runtime.terminal_ui import Conversation, ConversationEventDisplay

    view = Conversation()
    display = ConversationEventDisplay(view)
    for remaining in (15, 14):
        await display.emit(StreamEvent(type=EventType.BUDGET_UPDATE, turn_id="turn", data={
            "context_projection": {"effective_input_limit": 1015296, "auto_compact_limit": 943718},
            "budget_status": {"agent_steps_remaining": remaining},
        }))
    assert "1015296" not in view.plain_text() and "压缩阈值" not in view.plain_text()
    display.show_context_status()
    rendered = capsys.readouterr().out
    assert "1015296" in rendered and "943718" in rendered
    assert "剩余 agent 步骤=14" in rendered and "剩余 agent 步骤=15" not in rendered
    display.set_verbose(True)
    assert "剩余 agent 步骤=14" in view.plain_text()
    assert "剩余 agent 步骤=15" not in view.plain_text()
    await display.emit(StreamEvent(type=EventType.BUDGET_UPDATE, turn_id="turn", data={
        "context_projection": {"effective_input_limit": 999}, "budget_status": {},
    }))
    assert "输入上限=999" in view.plain_text()
    assert "输入上限=1015296" not in view.plain_text()
    display.set_verbose(False)
    assert "输入上限" not in view.plain_text()
    display.set_verbose(True)
    assert view.plain_text().count("输入上限=999") == 1
    display.clear_context_status()
    assert "输入上限" not in view.plain_text()
    display.show_context_status()
    assert capsys.readouterr().out == ""


def test_retained_exchange_is_semantically_covered_on_next_summary(tmp_path):
    from agent_runtime.harness import RolloutContextManager, RolloutStore
    from tests.agent.harness.test_compaction_consistency import long_history
    from tests.agent.harness.test_tool_result_elision import exchange

    with RolloutStore(tmp_path / "retained-coverage.db") as store:
        _, turn, _ = capacity_runner(store, tmp_path, messages=2000)
        long_history(store, turn)
        recent = exchange(store, turn.turn_id, "recent", "RECENT_CANONICAL_GOLD")
        manager = RolloutContextManager(store)
        digest, _ = manager.semantic_source(turn.turn_id)
        candidate = manager.semantic_candidate(
            turn.turn_id, source_hash=digest, summary="Recent work summarized.", retained_tail_messages=2
        )
        manager.commit_compaction(candidate)
        assert any(message.content == "RECENT_CANONICAL_GOLD" for message in manager.build(turn.turn_id))
        exchange(store, turn.turn_id, "new", "new evidence")
        _, source = manager.semantic_source(turn.turn_id)
        assert "Recent work summarized." in source and "new evidence" in source
        assert "RECENT_CANONICAL_GOLD" not in source
        import json

        assert recent.item_id in json.loads(candidate.payload_json)["covered_item_ids"]


def test_capacity_plan_reuses_completed_summary_after_crash_before_commit(tmp_path, monkeypatch):
    import asyncio

    from agent_runtime.harness import RolloutContextManager, RolloutStore, TurnExecutor
    from tests.agent.harness.test_compaction_consistency import Accept, long_history

    path = tmp_path / "crash.db"
    with RolloutStore(path) as store:
        runner, turn, original_gateway = capacity_runner(store, tmp_path)
        long_history(store, turn)
        context = runner.restore_turn_context(turn.turn_id)

        def crash(_candidate):
            raise RuntimeError("simulated crash after summary")

        monkeypatch.setattr(runner._context_manager, "commit_compaction", crash)
        with pytest.raises(RuntimeError, match="simulated crash"):
            asyncio.run(runner._prepare_compacted_step(context, step=1))
        assert len(original_gateway.requests) == 1
        [operation] = store.list_model_operations(turn.turn_id)
        assert operation.status == "completed"
        prepared_hash = operation.request_hash
        model = runner._model
    with RolloutStore(path) as store:
        reopened = TurnExecutor(
            thread_id=context.thread_id,
            store=store,
            model=model,
            context_manager=RolloutContextManager(store),
            completion_gate=Accept(),
        )
        result = asyncio.run(reopened.run_turn(reopened.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert len(original_gateway.requests) == 2  # cached summary, only continuation dispatched
        assert store.list_model_operations(turn.turn_id)[0].request_hash == prepared_hash
        assert store.verify().valid


def test_new_policy_insufficient_budget_preserves_originals_without_dispatch(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutStore
    from tests.agent.harness.test_compaction_consistency import long_history

    with RolloutStore(tmp_path / "no-budget.db") as store:
        runner, turn, model_gateway = capacity_runner(store, tmp_path, tokens=2000)
        long_history(store, turn)
        original_items = store.list_context_items(turn.turn_id)
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "failed"
        assert not model_gateway.requests
        assert not store.list_model_operations(turn.turn_id)
        assert all(store.read_item(item.item_id).payload == item.payload for item in original_items)
        assert not any(item.kind == "context_compaction" for item in store.list_items(turn.turn_id))


def test_new_policy_stale_candidate_and_incomplete_preserved_exchange_are_rejected(tmp_path):
    import json

    from agent_runtime.harness import RolloutContextManager, RolloutStore
    from agent_runtime.harness.protocol import ContextSourceChangedError
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed
    from tests.agent.harness.test_compaction_consistency import long_history
    from tests.agent.harness.test_tool_result_elision import exchange

    with RolloutStore(tmp_path / "cas.db") as store:
        _, turn, _ = capacity_runner(store, tmp_path, messages=2000)
        long_history(store, turn)
        recent = exchange(store, turn.turn_id, "recent", "recent evidence")
        manager = RolloutContextManager(store)
        source_hash, _ = manager.semantic_source(turn.turn_id)
        candidate = manager.semantic_candidate(
            turn.turn_id, source_hash=source_hash, summary="Evidence preserved.", retained_tail_messages=2
        )
        payload = json.loads(candidate.payload_json)
        with pytest.raises(ValueError, match="original call"):
            store.record_context_compaction(
                turn_id=turn.turn_id,
                covered_item_ids=tuple(payload["covered_item_ids"]),
                preserved_item_ids=(recent.item_id,),
                summary=payload["summary"],
                preserved_facts=payload["preserved_facts"],
                context_version=1,
                algorithm_revision="semantic-compaction-v6",
            )
        seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": "Concurrent new evidence."})
        with pytest.raises(ContextSourceChangedError):
            manager.commit_compaction(candidate)
        assert not any(item.kind == "context_compaction" for item in store.list_items(turn.turn_id))
        assert store.verify().valid

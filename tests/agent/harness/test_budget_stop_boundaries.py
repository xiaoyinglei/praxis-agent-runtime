"""Budget facts, lifecycle and optional stop checks have distinct contracts."""

from dataclasses import fields

import pytest

from agent_runtime.harness import CompletionDecision, RuntimeStore, turn_ops
from agent_runtime.harness.stop import StopChecks
from agent_runtime.result import AgentResult
from tests.agent.harness.test_completion_gate import _commit_tool_result, _proposal


def test_public_results_and_stop_decisions_have_no_task_validation_score():
    for model in (AgentResult, CompletionDecision):
        names = {field.name for field in fields(model)}
        assert "validation_status" not in names
        assert "validation_scope" not in names


def test_background_command_does_not_block_a_natural_turn_end(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "run.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="start a service",
            binding_manifest={"completion_policy": {"revision": "completion-v2"}},
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="service",
            tool_name="run_command",
            arguments={"command": "service"},
            result={"session_id": "service", "status": "running"},
        )
        decision = StopChecks(store).evaluate(_proposal(store, turn.turn_id))
        assert decision.action == "accept"


def test_running_acceptance_command_cannot_satisfy_the_stop_condition(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "run.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="check",
            binding_manifest={"completion_policy": {"revision": "completion-v2", "acceptance_command": "check"}},
        )
        _commit_tool_result(
            store,
            turn_id=turn.turn_id,
            call_id="check",
            tool_name="run_command",
            arguments={"command": "check"},
            result={"session_id": "check", "status": "running", "exit_code": 0},
        )
        decision = StopChecks(store).evaluate(_proposal(store, turn.turn_id))
        assert decision.action == "continue"
        assert "check" in decision.reason


@pytest.mark.anyio
async def test_restored_accept_decision_rechecks_later_child_mutation(tmp_path):
    from agent_runtime.harness import HarnessModelResponse, completion_ops
    from agent_runtime.harness.events import commit_and_publish
    from agent_runtime.harness.protocol import model_response_payload
    from agent_runtime.harness.stop import TurnStop
    from tests.agent.harness.legacy_fixtures import record_migrated_context_item
    from tests.agent.harness.test_completion_policy_v2 import COMMAND, _change, _child, _command, _start

    database = tmp_path / "restore.db"
    async with await RuntimeStore.open(database) as store:
        parent = _start(store, tmp_path, command=COMMAND)
        child = _child(store, parent)
        _command(store, parent.turn_id)
        source = record_migrated_context_item(
            store,
            turn_id=parent.turn_id,
            kind="model_response",
            payload=model_response_payload(HarnessModelResponse.from_parts(text="done", provider_response_id=None)),
        )
        proposal = completion_ops.record_final_proposal(
            store,
            turn_id=parent.turn_id,
            answer="done",
            source_item_id=source.item_id,
        )
        old = completion_ops.record_completion_decision(
            store,
            turn_id=parent.turn_id,
            proposal_item_id=proposal.item_id,
            action="accept",
            reason="command succeeded before process loss",
        )
        _change(store, child.turn_id, tmp_path)
    async with await RuntimeStore.open(database) as store:

        async def commit(operation):
            return await commit_and_publish(store, operation, None)

        stop = TurnStop(store=store, gate=StopChecks(store), commit=commit)
        result = await stop.finish(
            thread_id=parent.thread_id,
            turn_id=parent.turn_id,
            answer="done",
            input_revision=store.user_input_revision(parent.turn_id),
            source_item_id=source.item_id,
        )
        assert result is None
        assert store.read_turn(parent.turn_id).status == "running"
        decisions = [i for i in store.list_items(parent.turn_id) if i.kind == "completion_decision"]
        assert [i.payload["action"] for i in decisions] == ["accept", "continue"]
        assert decisions[-1].payload["reason_code"] == "acceptance_command_stale"
        assert store.read_item(old.item_id).payload["action"] == "accept"
        assert store.verify().valid


def test_budget_projection_reports_ledger_changes_and_restores_after_compaction(tmp_path):
    from agent_runtime.harness import ModelMessage, RolloutContextManager, TurnRunner
    from agent_runtime.harness.control import ContextSnapshot
    from tests.agent.harness.test_budget_pressure import _settle_prior_usage
    from tests.agent.harness.test_runtime_efficiency import PlanningModel

    with RuntimeStore.open_sync(tmp_path / "budget.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        record = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="task",
            binding_manifest={"model_token_budget_total": 10_000, "model_step_budget": 8},
        )
        runner = TurnRunner(
            thread_id=thread.thread_id,
            store=store,
            model=PlanningModel(),
            context_manager=RolloutContextManager(store),
            completion_gate=StopChecks(store),
        )
        turn = runner.restore_turn_context(record.turn_id)
        initial = (ModelMessage(role="user", body="task"),)
        first, _ = runner._requests._append_runtime_state(turn, initial, step=1, clock={})
        same, update = runner._requests._append_runtime_state(turn, first, step=1, clock={})
        assert same == first and update is None
        _settle_prior_usage(store, record.turn_id, 6_000)
        changed, update = runner._requests._append_runtime_state(turn, first, step=2, clock={})
        assert changed[: len(first)] == first and update is not None
        state = ContextSnapshot.from_messages(changed).runtime_state
        facts = state["Resource limits (runtime ledger)"]
        assert facts["tokens_remaining"] == 4_000
        assert facts["tokens_available_for_ordinary_work"] == 3_000
        assert facts["protected_tokens_for_recovery_and_response"] == 1_000
        assert facts["agent_requests_remaining_including_this"] == 7
        assert "Current execution environment" not in update.runtime_metadata["state"]
        compacted, restored = runner._requests._append_runtime_state(turn, initial, step=2, clock={})
        assert restored is not None
        assert ContextSnapshot.from_messages(compacted).runtime_state == state
        assert not any(
            phrase in str(changed) for phrase in ("Prefer a final answer", "Stop broad exploration", "Converge")
        )


def test_context_capacity_pressure_does_not_invent_a_missing_resource_budget(tmp_path):
    from agent_runtime.harness import RolloutContextManager, TurnRunner
    from agent_runtime.harness.control import ContextSnapshot
    from tests.agent.harness.test_runtime_efficiency import PlanningModel

    with RuntimeStore.open_sync(tmp_path / "no-budget.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        record = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="task", binding_manifest={})
        runner = TurnRunner(
            thread_id=thread.thread_id,
            store=store,
            model=PlanningModel(),
            context_manager=RolloutContextManager(store),
            completion_gate=StopChecks(store),
        )
        step = runner.capture_step_context(runner.restore_turn_context(record.turn_id), step=1, budget_pressure=True)
        assert step.budget_pressure is True  # host admission control is separate
        state = ContextSnapshot.from_messages(step.messages).runtime_state
        assert "Resource limits (runtime ledger)" not in state
        assert "resource_reminder" not in state
        assert "running low" not in str(step.messages)


@pytest.mark.anyio
async def test_new_stop_decision_and_completion_roll_back_together(tmp_path, monkeypatch):
    from agent_runtime.harness import completion_ops
    from agent_runtime.harness.events import commit_and_publish
    from agent_runtime.harness.stop import TurnStop

    database = tmp_path / "atomic.db"
    async with await RuntimeStore.open(database) as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="task", binding_manifest={})

        async def commit(operation):
            return await commit_and_publish(store, operation, None)

        stop = TurnStop(store=store, gate=StopChecks(store), commit=commit)
        original = completion_ops.record_completion_decision

        def crash_after_decision(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("process stopped after decision")

        with monkeypatch.context() as patch:
            patch.setattr(completion_ops, "record_completion_decision", crash_after_decision)
            with pytest.raises(RuntimeError, match="process stopped after decision"):
                await stop.finish(thread_id=thread.thread_id, turn_id=turn.turn_id, answer="done", input_revision=1)
        assert not any(i.kind in {"completion_decision", "final_proposal"} for i in store.list_items(turn.turn_id))
    async with await RuntimeStore.open(database) as store:

        async def commit(operation):
            return await commit_and_publish(store, operation, None)

        stop = TurnStop(store=store, gate=StopChecks(store), commit=commit)
        result = await stop.finish(thread_id=thread.thread_id, turn_id=turn.turn_id, answer="done", input_revision=1)
        assert result.status == "completed"
        assert len([i for i in store.list_items(turn.turn_id) if i.kind == "completion_decision"]) == 1
        assert store.verify().valid

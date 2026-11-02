"""Runtime completion is a lifecycle event; retain evidence without a task score."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from agent_runtime import Agent
from agent_runtime.cli import _display_agent_result
from agent_runtime.harness import CompletionDecision, RuntimeStore, Session, TurnResult, completion_ops, facts, turn_ops
from agent_runtime.result import AgentResult, AgentUsage
from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel


@pytest.mark.anyio
async def test_false_answer_ends_without_a_task_validation_claim(tmp_path, monkeypatch):
    class FalseAnswer(PublicHarnessModel):
        async def dispatch(self, prepared, *, event_sink=None):
            from agent_runtime.harness import HarnessModelResponse

            return HarnessModelResponse.from_parts(text="The moon is made of cheese.", provider_response_id=None)

    database = tmp_path / "research.sqlite"
    agent = Agent(checkpoint_db=database, workspace_path=tmp_path, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", FalseAnswer)
    result = await agent.run("Research the composition of the moon")
    assert result.status == "done"
    assert result.answer == "The moon is made of cheese."
    assert result.tool_calls == ()
    assert "validation_status" not in asdict(result)
    assert "validation_scope" not in asdict(await agent.read_result(result.turn_id))
    async with await RuntimeStore.open(database) as store:
        [decision] = [item for item in store.list_items(result.turn_id) if item.kind == "completion_decision"]
        assert set(decision.payload) == {"proposal_item_id", "action", "reason"}
        assert store.verify().valid


@pytest.mark.parametrize("action", ["accept", "continue", "pause", "fail"])
def test_custom_stop_decision_only_controls_lifecycle(action):
    assert asdict(CompletionDecision(action, "specific condition")) == {
        "action": action,
        "reason": "specific condition",
        "reason_code": None,
    }


@pytest.mark.parametrize("legacy_status", ["passed", "failed", "invalid-old-metadata"])
def test_historical_validation_metadata_is_preserved_but_not_projected(tmp_path, legacy_status):
    with RuntimeStore.open_sync(tmp_path / "old.sqlite") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="old", binding_manifest={})
        proposal = completion_ops.record_final_proposal(store, turn_id=turn.turn_id, answer="old answer")

        def seed():
            with store.persistence.transaction():
                facts.append_fact(
                    store,
                    thread_id=thread.thread_id,
                    turn_id=turn.turn_id,
                    record_type="item_started",
                    producer="verifier",
                    payload={"item_id": "legacy", "kind": "completion_decision"},
                )
                facts.append_fact(
                    store,
                    thread_id=thread.thread_id,
                    turn_id=turn.turn_id,
                    record_type="item_completed",
                    producer="verifier",
                    payload={
                        "item_id": "legacy",
                        "payload": {
                            "proposal_item_id": proposal.item_id,
                            "action": "accept",
                            "reason": "legacy",
                            "validation_status": legacy_status,
                            "validation_scope": "workspace_change",
                        },
                    },
                )

        store.run_sync(seed)
        turn_ops.complete_turn(store, turn_id=turn.turn_id, answer="old answer")
        result = AgentResult._from_harness(TurnResult(thread.thread_id, turn.turn_id, "old answer"), store=store)
        assert result.status == "done"
        assert "validation_status" not in asdict(result)
        assert store.read_item("legacy").payload["validation_status"] == legacy_status
        store.rebuild_projections()
        assert store.read_item("legacy").payload["validation_status"] == legacy_status
        assert store.verify().valid


@pytest.mark.anyio
async def test_explicit_stop_failure_preserves_the_specific_reason(tmp_path):
    class RejectAnswer:
        def evaluate(self, proposal):
            return CompletionDecision("fail", "required local check failed", reason_code="custom_check_failed")

    async with await Session.open(
        database=tmp_path / "failed.sqlite",
        workspace=tmp_path,
        model=PublicHarnessModel(),
        completion_gate=RejectAnswer(),
    ) as session:
        result = await session.submit("answer")
        assert result.status == "failed" and result.answer is None
        assert result.stop_reason == "custom_check_failed"
        assert any(d.message == "required local check failed" for d in result.diagnostics)
        assert session.store.verify().valid


def _done_result():
    return AgentResult(
        answer="research answer",
        status="done",
        files=(),
        tool_calls=(),
        evidence=(),
        citations=(),
        usage=AgentUsage(),
        diagnostics=(),
        turn_id="turn",
        stop_reason="completed",
        pause=None,
        workspace_path=None,
        groundedness=False,
        insufficient_evidence=False,
        plan=None,
        plan_events=(),
    )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
def test_cli_has_no_generic_validation_receipt(capsys, streamed, verbose):
    _display_agent_result(_done_result(), verbose=verbose, answer_streamed=streamed)
    output = capsys.readouterr().out
    assert "验证状态" not in output and "not_evaluated" not in output
    assert ("research answer" in output) is not streamed

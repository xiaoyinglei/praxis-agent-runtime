from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path

import pytest

from agent_runtime import Agent
from agent_runtime.cli import _display_agent_result
from agent_runtime.harness import CompletionDecision, RolloutStore, Session, TurnResult
from agent_runtime.result import AgentResult, AgentUsage
from agent_runtime.terminal_app import _ConversationOutput
from agent_runtime.terminal_ui import Conversation
from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel


@pytest.mark.anyio
async def test_false_research_answer_ends_without_becoming_validated(tmp_path, monkeypatch):
    class FalseAnswer(PublicHarnessModel):
        async def dispatch(self, prepared, *, delta_sink: object = None):
            from agent_runtime.harness import HarnessModelResponse
            return HarnessModelResponse(
                text="The moon is made of cheese.",
                provider_response_id="false-answer",
                usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            )

    database = tmp_path / "research.sqlite"
    agent = Agent(checkpoint_db=database, workspace_path=tmp_path)
    monkeypatch.setattr(agent, "_model_factory", lambda: FalseAnswer())
    result = await agent.run("Research the composition of the moon", require_workspace_change=False)
    assert result.status == "done", (result.stop_reason, result.diagnostics)
    assert result.answer == "The moon is made of cheese."
    assert result.tool_calls == ()
    assert result.validation_status == "not_evaluated"
    assert result.validation_scope is None
    replayed = await agent.read_result(result.turn_id)
    assert replayed.validation_status == "not_evaluated"
    assert replayed.validation_scope is None
    with RolloutStore(database) as store:
        [decision] = [item for item in store.list_items(result.turn_id) if item.kind == "completion_decision"]
        assert decision.payload["validation_status"] == "not_evaluated"
        assert decision.payload["validation_scope"] is None
        assert store.verify().valid


@pytest.mark.parametrize("action", ["accept", "continue", "pause", "fail"])
def test_legacy_custom_gate_action_does_not_invent_validation(action):
    decision = CompletionDecision(action, "legacy gate decision")
    assert decision.validation_status == "not_evaluated"
    assert decision.validation_scope is None


@pytest.mark.parametrize("validation_status", ["not_evaluated", "passed", "failed"])
def test_canonical_decision_projects_explicit_validation(tmp_path: Path, validation_status):
    with RolloutStore(tmp_path / "projection.sqlite") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="answer", binding_manifest={})
        proposal = store.record_final_proposal(turn_id=turn.turn_id, answer="answer")
        store.record_completion_decision(
            turn_id=turn.turn_id, proposal_item_id=proposal.item_id,
            action="accept", reason="custom decision",
            validation_status=validation_status, validation_scope="custom_check",
        )
        store.complete_turn(turn_id=turn.turn_id, answer="answer")
        result = AgentResult._from_harness(
            TurnResult(thread_id=thread.thread_id, turn_id=turn.turn_id, answer="answer"), store=store,
        )
        assert result.validation_status == validation_status
        assert result.validation_scope == "custom_check"
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("legacy_decision", [False, True])
async def test_read_result_of_old_turn_without_validation_metadata_is_not_evaluated(tmp_path, legacy_decision):
    database = tmp_path / "old.sqlite"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="old", binding_manifest={})
        if legacy_decision:
            proposal = store.record_final_proposal(turn_id=turn.turn_id, answer="old answer")
            # Replay the historical canonical shape, before validation metadata existed.
            with store._transaction():
                store._append_and_reduce(
                    thread_id=thread.thread_id, turn_id=turn.turn_id,
                    record_type="item_started", producer="verifier",
                    payload={"item_id": "legacy-decision", "kind": "completion_decision"},
                )
                store._append_and_reduce(
                    thread_id=thread.thread_id, turn_id=turn.turn_id,
                    record_type="item_completed", producer="verifier",
                    payload={"item_id": "legacy-decision", "payload": {
                        "proposal_item_id": proposal.item_id, "action": "accept", "reason": "legacy",
                    }},
                )
        store.complete_turn(turn_id=turn.turn_id, answer="old answer")
        assert store.verify().valid
    agent = Agent(checkpoint_db=database, workspace_path=tmp_path)
    result = await agent.read_result(turn.turn_id)
    assert result.validation_status == "not_evaluated"
    assert result.validation_scope is None


@pytest.mark.anyio
async def test_explicit_failed_gate_preserves_failed_validation_without_an_answer(tmp_path):
    class RejectAnswer:
        def evaluate(self, proposal):
            return CompletionDecision("fail", "claim contradicted", "failed", "custom_acceptance")

    async with await Session.open(
        database=tmp_path / "failed.sqlite", workspace=tmp_path,
        model=PublicHarnessModel(), completion_gate=RejectAnswer(),
    ) as session:
        result = await session.submit("answer")
        assert result.status == "failed"
        assert result.answer is None
        assert result.validation_status == "failed"
        assert result.validation_scope == "custom_acceptance"
        [decision] = [item for item in session.store.list_items(result.turn_id) if item.kind == "completion_decision"]
        assert decision.payload["validation_status"] == "failed"
        assert session.store.verify().valid


def _done_result():
    return AgentResult(
        answer="research answer", status="done", files=(), tool_calls=(), evidence=(), citations=(),
        usage=AgentUsage(), diagnostics=(), turn_id="turn", stop_reason="completed", pause=None,
        workspace_path=None, groundedness=False, insufficient_evidence=False, plan=None, plan_events=(),
    )


def test_streamed_cli_shows_unevaluated_status_without_verbose(capsys):
    _display_agent_result(_done_result(), verbose=False, answer_streamed=True)
    assert "not_evaluated" in capsys.readouterr().out
    _display_agent_result(_done_result(), verbose=True, answer_streamed=True)
    assert "not_evaluated" in capsys.readouterr().out


def test_terminal_chat_retains_validation_when_execution_is_hidden():
    view = Conversation()
    with redirect_stdout(_ConversationOutput(view)):
        _display_agent_result(_done_result(), verbose=False, answer_streamed=True, show_execution=False)
    assert "not_evaluated" in view.plain_text()
    with redirect_stdout(_ConversationOutput(view)):
        _display_agent_result(_done_result(), verbose=True, answer_streamed=True, show_execution=False)
    assert "not_evaluated" in view.plain_text()


@pytest.mark.parametrize("validation_status", ["passed", "failed"])
@pytest.mark.parametrize("answer_streamed", [False, True])
def test_cli_always_names_explicit_validation_and_scope(capsys, validation_status, answer_streamed):
    from dataclasses import replace
    result = replace(_done_result(), validation_status=validation_status, validation_scope="workspace_change")
    _display_agent_result(result, verbose=False, answer_streamed=answer_streamed)
    output = capsys.readouterr().out
    assert validation_status in output
    assert "workspace_change" in output

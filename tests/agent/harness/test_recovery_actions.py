from __future__ import annotations

import pytest

from agent_runtime import Agent
from agent_runtime.harness import RuntimeState
from agent_runtime.harness.ownership import OWNERSHIP_REVISION, execution_owner


def _start(store, workspace):
    thread = store.create_thread(workspace=workspace)
    return store.start_turn(thread_id=thread.thread_id, user_message="work",
                            binding_manifest={"execution_ownership": OWNERSHIP_REVISION})


@pytest.mark.anyio
async def test_public_recovery_inspection_is_read_only_and_never_loads_model(tmp_path, monkeypatch):
    database = tmp_path / "run.sqlite"
    async with (await RuntimeState.open(database)) as store:
        turn = _start(store, tmp_path)
        before = store.list_records(turn.thread_id)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: pytest.fail("inspection constructed model"))
    state = await agent.recovery_state(turn.turn_id)
    assert state["actions"] == ["continue", "abort"]
    assert state["reason_code"] == "before_first_dispatch"
    assert turn.turn_id in state["evidence_ids"]
    async with (await RuntimeState.open(database)) as store:
        assert store.list_records(turn.thread_id) == before
        store.pause_turn(turn_id=turn.turn_id, reason="no recoverable continuation")
    state = await agent.recovery_state(turn.turn_id)
    assert state["actions"] == ["abort"]
    assert state["reason_code"] == "no_resume_point"


@pytest.mark.anyio
async def test_recovery_answers_one_identified_request_and_reports_required_input(tmp_path):
    database = tmp_path / "run.sqlite"
    async with (await RuntimeState.open(database)) as store:
        turn = _start(store, tmp_path)
        request = store.request_clarification(turn_id=turn.turn_id, question="Which output?")
    state = await Agent(workspace_path=tmp_path, checkpoint_db=database).recovery_state(turn.turn_id)
    assert state["actions"] == ["continue", "abort"]
    assert state["request_id"] == request.request_id
    assert state["requires_input"] is True
    assert state["reason_code"] == "clarification_required"
    assert request.request_id in state["evidence_ids"]


@pytest.mark.anyio
async def test_active_owner_does_not_advertise_recovery_actions(tmp_path):
    database = tmp_path / "run.sqlite"
    async with (await RuntimeState.open(database)) as store:
        turn = _start(store, tmp_path)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database)
    with execution_owner(database, turn.thread_id):
        state = await agent.recovery_state(turn.turn_id)
    assert state["actions"] == []
    assert state["reason_code"] == "execution_active"


@pytest.mark.anyio
async def test_legacy_dispatch_keeps_its_unexpired_lease(tmp_path):
    database = tmp_path / "run.sqlite"
    async with (await RuntimeState.open(database)) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="old", binding_manifest={})
        operation = store.prepare_model_operation(turn_id=turn.turn_id, request_hash="r", context_hash="c",
            tool_hash="t", wire_hash="w", request_ref={"input_revision": 1, "request_id": "old"})
        store.dispatch_model_attempt(operation.operation_id)
    state = await Agent(workspace_path=tmp_path, checkpoint_db=database).recovery_state(turn.turn_id)
    assert state["actions"] == []
    assert state["reason_code"] == "execution_lease_active"


@pytest.mark.anyio
async def test_legacy_running_turn_without_owner_contract_does_not_offer_abort(tmp_path):
    database = tmp_path / "run.sqlite"
    async with (await RuntimeState.open(database)) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="old", binding_manifest={})
    state = await Agent(workspace_path=tmp_path, checkpoint_db=database).recovery_state(turn.turn_id)
    assert state["actions"] == []


@pytest.mark.anyio
@pytest.mark.parametrize("choice", [False, True])
async def test_cancelled_pending_input_offers_only_abort(tmp_path, choice):
    database = tmp_path / "run.sqlite"
    async with (await RuntimeState.open(database)) as store:
        turn = _start(store, tmp_path)
        if choice:
            store.request_choice(turn_id=turn.turn_id, question="Output?", options=("csv", "xlsx"))
        else:
            store.request_clarification(turn_id=turn.turn_id, question="Output?")
        store.request_turn_cancellation(turn_id=turn.turn_id)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database)
    state = await agent.recovery_state(turn.turn_id)
    assert state["actions"] == ["abort"]
    assert state["reason_code"] == "cancellation_pending"
    cancelled = await agent.resume(turn.turn_id, "abort")
    assert cancelled.stop_reason == "cancelled"


@pytest.mark.anyio
async def test_unknown_tool_effect_is_not_approval_or_retry(tmp_path):
    database = tmp_path / "run.sqlite"
    async with (await RuntimeState.open(database)) as store:
        turn = _start(store, tmp_path)
        values = dict(turn_id=turn.turn_id, operation_id="effect", tool_call_id="call", tool_name="write_file",
                      arguments_digest="args", execution_revision="write-v1", idempotent=False,
                      attempt_count=0, error_code=None, requires_reconciliation=False)
        store.record_tool_execution_state(status="prepared", **values)
        store.record_tool_execution_state(status="ready", **values)
        claim = store.claim_tool_operation(operation_id="effect", worker_id="lost", lease_seconds=30)
        store.commit_tool_operation_result(
            turn_id=turn.turn_id, operation_id="effect", claim_generation=claim.claim_generation,
            fencing_token=claim.fencing_token, status="unknown", attempt_count=1,
            error_code="cancelled_outcome_unknown", requires_reconciliation=True,
            result={"tool_call_id": "call", "tool_name": "write_file", "is_error": True,
                    "content": [], "attachments": []},
        )
    state = await Agent(workspace_path=tmp_path, checkpoint_db=database).recovery_state(turn.turn_id)
    assert state["actions"] == []
    assert state["reason_code"] == "tool_outcome_unknown"
    assert "effect" in state["evidence_ids"]


@pytest.mark.anyio
async def test_public_approval_plan_matches_execution_and_rejects_stale_request(tmp_path, monkeypatch):
    from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

    target = tmp_path / "value.txt"
    target.write_text("before")
    database = tmp_path / "run.sqlite"
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", PatchThenAnswerModel)
    paused = await agent.run("change value.txt", require_workspace_change=False)
    state = await agent.recovery_state(paused.turn_id)
    assert "allow_once" in state["actions"] and "continue" not in state["actions"]
    assert state["request_id"] == paused.pause.request_id
    with pytest.raises(ValueError, match="request"):
        await agent.resume(paused.turn_id, "allow_once", expected_request_id="stale-request")
    assert target.read_text() == "before"
    done = await agent.resume(paused.turn_id, "allow_once", expected_request_id=state["request_id"])
    assert done.status == "done"
    assert target.read_text() == "after"
    assert (await agent.recovery_state(done.turn_id))["actions"] == []


@pytest.mark.anyio
async def test_cancelled_approval_cannot_execute_the_previously_offered_action(tmp_path, monkeypatch):
    from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

    target = tmp_path / "value.txt"
    target.write_text("before")
    database = tmp_path / "run.sqlite"
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", PatchThenAnswerModel)
    paused = await agent.run("change value.txt", require_workspace_change=False)
    async with (await RuntimeState.open(database)) as store:
        store.request_turn_cancellation(turn_id=paused.turn_id)
    assert (await agent.recovery_state(paused.turn_id))["actions"] == ["abort"]
    with pytest.raises(RuntimeError, match="cancellation_pending"):
        await agent.resume(paused.turn_id, "allow_once")
    assert target.read_text() == "before"
    assert (await agent.resume(paused.turn_id, "abort")).stop_reason == "cancelled"

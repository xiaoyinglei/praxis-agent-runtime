"""A Turn has one explicit optional step limit, independent of session defaults."""

import pytest

from agent_runtime import Agent
from agent_runtime.harness import HarnessModelResponse, ModelToolCall, RuntimeStore, Session
from agent_runtime.tools.integrations.subagent import SubagentInput
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel


class ReadThenAnswer(PatchThenAnswerModel):
    def __init__(self, reads):
        self.reads = reads
        self.calls = 0
        self.step_limits = []

    def prepare(self, request):
        self.step_limits.append(request.binding_manifest.get("model_step_budget"))
        return super().prepare(request)

    async def dispatch(self, prepared, **kwargs):
        self.calls += 1
        return HarnessModelResponse.from_parts(
            text="done" if self.calls > self.reads else "read",
            tool_calls=()
            if self.calls > self.reads
            else (ModelToolCall(id=f"read-{self.calls}", name="read_file", arguments={"path": "source.txt"}),),
            provider_response_id=None,
            usage={"input_tokens": 1, "output_tokens": 1},
        )


@pytest.mark.anyio
async def test_default_run_has_no_implicit_128_step_ceiling(tmp_path, monkeypatch):
    (tmp_path / "source.txt").write_text("evidence")
    model = ReadThenAnswer(2)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    result = await agent.run("read the source")
    assert result.status == "done"
    assert model.calls == 3
    async with await RuntimeStore.open(tmp_path / "run.db") as store:
        assert store.read_turn(result.turn_id).binding_manifest.get("model_step_budget") is None
        assert store.verify().valid


@pytest.mark.anyio
async def test_explicit_step_limit_is_still_enforced(tmp_path, monkeypatch):
    (tmp_path / "source.txt").write_text("evidence")
    model = ReadThenAnswer(5)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    result = await agent.run("read the source", max_steps=3)
    assert result.status == "failed" and result.stop_reason == "model_step_budget_exhausted"
    assert model.calls == 3


@pytest.mark.anyio
@pytest.mark.parametrize("frozen_open", [True, False])
async def test_resume_missing_step_limit_does_not_acquire_16(tmp_path, frozen_open):
    from agent_runtime.harness import turn_ops

    (tmp_path / "source.txt").write_text("evidence")
    database = tmp_path / "run.db"
    model = ReadThenAnswer(17)
    from tests.agent.harness.test_tool_orchestrator import _read_tool

    tools = {"read_file": _read_tool(workspace=tmp_path)}
    async with await Session.open(
        database=database, workspace=tmp_path, model=model, tools=tools, _background_memory=False
    ) as original:
        binding = dict(original.binding_provider.snapshot(thread_id=original.thread_id, turn_id="old-turn"))
        binding.pop("model_step_budget", None)
        turn = turn_ops.start_turn(
            original.store,
            thread_id=original.thread_id,
            turn_id="old-turn",
            user_message="read",
            binding_manifest=binding,
        )
    async with await Session.open(
        database=database,
        workspace=tmp_path,
        model=model,
        **({"frozen_turn_id": turn.turn_id} if frozen_open else {"thread_id": turn.thread_id}),
        max_steps=2,
        tools=tools,
        _background_memory=False,
    ) as session:
        assert session.max_steps == (None if frozen_open else 2)
        result = await session._executor().drive(turn_id=turn.turn_id, start_step=1)
        assert result.status == "completed" and model.calls == 18, session.store.read_turn(
            turn.turn_id
        ).terminal_message
        assert model.step_limits == [None] * 18


def test_explicit_child_limit_above_50_is_not_rejected_by_tool_schema():
    assert SubagentInput(task="bounded task", max_turns=200).max_turns == 200


@pytest.mark.anyio
async def test_live_session_limit_cannot_shrink_a_frozen_turn(tmp_path):
    from agent_runtime.harness import turn_ops
    from tests.agent.harness.test_tool_orchestrator import _read_tool

    (tmp_path / "source.txt").write_text("evidence")
    database = tmp_path / "frozen.db"
    model = ReadThenAnswer(3)
    tools = {"read_file": _read_tool(workspace=tmp_path)}
    async with await Session.open(
        database=database, workspace=tmp_path, model=model, tools=tools, max_steps=4, _background_memory=False
    ) as original:
        binding = original.binding_provider.snapshot(thread_id=original.thread_id, turn_id="frozen-turn")
        turn = turn_ops.start_turn(
            original.store,
            thread_id=original.thread_id,
            turn_id="frozen-turn",
            user_message="read",
            binding_manifest=binding,
        )
    async with await Session.open(
        database=database,
        workspace=tmp_path,
        model=model,
        tools=tools,
        thread_id=turn.thread_id,
        max_steps=2,
        _background_memory=False,
    ) as session:
        assert session.max_steps == 2
        result = await session._executor().drive(turn_id=turn.turn_id, start_step=1)
        assert result.status == "completed" and model.calls == 4
        assert session.store.read_turn(turn.turn_id).binding_manifest["model_step_budget"] == 4


@pytest.mark.parametrize(
    "parent_limit, child_limit, expected",
    [
        (None, None, None),
        (None, 200, 200),
        (200, None, 200),
        (200, 100, 100),
    ],
)
def test_child_step_limit_inherits_or_narrows_parent(tmp_path, parent_limit, child_limit, expected):
    from agent_runtime.harness import budget_ops, turn_ops

    with RuntimeStore.open_sync(tmp_path / "child.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        parent = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="delegate",
            binding_manifest={"model_step_budget": parent_limit},
        )
        child = budget_ops.start_budgeted_child_turn(
            store,
            parent_turn_id=parent.turn_id,
            child_thread_id="child-thread",
            child_turn_id="child-turn",
            user_message="work",
            binding_manifest={"model_step_budget": child_limit},
            requested_tokens=None,
            requested_cost_micros=None,
        )
        assert child.binding_manifest["model_step_budget"] == expected
        assert store.verify().valid


def test_child_cannot_expand_explicit_parent_step_limit(tmp_path):
    from agent_runtime.harness import budget_ops, turn_ops

    with RuntimeStore.open_sync(tmp_path / "child.db") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        parent = turn_ops.start_turn(
            store, thread_id=thread.thread_id, user_message="delegate", binding_manifest={"model_step_budget": 3}
        )
        before = store.list_records(thread.thread_id)
        with pytest.raises(ValueError, match="cannot exceed"):
            budget_ops.start_budgeted_child_turn(
                store,
                parent_turn_id=parent.turn_id,
                child_thread_id="child-thread",
                child_turn_id="child-turn",
                user_message="work",
                binding_manifest={"model_step_budget": 4},
                requested_tokens=None,
                requested_cost_micros=None,
            )
        assert store.list_records(thread.thread_id) == before
        assert len(store.list_turns()) == 1


def test_one_ordinary_step_allows_its_measured_multi_node_compaction(tmp_path):
    import asyncio

    from agent_runtime.harness import turn_ops
    from tests.agent.harness.test_compaction_consistency import long_history
    from tests.agent.harness.test_context_capacity_policy import capacity_runner

    with RuntimeStore.open_sync(tmp_path / "summary.db") as store:
        runner, original, _ = capacity_runner(store, tmp_path, window=8192)
        turn_ops.fail_turn(store, turn_id=original.turn_id, reason_code="fixture", message="fixture")
        turn = turn_ops.start_turn(
            store,
            thread_id=original.thread_id,
            user_message="summarize evidence",
            binding_manifest={**original.binding_manifest, "model_step_budget": 1},
        )
        long_history(store, turn)
        result = asyncio.run(runner.drive(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        operations = store.list_model_operations(turn.turn_id)
        summaries = [op for op in operations if op.request_ref["purpose"] == "context_summary"]
        assert len(summaries) > 1
        assert all(op.request_ref["compaction_plan"]["max_summary_calls"] is None for op in summaries)
        assert len([op for op in operations if op.request_ref["purpose"] == "agent_step"]) == 1
        assert (
            store.read_budget_state(turn.turn_id).used.total_tokens
            <= store.read_budget_state(turn.turn_id).limits.tokens
        )
        assert store.verify().valid


@pytest.mark.anyio
async def test_recovered_unlimited_turn_dispatches_step_129(tmp_path):
    from agent_runtime.harness import completion_ops, model_ops, turn_ops
    from agent_runtime.harness.execution_facts import response_handled
    from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel

    async with await Session.open(
        database=tmp_path / "late-step.db", workspace=tmp_path, model=PublicHarnessModel(), _background_memory=False
    ) as session:
        binding = session.binding_provider.snapshot(thread_id=session.thread_id, turn_id="late-turn")
        turn = turn_ops.start_turn(
            session.store,
            thread_id=session.thread_id,
            turn_id="late-turn",
            user_message="continue remaining work",
            binding_manifest=binding,
        )
        runner = session._executor()
        step = runner.capture_step_context(runner.restore_turn_context(turn.turn_id), step=128)
        prepared = await runner._model_runner.prepare(step)
        operation = model_ops.prepare_model_operation(
            session.store,
            turn_id=turn.turn_id,
            request_hash=prepared.request_hash,
            context_hash=prepared.context_hash,
            tool_hash=prepared.tool_hash,
            wire_hash=prepared.wire_hash,
            request_ref={**prepared.request_ref, "request_id": "late-turn:step:128"},
        )
        attempt = model_ops.dispatch_model_attempt(session.store, operation.operation_id)
        model_ops.complete_model_attempt(
            session.store,
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            response=HarnessModelResponse.from_parts(text="prior response", provider_response_id=None),
        )
        operation = next(
            op for op in session.store.list_model_operations(turn.turn_id) if op.operation_id == operation.operation_id
        )
        proposal = completion_ops.record_final_proposal(
            session.store,
            turn_id=turn.turn_id,
            answer="prior response",
            source_item_id=operation.response_item_id,
        )
        decision = completion_ops.record_completion_decision(
            session.store,
            turn_id=turn.turn_id,
            proposal_item_id=proposal.item_id,
            action="continue",
            reason="required work remains",
        )
        completion_ops.record_completion_feedback(
            session.store, turn_id=turn.turn_id, source_item_id=decision.item_id, reason="required work remains"
        )
        response_handled(
            session.store,
            turn_id=turn.turn_id,
            operation_id=operation.operation_id,
            response_item_id=operation.response_item_id,
            disposition="answer_or_continue",
        )
        result = await runner.drive(turn_id=turn.turn_id)
        assert result.status == "completed"
        operations = session.store.list_model_operations(turn.turn_id)
        assert [op.request_ref["step_snapshot"]["step"] for op in operations] == [128, 129]
        assert session.store.read_turn(turn.turn_id).binding_manifest["model_step_budget"] is None
        assert session.store.verify().valid

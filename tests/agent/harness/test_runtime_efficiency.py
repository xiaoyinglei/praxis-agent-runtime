"""Regressions from long runs, independent of task and model wording."""
from dataclasses import replace

import pytest

from agent_runtime import Agent
from agent_runtime.harness import (
    GatewayHarnessModel,
    HarnessMessage,
    HarnessModelRequest,
    HarnessModelResponse,
    HarnessToolCall,
)
from agent_runtime.modeling.openai_wire import serialize_openai_request
from tests.agent.harness.test_model_adapter import CapturingGateway, _resolved_model
from tests.agent.test_update_plan_surfaces import _UpdatePlanThenAnswerModel

POLICY = {"revision": "working-context-v1"}


def test_clock_changes_preserve_system_project_and_history_prefix():
    model = GatewayHarnessModel(model_id="test-model", resolved=_resolved_model(gateway=CapturingGateway()),
                                instructions=("Static instructions",))
    request = HarnessModelRequest(thread_id="thread", turn_id="turn", messages=(
        HarnessMessage(role="context", content="Project rules"),
        HarnessMessage(role="user", content="Implement a change"),
        HarnessMessage(role="assistant", content="Prior work"),
    ), binding_manifest={"execution_context_policy": POLICY, "runtime_clock": {"current_time": "2032-01-01"}})
    first = model.prepare(request).dispatch_payload.request
    second = model.prepare(replace(request, binding_manifest={**request.binding_manifest,
                                   "runtime_clock": {"current_time": "2032-01-02"}})).dispatch_payload.request
    assert first.messages[:-1] == second.messages[:-1]
    assert first.messages[-1] != second.messages[-1]
    assert "Project rules" in str(serialize_openai_request(first))
    assert "Current runtime clock facts" not in first.messages[0].content


class PlanningModel(_UpdatePlanThenAnswerModel):
    def __init__(self, reconcile=True):
        self.requests = []
        self.reconcile = reconcile

    def prepare(self, request):
        self.requests.append(request)
        return super().prepare(request)

    async def dispatch(self, prepared, *, delta_sink=None):
        step = prepared.request_ref["step"]
        if step in (1, 3) and (step == 1 or self.reconcile):
            return HarnessModelResponse(text="", provider_response_id=None, usage={}, tool_calls=(
                HarnessToolCall(id=f"plan-{step}", name="update_plan", arguments={"plan": [
                    {"step": "Inspect source", "status": "in_progress" if step == 1 else "completed"},
                ]}),
            ))
        if step == 2:
            return HarnessModelResponse(text="", provider_response_id=None, usage={}, tool_calls=(
                HarnessToolCall(id="inspect", name="read_file", arguments={"path": "source.txt"}),
            ))
        return HarnessModelResponse(text="Verified source.", provider_response_id=None, usage={})


@pytest.mark.anyio
@pytest.mark.parametrize("reconcile,steps,status", [(True, 4, "complete"), (False, 3, "needs_replan")])
async def test_current_plan_is_visible_without_gating_completion(tmp_path, reconcile, steps, status):
    (tmp_path / "source.txt").write_text("source")
    model = PlanningModel(reconcile)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run("Inspect source", max_steps=8)
    assert result.status == "done"
    assert len(model.requests) == steps
    assert result.plan.status == status
    assert any("Current advisory plan" in m.content and "Inspect source" in m.content
               for m in model.requests[2].messages)
    assert not any("Plan reconciliation" in m.content for r in model.requests for m in r.messages)


@pytest.mark.anyio
async def test_unfinished_advisory_plan_does_not_exhaust_last_step(tmp_path):
    (tmp_path / "source.txt").write_text("source")
    model = PlanningModel(reconcile=False)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run("Inspect source", max_steps=3)
    assert result.status == "done"
    assert result.plan.status == "needs_replan"
    assert len(model.requests) == 3


def test_working_context_compacts_before_provider_window(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutContextManager, RolloutStore, TurnExecutor
    from agent_runtime.modeling.contracts import ContextCapacityPolicy
    from tests.agent.harness.test_compaction_consistency import Accept, long_history
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway

    with RolloutStore(tmp_path / "rollout.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="Preserve the public API", binding_manifest={
            "context_capacity_policy": ContextCapacityPolicy().model_dump(),
            "execution_context_policy": POLICY, "auto_compact_token_limit": 18000,
        })
        long_history(store, turn)
        gateway = BudgetAwareCapturingGateway(max_input_tokens=100000)
        model = GatewayHarnessModel(model_id="test-model", resolved=_resolved_model(
            gateway=gateway, token_accounting=gateway.token_accounting, context_window_tokens=1048576,
        ), instructions=("Follow task",))
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=RolloutContextManager(store), completion_gate=Accept())
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        compactions = [i for i in store.list_items(turn.turn_id) if i.kind == "context_compaction"]
        assert compactions
        assert compactions[-1].payload["capacity_diagnostics"]["trigger_reason"] == "working_context_pressure"
        assert "Preserve the public API" in str(gateway.requests[-1].messages)
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("same_resource", [True, False])
async def test_permission_failure_identity_survives_tool_and_argument_changes(tmp_path, same_resource):
    from agent_runtime.permission_policy import PathRule, PermissionPolicy

    class Model(PlanningModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            step = prepared.request_ref["step"]
            if step == 5:
                return HarnessModelResponse(
                    text="Resource denied; no changes made.", provider_response_id=None, usage={},
                )
            path = f"protected/{'same' if same_resource else step}.py"
            if step % 2:
                call = HarnessToolCall(id=f"write-{step}", name="apply_patch", arguments={
                    "file_path": path, "new_string": f"value = {step}",
                })
            else:
                call = HarnessToolCall(id=f"write-{step}", name="execute_python", arguments={
                    "code": f"pass # attempt {step}", "working_dir": ".", "workspace_write": True,
                    "output_paths": [path],
                })
            return HarnessModelResponse(text="", provider_response_id=None, usage={}, tool_calls=(call,))

    model = Model()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False,
                  permissions=PermissionPolicy(filesystem=(PathRule(path="protected", access="read"),)))
    agent._model_factory = lambda: model
    result = await agent.run("Write a small module", max_steps=8, allow_write_tools=True, allow_execute_tools=True)
    if same_resource:
        assert result.stop_reason == "repeated_tool_failure"
        assert len(model.requests) == 4
        assert any("repeated_tool_failure_recovery" in m.content for m in model.requests[3].messages)
    else:
        assert result.status == "done"
        assert len(model.requests) == 5
    assert not (tmp_path / "protected").exists()


def test_working_target_keeps_large_protected_user_task_usable(tmp_path):
    import asyncio

    from agent_runtime.harness import RolloutContextManager, RolloutStore, TurnExecutor
    from agent_runtime.modeling.contracts import ContextCapacityPolicy
    from tests.agent.harness.test_compaction_consistency import Accept
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway

    with RolloutStore(tmp_path / "rollout.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="protected task " * 2000, binding_manifest={
            "context_capacity_policy": ContextCapacityPolicy().model_dump(),
            "execution_context_policy": POLICY, "auto_compact_token_limit": 1000,
        })
        gateway = BudgetAwareCapturingGateway(max_input_tokens=100000)
        model = GatewayHarnessModel(model_id="test-model", resolved=_resolved_model(
            gateway=gateway, token_accounting=gateway.token_accounting, context_window_tokens=1048576,
        ), instructions=("Follow task",))
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=RolloutContextManager(store), completion_gate=Accept())
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert len(gateway.requests) == 1
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))


def test_auto_compact_limit_is_frozen_and_old_definitions_keep_their_bytes():
    import json

    from agent_runtime.core.llm_config import AgentModelsConfig, ModelSpec
    from agent_runtime.model_definition import (
        ModelExecutionDefinition,
        build_model_execution_definition,
        canonical_definition_json,
    )

    config = AgentModelsConfig(default_model="test", auto_compact_token_limit=65536, models={
        "test": ModelSpec(
            provider="openai_compatible", base_url="https://example.com/v1", context_window_tokens=1048576,
        ),
    })
    definition = build_model_execution_definition(model_id="test", spec=config.models["test"], config=config)
    assert definition.auto_compact_token_limit == 65536
    old = json.loads(canonical_definition_json(definition))
    del old["auto_compact_token_limit"]
    encoded = json.dumps(old, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert canonical_definition_json(ModelExecutionDefinition.model_validate_json(encoded)) == encoded


def test_plan_tool_has_one_active_step_invariant():
    from agent_runtime.tools.builtins.planning import create_update_plan_tool

    with pytest.raises(ValueError, match="At most one"):
        create_update_plan_tool(lambda _: None).validate_input({"plan": [
            {"step": "first", "status": "in_progress"}, {"step": "second", "status": "in_progress"},
        ]})


@pytest.mark.parametrize("mandatory", [False, True])
def test_incomplete_summary_preserves_originals_and_only_blocks_required_compaction(tmp_path, mandatory):
    import asyncio

    from agent_runtime.harness import RolloutContextManager, RolloutStore, TurnExecutor
    from agent_runtime.modeling.contracts import ContextCapacityPolicy
    from tests.agent.harness.test_compaction_consistency import Accept, long_history
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway

    with RolloutStore(tmp_path / "rollout.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="Finish the task", binding_manifest={
            "context_capacity_policy": ContextCapacityPolicy().model_dump(),
            "execution_context_policy": POLICY, "auto_compact_token_limit": 18000,
        })
        long_history(store, turn)
        manager = RolloutContextManager(store)
        original = manager.build(turn.turn_id)
        gateway = BudgetAwareCapturingGateway(max_input_tokens=12000 if mandatory else 100000)
        model = GatewayHarnessModel(model_id="test-model", resolved=_resolved_model(
            gateway=gateway, token_accounting=gateway.token_accounting, context_window_tokens=1048576,
        ), instructions=("Follow task",))
        summary_calls = []

        async def dispatch(prepared, *, delta_sink=None):
            if prepared.request_ref["purpose"] == "context_summary":
                summary_calls.append(prepared.request_ref["request_id"])
                return HarnessModelResponse(text="INCOMPLETE_NOT_MEMORY", provider_response_id=None, usage={},
                                            status="incomplete", incomplete_reason="max_output_tokens")
            return HarnessModelResponse(text="Completed from original evidence", provider_response_id=None, usage={})

        model.dispatch = dispatch
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=manager, completion_gate=Accept())
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == ("failed" if mandatory else "completed")
        if mandatory:
            assert store.read_turn(turn.turn_id).terminal_reason_code == "context_summary_incomplete"
            assert "max_output_tokens" in store.read_turn(turn.turn_id).terminal_message
        assert summary_calls
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))
        assert all(m in manager.build(turn.turn_id) for m in original)
        assert not any("INCOMPLETE_NOT_MEMORY" in m.content for m in manager.build(turn.turn_id))
        assert not runner._proactive_compaction_available(turn.turn_id)
        assert store.verify().valid


def test_failed_optional_compaction_state_survives_restart_and_resets_with_window(tmp_path):
    import asyncio

    from agent_runtime.harness import ContextBudgetExceededError, RolloutContextManager, RolloutStore, TurnExecutor
    from tests.agent.harness.test_compaction_consistency import Accept, long_history, model_for, start

    database = tmp_path / "rollout.db"
    with RolloutStore(database) as store:
        thread, turn = start(store, tmp_path)
        long_history(store, turn)
        manager = RolloutContextManager(store)
        model, _ = model_for(100000)

        async def dispatch(prepared, *, delta_sink=None):
            return HarnessModelResponse(text="partial", provider_response_id=None, usage={},
                                        status="incomplete", incomplete_reason="max_output_tokens")

        model.dispatch = dispatch
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=manager, completion_gate=Accept())
        with pytest.raises(ContextBudgetExceededError, match="Incomplete context summary"):
            asyncio.run(runner._semantic_summary(runner.restore_turn_context(turn.turn_id), step=1,
                                                source=manager.semantic_source(turn.turn_id)[1]))
        assert store.read_turn(turn.turn_id).status == "running"
        assert not runner._proactive_compaction_available(turn.turn_id)
    with RolloutStore(database) as store:
        manager = RolloutContextManager(store)
        model, _ = model_for(100000)
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=manager, completion_gate=Accept())
        assert not runner._proactive_compaction_available(turn.turn_id)
        source_hash, _ = manager.semantic_source(turn.turn_id)
        manager.commit_compaction(manager.semantic_candidate(
            turn.turn_id, source_hash=source_hash, summary="Preserve public API; prior findings remain archived.",
        ))
        assert runner._proactive_compaction_available(turn.turn_id)
        assert store.verify().valid


@pytest.mark.parametrize("budget,final_only", [(40000, True), (100000, False)])
def test_last_affordable_round_reserves_a_final_answer_instead_of_more_tools(tmp_path, budget, final_only):
    import asyncio

    from agent_runtime.harness import RolloutContextManager, RolloutStore, TurnExecutor
    from agent_runtime.modeling.contracts import ContextCapacityPolicy
    from tests.agent.harness.test_compaction_consistency import Accept
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway
    from tests.agent.test_single_tool_executor import _tool

    class Router:
        def select(self, **kwargs):
            return (_tool(),)

    with RolloutStore(tmp_path / "rollout.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="task " * 4000, binding_manifest={
            "context_capacity_policy": ContextCapacityPolicy().model_dump(),
            "execution_context_policy": {**POLICY, "finalization": "budget-admission-v1"},
            "model_token_budget_total": budget,
        })
        gateway = BudgetAwareCapturingGateway(max_input_tokens=100000)
        model = GatewayHarnessModel(model_id="test-model", resolved=_resolved_model(
            gateway=gateway, token_accounting=gateway.token_accounting, context_window_tokens=1048576,
        ), instructions=("Follow task",))
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model, tool_router=Router(),
                              context_manager=RolloutContextManager(store), completion_gate=Accept())
        result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
        assert result.status == "completed"
        assert bool(gateway.requests[-1].tools) is (not final_only)
        operation = store.list_model_operations(turn.turn_id)[-1]
        assert operation.request_ref.get("budget_finalization", False) is final_only
        if final_only:
            restored = runner.restore_step_context(runner.restore_turn_context(turn.turn_id), operation)
            repeated = asyncio.run(runner._prepare_step(restored))
            assert restored.tools == ()
            assert repeated.wire_hash == operation.wire_hash
        assert store.read_budget_state(turn.turn_id).used.total_tokens <= budget
        assert store.verify().valid


@pytest.mark.anyio
async def test_visible_plan_round_trips_through_tool_schema_without_title_alias(tmp_path):
    import json

    from agent_runtime.tools.builtins.planning import UpdatePlanInput

    (tmp_path / "source.txt").write_text("source")
    model = PlanningModel()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run("Inspect source", max_steps=8)
    assert result.status == "done"
    visible = next(m.content for m in model.requests[1].messages if m.content.startswith("Current advisory plan"))
    payload = json.loads(visible.split("\n")[1])
    UpdatePlanInput.model_validate({"plan": payload["plan"]})
    assert payload["plan"][0]["step"] == "Inspect source"
    assert "title" not in visible
    assert result.plan.steps[0].title == "Inspect source"

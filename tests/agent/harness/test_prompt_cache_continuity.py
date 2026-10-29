"""Cache invariants at the durable Turn / provider boundary."""
import asyncio
from dataclasses import replace

import pytest

from agent_runtime.harness import GatewayHarnessModel, HarnessModelRequest, ModelMessage, RuntimeState
from agent_runtime.modeling.openai_wire import serialize_openai_request
from tests.agent.harness.test_model_adapter import CapturingGateway, _resolved_model
from tests.agent.harness.test_runtime_efficiency import PlanningModel
from tests.provider.test_openai_wire import _tool

CACHE_POLICY = {"revision": "working-context-v1", "cache_continuity": "append-v1"}


@pytest.mark.anyio
async def test_real_turn_appends_state_and_restores_identical_saved_wire(tmp_path):
    from agent_runtime import Agent

    (tmp_path / "source.txt").write_text("synthetic source")
    model = PlanningModel()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run("Inspect source", max_steps=8)
    assert result.status == "done"
    adapter = GatewayHarnessModel(model_id="test-model", resolved=_resolved_model(gateway=CapturingGateway()),
                                  instructions=("Static instructions",))
    wires = [adapter.prepare(r).dispatch_payload.request for r in model.requests]
    for before, after in zip(wires, wires[1:], strict=False):
        assert after.messages[:len(before.messages)] == before.messages
        assert after.tools == before.tools
    assert "Current advisory plan" in str(wires[-1].messages)
    # Rebuilding a frozen request uses its saved state, regardless of wall clock.
    assert adapter.prepare(model.requests[0]).wire_hash == adapter.prepare(model.requests[0]).wire_hash
    async with (await RuntimeState.open(tmp_path / "rollout.db")) as store:
        updates = [i.payload["request_ref"].get("runtime_context_update")
                   for i in store.list_items(result.turn_id) if i.kind == "model_request"]
        assert len(updates) == len(wires) and updates[0]
        assert "Current advisory plan" not in (updates[2] or "")  # unchanged plan is not copied every step
        assert store.verify().valid


@pytest.mark.parametrize("effort", ["off", "low", "high"])
def test_summary_fork_preserves_messages_tools_and_generation_mode(effort):
    from agent_runtime.core.generation_settings import ReasoningCapabilities
    from agent_runtime.modeling.contracts import LLMStageBudget

    gateway = CapturingGateway()
    gateway.agent_output_policy = "model-aware-v1"
    gateway.stage_budget = lambda stage: LLMStageBudget(max_input_tokens=100000, max_output_tokens=256)
    resolved = _resolved_model(gateway=gateway)
    resolved.capabilities = replace(resolved.capabilities, reasoning=ReasoningCapabilities(
        supported_efforts=("low", "high"), default_effort="high", thinking_toggle="thinking"))
    adapter = GatewayHarnessModel(model_id="test-model", resolved=resolved,
                                  instructions=("Static instructions",))
    parent = HarnessModelRequest(thread_id="thread", turn_id="turn", messages=(
        ModelMessage(role="context", body="Project rules"),
        ModelMessage(role="user", body="Synthetic task"),
        ModelMessage(role="assistant", body="Evidence"),
    ), tools=(_tool("zeta"), _tool("alpha")), binding_manifest={"execution_context_policy": CACHE_POLICY,
        "request_settings": {"preferences": {"reasoning_effort": effort}}})
    before = adapter.prepare(parent).dispatch_payload.request
    summary = replace(parent, purpose="context_summary", messages=(*parent.messages,
        ModelMessage(role="user", body="Summarize the history. Do not call tools.")),
        binding_manifest={**parent.binding_manifest, "summary_cache_fork": True})
    prepared = adapter.prepare(summary)
    after = prepared.dispatch_payload.request
    assert after.messages[:len(before.messages)] == before.messages
    assert after.tools == before.tools
    assert (serialize_openai_request(after).payload["tool_choice"]
            == serialize_openai_request(before).payload["tool_choice"])
    assert after.settings.provider_options == before.settings.provider_options
    assert adapter.prepare(summary).wire_hash == prepared.wire_hash
    asyncio.run(adapter.dispatch(prepared))


def test_unmarked_internal_requests_still_reject_tools():
    adapter = GatewayHarnessModel(model_id="test-model", resolved=_resolved_model(gateway=CapturingGateway()),
                                  instructions=("Static instructions",))
    with pytest.raises(ValueError, match="must not expose tools"):
        adapter.prepare(HarnessModelRequest(thread_id="thread", turn_id="turn", purpose="context_summary",
            messages=(ModelMessage(role="user", body="summary"),), tools=(_tool("alpha"),),
            binding_manifest={}))


@pytest.mark.anyio
async def test_summary_uses_saved_parent_and_falls_back_when_fork_is_too_large(tmp_path):
    from types import SimpleNamespace

    from agent_runtime import Agent
    from agent_runtime.harness import RolloutContextManager, TurnExecutor
    from tests.agent.harness.test_compaction_consistency import Accept
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway

    (tmp_path / "source.txt").write_text("synthetic source")
    captures = PlanningModel()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: captures
    result = await agent.run("Inspect source", max_steps=8)
    gateway = BudgetAwareCapturingGateway(max_input_tokens=100000)
    adapter = GatewayHarnessModel(model_id="test-model", resolved=_resolved_model(
        gateway=gateway, token_accounting=gateway.token_accounting), instructions=("Static instructions",))
    async with (await RuntimeState.open(tmp_path / "rollout.db")) as store:
        manager = RolloutContextManager(store)
        runner = TurnExecutor(thread_id=result.thread_id, store=store, model=adapter, context_manager=manager,
                              completion_gate=Accept())
        runner._tool_orchestrator = SimpleNamespace(restore_tools=lambda revisions: captures.requests[-1].tools)
        turn = runner.restore_turn_context(result.turn_id)
        digest, source = manager.semantic_source(result.turn_id, retained_tail_messages=0)
        plan = {"source_hash": digest, "summary_output_limit": 256, "summary_input_limit": 100000,
                "summary_byte_limit": 4096}
        step = runner._summary_step(turn, step=5, source=source, plan=plan)
        assert step.binding_manifest["summary_cache_fork"] is True
        before = adapter.prepare(captures.requests[-1]).dispatch_payload.request
        fork_prepared = await runner._prepare_step(step)
        after = fork_prepared.dispatch_payload.request
        assert after.messages[:len(before.messages)] == before.messages
        assert after.tools == before.tools
        assert "synthetic source" in str(after.messages)
        restored = replace(step, messages=step.messages, binding_manifest=dict(step.binding_manifest))
        assert (await runner._prepare_step(restored)).wire_hash == (await runner._prepare_step(step)).wire_hash
        # The isolated source can fit a limit where replaying the parent cannot.
        fallback_binding = dict(step.binding_manifest)
        fallback_prompt = fallback_binding.pop("summary_fork_fallback")
        fallback_binding.pop("summary_cache_fork")
        fallback = replace(step, binding_manifest=fallback_binding, tools=(),
                           messages=(ModelMessage(role="user", body=fallback_prompt),))
        fallback_size = (await runner._prepare_step(fallback)).resource_request.input_tokens
        assert fallback_size < fork_prepared.resource_request.input_tokens
        admitted = await runner._prepare_step(replace(step, input_token_limit=fallback_size))
        assert admitted.dispatch_payload.request.tools == ()
        assert not admitted.request_ref["step_snapshot"]["binding_manifest"].get("summary_cache_fork")
        small = replace(step, input_token_limit=1)
        with pytest.raises(Exception, match="input|context"):
            await runner._prepare_step(small)
        # Fragment/merge probes never drag a full parent through a summary tree.
        fragment = runner._summary_step(turn, step=5, source="{}", plan=plan)
        assert not fragment.tools and not fragment.binding_manifest.get("summary_cache_fork")


def test_declared_cache_parameters_reach_gateway_with_stable_routing_key():
    from agent_runtime.core.messages import ModelMessage
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway

    gateway = BudgetAwareCapturingGateway(max_input_tokens=100000)
    resolved = _resolved_model(gateway=gateway, token_accounting=gateway.token_accounting)
    resolved.capabilities = replace(resolved.capabilities, supported_cache_parameters=("prompt_cache_key",))
    adapter = GatewayHarnessModel(model_id="test-model", resolved=resolved, instructions=("Static instructions",))
    first = adapter.prepare(HarnessModelRequest(thread_id="thread", turn_id="turn", messages=(
        ModelMessage(role="user", body="Task"),), binding_manifest={"execution_context_policy": CACHE_POLICY}))
    request = first.dispatch_payload.request
    later = replace(request, messages=(*request.messages, ModelMessage(role="user", body="Update")),
                    prompt_revision="dynamic-revision")
    wire = serialize_openai_request(request)
    assert wire.payload["prompt_cache_key"] == serialize_openai_request(later).payload["prompt_cache_key"]
    assert wire.payload["prompt_cache_key"] != request.prompt_revision
    assert "prompt_cache_retention" not in wire.payload
    # Legacy bindings do not gain new provider parameters on replay.
    old = adapter.prepare(HarnessModelRequest(thread_id="thread", turn_id="turn", messages=(
        ModelMessage(role="user", body="Task"),), binding_manifest={}))
    assert "prompt_cache_key" not in serialize_openai_request(old.dispatch_payload.request).payload

    from agent_runtime.modeling.contracts import LLMCallStage
    from agent_runtime.modeling.gateway import LLMGateway
    from tests.agent.harness.test_model_adapter import CharacterAccounting, NonStreamingProvider

    class CaptureProvider(NonStreamingProvider):
        def generate_with_tools(self, **kwargs):
            self.kwargs = kwargs
            return super().generate_with_tools(**kwargs)

    provider = CaptureProvider()
    real_gateway = LLMGateway(generator=provider, token_accounting=CharacterAccounting(),
                              model_context_tokens=100000)
    response = asyncio.run(real_gateway.agenerate_model_request(
        stage=LLMCallStage.AGENT_STEP, request=request, provider="openai_compatible", supports_native_tools=True))
    assert provider.kwargs["prompt_cache_key"] == wire.payload["prompt_cache_key"]
    assert response.provider_wire_hash == first.wire_hash


@pytest.mark.anyio
@pytest.mark.parametrize("legacy", [False, True])
async def test_capacity_metadata_keeps_full_prefix_and_saved_replay(tmp_path, monkeypatch, legacy):
    from types import SimpleNamespace

    from agent_runtime import Agent
    from agent_runtime.core.messages import ModelToolCall, StopReason
    from agent_runtime.harness import RolloutContextManager, TurnExecutor
    from agent_runtime.harness.services import SessionBindingProvider
    from tests.agent.harness.test_compaction_consistency import Accept
    from tests.agent.harness.test_model_adapter import BoundGatewayHarnessModel, BudgetAwareCapturingGateway

    if legacy:
        original = SessionBindingProvider._static_snapshot

        def legacy_snapshot(self, turn_id):
            binding = original(self, turn_id)
            binding["execution_context_policy"]["revision"] = "working-context-v1"
            return binding

        monkeypatch.setattr(SessionBindingProvider, "_static_snapshot", legacy_snapshot)

    class ReadThenAnswer(BudgetAwareCapturingGateway):
        async def agenerate_model_request(self, **kwargs):
            response = await super().agenerate_model_request(**kwargs)
            if len(self.requests) == 1:
                from agent_runtime.core.messages import outputs_from_parts

                return replace(response, stop_reason=StopReason.TOOL_USE, raw_stop_reason="tool_calls",
                               outputs=outputs_from_parts(tool_calls=(
                                   ModelToolCall(id="read", name="read_file", arguments={"path": "source.txt"}),
                               )))
            return response

    (tmp_path / "source.txt").write_text("synthetic source")
    from agent_runtime.modeling.contracts import ContextCapacityPolicy

    gateway = ReadThenAnswer(max_input_tokens=100000)
    gateway.context_policy = ContextCapacityPolicy()
    class CapturingModel(BoundGatewayHarnessModel):
        def prepare(self, request):
            captured.append(request)
            return super().prepare(request)

    captured = []
    resolved = _resolved_model(gateway=gateway, token_accounting=gateway.token_accounting,
                               context_window_tokens=1048576)
    resolved.capabilities = replace(resolved.capabilities, supported_cache_parameters=("prompt_cache_key",))
    model = CapturingModel(model_id="test-model", resolved=resolved, instructions=("Read and answer",))
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run("Read source.txt", max_steps=4)
    assert result.status == "done", result.diagnostics
    before, after = gateway.requests
    assert after.messages[:len(before.messages)] == before.messages
    assert sum(m.content.startswith("Runtime input limits:\n") for m in after.messages) == int(legacy)
    if not legacy:
        assert "Runtime budget status" not in str(after.messages)
        assert "requests_remaining_including_this" not in str(after.messages)
        assert "effective_input_limit" not in str(after.messages)
    async with (await RuntimeState.open(tmp_path / "rollout.db")) as store:
        operations = store.list_model_operations(result.turn_id)
        assert bool(operations[0].request_ref.get("runtime_input_update")) == legacy
        assert "runtime_input_update" not in operations[1].request_ref
        for operation in operations:
            assert operation.request_ref["context_projection"]["effective_input_limit"] > 0
            if not legacy:
                assert "tokens_remaining" in operation.request_ref["budget_status"]
        runner = TurnExecutor(thread_id=result.thread_id, store=store, model=model,
                              context_manager=RolloutContextManager(store), completion_gate=Accept())
        runner._tool_orchestrator = SimpleNamespace(restore_tools=lambda revisions: captured[0].tools)
        def forbid_clock():
            raise AssertionError("recovery must not recapture time")
        monkeypatch.setattr("agent_runtime.harness.turn.capture_runtime_clock", forbid_clock)
        for operation in operations:
            restored = runner.restore_step_context(runner.restore_turn_context(result.turn_id), operation)
            prepared = await runner._prepare_step(restored)
            assert prepared.wire_hash == operation.wire_hash
            assert prepared.request_hash == operation.request_hash
            assert prepared.context_hash == operation.context_hash
            assert prepared.tool_hash == operation.tool_hash
        assert store.verify().valid


def test_yaml_cache_capability_is_frozen_and_legacy_definition_bytes_stay_unchanged(tmp_path):
    import json

    from agent_runtime.core.llm_registry import ModelRegistry
    from agent_runtime.model_definition import (
        ModelExecutionDefinition,
        build_model_execution_definition,
        canonical_definition_json,
    )

    path = tmp_path / "models.yaml"
    path.write_text("""models:
  example:
    capability: chat
    provider: openai_compatible
    base_url: https://example.com/v1
    context_window_tokens: 8192
    supported_cache_parameters: [prompt_cache_key]
defaults:
  primary_model: example
""")
    config = ModelRegistry._load_yaml_file(path)
    definition = build_model_execution_definition(model_id="example", spec=config.models["example"], config=config)
    assert definition.capabilities.supported_cache_parameters == ("prompt_cache_key",)
    archived = json.loads(canonical_definition_json(definition))
    archived.pop("supported_cache_parameters")
    old_bytes = json.dumps(archived, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert canonical_definition_json(ModelExecutionDefinition.model_validate_json(old_bytes)) == old_bytes


@pytest.mark.anyio
async def test_summary_rejects_provider_tool_calls_and_keeps_original_history(tmp_path):
    from agent_runtime.core.messages import ModelToolCall, StopReason
    from agent_runtime.harness import ContextBudgetExceededError, RolloutContextManager, TurnExecutor
    from agent_runtime.harness.turn import StepContext
    from tests.agent.harness.test_compaction_consistency import Accept

    class MisbehavingGateway(CapturingGateway):
        async def agenerate_model_request(self, **kwargs):
            response = await super().agenerate_model_request(**kwargs)
            from agent_runtime.core.messages import outputs_from_parts

            return replace(response, stop_reason=StopReason.TOOL_USE, raw_stop_reason="tool_calls",
                           outputs=outputs_from_parts(text=response.turn.text, tool_calls=(
                               ModelToolCall(id="forbidden", name="side_effect", arguments={}),
                           )))

    tool = _tool("side_effect")
    gateway = MisbehavingGateway()
    adapter = GatewayHarnessModel(model_id="test-model", resolved=_resolved_model(gateway=gateway),
                                  instructions=("Summarize only",))
    async with (await RuntimeState.open(tmp_path / "rollout.db")) as store:
        thread = store.create_thread(workspace=tmp_path)
        record = store.start_turn(thread_id=thread.thread_id, user_message="synthetic task",
                                  binding_manifest={"execution_context_policy": CACHE_POLICY})
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=adapter,
                              context_manager=RolloutContextManager(store), completion_gate=Accept())
        turn = runner.restore_turn_context(record.turn_id)
        fork = StepContext(turn=turn, step=1, input_revision=store.user_input_revision(record.turn_id),
            binding_manifest={**turn.binding_manifest, "summary_cache_fork": True},
            messages=(ModelMessage(role="user", body="Summarize only. Do not call tools."),),
            tools=(tool,), model_token_budget_remaining=None, purpose="context_summary", request_id="summary-test")
        runner._summary_step = lambda *args, **kwargs: fork
        with pytest.raises(ContextBudgetExceededError, match="contain no tool calls"):
            await runner._semantic_summary(turn, step=1, source="{}")
        assert not store.list_tool_operations(turn.turn_id)
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))
        assert RolloutContextManager(store).build(turn.turn_id)[0].content == "synthetic task"
        assert store.verify().valid


def test_new_turn_clears_previous_acceptance_contract(tmp_path):
    import json

    from agent_runtime.harness import RolloutContextManager, TurnExecutor
    from tests.agent.harness.test_compaction_consistency import Accept

    with RuntimeState.open_sync(tmp_path / "rollout.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        record = store.start_turn(thread_id=thread.thread_id, user_message="new task", binding_manifest={})
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=PlanningModel(),
                              context_manager=RolloutContextManager(store), completion_gate=Accept())
        marker = "Runtime state update (changed fields; latest value wins):\n"
        previous = ModelMessage(role="context", body=marker + json.dumps({
            "acceptance_contract": {"acceptance_command": "previous-turn-command"},
        }))
        _, update = runner._append_runtime_state(runner.restore_turn_context(record.turn_id),
            (ModelMessage(role="user", body="old task"), previous), step=1, pressure=False, clock={})
        assert json.loads(update[len(marker):].split("\n", 1)[0])["acceptance_contract"] is None

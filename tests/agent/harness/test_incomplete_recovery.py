from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from agent_runtime import Agent, cli
from agent_runtime.budget import ResourceUsage
from agent_runtime.harness import (
    HarnessModelResponse,
    HarnessToolCall,
    ModelDispatchCancelledError,
    RolloutEventReader,
    RolloutStore,
    TurnExecutor,
)
from agent_runtime.streaming.events import EventType, ItemStatus, TurnItemKind
from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel


class TruncatedThenAnswer(PublicHarnessModel):
    def __init__(self, *, repeat=False, reason="max_output_tokens", calls=()):
        self.requests = []
        self.repeat = repeat
        self.reason = reason
        self.calls = calls

    def prepare(self, request):
        self.requests.append(request)
        return replace(super().prepare(request), resource_request=ResourceUsage(
            input_tokens=8, output_tokens=32, model_calls=1,
        ))

    async def dispatch(self, prepared, **kwargs):
        if self.repeat or prepared.request_ref["step"] == 1:
            return HarnessModelResponse(
                text="", provider_response_id=None,
                usage={"input_tokens": 8, "output_tokens": 32, "reasoning_tokens": 32, "total_tokens": 40},
                status="incomplete", incomplete_reason=self.reason, tool_calls=self.calls,
            )
        return HarnessModelResponse(text="complete answer", provider_response_id=None,
                                    usage={"input_tokens": 8, "output_tokens": 3, "total_tokens": 11})


@pytest.mark.parametrize("entrypoint", ["sdk", "cli"])
def test_output_limit_restarts_complete_step_through_public_entrypoints(tmp_path, monkeypatch, entrypoint):
    model = TruncatedThenAnswer(calls=(HarnessToolCall(
        id="discarded", name="apply_patch", arguments={"patch": "must never run"},
    ),))
    database = tmp_path / "rollout.db"
    if entrypoint == "sdk":
        agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
        monkeypatch.setattr(agent, "_model_factory", lambda: model)
        result = asyncio.run(agent.run("Explain", max_steps=3, max_tokens_total=1000))
        assert result.status == "done"
        assert result.answer == "complete answer"
        assert result.usage.model_calls == 2
        assert result.usage.total_tokens == 51
        assert result.diagnostics[0].severity == "warning"
    else:
        original = cli._create_agent_facade
        def create(**kwargs):
            agent = original(**kwargs)
            monkeypatch.setattr(agent, "_model_factory", lambda: model)
            return agent
        monkeypatch.setattr(cli, "_create_agent_facade", create)
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli.agent_app, [
            "run", "Explain", "--max-steps", "3", "--disable-workspace-mcp",
            "--non-interactive", "--rollout-db", str(database),
        ])
        assert result.exit_code == 0, result.output
    with RolloutStore(database) as store:
        turn = store.list_turns()[-1]
        assert store.verify().valid
        assert not store.list_tool_operations(turn.turn_id)
        assert len(store.list_model_operations(turn.turn_id)) == 2
    request = model.requests[-1]
    assert not any(m.tool_calls for m in request.messages)
    assert any("output limit" in m.content for m in request.messages if m.role == "context")


@pytest.mark.parametrize("reason, repeat, expected", [
    ("max_output_tokens", True, 2), ("content_filter", False, 1), ("unknown", False, 1),
])
def test_retry_is_bounded_and_only_for_known_output_limit(tmp_path, monkeypatch, reason, repeat, expected):
    model = TruncatedThenAnswer(reason=reason, repeat=repeat)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    result = asyncio.run(agent.run("Explain", max_steps=10))
    assert result.status == "failed"
    assert result.answer is None
    assert result.usage.model_calls == expected
    with RolloutStore(tmp_path / "run.db") as store:
        assert store.read_turn(result.turn_id).terminal_reason_code == "model_response_incomplete"
        assert not store.list_tool_operations(result.turn_id)
        assert store.verify().valid


def test_retry_does_not_reset_budget(tmp_path, monkeypatch):
    model = TruncatedThenAnswer()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    result = asyncio.run(agent.run("Explain", max_steps=10, max_tokens_total=40))
    assert result.status == "failed"
    assert result.usage.model_calls == 1
    assert result.usage.total_tokens == 40


def test_retry_spends_an_ordinary_step(tmp_path, monkeypatch):
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", TruncatedThenAnswer)
    result = asyncio.run(agent.run("Explain", max_steps=1))
    assert result.status == "failed"
    assert result.usage.model_calls == 1
    with RolloutStore(tmp_path / "run.db") as store:
        assert store.read_turn(result.turn_id).terminal_reason_code == "model_step_budget_exhausted"


def test_recovery_policy_metadata_does_not_disable_retry(tmp_path, monkeypatch):
    start = RolloutStore.start_turn

    def with_metadata(store, **kwargs):
        kwargs["binding_manifest"]["model_response_recovery"]["note"] = "diagnostic metadata"
        return start(store, **kwargs)

    monkeypatch.setattr(RolloutStore, "start_turn", with_metadata)
    model = TruncatedThenAnswer()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    result = asyncio.run(agent.run("Explain", max_steps=3))
    assert result.status == "done"
    assert result.usage.model_calls == 2


def test_legacy_turn_cannot_acquire_retry_policy_on_public_recovery(tmp_path, monkeypatch):
    class Crash(BaseException):
        pass
    original_start = RolloutStore.start_turn
    original_handle = TurnExecutor._handle_model_response
    def legacy_start(self, **kwargs):
        kwargs["binding_manifest"] = dict(kwargs["binding_manifest"])
        kwargs["binding_manifest"].pop("model_response_recovery", None)
        return original_start(self, **kwargs)
    async def crash_handle(self, **kwargs):
        raise Crash()
    monkeypatch.setattr(RolloutStore, "start_turn", legacy_start)
    monkeypatch.setattr(TurnExecutor, "_handle_model_response", crash_handle)
    database = tmp_path / "run.db"
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", TruncatedThenAnswer)
    with pytest.raises(Crash):
        asyncio.run(agent.run("Explain", max_steps=5))
    monkeypatch.setattr(RolloutStore, "start_turn", original_start)
    monkeypatch.setattr(TurnExecutor, "_handle_model_response", original_handle)
    with RolloutStore(database) as store:
        turn = store.list_turns()[-1]
        assert "model_response_recovery" not in turn.binding_manifest
    fresh = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(fresh, "_model_factory", TruncatedThenAnswer)
    result = asyncio.run(fresh.resume(turn.turn_id, "continue"))
    assert result.status == "failed"
    assert result.usage.model_calls == 1


@pytest.mark.parametrize("boundary", ["response_committed", "retry_prepared"])
@pytest.mark.parametrize("repeat", [False, True])
def test_restart_between_commits_preserves_retry_limit_and_usage(tmp_path, monkeypatch, boundary, repeat):
    class Crash(BaseException):
        pass

    database = tmp_path / "run.db"
    model = TruncatedThenAnswer()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    original_handle = TurnExecutor._handle_model_response
    original_dispatch = TurnExecutor._dispatch_prepared
    async def crash_handle(self, **kwargs):
        if boundary == "response_committed":
            raise Crash()
        return await original_handle(self, **kwargs)
    async def crash_dispatch(self, **kwargs):
        if boundary == "retry_prepared" and kwargs["operation"].request_ref["step"] == 2:
            raise Crash()
        return await original_dispatch(self, **kwargs)
    monkeypatch.setattr(TurnExecutor, "_handle_model_response", crash_handle)
    monkeypatch.setattr(TurnExecutor, "_dispatch_prepared", crash_dispatch)
    with pytest.raises(Crash):
        asyncio.run(agent.run("Explain", max_steps=5, max_tokens_total=1000))
    monkeypatch.setattr(TurnExecutor, "_handle_model_response", original_handle)
    monkeypatch.setattr(TurnExecutor, "_dispatch_prepared", original_dispatch)
    with RolloutStore(database) as store:
        turn_id = store.list_turns()[-1].turn_id
        assert store.read_budget_state(turn_id).used.total_tokens == 40
    fresh = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    model = TruncatedThenAnswer(repeat=repeat)
    monkeypatch.setattr(fresh, "_model_factory", lambda: model)
    result = asyncio.run(fresh.resume(turn_id, "continue"))
    assert result.status == ("failed" if repeat else "done")
    assert result.usage.model_calls == 2
    assert result.usage.total_tokens == (80 if repeat else 51)
    with RolloutStore(database) as store:
        assert store.read_budget_state(turn_id).limits.tokens == 1000
        assert store.read_turn(turn_id).binding_manifest["model_response_recovery"]["max_retries"] == 1
        assert store.verify().valid


def test_cancellation_during_retry_does_not_start_a_third_request(tmp_path, monkeypatch):
    class CancelRetry(TruncatedThenAnswer):
        async def dispatch(self, prepared, **kwargs):
            if prepared.request_ref["step"] == 2:
                raise ModelDispatchCancelledError("cancelled")
            return await super().dispatch(prepared, **kwargs)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", CancelRetry)
    result = asyncio.run(agent.run("Explain", max_steps=5))
    assert result.status == "failed"  # Public facade maps cancellation to failed + stop_reason.
    assert result.stop_reason == "cancelled"
    with RolloutStore(tmp_path / "run.db") as store:
        assert store.read_turn(result.turn_id).status == "cancelled"
        assert len(store.list_model_operations(result.turn_id)) == 2
        assert store.verify().valid


def test_successful_side_effect_is_not_replayed_after_truncation(tmp_path, monkeypatch):
    from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

    class PatchTruncateAnswer(PatchThenAnswerModel):
        async def dispatch(self, prepared, **kwargs):
            if prepared.request_ref["request_id"].endswith(":step:2"):
                return HarnessModelResponse(
                    text="discarded partial answer", provider_response_id=None,
                    usage={"input_tokens": 3, "output_tokens": 20, "total_tokens": 23},
                    status="incomplete", incomplete_reason="max_output_tokens",
                    tool_calls=(HarnessToolCall(id="must-not-run", name="apply_patch", arguments={
                        "file_path": "value.txt", "old_string": "after", "new_string": "wrong",
                    }),),
                )
            return await super().dispatch(prepared, **kwargs)

    database = tmp_path / "run.db"
    (tmp_path / "value.txt").write_text("before")
    first = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(first, "_model_factory", PatchTruncateAnswer)
    paused = asyncio.run(first.run("Update value.txt", max_steps=4))
    assert paused.status == "paused"
    fresh = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(fresh, "_model_factory", PatchTruncateAnswer)
    result = asyncio.run(fresh.resume(paused.turn_id, "allow_once"))
    assert result.status == "done"
    assert result.answer != "discarded partial answer"
    assert (tmp_path / "value.txt").read_text() == "after"
    with RolloutStore(database) as store:
        assert len(store.list_tool_operations(result.turn_id)) == 1
        assert len(store.list_model_operations(result.turn_id)) == 3
        events = RolloutEventReader(store).read(thread_id=result.thread_id)
        incomplete = [event.event for event in events
                      if event.event.type == EventType.ITEM_COMPLETED
                      and event.event.item_kind == TurnItemKind.AGENT_MESSAGE
                      and event.event.data.get("response_status") == "incomplete"]
        assert len(incomplete) == 1
        assert incomplete[0].status == ItemStatus.FAILED
        assert store.verify().valid


def test_real_streaming_protocol_discards_prefix_and_partial_tool_json(tmp_path):
    from agent_runtime.harness import GatewayHarnessModel, RolloutContextManager
    from agent_runtime.modeling.contracts import LLMCallStage, LLMStageBudget, LLMUsage
    from agent_runtime.modeling.gateway import LLMGateway
    from tests.agent.harness.test_budget_pressure import AcceptAnswer
    from tests.agent.harness.test_model_adapter import CharacterAccounting, _resolved_model

    class Generator:
        def __init__(self):
            self.requests = []
        def stream_with_tools(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                yield {"type": "text_delta", "content": "discarded prefix"}
                yield {"type": "tool_use_start", "tool_id": "frag", "tool_name": "write_file"}
                yield {"type": "tool_input_delta", "tool_id": "frag", "content": '{"path":'}
                yield {"type": "content_block_stop", "tool_id": "frag"}
                reason = "max_tokens"
            else:
                yield {"type": "text_delta", "content": "complete answer"}
                reason = "end_turn"
            yield {"type": "message_stop", "stop_reason": reason,
                   "usage": LLMUsage(input_tokens=10, output_tokens=20, total_tokens=30, source="provider")}

    generator = Generator()
    gateway = LLMGateway(generator=generator, token_accounting=CharacterAccounting(),
                         model_context_tokens=30_000, stage_budgets={
        LLMCallStage.AGENT_STEP: LLMStageBudget(
            max_input_tokens=20_000, max_output_tokens=128, model_context_tokens=30_000, safety_margin_tokens=0,
        ),
    })
    model = GatewayHarnessModel(model_id="test-model", resolved=_resolved_model(
        gateway=gateway, max_output_tokens=128,
    ), instructions=("Answer the user.",))
    with RolloutStore(tmp_path / "run.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        runner = TurnExecutor(thread_id=thread.thread_id, store=store, model=model,
                              context_manager=RolloutContextManager(store), completion_gate=AcceptAnswer())
        result = asyncio.run(runner.run(turn_id="stream-retry", user_message="Explain", binding_manifest={
            "model_id": "test-model", "model_step_budget": 4, "model_token_budget_total": 100_000,
            "model_response_recovery": {"revision": "restart-step-v1", "max_retries": 1},
        }))
        assert result.status == "completed"
        assert result.answer == "complete answer"
        assert store.read_budget_state(result.turn_id).used.total_tokens == 60
        assert len(generator.requests) == 2
        assert "discarded prefix" not in str(generator.requests[1]["messages"])
        assert "tool_calls" not in str(generator.requests[1]["messages"])
        assert not store.list_tool_operations(result.turn_id)
        completed = [e.event for e in RolloutEventReader(store).read(thread.thread_id)
                     if e.event.type == EventType.ITEM_COMPLETED
                     and e.event.item_kind == TurnItemKind.AGENT_MESSAGE]
        assert completed[0].status == ItemStatus.FAILED
        assert completed[-1].status == ItemStatus.SUCCESS
        assert store.verify().valid

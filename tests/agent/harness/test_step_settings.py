from __future__ import annotations

import hashlib
import json
from contextlib import AsyncExitStack
from dataclasses import asdict

import pytest

from agent_runtime.harness import CompletionDecision, HarnessModelResponse, PreparedModelCall, Session


class ChangingModel:
    def __init__(self):
        self.selected = "first"
        self.requests = []
        self.unknown_once = False

    def snapshot(self, *, thread_id, turn_id):
        return {"model_id": self.selected, "thread_id": thread_id, "turn_id": turn_id}

    def ensure_available(self, binding, *, thread_id, turn_id):
        assert binding["turn_id"] == turn_id

    def prepare(self, request):
        self.requests.append(request)
        digest = hashlib.sha256(
            json.dumps([dict(request.binding_manifest), [asdict(m) for m in request.messages]], sort_keys=True).encode()
        ).hexdigest()
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash=digest,
            wire_hash=digest,
            request_ref={"model_id": request.binding_manifest["model_id"]},
        )

    async def dispatch(self, prepared):
        self.selected = "second"
        if self.unknown_once:
            self.unknown_once = False
            raise ConnectionError("outcome unknown")
        return HarnessModelResponse(text="answer", provider_response_id=None, usage={})


class ContinueOnce:
    def __init__(self):
        self.calls = 0

    def evaluate(self, proposal):
        self.calls += 1
        return CompletionDecision(action="continue" if self.calls == 1 else "accept", reason="test")


def make_control_plane(tmp_path, model_ids=("first", "second")):
    from agent_runtime.core.llm_config import AgentModelsConfig
    from agent_runtime.core.llm_registry import ModelRegistry
    from agent_runtime.model_trust import ModelBindingTrustDomain, TrustedModelDefinitionArchive
    from agent_runtime.models import ModelControlPlane

    registry = ModelRegistry(
        AgentModelsConfig.model_validate(
            {
                "default_model": model_ids[0],
                "models": {
                    name: {
                        "provider": "openai_compatible",
                        "location": "local",
                        "base_url": "http://localhost:1/v1",
                        "context_window_tokens": 8192,
                    }
                    for name in model_ids
                },
            }
        )
    )
    trust_path = tmp_path.parent / (tmp_path.name + "-trust")
    trust = ModelBindingTrustDomain(trust_path / "trust.json", workspace=tmp_path, worktree=tmp_path)
    archive = TrustedModelDefinitionArchive(trust_path / "definitions", workspace=tmp_path, worktree=tmp_path)
    plane = ModelControlPlane.from_registry(registry, trust_domain=trust, definition_archive=archive, session_path=None)
    trust.initialize()
    return plane


def test_child_binding_keeps_source_definition_and_authenticates_new_identity(tmp_path):
    from agent_runtime.harness.model_adapter import ControlPlaneHarnessModel
    from agent_runtime.model_trust import BindingAuthenticationError

    plane = make_control_plane(tmp_path)
    try:
        model = ControlPlaneHarnessModel(control_plane=plane, instructions=("test",))
        source = model.snapshot(thread_id="parent", turn_id="parent-turn")
        plane.switch_model("second", requested_by="user", persist=False)
        child = model.rebind(source, thread_id="child", turn_id="child-turn")
        assert child["model_id"] == "first"
        assert child["binding"] == source["binding"]
        assert child["signature"] != source["signature"]
        assert plane.model_spec_for_frozen_binding(child, thread_id="child", turn_id="child-turn").id == "first"
        with pytest.raises(BindingAuthenticationError):
            plane.model_spec_for_frozen_binding(child, thread_id="parent", turn_id="parent-turn")
    finally:
        plane.close()


@pytest.mark.anyio
async def test_saved_removed_model_does_not_block_session_open_for_tool_recovery(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

    target = tmp_path / "value.txt"
    target.write_text("before")
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "db", enable_workspace_mcp=False)
    plane = make_control_plane(tmp_path)
    monkeypatch.setattr(agent, "_get_model_control_plane", lambda: plane)
    monkeypatch.setattr(agent, "_harness_model", PatchThenAnswerModel)
    bootstraps = []

    async def ready(spec):
        bootstraps.append(spec.id)

    monkeypatch.setattr("agent_runtime.local_runtime.ensure_local_provider_ready", ready)
    options = {"agent": agent, "require_workspace_change": False}
    async with await Session.open(**options) as session:
        paused = await session.submit("write")
        assert paused.status == "paused"
    plane = make_control_plane(tmp_path, model_ids=("second",))
    bootstraps.clear()
    async with await Session.open(**options, frozen_turn_id=paused.turn_id) as reopened:
        assert bootstraps == []
        reopened.switch_model("second")
        result = await reopened.resume(paused.turn_id, "approve")
        assert result.status == "done"
        assert target.read_text() == "after"


@pytest.mark.anyio
async def test_live_session_model_switch_captures_authenticated_step_bindings(tmp_path, monkeypatch):
    from agent_runtime import Agent

    plane = make_control_plane(tmp_path)
    model = ChangingModel()
    model.snapshot = plane.freeze_model_binding
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_get_model_control_plane", lambda: plane)
    monkeypatch.setattr(agent, "_harness_model", lambda: model)

    async def ready(spec):
        pass

    monkeypatch.setattr("agent_runtime.local_runtime.ensure_local_provider_ready", ready)
    async with await Session.open(agent=agent, completion_gate=ContinueOnce()) as session:
        original_dispatch = model.dispatch

        async def dispatch(prepared):
            session.switch_model("second")
            return await original_dispatch(prepared)

        model.dispatch = dispatch
        result = await session.submit("task")
        assert result.status == "done"
        assert [request.binding_manifest["model_id"] for request in model.requests] == ["first", "second"]
        assert all("signature" in request.binding_manifest for request in model.requests)
        assert session.store.read_thread(session.thread_id).settings["model_id"] == "second"
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_next_step_captures_current_model_without_rewriting_old_request(tmp_path):
    model = ChangingModel()
    async with await Session.open(
        database=tmp_path / "db", workspace=tmp_path, model=model, completion_gate=ContinueOnce()
    ) as session:
        result = await session.submit("task")
        assert result.status == "done"
        assert [request.binding_manifest["model_id"] for request in model.requests] == ["first", "second"]
        operations = session.store.list_model_operations(result.turn_id)
        assert [op.request_ref["step_snapshot"]["binding_manifest"]["model_id"] for op in operations] == [
            "first",
            "second",
        ]


@pytest.mark.anyio
@pytest.mark.parametrize("restart", [False, True])
async def test_unknown_request_restores_snapshot_then_next_step_captures_current_settings(tmp_path, restart):
    model = ChangingModel()
    model.unknown_once = True
    options = dict(database=tmp_path / "db", workspace=tmp_path, model=model, completion_gate=ContinueOnce())
    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(await Session.open(**options))
        paused = await session.submit("task")
        assert paused.status == "paused"
        assert session.store.list_model_operations(paused.turn_id)[0].request_ref.get("step_snapshot")
        if restart:
            await stack.aclose()
            session = await stack.enter_async_context(await Session.open(**options, thread_id=paused.thread_id))
        result = await session.resume(paused.turn_id, "retry")
        assert result.status == "done"
        assert [request.binding_manifest["model_id"] for request in model.requests] == ["first", "first", "second"]


@pytest.mark.anyio
async def test_policy_update_during_request_applies_to_next_step_and_survives_reopen(tmp_path):
    model = ChangingModel()
    async with await Session.open(
        database=tmp_path / "db", workspace=tmp_path, model=model, completion_gate=ContinueOnce()
    ) as session:
        original_dispatch = model.dispatch

        async def dispatch(prepared):
            session.update_tool_policy(allow_write_tools=True)
            return await original_dispatch(prepared)

        model.dispatch = dispatch
        result = await session.submit("task")
        assert result.status == "done"
        assert [
            request.binding_manifest["tool_execution_policy"]["allow_write_tools"] for request in model.requests
        ] == [False, True]
        thread_id = session.thread_id
        assert session.store.verify().valid
    async with await Session.open(
        database=tmp_path / "db", workspace=tmp_path, model=ChangingModel(), thread_id=thread_id
    ) as reopened:
        assert reopened.tool_execution_context.allow_write_tools is True


@pytest.mark.anyio
async def test_unrelated_policy_change_does_not_invalidate_pending_write(tmp_path):
    from tests.agent.harness.test_approval_resume import AcceptAnswer, WriteThenAnswerModel, _write_tool

    calls = []
    async with await Session.open(
        database=tmp_path / "db",
        workspace=tmp_path,
        model=WriteThenAnswerModel(),
        completion_gate=AcceptAnswer(),
        tools={"write_file": _write_tool(tmp_path, calls)},
    ) as session:
        paused = await session.submit("write")
        session.update_tool_policy(denied_tool_names=frozenset({"unrelated_tool"}))
        result = await session.resume(paused.turn_id, "approve")
        assert result.status == "done"
        assert calls == ["approved.txt"]


@pytest.mark.anyio
async def test_policy_change_while_sampling_does_not_rewrite_that_requests_tools(tmp_path):
    from tests.agent.harness.test_approval_resume import AcceptAnswer, WriteThenAnswerModel, _write_tool

    calls = []
    model = WriteThenAnswerModel()
    async with await Session.open(
        database=tmp_path / "db",
        workspace=tmp_path,
        model=model,
        completion_gate=AcceptAnswer(),
        tools={"write_file": _write_tool(tmp_path, calls)},
    ) as session:
        original_dispatch = model.dispatch

        async def dispatch(prepared):
            session.update_tool_policy(allow_write_tools=True)
            return await original_dispatch(prepared)

        model.dispatch = dispatch
        paused = await session.submit("write")
        assert paused.status == "paused"
        assert calls == []


@pytest.mark.anyio
async def test_committed_tool_response_rejects_changed_runner_before_io(tmp_path, monkeypatch):
    from dataclasses import replace

    from agent_runtime.harness import TurnExecutor
    from tests.agent.harness.test_approval_resume import AcceptAnswer, WriteThenAnswerModel, _write_tool

    calls = []
    tool = _write_tool(tmp_path, calls)

    async def crash(*args, **kwargs):
        raise RuntimeError("crash after model response commit")

    with monkeypatch.context() as patch:
        patch.setattr(TurnExecutor, "_handle_model_response", crash)
        async with await Session.open(
            database=tmp_path / "db",
            workspace=tmp_path,
            model=WriteThenAnswerModel(),
            completion_gate=AcceptAnswer(),
            tools={"write_file": tool},
        ) as session:
            with pytest.raises(RuntimeError, match="crash after"):
                await session.submit("write")
            turn_id = session.head_turn_id
            thread_id = session.thread_id
    async with await Session.open(
        database=tmp_path / "db",
        workspace=tmp_path,
        model=WriteThenAnswerModel(),
        thread_id=thread_id,
        completion_gate=AcceptAnswer(),
        tools={"write_file": replace(tool, execution_revision="changed")},
    ) as reopened:
        with pytest.raises(RuntimeError, match="original operation tool is unavailable"):
            await reopened.resume(turn_id, "continue")
        assert calls == []


@pytest.mark.anyio
async def test_child_inherits_originating_step_while_parent_next_step_uses_updates(tmp_path):
    from tests.agent.harness.test_subagent_path import ParentChildModel

    class Model(ParentChildModel):
        selected = "first"

        def snapshot(self, *, thread_id, turn_id):
            return {**super().snapshot(thread_id=thread_id, turn_id=turn_id), "model_id": self.selected}

        async def dispatch(self, prepared):
            request = prepared.dispatch_payload
            if request.step == 2 and not request.messages[0].content.startswith("child task"):
                self.selected = "second"
                session.update_tool_policy(allow_write_tools=False)
            return await super().dispatch(prepared)

    model = Model()
    async with await Session.open(
        database=tmp_path / "db",
        workspace=tmp_path,
        model=model,
        enable_subagents=True,
        max_tokens_total=500,
    ) as session:
        session.update_tool_policy(allow_write_tools=True)
        result = await session.submit("delegate")
        if result.status == "paused":
            result = await session.resume(result.turn_id, "approve")
        assert result.status == "done"
        assert [
            (
                request.binding_manifest["model_id"],
                request.binding_manifest["tool_execution_policy"]["allow_write_tools"],
            )
            for request in model.requests
        ] == [("first", True), ("first", True), ("first", True), ("second", False)]

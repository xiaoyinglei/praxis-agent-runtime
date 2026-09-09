from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from agent_runtime import Agent
from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel


def test_frozen_model_resolution_reuses_provider_and_closes_it_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from agent_runtime.models import ModelControlPlane

    plane = ModelControlPlane.from_env(workspace=tmp_path, session_path=None)
    plane._trust_domain.initialize()
    resolved = []
    closed = []

    def resolve(definition):
        value = SimpleNamespace(generator=SimpleNamespace(close=lambda: closed.append(definition.definition_revision)))
        resolved.append(value)
        return value

    monkeypatch.setattr(plane._registry, "resolve_definition", resolve)
    monkeypatch.setattr(plane, "_ensure_model_credentials", lambda spec: None)
    first = plane.freeze_model_binding(thread_id="thread-1", turn_id="turn-1")
    second = plane.freeze_model_binding(thread_id="thread-1", turn_id="turn-2")
    one = plane.resolve_frozen_binding(first, thread_id="thread-1", turn_id="turn-1")
    two = plane.resolve_frozen_binding(second, thread_id="thread-1", turn_id="turn-2")
    assert one is two, "Session model control must reuse the physical provider across Turns"
    from agent_runtime.model_trust import BindingAuthenticationError

    with pytest.raises(BindingAuthenticationError):
        plane.resolve_frozen_binding(first, thread_id="thread-1", turn_id="turn-2")
    assert len(resolved) == 1
    plane.close()
    plane.close()
    assert len(closed) == 1


def test_reused_provider_refreshes_when_credential_rotates(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from agent_runtime.core.llm_config import AgentModelsConfig
    from agent_runtime.core.llm_registry import ModelRegistry
    from agent_runtime.model_trust import ModelBindingTrustDomain, TrustedModelDefinitionArchive
    from agent_runtime.models import ModelControlPlane

    monkeypatch.setenv(
        "AGENT_MODELS",
        json.dumps(
            {
                "default_model": "cloud",
                "models": {
                    "cloud": {
                        "provider": "openai_compatible",
                        "base_url": "https://api.example.com/v1",
                        "api_key_env": "SESSION_TEST_KEY",
                        "location": "cloud",
                        "context_window_tokens": 8192,
                    }
                },
            }
        ),
    )
    monkeypatch.setenv("SESSION_TEST_KEY", "first-test-key")
    trust = ModelBindingTrustDomain(
        tmp_path.parent / (tmp_path.name + "-trust") / "trust.json", workspace=tmp_path, worktree=tmp_path
    )
    archive = TrustedModelDefinitionArchive(
        tmp_path.parent / (tmp_path.name + "-trust") / "definitions", workspace=tmp_path, worktree=tmp_path
    )
    registry = ModelRegistry(AgentModelsConfig.model_validate(json.loads(os.environ["AGENT_MODELS"])))
    plane = ModelControlPlane.from_registry(registry, trust_domain=trust, definition_archive=archive, session_path=None)
    plane._trust_domain.initialize()
    created = []
    closed = []

    def resolve(definition):
        assert definition.api_key_env == "SESSION_TEST_KEY"
        key = os.environ[definition.api_key_env]
        created.append(key)
        return SimpleNamespace(generator=SimpleNamespace(close=lambda: closed.append(key)))

    monkeypatch.setattr(plane._registry, "resolve_definition", resolve)
    binding = plane.freeze_model_binding(thread_id="thread-1", turn_id="turn-1")
    first = plane.resolve_frozen_binding(binding, thread_id="thread-1", turn_id="turn-1")
    monkeypatch.setenv("SESSION_TEST_KEY", "rotated-test-key")
    second = plane.resolve_frozen_binding(binding, thread_id="thread-1", turn_id="turn-1")
    assert first is not second
    assert plane.resolve_frozen_binding(binding, thread_id="thread-1", turn_id="turn-1") is second
    assert created == ["first-test-key", "rotated-test-key"]
    assert closed == []  # A child may still be using the earlier client.
    plane.close()
    assert set(closed) == set(created)


@pytest.mark.anyio
async def test_conversation_reuses_resources_until_session_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    models = []

    def make_model():
        model = PublicHarnessModel()
        models.append(model)
        return model

    monkeypatch.setattr(agent, "_harness_model", make_model)
    assert callable(getattr(agent, "session", None)), "Agent must create a conversation-scoped Session"
    async with agent.session(require_workspace_change=False) as session:
        resources = (session.store, session.model, session.tool_orchestrator, session.event_dispatcher)
        first = await session.submit("first")
        second = await session.submit("second")
        assert first.turn_id != second.turn_id
        assert session.store.read_turn(first.turn_id).thread_id == session.thread_id
        assert session.store.read_turn(second.turn_id).thread_id == session.thread_id
        assert session.head_turn_id == second.turn_id
        assert resources == (session.store, session.model, session.tool_orchestrator, session.event_dispatcher)
        assert len(models) == 1
        assert session.store.verify().valid
    with pytest.raises(RuntimeError, match="closed"):
        await session.submit("too late")


@pytest.mark.anyio
async def test_close_waits_for_active_turn_and_rejects_competing_submit(tmp_path, monkeypatch):
    from tests.agent.harness.test_public_agent_cutover import BlockingPublicModel

    model = BlockingPublicModel()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    async with agent.session(require_workspace_change=False) as session:
        running = asyncio.create_task(session.submit("first"))
        await asyncio.wait_for(model.dispatch_started.wait(), 1)
        try:
            with pytest.raises(RuntimeError, match="active Turn"):
                await session.submit("competing")
            closing = asyncio.create_task(session.close())
            await asyncio.sleep(0)
            assert not closing.done()
            with pytest.raises(RuntimeError, match="closed"):
                await session.submit("after close requested")
            assert len(session.store.list_turns()) == 1
        finally:
            model.release_dispatch.set()
        result = await running
        await closing
        assert result.status == "done"
    with pytest.raises(sqlite3.ProgrammingError):
        session.store.read_thread(session.thread_id)


@pytest.mark.anyio
async def test_mcp_process_lives_across_submissions_and_exits_on_close(tmp_path, monkeypatch):
    from agent_runtime.runtime.mcp import decide_mcp_config_trust
    from tests.agent.test_mcp_e2e import _MCP_SERVER_SCRIPT

    pid_log = tmp_path / "server-pids.txt"
    script = tmp_path / "server.py"
    script.write_text(
        "import os\nfrom pathlib import Path\n"
        f"with Path({str(pid_log)!r}).open('a') as log: log.write(str(os.getpid()) + '\\n')\n" + _MCP_SERVER_SCRIPT,
    )
    config = tmp_path / "configs" / "mcp_servers.yaml"
    config.parent.mkdir()
    config.write_text(
        "servers:\n  - name: test_server\n    transport: stdio\n"
        f"    command: {sys.executable}\n    args: [{script}]\n"
        "    tools_allowlist: [read_only_info]\n    enabled: true\n",
    )
    monkeypatch.delenv("AGENT_MCP_CONFIG", raising=False)
    trust = decide_mcp_config_trust(config, workspace_root=tmp_path, trust_workspace=True)
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", mcp_config_trust=trust)
    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)
    async with agent.session(require_workspace_change=False) as session:
        tool = session.tools["mcp__test_server__read_only_info"]
        first = await session.submit("first")
        first_reply = await tool.run({"field": "version"})
        second = await session.submit("second")
        second_reply = await tool.run({"field": "status"})
        assert first.turn_id != second.turn_id
        assert "version: 1.0.0" in str(first_reply)
        assert "status: 1.0.0" in str(second_reply)
        assert session.tools[tool.definition.name] is tool
        pids = pid_log.read_text().splitlines()
        assert len(pids) == 1
        pid = int(pids[0])
        os.kill(pid, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.anyio
async def test_failed_open_releases_store_and_model_control(tmp_path, monkeypatch):
    from agent_runtime.harness import RolloutStore
    from agent_runtime.harness import session as session_module

    stores = []
    planes = []
    original_store = RolloutStore.__init__
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    create_plane = agent._get_model_control_plane

    def record_store(self, *args, **kwargs):
        original_store(self, *args, **kwargs)
        stores.append(self)

    def record_plane():
        plane = create_plane()
        planes.append(plane)
        return plane

    def fail_services(*args, **kwargs):
        raise RuntimeError("assembly failed after resources opened")

    monkeypatch.setattr(RolloutStore, "__init__", record_store)
    monkeypatch.setattr(agent, "_get_model_control_plane", record_plane)
    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)
    monkeypatch.setattr(session_module, "configure_services", fail_services)
    with pytest.raises(RuntimeError, match="assembly failed"):
        async with agent.session():
            raise AssertionError("failed Session must not be returned")
    assert len(planes) == len(stores) == 1
    assert planes[0]._closed
    with pytest.raises(sqlite3.ProgrammingError):
        stores[0].list_threads()


@pytest.mark.anyio
async def test_agent_factory_sessions_have_independent_model_state_and_resources(tmp_path, monkeypatch):
    agent = Agent(
        workspace_path=tmp_path,
        checkpoint_db=tmp_path / "rollout.db",
        enable_workspace_mcp=False,
        model_session_path=None,
    )
    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)
    async with agent.session(require_workspace_change=False) as first:
        async with agent.session(require_workspace_change=False) as second:
            assert first.thread_id != second.thread_id
            assert first.store is not second.store
            assert first.model_control_plane is not second.model_control_plane
            assert first.model is not second.model
            assert first.tool_orchestrator is not second.tool_orchestrator
            initial = second.current_model().id
            alternative = next(spec.id for spec in first.models() if spec.id != initial)
            first.switch_model(alternative)
            assert second.current_model().id == initial
            await first.close()
            assert first.model_control_plane._closed
            assert not second.model_control_plane._closed
            result = await second.submit("still usable")
            assert result.status == "done"


@pytest.mark.anyio
async def test_cancelled_turn_keeps_session_open_and_can_be_aborted(tmp_path, monkeypatch):
    from tests.agent.harness.test_public_agent_cutover import BlockingPublicModel

    model = BlockingPublicModel()
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    async with agent.session(require_workspace_change=False) as session:
        running = asyncio.create_task(session.submit("cancel me"))
        await asyncio.wait_for(model.dispatch_started.wait(), 1)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        assert not session._closed
        [turn] = session.store.list_turns()
        with pytest.raises(RuntimeError, match="active Turn"):
            await session.submit("cannot bypass pending outcome")
        await session.resume(turn.turn_id, "abort")
        model.release_dispatch.set()
        result = await session.submit("next turn")
        assert result.turn_id != turn.turn_id
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_one_shot_stream_does_not_drop_buffered_events_after_session_close(tmp_path, monkeypatch):
    from agent_runtime.streaming.events import EventType

    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "rollout.db", enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)
    events = []
    async for event in agent.stream("slow consumer", require_workspace_change=False):
        events.append(event)
        await asyncio.sleep(0.02)
    assert [event.type for event in events] == [
        EventType.TURN_STARTED,
        EventType.ITEM_STARTED,
        EventType.ITEM_COMPLETED,
        EventType.TURN_COMPLETED,
    ]


@pytest.mark.anyio
async def test_close_waits_for_direct_child_and_its_budget_settlement(tmp_path):
    from agent_runtime.harness import RolloutStore, Session
    from tests.agent.harness.test_subagent_budget import AcceptAnswer, BlockingAnswerModel

    model = BlockingAnswerModel()
    database = tmp_path / "rollout.db"
    async with await Session.open(
        database=database, workspace=tmp_path, model=model, completion_gate=AcceptAnswer()
    ) as session:
        parent = session.store.start_turn(
            thread_id=session.thread_id,
            user_message="parent",
            binding_manifest={"model_id": "test-model", "model_token_budget_total": 100},
        )
        close_finished = asyncio.Event()

        async def run_child_then_wait_for_close():
            result = await session.run_child(
                parent_turn_id=parent.turn_id,
                user_message="child",
                max_steps=2,
                max_tokens_total=60,
            )
            await close_finished.wait()
            return result

        async def close_session():
            await session.close()
            close_finished.set()

        child = asyncio.create_task(run_child_then_wait_for_close())
        await asyncio.wait_for(model.started.wait(), 1)
        closing = asyncio.create_task(close_session())
        try:
            await asyncio.sleep(0.02)
            assert not closing.done(), "close must wait for child I/O and budget settlement"
        finally:
            model.release.set()
            await asyncio.wait_for(asyncio.gather(child, closing, return_exceptions=True), timeout=1)
        result = child.result()
        with RolloutStore(database) as store:
            assert store.read_child_budget_allocation(result.turn_id)["status"] == "settled"
            assert store.read_budget_state(parent.turn_id).used.total_tokens == 5

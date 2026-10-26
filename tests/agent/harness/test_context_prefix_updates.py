import json
from dataclasses import replace

import pytest

from agent_runtime import Agent
from agent_runtime.harness import GatewayHarnessModel, HarnessModelResponse, HarnessToolCall, Session
from agent_runtime.harness.memory import MemoryConfig
from agent_runtime.modeling.openai_wire import serialize_openai_request
from tests.agent.harness.test_model_adapter import CapturingGateway, _resolved_model
from tests.agent.harness.test_tool_orchestrator import _read_tool
from tests.agent.test_update_plan_surfaces import _UpdatePlanThenAnswerModel


class Probe(_UpdatePlanThenAnswerModel):
    def __init__(self, mode):
        self.mode = mode
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return super().prepare(request)

    async def dispatch(self, prepared, **kwargs):
        request = self.requests[-1]
        calls = ()
        first_turn = request.turn_id == self.requests[0].turn_id
        if first_turn and self.mode == "deferred" and request.step == 1:
            calls = (HarnessToolCall(id="find", name="find_tools", arguments={"query": "read", "limit": 1}),)
        elif (
            first_turn
            and self.mode != "answer"
            and request.step <= (2 if self.mode in {"nested", "nested_changed", "deferred"} else 1)
        ):
            calls = (
                HarnessToolCall(
                    id=f"read-{request.step}",
                    name="read_file",
                    arguments={"path": "pkg/source.txt" if self.mode in {"nested", "nested_changed"} else "source.txt"},
                ),
            )
        return HarnessModelResponse(
            text="" if calls else "answer", provider_response_id=None, usage={}, tool_calls=calls
        )


def wire(request):
    resolved = _resolved_model(gateway=CapturingGateway())
    resolved.capabilities = replace(resolved.capabilities, supported_cache_parameters=("prompt_cache_key",))
    adapter = GatewayHarnessModel(model_id="test-model", resolved=resolved, instructions=("Static policy",))
    return serialize_openai_request(adapter.prepare(request).dispatch_payload.request).payload


@pytest.mark.anyio
@pytest.mark.parametrize(
    "mode", ["plain", "nested", "nested_changed", "root_changed", "root_removed", "memory_version", "deferred"]
)
async def test_context_changes_append_to_serialized_history(tmp_path, mode):
    (tmp_path / "source.txt").write_text("source")
    (tmp_path / "AGENTS.md").write_text("root rules v1")
    if mode in {"nested", "nested_changed"}:
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg/AGENTS.md").write_text("nested rules")
        (tmp_path / "pkg/source.txt").write_text("source")
    model = Probe(mode)
    agent = Agent(
        workspace_path=tmp_path,
        checkpoint_db=tmp_path / "run.db",
        enable_workspace_mcp=False,
        memory=MemoryConfig(use_memories=True, generate_memories=False),
    )
    agent._model_factory = lambda: model
    if mode == "deferred":
        session = await Session.open(
            database=tmp_path / "run.db",
            workspace=tmp_path,
            model=model,
            tools={"read_file": _read_tool(workspace=tmp_path)},
            discoverable_tool_names=("read_file",),
            memory=MemoryConfig(use_memories=True, generate_memories=False),
            _background_memory=False,
        )
    else:
        session = agent.session(_background_memory=False)
    async with session as live:
        await live.submit("first task")
        last = model.requests[-1]
        if mode == "root_changed":
            (tmp_path / "AGENTS.md").write_text("root rules v2")
        if mode == "root_removed":
            (tmp_path / "AGENTS.md").unlink()
        if mode == "nested_changed":
            (tmp_path / "pkg/AGENTS.md").write_text("nested rules v2")
        if mode == "memory_version":
            state = live.store.read_memory_state(tmp_path)
            live.store.commit_memory_state(tmp_path, expected_version=state["version"], state=state)
        await live.submit("second task")
        second = model.requests[-1]
        before, after = wire(last), wire(second)
        assert after["messages"][: len(before["messages"])] == before["messages"]
        assert after.get("tools") == before.get("tools")
        assert after.get("prompt_cache_key") == before.get("prompt_cache_key")
        for previous, following in zip(model.requests, model.requests[1:], strict=False):
            a, b = wire(previous), wire(following)
            assert b["messages"][: len(a["messages"])] == a["messages"]
        if mode == "root_changed":
            tail = str(after["messages"][len(before["messages"]) :])
            assert "root rules v2" in tail
            assert "root rules v2" not in str(before["messages"])
        if mode == "root_removed":
            latest = next(
                m.content for m in reversed(second.messages) if m.content.startswith("Project instructions (")
            )
            assert json.loads(latest.split("\n", 1)[1]) == []
            assert "replaces earlier project instruction snapshots" in latest
        if mode in {"nested", "nested_changed"}:
            assert "nested rules" in str(wire(model.requests[1])["messages"])
            assert second.binding_manifest["project_instructions"]["entries"][-1]["scope"] == "pkg"
            if mode == "nested_changed":
                assert "nested rules v2" in str(after["messages"][len(before["messages"]) :])
        if mode == "memory_version":
            assert (
                second.binding_manifest["memory_snapshot"]["version"]
                > last.binding_manifest["memory_snapshot"]["version"]
            )
            assert sum("Cross-chat memory (" in message.content for message in second.messages) == 1
        assert live.store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["delete", "disable", "disable_legacy_gap", "delete_legacy_enabled_gap"])
async def test_memory_revocation_starts_a_new_context_window_and_survives_reopen(tmp_path, change):
    from agent_runtime.harness import RolloutStore
    from agent_runtime.harness.memory_management import remember_memory

    database = tmp_path / "run.db"
    with RolloutStore(database) as store:
        remember_memory(store, tmp_path, key="secret", title="Sensitive heading", text="PRIVATE_MEMORY_BODY")

    class MemoryReader(Probe):
        async def dispatch(self, prepared, **kwargs):
            request = self.requests[-1]
            if request.turn_id == self.requests[0].turn_id and request.step == 1:
                return HarnessModelResponse(
                    text="",
                    provider_response_id=None,
                    usage={},
                    tool_calls=(HarnessToolCall(id="memory-read", name="read_memory", arguments={"key": "secret"}),),
                )
            return HarnessModelResponse(text="answer", provider_response_id=None, usage={})

    model = MemoryReader("memory")
    enabled = MemoryConfig(use_memories=True, generate_memories=False)
    async with await Session.open(
        database=database, workspace=tmp_path, model=model, memory=enabled, _background_memory=False
    ) as session:
        await session.submit("first task")
        assert "PRIVATE_MEMORY_BODY" in str(wire(model.requests[-1])["messages"])
        thread_id = session.thread_id
        if change in {"delete", "delete_legacy_enabled_gap"}:
            session.delete_memory("secret")
        if change in {"disable_legacy_gap", "delete_legacy_enabled_gap"}:
            from agent_runtime.harness.memory import memory_snapshot

            # A legacy intermediate Turn may carry no snapshot, or an updated
            # snapshot, while still projecting earlier recalled data.
            legacy_binding = (
                {"memory_snapshot": memory_snapshot(session.store, tmp_path, navigation_only=True)}
                if change == "delete_legacy_enabled_gap"
                else {}
            )
            gap = session.store.start_turn(
                thread_id=thread_id, user_message="legacy intermediate task", binding_manifest=legacy_binding
            )
            session.store.complete_turn(turn_id=gap.turn_id, answer="intermediate answer")
        config = (
            enabled
            if change in {"delete", "delete_legacy_enabled_gap"}
            else MemoryConfig(use_memories=False, generate_memories=False)
        )
    async with await Session.open(
        database=database, workspace=tmp_path, model=model, thread_id=thread_id, memory=config, _background_memory=False
    ) as session:
        for task in ("second task", "third task"):
            result = await session.submit(task)
            visible = str(wire(model.requests[-1])["messages"])
            assert "PRIVATE_MEMORY_BODY" not in visible
            assert "Sensitive heading" not in visible
            assert session.store.verify().valid
        assert "PRIVATE_MEMORY_BODY" not in str(session.store.list_context_items(result.turn_id))
        assert "PRIVATE_MEMORY_BODY" in str(session.store.list_conversation_items(result.turn_id))
    async with await Session.open(
        database=database, workspace=tmp_path, model=model, thread_id=thread_id, memory=config, _background_memory=False
    ) as session:
        await session.submit("fourth task after reopening")
        assert "PRIVATE_MEMORY_BODY" not in str(wire(model.requests[-1])["messages"])


@pytest.mark.anyio
async def test_context_updates_survive_saved_request_replay_and_compaction(tmp_path, monkeypatch):
    from agent_runtime.harness import RolloutContextManager
    from agent_runtime.modeling.contracts import ContextCapacityPolicy
    from agent_runtime.runtime_clock import capture_runtime_clock

    clock = capture_runtime_clock()
    monkeypatch.setattr("agent_runtime.harness.turn.capture_runtime_clock", lambda: dict(clock))

    (tmp_path / "AGENTS.md").write_text("ROOT_BASELINE")
    model = Probe("answer")
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    async with agent.session(_background_memory=False) as session:
        result = await session.submit("first task")
        original = model.requests[0]
        executor = session._executor()
        operation = session.store.list_model_operations(result.turn_id)[0]
        assert any("ROOT_BASELINE" in update for update in operation.request_ref["context_updates"])
        (tmp_path / "AGENTS.md").write_text("CHANGED_ON_DISK")
        with monkeypatch.context() as guard:

            def forbid_clock():
                raise AssertionError("replay must use its saved clock")

            guard.setattr("agent_runtime.harness.turn.capture_runtime_clock", forbid_clock)
            restored = executor.restore_step_context(executor.restore_turn_context(result.turn_id), operation)
            assert wire(restored.model_request()) == wire(original)
        binding = {**original.binding_manifest, "context_capacity_policy": ContextCapacityPolicy().model_dump()}
        binding.pop("execution_ownership", None)
        turn = session.store.start_turn(thread_id=session.thread_id, user_message="continue", binding_manifest=binding)
        manager = RolloutContextManager(session.store)
        source_hash, source = manager.semantic_source(turn.turn_id)
        assert "ROOT_BASELINE" in source
        candidate = manager.semantic_candidate(turn.turn_id, source_hash=source_hash, summary="Previous task answered.")
        context = executor.restore_turn_context(turn.turn_id)
        planned = executor.capture_step_context(context, step=1, messages=candidate.messages)
        manager.commit_compaction(candidate)
        projected = executor.capture_step_context(context, step=1)
        assert projected.messages == planned.messages
        updates = [m for m in projected.messages if m.content.startswith("Project instructions (")]
        assert len(updates) == 1 and "ROOT_BASELINE" in updates[0].content
        assert "CHANGED_ON_DISK" not in str(projected.messages)
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_legacy_turn_keeps_its_context_layout(tmp_path):
    from agent_runtime.harness import HarnessMessage

    (tmp_path / "AGENTS.md").write_text("legacy rule")
    model = Probe("answer")
    async with await Session.open(
        database=tmp_path / "run.db",
        workspace=tmp_path,
        model=model,
        memory=MemoryConfig(generate_memories=False),
        _background_memory=False,
    ) as session:
        result = await session.submit("task")
        executor = session._executor()
        turn = executor.restore_turn_context(result.turn_id)
        policy = {
            key: value
            for key, value in turn.binding_manifest["execution_context_policy"].items()
            if key != "context_updates"
        }
        legacy = replace(turn, binding_manifest={**turn.binding_manifest, "execution_context_policy": policy})
        step = executor.capture_step_context(legacy, step=2, messages=(HarnessMessage(role="user", content="task"),))
        assert step.messages[0].content.startswith("Cross-chat memory (")
        assert json.loads(step.messages[0].content.split("\n", 1)[1])["version"] == 0
        assert step.messages[1].content.startswith("Project instructions (")
        assert "legacy rule" in step.messages[1].content
        assert "replaces earlier" not in step.messages[1].content
        assert "context_updates" not in step.binding_manifest


@pytest.mark.anyio
async def test_tool_discovery_survives_reopen_but_obeys_fork_cutoff(tmp_path):
    database = tmp_path / "run.db"
    (tmp_path / "source.txt").write_text("source")
    model = Probe("answer")
    options = dict(
        database=database,
        workspace=tmp_path,
        model=model,
        tools={"read_file": _read_tool(workspace=tmp_path)},
        discoverable_tool_names=("read_file",),
        _background_memory=False,
    )
    async with await Session.open(**options) as session:
        before_discovery = await session.submit("first task")
        model.requests.clear()
        model.mode = "deferred"
        discovered = await session.submit("discover and read")
        thread_id = session.thread_id
        active_wire = wire(model.requests[-1])
        assert "read_file" in [tool.definition.name for tool in model.requests[-1].tools]
    model.mode = "answer"
    async with await Session.open(**options, thread_id=thread_id) as session:
        await session.submit("continue")
        assert wire(model.requests[-1]).get("tools") == active_wire.get("tools")
        assert wire(model.requests[-1])["messages"][: len(active_wire["messages"])] == active_wire["messages"]
    for fork_at, expected in ((before_discovery.turn_id, False), (discovered.turn_id, True)):
        async with await Session.open(**options, previous_turn_id=fork_at) as session:
            await session.submit("forked task")
            assert ("read_file" in [tool.definition.name for tool in model.requests[-1].tools]) == expected
            assert session.store.verify().valid


@pytest.mark.anyio
async def test_memory_body_change_appends_navigation_even_when_title_is_unchanged(tmp_path):
    from agent_runtime.harness import RolloutStore
    from agent_runtime.harness.memory_management import edit_memory, remember_memory

    database = tmp_path / "run.db"
    with RolloutStore(database) as store:
        remember_memory(store, tmp_path, key="runner", title="Runner", text="old body")
    model = Probe("answer")
    async with await Session.open(
        database=database,
        workspace=tmp_path,
        model=model,
        memory=MemoryConfig(generate_memories=False),
        _background_memory=False,
    ) as session:
        await session.submit("first task")
        before = wire(model.requests[-1])
        edit_memory(session.store, tmp_path, key="runner", text="new body")
        await session.submit("second task")
        after = wire(model.requests[-1])
        assert after["messages"][: len(before["messages"])] == before["messages"]
        navigation = [
            json.loads(m.content.split("\n", 1)[1])
            for m in model.requests[-1].messages
            if m.content.startswith("Cross-chat memory (")
        ]
        assert len(navigation) == 2
        assert navigation[0]["entries"] == navigation[1]["entries"]
        assert navigation[0]["content_revision"] != navigation[1]["content_revision"]
        assert "new body" not in str(after["messages"])


def test_cli_chat_uses_the_same_append_policy_as_the_sdk(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from agent_runtime import cli

    (tmp_path / "AGENTS.md").write_text("CLI_RULE_V1")

    class Model(Probe):
        async def dispatch(self, prepared, **kwargs):
            response = await super().dispatch(prepared, **kwargs)
            (tmp_path / "AGENTS.md").write_text("CLI_RULE_V2")
            return response

    model = Model("answer")
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    monkeypatch.setattr(cli, "_create_agent_facade", lambda **_options: agent)
    result = CliRunner().invoke(
        cli.agent_app, ["chat", "--ui", "inline", "--no-web-tools"], input="first task\nsecond task\n/exit\n"
    )
    assert result.exit_code == 0, result.output
    assert len(model.requests) == 2, result.output
    before, after = map(wire, model.requests)
    assert after["messages"][: len(before["messages"])] == before["messages"]
    assert "CLI_RULE_V1" in str(before["messages"])
    assert "CLI_RULE_V2" in str(after["messages"][len(before["messages"]) :])

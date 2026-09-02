from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest

from agent_runtime import Agent
from agent_runtime.cli import _run_facade_command
from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness import (
    HarnessModelRequest,
    HarnessModelResponse,
    HarnessToolCall,
    ModelDispatchCancelledError,
    PreparedModelCall,
    RolloutEventReader,
    RolloutStore,
    RuntimeComposition,
)
from agent_runtime.harness import composition as harness_composition
from agent_runtime.streaming.events import EventType, StreamEvent, TurnItemKind
from agent_runtime.streaming.sink import TurnEventDispatcher


class PublicHarnessModel:
    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {
            "model_alias": "public-harness-model",
            "model_revision": "public-harness-v1",
        }

    def ensure_available(
        self,
        binding: Mapping[str, object],
        *,
        thread_id: str,
        turn_id: str,
    ) -> None:
        if binding.get("thread_id") != thread_id or binding.get("turn_id") != turn_id:
            raise RuntimeError("public harness binding belongs to a different Turn")
        if binding.get("model_revision") != "public-harness-v1":
            raise RuntimeError("public harness binding revision changed")

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        digest = hashlib.sha256(f"{request.turn_id}:{request.step}".encode()).hexdigest()
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash="public-tools",
            wire_hash=digest,
            request_ref={"step": request.step},
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        return HarnessModelResponse(
            text="answer from replacement harness",
            provider_response_id="public-response",
            usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
        )


class FailingPublicHarnessModel(PublicHarnessModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        raise ConnectionError("provider connection ended after dispatch")


class IncompletePublicHarnessModel(PublicHarnessModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        return HarnessModelResponse(
            text="partial public answer",
            provider_response_id="response-incomplete",
            usage={"input_tokens": 8, "output_tokens": 32, "total_tokens": 40},
            status="incomplete",
            incomplete_reason="max_output_tokens",
        )


@pytest.mark.anyio
async def test_public_read_result_after_disconnect_never_constructs_model_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    monkeypatch.setattr(agent, "_harness_model", lambda: PublicHarnessModel())
    committed = await agent.run(
        "commit before disconnect",
        require_workspace_change=False,
    )

    def fail_model_construction() -> object:
        raise AssertionError("read_result must not construct a model runtime")

    monkeypatch.setattr(agent, "_harness_model", fail_model_construction)
    replayed = await agent.read_result(committed.turn_id)

    assert replayed == committed
    with RolloutStore(database) as store:
        assert store.verify().valid is True


@pytest.mark.anyio
async def test_public_result_exposes_durable_unknown_model_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    monkeypatch.setattr(
        agent,
        "_harness_model",
        lambda: FailingPublicHarnessModel(),
    )

    result = await agent.run("dispatch failure", require_workspace_change=False)

    assert result.status == "paused"
    assert result.diagnostics == (result.diagnostics[0],)
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "model_dispatch_outcome_unknown"
    assert diagnostic.component == "model"
    assert diagnostic.error_type == "ConnectionError"
    assert diagnostic.message == "provider connection ended after dispatch"


@pytest.mark.anyio
async def test_public_result_exposes_durable_incomplete_model_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    monkeypatch.setattr(
        agent,
        "_harness_model",
        lambda: IncompletePublicHarnessModel(),
    )

    result = await agent.run("bounded model output", require_workspace_change=False)

    assert result.status == "failed"
    assert result.answer is None
    assert result.usage.model_calls == 1
    assert result.usage.total_tokens == 40
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "model_response_incomplete"
    assert diagnostic.component == "model"
    assert diagnostic.severity == "error"
    assert diagnostic.error_type == "IncompleteModelResponse"
    assert diagnostic.message == "Model response was incomplete: max_output_tokens."


@pytest.mark.anyio
async def test_public_agent_can_explicitly_disable_workspace_mcp_discovery_and_freezes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    config = workspace / "configs" / "mcp_servers.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("servers: {}\n", encoding="utf-8")
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(
        checkpoint_db=database,
        workspace_path=workspace,
        enable_workspace_mcp=False,
    )
    monkeypatch.setattr(agent, "_harness_model", lambda: PublicHarnessModel())

    result = await agent.run(
        "run without repository MCP servers",
        require_workspace_change=False,
    )

    assert result.status == "done"
    with RolloutStore(database) as store:
        assert store.read_turn(result.turn_id).binding_manifest["mcp_policy"] == {"workspace_discovery_enabled": False}


@pytest.mark.anyio
async def test_public_followup_restores_disabled_workspace_mcp_from_runtime_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    config = workspace / "configs" / "mcp_servers.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("servers: {}\n", encoding="utf-8")
    database = tmp_path / "praxis.sqlite3"
    first_agent = Agent(
        checkpoint_db=database,
        workspace_path=workspace,
        enable_workspace_mcp=False,
    )
    monkeypatch.setattr(
        first_agent,
        "_harness_model",
        lambda: PublicHarnessModel(),
    )
    first = await first_agent.run("first", require_workspace_change=False)
    fresh_process_agent = Agent(
        checkpoint_db=database,
        workspace_path=workspace,
    )
    monkeypatch.setattr(
        fresh_process_agent,
        "_harness_model",
        lambda: PublicHarnessModel(),
    )

    followup = await fresh_process_agent.run(
        "follow up",
        previous_turn_id=first.turn_id,
        require_workspace_change=False,
    )

    assert followup.status == "done"
    with RolloutStore(database) as store:
        assert store.read_turn(followup.turn_id).binding_manifest["mcp_policy"] == {
            "workspace_discovery_enabled": False
        }


class PatchThenAnswerModel:
    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {
            "model_alias": "patch-model",
            "model_revision": "patch-model-v1",
        }

    def ensure_available(
        self,
        binding: Mapping[str, object],
        *,
        thread_id: str,
        turn_id: str,
    ) -> None:
        if binding.get("thread_id") != thread_id or binding.get("turn_id") != turn_id:
            raise RuntimeError("patch-model binding belongs to a different Turn")
        if binding.get("model_revision") != "patch-model-v1":
            raise RuntimeError("patch-model binding revision changed")

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        digest = hashlib.sha256(f"step:{request.step}".encode()).hexdigest()
        toolset_revision = toolset_revision_for_tools(request.tools)
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash=toolset_revision,
            wire_hash=digest,
            request_ref={
                "request_id": f"{request.turn_id}:step:{request.step}",
                "toolset_revision": toolset_revision,
                "exposed_tool_names": [tool.definition.name for tool in request.tools],
            },
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        if prepared.request_ref["request_id"].endswith(":step:1"):
            return HarnessModelResponse(
                text="",
                provider_response_id="patch-call",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="patch-1",
                        name="apply_patch",
                        arguments={
                            "file_path": "value.txt",
                            "old_string": "before",
                            "new_string": "after",
                        },
                    ),
                ),
            )
        return HarnessModelResponse(
            text="patched through replacement harness",
            provider_response_id="patch-answer",
            usage={"input_tokens": 3, "output_tokens": 2},
        )


class DestructivePythonThenAnswerModel(PatchThenAnswerModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        if prepared.request_ref["request_id"].endswith(":step:1"):
            return HarnessModelResponse(
                text="",
                provider_response_id="python-call",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="python-1",
                        name="execute_python",
                        arguments={
                            "code": "from pathlib import Path\nPath('value.txt').write_text('after')",
                            "workspace_write": True,
                            "output_paths": ["value.txt"],
                        },
                    ),
                ),
            )
        return HarnessModelResponse(
            text="changed through managed Python",
            provider_response_id="python-answer",
            usage={"input_tokens": 3, "output_tokens": 2},
        )


class CapturingPublicModel(PublicHarnessModel):
    def __init__(self) -> None:
        self.requests: list[HarnessModelRequest] = []

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        self.requests.append(request)
        return super().prepare(request)


class BlockingPublicModel(PublicHarnessModel):
    def __init__(self) -> None:
        self.dispatch_started = asyncio.Event()
        self.release_dispatch = asyncio.Event()

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        self.dispatch_started.set()
        await self.release_dispatch.wait()
        return HarnessModelResponse(
            text="live stream answer",
            provider_response_id="live-stream-response",
            usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
        )


class AcknowledgedStreamCancelModel(PublicHarnessModel):
    def __init__(self) -> None:
        self.dispatch_started = asyncio.Event()

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        del prepared
        self.dispatch_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError as exc:
            raise ModelDispatchCancelledError(
                "provider acknowledged stream cancellation"
            ) from exc


class BlockingResumeModel(PatchThenAnswerModel):
    def __init__(self) -> None:
        self.resume_dispatch_started = asyncio.Event()
        self.release_resume_dispatch = asyncio.Event()

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        if prepared.request_ref["request_id"].endswith(":step:1"):
            return await super().dispatch(prepared)
        self.resume_dispatch_started.set()
        await self.release_resume_dispatch.wait()
        return HarnessModelResponse(
            text="resumed live",
            provider_response_id="resume-live-response",
            usage={"input_tokens": 3, "output_tokens": 2},
        )


class PlanThenAnswerModel(PatchThenAnswerModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        if prepared.request_ref["request_id"].endswith(":step:1"):
            return HarnessModelResponse(
                text="",
                provider_response_id="plan-call",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="plan-1",
                        name="update_plan",
                        arguments={
                            "plan": [
                                {
                                    "step_id": "inspect",
                                    "step": "Inspect the target",
                                    "status": "in_progress",
                                },
                                {
                                    "step_id": "verify",
                                    "step": "Verify the answer",
                                    "status": "pending",
                                },
                            ],
                            "explanation": "Use a bounded strategy.",
                        },
                    ),
                ),
            )
        return HarnessModelResponse(
            text="planned answer",
            provider_response_id="planned-answer",
            usage={"input_tokens": 3, "output_tokens": 2},
        )


@pytest.mark.anyio
async def test_public_agent_uses_rollout_harness_not_legacy_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    events: list[StreamEvent] = []

    class Sink:
        async def emit(self, event: StreamEvent) -> None:
            events.append(event)

    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)
    result = await agent.run(
        "answer through the public SDK",
        require_workspace_change=False,
        event_sink=Sink(),
    )

    assert result.answer == "answer from replacement harness"
    assert result.status == "done"
    assert result.thread_id != result.turn_id
    assert result.workspace_path == str(workspace.resolve())
    with RolloutStore(database) as store:
        assert store.read_turn(result.turn_id).thread_id == result.thread_id
        assert store.verify().valid is True
    assert [event.type for event in events] == [
        EventType.TURN_STARTED,
        EventType.ITEM_STARTED,
        EventType.ITEM_COMPLETED,
        EventType.TURN_COMPLETED,
    ]
    assert {event.turn_id for event in events} == {result.turn_id}


@pytest.mark.anyio
async def test_public_resume_approves_the_same_durable_harness_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "value.txt"
    target.write_text("before", encoding="utf-8")
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    model = PatchThenAnswerModel()
    monkeypatch.setattr(agent, "_harness_model", lambda: model)

    paused = await agent.run(
        "change value.txt",
        require_workspace_change=False,
    )

    assert paused.status == "paused"
    assert paused.pause is not None
    assert paused.pause.kind == "tool_approval"
    assert target.read_text(encoding="utf-8") == "before"
    pending = await agent.pending_input(paused.turn_id)
    assert pending == paused.pause

    resumed = await agent.resume(paused.turn_id, "allow_once")

    assert resumed.turn_id == paused.turn_id
    assert resumed.thread_id == paused.thread_id
    assert resumed.status == "done"
    assert resumed.answer == "patched through replacement harness"
    assert target.read_text(encoding="utf-8") == "after"
    with RolloutStore(database) as store:
        [operation] = store.list_tool_operations(paused.turn_id)
        assert operation.status == "succeeded"
        assert store.verify().valid is True


@pytest.mark.anyio
async def test_public_resume_restores_the_frozen_tool_execution_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "value.txt"
    target.write_text("before", encoding="utf-8")
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    monkeypatch.setattr(
        agent,
        "_harness_model",
        lambda: DestructivePythonThenAnswerModel(),
    )

    paused = await agent.run(
        "change value.txt",
        require_workspace_change=False,
        allow_write_tools=True,
        allow_execute_tools=True,
    )

    assert paused.status == "paused"
    with RolloutStore(database) as store:
        policy = store.read_turn(paused.turn_id).binding_manifest["tool_execution_policy"]
        assert policy["allow_write_tools"] is True
        assert policy["allow_execute_tools"] is True

    resumed = await agent.resume(paused.turn_id, "allow_once")

    assert resumed.status == "done"
    assert target.read_text(encoding="utf-8") == "after"


@pytest.mark.anyio
async def test_public_abort_cancels_only_a_safely_paused_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "value.txt"
    target.write_text("before", encoding="utf-8")
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    monkeypatch.setattr(agent, "_harness_model", lambda: PatchThenAnswerModel())
    paused = await agent.run(
        "change value.txt",
        require_workspace_change=False,
    )

    cancelled = await agent.resume(paused.turn_id, "abort")

    assert cancelled.status == "failed"
    assert target.read_text(encoding="utf-8") == "before"
    with RolloutStore(database) as store:
        assert store.read_turn(paused.turn_id).status == "cancelled"
        assert store.list_tool_operations(paused.turn_id)[0].status == "cancelled"
        assert store.list_interactions(paused.turn_id)[0].status == "resolved"
        assert store.verify().valid is True


@pytest.mark.anyio
async def test_public_agent_freezes_and_durably_enforces_model_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)

    result = await agent.run(
        "claim a change without making one",
        max_turns=1,
        max_tokens_total=10,
        require_workspace_change=True,
    )

    assert result.status == "failed"
    assert result.answer is None
    with RolloutStore(database) as store:
        turn = store.read_turn(result.turn_id)
        assert turn.status == "failed"
        assert turn.binding_manifest["model_step_budget"] == 1
        assert turn.binding_manifest["model_token_budget_total"] == 10
        assert len(store.list_model_operations(result.turn_id)) == 1
        assert store.verify().valid is True


@pytest.mark.anyio
async def test_public_agent_rejects_untrusted_workspace_mcp_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "mcp.yaml"
    config.write_text("servers: []\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MCP_CONFIG", str(config))
    agent = Agent(checkpoint_db=tmp_path / "praxis.sqlite3", workspace_path=workspace)
    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)

    with pytest.raises(PermissionError, match="workspace MCP config is not trusted"):
        await agent.run("do not start MCP", require_workspace_change=False)


@pytest.mark.anyio
async def test_public_agent_stages_files_as_canonical_input_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "brief.txt"
    source.write_text("trusted attachment", encoding="utf-8")
    database = tmp_path / "praxis.sqlite3"
    model = CapturingPublicModel()
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    monkeypatch.setattr(agent, "_harness_model", lambda: model)

    result = await agent.run(
        "use the attached brief",
        files=[str(source)],
        require_workspace_change=False,
    )

    assert result.files == (str(source),)
    attachment_messages = [
        message.content
        for message in model.requests[0].messages
        if message.role == "context" and "Attached input file" in message.content
    ]
    assert len(attachment_messages) == 1
    assert "brief.txt" in attachment_messages[0]
    with RolloutStore(database) as store:
        input_items = [item for item in store.list_items(result.turn_id) if item.kind == "input_file"]
        assert len(input_items) == 1
        staged = workspace / str(input_items[0].payload["workspace_path"])
        assert staged.read_text(encoding="utf-8") == "trusted attachment"
        assert input_items[0].payload["sha256"]
        assert store.verify().valid is True


@pytest.mark.anyio
async def test_public_astream_replays_events_from_canonical_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = Agent(
        checkpoint_db=tmp_path / "praxis.sqlite3",
        workspace_path=workspace,
    )
    monkeypatch.setattr(agent, "_harness_model", PublicHarnessModel)
    events = [
        event
        async for event in agent.stream(
            "stream replacement events",
            require_workspace_change=False,
        )
    ]

    assert [event.type for event in events] == [
        EventType.TURN_STARTED,
        EventType.ITEM_STARTED,
        EventType.ITEM_COMPLETED,
        EventType.TURN_COMPLETED,
    ]
    assert len({event.turn_id for event in events}) == 1


@pytest.mark.anyio
async def test_public_astream_emits_committed_turn_start_before_model_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = Agent(
        checkpoint_db=tmp_path / "praxis.sqlite3",
        workspace_path=workspace,
    )
    model = BlockingPublicModel()
    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    stream = agent.stream(
        "stream while the model is blocked",
        require_workspace_change=False,
    )

    first = await asyncio.wait_for(anext(stream), timeout=1.0)

    assert first.type is EventType.TURN_STARTED
    assert model.dispatch_started.is_set()
    assert not model.release_dispatch.is_set()
    model.release_dispatch.set()
    remaining = [event async for event in stream]
    assert remaining[-1].type is EventType.TURN_COMPLETED


@pytest.mark.anyio
async def test_confirmed_stream_cancel_commits_request_then_turn_aborted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    model = AcknowledgedStreamCancelModel()
    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    stream = agent.stream(
        "cancel the controlling stream",
        require_workspace_change=False,
    )

    started = await asyncio.wait_for(anext(stream), timeout=1.0)
    await asyncio.wait_for(model.dispatch_started.wait(), timeout=1.0)
    await stream.aclose()

    with RolloutStore(database) as store:
        replayed = [
            entry.event
            for entry in RolloutEventReader(store).read_global()
            if entry.event.turn_id == started.turn_id
        ]
        turn = store.read_turn(started.turn_id)
        thread = store.read_thread(turn.thread_id)

    assert [event.type for event in replayed] == [
        EventType.TURN_STARTED,
        EventType.TURN_CANCELLATION_REQUESTED,
        EventType.TURN_ABORTED,
    ]
    assert turn.status == "cancelled"
    assert thread.active_turn_id is None


@pytest.mark.anyio
async def test_unconfirmed_stream_cancel_retains_interrupted_turn_for_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    model = BlockingPublicModel()
    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    stream = agent.stream(
        "cancel with an unknown provider outcome",
        require_workspace_change=False,
    )

    started = await asyncio.wait_for(anext(stream), timeout=1.0)
    await asyncio.wait_for(model.dispatch_started.wait(), timeout=1.0)
    await stream.aclose()

    with RolloutStore(database) as store:
        replayed = [
            entry.event
            for entry in RolloutEventReader(store).read_global()
            if entry.event.turn_id == started.turn_id
        ]
        turn = store.read_turn(started.turn_id)
        thread = store.read_thread(turn.thread_id)

    assert [event.type for event in replayed] == [
        EventType.TURN_STARTED,
        EventType.TURN_CANCELLATION_REQUESTED,
        EventType.TURN_PAUSED,
    ]
    assert replayed[-1].data["reason"] == "outcome_unknown"
    assert turn.status == "interrupted"
    assert thread.active_turn_id == turn.turn_id


@pytest.mark.anyio
async def test_public_event_sink_receives_committed_events_during_the_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = Agent(
        checkpoint_db=tmp_path / "praxis.sqlite3",
        workspace_path=workspace,
    )
    model = BlockingPublicModel()
    turn_started = asyncio.Event()
    events: list[StreamEvent] = []

    class Sink:
        async def emit(self, event: StreamEvent) -> None:
            events.append(event)
            if event.type is EventType.TURN_STARTED:
                turn_started.set()

    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    run = asyncio.create_task(
        agent.run(
            "publish while the model is blocked",
            require_workspace_change=False,
            event_sink=Sink(),
        )
    )

    await asyncio.wait_for(turn_started.wait(), timeout=1.0)
    await asyncio.wait_for(model.dispatch_started.wait(), timeout=1.0)

    assert model.dispatch_started.is_set()
    assert not run.done()
    model.release_dispatch.set()
    result = await run
    assert result.answer == "live stream answer"
    assert [event.type for event in events] == [
        EventType.TURN_STARTED,
        EventType.ITEM_STARTED,
        EventType.ITEM_COMPLETED,
        EventType.TURN_COMPLETED,
    ]


@pytest.mark.anyio
async def test_public_stream_awaits_post_commit_batch_without_record_listener_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    model = BlockingPublicModel()
    agent = Agent(checkpoint_db=database, workspace_path=workspace)
    sink_started = asyncio.Event()
    release_sink = asyncio.Event()

    class BlockingSink:
        async def emit(self, event: StreamEvent) -> None:
            if event.type is EventType.TURN_STARTED:
                sink_started.set()
                await release_sink.wait()

    original_init = RolloutStore.__init__

    def reject_record_listener(self: RolloutStore, *args: object, **kwargs: object) -> None:
        assert kwargs.get("record_listener") is None
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(RolloutStore, "__init__", reject_record_listener)
    monkeypatch.setattr(harness_composition, "RolloutStore", RolloutStore)
    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    running = asyncio.create_task(
        agent.run(
            "await committed batch delivery",
            require_workspace_change=False,
            event_sink=BlockingSink(),
        )
    )

    await asyncio.wait_for(sink_started.wait(), timeout=0.5)
    assert model.dispatch_started.is_set() is False
    with RolloutStore(database) as committed_store:
        turns = committed_store.list_turns()
        assert len(turns) == 1
        assert turns[0].status == "running"
        assert committed_store.verify().valid is True

    release_sink.set()
    await asyncio.wait_for(model.dispatch_started.wait(), timeout=0.5)
    model.release_dispatch.set()
    result = await asyncio.wait_for(running, timeout=1.0)
    assert result.answer == "live stream answer"


@pytest.mark.anyio
async def test_interleaved_threads_publish_only_their_transaction_batches(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first_dispatcher = TurnEventDispatcher()
    second_dispatcher = TurnEventDispatcher()
    first_stream = first_dispatcher.subscribe_controlling()
    second_stream = second_dispatcher.subscribe_controlling()
    with RuntimeComposition.open(
        database=tmp_path / "praxis.sqlite3",
        workspace=workspace,
        model=PublicHarnessModel(),
        require_workspace_change=False,
    ) as runtime:
        first, second = await asyncio.gather(
            runtime.thread_manager.run(
                user_message="first interleaved thread",
                event_dispatcher=first_dispatcher,
            ),
            runtime.thread_manager.run(
                user_message="second interleaved thread",
                event_dispatcher=second_dispatcher,
            ),
        )

        first_events = []
        while not first_stream.empty:
            first_events.append(await first_stream.receive())
        second_events = []
        while not second_stream.empty:
            second_events.append(await second_stream.receive())

    assert first_events
    assert second_events
    assert {event.turn_id for event in first_events} == {first.turn_id}
    assert {event.turn_id for event in second_events} == {second.turn_id}


@pytest.mark.anyio
async def test_cli_arun_sink_and_astream_share_identical_v2_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class Capture:
        def __init__(self) -> None:
            self.events: list[StreamEvent] = []

        def begin_turn(self) -> None:
            return None

        def finish(self) -> None:
            return None

        async def emit(self, event: StreamEvent) -> None:
            self.events.append(event)

    def configured_agent(name: str) -> Agent:
        agent = Agent(
            checkpoint_db=tmp_path / f"{name}.sqlite3",
            workspace_path=workspace,
        )
        monkeypatch.setattr(agent, "_harness_model", lambda: PublicHarnessModel())
        return agent

    cli_agent = configured_agent("cli")
    cli_capture = Capture()
    cli_result = await _run_facade_command(
        cli_agent,
        task="CLI canonical lifecycle",
        files=(),
        max_tokens_total=None,
        interactive_approval=False,
        require_workspace_change=False,
        event_display=cli_capture,  # type: ignore[arg-type]
    )
    sink_agent = configured_agent("sink")
    sink_capture = Capture()
    sink_result = await sink_agent.run(
        "sink canonical lifecycle",
        require_workspace_change=False,
        event_sink=sink_capture,
    )
    stream_agent = configured_agent("stream")
    stream_events = [
        event
        async for event in stream_agent.stream(
            "astream canonical lifecycle",
            require_workspace_change=False,
        )
    ]

    all_live = (cli_capture.events, sink_capture.events, stream_events)
    assert (
        [event.type for event in all_live[0]]
        == [event.type for event in all_live[1]]
        == [event.type for event in all_live[2]]
    )
    assert (
        [event.item_kind for event in all_live[0]]
        == [event.item_kind for event in all_live[1]]
        == [event.item_kind for event in all_live[2]]
    )
    for database, result, live in (
        (tmp_path / "cli.sqlite3", cli_result, cli_capture.events),
        (tmp_path / "sink.sqlite3", sink_result, sink_capture.events),
    ):
        with RolloutStore(database) as store:
            replayed = RolloutEventReader(store).read(result.thread_id)
        assert [(event.type, event.item_id) for event in live] == [
            (entry.event.type, entry.event.item_id) for entry in replayed
        ]


@pytest.mark.anyio
async def test_public_runtime_emits_v2_only_without_implicit_legacy_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = Agent(
        checkpoint_db=tmp_path / "praxis.sqlite3",
        workspace_path=workspace,
    )
    monkeypatch.setattr(agent, "_harness_model", lambda: PublicHarnessModel())

    class Capture:
        def __init__(self) -> None:
            self.events: list[StreamEvent] = []

        async def emit(self, event: StreamEvent) -> None:
            self.events.append(event)

    capture = Capture()
    await agent.run(
        "canonical events only",
        require_workspace_change=False,
        event_sink=capture,
    )

    v2_types = {
        EventType.TURN_STARTED,
        EventType.TURN_PAUSED,
        EventType.TURN_RESUMED,
        EventType.TURN_CANCELLATION_REQUESTED,
        EventType.TURN_COMPLETED,
        EventType.TURN_ABORTED,
        EventType.ITEM_STARTED,
        EventType.ITEM_DELTA,
        EventType.ITEM_COMPLETED,
    }
    assert capture.events
    assert all(event.protocol_version == 2 for event in capture.events)
    assert all(event.type in v2_types for event in capture.events)
    assert len([event for event in capture.events if event.type is EventType.ITEM_COMPLETED]) == 1


@pytest.mark.anyio
async def test_public_resume_event_sink_is_live_and_does_not_replay_old_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "value.txt").write_text("before", encoding="utf-8")
    agent = Agent(
        checkpoint_db=tmp_path / "praxis.sqlite3",
        workspace_path=workspace,
    )
    model = BlockingResumeModel()
    monkeypatch.setattr(agent, "_harness_model", lambda: model)
    paused = await agent.run(
        "change value.txt",
        require_workspace_change=False,
    )
    assert paused.status == "paused"
    tool_started = asyncio.Event()
    events: list[StreamEvent] = []

    class Sink:
        async def emit(self, event: StreamEvent) -> None:
            events.append(event)
            if event.type is EventType.ITEM_STARTED and event.item_kind is TurnItemKind.TOOL:
                tool_started.set()

    resume = asyncio.create_task(agent.resume(paused.turn_id, "allow_once", event_sink=Sink()))

    await asyncio.wait_for(tool_started.wait(), timeout=1.0)
    await asyncio.wait_for(model.resume_dispatch_started.wait(), timeout=1.0)

    assert model.resume_dispatch_started.is_set()
    assert not resume.done()
    assert EventType.TURN_STARTED not in [event.type for event in events]
    assert EventType.HUMAN_INPUT_REQUIRED not in [event.type for event in events]
    model.release_resume_dispatch.set()
    result = await resume
    assert result.answer == "resumed live"
    assert events[-1].type is EventType.TURN_COMPLETED


@pytest.mark.anyio
async def test_public_abort_of_interrupted_turn_does_not_construct_model_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="legacy interrupted work",
            binding_manifest={
                "model_alias": "removed-legacy-model",
                "legacy_resume_compatible": False,
            },
        )
        store.interrupt_orphaned_turn(
            turn_id=turn.turn_id,
            reason="legacy worker stopped",
            maintenance_confirmed=True,
        )
    agent = Agent(checkpoint_db=database, workspace_path=workspace)

    def fail_model_construction() -> object:
        raise AssertionError("abort must not construct a model runtime")

    monkeypatch.setattr(agent, "_harness_model", fail_model_construction)

    result = await agent.resume(turn.turn_id, "abort")

    assert result.status == "failed"
    assert result.stop_reason == "cancelled"
    with RolloutStore(database) as store:
        assert store.read_turn(turn.turn_id).status == "cancelled"


@pytest.mark.anyio
async def test_public_resume_rejects_incompatible_migrated_turn_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "praxis.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="legacy approval",
            binding_manifest={
                "model_alias": "removed-legacy-model",
                "legacy_resume_compatible": False,
                "legacy_approvals_invalidated": True,
            },
        )
        store.pause_turn(turn_id=turn.turn_id, reason="legacy approval invalidated")
    agent = Agent(checkpoint_db=database, workspace_path=workspace)

    def fail_model_construction() -> object:
        raise AssertionError("incompatible legacy resume must fail before runtime")

    monkeypatch.setattr(agent, "_harness_model", fail_model_construction)

    with pytest.raises(RuntimeError, match="incompatible legacy Turn"):
        await agent.resume(turn.turn_id, "allow_once")

    with RolloutStore(database) as store:
        assert store.read_turn(turn.turn_id).status == "paused"
        assert store.read_thread(thread.thread_id).active_turn_id == turn.turn_id


@pytest.mark.anyio
async def test_public_result_projects_plan_from_committed_tool_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = Agent(
        checkpoint_db=tmp_path / "praxis.sqlite3",
        workspace_path=workspace,
    )
    monkeypatch.setattr(agent, "_harness_model", lambda: PlanThenAnswerModel())

    result = await agent.run(
        "plan this answer",
        require_workspace_change=False,
    )

    assert result.answer == "planned answer"
    assert result.plan is not None
    assert result.plan.objective == "plan this answer"
    assert result.plan.status == "complete"
    assert [step.step_id for step in result.plan.steps] == [
        "step_inspect",
        "step_verify",
    ]
    assert all(step.status == "completed" for step in result.plan.steps)
    assert [event.event_type for event in result.plan_events] == [
        "llm_update",
        "completed",
    ]

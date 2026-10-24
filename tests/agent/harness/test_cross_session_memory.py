from __future__ import annotations

import asyncio
import json

import pytest

from agent_runtime.harness import HarnessModelResponse, PreparedModelCall, RolloutStore, Session


class MemoryModel:
    def __init__(self):
        self.requests = []
        self.bad_output = False

    def snapshot(self, **identity):
        return {**identity, "model_id": "test"}

    def ensure_available(self, binding, **identity):
        assert binding["model_id"] == "test"

    def prepare(self, request):
        self.requests.append(request)
        return PreparedModelCall("r", "c", "t", "w", {"purpose": request.purpose})

    async def dispatch(self, prepared, *, delta_sink=None):
        request = self.requests[-1]
        if request.purpose == "memory_extract":
            source = json.loads(request.messages[-1].content.split("\nDATA:\n", 1)[1])
            self.extracted = [{"key": "environment", "title": "Python environment",
                               "text": "Use uv for this workspace.",
                               "source_item_ids": [source[0]["item_id"]]}]
            text = json.dumps({"entries": self.extracted})
        elif request.purpose == "memory_consolidate":
            data = json.loads(request.messages[-1].content.split("\nDATA:\n", 1)[1])
            entries = [{k: v for k, v in e.items() if k != "sources"} for e in data["candidates"]]
            text = "bad JSON" if self.bad_output else json.dumps({"entries": entries})
        else:
            text = "normal answer"
        return HarnessModelResponse(text, None, {"input_tokens": 10, "output_tokens": 20})


def seed(store, workspace, *, text="Use uv for Python commands.", contributes=True):
    thread = store.create_thread(workspace=workspace)
    turn = store.start_turn(thread_id=thread.thread_id, user_message=text,
                            binding_manifest={"memory_generation_allowed": contributes})
    store.complete_turn(turn_id=turn.turn_id, answer="acknowledged")
    return thread, turn


@pytest.mark.anyio
async def test_memory_update_is_durable_and_an_independent_chat_uses_a_frozen_snapshot(tmp_path):
    from agent_runtime.harness.memory import MemoryConfig, update_memory

    database = tmp_path / "r.db"
    model = MemoryModel()
    with RolloutStore(database) as store:
        seed(store, tmp_path)
        result = await update_memory(store, workspace=tmp_path, model=model, binding={"model_id": "test"})
        assert result["status"] == "updated"
        assert len(model.requests) == 2
        assert all(not request.tools for request in model.requests)
        assert store.verify().valid
        assert (await update_memory(store, workspace=tmp_path, model=model, binding={})) ["status"] == "unchanged"
        assert len(model.requests) == 2
    async with await Session.open(
        database=database, workspace=tmp_path, model=model,
        memory=MemoryConfig(generate_memories=False), require_workspace_change=False,
    ) as session:
        reply = await session.submit("new independent chat")
        snapshot = session.store.read_turn(reply.turn_id).binding_manifest["memory_snapshot"]
        assert snapshot["entries"][0] == {"key": "environment", "title": "Python environment"}
        detail = session.store.read_memory_state(tmp_path, version=snapshot["version"])["entries"][0]
        assert detail["text"] == "Use uv for this workspace."
        assert any("Python environment" in m.content and m.content.startswith("Cross-chat memory (")
                   for m in model.requests[-1].messages)
        frozen = json.dumps(snapshot)
        session.delete_memory("environment")
        assert json.dumps(session.store.read_turn(reply.turn_id).binding_manifest["memory_snapshot"]) == frozen
        next_reply = await session.submit("next message")
        assert session.store.read_turn(next_reply.turn_id).binding_manifest["memory_snapshot"]["entries"] == []


@pytest.mark.anyio
async def test_workspace_isolation_contribution_control_and_delete_does_not_resurrect(tmp_path):
    from agent_runtime.harness.memory import delete_memory, memory_snapshot, update_memory

    other = tmp_path / "other"
    other.mkdir()
    model = MemoryModel()
    with RolloutStore(tmp_path / "r.db") as store:
        seed(store, tmp_path, contributes=False)
        seed(store, other)
        assert (await update_memory(store, workspace=tmp_path, model=model, binding={})) ["status"] == "unchanged"
        seed(store, tmp_path)
        await update_memory(store, workspace=tmp_path, model=model, binding={})
        assert memory_snapshot(store, other)["entries"] == []
        delete_memory(store, tmp_path, "environment")
        seed(store, tmp_path, text="Same preference again: use uv")
        await update_memory(store, workspace=tmp_path, model=model, binding={})
        assert memory_snapshot(store, tmp_path)["entries"] == []


@pytest.mark.anyio
async def test_failed_consolidation_preserves_previous_memory_and_records_failure(tmp_path):
    from agent_runtime.harness.memory import memory_snapshot, update_memory

    model = MemoryModel()
    with RolloutStore(tmp_path / "r.db") as store:
        seed(store, tmp_path)
        await update_memory(store, workspace=tmp_path, model=model, binding={})
        before = memory_snapshot(store, tmp_path)
        seed(store, tmp_path, text="new evidence")
        model.bad_output = True
        result = await update_memory(store, workspace=tmp_path, model=model, binding={})
        assert result["status"] == "failed"
        assert memory_snapshot(store, tmp_path) == before
        assert result["maintenance_turn_ids"]
        assert store.verify().valid


@pytest.mark.anyio
async def test_model_cannot_forge_memory_source_ids(tmp_path):
    from agent_runtime.harness.memory import memory_snapshot, update_memory

    class ForgingModel(MemoryModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            return HarnessModelResponse(json.dumps({"entries": [{"key": "bad", "title": "bad", "text": "bad",
                                                                 "source_item_ids": ["unavailable"]}]}), None, {})

    with RolloutStore(tmp_path / "r.db") as store:
        seed(store, tmp_path)
        result = await update_memory(store, workspace=tmp_path, model=ForgingModel(), binding={})
        assert result["status"] == "failed"
        assert memory_snapshot(store, tmp_path)["entries"] == []


def test_sdk_can_explicitly_disable_memory_without_changing_model_config(tmp_path):
    from agent_runtime import Agent, MemoryConfig

    agent = Agent(workspace_path=tmp_path, memory=MemoryConfig(use_memories=False, generate_memories=False))
    assert not agent.memory.use_memories


def test_sdk_inspection_reports_explicit_controls_without_creating_database(tmp_path):
    from agent_runtime import Agent, MemoryConfig

    database = tmp_path / "not-created.sqlite"
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database,
                  memory=MemoryConfig(use_memories=True, generate_memories=False))
    status = agent.memories()
    assert status["use_memories"] is True and status["generate_memories"] is False
    assert status["entries"] == [] and not database.exists()


@pytest.mark.anyio
async def test_concurrent_maintenance_claims_only_one_worker(tmp_path):
    from agent_runtime.harness.memory import update_memory

    entered, release = asyncio.Event(), asyncio.Event()

    class WaitingModel(MemoryModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            entered.set()
            await release.wait()
            return await super().dispatch(prepared, delta_sink=delta_sink)

    database = tmp_path / "r.db"
    with RolloutStore(database) as first, RolloutStore(database) as second:
        seed(first, tmp_path)
        worker = asyncio.create_task(update_memory(first, workspace=tmp_path, model=WaitingModel(), binding={}))
        await entered.wait()
        contender = MemoryModel()
        assert (await update_memory(second, workspace=tmp_path, model=contender, binding={})) ["status"] == "busy"
        assert contender.requests == []
        release.set()
        assert (await worker)["status"] == "updated"


@pytest.mark.anyio
async def test_read_memory_uses_frozen_entries_and_cannot_read_raw_other_chat_items(tmp_path):
    from agent_runtime.harness.memory import create_memory_tool, delete_memory, memory_snapshot, update_memory
    from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin

    with RolloutStore(tmp_path / "r.db") as store:
        seed(store, tmp_path)
        await update_memory(store, workspace=tmp_path, model=MemoryModel(), binding={})
        thread = store.create_thread(workspace=tmp_path)
        snapshot = memory_snapshot(store, tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="recall",
                                binding_manifest={"memory_snapshot": snapshot})
        tool = create_memory_tool(store)
        executor = ToolOrchestrator(store=store, tools={"read_memory": tool},
                                   execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path))
        delete_memory(store, tmp_path, "environment")
        output = await executor.execute(turn_id=turn.turn_id, call=ToolCall(
            tool_call_id="recall-1", tool_name="read_memory", arguments={"key": "environment"},
            origin=ToolCallOrigin(request_id="r", toolset_revision="v", exposed_tool_names=("read_memory",))))
        assert not output.is_error
        result = json.loads(output.content[0].data["text"])
        assert result["version"] == snapshot["version"]
        assert result["entries"][0]["text"] == "Use uv for this workspace."
        invalid = await executor.execute(turn_id=turn.turn_id, call=ToolCall(
            tool_call_id="recall-2", tool_name="read_memory",
            arguments={"item_id": snapshot["entries"][0]["source_item_ids"][0]},
            origin=ToolCallOrigin(request_id="r", toolset_revision="v", exposed_tool_names=("read_memory",))))
        assert invalid.is_error


def test_memory_cli_status_is_readonly_and_controls_are_independent(tmp_path):
    from typer.testing import CliRunner

    from agent_runtime.cli import agent_app

    database = tmp_path / "r.db"
    common = ["--workspace", str(tmp_path), "--checkpoint-db", str(database)]
    runner = CliRunner()
    result = runner.invoke(agent_app, ["memory", "status", *common])
    assert result.exit_code == 0, result.output
    assert not database.exists()
    result = runner.invoke(agent_app, ["memory", "configure", "--use", "--no-generate", *common])
    assert result.exit_code == 0, result.output
    with RolloutStore(database, read_only=True) as store:
        state = store.read_memory_state(tmp_path)
        assert state["use_memories"] and not state["generate_memories"]


@pytest.mark.anyio
async def test_long_history_is_consumed_in_complete_batches_across_reopens(tmp_path):
    from agent_runtime.harness.memory import MAX_SOURCE_BYTES, update_memory

    database = tmp_path / "r.db"
    model = MemoryModel()
    with RolloutStore(database) as store:
        thread, turn = seed(store, tmp_path, text="中文 and escaped \"data\" " * 5000)
        original = next(i for i in store.list_items(turn.turn_id) if i.kind == "user_message")
        expected = json.dumps(dict(original.payload), ensure_ascii=False)
    fragments = []
    for _ in range(20):
        with RolloutStore(database) as store:
            result = await update_memory(store, workspace=tmp_path, model=model, binding={})
        assert result["status"] == "updated"
        request = next(r for r in reversed(model.requests) if r.purpose == "memory_extract")
        records = json.loads(request.messages[-1].content.split("\nDATA:\n", 1)[1])
        assert len(json.dumps(records, ensure_ascii=False).encode()) <= MAX_SOURCE_BYTES
        fragments.extend(r["payload_fragment"] for r in records if r["item_id"] == original.item_id)
        if not result["has_more"]:
            break
    else:
        pytest.fail("source must make bounded progress")
    assert len(fragments) > 1 and "".join(fragments) == expected
    with RolloutStore(database) as store:
        assert (await update_memory(store, workspace=tmp_path, model=model, binding={})) ["status"] == "unchanged"
        later = store.start_turn(thread_id=thread.thread_id, user_message="new allowed fact",
                                 binding_manifest={"memory_generation_allowed": True})
        store.complete_turn(turn_id=later.turn_id, answer="acknowledged")
        await update_memory(store, workspace=tmp_path, model=model, binding={})
    request = next(r for r in reversed(model.requests) if r.purpose == "memory_extract")
    assert "new allowed fact" in request.messages[-1].content
    assert original.item_id not in request.messages[-1].content


@pytest.mark.anyio
async def test_snapshot_failure_returns_diagnostic_without_phantom_turn_cleanup(tmp_path):
    from agent_runtime.harness.memory import update_memory

    class BrokenSnapshot(MemoryModel):
        def snapshot(self, **identity):
            raise ValueError("cannot bind model")

    with RolloutStore(tmp_path / "r.db") as store:
        seed(store, tmp_path)
        result = await update_memory(store, workspace=tmp_path, model=BrokenSnapshot(), binding={})
        assert result["status"] == "failed"
        assert result["error_type"] == "ValueError" and result["maintenance_turn_ids"] == []
        assert store.read_memory_state(tmp_path)["version"] == 0


@pytest.mark.anyio
async def test_reopened_chat_obeys_new_workspace_controls_without_extracting_disabled_turns(tmp_path):
    from agent_runtime.harness.memory import MemoryConfig, configure_memory, update_memory

    database = tmp_path / "r.db"
    model = MemoryModel()
    async with await Session.open(database=database, workspace=tmp_path, model=model,
                                  memory=MemoryConfig(use_memories=False, generate_memories=False),
                                  require_workspace_change=False) as session:
        await session.submit("private old turn")
        thread_id = session.thread_id
    with RolloutStore(database) as store:
        configure_memory(store, tmp_path, MemoryConfig())
    async with await Session.open(database=database, workspace=tmp_path, model=model, thread_id=thread_id,
                                  _background_memory=False, require_workspace_change=False) as session:
        assert session.memory_config.generate_memories
        assert session.store.read_thread(thread_id).settings["memory_generation_allowed"]
        await session.submit("new allowed turn")
    with RolloutStore(database) as store:
        result = await update_memory(store, workspace=tmp_path, model=model, binding={})
        assert result["status"] == "updated"
    extraction = next(r for r in model.requests if r.purpose == "memory_extract")
    assert "new allowed turn" in extraction.messages[-1].content
    assert "private old turn" not in extraction.messages[-1].content


@pytest.mark.anyio
async def test_memory_maintenance_runs_the_existing_model_ready_hook(tmp_path):
    from agent_runtime.harness.memory import update_memory

    calls = []

    async def ready(binding):
        calls.append(binding["maintenance_purpose"])

    with RolloutStore(tmp_path / "r.db") as store:
        seed(store, tmp_path)
        result = await update_memory(store, workspace=tmp_path, model=MemoryModel(), binding={}, prepare_binding=ready)
        assert result["status"] == "updated"
        assert calls == ["memory_extract", "memory_consolidate"]


@pytest.mark.anyio
async def test_later_correction_replaces_the_previous_entry(tmp_path):
    from agent_runtime.harness.memory import memory_snapshot, update_memory

    class CorrectionModel(MemoryModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            request = self.requests[-1]
            if request.purpose == "memory_extract":
                data = json.loads(request.messages[-1].content.split("\nDATA:\n", 1)[1])
                return HarnessModelResponse(json.dumps({"entries": [{
                    "key": "environment", "title": "Command runner", "text": data[0]["payload"]["text"],
                    "source_item_ids": [data[0]["item_id"]],
                }]}), None, {})
            return await super().dispatch(prepared, delta_sink=delta_sink)

    with RolloutStore(tmp_path / "r.db") as store:
        seed(store, tmp_path, text="Use pip.")
        model = CorrectionModel()
        await update_memory(store, workspace=tmp_path, model=model, binding={})
        seed(store, tmp_path, text="Correction: use uv instead.")
        await update_memory(store, workspace=tmp_path, model=model, binding={})
        entries = memory_snapshot(store, tmp_path)["entries"]
        assert len(entries) == 1
        assert entries[0]["text"] == "Correction: use uv instead."


@pytest.mark.anyio
async def test_memory_budget_rejects_dispatch_before_provider_call(tmp_path):
    from agent_runtime.budget import ResourceUsage
    from agent_runtime.harness.memory import MEMORY_TOKEN_BUDGET, update_memory

    class OverBudget(MemoryModel):
        def prepare(self, request):
            return PreparedModelCall("r", "c", "t", "w", {},
                                     resource_request=ResourceUsage(input_tokens=MEMORY_TOKEN_BUDGET + 1))

        async def dispatch(self, *args, **kwargs):
            pytest.fail("provider must not be called above the maintenance budget")

    with RolloutStore(tmp_path / "r.db") as store:
        seed(store, tmp_path)
        result = await update_memory(store, workspace=tmp_path, model=OverBudget(), binding={})
        assert result["status"] == "failed"
        assert store.read_memory_state(tmp_path)["version"] == 0


@pytest.mark.anyio
async def test_cancelled_provider_outcome_is_not_reissued_and_other_sources_can_continue(tmp_path):
    from agent_runtime.harness.memory import update_memory

    entered = asyncio.Event()

    class SlowModel(MemoryModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            entered.set()
            await asyncio.Event().wait()

    with RolloutStore(tmp_path / "r.db") as store:
        seed(store, tmp_path)
        task = asyncio.create_task(update_memory(store, workspace=tmp_path, model=SlowModel(), binding={}))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store.read_memory_state(tmp_path)["version"] == 0
        contender = MemoryModel()
        blocked = await update_memory(store, workspace=tmp_path, model=contender, binding={})
        assert blocked["status"] == "needs_reconciliation"
        assert contender.requests == []
        later, _ = seed(store, tmp_path, text="unrelated safe source")
        result = await update_memory(store, workspace=tmp_path, model=contender, binding={})
        assert result["status"] == "updated" and result["source_thread_id"] == later.thread_id


@pytest.mark.anyio
async def test_metadata_only_model_snapshot_gets_maintenance_identity(tmp_path):
    from agent_runtime.harness.memory import update_memory

    class MetadataOnly(MemoryModel):
        def snapshot(self, **identity):
            return {"model_id": "test"}

    async def ready(binding):
        assert binding["thread_id"].startswith("thread_")
        assert binding["turn_id"].startswith("turn_")

    with RolloutStore(tmp_path / "r.db") as store:
        seed(store, tmp_path)
        result = await update_memory(store, workspace=tmp_path, model=MetadataOnly(), binding={}, prepare_binding=ready)
        assert result["status"] == "updated"


@pytest.mark.anyio
async def test_terminal_chat_enqueues_durable_work_for_next_session(tmp_path):
    from agent_runtime.harness.memory import MemoryConfig

    database = tmp_path / "r.db"
    model = MemoryModel()
    async with await Session.open(database=database, workspace=tmp_path, model=model,
                                  memory=MemoryConfig(), _background_memory=False,
                                  require_workspace_change=False) as first:
        await first.submit("Remember a stable project convention")
        source_thread = first.thread_id
    with RolloutStore(database, read_only=True) as store:
        jobs = store.read_memory_jobs(tmp_path)
        assert jobs[source_thread]["status"] == "pending"
    async with await Session.open(database=database, workspace=tmp_path, model=model,
                                  memory=MemoryConfig(), _background_memory=False,
                                  require_workspace_change=False) as second:
        result = await second.refresh_memory()
        assert result["status"] == "updated"
        assert second.store.read_memory_jobs(tmp_path)[source_thread]["status"] == "done"


@pytest.mark.anyio
async def test_index_is_bounded_and_details_use_the_frozen_version(tmp_path):
    from agent_runtime.harness.memory import MemoryConfig, create_memory_tool
    from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin

    database = tmp_path / "r.db"
    with RolloutStore(database) as store:
        source, turn = seed(store, tmp_path)
        source_id = store.list_items(turn.turn_id)[0].item_id
        state = store.read_memory_state(tmp_path)
        state["entries"] = [{"key": f"topic-{n}", "title": f"Topic {n}", "text": f"private detail {n}",
                             "source_item_ids": [source_id], "sources": {source_id: source.thread_id}}
                            for n in range(45)]
        store.commit_memory_state(tmp_path, expected_version=0, state=state)
    async with await Session.open(database=database, workspace=tmp_path, model=MemoryModel(),
                                  memory=MemoryConfig(generate_memories=False),
                                  require_workspace_change=False) as session:
        reply = await session.submit("retrieve an old topic")
        snapshot = session.store.read_turn(reply.turn_id).binding_manifest["memory_snapshot"]
        assert snapshot["total_entries"] == 45 and len(snapshot["entries"]) <= 32
        assert "private detail" not in json.dumps(snapshot)
        session.delete_memory("topic-0")
        frozen_turn = session.store.start_turn(thread_id=session.thread_id, user_message="frozen read",
                                              binding_manifest={"memory_snapshot": snapshot})
        orchestrator = ToolOrchestrator(store=session.store, tools={"read_memory": create_memory_tool(session.store)},
                                       execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path))
        output = await orchestrator.execute(turn_id=frozen_turn.turn_id, call=ToolCall(
            tool_call_id="frozen-detail", tool_name="read_memory", arguments={"query": "private detail 0"},
            origin=ToolCallOrigin(request_id="r", toolset_revision="v", exposed_tool_names=("read_memory",))))
        assert not output.is_error
        payload = json.loads(output.content[0].data["text"])
        assert payload["entries"][0]["text"] == "private detail 0"
        assert payload["version"] == snapshot["version"]


@pytest.mark.anyio
async def test_old_size_skip_is_migrated_without_reextracting_normal_processed_history(tmp_path):
    import hashlib

    from agent_runtime.harness.memory import MAX_SOURCE_BYTES, update_memory

    with RolloutStore(tmp_path / "r.db") as store:
        thread, turn = seed(store, tmp_path, text="big" * 20000)
        turns = store.list_turns(thread.thread_id)
        revision = hashlib.sha256(json.dumps({
            "turns": [(t.turn_id, t.applied_thread_sequence) for t in turns],
            "source_byte_limit": MAX_SOURCE_BYTES,
        }, sort_keys=True).encode()).hexdigest()
        state = store.read_memory_state(tmp_path)
        state["processed"][thread.thread_id] = revision
        state["skipped_sources"] = [{"thread_id": thread.thread_id, "revision": revision,
                                     "reason": "source_bytes_exceeded", "limit": MAX_SOURCE_BYTES}]
        store.commit_memory_state(tmp_path, expected_version=0, state=state)
        result = await update_memory(store, workspace=tmp_path, model=MemoryModel(), binding={})
        assert result["status"] == "updated" and result["has_more"]
        assert store.read_memory_state(tmp_path)["skipped_sources"] == []
        assert (await update_memory(store, workspace=tmp_path, model=MemoryModel(), binding={}))["status"] == "updated"


def test_expired_worker_cannot_publish_cursor_or_mutate_job(tmp_path):
    import time

    from agent_runtime.harness.rollout import ContextSourceChangedError

    with RolloutStore(tmp_path / "r.db") as store:
        thread, _ = seed(store, tmp_path)
        assert store.claim_memory_job(tmp_path, owner="old", now=time.time() - 10, lease_seconds=1)
        assert store.claim_memory_job(tmp_path, owner="new", now=time.time(), lease_seconds=300)
        job = store.write_memory_job(tmp_path, thread.thread_id, {"status": "running"}, owner="new")
        with pytest.raises(ContextSourceChangedError):
            store.write_memory_job(tmp_path, thread.thread_id, {"status": "done"}, owner="old")
        with pytest.raises(ContextSourceChangedError):
            store.commit_memory_state(tmp_path, expected_version=0,
                                      state={**store.read_memory_state(tmp_path), "source_cursors": {"bad": 1}},
                                      claim_owner="old", job_thread_id=thread.thread_id,
                                      job_state={"status": "done"}, job_revision=job["revision"])
        assert store.read_memory_state(tmp_path)["version"] == 0
        assert store.read_memory_jobs(tmp_path)[thread.thread_id] == job


@pytest.mark.anyio
async def test_completed_extraction_is_reused_after_cancel_before_consolidation_dispatch(tmp_path):
    from agent_runtime.harness.memory import update_memory

    database = tmp_path / "r.db"
    entered = asyncio.Event()
    async def ready(binding):
        if binding["maintenance_purpose"] == "memory_consolidate":
            entered.set()
            await asyncio.Event().wait()
    first = MemoryModel()
    with RolloutStore(database) as store:
        thread, _ = seed(store, tmp_path)
        worker = asyncio.create_task(update_memory(store, workspace=tmp_path, model=first, binding={},
                                                  prepare_binding=ready))
        await entered.wait()
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        assert store.read_memory_jobs(tmp_path)[thread.thread_id]["status"] == "pending"
        assert store.read_memory_state(tmp_path)["version"] == 0
    second = MemoryModel()
    with RolloutStore(database) as store:
        result = await update_memory(store, workspace=tmp_path, model=second, binding={})
        assert result["status"] == "updated"
        assert [r.purpose for r in second.requests] == ["memory_consolidate"]
        assert len(result["maintenance_turn_ids"]) == 3


def test_version_zero_never_falls_back_to_current_memory(tmp_path):
    with RolloutStore(tmp_path / "r.db") as store:
        state = store.read_memory_state(tmp_path)
        store.commit_memory_state(tmp_path, expected_version=0, state={**state, "entries": [{"key": "new"}]})
        assert store.read_memory_state(tmp_path, version=0)["entries"] == []
        with pytest.raises(KeyError):
            store.read_memory_state(tmp_path, version=98765)


@pytest.mark.anyio
async def test_completed_extraction_recovers_across_candidate_cache_crash_window(tmp_path, monkeypatch):
    from agent_runtime.harness.memory import update_memory

    database = tmp_path / "r.db"
    with RolloutStore(database) as store:
        seed(store, tmp_path)
        write = store.write_memory_job
        def crash_before_cache(*args, **kwargs):
            if "candidates" in args[2]:
                raise asyncio.CancelledError()
            return write(*args, **kwargs)
        monkeypatch.setattr(store, "write_memory_job", crash_before_cache)
        with pytest.raises(asyncio.CancelledError):
            await update_memory(store, workspace=tmp_path, model=MemoryModel(), binding={})
    with RolloutStore(database) as store:
        model = MemoryModel()
        assert (await update_memory(store, workspace=tmp_path, model=model, binding={}))["status"] == "updated"
        assert [r.purpose for r in model.requests] == ["memory_consolidate"]


@pytest.mark.anyio
async def test_exhausted_batch_does_not_block_other_sources_or_reset_its_budget(tmp_path):
    from agent_runtime.harness.memory import update_memory

    class InvalidModel(MemoryModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            return HarnessModelResponse("bad JSON", None, {"input_tokens": 12000, "output_tokens": 0})
    with RolloutStore(tmp_path / "r.db") as store:
        blocked, _ = seed(store, tmp_path)
        safe, _ = seed(store, tmp_path, text="later safe convention")
        first = await update_memory(store, workspace=tmp_path, model=InvalidModel(), binding={})
        assert first["status"] == "failed"
        job = store.read_memory_jobs(tmp_path)[blocked.thread_id]
        store.write_memory_job(tmp_path, blocked.thread_id, {**job, "retry_after": 0})
        second = await update_memory(store, workspace=tmp_path, model=InvalidModel(), binding={})
        assert second["status"] == "budget_exhausted"
        failed_turns = store.read_memory_jobs(tmp_path)[blocked.thread_id]["turn_ids"]
        assert sum(store.read_budget_state(t).exposure.total_tokens for t in failed_turns) == 24000
        result = await update_memory(store, workspace=tmp_path, model=MemoryModel(), binding={})
        assert result["status"] == "updated" and result["source_thread_id"] == safe.thread_id
        assert store.read_memory_jobs(tmp_path)[blocked.thread_id]["turn_ids"] == failed_turns


@pytest.mark.anyio
async def test_new_topic_preserves_a_bank_larger_than_navigation_capacity(tmp_path):
    from agent_runtime.harness.memory import update_memory

    with RolloutStore(tmp_path / "r.db") as store:
        source, turn = seed(store, tmp_path)
        source_id = store.list_items(turn.turn_id)[0].item_id
        originals = [{"key": f"topic-{n}", "title": f"Topic {n}", "text": f"stable original {n}",
                      "source_item_ids": [source_id], "sources": {source_id: source.thread_id}}
                     for n in range(45)]
        state = store.read_memory_state(tmp_path)
        store.commit_memory_state(tmp_path, expected_version=0, state={**state, "entries": originals})
        result = await update_memory(store, workspace=tmp_path, model=MemoryModel(), binding={})
        assert result["status"] == "updated"
        entries = store.read_memory_state(tmp_path)["entries"]
        assert len(entries) == 46 and entries[:45] == originals

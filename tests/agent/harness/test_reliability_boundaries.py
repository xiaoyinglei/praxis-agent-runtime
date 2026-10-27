from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import replace

import pytest

from agent_runtime import Agent
from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, RolloutStore, TurnExecutor
from agent_runtime.harness import rollout as rollout_module
from agent_runtime.harness.events import RolloutEventReader
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import NormalizedToolOutput, ToolCall, ToolCallOrigin, ToolContentBlock
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel
from tests.agent.harness.test_tool_orchestrator import _read_tool


def _turn(store, workspace):
    thread = store.create_thread(workspace=workspace)
    return store.start_turn(thread_id=thread.thread_id, user_message="inspect", binding_manifest={})


def test_corrupt_artifact_does_not_block_unrelated_turn(tmp_path):
    with RolloutStore(tmp_path / "run.db") as store:
        old = _turn(store, tmp_path)
        artifact = store.commit_artifact(turn_id=old.turn_id, content=b"old", media_type="text/plain", name="old")
        store.artifact_blob_path(artifact.artifact_id).unlink()
        current = _turn(store, tmp_path)
        operation = store.prepare_model_operation(turn_id=current.turn_id, request_hash="r", context_hash="c",
                                                  tool_hash="t", wire_hash="w", request_ref={"request_id": "new"})
        assert store.dispatch_model_attempt(operation.operation_id).status == "dispatched"
        assert not store.verify().valid  # Global audit still diagnoses the damaged archive.


@pytest.mark.anyio
@pytest.mark.parametrize("kind", [TurnExecutor, ToolOrchestrator])
async def test_store_writer_wait_yields_to_event_loop(tmp_path, kind):
    with RolloutStore(tmp_path / "run.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        store.run_sync(lambda: store._connection.execute("PRAGMA busy_timeout=150").close())
        other = sqlite3.connect(tmp_path / "run.db", isolation_level=None)
        other.execute("BEGIN IMMEDIATE")
        executor = object.__new__(kind)
        executor._store, executor._event_dispatcher = store, None

        async def release():
            await asyncio.sleep(0.02)
            other.execute("ROLLBACK")

        task = asyncio.create_task(release())
        try:
            turn = await executor._commit(lambda: store.start_turn(
                thread_id=thread.thread_id, user_message="new", binding_manifest={}))
            assert turn.status == "running"
            assert task.done()
            assert store.run_sync(lambda: store._connection.execute("PRAGMA busy_timeout").fetchone())[0] == 150
        finally:
            await task
            other.close()


def test_live_projection_does_not_reload_completed_turns(tmp_path, monkeypatch):
    with RolloutStore(tmp_path / "run.db") as store:
        thread = store.create_thread(workspace=tmp_path)
        reader = RolloutEventReader(store)
        for _ in range(12):
            turn = store.start_turn(thread_id=thread.thread_id, user_message="old", binding_manifest={})
            store.complete_turn(turn_id=turn.turn_id, answer="old answer")
        mutation = store.capture_mutation(lambda: store.start_turn(
            thread_id=thread.thread_id, user_message="current", binding_manifest={}))
        original = store.list_records
        loaded = []

        def count(thread_id):
            records = original(thread_id)
            loaded.extend(records)
            return records

        monkeypatch.setattr(store, "list_records", count)
        events = reader.project_committed_batch(mutation.records)
        assert events
        assert not any(record.turn_id != mutation.value.turn_id for record in loaded)


@pytest.mark.parametrize("batches", [24, 48])
@pytest.mark.parametrize("split_items", [False, True])
def test_long_turn_live_projection_work_tracks_new_records(tmp_path, monkeypatch, batches, split_items):
    with RolloutStore(tmp_path / "run.db") as store:
        turn = _turn(store, tmp_path)
        early_completion = next(record for record in store.list_turn_records(turn.turn_id)
                                if record.record_type == "item_completed")
        loaded = []
        projected = []
        original_snapshot = rollout_module._record_snapshot
        original_project = RolloutEventReader._project_record

        def snapshot(row):
            record = original_snapshot(row)
            loaded.append(record.record_id)
            return record

        def project(reader, record, **kwargs):
            projected.append(record.record_id)
            return original_project(reader, record, **kwargs)

        monkeypatch.setattr(rollout_module, "_record_snapshot", snapshot)
        monkeypatch.setattr(RolloutEventReader, "_project_record", project)
        total_records = 0
        total_loaded = 0
        total_projected = 0
        sql_steps = 0

        def sql_step():
            nonlocal sql_steps
            sql_steps += 1
            return 0

        for _ in range(batches):
            mutation = store.capture_mutation(lambda: store.reserve_resource_unit(turn.turn_id, "reads", 256))
            # Count only live projection, independently of the write/reducer.
            loaded.clear()
            projected.clear()
            store.run_sync(lambda: store._connection.set_progress_handler(sql_step, 1))
            try:
                parts = tuple((record,) for record in mutation.records) if split_items else (mutation.records,)
                for part in parts:
                    assert RolloutEventReader(store).project_committed_batch(part) == ()
            finally:
                store.run_sync(lambda: store._connection.set_progress_handler(None, 0))
            total_records += len(mutation.records)
            total_loaded += len(loaded)
            total_projected += len(projected)

        assert total_loaded <= total_records, (batches, total_records, total_loaded)
        assert total_projected == total_records, (batches, total_records, total_projected)
        assert sql_steps <= batches * 100, (batches, sql_steps)
        # A delayed completion must find its old start by identity even after
        # many unrelated Items. Counting VM work catches hidden SQL full scans.
        sql_steps = 0
        store.run_sync(lambda: store._connection.set_progress_handler(sql_step, 1))
        try:
            assert RolloutEventReader(store).project_committed_batch((early_completion,)) == ()
        finally:
            store.run_sync(lambda: store._connection.set_progress_handler(None, 0))
        assert sql_steps <= 100, sql_steps


@pytest.mark.anyio
async def test_live_projection_matches_replay_across_tool_commits_and_reopened_turn(tmp_path, monkeypatch):
    database = tmp_path / "run.db"
    dispatcher = TurnEventDispatcher(capacity=128)
    stream = dispatcher.subscribe_controlling()

    def no_history(*args, **kwargs):
        pytest.fail("live delivery must not load Turn or Thread history")

    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        executor = object.__new__(TurnExecutor)
        executor._store, executor._event_dispatcher = store, dispatcher
        with monkeypatch.context() as patch:
            patch.setattr(store, "list_records", no_history)
            patch.setattr(store, "list_turn_records", no_history)
            turn = await executor._commit(lambda: store.start_turn(
                thread_id=thread.thread_id, user_message="inspect", binding_manifest={}))
            operation = await executor._commit(lambda: store.prepare_model_operation(
                turn_id=turn.turn_id, request_hash="r", context_hash="c", tool_hash="t", wire_hash="w",
                request_ref={"request_id": "model-request"},
            ))
            attempt = await executor._commit(lambda: store.dispatch_model_attempt(operation.operation_id))
            await executor._commit(lambda: store.complete_model_attempt(
                operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation,
                text="answer", provider_response_id="response", usage={"input_tokens": 1, "output_tokens": 1},
                reasoning_content="inspect the file", plan_content="read then answer",
            ))
            orchestrator = ToolOrchestrator(
                store=store, tools={"read_file": _read_tool(workspace=tmp_path)},
                execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path),
                event_dispatcher=dispatcher,
            )
            for index in range(4):
                result = await orchestrator.execute(turn_id=turn.turn_id, call=ToolCall(
                    tool_call_id=f"read-{index}", tool_name="read_file", arguments={"path": "present.txt"},
                    origin=ToolCallOrigin(request_id="tool-request", toolset_revision="tools",
                                          exposed_tool_names=("read_file",)),
                ))
                assert not result.is_error
            interaction = await executor._commit(lambda: store.request_clarification(
                turn_id=turn.turn_id, question="continue?"))

    with RolloutStore(database) as reopened:
        executor = object.__new__(TurnExecutor)
        executor._store, executor._event_dispatcher = reopened, dispatcher
        with monkeypatch.context() as patch:
            patch.setattr(reopened, "list_records", no_history)
            patch.setattr(reopened, "list_turn_records", no_history)
            await executor._commit(lambda: reopened.resolve_clarification(
                turn_id=turn.turn_id, request_id=interaction.request_id, response="continue"))
            source = reopened.list_model_operations(turn.turn_id)[0].response_item_id
            await executor._commit(lambda: reopened.complete_turn(
                turn_id=turn.turn_id, answer="answer", source_item_id=source,
            ))
        live = []
        while not stream.empty:
            live.append(stream.receive_nowait())
        replay = RolloutEventReader(reopened).read(thread.thread_id)
        assert [replace(event, sequence=0) for event in live] == [
            replace(event.event, sequence=0) for event in replay
        ]
        assert {event.type.value for event in live} >= {"turn_paused", "turn_resumed", "turn_completed"}
        assert sum(event.type.value == "item_completed" and event.item_kind.value == "tool" for event in live) == 4
        reopened.rebuild_projections()
        assert RolloutEventReader(reopened).read(thread.thread_id) == replay
        assert reopened.verify().valid
    dispatcher.close()


@pytest.mark.anyio
@pytest.mark.parametrize("parallel", [False, True])
async def test_externalized_output_commits_durable_artifact_and_reopens(tmp_path, parallel):
    database = tmp_path / "run.db"
    body = "retained result\n" + "x" * 10000 + "\nlast record"
    tool = replace(_read_tool(workspace=tmp_path), max_model_output_bytes=4096,
                   normalize_output=lambda _: NormalizedToolOutput(
                       content=(ToolContentBlock(type="text", data={"text": body}),)))
    with RolloutStore(database) as store:
        turn = _turn(store, tmp_path)
        orchestrator = ToolOrchestrator(store=store, tools={"read_file": tool},
                                       execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path))
        calls = tuple(ToolCall(
            tool_call_id=f"read-{index}", tool_name="read_file", arguments={"path": "present.txt"},
            origin=ToolCallOrigin(request_id="request", toolset_revision="tools", exposed_tool_names=("read_file",)),
        ) for index in range(2 if parallel else 1))
        results = (await orchestrator.execute_batch(turn_id=turn.turn_id, calls=calls) if parallel
                   else (await orchestrator.execute(turn_id=turn.turn_id, call=calls[0]),))
        assert all(not result.is_error and result.truncated for result in results)
        assert all(operation.status == "succeeded" for operation in store.list_tool_operations(turn.turn_id))
        references = [result.attachments[0].artifact_id for result in results]
        assert all(json.loads(store.read_artifact(reference))["content"][0]["data"]["text"] == body
                   for reference in references)
        assert store.verify().valid
        store.rebuild_projections()
        assert store.verify().valid
    with RolloutStore(database) as reopened:
        assert all(json.loads(reopened.read_artifact(reference))["content"][0]["data"]["text"] == body
                   for reference in references)
        assert reopened.verify().valid


@pytest.mark.anyio
async def test_public_agent_large_file_output_survives_next_dispatch_and_reopen(tmp_path, monkeypatch):
    # JSON escaping can exceed the model window even inside read_file's byte cap.
    body = "\\" * 600000
    (tmp_path / "large.txt").write_text(body)

    class ReadThenAnswer(PatchThenAnswerModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            first = prepared.request_ref["request_id"].endswith(":step:1")
            return HarnessModelResponse(
                text="" if first else "file inspected", provider_response_id="read" if first else "answer",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(HarnessToolCall(id="read-large", name="read_file", arguments={
                    "path": "large.txt", "max_bytes": 1000000,
                }),) if first else (),
            )

    database = tmp_path / "run.db"
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", ReadThenAnswer)
    result = await agent.run("inspect large.txt", require_workspace_change=False)
    assert result.status == "done", result.diagnostics
    assert len(result.tool_calls) == 1
    assert not result.tool_calls[0].is_error and result.tool_calls[0].truncated
    with RolloutStore(database) as store:
        [artifact] = store.list_artifacts(result.turn_id)
        assert json.loads(store.read_artifact(artifact.artifact_id))["structured_content"]["content"] == body
        assert store.verify().valid
        store.rebuild_projections()
        assert store.verify().valid
    assert await Agent(workspace_path=tmp_path, checkpoint_db=database).read_result(result.turn_id) == result


@pytest.mark.anyio
@pytest.mark.parametrize("batch", [False, True])
async def test_cancel_during_artifact_admission_commits_completed_result(tmp_path, batch):
    database = tmp_path / "run.db"
    with RolloutStore(database) as store:
        other = sqlite3.connect(database, isolation_level=None)
        turn = _turn(store, tmp_path)

        async def run(_arguments):
            other.execute("BEGIN IMMEDIATE")
            return {"text": "x" * 10000}

        tool = replace(_read_tool(workspace=tmp_path), run=run, max_model_output_bytes=4096)
        orchestrator = ToolOrchestrator(store=store, tools={"read_file": tool},
                                       execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path))
        original_sink = orchestrator._executor._artifact_sink
        reached_sink = asyncio.Event()

        async def blocked_sink(content, media_type, name):
            reached_sink.set()
            return await original_sink(content, media_type, name)

        orchestrator._executor._artifact_sink = blocked_sink
        call = ToolCall(tool_call_id="read", tool_name="read_file", arguments={"path": "present.txt"},
                        origin=ToolCallOrigin(request_id="r", toolset_revision="v", exposed_tool_names=("read_file",)))
        task = asyncio.create_task(orchestrator.execute_batch(turn_id=turn.turn_id, calls=(call,)) if batch
                                   else orchestrator.execute(turn_id=turn.turn_id, call=call))
        try:
            await asyncio.wait_for(reached_sink.wait(), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()  # Ownership is retained while the writer is busy.
        finally:
            other.execute("ROLLBACK")
            other.close()
            with pytest.raises(asyncio.CancelledError):
                await task
        [operation] = store.list_tool_operations(turn.turn_id)
        assert operation.status == "succeeded"
        [result] = [item for item in store.list_items(turn.turn_id) if item.kind == "tool_result"]
        assert not result.payload["is_error"]
        assert result.payload["metadata"]["execution_cancellation_requested"]
        [artifact] = store.list_artifacts(turn.turn_id)
        assert json.loads(store.read_artifact(artifact.artifact_id))["content"][0]["data"]["text"] == "x" * 10000
        assert store.verify().valid

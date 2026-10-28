from __future__ import annotations

from dataclasses import replace

import pytest

from agent_runtime import Agent
from agent_runtime.harness import RolloutContextManager, RolloutEventReader, RuntimeState
from agent_runtime.harness import rollout as rollout_module
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
from agent_runtime.history import read_chat_messages
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
from tests.agent.harness.test_public_result_projection import PublicAnswerModel
from tests.agent.harness.test_stream_projection import _complete_model_response
from tests.agent.harness.test_tool_orchestrator import _read_tool


def _events(values):
    return [(value.record_id, replace(value.event, sequence=0)) for value in values]


def _turn(store, workspace, thread_id=None):
    thread_id = thread_id or store.create_thread(workspace=workspace).thread_id
    return store.start_turn(thread_id=thread_id, user_message="answer", binding_manifest={})


def test_turn_completion_references_the_original_output_without_copying_it(tmp_path):
    with RuntimeState.open_sync(tmp_path / "events.db") as store:
        turn = _turn(store, tmp_path)
        _complete_model_response(store, turn_id=turn.turn_id, text="answer")
        source = store.list_model_operations(turn.turn_id)[0].response_item_id
        mutation = store.capture_mutation(lambda: store.complete_turn(
            turn_id=turn.turn_id, answer="answer", source_item_id=source,
        ))
        assert not any(record.record_type == "item_started" for record in mutation.records)
        assert mutation.records[-1].payload["answer_item_id"] == source
        assert store.read_turn_answer(turn.turn_id).item_id == source
        assert not any(item.kind == "agent_message" for item in store.list_items(turn.turn_id))
        assert store.verify().valid
        store.rebuild_projections()
        assert store.read_turn_answer(turn.turn_id).item_id == source


def test_equal_text_does_not_merge_independent_outputs(tmp_path):
    with RuntimeState.open_sync(tmp_path / "events.db") as store:
        turn = _turn(store, tmp_path)
        _complete_model_response(store, turn_id=turn.turn_id, text="same")
        store.complete_turn(turn_id=turn.turn_id, answer="same")
        outputs = [value.event for value in RolloutEventReader(store).read(turn.thread_id)
                   if value.event.type.value == "item_completed"]
        assert len(outputs) == 2
        assert len({event.item_id for event in outputs}) == 2
        assert read_chat_messages(tmp_path / "events.db", thread_id=turn.thread_id, workspace=tmp_path) == (
            ("user", "answer"), ("answer", "same"), ("answer", "same"),
        )


def test_repeated_answer_across_turns_survives_live_and_replay(tmp_path):
    with RuntimeState.open_sync(tmp_path / "events.db") as store:
        first = _turn(store, tmp_path)
        _complete_model_response(store, turn_id=first.turn_id, text="same")
        store.complete_turn(turn_id=first.turn_id, answer="same")
        second = _turn(store, tmp_path, first.thread_id)
        mutation = store.capture_mutation(lambda: store.complete_turn(turn_id=second.turn_id, answer="same"))
        reader = RolloutEventReader(store)
        live = reader.project_committed_batch(mutation.records)
        record_ids = {record.record_id for record in mutation.records}
        replay = tuple(value for value in reader.read(first.thread_id) if value.record_id in record_ids)
        assert any(value.event.type.value == "item_completed" for value in live)
        assert _events(live) == _events(replay)


def test_followup_context_uses_accepted_output_identity_and_excludes_unaccepted_text(tmp_path):
    with RuntimeState.open_sync(tmp_path / "events.db") as store:
        first = _turn(store, tmp_path)
        _complete_model_response(store, turn_id=first.turn_id, text="the answer is 42")
        source = store.list_model_operations(first.turn_id)[0].response_item_id
        store.complete_turn(turn_id=first.turn_id, answer="the answer is 42", source_item_id=source)
        second = _turn(store, tmp_path, first.thread_id)
        _complete_model_response(store, turn_id=second.turn_id, text="unaccepted candidate")
        messages = RolloutContextManager(store).build(second.turn_id)
        assert [(message.role, message.content) for message in messages] == [
            ("user", "answer"), ("assistant", "the answer is 42"), ("user", "answer"),
        ]


def test_fork_and_compaction_keep_the_accepted_output_as_the_original_item(tmp_path):
    with RuntimeState.open_sync(tmp_path / "events.db") as store:
        first = _turn(store, tmp_path)
        _complete_model_response(store, turn_id=first.turn_id, text="accepted answer")
        source = store.list_model_operations(first.turn_id)[0].response_item_id
        store.complete_turn(turn_id=first.turn_id, answer="accepted answer", source_item_id=source)
        fork = store.fork_thread(from_turn_id=first.turn_id)
        current = _turn(store, tmp_path, fork.thread_id)
        manager = RolloutContextManager(store)
        assert any(message.content == "accepted answer" for message in manager.build(current.turn_id))
        candidate = manager._candidate(current.turn_id, 1, summary_override="previous question and accepted answer",
                                       require_reduction=False)
        manager.commit_compaction(candidate)
        assert store.read_item(source).payload["text"] == "accepted answer"
        assert store.read_turn_answer(first.turn_id).item_id == source
        assert store.verify().valid


@pytest.mark.anyio
async def test_public_agent_reopens_with_one_original_answer_in_followup_context(tmp_path, monkeypatch):
    requests = []

    class CapturingModel(PublicAnswerModel):
        def prepare(self, request):
            requests.append(request)
            return super().prepare(request)

    database = tmp_path / "events.db"
    first_agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(first_agent, "_model_factory", CapturingModel)
    first = await first_agent.run("first question", require_workspace_change=False)
    assert first.status == "done"
    second_agent = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(second_agent, "_model_factory", CapturingModel)
    second = await second_agent.run("followup question", previous_turn_id=first.turn_id,
                                    require_workspace_change=False)
    assert second.status == "done"
    assert sum(message.role == "assistant" and message.content == first.answer
               for message in requests[-1].messages) == 1
    assert await second_agent.read_result(first.turn_id) == first
    assert await second_agent.read_result(second.turn_id) == second
    async with (await RuntimeState.open(database, read_only=True)) as store:
        assert not any(item.kind == "agent_message" for item in store.list_items(first.turn_id))
        assert not any(item.kind == "agent_message" for item in store.list_items(second.turn_id))


@pytest.mark.parametrize("wrong_turn", [False, True])
def test_answer_reference_rejection_rolls_back_the_entire_completion(tmp_path, wrong_turn):
    with RuntimeState.open_sync(tmp_path / "events.db") as store:
        first = _turn(store, tmp_path)
        _complete_model_response(store, turn_id=first.turn_id, text="answer")
        source = store.list_model_operations(first.turn_id)[0].response_item_id
        if wrong_turn:
            store.complete_turn(turn_id=first.turn_id, answer="answer", source_item_id=source)
            current = _turn(store, tmp_path, first.thread_id)
        else:
            current = first
        before = store.list_records(first.thread_id)
        with pytest.raises(ValueError, match="matching completed response"):
            store.complete_turn(turn_id=current.turn_id, answer="answer" if wrong_turn else "different",
                                source_item_id=source)
        assert store.list_records(first.thread_id) == before
        assert store.read_turn(current.turn_id).status == "running"
        assert store.verify().valid


def test_current_records_are_self_contained_even_after_their_projections_are_unavailable(tmp_path, monkeypatch):
    with RuntimeState.open_sync(tmp_path / "events.db") as store:
        turn = _turn(store, tmp_path)
        _complete_model_response(store, turn_id=turn.turn_id, text="answer")
        records = store.list_records(turn.thread_id)
        expected = RolloutEventReader(store).project_committed_batch(records)

        def no_dependency(*args, **kwargs):
            pytest.fail("current canonical records must not read historical or mutable projection state")

        for method in ("read_event_reference", "find_model_response_item", "read_item", "read_tool_operation"):
            monkeypatch.setattr(store, method, no_dependency)
        actual = tuple(value for record in records
                       for value in RolloutEventReader(store).project_committed_batch((record,)))
        assert _events(actual) == _events(expected)


@pytest.mark.anyio
async def test_tool_event_history_ignores_current_operation_state(tmp_path, monkeypatch):
    async with (await RuntimeState.open(tmp_path / "events.db")) as store:
        turn = _turn(store, tmp_path)
        orchestrator = ToolOrchestrator(
            store=store, tools={"read_file": _read_tool(workspace=tmp_path)},
            execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path),
        )
        result = await orchestrator.execute(turn_id=turn.turn_id, call=ToolCall(
            tool_call_id="read", tool_name="read_file", arguments={"path": "file"},
            origin=ToolCallOrigin(request_id="r", toolset_revision="t", exposed_tool_names=("read_file",)),
        ))
        assert not result.is_error
        records = store.list_records(turn.thread_id)
        reader = RolloutEventReader(store)
        expected = reader.project_committed_batch(records)

        def no_dependency(*args, **kwargs):
            pytest.fail("current event history must be independent of mutable operation state")

        for method in ("read_event_reference", "read_item", "read_tool_operation", "find_model_response_item"):
            monkeypatch.setattr(store, method, no_dependency)
        assert _events(reader.read(turn.thread_id)) == _events(expected)
        assert _events(tuple(event for record in records for event in reader.project_committed_batch((record,)))) == (
            _events(expected)
        )


@pytest.mark.parametrize("global_tail", [False, True])
def test_cursor_tail_loads_only_new_records_and_uses_a_storage_page(tmp_path, monkeypatch, global_tail):
    with RuntimeState.open_sync(tmp_path / "events.db") as store:
        first = _turn(store, tmp_path)
        for _ in range(48):
            store.reserve_resource_unit(first.turn_id, "reads", 256)
        store.complete_turn(turn_id=first.turn_id, answer="first")
        reader = RolloutEventReader(store)
        acknowledged = reader.read(first.thread_id)[-1]
        current = _turn(store, tmp_path, first.thread_id)
        store.complete_turn(turn_id=current.turn_id, answer="second")
        expected = tuple(value for value in reader.read(first.thread_id) if value.record_id > acknowledged.record_id)

        def no_history(*args, **kwargs):
            pytest.fail("cursor tail must not reload the complete Thread")

        monkeypatch.setattr(store, "list_records", no_history)
        loaded = []
        original_snapshot = rollout_module._record_snapshot

        def snapshot(row):
            record = original_snapshot(row)
            loaded.append(record.record_id)
            return record

        monkeypatch.setattr(rollout_module, "_record_snapshot", snapshot)
        if global_tail:
            actual = reader.read_global(after_record_id=acknowledged.record_id)
        else:
            actual = reader.read(first.thread_id, after=acknowledged.cursor)
        assert _events(actual) == _events(expected)
        assert all(record_id > acknowledged.record_id for record_id in loaded)
        if not global_tail:
            page = reader.read(first.thread_id, after=acknowledged.cursor, limit=1)
            assert len(page) == 1
            assert _events(page) == _events(expected[:1])


@pytest.mark.parametrize("global_tail", [False, True])
def test_history_read_freezes_its_upper_bound_before_concurrent_append(tmp_path, monkeypatch, global_tail):
    database = tmp_path / "events.db"
    with RuntimeState.open_sync(database) as store, RuntimeState.open_sync(database) as writer:
        turn = _turn(store, tmp_path)
        reader = RolloutEventReader(store)
        expected = reader.read(turn.thread_id)
        method = "list_global_records" if global_tail else "list_record_tail"
        original = getattr(store, method)
        appended = False

        def append_before_query(*args, **kwargs):
            nonlocal appended
            if not appended:
                appended = True
                writer.request_clarification(turn_id=turn.turn_id, question="continue?")
            return original(*args, **kwargs)

        monkeypatch.setattr(store, method, append_before_query)
        actual = reader.read_global() if global_tail else reader.read(turn.thread_id)
        assert _events(actual) == _events(expected)
        following = (reader.read_global(after_record_id=expected[-1].record_id) if global_tail
                     else reader.read(turn.thread_id, after=expected[-1].cursor))
        assert [event.event.type.value for event in following] == ["interaction_requested", "turn_paused"]
        assert following[0].data["kind"] == "clarification"
        assert following[0].data["request"] == {"question": "continue?"}
        assert following[1].data["request_id"] == following[0].data["request_id"]


@pytest.mark.parametrize("global_tail", [False, True])
def test_storage_pages_concatenate_to_the_same_event_history(tmp_path, global_tail):
    with RuntimeState.open_sync(tmp_path / "events.db") as store:
        turn = _turn(store, tmp_path)
        store.complete_turn(turn_id=turn.turn_id, answer="first")
        later = _turn(store, tmp_path, turn.thread_id)
        store.complete_turn(turn_id=later.turn_id, answer="second")
        reader = RolloutEventReader(store)
        expected = reader.read_global() if global_tail else reader.read(turn.thread_id)
        result = []
        while True:
            page = (reader.read_global(after_record_id=result[-1].record_id if result else 0, limit=2)
                    if global_tail else reader.read(
                        turn.thread_id, after=result[-1].cursor if result else None, limit=2,
                    ))
            if not page:
                break
            assert len(page) <= 2
            result.extend(page)
        assert _events(result) == _events(expected)

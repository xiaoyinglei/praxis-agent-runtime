from pathlib import Path

from agent_runtime.harness import RolloutEventReader, RuntimeState
from agent_runtime.harness.protocol import HarnessModelResponse, restore_model_response
from agent_runtime.history import read_chat_messages
from agent_runtime.modeling.events import ModelOutput


def test_response_persistence_has_one_ordered_source_and_no_aggregate_events(tmp_path: Path) -> None:
    path = tmp_path / "rollout.sqlite3"
    response = HarnessModelResponse(
        outputs=(
            ModelOutput("reason", "reasoning", "checked", provider_data={"signature": "private"}),
            ModelOutput("first", "text", "Found. "),
            ModelOutput(
                "call",
                "tool_call",
                tool_call_id="call-1",
                tool_name="inspect",
                arguments={"path": "src"},
                provider_data={"provider_id": "private-call"},
            ),
            ModelOutput("last", "text", "Fixed."),
        ),
        provider_response_id="response-1",
        usage={"output_tokens": 12},
        stop_reason="stop",
    )
    with RuntimeState.open_sync(path) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="fix", binding_manifest={})
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="r",
            context_hash="c",
            tool_hash="t",
            wire_hash="w",
            request_ref={"input_revision": 1, "request_id": "request-1"},
        )
        attempt = store.dispatch_model_attempt(operation.operation_id, worker_id="worker", lease_seconds=30)
        assert store.complete_model_attempt(
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            response=response,
        )
        saved = store.read_item(store.list_model_operations(turn.turn_id)[0].response_item_id)
        assert set(saved.payload) == {
            "outputs",
            "provider_response_id",
            "usage",
            "response_status",
            "incomplete_reason",
            "stop_reason",
        }
        store.complete_turn(turn_id=turn.turn_id, answer=response.text, source_item_id=saved.item_id)
        assert store.read_turn_answer(turn.turn_id).item_id == saved.item_id
        assert not any(item.kind == "agent_message" for item in store.list_items(turn.turn_id))

    with RuntimeState.open_sync(path) as store:
        answer = store.read_turn_answer(turn.turn_id)
        restored = restore_model_response(answer.payload)
        assert restored == response
        assert [output.output_id for output in restored.outputs] == ["reason", "first", "call", "last"]
        assert restored.outputs[0].provider_data["signature"] == "private"
        assert restored.tool_calls[0].id == "call-1"
        events = [entry.event for entry in RolloutEventReader(store).read(thread.thread_id)]
        completed_text = [
            event
            for event in events
            if event.type.value == "item_completed" and event.data.get("output_kind") == "text"
        ]
        assert [event.data["content"] for event in completed_text] == ["Found. ", "Fixed."]
        assert len({event.item_id for event in completed_text}) == 2
        assert not any(event.data.get("content") == response.text for event in events)
        assert all("private" not in str(event.data) for event in events)
    assert read_chat_messages(path, thread_id=thread.thread_id, workspace=tmp_path) == (
        ("user", "fix"),
        ("answer", "Found. Fixed."),
    )


def test_internal_summary_uses_outputs_without_public_output_lifecycle(tmp_path: Path) -> None:
    path = tmp_path / "rollout.sqlite3"
    response = HarnessModelResponse(
        outputs=(ModelOutput("summary", "text", "retained facts"),), status="incomplete", incomplete_reason="max_tokens"
    )
    with RuntimeState.open_sync(path) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="fix", binding_manifest={})
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="r",
            context_hash="c",
            tool_hash="t",
            wire_hash="w",
            request_ref={"input_revision": 1, "request_id": "summary-1", "purpose": "context_summary"},
        )
        attempt = store.dispatch_model_attempt(operation.operation_id, worker_id="worker", lease_seconds=30)
        assert store.complete_model_attempt(
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            response=response,
        )
        saved = store.read_item(store.list_model_operations(turn.turn_id)[0].response_item_id)
        assert saved.kind == "context_summary_response"
        assert "text" not in saved.payload
        assert restore_model_response(saved.payload) == response
        assert not any(item.kind == "model_output" for item in store.list_items(turn.turn_id))
        assert not any(
            entry.event.type.value in {"item_started", "item_completed"}
            for entry in RolloutEventReader(store).read(thread.thread_id)
        )
        record_count = len(store.list_records(thread.thread_id))
        assert not store.complete_model_attempt(
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            response=response,
        )
        assert len(store.list_records(thread.thread_id)) == record_count

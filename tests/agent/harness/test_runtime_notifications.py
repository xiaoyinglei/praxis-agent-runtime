from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.harness import RolloutEventReader, RuntimeState
from agent_runtime.harness.protocol import HarnessModelResponse
from agent_runtime.harness.records import RolloutRecord
from agent_runtime.streaming.events import EventType, StreamEvent, derive_model_public_item_id


def record(record_type: str, payload: dict, *, index: int = 1) -> RolloutRecord:
    return RolloutRecord(
        index,
        f"record-{index}",
        "thread-1",
        "turn-1",
        index,
        record_type,
        3,
        "runtime",
        payload,
        "unused",
        1700000000000 + index,
    )


@pytest.fixture
def reader(tmp_path: Path):
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as state:
        yield RolloutEventReader(state)


def test_interaction_request_and_resolution_are_actionable_and_replayable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RuntimeState.open_sync(database) as state:
        thread = state.create_thread(workspace=workspace)
        turn = state.start_turn(thread_id=thread.thread_id, user_message="ask first", binding_manifest={})
        interaction = state.request_clarification(turn_id=turn.turn_id, question="Which branch?")
        waiting = RolloutEventReader(state).read(thread.thread_id)
        requested = [event for event in waiting if event.event_type == "interaction_requested"]
        assert len(requested) == 1
        assert requested[0].data == {
            "request_id": interaction.request_id,
            "kind": "clarification",
            "operation_id": None,
            "request": {"question": "Which branch?"},
        }
        cursor = requested[0].cursor
        state.resolve_clarification(turn_id=turn.turn_id, request_id=interaction.request_id, response="private answer")
        tail = RolloutEventReader(state).read(thread.thread_id, after=cursor)
        resolved = [event for event in tail if event.event_type == "interaction_resolved"]
        assert len(resolved) == 1
        assert resolved[0].data["request_id"] == interaction.request_id
        assert resolved[0].data["status"] == "resolved"
        assert "private answer" not in str(resolved[0].data)
        expected = [(event.event_type, event.data, event.cursor) for event in tail]
    with RuntimeState.open_sync(database) as reopened:
        replay = RolloutEventReader(reopened).read(thread.thread_id, after=cursor)
        assert [(event.event_type, event.data, event.cursor) for event in replay] == expected


@pytest.mark.parametrize("action,status", [("abort", "cancelled"), ("recheck", "invalidated")])
def test_actual_approval_control_actions_preserve_interaction_outcome(tmp_path: Path, action: str, status: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as state:
        thread = state.create_thread(workspace=workspace)
        turn = state.start_turn(thread_id=thread.thread_id, user_message="write a file", binding_manifest={})
        approval = state.request_tool_approval(
            turn_id=turn.turn_id,
            operation_id="operation-1",
            tool_call_id="call-1",
            tool_name="write_file",
            arguments_digest="arguments-1",
            execution_revision="execution-1",
            idempotent=True,
            request={"tool_name": "write_file", "question": "Allow writing?"},
            effects=("write",),
        )
        if action == "abort":
            state.cancel_turn(turn_id=turn.turn_id)
        else:
            state.refresh_tool_approval(
                turn_id=turn.turn_id, request={"tool_name": "write_file", "question": "Allow again?"}
            )
        resolved = [
            event
            for event in RolloutEventReader(state).read(thread.thread_id)
            if event.event_type == "interaction_resolved" and event.data["request_id"] == approval.request_id
        ]
        assert len(resolved) == 1
        assert resolved[0].data["status"] == status
        assert resolved[0].data["action"] == action
        assert resolved[0].data["operation_id"] == "operation-1"
        assert resolved[0].data["kind"] == "tool_approval"


def test_actual_compaction_notification_keeps_version_and_refs_without_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as state:
        thread = state.create_thread(workspace=workspace)
        turn = state.start_turn(thread_id=thread.thread_id, user_message="compact history", binding_manifest={})
        covered = tuple(item.item_id for item in state.list_context_items(turn.turn_id) if item.status == "completed")
        source_revision = state.context_source_revision(turn.turn_id)
        state.record_context_compaction(
            turn_id=turn.turn_id,
            covered_item_ids=covered,
            summary="private summary text",
            preserved_facts={
                category: []
                for category in (
                    "architecture_and_safety_constraints",
                    "file_changes",
                    "verification_results",
                    "unresolved_work",
                    "uncertain_side_effects",
                )
            },
            context_version=1,
            expected_source_revision=source_revision,
            algorithm_revision="test-algorithm-v1",
        )
        completed = [
            event
            for event in RolloutEventReader(state).read(thread.thread_id)
            if event.event_type == "context_compaction_completed"
        ]
        assert len(completed) == 1
        assert completed[0].data["context_version"] == 1
        assert completed[0].data["source_revision"] == source_revision
        assert completed[0].data["algorithm_revision"] == "test-algorithm-v1"
        assert completed[0].data["covered_item_refs"][0]["item_id"] == covered[0]
        assert completed[0].data["covered_sequence_ranges"]
        assert "private summary text" not in str(completed[0].data)


def test_incomplete_response_remains_visible_without_invalidating_completed_output(tmp_path: Path) -> None:
    from agent_runtime.modeling.events import ModelOutput

    database = tmp_path / "rollout.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeState.open_sync(database) as state:
        thread = state.create_thread(workspace=workspace)
        turn = state.start_turn(thread_id=thread.thread_id, user_message="answer fully", binding_manifest={})
        operation = state.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="request-1",
            context_hash="context-1",
            tool_hash="tools-1",
            wire_hash="wire-1",
            request_ref={"input_revision": 1, "request_id": "request-1"},
        )
        attempt = state.dispatch_model_attempt(operation.operation_id, worker_id="worker-1", lease_seconds=30)
        assert state.complete_model_attempt(
            operation_id=operation.operation_id,
            attempt_id=attempt.attempt_id,
            generation=attempt.generation,
            response=HarnessModelResponse(
                provider_response_id="response-1",
                usage={"output_tokens": 10},
                outputs=(ModelOutput("output-1", "text", content="partial answer"),),
                status="incomplete",
                incomplete_reason="max_output_tokens",
                stop_reason="length",
            ),
        )
        events = RolloutEventReader(state).read(thread.thread_id)
        completed_attempt = [
            event
            for event in events
            if event.event_type == "model_attempt_updated" and event.data["status"] == "completed"
        ]
        assert len(completed_attempt) == 1
        assert completed_attempt[0].data["response_status"] == "incomplete"
        assert completed_attempt[0].data["incomplete_reason"] == "max_output_tokens"
        assert completed_attempt[0].data["stop_reason"] == "length"
        outputs = [
            event
            for event in events
            if event.event_type == "item_completed" and event.data.get("output_id") == "output-1"
        ]
        assert len(outputs) == 1
        assert outputs[0].event.status.value == "success"
        expected = [(event.event_type, event.data, event.cursor) for event in events]
    with RuntimeState.open_sync(database) as state:
        events = RolloutEventReader(state).read(thread.thread_id)
        assert [(event.event_type, event.data, event.cursor) for event in events] == expected


def test_runtime_model_attempt_notifications_do_not_leak_provider_request(reader) -> None:
    event = reader.project_committed_batch(
        (
            record(
                "model_operation_prepared",
                {
                    "operation_id": "op-1",
                    "attempt_id": "attempt-1",
                    "generation": 1,
                    "request_ref": {"messages": "private prompt", "api_key": "secret"},
                },
            ),
        )
    )[0].event
    assert event.type.value == "model_attempt_updated"
    assert event.data == {"operation_id": "op-1", "attempt_id": "attempt-1", "generation": 1, "status": "prepared"}


@pytest.mark.parametrize(
    "record_type,status",
    [
        ("model_attempt_dispatched", "dispatched"),
        ("model_attempt_completed", "completed"),
        ("model_attempt_rejected", "rejected"),
        ("model_attempt_cancelled", "cancelled"),
        ("model_attempt_unknown", "unknown"),
        ("model_retry_prepared", "retry_prepared"),
        ("model_attempt_late_response", "late_response"),
    ],
)
def test_all_model_attempt_transitions_are_visible(reader, record_type, status) -> None:
    payload = {"operation_id": "op-1", "attempt_id": "attempt-1", "generation": 2, "previous_attempt_id": "attempt-0"}
    event = reader.project_committed_batch((record(record_type, payload),))[0].event
    assert event.type.value == "model_attempt_updated"
    assert event.data["status"] == status
    assert event.data["attempt_id"] == "attempt-1"


@pytest.mark.parametrize("transition", ["prepared", "start_rejected", "claim_expired", "result_missing", "superseded"])
def test_tool_control_transitions_are_visible_without_reporting_execution(reader, transition) -> None:
    event = reader.project_committed_batch(
        (
            record(
                f"tool_operation_{transition}",
                {
                    "operation_id": "tool-1",
                    "request_id": "approval-1",
                    "reason": "authorization changed",
                    "arguments": {"command": "private command"},
                    "fencing_token": "internal token",
                    "claim_generation": 2,
                },
            ),
        )
    )[0].event
    assert event.type.value == "tool_operation_updated"
    assert event.data["transition"] == transition
    assert event.data["operation_id"] == "tool-1"
    assert event.data["claim_generation"] == 2
    assert "private command" not in str(event.data)
    assert "internal token" not in str(event.data)
    assert event.item_id is None


@pytest.mark.parametrize("record_type", ["turn_paused", "turn_resumed"])
def test_turn_waiting_and_resume_keep_committed_interaction_identity(reader, record_type) -> None:
    event = reader.project_committed_batch(
        (
            record(
                record_type,
                {
                    "turn_id": "turn-1",
                    "request_id": "request-1",
                    "operation_id": "operation-1",
                },
            ),
        )
    )[0].event
    assert event.data["request_id"] == "request-1"
    assert event.data["operation_id"] == "operation-1"


def test_completion_decision_preserves_action_separate_from_validation(reader) -> None:
    event = reader.project_committed_batch(
        (
            record(
                "item_completed",
                {
                    "item_id": "decision-1",
                    "kind": "completion_decision",
                    "payload": {
                        "proposal_item_id": "proposal-1",
                        "action": "accept",
                        "reason": "scope complete",
                        "validation_status": "failed",
                        "validation_scope": "tests",
                        "reason_code": "test_failure",
                    },
                },
            ),
        )
    )[0].event
    assert event.type.value == "completion_decided"
    assert event.data["action"] == "accept"
    assert event.data["validation_status"] == "failed"
    assert event.data["proposal_item_id"] == "proposal-1"


def test_verification_notification_preserves_committed_evidence_reference(reader) -> None:
    payload = {
        "operation_id": "tool-1",
        "tool_call_id": "call-1",
        "arguments_digest": "digest",
        "source_result_item_id": "result-1",
        "verification_kind": "test",
        "verifier": "pytest",
        "verified_resources": ["test_runtime.py"],
    }
    event = reader.project_committed_batch(
        (record("item_completed", {"item_id": "verification-1", "kind": "verification", "payload": payload}),)
    )[0].event
    assert event.type.value == "verification_completed"
    assert event.data["source_result_item_id"] == "result-1"
    assert "passed" not in event.data


def test_compaction_commit_has_start_and_completion_notifications(reader) -> None:
    events = reader.project_committed_batch(
        (
            record("item_started", {"item_id": "compact-1", "kind": "context_compaction"}),
            record(
                "item_completed",
                {
                    "item_id": "compact-1",
                    "kind": "context_compaction",
                    "payload": {
                        "covered_item_ids": ["old-1"],
                        "summary_item_id": "summary-1",
                        "capacity_diagnostics": {"before_input_tokens": 100},
                    },
                },
                index=2,
            ),
        )
    )
    assert [event.event_type for event in events] == ["context_compaction_started", "context_compaction_completed"]
    assert events[-1].data["status"] == "completed"
    assert events[-1].data["covered_item_ids"] == ["old-1"]


def test_output_identity_is_scoped_to_attempt_and_output() -> None:
    first = derive_model_public_item_id(
        turn_id="turn-1", model_attempt_id="attempt-1", channel="agent_message", output_id="text-1"
    )
    second = derive_model_public_item_id(
        turn_id="turn-1", model_attempt_id="attempt-1", channel="agent_message", output_id="text-2"
    )
    assert first != second
    assert first != derive_model_public_item_id(
        turn_id="turn-1", model_attempt_id="attempt-2", channel="agent_message", output_id="text-1"
    )


def test_generated_tool_call_is_a_proposal_not_execution_and_redacts_provider_data(reader) -> None:
    output = {
        "output_id": "output-1",
        "kind": "tool_call",
        "content": "",
        "tool_call_id": "call-1",
        "tool_name": "run_command",
        "arguments": {"command": "pwd"},
        "argument_error": None,
        "provider_data": {"signature": "private signature"},
    }
    public_id = derive_model_public_item_id(
        turn_id="turn-1", model_attempt_id="attempt-1", channel="tool_call", output_id="output-1"
    )
    events = reader.project_committed_batch(
        (
            record(
                "item_completed",
                {
                    "item_id": "internal-1",
                    "kind": "model_output",
                    "output_id": "output-1",
                    "output_kind": "tool_call",
                    "attempt_id": "attempt-1",
                    "operation_id": "op-1",
                    "public_item_id": public_id,
                    "status": "success",
                    "payload": {"output": output},
                },
            ),
        )
    )
    event = events[0].event
    assert event.item_kind.value == "tool_call"
    assert event.data["execution_started"] is False
    assert event.data["arguments"] == {"command": "pwd"}
    assert "private signature" not in str(event.data)


def test_aggregate_response_is_not_displayed_twice(reader) -> None:
    assert (
        reader.project_committed_batch(
            (
                record(
                    "item_completed",
                    {
                        "item_id": "aggregate-1",
                        "kind": "model_response",
                        "output_protocol": "model-outputs-v1",
                        "payload": {"text": "answer"},
                    },
                ),
            )
        )
        == ()
    )


@pytest.mark.parametrize("data", [{}, {"request_id": "request-1", "kind": "clarification", "request": "malformed"}])
def test_runtime_notification_payload_is_validated(data) -> None:
    with pytest.raises(ValueError):
        StreamEvent(type=EventType("interaction_requested"), turn_id="turn-1", data=data)


@pytest.mark.anyio
async def test_tool_proposal_display_does_not_count_as_execution(capsys) -> None:
    from agent_runtime.streaming.events import (
        ItemDeltaKind,
        ItemStatus,
        TurnItemKind,
        item_completed,
        item_delta,
        item_started,
    )
    from agent_runtime.terminal_render import TerminalToolEventDisplay

    display = TerminalToolEventDisplay(interactive=False)
    await display.emit(
        item_started(
            turn_id="turn-1", item_id="proposal-1", item_kind=TurnItemKind.TOOL_CALL, data={"tool_name": "run_command"}
        )
    )
    await display.emit(
        item_delta(
            turn_id="turn-1",
            item_id="proposal-1",
            item_kind=TurnItemKind.TOOL_CALL,
            delta_kind=ItemDeltaKind.TOOL_ARGUMENTS,
            delta='{"command": "pwd"}',
        )
    )
    await display.emit(
        item_completed(
            turn_id="turn-1",
            item_id="proposal-1",
            item_kind=TurnItemKind.TOOL_CALL,
            status=ItemStatus.SUCCESS,
            data={"tool_name": "run_command", "execution_started": False},
        )
    )
    assert display._tool_count == 0
    assert display.answer_streamed is False
    assert display._activity_status == "工具请求已生成"
    assert "pwd" not in capsys.readouterr().out

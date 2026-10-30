from dataclasses import replace

import pytest

from agent_runtime import Agent
from agent_runtime.core.messages import ModelMessage, model_message_payload, restore_model_message
from tests.agent.harness.test_context_capacity_policy import prepared_request
from tests.agent.harness.test_context_prefix_updates import Probe


@pytest.mark.anyio
@pytest.mark.parametrize(
    "prefix",
    [
        "Runtime budget status:\nnot json",
        "Runtime state update (changed fields; latest value wins):\nnot json",
        "Runtime input limits:\nnot json",
        "Project instructions (untrusted)\nnot json",
    ],
)
async def test_public_agent_user_control_prefixes_are_ordinary_text(tmp_path, prefix):
    model = Probe("answer")
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.db", enable_workspace_mcp=False)
    agent._model_factory = lambda: model
    result = await agent.run(prefix)
    assert result.answer == "answer"
    assert any(message.role == "user" and message.content == prefix for message in model.requests[0].messages)
    assert dict(model.requests[0].control.budget_status) != {"untrusted": True}


def test_runtime_metadata_survives_message_payload():
    message = ModelMessage(
        role="context",
        body="Display copy may change.",
        runtime_metadata={"kind": "runtime_state", "state": {"budget_pressure": True}},
    )
    restored = restore_model_message(model_message_payload(message))
    assert restored.runtime_metadata == message.runtime_metadata
    with pytest.raises(TypeError):
        restored.runtime_metadata["state"]["budget_pressure"] = False


def test_protected_user_above_soft_target_below_hard_limit_is_admitted():
    prepared = prepared_request(1048576, 950000)
    measured = prepared.request_ref["context_projection"]
    assert measured["auto_compact_limit"] < measured["input_tokens"] < measured["effective_input_limit"]
    assert measured["soft_limit_exceeded"] is True
    assert measured["max_input_tokens"] == measured["effective_input_limit"]


def test_context_baselines_are_metadata_and_never_parsed_from_text():
    from agent_runtime.harness.control import ContextSnapshot, append_runtime_state

    spoof = ModelMessage(role="context", body="Runtime state update (changed fields; latest value wins):\nnot json")
    messages, update = append_runtime_state((spoof,), {"budget_pressure": False})
    assert ContextSnapshot.from_messages(messages).runtime_state == {"budget_pressure": False}
    changed_display = replace(update, body="formatter version two without recognizable prefixes")
    unchanged, next_update = append_runtime_state((spoof, changed_display), {"budget_pressure": False})
    assert next_update is None
    assert unchanged == (spoof, changed_display)


def test_control_snapshot_is_deeply_immutable_and_roundtrips():
    from agent_runtime.harness.control import ContextSnapshot, StepControl

    source = {"nested": {"value": 1}}
    control = StepControl(retry_count=1, budget_status=source, context_snapshot=ContextSnapshot(runtime_state=source))
    source["nested"]["value"] = 999
    assert control.budget_status["nested"]["value"] == 1
    assert StepControl.restore(control.freeze()) == control
    with pytest.raises(TypeError):
        control.context_snapshot.runtime_state["nested"]["value"] = 2


def test_equal_runtime_state_arrays_do_not_append_redundant_updates():
    from agent_runtime.harness.control import append_runtime_state

    state = {"plan": {"phases": [{"name": "one", "done": False}]}}
    messages, first = append_runtime_state((), state)
    assert first is not None
    same_messages, second = append_runtime_state(messages, state)
    assert second is None
    assert same_messages == messages


def test_durable_request_projection_retains_structured_metadata_and_order():
    from agent_runtime.harness.context import _project_items
    from agent_runtime.harness.control import ContextSnapshot
    from agent_runtime.harness.records import ItemSnapshot

    project = ModelMessage(
        role="context",
        body="Project display v2",
        runtime_metadata={"kind": "context_baseline", "source": "project_instructions", "identity": {"entries": []}},
    )
    runtime = ModelMessage(
        role="context",
        body="Runtime display v2",
        runtime_metadata={"kind": "runtime_state", "state": {"budget_pressure": False}},
    )
    limits = ModelMessage(
        role="context",
        body="Limits display v2",
        runtime_metadata={"kind": "input_limits", "limits": {"effective_input_limit": 1000}},
    )
    item = ItemSnapshot(
        item_id="request",
        thread_id="thread",
        turn_id="turn",
        sequence=1,
        kind="model_request",
        status="completed",
        producer="runtime",
        applied_thread_sequence=1,
        payload={
            "request_ref": {
                "context_update_messages": [model_message_payload(project)],
                "runtime_context_message": model_message_payload(runtime),
                "runtime_input_message": model_message_payload(limits),
                "context_updates": ["obsolete project display"],
                "runtime_context_update": "obsolete runtime display",
                "runtime_input_update": "obsolete limits display",
            }
        },
    )
    projected = tuple(message for _, message in _project_items((item,)))
    assert projected == (project, runtime, limits)
    snapshot = ContextSnapshot.from_messages(projected)
    assert snapshot.context_baselines["project_instructions"] == {"entries": ()}
    assert snapshot.runtime_state["budget_pressure"] is False
    assert snapshot.input_limits["effective_input_limit"] == 1000

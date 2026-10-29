from __future__ import annotations

import pytest

from agent_runtime.modeling.events import (
    ModelOutput,
    ModelOutputCompleted,
    ModelOutputDelta,
    ModelOutputStarted,
    ModelResponseCompleted,
    ModelResponseStarted,
    ModelStreamState,
    output_from_payload,
    output_to_payload,
)


def test_model_stream_preserves_two_same_kind_objects_and_authoritative_completion() -> None:
    state = ModelStreamState()
    state.accept(ModelResponseStarted(provider_response_id="response-1"))
    state.accept(ModelOutputStarted(ModelOutput("a", "text")))
    state.accept(ModelOutputDelta("a", "text", "draft"))
    state.accept(ModelOutputCompleted(ModelOutput("a", "text", "final a")))
    state.accept(ModelOutputStarted(ModelOutput("b", "text")))
    state.accept(ModelOutputDelta("b", "text", "b"))
    state.accept(ModelOutputCompleted(ModelOutput("b", "text", "b")))
    state.accept(ModelResponseCompleted(provider_response_id="response-1", usage={}, stop_reason="stop"))
    assert [(output.output_id, output.content) for output in state.outputs] == [("a", "final a"), ("b", "b")]


def test_tool_fragments_and_signature_are_retained_without_becoming_text() -> None:
    state = ModelStreamState()
    state.accept(ModelResponseStarted())
    state.accept(ModelOutputStarted(ModelOutput("thinking-1", "reasoning")))
    state.accept(ModelOutputDelta("thinking-1", "reasoning", "inspect"))
    state.accept(ModelOutputDelta("thinking-1", "signature", "signed"))
    state.accept(ModelOutputStarted(ModelOutput("tool-1", "tool_call", tool_call_id="call-1", tool_name="read_file")))
    state.accept(ModelOutputDelta("tool-1", "tool_arguments", '{"path":'))
    state.accept(ModelOutputDelta("tool-1", "tool_arguments", '"a.py"}'))
    assert state.outputs[0].provider_data["signature"] == "signed"
    assert state.outputs[1].content == '{"path":"a.py"}'
    assert state.outputs[1].arguments is None


def test_output_payload_round_trip_detaches_opaque_provider_data() -> None:
    opaque = {"signature": "signed", "annotations": [{"source": "one"}]}
    output = ModelOutput("a", "reasoning", "visible", provider_data=opaque)
    opaque["annotations"][0]["source"] = "changed"
    payload = output_to_payload(output)
    restored = output_from_payload(payload)
    payload["provider_data"]["annotations"][0]["source"] = "again"
    assert restored.provider_data["annotations"][0]["source"] == "one"


@pytest.mark.parametrize(
    "event", [ModelOutputDelta("missing", "text", "x"), ModelOutputCompleted(ModelOutput("missing", "text", "x"))]
)
def test_model_stream_rejects_output_events_without_start(event) -> None:
    state = ModelStreamState()
    state.accept(ModelResponseStarted())
    with pytest.raises(ValueError, match="not started"):
        state.accept(event)


def test_response_cannot_finish_with_open_outputs_or_continue_after_terminal() -> None:
    state = ModelStreamState()
    state.accept(ModelResponseStarted())
    output = ModelOutput("a", "text")
    state.accept(ModelOutputStarted(output))
    terminal = ModelResponseCompleted(usage={}, stop_reason="stop")
    with pytest.raises(ValueError, match="open outputs"):
        state.accept(terminal)
    state.accept(ModelOutputCompleted(output))
    state.accept(terminal)
    with pytest.raises(ValueError, match="terminated"):
        state.accept(ModelOutputStarted(ModelOutput("b", "text")))


def test_response_projections_are_detached_from_mutable_constructor_inputs() -> None:
    from agent_runtime.harness.protocol import HarnessModelResponse, ModelToolCall

    arguments = {"path": "a.py"}
    response = HarnessModelResponse.from_parts(tool_calls=(ModelToolCall("call", "read_file", arguments),))
    arguments["path"] = "changed.py"
    assert response.tool_calls[0].arguments["path"] == "a.py"
    with pytest.raises(TypeError):
        response.tool_calls[0].arguments["path"] = "changed.py"


@pytest.mark.parametrize("payload", [{"provider_data": []}, {"arguments": []}])
def test_model_output_requires_object_metadata_and_tool_arguments(payload) -> None:
    with pytest.raises(TypeError, match="mapping"):
        ModelOutput("tool", "tool_call", tool_call_id="call", tool_name="read_file", **payload)


@pytest.mark.parametrize("returned_usage", [{"input_tokens": 1}, {}])
def test_terminal_usage_cannot_be_reduced_or_dropped_by_returned_response(returned_usage) -> None:
    state = ModelStreamState()
    state.accept(ModelResponseStarted())
    state.accept(ModelResponseCompleted(usage={"input_tokens": 100}, stop_reason="stop"))
    with pytest.raises(ValueError, match="usage"):
        state.validate_response(
            outputs=(),
            provider_response_id=None,
            status="completed",
            incomplete_reason=None,
            stop_reason="stop",
            usage=returned_usage,
        )


def test_returned_usage_can_add_pricing_and_latency_without_overwriting_provider_facts() -> None:
    provider_usage = {"input_tokens": 100, "raw_provider_usage": {"details": [1, {"cache": 20}]}}
    state = ModelStreamState()
    state.accept(ModelResponseStarted())
    state.accept(ModelResponseCompleted(usage=provider_usage, stop_reason="stop"))
    state.validate_response(
        outputs=(),
        provider_response_id=None,
        status="completed",
        incomplete_reason=None,
        stop_reason="stop",
        usage={**provider_usage, "cost_micros": 12, "pricing_revision": "test", "latency_ms": 1.2},
    )


@pytest.mark.parametrize("response_id", ["", 1, {}, True])
def test_model_response_start_rejects_invalid_provider_identity(response_id) -> None:
    with pytest.raises((TypeError, ValueError), match="response.*identity"):
        ModelResponseStarted(response_id)


@pytest.mark.parametrize(
    "payload",
    [
        {"stop_reason": ""},
        {"stop_reason": 1},
        {"provider_response_id": 1},
        {"incomplete_reason": ""},
        {"status": "incomplete", "incomplete_reason": 1},
    ],
)
def test_model_response_terminal_rejects_malformed_metadata(payload) -> None:
    with pytest.raises((TypeError, ValueError)):
        ModelResponseCompleted(**{"usage": {}, "stop_reason": "stop", **payload})


@pytest.mark.parametrize("argument_error", [None, "Provider declared argument error"])
def test_complete_tool_output_cannot_disagree_with_its_raw_arguments(argument_error) -> None:
    with pytest.raises(ValueError, match="arguments.*conflict"):
        ModelOutput(
            "out-1",
            "tool_call",
            '{"path":"shown.py"}',
            tool_call_id="call-1",
            tool_name="read_file",
            arguments={"path": "executed.py"},
            argument_error=argument_error,
        )


def test_complete_tool_output_accepts_semantically_equal_nested_json() -> None:
    output = ModelOutput(
        "out-1",
        "tool_call",
        '{ "items": [{"b": [1, 2], "a": true}], "name": "x" }',
        tool_call_id="call-1",
        tool_name="update_plan",
        arguments={"name": "x", "items": [{"a": True, "b": [1, 2]}]},
    )
    assert output_to_payload(output)["arguments"] == {"name": "x", "items": [{"a": True, "b": [1, 2]}]}


def test_tool_argument_consistency_keeps_json_boolean_distinct_from_integer() -> None:
    with pytest.raises(ValueError, match="arguments.*conflict"):
        ModelOutput(
            "out-1",
            "tool_call",
            '{"enabled":true}',
            tool_call_id="call-1",
            tool_name="configure",
            arguments={"enabled": 1},
        )


@pytest.mark.parametrize("raw", ['{"path":', '["not", "an", "object"]'])
def test_complete_tool_output_requires_valid_object_json_without_argument_error(raw) -> None:
    with pytest.raises(ValueError, match="arguments"):
        ModelOutput("out-1", "tool_call", raw, tool_call_id="call-1", tool_name="read_file", arguments={})


def test_malformed_tool_evidence_is_preserved_when_explicitly_marked_invalid() -> None:
    raw = '{"path":'
    output = ModelOutput(
        "out-1",
        "tool_call",
        raw,
        tool_call_id="call-1",
        tool_name="read_file",
        arguments={"_raw": raw},
        argument_error="Invalid JSON: incomplete object",
    )
    assert output.content == raw
    assert output_to_payload(output)["arguments"] == {"_raw": raw}
    assert output.argument_error == "Invalid JSON: incomplete object"


@pytest.mark.parametrize("argument_error", ["", " ", 1, True])
def test_malformed_tool_evidence_requires_a_nonempty_error_string(argument_error) -> None:
    with pytest.raises(ValueError, match="argument_error"):
        ModelOutput(
            "out-1",
            "tool_call",
            '{"path":',
            tool_call_id="call-1",
            tool_name="read_file",
            arguments={"_raw": '{"path":'},
            argument_error=argument_error,
        )


def test_tool_snapshot_without_raw_json_derives_content_from_arguments() -> None:
    import json

    output = ModelOutput(
        "out-1",
        "tool_call",
        tool_call_id="call-1",
        tool_name="update_plan",
        arguments={"items": [{"step": "read"}]},
    )
    assert json.loads(output.content) == {"items": [{"step": "read"}]}


def test_tool_output_started_can_retain_partial_unparsed_input() -> None:
    output = ModelOutput("out-1", "tool_call", '{"path":', tool_call_id="call-1", tool_name="read_file")
    assert output.arguments is None
    assert output.content == '{"path":'

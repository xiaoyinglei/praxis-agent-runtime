from dataclasses import fields, replace

import pytest

from agent_runtime.core.messages import ModelMessage, ModelToolCall, model_message_payload, restore_model_message
from agent_runtime.harness.protocol import (
    HarnessModelResponse,
)
from agent_runtime.modeling.events import ModelOutput


def test_public_model_types_are_the_same_canonical_values() -> None:
    from agent_runtime.harness import ModelMessage as PublicMessage
    from agent_runtime.harness import ModelToolCall as PublicCall

    assert PublicMessage is ModelMessage
    assert PublicCall is ModelToolCall


def test_message_has_one_body_and_derives_text_and_calls_without_losing_order() -> None:
    outputs = (
        ModelOutput("before", "text", "before"),
        ModelOutput("call", "tool_call", tool_call_id="c", tool_name="read_file", arguments={"path": "a"}),
        ModelOutput("after", "text", "after"),
    )
    message = ModelMessage(role="assistant", body=outputs)
    assert message.content == "beforeafter"
    assert message.tool_calls[0].id == "c"
    assert message.outputs == outputs
    assert {field.name for field in fields(message)} == {"role", "body", "tool_call_id", "context_category"}
    payload = model_message_payload(message)
    assert "content" not in payload and "outputs" not in payload and "tool_calls" not in payload
    assert restore_model_message(model_message_payload(message)) == message
    assert replace(message, body="replacement").content == "replacement"


def test_response_keeps_only_outputs_and_simple_factory_is_explicit() -> None:
    response = HarnessModelResponse.from_parts(text="hello", tool_calls=(ModelToolCall("c", "read_file", {}),))
    assert response.text == "hello"
    assert response.tool_calls[0].id == "c"
    assert not {"text", "tool_calls", "reasoning_content", "plan_content"} & {f.name for f in fields(response)}
    with pytest.raises(TypeError):
        HarnessModelResponse(text="contradiction", outputs=response.outputs)


def test_message_reopen_retains_private_metadata_and_multiple_same_kind_outputs() -> None:
    outputs = (
        ModelOutput("a", "reasoning", "inspect", provider_data={"signature": "private"}),
        ModelOutput("b", "text", "one"),
        ModelOutput("c", "text", "two"),
    )
    message = ModelMessage(role="assistant", body=outputs)
    restored = restore_model_message(model_message_payload(message))
    assert restored == message
    assert restored.content == "onetwo"
    assert restored.outputs[0].provider_data["signature"] == "private"


def test_tool_results_and_context_do_not_accept_assistant_output_blocks() -> None:
    with pytest.raises(ValueError, match="assistant"):
        ModelMessage(role="tool", body=(ModelOutput("x", "text", "x"),), tool_call_id="call")
    message = ModelMessage(role="context", body="summary", context_category="continuation_summary")
    assert message.content == "summary" and message.outputs == ()


def test_incomplete_response_preserves_output_facts_but_exposes_no_executable_calls() -> None:
    response = HarnessModelResponse(
        outputs=(ModelOutput("tool", "tool_call", tool_call_id="c", tool_name="read_file", arguments={}),),
        status="incomplete",
        incomplete_reason="max_output_tokens",
    )
    assert len(response.outputs) == 1 and response.tool_calls == ()


def test_wire_projection_includes_each_body_once_and_keeps_private_replay_metadata_out() -> None:
    from agent_runtime.core.model_request import ModelSettings, build_model_request, build_stable_context
    from agent_runtime.modeling.local_agent_wire import render_local_agent_request
    from agent_runtime.modeling.openai_wire import serialize_openai_request

    marker = "visible-answer-sentinel"
    message = ModelMessage(role="assistant", body=(
        ModelOutput("text", "text", marker, provider_data={"signature": "private-signature-sentinel"}),
    ))
    context = build_stable_context(instructions=("Inspect.",), frozen_run_context=(), initial_user_task="inspect")
    request = build_model_request(request_id="request", context=context.append_message(message), selected_tools=(),
                                  settings=ModelSettings(model="test"))
    for wire in (render_local_agent_request(request, provider="mlx").prompt,
                 serialize_openai_request(request).serialized_json):
        assert wire.count(marker) == 1
        assert "private-signature-sentinel" not in wire
    assert message.outputs[0].provider_data["signature"] == "private-signature-sentinel"


def test_response_usage_is_an_immutable_snapshot() -> None:
    usage = {"output_tokens": 3, "details": {"cached": 1}}
    response = HarnessModelResponse(outputs=(ModelOutput("t", "text", "answer"),), usage=usage)
    usage["details"]["cached"] = 99
    assert response.usage["details"]["cached"] == 1
    with pytest.raises(TypeError):
        response.usage["output_tokens"] = 99


@pytest.mark.anyio
async def test_adapter_rejects_end_turn_with_tool_blocks_before_runtime_execution(tmp_path) -> None:
    from agent_runtime.core.messages import StopReason
    from agent_runtime.harness import GatewayHarnessModel, HarnessModelRequest, Session
    from agent_runtime.modeling.contracts import normalize_llm_usage
    from agent_runtime.modeling.gateway import AgentModelResponse
    from agent_runtime.modeling.openai_wire import serialize_openai_request
    from tests.agent.harness.test_model_adapter import _resolved_model
    from tests.agent.test_canonical_model_request import _tool

    class Gateway:
        async def agenerate_model_request(self, *, request, **kwargs):
            return AgentModelResponse(
                outputs=(
                    ModelOutput("text", "text", "done"),
                    ModelOutput(
                        "call",
                        "tool_call",
                        tool_call_id="c",
                        tool_name="write_file",
                        arguments={"path": "a"},
                    ),
                ),
                stop_reason=StopReason.END_TURN,
                raw_stop_reason="stop",
                usage=normalize_llm_usage(
                    input_tokens=1, output_tokens=1, input_tokens_include_cache=True, usage_source="provider"
                ),
                provider_wire_hash=serialize_openai_request(request).provider_wire_hash,
                serializer_revision="test",
                wire_kind="openai",
            )

    class BoundModel(GatewayHarnessModel):
        def snapshot(self, **identity):
            return {**identity, "model_id": "test"}

        def ensure_available(self, *args, **kwargs):
            pass

    model = BoundModel(model_id="test", resolved=_resolved_model(gateway=Gateway()), instructions=("Inspect.",))
    prepared = model.prepare(
        HarnessModelRequest(
            thread_id="t",
            turn_id="u",
            messages=(ModelMessage(role="user", body="inspect"),),
            binding_manifest={},
        )
    )
    with pytest.raises(RuntimeError, match="end-turn.*tool calls"):
        await model.dispatch(prepared)

    marker = tmp_path / "executed"
    tool = replace(_tool("write_file", schema={"type": "object"}), run=lambda _: marker.write_text("executed"))
    async with await Session.open(
        database=tmp_path / "run.db",
        workspace=tmp_path,
        model=model,
        tools={"write_file": tool},
        _background_memory=False,
        require_workspace_change=False,
    ) as session:
        result = await session.submit("inspect")
        assert result.status == "failed"
        assert session.store.list_tool_operations(result.turn_id) == ()
    assert not marker.exists()

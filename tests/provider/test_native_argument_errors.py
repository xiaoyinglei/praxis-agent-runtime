"""Malformed native arguments retain their real error and original bytes."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from agent_runtime.core.messages import ModelMessage, snapshot_model_message
from agent_runtime.modeling.contracts import LLMCallStage, LLMUsage
from agent_runtime.modeling.openai_wire import parse_openai_response
from tests.provider.test_llm_gateway import _canonical_request, _gateway


def native_response(arguments):
    return parse_openai_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "patch",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )


@pytest.mark.parametrize("raw", ['{"file_path":"result.txt","new_string":"unfinished', "[1,2]", "null"])
def test_non_object_or_malformed_native_arguments_have_explicit_error(raw):
    call = native_response(raw).tool_calls[0]
    assert call.argument_error
    assert "JSON" in call.argument_error
    assert call.input == {"_raw": raw}
    for copied in (
        deepcopy(call),
        snapshot_model_message(
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=(call,),
            )
        ).tool_calls[0],
    ):
        assert copied.argument_error == call.argument_error
        assert copied.input == call.input


def test_large_malformed_arguments_are_not_silently_cut_to_twenty_k():
    raw = '{"file_path":"result.txt","new_string":"' + "x" * 30_000
    call = native_response(raw).tool_calls[0]
    assert call.input["_raw"] == raw
    assert "line 1" in call.argument_error
    assert "column" in call.argument_error


def test_valid_raw_named_argument_is_ordinary_business_data():
    call = native_response(json.dumps({"_raw": "business value"})).tool_calls[0]
    assert call.argument_error is None
    assert call.input == {"_raw": "business value"}


@pytest.mark.anyio
async def test_streamed_arguments_report_the_same_error_as_nonstreaming():
    raw = '{"file_path":"result.txt","new_string":"unfinished'

    class Generator:
        def stream_with_tools(self, **kwargs):
            yield {"type": "tool_use_start", "tool_id": "patch", "tool_name": "apply_patch"}
            for fragment in (raw[:20], raw[20:]):
                yield {"type": "tool_input_delta", "tool_id": "patch", "content": fragment}
            yield {"type": "content_block_stop", "tool_id": "patch"}
            yield {
                "type": "message_stop",
                "stop_reason": "tool_use",
                "usage": LLMUsage(
                    input_tokens=10,
                    output_tokens=20,
                    total_tokens=30,
                    source="provider",
                ),
            }

    result = await _gateway(Generator(), max_input_tokens=2000, model_context_tokens=4000).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=_canonical_request(),
        provider="openai-compatible",
        supports_native_tools=True,
        stream=True,
    )
    assert result.turn.tool_calls[0].argument_error == native_response(raw).tool_calls[0].argument_error
    assert result.turn.tool_calls[0].input == {"_raw": raw}

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_runtime.modeling.chat import OpenAICompatibleChatGenerator
from agent_runtime.modeling.openai_wire import parse_openai_response


@pytest.mark.parametrize("stream", [False, True])
def test_mlx_reasoning_field_stays_in_reasoning_channel(stream):
    message = SimpleNamespace(content="answer", reasoning="provider reasoning", tool_calls=None)
    choice = SimpleNamespace(finish_reason="stop", **({"delta": message} if stream else {"message": message}))
    response = SimpleNamespace(choices=[choice], usage=None)
    generator = OpenAICompatibleChatGenerator.__new__(OpenAICompatibleChatGenerator)
    generator.chat_model_name = "test"
    generator._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: iter([response]) if stream else response,
            )
        )
    )
    chunks = list(generator.stream_with_tools(messages=[], tools=[]))
    assert {c["type"]: c.get("content") for c in chunks} == {
        "thinking_delta": "provider reasoning",
        "text_delta": "answer",
        "message_stop": None,
    }
    if not stream:
        turn = parse_openai_response(response)
        assert turn.reasoning_content == "provider reasoning"
        assert turn.text == "answer"


def test_qwen_mlx_template_enables_thinking():
    from agent_runtime.models import ModelCatalog

    definition = ModelCatalog.from_config_file(Path("configs/models.yaml")).definition("mlx-community/Qwen3.5-9B-4bit")
    assert definition.defaults.provider_options.chat_template_kwargs.enable_thinking is True


def test_reasoning_tool_markup_does_not_become_executable_call():
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"reasoning": '<tool_call>{"name":"run_command"}</tool_call>'},
            }
        ],
    }
    turn = parse_openai_response(response)
    assert turn.text == ""
    assert turn.tool_calls == []
    assert turn.reasoning_content == response["choices"][0]["message"]["reasoning"]


def test_usage_only_tail_is_preserved_and_missing_stop_is_recoverable():
    from agent_runtime.harness.model_adapter import _is_uncertain_transport_failure

    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="4", tool_calls=None), finish_reason="stop")],
            usage=None,
        ),
        SimpleNamespace(choices=[], usage={"prompt_tokens": 4980, "completion_tokens": 20, "total_tokens": 5000}),
    ]
    generator = OpenAICompatibleChatGenerator.__new__(OpenAICompatibleChatGenerator)
    generator.chat_model_name = "test"
    generator._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: iter(chunks)))
    )
    result = list(generator.stream_with_tools(messages=[], tools=[]))
    assert result[-1]["usage"].input_tokens == 4980
    assert sum(c["type"] == "message_stop" for c in result) == 1
    chunks[:] = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="partial", tool_calls=None), finish_reason=None)],
            usage=None,
        )
    ]
    with pytest.raises(ConnectionError) as error:
        list(generator.stream_with_tools(messages=[], tools=[]))
    assert _is_uncertain_transport_failure(error.value)


def test_token_estimate_accounts_for_complete_tool_schema():
    from agent_runtime.modeling.gateway import _account_messages

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read workspace data",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Exact path to the file"}},
                },
            },
        }
    ]
    counted = _account_messages([], tools)
    assert "Exact path to the file" in counted
    assert "Read workspace data" in counted

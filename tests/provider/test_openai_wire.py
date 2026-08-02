from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from rag.agent.core.messages import (
    ModelMessage,
    StopReason,
    ToolUseResult,
    context_event_message,
)
from rag.agent.core.model_request import (
    ContextBlock,
    ModelSettings,
    ToolChoice,
    build_model_request,
    build_stable_context,
)
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    json_schema_input,
)
from rag.providers import openai_wire as openai_wire_module
from rag.providers.openai_wire import (
    parse_openai_response,
    parse_openai_usage,
    serialize_openai_request,
)


def _tool(name: str, schema: Mapping[str, JsonValue] | None = None) -> Tool:
    input_schema = schema or {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Description for {name}.",
            input_schema=input_schema,
        ),
        validate_input=json_schema_input(input_schema),
        run=lambda _arguments: {},
        normalize_output=lambda _raw: NormalizedToolOutput(),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision=f"test-{name}-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=10_000,
    )


def _request(
    *,
    top_p: float | None = 0.9,
    seed: int | None = 42,
    provider_options: Mapping[str, JsonValue] | None = None,
):
    schema = {
        "$defs": {
            "Target": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ]
            }
        },
        "type": "object",
        "properties": {
            "target": {"$ref": "#/$defs/Target"},
            "mode": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["target"],
        "additionalProperties": False,
    }
    context = build_stable_context(
        instructions=("Be precise.",),
        frozen_run_context=(ContextBlock("workspace", {"root": "."}),),
        initial_user_task="Use the selected tool.",
    ).append_message(ModelMessage(role="assistant", content="Prior response."))
    tools = (
        _tool("mcp__zeta__lookup", schema),
        _tool("mcp__alpha__lookup"),
    )
    return build_model_request(
        request_id="req-openai-wire",
        context=context,
        selected_tools=tools,
        settings=ModelSettings(
            model="gpt-compatible",
            max_output_tokens=777,
            temperature=0.1,
            top_p=top_p,
            parallel_tool_calls=True,
            seed=seed,
            provider_options=provider_options or {},
        ),
        tool_choice=ToolChoice.named("mcp__zeta__lookup"),
    )


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def test_openai_wire_preserves_canonical_message_and_tool_order() -> None:
    request = _request()

    wire = serialize_openai_request(request)
    payload = _thaw(wire.payload)
    assert isinstance(payload, dict)

    assert payload["model"] == "gpt-compatible"
    assert payload["max_completion_tokens"] == 777
    assert payload["temperature"] == 0.1
    assert payload["top_p"] == 0.9
    assert payload["parallel_tool_calls"] is True
    assert payload["seed"] == 42
    assert [item["function"]["name"] for item in payload["tools"]] == [
        "mcp__zeta__lookup",
        "mcp__alpha__lookup",
    ]
    parameters = payload["tools"][0]["function"]["parameters"]
    assert parameters["properties"]["target"] == {"$ref": "#/$defs/Target"}
    assert parameters["$defs"]["Target"]["oneOf"] == [
        {"type": "string"},
        {"type": "integer"},
    ]
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "mcp__zeta__lookup"},
    }
    messages = payload["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert "Be precise." in messages[0]["content"]
    assert "frozen_run_context" in messages[0]["content"]
    assert sum(message["role"] == "system" for message in messages) == 1


def test_openai_wire_serializes_later_context_as_user_event() -> None:
    context = build_stable_context(
        instructions=("Be precise.",),
        initial_user_task="Inspect the repository.",
        transcript=(
            ModelMessage(role="assistant", content="I will inspect it."),
            context_event_message(
                "runtime_diagnostic",
                {"detail": "retry with the published schema"},
            ),
        ),
    )
    request = build_model_request(
        request_id="req-later-context",
        context=context,
        selected_tools=(),
        settings=ModelSettings(model="gpt-compatible"),
    )

    payload = _thaw(serialize_openai_request(request).payload)
    messages = payload["messages"]

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "runtime_diagnostic" in messages[-1]["content"]
    assert sum(message["role"] == "system" for message in messages) == 1


def test_openai_wire_round_trips_assistant_reasoning_for_tool_continuation() -> None:
    context = build_stable_context(
        instructions=("Use tools.",),
        initial_user_task="Inspect the repository.",
        transcript=(
            ModelMessage(
                role="assistant",
                content="",
                reasoning_content="I need to inspect the source first.",
                tool_calls=(),
            ),
        ),
    )
    request = build_model_request(
        request_id="req-reasoning-continuation",
        context=context,
        selected_tools=(),
        settings=ModelSettings(model="gpt-compatible"),
    )

    payload = _thaw(serialize_openai_request(request).payload)

    assert payload["messages"][-1] == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I need to inspect the source first.",
    }


def test_openai_wire_rejects_non_leading_system_message() -> None:
    context = build_stable_context(
        instructions=("Be precise.",),
        initial_user_task="Inspect the repository.",
        transcript=(
            ModelMessage(role="assistant", content="I will inspect it."),
            ModelMessage(role="system", content="Late system override."),
        ),
    )
    request = build_model_request(
        request_id="req-later-system",
        context=context,
        selected_tools=(),
        settings=ModelSettings(model="gpt-compatible"),
    )

    with pytest.raises(ValueError, match="non-leading system"):
        serialize_openai_request(request)


def test_openai_wire_hash_is_final_serialized_hash_and_cache_options_are_gated() -> None:
    request = _request()
    supported = frozenset({"prompt_cache_key", "prompt_cache_retention"})
    cache_parameters = {
        "prompt_cache_retention": "24h",
        "unsupported_cache_hint": "must-not-leak",
    }

    first = serialize_openai_request(
        request,
        cache_parameters=cache_parameters,
        supported_cache_parameters=supported,
    )
    second = serialize_openai_request(
        request,
        cache_parameters={
            "unsupported_cache_hint": "must-not-leak",
            "prompt_cache_retention": "24h",
        },
        supported_cache_parameters=supported,
    )
    without_cache = serialize_openai_request(request)

    assert first.provider_wire_hash == second.provider_wire_hash
    assert first.provider_wire_hash != without_cache.provider_wire_hash
    assert first.payload["prompt_cache_key"] == request.prompt_revision
    assert first.payload["prompt_cache_retention"] == "24h"
    assert "unsupported_cache_hint" not in first.payload
    assert first.serialized_json == second.serialized_json
    assert json.loads(first.serialized_json)["prompt_cache_key"] == (request.prompt_revision)


def test_openai_wire_omits_unset_optional_generation_fields() -> None:
    payload = serialize_openai_request(_request(top_p=None, seed=None)).payload

    assert "top_p" not in payload
    assert "seed" not in payload


@pytest.mark.parametrize("field_name", ["prompt_cache_key", "prompt_cache_retention"])
def test_provider_options_cannot_override_serializer_owned_cache_fields(
    field_name: str,
) -> None:
    request = _request(provider_options={field_name: "caller-owned"})

    with pytest.raises(ValueError, match="cannot override serializer-owned field"):
        serialize_openai_request(
            request,
            supported_cache_parameters={field_name},
        )


def test_openai_response_parser_returns_the_provider_neutral_model_turn() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="I will inspect it.",
                    reasoning_content="The repository must be inspected before editing.",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(
                                name="mcp__zeta__lookup",
                                arguments='{"target": 7}',
                            ),
                        )
                    ],
                ),
            )
        ]
    )

    turn = parse_openai_response(response)

    assert isinstance(turn, ToolUseResult)
    assert turn.stop_reason is StopReason.TOOL_USE
    assert turn.raw_stop_reason == "tool_calls"
    assert turn.text == "I will inspect it."
    assert turn.reasoning_content == (
        "The repository must be inspected before editing."
    )
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].name == "mcp__zeta__lookup"
    assert turn.tool_calls[0].input == {"target": 7}


def test_openai_response_parser_accepts_mapping_responses_deterministically() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "Done.", "tool_calls": []},
            }
        ]
    }

    first = parse_openai_response(response)
    second = parse_openai_response(response)

    assert first == second
    assert first.stop_reason is StopReason.END_TURN
    assert first.text == "Done."


def test_openai_response_parser_accepts_sdk_null_tool_calls(
    chat_completion_factory: Callable[..., ChatCompletion],
) -> None:
    response = chat_completion_factory(content="Done.", tool_calls=None)

    turn = parse_openai_response(response)

    assert turn.stop_reason is StopReason.END_TURN
    assert turn.text == "Done."
    assert turn.tool_calls == []


def test_openai_usage_treats_cached_tokens_as_part_of_total_input() -> None:
    usage = parse_openai_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 40},
            }
        }
    )

    assert usage is not None
    assert usage.logical_input_tokens == 100
    assert usage.uncached_input_tokens == 60
    assert usage.cache_read_input_tokens == 40
    assert usage.cache_write_input_tokens is None
    assert usage.output_tokens == 20
    assert usage.usage_source == "provider"
    assert usage.raw_provider_usage == {
        "completion_tokens": 20,
        "prompt_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 40},
    }


def test_openai_usage_preserves_reported_cache_write_details() -> None:
    usage = parse_openai_usage(
        {
            "usage": {
                "prompt_tokens": 90,
                "completion_tokens": 8,
                "prompt_tokens_details": {
                    "cached_tokens": 30,
                    "cache_write_tokens": 12,
                },
            }
        }
    )

    assert usage is not None
    assert usage.logical_input_tokens == 90
    assert usage.uncached_input_tokens == 48
    assert usage.cache_read_input_tokens == 30
    assert usage.cache_write_input_tokens == 12
    assert usage.logical_input_tokens == (
        usage.uncached_input_tokens
        + usage.cache_read_input_tokens
        + usage.cache_write_input_tokens
    )


def test_openai_usage_keeps_unreported_cache_values_unknown() -> None:
    usage = parse_openai_usage({"usage": {"prompt_tokens": 17, "completion_tokens": 3}})

    assert usage is not None
    assert usage.logical_input_tokens == 17
    assert usage.uncached_input_tokens is None
    assert usage.cache_read_input_tokens is None
    assert usage.cache_write_input_tokens is None
    assert parse_openai_usage({"choices": []}) is None


def test_openai_usage_accepts_sdk_style_objects() -> None:
    usage = parse_openai_usage(
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=2,
                prompt_tokens_details=SimpleNamespace(cached_tokens=5),
            )
        )
    )

    assert usage is not None
    assert usage.logical_input_tokens == 12
    assert usage.uncached_input_tokens == 7
    assert usage.cache_read_input_tokens == 5
    assert usage.raw_provider_usage == {
        "completion_tokens": 2,
        "prompt_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 5},
    }


def test_openai_wire_is_only_a_serializer_and_parser() -> None:
    module_path = Path(openai_wire_module.__file__ or "")
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}

    assert not any(
        module.startswith(
            (
                "rag.agent.tools.selection",
                "rag.agent.tools.executor",
                "rag.agent.tools.registry",
                "rag.agent.loop",
                "rag.agent.tooling",
            )
        )
        for module in imports
    )
    assert "select_tools" not in source
    assert "can_use_tool" not in source
    assert "ToolExecutor" not in source

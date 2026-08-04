from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_runtime.core.messages import ModelMessage, StopReason, ToolUseResult
from agent_runtime.core.model_request import (
    ContextBlock,
    ModelSettings,
    build_model_request,
    build_stable_context,
)
from agent_runtime.modeling.local_agent_wire import (
    LocalAgentWireMode,
    estimate_local_agent_usage,
    parse_local_agent_response,
    render_local_agent_request,
    resolve_local_agent_wire,
)
from agent_runtime.modeling.openai_wire import parse_openai_response
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    json_schema_input,
)
from rag.providers import local_agent_wire as local_wire_module


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
    top_p: float | None = 0.95,
    seed: int | None = None,
):
    read_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative file path.",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    context = build_stable_context(
        instructions=("Be precise and return one response envelope.",),
        frozen_run_context=(ContextBlock("workspace", {"root": "."}),),
        initial_user_task="Read README.md and summarize it.",
    ).append_message(ModelMessage(role="assistant", content="I will inspect it."))
    return build_model_request(
        request_id="req-local-wire",
        context=context,
        selected_tools=(
            _tool("read_file", read_schema),
            _tool("mcp__zeta__lookup"),
            _tool("mcp__alpha__lookup"),
        ),
        settings=ModelSettings(
            model="models--mlx-community--Qwen3-14B-4bit",
            max_output_tokens=512,
            temperature=0.0,
            top_p=top_p,
            parallel_tool_calls=True,
            seed=seed,
        ),
    )


@pytest.mark.parametrize("provider", ["mlx", "ollama"])
def test_mlx_and_ollama_resolve_the_same_flat_json_adapter(provider: str) -> None:
    assert resolve_local_agent_wire(provider, supports_native_tools=False) is LocalAgentWireMode.FLAT_JSON
    assert resolve_local_agent_wire(provider, supports_native_tools=True) is LocalAgentWireMode.NATIVE


def test_unknown_local_provider_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unsupported local model provider"):
        resolve_local_agent_wire("unknown", supports_native_tools=False)


@pytest.mark.parametrize("provider", ["mlx", "ollama"])
def test_local_flat_prompt_contains_every_selected_schema_in_order(
    provider: str,
) -> None:
    request = _request()

    wire = render_local_agent_request(request, provider=provider)

    assert wire.provider == provider
    assert wire.prompt.index('"name":"read_file"') < wire.prompt.index('"name":"mcp__zeta__lookup"')
    assert wire.prompt.index('"name":"mcp__zeta__lookup"') < wire.prompt.index('"name":"mcp__alpha__lookup"')
    assert '"required":["path"]' in wire.prompt
    assert '"additionalProperties":false' in wire.prompt
    assert "[Selected Tools]" in wire.prompt
    assert "[Canonical Messages]" in wire.prompt
    assert "Return exactly one JSON object" in wire.prompt
    assert (
        '{"text":"","tool_calls":[{"id":"call_1","name":"read_file","arguments":{"path":"README.md"}}]}'
    ) in wire.prompt
    assert wire.generation_options["model"] == ("models--mlx-community--Qwen3-14B-4bit")
    assert wire.generation_options["max_tokens"] == 512
    assert wire.generation_options["temperature"] == 0.0
    assert wire.generation_options["top_p"] == 0.95


def test_mlx_and_ollama_use_identical_prompt_semantics() -> None:
    request = _request()

    mlx = render_local_agent_request(request, provider="mlx")
    ollama = render_local_agent_request(request, provider="ollama")
    mlx_again = render_local_agent_request(request, provider="mlx")

    assert mlx.prompt == ollama.prompt
    assert mlx.provider_wire_hash != ollama.provider_wire_hash
    assert mlx.provider_wire_hash == mlx_again.provider_wire_hash
    assert mlx.serialized_json == mlx_again.serialized_json


def test_local_wire_omits_unset_optional_generation_fields() -> None:
    without_optional = render_local_agent_request(
        _request(top_p=None, seed=None),
        provider="mlx",
    )
    with_optional = render_local_agent_request(
        _request(top_p=0.8, seed=7),
        provider="mlx",
    )

    assert "top_p" not in without_optional.generation_options
    assert "seed" not in without_optional.generation_options
    assert with_optional.generation_options["top_p"] == 0.8
    assert with_optional.generation_options["seed"] == 7


def test_local_response_parser_returns_the_same_provider_neutral_turn_type() -> None:
    raw = json.dumps(
        {
            "text": "I will read it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                }
            ],
        }
    )
    local_turn = parse_local_agent_response(raw)
    openai_turn = parse_openai_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "I will read it.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": {"path": "README.md"},
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )

    assert isinstance(local_turn, ToolUseResult)
    assert type(local_turn) is type(openai_turn)
    assert local_turn.stop_reason is StopReason.TOOL_USE
    assert local_turn.text == openai_turn.text
    assert local_turn.tool_calls == openai_turn.tool_calls


def test_local_tokenizer_estimate_never_fabricates_cache_hits() -> None:
    usage = estimate_local_agent_usage(input_tokens=14, output_tokens=5)

    assert usage.logical_input_tokens == 14
    assert usage.uncached_input_tokens is None
    assert usage.cache_read_input_tokens is None
    assert usage.cache_write_input_tokens is None
    assert usage.output_tokens == 5
    assert usage.usage_source == "tokenizer_estimate"
    assert usage.raw_provider_usage is None


def test_local_response_parser_requires_one_strict_validated_envelope() -> None:
    with pytest.raises((ValueError, ValidationError, json.JSONDecodeError)):
        parse_local_agent_response('```json\n{"text":"Done","tool_calls":[]}\n```')
    with pytest.raises((ValueError, ValidationError)):
        parse_local_agent_response(
            {
                "text": "Done",
                "tool_calls": [],
                "unexpected": True,
            }
        )


def test_local_wire_is_serialization_not_a_second_runtime() -> None:
    module_path = Path(local_wire_module.__file__ or "")
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}

    assert not any(
        module.startswith(
            (
                "agent_runtime.tools.selection",
                "agent_runtime.tools.executor",
                "agent_runtime.tools.registry",
                "agent_runtime.loop",
                "agent_runtime.tooling",
            )
        )
        for module in imports
    )
    assert "select_tools" not in source
    assert "ToolExecutor" not in source
    assert "AgentLoop" not in source

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from rag.agent.core import model_request as model_request_module
from rag.agent.core.messages import (
    ModelMessage,
    context_event_message,
)
from rag.agent.core.messages import (
    ToolCall as ModelToolCall,
)
from rag.agent.core.model_request import (
    ContextBlock,
    ModelSettings,
    StableModelContext,
    ToolChoice,
    build_model_request,
    build_stable_context,
    canonical_model_request_json,
    stable_context_json,
)
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
    json_schema_input,
)


def _tool(
    name: str,
    *,
    description: str | None = None,
    schema: Mapping[str, JsonValue] | None = None,
    execution_revision: str = "runner-v1",
) -> Tool:
    input_schema: Mapping[str, JsonValue] = schema or {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=description or f"Use {name} for its documented operation.",
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
        execution_revision=f"test-{name}-{execution_revision}",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=10_000,
    )


def _context():
    return build_stable_context(
        instructions=(
            "You are a concise coding agent.",
            "Use structured tools only when they advance the task.",
        ),
        frozen_run_context=(
            ContextBlock(
                name="workspace",
                content={"root": ".", "write_scope": ("scratch", "reports")},
            ),
        ),
        initial_user_task="Inspect the repository and report the result.",
    )


def _settings(**overrides: object) -> ModelSettings:
    values: dict[str, object] = {
        "model": "test-model",
        "max_output_tokens": 1024,
        "temperature": 0.0,
        "top_p": 1.0,
        "parallel_tool_calls": True,
        "provider_options": {"reasoning_effort": "low"},
    }
    values.update(overrides)
    return ModelSettings(**values)  # type: ignore[arg-type]


def test_same_snapshot_builds_identical_canonical_request_and_revisions() -> None:
    schema = {
        "type": "object",
        "properties": {
            "zeta": {
                "type": "string",
                "enum": ["second", "first"],
            },
            "alpha": {"type": "integer"},
        },
        "additionalProperties": False,
    }
    selected_tools = (
        _tool("list_files"),
        _tool("mcp__server__second", schema=schema),
        _tool("mcp__server__first"),
    )
    context = _context()

    first = build_model_request(
        request_id="req-stable",
        context=context,
        selected_tools=selected_tools,
        settings=_settings(),
        tool_choice=ToolChoice.auto(),
    )
    second = build_model_request(
        request_id="req-stable",
        context=context,
        selected_tools=selected_tools,
        settings=_settings(),
        tool_choice=ToolChoice.auto(),
    )

    assert canonical_model_request_json(first) == canonical_model_request_json(second)
    assert first.prompt_revision == second.prompt_revision
    assert first.toolset_revision == second.toolset_revision
    assert first.exposed_tool_names == (
        "list_files",
        "mcp__server__second",
        "mcp__server__first",
    )
    assert tuple(tool.name for tool in first.tools) == first.exposed_tool_names

    payload = json.loads(canonical_model_request_json(first))
    assert payload["tools"][1]["input_schema"]["properties"]["zeta"]["enum"] == [
        "second",
        "first",
    ]
    properties_fragment = canonical_model_request_json(first).split(
        '"properties":',
        maxsplit=1,
    )[1]
    assert properties_fragment.index('"alpha"') < properties_fragment.index('"zeta"')


def test_mapping_key_order_does_not_change_canonical_hashes() -> None:
    first_schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {
            "beta": {"type": "string"},
            "alpha": {"type": "integer"},
        },
        "required": ["beta", "alpha"],
    }
    second_schema: Mapping[str, JsonValue] = {
        "required": ["beta", "alpha"],
        "properties": {
            "alpha": {"type": "integer"},
            "beta": {"type": "string"},
        },
        "type": "object",
    }

    first = build_model_request(
        request_id="req-order",
        context=_context(),
        selected_tools=(_tool("ordered", schema=first_schema),),
        settings=_settings(),
    )
    second = build_model_request(
        request_id="req-order",
        context=_context(),
        selected_tools=(_tool("ordered", schema=second_schema),),
        settings=_settings(),
    )

    assert first.toolset_revision == second.toolset_revision
    assert first.prompt_revision == second.prompt_revision
    assert canonical_model_request_json(first) == canonical_model_request_json(second)


def test_dynamic_runtime_diagnostics_do_not_change_the_stable_prefix() -> None:
    context = _context()
    tools = (_tool("list_files"),)
    first_tail = (
        context_event_message(
            "runtime_diagnostic",
            {
                "iteration": 1,
                "tool_success_count": 2,
                "tool_error_count": 0,
                "timestamp": "2026-07-13T10:00:00+08:00",
                "run_id": "run-a",
            },
        ),
    )
    second_tail = (
        context_event_message(
            "runtime_diagnostic",
            {
                "iteration": 99,
                "tool_success_count": 40,
                "tool_error_count": 7,
                "timestamp": "2030-01-01T00:00:00Z",
                "run_id": "run-b",
            },
        ),
    )

    first = build_model_request(
        request_id="req-a",
        context=context,
        selected_tools=tools,
        settings=_settings(),
        dynamic_tail=first_tail,
    )
    second = build_model_request(
        request_id="req-b",
        context=context,
        selected_tools=tools,
        settings=_settings(),
        dynamic_tail=second_tail,
    )

    assert stable_context_json(context) == stable_context_json(context)
    assert first.prompt_revision == second.prompt_revision
    assert first.toolset_revision == second.toolset_revision
    assert canonical_model_request_json(first) != canonical_model_request_json(second)
    stable_text = "\n".join(message.content for message in context.stable_messages)
    assert "run-a" not in stable_text
    assert "run-b" not in stable_text
    assert "Iteration" not in stable_text
    assert "list_files" not in stable_text


def test_selected_tool_order_is_semantic_and_never_globally_sorted() -> None:
    context = _context()
    selected = (
        _tool("list_files"),
        _tool("mcp__zeta__lookup"),
        _tool("mcp__alpha__lookup"),
    )

    request = build_model_request(
        request_id="req-activation-order",
        context=context,
        selected_tools=selected,
        settings=_settings(),
    )
    reordered = build_model_request(
        request_id="req-activation-order",
        context=context,
        selected_tools=(selected[0], selected[2], selected[1]),
        settings=_settings(),
    )

    assert request.exposed_tool_names == (
        "list_files",
        "mcp__zeta__lookup",
        "mcp__alpha__lookup",
    )
    assert request.toolset_revision != reordered.toolset_revision
    assert request.prompt_revision != reordered.prompt_revision


def test_cache_relevant_settings_change_the_prompt_revision() -> None:
    tools = (_tool("list_files"),)
    first = build_model_request(
        request_id="req-settings",
        context=_context(),
        selected_tools=tools,
        settings=_settings(temperature=0.0),
    )
    second = build_model_request(
        request_id="req-settings",
        context=_context(),
        selected_tools=tools,
        settings=_settings(temperature=0.2),
    )

    assert first.toolset_revision == second.toolset_revision
    assert first.prompt_revision != second.prompt_revision


def test_tool_result_model_content_is_snapshotted_once_in_the_transcript() -> None:
    context = _context()
    result = ToolResult(
        tool_call_id="call_read",
        tool_name="read_file",
        content=(ToolContentBlock(type="text", data={"text": "original result"}),),
        structured_content={"path": "README.md", "lines": 12},
        metadata={"formatter_revision": "old"},
    )

    appended = context.append_tool_result(result)
    stored_content = appended.transcript[-1].content

    def later_formatter(_result: ToolResult) -> str:
        return "changed formatter output"

    request = build_model_request(
        request_id="req-tool-result",
        context=appended,
        selected_tools=(_tool("read_file"),),
        settings=_settings(),
    )

    assert later_formatter(result) == "changed formatter output"
    assert appended.transcript[-1].content == stored_content
    assert request.messages[-1].content == stored_content
    assert "original result" in stored_content
    assert "changed formatter output" not in canonical_model_request_json(request)
    assert "formatter_revision" not in stored_content


def test_skill_activation_appends_a_canonical_event_without_new_revision() -> None:
    context = _context().append_message(ModelMessage(role="assistant", content="I will load the matching skill."))

    activated = context.append_skill_activation(
        {
            "event_type": "skill_activation",
            "success": True,
            "name": "demo",
            "skill_id": "project:demo",
            "instructions": "Follow the demo workflow.",
        }
    )

    assert context.transcript == (ModelMessage(role="assistant", content="I will load the matching skill."),)
    assert activated.transcript[:-1] == context.transcript
    assert activated.transcript[-1].role == "context"
    event = json.loads(activated.transcript[-1].content)
    assert event["event_type"] == "skill_activation"
    assert event["payload"]["skill_id"] == "project:demo"
    assert activated.context_revision == context.context_revision


def test_compaction_closes_the_old_revision_and_creates_a_new_one() -> None:
    context = _context()
    for index in range(3):
        context = context.append_message(ModelMessage(role="assistant", content=f"turn-{index}"))
    old_request = build_model_request(
        request_id="req-before-compact",
        context=context,
        selected_tools=(_tool("list_files"),),
        settings=_settings(),
    )

    compacted = context.compact(
        summary="Turns zero and one established the repository layout.",
        retained_tail=context.transcript[-1:],
    )
    new_request = build_model_request(
        request_id="req-after-compact",
        context=compacted,
        selected_tools=(_tool("list_files"),),
        settings=_settings(),
    )

    assert compacted.context_revision != context.context_revision
    assert compacted.parent_context_revision == context.context_revision
    assert compacted.revision_reason == "compaction"
    assert compacted.transcript[-1] == context.transcript[-1]
    assert "context_compaction" in compacted.transcript[0].content
    assert old_request.prompt_revision != new_request.prompt_revision
    assert context.transcript == tuple(ModelMessage(role="assistant", content=f"turn-{index}") for index in range(3))


def test_context_and_request_snapshot_caller_owned_mutable_values() -> None:
    block_content: dict[str, JsonValue] = {"paths": ["src"]}
    arguments = {"path": "README.md"}
    context = build_stable_context(
        instructions=("Stable instruction",),
        frozen_run_context=(ContextBlock("workspace", block_content),),
        initial_user_task="Read the file.",
    )
    request = build_model_request(
        request_id="req-snapshot",
        context=context,
        selected_tools=(_tool("read_file"),),
        settings=_settings(),
        dynamic_tail=(
            ModelMessage(
                role="assistant",
                content="",
                tool_calls=(ModelToolCall(id="call_1", name="read_file", input=arguments),),
            ),
        ),
    )
    before = canonical_model_request_json(request)

    block_content["paths"] = ("changed",)
    arguments["path"] = "CHANGED.md"

    assert canonical_model_request_json(request) == before
    assert "CHANGED.md" not in before


def test_direct_stable_context_construction_snapshots_caller_owned_sequences() -> None:
    instructions = ["Stable instruction"]
    blocks = [ContextBlock("workspace", {"paths": ["src"]})]
    transcript = [ModelMessage(role="assistant", content="Stable turn")]

    context = StableModelContext(
        instructions=instructions,  # type: ignore[arg-type]
        frozen_run_context=blocks,  # type: ignore[arg-type]
        initial_user_task="Inspect the repository.",
        transcript=transcript,  # type: ignore[arg-type]
        context_revision="context_persisted",
    )
    instructions[0] = "changed"
    blocks.clear()
    transcript.clear()

    assert context.instructions == ("Stable instruction",)
    assert tuple(block.name for block in context.frozen_run_context) == ("workspace",)
    assert context.transcript == (ModelMessage(role="assistant", content="Stable turn"),)


def test_canonical_message_snapshot_rejects_invalid_role_contracts() -> None:
    with pytest.raises(ValueError, match="tool messages require tool_call_id"):
        build_stable_context(
            instructions=("Stable instruction",),
            initial_user_task="Inspect the repository.",
            transcript=(ModelMessage(role="tool", content="result"),),
        )

    with pytest.raises(ValueError, match="only assistant messages may contain tool calls"):
        build_stable_context(
            instructions=("Stable instruction",),
            initial_user_task="Inspect the repository.",
            transcript=(
                ModelMessage(
                    role="user",
                    content="invalid call",
                    tool_calls=(ModelToolCall(id="call_1", name="read_file", input={}),),
                ),
            ),
        )


def test_canonical_request_module_is_dormant_and_provider_neutral() -> None:
    module_path = Path(model_request_module.__file__ or "")
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}

    assert not any(
        module.startswith(
            (
                "rag.agent.loop",
                "rag.agent.service",
                "rag.agent.tooling",
                "agent_runtime",
            )
        )
        for module in imports
    )
    assert "AgentMessageAssembler" not in source
    assert "LLMLoopModelTurnProvider" not in source
    assert "openai" not in source.lower()

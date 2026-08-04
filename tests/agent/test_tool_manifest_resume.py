from __future__ import annotations

import ast
import copy
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_runtime.core import checkpointing as checkpointing_module
from agent_runtime.core.checkpointing import (
    CanonicalToolCheckpoint,
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
    decode_legacy_tool_state_v1,
    decode_tool_checkpoint,
    encode_tool_checkpoint,
    reconcile_tool_manifest,
)
from agent_runtime.core.context import AgentRunConfig
from agent_runtime.core.messages import ModelMessage
from agent_runtime.core.model_request import (
    ModelSettings,
    bind_model_call_record,
    build_model_request,
    build_stable_context,
    build_tool_manifest,
    model_call_record_payload,
)
from agent_runtime.core.turn_contracts import ToolCallPlan, ToolManifestDriftStatus
from agent_runtime.loop.state import create_loop_state
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolDefinition,
    json_schema_input,
)
from rag.schema.llm import normalize_llm_usage

_FIXTURE = Path(__file__).parent / "fixtures/checkpoints/legacy_tool_state_v1.json"


def _tool(
    name: str,
    *,
    description: str | None = None,
    execution_revision: str = "runner-v1",
) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=description or f"Use {name}.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=lambda _arguments: {},
        normalize_output=lambda _raw: NormalizedToolOutput(),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision=execution_revision,
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=10_000,
    )


def _manifest(
    tools: tuple[Tool, ...],
    *,
    resident: tuple[str, ...],
    explicit: tuple[str, ...] = (),
    active: tuple[str, ...] = (),
    serializer_revision: str = "openai-compatible-chat-v1",
):
    return build_tool_manifest(
        tools=tools,
        resident_tool_names=resident,
        explicit_tool_names=explicit,
        active_tool_names=active,
        provider_serializer_revision=serializer_revision,
    )


def _origin_call(
    *,
    request_id: str,
    toolset_revision: str,
    exposed_tool_names: tuple[str, ...],
    tool_name: str,
    tool_call_id: str,
) -> ToolCall:
    return ToolCall(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments={"value": "x"},
        origin=ToolCallOrigin(
            request_id=request_id,
            toolset_revision=toolset_revision,
            exposed_tool_names=exposed_tool_names,
        ),
    )


def test_checkpoint_roundtrip_retains_the_originating_request_not_current_turn() -> None:
    tool_a = _tool("read_file")
    tool_b = _tool("list_files")
    context = build_stable_context(
        instructions=("Be precise.",),
        initial_user_task="Read the file.",
    )
    request_a = build_model_request(
        request_id="request-a",
        context=context,
        selected_tools=(tool_a,),
        settings=ModelSettings(model="test-model"),
    )
    request_b = build_model_request(
        request_id="request-b",
        context=context,
        selected_tools=(tool_b,),
        settings=ModelSettings(model="test-model"),
    )
    call = _origin_call(
        request_id=request_a.request_id,
        toolset_revision=request_a.toolset_revision,
        exposed_tool_names=request_a.exposed_tool_names,
        tool_name="read_file",
        tool_call_id="call-a",
    )
    usage = normalize_llm_usage(
        input_tokens=20,
        output_tokens=3,
        cache_read_input_tokens=5,
        input_tokens_include_cache=True,
        usage_source="provider",
        raw_provider_usage={"prompt_tokens": 20, "completion_tokens": 3},
    )
    record = bind_model_call_record(
        request=request_a,
        provider_wire_hash="wire-request-a",
        usage=usage,
    )
    checkpoint = CanonicalToolCheckpoint(
        context_revision=context.context_revision,
        prompt_revision=request_b.prompt_revision,
        transcript=(ModelMessage(role="user", content="Read the file."),),
        manifest=_manifest((tool_b,), resident=("list_files",)),
        pending_tool_calls=(call,),
        paused_tool_calls=(),
        model_call_records=(record,),
    )

    encoded = encode_tool_checkpoint(checkpoint)
    json.dumps(encoded, ensure_ascii=False, allow_nan=False)
    restored = decode_tool_checkpoint(encoded)

    assert request_b.request_id != request_a.request_id
    assert request_b.exposed_tool_names != request_a.exposed_tool_names
    assert restored.pending_tool_calls[0].origin.request_id == "request-a"
    assert restored.pending_tool_calls[0].origin.toolset_revision == request_a.toolset_revision
    assert restored.pending_tool_calls[0].origin.exposed_tool_names == ("read_file",)
    assert model_call_record_payload(restored.model_call_records[0]) == model_call_record_payload(record)


def test_checkpoint_rejects_conflicting_origin_for_the_same_pending_call() -> None:
    tool = _tool("read_file")
    manifest = _manifest((tool,), resident=("read_file",))
    accepted = _origin_call(
        request_id="request-a",
        toolset_revision=manifest.toolset_revision,
        exposed_tool_names=("read_file",),
        tool_name="read_file",
        tool_call_id="same-call",
    )
    overwritten = _origin_call(
        request_id="request-b",
        toolset_revision="tools-request-b",
        exposed_tool_names=("list_files",),
        tool_name="read_file",
        tool_call_id="same-call",
    )

    with pytest.raises(ValueError, match="must exactly match"):
        CanonicalToolCheckpoint(
            context_revision="context-a",
            prompt_revision="prompt-a",
            transcript=(ModelMessage(role="user", content="Read the file."),),
            manifest=manifest,
            tool_calls=(accepted,),
            pending_tool_calls=(overwritten,),
        )


def test_checkpoint_decode_rejects_inconsistent_normalized_usage() -> None:
    tool = _tool("read_file")
    raw = encode_tool_checkpoint(
        CanonicalToolCheckpoint(
            context_revision="context-a",
            prompt_revision="prompt-a",
            transcript=(ModelMessage(role="user", content="Read the file."),),
            manifest=_manifest((tool,), resident=("read_file",)),
        )
    )
    raw["model_call_records"] = [
        {
            "request_id": "request-a",
            "prompt_revision": "prompt-a",
            "toolset_revision": "tools-a",
            "provider_wire_hash": "wire-a",
            "usage": {
                "logical_input_tokens": 10,
                "uncached_input_tokens": 9,
                "cache_read_input_tokens": 5,
                "cache_write_input_tokens": None,
                "output_tokens": 2,
                "usage_source": "provider",
                "raw_provider_usage": None,
            },
        }
    ]

    with pytest.raises(ValueError, match="inconsistent normalized usage"):
        decode_tool_checkpoint(raw)


def test_matching_manifest_retains_revision_and_wire_hash_guarantee() -> None:
    tools = (
        _tool("read_file"),
        _tool("run_command"),
        _tool("mcp__docs__search"),
    )
    persisted = _manifest(
        tools,
        resident=("read_file",),
        explicit=("run_command",),
        active=("mcp__docs__search",),
    )
    rebuilt = _manifest(
        tools,
        resident=("read_file",),
        explicit=("run_command",),
        active=("mcp__docs__search",),
    )

    decision = reconcile_tool_manifest(
        persisted=persisted,
        rebuilt=rebuilt,
        pending_tool_calls=(),
        paused_tool_calls=(),
    )

    assert decision.status is ToolManifestDriftStatus.MATCH
    assert decision.toolset_revision == persisted.toolset_revision
    assert decision.active_tool_names == ("mcp__docs__search",)
    assert decision.provider_wire_hash_guaranteed is True
    assert decision.dependent_tool_calls == ()
    assert tuple(entry.name for entry in persisted.entries) == (
        "read_file",
        "run_command",
        "mcp__docs__search",
    )


def test_changed_tool_with_pending_call_requires_reconciliation() -> None:
    persisted = _manifest(
        (_tool("read_file", execution_revision="runner-v1"),),
        resident=("read_file",),
    )
    rebuilt = _manifest(
        (_tool("read_file", execution_revision="runner-v2"),),
        resident=("read_file",),
    )
    pending = _origin_call(
        request_id="request-a",
        toolset_revision=persisted.toolset_revision,
        exposed_tool_names=("read_file",),
        tool_name="read_file",
        tool_call_id="call-pending",
    )

    decision = reconcile_tool_manifest(
        persisted=persisted,
        rebuilt=rebuilt,
        pending_tool_calls=(pending,),
        paused_tool_calls=(),
    )

    assert decision.status is ToolManifestDriftStatus.RECONCILIATION_REQUIRED
    assert decision.reason == "tool_definition_changed"
    assert decision.toolset_revision == persisted.toolset_revision
    assert decision.dependent_tool_calls == (pending,)
    assert decision.dependent_tool_calls[0].origin.request_id == "request-a"
    assert decision.provider_wire_hash_guaranteed is False
    payload = decision.model_dump(mode="json")
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert payload["dependent_tool_calls"][0]["origin"]["request_id"] == "request-a"


def test_missing_active_tool_with_paused_call_retains_origin_evidence() -> None:
    persisted_tools = (_tool("read_file"), _tool("mcp__docs__search"))
    persisted = _manifest(
        persisted_tools,
        resident=("read_file",),
        active=("mcp__docs__search",),
    )
    rebuilt = _manifest((_tool("read_file"),), resident=("read_file",))
    paused = _origin_call(
        request_id="request-paused",
        toolset_revision=persisted.toolset_revision,
        exposed_tool_names=("read_file", "mcp__docs__search"),
        tool_name="mcp__docs__search",
        tool_call_id="call-paused",
    )

    decision = reconcile_tool_manifest(
        persisted=persisted,
        rebuilt=rebuilt,
        pending_tool_calls=(),
        paused_tool_calls=(paused,),
    )

    assert decision.status is ToolManifestDriftStatus.RECONCILIATION_REQUIRED
    assert decision.active_tool_names == ("mcp__docs__search",)
    assert decision.missing_tool_names == ("mcp__docs__search",)
    assert decision.dependent_tool_calls == (paused,)
    assert decision.dependent_tool_calls[0].origin.exposed_tool_names == (
        "read_file",
        "mcp__docs__search",
    )


def test_drift_without_dependent_call_removes_missing_active_and_creates_revision() -> None:
    persisted = _manifest(
        (_tool("read_file"), _tool("mcp__docs__search")),
        resident=("read_file",),
        active=("mcp__docs__search",),
    )
    rebuilt = _manifest((_tool("read_file"),), resident=("read_file",))

    decision = reconcile_tool_manifest(
        persisted=persisted,
        rebuilt=rebuilt,
        pending_tool_calls=(),
        paused_tool_calls=(),
    )

    assert decision.status is ToolManifestDriftStatus.NEW_REVISION_REQUIRED
    assert decision.toolset_revision == rebuilt.toolset_revision
    assert decision.active_tool_names == ()
    assert decision.missing_tool_names == ("mcp__docs__search",)
    assert decision.provider_wire_hash_guaranteed is False


def test_serializer_revision_change_ends_old_wire_hash_guarantee() -> None:
    tools = (_tool("read_file"),)
    persisted = _manifest(
        tools,
        resident=("read_file",),
        serializer_revision="openai-compatible-chat-v1",
    )
    rebuilt = _manifest(
        tools,
        resident=("read_file",),
        serializer_revision="openai-compatible-chat-v2",
    )

    decision = reconcile_tool_manifest(
        persisted=persisted,
        rebuilt=rebuilt,
        pending_tool_calls=(),
        paused_tool_calls=(),
    )

    assert decision.status is ToolManifestDriftStatus.NEW_REVISION_REQUIRED
    assert decision.reason == "provider_serializer_changed"
    assert decision.provider_wire_hash_guaranteed is False


def test_legacy_fixture_rebuilds_canonical_transcript_once_without_active_wiring() -> None:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    original = copy.deepcopy(raw)

    with pytest.raises(ValueError, match="unsupported tool checkpoint"):
        decode_tool_checkpoint(raw)
    migrated = decode_legacy_tool_state_v1(raw)
    encoded = encode_tool_checkpoint(migrated)
    restored = decode_tool_checkpoint(encoded)

    assert raw == original
    assert migrated.legacy_migrated is True
    assert [message.role for message in migrated.transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "legacy read result" in migrated.transcript[2].content
    assert migrated.manifest.active_tool_names == ("mcp__docs__search",)
    assert migrated.paused_tool_calls[0].origin.request_id == "legacy_req_a"
    assert encoded["format_version"] == 2
    assert "tool_call_ledger" not in encoded
    assert "tool_results" not in encoded
    assert encode_tool_checkpoint(restored) == encoded

    module_path = Path(checkpointing_module.__file__ or "")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    active_functions = {
        "_normalize_loaded_state",
        "_migrate_legacy_state",
    }
    active_calls = {
        node.func.id
        for function in tree.body
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and function.name in active_functions
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "decode_legacy_tool_state_v1" not in active_calls
    legacy_function = next(
        function
        for function in tree.body
        if isinstance(function, ast.FunctionDef) and function.name == "decode_legacy_tool_state_v1"
    )
    legacy_source = ast.get_source_segment(
        module_path.read_text(encoding="utf-8"),
        legacy_function,
    )
    assert legacy_source is not None
    assert "formatter" not in legacy_source
    assert "_migrate_legacy_state" not in legacy_source


@pytest.mark.anyio
async def test_real_checkpoint_store_uses_v2_codec_and_preserves_call_origin() -> None:
    run_id = "real-canonical-checkpoint"
    tool_a = _tool("read_file")
    tool_b = _tool("list_files")
    context = build_stable_context(
        instructions=("Be precise.",),
        initial_user_task="Read the file.",
    )
    request_a = build_model_request(
        request_id="request-a",
        context=context,
        selected_tools=(tool_a,),
        settings=ModelSettings(model="test-model"),
    )
    request_b = build_model_request(
        request_id="request-b",
        context=context,
        selected_tools=(tool_b,),
        settings=ModelSettings(model="test-model"),
    )
    call = _origin_call(
        request_id=request_a.request_id,
        toolset_revision=request_a.toolset_revision,
        exposed_tool_names=request_a.exposed_tool_names,
        tool_name="read_file",
        tool_call_id="call-a",
    )
    record = bind_model_call_record(
        request=request_a,
        provider_wire_hash="wire-request-a",
        usage=normalize_llm_usage(
            input_tokens=20,
            output_tokens=3,
            cache_read_input_tokens=5,
            cache_write_input_tokens=0,
            input_tokens_include_cache=True,
            usage_source="provider",
            raw_provider_usage={"prompt_tokens": 20},
        ),
    )
    config = AgentRunConfig(
        turn_id=run_id,
    )
    state = create_loop_state(
        current_message="Read the file.",
        run_config=config,
        pending_tool_calls=(
            ToolCallPlan(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                arguments={"value": "x"},
            ),
        ),
    )
    state["turn_transcript"] = [ModelMessage(role="user", content="Read the file.")]  # type: ignore[typeddict-unknown-key]
    state["context_revision"] = context.context_revision  # type: ignore[typeddict-unknown-key]
    state["prompt_revision"] = request_b.prompt_revision  # type: ignore[typeddict-unknown-key]
    state["tool_manifest"] = _manifest((tool_b,), resident=("list_files",))  # type: ignore[typeddict-unknown-key]
    state["canonical_tool_calls"] = {call.tool_call_id: call}  # type: ignore[typeddict-unknown-key]
    state["model_call_records"] = [record]  # type: ignore[typeddict-unknown-key]
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    store = LangGraphCheckpointStore(checkpointer, run_config=config)

    await store.save_snapshot(state, reason="canonical-origin")
    loaded = await store.load_latest()

    assert loaded is not None
    restored = decode_tool_checkpoint(loaded["tool_checkpoint"])  # type: ignore[typeddict-item]
    assert restored.prompt_revision == request_b.prompt_revision
    assert restored.pending_tool_calls == (call,)
    assert restored.pending_tool_calls[0].origin.request_id == "request-a"
    assert restored.pending_tool_calls[0].origin.toolset_revision == request_a.toolset_revision
    assert restored.pending_tool_calls[0].origin.exposed_tool_names == ("read_file",)
    assert restored.model_call_records == (record,)

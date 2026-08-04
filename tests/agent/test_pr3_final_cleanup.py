from __future__ import annotations

import ast
import base64
import importlib
from pathlib import Path

import pytest

from agent_runtime.core.checkpointing import (
    _migrate_legacy_state,
    agent_checkpoint_serde,
)
from agent_runtime.core.context import AgentRunConfig, TurnRegistry
from agent_runtime.core.messages import (
    ModelMessage,
    tool_result_message,
)
from agent_runtime.core.messages import (
    ToolCall as ModelToolCall,
)
from agent_runtime.core.model_request import StableModelContext
from agent_runtime.core.turn_contracts import ToolCallPlan
from agent_runtime.loop.state import (
    PendingToolCall,
    ToolCallLedger,
    create_loop_state,
)
from agent_runtime.loop.substate import MemoryState
from agent_runtime.tools.tool import ToolContentBlock, ToolResult
from rag.models.config import GenerationConfig
from rag.schema.llm import LLMCallStage


def _production_source(relative_path: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / relative_path).read_text(encoding="utf-8")


def _config() -> AgentRunConfig:
    return AgentRunConfig(
        turn_id="test-pr3-cleanup",
        llm_budget_total=100,
    )


def test_runtime_helpers_have_one_canonical_owner() -> None:
    cli_source = _production_source("agent_runtime/cli.py")
    loop_source = _production_source("agent_runtime/loop/runtime.py")
    sink_source = _production_source("agent_runtime/streaming/sink.py")

    assert "def _build_model_control_plane(" not in cli_source
    assert "def _format_public_tool_summary(" not in cli_source
    assert "class NoopStreamEventSink" not in sink_source
    for duplicate in (
        "_stream_turn_start",
        "_stream_turn_end",
        "_stream_loop_end",
        "_stream_tool_use_start",
        "_stream_tool_use_result",
        "_stream_tool_use_error",
        "_stream_compact_layer",
        "_stream_recovery",
    ):
        assert f"def {duplicate}(" not in loop_source


def test_substate_does_not_import_its_owner_module() -> None:
    tree = ast.parse(_production_source("agent_runtime/loop/substate.py"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "agent_runtime.loop.state" not in imports


def test_local_runtime_only_type_checks_public_model_contract() -> None:
    tree = ast.parse(_production_source("agent_runtime/local_runtime.py"))
    runtime_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }

    assert "agent_runtime.models" not in runtime_imports


def test_memory_digest_does_not_depend_on_checkpointing() -> None:
    compactor_tree = ast.parse(_production_source("agent_runtime/memory/compactor.py"))
    compactor_imports = {
        node.module
        for node in ast.walk(compactor_tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "agent_runtime.core.checkpointing" not in compactor_imports
    assert "def _digest_text(" not in _production_source(
        "agent_runtime/core/checkpointing.py"
    )


def test_orphaned_persistent_memory_capability_is_removed() -> None:
    root = Path(__file__).resolve().parents[2]
    state = create_loop_state(current_message="cleanup", run_config=_config())
    persistent_sources = root / "agent_runtime/memory/persistent"

    assert not tuple(persistent_sources.glob("*.py"))
    assert not (root / "tests/agent/test_persistent_memory.py").exists()
    assert "persistent_memories" not in state
    assert "memory_index" not in state
    assert "persistent" not in MemoryState.model_fields
    assert "initial_memory" not in StableModelContext.__dataclass_fields__
    assert {
        "memory_select",
        "memory_extract",
        "memory_consolidate",
    }.isdisjoint(stage.value for stage in LLMCallStage)
    assert {
        "memory_select",
        "memory_extract",
        "memory_consolidate",
    }.isdisjoint(GenerationConfig.__dataclass_fields__)


def test_checkpoint_decode_rejects_legacy_memory_state_identity() -> None:
    legacy_payload = base64.b64decode(
        "yAFdBZS3cmFnLmFnZW50Lmxvb3Auc3Vic3RhdGWrTWVtb3J5U3RhdGWL"
        "r3dvcmtpbmdfc3VtbWFyecCvZXh0cmFjdGVkX2ZhY3RzkLNyZWNlbnRf"
        "b2JzZXJ2YXRpb25zkLh2ZXJpZmllZF93b3Jrc3BhY2VfcGF0aHOQrmtu"
        "b3duX2xvY2F0b3JzkK5jb250ZXh0X2J1ZGdldMCrbWVtb3J5X3JlZnOQ"
        "rW1lbW9yeV9idWRnZXTAr21lbW9yeV93YXJuaW5nc5GmbGVnYWN5tXJl"
        "YWN0aXZlX2NvbXBhY3RfdXNlZMKqcGVyc2lzdGVudISpaW5kZXhfcmVm"
        "qU1FTU9SWS5tZKxpbmRleF9kaWdlc3StbGVnYWN5IGRpZ2VzdK5zZWxl"
        "Y3RlZF9jb3VudAKyc2VsZWN0ZWRfc3VtbWFyaWVzkqNvbmWjdHdvs21v"
        "ZGVsX3ZhbGlkYXRlX2pzb24="
    )

    restored = agent_checkpoint_serde().loads_typed(
        ("msgpack", legacy_payload)
    )

    assert isinstance(restored, dict)
    assert not isinstance(restored, MemoryState)
    assert restored["memory_warnings"] == ["legacy"]
    assert restored["persistent"]["index_ref"] == "MEMORY.md"


def test_model_provider_contracts_do_not_import_loop_implementation() -> None:
    for relative_path in (
        "agent_runtime/core/llm_providers.py",
        "agent_runtime/core/model_provider_runtime.py",
    ):
        tree = ast.parse(_production_source(relative_path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }

        assert "agent_runtime.loop.runtime" not in imports


def test_checkpoint_legacy_alias_does_not_import_public_facade() -> None:
    tree = ast.parse(_production_source("agent_runtime/core/checkpointing.py"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "agent_runtime" not in imports


def test_json_contract_helpers_are_not_reimplemented_by_consumers() -> None:
    for relative_path in (
        "agent_runtime/core/messages.py",
        "agent_runtime/core/model_request.py",
        "agent_runtime/tools/selection.py",
    ):
        source = _production_source(relative_path)
        assert "def _thaw_json(" not in source
        assert "def _require_non_empty_string(" not in source
        assert "def _require_bool(" not in source


_DEPRECATED_FIELDS = frozenset(
    {
        "retrieval_signals",
        "retrieval_signals_debug",
        "evidence",
        "citations",
        "evidence_refs",
        "answer_candidates",
        "computation_results",
        "structured_observations",
        "context_units",
        "context_bindings",
        "locators",
        "asset_refs",
        "persistent_memories",
        "memory_index",
    }
)


def _result(plan: ToolCallPlan) -> ToolResult:
    return ToolResult(
        tool_call_id=plan.tool_call_id,
        tool_name=plan.tool_name,
        content=(ToolContentBlock(type="text", data={"text": "result ok"}),),
        structured_content={"result": "ok"},
    )


def test_pending_single_track_roundtrip() -> None:
    plan = ToolCallPlan.create("search_knowledge", {"query": "policy"})
    pending = PendingToolCall(
        plan=plan,
        status="approved",
        summary="test summary",
    )

    serde = agent_checkpoint_serde()
    restored = serde.loads_typed(serde.dumps_typed(pending))

    assert isinstance(restored, PendingToolCall)
    assert restored.tool_call_id == plan.tool_call_id
    assert restored.tool_name == "search_knowledge"
    assert restored.status == "approved"
    assert restored.plan.arguments == {"query": "policy"}


def test_tool_call_ledger_is_bounded_fifo_and_preserves_active_calls() -> None:
    ledger = ToolCallLedger(max_entries=128)
    active_ids: set[str] = set()
    for index in range(130):
        plan = ToolCallPlan.create(f"tool_{index}", {"index": index})
        if index >= 125:
            active_ids.add(plan.tool_call_id)
        ledger.append_plans([plan], turn=1)

    ledger.trim(active_tool_call_ids=active_ids)

    assert len(ledger.entries) == 128
    assert active_ids.issubset(entry.plan.tool_call_id for entry in ledger.entries)


def test_canonical_transcript_preserves_calls_arguments_and_results() -> None:
    state = create_loop_state(current_message="transcript", run_config=_config())
    plans = [
        ToolCallPlan.create("search", {"query": "Paris", "top_k": 5}),
        ToolCallPlan.create("analyze", {"data": [1, 2, 3]}),
    ]
    state["tool_call_ledger"].append_plans(plans, turn=1)
    results = [_result(plan) for plan in plans]
    state["tool_results"] = results
    transcript: list[ModelMessage] = []
    for plan, result in zip(plans, results, strict=True):
        transcript.extend(
            [
                ModelMessage(
                    role="assistant",
                    content="",
                    tool_calls=(
                        ModelToolCall(
                            id=plan.tool_call_id,
                            name=plan.tool_name,
                            input=dict(plan.arguments),
                        ),
                    ),
                ),
                tool_result_message(result),
            ]
        )
    state["turn_transcript"] = transcript

    assert len(transcript) == 4
    assert transcript[0].tool_calls[0].name == "search"
    assert transcript[0].tool_calls[0].input == {
        "query": "Paris",
        "top_k": 5,
    }
    assert transcript[1].role == "tool"
    assert transcript[1].tool_call_id == plans[0].tool_call_id
    assert transcript[2].tool_calls[0].input == {"data": [1, 2, 3]}
    assert transcript[3].tool_call_id == plans[1].tool_call_id


def test_deprecated_fields_are_absent_from_new_and_migrated_state() -> None:
    state = create_loop_state(current_message="new state", run_config=_config())
    assert _DEPRECATED_FIELDS.isdisjoint(state)

    raw: dict[str, object] = dict(state)
    raw.update(
        {
            "loop_messages": [{"type": "human", "content": "old"}],
            "tool_result_store": {"old": "value"},
            "evidence": [],
            "citations": [],
            "retrieval_signals": None,
        }
    )
    migrated = _migrate_legacy_state(raw)

    assert _DEPRECATED_FIELDS.isdisjoint(migrated)
    assert "loop_messages" not in migrated
    assert "tool_result_store" not in migrated


def test_removed_agent_state_module_stays_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_runtime.state")


def test_live_state_serde_preserves_final_result_and_ledger() -> None:
    plan = ToolCallPlan.create(
        "search_knowledge",
        {"query": "capital of France"},
    )
    state = create_loop_state(current_message="capital", run_config=_config())
    state["pending_tool_calls"] = [PendingToolCall(plan=plan, status="completed", summary="Paris")]
    state["tool_call_ledger"].append_plans([plan], turn=1)
    state["tool_results"] = [
        ToolResult(
            tool_call_id=plan.tool_call_id,
            tool_name=plan.tool_name,
            structured_content={
                "answer_text": "Paris",
                "results": [{"evidence_id": "ev-1"}],
            },
        )
    ]

    serde = agent_checkpoint_serde()
    restored = serde.loads_typed(serde.dumps_typed(state))

    assert _DEPRECATED_FIELDS.isdisjoint(restored)
    result = restored["tool_results"][0]
    assert isinstance(result, ToolResult)
    assert result.structured_content["results"][0]["evidence_id"] == "ev-1"
    assert len(restored["tool_call_ledger"].entries) == 1
    assert restored["pending_tool_calls"][0].status == "completed"


@pytest.fixture(autouse=True)
def _cleanup_run_registry() -> None:
    yield
    TurnRegistry.remove("test-pr3-cleanup")

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from rag.agent.core.checkpointing import (
    _migrate_legacy_state,
    agent_checkpoint_serde,
)
from rag.agent.core.context import AgentRunConfig, TurnRegistry
from rag.agent.core.messages import (
    ModelMessage,
    tool_result_message,
)
from rag.agent.core.messages import (
    ToolCall as ModelToolCall,
)
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.state import (
    PendingToolCall,
    ToolCallLedger,
    create_loop_state,
)
from rag.agent.tools.tool import ToolContentBlock, ToolResult


def _production_source(relative_path: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / relative_path).read_text(encoding="utf-8")


def _config() -> AgentRunConfig:
    return AgentRunConfig(
        turn_id="test-pr3-cleanup",
        llm_budget_total=100,
    )


def test_runtime_helpers_have_one_canonical_owner() -> None:
    cli_source = _production_source("rag/agent/cli.py")
    loop_source = _production_source("rag/agent/loop/runtime.py")
    sink_source = _production_source("rag/agent/streaming/sink.py")

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
    tree = ast.parse(_production_source("rag/agent/loop/substate.py"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "rag.agent.loop.state" not in imports


def test_local_runtime_only_type_checks_public_model_contract() -> None:
    tree = ast.parse(_production_source("agent_runtime/local_runtime.py"))
    runtime_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }

    assert "agent_runtime.models" not in runtime_imports


def test_memory_digest_does_not_depend_on_checkpointing() -> None:
    compactor_tree = ast.parse(_production_source("rag/agent/memory/compactor.py"))
    compactor_imports = {
        node.module
        for node in ast.walk(compactor_tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "rag.agent.core.checkpointing" not in compactor_imports
    assert "def _digest_text(" not in _production_source(
        "rag/agent/core/checkpointing.py"
    )
    assert "def _digest_text(" not in _production_source(
        "rag/agent/memory/persistent/runtime.py"
    )


def test_model_provider_contracts_do_not_import_loop_implementation() -> None:
    for relative_path in (
        "rag/agent/core/llm_providers.py",
        "rag/agent/core/model_provider_runtime.py",
    ):
        tree = ast.parse(_production_source(relative_path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }

        assert "rag.agent.loop.runtime" not in imports


def test_checkpoint_legacy_alias_does_not_import_public_facade() -> None:
    tree = ast.parse(_production_source("rag/agent/core/checkpointing.py"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "agent_runtime" not in imports


def test_json_contract_helpers_are_not_reimplemented_by_consumers() -> None:
    for relative_path in (
        "rag/agent/core/messages.py",
        "rag/agent/core/model_request.py",
        "rag/agent/tools/selection.py",
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
        importlib.import_module("rag.agent.state")


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

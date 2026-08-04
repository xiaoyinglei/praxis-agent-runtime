from __future__ import annotations

import asyncio
import logging
from dataclasses import make_dataclass
from pathlib import Path

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import create_model

from agent_runtime.core.checkpointing import (
    _normalize_loaded_state,
    aclose_agent_checkpointer,
    agent_checkpoint_serde,
    create_agent_checkpointer,
)
from agent_runtime.core.context import AgentRunConfig
from agent_runtime.core.messages import ModelMessage
from agent_runtime.core.runtime_diagnostics import AgentLatencyProfile, ToolCallMetrics
from agent_runtime.core.turn_contracts import ToolCallPlan
from agent_runtime.loop.state import PendingToolCall, create_loop_state
from agent_runtime.memory.models import ExternalizedToolOutput, MemoryRef
from agent_runtime.planning import AgentPlan, PlanStep
from agent_runtime.primitive_ops import (
    CandidateHeaderRow,
    StructuredProbeOutput,
    StructuredTableProbe,
)
from agent_runtime.tools.tool import ToolContentBlock, ToolResult
from rag.schema.runtime import AccessPolicy


def _round_trip_without_warning(
    value: object,
    caplog: pytest.LogCaptureFixture,
) -> object:
    serde = agent_checkpoint_serde()
    with caplog.at_level(
        logging.WARNING,
        logger="langgraph.checkpoint.serde.jsonplus",
    ):
        restored = serde.loads_typed(serde.dumps_typed(value))
    assert "Deserializing unregistered type" not in caplog.text
    return restored


def test_checkpoint_serde_restores_canonical_tool_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = ToolResult(
        tool_call_id="tc-list",
        tool_name="list_files",
        content=(ToolContentBlock(type="text", data={"text": "sales.csv"}),),
        structured_content={
            "entries": [
                {
                    "name": "sales.csv",
                    "path": "input_files/sales.csv",
                    "size": 26,
                    "is_dir": False,
                }
            ]
        },
        metadata={"latency_ms": 0.5},
    )

    restored = _round_trip_without_warning(result, caplog)

    assert isinstance(restored, ToolResult)
    assert restored.content == result.content
    assert restored.structured_content == result.structured_content
    assert restored.metadata == result.metadata


def test_checkpoint_serde_restores_agent_plan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    plan = AgentPlan(
        objective="Answer with a bounded plan.",
        active_step_id="step_answer",
        steps=[
            PlanStep(
                step_id="step_answer",
                title="Produce the final answer",
                status="in_progress",
            )
        ],
    )

    restored = _round_trip_without_warning(plan, caplog)

    assert isinstance(restored, AgentPlan)
    assert restored == plan


def test_checkpoint_serde_rejects_legacy_plan_module_identity() -> None:
    legacy_plan_type = create_model(
        "AgentPlan",
        __module__="rag." + "agent.planning",
        objective=(str, ...),
        status=(str, "active"),
        revision=(int, 0),
        active_step_id=(str | None, None),
        steps=(list[dict[str, object]], []),
        summary=(str | None, None),
    )
    legacy_plan = legacy_plan_type(
        objective="Legacy checkpoint plan.",
        active_step_id="step_migrate",
        steps=[
            {
                "step_id": "step_migrate",
                "title": "Migrate the legacy plan.",
            }
        ],
    )
    encoded = JsonPlusSerializer(
        allowed_msgpack_modules=True,
    ).dumps_typed(legacy_plan)

    restored = agent_checkpoint_serde().loads_typed(encoded)

    assert isinstance(restored, dict)
    assert not isinstance(restored, AgentPlan)
    assert restored["objective"] == "Legacy checkpoint plan."
    assert restored["steps"][0]["step_id"] == "step_migrate"


def test_checkpoint_serde_restores_checkpoint_stable_structured_preview(
    caplog: pytest.LogCaptureFixture,
) -> None:
    preview = StructuredProbeOutput(
        path="input_files/sales.csv",
        file_kind="text",
        mime_type="text/csv",
        tables=[
            StructuredTableProbe(
                table_index=0,
                name="sales.csv",
                used_range="A1:B2",
                row_count=2,
                column_count=2,
                sample_rows=[["name", "amount"], ["Alice", 100]],
                candidate_header_rows=[
                    CandidateHeaderRow(
                        row_index=1,
                        confidence=0.95,
                        reason="fixture",
                    )
                ],
                data_start_row=2,
            )
        ],
    )

    restored = _round_trip_without_warning(preview, caplog)

    assert isinstance(restored, StructuredProbeOutput)
    assert restored == preview


def test_checkpoint_serde_normalizes_removed_primitive_model_to_mapping() -> None:
    legacy_model = create_model(
        "ListFilesOutput",
        __module__="rag." + "agent.primitive_ops",
        files=(list[dict[str, object]], ...),
        truncated=(bool, False),
    )
    legacy_output = legacy_model(files=[{"path": "input_files/sales.csv", "is_dir": False}])
    permissive_serde = JsonPlusSerializer(allowed_msgpack_modules=True)
    encoded = permissive_serde.dumps_typed(legacy_output)

    restored = agent_checkpoint_serde().loads_typed(encoded)

    assert restored == {
        "files": [{"path": "input_files/sales.csv", "is_dir": False}],
        "truncated": False,
    }


def test_checkpoint_serde_rejects_legacy_run_config_identity() -> None:
    legacy_config_type = make_dataclass(
        "AgentRunConfig",
        (
            ("run_id", str),
            ("thread_id", str),
            ("llm_budget_total", int | None),
            ("max_depth", int),
            ("access_policy", AccessPolicy),
            ("agent_type", str | None),
            ("source_scope", tuple[str, ...]),
        ),
        frozen=True,
        module="rag." + "agent.core.context",
    )
    config = legacy_config_type(
        run_id="legacy-run-config",
        thread_id="legacy-run-config",
        llm_budget_total=100,
        max_depth=1,
        access_policy=AccessPolicy.default(),
        agent_type="generic",
        source_scope=("legacy-source",),
    )
    encoded = JsonPlusSerializer(
        allowed_msgpack_modules=True,
    ).dumps_typed(config)

    restored = agent_checkpoint_serde().loads_typed(encoded)

    assert isinstance(restored, dict)
    assert not isinstance(restored, AgentRunConfig)
    assert restored["run_id"] == "legacy-run-config"
    assert restored["thread_id"] == "legacy-run-config"
    assert restored["llm_budget_total"] == 100


def test_legacy_message_context_migrates_without_losing_conversation_history() -> None:
    current = ModelMessage(role="user", content="what was the token?")
    restored = _normalize_loaded_state(  # type: ignore[arg-type]
        {
            "task": "remember alpha",
            "canonical_transcript": [
                ModelMessage(role="assistant", content="remembered"),
                current,
            ],
            "turn_transcript": [current],
            "runtime_diagnostics": [],
        }
    )

    assert restored["current_message"] == "what was the token?"
    assert restored["conversation_history"] == [
        ModelMessage(role="user", content="remember alpha"),
        ModelMessage(role="assistant", content="remembered"),
    ]
    assert restored["turn_transcript"] == [current]
    assert "task" not in restored
    assert "canonical_transcript" not in restored


def test_checkpoint_serde_restores_externalized_output_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    output = ExternalizedToolOutput(
        original_output_model="agent_runtime.tools.builtins.shell.RunCommandOutput",
        summary="run_command ok=True exit_code=0 stdout_preview=large",
        ref=MemoryRef(
            ref_id="mem_abc",
            path=".agent_memory/records/mem_abc.json",
            summary="run_command ok=True exit_code=0",
            source_tool_call_id="tc-big",
            source_tool_name="run_command",
            size_bytes=1024,
        ),
        status="available",
    )

    restored = _round_trip_without_warning(output, caplog)

    assert isinstance(restored, ExternalizedToolOutput)
    assert restored.ref.ref_id == "mem_abc"


@pytest.mark.parametrize(
    "value",
    [
        ToolCallMetrics(native_calls=2, native_latency_ms_total=15.5),
        AgentLatencyProfile(total_ms=12.5, model_latency_ms=4.0),
    ],
)
def test_checkpoint_serde_restores_runtime_diagnostics(
    value: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    restored = _round_trip_without_warning(value, caplog)

    assert restored == value


def test_sqlite_checkpointer_constructs_without_running_event_loop(
    tmp_path: Path,
) -> None:
    checkpointer = create_agent_checkpointer(tmp_path / "agent-checkpoints.sqlite")

    assert checkpointer is not None
    asyncio.run(aclose_agent_checkpointer(checkpointer))


def test_sqlite_checkpointer_deletes_one_turn_thread(tmp_path: Path) -> None:
    async def exercise() -> None:
        checkpointer = create_agent_checkpointer(tmp_path / "agent-checkpoints.sqlite")
        config = {"configurable": {"thread_id": "turn-delete", "checkpoint_ns": ""}}
        saved = await checkpointer.aput(
            config,
            empty_checkpoint(),
            {},
            {},
        )
        assert await checkpointer.aget_tuple(saved) is not None

        await checkpointer.adelete_thread("turn-delete")

        assert await checkpointer.aget_tuple(saved) is None
        await aclose_agent_checkpointer(checkpointer)

    asyncio.run(exercise())


def test_sqlite_checkpointer_delete_is_safe_before_first_checkpoint(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        checkpointer = create_agent_checkpointer(tmp_path / "agent-checkpoints.sqlite")

        await checkpointer.adelete_thread("missing-turn")

        await aclose_agent_checkpointer(checkpointer)

    asyncio.run(exercise())


def test_live_loop_state_serde_preserves_final_tool_result() -> None:
    serde = agent_checkpoint_serde()
    plan = ToolCallPlan.create(
        "search_knowledge",
        {"query": "What is the capital of France?"},
    )
    config = AgentRunConfig(
        turn_id="test-serde-live",
        llm_budget_total=100,
    )
    result = ToolResult(
        tool_call_id=plan.tool_call_id,
        tool_name="search_knowledge",
        structured_content={
            "answer_text": "Paris is the capital of France.",
            "results": [
                {
                    "evidence_id": "ev_1",
                    "doc_id": 1,
                    "citation_anchor": "paris-capital",
                    "text": "Paris is the capital of France.",
                    "score": 0.95,
                    "source_type": "section",
                    "file_name": "france.pdf",
                }
            ],
            "citations": ["paris-capital"],
            "groundedness_flag": True,
            "insufficient_evidence": False,
            "total_found": 1,
        },
    )
    state = create_loop_state(
        current_message="What is the capital of France?",
        run_config=config,
    )
    state["pending_tool_calls"] = [
        PendingToolCall(
            plan=plan,
            status="completed",
            summary="Paris is the capital of France.",
        )
    ]
    state["tool_call_ledger"].append_plans([plan], turn=1)
    state["tool_results"] = [result]

    restored = serde.loads_typed(serde.dumps_typed(state))

    assert "evidence" not in restored
    assert "citations" not in restored
    restored_result = restored["tool_results"][0]
    assert isinstance(restored_result, ToolResult)
    assert restored_result.structured_content["results"][0]["evidence_id"] == ("ev_1")
    assert len(restored["tool_call_ledger"].entries) == 1
    assert restored["pending_tool_calls"][0].status == "completed"

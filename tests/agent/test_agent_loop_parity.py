from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.agent.parity.loop_scenarios import run_loop_scenarios

BASELINE_PATH = Path(__file__).parent / "parity" / "baselines" / "legacy_graph_v1.json"
PARITY_FILES = (
    Path(__file__).parent / "parity" / "fixtures.py",
    Path(__file__).parent / "parity" / "loop_scenarios.py",
    Path(__file__).parent / "parity" / "normalize.py",
)


def _load_baseline() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(BASELINE_PATH.read_text(encoding="utf-8")),
    )


def _tool_outcomes(state: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (result["tool_name"], result.get("outcome", result.get("status")))
        for result in state["tool_results"]
    ]


@pytest.mark.anyio
async def test_final_loop_preserves_frozen_user_visible_capabilities() -> None:
    baseline = _load_baseline()
    assert baseline["source_commit"] == "dbd746b9"
    assert baseline["schema_version"] == 1

    legacy = baseline["scenarios"]
    current = cast(dict[str, dict[str, Any]], await run_loop_scenarios())
    assert set(current) == set(legacy) - {"child_agent"}

    for name in (
        "single_tool",
        "multiple_tools",
        "rag_grounding",
        "structured_output",
    ):
        assert current[name]["status"] == legacy[name]["status"] == "done"
        assert current[name]["final_answer"] == legacy[name]["final_answer"]
        assert _tool_outcomes(current[name]) == _tool_outcomes(legacy[name])

    retry = current["tool_retry"]
    assert retry["status"] == legacy["tool_retry"]["status"] == "done"
    assert retry["final_answer"] == legacy["tool_retry"]["final_answer"]
    assert _tool_outcomes(retry)[-1] == _tool_outcomes(
        legacy["tool_retry"]
    )[-1]

    paused = current["approval_resume"]["paused"]
    resumed = current["approval_resume"]["resumed"]
    legacy_resumed = legacy["approval_resume"]["resumed"]
    assert paused["status"] == "paused"
    assert paused["observed"] == {"runner_calls": []}
    assert paused["pending_tool_names"] == ["write_tool"]
    assert resumed["status"] == legacy_resumed["status"] == "done"
    assert resumed["final_answer"] == legacy_resumed["final_answer"]
    assert _tool_outcomes(resumed) == _tool_outcomes(legacy_resumed)
    assert resumed["observed"] == {"runner_calls": ["approved"]}


@pytest.mark.anyio
async def test_final_loop_preserves_control_and_canonical_invariants() -> None:
    current = cast(dict[str, dict[str, Any]], await run_loop_scenarios())

    plain = current["plain_without_tools"]
    assert plain["status"] == "done"
    assert plain["final_answer"] == "Direct answer."
    assert plain["tool_results"] == []

    goal = current["explicit_goal_spec"]
    assert goal["status"] == "paused"
    assert goal["feedback_codes"] == ["goal_contract_unsatisfied"]
    assert goal["pause_reason"] == (
        "Explicit goal contract still requires traceable evidence."
    )

    compaction = current["message_compaction"]
    assert compaction["status"] == "paused"
    assert len(compaction["messages"]) == 6
    assert compaction["working_summary"] is None
    assert compaction["transcript_roles"][0:2] == ["user", "context"]

    fallback = current["model_fallback"]
    assert fallback["status"] == "paused"
    assert "fallback" in fallback["observed"]["transition_reasons"]

    structured = current["structured_output"]
    assert structured["final_output"]["data"] == {
        "answer": "structured:policy",
        "confidence": 0.91,
    }

    retry = current["tool_retry"]
    assert retry["observed"] == {"attempts": ["retry", "retry"]}
    assert [result["outcome"] for result in retry["tool_results"]] == [
        "error",
        "ok",
    ]

    rag = current["rag_grounding"]
    assert rag["tool_results"][-1]["structured_content"][
        "groundedness_flag"
    ] is True

    for scenario in current.values():
        states = (
            (scenario["paused"], scenario["resumed"])
            if "paused" in scenario
            else (scenario,)
        )
        for state in states:
            for origin in state["call_origins"]:
                assert origin["request_id"]
                assert origin["toolset_revision"]
                tool_name = next((
                    result["tool_name"]
                    for result in state["tool_results"]
                    if result["tool_call_id"] == origin["tool_call_id"]
                ), None)
                if tool_name is None:
                    assert state["status"] == "paused"
                    assert state["pending_tool_names"]
                    continue
                assert tool_name in origin["exposed_tool_names"]


def test_parity_fixtures_do_not_import_deleted_tool_contracts() -> None:
    forbidden = (
        "ToolSpec",
        "ToolExecutionService",
        "rag.agent.tools.spec",
        "rag.agent.core.tool_execution",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in PARITY_FILES)
    for token in forbidden:
        assert token not in source

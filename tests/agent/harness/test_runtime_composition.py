from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    HarnessAgent,
    HarnessModelRequest,
    HarnessModelResponse,
    PreparedModelCall,
    RolloutStore,
    RuntimeComposition,
)


class PlainModel:
    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {"model_alias": "test-model", "model_revision": "test-model-v1"}

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        digest = hashlib.sha256(request.messages[-1].content.encode()).hexdigest()
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash="no-tools",
            wire_hash=digest,
            request_ref={"message_count": len(request.messages)},
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        return HarnessModelResponse(
            text="composed answer",
            provider_response_id="response-composed",
            usage={"input_tokens": 1, "output_tokens": 1},
        )


class AcceptAnswer:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        return CompletionDecision(action="accept", reason="accepted by test verifier")


def test_harness_facade_uses_composed_thread_manager_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"

    with RuntimeComposition.open(
        database=database,
        workspace=workspace,
        model=PlainModel(),
    ) as runtime:
        agent = HarnessAgent(runtime.thread_manager)

        result = agent.run("answer through the public facade")

        assert result.answer == "composed answer"
        binding = runtime.store.read_turn(result.turn_id).binding_manifest
        assert binding["model_alias"] == "test-model"
        assert binding["model_revision"] == "test-model-v1"
        assert binding["toolset_revision"] == toolset_revision_for_tools(())
        assert binding["tool_execution_revisions"] == {}
        assert binding["completion_policy"] == {
            "require_workspace_change": False
        }
        assert binding["mcp_policy"] == {
            "workspace_discovery_enabled": True
        }
        assert binding["tool_execution_policy"] == {
            "active_skill_ids": [],
            "allow_execute_tools": False,
            "allow_write_tools": False,
            "auto_approve_sandboxed": False,
            "denied_tool_names": [],
            "deny_effects": [],
            "max_parallel_calls": 4,
            "require_confirmation_for": [],
        }
        assert runtime.store.read_turn(result.turn_id).status == "completed"
        assert runtime.store.verify().valid is True


def test_composition_refuses_to_run_with_projection_log_drift(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="canonical",
            binding_manifest={"model_alias": "test-model"},
        )
        item = store.list_items(turn.turn_id)[0]
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE items SET payload_json = ? WHERE item_id = ?",
            ('{"text":"projection drift"}', item.item_id),
        )

    with pytest.raises(RuntimeError, match="projection integrity"):
        RuntimeComposition.open(
            database=database,
            workspace=workspace,
            model=PlainModel(),
            completion_gate=AcceptAnswer(),
        )

    with RolloutStore(database) as store:
        assert store.verify().valid is False

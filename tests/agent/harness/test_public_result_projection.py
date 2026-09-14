from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    HarnessModelRequest,
    HarnessModelResponse,
    PreparedModelCall,
    RolloutStore,
    Session,
    TurnResult,
)
from agent_runtime.result import AgentResult, AgentToolCall


def test_public_result_dto_is_frozen_and_keeps_thread_turn_identity() -> None:
    assert tuple(field.name for field in fields(AgentResult)) == (
        "answer",
        "status",
        "files",
        "tool_calls",
        "evidence",
        "citations",
        "usage",
        "diagnostics",
        "turn_id",
        "stop_reason",
        "pause",
        "workspace_path",
        "groundedness",
        "insufficient_evidence",
        "plan",
        "plan_events",
        "needs_user_input",
        "thread_id",
    )
    call = AgentToolCall(
        tool_call_id="call-1",
        tool_name="read_file",
        arguments={"path": "README.md", "range": [1, 10]},
    )
    with pytest.raises(FrozenInstanceError):
        call.tool_name = "changed"  # type: ignore[misc]
    assert call.arguments == {"path": "README.md", "range": (1, 10)}
    with pytest.raises(TypeError):
        call.arguments["path"] = "changed"  # type: ignore[index]


class PublicAnswerModel:
    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {"model_id": "public-model", "model_revision": "public-v1"}

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
        del prepared
        return HarnessModelResponse(
            text="public harness answer",
            provider_response_id="provider-response-1",
            usage={
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "usage_source": "provider",
            },
        )


class AcceptPublicAnswer:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        return CompletionDecision(action="accept", reason="public answer accepted")


@pytest.mark.anyio
async def test_interrupted_model_exposes_public_retry_prompt(tmp_path):
    class InterruptedModel(PublicAnswerModel):
        attempts = 0

        async def dispatch(self, prepared):
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("OpenAI-compatible stream ended without a finish reason")
            return await super().dispatch(prepared)

    model = InterruptedModel()
    async with await Session.open(database=tmp_path / "retry.sqlite", workspace=tmp_path,
                                  model=model, completion_gate=AcceptPublicAnswer()) as session:
        paused = await session.submit("answer")
        assert paused.status == "paused"
        assert paused.pause is not None
        assert paused.pause.kind == "model_retry"
        assert paused.pause.options == ("retry", "abort")
        resumed = await session.resume(paused.turn_id, "retry")
        assert resumed.status == "done"
        assert model.attempts == 2
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_harness_turn_projects_to_the_stable_public_agent_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with await Session.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=PublicAnswerModel(),
        completion_gate=AcceptPublicAnswer(),
    ) as runtime:
        internal = await runtime.submit("answer publicly")

        result = internal

        assert result.answer == "public harness answer"
        assert result.status == "done"
        assert result.thread_id == internal.thread_id
        assert result.turn_id == internal.turn_id
        assert result.workspace_path == str(workspace.resolve())
        assert result.usage.input_tokens == 7
        assert result.usage.output_tokens == 3
        assert result.usage.total_tokens == 10
        assert result.usage.model_calls == 1
        assert result.usage.usage_source == "provider"
        assert result.tool_calls == ()
        assert result.pause is None
        assert result.stop_reason == "completed"


@pytest.mark.anyio
async def test_failed_result_projects_canonical_terminal_reason(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with await Session.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=PublicAnswerModel(),
        completion_gate=AcceptPublicAnswer(),
    ) as runtime:
        thread = runtime.store.create_thread(workspace=workspace)
        turn = runtime.store.start_turn(
            thread_id=thread.thread_id,
            user_message="exhaust the step budget",
            binding_manifest={"model_id": "public-model"},
        )
        failed = runtime.store.fail_turn(
            turn_id=turn.turn_id,
            reason_code="model_step_budget_exhausted",
            message="Turn exhausted its frozen model step budget.",
        )

        result = AgentResult._from_harness(
            TurnResult(
                thread_id=thread.thread_id,
                turn_id=failed.turn_id,
                answer=None,
                status="failed",
            ),
            store=runtime.store,
        )

        assert result.status == "failed"
        assert result.stop_reason == "model_step_budget_exhausted"
        assert [(diagnostic.code, diagnostic.message) for diagnostic in result.diagnostics] == [
            (
                "model_step_budget_exhausted",
                "Turn exhausted its frozen model step budget.",
            )
        ]


@pytest.mark.anyio
async def test_public_pause_projects_the_frozen_choice_question_and_options(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async with await Session.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=PublicAnswerModel(),
        completion_gate=AcceptPublicAnswer(),
    ) as runtime:
        thread = runtime.store.create_thread(workspace=workspace)
        turn = runtime.store.start_turn(
            thread_id=thread.thread_id,
            user_message="pick a target",
            binding_manifest={"model_id": "public-model"},
        )
        interaction = runtime.store.request_choice(
            turn_id=turn.turn_id,
            question="Which target?",
            options=("staging", "production"),
        )

        result = AgentResult._from_harness(
            TurnResult(
                thread_id=thread.thread_id,
                turn_id=turn.turn_id,
                answer=None,
                status="paused",
                interaction_id=interaction.request_id,
            ),
            store=runtime.store,
        )

        assert result.pause is not None
        assert result.pause.kind == "choice"
        assert result.pause.question == "Which target?"
        assert result.pause.options == ("staging", "production")
        assert result.needs_user_input == "Which target?"


@pytest.mark.anyio
async def test_completed_result_is_read_after_disconnect_without_rerunning_model_or_gate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    async with await Session.open(
        database=database,
        workspace=workspace,
        model=PublicAnswerModel(),
        completion_gate=AcceptPublicAnswer(),
    ) as initial:
        committed = await initial.submit("commit before disconnect")

    from unittest.mock import patch

    from agent_runtime import Agent

    reader = Agent(checkpoint_db=database, workspace_path=workspace)
    with patch.object(reader, "_harness_model", side_effect=AssertionError("read must not open a model")):
        replayed = await reader.read_result(committed.turn_id)
    assert replayed == committed
    with RolloutStore(database) as store:
        assert store.verify().valid is True

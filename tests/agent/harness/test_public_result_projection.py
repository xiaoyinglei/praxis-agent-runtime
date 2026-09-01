from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    HarnessAgent,
    HarnessModelRequest,
    HarnessModelResponse,
    PreparedModelCall,
    RuntimeComposition,
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
        return {"model_alias": "public-model", "model_revision": "public-v1"}

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


def test_harness_turn_projects_to_the_stable_public_agent_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=PublicAnswerModel(),
        completion_gate=AcceptPublicAnswer(),
    ) as runtime:
        internal = HarnessAgent(runtime.thread_manager).run("answer publicly")

        result = AgentResult._from_harness(internal, store=runtime.store)

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


def test_public_pause_projects_the_frozen_choice_question_and_options(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=PublicAnswerModel(),
        completion_gate=AcceptPublicAnswer(),
    ) as runtime:
        thread = runtime.store.create_thread(workspace=workspace)
        turn = runtime.store.start_turn(
            thread_id=thread.thread_id,
            user_message="pick a target",
            binding_manifest={"model_alias": "public-model"},
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


def test_completed_result_is_read_after_disconnect_without_rerunning_model_or_gate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RuntimeComposition.open(
        database=database,
        workspace=workspace,
        model=PublicAnswerModel(),
        completion_gate=AcceptPublicAnswer(),
    ) as initial:
        committed = HarnessAgent(initial.thread_manager).run("commit before disconnect")

    class NeverCalledModel(PublicAnswerModel):
        def __init__(self) -> None:
            self.prepare_calls = 0

        def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
            del request
            self.prepare_calls += 1
            raise AssertionError("read_result must not prepare a model call")

    class NeverCalledGate:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
            del proposal
            self.calls += 1
            raise AssertionError("read_result must not rerun CompletionGate")

    model = NeverCalledModel()
    gate = NeverCalledGate()
    with RuntimeComposition.open(
        database=database,
        workspace=workspace,
        model=model,
        completion_gate=gate,
    ) as reconnected:
        replayed = HarnessAgent(reconnected.thread_manager).read_result(committed.turn_id)

        assert replayed == committed
        assert model.prepare_calls == 0
        assert gate.calls == 0
        assert reconnected.store.verify().valid is True

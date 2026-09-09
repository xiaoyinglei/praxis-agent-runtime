from __future__ import annotations

import hashlib
import json
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
)


class EchoHistoryModel:
    def __init__(self) -> None:
        self.requests: list[HarnessModelRequest] = []

    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {"model_id": f"model-v{len(self.requests) + 1}", "thread_id": thread_id, "turn_id": turn_id}

    def ensure_available(self, binding, *, thread_id, turn_id):
        if binding.get("model_id") == "unavailable":
            raise RuntimeError("frozen binding unavailable")

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        self.requests.append(request)
        encoded = json.dumps(
            [(message.role, message.content) for message in request.messages],
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash="no-tools",
            wire_hash=digest,
            request_ref={"message_count": len(request.messages)},
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        return HarnessModelResponse(
            text=f"answer-{len(self.requests)}",
            provider_response_id=f"response-{len(self.requests)}",
            usage={"input_tokens": len(self.requests[-1].messages), "output_tokens": 1},
        )


class AcceptPlainAnswer:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        return CompletionDecision(action="accept", reason="plain answer accepted")


class RotatingBindingProvider:
    def __init__(self) -> None:
        self.revision = 0
        self.identities: list[tuple[str, str]] = []

    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        self.revision += 1
        self.identities.append((thread_id, turn_id))
        return {
            "model_id": f"model-v{self.revision}",
            "thread_id": thread_id,
            "turn_id": turn_id,
        }


@pytest.mark.anyio
async def test_session_creates_distinct_turns_with_fresh_bindings_and_shared_history(tmp_path: Path) -> None:
    model = EchoHistoryModel()
    async with await Session.open(database=tmp_path / "rollout.sqlite3", workspace=tmp_path, model=model) as session:
        first = await session.submit("first")
        second = await session.submit("second")
        assert first.thread_id == second.thread_id == session.thread_id
        assert first.turn_id != second.turn_id
        assert [(m.role, m.content) for m in model.requests[1].messages] == [
            ("user", "first"),
            ("assistant", "answer-1"),
            ("user", "second"),
        ]
        assert session.store.read_turn(second.turn_id).predecessor_turn_id == first.turn_id
        assert session.store.read_turn(first.turn_id).binding_manifest["model_id"] == "model-v1"
        assert session.store.read_turn(second.turn_id).binding_manifest["model_id"] == "model-v2"
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_clarification_continues_with_current_model_when_initial_model_is_unavailable(tmp_path: Path) -> None:
    model = EchoHistoryModel()
    async with await Session.open(database=tmp_path / "rollout.sqlite3", workspace=tmp_path, model=model) as session:
        turn = session.store.start_turn(
            thread_id=session.thread_id,
            user_message="ambiguous",
            binding_manifest={"model_id": "unavailable"},
        )
        session.store.request_clarification(turn_id=turn.turn_id, question="which target?")
        resumed = await session.resume(turn.turn_id, "continue", user_input="A")
        assert resumed.status == "done"
        assert len(model.requests) == 1
        assert model.requests[0].binding_manifest["model_id"] != "unavailable"
        assert session.store.read_thread(session.thread_id).active_turn_id is None



@pytest.mark.anyio
async def test_opening_from_non_head_turn_forks_with_exact_history(tmp_path: Path) -> None:
    model = EchoHistoryModel()
    options = dict(database=tmp_path / "rollout.sqlite3", workspace=tmp_path, model=model)
    async with await Session.open(**options) as source:
        first = await source.submit("first")
        second = await source.submit("second")
        async with await Session.open(**options, previous_turn_id=first.turn_id) as branch:
            result = await branch.submit("branch")
            assert result.thread_id != first.thread_id
            assert branch.store.read_thread(branch.thread_id).fork_turn_id == first.turn_id
            assert [(m.role, m.content) for m in model.requests[2].messages] == [
                ("user", "first"),
                ("assistant", "answer-1"),
                ("user", "branch"),
            ]
            assert source.store.read_thread(source.thread_id).head_turn_id == second.turn_id


@pytest.mark.anyio
async def test_child_binding_and_durable_turn_share_identity(tmp_path: Path) -> None:
    model = EchoHistoryModel()
    async with await Session.open(database=tmp_path / "rollout.sqlite3", workspace=tmp_path, model=model) as session:
        child = await session.run_child(user_message="isolated child", max_steps=2, max_tokens_total=100)
        binding = session.store.read_turn(child.turn_id).binding_manifest
        assert binding["thread_id"] == child.thread_id
        assert binding["turn_id"] == child.turn_id
        assert child.thread_id != session.thread_id


def test_fork_from_non_head_turn_has_an_exact_history_cutoff(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        source = store.create_thread(workspace=workspace)
        first = store.start_turn(
            thread_id=source.thread_id,
            user_message="source-one",
            binding_manifest={"model_id": "model-v1"},
        )
        store.complete_turn(turn_id=first.turn_id, answer="answer-one")
        second = store.start_turn(
            thread_id=source.thread_id,
            user_message="source-two",
            binding_manifest={"model_id": "model-v1"},
        )
        store.complete_turn(turn_id=second.turn_id, answer="answer-two")
        third = store.start_turn(
            thread_id=source.thread_id,
            user_message="source-three",
            binding_manifest={"model_id": "model-v1"},
        )
        store.complete_turn(turn_id=third.turn_id, answer="answer-three")

        fork = store.fork_thread(from_turn_id=first.turn_id)
        branch = store.start_turn(
            thread_id=fork.thread_id,
            user_message="branch-only",
            binding_manifest={"model_id": "model-v2"},
        )

        context = store.list_context_items(branch.turn_id)
        assert [item.payload.get("text") for item in context] == [
            "source-one",
            "answer-one",
            "branch-only",
        ]
        assert fork.parent_thread_id == source.thread_id
        assert fork.fork_turn_id == first.turn_id
        assert branch.predecessor_turn_id == first.turn_id
        assert store.read_thread(source.thread_id).head_turn_id == third.turn_id
        assert len({item.item_id for item in context}) == len(context)
        assert store.verify().valid is True

        store.rebuild_projections()
        rebuilt = store.read_thread(fork.thread_id)
        assert rebuilt.parent_thread_id == source.thread_id
        assert rebuilt.fork_turn_id == first.turn_id
        assert store.verify().valid is True

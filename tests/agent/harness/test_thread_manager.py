from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    HarnessAgent,
    HarnessModelRequest,
    HarnessModelResponse,
    PreparedModelCall,
    RolloutContextManager,
    RolloutStore,
    Session,
    ThreadManager,
    TurnResult,
)


class EchoHistoryModel:
    def __init__(self) -> None:
        self.requests: list[HarnessModelRequest] = []

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
            "model_alias": f"model-v{self.revision}",
            "thread_id": thread_id,
            "turn_id": turn_id,
        }


class RecoveryEntryRunner:
    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.calls: list[tuple[object, ...]] = []

    async def respond_interaction(
        self,
        *,
        turn_id: str,
        request_id: str,
        response: str,
    ) -> TurnResult:
        self.calls.append(("respond", turn_id, request_id, response))
        return TurnResult(thread_id="thread", turn_id=turn_id, answer="resumed")

    async def retry_unknown_model(self, *, turn_id: str) -> TurnResult:
        self.calls.append(("retry", turn_id))
        return TurnResult(thread_id="thread", turn_id=turn_id, answer="retried")


def test_thread_manager_creates_thread_and_reuses_it_for_followup(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        model = EchoHistoryModel()
        bindings = RotatingBindingProvider()
        opened_sessions: list[str] = []

        def open_session(thread_id: str) -> Session:
            opened_sessions.append(thread_id)
            return Session(
                thread_id=thread_id,
                store=store,
                model=model,
                context_manager=RolloutContextManager(store),
                completion_gate=AcceptPlainAnswer(),
            )

        manager = ThreadManager(
            store=store,
            session_factory=open_session,
            workspace=workspace,
            binding_provider=bindings,
        )

        first = asyncio.run(manager.run(user_message="first"))
        second = asyncio.run(
            manager.run(
                thread_id=first.thread_id,
                user_message="second",
            )
        )

        assert second.thread_id == first.thread_id
        assert opened_sessions == [first.thread_id]
        assert second.turn_id != first.turn_id
        assert bindings.identities == [
            (first.thread_id, first.turn_id),
            (second.thread_id, second.turn_id),
        ]
        assert [(message.role, message.content) for message in model.requests[1].messages] == [
            ("user", "first"),
            ("assistant", "answer-1"),
            ("user", "second"),
        ]
        assert store.read_turn(second.turn_id).predecessor_turn_id == first.turn_id
        assert store.read_turn(first.turn_id).binding_manifest["model_alias"] == "model-v1"
        assert store.read_turn(second.turn_id).binding_manifest["model_alias"] == "model-v2"
        assert store.verify().valid is True


def test_public_recovery_entries_validate_frozen_binding_before_runner_io(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="ambiguous",
            binding_manifest={"model_alias": "model-v1"},
        )
        clarification = store.request_clarification(
            turn_id=turn.turn_id,
            question="which target?",
        )
        runner = RecoveryEntryRunner(thread.thread_id)
        validated: list[tuple[dict[str, object], str, str]] = []
        manager = ThreadManager(
            store=store,
            session_factory=lambda _thread_id: runner,  # type: ignore[return-value]
            workspace=workspace,
            binding_provider=RotatingBindingProvider(),
            binding_validator=lambda binding, *, thread_id, turn_id: validated.append(
                (dict(binding), thread_id, turn_id)
            ),
        )
        agent = HarnessAgent(manager)

        response = agent.respond_interaction(
            turn.turn_id,
            clarification.request_id,
            "target A",
        )
        retry = agent.retry_unknown_model(turn.turn_id)

        assert response.answer == "resumed"
        assert retry.answer == "retried"
        assert runner.calls == [
            ("respond", turn.turn_id, clarification.request_id, "target A"),
            ("retry", turn.turn_id),
        ]
        assert validated == [
            ({"model_alias": "model-v1"}, thread.thread_id, turn.turn_id),
            ({"model_alias": "model-v1"}, thread.thread_id, turn.turn_id),
        ]


def test_interrupted_turn_can_be_cancelled_without_restoring_legacy_binding(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="legacy interrupted work",
            binding_manifest={"legacy_resume_compatible": False},
        )
        store.interrupt_orphaned_turn(
            turn_id=turn.turn_id,
            reason="legacy worker is gone",
            maintenance_confirmed=True,
        )
        manager = ThreadManager(
            store=store,
            session_factory=lambda thread_id: RecoveryEntryRunner(thread_id),  # type: ignore[return-value]
            workspace=workspace,
            binding_provider=RotatingBindingProvider(),
            binding_validator=lambda _binding, **_identity: (_ for _ in ()).throw(
                RuntimeError("legacy binding unavailable")
            ),
        )

        result = manager.cancel(turn_id=turn.turn_id)

        assert result.status == "cancelled"
        assert store.read_turn(turn.turn_id).status == "cancelled"
        assert store.read_thread(thread.thread_id).active_turn_id is None


def test_fork_from_non_head_turn_has_an_exact_history_cutoff(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        source = store.create_thread(workspace=workspace)
        first = store.start_turn(
            thread_id=source.thread_id,
            user_message="source-one",
            binding_manifest={"model_alias": "model-v1"},
        )
        store.complete_turn(turn_id=first.turn_id, answer="answer-one")
        second = store.start_turn(
            thread_id=source.thread_id,
            user_message="source-two",
            binding_manifest={"model_alias": "model-v1"},
        )
        store.complete_turn(turn_id=second.turn_id, answer="answer-two")
        third = store.start_turn(
            thread_id=source.thread_id,
            user_message="source-three",
            binding_manifest={"model_alias": "model-v1"},
        )
        store.complete_turn(turn_id=third.turn_id, answer="answer-three")

        fork = store.fork_thread(from_turn_id=first.turn_id)
        branch = store.start_turn(
            thread_id=fork.thread_id,
            user_message="branch-only",
            binding_manifest={"model_alias": "model-v2"},
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


def test_previous_turn_compatibility_forks_when_predecessor_is_not_head(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        model = EchoHistoryModel()
        bindings = RotatingBindingProvider()
        manager = ThreadManager(
            store=store,
            session_factory=lambda thread_id: Session(
                thread_id=thread_id,
                store=store,
                model=model,
                context_manager=RolloutContextManager(store),
                completion_gate=AcceptPlainAnswer(),
            ),
            workspace=workspace,
            binding_provider=bindings,
        )
        first = asyncio.run(manager.run(user_message="first"))
        second = asyncio.run(manager.run(thread_id=first.thread_id, user_message="second"))

        branch = asyncio.run(
            manager.run(
                previous_turn_id=first.turn_id,
                user_message="branch",
            )
        )

        assert branch.thread_id != first.thread_id
        assert bindings.identities[-1] == (branch.thread_id, branch.turn_id)
        assert store.read_thread(branch.thread_id).fork_turn_id == first.turn_id
        assert [(message.role, message.content) for message in model.requests[2].messages] == [
            ("user", "first"),
            ("assistant", "answer-1"),
            ("user", "branch"),
        ]
        assert store.read_thread(first.thread_id).head_turn_id == second.turn_id


def test_child_turn_uses_one_identity_for_binding_and_durable_start(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        model = EchoHistoryModel()
        bindings = RotatingBindingProvider()
        manager = ThreadManager(
            store=store,
            session_factory=lambda thread_id: Session(
                thread_id=thread_id,
                store=store,
                model=model,
                context_manager=RolloutContextManager(store),
                completion_gate=AcceptPlainAnswer(),
            ),
            workspace=workspace,
            binding_provider=bindings,
        )

        child = asyncio.run(
            manager.run_child(
                user_message="isolated child",
                max_steps=2,
                max_tokens_total=100,
            )
        )

        assert bindings.identities == [(child.thread_id, child.turn_id)]
        binding = store.read_turn(child.turn_id).binding_manifest
        assert binding["thread_id"] == child.thread_id
        assert binding["turn_id"] == child.turn_id

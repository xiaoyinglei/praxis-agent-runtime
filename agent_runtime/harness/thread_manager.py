"""Durable Thread lifecycle owner for the replacement Harness."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime.budget import ResourceUsage
from agent_runtime.harness.protocol import BindingProvider, BindingValidator, TurnResult
from agent_runtime.harness.rollout import RolloutStore, TurnSnapshot
from agent_runtime.harness.session import Session
from agent_runtime.streaming.sink import TurnEventDispatcher


class ThreadManager:
    def __init__(
        self,
        *,
        store: RolloutStore,
        session_factory: Callable[[str], Session],
        workspace: Path,
        binding_provider: BindingProvider,
        binding_validator: BindingValidator | None = None,
        event_dispatcher: TurnEventDispatcher | None = None,
    ) -> None:
        resolved_workspace = Path(workspace).resolve()
        if not resolved_workspace.is_dir():
            raise ValueError("workspace must be an existing directory")
        self._store = store
        self._session_factory = session_factory
        self._sessions: dict[str, Session] = {}
        self._workspace = resolved_workspace
        self._binding_provider = binding_provider
        self._binding_validator = binding_validator
        self._event_dispatcher = event_dispatcher

    async def run(
        self,
        *,
        user_message: str,
        thread_id: str | None = None,
        previous_turn_id: str | None = None,
        input_files: tuple[Mapping[str, Any], ...] = (),
        event_dispatcher: TurnEventDispatcher | None = None,
    ) -> TurnResult:
        if thread_id is not None and previous_turn_id is not None:
            raise ValueError("thread_id and previous_turn_id are mutually exclusive")
        if previous_turn_id is not None:
            predecessor = self._store.read_turn(previous_turn_id)
            if predecessor.status not in {"completed", "failed", "cancelled"}:
                raise RuntimeError("non-terminal predecessor must be resumed, cancelled, or abandoned")
            source_thread = self._store.read_thread(predecessor.thread_id)
            if Path(source_thread.workspace) != self._workspace:
                raise RuntimeError("predecessor belongs to a different workspace security domain")
            thread_id = (
                source_thread.thread_id
                if source_thread.head_turn_id == previous_turn_id
                else self._store.fork_thread(from_turn_id=previous_turn_id).thread_id
            )
        if thread_id is None:
            thread_id = self._store.create_thread(workspace=self._workspace).thread_id
        else:
            thread = self._store.read_thread(thread_id)
            if Path(thread.workspace) != self._workspace:
                raise RuntimeError("thread belongs to a different workspace security domain")
        turn_id = f"turn_{uuid4().hex}"
        binding = dict(
            self._binding_provider.snapshot(
                thread_id=thread_id,
                turn_id=turn_id,
            )
        )
        binding["budget_root_turn_id"] = turn_id
        return await self._session_for_thread(
            thread_id,
            event_dispatcher=event_dispatcher or self._event_dispatcher,
        ).run(
            turn_id=turn_id,
            user_message=user_message,
            binding_manifest=binding,
            input_files=input_files,
        )

    async def resume(self, *, turn_id: str, decision: str) -> TurnResult:
        self._validate_recovery_turn(turn_id)
        return await self._session_for_turn(turn_id).resume(
            turn_id=turn_id,
            decision=decision,
        )

    async def run_child(
        self,
        *,
        user_message: str,
        max_steps: int | None,
        max_tokens_total: int | None,
        max_cost_micros: int | None = None,
        parent_turn_id: str | None = None,
    ) -> TurnResult:
        """Start a child Thread; budgeted children inherit capacity from a parent.

        parent_turn_id=None is retained only for explicit standalone child runs.
        The model-visible subagent tool always supplies the trusted parent Turn.
        """

        thread_id = f"thread_{uuid4().hex}"
        turn_id = f"turn_{uuid4().hex}"
        binding = dict(
            self._binding_provider.snapshot(
                thread_id=thread_id,
                turn_id=turn_id,
            )
        )
        if max_steps is not None:
            binding["model_step_budget"] = max_steps
        binding["completion_policy"] = {"require_workspace_change": False}

        if parent_turn_id is None:
            # Compatibility for non-delegated callers. This path is not used by
            # the `task` tool and therefore does not claim parent inheritance.
            self._store.create_thread(
                workspace=self._workspace,
                thread_id=thread_id,
            )
            if max_tokens_total is not None:
                binding["model_token_budget_total"] = max_tokens_total
            if max_cost_micros is not None:
                binding["model_cost_budget_total_micros"] = max_cost_micros
            binding["budget_root_turn_id"] = turn_id
            return await self._session_for_thread(thread_id).run(
                turn_id=turn_id,
                user_message=user_message,
                binding_manifest=binding,
            )

        parent = self._validate_turn_workspace(parent_turn_id)
        if parent.status != "running":
            raise RuntimeError("subagent parent Turn must still be running")
        child_turn = self._store.start_budgeted_child_turn(
            parent_turn_id=parent_turn_id,
            child_thread_id=thread_id,
            child_turn_id=turn_id,
            user_message=user_message,
            binding_manifest=binding,
            requested_tokens=max_tokens_total,
            requested_cost_micros=max_cost_micros,
        )
        session = self._session_for_thread(thread_id)
        result = await session.run_turn(
            session.restore_turn_context(child_turn.turn_id),
            start_step=1,
        )

        if result.status in {"completed", "failed", "cancelled"}:
            state = self._store.read_budget_state(child_turn.turn_id)
            unresolved = state.reserved + state.uncertain + state.child_reserved
            if unresolved == ResourceUsage():
                self._store.settle_child_budget(child_turn_id=child_turn.turn_id)
        return result

    async def respond_interaction(
        self,
        *,
        turn_id: str,
        request_id: str,
        response: str,
    ) -> TurnResult:
        self._validate_recovery_turn(turn_id)
        return await self._session_for_turn(turn_id).respond_interaction(
            turn_id=turn_id,
            request_id=request_id,
            response=response,
        )

    async def retry_unknown_model(self, *, turn_id: str) -> TurnResult:
        self._validate_recovery_turn(turn_id)
        return await self._session_for_turn(turn_id).retry_unknown_model(turn_id=turn_id)

    async def recover_committed_model_response(self, *, turn_id: str) -> TurnResult:
        self._validate_recovery_turn(turn_id)
        return await self._session_for_turn(turn_id).recover_committed_model_response(turn_id=turn_id)

    def cancel(self, *, turn_id: str) -> TurnResult:
        self._validate_turn_workspace(turn_id)
        turn = self._store.cancel_turn(turn_id=turn_id)
        return TurnResult(
            thread_id=turn.thread_id,
            turn_id=turn.turn_id,
            answer=None,
            status="cancelled",
        )

    def read_result(self, *, turn_id: str) -> TurnResult:
        turn = self._validate_turn_workspace(turn_id)
        answer: str | None = None
        if turn.status == "completed":
            answers = [
                item.payload.get("text")
                for item in self._store.list_items(turn_id)
                if item.kind == "agent_message"
                and item.status == "completed"
                and isinstance(item.payload.get("text"), str)
            ]
            if len(answers) != 1:
                raise RuntimeError("completed Turn has no unique canonical answer")
            answer = answers[0]
        interaction_id = next(
            (
                interaction.request_id
                for interaction in reversed(self._store.list_interactions(turn_id))
                if interaction.status == "pending"
            ),
            None,
        )
        return TurnResult(
            thread_id=turn.thread_id,
            turn_id=turn.turn_id,
            answer=answer,
            status=turn.status,
            interaction_id=interaction_id,
        )

    def _validate_recovery_turn(self, turn_id: str) -> None:
        turn = self._validate_turn_workspace(turn_id)
        if self._binding_validator is not None:
            self._binding_validator(
                turn.binding_manifest,
                thread_id=turn.thread_id,
                turn_id=turn.turn_id,
            )

    def _session_for_turn(self, turn_id: str) -> Session:
        turn = self._store.read_turn(turn_id)
        return self._session_for_thread(
            turn.thread_id,
            event_dispatcher=self._event_dispatcher,
        )

    def _session_for_thread(
        self,
        thread_id: str,
        *,
        event_dispatcher: TurnEventDispatcher | None = None,
    ) -> Session:
        session = self._sessions.get(thread_id)
        if session is None:
            session = self._session_factory(thread_id)
            if session.thread_id != thread_id:
                raise RuntimeError("Session factory returned the wrong thread")
            self._sessions[thread_id] = session
        if event_dispatcher is not None:
            session.attach_event_dispatcher(event_dispatcher)
        return session

    def _validate_turn_workspace(self, turn_id: str) -> TurnSnapshot:
        turn = self._store.read_turn(turn_id)
        thread = self._store.read_thread(turn.thread_id)
        if Path(thread.workspace) != self._workspace:
            raise RuntimeError("turn belongs to a different workspace security domain")
        return turn

"""Public Agent SDK backed exclusively by the Rollout Harness."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.models import (
    ModelControlPlane,
    ModelSpec,
    ModelSwitchRequester,
    validate_model_switch_requester,
)
from agent_runtime.result import AgentPause, AgentResult
from agent_runtime.streaming.events import StreamEvent
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.workspace import DEFAULT_CHECKPOINT_PATH, DEFAULT_MODEL_SESSION_PATH

if TYPE_CHECKING:
    from agent_runtime.harness import BoundHarnessModel, Session
    from agent_runtime.runtime.mcp import MCPConfigTrustDecision

logger = logging.getLogger(__name__)
_RUNTIME_CLOSE_GRACE_SECONDS = 5.0


class AgentEventSink(Protocol):
    """Receive durable lifecycle events derived from the Rollout log."""

    async def emit(self, event: StreamEvent) -> None: ...


class Agent:
    def __init__(
        self,
        *,
        model: str | None = None,
        checkpoint_db: Path | None = DEFAULT_CHECKPOINT_PATH,
        workspace_path: Path | str | None = None,
        model_session_path: Path | None = DEFAULT_MODEL_SESSION_PATH,
        knowledge: RAGKnowledgeConfig | None = None,
        enable_workspace_mcp: bool = True,
        mcp_config_trust: MCPConfigTrustDecision | None = None,
        _selection_requester: ModelSwitchRequester = "system",
    ) -> None:
        if knowledge is not None and not isinstance(knowledge, RAGKnowledgeConfig):
            raise TypeError("knowledge must be RAGKnowledgeConfig or None")
        if not isinstance(enable_workspace_mcp, bool):
            raise TypeError("enable_workspace_mcp must be bool")
        if mcp_config_trust is not None:
            from agent_runtime.runtime.mcp import MCPConfigTrustDecision

            if not isinstance(mcp_config_trust, MCPConfigTrustDecision):
                raise TypeError("mcp_config_trust must be MCPConfigTrustDecision or None")
        selection_requester = validate_model_switch_requester(_selection_requester)
        self.model = model
        self.checkpoint_db = checkpoint_db
    
        workspace = (Path.cwd() if workspace_path is None else Path(workspace_path))
        self.workspace_path = (workspace.expanduser().resolve())
        self.model_session_path = model_session_path
        self.knowledge = knowledge
        self.enable_workspace_mcp = enable_workspace_mcp
        self.mcp_config_trust = mcp_config_trust
        self._selection_requester = selection_requester
        self._followup_model_id: str | None = None

    def models(self) -> list[ModelSpec]:
        return self._get_model_control_plane().list_models()

    def current_model(self) -> ModelSpec:
        return self._get_model_control_plane().current_model()

    def switch_model(self, model_id: str) -> ModelSpec:
        spec = self._get_model_control_plane().switch_model(
            model_id,
            requested_by="user",
            persist=self.model_session_path is not None,
        )
        self.model = spec.id
        self._followup_model_id = spec.id
        self._selection_requester = "user"
        return spec

    def _request_model_switch(self, model_id: str) -> ModelSpec:
        spec = self._get_model_control_plane().request_model_switch(model_id)
        self.model = spec.id
        self._followup_model_id = spec.id
        self._selection_requester = "agent"
        return spec

    async def run(
        self,
        task: str,
        *,
        previous_turn_id: str | None = None,
        files: Sequence[str] | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        max_cost_micros: int | None = None,
        require_workspace_change: bool = True,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        event_sink: AgentEventSink | None = None,
        _event_dispatcher: TurnEventDispatcher | None = None,
    ) -> AgentResult:
        async with self.session(
            previous_turn_id=previous_turn_id,
            require_workspace_change=require_workspace_change,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            max_turns=max_turns, max_tokens_total=max_tokens_total,
            max_cost_micros=max_cost_micros,
            event_sink=event_sink, _event_dispatcher=_event_dispatcher,
        ) as session:
            return await session.submit(task, files=files)

    @asynccontextmanager
    async def session(
        self, *, previous_turn_id: str | None = None,
        require_workspace_change: bool = True,
        allow_write_tools: bool = False, allow_execute_tools: bool = False,
        max_turns: int | None = None, max_tokens_total: int | None = None,
        max_cost_micros: int | None = None,
        event_sink: AgentEventSink | None = None,
        _event_dispatcher: TurnEventDispatcher | None = None,
        _frozen_turn_id: str | None = None,
    ) -> AsyncIterator[Session]:
        from agent_runtime.harness.session import Session

        session = await Session.open(
            agent=self, previous_turn_id=previous_turn_id, frozen_turn_id=_frozen_turn_id,
            require_workspace_change=require_workspace_change,
            allow_write_tools=allow_write_tools, allow_execute_tools=allow_execute_tools,
            max_steps=16 if max_turns is None else max_turns,
            max_tokens_total=max_tokens_total, max_cost_micros=max_cost_micros,
            event_sink=event_sink, event_dispatcher=_event_dispatcher,
        )
        async with session:
            yield session

    async def resume(
        self,
        turn_id: str,
        action: str,
        *,
        user_input: str | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        from agent_runtime.harness import RolloutStore, TurnResult

        if action == "abort":
            from agent_runtime.harness import RolloutEventReader

            with RolloutStore(self._harness_database()) as store:
                turn = store.read_turn(turn_id)
                thread = store.read_thread(turn.thread_id)
                if Path(thread.workspace).resolve() != self._workspace_path():
                    raise RuntimeError("turn belongs to a different workspace security domain")
                mutation = store.capture_mutation(lambda: store.cancel_turn(turn_id=turn_id))
                cancelled = mutation.value
                result = AgentResult._from_harness(
                    TurnResult(
                        thread_id=cancelled.thread_id,
                        turn_id=cancelled.turn_id,
                        answer=None,
                        status="cancelled",
                    ),
                    store=store,
                )
                if event_sink is not None:
                    dispatcher = TurnEventDispatcher()
                    dispatcher.subscribe_controlling_sink(event_sink)
                    for replayed in RolloutEventReader(store).project_committed_batch(mutation.records):
                        await dispatcher.emit(replayed.event, cursor=replayed.cursor)
            return result
        async with self.session(_frozen_turn_id=turn_id, event_sink=event_sink) as session:
            return await session.resume(turn_id, action, user_input=user_input)

    async def read_result(self, turn_id: str) -> AgentResult:
        from agent_runtime.harness import RolloutStore, TurnResult

        with RolloutStore(self._harness_database()) as store:
            turn = store.read_turn(turn_id)
            thread = store.read_thread(turn.thread_id)
            if Path(thread.workspace).resolve() != self._workspace_path():
                raise RuntimeError("turn belongs to a different workspace security domain")
            answer: str | None = None
            if turn.status == "completed":
                answers = [
                    item.payload.get("text")
                    for item in store.list_items(turn_id)
                    if item.kind == "agent_message"
                    and item.status == "completed"
                    and isinstance(item.payload.get("text"), str)
                ]
                if len(answers) != 1:
                    raise RuntimeError("completed Turn has no unique canonical answer")
                answer = answers[0]
            pending = next(
                (
                    interaction.request_id
                    for interaction in reversed(store.list_interactions(turn_id))
                    if interaction.status == "pending"
                ),
                None,
            )
            return AgentResult._from_harness(
                TurnResult(
                    thread_id=turn.thread_id,
                    turn_id=turn.turn_id,
                    answer=answer,
                    status=turn.status,
                    interaction_id=pending,
                ),
                store=store,
            )

    async def stream(
        self,
        task: str,
        *,
        previous_turn_id: str | None = None,
        files: Sequence[str] | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        max_cost_micros: int | None = None,
        require_workspace_change: bool = True,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Yield committed durable events while the Turn is still running."""

        dispatcher = TurnEventDispatcher()
        stream = dispatcher.subscribe_controlling()
        run_task = asyncio.create_task(
            self.run(
                task,
                previous_turn_id=previous_turn_id,
                files=files,
                max_turns=max_turns,
                max_tokens_total=max_tokens_total,
                max_cost_micros=max_cost_micros,
                require_workspace_change=require_workspace_change,
                allow_write_tools=allow_write_tools,
                allow_execute_tools=allow_execute_tools,
                _event_dispatcher=dispatcher,
            )
        )
        try:
            while True:
                if not stream.empty:
                    yield stream.receive_nowait()
                    continue
                if run_task.done():
                    break
                next_event = asyncio.create_task(stream.wait_available())
                done, _pending = await asyncio.wait(
                    {run_task, next_event},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if next_event in done:
                    continue
                else:
                    next_event.cancel()
                    await asyncio.gather(next_event, return_exceptions=True)
            await run_task
        finally:
            if not run_task.done():
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
            dispatcher.close()

    async def pending_input(self, turn_id: str) -> AgentPause | None:
        from agent_runtime.harness import RolloutStore, TurnResult

        with RolloutStore(self._harness_database()) as store:
            turn = store.read_turn(turn_id)
            thread = store.read_thread(turn.thread_id)
            if Path(thread.workspace).resolve() != self._workspace_path():
                raise RuntimeError("turn belongs to a different workspace security domain")
            if turn.status != "paused":
                return None
            projected = AgentResult._from_harness(
                TurnResult(
                    thread_id=thread.thread_id,
                    turn_id=turn_id,
                    answer=None,
                    status="paused",
                ),
                store=store,
            )
            return projected.pause

    def _harness_model(self) -> BoundHarnessModel:
        from agent_runtime.builtin.generic import GENERIC_SYSTEM_PROMPT
        from agent_runtime.harness import ControlPlaneHarnessModel

        return ControlPlaneHarnessModel(
            control_plane=self._get_model_control_plane(),
            instructions=(GENERIC_SYSTEM_PROMPT,),
        )

    def _harness_database(self) -> Path:
        if self.checkpoint_db is not None:
            return Path(self.checkpoint_db).expanduser().resolve()
        return self._workspace_path() / ".praxis" / "runtime" / "rollout.sqlite3"

    def _workspace_path(self) -> Path:
        return self.workspace_path

    def _stage_harness_files(
        self,
        files: Sequence[str],
    ) -> tuple[dict[str, object], ...]:
        from agent_runtime.workspace import import_files, open_workspace

        if not files:
            return ()
        workspace = open_workspace(self._workspace_path(), create=True)
        staged = import_files(
            workspace,
            list(files),
            namespace=f"turn_{uuid4().hex}",
        )
        values: list[dict[str, object]] = []
        for original, path in zip(files, staged, strict=True):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            values.append(
                {
                    "original_path": str(Path(original).expanduser().resolve()),
                    "workspace_path": path.relative_to(workspace.root).as_posix(),
                    "sha256": digest.hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        return tuple(values)
    
    def _get_model_control_plane(self) -> ModelControlPlane:
        from agent_runtime.model_config_io import discover_git_worktree

        workspace = self._workspace_path()
        session_path = self.model_session_path
        if session_path is not None and not session_path.is_absolute():
            session_path = workspace / session_path
        return ModelControlPlane.from_env(
            initial_model_id=self.model,
            initial_selection_requester=self._selection_requester,
            session_path=session_path, workspace=workspace,
            worktree=discover_git_worktree(workspace),
        )


def _positive_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


async def _close_owned_sync_resource(resource: object, *, label: str) -> None:
    close_method = getattr(resource, "close", None)
    if not callable(close_method):
        return
    try:
        await asyncio.wait_for(
            asyncio.to_thread(close_method),
            timeout=_RUNTIME_CLOSE_GRACE_SECONDS,
        )
    except TimeoutError:
        logger.warning(
            "%s close exceeded %.1fs grace period",
            label,
            _RUNTIME_CLOSE_GRACE_SECONDS,
        )
    except Exception as exc:
        logger.warning("%s close failed (%s)", label, type(exc).__name__[:120])

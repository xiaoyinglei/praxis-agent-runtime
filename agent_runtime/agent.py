"""Public Agent SDK backed exclusively by the Rollout Harness."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
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
    from agent_runtime.harness import BoundHarnessModel, RuntimeComposition
    from agent_runtime.tools.tool import Tool

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
        _selection_requester: ModelSwitchRequester = "system",
    ) -> None:
        if knowledge is not None and not isinstance(knowledge, RAGKnowledgeConfig):
            raise TypeError("knowledge must be RAGKnowledgeConfig or None")
        if not isinstance(enable_workspace_mcp, bool):
            raise TypeError("enable_workspace_mcp must be bool")
        selection_requester = validate_model_switch_requester(_selection_requester)
        self.model = model
        self.checkpoint_db = checkpoint_db
        self.workspace_path = None if workspace_path is None else Path(workspace_path).expanduser().resolve()
        self.model_session_path = model_session_path
        self.knowledge = knowledge
        self.enable_workspace_mcp = enable_workspace_mcp
        self._selection_requester = selection_requester
        self._model_control_plane: ModelControlPlane | None = None
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
        return spec

    def _request_model_switch(self, model_id: str) -> ModelSpec:
        spec = self._get_model_control_plane().request_model_switch(model_id)
        self.model = spec.id
        self._followup_model_id = spec.id
        return spec

    def run(
        self,
        task: str,
        *,
        previous_turn_id: str | None = None,
        files: Sequence[str] | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        require_workspace_change: bool = True,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        event_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        return asyncio.run(
            self.arun(
                task,
                previous_turn_id=previous_turn_id,
                files=files,
                max_turns=max_turns,
                max_tokens_total=max_tokens_total,
                require_workspace_change=require_workspace_change,
                allow_write_tools=allow_write_tools,
                allow_execute_tools=allow_execute_tools,
                event_sink=event_sink,
            )
        )

    async def arun(
        self,
        task: str,
        *,
        previous_turn_id: str | None = None,
        files: Sequence[str] | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        require_workspace_change: bool = True,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        event_sink: AgentEventSink | None = None,
        _event_dispatcher: TurnEventDispatcher | None = None,
    ) -> AgentResult:
        runtime_agent = (
            self if previous_turn_id is None else self._harness_agent_for_turn(previous_turn_id, followup=True)
        )
        input_file_items = runtime_agent._stage_harness_files(files or ())
        dispatcher = _event_dispatcher or TurnEventDispatcher()
        if event_sink is not None:
            dispatcher.subscribe_controlling_sink(event_sink)
        async with runtime_agent._open_harness_runtime(
            require_workspace_change=require_workspace_change,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            max_steps=max_turns,
            max_tokens_total=max_tokens_total,
            event_dispatcher=dispatcher,
        ) as runtime:
            internal = await runtime.thread_manager.run(
                user_message=task,
                previous_turn_id=previous_turn_id,
                input_files=input_file_items,
            )
            return AgentResult._from_harness(
                internal,
                store=runtime.store,
                files=tuple(files or ()),
            )

    def resume(
        self,
        turn_id: str,
        action: str,
        *,
        user_input: str | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        return asyncio.run(
            self.aresume(
                turn_id,
                action,
                user_input=user_input,
                event_sink=event_sink,
            )
        )

    async def aresume(
        self,
        turn_id: str,
        action: str,
        *,
        user_input: str | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        from agent_runtime.harness import RolloutStore, TurnResult

        runtime_agent = self._harness_agent_for_turn(turn_id, followup=False)
        with RolloutStore(self._harness_database()) as store:
            turn = store.read_turn(turn_id)
            pending = tuple(item for item in store.list_interactions(turn_id) if item.status == "pending")
            completion_policy = turn.binding_manifest.get("completion_policy")
            require_change = (
                isinstance(completion_policy, Mapping) and completion_policy.get("require_workspace_change") is True
            )
            tool_execution_policy = turn.binding_manifest.get("tool_execution_policy")
            allow_write_tools = (
                isinstance(tool_execution_policy, Mapping) and tool_execution_policy.get("allow_write_tools") is True
            )
            allow_execute_tools = (
                isinstance(tool_execution_policy, Mapping) and tool_execution_policy.get("allow_execute_tools") is True
            )
            step_budget = _positive_integer(turn.binding_manifest.get("model_step_budget"))
            token_budget = _positive_integer(turn.binding_manifest.get("model_token_budget_total"))
            unknown_model = tuple(
                operation for operation in store.list_model_operations(turn_id) if operation.status == "unknown"
            )
            model_operations = store.list_model_operations(turn_id)
            resolved_approved_ready = any(
                interaction.kind == "tool_approval"
                and interaction.status == "resolved"
                and interaction.response.get("decision") == "approve"
                and interaction.operation_id is not None
                and store.read_tool_operation(interaction.operation_id).status == "ready"
                for interaction in store.list_interactions(turn_id)
            )
            recoverable_committed_response = False
            if (
                turn.status == "running"
                and model_operations
                and model_operations[-1].status == "completed"
                and model_operations[-1].response_item_id is not None
            ):
                response_item = store.read_item(model_operations[-1].response_item_id)
                calls = response_item.payload.get("tool_calls")
                recoverable_committed_response = bool(isinstance(calls, (list, tuple)) and calls)
        if action != "abort" and turn.binding_manifest.get("legacy_resume_compatible") is False:
            raise RuntimeError(
                "incompatible legacy Turn cannot resume; its approvals were "
                "invalidated during migration, so abort the Turn explicitly"
            )
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
        dispatcher = TurnEventDispatcher()
        if event_sink is not None:
            dispatcher.subscribe_controlling_sink(event_sink)
        async with runtime_agent._open_harness_runtime(
            require_workspace_change=require_change,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            max_steps=step_budget,
            max_tokens_total=token_budget,
            event_dispatcher=dispatcher,
        ) as runtime:
            if len(pending) == 1 and pending[0].kind == "tool_approval":
                decision = {
                    "allow_once": "approve",
                    "approve": "approve",
                    "deny": "deny",
                }.get(action)
                if decision is None:
                    raise ValueError("tool approval action must be allow_once, approve, or deny")
                internal = await runtime.thread_manager.resume(
                    turn_id=turn_id,
                    decision=decision,
                )
            elif len(pending) == 1 and pending[0].kind in {
                "clarification",
                "choice",
            }:
                if action != "continue" or user_input is None:
                    raise ValueError(f"{pending[0].kind} resume requires action=continue and user_input")
                internal = await runtime.thread_manager.respond_interaction(
                    turn_id=turn_id,
                    request_id=pending[0].request_id,
                    response=user_input,
                )
            elif (
                not pending
                and len(unknown_model) == 1
                and action
                in {
                    "continue",
                    "retry",
                }
            ):
                internal = await runtime.thread_manager.retry_unknown_model(turn_id=turn_id)
            elif (
                not pending
                and resolved_approved_ready
                and action
                in {
                    "allow_once",
                    "approve",
                    "continue",
                    "retry",
                }
            ):
                internal = await runtime.thread_manager.resume(
                    turn_id=turn_id,
                    decision="approve",
                )
            elif (
                not pending
                and recoverable_committed_response
                and action
                in {
                    "continue",
                    "retry",
                }
            ):
                internal = await runtime.thread_manager.recover_committed_model_response(turn_id=turn_id)
            else:
                raise RuntimeError("resume action does not match the Turn's durable pending state")
            return AgentResult._from_harness(internal, store=runtime.store)

    def read_result(self, turn_id: str) -> AgentResult:
        return asyncio.run(self.aread_result(turn_id))

    async def aread_result(self, turn_id: str) -> AgentResult:
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

    async def astream(
        self,
        task: str,
        *,
        previous_turn_id: str | None = None,
        files: Sequence[str] | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        require_workspace_change: bool = True,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Yield committed durable events while the Turn is still running."""

        dispatcher = TurnEventDispatcher()
        stream = dispatcher.subscribe_controlling()
        run_task = asyncio.create_task(
            self.arun(
                task,
                previous_turn_id=previous_turn_id,
                files=files,
                max_turns=max_turns,
                max_tokens_total=max_tokens_total,
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

    def pending_input(self, turn_id: str) -> AgentPause | None:
        return asyncio.run(self.apending_input(turn_id))

    async def apending_input(self, turn_id: str) -> AgentPause | None:
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

    def _harness_agent_for_turn(self, turn_id: str, *, followup: bool) -> Agent:
        from agent_runtime.harness import RolloutStore

        with RolloutStore(self._harness_database()) as store:
            turn = store.read_turn(turn_id)
            thread = store.read_thread(turn.thread_id)
            alias = (
                None
                if "authentication_schema_version" in turn.binding_manifest
                else turn.binding_manifest.get("model_alias")
            )
            knowledge_value = turn.binding_manifest.get("knowledge_config")
            mcp_policy = turn.binding_manifest.get("mcp_policy")
        frozen_knowledge = (
            RAGKnowledgeConfig.model_validate(knowledge_value) if isinstance(knowledge_value, Mapping) else None
        )
        if not followup:
            model = None
        elif self._followup_model_id is not None:
            model = self._followup_model_id
        elif isinstance(alias, str) and alias:
            model = alias
        else:
            model = self.model
        restored = Agent(
            model=model,
            checkpoint_db=self.checkpoint_db,
            workspace_path=thread.workspace,
            model_session_path=self.model_session_path,
            knowledge=frozen_knowledge,
            enable_workspace_mcp=(
                bool(mcp_policy.get("workspace_discovery_enabled"))
                if isinstance(mcp_policy, Mapping) and isinstance(mcp_policy.get("workspace_discovery_enabled"), bool)
                else self.enable_workspace_mcp
            ),
            _selection_requester=self._selection_requester,
        )
        override = self.__dict__.get("_harness_model")
        if callable(override):
            restored.__dict__["_harness_model"] = override
        return restored

    def _harness_database(self) -> Path:
        if self.checkpoint_db is not None:
            return Path(self.checkpoint_db).expanduser().resolve()
        return self._workspace_path() / ".praxis" / "runtime" / "rollout.sqlite3"

    def _workspace_path(self) -> Path:
        return (self.workspace_path or Path.cwd()).expanduser().resolve()

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

    @asynccontextmanager
    async def _open_harness_runtime(
        self,
        *,
        require_workspace_change: bool,
        allow_write_tools: bool,
        allow_execute_tools: bool,
        max_steps: int | None,
        max_tokens_total: int | None,
        event_dispatcher: TurnEventDispatcher | None = None,
    ) -> AsyncIterator[RuntimeComposition]:
        from agent_runtime.harness import RuntimeComposition
        from agent_runtime.runtime.mcp import (
            decide_mcp_config_trust,
            open_trusted_product_mcp_tools,
            resolve_product_mcp_config,
        )
        from agent_runtime.skills.catalog import SkillCatalog
        from agent_runtime.skills.loader import scan_and_load_skills
        from agent_runtime.skills.policy import SkillPolicy
        from agent_runtime.skills.runtime import SkillRuntime
        from agent_runtime.tools.builtins import create_resident_coding_tools
        from agent_runtime.tools.permissions import ToolExecutionContext
        from agent_runtime.workspace import open_workspace

        workspace = open_workspace(self._workspace_path(), create=True)

        def acknowledge_plan_update(_arguments: object) -> dict[str, object]:
            return {
                "accepted": True,
                "revision": 0,
                "message": "Plan update recorded as a ToolResult.",
            }

        resident = create_resident_coding_tools(
            workspace,
            plan_updater=acknowledge_plan_update,
        )
        skill_policy = SkillPolicy()
        manifests = [
            manifest
            for manifest in scan_and_load_skills(
                workspace.root,
                repo_root=workspace.root,
            )
            if skill_policy.is_skill_enabled(manifest)
        ]
        candidate_skill_runtime = SkillRuntime(
            SkillCatalog(manifests),
            policy=skill_policy,
        )
        skill_runtime = candidate_skill_runtime if candidate_skill_runtime.has_model_invocable_skills else None
        provider: object | None = None
        knowledge_revision: str | None = None
        knowledge_runner: object | None = None
        if self.knowledge is not None:
            from agent_runtime.knowledge_providers.rag import (
                LazyRAGKnowledgeProvider,
            )

            provider = LazyRAGKnowledgeProvider(
                config=self.knowledge,
                model_alias=self.model,
                vector_dsn=os.environ.get("AGENT_VECTOR_DSN"),
            )
            knowledge_runner = provider.search_knowledge
            knowledge_revision = "rag_" + hashlib.sha256(self.knowledge.model_dump_json().encode()).hexdigest()[:16]

        async with AsyncExitStack() as stack:
            config_path = resolve_product_mcp_config(workspace.root) if self.enable_workspace_mcp else None
            mcp_tools: tuple[Tool, ...] = ()
            if config_path is not None:
                trust = decide_mcp_config_trust(
                    config_path,
                    workspace_root=workspace.root,
                    trust_workspace=False,
                )
                mcp_tools = await stack.enter_async_context(open_trusted_product_mcp_tools(config_path, trust=trust))
            tools = {tool.definition.name: tool for tool in (*resident, *mcp_tools)}
            runtime = RuntimeComposition.open(
                database=self._harness_database(),
                workspace=workspace.root,
                model=self._harness_model(),
                tools=tools,
                tool_execution_context=ToolExecutionContext(
                    workspace_root=workspace.root,
                    cwd=workspace.root,
                    allow_write_tools=allow_write_tools,
                    allow_execute_tools=allow_execute_tools,
                ),
                knowledge_runner=(knowledge_runner if callable(knowledge_runner) else None),
                knowledge_revision=knowledge_revision,
                knowledge_config=(None if self.knowledge is None else self.knowledge.model_dump(mode="json")),
                discoverable_tool_names=tuple(tool.definition.name for tool in mcp_tools),
                workspace_mcp_enabled=self.enable_workspace_mcp,
                require_workspace_change=require_workspace_change,
                enable_subagents=True,
                skill_runtime=skill_runtime,
                max_steps=16 if max_steps is None else max_steps,
                max_tokens_total=max_tokens_total,
                event_dispatcher=event_dispatcher,
            )
            try:
                yield runtime
            finally:
                runtime.close()
                if provider is not None:
                    await _close_owned_sync_resource(
                        provider,
                        label="knowledge provider",
                    )
                await self._close_model_control_plane()

    def _get_model_control_plane(self) -> ModelControlPlane:
        if self._model_control_plane is None:
            from agent_runtime.model_config_io import discover_git_worktree

            workspace = self._workspace_path()
            session_path = self.model_session_path
            if session_path is not None and not session_path.is_absolute():
                session_path = workspace / session_path
            self._model_control_plane = ModelControlPlane.from_env(
                initial_model_id=self.model,
                initial_selection_requester=self._selection_requester,
                session_path=session_path,
                workspace=workspace,
                worktree=discover_git_worktree(workspace),
            )
        return self._model_control_plane

    async def _close_model_control_plane(self) -> None:
        control_plane = self._model_control_plane
        self._model_control_plane = None
        if control_plane is not None:
            await _close_owned_sync_resource(
                control_plane,
                label="model control plane",
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

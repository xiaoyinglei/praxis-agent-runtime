"""One live conversation: owned resources, persistent Turns, temporary Steps."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from agent_runtime.budget import ResourceUsage
from agent_runtime.harness.protocol import BoundHarnessModel, CompletionGate, ContextManager, ToolRouter, TurnResult
from agent_runtime.harness.rollout import RolloutStore
from agent_runtime.harness.services import SessionBindingProvider, _tool_execution_policy_snapshot, configure_services
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator, current_tool_binding
from agent_runtime.harness.turn import TurnExecutor
from agent_runtime.result import AgentResult
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolEffect

if TYPE_CHECKING:
    from agent_runtime.agent import Agent, AgentEventSink
    from agent_runtime.models import ModelControlPlane, ModelSpec
    from agent_runtime.tools.tool import Tool


class Session:
    """Open once per conversation; submit creates a Turn without reopening services.

    Use Agent.session() or async with await Session.open(...). close() stops
    accepting submissions and waits for the active call before releasing resources.
    A paused Turn must be resumed or aborted before another submission.
    An explicitly supplied event dispatcher belongs to its caller, which drains
    and closes it after the Session finishes.
    """

    tool_execution_context: ToolExecutionContext
    store: RolloutStore
    database: Path
    workspace_path: Path
    thread_id: str
    head_turn_id: str | None
    model: BoundHarnessModel
    model_control_plane: ModelControlPlane | None
    context_manager: ContextManager
    completion_gate: CompletionGate
    tool_router: ToolRouter | None
    tool_orchestrator: ToolOrchestrator | None
    tools: Mapping[str, Tool]
    worker_id: str
    max_steps: int
    binding_provider: SessionBindingProvider
    event_dispatcher: TurnEventDispatcher

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._active_turn_lock = asyncio.Lock()
        self._active_task: asyncio.Task[Any] | None = None
        self._child_calls: dict[asyncio.Task[Any], asyncio.Event] = {}
        self._closed = False
        self._closing = False
        self._agent: Agent | None = None
        self.model_control_plane = None
        self._ready_model_id: str | None = None
        self._service_options: dict[str, Any] = {}

    @classmethod
    async def open(
        cls,
        *,
        agent: Agent | None = None,
        database: Path | None = None,
        workspace: Path | None = None,
        model: BoundHarnessModel | None = None,
        previous_turn_id: str | None = None,
        frozen_turn_id: str | None = None,
        thread_id: str | None = None,
        event_dispatcher: TurnEventDispatcher | None = None,
        event_sink: AgentEventSink | None = None,
        **service_options: Any,
    ) -> Self:
        self = cls()
        explicit_tool_context = "tool_execution_context" in service_options
        agent = copy.copy(agent) if agent is not None else None
        self._agent = agent
        self.event_dispatcher = event_dispatcher or TurnEventDispatcher()
        if event_sink is not None:
            self.event_dispatcher.subscribe_controlling_sink(event_sink)
        if event_dispatcher is None:
            self._stack.callback(self.event_dispatcher.close)
        try:
            if agent is not None:
                database = agent._harness_database()
                workspace = agent.workspace_path
            if database is None or workspace is None:
                raise ValueError("Session requires an Agent or database and workspace")
            self.database = Path(database)
            self.workspace_path = Path(workspace).resolve()
            self.store = self._stack.enter_context(RolloutStore(self.database))
            integrity = self.store.verify()
            if not integrity.valid:
                raise RuntimeError("Rollout projection integrity check failed: " + "; ".join(integrity.errors))
            if sum(value is not None for value in (previous_turn_id, frozen_turn_id, thread_id)) > 1:
                raise ValueError("choose one conversation anchor")
            anchor = frozen_turn_id or previous_turn_id
            if anchor is not None:
                turn = self.store.read_turn(anchor)
                thread = self.store.read_thread(turn.thread_id)
                if Path(thread.workspace).resolve() != self.workspace_path:
                    raise RuntimeError("turn belongs to a different workspace security domain")
                if previous_turn_id is not None:
                    if turn.status not in {"completed", "failed", "cancelled"}:
                        raise RuntimeError("non-terminal predecessor must be resumed, cancelled, or abandoned")
                    if thread.head_turn_id != previous_turn_id:
                        thread = self.store.fork_thread(from_turn_id=previous_turn_id)
                thread_id = thread.thread_id
                if frozen_turn_id is not None:
                    binding = turn.binding_manifest
                    if binding.get("legacy_resume_compatible") is False:
                        raise RuntimeError("incompatible legacy Turn cannot resume; abort the Turn explicitly")
                    completion = binding.get("completion_policy", {})
                    service_options.update(
                        require_workspace_change=completion.get("require_workspace_change") is True,
                        max_steps=binding.get("model_step_budget") or 16,
                        max_tokens_total=binding.get("model_token_budget_total"),
                        max_cost_micros=binding.get("model_cost_budget_total_micros"),
                    )
            if thread_id is not None:
                thread = self.store.read_thread(thread_id)
                if Path(thread.workspace).resolve() != self.workspace_path:
                    raise RuntimeError("thread belongs to a different workspace security domain")
            if agent is not None:
                model, resource_options = await self._open_agent_resources(
                    agent,
                    allow_write_tools=service_options.pop("allow_write_tools", False),
                    allow_execute_tools=service_options.pop("allow_execute_tools", False),
                )
                service_options.update(resource_options)
            if model is None:
                raise ValueError("Session requires a bound model")
            self._service_options = dict(service_options)
            configure_services(self, workspace=self.workspace_path, model=model, **service_options)
            self.thread_id = (
                thread_id
                if thread_id is not None
                else self.store.create_thread(workspace=self.workspace_path).thread_id
            )
            thread = self.store.read_thread(self.thread_id)
            self.head_turn_id = thread.head_turn_id
            settings = thread.settings
            if settings:
                policy = dict(settings["tool_execution_policy"])
                policy["deny_effects"] = frozenset(ToolEffect(value) for value in policy["deny_effects"])
                if not explicit_tool_context:
                    self.tool_execution_context = replace(self.tool_execution_context, **policy)
                if self.model_control_plane is not None and settings.get("model_id"):
                    self.model_control_plane.state = replace(
                        self.model_control_plane.state,
                        current_model_id=settings["model_id"],
                        selection_requester=settings["model_selection_requester"],
                    )
                self._apply_tool_context()
                if explicit_tool_context:
                    self._persist_settings()
            else:
                self._persist_settings()
            if agent is not None and agent._followup_model_id is not None:
                assert self.model_control_plane is not None
                self.model_control_plane.switch_model(
                    agent._followup_model_id,
                    requested_by=agent._selection_requester,
                    persist=False,
                )
                self._persist_settings()
            if self.model_control_plane is not None and not settings and frozen_turn_id is None:
                await self._bootstrap_model_provider()
            return self
        except BaseException:
            await self._stack.aclose()
            self._closed = True
            raise

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("Session is closed")

    async def close(self) -> None:
        if self._closed:
            return
        if self._active_task is asyncio.current_task() or asyncio.current_task() in self._child_calls:
            raise RuntimeError("cannot close Session from its active Turn")
        self._closing = True
        async with self._active_turn_lock:
            if not self._closed:
                if self._child_calls:
                    await asyncio.gather(*(done.wait() for done in tuple(self._child_calls.values())))
                try:
                    await self._stack.aclose()
                finally:
                    self._closed = True

    @asynccontextmanager
    async def _active_turn(self, event_sink: AgentEventSink | None = None) -> AsyncIterator[None]:
        self._ensure_open()
        if self._active_turn_lock.locked():
            raise RuntimeError("Session already has an active Turn")
        async with self._active_turn_lock:
            self._ensure_open()
            self._active_task = asyncio.current_task()
            if event_sink is not None:
                self.event_dispatcher.subscribe_controlling_sink(event_sink)
            try:
                yield
            finally:
                if event_sink is not None:
                    self.event_dispatcher.unsubscribe_controlling_sink(event_sink)
                if self.tool_orchestrator is not None:
                    self.tool_orchestrator.release_turn_state()
                self.head_turn_id = self.store.read_thread(self.thread_id).head_turn_id
                self._active_task = None

    def _executor(self) -> TurnExecutor:
        executor = TurnExecutor(
            thread_id=self.thread_id,
            store=self.store,
            model=self.model,
            context_manager=self.context_manager,
            completion_gate=self.completion_gate,
            tool_router=self.tool_router,
            tool_orchestrator=self.tool_orchestrator,
            worker_id=self.worker_id,
            max_steps=self.max_steps,
            binding_provider=self.binding_provider,
            prepare_binding=self._prepare_step_binding,
        )
        executor.attach_event_dispatcher(self.event_dispatcher)
        return executor

    async def submit(
        self,
        task: str,
        *,
        files: Sequence[str] | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        async with self._active_turn(event_sink):
            if self.store.read_thread(self.thread_id).active_turn_id is not None:
                raise RuntimeError("active Turn must be resumed or aborted before submit")
            if self.model_control_plane is not None and self.current_model().id != self._ready_model_id:
                await self._bootstrap_model_provider()
            input_files: tuple[dict[str, object], ...] = ()
            if files:
                if self._agent is None:
                    raise ValueError("file staging requires an Agent-configured Session")
                input_files = self._agent._stage_harness_files(files)
            turn_id = f"turn_{uuid4().hex}"
            binding = dict(self.binding_provider.snapshot(thread_id=self.thread_id, turn_id=turn_id))
            binding["budget_root_turn_id"] = turn_id
            result = await self._executor().run(
                turn_id=turn_id,
                user_message=task,
                binding_manifest=binding,
                input_files=input_files,
            )
            return AgentResult._from_harness(result, store=self.store, files=tuple(files or ()))

    def current_model(self) -> ModelSpec:
        if self.model_control_plane is None:
            raise RuntimeError("Session has no model control plane")
        return self.model_control_plane.current_model()

    def models(self) -> list[ModelSpec]:
        if self.model_control_plane is None:
            raise RuntimeError("Session has no model control plane")
        return self.model_control_plane.list_models()

    def switch_model(self, model_id: str) -> ModelSpec:
        self._ensure_open()
        if self.model_control_plane is None:
            raise RuntimeError("Session has no model control plane")
        previous = copy.copy(self.model_control_plane.state)
        spec = self.model_control_plane.switch_model(model_id, requested_by="user", persist=False)
        try:
            self._persist_settings()
        except BaseException:
            self.model_control_plane.state = previous
            raise
        return spec

    def _persist_settings(self) -> None:
        self.store.update_session_settings(
            self.thread_id,
            {
                "model_id": None
                if self.model_control_plane is None
                else self.model_control_plane.state.current_model_id,
                "model_selection_requester": (
                    None if self.model_control_plane is None else self.model_control_plane.state.selection_requester
                ),
                "tool_execution_policy": _tool_execution_policy_snapshot(self.tool_execution_context),
            },
        )

    def _apply_tool_context(self) -> None:
        if self.tool_orchestrator is not None:
            self.tool_orchestrator.update_execution_context(self.tool_execution_context)

    def update_tool_policy(
        self,
        *,
        allow_write_tools: bool | None = None,
        allow_execute_tools: bool | None = None,
        require_confirmation_for: frozenset[str] | None = None,
        denied_tool_names: frozenset[str] | None = None,
        deny_effects: frozenset[ToolEffect] | None = None,
        auto_approve_sandboxed: bool | None = None,
    ) -> None:
        self._ensure_open()
        updates: dict[str, Any] = {
            name: value
            for name, value in {
                "allow_write_tools": allow_write_tools,
                "allow_execute_tools": allow_execute_tools,
                "require_confirmation_for": require_confirmation_for,
                "denied_tool_names": denied_tool_names,
                "deny_effects": deny_effects,
                "auto_approve_sandboxed": auto_approve_sandboxed,
            }.items()
            if value is not None
        }
        previous = self.tool_execution_context
        self.tool_execution_context = replace(previous, **updates)
        try:
            self._persist_settings()
        except BaseException:
            self.tool_execution_context = previous
            raise
        self._apply_tool_context()

    async def _open_agent_resources(
        self,
        agent: Agent,
        *,
        allow_write_tools: bool,
        allow_execute_tools: bool,
    ) -> tuple[BoundHarnessModel, dict[str, Any]]:
        from agent_runtime.agent import _close_owned_sync_resource
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

        workspace = open_workspace(agent._workspace_path(), create=True)
        self.model_control_plane = agent._get_model_control_plane()
        self._stack.push_async_callback(
            _close_owned_sync_resource,
            self.model_control_plane,
            label="model control plane",
        )

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
        if agent.knowledge is not None:
            from agent_runtime.knowledge_providers.rag import (
                LazyRAGKnowledgeProvider,
            )

            provider = LazyRAGKnowledgeProvider(
                config=agent.knowledge,
                model_id=agent.model,
                vector_dsn=os.environ.get("AGENT_VECTOR_DSN"),
            )
            self._stack.push_async_callback(_close_owned_sync_resource, provider, label="knowledge provider")
            knowledge_runner = provider.search_knowledge
            knowledge_revision = "rag_" + hashlib.sha256(agent.knowledge.model_dump_json().encode()).hexdigest()[:16]

        config_path = resolve_product_mcp_config(workspace.root) if agent.enable_workspace_mcp else None
        mcp_tools: tuple[Tool, ...] = ()
        mcp_trust_binding: Mapping[str, object] | None = None
        if config_path is not None:
            trust = agent.mcp_config_trust or decide_mcp_config_trust(
                config_path,
                workspace_root=workspace.root,
                trust_workspace=False,
            )
            mcp_tools = await self._stack.enter_async_context(open_trusted_product_mcp_tools(config_path, trust=trust))
            mcp_trust_binding = {
                "config_path": str(trust.config_path),
                "config_source": trust.source,
                "config_sha256": trust.config_sha256,
            }
        tools = {tool.definition.name: tool for tool in (*resident, *mcp_tools)}

        from agent_runtime.builtin.generic import GENERIC_SYSTEM_PROMPT
        from agent_runtime.harness.model_adapter import ControlPlaneHarnessModel

        override = agent.__dict__.get("_harness_model")
        model = (
            override()
            if callable(override)
            else ControlPlaneHarnessModel(
                control_plane=self.model_control_plane,
                instructions=(GENERIC_SYSTEM_PROMPT,),
            )
        )
        return model, dict(
            tools=tools,
            tool_execution_context=ToolExecutionContext(
                workspace_root=workspace.root,
                cwd=workspace.root,
                allow_write_tools=allow_write_tools,
                allow_execute_tools=allow_execute_tools,
            ),
            knowledge_runner=knowledge_runner if callable(knowledge_runner) else None,
            knowledge_revision=knowledge_revision,
            knowledge_config=None if agent.knowledge is None else agent.knowledge.model_dump(mode="json"),
            discoverable_tool_names=tuple(tool.definition.name for tool in mcp_tools),
            workspace_mcp_enabled=agent.enable_workspace_mcp,
            mcp_config_trust=mcp_trust_binding,
            enable_subagents=True,
            skill_runtime=skill_runtime,
        )

    async def _prepare_step_binding(self, binding: Mapping[str, Any]) -> None:
        if self.model_control_plane is None or "authentication_schema_version" not in binding:
            validator = getattr(self.model, "ensure_available", None)
            if callable(validator):
                validator(binding, thread_id=binding["thread_id"], turn_id=binding["turn_id"])
            return
        from agent_runtime.local_runtime import ensure_local_provider_ready

        spec = self.model_control_plane.model_spec_for_frozen_binding(
            binding,
            thread_id=binding["thread_id"],
            turn_id=binding["turn_id"],
        )
        if spec.id != self._ready_model_id:
            await ensure_local_provider_ready(spec)
            self._ready_model_id = spec.id

    async def _bootstrap_model_provider(self) -> None:
        from agent_runtime.local_runtime import ensure_local_provider_ready

        assert self.model_control_plane is not None
        spec = self.model_control_plane.current_model()
        await ensure_local_provider_ready(spec)
        self._ready_model_id = spec.id

    async def resume(
        self,
        turn_id: str,
        action: str,
        *,
        user_input: str | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        async with self._active_turn(event_sink):
            turn = self.store.read_turn(turn_id)
            if turn.thread_id != self.thread_id:
                raise RuntimeError("Turn belongs to a different Session")
            executor = self._executor()
            if action == "abort":
                cancelled = await executor._commit(lambda: self.store.cancel_turn(turn_id=turn_id))
                return AgentResult._from_harness(
                    TurnResult(
                        thread_id=self.thread_id,
                        turn_id=turn_id,
                        answer=None,
                        status=cancelled.status,
                    ),
                    store=self.store,
                )
            pending = tuple(item for item in self.store.list_interactions(turn_id) if item.status == "pending")
            approvals = [item for item in self.store.list_interactions(turn_id) if item.kind == "tool_approval"]
            if turn.status == "completed" and approvals and action in {"allow_once", "approve", "deny"}:
                internal = await executor.resume(
                    turn_id=turn_id,
                    decision="approve" if action == "allow_once" else action,
                )
                return AgentResult._from_harness(internal, store=self.store)
            model_operations = self.store.list_model_operations(turn_id)
            unknown_model = tuple(operation for operation in model_operations if operation.status == "unknown")
            resolved_approved_ready = any(
                interaction.kind == "tool_approval"
                and interaction.status == "resolved"
                and interaction.response.get("decision") == "approve"
                and interaction.operation_id is not None
                and self.store.read_tool_operation(interaction.operation_id).status == "ready"
                for interaction in self.store.list_interactions(turn_id)
            )
            recoverable_committed_response = False
            if (
                turn.status == "running"
                and model_operations
                and model_operations[-1].status == "completed"
                and model_operations[-1].response_item_id is not None
            ):
                response = self.store.read_item(model_operations[-1].response_item_id)
                calls = response.payload.get("tool_calls")
                recoverable_committed_response = bool(isinstance(calls, (list, tuple)) and calls)
            if len(pending) == 1 and pending[0].kind == "tool_approval":
                decision = {
                    "allow_once": "approve",
                    "approve": "approve",
                    "deny": "deny",
                }.get(action)
                if decision is None:
                    raise ValueError("tool approval action must be allow_once, approve, or deny")
                internal = await executor.resume(
                    turn_id=turn_id,
                    decision=decision,
                )
            elif len(pending) == 1 and pending[0].kind in {
                "clarification",
                "choice",
            }:
                if action != "continue" or user_input is None:
                    raise ValueError(f"{pending[0].kind} resume requires action=continue and user_input")
                internal = await executor.respond_interaction(
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
                internal = await executor.retry_unknown_model(turn_id=turn_id)
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
                internal = await executor.resume(
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
                internal = await executor.recover_committed_model_response(turn_id=turn_id)
            else:
                raise RuntimeError("resume action does not match the Turn's durable pending state")
            return AgentResult._from_harness(internal, store=self.store)

    async def run_child(
        self,
        *,
        user_message: str,
        max_steps: int | None,
        max_tokens_total: int | None,
        max_cost_micros: int | None = None,
        parent_turn_id: str | None = None,
    ) -> TurnResult:
        self._ensure_open()
        task = asyncio.current_task()
        assert task is not None
        completed = asyncio.Event()
        self._child_calls[task] = completed
        try:
            thread_id = f"thread_{uuid4().hex}"
            turn_id = f"turn_{uuid4().hex}"
            source_binding = current_tool_binding()
            if source_binding is None:
                source_binding = self.binding_provider.snapshot(
                    thread_id=self.thread_id, turn_id=parent_turn_id or turn_id
                )
            binding = dict(
                self.binding_provider.snapshot(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    model_binding=source_binding,
                )
            )
            policy_value = source_binding["tool_execution_policy"]
            if not isinstance(policy_value, Mapping):
                raise RuntimeError("source Step has no tool execution policy")
            policy = dict(policy_value)
            binding["tool_execution_policy"] = dict(policy)
            policy["deny_effects"] = frozenset(ToolEffect(value) for value in policy["deny_effects"])
            child_options = {
                **self._service_options,
                "model_binding": source_binding,
                "tool_execution_context": replace(self.tool_execution_context, **policy),
            }
            # Borrow providers through the source binding, independent of the parent's next selection.
            await self._prepare_step_binding(source_binding)
            if max_steps is not None:
                binding["model_step_budget"] = max_steps
            binding["completion_policy"] = {"require_workspace_change": False}
            if parent_turn_id is None:
                self.store.create_thread(workspace=self.workspace_path, thread_id=thread_id)
                binding["budget_root_turn_id"] = turn_id
                if max_tokens_total is not None:
                    binding["model_token_budget_total"] = max_tokens_total
                if max_cost_micros is not None:
                    binding["model_cost_budget_total_micros"] = max_cost_micros
            else:
                parent = self.store.read_turn(parent_turn_id)
                if parent.thread_id != self.thread_id or parent.status != "running":
                    raise RuntimeError("subagent parent Turn must still be running in this Session")
                self.store.start_budgeted_child_turn(
                    parent_turn_id=parent_turn_id,
                    child_thread_id=thread_id,
                    child_turn_id=turn_id,
                    user_message=user_message,
                    binding_manifest=binding,
                    requested_tokens=max_tokens_total,
                    requested_cost_micros=max_cost_micros,
                )
            # Child services and Store belong to a separate Session. The parent's
            # live model and installed external tools are borrowed for this awaited call.
            async with await Session.open(
                database=self.database,
                workspace=self.workspace_path,
                model=self.model,
                thread_id=thread_id,
                **child_options,
            ) as child:
                async with child._active_turn():
                    executor = child._executor()
                    if parent_turn_id is None:
                        result = await executor.run(
                            turn_id=turn_id,
                            user_message=user_message,
                            binding_manifest=binding,
                        )
                    else:
                        result = await executor.run_turn(executor.restore_turn_context(turn_id), start_step=1)
                if parent_turn_id is not None and result.status in {"completed", "failed", "cancelled"}:
                    state = self.store.read_budget_state(turn_id)
                    if state.reserved + state.uncertain + state.child_reserved == ResourceUsage():
                        self.store.settle_child_budget(child_turn_id=turn_id)
                return result
        finally:
            self._child_calls.pop(task)
            completed.set()

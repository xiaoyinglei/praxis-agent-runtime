"""One live conversation: owned resources, persistent Turns, temporary Steps."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from agent_runtime.approval import ApprovalMode, CommandGrant
from agent_runtime.budget import ResourceUsage
from agent_runtime.core.generation_settings import ModelRequestPreferences
from agent_runtime.harness.protocol import (
    BoundHarnessModel,
    CompletionDecision,
    CompletionGate,
    CompletionProposal,
    ContextManager,
    ToolRouter,
    TurnResult,
)
from agent_runtime.harness.rollout import ItemSnapshot, RolloutStore
from agent_runtime.harness.services import (
    SessionBindingProvider,
    _tool_execution_policy_snapshot,
    configure_services,
    restore_tool_policy,
)
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator, current_tool_binding
from agent_runtime.harness.turn import TurnExecutor
from agent_runtime.permission_policy import NetworkPolicy, PermissionPolicy, ResolvedPermissions
from agent_runtime.result import AgentResult
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolEffect

if TYPE_CHECKING:
    from agent_runtime.agent import Agent, AgentEventSink
    from agent_runtime.harness.memory import MemoryConfig
    from agent_runtime.models import ModelControlPlane, ModelSpec
    from agent_runtime.tools.tool import Tool


class _MemoryCompletion:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        return CompletionDecision("accept", "memory output ready for schema and source validation")


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
    memory_config: MemoryConfig
    _background_memory: bool
    _memory_control_version: int

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._active_turn_lock = asyncio.Lock()
        self._active_task: asyncio.Task[Any] | None = None
        self._child_calls: dict[asyncio.Task[Any], asyncio.Event] = {}
        self._closed = False
        self._closing = False
        self._persisted_request_revision = 0
        self._agent: Agent | None = None
        self.model_control_plane = None
        self._ready_model_id: str | None = None
        self._service_options: dict[str, Any] = {}
        self._memory_task: asyncio.Task[Any] | None = None
        self._mcp_config_path: Path | None = None
        self._mcp_catalog_revision: str | None = None
        self._cancel_live_commands: Callable[[], Awaitable[None]] | None = None
        self._command_sessions: Any = None

    def connections_changed(self) -> bool:
        """Report host setup changes; never mutate the tools of an active Turn."""
        if self._agent is None or not self._agent.enable_workspace_mcp:
            return False
        if self._resource_permissions().tool_decision("mcp__*") == "deny":
            return False
        from agent_runtime.runtime.mcp import resolve_product_mcp_config
        from agent_runtime.runtime.mcp_catalog import catalog_revision
        path = resolve_product_mcp_config(self.workspace_path)
        return path != self._mcp_config_path or (
            path is not None and catalog_revision(path) != self._mcp_catalog_revision
        )

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
        memory: MemoryConfig | None = None,
        _background_memory: bool = True,
        **service_options: Any,
    ) -> Self:
        self = cls()
        explicit_tool_context = "tool_execution_context" in service_options
        approval_mode = service_options.pop("approval_mode", None)
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
            from agent_runtime.harness.memory import MemoryConfig

            configured_memory = agent.memory if agent is not None and agent.memory is not None else memory
            state = self.store.read_memory_state(self.workspace_path)
            self._memory_control_version = state.get("control_version", 0)
            self.memory_config = configured_memory or MemoryConfig(
                use_memories=state["use_memories"], generate_memories=state["generate_memories"],
            )
            self._background_memory = _background_memory
            if frozen_turn_id is not None:
                original = self.store.read_turn(frozen_turn_id).binding_manifest
                self.memory_config = self.memory_config.model_copy(update={
                    "use_memories": "memory_snapshot" in original,
                })
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
                    if (agent is not None
                            and binding.get("mcp_policy", {}).get("workspace_discovery_enabled") is False):
                        # Reconcile the frozen capability before discovery can
                        # read configuration or open any server resources.
                        agent.enable_workspace_mcp = False
                    completion = binding.get("completion_policy", {})
                    service_options.update(
                        require_workspace_change=completion.get("require_workspace_change") is True,
                        acceptance_command=completion.get("acceptance_command"),
                        max_steps=binding.get("model_step_budget") or 16,
                        max_tokens_total=binding.get("model_token_budget_total"),
                        max_cost_micros=binding.get("model_cost_budget_total_micros"),
                    )
            if thread_id is not None:
                thread = self.store.read_thread(thread_id)
                if Path(thread.workspace).resolve() != self.workspace_path:
                    raise RuntimeError("thread belongs to a different workspace security domain")
                if (configured_memory is None and frozen_turn_id is None
                        and "memory_use_allowed" in thread.settings
                        and thread.settings.get("memory_control_version") == self._memory_control_version):
                    self.memory_config = MemoryConfig(
                        use_memories=thread.settings["memory_use_allowed"],
                        generate_memories=thread.settings["memory_generation_allowed"],
                    )
            if agent is not None:
                if thread_id is not None:
                    selected_files = thread.settings.get("input_read_files", ())
                    if not isinstance(selected_files, (list, tuple)) or any(
                        not isinstance(path, str) or not Path(path).is_absolute() for path in selected_files
                    ):
                        raise ValueError("invalid saved input file references")
                    agent._input_read_files = tuple(selected_files)
                model, resource_options = await self._open_agent_resources(
                    agent,
                    allow_write_tools=service_options.pop("allow_write_tools", False),
                    allow_execute_tools=service_options.pop("allow_execute_tools", False),
                    allow_web_tools=service_options.pop("allow_web_tools", False),
                    initial_tool_context=service_options.get("tool_execution_context"),
                    saved_tool_policy=(thread.settings.get("tool_execution_policy")
                                       if thread_id is not None and not explicit_tool_context else None),
                    discovery_ceiling=(original.get("tool_execution_policy", {}).get("permissions")
                                       if frozen_turn_id is not None else None),
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
            self._persisted_request_revision = settings.get("request_settings_revision", 0)
            if settings:
                if not explicit_tool_context:
                    self.tool_execution_context = restore_tool_policy(
                        self.tool_execution_context, settings["tool_execution_policy"], restore_session_grants=True,
                    )
                if self.model_control_plane is not None and settings.get("model_id"):
                    self.model_control_plane.state = replace(
                        self.model_control_plane.state,
                        current_model_id=settings["model_id"],
                        selection_requester=settings["model_selection_requester"],
                        request_preferences={key: ModelRequestPreferences.model_validate(value)
                                             for key, value in settings.get("request_preferences", {}).items()},
                        request_settings_revision=settings.get("request_settings_revision", 0),
                    )
                self._apply_tool_context()
                if explicit_tool_context:
                    self._persist_settings()
                elif (configured_memory is not None
                      or settings.get("memory_control_version") != self._memory_control_version):
                    self._persist_settings()
            else:
                self._persist_settings()
            saved_mode = settings.get("tool_execution_policy", {}).get("approval_mode")
            selected_mode = (approval_mode if approval_mode is not None
                             else saved_mode if not explicit_tool_context else None)
            if selected_mode is not None:
                if approval_mode == "ask":
                    self.update_tool_policy(approval_mode=selected_mode, allow_write_tools=False,
                                            allow_execute_tools=False, allow_web_tools=False,
                                            auto_approve_sandboxed=False)
                else:
                    self.update_tool_policy(approval_mode=selected_mode)
            if agent is not None and agent._followup_model_id is not None:
                assert self.model_control_plane is not None
                self.model_control_plane.switch_model(
                    agent._followup_model_id,
                    requested_by=agent._selection_requester,
                    persist=False,
                )
                self._persist_settings()
            if agent is not None and agent._pending_request_settings:
                pending = agent._pending_request_settings.get(self.current_model().id)
                if pending is not None:
                    self.configure_request(**pending["options"])
                    agent._pending_request_settings.pop(self.current_model().id)
            if self.model_control_plane is not None and not settings and frozen_turn_id is None:
                await self._bootstrap_model_provider()
            self._start_memory_update()
            if self._command_sessions is not None and self.head_turn_id is not None:
                assert self.tool_orchestrator is not None
                await self.tool_orchestrator.restore_command_sessions(self._command_sessions, self.head_turn_id)
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
        if self._memory_task is not None:
            self._memory_task.cancel()
            await asyncio.gather(self._memory_task, return_exceptions=True)
        async with self._active_turn_lock:
            if not self._closed:
                if self._child_calls:
                    await asyncio.gather(*(done.wait() for done in tuple(self._child_calls.values())))
                try:
                    await self._stack.aclose()
                finally:
                    self.tool_execution_context = replace(self.tool_execution_context, command_grants=())
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
                try:
                    if event_sink is not None:
                        self.event_dispatcher.unsubscribe_controlling_sink(event_sink)
                    if self.tool_orchestrator is not None:
                        self.tool_orchestrator.release_turn_state()
                    self.head_turn_id = self.store.read_thread(self.thread_id).head_turn_id
                    if (self._cancel_live_commands is not None and self.head_turn_id is not None
                            and self.store.read_turn(self.head_turn_id).status in {"completed", "failed", "cancelled"}):
                        await self._cancel_live_commands()
                finally:
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
        if not isinstance(task, str) or not task.strip():
            raise ValueError("user_message must be non-empty")
        async with self._active_turn(event_sink):
            if self.store.read_thread(self.thread_id).active_turn_id is not None:
                raise RuntimeError("active Turn must be resumed or aborted before submit")
            if self.model_control_plane is not None and self.current_model().id != self._ready_model_id:
                await self._bootstrap_model_provider()
            input_files: tuple[dict[str, object], ...] = ()
            if files:
                input_files = self.attach_files(files)
            turn_id = f"turn_{uuid4().hex}"
            binding = dict(self.binding_provider.snapshot(thread_id=self.thread_id, turn_id=turn_id))
            binding["budget_root_turn_id"] = turn_id
            if self._agent is not None:
                binding["web_limits"] = self._agent.web_limits.model_dump(mode="json")
            binding["memory_generation_allowed"] = self.memory_config.generate_memories
            if self.memory_config.use_memories:
                from agent_runtime.harness.memory import memory_snapshot

                binding["memory_snapshot"] = memory_snapshot(self.store, self.workspace_path, navigation_only=True)
            result = await self._executor().run(
                turn_id=turn_id,
                user_message=task,
                binding_manifest=binding,
                input_files=input_files,
            )
            if (self.memory_config.generate_memories and result.status in {"completed", "failed", "cancelled"}
                    and self.store.read_thread(self.thread_id).parent_thread_id is None):
                self.store.enqueue_memory_job(self.workspace_path, self.thread_id)
            return AgentResult._from_harness(result, store=self.store, files=tuple(files or ()))

    def attach_files(self, files: Sequence[str]) -> tuple[dict[str, object], ...]:
        """Grant selected original files read access for this conversation, without copies."""
        self._ensure_open()
        if self._agent is None:
            raise ValueError("file references require an Agent-configured Session")
        values = self._agent._reference_harness_files(files, permission_ceiling=self.permissions())
        selected = tuple(str(value["path"]) for value in values)
        previous = self.tool_execution_context
        previous_files = self._agent._input_read_files
        self._agent._input_read_files = tuple(dict.fromkeys((*previous_files, *selected)))
        try:
            permissions = previous.permissions
            self.tool_execution_context = replace(previous, permissions=(
                None if permissions is None else permissions.with_read_files(selected)
            ))
            for path in selected:
                if not self.permissions().path_access(path):
                    raise PermissionError(self.permissions().path_reason(path))
            self._persist_settings()
        except BaseException:
            self._agent._input_read_files = previous_files
            self.tool_execution_context = previous
            raise
        self._apply_tool_context()
        return values

    async def steer(self, text: str, *, expected_turn_id: str, files: Sequence[str] | None = None) -> ItemSnapshot:
        """Receive guidance for the named active Turn without starting another loop.

        The returned Item is committed before acknowledgement. Running tools
        finish at their safe boundary; use resume(..., 'abort') after stopping
        the active submit task to cancel work. Text never grants tool approval.
        """
        self._ensure_open()
        def receive() -> ItemSnapshot:
            previous = self.tool_execution_context
            previous_files = () if self._agent is None else self._agent._input_read_files
            try:
                # Settings, input revision and file Items commit together. A rejected
                # input must not leave an unacknowledged permission grant behind.
                with self.store._transaction():
                    if self.store.read_thread(self.thread_id).active_turn_id != expected_turn_id:
                        raise RuntimeError("expected Turn is no longer active; input was not accepted")
                    values = self.attach_files(files) if files else ()
                    return self.store.receive_turn_input(
                        thread_id=self.thread_id, turn_id=expected_turn_id, text=text, input_files=values,
                    )
            except BaseException:
                self.tool_execution_context = previous
                if self._agent is not None:
                    self._agent._input_read_files = previous_files
                self._apply_tool_context()
                raise
        return await self._executor()._commit(receive)

    def input_status(self, item_id: str) -> str:
        """pending/applied means received/dispatched, not model understanding."""
        self._ensure_open()
        item = self.store.read_item(item_id)
        if item.thread_id != self.thread_id or item.kind != "user_message" or "input_revision" not in item.payload:
            raise ValueError("Item is not a steering receipt in this Session")
        return self.store.turn_input_status(item_id)

    def _start_memory_update(self) -> None:
        if not self.memory_config.generate_memories or not self._background_memory:
            return
        self._memory_task = asyncio.create_task(self.refresh_memory())

    async def refresh_memory(self) -> dict[str, Any]:
        """Explicit bounded maintenance; diagnostics include source, version and Turn IDs."""
        from agent_runtime.harness.memory import update_memory

        self._ensure_open()
        if not self.memory_config.generate_memories:
            return {"status": "disabled", "maintenance_turn_ids": []}
        identity = f"turn_{uuid4().hex}"
        binding = self.binding_provider.snapshot(thread_id=self.thread_id, turn_id=identity)
        return await update_memory(self.store, workspace=self.workspace_path, model=self.model,
                                   binding=binding, exclude_thread_id=self.thread_id,
                                   prepare_binding=self._prepare_step_binding)

    @staticmethod
    async def run_memory_turn(
        store: RolloutStore, model: BoundHarnessModel, turn_id: str,
        *, prepare_binding: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ) -> TurnResult:
        """Compose a tool-free maintenance Turn through the same execution owner."""
        from agent_runtime.harness.context import RolloutContextManager

        executor = TurnExecutor(
            thread_id=store.read_turn(turn_id).thread_id, store=store, model=model,
            context_manager=RolloutContextManager(store), completion_gate=_MemoryCompletion(),
            max_steps=1, prepare_binding=prepare_binding,
        )
        return await executor.run_turn(executor.restore_turn_context(turn_id), start_step=1)

    def delete_memory(self, key: str) -> dict[str, Any]:
        """Delete a committed workspace memory; existing Turn snapshots stay unchanged."""
        from agent_runtime.harness.memory import delete_memory

        self._ensure_open()
        return delete_memory(self.store, self.workspace_path, key)

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

    def request_settings(self) -> dict[str, Any]:
        self._ensure_open()
        if self.model_control_plane is None:
            raise RuntimeError("Session has no model control plane")
        return self.model_control_plane.request_settings()

    def configure_request(
        self, *, reasoning_effort: str | None = None, max_output_tokens: int | str | None = None,
    ) -> dict[str, Any]:
        """Persist a user change before the next request captures its settings."""
        self._ensure_open()
        if self.model_control_plane is None:
            raise RuntimeError("Session has no model control plane")
        previous = copy.deepcopy(self.model_control_plane.state)
        settings = self.model_control_plane.configure_request(
            reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens, persist=False,
        )
        try:
            self._persist_settings()
        except BaseException:
            self.model_control_plane.state = previous
            raise
        return settings

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
                "request_preferences": ({} if self.model_control_plane is None else {
                    key: value.model_dump() for key, value in self.model_control_plane.state.request_preferences.items()
                }),
                "request_settings_revision": (0 if self.model_control_plane is None else
                                              self.model_control_plane.state.request_settings_revision),
                "tool_execution_policy": _tool_execution_policy_snapshot(self.tool_execution_context),
                "input_read_files": [] if self._agent is None else list(self._agent._input_read_files),
                "memory_generation_allowed": self.memory_config.generate_memories,
                "memory_use_allowed": self.memory_config.use_memories,
                "memory_control_version": self._memory_control_version,
            },
            expected_request_revision=self._persisted_request_revision,
        )
        self._persisted_request_revision = (0 if self.model_control_plane is None else
                                            self.model_control_plane.state.request_settings_revision)

    def _apply_tool_context(self) -> None:
        if self.tool_orchestrator is not None:
            self.tool_orchestrator.update_execution_context(self.tool_execution_context)

    def permissions(self) -> ResolvedPermissions:
        """Return current effective authority, including administrator ceilings."""
        self._ensure_open()
        return self.tool_execution_context.effective_permissions or PermissionPolicy().resolve(self.workspace_path)

    def update_permissions(self, policy: PermissionPolicy | Path | str) -> None:
        """Explicit user/host control; the model has no tool to widen authority."""
        self._ensure_open()
        if not isinstance(policy, (PermissionPolicy, Path, str)):
            raise TypeError("policy must be PermissionPolicy or a policy file path")
        previous = self.tool_execution_context
        previous_agent_policy = None if self._agent is None else self._agent.permissions
        if self._agent is not None:
            self._agent.permissions = policy
        try:
            from agent_runtime.permission_policy import load_permission_policy

            effective = (self._agent.effective_permissions() if self._agent is not None else
                         policy.resolve(self.workspace_path) if isinstance(policy, PermissionPolicy) else
                         load_permission_policy(policy, self.workspace_path))
            self.tool_execution_context = replace(previous, permissions=effective)
            self._persist_settings()
        except BaseException:
            self.tool_execution_context = previous
            if self._agent is not None:
                self._agent.permissions = previous_agent_policy
            raise
        self._apply_tool_context()

    def _resource_permissions(self) -> ResolvedPermissions:
        # Resident tools are borrowed by child Sessions. The active Step supplies
        # the child's frozen ceiling while the parent supplies live revocations.
        # Resource cleanup still checks revocations after public Session access
        # has closed. It must not call the public open-only permissions method.
        effective = self.tool_execution_context.effective_permissions or PermissionPolicy().resolve(self.workspace_path)
        network_denied = ToolEffect.NETWORK in self.tool_execution_context.deny_effects
        binding = current_tool_binding()
        if binding is not None:
            saved_policy = binding.get("tool_execution_policy", {})
            network_denied = network_denied or ToolEffect.NETWORK.value in saved_policy.get("deny_effects", ())
            saved = saved_policy.get("permissions")
            if saved is not None:
                effective = effective.intersection(ResolvedPermissions.model_validate(saved))
        if network_denied:
            effective = effective.model_copy(update={"network": NetworkPolicy(decision="deny")})
        return effective

    def reload_permissions(self) -> None:
        """Accept newly granted source authority for future calls after user reload."""
        self._ensure_open()
        if self._agent is None:
            raise ValueError("reload requires an Agent-configured Session")
        previous = self.tool_execution_context
        self.tool_execution_context = replace(previous, permissions=self._agent.effective_permissions())
        try:
            self._persist_settings()
        except BaseException:
            self.tool_execution_context = previous
            raise
        self._apply_tool_context()

    def update_tool_policy(
        self,
        *,
        allow_write_tools: bool | None = None,
        allow_execute_tools: bool | None = None,
        allow_web_tools: bool | None = None,
        require_confirmation_for: frozenset[str] | None = None,
        denied_tool_names: frozenset[str] | None = None,
        deny_effects: frozenset[ToolEffect] | None = None,
        auto_approve_sandboxed: bool | None = None,
        approval_mode: ApprovalMode | None = None,
    ) -> None:
        self._ensure_open()
        updates: dict[str, Any] = {
            name: value
            for name, value in {
                "allow_write_tools": allow_write_tools,
                "allow_execute_tools": allow_execute_tools,
                "allow_web_tools": allow_web_tools,
                "require_confirmation_for": require_confirmation_for,
                "denied_tool_names": denied_tool_names,
                "deny_effects": deny_effects,
                "auto_approve_sandboxed": auto_approve_sandboxed,
                "approval_mode": approval_mode,
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

    def approval_grants(self) -> tuple[CommandGrant, ...]:
        """Current process-local grants; never restored from a checkpoint."""
        self._ensure_open()
        return self.tool_execution_context.command_grants

    def revoke_approval_grants(self) -> None:
        self._ensure_open()
        self.tool_execution_context = replace(self.tool_execution_context, command_grants=())
        self._apply_tool_context()

    def _add_approval_grant(self, grant: CommandGrant) -> None:
        grants = self.tool_execution_context.command_grants
        if grant not in grants:
            self.tool_execution_context = replace(self.tool_execution_context, command_grants=(*grants, grant))
            self._apply_tool_context()

    async def _open_agent_resources(
        self,
        agent: Agent,
        *,
        allow_write_tools: bool,
        allow_execute_tools: bool,
        allow_web_tools: bool,
        initial_tool_context: ToolExecutionContext | None = None,
        saved_tool_policy: Mapping[str, Any] | None = None,
        discovery_ceiling: Mapping[str, Any] | None = None,
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
        # Discovery can invoke permission providers before service assembly.
        self.tool_execution_context = ToolExecutionContext(
            workspace_root=workspace.root, cwd=workspace.root,
            allow_write_tools=allow_write_tools, allow_execute_tools=allow_execute_tools,
            allow_web_tools=allow_web_tools, permissions=agent.effective_permissions(),
            permission_provider=agent.effective_permissions,
        )
        if initial_tool_context is not None:
            if (initial_tool_context.workspace_root is not None
                    and Path(initial_tool_context.workspace_root).resolve() != workspace.root):
                raise ValueError("tool context belongs to a different workspace security domain")
            current = self.tool_execution_context.effective_permissions
            assert current is not None
            supplied = initial_tool_context.effective_permissions
            def current_permissions() -> ResolvedPermissions:
                live = agent.effective_permissions()
                supplied_live = initial_tool_context.effective_permissions
                return live if supplied_live is None else live.intersection(supplied_live)

            self.tool_execution_context = replace(
                initial_tool_context, workspace_root=workspace.root,
                permissions=current if supplied is None else current.intersection(supplied),
                permission_provider=current_permissions,
            )
        elif saved_tool_policy is not None:
            self.tool_execution_context = restore_tool_policy(
                self.tool_execution_context, saved_tool_policy, restore_session_grants=True,
            )
        workspace.permission_provider = self._resource_permissions
        self.model_control_plane = agent._get_model_control_plane()
        self._stack.push_async_callback(
            _close_owned_sync_resource,
            self.model_control_plane,
            label="model control plane",
        )
        self.model_control_plane.ensure_model_binding_trust(
            has_existing_bindings=any(
                "authentication_schema_version" in turn.binding_manifest for turn in self.store.list_turns()
            ),
        )

        def acknowledge_plan_update(_arguments: object) -> dict[str, object]:
            return {
                "accepted": True,
                "revision": 0,
                "message": "Plan update recorded as a ToolResult.",
            }

        from agent_runtime.tools.builtins.command_sessions import DurableCommandSessions

        command_sessions = DurableCommandSessions(workspace)
        self._command_sessions = command_sessions

        async def close_command_sessions(*, close: bool = True) -> None:
            from agent_runtime.tools.builtins.shell import _normalize_command_output
            outcomes = await (command_sessions.close_results() if close else command_sessions.cancel_results())
            for continuation, output in outcomes:
                if self.tool_orchestrator is not None:
                    normalized = _normalize_command_output(output)
                    await self.tool_orchestrator.close_continuation(continuation, {
                        "structured_content": output.model_dump(mode="json"),
                        "is_error": normalized.is_error,
                        "error_code": normalized.error_code,
                        "error_message": normalized.error_message,
                    })

        async def cancel_live_commands() -> None:
            await close_command_sessions(close=False)

        self._cancel_live_commands = cancel_live_commands
        self._stack.push_async_callback(close_command_sessions)
        resident = create_resident_coding_tools(
            workspace,
            plan_updater=acknowledge_plan_update,
            command_sessions=command_sessions,
        )
        from agent_runtime.harness.tool_orchestrator import current_tool_turn_id
        from agent_runtime.tools.builtins.web import WEB_SOURCE_MEDIA_TYPE, create_web_tools, load_search_key
        from agent_runtime.tools.web_http import PublicWebClient, PublicWebError
        from agent_runtime.tools.web_limits import WebLimits

        def authorize_web_url(url: str) -> None:
            if self._resource_permissions().network_decision(url) == "deny":
                raise PublicWebError("network_policy_denied", "destination denied by current permission policy")

        web_client = (PublicWebClient(proxy_url=agent.web_proxy_url, no_proxy=agent.web_no_proxy,
                                     authorize_url=authorize_web_url)
                      if agent.web_proxy_url is not None else PublicWebClient(authorize_url=authorize_web_url))
        self._stack.push_async_callback(web_client.aclose)

        def save_web_source(content: bytes) -> str:
            turn_id = current_tool_turn_id()
            if turn_id is None:
                raise RuntimeError("web source requires an active tool Turn")
            return self.store.commit_artifact(
                turn_id=turn_id, content=content, media_type=WEB_SOURCE_MEDIA_TYPE, name="web-source.json",
            ).artifact_id

        def load_web_source(source_id: str) -> bytes:
            artifact = self.store.read_artifact_metadata(source_id)
            thread = self.store.read_thread(artifact.thread_id)
            if artifact.media_type != WEB_SOURCE_MEDIA_TYPE or Path(thread.workspace).resolve() != workspace.root:
                raise KeyError("web source belongs to another security domain")
            return self.store.read_artifact(source_id)

        def current_web_limits() -> WebLimits:
            turn = self.store.read_turn(current_tool_turn_id())
            return WebLimits.model_validate(turn.binding_manifest.get("web_limits", {}))

        def persisted_web_attempts() -> int:
            turn_id = current_tool_turn_id()
            # New Turns bind WebLimits and use durable unit reservations from
            # their first request. Only pre-upgrade Turns need legacy counting.
            if "web_limits" in self.store.read_turn(turn_id).binding_manifest:
                return 0
            attempts = 0
            for operation in self.store.list_tool_operations(turn_id):
                if (operation.tool_name not in {"web_search", "web_fetch"}
                        or "network" not in operation.effects or not operation.attempt_count):
                    continue
                if operation.result_item_id:
                    payload = self.store.read_item(operation.result_item_id).payload
                    metadata = payload.get("metadata", {})
                    if (metadata.get("web_cache_hit") is True
                            or metadata.get("web_request_attempted") is False):
                        # Only the last attempt is known not to have reached the
                        # network; earlier recovered attempts remain unknown.
                        attempts += max(0, operation.attempt_count - 1)
                        continue
                # Current running attempts have not reserved network yet. Previous
                # attempts and unknown outcomes after recovery count conservatively.
                attempts += (max(0, operation.attempt_count - 1)
                             if operation.status == "running" and operation.claim_owner == self.worker_id
                             else operation.attempt_count)
            return attempts

        def web_budget_status() -> dict[str, Any]:
            limit = current_web_limits().turn_requests
            usage = self.store.read_resource_usage(
                current_tool_turn_id(), "public_web_requests", initial_used=persisted_web_attempts(),
            )
            attempts = usage["used"]
            return {"limit": limit, "used": min(attempts, limit), "remaining": max(0, limit - attempts),
                    "unit": "network_tool_attempts", "unknown_attempts_counted": True,
                    "last_reservation_item_id": usage["item_id"]}

        def check_web_budget() -> None:
            limit = current_web_limits().turn_requests
            try:
                # Commit a unit before network I/O; the SQLite transaction also
                # serializes separate Sessions/processes sharing this Turn.
                self.store.reserve_resource_unit(
                    current_tool_turn_id(), "public_web_requests", limit=limit,
                    initial_used=persisted_web_attempts(),
                )
            except ValueError as error:
                if str(error) != "resource_limit_exceeded":
                    raise
                raise PublicWebError(
                    "web_request_budget_exceeded", f"This Turn permits {limit} web request attempts.",
                ) from None

        def saved_fetch_failure(url: str, mode: str) -> dict[str, Any] | None:
            from agent_runtime.tools.builtins.web import WebFetchInput

            items = self.store.list_items(current_tool_turn_id())
            arguments = {
                item.payload.get("tool_call_id"): item.payload.get("arguments")
                for item in items if item.kind == "tool_call" and item.payload.get("tool_name") == "web_fetch"
            }
            for item in reversed(items):
                if item.kind != "tool_result" or item.payload.get("tool_name") != "web_fetch":
                    continue
                prior_args = arguments.get(item.payload.get("tool_call_id"))
                if not isinstance(prior_args, dict):
                    continue
                try:
                    request = WebFetchInput.model_validate(prior_args)
                except ValueError:
                    # Rejected input is not acquisition evidence and must not
                    # prevent unrelated valid requests after continuation.
                    continue
                prior_mode = ("auto" if request.render == "auto" and request.view != "raw" else
                              "browser" if request.render is True else "http")
                if request.url != url or prior_mode != mode:
                    continue
                output = item.payload.get("structured_content")
                # The newest acquisition supersedes older failure evidence, including explicit refresh.
                if not isinstance(output, dict) or not output.get("error_code"):
                    return None
                return {**output, "failure_origin": {"item_id": item.item_id,
                                                     "tool_call_id": str(item.payload.get("tool_call_id"))}}
            return None

        web_tools = create_web_tools(
            web_client, save_source=save_web_source, load_source=load_web_source,
            search_key=load_search_key(agent.web_search_key_file, workspace=workspace.root),
            search_provider=agent.web_search_provider,
            before_request=check_web_budget, cache_scope=current_tool_turn_id,
            resource_limits=current_web_limits, budget_status=web_budget_status,
            failure_lookup=saved_fetch_failure,
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

        startup_ceiling = None if discovery_ceiling is None else ResolvedPermissions.model_validate(discovery_ceiling)

        def mcp_permissions() -> ResolvedPermissions:
            current = self._resource_permissions()
            return current if startup_ceiling is None else current.intersection(startup_ceiling)

        policy = mcp_permissions()
        mcp_allowed = policy.tool_decision("mcp__*") != "deny"
        config_path = resolve_product_mcp_config(workspace.root) if agent.enable_workspace_mcp and mcp_allowed else None
        mcp_tools: tuple[Tool, ...] = ()
        mcp_trust_binding: Mapping[str, object] | None = None
        if config_path is not None:
            trust = agent.mcp_config_trust or decide_mcp_config_trust(
                config_path,
                workspace_root=workspace.root,
                trust_workspace=False,
            )
            mcp_tools = await self._stack.enter_async_context(open_trusted_product_mcp_tools(
                config_path, trust=trust, permissions=policy, permission_provider=mcp_permissions,
                lazy=True,
            ))
            from agent_runtime.runtime.mcp_catalog import catalog_revision
            self._mcp_config_path = config_path
            self._mcp_catalog_revision = catalog_revision(config_path)
            mcp_trust_binding = {
                "config_path": str(trust.config_path),
                "config_source": trust.source,
                "config_sha256": trust.config_sha256,
            }
        tools = {tool.definition.name: tool for tool in (*resident, *web_tools, *mcp_tools)}

        from agent_runtime.builtin.generic import coding_instructions
        from agent_runtime.harness.model_adapter import ControlPlaneHarnessModel

        override = agent._model_factory
        model = (
            override()
            if callable(override)
            else ControlPlaneHarnessModel(
                control_plane=self.model_control_plane,
                instructions=coding_instructions(workspace.root),
            )
        )
        return model, dict(
            tools=tools,
            tool_execution_context=self.tool_execution_context,
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
            self.model.ensure_available(binding, thread_id=binding["thread_id"], turn_id=binding["turn_id"])
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
                if self._cancel_live_commands is not None:
                    await self._cancel_live_commands()
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
            recoverable_summary = (
                turn.status == "running"
                and bool(model_operations)
                and model_operations[-1].status == "completed"
                and model_operations[-1].request_ref.get("purpose") == "context_summary"
                and model_operations[-1].response_item_id is not None
            )
            recoverable_committed_response = False
            if (
                turn.status == "running"
                and model_operations
                and model_operations[-1].status == "completed"
                and model_operations[-1].response_item_id is not None
            ):
                response = self.store.read_item(model_operations[-1].response_item_id)
                recoverable_committed_response = response.kind == "model_response"
            if len(pending) == 1 and pending[0].kind == "tool_approval":
                decision = {
                    "allow_once": "approve",
                    "approve": "approve",
                    "allow_session": "approve",
                    "deny": "deny",
                }.get(action)
                if (decision is None and action == "continue" and model_operations
                        and self.store.request_input_revision(turn_id, model_operations[-1].request_ref) is not None
                        and self.store.user_input_revision(turn_id)
                        != self.store.request_input_revision(turn_id, model_operations[-1].request_ref)):
                    decision = "deny"
                if decision is None:
                    raise ValueError("tool approval action must be allow_once, approve, allow_session, or deny")
                internal = await executor.resume(
                    turn_id=turn_id,
                    decision=decision,
                    on_session_grant=self._add_approval_grant if action == "allow_session" else None,
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
            elif (
                not pending
                and turn.status == "running"
                and model_operations
                and model_operations[-1].status == "prepared"
                and action in {"continue", "retry"}
            ):
                internal = await executor.recover_prepared_model(turn_id=turn_id)
            elif not pending and executor.pending_context_overflow(turn_id) and action in {"continue", "retry"}:
                internal = await executor.run_turn(
                    executor.restore_turn_context(turn_id),
                    start_step=executor.agent_step_count(turn_id),
                )
            elif not pending and recoverable_summary and action in {"continue", "retry"}:
                internal = await executor.recover_committed_model_response(turn_id=turn_id)
            elif (not pending and turn.status == "running" and action in {"continue", "retry"}
                  and executor.pending_input_supersession(turn_id)):
                internal = await executor.run_turn(
                    executor.restore_turn_context(turn_id), start_step=executor.agent_step_count(turn_id) + 1,
                )
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
            from agent_runtime.tools.web_limits import WebLimits

            web_limits = source_binding.get("web_limits")
            if web_limits is None and parent_turn_id is not None:
                web_limits = self.store.read_turn(parent_turn_id).binding_manifest.get("web_limits")
            if web_limits is None and self._agent is not None:
                web_limits = self._agent.web_limits.model_dump(mode="json")
            binding["web_limits"] = WebLimits.model_validate(web_limits or {}).model_dump(mode="json")
            policy_value = source_binding["tool_execution_policy"]
            if not isinstance(policy_value, Mapping):
                raise RuntimeError("source Step has no tool execution policy")
            binding["tool_execution_policy"] = dict(policy_value)
            child_options = {
                **self._service_options,
                "model_binding": source_binding,
                "tool_execution_context": replace(restore_tool_policy(
                    self.tool_execution_context, policy_value, restore_session_grants=True,
                ), command_grants=()),
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

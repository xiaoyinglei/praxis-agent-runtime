"""Session tool assembly and request binding snapshots."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from agent_runtime.checkpoints import CheckpointService
from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness.context import RolloutContextManager
from agent_runtime.harness.context_recall import READ_CONTEXT_NAME, create_context_recall_tool
from agent_runtime.harness.execution_context import step_limit_from_binding
from agent_runtime.harness.protocol import BoundHarnessModel, CompletionGate
from agent_runtime.harness.retry_policy import RetryPolicy
from agent_runtime.harness.stop import StopChecks
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator, current_tool_turn_id
from agent_runtime.harness.tool_router import DurableToolRouter
from agent_runtime.project_instructions import instruction_snapshot
from agent_runtime.tools.integrations.knowledge import (
    KnowledgeSearchInput,
    create_knowledge_tools,
)
from agent_runtime.tools.integrations.skills import create_skill_tools
from agent_runtime.tools.integrations.subagent import (
    ChildHandleInput,
    IntegrateAgentInput,
    SendAgentMessageInput,
    StandardSubagentOutput,
    SubagentInput,
    WaitAgentInput,
    create_async_subagent_tools,
    create_subagent_tool,
)
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.permissions import restore_tool_policy as restore_tool_policy
from agent_runtime.tools.registry import build_tool_registry
from agent_runtime.tools.selection import create_find_tools_tool, find_tools
from agent_runtime.tools.tool import JsonValue, Tool
from agent_runtime.workspace import open_workspace


class HarnessSkillRuntime(Protocol):
    @property
    def catalog_revision(self) -> str: ...

    @property
    def model_invocable_skill_ids(self) -> tuple[str, ...]: ...

    def invoke_skill(
        self,
        arguments: Mapping[str, object],
    ) -> dict[str, object]: ...

    def skill_root(self, skill_id: str) -> Path | None: ...


def _tool_execution_policy_snapshot(
    context: ToolExecutionContext,
) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "allow_write_tools": context.allow_write_tools,
        "allow_execute_tools": context.allow_execute_tools,
        "allow_web_tools": context.allow_web_tools,
        "active_skill_ids": sorted(context.active_skill_ids),
        "deny_effects": sorted(effect.value for effect in context.deny_effects),
        "max_parallel_calls": context.max_parallel_calls,
        "require_confirmation_for": sorted(context.require_confirmation_for),
        "denied_tool_names": sorted(context.denied_tool_names),
        "auto_approve_sandboxed": context.auto_approve_sandboxed,
        "approval_mode": context.approval_mode,
    }
    if context.effective_permissions is not None:
        snapshot["permissions"] = context.effective_permissions.model_dump(mode="json")
    return snapshot


class SessionBindingProvider:
    def __init__(
        self,
        *,
        model: BoundHarnessModel,
        tools: Mapping[str, Tool],
        knowledge_revision: str | None,
        knowledge_config: Mapping[str, object] | None,
        tool_execution_policy: Callable[[], Mapping[str, object]],
        workspace_mcp_enabled: bool,
        mcp_config_trust: Mapping[str, object] | None,
        require_workspace_change: bool,
        acceptance_command: str | None,
        model_step_budget: int | None,
        model_token_budget_total: int | None,
        model_cost_budget_total_micros: int | None,
        project_instructions: Callable[[str], Mapping[str, Any] | None] | None = None,
        model_binding: Mapping[str, Any] | None = None,
        request_settings: Callable[[], Mapping[str, Any]] | None = None,
        subagent_limits: Mapping[str, int] | None = None,
        delivery_tools_enabled: bool = False,
    ) -> None:
        from agent_runtime.harness.subagents import child_limits

        self._subagent_limits = child_limits(subagent_limits)
        self._delivery_tools_enabled = delivery_tools_enabled
        self._request_settings = request_settings
        self._model = model
        self._model_binding = model_binding
        self._tools = tuple(tools.values())
        self._knowledge_revision = knowledge_revision
        self._knowledge_config = None if knowledge_config is None else dict(knowledge_config)
        self._tool_execution_policy = tool_execution_policy
        self._workspace_mcp_enabled = workspace_mcp_enabled
        self._mcp_config_trust = None if mcp_config_trust is None else dict(mcp_config_trust)
        self._require_workspace_change = require_workspace_change
        self._acceptance_command = acceptance_command
        self._model_step_budget = model_step_budget
        self._model_token_budget_total = model_token_budget_total
        self._model_cost_budget_total_micros = model_cost_budget_total_micros
        self._project_instructions = project_instructions

    def snapshot(
        self,
        *,
        thread_id: str,
        turn_id: str,
        model_binding: Mapping[str, Any] | None = None,
    ) -> Mapping[str, object]:
        source = self._model_binding if model_binding is None else model_binding
        if source is None:
            binding = dict(self._model.snapshot(thread_id=thread_id, turn_id=turn_id))
        elif "authentication_schema_version" in source:
            rebind = getattr(self._model, "rebind", None)
            if not callable(rebind):
                raise RuntimeError("authenticated model cannot bind a child Session")
            binding = dict(rebind(source, thread_id=thread_id, turn_id=turn_id))
        else:
            binding = {**source, "thread_id": thread_id, "turn_id": turn_id}
        for field_name, expected in (("thread_id", thread_id), ("turn_id", turn_id)):
            observed = binding.get(field_name)
            if observed is not None and observed != expected:
                raise RuntimeError(f"model binding returned the wrong {field_name}")
            binding[field_name] = expected
        binding.update(self._static_snapshot(turn_id))
        if self._request_settings is not None:
            binding["request_settings"] = dict(self._request_settings())
        return binding

    def _static_snapshot(self, turn_id: str) -> dict[str, object]:
        binding: dict[str, object] = {}
        binding["subagent_limits"] = dict(self._subagent_limits)
        binding["toolset_revision"] = toolset_revision_for_tools(self._tools)
        binding["tool_execution_revisions"] = {tool.definition.name: tool.execution_revision for tool in self._tools}
        binding["tool_execution_policy"] = dict(self._tool_execution_policy())
        mcp_policy: dict[str, object] = {"workspace_discovery_enabled": self._workspace_mcp_enabled}
        if self._mcp_config_trust is not None:
            mcp_policy.update(self._mcp_config_trust)
        binding["mcp_policy"] = mcp_policy
        completion_policy: dict[str, object] = {
            "revision": "completion-v2",
            "require_workspace_change": self._require_workspace_change,
            "acceptance_command": self._acceptance_command,
        }
        binding["completion_policy"] = completion_policy
        binding["delivery_policy"] = {"revision": "delivery-v1", "model_tools": self._delivery_tools_enabled}
        binding["model_retry_policy"] = RetryPolicy().to_payload()
        binding["model_response_recovery"] = {"revision": "restart-step-v1", "max_retries": 1}
        binding["tool_failure_recovery"] = {"revision": "one-replan-v1"}
        binding["completion_continuation_policy"] = {"revision": "bounded-v1", "max_continuations": 8}
        binding["execution_context_policy"] = {
            "revision": "working-context-v2",
            "cache_continuity": "append-v1",
            "context_updates": "append-v1",
            "tool_result_order": "call-v1",
        }
        binding["model_step_budget"] = self._model_step_budget
        binding["model_token_budget_total"] = self._model_token_budget_total
        binding["model_cost_budget_total_micros"] = self._model_cost_budget_total_micros
        if self._project_instructions is not None:
            instructions = self._project_instructions(turn_id)
            if instructions is not None:
                binding["project_instructions"] = dict(instructions)
        if self._knowledge_revision is not None:
            binding["knowledge_revision"] = self._knowledge_revision
            binding["knowledge_config"] = self._knowledge_config
        return binding


if TYPE_CHECKING:
    from agent_runtime.harness.session import Session


def _memory_tools(session: Session) -> tuple[Tool, ...]:
    if not session.memory_config.use_memories:
        return ()
    from agent_runtime.harness.memory import create_memory_tool
    from agent_runtime.harness.memory_management import create_memory_management_tool

    return (create_memory_tool(session.store), create_memory_management_tool(session.store))


def configure_services(
    session: Session,
    *,
    workspace: Path,
    model: BoundHarnessModel,
    model_binding: Mapping[str, Any] | None = None,
    completion_gate: CompletionGate | None = None,
    require_workspace_change: bool = False,
    acceptance_command: str | None = None,
    tools: Mapping[str, Tool] | None = None,
    tool_execution_context: ToolExecutionContext | None = None,
    knowledge_runner: Callable[..., object] | None = None,
    knowledge_revision: str | None = None,
    knowledge_config: Mapping[str, object] | None = None,
    discoverable_tool_names: tuple[str, ...] = (),
    workspace_mcp_enabled: bool = True,
    mcp_config_trust: Mapping[str, object] | None = None,
    enable_subagents: bool = False,
    enable_delivery_tools: bool = False,
    subagent_limits: Mapping[str, int] | None = None,
    skill_runtime: HarnessSkillRuntime | None = None,
    max_steps: int | None = None,
    max_tokens_total: int | None = None,
    max_cost_micros: int | None = None,
) -> None:
    if acceptance_command is not None:
        if not isinstance(acceptance_command, str) or not acceptance_command.strip():
            raise ValueError("acceptance_command must be a non-empty string or None")
        acceptance_command = acceptance_command.strip()
    step_limit_from_binding({"model_step_budget": max_steps})
    if max_tokens_total is not None and (
        isinstance(max_tokens_total, bool) or not isinstance(max_tokens_total, int) or max_tokens_total < 1
    ):
        raise ValueError("max_tokens_total must be a positive integer or None")
    if max_cost_micros is not None and (
        isinstance(max_cost_micros, bool) or not isinstance(max_cost_micros, int) or max_cost_micros < 1
    ):
        raise ValueError("max_cost_micros must be a positive integer or None")
    if (knowledge_runner is None) != (knowledge_revision is None):
        raise ValueError("knowledge_runner and knowledge_revision must be configured together")
    if (knowledge_revision is None) != (knowledge_config is None):
        raise ValueError("knowledge_config must be configured with knowledge_revision")
    if knowledge_revision is not None and not knowledge_revision.strip():
        raise ValueError("knowledge_revision must be non-empty")
    execution_context = tool_execution_context or ToolExecutionContext(
        workspace_root=workspace,
        cwd=workspace,
    )
    store = session.store
    from agent_runtime.harness.subagents import child_limits

    session._subagent_limits = child_limits(subagent_limits)

    async def run_subagent(arguments: Mapping[str, JsonValue]) -> object:
        payload = SubagentInput.model_validate(arguments)
        child_task = payload.task
        if payload.context_summary:
            child_task += "\n\nContext supplied by the parent agent:\n" + payload.context_summary
        child = await session.run_child(
            parent_turn_id=current_tool_turn_id(),
            user_message=child_task,
            max_steps=payload.max_turns,
            max_tokens_total=payload.llm_budget_total,
            max_cost_micros=payload.llm_cost_budget_micros,
            workspace_mode=payload.workspace_mode,
        )
        return await store.run_async(session.subagents._output, current_tool_turn_id(), child.turn_id)

    async def spawn_agent(arguments: Mapping[str, JsonValue]) -> object:
        payload = SubagentInput.model_validate(arguments)
        message = payload.task
        if payload.context_summary:
            message += "\n\nContext supplied by the parent agent:\n" + payload.context_summary
        return await session.subagents.spawn(
            parent_turn_id=current_tool_turn_id(),
            user_message=message,
            max_steps=payload.max_turns,
            max_tokens_total=payload.llm_budget_total,
            max_cost_micros=payload.llm_cost_budget_micros,
            workspace_mode=payload.workspace_mode,
        )

    async def send_agent_message(arguments: Mapping[str, JsonValue]) -> object:
        payload = SendAgentMessageInput.model_validate(arguments)
        return await session.subagents.send(
            parent_turn_id=current_tool_turn_id(),
            child_turn_id=payload.child_turn_id,
            message=payload.message,
            message_id=payload.message_id,
        )

    async def wait_agent(arguments: Mapping[str, JsonValue]) -> object:
        payload = WaitAgentInput.model_validate(arguments)
        return await session.subagents.wait(
            parent_turn_id=current_tool_turn_id(),
            child_turn_id=payload.child_turn_id,
            timeout_seconds=payload.timeout_seconds,
        )

    async def cancel_agent(arguments: Mapping[str, JsonValue]) -> object:
        payload = ChildHandleInput.model_validate(arguments)
        return await session.subagents.cancel(
            parent_turn_id=current_tool_turn_id(), child_turn_id=payload.child_turn_id
        )

    async def integrate_agent(arguments: Mapping[str, JsonValue]) -> object:
        payload = IntegrateAgentInput.model_validate(arguments)
        return await session.subagents.integrate(
            parent_turn_id=current_tool_turn_id(), child_turn_id=payload.child_turn_id,
            action=payload.action, artifact_id=payload.artifact_id,
            verification_tool_call_id=payload.verification_tool_call_id,
            resolution_note=payload.resolution_note,
            diff_offset=payload.diff_offset, diff_limit=payload.diff_limit,
        )

    async def search_knowledge(arguments: Mapping[str, JsonValue]) -> object:
        if knowledge_runner is None:
            raise RuntimeError("knowledge search is not configured")
        payload = KnowledgeSearchInput.model_validate(arguments)
        result = knowledge_runner(
            payload,
            execution_context=execution_context,
        )
        return await result if inspect.isawaitable(result) else result

    knowledge_tools = create_knowledge_tools(
        search_knowledge if knowledge_runner is not None else None,
        execution_revision=knowledge_revision or "unconfigured",
    )
    skill_tools = (
        create_skill_tools(
            open_workspace(workspace),
            invoke_skill=skill_runtime.invoke_skill,
            active_skill_root=skill_runtime.skill_root,
            invoke_execution_revision=skill_runtime.catalog_revision,
            available_skill_ids=skill_runtime.model_invocable_skill_ids,
        )
        if skill_runtime is not None
        else ()
    )
    from agent_runtime.harness.delivery import create_delivery_tool

    registry = build_tool_registry(
        tuple((tools or {}).values()),
        (create_delivery_tool(session.delivery),) if enable_delivery_tools else (),
        create_context_recall_tool(store),
        _memory_tools(session),
        knowledge_tools,
        skill_tools,
        (
            create_subagent_tool(
                run_subagent,
                execution_revision="harness-child-thread-v5",
                output_model=StandardSubagentOutput,
            ),
            *create_async_subagent_tools(
                spawn=spawn_agent, send=send_agent_message, wait=wait_agent, cancel=cancel_agent,
                integrate=integrate_agent, workspace=workspace,
            ),
        )
        if enable_subagents
        else (),
    )
    discoverable_names = (
        *tuple(discoverable_tool_names),
        *(
            ("task", "spawn_agent", "send_agent_message", "wait_agent", "cancel_agent", "integrate_agent")
            if enable_subagents else ()
        ),
    )
    if READ_CONTEXT_NAME in discoverable_names:
        raise ValueError("read_context is a resident tool owned by the composition root")
    if len(set(discoverable_names)) != len(discoverable_names):
        raise ValueError("discoverable tool names must be unique")
    installed_names = tuple(tool.definition.name for tool in registry.list_all())
    missing = tuple(name for name in discoverable_names if name not in installed_names)
    if missing:
        raise ValueError(f"discoverable tools are not installed: {missing}")
    if "find_tools" in installed_names or "find_tools" in discoverable_names:
        raise ValueError("find_tools is owned by the composition root")
    resident_names = tuple(name for name in installed_names if name not in set(discoverable_names))
    if discoverable_names:
        snapshot_ref: dict[str, Mapping[str, Tool]] = {}

        def search_hidden_tools(query: str, limit: int) -> object:
            return find_tools(
                snapshot_ref["tools"],
                query=query,
                discoverable_names=discoverable_names,
                resident_names=("find_tools", *resident_names),
                limit=limit,
            )

        registry.register(
            create_find_tools_tool(
                search_hidden_tools,
                execution_revision=toolset_revision_for_tools(registry.list_all()),
            )
        )
    tool_snapshot = registry.freeze()
    if discoverable_names:
        snapshot_ref["tools"] = tool_snapshot
    context_manager = RolloutContextManager(store)
    worker_id = f"worker_{uuid4().hex}"
    tool_router = DurableToolRouter(
        store=store,
        tools=tool_snapshot,
        resident_names=(
            *(("find_tools",) if discoverable_names else ()),
            *resident_names,
        ),
        discoverable_names=discoverable_names,
    )
    resolved_completion_gate = completion_gate or StopChecks(store)

    from agent_runtime.workspace_snapshot import runtime_excluded_paths

    execution_context = replace(
        execution_context,
        context_provider=execution_context.context_provider or (lambda: session.tool_execution_context),
        snapshot_excluded_paths=runtime_excluded_paths(session.database),
    )
    session.tool_orchestrator = ToolOrchestrator(
        store=store,
        tools=tool_snapshot,
        execution_context=execution_context,
        worker_id=worker_id,
        checkpoints=CheckpointService(workspace),
        dispatch_guard=session._guard_dispatch,
    )
    session.tool_execution_context = execution_context
    session.model = model
    session.context_manager = context_manager
    session.completion_gate = resolved_completion_gate
    session.tool_router = tool_router
    session.worker_id = worker_id
    session.max_steps = max_steps
    session.tools = tool_snapshot

    def project_instructions_for_turn(turn_id: str) -> Mapping[str, Any] | None:
        try:
            return store.read_turn(turn_id).binding_manifest.get("project_instructions")
        except KeyError:
            from agent_runtime.project_instructions import extend_instructions, restored_instructions

            snapshot = instruction_snapshot(session.tool_execution_context)
            thread = store.read_thread(session.thread_id)
            previous = thread.head_turn_id or thread.fork_turn_id
            if previous is not None:
                prior = restored_instructions(store.read_turn(previous).binding_manifest, store.list_items(previous))
                if prior is not None and prior["root"] == snapshot["root"]:
                    # Refresh known scopes once for a new Turn. Do not silently
                    # drop nested guidance when only the root was bootstrapped.
                    snapshot = extend_instructions(
                        snapshot,
                        session.tool_execution_context,
                        (Path(snapshot["root"]) / scope / "AGENTS.md" for scope in prior["directories"]),
                    )
            return snapshot

    session.binding_provider = SessionBindingProvider(
        model=model,
        model_binding=model_binding,
        tools=tool_snapshot,
        knowledge_revision=knowledge_revision,
        knowledge_config=knowledge_config,
        tool_execution_policy=lambda: _tool_execution_policy_snapshot(session.tool_execution_context),
        workspace_mcp_enabled=workspace_mcp_enabled,
        mcp_config_trust=mcp_config_trust,
        require_workspace_change=require_workspace_change,
        acceptance_command=acceptance_command,
        model_step_budget=max_steps,
        model_token_budget_total=max_tokens_total,
        model_cost_budget_total_micros=max_cost_micros,
        subagent_limits=session._subagent_limits,
        delivery_tools_enabled=enable_delivery_tools,
        project_instructions=project_instructions_for_turn,
        request_settings=(
            None if session.model_control_plane is None else session.model_control_plane.request_settings
        ),
    )

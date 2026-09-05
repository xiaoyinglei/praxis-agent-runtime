"""Resource-only composition root for the replacement Harness."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness.completion import DeliveryCompletionGate
from agent_runtime.harness.context import RolloutContextManager
from agent_runtime.harness.protocol import BoundHarnessModel, CompletionGate
from agent_runtime.harness.rollout import RolloutStore
from agent_runtime.harness.session import Session
from agent_runtime.harness.thread_manager import ThreadManager
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator, current_tool_turn_id
from agent_runtime.harness.tool_router import DurableToolRouter
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.tools.integrations.knowledge import (
    KnowledgeSearchInput,
    create_knowledge_tools,
)
from agent_runtime.tools.integrations.skills import create_skill_tools
from agent_runtime.tools.integrations.subagent import (
    SubagentInput,
    create_subagent_tool,
)
from agent_runtime.tools.permissions import ToolExecutionContext
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
    return {
        "allow_write_tools": context.allow_write_tools,
        "allow_execute_tools": context.allow_execute_tools,
        "active_skill_ids": sorted(context.active_skill_ids),
        "deny_effects": sorted(effect.value for effect in context.deny_effects),
        "max_parallel_calls": context.max_parallel_calls,
        "require_confirmation_for": sorted(context.require_confirmation_for),
        "denied_tool_names": sorted(context.denied_tool_names),
        "auto_approve_sandboxed": context.auto_approve_sandboxed,
    }


class _CompositionBindingProvider:
    def __init__(
        self,
        *,
        model: BoundHarnessModel,
        tools: Mapping[str, Tool],
        knowledge_revision: str | None,
        knowledge_config: Mapping[str, object] | None,
        tool_execution_policy: Mapping[str, object],
        workspace_mcp_enabled: bool,
        require_workspace_change: bool,
        model_step_budget: int,
        model_token_budget_total: int | None,
        model_cost_budget_total_micros: int | None,
    ) -> None:
        self._model = model
        self._tools = tuple(tools.values())
        self._knowledge_revision = knowledge_revision
        self._knowledge_config = None if knowledge_config is None else dict(knowledge_config)
        self._tool_execution_policy = dict(tool_execution_policy)
        self._workspace_mcp_enabled = workspace_mcp_enabled
        self._require_workspace_change = require_workspace_change
        self._model_step_budget = model_step_budget
        self._model_token_budget_total = model_token_budget_total
        self._model_cost_budget_total_micros = model_cost_budget_total_micros

    def snapshot(self, *, thread_id: str, turn_id: str) -> Mapping[str, object]:
        binding = dict(
            self._model.snapshot(
                thread_id=thread_id,
                turn_id=turn_id,
            )
        )
        for field_name, expected in (("thread_id", thread_id), ("turn_id", turn_id)):
            observed = binding.get(field_name)
            if observed is not None and observed != expected:
                raise RuntimeError(f"model binding returned the wrong {field_name}")
            binding[field_name] = expected
        binding.update(self._static_snapshot())
        return binding

    def _static_snapshot(self) -> dict[str, object]:
        binding: dict[str, object] = {}
        binding["toolset_revision"] = toolset_revision_for_tools(self._tools)
        binding["tool_execution_revisions"] = {tool.definition.name: tool.execution_revision for tool in self._tools}
        binding["tool_execution_policy"] = self._tool_execution_policy
        binding["mcp_policy"] = {"workspace_discovery_enabled": self._workspace_mcp_enabled}
        binding["completion_policy"] = {"require_workspace_change": self._require_workspace_change}
        binding["model_step_budget"] = self._model_step_budget
        binding["model_token_budget_total"] = self._model_token_budget_total
        binding["model_cost_budget_total_micros"] = self._model_cost_budget_total_micros
        if self._knowledge_revision is not None:
            binding["knowledge_revision"] = self._knowledge_revision
            binding["knowledge_config"] = self._knowledge_config
        return binding

    def ensure_available(
        self,
        binding: Mapping[str, object],
        *,
        thread_id: str,
        turn_id: str,
    ) -> None:
        stored_thread_id = binding.get("thread_id")
        stored_turn_id = binding.get("turn_id")
        if (
            stored_thread_id is not None
            or stored_turn_id is not None
        ) and (stored_thread_id != thread_id or stored_turn_id != turn_id):
            raise RuntimeError("frozen runtime binding belongs to a different Turn")
        model_validator = getattr(self._model, "ensure_available", None)
        if callable(model_validator):
            model_validator(
                binding,
                thread_id=thread_id,
                turn_id=turn_id,
            )
        else:
            raise RuntimeError("bound model cannot validate frozen bindings")
        current = self._static_snapshot()
        for key in (
            "toolset_revision",
            "tool_execution_revisions",
            "knowledge_revision",
            "knowledge_config",
            "mcp_policy",
        ):
            if binding.get(key) != current.get(key):
                raise RuntimeError(f"frozen runtime binding is unavailable: {key} changed")

        parent_turn_id = binding.get("budget_parent_turn_id")
        if isinstance(parent_turn_id, str) and parent_turn_id:
            completion_policy = binding.get("completion_policy")
            if completion_policy != {"require_workspace_change": False}:
                raise RuntimeError("frozen child completion policy is invalid")
            child_steps = binding.get("model_step_budget")
            current_steps = current.get("model_step_budget")
            if (
                isinstance(child_steps, bool)
                or not isinstance(child_steps, int)
                or child_steps < 1
                or isinstance(current_steps, bool)
                or not isinstance(current_steps, int)
                or child_steps > current_steps
            ):
                raise RuntimeError("frozen child model step budget is invalid")
            child_tokens = binding.get("model_token_budget_total")
            current_tokens = current.get("model_token_budget_total")
            if child_tokens is not None and (
                isinstance(child_tokens, bool)
                or not isinstance(child_tokens, int)
                or child_tokens < 1
            ):
                raise RuntimeError("frozen child model token budget is invalid")
            if (
                isinstance(current_tokens, int)
                and not isinstance(current_tokens, bool)
                and child_tokens is not None
                and child_tokens > current_tokens
            ):
                raise RuntimeError("frozen child model token budget exceeds runtime ceiling")
            child_cost = binding.get("model_cost_budget_total_micros")
            current_cost = current.get("model_cost_budget_total_micros")
            if child_cost is not None and (
                isinstance(child_cost, bool)
                or not isinstance(child_cost, int)
                or child_cost < 1
            ):
                raise RuntimeError("frozen child model cost budget is invalid")
            if (
                isinstance(current_cost, int)
                and not isinstance(current_cost, bool)
                and child_cost is not None
                and child_cost > current_cost
            ):
                raise RuntimeError("frozen child model cost budget exceeds runtime ceiling")
        else:
            for key in (
                "completion_policy",
                "model_step_budget",
                "model_token_budget_total",
                "model_cost_budget_total_micros",
            ):
                if binding.get(key) != current.get(key):
                    raise RuntimeError(f"frozen runtime binding is unavailable: {key} changed")


class RuntimeComposition:
    def __init__(
        self,
        *,
        store: RolloutStore,
        thread_manager: ThreadManager,
    ) -> None:
        self.store = store
        self.thread_manager = thread_manager
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        database: Path,
        workspace: Path,
        model: BoundHarnessModel,
        completion_gate: CompletionGate | None = None,
        require_workspace_change: bool = False,
        tools: Mapping[str, Tool] | None = None,
        tool_execution_context: ToolExecutionContext | None = None,
        knowledge_runner: Callable[..., object] | None = None,
        knowledge_revision: str | None = None,
        knowledge_config: Mapping[str, object] | None = None,
        discoverable_tool_names: tuple[str, ...] = (),
        workspace_mcp_enabled: bool = True,
        enable_subagents: bool = False,
        skill_runtime: HarnessSkillRuntime | None = None,
        max_steps: int = 16,
        max_tokens_total: int | None = None,
        max_cost_micros: int | None = None,
        event_dispatcher: TurnEventDispatcher | None = None,
    ) -> RuntimeComposition:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if max_tokens_total is not None and (
            isinstance(max_tokens_total, bool) or not isinstance(max_tokens_total, int) or max_tokens_total < 1
        ):
            raise ValueError("max_tokens_total must be a positive integer or None")
        if max_cost_micros is not None and (
            isinstance(max_cost_micros, bool)
            or not isinstance(max_cost_micros, int)
            or max_cost_micros < 1
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
        thread_manager_ref: dict[str, ThreadManager] = {}

        async def run_subagent(arguments: Mapping[str, JsonValue]) -> object:
            manager = thread_manager_ref.get("manager")
            if manager is None:
                raise RuntimeError("subagent runtime is not fully composed")
            payload = SubagentInput.model_validate(arguments)
            child_task = payload.task
            if payload.context_summary:
                child_task += "\n\nContext supplied by the parent agent:\n" + payload.context_summary
            child = await manager.run_child(
                parent_turn_id=current_tool_turn_id(),
                user_message=child_task,
                max_steps=payload.max_turns,
                max_tokens_total=payload.llm_budget_total,
                max_cost_micros=payload.llm_cost_budget_micros,
            )
            return {
                "conclusion": child.answer or child.interaction_id or "",
                "key_facts": [],
                "evidence_refs": [],
                "citations": [],
                "status": (
                    "done" if child.status == "completed" else "paused" if child.status == "paused" else "failed"
                ),
                "child_turn_id": child.turn_id,
                "stop_reason": None,
            }

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
        registry = build_tool_registry(
            tuple((tools or {}).values()),
            knowledge_tools,
            skill_tools,
            (
                create_subagent_tool(
                    run_subagent,
                    execution_revision="harness-child-thread-v3",
                ),
            )
            if enable_subagents
            else (),
        )
        discoverable_names = (
            *tuple(discoverable_tool_names),
            *(("task",) if enable_subagents else ()),
        )
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
        store = RolloutStore(database)
        integrity = store.verify()
        if not integrity.valid:
            store.close()
            raise RuntimeError("Rollout projection integrity check failed: " + "; ".join(integrity.errors))
        context_manager = RolloutContextManager(store)
        worker_id = f"worker_{uuid4().hex}"
        tool_router = (
            None
            if not tool_snapshot
            else DurableToolRouter(
                store=store,
                tools=tool_snapshot,
                resident_names=(
                    *(("find_tools",) if discoverable_names else ()),
                    *resident_names,
                ),
                discoverable_names=discoverable_names,
            )
        )
        resolved_completion_gate = completion_gate or DeliveryCompletionGate(store)

        def open_session(thread_id: str) -> Session:
            tool_orchestrator = (
                None
                if not tool_snapshot
                else ToolOrchestrator(
                    store=store,
                    tools=tool_snapshot,
                    execution_context=execution_context,
                    worker_id=worker_id,
                )
            )
            return Session(
                thread_id=thread_id,
                store=store,
                model=model,
                context_manager=context_manager,
                completion_gate=resolved_completion_gate,
                tool_router=tool_router,
                tool_orchestrator=tool_orchestrator,
                worker_id=worker_id,
                max_steps=max_steps,
            )

        binding_provider = _CompositionBindingProvider(
            model=model,
            tools=tool_snapshot,
            knowledge_revision=knowledge_revision,
            knowledge_config=knowledge_config,
            tool_execution_policy=_tool_execution_policy_snapshot(execution_context),
            workspace_mcp_enabled=workspace_mcp_enabled,
            require_workspace_change=require_workspace_change,
            model_step_budget=max_steps,
            model_token_budget_total=max_tokens_total,
            model_cost_budget_total_micros=max_cost_micros,
        )
        thread_manager = ThreadManager(
            store=store,
            session_factory=open_session,
            workspace=workspace,
            binding_provider=binding_provider,
            binding_validator=binding_provider.ensure_available,
            event_dispatcher=event_dispatcher,
        )
        thread_manager_ref["manager"] = thread_manager
        return cls(store=store, thread_manager=thread_manager)

    def __enter__(self) -> RuntimeComposition:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.store.close()
            self._closed = True

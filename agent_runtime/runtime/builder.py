from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from agent_runtime.models import ModelControlPlane

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic
    from agent_runtime.service import AgentService
    from agent_runtime.skills.runtime import SkillRuntime
    from agent_runtime.streaming.sink import StreamEventSink
    from agent_runtime.tools.tool import Tool
    from agent_runtime.turns import RuntimeBinding, TurnStore
    from agent_runtime.workspace import WorkspaceRuntime

logger = logging.getLogger(__name__)


def build_model_control_plane(
    *,
    model_alias: str | None = None,
    session_path: Path | None = None,
) -> ModelControlPlane:
    return ModelControlPlane.from_env(
        initial_model_id=model_alias,
        session_path=session_path,
    )


def build_agent_service(
    runtime: WorkspaceRuntime | None,
    *,
    checkpoint_db: Path | None = None,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    model_alias: str | None = None,
    model_control_plane: ModelControlPlane | None = None,
    runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
    knowledge_runner: Callable[..., object] | None = None,
    mcp_tools: Sequence[Tool] = (),
    skill_tools: Sequence[Tool] = (),
    subagent_tools: Sequence[Tool] = (),
    discoverable_tools: Sequence[Tool] = (),
    skill_runtime: SkillRuntime | None = None,
    stream_sink: StreamEventSink | None = None,
    strict_model_provider: bool = True,
    startup_ms: float = 0.0,
    turn_store: TurnStore | None = None,
    runtime_binding: RuntimeBinding | None = None,
) -> AgentService:
    """Build one CLI/SDK runtime from ordinary canonical Tool values."""
    from agent_runtime.builtin.generic import GENERIC_AGENT
    from agent_runtime.core.checkpointing import create_agent_checkpointer
    from agent_runtime.core.runtime_diagnostics import (
        AgentLatencyProfile,
        RuntimeDiagnostic,
    )
    from agent_runtime.service import AgentService
    from agent_runtime.tools.builtins import create_resident_coding_tools
    from agent_runtime.tools.integrations.knowledge import (
        KnowledgeSearchInput,
        create_knowledge_tools,
    )
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.registry import build_tool_registry
    from agent_runtime.tools.selection import (
        create_find_tools_tool,
        find_tools,
    )
    from agent_runtime.workspace import WorkspaceRuntime, create_temp_workspace

    build_started_at = time.perf_counter()
    model_ready_ms = 0.0
    definition = GENERIC_AGENT
    workspace = runtime if isinstance(runtime, WorkspaceRuntime) else create_temp_workspace()

    def acknowledge_plan_update(
        arguments: Mapping[str, object],
    ) -> dict[str, object]:
        del arguments
        return {
            "accepted": True,
            "revision": 0,
            "message": "Plan update accepted for runtime persistence.",
        }

    resident_tools = create_resident_coding_tools(
        workspace,
        plan_updater=acknowledge_plan_update,
    )

    async def search_knowledge(arguments: Mapping[str, object]) -> object:
        if knowledge_runner is None:
            raise RuntimeError("knowledge search is not configured")
        payload = KnowledgeSearchInput.model_validate(arguments)
        result = knowledge_runner(
            payload,
            execution_context=ToolExecutionContext(),
        )
        if hasattr(result, "__await__"):
            return await result
        return result

    knowledge_tools = create_knowledge_tools(search_knowledge if knowledge_runner is not None else None)
    tool_registry = build_tool_registry(
        resident_tools,
        knowledge_tools,
        mcp_tools,
        skill_tools,
        subagent_tools,
        discoverable_tools,
    )
    configured_resident_names = tuple(
        tool.definition.name for source in (knowledge_tools, skill_tools) for tool in source
    )
    discoverable_tool_names = tuple(
        tool.definition.name for source in (mcp_tools, subagent_tools, discoverable_tools) for tool in source
    )
    if discoverable_tool_names:
        discovery_snapshot = MappingProxyType({tool.definition.name: tool for tool in tool_registry.list_all()})

        def search_hidden_tools(query: str, limit: int) -> object:
            return find_tools(
                discovery_snapshot,
                query=query,
                discoverable_names=discoverable_tool_names,
                limit=limit,
                max_active_tools=definition.max_active_deferred_tools,
            )

        tool_registry.register(create_find_tools_tool(search_hidden_tools))
    definition = replace(
        definition,
        core_tool_names=tuple(tool.definition.name for tool in resident_tools),
        deferred_tool_names=(
            *configured_resident_names,
            *discoverable_tool_names,
        ),
    )
    diagnostics: tuple[RuntimeDiagnostic, ...] = tuple(runtime_diagnostics)
    try:
        model_ready_started_at = time.perf_counter()
        model_registry = model_control_plane or build_model_control_plane(
            model_alias=model_alias,
        )
        model_ready_ms = (time.perf_counter() - model_ready_started_at) * 1000
    except Exception as exc:
        if strict_model_provider:
            raise
        model_registry = None
        diagnostics = (
            *diagnostics,
            RuntimeDiagnostic.from_exception(
                code="model_registry_initialization_failed",
                component="model_registry",
                error=exc,
            ),
        )

    return AgentService(
        definition=definition,
        tool_registry=tool_registry,
        model_registry=model_registry,
        checkpointer=(checkpointer if checkpointer is not None else create_agent_checkpointer(checkpoint_db)),
        runtime_diagnostics=diagnostics,
        strict_model_provider=strict_model_provider,
        latency_profile=AgentLatencyProfile(
            startup_ms=startup_ms,
            build_service_ms=(time.perf_counter() - build_started_at) * 1000,
            model_ready_ms=model_ready_ms,
        ),
        workspace=workspace,
        configured_resident_tool_names=configured_resident_names,
        discoverable_tool_names=discoverable_tool_names,
        skill_runtime=skill_runtime,
        stream_sink=stream_sink,
        turn_store=turn_store,
        runtime_binding=runtime_binding,
    )

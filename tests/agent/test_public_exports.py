from __future__ import annotations

from pathlib import Path

_LEGACY_CLOSURE_EXPORTS = {
    "AgentAsToolRunner",
    "AgentDelegationRequest",
    "AgentServiceFactory",
    "BuiltinSubAgentRunner",
    "DelegatedAgentRunner",
    "GraphCompiler",
}

_LEGACY_CLOSURE_MODULES = (
    "rag.agent.core.agent_as_tool",
    "rag.agent.core.agent_service_factory",
    "rag.agent.core.agent_tool_contract",
    "rag.agent.core.compiler",
    "rag.agent.core.delegation",
    "rag.agent.core.runtime_ports",
    "rag.agent.core.registry",
    "rag.agent.core.subagent_runner",
    "rag.agent.graphs.base",
)

_LEGACY_CLOSURE_PATHS = (
    "rag/agent/core/agent_service_factory.py",
    "rag/agent/core/compiler.py",
    "rag/agent/core/subagent_runner.py",
    "rag/agent/core/agent_as_tool.py",
    "rag/agent/core/agent_tool_contract.py",
    "rag/agent/core/delegation.py",
    "rag/agent/core/runtime_ports.py",
    "rag/agent/core/registry.py",
    "rag/agent/graphs/base.py",
    "rag/agent/graphs/__init__.py",
    "rag/agent/graphs/nodes/__init__.py",
)

_ORPHANED_AGENT_PATHS = (
    "rag/agent/binding_providers.py",
    "rag/agent/capabilities/__init__.py",
)


def test_agent_package_exports_new_contract_surface_only() -> None:
    import rag.agent as agent
    import rag.agent.core as core

    assert hasattr(agent, "AgentRuntimePolicy")
    assert not hasattr(agent, "AgentRegistry")
    assert hasattr(agent, "AgentRunConfig")
    assert hasattr(agent, "AgentRunRequest")
    assert hasattr(agent, "AgentRunResult")
    assert hasattr(agent, "AgentService")
    assert hasattr(agent, "AgentState")
    assert hasattr(agent, "Tool")
    assert hasattr(agent, "ToolRegistry")
    assert hasattr(agent, "ToolResult")
    assert hasattr(agent, "TurnRegistry")
    assert not hasattr(agent, "derive_child_config")
    assert _LEGACY_CLOSURE_EXPORTS.isdisjoint(agent.__all__)
    assert _LEGACY_CLOSURE_EXPORTS.isdisjoint(core.__all__)
    assert not hasattr(agent, "AgentGraphCompiler")
    assert not hasattr(agent, "PlanController")
    assert not hasattr(agent, "RuntimeRegistry")
    assert not hasattr(agent, "TaskDAG")
    assert not hasattr(agent, "AnalysisAgentService")
    assert not hasattr(agent, "AgentRunState")
    assert not hasattr(agent, "AgentToolSpec")
    assert not hasattr(agent, "ToolSpec")
    assert not hasattr(agent, "AgentPlan")
    assert not hasattr(agent, "PlanEvent")
    assert not hasattr(agent, "PlanTracker")


def test_root_package_exports_new_agent_contract_surface() -> None:
    from rag import (
        AgentRunConfig,
        AgentRunRequest,
        AgentRuntimePolicy,
        AgentService,
        AgentState,
        Tool,
        ToolRegistry,
        ToolResult,
    )

    assert AgentRuntimePolicy is not None
    assert AgentRunConfig is not None
    assert AgentRunRequest is not None
    assert AgentService is not None
    assert AgentState is not None
    assert Tool is not None
    assert ToolRegistry is not None
    assert ToolResult is not None


def test_legacy_agent_service_module_no_longer_exports_old_service() -> None:
    import importlib

    service = importlib.import_module("rag.agent.service")
    assert not hasattr(service, "AnalysisAgentService")


def test_legacy_agent_modules_are_removed() -> None:
    import importlib.util

    legacy_modules = (
        "rag.agent.planner",
        "rag.agent.executor",
        "rag.agent.critic",
        "rag.agent.synthesizer",
        "rag.agent.understanding",
        "rag.agent.report",
        "rag.agent.schema",
        *_LEGACY_CLOSURE_MODULES,
    )

    for module_name in legacy_modules:
        try:
            module_spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            module_spec = None
        assert module_spec is None


def test_legacy_agent_closure_files_are_removed() -> None:
    root = Path(__file__).resolve().parents[2]

    assert [relative for relative in _LEGACY_CLOSURE_PATHS if (root / relative).exists()] == []


def test_orphaned_agent_paths_are_removed() -> None:
    root = Path(__file__).resolve().parents[2]

    assert [relative for relative in _ORPHANED_AGENT_PATHS if (root / relative).exists()] == []


def test_production_tree_has_no_legacy_agent_closure_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden = (*_LEGACY_CLOSURE_MODULES, "rag.agent.graphs")
    offenders: dict[str, tuple[str, ...]] = {}

    for production_root in (root / "rag", root / "agent_runtime"):
        for path in production_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            matches = tuple(module for module in forbidden if module in source)
            if matches:
                offenders[str(path.relative_to(root))] = matches

    assert offenders == {}

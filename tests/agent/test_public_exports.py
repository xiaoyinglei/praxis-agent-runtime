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
    "agent_runtime.core.agent_as_tool",
    "agent_runtime.core.agent_service_factory",
    "agent_runtime.core.agent_tool_contract",
    "agent_runtime.core.compiler",
    "agent_runtime.core.delegation",
    "agent_runtime.core.runtime_ports",
    "agent_runtime.core.registry",
    "agent_runtime.core.subagent_runner",
    "agent_runtime.graphs.base",
)

_LEGACY_CLOSURE_PATHS = (
    "agent_runtime/core/agent_service_factory.py",
    "agent_runtime/core/compiler.py",
    "agent_runtime/core/subagent_runner.py",
    "agent_runtime/core/agent_as_tool.py",
    "agent_runtime/core/agent_tool_contract.py",
    "agent_runtime/core/delegation.py",
    "agent_runtime/core/runtime_ports.py",
    "agent_runtime/core/registry.py",
    "agent_runtime/graphs/base.py",
    "agent_runtime/graphs/__init__.py",
    "agent_runtime/graphs/nodes/__init__.py",
)

_ORPHANED_AGENT_PATHS = (
    "agent_runtime/binding_providers.py",
    "agent_runtime/capabilities/__init__.py",
)


def test_agent_runtime_internal_contracts_use_explicit_modules() -> None:
    import agent_runtime as public_api
    import agent_runtime.core as core
    from agent_runtime.loop.state import LoopState
    from agent_runtime.service import AgentRunRequest, AgentRunResult, AgentService
    from agent_runtime.tools import Tool, ToolRegistry, ToolResult

    assert core.AgentRuntimePolicy is not None
    assert core.AgentRunConfig is not None
    assert core.TurnRegistry is not None
    assert AgentRunRequest is not None
    assert AgentRunResult is not None
    assert AgentService is not None
    assert LoopState is not None
    assert Tool is not None
    assert ToolRegistry is not None
    assert ToolResult is not None
    for internal_name in (
        "AgentRuntimePolicy",
        "AgentRunConfig",
        "AgentRunRequest",
        "AgentRunResult",
        "AgentService",
        "AgentState",
        "Tool",
        "ToolRegistry",
        "ToolResult",
        "TurnRegistry",
    ):
        assert not hasattr(public_api, internal_name)
    assert not hasattr(core, "AgentRegistry")
    assert not hasattr(core, "derive_child_config")
    assert _LEGACY_CLOSURE_EXPORTS.isdisjoint(public_api.__all__)
    assert _LEGACY_CLOSURE_EXPORTS.isdisjoint(core.__all__)
    for removed_name in (
        "AgentGraphCompiler",
        "PlanController",
        "RuntimeRegistry",
        "TaskDAG",
        "AnalysisAgentService",
        "AgentRunState",
        "AgentToolSpec",
        "ToolSpec",
        "AgentPlan",
        "PlanEvent",
        "PlanTracker",
    ):
        assert not hasattr(public_api, removed_name)


def test_rag_root_does_not_export_agent_contract_surface() -> None:
    import rag

    for name in (
        "AgentRunConfig",
        "AgentRunRequest",
        "AgentRuntimePolicy",
        "AgentService",
        "AgentState",
        "Tool",
        "ToolRegistry",
        "ToolResult",
    ):
        assert not hasattr(rag, name)


def test_legacy_agent_service_module_no_longer_exports_old_service() -> None:
    import importlib

    service = importlib.import_module("agent_runtime.service")
    assert not hasattr(service, "AnalysisAgentService")


def test_legacy_agent_modules_are_removed() -> None:
    import importlib.util

    legacy_modules = (
        "agent_runtime.planner",
        "agent_runtime.executor",
        "agent_runtime.critic",
        "agent_runtime.synthesizer",
        "agent_runtime.understanding",
        "agent_runtime.report",
        "agent_runtime.schema",
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
    forbidden = (*_LEGACY_CLOSURE_MODULES, "agent_runtime.graphs")
    offenders: dict[str, tuple[str, ...]] = {}

    for production_root in (root / "rag", root / "agent_runtime"):
        for path in production_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            matches = tuple(module for module in forbidden if module in source)
            if matches:
                offenders[str(path.relative_to(root))] = matches

    assert offenders == {}

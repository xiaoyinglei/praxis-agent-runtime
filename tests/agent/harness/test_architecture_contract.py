from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[3]
RUNTIME = ROOT / "agent_runtime"


def _sources_containing(needle: str) -> set[Path]:
    return {path.relative_to(ROOT) for path in RUNTIME.rglob("*.py") if needle in path.read_text(encoding="utf-8")}


def test_rollout_reducer_and_tool_runtime_each_have_one_behavior_owner() -> None:
    assert _sources_containing("self._append_and_reduce(") == {Path("agent_runtime/harness/rollout.py")}
    assert _sources_containing("ToolExecutor(tools)") == {Path("agent_runtime/harness/tool_orchestrator.py")}
    assert _sources_containing("tool.run(arguments)") == {Path("agent_runtime/tools/executor.py")}
    assert _sources_containing("return Session(") == {Path("agent_runtime/harness/composition.py")}
    assert _sources_containing("thread_manager = ThreadManager(") == {Path("agent_runtime/harness/composition.py")}


def test_live_execution_spine_is_session_turn_context_step_context() -> None:
    public = (RUNTIME / "harness" / "__init__.py").read_text(encoding="utf-8")
    assert '"Session"' in public
    assert '"TurnContext"' in public
    assert '"StepContext"' in public
    assert '"TurnRunner"' not in public

    session = (RUNTIME / "harness" / "session.py").read_text(encoding="utf-8")
    assert "class Session:" in session
    assert "class TurnContext:" in session
    assert "class StepContext:" in session
    assert "class TurnRunner:" not in session


def test_composition_and_thread_manager_do_not_steal_transition_or_runtime_owners() -> None:
    composition = (RUNTIME / "harness" / "composition.py").read_text(encoding="utf-8")
    manager = (RUNTIME / "harness" / "thread_manager.py").read_text(encoding="utf-8")
    for transition in (
        "._append_and_reduce(",
        ".start_turn(",
        ".complete_turn(",
        ".fail_turn(",
        ".pause_turn(",
    ):
        assert transition not in composition
    for forbidden in (
        "sqlite3",
        "GatewayHarnessModel",
        "ToolExecutor",
        "ToolRegistry",
        "MCPServerRuntime",
        "CompletionDecision",
    ):
        assert forbidden not in manager


def test_deleted_legacy_orchestration_cannot_be_imported_by_public_runtime() -> None:
    for relative in (
        "agent_runtime/service.py",
        "agent_runtime/turns.py",
        "agent_runtime/loop",
        "agent_runtime/memory",
        "agent_runtime/core/checkpointing.py",
    ):
        assert not (ROOT / relative).exists()
    public_source = "\n".join((RUNTIME / relative).read_text(encoding="utf-8") for relative in ("agent.py", "cli.py"))
    for legacy_name in ("AgentService", "AgentLoop", "LoopState", "langgraph"):
        assert legacy_name not in public_source


def test_provider_wire_and_model_dispatch_ownership_is_explicit() -> None:
    assert _sources_containing("def serialize_openai_request(") == {Path("agent_runtime/modeling/openai_wire.py")}
    assert _sources_containing("def render_local_agent_request(") == {
        Path("agent_runtime/modeling/local_agent_wire.py")
    }
    for component in ("context.py", "tool_router.py"):
        source = (RUNTIME / "harness" / component).read_text(encoding="utf-8")
        assert "serialize_openai_request" not in source
        assert "render_local_agent_request" not in source
    assert _sources_containing("dispatch = self._model.dispatch") == {
        Path("agent_runtime/harness/session.py")
    }

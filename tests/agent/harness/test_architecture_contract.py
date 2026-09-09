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
    assert _sources_containing("executor = TurnExecutor(") == {Path("agent_runtime/harness/session.py")}


def test_session_owns_resources_and_turn_executor_borrows_them() -> None:
    session = (RUNTIME / "harness" / "session.py").read_text()
    turn = (RUNTIME / "harness" / "turn.py").read_text()
    assert "class Session:" in session
    assert "class TurnExecutor:" in turn
    assert "class TurnContext:" in turn
    assert "class StepContext:" in turn
    assert "self._stack = AsyncExitStack()" in session
    assert "self._active_turn_lock = asyncio.Lock()" in session
    assert "RolloutStore(" not in turn
    assert "AsyncExitStack" not in turn
    for removed in ("composition.py", "thread_manager.py", "facade.py"):
        assert not (RUNTIME / "harness" / removed).exists()
    assert not _sources_containing("_open_harness_runtime")


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
    model_adapter = (RUNTIME / "harness" / "model_adapter.py").read_text(encoding="utf-8")
    assert "LLMBudgetLedger" not in model_adapter
    assert "inherit_budget_ledger=False" in model_adapter
    assert _sources_containing("def serialize_openai_request(") == {Path("agent_runtime/modeling/openai_wire.py")}
    assert _sources_containing("def render_local_agent_request(") == {
        Path("agent_runtime/modeling/local_agent_wire.py")
    }
    for component in ("context.py", "tool_router.py"):
        source = (RUNTIME / "harness" / component).read_text(encoding="utf-8")
        assert "serialize_openai_request" not in source
        assert "render_local_agent_request" not in source
    assert _sources_containing("dispatch = self._model.dispatch") == {
        Path("agent_runtime/harness/turn.py")
    }

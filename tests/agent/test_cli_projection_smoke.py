from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_smoke_module():
    script_path = Path(__file__).parents[2] / "scripts" / "agent_cli_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "agent_cli_smoke",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_projection_smoke_covers_the_delivery_surface() -> None:
    module = _load_smoke_module()

    result = module.run_smoke()

    assert result.passed, result.failures
    assert set(result.checks) == {
        "command_surface",
        "diff",
        "interactive_commands",
        "model_binding",
        "plan",
        "recovery",
        "recovery_commands",
        "text",
        "tool_error",
        "tool_result",
    }


def test_cli_projection_smoke_main_reports_pass(capsys) -> None:
    module = _load_smoke_module()

    exit_code = module.main()

    assert exit_code == 0
    assert "PASS cli_projection" in capsys.readouterr().out


def test_tool_summary_preserves_long_error_message() -> None:
    from agent_runtime.cli import _format_tool_summary
    from agent_runtime.result import AgentResult, AgentToolCall, AgentUsage

    message = "参数校验失败：" + "详细说明" * 30 + "；limit 必须小于等于 5"
    result = AgentResult(
        status="failed",
        answer=None, files=(), evidence=(), citations=(), usage=AgentUsage(),
        diagnostics=(), turn_id="turn-1", stop_reason=None, pause=None,
        workspace_path=None, groundedness=False, insufficient_evidence=False,
        plan=None, plan_events=(),
        tool_calls=(AgentToolCall(
            tool_call_id="call-1", tool_name="find_tools", is_error=True,
            error_code="invalid_arguments", error_message=message,
        ),),
    )
    assert message in _format_tool_summary(result)

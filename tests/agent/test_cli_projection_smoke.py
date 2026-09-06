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

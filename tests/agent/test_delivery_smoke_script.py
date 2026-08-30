from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_smoke_module():
    script_path = Path(__file__).parents[2] / "scripts" / "agent_delivery_smoke.py"
    spec = importlib.util.spec_from_file_location("agent_delivery_smoke", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_delivery_cases_target_the_public_replacement_harness() -> None:
    module = _load_smoke_module()
    cases = {case.name: case for case in module.build_cases()}

    assert set(cases) == {"direct_answer", "praxis_demo"}
    assert cases["praxis_demo"].expected_tools == (
        "read_file",
        "apply_patch",
        "read_file",
    )


@pytest.mark.anyio
async def test_fake_delivery_matrix_runs_public_sdk_and_approval_resume() -> None:
    module = _load_smoke_module()

    results = await module.run_matrix(model="fake", fake_model=True)

    assert all(result.passed for result in results)
    demo = next(result for result in results if result.name == "praxis_demo")
    assert demo.status == "done"
    assert demo.tools == ("read_file", "apply_patch", "read_file")
    assert any(line.startswith("[patch]") for line in demo.event_lines)
    assert demo.event_lines[-1].startswith("[complete]")


def test_demo_trace_rejects_absolute_paths() -> None:
    module = _load_smoke_module()

    assert module._contains_demo_absolute_path("read /Users/alice/private.py")
    assert not module._contains_demo_absolute_path("read fixture.py")

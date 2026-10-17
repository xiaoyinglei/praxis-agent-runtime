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

    assert set(cases) == {"direct_answer", "praxis_demo", "checkout_demo"}
    assert cases["praxis_demo"].expected_tools == (
        "read_file",
        "apply_patch",
        "read_file",
    )


def test_fake_delivery_binding_uses_only_model_id() -> None:
    module = _load_smoke_module()
    case = module.build_cases()[0]
    model = module._SmokeModel(case)

    binding = model.snapshot(thread_id="thread-smoke", turn_id="turn-smoke")

    assert binding["model_id"] == f"smoke-{case.name}"
    assert "model_alias" not in binding
    with pytest.raises(RuntimeError, match="legacy model_alias"):
        model.ensure_available(
            {
                **binding,
                "model_alias": f"smoke-{case.name}",
                "thread_id": "thread-smoke",
                "turn_id": "turn-smoke",
            },
            thread_id="thread-smoke",
            turn_id="turn-smoke",
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
    checkout = next(result for result in results if result.name == "checkout_demo")
    assert checkout.tools == ("read_file", "run_command", "apply_patch", "run_command")
    assert "FAILED (failures=2)" in checkout.check_outputs[0]
    assert "Ran 3 tests" in checkout.check_outputs[1]
    assert "\nOK\n" in checkout.check_outputs[1]
    assert "+    return round(subtotal * (1 - discount_percent / 100), 2)" in checkout.workspace_diff
    assert checkout.validation_status == "passed"


def test_demo_trace_rejects_absolute_paths() -> None:
    module = _load_smoke_module()

    assert module._contains_demo_absolute_path("read /Users/alice/private.py")
    assert not module._contains_demo_absolute_path("read fixture.py")


@pytest.mark.anyio
@pytest.mark.parametrize('scenario', ['no_change', 'verify_before_change', 'unreached_assertion'])
async def test_delivery_smoke_explicit_acceptance_rejects_missing_or_unexecuted_checks(scenario):
    module = _load_smoke_module()
    case = module.build_cases()[2]
    read = module.HarnessToolCall(id='read', name='read_file', arguments={'path': 'pricing.py'})
    verify = module.HarnessToolCall(id='verify', name='run_command', arguments={'command': module.CHECKOUT_COMMAND})
    patch = module.HarnessToolCall(id='patch', name='apply_patch', arguments={
        'file_path': 'pricing.py', 'old_string': module.CHECKOUT_BEFORE, 'new_string': module.CHECKOUT_AFTER})
    unexecuted = module.HarnessToolCall(id='unchecked', name='execute_python', arguments={
        'code': 'def never_called():\n    assert False\nprint("no check ran")'})
    calls = {'no_change': (read,), 'verify_before_change': (verify, patch),
             'unreached_assertion': (patch, unexecuted)}[scenario]

    class Model(module._SmokeModel):
        async def dispatch(self, prepared, *, delta_sink=None):
            step = int(prepared.request_ref['step'])
            if step <= len(calls):
                return module._response(text='', step=step, tool_calls=(calls[step - 1],))
            return module._response(text='praxis demo complete', step=step)

    result = await module._run_case(case, model=Model(case))
    assert not result.passed
    assert result.status != 'done'
    assert result.validation_status == 'failed'
    assert result.validation_scope == ('workspace_change_recorded' if scenario == 'no_change' else 'acceptance_command')
    if scenario == 'no_change':
        assert result.workspace_diff == ''
    else:
        assert "-    return round(subtotal * (1 - discount_percent), 2)" in result.workspace_diff
        assert "+    return round(subtotal * (1 - discount_percent / 100), 2)" in result.workspace_diff

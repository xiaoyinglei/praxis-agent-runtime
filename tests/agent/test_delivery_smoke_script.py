from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from agent_runtime.tools.builtins import RESIDENT_CODING_TOOL_NAMES


def _load_smoke_module():
    script_path = Path(__file__).parents[2] / "scripts" / "agent_delivery_smoke.py"
    spec = importlib.util.spec_from_file_location(
        "agent_delivery_smoke",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_delivery_cases_cover_the_locked_public_matrix() -> None:
    module = _load_smoke_module()
    cases = {case.name: case for case in module.build_cases()}

    assert {
        "direct_answer",
        "find_agent_service",
        "patch_fixture",
        "echo_hello",
        "missing_file_recovery",
        "repeated_failure_circuit",
        "hidden_mcp_disabled",
        "hidden_mcp_discovery",
        "approval_resume",
        "cache_usage",
        "mlx_local_envelope",
        "ollama_local_envelope",
    } == set(cases)
    assert cases["direct_answer"].expected_tools == ()
    assert cases["direct_answer"].expected_answer_exact == "4"
    assert cases["direct_answer"].expected_initial_tools == (
        RESIDENT_CODING_TOOL_NAMES
    )
    assert cases["find_agent_service"].expected_tools == (
        "search_text",
        "read_file",
    )
    assert cases["patch_fixture"].expected_tools == ("apply_patch",)
    assert cases["patch_fixture"].expected_answer_contains == ()
    assert cases["echo_hello"].expected_tools == ("run_command",)
    assert cases["missing_file_recovery"].expected_tool_errors == (
        "read_file:runner_failed",
    )
    assert cases["repeated_failure_circuit"].expected_tool_errors == (
        "read_file:runner_failed",
        "read_file:repeated_tool_failure",
    )
    assert cases["hidden_mcp_disabled"].allow_discovery_tools is False
    assert cases["hidden_mcp_disabled"].expected_answer_exact == (
        "hidden_disabled"
    )
    assert cases["hidden_mcp_discovery"].allow_discovery_tools is True
    assert cases["approval_resume"].expect_origin_retained is True
    assert cases["approval_resume"].expected_answer_contains == ()
    assert cases["cache_usage"].expected_answer_exact == "cache_visible"
    assert cases["mlx_local_envelope"].provider == "mlx"
    assert cases["ollama_local_envelope"].provider == "ollama"


def test_exact_answer_assertion_rejects_substring_false_positive() -> None:
    module = _load_smoke_module()
    case = module.SmokeCase(
        name="strict-answer",
        task="Answer exactly cache_visible.",
        expected_answer_exact="cache_visible",
    )
    result = module.SmokeResult(
        name=case.name,
        passed=False,
        status="done",
        answer="I cannot provide cache_visible.",
        tools=(),
        visible_tools=(),
        workspace_path=None,
    )

    error = module._validate_result(case, result)

    assert "expected exact answer" in error


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_fake_delivery_matrix_proves_public_runtime_invariants() -> None:
    module = _load_smoke_module()

    results = await module.run_matrix(model="fake", fake_model=True)
    by_name = {result.name: result for result in results}

    assert len(results) == 12
    assert all(result.passed for result in results), {
        result.name: result.error for result in results if not result.passed
    }

    direct = by_name["direct_answer"]
    assert direct.tools == ()
    assert direct.visible_tools == (RESIDENT_CODING_TOOL_NAMES,)
    assert direct.schema_bytes > 0

    find_service = by_name["find_agent_service"]
    assert find_service.tools == ("search_text", "read_file")
    assert find_service.visible_tools[0] == ("search_text", "read_file")
    assert find_service.tool_errors == ()

    patched = by_name["patch_fixture"]
    assert patched.tools == ("apply_patch",)
    assert patched.workspace_assertions_passed is True

    command = by_name["echo_hello"]
    assert command.tools == ("run_command",)
    assert command.tool_errors == ()

    missing = by_name["missing_file_recovery"]
    assert missing.status == "done"
    assert missing.tool_errors == (
        "read_file:runner_failed: tool runner failed",
    )

    circuit = by_name["repeated_failure_circuit"]
    assert circuit.status == "done"
    assert circuit.tools == ("read_file", "read_file", "read_file")
    assert sum("runner_failed" in error for error in circuit.tool_errors) == 2
    assert any(
        "repeated_tool_failure" in error
        for error in circuit.tool_errors
    )
    assert any(
        diagnostic.startswith("repeated_tool_failure:")
        for diagnostic in circuit.diagnostics
    )

    hidden_off = by_name["hidden_mcp_disabled"]
    assert "find_tools" not in hidden_off.visible_tools[0]
    assert "mcp__docs__search" not in hidden_off.visible_tools[0]
    assert hidden_off.tools == ()

    hidden_on = by_name["hidden_mcp_discovery"]
    assert hidden_on.tools == ("find_tools", "mcp__docs__search")
    assert "find_tools" in hidden_on.visible_tools[0]
    assert "mcp__docs__search" not in hidden_on.visible_tools[0]
    assert "mcp__docs__search" in hidden_on.visible_tools[1]

    approval = by_name["approval_resume"]
    assert approval.approval_count == 1
    assert approval.origin_retained is True
    assert approval.origin_toolset_revisions == (
        approval.toolset_revisions[0],
    )

    cache = by_name["cache_usage"]
    assert cache.usage_source == "provider"
    assert cache.cache_read_input_tokens == 7
    assert cache.cache_write_input_tokens == 3
    assert cache.prompt_revisions
    assert cache.toolset_revisions
    assert cache.provider_wire_hashes[0].startswith("wire_")
    assert cache.schema_bytes == direct.schema_bytes
    assert cache.toolset_revisions == direct.toolset_revisions
    evidence = module.delivery_metric_evidence(results)
    assert evidence.recovery_successes == 2
    assert evidence.recovery_cases == 2

    for name, wire_kind in (
        ("mlx_local_envelope", "mlx"),
        ("ollama_local_envelope", "ollama"),
    ):
        local = by_name[name]
        assert local.provider_wire_kind == wire_kind
        assert local.tools == ("read_file",)
        assert local.tool_errors == ()
        assert local.result_content_kinds == ("structured",)


def test_verbose_output_reports_revision_wire_schema_error_and_cache_data() -> None:
    module = _load_smoke_module()
    result = module.SmokeResult(
        name="cache_usage",
        passed=True,
        status="done",
        answer="cached",
        tools=(),
        visible_tools=(RESIDENT_CODING_TOOL_NAMES,),
        workspace_path="/tmp/workspace",
        schema_bytes=1234,
        request_ids=("request-1",),
        prompt_revisions=("prompt-1",),
        toolset_revisions=("tools-1",),
        provider_wire_hashes=("wire-1",),
        provider_wire_kind="openai",
        serializer_revision="openai-compatible-chat-v1",
        usage_source="provider",
        cache_read_input_tokens=7,
        cache_write_input_tokens=3,
        tool_errors=("read_file:runner_failed: tool runner failed",),
    )

    rendered = "\n".join(module._format_result(result, verbose=True))

    assert "schema_bytes: 1234" in rendered
    assert "request=request-1" in rendered
    assert "prompt=prompt-1" in rendered
    assert "toolset=tools-1" in rendered
    assert "wire_hash=wire-1" in rendered
    assert "wire_kind=openai" in rendered
    assert "serializer=openai-compatible-chat-v1" in rendered
    assert "usage_source=provider" in rendered
    assert "cache_read=7" in rendered
    assert "cache_write=3" in rendered
    assert "tool_error: read_file:runner_failed: tool runner failed" in rendered


def test_smoke_source_no_longer_imports_legacy_tool_surface() -> None:
    source = (
        Path(__file__).parents[2] / "scripts" / "agent_delivery_smoke.py"
    ).read_text(encoding="utf-8")

    assert "agent_runtime.tooling" not in source
    assert "ToolSurfaceRequest" not in source
    assert "tool_surface_request" not in source

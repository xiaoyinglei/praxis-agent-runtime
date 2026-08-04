#!/usr/bin/env python
"""Measure the Single Tool Runtime ACI with deterministic fake-model cases."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

DEFAULT_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "agent"
    / "fixtures"
    / "tool_aci_cases.json"
)


@dataclass(frozen=True, slots=True)
class ACICase:
    case_id: str
    category: str
    prompt: str
    tools: tuple[str, ...] | None
    allow_discovery_tools: bool
    expected_surface: tuple[str, ...]
    expected_tool: str | None
    fake_tool: str | None
    fake_arguments: Mapping[str, object]
    discovery_query: str | None = None
    expected_discovery: str | None = None
    recoverable: bool = False
    fake_recovery_present: bool = False
    fake_recovery_tool: str | None = None


@dataclass(frozen=True, slots=True)
class _RuntimeFixture:
    snapshot: Mapping[str, Any]
    default_resident_names: tuple[str, ...]
    discoverable_names: tuple[str, ...]
    workspace_root: Path


def load_cases(fixture_path: Path = DEFAULT_FIXTURE_PATH) -> tuple[ACICase, ...]:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("ACI fixture must use schema_version 1")
    if "thresholds" in payload:
        raise ValueError("ACI fixture must not define quality thresholds")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("ACI fixture cases must be a non-empty list")

    cases: list[ACICase] = []
    seen_ids: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("each ACI case must be an object")
        if "threshold" in raw or "thresholds" in raw:
            raise ValueError("ACI cases must not define quality thresholds")
        case_id = _required_text(raw, "id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate ACI case id: {case_id}")
        seen_ids.add(case_id)
        tools_value = raw.get("tools")
        tools = None if tools_value is None else _text_tuple(tools_value, "tools")
        fake_arguments = raw.get("fake_arguments", {})
        if not isinstance(fake_arguments, dict):
            raise ValueError(f"ACI case {case_id!r} fake_arguments must be an object")
        cases.append(
            ACICase(
                case_id=case_id,
                category=_required_text(raw, "category"),
                prompt=_required_text(raw, "prompt"),
                tools=tools,
                allow_discovery_tools=_required_bool(
                    raw,
                    "allow_discovery_tools",
                ),
                expected_surface=_text_tuple(
                    raw.get("expected_surface"),
                    "expected_surface",
                ),
                expected_tool=_optional_text(raw.get("expected_tool")),
                fake_tool=_optional_text(raw.get("fake_tool")),
                fake_arguments=dict(fake_arguments),
                discovery_query=_optional_text(raw.get("discovery_query")),
                expected_discovery=_optional_text(
                    raw.get("expected_discovery")
                ),
                recoverable=bool(raw.get("recoverable", False)),
                fake_recovery_present="fake_recovery" in raw,
                fake_recovery_tool=_fake_recovery_tool(raw, case_id=case_id),
            )
        )
    return tuple(cases)


async def run_evaluation(
    *,
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    fake_model: bool = False,
) -> dict[str, object]:
    """Run the offline benchmark and return a JSON-safe metric report."""

    if not fake_model:
        raise ValueError(
            "the deterministic ACI suite requires fake_model=True; "
            "live-provider delivery checks are opt-in via agent_delivery_smoke.py"
        )
    cases = load_cases(fixture_path)
    runtime = _build_runtime_fixture()
    try:
        case_results = [_evaluate_case(case, runtime) for case in cases]
        delivery = await _measure_delivery_evidence()
    finally:
        shutil.rmtree(runtime.workspace_root, ignore_errors=True)

    surface_expected = sum(
        len(result["expected_surface"]) for result in case_results
    )
    surface_actual = sum(
        len(result["actual_surface"]) for result in case_results
    )
    surface_matches = sum(
        len(
            set(result["expected_surface"])
            & set(result["actual_surface"])
        )
        for result in case_results
    )
    predicted_calls = [
        result for result in case_results if result["predicted_tool"] is not None
    ]
    no_call_cases = [
        result for result in case_results if result["expected_tool"] is None
    ]
    discovery_cases = [
        result
        for result in case_results
        if result["expected_discovery"] is not None
    ]
    fixture_recovery_cases = [
        result for result in case_results if result["recoverable"]
    ]
    recovery_successes = delivery.recovery_successes + sum(
        bool(result["recovery_succeeded"])
        for result in fixture_recovery_cases
    )
    recovery_cases = delivery.recovery_cases + len(fixture_recovery_cases)
    schema_bytes_total = sum(
        int(result["schema_bytes"]) for result in case_results
    )
    schema_tokens_total = sum(
        int(result["schema_tokens"]) for result in case_results
    )

    metrics: dict[str, object] = {
        "surface_recall": _rate(surface_matches, surface_expected),
        "surface_precision": _rate(surface_matches, surface_actual),
        "tool_choice_accuracy": _rate(
            sum(bool(result["tool_choice_correct"]) for result in case_results),
            len(case_results),
        ),
        "argument_validity": _rate(
            sum(bool(result["arguments_valid"]) for result in predicted_calls),
            len(predicted_calls),
        ),
        "unnecessary_call_rate": _rate(
            sum(result["predicted_tool"] is not None for result in no_call_cases),
            len(no_call_cases),
        ),
        "discovery_recall_at_5": _rate(
            sum(bool(result["discovery_hit_at_5"]) for result in discovery_cases),
            len(discovery_cases),
        ),
        "recovery_rate": _rate(recovery_successes, recovery_cases),
        "schema_bytes": _mean(schema_bytes_total, len(case_results)),
        "schema_tokens": _mean(schema_tokens_total, len(case_results)),
        "cache_read_tokens": delivery.cache_read_tokens,
        "cache_write_tokens": delivery.cache_write_tokens,
        "cache_usage_source": delivery.cache_usage_source,
    }
    metric_counts: dict[str, object] = {
        "surface": {
            "matches": surface_matches,
            "expected": surface_expected,
            "actual": surface_actual,
        },
        "tool_choice": {
            "correct": sum(
                bool(result["tool_choice_correct"])
                for result in case_results
            ),
            "cases": len(case_results),
        },
        "arguments": {
            "valid": sum(
                bool(result["arguments_valid"])
                for result in predicted_calls
            ),
            "calls": len(predicted_calls),
        },
        "unnecessary_calls": {
            "calls": sum(
                result["predicted_tool"] is not None
                for result in no_call_cases
            ),
            "no_call_cases": len(no_call_cases),
        },
        "discovery": {
            "hits": sum(
                bool(result["discovery_hit_at_5"])
                for result in discovery_cases
            ),
            "cases": len(discovery_cases),
            "k": 5,
        },
        "recovery": {
            "successes": recovery_successes,
            "cases": recovery_cases,
        },
        "schema": {
            "bytes_total": schema_bytes_total,
            "tokens_total": schema_tokens_total,
            "requests": len(case_results),
            "aggregation": "mean_per_initial_request",
            "tokenizer": "simple",
        },
    }
    return {
        "benchmark": "single-tool-runtime-aci-v1",
        "mode": "fake-model",
        "fixture": str(fixture_path),
        "case_count": len(case_results),
        "metrics": metrics,
        "metric_counts": metric_counts,
        "cases": case_results,
    }


def _build_runtime_fixture() -> _RuntimeFixture:
    from agent_runtime.tools.builtins import (
        RESIDENT_CODING_TOOL_NAMES,
        create_resident_coding_tools,
    )
    from agent_runtime.tools.integrations.knowledge import (
        create_search_knowledge_tool,
    )
    from agent_runtime.tools.integrations.mcp import (
        MCPToolDescriptor,
        create_mcp_tools,
    )
    from agent_runtime.tools.integrations.subagent import create_subagent_tool
    from agent_runtime.tools.registry import build_tool_registry
    from agent_runtime.tools.selection import create_find_tools_tool, find_tools
    from agent_runtime.workspace import create_temp_workspace

    workspace = create_temp_workspace(prefix="agent_tool_aci_")
    (workspace.input_files / "README.md").write_text(
        "# Runtime\nSingle Tool Runtime fixture.\n",
        encoding="utf-8",
    )
    (workspace.input_files / "fixture.txt").write_text(
        "before\n",
        encoding="utf-8",
    )
    (workspace.input_files / "service.py").write_text(
        "class AgentService:\n    pass\n",
        encoding="utf-8",
    )

    plan_revision = 0

    def update_plan(_arguments: Mapping[str, object]) -> dict[str, object]:
        nonlocal plan_revision
        plan_revision += 1
        return {
            "accepted": True,
            "revision": plan_revision,
            "message": "Plan updated.",
        }

    resident_tools = create_resident_coding_tools(
        workspace,
        plan_updater=update_plan,
    )
    knowledge_tool = create_search_knowledge_tool(
        lambda _arguments: {
            "results": [],
            "answer_text": "Approval requires policy evidence.",
            "citations": ["policy#approval"],
            "groundedness_flag": True,
            "insufficient_evidence": False,
            "total_found": 1,
        },
        execution_revision="aci-v1",
    )
    mcp_tools = create_mcp_tools(
        (
            MCPToolDescriptor(
                server_name="docs",
                tool_name="search",
                description=(
                    "Search external runtime documentation. "
                    "查询外部文档，搜索运行时资料。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1}
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                read_only_hint=True,
                idempotent_hint=True,
                execution_revision="aci-v1",
            ),
        ),
        lambda _server, _tool, _arguments: {
            "content": [{"type": "text", "text": "runtime docs"}]
        },
    )
    subagent_tool = create_subagent_tool(
        lambda _arguments: {
            "conclusion": "Boundary reviewed.",
            "key_facts": ["one Tool contract"],
            "evidence_refs": [],
            "citations": [],
            "status": "done",
            "child_run_id": "aci-child",
            "stop_reason": None,
        },
        execution_revision="aci-v1",
    )

    snapshot: Mapping[str, Any] = {}
    discoverable_names = (
        knowledge_tool.definition.name,
        *(tool.definition.name for tool in mcp_tools),
        subagent_tool.definition.name,
    )

    def search_hidden(query: str, limit: int) -> object:
        return find_tools(
            snapshot,
            query=query,
            discoverable_names=discoverable_names,
            resident_names=RESIDENT_CODING_TOOL_NAMES,
            limit=limit,
        )

    find_tool = create_find_tools_tool(
        search_hidden,
        execution_revision="aci-v1",
    )
    registry = build_tool_registry(
        resident_tools,
        find_tool,
        (knowledge_tool,),
        mcp_tools,
        (subagent_tool,),
    )
    snapshot = registry.freeze()
    return _RuntimeFixture(
        snapshot=snapshot,
        default_resident_names=RESIDENT_CODING_TOOL_NAMES,
        discoverable_names=discoverable_names,
        workspace_root=workspace.root,
    )


def _evaluate_case(
    case: ACICase,
    runtime: _RuntimeFixture,
) -> dict[str, object]:
    from agent_runtime.core.model_request import (
        canonical_json_text,
        tool_definition_payload,
    )
    from agent_runtime.tools.selection import (
        find_tools,
        resolve_tool_options,
        select_tools,
    )
    from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract

    options = resolve_tool_options(
        runtime.snapshot,
        default_resident_names=runtime.default_resident_names,
        tools=case.tools,
        allow_discovery_tools=case.allow_discovery_tools,
    )
    selected = select_tools(
        runtime.snapshot,
        resident_names=options.resident_names,
        disabled_names=options.disabled_names,
    )
    actual_surface = tuple(tool.definition.name for tool in selected)
    schema_text = canonical_json_text(
        tuple(
            tool_definition_payload(tool.definition)
            for tool in selected
        )
    )
    token_accounting = TokenAccountingService(
        TokenizerContract(
            embedding_model_name="aci-simple",
            tokenizer_model_name="aci-simple",
            chunking_tokenizer_model_name="aci-simple",
            tokenizer_backend="simple",
        )
    )

    arguments_valid: bool | None = None
    if case.fake_tool is not None:
        predicted = runtime.snapshot.get(case.fake_tool)
        if predicted is None or case.fake_tool not in actual_surface:
            arguments_valid = False
        else:
            try:
                predicted.validate_input(case.fake_arguments)
            except Exception:
                arguments_valid = False
            else:
                arguments_valid = True

    discovery_names: tuple[str, ...] = ()
    discovery_hit = False
    if case.discovery_query is not None:
        found = find_tools(
            runtime.snapshot,
            query=case.discovery_query,
            discoverable_names=runtime.discoverable_names,
            resident_names=runtime.default_resident_names,
            limit=5,
        )
        discovery_names = tuple(match.name for match in found.matches)
        discovery_hit = case.expected_discovery in discovery_names

    initial_rejected = (
        case.fake_tool is not None and case.fake_tool not in actual_surface
    )
    recovery_succeeded = (
        case.recoverable
        and initial_rejected
        and case.fake_recovery_present
        and case.fake_recovery_tool == case.expected_tool
    )
    return {
        "id": case.case_id,
        "category": case.category,
        "prompt": case.prompt,
        "expected_surface": list(case.expected_surface),
        "actual_surface": list(actual_surface),
        "expected_tool": case.expected_tool,
        "predicted_tool": case.fake_tool,
        "tool_choice_correct": case.fake_tool == case.expected_tool,
        "arguments_valid": arguments_valid,
        "expected_discovery": case.expected_discovery,
        "discovery_top_5": list(discovery_names),
        "discovery_hit_at_5": discovery_hit,
        "recoverable": case.recoverable,
        "fake_recovery_present": case.fake_recovery_present,
        "recovery_succeeded": recovery_succeeded,
        "schema_bytes": len(schema_text.encode("utf-8")),
        "schema_tokens": token_accounting.count(schema_text),
    }


async def _measure_delivery_evidence() -> Any:
    module = _load_delivery_smoke_module()
    results = await module.run_matrix(
        model="fake",
        fake_model=True,
        only={
            "cache_usage",
            "missing_file_recovery",
            "repeated_failure_circuit",
        },
    )
    failed = [result for result in results if not result.passed]
    if failed:
        details = "; ".join(
            f"{result.name}: {result.error}" for result in failed
        )
        raise RuntimeError(f"delivery evidence failed: {details}")
    return module.delivery_metric_evidence(results)


def _load_delivery_smoke_module() -> ModuleType:
    script_path = Path(__file__).with_name("agent_delivery_smoke.py")
    module_name = "_agent_delivery_smoke_for_aci"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load agent_delivery_smoke.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _required_text(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ACI case field {key!r} must be non-empty text")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("optional ACI text values must be non-empty or null")
    return value


def _fake_recovery_tool(
    raw: Mapping[str, object],
    *,
    case_id: str,
) -> str | None:
    if "fake_recovery" not in raw:
        return None
    recovery = raw["fake_recovery"]
    if not isinstance(recovery, dict) or set(recovery) != {"tool"}:
        raise ValueError(
            f"ACI case {case_id!r} fake_recovery must contain only tool"
        )
    return _optional_text(recovery["tool"])


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"ACI case field {field_name!r} must be a text list")
    return tuple(value)


def _required_bool(raw: Mapping[str, object], key: str) -> bool:
    value = raw.get(key)
    if type(value) is not bool:
        raise ValueError(f"ACI case field {key!r} must be a bool")
    return value


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _mean(total: int, count: int) -> int:
    return 0 if count == 0 else round(total / count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Path to a schema-version-1 ACI case fixture.",
    )
    parser.add_argument(
        "--fake-model",
        action="store_true",
        help="Run the deterministic offline model decisions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable report.",
    )
    args = parser.parse_args()
    if not args.fake_model:
        parser.error(
            "--fake-model is required; live-provider checks are opt-in via "
            "scripts/agent_delivery_smoke.py"
        )
    report = asyncio.run(
        run_evaluation(
            fixture_path=args.fixture,
            fake_model=args.fake_model,
        )
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(f"benchmark: {report['benchmark']}")
        print(f"mode: {report['mode']}")
        print(f"cases: {report['case_count']}")
        for name, value in report["metrics"].items():
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

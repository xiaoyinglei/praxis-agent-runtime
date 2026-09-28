"""Frozen Session-level code task; opt-in DeepSeek Flash only, no seeded conversation."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent_runtime.builtin.generic import coding_instructions
from agent_runtime.harness import GatewayHarnessModel, RolloutContextManager, Session
from agent_runtime.model_definition import ProviderOptionsDefinition, ThinkingOptionsDefinition
from agent_runtime.models import ModelControlPlane
from agent_runtime.tools.builtins.filesystem import create_apply_patch_tool, create_read_file_tool
from agent_runtime.tools.builtins.shell import create_run_command_tool
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.workspace import open_workspace
from evals.context_management.functional_live import RecordAnswer, evidence

BYTE_LIMIT = 14000
MAX_STEPS = 16
TOKEN_BUDGET = 100000


def verification_succeeded(items, operations):
    """Accept the fixture's verification command, not arbitrary successful shell use."""
    verification_calls = set()
    for item in items:
        if item.kind != "tool_call":
            continue
        command = item.payload.get("arguments", {}).get("command", "")
        try:
            argv = shlex.split(command)
        except ValueError:
            continue
        # This frozen task explicitly requests python3 verify.py. Compound shell
        # expressions need richer evidence; never infer success from a substring.
        if argv in (["python3", "verify.py"], ["python3", "./verify.py"]):
            verification_calls.add(item.payload["tool_call_id"])
    return any(
        op.tool_name == "run_command" and op.status == "succeeded" and op.tool_call_id in verification_calls
        for op in operations
    )

PHASES = (
    "Implement reconcile(events) in ledger.py from identity.md, deletion.md and ordering.md. "
    "First read those three documents together with three read_file calls in ONE response. "
    "Preserve event dictionaries and the input list; do not change specification documents. "
    "Add and run verify.py covering the specification, using python3. Finish only after checking the changed file.",
    "Continue the same ledger project. Add totals(events) in ledger.py from aggregation.md, empty.md and amounts.md. "
    "First read those three new documents together with three read_file calls in ONE response. "
    "All previously agreed reconcile behavior remains required. Do not change any specification documents. "
    "Extend and run verify.py, using python3; verify both APIs and inspect the final changed file.",
)


def create_fixture(root: Path):
    (root / "ledger.py").write_text('"""Tenant event ledger."""\n\ndef reconcile(events):\n    return list(events)\n')
    specs = {
        "identity.md": "reconcile(events) returns a new list of winning event dictionaries. "
        "Identity is (tenant, id), never id alone. "
        "Select greatest seq. Equal seq: the later INPUT POSITION wins. Never modify input list or dictionaries.",
        "deletion.md": "Apply winner selection before deletion. "
        "A winning event with deleted=true removes that identity. "
        "A later nondeleted event can resurrect an identity. Default deleted is false. No other filtering.",
        "ordering.md": "Sort live winners lexicographically by (tenant, id). "
        "Input uses string tenant/id and integer seq/amount. "
        "Preserve every field in the winning dictionary. Empty input yields [].",
        "aggregation.md": "totals(events) first applies reconcile. "
        "Return dict mapping tenant to SUM of live winning amounts. "
        "Never count discarded duplicates or deleted winners. Keep reconcile API and semantics unchanged.",
        "empty.md": "A tenant with only deleted winners is absent, not zero. "
        "A tenant with a live zero total is present with 0. "
        "Empty input returns {}. Both APIs must leave all input data unchanged.",
        "amounts.md": "Amounts are integers, including negative values and integers larger than 2**53. "
        "Do not convert to float or string. Missing amount means 0. Preserve arbitrary fields in reconcile winners.",
    }
    for name, rules in specs.items():
        # Actual sample data rather than injected conversation padding. All examples obey the same contract.
        examples = [
            json.dumps(
                {
                    "tenant": f"tenant-{i % 7}",
                    "id": f"event-{i}",
                    "seq": i,
                    "amount": (i - 25) * 9007199254740993,
                    "deleted": i % 9 == 0,
                    "note": f"audit sample {i}; preserve this field",
                }
            )
            for i in range(50)
        ]
        (root / name).write_text(rules + "\n\nExample input records (independent identities):\n" + "\n".join(examples))
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in specs}


# Hidden from the model workspace and prompts; fixed before any live request.
ORACLE = r"""
import copy, importlib.util, json, sys
spec = importlib.util.spec_from_file_location("ledger", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a = [
 {"tenant":"b","id":"x","seq":2,"amount":7,"extra":"keep"},
 {"tenant":"a","id":"x","seq":3,"amount":2},
 {"tenant":"a","id":"x","seq":1,"amount":900},
 {"tenant":"a","id":"x","seq":3,"amount":-2,"extra":"later tie"},
 {"tenant":"a","id":"y","seq":2,"deleted":True,"amount":999},
 {"tenant":"a","id":"y","seq":1,"amount":500},
 {"tenant":"c","id":"z","seq":1,"deleted":True},
 {"tenant":"c","id":"z","seq":2,"amount":0},
 {"tenant":"b","id":"large","seq":1,"amount":9007199254740993},
 {"tenant":"d","id":"gone","seq":1,"deleted":True},
 {"tenant":"c","id":"missing","seq":1},
]
original=copy.deepcopy(a)
expected=[a[3],a[8],a[0],a[10],a[7]]
assert m.reconcile(a)==expected, (m.reconcile(a),expected)
assert a==original and m.reconcile([])==[]
if sys.argv[2]=="2":
 assert m.totals(a)=={"a":-2,"b":9007199254741000,"c":0}, m.totals(a)
 assert m.totals([])=={} and a==original
print(json.dumps({"pass":True,"phase":int(sys.argv[2])}))
"""


async def main():
    load_dotenv()
    output = Path("evals/context_management") / f"deepseek-session-code-{datetime.now():%Y%m%d-%H%M%S}"
    output.mkdir()
    root = (output / "workspace").resolve()
    root.mkdir()
    hashes = create_fixture(root)
    report = {
        "model": "deepseek-flash",
        "thinking": "disabled",
        "byte_limit": BYTE_LIMIT,
        "steps_per_turn": MAX_STEPS,
        "tokens_per_turn": TOKEN_BUDGET,
        "task_sha256": hashlib.sha256(json.dumps(PHASES).encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "spec_hashes": hashes,
        "phases": [],
    }
    (output / "frozen.json").write_text(json.dumps(report, indent=2))
    (output / "oracle.py").write_text(ORACLE)
    control = ModelControlPlane.from_config_file(Path("configs/models.yaml"), initial_model_id="deepseek-flash")
    spans = []
    try:
        resolved = control.resolve("deepseek-flash")
        report["token_count_source"] = resolved.token_accounting.budget_count_source()
        (output / "frozen.json").write_text(json.dumps(report, indent=2))
        resolved = replace(
            resolved,
            capabilities=replace(resolved.capabilities, max_output_tokens=4096),
            request_defaults=resolved.request_defaults.model_copy(
                update={
                    "parallel_tool_calls": True,
                    "provider_options": ProviderOptionsDefinition(thinking=ThinkingOptionsDefinition(type="disabled")),
                }
            ),
        )
        sdk_create = resolved.generator._client.chat.completions.create
        report["wire_trace_files"] = []

        def traced_create(**kwargs):
            assert kwargs.get("model") == "deepseek-flash"
            trace = {
                key: kwargs[key]
                for key in (
                    "model",
                    "messages",
                    "tools",
                    "tool_choice",
                    "max_tokens",
                    "stream",
                    "extra_body",
                    "parallel_tool_calls",
                    "temperature",
                    "top_p",
                )
                if key in kwargs
            }
            name = f"sdk-request-{len(report['wire_trace_files']) + 1}.json"
            (output / name).write_text(json.dumps(trace, ensure_ascii=False, indent=2))
            report["wire_trace_files"].append(name)
            return sdk_create(**kwargs)

        resolved.generator._client.chat.completions.create = traced_create
        workspace = open_workspace(root)
        read = create_read_file_tool(workspace)

        async def measured_read(arguments):
            start = time.perf_counter()
            try:
                await asyncio.sleep(0.3)
                value = read.run(arguments)
                return await value if inspect.isawaitable(value) else value
            finally:
                spans.append({"path": arguments["path"], "start": start, "end": time.perf_counter()})

        tools = [
            replace(read, run=measured_read),
            create_apply_patch_tool(workspace),
            create_run_command_tool(workspace),
        ]
        thread_id = None
        for phase, task in enumerate(PHASES, 1):
            before_spans = len(spans)
            async with await Session.open(
                workspace=root,
                database=output / "session.db",
                thread_id=thread_id,
                model=GatewayHarnessModel(
                    model_id="deepseek-flash", resolved=resolved, instructions=coding_instructions(root)
                ),
                model_binding={"model_id": "deepseek-flash"},
                tools={tool.definition.name: tool for tool in tools},
                completion_gate=RecordAnswer(),
                tool_execution_context=ToolExecutionContext(
                    workspace_root=root,
                    cwd=root,
                    allow_write_tools=True,
                    allow_execute_tools=True,
                    max_parallel_calls=3,
                ),
                max_steps=MAX_STEPS,
                max_tokens_total=TOKEN_BUDGET,
            ) as session:
                thread_id = session.thread_id
                session.context_manager = RolloutContextManager(session.store, max_total_bytes=BYTE_LIMIT)
                result = await asyncio.wait_for(session.submit(task), timeout=360)
                info = evidence(session.store, result.turn_id)
                oracle = subprocess.run(
                    [sys.executable, str(output / "oracle.py"), str(root / "ledger.py"), str(phase)],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                phase_spans = spans[before_spans:]
                document_spans = [s for s in phase_spans if s["path"].endswith(".md")]
                overlap = max(
                    (
                        sum(s["start"] <= p < s["end"] for s in document_spans)
                        for p in (s["start"] for s in document_spans)
                    ),
                    default=0,
                )
                checks = {
                    "oracle_passed": oracle.returncode == 0,
                    "completed": info["turn_status"] == "completed",
                    "semantic_compaction": info["summary_calls"] > 0
                    and any(
                        c.get("algorithm_revision")
                        in {"semantic-compaction-v3", "semantic-compaction-v4", "semantic-compaction-v5"}
                        for c in info["compactions"]
                    ),
                    "parallel_reads": overlap >= 3,
                    "specs_unchanged": all(
                        hashlib.sha256((root / n).read_bytes()).hexdigest() == h for n, h in hashes.items()
                    ),
                    "verified_rollout": info["rollout_valid"],
                    "model_ran_verification": verification_succeeded(
                        session.store.list_items(result.turn_id),
                        session.store.list_tool_operations(result.turn_id),
                    ),
                }
                (output / f"ledger-phase-{phase}.py").write_text((root / "ledger.py").read_text())
                report["phases"].append(
                    {
                        "phase": phase,
                        "checks": checks,
                        "oracle_stdout": oracle.stdout,
                        "oracle_stderr": oracle.stderr,
                        "read_intervals": phase_spans,
                        **info,
                    }
                )
                report["pass"] = len(report["phases"]) == 2 and all(all(p["checks"].values()) for p in report["phases"])
                (output / "report.json").write_text(json.dumps(report, indent=2))
                print(json.dumps({"phase": phase, "checks": checks, "report": str(output / "report.json")}), flush=True)
                if info["turn_status"] != "completed":
                    break
    except Exception as exc:
        report["pass"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"error": report["error"], "report": str(output / "report.json")}), flush=True)
    finally:
        control.close()


if __name__ == "__main__":
    asyncio.run(main())

"""Replay summary planning on a SQLite backup; block all model dispatch."""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from agent_runtime.builtin.generic import coding_instructions
from agent_runtime.harness import GatewayHarnessModel, RolloutContextManager, RolloutStore, TurnExecutor
from agent_runtime.harness.context_recall import create_context_recall_tool
from agent_runtime.model_definition import ProviderOptionsDefinition, ThinkingOptionsDefinition
from agent_runtime.models import ModelControlPlane
from agent_runtime.tools.builtins.filesystem import create_apply_patch_tool, create_read_file_tool
from agent_runtime.tools.builtins.shell import create_run_command_tool
from agent_runtime.workspace import open_workspace
from evals.context_management.functional_live import RecordAnswer

parser = argparse.ArgumentParser(description="Offline replay of a failed Session summary plan; never dispatches.")
parser.add_argument("directory", type=Path)
parser.add_argument("--legacy-summary-input", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ.setdefault("DEEPSEEK_API_KEY", "offline-no-dispatch")
source = args.directory.resolve()
report = json.loads((source / "report.json").read_text())
tid = report["phases"][1]["turn_id"]
db = Path(tempfile.mkdtemp(prefix="praxis-plan-")) / "replay.db"
with sqlite3.connect(f"file:{source}/session.db?mode=ro", uri=True) as src, sqlite3.connect(db) as dst:
    src.backup(dst)
control = ModelControlPlane.from_config_file(Path("configs/models.yaml"), initial_model_id="deepseek-flash")
r = control.resolve("deepseek-flash")
r = replace(
    r,
    capabilities=replace(r.capabilities, max_output_tokens=4096),
    request_defaults=r.request_defaults.model_copy(
        update={
            "parallel_tool_calls": True,
            "provider_options": ProviderOptionsDefinition(thinking=ThinkingOptionsDefinition(type="disabled")),
        }
    ),
)


def no_network(*a, **kw):
    raise AssertionError("Network forbidden in offline replay")


r.generator._client.chat.completions.create = no_network
if args.legacy_summary_input:
    from agent_runtime.modeling.contracts import LLMCallStage

    original_budget = r.gateway._stage_budget

    def old_input_policy(stage):
        return original_budget(LLMCallStage.LLM_SUMMARIZE if stage == LLMCallStage.CONTEXT_COMPACTION else stage)

    r.gateway._stage_budget = old_input_policy
with RolloutStore(db) as store:
    t = store.read_turn(tid)
    ws = open_workspace(source / "workspace")
    ts = (
        create_read_file_tool(ws),
        create_apply_patch_tool(ws),
        create_run_command_tool(ws),
        create_context_recall_tool(store),
    )

    class Router:
        def select(self, **kw):
            return ts

    m = RolloutContextManager(store, max_total_bytes=14000)
    runner = TurnExecutor(
        thread_id=t.thread_id,
        store=store,
        model=GatewayHarnessModel(
            model_id="deepseek-flash", resolved=r, instructions=coding_instructions(source / "workspace")
        ),
        context_manager=m,
        completion_gate=RecordAnswer(),
        tool_router=Router(),
    )
    result = {"network_calls": 0, "source": str(source), "remaining": store.read_budget_state(tid).remaining("tokens")}

    class PlanMeasuredError(Exception):
        pass

    async def capture_plan(*a, **kw):
        result["plan"] = dict(kw["plan"])
        raise PlanMeasuredError("Stopped before summary operation creation/dispatch")

    runner._semantic_summary = capture_plan

    def trace(frame, event, arg):
        if frame.f_code.co_name == "_prepare_compacted_step" and event == "exception":
            d = frame.f_locals
            if "fixed" in d:
                result.update(
                    {
                        k: d[k]
                        for k in [
                            "fixed",
                            "count",
                            "cap",
                            "continuation_input_limit",
                            "summary_input_limit",
                            "byte_limit",
                            "base_tokens",
                            "output",
                            "remaining",
                        ]
                        if k in d
                    }
                )
                result["leaves"] = [p.resource_request.input_tokens for p in d["leaves"]]
                result["source_bytes"] = len(d["source"].encode())
                result["history_entries"] = len(json.loads(d["source"])["history"])
        return trace

    sys.settrace(trace)
    try:
        asyncio.run(runner._prepare_compacted_step(runner.restore_turn_context(tid), step=2))
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        sys.settrace(None)
    result["legacy_summary_input"] = args.legacy_summary_input
    if "fixed" in result:
        result["required_tokens"] = result["fixed"] + result["count"] * result["cap"]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.output)
control.close()

"""Opt-in live, synthetic functional acceptance; keeps the databases and workspace evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agent_runtime.builtin.generic import coding_instructions
from agent_runtime.harness import (
    CompletionDecision,
    GatewayHarnessModel,
    RolloutContextManager,
    RolloutStore,
    TurnExecutor,
)
from agent_runtime.harness.context_recall import create_context_recall_tool
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
from agent_runtime.harness.tool_router import StaticToolRouter
from agent_runtime.model_definition import ProviderOptionsDefinition, ThinkingOptionsDefinition
from agent_runtime.modeling.gateway import model_request_input_text
from agent_runtime.models import ModelControlPlane
from agent_runtime.tools.builtins.filesystem import create_apply_patch_tool, create_read_file_tool
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.workspace import open_workspace
from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed


class RecordAnswer:
    """The external oracle, not this gate or the runtime status, decides acceptance."""

    def evaluate(self, proposal):
        return CompletionDecision(action="accept", reason="Answer recorded; independent functional oracle follows.")


def parsed(answer):
    try:
        text = answer or ""
        return json.loads(text[text.index("{") : text.rindex("}") + 1])
    except (ValueError, TypeError):
        return {}


def runner(store, thread, model, root, tools=(), byte_limit=5000):
    tool_map = {tool.definition.name: tool for tool in (*tools, create_context_recall_tool(store))}
    return TurnExecutor(
        thread_id=thread.thread_id,
        store=store,
        model=model(root),
        context_manager=RolloutContextManager(store, max_total_bytes=byte_limit),
        completion_gate=RecordAnswer(),
        tool_router=StaticToolRouter(tool_map),
        tool_orchestrator=ToolOrchestrator(
            store=store,
            tools=tool_map,
            execution_context=ToolExecutionContext(
                workspace_root=root, cwd=root, allow_write_tools=True, max_parallel_calls=3
            ),
        ),
        max_steps=16,
    )


def binding():
    return {
        "model_id": "deepseek-flash",
        "model_step_budget": 16,
        "model_token_budget_total": 100000,
        "tool_execution_policy": {"max_parallel_calls": 3},
    }


def evidence(store, turn_id):
    operations = store.list_model_operations(turn_id)
    return {
        "turn_id": turn_id,
        "turn_status": store.read_turn(turn_id).status,
        "terminal_reason": store.read_turn(turn_id).terminal_reason_code,
        "usage": asdict(store.read_budget_state(turn_id).used),
        "rollout_valid": store.verify().valid,
        "summary_calls": sum(op.request_ref.get("purpose") == "context_summary" for op in operations),
        "model_operation_statuses": [op.status for op in operations],
        "tools": [
            {"name": op.tool_name, "status": op.status, "arguments_digest": op.arguments_digest}
            for op in store.list_tool_operations(turn_id)
        ],
        "compactions": [dict(item.payload) for item in store.list_items(turn_id) if item.kind == "context_compaction"],
    }


async def repeated_history(model, directory):
    root = directory / "history-workspace"
    root.mkdir()
    database = directory / "history.db"
    results = []
    thread_id = None
    for index, ttl in enumerate((17, 23, 31)):
        with RolloutStore(database) as store:
            thread = store.create_thread(workspace=root) if thread_id is None else store.read_thread(thread_id)
            thread_id = thread.thread_id
            turn = store.start_turn(
                thread_id=thread_id,
                user_message=(
                    "Report the CURRENT decision from history. Output only JSON with ttl (integer), reason (string), "
                    "migration_applied (boolean), implementation_applied (boolean), "
                    'verification ("pending" or "passed"). '
                    "A decision is not implementation. Do not invent execution evidence."
                ),
                binding_manifest=binding(),
            )
            texts = [
                "Database migration REJECTED: legacy clients require the old schema. No migration was applied.",
                "Routine inspection: unchanged, no code edit and no test executed. " * 160,
                f"Latest user-approved decision supersedes all older TTL values: TTL={ttl} seconds because "
                f"upstream invalidation takes {ttl - 1} seconds. This is ONLY a decision: implementation is NOT done.",
                "Restart durability verification is still PENDING; "
                "it must prove journal persistence before acknowledgment.",
            ]
            for text in texts:
                seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": text})
            executor = runner(store, thread, model, root)
            result = await executor.run_turn(executor.restore_turn_context(turn.turn_id), start_step=1)
            answer = parsed(result.answer)
            checks = {
                "latest_decision": answer.get("ttl") == ttl,
                "reason": str(ttl - 1) in str(answer.get("reason")),
                "no_invented_migration": answer.get("migration_applied") is False,
                "no_invented_implementation": answer.get("implementation_applied") is False,
                "pending_stays_pending": answer.get("verification") == "pending",
            }
            info = evidence(store, turn.turn_id)
            checks["actually_semantically_compacted"] = info["summary_calls"] > 0 and bool(info["compactions"])
            # This small reporting task must converge, not merely eventually
            # answer after repeatedly rereading the same immutable records.
            checks["bounded_context_work"] = (
                info["summary_calls"] <= 8
                and len(info["model_operation_statuses"]) - info["summary_calls"] <= 4
            )
            results.append({"round": index + 1, "answer": result.answer, "checks": checks, **info})
            if result.status != "completed":
                break
    return {"pass": len(results) == 3 and all(all(r["checks"].values()) for r in results), "rounds": results}


async def recall_history(model, directory, *, hidden_detail=False):
    root = directory / "recall-workspace"
    root.mkdir()
    with RolloutStore(directory / "recall.db") as store:
        thread = store.create_thread(workspace=root)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message=(
                "Find the exact calibration code for sensor Helios in the earlier archived tool output. "
                'Use read_context to inspect original evidence; do not guess. Return JSON {"code": "..."}.'
            ),
            binding_manifest=binding(),
        )
        for index in range(4):
            seed(
                store,
                turn_id=turn.turn_id,
                kind="model_response",
                payload={
                    "text": "Historical sensor log read",
                    "tool_calls": [{"id": f"old-{index}", "name": "read_log", "arguments": {}}],
                },
            )
            content = (
                "Routine calibration sample; nothing changed. " * (600 if hidden_detail else 120)
                + "\nSensor Helios exact calibration code: HX-7294-KAPPA\n"
                + "Routine sample. " * 1800
                if index == 0
                else "Recent unrelated sample; no calibration code here."
            )
            seed(
                store,
                turn_id=turn.turn_id,
                kind="tool_result",
                payload={"tool_call_id": f"old-{index}", "model_content": content},
            )
        first_request_hid_code = None

        def checked_model(workspace):
            instance = model(workspace)
            dispatch = instance.dispatch

            async def checked_dispatch(prepared, **kwargs):
                nonlocal first_request_hid_code
                if first_request_hid_code is None and prepared.request_ref["purpose"] == "agent_step":
                    wire = model_request_input_text(
                        prepared.dispatch_payload.request,
                        provider="deepseek",
                        supports_native_tools=True,
                    )
                    first_request_hid_code = "HX-7294-KAPPA" not in wire
                    if hidden_detail and not first_request_hid_code:
                        raise AssertionError("Hidden recall fixture leaked the target into the first request")
                return await dispatch(prepared, **kwargs)

            instance.dispatch = checked_dispatch
            return instance

        executor = runner(store, thread, checked_model, root, byte_limit=18000)
        result = await executor.run_turn(executor.restore_turn_context(turn.turn_id), start_step=1)
        info = evidence(store, turn.turn_id)
        calls = sum(op["name"] == "read_context" and op["status"] == "succeeded" for op in info["tools"])
        checks = {
            "exact_code": parsed(result.answer).get("code") == "HX-7294-KAPPA",
            "actual_recall_calls": calls > 0,
            "actually_archived": bool(info["compactions"]),
        }
        if hidden_detail:
            checks["answer_absent_before_recall"] = first_request_hid_code is True
        return {"pass": all(checks.values()), "answer": result.answer, "checks": checks, **info}


async def recall_hidden_history(model, directory):
    """Separate stronger fixture; leave the original recall case and result intact."""
    return await recall_history(model, directory, hidden_detail=True)


async def parallel_edit(model, directory):
    root = directory / "edit-workspace"
    root.mkdir()
    for name, value in {"alpha.txt": 19, "beta.txt": 37, "gamma.txt": 61}.items():
        (root / name).write_text(str(value) + "\n")
    (root / "result.json").write_text('{"total": 0}\n')
    workspace = open_workspace(root)
    original = create_read_file_tool(workspace)
    spans = []

    async def measured_read(arguments):
        start = time.perf_counter()
        try:
            # Controlled read latency makes actual overlap measurable; content is read by the production tool.
            await asyncio.sleep(0.3)
            value = original.run(arguments)
            return await value if inspect.isawaitable(value) else value
        finally:
            spans.append({"path": arguments["path"], "start": start, "end": time.perf_counter()})

    tools = (replace(original, run=measured_read), create_apply_patch_tool(workspace))
    with RolloutStore(directory / "edit.db") as store:
        thread = store.create_thread(workspace=root)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message=(
                "Read alpha.txt, beta.txt and gamma.txt together using three read_file calls in ONE response. "
                "Calculate their sum. Update result.json total from 0 to the sum "
                "without changing the three input files. "
                "Read result.json once after the write to verify it, then finish. Do not invent the inputs."
            ),
            binding_manifest=binding(),
        )
        executor = runner(store, thread, model, root, tools=tools, byte_limit=100000)
        result = await executor.run_turn(executor.restore_turn_context(turn.turn_id), start_step=1)
        inputs = [span for span in spans if span["path"] in {"alpha.txt", "beta.txt", "gamma.txt"}]
        simultaneous = max(
            (sum(s["start"] <= point < s["end"] for s in inputs) for point in [s["start"] for s in inputs]), default=0
        )
        checks = {
            "correct_file": json.loads((root / "result.json").read_text()).get("total") == 117,
            "inputs_unchanged": all(
                (root / name).read_text() == f"{value}\n"
                for name, value in {"alpha.txt": 19, "beta.txt": 37, "gamma.txt": 61}.items()
            ),
            "three_actually_overlapped": simultaneous == 3,
            "post_write_read": any(s["path"] == "result.json" for s in spans),
        }
        return {
            "pass": all(checks.values()),
            "checks": checks,
            "answer": result.answer,
            "maximum_simultaneous_reads": simultaneous,
            "read_intervals": spans,
            **evidence(store, turn.turn_id),
        }


async def main():
    args = argparse.ArgumentParser()
    args.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    args.add_argument("--prompt", choices=("production", "minimal"), default="production")
    args.add_argument("--case", choices=("repeated_compaction_restart", "archived_detail_recall",
                                       "parallel_read_edit_verify", "archived_detail_recall_hidden"))
    options = args.parse_args()
    mode = options.thinking
    load_dotenv()
    output = Path("evals/context_management") / f"deepseek-functional-{datetime.now():%Y%m%d-%H%M%S}-{mode}"
    output.mkdir()
    control = ModelControlPlane.from_config_file(Path("configs/models.yaml"), initial_model_id="deepseek-flash")
    try:
        resolved = control.resolve("deepseek-flash")
        resolved = replace(
            resolved,
            capabilities=replace(resolved.capabilities, max_output_tokens=4096),
            request_defaults=resolved.request_defaults.model_copy(
                update={
                    "parallel_tool_calls": True,
                    "provider_options": ProviderOptionsDefinition(thinking=ThinkingOptionsDefinition(type=mode)),
                }
            ),
        )

        def model(root):
            return GatewayHarnessModel(
                model_id="deepseek-flash",
                resolved=resolved,
                instructions=coding_instructions(root)
                if options.prompt == "production"
                else ("Follow the task and use tool evidence. Never claim unexecuted work or fabricate results.",),
            )

        report = {
            "model": "deepseek-flash",
            "thinking": mode,
            "prompt": options.prompt,
            "synthetic_only": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "selected_case": options.case,
            "cases": {},
        }
        (output / "frozen.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        for name, case in (
            ("repeated_compaction_restart", repeated_history),
            ("archived_detail_recall", recall_history),
            ("parallel_read_edit_verify", parallel_edit),
            ("archived_detail_recall_hidden", recall_hidden_history),
        ):
            if options.case is not None and options.case != name:
                continue
            if options.case is None and name == "archived_detail_recall_hidden":
                continue
            try:
                report["cases"][name] = await asyncio.wait_for(case(model, output), timeout=240)
            except Exception as exc:
                report["cases"][name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(
                json.dumps(
                    {"case": name, "pass": report["cases"][name]["pass"], "report": str(output / "report.json")}
                ),
                flush=True,
            )
    finally:
        control.close()


if __name__ == "__main__":
    asyncio.run(main())

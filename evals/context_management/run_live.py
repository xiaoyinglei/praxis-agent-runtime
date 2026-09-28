import asyncio
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from agent_runtime.harness import (
    CompletionDecision,
    GatewayHarnessModel,
    RolloutContextManager,
    RolloutStore,
    TurnExecutor,
)
from agent_runtime.models import ModelControlPlane
from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed

load_dotenv(Path.cwd() / ".env")


class Accept:
    def evaluate(self, proposal):
        return CompletionDecision(action="accept", reason="synthetic summary evaluation")


async def main():
    cp = ModelControlPlane.from_config_file(Path("configs/models.yaml"), initial_model_id="openai/gpt-oss-120b")
    try:
        resolved = cp.resolve("openai/gpt-oss-120b")
        resolved = replace(resolved, capabilities=replace(resolved.capabilities, max_output_tokens=2048))
        model = GatewayHarnessModel(
            model_id="openai/gpt-oss-120b", resolved=resolved, instructions=("Answer briefly using recorded evidence.",)
        )
        with TemporaryDirectory(prefix="praxis-summary-live-") as directory:
            root = Path(directory)
            with RolloutStore(root / "rollout.db") as store:
                thread = store.create_thread(workspace=root)
                turn = store.start_turn(
                    thread_id=thread.thread_id,
                    user_message=(
                        "Report the earlier cache decision, its reason, the rejected migration and why, "
                        "and the remaining verification."
                    ),
                    binding_manifest={
                        "model_id": "openai/gpt-oss-120b",
                        "model_step_budget": 1,
                        "model_token_budget_total": 30000,
                    },
                )
                for text in (
                    "Decision: cache TTL is 17 seconds because invalidation events arrive within 16 seconds.",
                    "Inspection completed. No files changed. " * 120,
                    "Rejected database migration: legacy clients depend on the existing schema; keep it unchanged.",
                    "Repeated file inspection completed without new findings. " * 120,
                    (
                        "Pending verification: restart recovery must prove that journal entries "
                        "are persisted BEFORE acknowledgment."
                    ),
                ):
                    seed(store, turn_id=turn.turn_id, kind="agent_message", payload={"text": text})
                runner = TurnExecutor(
                    thread_id=thread.thread_id,
                    store=store,
                    model=model,
                    context_manager=RolloutContextManager(store, max_total_bytes=5000),
                    completion_gate=Accept(),
                )
                result = await runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1)
                summaries = [
                    i.payload["text"] for i in store.list_items(turn.turn_id) if i.kind == "context_summary_response"
                ]
                output = {
                    "model": "openai/gpt-oss-120b",
                    "provider": "groq",
                    "synthetic_only": True,
                    "input_limit_bytes": 5000,
                    "output_limit_tokens": 2048,
                    "turn_token_budget": 30000,
                    "reason": store.read_turn(turn.turn_id).terminal_reason_code,
                    "errors": [
                        r.payload.get("reason")
                        for r in store.list_records(thread.thread_id)
                        if r.record_type == "model_attempt_rejected"
                    ],
                    "status": result.status,
                    "answer": result.answer,
                    "summaries": summaries,
                    "operations": len(store.list_model_operations(turn.turn_id)),
                    "usage": str(store.read_budget_state(turn.turn_id).used),
                    "verified": store.verify().valid,
                }
                Path("evals/context_management/semantic-live-latest.json").write_text(
                    json.dumps(output, ensure_ascii=False, indent=2)
                )
                print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        cp.close()


if __name__ == "__main__":
    asyncio.run(main())

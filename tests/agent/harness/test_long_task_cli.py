"""Public CLI, real tools and separate processes; no network/model credentials."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent_runtime.core.messages import ModelToolCall, StopReason, ToolUseResult
from agent_runtime.harness import RuntimeStore
from agent_runtime.harness.protocol import item_text
from agent_runtime.modeling.contracts import ContextCapacityPolicy
from agent_runtime.modeling.gateway import LLMGateway
from tests.agent.harness.test_context_capacity_cli_replay import ReplayModel, SyntheticProvider
from tests.agent.harness.test_model_adapter import CharacterAccounting, _resolved_model


def _main():
    from agent_runtime import cli
    from agent_runtime.harness import compaction_jobs

    recovering = sys.argv[1] == "recover"
    control = Path(os.environ["PRAXIS_TEST_CONTROL"])
    failing = os.environ.get("PRAXIS_TEST_FAILURE") == "1"

    class Provider(SyntheticProvider):
        def _turn(self, *, messages, **kwargs):
            text = json.dumps(messages, ensure_ascii=False)
            assert "Keep API_NAME unchanged" in text
            assert "Verify all three changes" in text
            summary = "Create a concise context checkpoint for another LLM" in text
            with (control / "calls.jsonl").open("a") as log:
                log.write(json.dumps({"summary": summary, "text": text, "tools": bool(kwargs.get("tools"))}) + "\n")
            if summary:
                assert not kwargs.get("tools")
                assert "DECISION_A: preserve the ABI" in text
                if failing:
                    return ToolUseResult(raw_stop_reason="stop", stop_reason=StopReason.END_TURN, text="x" * 34000)
                stage = 2 if "STAGE = 2" in text or "stage=2" in text else 1
                return ToolUseResult(
                    raw_stop_reason="stop",
                    stop_reason=StopReason.END_TURN,
                    text=f"CHECKPOINT stage={stage}. DECISION_A: preserve the ABI. "
                    "Applied earlier changes; final stage still pending. Original EVIDENCE_KEY is archived. "
                    "Recall the original evidence before the final change, then verify all three changes.",
                )
            self.agent_calls += 1
            call_id = f"{'reopen' if recovering else 'run'}-{self.agent_calls}"
            if recovering:
                assert "stage=2" in text
                assert "DECISION_A: preserve the ABI" in text
                if self.agent_calls == 1:
                    assert Path("config.py").read_text() == 'API_NAME = "stable"\nSTAGE = 2\n'
                    name, arguments = (
                        "read_context",
                        {"item_id": "history", "query": "EVIDENCE_KEY=3", "max_chars": 500},
                    )
                elif self.agent_calls == 2:
                    result = next(m["content"] for m in messages if m.get("tool_call_id") == "reopen-1")
                    identity = re.search(r"item_[0-9a-f]{32}", result).group()
                    name, arguments = "read_context", {"item_id": identity, "query": "EVIDENCE_KEY=3", "max_chars": 500}
                elif self.agent_calls == 3:
                    assert any(
                        "EVIDENCE_KEY=3" in m.get("content", "") and m.get("tool_call_id") == "reopen-2"
                        for m in messages
                    )
                    name, arguments = (
                        "apply_patch",
                        {"file_path": "config.py", "old_string": "STAGE = 2", "new_string": "STAGE = 3"},
                    )
                elif self.agent_calls == 4:
                    name, arguments = "read_file", {"path": "config.py"}
                else:
                    verified = next(m["content"] for m in messages if m.get("tool_call_id") == "reopen-4")
                    assert json.loads(verified)["structured_content"]["content"] == 'API_NAME = "stable"\nSTAGE = 3\n'
                    return ToolUseResult(
                        raw_stop_reason="stop",
                        stop_reason=StopReason.END_TURN,
                        text="Three changes verified against original evidence.",
                    )
            elif self.agent_calls == 1:
                name, arguments = (
                    "apply_patch",
                    {"file_path": "config.py", "old_string": "STAGE = 0", "new_string": "STAGE = 1"},
                )
            elif self.agent_calls == 2:
                name, arguments = "read_file", {"path": "archive.txt"}
            elif "CHECKPOINT stage=1" in text and "STAGE = 2" not in text:
                assert Path("config.py").read_text().endswith("STAGE = 1\n")
                name, arguments = (
                    "apply_patch",
                    {"file_path": "config.py", "old_string": "STAGE = 1", "new_string": "STAGE = 2"},
                )
            else:
                assert self.agent_calls < 30
                name, arguments = "read_file", {"path": f"note{self.agent_calls}.txt"}
            return ToolUseResult(
                raw_stop_reason="tool_calls",
                stop_reason=StopReason.TOOL_USE,
                text="DECISION_A: preserve the ABI" if self.agent_calls == 1 else "Continue the unfinished work.",
                tool_calls=[ModelToolCall(id=call_id, name=name, arguments=arguments)],
            )

    provider = Provider("long-task")
    accounting = CharacterAccounting()
    gateway = LLMGateway(
        generator=provider,
        token_accounting=accounting,
        model_context_tokens=1048576,
        context_policy=ContextCapacityPolicy(max_messages=16),
    )
    model = ReplayModel(
        model_id="synthetic-local",
        resolved=_resolved_model(
            gateway=gateway,
            generator=provider,
            token_accounting=accounting,
            context_window_tokens=1048576,
        ),
        instructions=("Use actual execution evidence.",),
    )
    create_facade = cli._create_agent_facade

    def local_facade(**options):
        agent = create_facade(**options)
        agent._model_factory = lambda: model
        return agent

    cli._create_agent_facade = local_facade
    install = compaction_jobs.install_candidate

    def crash_after_second_checkpoint(store, **kwargs):
        install(store, **kwargs)
        boundaries = [
            i
            for i in store.list_items(kwargs["turn_id"])
            if i.kind == "context_compaction" and i.payload.get("algorithm_revision") == "semantic-compaction-v7"
        ]
        if not recovering and len(boundaries) == 2:
            # The whole transaction has committed, including the continuation.
            os._exit(73)

    compaction_jobs.install_candidate = crash_after_second_checkpoint
    finish_job = compaction_jobs.finish_job

    def crash_after_failed_job(store, **kwargs):
        finish_job(store, **kwargs)
        if failing and not recovering and kwargs["status"] == "failed":
            os._exit(74)

    compaction_jobs.finish_job = crash_after_failed_job
    cli.agent_app()


def test_cli_multiple_semantic_checkpoints_then_reopens_unfinished_turn(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config.py").write_text('API_NAME = "stable"\nSTAGE = 0\n')
    (workspace / "archive.txt").write_text("EVIDENCE_KEY=3\n" + "original evidence " * 400)
    for index in range(3, 30):
        (workspace / f"note{index}.txt").write_text(f"Inspection {index} completed. Keep the interface stable.\n")
    database = tmp_path / "rollout.sqlite"
    entry = [sys.executable, "-c", f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r})['_main']()"]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3]), "PRAXIS_TEST_CONTROL": str(tmp_path)}

    def invoke(*args):
        return subprocess.run(
            [*entry, *args, "--checkpoint-db", str(database)],
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )

    first = invoke(
        "run",
        "Keep API_NAME unchanged. Verify all three changes before finishing.",
        "--non-interactive",
        "--disable-workspace-mcp",
        "--allow-write-tools",
    )
    assert first.returncode == 73, first.stdout + first.stderr
    assert (workspace / "config.py").read_text().endswith("STAGE = 2\n")
    with RuntimeStore.open_sync(database) as store:
        [turn] = store.list_turns()
        boundaries = [i for i in store.list_items(turn.turn_id) if i.kind == "context_compaction"]
        assert len(boundaries) == 2
        assert all(i.payload["capacity_diagnostics"]["summary_calls"] >= 1 for i in boundaries)
        patches = [
            i
            for i in store.list_items(turn.turn_id)
            if i.kind == "tool_result" and i.payload["tool_name"] == "apply_patch"
        ]
        assert len(patches) == 2
        assert patches[0].sequence < boundaries[0].sequence < patches[1].sequence < boundaries[1].sequence
        assert store.list_model_operations(turn.turn_id)[-1].status == "prepared"
        original = next(
            i
            for i in store.list_items(turn.turn_id)
            if i.kind == "tool_result" and i.payload.get("tool_call_id") == "run-2"
        )

    second = invoke("recover", turn.turn_id, "--action", "continue")
    assert second.returncode == 0, second.stdout + second.stderr
    with RuntimeStore.open_sync(database) as store:
        [finished] = store.list_turns()
        assert finished.turn_id == turn.turn_id and finished.status == "completed"
        assert item_text(store.read_turn_answer(turn.turn_id)) == "Three changes verified against original evidence."
        assert store.read_item(original.item_id).payload == original.payload
        recalls = [
            i
            for i in store.list_items(turn.turn_id)
            if i.kind == "tool_result" and i.payload["tool_name"] == "read_context"
        ]
        assert len(recalls) == 2 and all(not i.payload["is_error"] for i in recalls)
        assert original.item_id in str(recalls[-1].payload) and "EVIDENCE_KEY=3" in str(recalls[-1].payload)
        assert store.verify().valid
    assert (workspace / "config.py").read_text() == 'API_NAME = "stable"\nSTAGE = 3\n'
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    summaries = [call for call in calls if call["summary"]]
    assert len(summaries) >= 2 and all(not call["tools"] for call in summaries)
    assert "CHECKPOINT stage=1" in summaries[1]["text"]


def test_cli_does_not_dispatch_failed_summary_again_after_process_exit(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config.py").write_text('API_NAME = "stable"\nSTAGE = 0\n')
    (workspace / "archive.txt").write_text("EVIDENCE_KEY=3\n")
    for index in range(3, 30):
        (workspace / f"note{index}.txt").write_text(f"Evidence {index}.\n")
    database = tmp_path / "failed.sqlite"
    entry = [sys.executable, "-c", f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r})['_main']()"]
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "PRAXIS_TEST_CONTROL": str(tmp_path),
        "PRAXIS_TEST_FAILURE": "1",
    }

    def invoke(*args):
        return subprocess.run(
            [*entry, *args, "--checkpoint-db", str(database)],
            cwd=workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )

    first = invoke(
        "run",
        "Keep API_NAME unchanged. Verify all three changes before finishing.",
        "--non-interactive",
        "--disable-workspace-mcp",
        "--allow-write-tools",
    )
    assert first.returncode == 74, first.stdout + first.stderr
    with RuntimeStore.open_sync(database) as store:
        [turn] = store.list_turns()
        summaries = [
            op for op in store.list_model_operations(turn.turn_id) if op.request_ref["purpose"] == "context_summary"
        ]
        assert len(summaries) == 1
        used = store.read_budget_state(turn.turn_id).used
    second = invoke("recover", turn.turn_id, "--action", "continue")
    assert second.returncode != 0
    with RuntimeStore.open_sync(database) as store:
        assert store.read_turn(turn.turn_id).terminal_reason_code == "context_summary_previously_failed", (
            second.stdout + second.stderr
        )[-4000:]
        assert store.read_budget_state(turn.turn_id).used == used
        assert not any(i.kind == "context_compaction" for i in store.list_items(turn.turn_id))
        assert store.verify().valid
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert sum(call["summary"] for call in calls) == 1

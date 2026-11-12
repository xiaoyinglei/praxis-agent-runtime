"""Controlled public CLI evidence with synthetic provider and real local tools."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from agent_runtime.cli import agent_app
from agent_runtime.core.messages import ModelToolCall, StopReason, ToolUseResult
from agent_runtime.harness import GatewayHarnessModel, RuntimeStore, turn_ops
from agent_runtime.harness.protocol import item_text
from agent_runtime.modeling.contracts import ContextCapacityPolicy, LLMCallStage, LLMProviderResult, LLMStageBudget
from agent_runtime.modeling.gateway import LLMGateway
from tests.agent.harness.test_model_adapter import CharacterAccounting, _resolved_model


class SyntheticProvider:
    def __init__(self, task):
        self.task = task
        self.agent_calls = 0
        self.agent_tools = []
        self.summary_sources = []
        self.summary_tools = []

    def generate_with_tools(self, *, messages, **kwargs):
        turn = self._turn(messages=messages, **kwargs)
        return LLMProviderResult(
            value={
                "choices": [
                    {
                        "message": {
                            "content": turn.text,
                            "tool_calls": [
                                {
                                    "id": call.id,
                                    "type": "function",
                                    "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
                                }
                                for call in turn.tool_calls
                            ],
                        },
                        "finish_reason": "tool_calls" if turn.tool_calls else "stop",
                    }
                ]
            }
        )

    def _turn(self, *, messages, tools=None, **kwargs):
        text = json.dumps(messages, ensure_ascii=False)
        if (
            "Summarize the supplied execution history faithfully." in text
            or "Create a concise context checkpoint for another LLM that will resume this task." in text
            or "Return only a checkpoint; do not continue the task or call tools." in text
        ):
            self.summary_sources.append(text)
            self.summary_tools.append(tools)
            facts = [fact for fact in ["GOLD_WEB_FACT=17", "GOLD_CODE_FACT=17", "GOLD_DIALOG_FACT=17"] if fact in text]
            return ToolUseResult(
                raw_stop_reason="synthetic",
                text="Verified synthetic evidence: " + ", ".join(facts),
                stop_reason=StopReason.END_TURN,
            )
        self.agent_calls += 1
        self.agent_tools.append(tools)
        if self.task == "web" and self.agent_calls <= 2:
            arguments = {
                "url": "https://example.com/frozen",
                "render": False,
                "max_lines": 50,
                "max_bytes": 16000,
                "max_tokens": 16000,
            }
            if self.agent_calls == 2:
                source_id = re.search(r"artifact_[0-9a-f]{32}", text).group()
                arguments = {
                    "source_id": source_id,
                    "start_line": 1,
                    "render": False,
                    "max_lines": 50,
                    "max_bytes": 16000,
                    "max_tokens": 16000,
                }
            return ToolUseResult(
                raw_stop_reason="synthetic",
                text="Explicitly recheck the same saved evidence.",
                stop_reason=StopReason.TOOL_USE,
                tool_calls=[ModelToolCall(id=f"fetch-{self.agent_calls}", name="web_fetch", arguments=arguments)],
            )
        if self.task == "code" and self.agent_calls <= 3:
            name = "apply_patch" if self.agent_calls == 2 else "read_file"
            arguments = {"path": "sample.py"}
            if name == "apply_patch":
                arguments = {
                    "file_path": "sample.py",
                    "old_string": "# old evidence ",
                    "new_string": "VALUE = 17\n# old evidence ",
                }
            return ToolUseResult(
                raw_stop_reason="synthetic",
                text="Apply and recheck the synthetic change.",
                stop_reason=StopReason.TOOL_USE,
                tool_calls=[ModelToolCall(id=f"file-{self.agent_calls}", name=name, arguments=arguments)],
            )
        expected = f"GOLD_{self.task.upper() if self.task != 'code' else 'CODE'}_FACT=17"
        if self.task == "dialog":
            expected = "GOLD_DIALOG_FACT=17"
        self.final_context = text
        return ToolUseResult(
            raw_stop_reason="synthetic",
            text=expected if expected in text else "synthetic fact unavailable",
            stop_reason=StopReason.END_TURN,
        )


class ReplayModel(GatewayHarnessModel):
    def snapshot(self, *, thread_id, turn_id):
        return {
            "thread_id": thread_id,
            "turn_id": turn_id,
            "model_id": "synthetic-local",
            **(
                {"context_capacity_policy": self._resolved.gateway.context_policy.model_dump()}
                if self._resolved.gateway.context_policy is not None
                else {}
            ),
        }

    def ensure_available(self, binding, *, thread_id, turn_id):
        pass


@pytest.mark.parametrize("task", ["web", "code", "dialog"])
@pytest.mark.parametrize("mode", ["capacity", "legacy"])
def test_public_cli_controlled_context_replay(tmp_path, monkeypatch, task, mode):
    policy = ContextCapacityPolicy()
    provider = SyntheticProvider(task)
    accounting = CharacterAccounting()
    gateway = LLMGateway(
        generator=provider,
        token_accounting=accounting,
        model_context_tokens=1048576,
        stage_budgets={
            LLMCallStage.AGENT_STEP: LLMStageBudget(max_input_tokens=None, max_output_tokens=256),
            LLMCallStage.CONTEXT_COMPACTION: LLMStageBudget(max_input_tokens=None, max_output_tokens=256),
        },
        context_policy=policy if mode == "capacity" else None,
    )
    model = ReplayModel(
        model_id="synthetic-local",
        resolved=_resolved_model(
            gateway=gateway, generator=provider, token_accounting=accounting, context_window_tokens=1048576
        ),
        instructions=("Use only actual execution evidence.",),
    )
    from agent_runtime import cli as cli_module

    original_facade = cli_module._create_agent_facade

    def create_facade(**options):
        agent = original_facade(**options)
        monkeypatch.setattr(agent, "_model_factory", lambda: model)
        return agent

    monkeypatch.setattr(cli_module, "_create_agent_facade", create_facade)
    # Inject the same low projection guard to trigger both algorithms. Summary
    # fragments retain their frozen policy materialization bounds. Leave room
    # for a protected 12K work unit plus live facts; two such units still
    # exceed the 20K guard and exercise compaction in both modes.
    from agent_runtime.harness import services
    from agent_runtime.harness.context import RolloutContextManager

    monkeypatch.setattr(
        services, "RolloutContextManager", lambda store: RolloutContextManager(store, max_total_bytes=20000)
    )
    monkeypatch.chdir(tmp_path)
    database = tmp_path / "replay.sqlite"
    (tmp_path / "sample.py").write_text("# old evidence " + "x" * 12000 + "\n# GOLD_CODE_FACT=17\n")
    import agent_runtime.tools.web_http as web_http

    original_client = web_http.PublicWebClient
    body = "\n".join("line " + str(i) + " " + "x" * 250 for i in range(49)) + "\nGOLD_WEB_FACT=17"
    monkeypatch.setattr(
        web_http,
        "PublicWebClient",
        lambda **options: original_client(
            **options,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=body, headers={"content-type": "text/plain"})
            ),
        ),
    )
    args = [
        "run",
        "Use synthetic evidence; explicitly recheck the saved source.",
        "--checkpoint-db",
        str(database),
        "--non-interactive",
        "--disable-workspace-mcp",
        "--no-require-workspace-change",
        "--max-tokens-total",
        # CharacterAccounting counts serialized characters rather than model tokens.
        # This case tests context retention; cumulative-budget stops have separate cases.
        "500000",
        "--allow-write-tools",
    ]
    if task == "dialog":
        from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed

        with RuntimeStore.open_sync(database) as store:
            thread = turn_ops.create_thread(store, workspace=tmp_path)
            previous = turn_ops.start_turn(
                store, thread_id=thread.thread_id, user_message="Synthetic dialogue", binding_manifest={}
            )
            for _index in range(8):
                seed(store, turn_id=previous.turn_id, kind="agent_message", payload={"text": "x" * 2000})
            seed(store, turn_id=previous.turn_id, kind="agent_message", payload={"text": "GOLD_DIALOG_FACT=17"})
            turn_ops.complete_turn(store, turn_id=previous.turn_id, answer="Continue later.")
            args.extend(["--previous-turn-id", previous.turn_id])
    result = CliRunner().invoke(agent_app, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    expected = f"GOLD_{task.upper()}_FACT=17"
    preserves_fact = mode == "capacity" or task == "dialog"
    expect_summary = preserves_fact and (mode, task) != ("capacity", "code")
    if expect_summary:
        assert provider.summary_sources and expected in str(provider.summary_sources)
        if mode == "capacity":
            assert all(not tools for tools in provider.summary_tools)
            assert any(tools for tools in provider.agent_tools)
    else:
        if provider.summary_sources:
            assert expected in str(provider.summary_sources)
    if task == "code":
        assert (tmp_path / "sample.py").read_text().startswith("VALUE = 17\n# old evidence ")
    with RuntimeStore.open_sync(database) as store:
        turn = store.list_turns()[-1]
        items = store.list_items(turn.turn_id)
        compactions = [item for item in items if item.kind == "context_compaction"]
        assert compactions
        if mode == "capacity":
            expected_algorithm = "semantic-compaction-v7" if expect_summary else "tool-output-elision-v5"
            assert all(item.payload["algorithm_revision"] == expected_algorithm for item in compactions)
            if not expect_summary:
                assert not provider.summary_sources
        accepted = store.read_turn_answer(turn.turn_id)
        answer = item_text(accepted)
        assert accepted.kind == "model_response"
        assert not any(item.kind == "agent_message" for item in items)
        if preserves_fact:
            assert answer == expected
        else:
            assert (answer == expected) == (expected in provider.final_context)
        assert store.verify().valid
        tool_results = [item for item in items if item.kind == "tool_result"]
        assert all(not item.payload.get("is_error") for item in tool_results)
        if task == "web":
            assert len(tool_results) == 2  # explicit reread remains permitted
            assert store.read_resource_usage(turn.turn_id, "public_web_requests")["used"] == 1
        report = {
            "task": task,
            "provider": "synthetic-local",
            "cloud_calls": 0,
            "mode": mode,
            "effective_input_limit": 1047808,
            "fact_present": answer == expected,
            "agent_steps": provider.agent_calls,
            "summary_calls": len(provider.summary_sources),
            "tools": len(tool_results),
            "compaction_reasons": [
                item.payload.get("capacity_diagnostics", {}).get("trigger_reason", "legacy") for item in compactions
            ],
            "algorithms": [item.payload["algorithm_revision"] for item in compactions],
            "fetched_ranges": [
                [item.payload["structured_content"]["start_line"], item.payload["structured_content"]["next_line"]]
                for item in tool_results
                if item.payload.get("tool_name") == "web_fetch"
            ],
            "cli_output": result.output,
        }
        (tmp_path / "replay-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def _continuity_cli():
    """Subprocess fixture. Only the provider and the context threshold are synthetic."""
    from agent_runtime import cli, harness
    from agent_runtime.harness import services

    recovering = sys.argv[1] == "recover"
    semantic = os.environ.get("PRAXIS_TEST_SEMANTIC") == "1"

    class ContinuityProvider(SyntheticProvider):
        def _turn(self, *, messages, **kwargs):
            text = json.dumps(messages)
            assert "Keep API_NAME unchanged" in text
            assert "Finish by verifying both edits" in text
            if "Create a concise context checkpoint for another LLM" in text:
                assert semantic and not kwargs.get("tools")
                assert "ARCHIVED_FACT=42" in text
                assert Path("sample.py").read_text().splitlines()[1] == "FIRST = 1"
                return ToolUseResult(
                    raw_stop_reason="stop",
                    text="First edit committed. Apply the second edit and verify both. " + "进度已核实。" * 24,
                    stop_reason=StopReason.END_TURN,
                )
            self.agent_calls += 1
            if recovering:
                assert "ARCHIVED_FACT=42" not in text or self.agent_calls > 1
                if self.agent_calls == 1:
                    assert "patch-after-compaction" in text
                    if semantic:
                        assert "First edit committed." in text
                        return self._call(
                            "read_context", {"item_id": "history", "query": "ARCHIVED_FACT", "max_chars": 250}
                        )
                    [archived] = [message["content"] for message in messages if message.get("tool_call_id") == "run-2"]
                    item_id = re.search(r"read_context\(item_id='([^']+)'\)", archived).group(1)
                    name, arguments = "read_context", {"item_id": item_id, "query": "ARCHIVED_FACT", "max_chars": 80}
                elif self.agent_calls == 2:
                    assert "ARCHIVED_FACT=42" in text
                    name, arguments = "read_file", {"path": "sample.py"}
                else:
                    [verified] = [
                        json.loads(message["content"])["structured_content"]["content"]
                        for message in messages
                        if message.get("tool_call_id") == "reopen-2"
                    ]
                    assert verified == 'API_NAME = "stable"\nFIRST = 1\nSECOND = 42\n'
                    return ToolUseResult(
                        raw_stop_reason="stop", text="Both edits verified.", stop_reason=StopReason.END_TURN
                    )
            else:
                edit_step = 8 if semantic else 4
                if self.agent_calls == 1:
                    name = "apply_patch"
                    arguments = {"file_path": "sample.py", "old_string": "FIRST = 0", "new_string": "FIRST = 1"}
                elif self.agent_calls < edit_step:
                    name = "read_file"
                    path = {2: "archive.txt", edit_step - 1: "recent.txt"}.get(
                        self.agent_calls, f"note{self.agent_calls}.txt"
                    )
                    arguments = {"path": path}
                else:
                    assert self.agent_calls == edit_step
                    assert ("First edit committed." if semantic else "Tool output archived") in text
                    assert "ARCHIVED_FACT=42" not in text
                    assert "RECENT_FACT=42" in text
                    assert Path("sample.py").read_text().startswith('API_NAME = "stable"\nFIRST = 1')
                    name = "apply_patch"
                    arguments = {"file_path": "sample.py", "old_string": "SECOND = 0", "new_string": "SECOND = 42"}
            return self._call(name, arguments)

        def _call(self, name, arguments):
            call_id = f"{'reopen' if recovering else 'run'}-{self.agent_calls}"
            return ToolUseResult(
                raw_stop_reason="tool_calls",
                text="patch-after-compaction"
                if not recovering and name == "apply_patch" and self.agent_calls > 1
                else "",
                stop_reason=StopReason.TOOL_USE,
                tool_calls=[ModelToolCall(id=call_id, name=name, arguments=arguments)],
            )

    class ContinuityModel(ReplayModel):
        def prepare(self, request):
            if not recovering and request.purpose == "agent_step" and "SECOND = 42" in Path("sample.py").read_text():
                # The second write has committed. Exit before preparing the
                # next operation so recover has to project the persisted boundary.
                os._exit(73)
            return super().prepare(request)

    provider = ContinuityProvider("continuity")
    accounting = CharacterAccounting()
    gateway = LLMGateway(
        generator=provider,
        token_accounting=accounting,
        model_context_tokens=1048576,
        context_policy=ContextCapacityPolicy(),
    )
    model = ContinuityModel(
        model_id="synthetic-local",
        resolved=_resolved_model(
            gateway=gateway, generator=provider, token_accounting=accounting, context_window_tokens=1048576
        ),
        instructions=("Use only actual execution evidence.",),
    )
    create_facade = cli._create_agent_facade

    def local_facade(**options):
        agent = create_facade(**options)
        agent._model_factory = lambda: model
        return agent

    cli._create_agent_facade = local_facade
    services.RolloutContextManager = lambda store: harness.RolloutContextManager(
        store, max_total_bytes=17000 if not semantic else 48000, max_messages=16 if semantic else None
    )
    cli.agent_app()


@pytest.mark.parametrize("semantic", [False, True], ids=["archive", "summary"])
def test_cli_recovers_same_turn_after_work_on_both_sides_of_compaction(tmp_path, semantic):
    (tmp_path / "sample.py").write_text('API_NAME = "stable"\nFIRST = 0\nSECOND = 0\n')
    (tmp_path / "archive.txt").write_text("ARCHIVED_FACT=42\n" + "a" * 9800)
    (tmp_path / "recent.txt").write_text("RECENT_FACT=42\n" + "b" * 7800)
    (tmp_path / "note3.txt").write_text("Preserve the public interface.")
    (tmp_path / "note4.txt").write_text("Check both lines after writing.")
    (tmp_path / "note5.txt").write_text("Keep verification evidence.")
    (tmp_path / "note6.txt").write_text("The second edit remains pending.")
    database = tmp_path / "continuity.sqlite"
    entry = [
        sys.executable,
        "-c",
        f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r})['_continuity_cli']()",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "PRAXIS_TEST_SEMANTIC": str(int(semantic)),
    }

    def cli(*args):
        result = subprocess.run(
            [*entry, *args, "--checkpoint-db", str(database)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        return result.returncode, result.stdout + result.stderr

    code, output = cli(
        "run",
        "Keep API_NAME unchanged. Finish by verifying both edits.",
        "--non-interactive",
        "--disable-workspace-mcp",
        "--allow-write-tools",
        "--no-require-workspace-change",
    )
    assert code == 73, output
    with RuntimeStore.open_sync(database) as store:
        [turn] = store.list_turns()
        items = store.list_items(turn.turn_id)
        [boundary] = [item for item in items if item.kind == "context_compaction"]
        assert boundary.payload["algorithm_revision"] == (
            "semantic-compaction-v7" if semantic else "tool-output-elision-v5"
        )
        assert boundary.payload["capacity_diagnostics"]["summary_calls"] == int(semantic)
        archived_result = next(
            item for item in items if item.payload.get("tool_call_id") == "run-2" and item.kind == "tool_result"
        )
        assert "ARCHIVED_FACT=42" in str(archived_result.payload)
        if semantic:
            assert any(entry["item_id"] is None for entry in boundary.payload["replacement_history"])
            assert archived_result.item_id not in {
                entry["item_id"] for entry in boundary.payload["replacement_history"]
            }
        patch_results = [
            item for item in items if item.kind == "tool_result" and item.payload["tool_name"] == "apply_patch"
        ]
        assert len(patch_results) == 2 and all(not item.payload["is_error"] for item in patch_results)
        assert (
            patch_results[0].applied_thread_sequence
            < boundary.applied_thread_sequence
            < patch_results[1].applied_thread_sequence
        )

    code, output = cli("recover", turn.turn_id, "--action", "continue")
    assert code == 0, output
    with RuntimeStore.open_sync(database) as store:
        [recovered] = store.list_turns()
        assert recovered.turn_id == turn.turn_id
        assert recovered.status == "completed"
        items = store.list_items(recovered.turn_id)
        assert [item.item_id for item in items if item.kind == "context_compaction"] == [boundary.item_id]
        [recall] = [
            item for item in items if item.kind == "tool_result" and item.payload["tool_name"] == "read_context"
        ]
        assert not recall.payload["is_error"]
        assert "ARCHIVED_FACT=42" in str(recall.payload)
        if semantic:
            assert archived_result.item_id in str(recall.payload)
        assert store.read_item(archived_result.item_id).payload == archived_result.payload
        [verification] = [
            item for item in items if item.payload.get("tool_call_id") == "reopen-2" and item.kind == "tool_result"
        ]
        assert not verification.payload["is_error"]
        assert verification.payload["structured_content"]["content"] == 'API_NAME = "stable"\nFIRST = 1\nSECOND = 42\n'
        assert item_text(store.read_turn_answer(recovered.turn_id)) == "Both edits verified."
        assert store.verify().valid

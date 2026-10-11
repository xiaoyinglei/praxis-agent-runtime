"""Controlled public CLI evidence with synthetic provider and real local tools."""

from __future__ import annotations

import json
import re

import httpx
import pytest
from typer.testing import CliRunner

from agent_runtime.cli import agent_app
from agent_runtime.core.messages import StopReason, ToolCall, ToolUseResult
from agent_runtime.harness import GatewayHarnessModel, RolloutStore
from agent_runtime.modeling.contracts import ContextCapacityPolicy, LLMCallStage, LLMProviderResult, LLMStageBudget
from agent_runtime.modeling.gateway import LLMGateway
from tests.agent.harness.test_model_adapter import CharacterAccounting, _resolved_model


class SyntheticProvider:
    def __init__(self, task):
        self.task = task
        self.agent_calls = 0
        self.summary_sources = []

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
                                    "function": {"name": call.name, "arguments": json.dumps(call.input)},
                                }
                                for call in turn.tool_calls
                            ],
                        },
                        "finish_reason": "tool_calls" if turn.tool_calls else "stop",
                    }
                ]
            }
        )

    def _turn(self, *, messages, **kwargs):
        del kwargs
        text = json.dumps(messages, ensure_ascii=False)
        if "Summarize the supplied execution history faithfully." in text:
            self.summary_sources.append(text)
            facts = [fact for fact in ["GOLD_WEB_FACT=17", "GOLD_CODE_FACT=17", "GOLD_DIALOG_FACT=17"] if fact in text]
            return ToolUseResult(
                raw_stop_reason="synthetic",
                text="Verified synthetic evidence: " + ", ".join(facts),
                stop_reason=StopReason.END_TURN,
            )
        self.agent_calls += 1
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
                tool_calls=[ToolCall(id=f"fetch-{self.agent_calls}", name="web_fetch", input=arguments)],
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
                tool_calls=[ToolCall(id=f"file-{self.agent_calls}", name=name, input=arguments)],
            )
        expected = f"GOLD_{self.task.upper() if self.task != 'code' else 'CODE'}_FACT=17"
        if self.task == "dialog":
            expected = "GOLD_DIALOG_FACT=17"
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
        monkeypatch.setattr(agent, "_harness_model", lambda: model)
        return agent

    monkeypatch.setattr(cli_module, "_create_agent_facade", create_facade)
    # Inject the same low projection guard to trigger both algorithms. Summary
    # fragments retain their frozen policy materialization bounds; an 8K bound
    # there would correctly reject these indivisible 12K tool work units.
    from agent_runtime.harness import services
    from agent_runtime.harness.context import RolloutContextManager

    monkeypatch.setattr(
        services, "RolloutContextManager", lambda store: RolloutContextManager(store, max_total_bytes=8000)
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
        "200000",
        "--allow-write-tools",
    ]
    if task == "dialog":
        from tests.agent.harness.legacy_fixtures import record_migrated_context_item as seed

        with RolloutStore(database) as store:
            thread = store.create_thread(workspace=tmp_path)
            previous = store.start_turn(
                thread_id=thread.thread_id, user_message="Synthetic dialogue", binding_manifest={}
            )
            for _index in range(8):
                seed(store, turn_id=previous.turn_id, kind="agent_message", payload={"text": "x" * 2000})
            seed(store, turn_id=previous.turn_id, kind="agent_message", payload={"text": "GOLD_DIALOG_FACT=17"})
            store.complete_turn(turn_id=previous.turn_id, answer="Continue later.")
            args.extend(["--previous-turn-id", previous.turn_id])
    result = CliRunner().invoke(agent_app, args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    expected = f"GOLD_{task.upper()}_FACT=17"
    preserves_fact = mode == "capacity" or task == "dialog"
    if preserves_fact:
        assert provider.summary_sources and expected in str(provider.summary_sources)
    else:
        assert not provider.summary_sources
    if task == "code":
        assert (tmp_path / "sample.py").read_text().startswith("VALUE = 17\n# old evidence ")
    with RolloutStore(database) as store:
        turn = store.list_turns()[-1]
        items = store.list_items(turn.turn_id)
        compactions = [item for item in items if item.kind == "context_compaction"]
        assert compactions
        if mode == "capacity":
            assert all(item.payload["algorithm_revision"] == "semantic-compaction-v6" for item in compactions)
        answer = next(item.payload["text"] for item in reversed(items) if item.kind == "agent_message")
        assert (answer == expected) == preserves_fact
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

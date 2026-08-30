#!/usr/bin/env python
"""Deterministically verify the CLI projection without calling a model."""

from __future__ import annotations

import asyncio
import io
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import click
from typer.main import get_command

from agent_runtime.cli import (
    _CLIToolEventDisplay,
    _display_pending_recovery,
    _print_chat_help,
    agent_app,
)
from agent_runtime.harness import RolloutStore
from agent_runtime.result import AgentPause, AgentToolSummary
from agent_runtime.streaming.events import (
    EventType,
    StreamEvent,
    recovery_event,
    text_delta,
    tool_use_error,
    tool_use_start,
)


@dataclass(frozen=True, slots=True)
class CLISmokeResult:
    checks: dict[str, bool]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


async def _render_canonical_events() -> str:
    display = _CLIToolEventDisplay()
    output = io.StringIO()
    with redirect_stdout(output):
        display.begin_turn()
        await display.emit(text_delta("live text"))
        await display.emit(
            StreamEvent(
                type=EventType.PLAN_UPDATED,
                data={
                    "plan": {
                        "revision": 2,
                        "steps": [
                            {"title": "Inspect source", "status": "completed"},
                            {"title": "Verify CLI", "status": "in_progress"},
                        ],
                    }
                },
            )
        )
        await display.emit(
            tool_use_start(
                "apply_patch",
                "call_patch",
                input_preview="file_path='fixture.txt'",
            )
        )
        await display.emit(
            StreamEvent(
                type=EventType.TOOL_USE_RESULT,
                data={
                    "tool_name": "apply_patch",
                    "tool_id": "call_patch",
                    "result": "patched fixture.txt",
                    "details": {"diff": ("--- a/fixture.txt\n+++ b/fixture.txt\n@@ -1 +1 @@\n-before\n+after\n")},
                },
            )
        )
        await display.emit(recovery_event("retry", "continued from persisted checkpoint"))
        await display.emit(tool_use_start("read_file", "call_missing"))
        await display.emit(tool_use_error("call_missing", "file not found"))
        display.finish()
    return output.getvalue()


def _render_recovery_commands(turn_id: str, checkpoint_db: Path) -> str:
    request = AgentPause(
        request_id="hir_cli_smoke",
        kind="tool_approval",
        question="Allow patch?",
        tool_calls=(
            AgentToolSummary(
                tool_call_id="call_patch",
                tool_name="apply_patch",
                args_preview="file_path='fixture.txt'",
                risk_level="medium",
                reason="Writes the workspace",
            ),
        ),
        options=("allow_once", "deny", "abort"),
    )
    output = io.StringIO()
    with redirect_stdout(output):
        _display_pending_recovery(
            request,
            turn_id=turn_id,
            checkpoint_db=checkpoint_db,
        )
    return output.getvalue()


def run_smoke() -> CLISmokeResult:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="agent_cli_smoke_") as temp_dir:
        root = Path(temp_dir)
        workspace = root / "workspace"
        workspace.mkdir()
        checkpoint_db = root / "agent.sqlite"
        with RolloutStore(checkpoint_db) as store:
            thread = store.create_thread(workspace=workspace)
            turn = store.start_turn(
                thread_id=thread.thread_id,
                user_message="verify CLI projection",
                binding_manifest={"model_alias": "smoke-model"},
            )
            turn = store.complete_turn(
                turn_id=turn.turn_id,
                answer="CLI smoke fixture",
            )

        rendered = asyncio.run(_render_canonical_events())
        checks.update(
            {
                "text": "live text" in rendered,
                "plan": (
                    "计划 (revision 2)" in rendered and "✓ Inspect source" in rendered and "→ Verify CLI" in rendered
                ),
                "tool_result": ("→ apply_patch" in rendered and "✓ apply_patch: patched fixture.txt" in rendered),
                "tool_error": "✗ read_file: file not found" in rendered,
                "diff": ("--- a/fixture.txt" in rendered and "+after" in rendered),
                "recovery": "↻ 恢复: retry" in rendered,
            }
        )

        root_command = get_command(agent_app)
        commands = root_command.commands if isinstance(root_command, click.Group) else {}
        resume_command = commands.get("resume")
        resume_options = {
            option
            for parameter in (() if resume_command is None else resume_command.params)
            if isinstance(parameter, click.Option)
            for option in (*parameter.opts, *parameter.secondary_opts)
        }
        checks["command_surface"] = {"chat", "run", "resume", "model"} <= set(commands) and {
            "--last",
            "--action",
        } <= resume_options

        slash_output = io.StringIO()
        with redirect_stdout(slash_output):
            _print_chat_help()
        slash_text = slash_output.getvalue()
        checks["interactive_commands"] = all(
            command in slash_text for command in ("/status", "/new", "/model", "/exit")
        )

        recovery_text = _render_recovery_commands(turn.turn_id, checkpoint_db)
        checks["recovery_commands"] = (
            "Allow patch?" in recovery_text
            and "Writes the workspace" in recovery_text
            and "--action allow_once" in recovery_text
        )

    failures = tuple(name for name, passed in checks.items() if not passed)
    return CLISmokeResult(checks=checks, failures=failures)


def main() -> int:
    result = run_smoke()
    marker = "PASS" if result.passed else "FAIL"
    print(f"{marker} cli_projection checks={len(result.checks)}")
    for name, passed in result.checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

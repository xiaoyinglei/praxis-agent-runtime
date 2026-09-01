from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import pytest
import typer

from agent_runtime import cli
from agent_runtime.agent import Agent
from agent_runtime.harness import RolloutStore
from agent_runtime.result import (
    AgentPause,
    AgentResult,
    AgentToolSummary,
    AgentUsage,
)


def _persist_cli_turn(
    database: Path,
    workspace: Path,
    *,
    binding_manifest: Mapping[str, object] | None = None,
) -> str:
    workspace.mkdir(exist_ok=True)
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="CLI resume fixture",
            binding_manifest=(
                {"model_alias": None}
                if binding_manifest is None
                else binding_manifest
            ),
        )
    return turn.turn_id


def _result(
    *,
    turn_id: str,
    status: str = "done",
    answer: str | None = None,
    pause: AgentPause | None = None,
) -> AgentResult:
    return AgentResult(
        answer=answer,
        status=status,  # type: ignore[arg-type]
        files=(),
        tool_calls=(),
        evidence=(),
        citations=(),
        usage=AgentUsage(),
        diagnostics=(),
        turn_id=turn_id,
        stop_reason=None,
        pause=pause,
        workspace_path=None,
        groundedness=False,
        insufficient_evidence=False,
        plan=None,
        plan_events=(),
    )


def test_agent_resume_uses_public_facade_and_stable_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumed: list[tuple[str, str, object]] = []
    facade_options: list[dict[str, object]] = []

    class _Facade:
        async def aresume(
            self,
            turn_id: str,
            action: str,
            *,
            user_input: str | None = None,
            event_sink: object,
        ) -> AgentResult:
            assert action == "allow_once"
            assert user_input is None
            resumed.append((turn_id, action, event_sink))
            return _result(turn_id=turn_id, answer="resumed")

    def create_facade(**kwargs: object) -> _Facade:
        facade_options.append(kwargs)
        return _Facade()

    monkeypatch.setattr(cli, "_create_agent_facade", create_facade)

    database = tmp_path / "agent.sqlite"
    turn_id = _persist_cli_turn(database, tmp_path / "workspace")
    cli.agent_resume(
        turn_id=turn_id,
        checkpoint_db=database,
        action="allow_once",
    )

    assert len(resumed) == 1
    assert resumed[0][:2] == (turn_id, "allow_once")
    assert isinstance(resumed[0][2], cli._CLIToolEventDisplay)
    assert facade_options == [
        {
            "model": None,
            "checkpoint_db": database,
            "workspace_path": str((tmp_path / "workspace").resolve()),
            "knowledge": None,
        }
    ]


def test_schema_v2_resume_ignores_unsigned_top_level_model_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "agent.sqlite"
    workspace = tmp_path / "workspace"
    turn_id = _persist_cli_turn(
        database,
        workspace,
        binding_manifest={
            "authentication_schema_version": 1,
            "selection_requester": "user",
            "binding": {"schema_version": 2, "alias": "removed-alias"},
            "model_alias": "forged-top-level-alias",
        },
    )
    facade_options: list[dict[str, object]] = []

    class _Facade:
        async def aresume(self, *_args: object, **_kwargs: object) -> AgentResult:
            return _result(turn_id=turn_id, answer="resumed")

    def create_facade(**kwargs: object) -> _Facade:
        facade_options.append(kwargs)
        return _Facade()

    monkeypatch.setattr(cli, "_create_agent_facade", create_facade)

    cli.agent_resume(
        turn_id=turn_id,
        checkpoint_db=database,
        action="continue",
    )

    assert facade_options[0]["model"] is None

    agent = Agent(
        model="definitely-removed-alias",
        checkpoint_db=database,
        workspace_path=workspace,
    )
    restored = agent._harness_agent_for_turn(turn_id, followup=False)
    assert restored.model is None


@pytest.mark.anyio
async def test_public_legacy_resume_reaches_exact_provider_resume_error(
    tmp_path: Path,
) -> None:
    database = tmp_path / "agent.sqlite"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="legacy paused turn",
            binding_manifest={
                "schema_version": 1,
                "model_alias": "definitely-removed-alias",
            },
        )
        clarification = store.request_clarification(
            turn_id=turn.turn_id,
            question="which target?",
        )

    agent = Agent(
        model="definitely-removed-alias",
        checkpoint_db=database,
        workspace_path=workspace,
        enable_workspace_mcp=False,
    )

    with pytest.raises(
        RuntimeError,
        match="legacy model binding is incomplete and cannot be resumed safely",
    ):
        await agent.aresume(
            turn.turn_id,
            "continue",
            user_input="target A",
        )
    assert clarification.status == "pending"


def test_agent_resume_without_action_prints_pending_recovery_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = AgentPause(
        request_id="hir_pending",
        kind="tool_approval",
        question="Allow the command?",
        tool_calls=(
            AgentToolSummary(
                tool_call_id="tc_pending",
                tool_name="run_command",
                args_preview="command='pytest'",
                risk_level="medium",
                reason="Runs local tests",
            ),
        ),
        options=("allow_once", "deny", "abort"),
    )
    resumed: list[bool] = []

    class _Facade:
        async def apending_input(
            self,
            passed_turn_id: str,
        ) -> AgentPause:
            assert passed_turn_id == turn_id
            return request

        async def aresume(self, **_kwargs: object) -> AgentResult:
            resumed.append(True)
            raise AssertionError("inspection must not mutate the Turn")

    monkeypatch.setattr(cli, "_create_agent_facade", lambda **_kwargs: _Facade())
    database = tmp_path / "agent.sqlite"
    turn_id = _persist_cli_turn(database, tmp_path / "workspace")

    with pytest.raises(typer.Exit) as exc_info:
        cli.agent_resume(
            turn_id=turn_id,
            checkpoint_db=database,
            action=None,
        )

    output = capsys.readouterr().out
    assert exc_info.value.exit_code == 2
    assert "Allow the command?" in output
    assert "run_command" in output
    assert "Runs local tests" in output
    assert "--action allow_once" in output
    assert resumed == []


def test_agent_resume_without_pending_request_offers_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "agent.sqlite"
    turn_id = _persist_cli_turn(database, tmp_path / "workspace")

    class _Facade:
        async def apending_input(
            self,
            passed_turn_id: str,
        ) -> None:
            assert passed_turn_id == turn_id
            return None

    monkeypatch.setattr(cli, "_create_agent_facade", lambda **_kwargs: _Facade())

    with pytest.raises(typer.Exit) as exc_info:
        cli.agent_resume(
            turn_id=turn_id,
            checkpoint_db=database,
            action=None,
        )

    output = capsys.readouterr().out
    assert exc_info.value.exit_code == 2
    assert "中断执行" in output
    assert "--action continue" in output


def test_interactive_terminal_fails_closed_for_ci_or_non_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Stream:
        def __init__(self, value: bool) -> None:
            self.value = value

        def isatty(self) -> bool:
            return self.value

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(cli.sys, "stdin", _Stream(True))
    monkeypatch.setattr(cli.sys, "stdout", _Stream(True))
    assert cli._is_interactive_terminal() is True

    monkeypatch.setattr(cli.sys, "stdin", _Stream(False))
    assert cli._is_interactive_terminal() is False

    monkeypatch.setattr(cli.sys, "stdin", _Stream(True))
    monkeypatch.setenv("CI", "true")
    assert cli._is_interactive_terminal() is False


@pytest.mark.anyio
async def test_inline_approval_uses_public_execution_chain(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pause = AgentPause(
        request_id="hir_inline",
        kind="tool_approval",
        question="approve?",
        tool_calls=(
            AgentToolSummary(
                tool_call_id="tc_inline",
                tool_name="run_command",
                args_preview="command='pytest'",
            ),
        ),
        options=("allow_once", "deny", "abort"),
    )
    calls: list[str] = []
    displays: list[object] = []

    turn_id = str(uuid4())

    class _Facade:
        async def arun(self, task: str, **kwargs: object) -> AgentResult:
            assert task == "run tests"
            displays.append(kwargs["event_sink"])
            calls.append("execute")
            return _result(turn_id=turn_id, status="paused", pause=pause)

        async def aresume(
            self,
            turn_id: str,
            action: str,
            *,
            user_input: str | None = None,
            event_sink: object,
        ) -> AgentResult:
            assert action == "allow_once"
            assert user_input is None
            displays.append(event_sink)
            calls.append("resume")
            return _result(turn_id=turn_id, answer="continued")

    monkeypatch.setattr(
        cli,
        "_handle_pause",
        lambda _result: "allow_once",
    )

    display = cli._CLIToolEventDisplay()
    result = await cli._run_facade_command(
        _Facade(),
        task="run tests",
        files=(),
        max_tokens_total=None,
        interactive_approval=True,
        event_display=display,
    )

    assert result.answer == "continued"
    assert calls == ["execute", "resume"]
    assert displays == [display, display]

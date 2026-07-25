from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pytest import MonkeyPatch

from agent_runtime.result import AgentResult, AgentUsage
from rag.agent import cli
from rag.agent.turns import RuntimeBinding, TurnStatus, TurnStore


def _result(*, turn_id: str | None = None, answer: str = "bounded") -> AgentResult:
    return AgentResult(
        answer=answer,
        status="done",
        files=(),
        tool_calls=(),
        evidence=(),
        citations=(),
        usage=AgentUsage(),
        diagnostics=(),
        turn_id=turn_id or str(uuid4()),
        stop_reason=None,
        pause=None,
        workspace_path=None,
        groundedness=False,
        insufficient_evidence=False,
        plan=None,
        plan_events=(),
    )


@pytest.mark.anyio
async def test_chat_slash_commands_do_not_reach_the_agent(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "agent.sqlite"
    store = TurnStore(database)
    previous = store.begin_turn(
        "first",
        RuntimeBinding(
            model_alias="fake-model",
            workspace_path=str(workspace.resolve()),
        ),
    )
    store.mark_terminal(previous.turn_id, TurnStatus.COMPLETED)
    store.close()
    turn_calls: list[object] = []

    class _Facade:
        checkpoint_db = database
        workspace_path = workspace

        def current_model(self) -> SimpleNamespace:
            return SimpleNamespace(id="fake-model")

        async def arun(self, *args: object, **kwargs: object) -> AgentResult:
            turn_calls.append((args, kwargs))
            raise AssertionError("slash commands must not reach the agent")

    commands = iter(["/status", "/new", "/status", "/help", "/unknown", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    await cli._chat_facade_loop(
        _Facade(),
        max_tokens_total=None,
        previous_turn_id=previous.turn_id,
    )

    output = capsys.readouterr().out
    assert f"Previous Turn: {previous.turn_id}" in output
    assert "下一条消息将使用空历史" in output
    assert "Previous Turn: (none)" in output
    assert "/new" in output
    assert "未知命令: /unknown" in output
    assert turn_calls == []


@pytest.mark.anyio
async def test_chat_loop_carries_the_previous_turn_automatically(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []
    result_ids = [str(uuid4()), str(uuid4())]

    class _Facade:
        checkpoint_db = tmp_path / "agent.sqlite"
        workspace_path = workspace

        def current_model(self) -> SimpleNamespace:
            return SimpleNamespace(id="fake-model")

        async def arun(
            self,
            message: str,
            **kwargs: object,
        ) -> AgentResult:
            calls.append((message, kwargs))
            return _result(turn_id=result_ids[len(calls) - 1])

    commands = iter(["hello", "continue", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    await cli._chat_facade_loop(
        _Facade(),
        max_tokens_total=None,
        max_turns=3,
    )

    assert [message for message, _kwargs in calls] == ["hello", "continue"]
    assert calls[0][1]["previous_turn_id"] is None
    assert calls[1][1]["previous_turn_id"] == result_ids[0]
    assert calls[0][1]["max_turns"] == 3
    assert calls[0][1]["require_workspace_change"] is False
    assert isinstance(calls[0][1]["event_sink"], cli._CLIToolEventDisplay)

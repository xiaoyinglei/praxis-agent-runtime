from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pytest import MonkeyPatch

from agent_runtime import cli
from agent_runtime.core.llm_registry import UnknownModelAliasError
from agent_runtime.harness import RolloutStore
from agent_runtime.models import ModelSpec
from agent_runtime.result import AgentResult, AgentUsage
from agent_runtime.streaming.events import ItemStatus, TurnItemKind, item_completed


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


def _model_spec(model_id: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider=f"provider-{model_id}",
        provider_model=f"provider/{model_id}",
        context_window=32_768,
        supports_tools=True,
        supports_structured_output=True,
        location="cloud",
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
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        previous = store.start_turn(
            thread_id=thread.thread_id,
            user_message="first",
            binding_manifest={"model_alias": "fake-model"},
        )
        previous = store.complete_turn(
            turn_id=previous.turn_id,
            answer="first answer",
        )
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


@pytest.mark.anyio
async def test_verbose_command_expands_subsequent_turn_tool_results(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Facade:
        checkpoint_db = tmp_path / "agent.sqlite"
        workspace_path = tmp_path

        def current_model(self) -> SimpleNamespace:
            return SimpleNamespace(id="fake-model")

        async def arun(self, message: str, **kwargs: object) -> AgentResult:
            assert message == "inspect"
            sink = kwargs["event_sink"]
            await sink.emit(  # type: ignore[union-attr]
                item_completed(
                    turn_id="turn-verbose",
                    item_id="tool-verbose",
                    item_kind=TurnItemKind.TOOL,
                    status=ItemStatus.SUCCESS,
                    data={
                        "result": {
                            "tool_name": "read_file",
                            "structured_content": {
                                "lines": [f"line {index}" for index in range(12)]
                            },
                        }
                    },
                )
            )
            return _result(turn_id="turn-verbose", answer="")

    commands = iter(["/verbose", "inspect", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    await cli._chat_facade_loop(_Facade(), max_tokens_total=None)  # type: ignore[arg-type]

    output = capsys.readouterr().out
    assert "详细输出: 开" in output
    assert "line 0" in output
    assert "line 11" in output
    assert "/verbose 查看完整结果" not in output


@pytest.mark.anyio
async def test_bare_model_command_shows_current_available_and_switch_usage(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = (_model_spec("model-a"), _model_spec("model-b"))

    class _Facade:
        checkpoint_db = tmp_path / "agent.sqlite"
        workspace_path = tmp_path

        def current_model(self) -> ModelSpec:
            return models[0]

        def models(self) -> list[ModelSpec]:
            return list(models)

    commands = iter(["/model", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    await cli._chat_facade_loop(
        _Facade(),  # type: ignore[arg-type]
        max_tokens_total=None,
    )

    output = capsys.readouterr().out
    assert "当前模型: model-a" in output
    assert "* model-a" in output
    assert "  model-b" in output
    assert "切换: /model <alias>" in output


@pytest.mark.anyio
async def test_model_switch_after_completed_turn_keeps_history_and_changes_next_turn(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    models = {
        "model-a": _model_spec("model-a"),
        "model-b": _model_spec("model-b"),
    }
    selected = "model-a"
    calls: list[tuple[str, str, object]] = []
    result_ids = [str(uuid4()), str(uuid4())]

    class _Facade:
        checkpoint_db = tmp_path / "agent.sqlite"
        workspace_path = tmp_path

        def current_model(self) -> ModelSpec:
            return models[selected]

        def models(self) -> list[ModelSpec]:
            return list(models.values())

        def switch_model(self, model_id: str) -> ModelSpec:
            nonlocal selected
            selected = model_id
            return models[model_id]

        async def arun(
            self,
            message: str,
            **kwargs: object,
        ) -> AgentResult:
            calls.append((message, selected, kwargs["previous_turn_id"]))
            return _result(turn_id=result_ids[len(calls) - 1])

    commands = iter(["remember cobalt", "/model model-b", "what did I say?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    await cli._chat_facade_loop(
        _Facade(),  # type: ignore[arg-type]
        max_tokens_total=None,
    )

    assert calls == [
        ("remember cobalt", "model-a", None),
        ("what did I say?", "model-b", result_ids[0]),
    ]


@pytest.mark.anyio
async def test_invalid_model_alias_keeps_current_lists_aliases_and_starts_no_turn(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = (_model_spec("model-a"), _model_spec("model-b"))
    selected = "model-a"
    switch_attempts: list[str] = []
    turn_calls: list[str] = []

    class _Facade:
        checkpoint_db = tmp_path / "agent.sqlite"
        workspace_path = tmp_path

        def current_model(self) -> ModelSpec:
            return next(spec for spec in models if spec.id == selected)

        def models(self) -> list[ModelSpec]:
            return list(models)

        def switch_model(self, model_id: str) -> ModelSpec:
            switch_attempts.append(model_id)
            raise UnknownModelAliasError(f"Model alias {model_id!r} not found in catalog")

        async def arun(self, message: str, **kwargs: object) -> AgentResult:
            del kwargs
            turn_calls.append(message)
            return _result()

    commands = iter(["/model missing", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    await cli._chat_facade_loop(
        _Facade(),  # type: ignore[arg-type]
        max_tokens_total=None,
    )

    output = capsys.readouterr().out
    assert "模型切换失败" in output
    assert "missing" in output
    assert "可用模型:" in output
    assert "model-a" in output
    assert "model-b" in output
    assert selected == "model-a"
    assert switch_attempts == ["missing"]
    assert turn_calls == []

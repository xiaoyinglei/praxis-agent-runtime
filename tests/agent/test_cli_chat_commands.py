from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pytest import MonkeyPatch

from agent_runtime import cli
from agent_runtime.core.llm_registry import UnknownModelIdError
from agent_runtime.harness import RolloutStore
from agent_runtime.models import ModelSpec
from agent_runtime.result import AgentResult, AgentUsage
from agent_runtime.streaming.events import ItemStatus, TurnItemKind, item_completed


class _SessionFactory:
    opened = 0
    closed = 0

    @asynccontextmanager
    async def session(self, **options):
        self.opened += 1
        self.options = options
        try:
            yield self
        finally:
            self.closed += 1


def test_local_provider_startup_failure_has_readable_message():
    import click

    from agent_runtime.local_runtime import LocalRuntimeError

    async def unavailable():
        raise LocalRuntimeError("RAW http://127.0.0.1:8080/v1/models")

    with pytest.raises(click.ClickException, match="本地模型服务尚未就绪") as exc:
        cli._run_cli_async(unavailable())
    assert "127.0.0.1" not in str(exc.value)
    assert "RAW" not in str(exc.value)


def test_failure_details_are_available_only_in_verbose_output(capsys):
    from agent_runtime.result import AgentDiagnostic

    diagnostic = AgentDiagnostic(code="model_step_budget_exhausted", component="runtime", message="RAW diagnosis")
    cli._display_failure(stop_reason=diagnostic.code, diagnostics=[diagnostic], verbose=False)
    output = capsys.readouterr().out
    assert "尝试次数上限" in output
    assert "RAW diagnosis" not in output
    cli._display_failure(stop_reason=diagnostic.code, diagnostics=[diagnostic], verbose=True)
    assert "RAW diagnosis" in capsys.readouterr().out


@pytest.mark.anyio
async def test_chat_model_picker_switches_session_without_provider_request(tmp_path):
    from agent_runtime.terminal_app import TerminalChatApp

    class Facade(_SessionFactory):
        workspace_path = tmp_path
        selected = "first"

        def current_model(self):
            return _model_spec(self.selected)

        def models(self):
            return [_model_spec("first"), _model_spec("second")]

        def switch_model(self, model_id):
            self.selected = model_id
            return self.current_model()

        async def submit(self, *args, **kwargs):
            pytest.fail("Model selection must not call the provider")

    class UI(TerminalChatApp):
        commands = iter(["/model", "/exit"])

        async def prompt(self, **kwargs):
            return next(self.commands)

        async def choose_model(self, choices, current):
            assert choices == ["first", "second"]
            assert current == "first"
            return "second"

    facade = Facade()
    ui = UI(model="first", workspace=str(tmp_path))
    await cli._chat_facade_loop(facade, max_tokens_total=None, _ui=ui)
    assert facade.selected == ui.model == "second"


@pytest.mark.anyio
async def test_connections_command_reopens_changed_tools_between_turns_with_same_history(tmp_path):
    from agent_runtime.terminal_app import TerminalChatApp
    commands = iter(["first", "/connections", "second", "/exit"])
    opened = []
    closed = []
    state = {"changed": False}
    results = [_result(), _result()]

    class Session:
        def __init__(self, index):
            self.index = index

        def current_model(self):
            return _model_spec("test")

        def connections_changed(self):
            return state["changed"]

        async def close(self):
            closed.append(self.index)

        async def submit(self, query, **kwargs):
            return results[self.index]

    class Facade:
        workspace_path = tmp_path

        def current_model(self):
            return _model_spec("test")

        @asynccontextmanager
        async def session(self, **kwargs):
            session = Session(len(opened))
            opened.append(kwargs)
            state["changed"] = False
            yield session

    class UI(TerminalChatApp):
        async def prompt(self, **kwargs):
            return next(commands)

        async def open_connections(self):
            state["changed"] = True  # Host finished a check; active Turn tools were not changed.

    await cli._chat_facade_loop(Facade(), max_tokens_total=None, _ui=UI(model="test", workspace=str(tmp_path)))
    assert closed == [0]
    assert len(opened) == 2
    assert opened[1]["previous_turn_id"] == results[0].turn_id



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
            binding_manifest={"model_id": "fake-model"},
        )
        previous = store.complete_turn(
            turn_id=previous.turn_id,
            answer="first answer",
        )
    turn_calls: list[object] = []

    class _Facade(_SessionFactory):
        checkpoint_db = database
        workspace_path = workspace

        def current_model(self) -> SimpleNamespace:
            return SimpleNamespace(id="fake-model")

        async def submit(self, *args: object, **kwargs: object) -> AgentResult:
            turn_calls.append((args, kwargs))
            raise AssertionError("slash commands must not reach the agent")

    commands = iter(["/status", "/copy", "/new", "/status", "/details", "/help", "/unknown", "/exit"])
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
    assert "未知命令: /details" in output
    assert "Ctrl+O 切换最近记录" in output
    assert "暂无可复制的回答" in output
    assert "/transcript" not in output
    assert turn_calls == []


def test_ordinary_chat_shows_validation_status_and_scope(capsys):
    from dataclasses import replace

    result = _result(answer="你好")
    cli._display_agent_result(result, verbose=False)
    output = capsys.readouterr().out
    assert "你好" in output
    assert "not_evaluated" in output and "任务验收尚未评估" in output
    cli._display_agent_result(result, verbose=True)
    assert "not_evaluated" in capsys.readouterr().out
    cli._display_agent_result(replace(result, validation_status="passed", validation_scope="workspace_change"),
                              verbose=False)
    output = capsys.readouterr().out
    assert "passed" in output and "工作区修改" in output


@pytest.mark.anyio
async def test_status_retains_unevaluated_result_after_quiet_chat(tmp_path, monkeypatch, capsys):
    class Facade(_SessionFactory):
        workspace_path = tmp_path

        def current_model(self):
            return SimpleNamespace(id="fake-model")

        async def submit(self, message, **kwargs):
            return _result(answer="ordinary answer")

    commands = iter(["hello", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))
    await cli._chat_facade_loop(Facade(), max_tokens_total=None)
    output = capsys.readouterr().out
    assert "ordinary answer" in output
    assert output.count("验证状态: not_evaluated") == 2
    assert "任务验收尚未评估" in output


@pytest.mark.anyio
async def test_chat_loop_reuses_one_session_for_all_submissions(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []
    result_ids = [str(uuid4()), str(uuid4())]

    class _Facade(_SessionFactory):
        checkpoint_db = tmp_path / "agent.sqlite"
        workspace_path = workspace

        def current_model(self) -> SimpleNamespace:
            return SimpleNamespace(id="fake-model")

        async def submit(
            self,
            message: str,
            **kwargs: object,
        ) -> AgentResult:
            calls.append((message, kwargs))
            return _result(turn_id=result_ids[len(calls) - 1])

    commands = iter(["hello", "continue", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    facade = _Facade()
    await cli._chat_facade_loop(facade, max_tokens_total=None, max_turns=3)

    assert [message for message, _kwargs in calls] == ["hello", "continue"]
    assert facade.opened == facade.closed == 1
    assert all("previous_turn_id" not in kwargs for _, kwargs in calls)
    assert facade.options["max_turns"] == 3
    assert facade.options["require_workspace_change"] is False
    assert isinstance(calls[0][1]["event_sink"], cli._CLIToolEventDisplay)


@pytest.mark.anyio
async def test_verbose_command_expands_subsequent_turn_tool_results(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Facade(_SessionFactory):
        checkpoint_db = tmp_path / "agent.sqlite"
        workspace_path = tmp_path

        def current_model(self) -> SimpleNamespace:
            return SimpleNamespace(id="fake-model")

        async def submit(self, message: str, **kwargs: object) -> AgentResult:
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

    class _Facade(_SessionFactory):
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
    assert "切换: /model <model_id>" in output


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

    class _Facade(_SessionFactory):
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

        async def submit(
            self,
            message: str,
            **kwargs: object,
        ) -> AgentResult:
            calls.append((message, selected, self))
            return _result(turn_id=result_ids[len(calls) - 1])

    commands = iter(["remember cobalt", "/model model-b", "what did I say?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))

    await cli._chat_facade_loop(
        _Facade(),  # type: ignore[arg-type]
        max_tokens_total=None,
    )

    assert [(message, model) for message, model, _ in calls] == [
        ("remember cobalt", "model-a"), ("what did I say?", "model-b"),
    ]
    assert calls[0][2] is calls[1][2]


@pytest.mark.anyio
async def test_invalid_model_id_keeps_current_lists_ids_and_starts_no_turn(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = (_model_spec("model-a"), _model_spec("model-b"))
    selected = "model-a"
    switch_attempts: list[str] = []
    turn_calls: list[str] = []

    class _Facade(_SessionFactory):
        checkpoint_db = tmp_path / "agent.sqlite"
        workspace_path = tmp_path

        def current_model(self) -> ModelSpec:
            return next(spec for spec in models if spec.id == selected)

        def models(self) -> list[ModelSpec]:
            return list(models)

        def switch_model(self, model_id: str) -> ModelSpec:
            switch_attempts.append(model_id)
            raise UnknownModelIdError(f"Model ID {model_id!r} not found in catalog")

        async def submit(self, message: str, **kwargs: object) -> AgentResult:
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


@pytest.mark.anyio
async def test_new_session_still_cleans_display_after_submit_failure(tmp_path, monkeypatch):
    class Display(cli._CLIToolEventDisplay):
        active = False

        def begin_turn(self, *, reset=True):
            assert not self._context_status  # /new clears the previous chat before submission.
            super().begin_turn(reset=reset)
            self.active = True

        def finish(self):
            super().finish()
            self.active = False

    display = Display(interactive=False)
    display._render_context_status("previous-chat", "stale context")

    class Facade(_SessionFactory):
        workspace_path = tmp_path

        def current_model(self):
            return SimpleNamespace(id="fake-model")

        async def submit(self, *args, **kwargs):
            raise RuntimeError("submit failed")

    commands = iter(["/new", "run"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))
    monkeypatch.setattr(cli, "_CLIToolEventDisplay", lambda **kwargs: display)
    with pytest.raises(RuntimeError, match="submit failed"):
        await cli._chat_facade_loop(Facade(), max_tokens_total=None)
    assert not display.active
    assert display._started_at is None


@pytest.mark.anyio
async def test_live_ui_cancel_cleans_turn_and_can_submit_again(tmp_path):
    import asyncio

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.streaming.events import item_started
    from agent_runtime.terminal_app import TerminalChatApp

    calls = []
    started = asyncio.Event()

    class Facade(_SessionFactory):
        workspace_path = tmp_path

        def current_model(self):
            return SimpleNamespace(id="fake-model")

        async def submit(self, message, *, event_sink):
            calls.append(message)
            if message == "slow":
                await event_sink.emit(item_started(turn_id="slow-turn", item_id="cmd",
                                                  item_kind=TurnItemKind.COMMAND,
                                                  data={"tool_name": "run_command", "input_preview": "sleep 10"}))
                started.set()
                await asyncio.Event().wait()
            return _result(answer="second answer")

        async def resume(self, turn_id, action, **kwargs):
            calls.append((turn_id, action))
            return _result(turn_id=turn_id)

    async def wait_for(predicate):
        async with asyncio.timeout(3):
            while not predicate():
                await asyncio.sleep(.01)

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake-model", workspace=str(tmp_path), input=pipe, output=DummyOutput())
        facade = Facade()
        task = asyncio.create_task(ui.run(lambda: cli._chat_facade_loop(facade, max_tokens_total=None, _ui=ui)))
        try:
            await wait_for(lambda: ui.waiting)
            pipe.send_text("slow\r")
            await asyncio.wait_for(started.wait(), 3)
            pipe.send_text("\x03")
            await wait_for(lambda: ui.waiting)
            pipe.send_text("next\r")
            await wait_for(lambda: "second answer" in ui.view.plain_text())
            await wait_for(lambda: ui.waiting)
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                from contextlib import suppress
                with suppress(asyncio.CancelledError):
                    await task
        assert calls == ["slow", ("slow-turn", "abort"), "next"]
        assert facade.opened == facade.closed == 1
        assert "已取消当前执行" in ui.view.plain_text()


@pytest.mark.anyio
async def test_live_ui_approval_validates_choice_and_resumes_same_turn(tmp_path):
    import asyncio
    from dataclasses import replace

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.result import AgentPause
    from agent_runtime.terminal_app import TerminalChatApp

    calls = []
    pause = AgentPause(request_id="approval", kind="tool_approval", question="允许执行命令？", options=("deny",))

    class Facade(_SessionFactory):
        workspace_path = tmp_path

        def current_model(self):
            return SimpleNamespace(id="fake-model")

        async def submit(self, message, *, event_sink):
            return replace(_result(turn_id="paused-turn"), status="paused", pause=pause)

        async def resume(self, turn_id, action, **kwargs):
            calls.append((turn_id, action))
            return _result(turn_id=turn_id, answer="已拒绝")

    async def wait_for(predicate):
        async with asyncio.timeout(3):
            while not predicate():
                await asyncio.sleep(.01)

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake-model", workspace=str(tmp_path), input=pipe, output=DummyOutput())
        task = asyncio.create_task(ui.run(lambda: cli._chat_facade_loop(Facade(), max_tokens_total=None, _ui=ui)))
        try:
            await wait_for(lambda: ui.waiting)
            pipe.send_text("run\r")
            await wait_for(lambda: ui.status == "等待确认" and ui.waiting)
            pipe.send_text("Y")  # Shortcut cannot authorize an option that was not offered.
            await asyncio.sleep(.05)
            assert calls == []
            pipe.send_text("N")
            await wait_for(lambda: "已拒绝" in ui.view.plain_text() and ui.waiting)
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                from contextlib import suppress
                with suppress(asyncio.CancelledError):
                    await task
        assert calls == [("paused-turn", "deny")]


@pytest.mark.anyio
async def test_live_ui_clarification_resumes_with_text_instead_of_aborting(tmp_path):
    import asyncio
    from dataclasses import replace

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.result import AgentPause
    from agent_runtime.terminal_app import TerminalChatApp

    calls = []
    request = AgentPause(request_id="question", kind="clarification", question="你的工作年限？")

    class Facade(_SessionFactory):
        workspace_path = tmp_path

        def current_model(self):
            return SimpleNamespace(id="fake-model")

        async def submit(self, message, **kwargs):
            return replace(_result(turn_id="question-turn"), status="paused", pause=request)

        async def resume(self, turn_id, action, **kwargs):
            calls.append((turn_id, action, kwargs.get("user_input")))
            return _result(turn_id=turn_id, answer="已收到条件")

    async def wait_for(predicate):
        async with asyncio.timeout(3):
            while not predicate():
                await asyncio.sleep(.01)

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake-model", workspace=str(tmp_path), input=pipe, output=DummyOutput())
        task = asyncio.create_task(ui.run(lambda: cli._chat_facade_loop(Facade(), max_tokens_total=None, _ui=ui)))
        try:
            await wait_for(lambda: ui.waiting)
            pipe.send_text("run\r")
            await wait_for(lambda: ui.status == "等待确认" and ui.waiting)
            pipe.send_text("两年\r")
            await wait_for(lambda: "已收到条件" in ui.view.plain_text() and ui.waiting)
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                from contextlib import suppress
                with suppress(asyncio.CancelledError):
                    await task
        assert calls == [("question-turn", "continue", "两年")]


@pytest.mark.anyio
@pytest.mark.parametrize("kind,options,typed,action,user_input", [
    ("clarification", (), "两年", "continue", "两年"),
    ("choice", ("两年", "五年"), "两年", "continue", "两年"),
    ("tool_approval", ("approve", "deny"), "Y", "approve", None),
])
async def test_interactive_run_preserves_pause_action_and_text(
    monkeypatch, kind, options, typed, action, user_input,
):
    from dataclasses import replace

    from agent_runtime.result import AgentPause

    calls = []
    request = AgentPause(request_id="question", kind=kind, question="需要确认", options=options)

    class Facade:
        async def run(self, *args, **kwargs):
            return replace(_result(turn_id="run-turn"), status="paused", pause=request)

        async def resume(self, turn_id, action, **kwargs):
            calls.append((turn_id, action, kwargs.get("user_input")))
            return _result(turn_id=turn_id)

    monkeypatch.setattr("builtins.input", lambda _: typed)
    result = await cli._run_facade_command(
        Facade(), task="task", files=(), max_tokens_total=None, interactive_approval=True,
    )
    assert result.status == "done"
    assert calls == [("run-turn", action, user_input)]

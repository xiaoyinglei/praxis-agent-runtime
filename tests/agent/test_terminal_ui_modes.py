from __future__ import annotations

import asyncio
import base64
import io
import logging
import sys
from contextlib import suppress

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from typer.main import get_command
from typer.testing import CliRunner

from agent_runtime import cli
from agent_runtime.cli import agent_app
from agent_runtime.terminal_app import TerminalChatApp


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["inline", "fullscreen"])
@pytest.mark.parametrize("configured_handler", [False, True, "dynamic"])
async def test_background_errors_stay_in_conversation_without_overwriting_composer(
    mode, configured_handler, monkeypatch,
):
    terminal = io.StringIO()
    stderr = sys.stderr
    logger = logging.Logger("terminal-background-warning")
    monkeypatch.setitem(logging.Logger.manager.loggerDict, logger.name, logger)
    handler = logging.StreamHandler(stderr)
    if configured_handler == "dynamic":
        handler = logging.lastResort
    if configured_handler:
        logger.addHandler(handler)
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(
            model="preview", workspace="/tmp", ui_mode=mode, input=pipe,
            output=Vt100_Output(terminal, lambda: Size(rows=24, columns=100), enable_cpr=False),
        )
        emit = asyncio.Event()

        async def conversation():
            await emit.wait()
            print("background stderr diagnostic", file=sys.stderr)
            logger.warning("MCP server notion is unavailable: simulated connection failure")
            await ui.prompt()

        task = asyncio.create_task(ui.run(conversation))
        try:
            async with asyncio.timeout(3):
                while ui.app.renderer._last_screen is None:
                    await asyncio.sleep(0.01)
                ui.editor.text = "unfinished 中文 question"
                emit.set()
                while not ui.waiting:
                    await asyncio.sleep(0.01)
            assert "background stderr diagnostic" in ui.view.plain_text()
            assert ui.view.plain_text().count("MCP server notion is unavailable") == 1
            assert ui.editor.text == "unfinished 中文 question"
            ui.invalidate()
            await asyncio.sleep(0.08)
            screen = ui.app.renderer._last_screen
            rows = ["".join(row[x].char for x in sorted(row)) for _, row in sorted(screen.data_buffer.items())]
            assert sum("/help" in row for row in rows) == 1
            assert "unfinished 中文 question" in "\n".join(rows)
            pipe.send_text("\r")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        assert sys.stderr is stderr
        assert handler.stream is stderr


@pytest.mark.anyio
@pytest.mark.parametrize("finish", ["exception", "cancel"])
async def test_diagnostic_streams_restore_after_abnormal_exit(finish, monkeypatch, tmp_path):
    stderr, stdout = sys.stderr, sys.stdout
    logger = logging.Logger("terminal-restore-warning")
    monkeypatch.setitem(logging.Logger.manager.loggerDict, logger.name, logger)
    console = logging.StreamHandler(stderr)
    log_path = tmp_path / "agent.log"
    file_handler = logging.FileHandler(log_path)
    custom_stream = io.StringIO()
    custom_handler = logging.StreamHandler(custom_stream)
    for handler in (console, file_handler, custom_handler):
        logger.addHandler(handler)
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="preview", workspace="/tmp", input=pipe,
                             output=Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=80), enable_cpr=False))
        ready = asyncio.Event()

        async def conversation():
            logger.warning("diagnostic before exit")
            ready.set()
            if finish == "exception":
                raise RuntimeError("simulated failure")
            await asyncio.Future()

        task = asyncio.create_task(ui.run(conversation))
        try:
            await asyncio.wait_for(ready.wait(), 3)
            if finish == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(RuntimeError, match="simulated failure"):
                    await asyncio.wait_for(task, 3)
            assert sys.stderr is stderr and sys.stdout is stdout
            assert console.stream is stderr
            assert custom_handler.stream is custom_stream
            assert log_path.read_text() == "diagnostic before exit\n"
            assert custom_stream.getvalue() == "diagnostic before exit\n"
            assert ui.view.plain_text().count("diagnostic before exit") == 1
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            file_handler.close()


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["inline", "fullscreen"])
@pytest.mark.parametrize("exit_keys", ["/exit\r", "\x04"])
async def test_modes_restore_terminal_and_keep_completed_messages_available(mode, exit_keys, monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "preview")
    stream = io.StringIO()
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(
            model="preview", workspace="/tmp", ui_mode=mode, input=pipe,
            output=Vt100_Output(stream, lambda: Size(rows=24, columns=80), enable_cpr=False),
        )

        async def conversation():
            await ui.replay_messages((("user", "saved question"), ("answer", "saved answer")))
            ui.view.message("completed answer\n")
            await ui.flush_scrollback()
            await ui.show_text("status details")
            await ui.copy_answer("clipboard preview")
            await ui.prompt()

        task = asyncio.create_task(ui.run(conversation))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(0.01)
            visible = "".join(text for _, text, *_ in ui.fragments())
            output = stream.getvalue()
            assert "\x1b[3J" not in output  # Never erase existing shell scrollback.
            assert "你 ›" not in output and "你 ›" not in visible
            assert base64.b64encode(b"clipboard preview").decode() in output
            if mode == "fullscreen":
                assert "saved answer" in visible and "completed answer" in visible
                assert "status details" in visible
                assert "clipboard preview" not in output and "clipboard preview" not in visible
                assert not any(block.in_scrollback for block in ui.view.blocks)
                assert "\x1b[?1049h" in output
                screen = ui.app.renderer._last_screen
                assert screen.height == 24
            else:
                assert "\x1b[2J" in output
                assert "\x1b[?1049h" not in output
                assert "completed answer" in output and "completed answer" not in visible
            pipe.send_text(exit_keys)
            await asyncio.wait_for(task, 3)
            assert ("\x1b[?1049l" in stream.getvalue()) == (mode == "fullscreen")
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


@pytest.mark.anyio
async def test_fullscreen_restores_shell_after_exception():
    stream = io.StringIO()
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(
            model="preview", workspace="/tmp", ui_mode="fullscreen", input=pipe,
            output=Vt100_Output(stream, lambda: Size(rows=24, columns=80), enable_cpr=False),
        )

        async def fail():
            raise RuntimeError("preview failure")

        with pytest.raises(RuntimeError, match="preview failure"):
            await ui.run(fail)
        assert "\x1b[?1049h" in stream.getvalue()
        assert "\x1b[?1049l" in stream.getvalue()


@pytest.mark.anyio
async def test_fullscreen_restores_shell_on_cancel_and_keeps_input_at_bottom_after_resize():
    stream = io.StringIO()
    size = [Size(rows=24, columns=80)]
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(
            model="preview", workspace="/tmp", ui_mode="fullscreen", input=pipe,
            output=Vt100_Output(stream, lambda: size[0], enable_cpr=False),
        )
        task = asyncio.create_task(ui.run(ui.prompt))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(0.01)
            # The composer sits above the separator, two status rows, and help row.
            screen = ui.app.renderer._last_screen
            assert screen.visible_windows_to_write_positions[ui.editor.window].ypos == size[0].rows - 5
            rendered = ["".join(row[x].char for x in sorted(row)) for _, row in sorted(screen.data_buffer.items())]
            assert "preview" in rendered[-3]
            assert "待报告" in rendered[-2]
            assert "Enter" in rendered[-1]
            size[0] = Size(rows=16, columns=40)
            ui.app._on_resize()
            await asyncio.sleep(0.08)
            screen = ui.app.renderer._last_screen
            assert screen.visible_windows_to_write_positions[ui.editor.window].ypos == size[0].rows - 5
            pipe.send_text("\x1b[200~首行\n尾行\x1b[201~")
            await asyncio.sleep(0.08)
            assert ui.editor.text == "首行\n尾行"
            screen = ui.app.renderer._last_screen
            assert screen.visible_windows_to_write_positions[ui.editor.window].ypos == size[0].rows - 6
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        assert "\x1b[?1049l" in stream.getvalue()


def test_chat_ui_option_rejects_invalid_mode_before_runtime_creation():
    command = get_command(agent_app).commands["chat"]
    option = next(param for param in command.params if param.name == "ui_mode")
    assert set(option.type.choices) == {"inline", "fullscreen"}
    result = CliRunner().invoke(agent_app, ["chat", "--ui", "invalid"])
    assert result.exit_code == 2


@pytest.mark.parametrize("mode", ["inline", "fullscreen"])
def test_chat_ui_choice_reaches_chat_loop(mode, monkeypatch):
    received = []
    monkeypatch.setattr(cli, "_create_agent_facade", lambda **kwargs: object())

    async def loop(facade, **kwargs):
        received.append(kwargs["ui_mode"])

    monkeypatch.setattr(cli, "_chat_facade_loop", loop)
    result = CliRunner().invoke(agent_app, ["chat", "--ui", mode])
    assert result.exit_code == 0, result.output
    assert received == [mode]


@pytest.mark.anyio
async def test_cleared_composer_does_not_show_command_menu_over_answer():
    from prompt_toolkit.buffer import CompletionState
    from prompt_toolkit.completion import Completion
    from prompt_toolkit.document import Document

    stream = io.StringIO()
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(
            model="preview", workspace="/tmp", ui_mode="fullscreen", input=pipe,
            output=Vt100_Output(stream, lambda: Size(rows=24, columns=80), enable_cpr=False),
        )
        task = asyncio.create_task(ui.run(ui.prompt))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(0.01)
            # A completion worker may finish after Enter cleared the composer.
            ui.editor.buffer.complete_state = CompletionState(
                Document(""), [Completion("/help", display_meta="stale command popup")],
            )
            ui.invalidate()
            await asyncio.sleep(0.08)
            screen = ui.app.renderer._last_screen
            visible = "\n".join("".join(cell.char for cell in row.values()) for row in screen.data_buffer.values())
            assert "stale command popup" not in visible
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["inline", "fullscreen"])
async def test_preview_ctrl_c_interrupts_output_and_returns_to_input(mode):
    from scripts.terminal_ui_demo import preview

    stream = io.StringIO()
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(
            model="preview", workspace="/tmp", ui_mode=mode, input=pipe,
            output=Vt100_Output(stream, lambda: Size(rows=24, columns=80), enable_cpr=False),
        )
        task = asyncio.create_task(ui.run(lambda: preview(ui)))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting:
                    await asyncio.sleep(0.01)
                ui.editor.text = "/long"
                pipe.send_text("\r")
                while ui._operation is None:
                    await asyncio.sleep(0.01)
                pipe.send_text("\x03")
                while not ui.waiting:
                    await asyncio.sleep(0.01)
            ui.editor.text = "/exit"
            pipe.send_text("\r")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


@pytest.mark.anyio
async def test_replay_preserves_answer_starting_with_a_literal_bullet():
    stream = io.StringIO()
    ui = TerminalChatApp(
        model="preview", workspace="/tmp",
        output=Vt100_Output(stream, lambda: Size(rows=24, columns=80), enable_cpr=False),
    )
    await ui.replay_messages((("answer", "● important finding"),))
    # The display marker and the literal answer bullet should both survive.
    visible = "".join(value for _, value, *_ in to_formatted_text(ANSI(stream.getvalue())))
    assert "● ● important finding" in visible

from __future__ import annotations

import asyncio
import io
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from agent_runtime.streaming.events import ItemStatus, TurnItemKind, item_completed
from agent_runtime.terminal_app import TerminalChatApp


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["keyboard", "mouse"])
async def test_connection_prompt_waits_for_human_action_and_preserves_draft(monkeypatch, action):
    opened = AsyncMock()
    monkeypatch.setattr(TerminalChatApp, "open_connections", opened, raising=False)
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="test", workspace="/tmp", ui_mode="fullscreen", input=pipe,
                             output=Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=100), enable_cpr=False))
        task = asyncio.create_task(ui.run(ui.prompt))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(0.01)
            ui.editor.text = "unfinished question"
            ui.display.begin_turn()
            await ui.display.emit(item_completed(
                turn_id="t", item_id="connect", item_kind=TurnItemKind.TOOL, status=ItemStatus.FAILED,
                error="Please connect",
                data={"result": {"tool_name": "mcp__notion__connect", "error_code": "mcp_connection_required",
                                 "error_message": "Please connect", "metadata": {"mcp_server": "notion"}}},
            ))
            await asyncio.sleep(0.08)
            opened.assert_not_awaited()
            screen = ui.app.renderer._last_screen
            visible = "\n".join("".join(cell.char for cell in row.values()) for row in screen.data_buffer.values())
            assert "打开连接管理" in visible
            if action == "keyboard":
                pipe.send_text("\x1bOR")  # F3: human action, not a model tool or startup effect.
            else:
                window = next(window for window in screen.visible_windows_to_write_positions
                              if getattr(window.content, "text", None) == ui._connection_fragments)
                row = screen.visible_windows_to_write_positions[window].ypos + 1
                pipe.send_text(f"\x1b[<0;4;{row}M\x1b[<0;4;{row}m")
            async with asyncio.timeout(3):
                while opened.await_count == 0:
                    await asyncio.sleep(0.01)
            assert ui.editor.text == "unfinished question"
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.anyio
async def test_human_opened_connection_manager_is_reused_and_closed_with_chat(monkeypatch):
    browser_urls = []
    monkeypatch.setattr("agent_runtime.terminal_app.webbrowser.open", lambda url: browser_urls.append(url) or True)
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="test", workspace="/tmp", input=pipe,
                             output=Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=80), enable_cpr=False))
        task = asyncio.create_task(ui.run(ui.prompt))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting:
                    await asyncio.sleep(0.01)
            assert browser_urls == []
            await ui.open_connections()
            server = ui._connection_server
            assert server is not None and ui._connection_thread.is_alive()
            await ui.open_connections()
            assert ui._connection_server is server
            assert browser_urls == [server.browser_url, server.browser_url]
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        assert not ui._connection_thread.is_alive()
        assert server.fileno() == -1

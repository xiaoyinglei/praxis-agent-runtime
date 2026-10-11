from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agent_runtime import cli
from agent_runtime.result import AgentPause
from agent_runtime.terminal_app import TerminalChatApp


def pause(options=("approve", "deny"), *, request_id="request"):
    return AgentPause(request_id=request_id, kind="tool_approval", question="允许执行此操作？", options=options)


async def wait_for(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.01)


@pytest.mark.parametrize("typed,options,expected", [
    ("Y", ("approve", "deny"), "approve"),
    ("y", ("approve", "deny"), "approve"),
    ("yes", ("approve", "deny"), "approve"),
    ("Y", ("allow_once", "deny"), "allow_once"),
    ("N", ("approve", "deny"), "deny"),
])
def test_plain_approval_shortcut_matches_offered_option(monkeypatch, typed, options, expected):
    replies = iter([typed, "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cli._handle_pause(SimpleNamespace(pause=pause(options))) == expected


@pytest.mark.parametrize("invalid", ["Y", "y", "allow_once", "approve", "c"])
def test_plain_approval_cannot_choose_unoffered_action(monkeypatch, invalid):
    replies = iter([invalid, "N", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cli._handle_pause(SimpleNamespace(pause=pause(("deny",)))) == "deny"


@pytest.mark.anyio
@pytest.mark.parametrize("key,expected", [("y", "approve"), ("Y", "approve"), ("n", "deny"),
                                          ("N", "deny"), ("\x1b", None), ("\x03", None), ("\r", "deny")])
async def test_approval_keys_work_without_typing_action_or_enter(key, expected):
    selected = []
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake", workspace="/tmp", input=pipe, output=DummyOutput())
        ui.editor.text = "未发送的草稿"

        async def conversation():
            selected.append(await ui.choose_approval(pause()))

        task = asyncio.create_task(ui.run(conversation))
        try:
            await wait_for(lambda: ui.status == "等待确认")
            pipe.send_text(key)
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert selected == [expected]
        assert ui.editor.text == "未发送的草稿"


@pytest.mark.anyio
async def test_deny_only_menu_ignores_yes_and_normal_chat_keeps_y():
    selected = []
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake", workspace="/tmp", input=pipe, output=DummyOutput())

        async def conversation():
            selected.append(await ui.choose_approval(pause(("deny",))))
            selected.append(await ui.prompt())

        task = asyncio.create_task(ui.run(conversation))
        try:
            await wait_for(lambda: ui.status == "等待确认")
            pipe.send_text("Y")
            await asyncio.sleep(.05)
            assert selected == []
            pipe.send_text("N")
            await wait_for(lambda: selected == ["deny"] and ui.waiting)
            pipe.send_text("Y\r")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert selected == ["deny", "Y"]


@pytest.mark.anyio
async def test_mouse_confirmation_is_bound_to_request_and_rejects_drag():
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    def event(kind, x=0):
        ui.app.key_processor.before_key_press.fire()
        return MouseEvent(Point(x, 0), kind, MouseButton.LEFT, frozenset())

    selected = []
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake", workspace="/tmp", input=pipe, output=DummyOutput())

        async def conversation():
            selected.append(await ui.choose_approval(pause(request_id="first")))
            selected.append(await ui.choose_approval(pause(request_id="second")))

        task = asyncio.create_task(ui.run(conversation))
        try:
            await wait_for(lambda: ui.status == "等待确认")
            old = next(f[2] for f in ui._approval_fragments() if "允许" in f[1])
            old(event(MouseEventType.MOUSE_UP))
            assert selected == []
            old(event(MouseEventType.MOUSE_DOWN))
            old(event(MouseEventType.MOUSE_MOVE, 5))
            old(event(MouseEventType.MOUSE_UP, 5))
            assert selected == []
            old(event(MouseEventType.MOUSE_DOWN))
            old(event(MouseEventType.MOUSE_UP))
            await wait_for(lambda: selected == ["approve"] and ui.status == "等待确认")
            old(event(MouseEventType.MOUSE_DOWN))
            old(event(MouseEventType.MOUSE_UP))
            await asyncio.sleep(.05)
            assert selected == ["approve"]
            pipe.send_text("N")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert selected == ["approve", "deny"]


@pytest.mark.anyio
@pytest.mark.parametrize("rows,columns", [(8, 40), (10, 60), (24, 100)])
async def test_rendered_approval_keeps_buttons_visible_scrolls_and_accepts_mouse(rows, columns):
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    from agent_runtime.result import AgentToolSummary

    selected = []
    with create_pipe_input() as pipe:
        output = Vt100_Output(io.StringIO(), lambda: Size(rows=rows, columns=columns), enable_cpr=True)
        ui = TerminalChatApp(model="fake", workspace="/tmp", input=pipe, output=output)
        request = AgentPause(request_id="long", kind="tool_approval", question="允许读取公开网页？",
                            options=("approve", "deny"), tool_calls=(
                                AgentToolSummary(tool_call_id="call", tool_name="web_fetch",
                                                 args_preview="参数预览\n" + "长参数内容\n" * 50),))

        async def conversation():
            selected.append(await ui.choose_approval(request))

        task = asyncio.create_task(ui.run(conversation))
        try:
            await wait_for(lambda: ui.status == "等待确认" and ui.app.renderer._last_screen is not None)
            pipe.send_text("\x1b[1;1R")
            await asyncio.sleep(.03)
            ui.app._redraw()
            screen = ui.app.renderer._last_screen
            visible = "\n".join("".join(cell.char for _, cell in sorted(line.items()))
                                for _, line in sorted(screen.data_buffer.items()))
            assert "Window too small" not in visible
            assert all(label in visible for label in ("允许一次", "拒绝", "取消"))
            pipe.send_text("\t\x1b[6~")  # Tab to the read-only body, PageDown to inspect longer arguments.
            await asyncio.sleep(.05)
            ui.app._redraw()
            body = next(w for w in ui.app.layout.find_all_windows() if w.content is ui.approval_body.control)
            assert max(body.render_info.displayed_lines) > 0
            buttons = next(w for w in ui.app.layout.find_all_windows() if w.content is ui.approval_buttons)
            y, x = buttons.render_info._rowcol_to_yx[(0, 4)]
            y += ui.app.renderer.rows_above_layout
            body_y, body_x = min(body.render_info._rowcol_to_yx.values())
            body_y += ui.app.renderer.rows_above_layout
            pipe.send_text(f"\x1b[<0;{x + 1};{y + 1}M"
                           f"\x1b[<32;{body_x + 1};{body_y + 1}M"
                           f"\x1b[<0;{x + 1};{y + 1}m")
            await asyncio.sleep(.05)
            assert selected == []  # Drag across controls and back must not confirm.
            pipe.send_text(f"\x1b[<0;{x + 1};{y + 1}M\x1b[<0;{x + 1};{y + 1}m")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert selected == ["approve"]


@pytest.mark.anyio
async def test_arrow_selection_and_enter_work_when_mouse_is_disabled():
    selected = []
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake", workspace="/tmp", input=pipe, output=DummyOutput())

        async def conversation():
            selected.append(await ui.choose_approval(pause()))

        task = asyncio.create_task(ui.run(conversation))
        try:
            await wait_for(lambda: ui.status == "等待确认")
            pipe.send_text("\x1bOQ\x1b[A\r")  # F2 disables mouse; Up selects allow, Enter confirms.
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert not ui.app.mouse_support()
        assert selected == ["approve"]


@pytest.mark.anyio
async def test_empty_options_never_synthesizes_permission():
    ui = TerminalChatApp(model="fake", workspace="/tmp", output=DummyOutput())
    assert await ui.choose_approval(pause(())) is None
    assert cli._handle_pause(SimpleNamespace(pause=pause(()))) is None


def test_empty_option_clarification_remains_a_text_answer(monkeypatch):
    request = AgentPause(request_id="question", kind="clarification", question="你的工作年限？")
    monkeypatch.setattr("builtins.input", lambda _: "两年")
    assert cli._handle_pause(SimpleNamespace(pause=request)) == "两年"

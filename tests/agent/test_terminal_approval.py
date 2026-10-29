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
    ("S", ("approve", "allow_session", "deny"), "allow_session"),
])
def test_plain_approval_shortcut_matches_offered_option(monkeypatch, typed, options, expected):
    replies = iter([typed, "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cli._handle_pause(SimpleNamespace(pause=pause(options))) == expected


@pytest.mark.parametrize("invalid", ["Y", "y", "allow_once", "approve", "c", "s"])
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
@pytest.mark.parametrize("mode", ["inline", "fullscreen"])
async def test_arrow_selection_and_enter_work_when_mouse_is_disabled(mode):
    selected = []
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake", workspace="/tmp", ui_mode=mode, input=pipe, output=DummyOutput())
        ui._mouse_enabled = False  # Explicit native-selection mode, independently of each mode's default.

        async def conversation():
            selected.append(await ui.choose_approval(pause()))

        task = asyncio.create_task(ui.run(conversation))
        try:
            await wait_for(lambda: ui.status == "等待确认")
            assert ui.app.mouse_support() is False
            # Keyboard selection works without enabling mouse reporting in either mode.
            pipe.send_text("\x1b[A\r")
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


@pytest.mark.anyio
async def test_session_grant_selector_and_visible_scope():
    selected = []
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake", workspace="/tmp", input=pipe, output=DummyOutput())
        request = AgentPause(request_id="scoped", kind="tool_approval", question="允许命令？",
                             options=("approve", "allow_session", "deny"),
                             context={"session_grant_label": "git status [可追加参数]；目录 /tmp；只读沙箱"})

        async def conversation():
            selected.append(await ui.choose_approval(request))

        task = asyncio.create_task(ui.run(conversation))
        await wait_for(lambda: ui.status == "等待确认")
        assert "git status" in ui.approval_body.text
        labels = "".join(fragment[1] for fragment in ui._approval_fragments())
        assert "本会话" in labels
        assert " · S" not in labels and " · Y" not in labels and " · N" not in labels
        assert "Enter" in "".join(fragment[1] for fragment in ui._footer())
        pipe.send_text("\x1b[A\r")  # Move from default deny to the grant option and confirm.
        await asyncio.wait_for(task, 3)
        assert selected == ["allow_session"]


@pytest.mark.anyio
async def test_run_approval_uses_selection_dialog_without_typed_answer(monkeypatch):
    requests = []

    class Dialog:
        async def run_async(self):
            return "approve"

    def dialog(**kwargs):
        requests.append(kwargs)
        return Dialog()

    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda: True)
    monkeypatch.setattr("prompt_toolkit.shortcuts.radiolist_dialog", dialog)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("approval must use the selector"))
    result = SimpleNamespace(pause=pause())
    assert await cli._handle_cli_pause(result) == "approve"
    assert requests[0]["values"] == [("approve", "允许一次"), ("deny", "拒绝")]


@pytest.mark.anyio
@pytest.mark.parametrize("selection", ["full", None])
async def test_run_approval_can_change_mode_or_return_without_approving(monkeypatch, selection):
    requests = []
    choices = iter(["permissions", selection, "deny"])
    changes = []

    class Dialog:
        async def run_async(self):
            return next(choices)

    def dialog(**kwargs):
        requests.append(kwargs)
        return Dialog()

    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda: True)
    monkeypatch.setattr("prompt_toolkit.shortcuts.radiolist_dialog", dialog)
    action = await cli._handle_cli_pause(SimpleNamespace(pause=pause()),
        approval_mode="auto", set_approval_mode=changes.append)
    assert action == ("recheck" if selection else "deny")
    assert changes == (["full"] if selection else [])
    assert ("permissions", "更改审批模式…") in requests[0]["values"]
    assert requests[1]["default"] == "auto"


def test_empty_option_clarification_remains_a_text_answer(monkeypatch):
    request = AgentPause(request_id="question", kind="clarification", question="你的工作年限？")
    monkeypatch.setattr("builtins.input", lambda _: "两年")
    assert cli._handle_pause(SimpleNamespace(pause=request)) == "两年"


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_menu", [False, True])
@pytest.mark.parametrize("entry", ["f4", "button"])
async def test_permissions_picker_during_approval_preserves_request_and_draft(cancel_menu, entry):
    selected = []
    changes = []
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake", workspace="/tmp", input=pipe, output=DummyOutput())
        ui.editor.text = "原有草稿"

        async def permissions():
            choice = await ui.choose_menu(["ask", "auto", "full"], title="选择审批模式", current="auto")
            if choice is None:
                return False
            changes.append(choice)
            ui.approval_mode = choice
            return True

        ui.on_permissions = permissions

        async def conversation():
            selected.append(await ui.choose_approval(pause()))

        task = asyncio.create_task(ui.run(conversation))
        try:
            await wait_for(lambda: ui.status == "等待确认")
            assert "帮我批准" in ui.approval_body.text
            assert "更改审批模式" in "".join(f[1] for f in ui._approval_fragments())
            # Either F4 or the visible button opens the same picker.
            pipe.send_text("\x1bOS" if entry == "f4" else "\x1b[B\r")
            await wait_for(lambda: ui._model_pending is not None)
            pipe.send_text("\x1b" if cancel_menu else "\x1b[B\r")
            if cancel_menu:
                await wait_for(lambda: ui._model_pending is None)
                assert selected == [] and changes == []
                assert ui.app.layout.has_focus(ui.approval_buttons)
                pipe.send_text("n")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert selected == (["deny"] if cancel_menu else ["recheck"])
        assert changes == ([] if cancel_menu else ["full"])
        assert ui.editor.text == "原有草稿"


@pytest.mark.anyio
async def test_idle_f4_picker_is_visible_and_keeps_unsent_input():
    selected = []
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake", workspace="/tmp", input=pipe, output=DummyOutput())

        async def permissions():
            selected.append(await ui.choose_menu(["ask", "auto", "full"], title="选择审批模式", current="auto"))
            return False

        ui.on_permissions = permissions

        async def conversation():
            assert await ui.prompt() == "原有草稿"

        task = asyncio.create_task(ui.run(conversation))
        try:
            await wait_for(lambda: ui.waiting)
            ui.editor.text = "原有草稿"
            footer = "".join(f[1] for f in ui._footer())
            assert "帮我批准" in footer and "F4" in footer
            pipe.send_text("\x1bOS")
            await wait_for(lambda: ui._model_pending is not None)
            pipe.send_text("\x1b")
            await wait_for(lambda: selected == [None])
            pipe.send_text("\r")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task


@pytest.mark.anyio
async def test_new_approval_does_not_steal_focus_from_visible_permissions_menu():
    selected = []
    choices = []
    ready = asyncio.Event()
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake", workspace="/tmp", input=pipe, output=DummyOutput())

        async def permissions():
            choices.append(await ui.choose_menu(["ask", "auto", "full"], title="选择审批模式", current="auto"))
            return False

        ui.on_permissions = permissions

        async def conversation():
            await ready.wait()
            selected.append(await ui.choose_approval(pause()))

        task = asyncio.create_task(ui.run(conversation))
        try:
            pipe.send_text("\x1bOS")
            await wait_for(lambda: ui._model_pending is not None)
            ready.set()
            await wait_for(lambda: ui._approval is not None)
            assert ui.app.layout.has_focus(ui.model_menu)
            pipe.send_text("\x1b[A\r")
            await wait_for(lambda: choices == ["ask"])
            assert selected == []
            pipe.send_text("n")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert selected == ["deny"]


@pytest.mark.anyio
async def test_new_prompt_does_not_steal_focus_from_visible_permissions_menu():
    choices = []
    ready = asyncio.Event()
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="fake", workspace="/tmp", input=pipe, output=DummyOutput())

        async def permissions():
            choices.append(await ui.choose_menu(["ask", "auto", "full"], title="选择审批模式", current="auto"))
            return False

        ui.on_permissions = permissions

        async def conversation():
            await ready.wait()
            assert await ui.prompt() == "continue"

        task = asyncio.create_task(ui.run(conversation))
        try:
            pipe.send_text("\x1bOS")
            await wait_for(lambda: ui._model_pending is not None)
            ready.set()
            await wait_for(lambda: ui._pending is not None)
            assert ui.app.layout.has_focus(ui.model_menu)
            pipe.send_text("\x1b[A\r")
            await wait_for(lambda: choices == ["ask"])
            pipe.send_text("continue\r")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task


@pytest.mark.anyio
@pytest.mark.parametrize("ui_mode", ["inline", "fullscreen"])
async def test_small_terminal_shows_mode_entry_and_picker_with_live_input(ui_mode):
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    with create_pipe_input() as pipe:
        output = Vt100_Output(io.StringIO(), lambda: Size(rows=8, columns=40), enable_cpr=False)
        ui = TerminalChatApp(model="fake", workspace="/tmp", ui_mode=ui_mode, input=pipe, output=output)

        async def permissions():
            await ui.choose_menu(["请求批准", "帮我批准", "完全访问"], title="选择审批模式", current="帮我批准")
            return False

        ui.on_permissions = permissions
        ui.on_input = lambda _: asyncio.sleep(0)

        def visible():
            ui.app._redraw()
            screen = ui.app.renderer._last_screen
            return "\n".join("".join(cell.char for _, cell in sorted(line.items()))
                             for _, line in sorted(screen.data_buffer.items()))

        async def conversation():
            assert await ui.choose_approval(pause()) == "deny"

        task = asyncio.create_task(ui.run(conversation))
        try:
            await wait_for(lambda: ui._approval is not None and ui.app.renderer._last_screen is not None)
            screen = visible()
            assert "Window too small" not in screen
            assert all(label in screen for label in ("允许一次", "更改审批模式", "拒绝", "取消", "F4"))
            pipe.send_text("\x1bOS")
            await wait_for(lambda: ui._model_pending is not None)
            screen = visible()
            assert "Window too small" not in screen and "完全访问" in screen
            pipe.send_text("\x1b")
            await wait_for(lambda: ui._model_pending is None)
            pipe.send_text("n")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

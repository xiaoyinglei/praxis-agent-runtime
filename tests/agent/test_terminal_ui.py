from __future__ import annotations

import pytest

from agent_runtime.streaming.events import ItemStatus, TurnItemKind, item_completed, item_started, text_delta
from agent_runtime.terminal_ui import Conversation, ConversationEventDisplay


def test_nested_folds_toggle_in_place_without_duplicating_output():
    view = Conversation()
    group = view.add_group()
    tool = view.add_tool(group, "读取文件")
    tool.append("unique detail")
    group.expanded = False
    assert "unique detail" not in view.plain_text()
    view.toggle(group)
    assert "读取文件" in view.plain_text()
    assert "unique detail" not in view.plain_text()
    view.toggle(tool)
    assert view.plain_text().count("unique detail") == 1
    view.toggle(group)
    assert "unique detail" not in view.plain_text()
    view.toggle(group)
    assert view.plain_text().count("unique detail") == 1
    view.toggle(tool)
    assert "unique detail" not in view.plain_text()


@pytest.mark.anyio
async def test_events_preserve_answer_and_tool_details_without_stdout(capsys):
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    start = item_started(
        turn_id="turn",
        item_id="tool",
        item_kind=TurnItemKind.TOOL,
        data={"tool_name": "read_file", "input_preview": "hello.py"},
    )
    await display.emit(start)
    await display.emit(start)
    await display.emit(
        item_completed(
            turn_id="turn",
            item_id="tool",
            item_kind=TurnItemKind.TOOL,
            status=ItemStatus.SUCCESS,
            data={"result": {"tool_name": "read_file", "structured_content": "unique body"}},
        )
    )
    await display.emit(text_delta("完成", turn_id="turn"))
    display.finish()
    assert capsys.readouterr().out == ""
    assert "完成" in view.plain_text()
    group = display.group
    assert group is not None and len(group.children) == 1
    assert not group.expanded
    view.toggle(group)
    view.toggle(group.children[0])
    assert view.plain_text().count("unique body") == 1
    assert display.answer_streamed


def test_tool_details_are_bounded_and_report_omission():
    view = Conversation()
    tool = view.add_tool(view.add_group(), "command")
    tool.append("x" * 200_000)
    tool.expanded = True
    assert len(tool.text) < 140_000
    assert "省略" in view.plain_text()


@pytest.mark.anyio
async def test_real_application_keyboard_toggles_and_multiline_submission():
    import asyncio

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="test", workspace="/tmp", input=pipe, output=DummyOutput())
        group = ui.view.add_group()
        group.expanded = False
        tool = ui.view.add_tool(group, "读取文件")
        tool.append("secret detail")
        submitted = []

        async def conversation():
            submitted.append(await ui.prompt())

        task = asyncio.create_task(ui.run(conversation))
        for _ in range(100):
            if ui.waiting:
                break
            await asyncio.sleep(0.01)
        pipe.send_text("\x0f")  # Ctrl+O: toggle latest group without leaving input.
        await asyncio.sleep(0.05)
        assert group.expanded
        pipe.send_text("\x0f")
        await asyncio.sleep(0.05)
        assert not group.expanded
        pipe.send_text("你好\x1b\r第二行\r")
        await asyncio.wait_for(task, 3)
        assert submitted == ["你好\n第二行"]


def test_mouse_handler_repeatedly_toggles_same_block():
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp

    ui = TerminalChatApp(model="test", workspace="/tmp", output=DummyOutput())
    group = ui.view.add_group()
    click = MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset())
    handler = next(f[2] for f in ui.fragments() if len(f) == 3 and "执行记录" in f[1])
    handler(click)
    assert not group.expanded
    handler(click)
    assert group.expanded


@pytest.mark.anyio
async def test_expanding_tool_preserves_middle_of_long_result():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await display.emit(
        item_completed(
            turn_id="t",
            item_id="read",
            item_kind=TurnItemKind.TOOL,
            status=ItemStatus.SUCCESS,
            data={
                "result": {"tool_name": "read_file", "structured_content": "\n".join(f"row-{i:03}" for i in range(100))}
            },
        )
    )
    display.finish()
    display.set_verbose(True)
    assert "row-050" in view.plain_text()
    assert view.plain_text().count("row-050") == 1


@pytest.mark.anyio
async def test_command_chunks_stay_with_their_tool_when_interleaved():
    from agent_runtime.streaming.events import ItemDeltaKind, item_delta

    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    for identity in ("one", "two"):
        await display.emit(
            item_started(
                turn_id="t", item_id=identity, item_kind=TurnItemKind.COMMAND, data={"tool_name": "run_command"}
            )
        )
    for identity, content in (("two", "second\n"), ("one", "first\n"), ("two", "tail")):
        await display.emit(
            item_delta(
                turn_id="t",
                item_id=identity,
                item_kind=TurnItemKind.COMMAND,
                delta_kind=ItemDeltaKind.COMMAND_STDOUT,
                delta=content,
            )
        )
    for identity in ("one", "two"):
        await display.emit(
            item_completed(
                turn_id="t",
                item_id=identity,
                item_kind=TurnItemKind.COMMAND,
                status=ItemStatus.SUCCESS,
                data={"result": {"tool_name": "run_command"}},
            )
        )
    one, two = display.group.children
    assert "first" in one.text and "second" not in one.text
    assert two.text.count("second") == two.text.count("tail") == 1


@pytest.mark.anyio
async def test_error_remains_visible_when_user_collapses_group():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await display.emit(
        item_completed(
            turn_id="t",
            item_id="bad",
            item_kind=TurnItemKind.TOOL,
            status=ItemStatus.FAILED,
            error="missing-file",
            data={"result": {"tool_name": "read_file", "error_message": "missing-file"}},
        )
    )
    display.finish()
    display.group.expanded = False
    assert "missing-file" in view.plain_text()


@pytest.mark.anyio
async def test_rendered_mouse_coordinates_and_resize():
    import asyncio
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    from agent_runtime.terminal_app import TerminalChatApp

    terminal_size = [Size(rows=24, columns=80)]
    output = Vt100_Output(io.StringIO(), lambda: terminal_size[0], enable_cpr=True)
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="demo", workspace="/tmp/中文工作区", input=pipe, output=output)
        ui.view.message("检查测试\n", "user")
        group = ui.view.add_group()
        child = ui.view.add_tool(group, "读取文件 · demo.py")
        child.append("unique-result\n" + "中文长输出" * 30)
        group.expanded = False
        task = asyncio.create_task(ui.run(ui.prompt))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(0.01)

            async def click(block):
                pipe.send_text("\x1b[1;1R")
                await asyncio.sleep(0.03)
                ui.app._redraw()
                window = next(w for w in ui.app.layout.find_all_windows() if w.content is ui.body)
                row = next(row for row, current in ui._fold_rows.items() if current is block)
                y, x = window.render_info._rowcol_to_yx[(row, 0)]
                y += ui.app.renderer.rows_above_layout
                pipe.send_text(f"\x1b[<0;{x + 2};{y + 1}M\x1b[<0;{x + 2};{y + 1}m")
                await asyncio.sleep(0.08)

            await click(group)
            assert group.expanded
            await click(child)
            assert child.expanded
            ui.app._redraw()
            screen = ui.app.renderer._last_screen
            visible = "\n".join(
                "".join(cell.char for _, cell in sorted(row.items())) for _, row in sorted(screen.data_buffer.items())
            )
            assert "unique-result" in visible
            terminal_size[0] = Size(rows=16, columns=40)
            ui.app._on_resize()
            await asyncio.sleep(0.08)
            await click(child)
            assert not child.expanded
            await click(group)
            assert not group.expanded
            pipe.send_text("done\r")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                from contextlib import suppress

                with suppress(asyncio.CancelledError):
                    await task


def test_conversation_has_a_total_detail_budget():
    view = Conversation(max_characters=300)
    group = view.add_group()
    for number in range(10):
        view.add_tool(group, f"tool-{number}").append("x" * 100)
    view.invalidate()
    assert sum(len(child.text) for child in group.children) <= 300
    assert group.children[0].omitted > 0
    assert group.children[-1].text == "x" * 100


@pytest.mark.anyio
async def test_tool_index_evicts_old_records_and_late_output_remains_visible():
    from agent_runtime.streaming.events import ItemDeltaKind, item_delta

    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    for number in range(270):
        await display.emit(
            item_started(turn_id="t", item_id=str(number), item_kind=TurnItemKind.TOOL, data={"tool_name": "read_file"})
        )
    await display.emit(
        item_delta(
            turn_id="t",
            item_id="0",
            item_kind=TurnItemKind.TOOL,
            delta_kind=ItemDeltaKind.TOOL_PROGRESS,
            delta="late output",
        )
    )
    assert len(display._tool_blocks) == len(display.group.children) == 256
    assert "late output" in display.group.children[-1].text


def test_assistant_role_marker_does_not_break_markdown_heading():
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp

    ui = TerminalChatApp(model="test", workspace="/tmp", output=DummyOutput())
    ui.view.message("# Heading\n\n```python\nprint(1)\n```")
    visible = "".join(fragment[1] for fragment in ui.fragments())
    assert "Heading" in visible and "# Heading" not in visible
    assert "```" not in visible


@pytest.mark.anyio
async def test_nested_tool_events_do_not_change_cancellation_target():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    for turn_id in ("parent-turn", "child-turn"):
        await display.emit(
            item_started(turn_id=turn_id, item_id="tool", item_kind=TurnItemKind.TOOL, data={"tool_name": "read_file"})
        )
    assert display.turn_id == "parent-turn"


@pytest.mark.anyio
async def test_whitespace_does_not_create_answer_or_replace_thinking_status():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await display.emit(text_delta("\n\n", turn_id="t"))
    assert not display.answer_streamed
    assert not any(block.kind == "answer" for block in view.blocks)
    assert "正在回答" not in display.activity_text()
    await display.emit(text_delta("    code", turn_id="t"))
    assert display.answer_streamed
    assert view.blocks[-1].text == "\n\n    code"


def test_scrolling_reuses_long_history_layout(monkeypatch):
    from prompt_toolkit.output import DummyOutput

    from agent_runtime import terminal_app

    ui = terminal_app.TerminalChatApp(model="test", workspace="/tmp", output=DummyOutput())
    for number in range(40):
        ui.view.message(str(number), "user")
        ui.view.message(f"## Answer {number}\n\n" + "long answer " * 20)
    ui.fragments()
    def unexpected(*args, **kwargs):
        pytest.fail("Scrolling unchanged history must not repeat Markdown layout")
    monkeypatch.setattr(terminal_app, "_markdown", unexpected)
    ui._move(-10)
    ui.fragments()


@pytest.mark.anyio
async def test_model_picker_arrows_enter_and_escape():
    import asyncio

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="first", workspace="/tmp", input=pipe, output=DummyOutput())
        selected = []
        async def conversation():
            selected.append(await ui.choose_model(["first", "second", "third"], "first"))
            selected.append(await ui.choose_model(["first", "second"], "second"))
        task = asyncio.create_task(ui.run(conversation))
        await asyncio.sleep(.05)
        pipe.send_text("\x1b[B\x1b[B\x1b[A\r")
        await asyncio.sleep(.1)
        pipe.send_text("\x1b")
        await asyncio.wait_for(task, 2)
        assert selected == ["second", None]


@pytest.mark.anyio
async def test_model_picker_fits_terminal_and_scrolls_after_resize():
    import asyncio
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    from agent_runtime.terminal_app import TerminalChatApp

    size = [Size(rows=24, columns=80)]
    with create_pipe_input() as pipe:
        output = Vt100_Output(io.StringIO(), lambda: size[0], enable_cpr=True)
        ui = TerminalChatApp(model="model-0", workspace="/tmp", input=pipe, output=output)
        selected = []

        async def conversation():
            selected.append(await ui.choose_model([f"model-{i}" for i in range(12)], "model-0"))

        task = asyncio.create_task(ui.run(conversation))
        try:
            async with asyncio.timeout(3):
                while ui._model_pending is None or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(.01)
            for height in (24, 10, 12, 40):
                size[0] = Size(rows=height, columns=60)
                ui.app._on_resize()
                pipe.send_text("\x1b[1;1R")
                await asyncio.sleep(.06)
                ui.app._redraw()
                screen = ui.app.renderer._last_screen
                visible = "\n".join(
                    "".join(cell.char for _, cell in sorted(row.items()))
                    for _, row in sorted(screen.data_buffer.items())
                )
                assert "Window too small" not in visible
                assert "model-0" in visible
            size[0] = Size(rows=10, columns=60)
            ui.app._on_resize()
            pipe.send_text("\x1b[1;1R" + "\x1b[B" * 11)
            await asyncio.sleep(.08)
            ui.app._redraw()
            window = next(w for w in ui.app.layout.find_all_windows() if w.content is ui.model_menu)
            assert 11 in window.render_info.displayed_lines
            pipe.send_text("\r")
            await asyncio.wait_for(task, 3)
            assert selected == ["model-11"]
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


def test_drag_selection_copies_unicode_without_toggling(monkeypatch):
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp

    ui = TerminalChatApp(model="test", workspace="/tmp", output=DummyOutput())
    ui.view.message("你好 Python", "notice")
    ui.fragments()
    copied = []
    monkeypatch.setattr(ui, "_copy_text", copied.append)
    mouse = ui._handler(None, 0)
    for kind, x in [(MouseEventType.MOUSE_DOWN, 0), (MouseEventType.MOUSE_MOVE, 5), (MouseEventType.MOUSE_UP, 5)]:
        mouse(MouseEvent(Point(x, 0), kind, MouseButton.LEFT, frozenset()))
    assert copied == ["你好 Py"]
    mouse(MouseEvent(Point(9, 0), MouseEventType.MOUSE_MOVE, MouseButton.NONE, frozenset()))
    assert ui._selected_text() == "你好 Py"


@pytest.mark.anyio
async def test_vt100_drag_paste_and_native_mouse_toggle(monkeypatch):
    import asyncio
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    from agent_runtime.terminal_app import TerminalChatApp

    with create_pipe_input() as pipe:
        output = Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=80), enable_cpr=True)
        ui = TerminalChatApp(model="test", workspace="/tmp", input=pipe, output=output)
        ui.view.message("你好 Python", "notice")
        copied = []
        monkeypatch.setattr(ui, "_copy_text", copied.append)
        task = asyncio.create_task(ui.run(ui.prompt))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(.01)
            pipe.send_text("\x1b[1;1R")
            await asyncio.sleep(.03)
            ui.app._redraw()
            window = next(w for w in ui.app.layout.find_all_windows() if w.content is ui.body)
            y, start = window.render_info._rowcol_to_yx[(0, 0)]
            _, end = window.render_info._rowcol_to_yx[(0, 5)]
            y += ui.app.renderer.rows_above_layout
            pipe.send_text(f"\x1b[<0;{start + 1};{y + 1}M\x1b[<32;{end + 1};{y + 1}M\x1b[<0;{end + 1};{y + 1}m")
            await asyncio.sleep(.08)
            assert copied == ["你好 Py"]
            pipe.send_text("\x1b[200~第一行\n第二行\x1b[201~")
            await asyncio.sleep(.05)
            assert ui.editor.text == "第一行\n第二行"
            assert ui.waiting  # Pasting must never auto-submit.
            pipe.send_text("\x1bOQ")  # F2 releases mouse reporting to the terminal.
            await asyncio.sleep(.05)
            assert not ui.app.mouse_support()
            pipe.send_text("\r")
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

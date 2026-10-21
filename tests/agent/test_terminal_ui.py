from __future__ import annotations

import pytest

from agent_runtime.streaming.events import ItemStatus, TurnItemKind, item_completed, item_started, text_delta
from agent_runtime.terminal_ui import Conversation, ConversationEventDisplay


@pytest.mark.anyio
async def test_web_error_details_are_retained_but_collapsed():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await display.emit(item_completed(
        turn_id="t", item_id="s", item_kind=TurnItemKind.TOOL, status=ItemStatus.FAILED,
        error="RAW detail", data={"result": {"tool_name": "web_search", "error_code": "timeout",
            "error_message": "RAW detail", "structured_content": {"error_message": "RAW detail"}}},
    ))
    display.finish()
    assert "联网请求超时" in view.plain_text()
    assert "RAW detail" not in view.plain_text()
    display.set_verbose(True)
    assert "RAW detail" in view.plain_text()


@pytest.mark.anyio
async def test_failed_tool_keeps_code_in_details_and_readable_preview():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await display.emit(item_completed(
        turn_id="t", item_id="failed", item_kind=TurnItemKind.TOOL, status=ItemStatus.FAILED,
        error="tool runner failed",
        data={"result": {"tool_name": "web_fetch", "error_code": "runner_failed",
                         "error_message": "tool runner failed", "structured_content": None}},
    ))
    title = display.group.children[0].title
    assert "runner_failed" not in title
    assert "网页工具执行异常" in view.plain_text()
    display.set_verbose(True)
    assert "runner_failed" in view.plain_text()
    assert "tool runner failed" in view.plain_text()


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
async def test_completed_answers_enter_native_scrollback_once_and_remain_inspectable():
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    from agent_runtime.terminal_app import TerminalChatApp

    stream = io.StringIO()
    ui = TerminalChatApp(
        model="test", workspace="/tmp",
        output=Vt100_Output(stream, lambda: Size(rows=24, columns=80), enable_cpr=False),
    )
    ui.view.message("first question\n", "user")
    ui.view.message("first complete answer\n")
    await ui.flush_scrollback()
    first = stream.getvalue()
    assert "first complete answer" in first
    assert "first complete answer" not in "".join(t for _, t, *_ in ui.fragments())
    await ui.flush_scrollback()
    assert stream.getvalue() == first
    # Adjacent messages of the same role must not append to a committed block.
    ui.view.message("second complete answer\n")
    assert not ui.view.blocks[-1].in_scrollback
    await ui.flush_scrollback()
    assert stream.getvalue().count("first complete answer") == 1
    assert stream.getvalue().count("second complete answer") == 1
    assert "first complete answer" in ui.view.plain_text()
    ui._move(-10)
    assert "first complete answer" in "".join(t for _, t, *_ in ui.fragments())


@pytest.mark.anyio
async def test_saved_messages_replay_in_full_even_when_inspection_view_is_bounded():
    import io
    import re

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    from agent_runtime.terminal_app import TerminalChatApp
    from agent_runtime.terminal_ui import MAX_TEXT

    stream = io.StringIO()
    ui = TerminalChatApp(
        model="test", workspace="/tmp",
        output=Vt100_Output(stream, lambda: Size(rows=24, columns=80), enable_cpr=False),
    )
    text = "begin saved answer\n" + "middle saved line\n" * (MAX_TEXT // 16) + "end saved answer"
    await ui.replay_messages((("user", "saved question"), ("answer", text)))
    rendered = stream.getvalue()
    assert "saved question" in rendered and "begin saved answer" in rendered and "end saved answer" in rendered
    # Markdown soft breaks and terminal wrapping may split a phrase across rows.
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", rendered)
    assert " ".join(plain.split()).count("middle saved line") == text.count("middle saved line")
    assert ui.view.blocks[-1].omitted > 0
    assert all(block.in_scrollback for block in ui.view.blocks)
    await ui.flush_scrollback()
    assert stream.getvalue() == rendered


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
        ui._mouse_enabled = True  # Explicit application mouse mode; native selection is the default.
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


def test_markdown_inline_and_fenced_code_do_not_force_background_colors():
    from agent_runtime.terminal_app import _markdown

    fragments = _markdown("Use `/home/admin/praxis`.\n\n```python\nprint('你好')\n```", 80, True)
    assert "你好" in "".join(text for _, text, *_ in fragments)
    assert all("bg:" not in style for style, *_ in fragments)


@pytest.mark.parametrize("term", ["xterm-256color", "dumb"])
def test_web_references_keep_clickable_targets_without_visible_control_bytes(monkeypatch, term):
    from agent_runtime.terminal_app import _markdown

    monkeypatch.setenv("TERM", term)
    _markdown.cache_clear()
    url = "https://docs.example.org/research?id=123"
    fragments = _markdown(f"A claim from [Original report]({url}).\n\nAgain [the report]({url}).", 80, True)
    visible = "".join(text for style, text, *_ in fragments if "[ZeroWidthEscape]" not in style)
    controls = "".join(text for style, text, *_ in fragments if "[ZeroWidthEscape]" in style)
    assert "Original report" in visible and "网页引用" in visible and "docs.example.org" in visible
    assert visible.count("[1]") == 1 and "[2]" not in visible
    assert "\x1b" not in visible and "]8;id=" not in visible and url in visible
    if term == "dumb":
        assert not controls
    else:
        assert url in controls and "\x1b]8;" in controls


@pytest.mark.parametrize("target", ["file:///tmp/report", "javascript:alert(1)", "mailto:user@example.org"])
def test_non_web_destinations_are_never_terminal_hyperlinks(target):
    from agent_runtime.terminal_app import _markdown

    fragments = _markdown(f"[Readable label]({target})", 80, True)
    assert "Readable label" in "".join(text for _, text, *_ in fragments)
    assert not any("[ZeroWidthEscape]" in style for style, *_ in fragments)


def test_links_in_code_blocks_are_not_presented_as_cited_sources():
    from agent_runtime.terminal_app import _markdown

    fragments = _markdown("```md\n[example](https://example.org)\n```", 80, False)
    assert "网页引用" not in "".join(text for _, text, *_ in fragments)
    assert not any("[ZeroWidthEscape]" in style for style, *_ in fragments)


def test_reference_controls_do_not_pollute_mouse_selection_text():
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp

    ui = TerminalChatApp(model="test", workspace="/tmp", output=DummyOutput())
    ui.view.message("Read [Source](https://example.org/report).")
    ui.fragments()
    assert "网页引用" in "\n".join(ui._plain_lines)
    assert all("\x1b" not in line and "id=" not in line for line in ui._plain_lines)


@pytest.mark.anyio
async def test_all_slash_commands_complete_before_enter_can_submit():
    import asyncio

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp
    from agent_runtime.terminal_input import CHAT_COMMANDS

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="test", workspace="/tmp", input=pipe, output=DummyOutput())
        submitted = []

        async def conversation():
            for _command in CHAT_COMMANDS:
                submitted.append(await ui.prompt())

        task = asyncio.create_task(ui.run(conversation))

        async def wait_for(condition):
            for _ in range(100):
                if condition():
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("Completion state did not become ready")

        for index, command in enumerate(CHAT_COMMANDS):
            await wait_for(lambda index=index: ui.waiting and len(submitted) == index)
            pipe.send_text(command[:-1])
            await wait_for(lambda: ui.editor.buffer.complete_state is not None)
            pipe.send_text("\r")
            await wait_for(lambda command=command:
                           ui.editor.text == command and ui.editor.buffer.complete_state is None)
            assert len(submitted) == index
            pipe.send_text("\r")
        await asyncio.wait_for(task, 3)
        assert submitted == list(CHAT_COMMANDS)


@pytest.mark.anyio
async def test_native_mouse_and_idle_prompt_preserve_multiline_paste():
    import asyncio
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    from agent_runtime.terminal_app import TerminalChatApp

    stream = io.StringIO()
    with create_pipe_input() as pipe:
        output = Vt100_Output(stream, lambda: Size(rows=24, columns=80), enable_cpr=False)
        ui = TerminalChatApp(model="test", workspace="/tmp", input=pipe, output=output)
        submitted = []

        async def conversation():
            submitted.append(await ui.prompt())

        task = asyncio.create_task(ui.run(conversation))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(.01)
            assert not ui.app.mouse_support()
            assert "\x1b[?1000h" not in stream.getvalue()
            pipe.send_text("\x1b[200~你好\n第二行👩‍💻\x1b[201~")
            await asyncio.sleep(.1)
            assert ui.waiting and not submitted
            assert ui.editor.text == "你好\n第二行👩‍💻"
            pipe.send_text("\x7f\x0a尾行")  # Delete an entire grapheme, Ctrl+J adds a newline.
            await asyncio.sleep(.1)
            assert ui.editor.text == "你好\n第二行\n尾行"
            renders = ui.app.render_counter
            await asyncio.sleep(.3)
            assert ui.app.render_counter == renders  # No idle redraw clearing native selection.
            pipe.send_text("\r")
            await asyncio.wait_for(task, 3)
            assert submitted == ["你好\n第二行\n尾行"]
        finally:
            if not task.done():
                task.cancel()
                from contextlib import suppress

                with suppress(asyncio.CancelledError):
                    await task


@pytest.mark.anyio
async def test_ssh_copy_requests_client_clipboard_without_claiming_success(monkeypatch):
    import asyncio
    import base64
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    from agent_runtime import terminal_app

    monkeypatch.setenv("SSH_CONNECTION", "client server")
    monkeypatch.setattr(terminal_app.sys, "platform", "linux")

    async def unexpected(*args, **kwargs):
        pytest.fail("An SSH server must not invoke a local clipboard program")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected)
    stream = io.StringIO()
    ui = terminal_app.TerminalChatApp(
        model="test", workspace="/tmp",
        output=Vt100_Output(stream, lambda: Size(rows=24, columns=80), enable_cpr=False),
    )
    await ui.copy_answer("# 你好\n完整的回答")
    encoded = base64.b64encode("# 你好\n完整的回答".encode()).decode()
    assert f"\x1b]52;c;{encoded}\x07" in stream.getvalue()
    assert "完整的回答" in stream.getvalue()  # Native scrollback fallback, no stdout redirection.
    assert "已请求" in ui.status and ui.status != "已复制"
    ui.app.clipboard.set_text("old kill ring")
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

    ui._mouse_enabled = True
    ui._handler(None, 0)(MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.RIGHT, frozenset()))
    assert not ui.app.mouse_support()  # The next right-click belongs to the client menu.
    assert "再次右键" in ui.status
    assert ui.editor.text == ""  # Never silently paste the server's kill ring instead.


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
        ui._mouse_enabled = True
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


def test_long_reference_url_remains_complete_in_native_scrollback():
    from agent_runtime.terminal_app import _markdown

    url = "https://example.org/" + "long-path/" * 30 + "?record=id123"
    fragments = _markdown(f"Read [Source]({url}).", 40, False)
    visible = "".join(text for style, text, *_ in fragments if "[ZeroWidthEscape]" not in style)
    assert url in visible
    assert "\x1b" not in visible


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["inline", "fullscreen"])
async def test_request_settings_and_usage_are_available_in_details_without_transcript_notices(mode):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.result import AgentUsage
    from agent_runtime.streaming.events import EventType, StreamEvent
    from agent_runtime.terminal_app import TerminalChatApp

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="demo", workspace="/tmp", ui_mode=mode, input=pipe, output=DummyOutput())
        ui.display.begin_turn()
        await ui.display.emit(StreamEvent(type=EventType.BUDGET_UPDATE, turn_id="t", data={
            "purpose": "agent_step", "effective_settings": {"reasoning_effort": "high",
                "max_output_tokens": 4096, "requested_output_tokens": 4096},
        }))
        ui.display.update_usage(AgentUsage(input_tokens=100, output_tokens=20, total_tokens=120,
                                          cache_read_input_tokens=80))
        ui.display.finish()
        assert ui._status_fragments() == []
        ui.display.set_verbose(True)
        status = "".join(fragment[1] for fragment in ui._status_fragments())
        assert "demo" in status and "high" in status
        assert "本轮累计" in status and "缓存" in status and "80" in status
        assert status.count("\n") == 1
        assert "high" not in ui.view.plain_text()
        assert "本轮累计" not in ui.view.plain_text()
        assert "Enter" in "".join(fragment[1] for fragment in ui._footer())


def test_usage_update_invalidates_retained_status_and_compact_status_keeps_cache():
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output
    from wcwidth import wcswidth

    from agent_runtime.result import AgentUsage
    from agent_runtime.terminal_app import TerminalChatApp

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="demo", workspace="/tmp", input=pipe, output=Vt100_Output(
            io.StringIO(), lambda: Size(rows=24, columns=80), enable_cpr=False))
        invalidated = []
        ui.view.invalidate = lambda: invalidated.append(True)
        ui.display.update_usage(AgentUsage(input_tokens=4000, output_tokens=500, total_tokens=4500,
                                          cache_read_input_tokens=3000, reasoning_tokens=100))
        assert invalidated
        ui.display.set_verbose(True)
        lines = "".join(fragment[1] for fragment in ui._status_fragments()).splitlines()
        assert all(wcswidth(line) <= 80 for line in lines)
        assert "缓存" in lines[1] and "推理" in lines[1]


@pytest.mark.anyio
async def test_effort_menu_reuses_picker_and_preserves_status_on_selection_and_cancel():
    import asyncio
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    from agent_runtime.result import AgentUsage
    from agent_runtime.terminal_app import TerminalChatApp

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="demo", workspace="/tmp", ui_mode="fullscreen", input=pipe,
            output=Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=100), enable_cpr=False))
        ui.display.update_usage(AgentUsage(input_tokens=100, output_tokens=20, total_tokens=120))
        selected = []
        async def conversation():
            selected.append(await ui.choose_menu(["default", "medium", "high"],
                                                 title="选择思考强度", current="medium"))
            selected.append(await ui.choose_menu(["default", "high"], title="选择思考强度", current="high"))
        task = asyncio.create_task(ui.run(conversation))
        try:
            async with asyncio.timeout(3):
                while ui._model_pending is None or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(.01)
            assert ui._model_index == 1
            assert "选择思考强度" in "".join(fragment[1] for fragment in ui._footer())
            assert ui._model_choices == ["default", "medium", "high"]
            assert "off" not in "".join(fragment[1] for fragment in ui._model_fragments())
            pipe.send_text("\x1b[B\r")
            async with asyncio.timeout(3):
                while not selected or ui._model_choices != ["default", "high"]:
                    await asyncio.sleep(.01)
            pipe.send_text("\x1b")
            await asyncio.wait_for(task, 3)
            assert selected == ["high", None]
            assert ui._status_fragments() == []
            ui.display.set_verbose(True)
            assert "120" in "".join(fragment[1] for fragment in ui._status_fragments())
            assert "选择思考强度" not in ui.view.plain_text()
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


@pytest.mark.anyio
@pytest.mark.parametrize("columns", [60, 80])
async def test_large_token_status_keeps_model_effort_context_cache_and_reasoning_visible(columns):
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output
    from wcwidth import wcswidth

    from agent_runtime.result import AgentUsage
    from agent_runtime.streaming.events import EventType, StreamEvent
    from agent_runtime.terminal_app import TerminalChatApp

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="深度思考模型👩‍💻-deepseek-flash", workspace="/tmp", input=pipe,
            output=Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=columns), enable_cpr=False))
        await ui.display.emit(StreamEvent(type=EventType.BUDGET_UPDATE, turn_id="t", data={
            "purpose": "agent_step",
            "effective_settings": {"reasoning_effort": "high", "max_output_tokens": 65536,
                                   "requested_output_tokens": 65536},
            "context_projection": {"input_tokens": 18456, "max_input_tokens": 120000,
                                   "count_source": "utf8_bytes_conservative"},
        }))
        ui.display.update_usage(AgentUsage(input_tokens=2500000, output_tokens=1000000,
            total_tokens=3500000, cache_read_input_tokens=1800000, reasoning_tokens=500000))
        ui.display.set_verbose(True)
        lines = "".join(fragment[1] for fragment in ui._status_fragments()).splitlines()
        assert len(lines) == 2 and all(wcswidth(line) <= columns for line in lines)
        assert "深" in lines[0] and "high" in lines[0] and "上下文" in lines[0]
        assert "估" in lines[0]
        assert "入" in lines[1] and "出" in lines[1] and "缓存" in lines[1] and "推理" in lines[1]
        assert "3.5M" in lines[1] and "1.8M" in lines[1]
        assert "≈" in lines[1]  # Scaled status values are rounded, never exact accounting.


@pytest.mark.anyio
async def test_bracketed_file_drop_after_existing_instruction_stays_in_its_own_line(tmp_path):
    import asyncio
    import shlex

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.local_files import parse_file_input
    from agent_runtime.terminal_app import TerminalChatApp

    attachment = tmp_path / "季度 报告.txt"
    attachment.write_text("source")
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="demo", workspace=str(tmp_path), input=pipe, output=DummyOutput())
        submissions = []
        async def conversation():
            submissions.append(await ui.prompt())
        task = asyncio.create_task(ui.run(conversation))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting:
                    await asyncio.sleep(.01)
            pipe.send_text("分析这个")
            await asyncio.sleep(.03)
            pipe.send_text("\x1b[200~" + shlex.quote(str(attachment)) + "\x1b[201~")
            await asyncio.sleep(.08)
            assert not task.done() and ui.waiting
            guidance, files = parse_file_input(ui.editor.text, base=tmp_path)
            assert guidance == "分析这个" and files == [str(attachment)]
            pipe.send_text("继续解释")
            await asyncio.sleep(.03)
            guidance, files = parse_file_input(ui.editor.text, base=tmp_path)
            assert guidance == "分析这个\n继续解释" and files == [str(attachment)]
            pipe.send_text("\r")
            await asyncio.wait_for(task, 3)
            assert parse_file_input(submissions[0], base=tmp_path) == (guidance, files)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


@pytest.mark.anyio
async def test_busy_composer_delivers_request_settings_commands_and_keeps_other_slashes_in_draft():
    import asyncio

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="demo", workspace="/tmp", input=pipe, output=DummyOutput())
        finish = asyncio.Event()
        received = []
        async def on_input(text):
            received.append(text)
        ui.on_input = on_input
        async def conversation():
            await ui.operation(finish.wait())
        task = asyncio.create_task(ui.run(conversation))
        try:
            async with asyncio.timeout(3):
                while ui._operation is None:
                    await asyncio.sleep(.01)
            pipe.send_text("/model\r")
            await asyncio.sleep(.08)
            assert received == [] and ui.editor.text == "/model"
            ui.editor.text = ""
            pipe.send_text("/effort\r")
            async with asyncio.timeout(3):
                while received != ["/effort"]:
                    await asyncio.sleep(.01)
            pipe.send_text("/output 4096\r")
            async with asyncio.timeout(3):
                while len(received) != 2:
                    await asyncio.sleep(.01)
            assert received == ["/effort", "/output 4096"] and ui.editor.text == ""
            assert not ui._input_receipts
            finish.set()
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

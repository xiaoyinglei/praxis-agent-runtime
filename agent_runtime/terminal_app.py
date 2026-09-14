"""One prompt_toolkit application owns chat input, folding, and repainting."""

from __future__ import annotations

import asyncio
import io
import os
import sys
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import redirect_stdout, suppress
from functools import lru_cache
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import ConditionalKeyBindings, KeyBindings, KeyPressEvent, merge_key_bindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from rich.console import Console
from rich.markdown import Markdown

from agent_runtime.terminal_input import _COMPOSER_BINDINGS, BoundedPromptHistory
from agent_runtime.terminal_render import display_rows, safe_terminal_text
from agent_runtime.terminal_ui import Block, Conversation, ConversationEventDisplay


@lru_cache(maxsize=128)
def _markdown(text: str, width: int, color: bool) -> StyleAndTextTuples:
    stream = io.StringIO()
    console = Console(file=stream, width=width, force_terminal=True, no_color=not color)
    console.print(Markdown(text, code_theme="ansi_dark"), end="")
    return list(to_formatted_text(ANSI(stream.getvalue().rstrip("\n"))))


class _ConversationOutput(io.TextIOBase):
    def __init__(self, view: Conversation) -> None:
        self.view = view

    def write(self, value: str) -> int:
        self.view.message(value, "notice")
        return len(value)

    def flush(self) -> None:
        pass


class _ConversationControl(FormattedTextControl):
    def __init__(self, owner: TerminalChatApp) -> None:
        self.owner = owner
        super().__init__(owner.fragments, focusable=True, get_cursor_position=owner._cursor)

    def mouse_handler(self, mouse_event: MouseEvent) -> None:
        # Window already translates screen cells (including CJK) to text columns.
        # Avoid splitting the entire history into lines for every drag event.
        row = mouse_event.position.y
        self.owner._handler(self.owner._fold_rows.get(row), row)(mouse_event)


class TerminalChatApp:
    def __init__(self, *, model: str, workspace: str, input: Input | None = None, output: Output | None = None) -> None:
        self.model = model
        self.workspace = workspace
        self.status = "就绪"
        self.view = Conversation(self.invalidate)
        self.display = ConversationEventDisplay(self.view)
        self._pending: asyncio.Future[str] | None = None
        self._operation: asyncio.Task[Any] | None = None
        self._row = 0
        self._follow = True
        self._fold_rows: dict[int, Block] = {}
        self._line_count = 1
        self._layout_key: tuple[int, int] | None = None
        self._fragments: StyleAndTextTuples = []
        self._plain_lines: list[str] = []
        self._selection_start: Point | None = None
        self._selection_end: Point | None = None
        self._selecting = False
        self._highlight_key: object = None
        self._highlighted: StyleAndTextTuples = []
        self._mouse_enabled = True
        self._model_choices: list[str] = []
        self._model_index = 0
        self._model_pending: asyncio.Future[str | None] | None = None
        self.model_menu = FormattedTextControl(
            self._model_fragments, focusable=True,
            get_cursor_position=lambda: Point(0, self._model_index),
        )
        self.body = _ConversationControl(self)
        self.editor = TextArea(
            prompt="❯ ",
            multiline=True,
            wrap_lines=True,
            height=3,
            history=BoundedPromptHistory(),
            completer=WordCompleter(["/help", "/status", "/new", "/clear", "/model", "/verbose", "/exit"]),
            complete_while_typing=Condition(lambda: self.editor.text.startswith("/")),
        )
        keys = KeyBindings()

        @keys.add("up", filter=has_focus(self.model_menu))
        @keys.add("down", filter=has_focus(self.model_menu))
        def select_model(event: KeyPressEvent) -> None:
            direction = -1 if event.key_sequence[0].key == "up" else 1
            self._model_index = (self._model_index + direction) % len(self._model_choices)

        @keys.add("enter", filter=has_focus(self.model_menu))
        def confirm_model(event: KeyPressEvent) -> None:
            self._resolve_model(self._model_choices[self._model_index])

        @keys.add("escape", filter=has_focus(self.model_menu), eager=True)
        def close_model(event: KeyPressEvent) -> None:
            self._resolve_model(None)

        @keys.add("f2")
        def native_mouse(event: KeyPressEvent) -> None:
            self._mouse_enabled = not self._mouse_enabled

        @keys.add("enter", filter=has_focus(self.editor))
        def send(event: KeyPressEvent) -> None:
            if self.waiting and self.editor.text.strip():
                value = self.editor.text
                self.editor.buffer.append_to_history()
                self.editor.text = ""
                assert self._pending is not None
                self._pending.set_result(value)
                self._follow = True

        @keys.add("escape", "enter", filter=has_focus(self.editor))
        def newline(event: KeyPressEvent) -> None:
            self.editor.buffer.insert_text("\n")

        @keys.add("tab", filter=has_focus(self.editor))
        def focus_records(event: KeyPressEvent) -> None:
            if self.editor.text.startswith("/"):
                self.editor.buffer.complete_next()
                return
            event.app.layout.focus(self.body)
            self._follow = False
            self._row = max(self._fold_rows, default=0)

        @keys.add("tab", filter=has_focus(self.body))
        @keys.add("escape", filter=has_focus(self.body))
        def focus_input(event: KeyPressEvent) -> None:
            event.app.layout.focus(self.editor)

        @keys.add("up", filter=has_focus(self.body))
        def up(event: KeyPressEvent) -> None:
            self._move(-1)

        @keys.add("down", filter=has_focus(self.body))
        def down(event: KeyPressEvent) -> None:
            self._move(1)

        @keys.add("pageup")
        def page_up(event: KeyPressEvent) -> None:
            self._move(-max(3, self.app.output.get_size().rows // 2))

        @keys.add("pagedown")
        def page_down(event: KeyPressEvent) -> None:
            self._move(max(3, self.app.output.get_size().rows // 2))

        @keys.add("c-end")
        def latest(event: KeyPressEvent) -> None:
            self._follow = True

        @keys.add("enter", filter=has_focus(self.body))
        @keys.add(" ", filter=has_focus(self.body))
        def toggle_selected(event: KeyPressEvent) -> None:
            block = self._fold_rows.get(self._row)
            if block:
                self.view.toggle(block)

        @keys.add("c-o")
        def toggle_latest(event: KeyPressEvent) -> None:
            block = self._fold_rows.get(self._row) if event.app.layout.has_focus(self.body) else None
            if block is None:
                block = next((b for b in reversed(self.view.blocks) if b.kind == "group"), None)
            if block:
                self.view.toggle(block)

        @keys.add("c-c")
        def cancel(event: KeyPressEvent) -> None:
            if self._model_pending is not None:
                self._resolve_model(None)
            elif self._selected_text():
                self._copy_text(self._selected_text())
                self._selection_start = self._selection_end = None
            elif self._operation and not self._operation.done():
                self.status = "正在取消，等待执行清理"
                if not self._operation.cancelling():
                    self._operation.cancel()
            elif self.editor.text:
                self.editor.text = ""
            elif self.waiting:
                assert self._pending is not None
                self._pending.set_result("/exit")

        @keys.add("c-d", filter=has_focus(self.editor))
        def eof(event: KeyPressEvent) -> None:
            if self.waiting and not self.editor.text:
                assert self._pending is not None
                self._pending.set_result("/exit")

        body_window = Window(
            self.body,
            wrap_lines=False,
            height=lambda: Dimension(min=0, preferred=max(0, self.app.output.get_size().rows - 8)),
        )
        layout = HSplit(
            [
                Window(
                    FormattedTextControl(
                        lambda: [
                            ("class:header", f" Agent · {safe_terminal_text(self.model)}\n"),
                            ("class:dim", safe_terminal_text(self.workspace)),
                        ]
                    ),
                    height=2,
                ),
                body_window,
                ConditionalContainer(
                    Window(
                        self.model_menu,
                        height=lambda: Dimension(min=1, preferred=min(8, len(self._model_choices)), max=8),
                    ),
                    filter=Condition(lambda: self._model_pending is not None),
                ),
                Window(height=1, char="─", style="class:dim"),
                self.editor,
                Window(FormattedTextControl(self._footer), height=1),
            ]
        )
        self.app: Application[None] = Application(
            layout=Layout(layout, focused_element=self.editor),
            key_bindings=merge_key_bindings(
                [
                    ConditionalKeyBindings(_COMPOSER_BINDINGS, has_focus(self.editor)),
                    keys,
                ]
            ),
            mouse_support=Condition(lambda: self._mouse_enabled),
            full_screen=False,
            input=input,
            output=output,
            refresh_interval=0.2,
            min_redraw_interval=0.05,
            style=Style.from_dict(
                {
                    "header": "bold",
                    "user": "bold ansicyan",
                    "group": "ansicyan",
                    "tool": "",
                    "dim": "ansibrightblack",
                    "error": "ansired",
                    "added": "ansigreen",
                    "removed": "ansired",
                }
            )
            if "NO_COLOR" not in os.environ
            else Style.from_dict({}),
        )

    @property
    def waiting(self) -> bool:
        return self._pending is not None and not self._pending.done()

    def invalidate(self) -> None:
        if hasattr(self, "app"):
            self.app.invalidate()

    def _cursor(self) -> Point:
        return Point(0, max(0, self._line_count - 1) if self._follow else min(self._row, max(0, self._line_count - 1)))

    def _move(self, amount: int) -> None:
        self._row = max(0, min(self._cursor().y + amount, self._line_count - 1))
        self._follow = False
        self.invalidate()

    def _handler(self, block: Block | None, row: int) -> Callable[[MouseEvent], None]:
        def mouse(event: MouseEvent) -> None:
            if event.event_type == MouseEventType.SCROLL_UP:
                self._move(-3)
            elif event.event_type == MouseEventType.SCROLL_DOWN:
                self._move(3)
            elif event.event_type == MouseEventType.MOUSE_DOWN and event.button == MouseButton.LEFT:
                self._selection_start = self._selection_end = event.position
                self._selecting = True
                self._row = event.position.y
                self._follow = False
            elif event.event_type == MouseEventType.MOUSE_MOVE and self._selecting:
                self._selection_end = event.position
            elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                self._selecting = False
                self._selection_end = event.position
                selected = self._selected_text()
                if selected:
                    self._copy_text(selected)
                elif block:
                    self._row = row
                    self._follow = False
                    self.view.toggle(block)
            elif event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.RIGHT:
                self.app.create_background_task(self._paste_text())
            else:
                return None
            self.invalidate()
            return None

        return mouse

    def fragments(self) -> StyleAndTextTuples:
        width = max(4, self.app.output.get_size().columns - 1) if hasattr(self, "app") else 80
        key = (self.view.revision, width)
        if self._layout_key == key:
            return self._highlight_selection(self._fragments)
        if self._layout_key is not None and self._layout_key[1] != width:
            self._selection_start = self._selection_end = None
        fragments: StyleAndTextTuples = []
        line = 0
        self._fold_rows = {}
        for text, block, style in self.view.rows():
            if style == "answer" and text.strip():
                rendered = _markdown(text.removeprefix("● "), max(1, width - 2), "NO_COLOR" not in os.environ)
                handler = self._handler(None, line)
                fragments.append(("", "● ", handler))
                fragments.extend((s, t.replace("\n", "\n  "), handler) for s, t, *_ in rendered)
                fragments.append(("", "\n", handler))
                line += sum(t.count("\n") for _, t, *_ in rendered) + 1
                continue
            for value in text.split("\n"):
                for row in display_rows(value, width=width):
                    if block:
                        self._fold_rows[line] = block
                    row_style = style
                    if style == "detail":
                        if row.lstrip().startswith("+"):
                            row_style = "added"
                        elif row.lstrip().startswith("-"):
                            row_style = "removed"
                    fragments.append(
                        (
                            f"class:{row_style}" if row_style else "",
                            row + "\n",
                            self._handler(block, line),
                        )
                    )
                    line += 1
        self._line_count = max(1, line)
        self._layout_key = key
        self._fragments = fragments
        self._plain_lines = "".join(t for _, t, *_ in fragments).split("\n")
        return self._highlight_selection(fragments)

    def _selection_bounds(self) -> tuple[Point, Point] | None:
        if self._selection_start is None or self._selection_end is None:
            return None
        start, end = sorted((self._selection_start, self._selection_end), key=lambda p: (p.y, p.x))
        return (start, end) if start != end else None

    def _selected_text(self) -> str:
        bounds = self._selection_bounds()
        if bounds is None:
            return ""
        start, end = bounds
        lines = []
        for row in range(max(0, start.y), min(len(self._plain_lines), end.y + 1)):
            selected = []
            for x, char in enumerate(self._plain_lines[row]):
                if (row, x) >= (start.y, start.x) and (row, x) < (end.y, end.x):
                    selected.append(char)
            lines.append("".join(selected))
        return "\n".join(lines)

    def _highlight_selection(self, fragments: StyleAndTextTuples) -> StyleAndTextTuples:
        bounds = self._selection_bounds()
        focused_row = self._row if self.app.layout.has_focus(self.body) else None
        if bounds is None and focused_row is None:
            return fragments
        key = (self._layout_key, bounds, focused_row)
        if self._highlight_key == key:
            return self._highlighted
        start, end = bounds or (Point(0, -1), Point(0, -1))
        output: StyleAndTextTuples = []
        x = y = 0
        for fragment in fragments:
            style, text = fragment[0], fragment[1]
            handler = fragment[2] if len(fragment) == 3 else None
            # Keep whole unaffected spans, especially large historical answers.
            last_y = y + text.count("\n")
            if not (start.y <= last_y and end.y >= y) and not (focused_row is not None and y <= focused_row <= last_y):
                output.append((style, text, handler) if handler else (style, text))
                x = len(text.rsplit("\n", 1)[-1]) if last_y != y else x + len(text)
                y = last_y
                continue
            run_style, run_text = style, ""
            for char in text:
                selected = (start.y, start.x) <= (y, x) < (end.y, end.x) or y == focused_row
                next_style = style + (" reverse" if selected else "")
                if next_style != run_style and run_text:
                    output.append((run_style, run_text, handler) if handler else (run_style, run_text))
                    run_text = ""
                run_style = next_style
                run_text += char
                if char == "\n":
                    x, y = 0, y + 1
                else:
                    x += 1
            if run_text:
                output.append((run_style, run_text, handler) if handler else (run_style, run_text))
        self._highlight_key, self._highlighted = key, output
        return output

    async def _paste_text(self) -> None:
        text = self.app.clipboard.get_data().text
        if sys.platform == "darwin":
            try:
                process = await asyncio.create_subprocess_exec("pbpaste", stdout=asyncio.subprocess.PIPE)
                data, _ = await process.communicate()
                if process.returncode == 0:
                    text = data.decode("utf-8", errors="replace")
            except OSError:
                pass
        self.app.layout.focus(self.editor)
        self.editor.buffer.insert_text(text)
        self.invalidate()

    def _copy_text(self, text: str) -> None:
        self.app.clipboard.set_data(ClipboardData(text))
        async def copy() -> None:
            if sys.platform == "darwin":
                try:
                    process = await asyncio.create_subprocess_exec("pbcopy", stdin=asyncio.subprocess.PIPE)
                    await process.communicate(text.encode("utf-8"))
                    self.status = "已复制" if process.returncode == 0 else "系统复制失败，可按 F2 使用终端原生选择"
                except OSError:
                    self.status = "系统复制失败，可按 F2 使用终端原生选择"
            else:
                self.status = "已选择；F2 切换终端原生复制"
            self.invalidate()
        self.app.create_background_task(copy())

    def _resolve_model(self, choice: str | None) -> None:
        if self._model_pending is not None and not self._model_pending.done():
            self._model_pending.set_result(choice)

    def _model_fragments(self) -> StyleAndTextTuples:
        result: StyleAndTextTuples = []
        for index, choice in enumerate(self._model_choices):
            def click(event: MouseEvent, selected: str = choice) -> None:
                if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                    self._resolve_model(selected)
            result.append((
                "reverse" if index == self._model_index else "",
                f" {'❯' if index == self._model_index else ' '} {safe_terminal_text(choice)}\n", click,
            ))
        return result

    async def choose_model(self, choices: list[str], current: str) -> str | None:
        if not choices:
            return None
        self._model_choices = choices
        self._model_index = choices.index(current) if current in choices else 0
        self._model_pending = asyncio.get_running_loop().create_future()
        self.app.layout.focus(self.model_menu)
        self.invalidate()
        try:
            return await self._model_pending
        finally:
            self._model_pending = None
            self.app.layout.focus(self.editor)
            self.invalidate()

    def _footer(self) -> StyleAndTextTuples:
        if self._model_pending is not None:
            return [("class:dim", " 选择模型 · ↑↓ 移动 · Enter 确认 · Esc 取消 · 点击选择")]
        activity = self.display.activity_text() if self._operation and not self._operation.cancelling() else self.status
        mouse_hint = "拖选复制 · 右键粘贴 · F2 原生选择" if self._mouse_enabled else "原生鼠标选择 · F2 恢复折叠点击"
        return [
            (
                "class:dim",
                f" {activity} · {mouse_hint} · Ctrl+O 折叠 · Enter 发送 · Ctrl+C 取消",
            )
        ]

    async def prompt(self, *, approval: bool = False) -> str:
        self.status = "等待确认" if approval else "就绪"
        self._pending = asyncio.get_running_loop().create_future()
        self.app.layout.focus(self.editor)
        self.invalidate()
        try:
            value = await self._pending
            if not approval:
                self.view.message(value + "\n", "user")
            return value
        finally:
            self._pending = None

    async def operation(self, awaitable: Coroutine[Any, Any, Any]) -> Any:
        self._operation = asyncio.create_task(awaitable)
        self.status = "工作中"
        try:
            return await self._operation
        finally:
            self._operation = None
            self.status = "就绪"
            self.invalidate()

    async def run(self, conversation: Callable[[], Awaitable[None]]) -> None:
        worker: asyncio.Task[None] | None = None

        async def work() -> None:
            try:
                with redirect_stdout(_ConversationOutput(self.view)):
                    await conversation()
            except Exception as exc:
                self.app.exit(exception=exc)
            else:
                self.app.exit()

        def start() -> None:
            nonlocal worker
            worker = asyncio.create_task(work())

        try:
            await self.app.run_async(pre_run=start)
        finally:
            if worker:
                if not worker.done():
                    worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker

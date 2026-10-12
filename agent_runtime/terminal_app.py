"""One prompt_toolkit application owns chat input, folding, and repainting."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any

from prompt_toolkit.application import Application, run_in_terminal
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
from prompt_toolkit.renderer import print_formatted_text
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

from agent_runtime.result import AgentPause
from agent_runtime.terminal_input import _COMPOSER_BINDINGS, BoundedPromptHistory, resolve_pause_choice
from agent_runtime.terminal_render import display_rows, safe_terminal_text
from agent_runtime.terminal_ui import Block, Conversation, ConversationEventDisplay


@lru_cache(maxsize=128)
def _markdown(text: str, width: int, color: bool) -> StyleAndTextTuples:
    stream = io.StringIO()
    console = Console(
        file=stream, width=width, force_terminal=True, no_color=not color,
        theme=Theme({"markdown.code": "bold", "markdown.code_block": "none",
                     "markdown.link": "underline", "markdown.link_url": "none"}),
    )
    console.print(Markdown(text, code_theme="ansi_light"), end="")
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


@dataclass
class _ApprovalState:
    request_id: str
    options: tuple[str, ...]
    future: asyncio.Future[str | None]
    selected: int | None
    pressed: tuple[int, Point, int] | None = None


def _approval_label(choice: str | None) -> str:
    return {
        "approve": "允许一次 · Y", "allow_once": "允许一次 · Y",
        "deny": "拒绝 · N", None: "取消 · Esc",
    }.get(choice, safe_terminal_text(str(choice)))


class TerminalChatApp:
    def __init__(self, *, model: str, workspace: str, input: Input | None = None, output: Output | None = None) -> None:
        self.model = model
        self.workspace = workspace
        self.status = "就绪"
        self.view = Conversation(self.invalidate)
        self.display = ConversationEventDisplay(self.view)
        self._pending: asyncio.Future[str] | None = None
        self._approval: _ApprovalState | None = None
        self._input_sequence = 0
        self._operation: asyncio.Task[Any] | None = None
        self._row = 0
        self._follow = True
        self._show_history = False
        self._fold_rows: dict[int, Block] = {}
        self._line_count = 1
        self._layout_key: tuple[int, int, bool] | None = None
        self._fragments: StyleAndTextTuples = []
        self._plain_lines: list[str] = []
        self._selection_start: Point | None = None
        self._selection_end: Point | None = None
        self._selecting = False
        self._highlight_key: object = None
        self._highlighted: StyleAndTextTuples = []
        # The client terminal owns selection and its clipboard, including over SSH.
        self._mouse_enabled = False
        self._model_choices: list[str] = []
        self._model_index = 0
        self._model_pending: asyncio.Future[str | None] | None = None
        self._menu_title = "选择模型"
        self.model_menu = FormattedTextControl(
            self._model_fragments, focusable=True,
            get_cursor_position=lambda: Point(0, self._model_index),
        )
        self.body = _ConversationControl(self)
        self.editor = TextArea(
            prompt="❯ ",
            multiline=True,
            wrap_lines=True,
            height=lambda: Dimension(min=3, preferred=min(8, self.editor.document.line_count + 1), max=8),
            style="class:input",
            history=BoundedPromptHistory(),
            completer=WordCompleter(["/help", "/status", "/new", "/clear", "/model", "/verbose",
                                     "/copy", "/resume", "/exit"]),
            complete_while_typing=Condition(lambda: self.editor.text.startswith("/")),
        )
        self.approval_body = TextArea(
            read_only=True, focusable=True, wrap_lines=True, scrollbar=True,
            height=lambda: Dimension(min=1, preferred=min(6, max(1, self.app.output.get_size().rows - 8))),
            style="class:approval-body",
        )
        self.approval_buttons = FormattedTextControl(self._approval_fragments, focusable=True)
        approving = Condition(lambda: self._approval is not None)
        keys = KeyBindings()

        @keys.add("y", filter=approving, eager=True)
        @keys.add("Y", filter=approving, eager=True)
        @keys.add("n", filter=approving, eager=True)
        @keys.add("N", filter=approving, eager=True)
        def approval_shortcut(event: KeyPressEvent) -> None:
            state = self._approval
            if state is not None:
                choice = resolve_pause_choice(event.data, state.options)
                if choice is not None:
                    self._resolve_approval(state, choice)

        @keys.add("escape", filter=approving, eager=True)
        def cancel_approval(event: KeyPressEvent) -> None:
            if self._approval is not None:
                self._resolve_approval(self._approval, None)

        @keys.add("left", filter=has_focus(self.approval_buttons))
        @keys.add("up", filter=has_focus(self.approval_buttons))
        @keys.add("right", filter=has_focus(self.approval_buttons))
        @keys.add("down", filter=has_focus(self.approval_buttons))
        def select_approval(event: KeyPressEvent) -> None:
            state = self._approval
            if state is not None:
                direction = -1 if event.key_sequence[0].key in {"left", "up"} else 1
                state.selected = ((state.selected + direction) % (len(state.options) + 1)
                                  if state.selected is not None else 0)
                state.pressed = None

        @keys.add("enter", filter=has_focus(self.approval_buttons))
        def confirm_approval(event: KeyPressEvent) -> None:
            state = self._approval
            if state is not None and state.selected is not None:
                self._resolve_approval(state, (*state.options, None)[state.selected])

        @keys.add("tab", filter=approving)
        @keys.add("s-tab", filter=approving)
        def approval_focus(event: KeyPressEvent) -> None:
            event.app.layout.focus(self.approval_body if event.app.layout.has_focus(self.approval_buttons)
                                   else self.approval_buttons)

        @keys.add("pageup", filter=has_focus(self.approval_body))
        @keys.add("pagedown", filter=has_focus(self.approval_body))
        def approval_page(event: KeyPressEvent) -> None:
            info = self.approval_body.window.render_info
            count = max(1, info.window_height - 1) if info is not None else 1
            if event.key_sequence[0].key == "pageup":
                self.approval_body.buffer.cursor_up(count=count)
            else:
                self.approval_body.buffer.cursor_down(count=count)

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
            self._show_history = self._mouse_enabled
            self._follow = True
            self._selection_start = self._selection_end = None
            self.status = "鼠标交互已开启" if self._mouse_enabled else "终端原生选择已开启"

        @keys.add("enter", filter=has_focus(self.editor))
        def send(event: KeyPressEvent) -> None:
            if self.waiting and self.editor.text.strip():
                value = self.editor.text
                self.editor.buffer.append_to_history()
                self.editor.text = ""
                assert self._pending is not None
                self._pending.set_result(value)
                self._show_history = False
                self._follow = True

        @keys.add("escape", "enter", filter=has_focus(self.editor))
        @keys.add("c-j", filter=has_focus(self.editor))
        def newline(event: KeyPressEvent) -> None:
            self.editor.buffer.insert_text("\n")

        @keys.add("tab", filter=has_focus(self.editor))
        def focus_records(event: KeyPressEvent) -> None:
            if self.editor.text.startswith("/"):
                self.editor.buffer.complete_next()
                return
            event.app.layout.focus(self.body)
            self._show_history = True
            self.fragments()
            self._follow = False
            self._row = max(self._fold_rows, default=0)

        @keys.add("tab", filter=has_focus(self.body))
        @keys.add("escape", filter=has_focus(self.body))
        def focus_input(event: KeyPressEvent) -> None:
            event.app.layout.focus(self.editor)
            self._show_history = False
            self._follow = True

        @keys.add("up", filter=has_focus(self.body))
        def up(event: KeyPressEvent) -> None:
            self._move(-1)

        @keys.add("down", filter=has_focus(self.body))
        def down(event: KeyPressEvent) -> None:
            self._move(1)

        @keys.add("pageup", filter=~approving)
        def page_up(event: KeyPressEvent) -> None:
            self._move(-max(3, self.app.output.get_size().rows // 2))

        @keys.add("pagedown", filter=~approving)
        def page_down(event: KeyPressEvent) -> None:
            self._move(max(3, self.app.output.get_size().rows // 2))

        @keys.add("c-end")
        def latest(event: KeyPressEvent) -> None:
            self._show_history = False
            self._follow = True

        @keys.add("enter", filter=has_focus(self.body))
        @keys.add(" ", filter=has_focus(self.body))
        def toggle_selected(event: KeyPressEvent) -> None:
            block = self._fold_rows.get(self._row)
            if block:
                self.view.toggle(block)
                self._show_history = True

        @keys.add("c-o")
        def toggle_latest(event: KeyPressEvent) -> None:
            block = self._fold_rows.get(self._row) if event.app.layout.has_focus(self.body) else None
            if block is None:
                block = next((b for b in reversed(self.view.blocks) if b.kind == "group"), None)
            if block:
                self.view.toggle(block)
                self._show_history = True

        @keys.add("c-c")
        def cancel(event: KeyPressEvent) -> None:
            if self._approval is not None:
                self._resolve_approval(self._approval, None)
            elif self._model_pending is not None:
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
            height=lambda: Dimension(min=0, max=max(0, self.app.output.get_size().rows - 8)),
            dont_extend_height=True,
        )
        layout = HSplit(
            [
                Window(
                    FormattedTextControl(self._header),
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
                ConditionalContainer(
                    HSplit([
                        self.approval_body,
                        Window(self.approval_buttons, height=lambda: len(self._approval.options) + 1
                               if self._approval is not None else 1),
                    ]), filter=approving,
                ),
                ConditionalContainer(self.editor, filter=~approving),
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
            # Repainting an idle prompt can clear the terminal's native selection.
            refresh_interval=None,
            min_redraw_interval=0.05,
            style=Style.from_dict(
                {
                    "header": "bold ansiblue",
                    "assistant": "bold",
                    "user": "bold ansiblue",
                    "group": "bold",
                    "tool": "",
                    "dim": "ansibrightblack",
                    "error": "ansired",
                    "added": "ansigreen",
                    "removed": "ansired",
                    "approval-body": "",
                    "approval-allow": "ansigreen bold",
                    "approval-deny": "ansiyellow bold",
                    "approval-cancel": "ansibrightblack",
                    "approval-selected": "reverse bold",
                    "input": "",
                }
            )
            if "NO_COLOR" not in os.environ
            else Style.from_dict({}),
        )
        self.app.key_processor.before_key_press += self._advance_input

    def _advance_input(self, _processor: object) -> None:
        # Include movement handled by other controls without parsing terminal packets again.
        self._input_sequence += 1

    @property
    def waiting(self) -> bool:
        return (self._pending is not None and not self._pending.done()
                or self._approval is not None and not self._approval.future.done())

    def invalidate(self) -> None:
        if hasattr(self, "app"):
            self.app.invalidate()

    def _cursor(self) -> Point:
        return Point(0, max(0, self._line_count - 1) if self._follow else min(self._row, max(0, self._line_count - 1)))

    def _move(self, amount: int) -> None:
        if not self._show_history and any(block.in_scrollback for block in self.view.blocks):
            self._show_history = True
            self.fragments()
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
        history = self._show_history or self._mouse_enabled
        key = (self.view.revision, width, history)
        if self._layout_key == key:
            return self._highlight_selection(self._fragments)
        if self._layout_key is not None and self._layout_key[1] != width:
            self._selection_start = self._selection_end = None
        fragments: StyleAndTextTuples = []
        line = 0
        self._fold_rows = {}
        for text, block, style in self.view.rows(include_scrollback=history):
            if style == "answer" and text.strip():
                rendered = _markdown(text.removeprefix("● "), max(1, width - 2), "NO_COLOR" not in os.environ)
                handler = self._handler(None, line)
                fragments.append(("class:assistant", "Praxis\n", handler))
                fragments.append(("", "  ", handler))
                fragments.extend((s, t.replace("\n", "\n  "), handler) for s, t, *_ in rendered)
                fragments.append(("", "\n", handler))
                line += sum(t.count("\n") for _, t, *_ in rendered) + 2
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
        # A server's clipboard (or our kill ring) is not the SSH client's clipboard.
        if sys.platform != "darwin" or os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
            self.status = "请用终端粘贴快捷键；F2 可恢复原生右键菜单"
            self.invalidate()
            return
        try:
            process = await asyncio.create_subprocess_exec("pbpaste", stdout=asyncio.subprocess.PIPE)
            try:
                data, _ = await asyncio.wait_for(process.communicate(), 3)
            except (TimeoutError, asyncio.CancelledError) as exc:
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                await process.communicate()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                self.status = "剪贴板读取超时，请用终端粘贴快捷键"
                return
            if process.returncode != 0:
                self.status = "请用终端粘贴快捷键"
                return
            text = data.decode("utf-8", errors="replace")
        except OSError:
            self.status = "请用终端粘贴快捷键"
            return
        finally:
            self.invalidate()
        self.app.layout.focus(self.editor)
        self.editor.buffer.insert_text(text)
        self.invalidate()

    def _copy_text(self, text: str) -> None:
        self.app.clipboard.set_data(ClipboardData(text))
        self.app.create_background_task(self._write_clipboard(text))

    async def _write_clipboard(self, text: str) -> None:
        if sys.platform == "darwin" and not (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")):
            try:
                process = await asyncio.create_subprocess_exec("pbcopy", stdin=asyncio.subprocess.PIPE)
                try:
                    await asyncio.wait_for(process.communicate(text.encode("utf-8")), 3)
                except (TimeoutError, asyncio.CancelledError) as exc:
                    if process.returncode is None:
                        with suppress(ProcessLookupError):
                            process.kill()
                    await process.communicate()
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    self.status = "复制超时，请用终端原生选择"
                    self.invalidate()
                    return
                self.status = "已复制" if process.returncode == 0 else "复制失败，请用终端原生选择"
            except OSError:
                self.status = "复制失败，请用终端原生选择"
        elif len(text.encode("utf-8")) <= 64 * 1024:
            # OSC 52 targets the client terminal without reading its clipboard.
            # Terminals may reject it, so never claim an acknowledged copy.
            payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
            def request() -> None:
                self.app.output.write_raw(f"\x1b]52;c;{payload}\x07")
                self.app.output.flush()
            await run_in_terminal(request)
            self.status = "已请求终端复制；未生效时请原生选取"
        else:
            self.status = "回答较长，请从终端滚动历史中原生选取"
        self.invalidate()

    async def show_text(self, text: str) -> None:
        """Write above the UI into native scrollback, bypassing redirected stdout."""
        await run_in_terminal(lambda: print_formatted_text(
            self.app.output, [("", safe_terminal_text(text) + "\n")], self.app.style or Style.from_dict({}),
        ))

    async def replay_messages(self, messages: Sequence[tuple[str, str]]) -> None:
        """Publish saved messages in native scrollback; retain a bounded inspection view."""
        await self.flush_scrollback()
        for kind, text in messages:
            width = max(4, self.app.output.get_size().columns - 1)
            fragments: StyleAndTextTuples = (
                [("class:user", "你 › " + safe_terminal_text(text) + "\n\n")]
                if kind == "user" else
                [("class:assistant", "Praxis\n"),
                 *_markdown(safe_terminal_text(text), width, "NO_COLOR" not in os.environ), ("", "\n\n")]
            )
            await run_in_terminal(partial(
                print_formatted_text, self.app.output, fragments, self.app.style or Style.from_dict({}),
            ))
            self.view.message(text + "\n", kind)
            self.view.blocks[-1].in_scrollback = True
        self._show_history = False
        self._follow = True
        self.view.invalidate()

    async def flush_scrollback(self) -> None:
        """Commit stable output once; keep the viewport for streaming and inspection."""
        pending = [block for block in self.view.blocks if not block.in_scrollback]
        if not pending:
            return
        width = max(4, self.app.output.get_size().columns - 1)
        fragments: StyleAndTextTuples = []
        for text, _, style in self.view.rows(include_scrollback=False):
            if style == "answer" and text.strip():
                fragments.append(("class:assistant", "Praxis\n"))
                fragments.extend(_markdown(text.removeprefix("● "), width, "NO_COLOR" not in os.environ))
                fragments.append(("", "\n"))
            else:
                fragments.append((f"class:{style}" if style else "", text + "\n"))
        for block in pending:
            block.in_scrollback = True
        self._show_history = False
        self._follow = True
        self.view.invalidate()
        try:
            # The resumed UI must already be compact, otherwise its repaint would
            # push the just-printed answer off the visible terminal again.
            await run_in_terminal(lambda: print_formatted_text(
                self.app.output, fragments, self.app.style or Style.from_dict({}),
            ))
        except BaseException:
            for block in pending:
                block.in_scrollback = False
            self.view.invalidate()
            raise

    async def copy_answer(self, text: str) -> None:
        # Native scrollback is available even when the client rejects OSC 52.
        await self.show_text("最近回答（原始 Markdown，可在终端滚动历史中选取）\n\n" + text + "\n")
        await self._write_clipboard(text)
        self.view.message(self.status + "\n", "notice")

    def _resolve_model(self, choice: str | None) -> None:
        if self._model_pending is not None and not self._model_pending.done():
            self._model_pending.set_result(choice)

    def _header(self) -> StyleAndTextTuples:
        width = max(1, self.app.output.get_size().columns - 2)
        title = display_rows(f" PRAXIS  ·  {safe_terminal_text(self.model)}  ·  {self.status}", width=width)[0]
        workspace = os.path.basename(self.workspace.rstrip("/")) or self.workspace
        return [("class:header", title + "\n"),
                ("class:dim", f" 工作区 {safe_terminal_text(workspace)} · /status 查看完整路径")]

    def _resolve_approval(self, state: _ApprovalState, choice: str | None) -> None:
        if self._approval is state and not state.future.done() and (choice is None or choice in state.options):
            state.pressed = None
            state.future.set_result(choice)

    def _approval_fragments(self) -> StyleAndTextTuples:
        state = self._approval
        if state is None:
            return []
        fragments: StyleAndTextTuples = []
        for index, choice in enumerate((*state.options, None)):
            def click(event: MouseEvent, selected: str | None = choice, button: int = index) -> None:
                if self._approval is not state or state.future.done():
                    return
                if event.event_type == MouseEventType.MOUSE_MOVE:
                    state.pressed = None
                elif event.button == MouseButton.LEFT:
                    if event.event_type == MouseEventType.MOUSE_DOWN:
                        state.pressed = (button, event.position, self._input_sequence)
                    elif event.event_type == MouseEventType.MOUSE_UP:
                        pressed = state.pressed
                        state.pressed = None
                        if (pressed is not None and pressed[:2] == (button, event.position)
                                and pressed[2] + 1 == self._input_sequence):
                            self._resolve_approval(state, selected)

            kind = "allow" if choice in {"approve", "allow_once"} else "deny" if choice == "deny" else "cancel"
            style = f"class:approval-{kind}" + (" class:approval-selected" if state.selected == index else "")
            prefix = "❯" if state.selected == index else " "
            fragments.append((style, f" {prefix} [ {_approval_label(choice)} ]\n", click))
        return fragments

    async def choose_approval(self, request: AgentPause) -> str | None:
        if request.kind != "tool_approval":
            raise ValueError("approval panel requires a tool_approval request")
        if self._approval is not None:
            raise RuntimeError("another approval is already pending")
        if not request.options:
            self.view.message("审批请求没有提供可选操作，已取消。\n", "error")
            return None
        selected = request.options.index("deny") if "deny" in request.options else None
        state = _ApprovalState(
            request.request_id, request.options, asyncio.get_running_loop().create_future(), selected,
        )
        details = ["需要你确认", "", request.question]
        for call in request.tool_calls:
            details.extend(["", call.tool_name, "参数预览：" + call.args_preview])
            if call.reason and call.reason != request.question:
                details.append("原因：" + call.reason)
        detail = safe_terminal_text("\n".join(details))
        history = self.view.add_group()
        history.title = "人工审批 · 等待确认"
        history.expanded = False
        self.view.add_tool(history, request.question).append(detail)
        self.approval_body.text = detail
        self._approval = state
        self.status = "等待确认"
        self.app.layout.focus(self.approval_buttons)
        self.invalidate()
        try:
            choice = await state.future
            history.title = "人工审批 · " + ("已取消" if choice is None else _approval_label(choice).split(" · ")[0])
            return choice
        finally:
            state.pressed = None
            if self._approval is state:
                self._approval = None
                self.approval_body.text = ""
                self.status = "就绪"
                self.app.layout.focus(self.editor)
                self.invalidate()

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

    async def choose_menu(self, choices: list[str], *, title: str, current: str = "") -> str | None:
        if not choices:
            return None
        self._model_choices = choices
        self._menu_title = title
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

    async def choose_model(self, choices: list[str], current: str) -> str | None:
        return await self.choose_menu(choices, title="选择模型", current=current)

    def _footer(self) -> StyleAndTextTuples:
        if self._approval is not None:
            hints = []
            if resolve_pause_choice("y", self._approval.options) is not None:
                hints.append("Y允许")
            if "deny" in self._approval.options:
                hints.append("N拒绝")
            return [("class:dim", " " + " · ".join([*hints, "Esc取消", "Tab详情"]))]
        if self._model_pending is not None:
            return [("class:dim", f" {self._menu_title} · ↑↓ 移动 · Enter 确认 · Esc 取消")]
        activity = self.display.activity_text() if self._operation and not self._operation.cancelling() else self.status
        mouse_hint = "F2 原生选择" if self._mouse_enabled else "拖选复制"
        return [
            (
                "class:dim",
                f" {activity} · Enter 发送 · Ctrl+J 换行 · {mouse_hint}",
            )
        ]

    async def prompt(self, *, approval: bool = False) -> str:
        if not approval:
            await self.flush_scrollback()
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
        async def refresh() -> None:
            while True:
                self.invalidate()
                await asyncio.sleep(0.2)
        refresh_task = asyncio.create_task(refresh())
        try:
            return await self._operation
        finally:
            refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await refresh_task
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

from __future__ import annotations

import builtins
import sys
from collections.abc import Iterable
from typing import Protocol

import regex
from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

DEFAULT_HISTORY_ENTRIES = 100
DEFAULT_HISTORY_BYTES = 64 * 1024
_GRAPHEME = regex.compile(r"\X")


_COMPOSER_BINDINGS = KeyBindings()


@_COMPOSER_BINDINGS.add("backspace", eager=True)
def _delete_previous_grapheme(event: KeyPressEvent) -> None:
    buffer = event.current_buffer
    if buffer.selection_state is not None:
        data = buffer.cut_selection()
        event.app.clipboard.set_data(data)
        return
    cursor = buffer.cursor_position
    if cursor == 0:
        return
    for match in _GRAPHEME.finditer(buffer.text):
        if match.start() < cursor <= match.end():
            buffer.document = Document(
                buffer.text[: match.start()] + buffer.text[match.end() :],
                cursor_position=match.start(),
            )
            return


@_COMPOSER_BINDINGS.add("delete", eager=True)
def _delete_next_grapheme(event: KeyPressEvent) -> None:
    buffer = event.current_buffer
    if buffer.selection_state is not None:
        data = buffer.cut_selection()
        event.app.clipboard.set_data(data)
        return
    cursor = buffer.cursor_position
    for match in _GRAPHEME.finditer(buffer.text):
        if match.start() <= cursor < match.end():
            buffer.document = Document(
                buffer.text[: match.start()] + buffer.text[match.end() :],
                cursor_position=match.start(),
            )
            return


@_COMPOSER_BINDINGS.add("left", eager=True)
def _move_left_one_grapheme(event: KeyPressEvent) -> None:
    buffer = event.current_buffer
    cursor = buffer.cursor_position
    for match in _GRAPHEME.finditer(buffer.text):
        if match.start() < cursor <= match.end():
            buffer.cursor_position = match.start()
            if buffer.selection_state is not None:
                buffer.exit_selection()
            return


@_COMPOSER_BINDINGS.add("right", eager=True)
def _move_right_one_grapheme(event: KeyPressEvent) -> None:
    buffer = event.current_buffer
    cursor = buffer.cursor_position
    for match in _GRAPHEME.finditer(buffer.text):
        if match.start() <= cursor < match.end():
            buffer.cursor_position = match.end()
            if buffer.selection_state is not None:
                buffer.exit_selection()
            return


class _PromptSession(Protocol):
    def prompt(self, prompt: str) -> str: ...


class _BuiltinInputSession:
    def prompt(self, prompt: str) -> str:
        return builtins.input(prompt)


class BoundedPromptHistory(History):
    """Process-local prompt history with deterministic memory bounds."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_HISTORY_ENTRIES,
        max_bytes: int = DEFAULT_HISTORY_BYTES,
    ) -> None:
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("history limits must be positive")
        super().__init__()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._storage: list[str] = []
        self._storage_bytes = 0

    def load_history_strings(self) -> Iterable[str]:
        yield from reversed(self._storage)

    def store_string(self, string: str) -> None:
        # append_string owns storage so it can update both prompt_toolkit's
        # loaded view and the byte budget atomically.
        del string

    def append_string(self, string: str) -> None:
        if not string.strip():
            return
        encoded_bytes = len(string.encode("utf-8"))
        if encoded_bytes > self._max_bytes:
            return
        self._storage.append(string)
        self._storage_bytes += encoded_bytes
        self._loaded_strings.insert(0, string)
        self._loaded = True
        while (
            len(self._storage) > self._max_entries
            or self._storage_bytes > self._max_bytes
        ):
            removed = self._storage.pop(0)
            self._storage_bytes -= len(removed.encode("utf-8"))
            self._loaded_strings.pop()


class TerminalComposer:
    """Unicode-aware line editor used by the interactive chat loop."""

    def __init__(
        self,
        *,
        session: _PromptSession | None = None,
        history: BoundedPromptHistory | None = None,
    ) -> None:
        self.history = history or BoundedPromptHistory()
        if session is not None:
            self._session = session
        elif sys.stdin.isatty() and sys.stdout.isatty():
            self._session = PromptSession(
                history=self.history,
                enable_history_search=True,
                key_bindings=_COMPOSER_BINDINGS,
            )
        else:
            self._session = _BuiltinInputSession()

    def prompt(self, prompt: str = "> ") -> str:
        return self._session.prompt(prompt)

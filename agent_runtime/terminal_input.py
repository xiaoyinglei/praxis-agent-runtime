from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from prompt_toolkit import PromptSession
from prompt_toolkit.history import History

DEFAULT_HISTORY_ENTRIES = 100
DEFAULT_HISTORY_BYTES = 64 * 1024


class _PromptSession(Protocol):
    def prompt(self, prompt: str) -> str: ...


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
        self._session: _PromptSession = session or PromptSession(
            history=self.history,
            enable_history_search=True,
        )

    def prompt(self, prompt: str = "> ") -> str:
        return self._session.prompt(prompt)

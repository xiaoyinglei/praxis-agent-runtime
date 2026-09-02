from __future__ import annotations

import os
import pty
import select
import signal
import sys
import time
from pathlib import Path

import pytest

from agent_runtime.terminal_input import BoundedPromptHistory, TerminalComposer


def _history_entries(history: BoundedPromptHistory) -> list[str]:
    return list(history.get_strings())


def test_history_evicts_oldest_by_entry_limit() -> None:
    history = BoundedPromptHistory(max_entries=3, max_bytes=1_000)

    for value in ("one", "two", "three", "four"):
        history.append_string(value)

    assert _history_entries(history) == ["two", "three", "four"]


def test_history_evicts_oldest_by_utf8_byte_limit() -> None:
    history = BoundedPromptHistory(max_entries=100, max_bytes=7)

    history.append_string("a")
    history.append_string("你好")
    history.append_string("b")

    assert _history_entries(history) == ["你好", "b"]


def test_empty_submission_is_not_recorded(tmp_path: Path) -> None:
    history = BoundedPromptHistory()

    history.append_string("")
    history.append_string("   ")
    history.append_string("kept")

    assert _history_entries(history) == ["kept"]
    assert list(tmp_path.iterdir()) == []


class _FakeSession:
    def __init__(self, values: list[object]) -> None:
        self._values = iter(values)
        self.prompts: list[str] = []

    def prompt(self, prompt: str) -> str:
        self.prompts.append(prompt)
        value = next(self._values)
        if isinstance(value, BaseException):
            raise value
        assert isinstance(value, str)
        return value


def test_composer_delegates_to_one_session() -> None:
    session = _FakeSession(["first", "second"])
    composer = TerminalComposer(session=session)

    assert composer.prompt("> ") == "first"
    assert composer.prompt("> ") == "second"
    assert session.prompts == ["> ", "> "]


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_composer_propagates_terminal_exit(error: BaseException) -> None:
    composer = TerminalComposer(session=_FakeSession([error]))

    with pytest.raises(type(error)):
        composer.prompt("> ")


def _pty_exchange(
    program: str,
    *,
    typed_text: str = "你好",
    backspaces: int = 2,
    edit_keys: bytes | None = None,
    wait_for_raw_mode: bool = False,
) -> str:
    env = os.environ.copy()
    env["PROMPT_TOOLKIT_NO_CPR"] = "1"

    pid, master = pty.fork()

    if pid == 0:
        os.execvpe(
            sys.executable,
            [sys.executable, "-c", program],
            env,
        )
        raise AssertionError("execvpe unexpectedly returned")

    os.set_blocking(master, False)

    output = bytearray()
    deadline = time.monotonic() + 5.0
    child_reaped = False

    def read_chunk() -> bytes | None:
        try:
            return os.read(master, 4_096)
        except (BlockingIOError, OSError):
            return None

    try:
        # Wait until the child has actually entered its interactive
        # terminal mode before injecting keystrokes.
        while time.monotonic() < deadline:
            readable, _, _ = select.select(
                [master],
                [],
                [],
                0.1,
            )

            if readable:
                chunk = read_chunk()
                if chunk:
                    output.extend(chunk)

            if wait_for_raw_mode:
                if (
                    b"\x1b[?2004h" in output
                    and b">" in output
                ):
                    break
            elif b"> " in output:
                break

        os.write(
            master,
            typed_text.encode("utf-8"),
        )

        os.write(
            master,
            edit_keys
            if edit_keys is not None
            else (b"\x7f" * backspaces) + b"\r",
        )

        while time.monotonic() < deadline:
            readable, _, _ = select.select(
                [master],
                [],
                [],
                0.1,
            )

            if readable:
                chunk = read_chunk()
                if chunk:
                    output.extend(chunk)

            waited_pid, _status = os.waitpid(
                pid,
                os.WNOHANG,
            )

            if waited_pid == pid:
                child_reaped = True
                break

        # Drain bytes emitted immediately before process exit.
        while True:
            readable, _, _ = select.select(
                [master],
                [],
                [],
                0,
            )

            if not readable:
                break

            chunk = read_chunk()

            if not chunk:
                break

            output.extend(chunk)

    finally:
        if not child_reaped:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass

        os.close(master)

    return output.decode(
        "utf-8",
        errors="backslashreplace",
    )

@pytest.mark.skipif(sys.platform != "darwin", reason="macOS libedit PTY regression")
def test_composer_backspace_removes_whole_chinese_characters() -> None:
    bare = _pty_exchange("value=input('> '); print('RESULT=' + ascii(value))")
    assert (
    "\\udc" in bare
    or "UnicodeDecodeError" in bare
)

    composed = _pty_exchange(
        "from agent_runtime.terminal_input import TerminalComposer; "
        "value=TerminalComposer().prompt('> '); print('RESULT=' + ascii(value))",
        wait_for_raw_mode=True,
    )

    assert "RESULT=''" in composed
    assert "\\udc" not in composed


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS PTY integration")
@pytest.mark.parametrize("typed_text", ["e\u0301", "👨‍👩‍👧‍👦"])
def test_composer_backspace_removes_one_grapheme_cluster(typed_text: str) -> None:
    composed = _pty_exchange(
        "from agent_runtime.terminal_input import TerminalComposer; "
        "value=TerminalComposer().prompt('> '); print('RESULT=' + ascii(value))",
        typed_text=typed_text,
        backspaces=1,
        wait_for_raw_mode=True,
    )

    assert "RESULT=''" in composed


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS PTY integration")
@pytest.mark.parametrize("typed_text", ["e\u0301", "👨‍👩‍👧‍👦"])
def test_composer_forward_delete_removes_one_grapheme_cluster(typed_text: str) -> None:
    composed = _pty_exchange(
        "from agent_runtime.terminal_input import TerminalComposer; "
        "value=TerminalComposer().prompt('> '); print('RESULT=' + ascii(value))",
        typed_text=typed_text,
        edit_keys=b"\x1b[H\x1b[3~\r",
        wait_for_raw_mode=True,
    )

    assert "RESULT=''" in composed


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS PTY integration")
@pytest.mark.parametrize("typed_text", ["e\u0301", "👨‍👩‍👧‍👦"])
def test_composer_cursor_moves_across_one_grapheme_cluster(typed_text: str) -> None:
    composed = _pty_exchange(
        "from agent_runtime.terminal_input import TerminalComposer; "
        "value=TerminalComposer().prompt('> '); print('RESULT=' + ascii(value))",
        typed_text=typed_text,
        edit_keys=b"\x1b[DX\r",
        wait_for_raw_mode=True,
    )

    assert f"RESULT={ascii('X' + typed_text)}" in composed

    composed = _pty_exchange(
        "from agent_runtime.terminal_input import TerminalComposer; "
        "value=TerminalComposer().prompt('> '); print('RESULT=' + ascii(value))",
        typed_text=typed_text,
        edit_keys=b"\x1b[H\x1b[CX\r",
        wait_for_raw_mode=True,
    )

    assert f"RESULT={ascii(typed_text + 'X')}" in composed

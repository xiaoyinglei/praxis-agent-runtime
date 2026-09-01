from __future__ import annotations

import os
import select
import subprocess
import sys
import termios
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


def _pty_exchange(program: str, *, wait_for_raw_mode: bool = False) -> str:
    master, slave = os.openpty()
    os.set_blocking(master, False)
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    output = bytearray()
    deadline = time.monotonic() + 5

    def read_chunk() -> bytes | None:
        try:
            return os.read(master, 4_096)
        except (BlockingIOError, OSError):
            return None

    try:
        while b"> " not in output and time.monotonic() < deadline:
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                chunk = read_chunk()
                if chunk:
                    output.extend(chunk)
        while wait_for_raw_mode and time.monotonic() < deadline:
            if not termios.tcgetattr(slave)[3] & termios.ICANON:
                break
            time.sleep(0.01)
        os.write(master, "你好".encode())
        os.write(master, b"\x7f\x7f\n")
        os.close(slave)
        slave = -1
        while process.poll() is None and time.monotonic() < deadline:
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                chunk = read_chunk()
                if chunk:
                    output.extend(chunk)
        process.wait(timeout=1)
        while True:
            readable, _, _ = select.select([master], [], [], 0)
            if not readable:
                break
            chunk = read_chunk()
            if not chunk:
                break
            output.extend(chunk)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if slave >= 0:
            os.close(slave)
        os.close(master)
    return output.decode("utf-8", errors="backslashreplace")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS libedit PTY regression")
def test_composer_backspace_removes_whole_chinese_characters() -> None:
    bare = _pty_exchange("value=input('> '); print('RESULT=' + ascii(value))")
    assert "\\udc" in bare

    composed = _pty_exchange(
        "from agent_runtime.terminal_input import TerminalComposer; "
        "value=TerminalComposer().prompt('> '); print('RESULT=' + ascii(value))",
        wait_for_raw_mode=True,
    )

    assert "RESULT=''" in composed
    assert "\\udc" not in composed

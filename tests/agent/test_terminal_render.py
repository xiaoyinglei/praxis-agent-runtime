from __future__ import annotations

from agent_runtime.terminal_render import (
    bounded_result_lines,
    display_rows,
    safe_terminal_text,
)


def test_safe_terminal_text_removes_ansi_and_unsafe_controls() -> None:
    value = "\x1b[31mred\x1b[0m\x00\x07\nnext\tvalue"

    rendered = safe_terminal_text(value)

    assert rendered == "red\nnext\tvalue"


def test_display_rows_respects_cjk_cell_width() -> None:
    assert display_rows("你好a", width=4) == ["你好", "a"]


def test_display_rows_does_not_split_combining_graphemes() -> None:
    assert display_rows("e\u0301e\u0301", width=1) == ["e\u0301", "e\u0301"]


def test_display_rows_measures_zwj_emoji_as_one_cluster() -> None:
    family = "👨‍👩‍👧‍👦"

    assert display_rows(f"{family}{family}", width=2) == [family, family]


def test_bounded_result_lines_formats_short_json() -> None:
    assert bounded_result_lines({"ok": True}, width=80) == ["{", '  "ok": true', "}"]


def test_bounded_result_lines_keeps_head_and_tail_with_exact_marker() -> None:
    value = "\n".join(f"line {index}" for index in range(12))

    lines = bounded_result_lines(value, width=80, max_rows=8)

    assert lines == [
        "line 0",
        "line 1",
        "line 2",
        "… +5 lines (/verbose 查看完整结果)",
        "line 8",
        "line 9",
        "line 10",
        "line 11",
    ]


def test_bounded_result_lines_verbose_returns_every_row() -> None:
    value = "\n".join(f"line {index}" for index in range(12))

    lines = bounded_result_lines(value, width=80, max_rows=8, verbose=True)

    assert lines == [f"line {index}" for index in range(12)]

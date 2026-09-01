from __future__ import annotations

from agent_runtime.terminal_render import (
    BoundedCommandPreview,
    BoundedProgressPreview,
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


def test_command_preview_preserves_chunk_order_and_equal_deltas() -> None:
    preview = BoundedCommandPreview(width=80)

    visible = [
        *preview.feed("same\n"),
        *preview.feed("same\nerr"),
        *preview.feed("or\nlast\n"),
        *preview.finish(),
    ]

    assert visible == ["same", "same", "error", "last"]


def test_command_preview_keeps_six_head_and_three_tail_rows() -> None:
    preview = BoundedCommandPreview(width=80)
    visible: list[str] = []

    for index in range(12):
        visible.extend(preview.feed(f"line {index}\n"))
    visible.extend(preview.finish())

    assert visible == [
        "line 0",
        "line 1",
        "line 2",
        "line 3",
        "line 4",
        "line 5",
        "… +3 lines (/verbose 查看完整结果)",
        "line 9",
        "line 10",
        "line 11",
    ]


def test_command_preview_bounds_a_long_line_and_reports_omitted_characters() -> None:
    preview = BoundedCommandPreview(width=80, max_partial_bytes=16 * 1024)

    visible = [*preview.feed("x" * 100_000)]

    assert visible == []
    assert preview.retained_bytes <= 16 * 1024

    visible.extend(preview.finish())

    assert any("… +83616 chars" in line for line in visible)
    assert visible[0] == "x" * 80
    assert visible[-1] == "x" * 32


def test_progress_preview_limits_messages_and_reports_suppression() -> None:
    preview = BoundedProgressPreview(max_messages=8)

    visible = [preview.feed(f"progress {index}") for index in range(11)]

    assert visible[:8] == [f"progress {index}" for index in range(8)]
    assert visible[8:] == [None, None, None]
    assert preview.finish() == "… +3 progress updates"

from __future__ import annotations

import json

import regex
from wcwidth import wcswidth

DEFAULT_RESULT_ROWS = 8
DEFAULT_TERMINAL_WIDTH = 100

_ANSI_ESCAPE = regex.compile(
    r"(?:\x1b\][^\x07]*(?:\x07|\x1b\\))|(?:\x1b\[[0-?]*[ -/]*[@-~])"
)
_UNSAFE_CONTROL = regex.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_GRAPHEME = regex.compile(r"\X")


def safe_terminal_text(value: str) -> str:
    """Remove terminal control sequences without changing durable data."""

    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    without_ansi = _ANSI_ESCAPE.sub("", normalized)
    return _UNSAFE_CONTROL.sub("", without_ansi)


def display_rows(value: str, *, width: int) -> list[str]:
    """Wrap text by terminal cells while preserving grapheme clusters."""

    if width < 1:
        raise ValueError("terminal width must be positive")
    rows: list[str] = []
    for logical_line in safe_terminal_text(value).expandtabs(4).split("\n"):
        current: list[str] = []
        current_width = 0
        for cluster in _GRAPHEME.findall(logical_line):
            cluster_width = wcswidth(cluster)
            if cluster_width < 0:
                cluster = "�"
                cluster_width = 1
            if current and current_width + cluster_width > width:
                rows.append("".join(current))
                current = []
                current_width = 0
            current.append(cluster)
            current_width += cluster_width
        rows.append("".join(current))
    return rows


def _structured_text(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def bounded_result_lines(
    value: object,
    *,
    width: int = DEFAULT_TERMINAL_WIDTH,
    max_rows: int = DEFAULT_RESULT_ROWS,
    verbose: bool = False,
) -> list[str]:
    """Format a result with Codex-style middle omission for terminal display."""

    if max_rows < 1:
        raise ValueError("result row budget must be positive")
    rows = display_rows(_structured_text(value), width=width)
    if verbose or len(rows) <= max_rows:
        return rows
    retained = max_rows - 1
    head_count = retained // 2
    tail_count = retained - head_count
    omitted = len(rows) - head_count - tail_count
    marker = f"… +{omitted} lines (/verbose 查看完整结果)"
    tail = rows[-tail_count:] if tail_count else []
    return [*rows[:head_count], marker, *tail]

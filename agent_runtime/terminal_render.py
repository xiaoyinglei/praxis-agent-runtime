from __future__ import annotations

import json
from collections import deque

import regex
from wcwidth import wcswidth

DEFAULT_RESULT_ROWS = 8
DEFAULT_TERMINAL_WIDTH = 100
DEFAULT_COMMAND_HEAD_ROWS = 6
DEFAULT_COMMAND_TAIL_ROWS = 3
DEFAULT_PARTIAL_ROW_BYTES = 16 * 1024
DEFAULT_PROGRESS_MESSAGES = 8

_ANSI_ESCAPE = regex.compile(
    r"(?:\x1b\][^\x07]*(?:\x07|\x1b\\))|(?:\x1b\[[0-?]*[ -/]*[@-~])"
)
_UNSAFE_CONTROL = regex.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_GRAPHEME = regex.compile(r"\X")


class _BoundedPartialRow:
    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._head = ""
        self._tail = ""
        self._omitted = 0

    @property
    def has_content(self) -> bool:
        return bool(self._head or self._tail or self._omitted)

    @property
    def retained_bytes(self) -> int:
        return len(self._head.encode("utf-8")) + len(self._tail.encode("utf-8"))

    @property
    def omitted(self) -> int:
        return self._omitted

    def append(self, value: str) -> None:
        if not value:
            return
        if self._omitted == 0:
            combined = self._head + value
            if len(combined.encode("utf-8")) <= self._max_bytes:
                self._head = combined
                return
            clusters = _GRAPHEME.findall(combined)
            self._head = _prefix_within_bytes(clusters, self._max_bytes // 2)
            self._tail = _suffix_within_bytes(clusters, self._max_bytes // 2)
            retained = len(_GRAPHEME.findall(self._head)) + len(
                _GRAPHEME.findall(self._tail)
            )
            self._omitted = len(clusters) - retained
            return

        clusters = [*_GRAPHEME.findall(self._tail), *_GRAPHEME.findall(value)]
        tail = _suffix_within_bytes(clusters, self._max_bytes // 2)
        retained = len(_GRAPHEME.findall(tail))
        self._omitted += len(clusters) - retained
        self._tail = tail

    def parts(self) -> tuple[str, int, str]:
        return self._head, self._omitted, self._tail


def _prefix_within_bytes(clusters: list[str], budget: int) -> str:
    retained: list[str] = []
    used = 0
    for cluster in clusters:
        size = len(cluster.encode("utf-8"))
        if used + size > budget:
            break
        retained.append(cluster)
        used += size
    return "".join(retained)


def _suffix_within_bytes(clusters: list[str], budget: int) -> str:
    retained: deque[str] = deque()
    used = 0
    for cluster in reversed(clusters):
        size = len(cluster.encode("utf-8"))
        if used + size > budget:
            break
        retained.appendleft(cluster)
        used += size
    return "".join(retained)


class BoundedCommandPreview:
    """Bounded plain-terminal projection for ordered command deltas."""

    def __init__(
        self,
        *,
        width: int,
        head_rows: int = DEFAULT_COMMAND_HEAD_ROWS,
        tail_rows: int = DEFAULT_COMMAND_TAIL_ROWS,
        max_partial_bytes: int = DEFAULT_PARTIAL_ROW_BYTES,
    ) -> None:
        if width < 1 or head_rows < 0 or tail_rows < 0 or max_partial_bytes < 1:
            raise ValueError("command preview limits are invalid")
        self._width = width
        self._head_rows = head_rows
        self._max_partial_bytes = max_partial_bytes
        self._tail: deque[str] = deque(maxlen=tail_rows)
        self._partial = _BoundedPartialRow(max_partial_bytes)
        self._rows = 0
        self._suppressed_rows = 0
        self._forced_markers: list[str] = []
        self._finished = False

    @property
    def retained_bytes(self) -> int:
        return self._partial.retained_bytes + sum(
            len(line.encode("utf-8")) for line in self._tail
        )

    def feed(self, delta: str) -> list[str]:
        if self._finished:
            return []
        visible: list[str] = []
        parts = safe_terminal_text(delta).split("\n")
        for part in parts[:-1]:
            self._partial.append(part)
            visible.extend(self._complete_partial())
        self._partial.append(parts[-1])
        return visible

    def finish(self) -> list[str]:
        if self._finished:
            return []
        visible = self._complete_partial() if self._partial.has_content else []
        visible.extend(self._forced_markers)
        omitted = self._suppressed_rows - len(self._tail)
        if omitted > 0:
            visible.append(f"… +{omitted} lines (/verbose 查看完整结果)")
        visible.extend(self._tail)
        self._tail.clear()
        self._forced_markers.clear()
        self._finished = True
        return visible

    def _complete_partial(self) -> list[str]:
        head, omitted_chars, tail = self._partial.parts()
        self._partial = _BoundedPartialRow(self._max_partial_bytes)
        visible: list[str] = []
        for row in display_rows(head, width=self._width):
            visible.extend(self._accept_row(row))
        if omitted_chars:
            self._forced_markers.append(f"… +{omitted_chars} chars")
            for row in display_rows(tail, width=self._width):
                visible.extend(self._accept_row(row))
        return visible

    def _accept_row(self, row: str) -> list[str]:
        self._rows += 1
        if self._rows <= self._head_rows:
            return [row]
        self._suppressed_rows += 1
        self._tail.append(row)
        return []


class BoundedProgressPreview:
    def __init__(self, *, max_messages: int = DEFAULT_PROGRESS_MESSAGES) -> None:
        if max_messages < 1:
            raise ValueError("progress message limit must be positive")
        self._max_messages = max_messages
        self._seen = 0
        self._suppressed = 0

    def feed(self, message: str) -> str | None:
        self._seen += 1
        if self._seen <= self._max_messages:
            return safe_terminal_text(message)
        self._suppressed += 1
        return None

    def finish(self) -> str | None:
        if not self._suppressed:
            return None
        return f"… +{self._suppressed} progress updates"


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

from __future__ import annotations

import json
import shutil
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import regex
from wcwidth import wcswidth

from agent_runtime.streaming.events import (
    EventType,
    ItemDeltaKind,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
)

DEFAULT_RESULT_ROWS = 8
DEFAULT_TERMINAL_WIDTH = 100
DEFAULT_COMMAND_HEAD_ROWS = 6
DEFAULT_COMMAND_TAIL_ROWS = 3
DEFAULT_PARTIAL_ROW_BYTES = 16 * 1024
DEFAULT_PROGRESS_MESSAGES = 8

_ANSI_ESCAPE = regex.compile(
    r"(?:\x1b\][^\x07]*(?:\x07|\x1b\\))|(?:\x1b\[[0-?]*[ -/]*[@-~])"
)
_UNSAFE_CONTROL = regex.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
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
        self._pending_cr = False
        self._finished = False

    @property
    def retained_bytes(self) -> int:
        return self._partial.retained_bytes + sum(
            len(line.encode("utf-8")) for line in self._tail
        )

    @property
    def has_output(self) -> bool:
        return self._rows > 0 or self._partial.has_content or self._pending_cr

    def feed(self, delta: str) -> list[str]:
        if self._finished:
            return []
        visible: list[str] = []
        if self._pending_cr:
            delta = "\n" + (delta[1:] if delta.startswith("\n") else delta)
            self._pending_cr = False
        if delta.endswith("\r"):
            delta = delta[:-1]
            self._pending_cr = True
        parts = safe_terminal_text(delta).split("\n")
        for part in parts[:-1]:
            self._partial.append(part)
            visible.extend(self._complete_partial())
        self._partial.append(parts[-1])
        return visible

    def finish(self) -> list[str]:
        if self._finished:
            return []
        if self._pending_cr:
            self._pending_cr = False
            visible = self._complete_partial()
        else:
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


@dataclass(slots=True)
class _ItemDisplayState:
    name: str
    kind: TurnItemKind
    command: BoundedCommandPreview | None = None
    progress: BoundedProgressPreview | None = None


class TerminalToolEventDisplay:
    """Project canonical and legacy stream events into bounded terminal output."""

    def __init__(
        self,
        *,
        width: int | None = None,
        max_lifecycle_keys: int = 256,
    ) -> None:
        if max_lifecycle_keys < 1:
            raise ValueError("lifecycle key limit must be positive")
        terminal_width = shutil.get_terminal_size(fallback=(100, 24)).columns
        self._width = max(20, width if width is not None else terminal_width)
        self._max_lifecycle_keys = max_lifecycle_keys
        self._lifecycle: OrderedDict[
            tuple[str, str, EventType], None
        ] = OrderedDict()
        self._items: OrderedDict[tuple[str, str], _ItemDisplayState] = OrderedDict()
        self._plans: OrderedDict[tuple[str, int | str], None] = OrderedDict()
        self._verbose = False
        self._line_open = False
        self.answer_streamed = False

    @property
    def lifecycle_key_count(self) -> int:
        return len(self._lifecycle)

    @property
    def active_item_count(self) -> int:
        return len(self._items)

    def set_verbose(self, verbose: bool) -> None:
        self._verbose = verbose

    async def emit(self, event: StreamEvent) -> None:
        if event.type is EventType.ITEM_STARTED:
            self._render_item_start(event)
        elif event.type is EventType.ITEM_DELTA:
            self._render_item_delta(event)
        elif event.type is EventType.ITEM_COMPLETED:
            self._render_item_completed(event)
        elif event.type is EventType.TEXT_DELTA:
            self._render_text(event.data.get("text"))
        elif event.type is EventType.TOOL_USE_START:
            self._render_legacy_start(event)
        elif event.type is EventType.TOOL_USE_PROGRESS:
            self._render_legacy_progress(event)
        elif event.type is EventType.TOOL_USE_RESULT:
            self._render_legacy_result(event)
        elif event.type is EventType.TOOL_USE_ERROR:
            self._render_legacy_error(event)
        elif event.type is EventType.PLAN_UPDATED:
            self._render_plan(event)
        elif event.type is EventType.RECOVERY:
            strategy = event.data.get("strategy")
            if isinstance(strategy, str) and strategy:
                detail = event.data.get("detail")
                suffix = f" — {detail}" if isinstance(detail, str) and detail else ""
                self._write_line(f"↻ 恢复: {strategy}{suffix}")

    def begin_turn(self) -> None:
        self.finish()
        self.answer_streamed = False

    def finish(self) -> None:
        if self._line_open:
            print(flush=True)
            self._line_open = False

    def _render_item_start(self, event: StreamEvent) -> None:
        if event.item_kind not in {TurnItemKind.TOOL, TurnItemKind.COMMAND}:
            return
        if event.item_id is None or not self._remember(event, event.item_id):
            return
        name_value = event.data.get("tool_name")
        name = (
            name_value
            if isinstance(name_value, str) and name_value
            else event.item_kind.value
        )
        state = _ItemDisplayState(name=name, kind=event.item_kind)
        if event.item_kind is TurnItemKind.COMMAND:
            state.command = BoundedCommandPreview(width=self._width - 2)
        else:
            state.progress = BoundedProgressPreview()
        self._store_item((event.turn_id, event.item_id), state)
        self._render_start_line(name, event.data.get("input_preview"))

    def _render_item_delta(self, event: StreamEvent) -> None:
        delta = event.data.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        if event.delta_kind is ItemDeltaKind.TEXT:
            self._render_text(delta)
            return
        if event.item_id is None:
            return
        key = (event.turn_id, event.item_id)
        state = self._items.get(key)
        if event.delta_kind in {
            ItemDeltaKind.COMMAND_STDOUT,
            ItemDeltaKind.COMMAND_STDERR,
        }:
            if self._verbose:
                self._render_text(safe_terminal_text(delta), answer=False)
                return
            if state is None:
                state = _ItemDisplayState(
                    name="command",
                    kind=TurnItemKind.COMMAND,
                    command=BoundedCommandPreview(width=self._width - 2),
                )
                self._store_item(key, state)
            if state.command is None:
                state.command = BoundedCommandPreview(width=self._width - 2)
            for line in state.command.feed(delta):
                self._write_line(f"  {line}")
            return
        if event.delta_kind is ItemDeltaKind.TOOL_PROGRESS:
            if state is None:
                state = _ItemDisplayState(
                    name="tool",
                    kind=TurnItemKind.TOOL,
                    progress=BoundedProgressPreview(),
                )
                self._store_item(key, state)
            if state.progress is None:
                state.progress = BoundedProgressPreview()
            progress = state.progress.feed(delta)
            if progress is not None:
                self._write_line(f"… {state.name}: {progress}")

    def _render_item_completed(self, event: StreamEvent) -> None:
        if event.item_kind is TurnItemKind.PLAN:
            self._render_plan(event)
            return
        if event.item_kind not in {TurnItemKind.TOOL, TurnItemKind.COMMAND}:
            return
        if event.item_id is None or not self._remember(event, event.item_id):
            return
        key = (event.turn_id, event.item_id)
        state = self._items.pop(key, None)
        result_value = event.data.get("result")
        result = result_value if isinstance(result_value, Mapping) else {}
        name_value = result.get("tool_name")
        name = (
            name_value
            if isinstance(name_value, str) and name_value
            else state.name
            if state is not None
            else event.item_kind.value
        )
        self._flush_item_state(state)
        command_output_streamed = (
            state is not None
            and state.command is not None
            and state.command.has_output
        )
        if event.status is ItemStatus.SUCCESS:
            if event.item_kind is TurnItemKind.COMMAND or name == "run_command":
                self._write_line(f"✓ {name}{self._command_suffix(result)}")
                if not command_output_streamed:
                    structured = result.get("structured_content")
                    if structured is not None:
                        self._write_result(structured)
            else:
                self._write_line(f"✓ {name}")
                structured = result.get("structured_content")
                if structured is not None:
                    self._write_result(structured)
            self._render_truncation_warnings(name, result)
            self._render_diff(result.get("metadata"))
            return
        error = event.error or result.get("error_message")
        suffix = f": {safe_terminal_text(error)}" if isinstance(error, str) else ""
        command_suffix = (
            self._command_suffix(result)
            if event.item_kind is TurnItemKind.COMMAND or name == "run_command"
            else ""
        )
        self._write_line(f"✗ {name}{command_suffix}{suffix}")
        structured = result.get("structured_content")
        if structured is not None and (
            event.item_kind is not TurnItemKind.COMMAND and name != "run_command"
            or not command_output_streamed
        ):
            self._write_result(structured)
        self._render_truncation_warnings(name, result)
        self._render_diff(result.get("metadata"))

    def _render_legacy_start(self, event: StreamEvent) -> None:
        tool_id = event.data.get("tool_id")
        name = event.data.get("tool_name")
        if not isinstance(tool_id, str) or not tool_id or not isinstance(name, str):
            return
        if not self._remember(event, tool_id):
            return
        self._store_item(
            (event.turn_id, tool_id),
            _ItemDisplayState(
                name=name,
                kind=TurnItemKind.TOOL,
                progress=BoundedProgressPreview(),
            ),
        )
        self._render_start_line(name, event.data.get("input_preview"))

    def _render_legacy_progress(self, event: StreamEvent) -> None:
        tool_id = event.data.get("tool_id")
        progress = event.data.get("progress")
        if not isinstance(tool_id, str) or not isinstance(progress, str):
            return
        state = self._items.get((event.turn_id, tool_id))
        if state is None:
            state = _ItemDisplayState(
                name="tool",
                kind=TurnItemKind.TOOL,
                progress=BoundedProgressPreview(),
            )
            self._store_item((event.turn_id, tool_id), state)
        if state.progress is None:
            state.progress = BoundedProgressPreview()
        name = state.name
        percent = event.data.get("percent")
        percent_text = f" ({percent:g}%)" if isinstance(percent, (int, float)) else ""
        visible = state.progress.feed(f"{progress}{percent_text}")
        if visible is not None:
            self._write_line(f"… {name}: {visible}")

    def _render_legacy_result(self, event: StreamEvent) -> None:
        tool_id = event.data.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id or not self._remember(event, tool_id):
            return
        state = self._items.pop((event.turn_id, tool_id), None)
        name_value = event.data.get("tool_name")
        name = (
            name_value
            if isinstance(name_value, str) and name_value
            else state.name
            if state is not None
            else "tool"
        )
        self._flush_item_state(state)
        self._write_line(f"✓ {name}")
        result = event.data.get("result")
        if result is not None and result != "":
            self._write_result(result)
        self._render_diff(event.data.get("details"))

    def _render_legacy_error(self, event: StreamEvent) -> None:
        tool_id = event.data.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id or not self._remember(event, tool_id):
            return
        state = self._items.pop((event.turn_id, tool_id), None)
        name = state.name if state is not None else "tool"
        self._flush_item_state(state)
        error = event.data.get("error")
        suffix = f": {safe_terminal_text(error)}" if isinstance(error, str) else ""
        self._write_line(f"✗ {name}{suffix}")

    def _render_plan(self, event: StreamEvent) -> None:
        plan = event.data.get("plan")
        if not isinstance(plan, Mapping):
            return
        revision = plan.get("revision")
        if not isinstance(revision, (int, str)):
            return
        key = (event.turn_id, revision)
        if key in self._plans:
            return
        self._plans[key] = None
        if len(self._plans) > self._max_lifecycle_keys:
            self._plans.popitem(last=False)
        self._write_line(f"计划 (revision {revision})")
        steps = plan.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            return
        symbols = {"completed": "✓", "in_progress": "→", "failed": "✗"}
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            title = step.get("title")
            if not isinstance(title, str) or not title:
                continue
            status = step.get("status")
            symbol = symbols.get(status, "○") if isinstance(status, str) else "○"
            self._write_line(f"  {symbol} {safe_terminal_text(title)}")

    def _render_start_line(self, name: str, preview: object) -> None:
        if not isinstance(preview, str) or not preview:
            self._write_line(f"→ {name}")
            return
        name_width = max(wcswidth(safe_terminal_text(name)), 0)
        lines = bounded_result_lines(
            preview,
            width=max(1, self._width - name_width - 4),
            max_rows=2,
        )
        self._write_line(f"→ {name}: {lines[0]}")
        for line in lines[1:]:
            self._write_line(f"  {line}")

    def _write_result(self, value: object) -> None:
        for line in bounded_result_lines(
            value,
            width=self._width - 2,
            verbose=self._verbose,
        ):
            self._write_line(f"  {line}")

    def _flush_item_state(self, state: _ItemDisplayState | None) -> None:
        if state is None:
            return
        if state.command is not None and not self._verbose:
            for line in state.command.finish():
                self._write_line(f"  {line}")
        if state.progress is not None:
            marker = state.progress.finish()
            if marker is not None:
                self._write_line(f"… {state.name}: {marker}")

    def _render_truncation_warnings(
        self,
        name: str,
        result: Mapping[str, object],
    ) -> None:
        if result.get("truncated") is True:
            self._write_line("⚠ 工具结果已由 ACI 截断。")
        structured = result.get("structured_content")
        if (
            name == "run_command"
            and isinstance(structured, Mapping)
            and structured.get("truncated") is True
        ):
            self._write_line("⚠ 命令输出达到工具预算，已保留头尾。")

    def _render_diff(self, metadata: object) -> None:
        if not isinstance(metadata, Mapping):
            return
        diff = metadata.get("diff")
        if isinstance(diff, str) and diff:
            self._write_block(safe_terminal_text(diff))

    def _command_suffix(self, result: Mapping[str, object]) -> str:
        structured = result.get("structured_content")
        if not isinstance(structured, Mapping):
            return ""
        fields: list[str] = []
        for key in ("exit_code", "timed_out", "duration_ms"):
            value = structured.get(key)
            if isinstance(value, (str, int, float, bool)):
                fields.append(f"{key}={value}")
        return f": {', '.join(fields)}" if fields else ""

    def _remember(self, event: StreamEvent, identity: str) -> bool:
        key = (event.turn_id, identity, event.type)
        if key in self._lifecycle:
            self._lifecycle.move_to_end(key)
            return False
        self._lifecycle[key] = None
        if len(self._lifecycle) > self._max_lifecycle_keys:
            self._lifecycle.popitem(last=False)
        return True

    def _store_item(
        self,
        key: tuple[str, str],
        state: _ItemDisplayState,
    ) -> None:
        self._items[key] = state
        self._items.move_to_end(key)
        if len(self._items) > self._max_lifecycle_keys:
            self._items.popitem(last=False)

    def _render_text(self, value: object, *, answer: bool = True) -> None:
        if not isinstance(value, str) or not value:
            return
        rendered = safe_terminal_text(value)
        if not rendered:
            return
        print(rendered, end="", flush=True)
        if answer:
            self.answer_streamed = True
        self._line_open = not rendered.endswith("\n")

    def _write_line(self, value: str) -> None:
        if self._line_open:
            print()
        print(safe_terminal_text(value), flush=True)
        self._line_open = False

    def _write_block(self, value: str) -> None:
        if self._line_open:
            print()
        print(safe_terminal_text(value).rstrip("\n"), flush=True)
        self._line_open = False

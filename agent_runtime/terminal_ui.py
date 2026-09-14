"""Retained terminal conversation and event projection; no terminal I/O here."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agent_runtime.streaming.events import EventType, StreamEvent, TurnItemKind
from agent_runtime.terminal_render import TerminalToolEventDisplay, bounded_result_lines, safe_terminal_text

MAX_TEXT = 128 * 1024


@dataclass(eq=False)
class Block:
    kind: str
    title: str = ""
    text: str = ""
    expanded: bool = False
    children: list[Block] = field(default_factory=list)
    omitted: int = 0
    failed: bool = False

    def append(self, text: str) -> None:
        self.text += safe_terminal_text(text)
        self.trim(MAX_TEXT)

    def trim(self, limit: int) -> None:
        if len(self.text) > limit:
            self.omitted += len(self.text) - limit
            head = limit // 2
            tail = limit - head
            self.text = self.text[:head] + (self.text[-tail:] if tail else "")

    def detail(self) -> str:
        if not self.omitted:
            return self.text
        middle = len(self.text) // 2
        return self.text[:middle] + f"\n… 已省略 {self.omitted} 字符 …\n" + self.text[middle:]


class Conversation:
    def __init__(self, invalidate: Callable[[], None] = lambda: None, *, max_characters: int = 8 * 1024 * 1024) -> None:
        self.blocks: list[Block] = []
        self._invalidate = invalidate
        self._max_characters = max_characters
        self.omitted_blocks = 0
        self.revision = 0

    def invalidate(self) -> None:
        self.revision += 1
        retained = [node for block in self.blocks for node in (block, *block.children)]
        excess = sum(len(node.text) for node in retained) - self._max_characters
        for node in retained:
            if excess <= 0:
                break
            remove = min(len(node.text), excess)
            node.trim(len(node.text) - remove)
            excess -= remove
        self._invalidate()

    def _add(self, block: Block) -> Block:
        self.blocks.append(block)
        if len(self.blocks) > 100:
            self.blocks.pop(0)
            self.omitted_blocks += 1
        self.invalidate()
        return block

    def message(self, text: str, kind: str = "answer") -> None:
        if self.blocks and self.blocks[-1].kind == kind and kind != "user":
            self.blocks[-1].append(text)
        else:
            block = self._add(Block(kind))
            block.append(text)
        self.invalidate()

    def add_group(self) -> Block:
        return self._add(Block("group", "执行记录", expanded=True))

    def add_tool(self, group: Block, title: str) -> Block:
        block = Block("tool", title)
        if len(group.children) >= 256:
            group.children.pop(0)
            group.omitted += 1
        group.children.append(block)
        self.invalidate()
        return block

    def toggle(self, block: Block) -> None:
        block.expanded = not block.expanded
        self.invalidate()

    def rows(self) -> list[tuple[str, Block | None, str]]:
        rows: list[tuple[str, Block | None, str]] = []
        if self.omitted_blocks:
            rows.append((f"… 已省略 {self.omitted_blocks} 段较早记录", None, "dim"))
        for block in self.blocks:
            if block.kind != "group":
                prefix = "❯ " if block.kind == "user" else "● " if block.kind == "answer" else ""
                rows.append((prefix + block.detail(), None, block.kind))
            else:
                errors = sum(child.failed for child in block.children)
                suffix = f" · {errors} 项异常" if errors else ""
                rows.append(
                    (("▾ " if block.expanded else "▸ ") + block.title + suffix, block, "error" if errors else "group")
                )
                if block.expanded:
                    if block.omitted:
                        rows.append((f"  … 已省略 {block.omitted} 项较早操作", None, "dim"))
                    if block.text:
                        rows.append(("  " + block.detail(), None, "dim"))
                    for child in block.children:
                        rows.append(
                            (
                                "  " + ("▾ " if child.expanded else "▸ ") + child.title,
                                child,
                                "error" if child.failed else "tool",
                            )
                        )
                        if child.expanded and child.text:
                            rows.append(
                                ("\n".join("    " + line for line in child.detail().splitlines()), None, "detail")
                            )
            rows.append(("", None, ""))
        return rows

    def plain_text(self) -> str:
        return "\n".join(text for text, _, _ in self.rows())


_TOOL_LABELS = {
    "read_file": "读取文件",
    "write_file": "写入文件",
    "apply_patch": "修改文件",
    "run_command": "执行命令",
    "list_directory": "列出目录",
    "search_files": "搜索文件",
}


class ConversationEventDisplay(TerminalToolEventDisplay):
    """Reuse the CLI formatter, replacing append-only writes with retained blocks."""

    def __init__(self, view: Conversation) -> None:
        super().__init__(interactive=False)
        self.view = view
        self.group: Block | None = None
        self._target: Block | None = None
        self._tool_blocks: dict[tuple[str, str], Block] = {}
        self.turn_id: str | None = None
        self._new_answer = True

    def begin_turn(self, *, reset: bool = True) -> None:
        super().begin_turn(reset=reset)
        if reset or self.group is None:
            self.group = None  # Create only when an execution event actually arrives.
            self._tool_blocks.clear()
            self.turn_id = None
        elif self.group:
            self.group.expanded = True
        self.view.invalidate()

    async def emit(self, event: StreamEvent) -> None:
        if self.turn_id is None and event.turn_id:
            self.turn_id = event.turn_id
        identity = event.item_id or event.data.get("tool_id")
        is_tool = event.item_kind in {TurnItemKind.TOOL, TurnItemKind.COMMAND} or event.type in {
            EventType.TOOL_USE_START,
            EventType.TOOL_USE_PROGRESS,
            EventType.TOOL_USE_RESULT,
            EventType.TOOL_USE_ERROR,
        }
        self._target = None
        if is_tool and isinstance(identity, str):
            if self.group is None:
                self.group = self.view.add_group()
            key = (event.turn_id, identity)
            if key not in self._tool_blocks:
                result = event.data.get("result")
                result_name = result.get("tool_name") if isinstance(result, dict) else None
                name = str(event.data.get("tool_name") or result_name or event.item_kind or "工具")
                if event.data.get("execution_started") is False:
                    name = "未执行 · " + name
                self._tool_blocks[key] = self.view.add_tool(self.group, name)
                retained = set(self.group.children)
                self._tool_blocks = {k: v for k, v in self._tool_blocks.items() if v in retained}
            self._target = self._tool_blocks[key]
        if event.type is EventType.ITEM_STARTED and event.item_kind is TurnItemKind.AGENT_MESSAGE:
            self._new_answer = True
        try:
            await super().emit(event)
        finally:
            self._target = None
        if self.group and self._started_at is not None:
            self.group.title = f"执行记录 · {self._tool_count} 次工具"
        self.view.invalidate()

    def _render_start_line(self, name: str, preview: object) -> None:
        self._answer_has_text = False
        self._new_answer = True
        self._pending_answer_whitespace = ""
        label = _TOOL_LABELS.get(name, name)
        self._activity_status = f"正在{label}"
        if self._target:
            title = safe_terminal_text(str(preview or "")).replace("\n", " ")
            self._target.title = f"● {label}" + (f" · {title[:180]}" if title else "")
            self._target.append(f"{name}\n{safe_terminal_text(str(preview or ''))}\n")

    def _record_detail(self, value: str) -> None:
        if self._target:
            self._target.append(value + "\n")
        else:
            self.view.message(value + "\n", "notice")

    def _write_line(self, value: str, *, record: bool = True) -> None:
        if not record:  # Command preview duplicates the separately retained command rows.
            return
        if self._target and value.startswith(("✓", "✗")):
            self._target.title = value[0] + " " + self._target.title.removeprefix("● ")
            self._target.failed = value.startswith("✗")
            if self._target.failed:
                self._target.expanded = True
                self.view.message(value + "\n", "error")
        self._record_detail(value)

    def _write_block(self, value: str) -> None:
        self._record_detail(value)

    def _write_result(self, value: object) -> None:
        # Folding hides retained content; it must not reuse the eight-row plain CLI preview.
        for line in bounded_result_lines(value, width=self._width, max_rows=2000):
            self._record_detail(line.replace("(/verbose 查看完整结果)", "(界面显示上限)"))

    def _render_text(self, value: object, *, answer: bool = True) -> None:
        if not isinstance(value, str) or not value:
            return
        rendered = safe_terminal_text(value)
        if not rendered:
            return
        if answer:
            if not self._answer_has_text:
                if not rendered.strip():
                    self._pending_answer_whitespace = (self._pending_answer_whitespace + rendered)[-4096:]
                    return
                rendered = self._pending_answer_whitespace + rendered
                self._pending_answer_whitespace = ""
                self._answer_has_text = True
                if self._new_answer and self.view.blocks and self.view.blocks[-1].kind == "answer":
                    self.view._add(Block("answer"))
                self._new_answer = False
            self.answer_streamed = True
            self.view.message(rendered)
            self._activity_status = "正在回答"
        elif self._target:
            self._target.append(value)
        self.view.invalidate()

    def set_verbose(self, verbose: bool) -> None:
        # Verbose is a view preference; it must not change event retention.
        for block in self.view.blocks:
            if block.kind == "group":
                block.expanded = verbose
                for child in block.children:
                    child.expanded = verbose
        self.view.invalidate()

    def interrupted(self) -> None:
        self.finish()
        if self.group:
            self.group.title = self.group.title.replace("执行记录", "执行已中断", 1)
            for child in self.group.children:
                if child.title.startswith("●"):
                    child.title = "■ 结果未确认 · " + child.title.removeprefix("● ")
                    child.failed = True
            self.group.expanded = True
        self.view.invalidate()

    def finish(self) -> None:
        running = self._started_at is not None
        super().finish()
        if running and self.group:
            self.group.title = f"执行记录 · {self._tool_count} 次工具 · {self._elapsed_seconds:.1f}s"
            self.group.expanded = any(child.failed for child in self.group.children)
            for path, change in self._changed_files.items():
                self.view.message(f"修改：{path}{change}\n", "notice")
        self.view.invalidate()

"""Retained terminal conversation and event projection; no terminal I/O here."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from agent_runtime.result import AgentResult, AgentUsage
from agent_runtime.streaming.events import EventType, ItemDeltaKind, ItemStatus, StreamEvent, TurnItemKind
from agent_runtime.terminal_render import (
    TerminalToolEventDisplay,
    bounded_result_lines,
    display_rows,
    format_usage_summary,
    safe_terminal_text,
    web_error_message,
)

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
    in_scrollback: bool = False
    inline: bool = False
    operation_status: str = ""
    preview: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    session_id: str = ""

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
        self.details = False

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
        if self.blocks and self.blocks[-1].kind == kind and kind != "user" and not self.blocks[-1].in_scrollback:
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

    def rows(self, *, include_scrollback: bool = True, width: int | None = None) -> list[tuple[str, Block | None, str]]:
        def preview_row(text: str) -> str:
            if width is None or len(display_rows(text, width=max(1, width))) <= 1:
                return text
            return display_rows(text, width=max(1, width - 1))[0] + "…"

        rows: list[tuple[str, Block | None, str]] = []
        if self.omitted_blocks:
            rows.append((f"… 已省略 {self.omitted_blocks} 段较早记录", None, "dim"))
        for block in self.blocks:
            if block.in_scrollback and not include_scrollback:
                continue
            if block.kind == "diagnostic" and not self.details:
                continue
            if block.kind == "group" and block.inline and not self.details:
                children = block.children
                visible = [child for i, child in enumerate(children)
                           if i >= len(children) - 5 or child.failed or child.operation_status == "running"]
                hidden = len(children) - len(visible)
                if hidden:
                    rows.append((f"▸ 另有 {hidden} 项操作 · Ctrl+O 详情", block, "dim"))
                for child in visible:
                    rows.append((preview_row(child.title), child, "warning" if child.failed else "tool"))
                    if child.expanded:
                        rows.append(("\n".join("  " + line for line in child.detail().splitlines()), None, "detail"))
                    elif child.preview:
                        for line in child.preview.splitlines()[:2]:
                            rows.append((preview_row("  ⎿ " + line), None, "warning" if child.failed else "dim"))
                rows.append(("", None, ""))
                continue
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
    "list_files": "查看目录",
    "execute_python": "执行 Python",
    "manage_command": "管理命令",
    "update_plan": "更新计划",
    "search_files": "搜索文件",
    "web_search": "搜索网页",
    "web_fetch": "读取网页",
}


class ConversationEventDisplay(TerminalToolEventDisplay):
    """Project canonical lifecycle facts into retained, compact terminal operations."""

    def __init__(self, view: Conversation, *, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(interactive=False, clock=clock)
        self.view = view
        self.group: Block | None = None
        self._target: Block | None = None
        self._tool_blocks: dict[tuple[str, str], Block] = {}
        self._projected: dict[tuple[str, str, EventType], None] = {}
        self._receipt: Block | None = None
        self._context_detail: Block | None = None
        self.turn_id: str | None = None
        self.connection_required = False
        self.search_configuration_required = False
        self._new_answer = True
        self._last_activity = clock()

    def update_usage(self, usage: AgentUsage) -> None:
        super().update_usage(usage)
        self.view.invalidate()

    def update_request_settings(self, snapshot: Mapping[str, object]) -> None:
        super().update_request_settings(snapshot)
        self.view.invalidate()

    def begin_turn(self, *, reset: bool = True) -> None:
        super().begin_turn(reset=reset)
        self._activity_status = "等待模型响应"
        self._last_activity = self._clock()
        # A resumed segment appends its own outcome after its new content. The
        # pause-time snapshot stays immutable while Turn usage remains cumulative.
        self._receipt = None
        if reset:
            if self._context_detail is not None:
                self._context_detail.text = "\n".join(
                    value for key, value in self._context_status.items() if key != "request_settings"
                )
            self.group = None
            self._tool_blocks.clear()
            self._projected.clear()
            self.turn_id = None
        self.view.invalidate()

    def activity_text(self) -> str:
        now = self._clock()
        active = len({b.session_id or identity for identity, b in self._tool_blocks.items()
                      if b.operation_status == "running"})
        status = self._activity_status
        if active > 1:
            status += f" · 另有 {active - 1} 项进行中"
        elif active and status == "这一步失败，继续处理中":
            status += " · 仍有操作进行中"
        idle = now - self._last_activity
        if self._started_at is not None and idle >= 15:
            status += f" · 距上次活动 {int(idle)} 秒"
        return status

    def runtime_stats(self, *, compact: bool = False) -> str:
        elapsed = max(0.0, self._elapsed_seconds + (
            0.0 if self._started_at is None else self._clock() - self._started_at))
        timer = f"{elapsed:.0f} 秒"
        usage = self._usage
        if usage is None:
            return f"{timer} · Token 待报告" + ("" if compact else " · 缓存 未知")
        cached = "未知" if usage.cache_read_input_tokens is None else f"{usage.cache_read_input_tokens:,}"
        label = "已报告" if usage.usage_complete else "已报告部分"
        if compact:
            return f"{timer} · {label} {self._usage_count_text('total_tokens')} Token · 缓存 {cached}"
        return (f"{timer} · {label} 输入 {self._usage_count_text('input_tokens')}"
                f" · 输出 {self._usage_count_text('output_tokens')}"
                f" · 缓存 {cached}")

    def activity_frame(self) -> str:
        # A small, fixed-width pulse driven by the same monotonic task clock.
        frames = ("·", "•", "●", "•")
        elapsed = self._elapsed_seconds + (0.0 if self._started_at is None else self._clock() - self._started_at)
        return frames[int(max(0.0, elapsed) * 6) % len(frames)]

    async def emit(self, event: StreamEvent) -> None:
        self._last_activity = self._clock()
        if self.turn_id is None and event.turn_id:
            self.turn_id = event.turn_id
        identity = event.item_id or event.data.get("tool_id")
        is_tool = event.item_kind in {TurnItemKind.TOOL, TurnItemKind.COMMAND} or event.type in {
            EventType.TOOL_USE_START, EventType.TOOL_USE_PROGRESS,
            EventType.TOOL_USE_RESULT, EventType.TOOL_USE_ERROR,
        }
        self._target = None
        if is_tool and isinstance(identity, str):
            lifecycle = event.type in {EventType.ITEM_STARTED, EventType.ITEM_COMPLETED,
                                      EventType.TOOL_USE_START, EventType.TOOL_USE_RESULT, EventType.TOOL_USE_ERROR}
            projected = (event.turn_id, identity, event.type)
            if lifecycle and projected in self._projected:
                return
            if lifecycle:
                self._projected[projected] = None
                if len(self._projected) > 1024:
                    self._projected.pop(next(iter(self._projected)))
            key = (event.turn_id, identity)
            if key not in self._tool_blocks:
                if self.group is None or not self.view.blocks or self.view.blocks[-1] is not self.group:
                    self.group = self.view.add_group()
                    self.group.inline = True
                result = event.data.get("result")
                name = str(event.data.get("tool_name") or
                           (result.get("tool_name") if isinstance(result, Mapping) else None) or "工具")
                block = self.view.add_tool(self.group, "● " + _TOOL_LABELS.get(name, name))
                block.tool_name = name
                self._tool_blocks[key] = block
                retained = {child for group in self.view.blocks for child in group.children}
                self._tool_blocks = {k: v for k, v in self._tool_blocks.items() if v in retained}
            self._target = self._tool_blocks[key]
            call_id = event.data.get("tool_call_id")
            if not isinstance(call_id, str):
                result_data = event.data.get("result")
                call_id = result_data.get("tool_call_id") if isinstance(result_data, Mapping) else None
            if isinstance(call_id, str):
                self._target.tool_call_id = call_id
        if event.type is EventType.ITEM_STARTED and event.item_kind is TurnItemKind.AGENT_MESSAGE:
            self._new_answer = True
        try:
            if event.type is EventType.RECOVERY:
                self._activity_status = "正在恢复"
                self.view.message(json.dumps(event.data, ensure_ascii=False) + "\n", "diagnostic")
            else:
                await super().emit(event)
            if self._target:
                self._project_tool(event, self._target)
        finally:
            self._target = None
        lifecycle_labels = {
            EventType.TURN_PAUSED: "等待你的处理",
            EventType.HUMAN_INPUT_REQUIRED: "等待你的确认或回答",
            EventType.TURN_CANCELLATION_REQUESTED: "正在停止，等待执行清理",
            EventType.TURN_ABORTED: "本轮已中止",
            EventType.TURN_COMPLETED: "本轮已结束",
            EventType.TURN_RESUMED: "等待模型响应",
        }
        if event.type in lifecycle_labels:
            self._activity_status = lifecycle_labels[event.type]
        if self.group and self._started_at is not None:
            self.group.title = f"执行记录 · {self._tool_count} 次操作"
        self.view.invalidate()

    def _project_tool(self, event: StreamEvent, block: Block) -> None:
        if event.type in {EventType.ITEM_STARTED, EventType.TOOL_USE_START}:
            block.operation_status = "running"
            block.preview = "进行中"
            if event.data.get("execution_started") is False:
                block.operation_status = "not_started"
                block.preview = "未执行"
            return
        if event.type not in {EventType.ITEM_COMPLETED, EventType.TOOL_USE_RESULT, EventType.TOOL_USE_ERROR}:
            if event.delta_kind in {ItemDeltaKind.COMMAND_STDOUT, ItemDeltaKind.COMMAND_STDERR}:
                delta = event.data.get("delta")
                if isinstance(delta, str):
                    block.preview = "\n".join(safe_terminal_text(delta).splitlines()[-2:])[:400]
            return
        result = event.data.get("result", event.data)
        if not isinstance(result, Mapping):
            result = {}
        output = result.get("structured_content")
        process = output if isinstance(output, Mapping) else {}
        is_command = block.tool_name in {"run_command", "manage_command"}
        process_status = process.get("status") if is_command else None
        session_id = process.get("session_id") if is_command else None
        if isinstance(session_id, str):
            block.session_id = session_id
        state = "success"
        if event.status is ItemStatus.OUTCOME_UNKNOWN or process_status == "outcome_unknown":
            state = "unknown"
        elif (event.status is ItemStatus.CANCELLED or result.get("error_code") == "cancelled"
              or process_status in {"cancelled", "canceled"}):
            state = "cancelled"
        elif event.status is ItemStatus.FAILED or event.type is EventType.TOOL_USE_ERROR or result.get("is_error"):
            state = "failed"
        elif block.tool_name in {"run_command", "manage_command"}:
            if process_status == "running":
                state = "running"
            elif isinstance(process.get("exit_code"), int) and process["exit_code"] != 0:
                state = "failed"
        block.operation_status = state
        block.failed = state in {"failed", "unknown"}
        error = event.error or result.get("error_message")
        code = result.get("error_code")
        if code == "mcp_connection_required":
            self.connection_required = True
        if code == "web_search_not_configured":
            self.search_configuration_required = True
        if state == "failed":
            known = {
                "command_failed": "命令执行失败；Ctrl+O 查看具体原因",
                "python_failed": "Python 执行失败；Ctrl+O 查看具体原因",
                "permission_denied": "系统拒绝访问，请检查文件或授权范围",
                "file_not_found": "找不到文件，请检查文件路径",
                "mcp_connection_required": "需要连接服务，请打开连接管理",
            }
            if block.tool_name in {"web_search", "web_fetch"}:
                summary = web_error_message(code)
            else:
                summary = (known.get(str(code)) or known.get(str(error))
                           or safe_terminal_text(str(error or "这一步未成功"))[:200])
            block.preview = summary
            self._activity_status = "这一步失败，继续处理中"
        elif state == "unknown":
            block.preview = "结果未确认；请检查实际结果后再继续"
            self._activity_status = "操作结果未确认"
        elif state == "cancelled":
            block.preview = "已取消"
            self._activity_status = "操作已取消"
        elif state == "running":
            block.preview = "命令运行中" if block.session_id else "进行中"
            self._activity_status = "命令运行中" if block.session_id else "正在处理"
        else:
            content = (output if isinstance(output, str)
                       else process.get("stdout", process.get("content", process.get("text"))))
            block.preview = ("\n".join(safe_terminal_text(content).splitlines()[:2])[:400]
                             if isinstance(content, str) and content.strip() else "操作完成")
            self._activity_status = "等待模型响应"
        if block.session_id and state != "running":
            for previous in (child for group in self.view.blocks for child in group.children):
                if (previous is not block and previous.session_id == block.session_id
                        and previous.operation_status == "running"):
                    previous.operation_status = state
                    previous.failed = block.failed
                    previous.preview = block.preview
        block.expanded = self.view.details
        if error:
            # The formatter may summarize known errors; retain the original separately.
            raw = safe_terminal_text(str(error))
            if raw not in block.text:
                block.append("错误原文：" + raw + "\n")
        if code:
            block.append("错误代码：" + safe_terminal_text(str(code)) + "\n")

    def _render_start_line(self, name: str, preview: object) -> None:
        self._answer_has_text = False
        self._new_answer = True
        self._pending_answer_whitespace = ""
        label = _TOOL_LABELS.get(name, name)
        self._activity_status = f"正在{label}"
        if self._target:
            title = safe_terminal_text(str(preview or "")).replace("\n", " ")
            self._target.title = f"● {label}" + (f" · {title[:180]}" if title else "")
            self._target.append(f"{name}\n{safe_terminal_text(str(preview or '参数未提供'))}\n")

    def _record_detail(self, value: str) -> None:
        if self._target:
            self._target.append(value + "\n")
        else:
            self.view.message(value + "\n", "notice")

    def _render_context_status(self, key: str, value: str) -> None:
        self._context_status[key] = value
        if self._context_detail is None or self._context_detail not in self.view.blocks:
            self._context_detail = self.view._add(Block("diagnostic", title="上下文状态"))
        self._context_detail.text = safe_terminal_text("\n".join(
            line for name, line in self._context_status.items() if name != "request_settings"
        ))
        self._context_detail.trim(MAX_TEXT)
        self.view.invalidate()

    def clear_context_status(self) -> None:
        super().clear_context_status()
        if self._context_detail is not None and self._context_detail in self.view.blocks:
            self.view.blocks.remove(self._context_detail)
        self._context_detail = None
        self.view.invalidate()

    def _write_line(self, value: str, *, record: bool = True) -> None:
        if record:
            self._record_detail(value)

    def _write_block(self, value: str) -> None:
        self._record_detail(value)

    def _write_result(self, value: object) -> None:
        for line in bounded_result_lines(value, width=self._width, max_rows=2000):
            self._record_detail(line.replace("(/verbose 查看完整结果)", "(界面显示上限)"))

    def _write_error_result(self, value: object) -> None:
        self._write_result(value)

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
        # Command deltas are retained by the formatter before its verbose write.
        self.view.invalidate()

    def set_verbose(self, verbose: bool) -> None:
        super().set_verbose(verbose)
        self.view.details = verbose
        for block in self.view.blocks:
            if block.kind == "group":
                block.expanded = verbose
                for child in block.children:
                    child.expanded = verbose
        self.view.invalidate()

    def stopping(self) -> None:
        self._activity_status = "正在停止，等待执行清理"
        self.view.invalidate()

    def interrupted(self) -> None:
        self.finish()
        self._activity_status = "执行已中断；未完成操作的结果未确认"
        for block in self._tool_blocks.values():
            if block.operation_status == "running":
                block.operation_status = "unknown"
                block.preview = "结果未确认；已发生的修改不会回滚"
                block.failed = True
        self.view.invalidate()

    def present_result(self, result: AgentResult) -> None:
        self.update_usage(result.usage)
        for call in result.tool_calls:
            for block in self._tool_blocks.values():
                if block.tool_call_id == call.tool_call_id and call.arguments is not None:
                    arguments = json.dumps(dict(call.arguments), ensure_ascii=False, default=str)
                    detail = "参数：" + safe_terminal_text(arguments)
                    if detail not in block.text:
                        block.append(detail + "\n")
                    for key in ("file_path", "path", "command"):
                        value = call.arguments.get(key)
                        if isinstance(value, str):
                            label = _TOOL_LABELS.get(block.tool_name, block.tool_name)
                            preview = safe_terminal_text(value).replace("\n", " ")[:180]
                            block.title = f"● {label} · {preview}"
                            break
        if self._receipt is None or self._receipt.in_scrollback:
            self._receipt = self.view._add(Block("receipt"))
        self._receipt.kind = "receipt"
        self._receipt.text = format_usage_summary(result.usage) + "\n"
        if result.status == "paused":
            self._receipt.text = "暂停时，" + self._receipt.text
            self._activity_status = "等待你的确认或回答"
            self.view.invalidate()
            return
        if result.stop_reason == "cancelled":
            self._activity_status = "已停止"
            text = "已停止 · 已发生的文件修改不会回滚"
            for block in self._tool_blocks.values():
                if block.operation_status == "running":
                    block.operation_status = "cancelled"
                    block.preview = "已取消"
        elif result.status == "failed":
            self._activity_status = "本轮未完成"
            message = next((d.message for d in result.diagnostics if d.severity == "error"), "")
            summary = safe_terminal_text(message).replace("\n", " ")[:200]
            text = "! 本轮未完成" + ("：" + summary if summary else "；Ctrl+O 查看原因")
            self._receipt.kind = "error"
        else:
            self._activity_status = "本轮已结束"
            scopes = {
                "acceptance_command": "用户指定的验收命令",
                "workspace_change": "工作区修改及修改后的检查",
                "workspace_change_recorded": "仅记录工作区修改",
            }
            scope = scopes.get(result.validation_scope or "", result.validation_scope or "未声明范围")
            if result.validation_scope == "workspace_change_recorded":
                text = "已记录工作区修改"
            elif result.validation_status == "passed":
                text = "✓ 已验证 · " + safe_terminal_text(scope)
            elif result.validation_status == "failed":
                text = "! 验收未通过 · " + safe_terminal_text(scope)
            else:
                text = ""
        if result.files:
            text += ("\n" if text else "") + "文件引用：\n" + "\n".join(
                safe_terminal_text(path) for path in result.files
            )
        self._receipt.text = (text + "\n" if text else "") + format_usage_summary(result.usage) + "\n"
        diagnostics = [f"{d.code}: {d.message}" for d in result.diagnostics]
        if result.stop_reason:
            diagnostics.append("结束原因：" + result.stop_reason)
        if diagnostics:
            # Final technical causes remain inspectable without duplicating the notice.
            if not any(block.kind == "diagnostic" and block.title == result.turn_id for block in self.view.blocks):
                self.view._add(Block("diagnostic", title=result.turn_id,
                                     text=safe_terminal_text("\n".join(diagnostics)) + "\n"))
        if any(block.operation_status == "running" for block in self._tool_blocks.values()):
            self._receipt.text += "仍有命令运行中；可在详情查看会话。\n"
        self.view.invalidate()

    def finish(self) -> None:
        running = self._started_at is not None
        super().finish()
        if running and self.group:
            self.group.title = f"执行记录 · {self._tool_count} 次操作 · {self._elapsed_seconds:.1f}s"
            self.group.expanded = self.view.details
        self.view.invalidate()
